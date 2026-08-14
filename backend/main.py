"""FastAPI application entry point for the Sarvam Cloud Lead Agent."""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.api.call_routes import router as call_router
from backend.api.routes import router
from backend.config import Settings, get_settings
from backend.conversation import ConversationEngine
from backend.database import create_engine_and_session, make_database_url
from backend.errors import AppError
from backend.providers.llm_client import LlmClient
from backend.providers.sarvam_client import SarvamClient
from backend.rate_limit import RateLimiter
from backend.telephony.call_manager import CallRegistry
from backend.telephony.exotel_service import ExotelService
from backend.telephony.turn_flow import TurnFlow
from backend.telephony.twilio_service import TwilioService
from backend.utils.logging import setup_logging

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


def build_app(
    settings: Settings | None = None,
    *,
    session_factory=None,
    sarvam_client: SarvamClient | None = None,
    llm_client: LlmClient | None = None,
    twilio_client: TwilioService | None = None,
    exotel_client: ExotelService | None = None,
    call_registry: CallRegistry | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.debug)
    from backend.pipeline_trace import configure_pipeline_trace

    configure_pipeline_trace(
        enabled=settings.pipeline_trace_enabled,
        max_chars=settings.pipeline_trace_max_chars,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Pre-build the greeting TwiML in the background so the first /twiml
        # webhook (Twilio's ~15s budget) is served from cache, not from a slow
        # synchronous LLM + TTS call.
        startup_tasks: list[asyncio.Task] = []
        startup_tasks.append(asyncio.create_task(app.state.turn_flow.warm_greeting()))
        try:
            yield
        finally:
            for task in startup_tasks:
                task.cancel()
            if startup_tasks:
                await asyncio.gather(*startup_tasks, return_exceptions=True)
            for client_name in (
                "sarvam_client",
                "llm_client",
                "twilio_client",
                "exotel_client",
            ):
                client = getattr(app.state, client_name, None)
                if client is not None and hasattr(client, "aclose"):
                    await client.aclose()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url=None,
        openapi_url="/api/openapi.json" if settings.debug else None,
    )

    if settings.cors_enabled and settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=False,
            allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
            allow_headers=["*"],
        )

    db_url = make_database_url(settings)
    _, factory = create_engine_and_session(db_url)
    logger.info(
        "Database ready (%s)",
        "postgresql" if "postgresql" in db_url else "sqlite",
    )
    app.state.settings = settings
    app.state.session_factory = session_factory or factory
    app.state.sarvam_client = sarvam_client or SarvamClient(settings)
    app.state.llm_client = llm_client or LlmClient(settings)
    app.state.conversation_engine = ConversationEngine(
        app.state.llm_client,
        business_name=settings.business_name,
        business_description=settings.business_description,
        agent_name=settings.agent_name,
        disclose_ai_assistant=settings.disclose_ai_assistant,
        settings=settings,
    )
    app.state.rate_limiter = RateLimiter(
        enabled=settings.rate_limit_enabled,
        per_minute=settings.rate_limit_per_minute,
    )
    app.state.call_rate_limiter = RateLimiter(
        enabled=settings.rate_limit_enabled,
        per_minute=settings.call_rate_limit_per_minute,
    )
    app.state.twilio_client = twilio_client or TwilioService(settings)
    app.state.exotel_client = exotel_client or ExotelService(settings)
    app.state.call_registry = call_registry or CallRegistry()
    app.state.turn_flow = TurnFlow(
        settings=settings,
        session_factory=app.state.session_factory,
        engine=app.state.conversation_engine,
        sarvam=app.state.sarvam_client,
        twilio=app.state.twilio_client,
        registry=app.state.call_registry,
    )

    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if request.url.path == "/api/calls" and request.method == "POST":
            limiter: RateLimiter = request.app.state.call_rate_limiter
        else:
            limiter: RateLimiter = request.app.state.rate_limiter
        if request.url.path.startswith("/api/") and request.method in {"POST", "DELETE"}:
            client_ip = request.client.host if request.client else "unknown"
            allowed, retry_after = limiter.check(client_ip)
            if not allowed:
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": {
                            "code": "RATE_LIMITED",
                            "message": "Too many requests. Please slow down.",
                            "retryable": True,
                            "details": None,
                        }
                    },
                    headers={"Retry-After": str(retry_after)},
                )
        return await call_next(request)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        settings_obj: Settings = request.app.state.settings
        logger.warning(
            "Request failed: %s %s -> %s", request.method, request.url.path, exc.code
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                    "details": exc.details,
                }
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception):
        settings_obj: Settings = request.app.state.settings
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        details = f"{type(exc).__name__}: {exc}" if settings_obj.debug else None
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": "Unexpected server error. See logs for details.",
                    "retryable": True,
                    "details": details,
                }
            },
        )

    app.include_router(router)
    app.include_router(call_router)

    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static") or request.url.path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
        return response

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> HTMLResponse:
            return HTMLResponse(FRONTEND_DIR.joinpath("index.html").read_text(encoding="utf-8"))

    return app


app = build_app()
