"""Exotel Connect Voice AI integration for the Mumbai region."""

from __future__ import annotations

import base64

import httpx

from backend.config import Settings
from backend.errors import AppError, ProviderUnavailableError
from backend.telephony.phone import normalize_e164
from backend.telephony.twilio_service import OutboundCallResult
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class ExotelNotConfiguredError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="CALL_NOT_CONFIGURED",
            message=(
                "Exotel calling is not configured. Set the Exotel account, API "
                "credentials and caller ID in the server environment."
            ),
            retryable=False,
            status_code=400,
        )


class ExotelService:
    """Create direct bidirectional AgentStream calls without a dashboard flow."""

    def __init__(
        self, settings: Settings, http_client: httpx.AsyncClient | None = None
    ) -> None:
        self._settings = settings
        self._own_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=settings.exotel_call_timeout
        )

    @property
    def configured(self) -> bool:
        settings = self._settings
        return bool(
            settings.exotel_account_sid
            and settings.exotel_api_key
            and settings.exotel_api_token
            and settings.exotel_caller_id
        )

    def public_base(self) -> str:
        base = self._settings.public_base.rstrip("/")
        if not base:
            raise AppError(
                code="CALL_NOT_CONFIGURED",
                message=(
                    "PUBLIC_BASE_URL is empty. Configure the public HTTPS URL "
                    "before placing an Exotel call."
                ),
                retryable=False,
                status_code=400,
            )
        return base

    def stream_ws_url(self) -> str:
        base = self.public_base()
        if base.startswith("https://"):
            base = "wss://" + base[len("https://") :]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://") :]
        elif not base.startswith(("ws://", "wss://")):
            base = "wss://" + base
        return base + "/api/calls/exotel/stream"

    def _connect_url(self) -> str:
        base = self._settings.exotel_base_url.rstrip("/")
        account_sid = self._settings.exotel_account_sid
        if self._settings.exotel_flow_id:
            return f"{base}/v1/Accounts/{account_sid}/Calls/connect"
        return f"{base}/v1/accounts/{account_sid}/calls/connect"

    def _auth_header(self) -> str:
        raw = (
            f"{self._settings.exotel_api_key}:{self._settings.exotel_api_token}"
        ).encode("utf-8")
        return "Basic " + base64.b64encode(raw).decode("ascii")

    async def start_call(self, to_number: str) -> OutboundCallResult:
        if not self.configured:
            raise ExotelNotConfiguredError()
        to = normalize_e164(to_number)
        if self._settings.exotel_flow_id:
            account_sid = self._settings.exotel_account_sid
            flow_id = self._settings.exotel_flow_id
            data = {
                "From": to,
                "CallerId": self._settings.exotel_caller_id,
                "Url": (
                    f"https://my.exotel.com/{account_sid}/exoml/"
                    f"start_voice/{flow_id}"
                ),
                "CallType": "trans",
            }
            if self._settings.exotel_status_callback_url:
                data["StatusCallback"] = (
                    self._settings.exotel_status_callback_url
                )
                data["StatusCallbackEvents"] = "terminal"
        else:
            stream_url = self.stream_ws_url()
            data = {
                "from": to,
                "callerid": self._settings.exotel_caller_id,
                "streamurl": stream_url,
                "streamtype": "bidirectional",
            }
            if self._settings.exotel_status_callback_url:
                data["statuscallback"] = (
                    self._settings.exotel_status_callback_url
                )
                data["statuscallbackevents[]"] = "terminal"
        try:
            response = await self._client.post(
                self._connect_url(),
                data=data,
                headers={"Authorization": self._auth_header()},
            )
            if response.status_code >= 400:
                detail = response.text[:500]
                for sensitive in (
                    to,
                    self._settings.exotel_caller_id,
                    self._settings.exotel_account_sid,
                ):
                    if sensitive:
                        detail = detail.replace(sensitive, "[REDACTED]")
                logger.warning(
                    "Exotel call rejected: status=%s detail=%s",
                    response.status_code,
                    detail,
                )
            response.raise_for_status()
            payload = response.json()
            call = payload.get("call") or payload.get("Call") or payload
            call_sid = str(call.get("sid") or call.get("Sid") or "")
            if not call_sid:
                raise ValueError("Exotel response did not contain a call SID")
            status = str(call.get("status") or call.get("Status") or "queued")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            logger.warning(
                "Exotel call creation failed: %s", type(exc).__name__
            )
            raise ProviderUnavailableError(
                "Exotel could not place the call. Please try again in a moment."
            ) from exc
        logger.info("Exotel outbound call initiated: call_sid=%s", call_sid)
        return OutboundCallResult(
            call_sid=call_sid,
            status=status,
            to=to,
            from_=self._settings.exotel_caller_id,
        )

    async def complete_call(self, call_sid: str) -> None:
        """Close is driven by AgentStream/Exotel; avoid an undocumented API call."""

        if call_sid:
            logger.info("Exotel call completion requested: call_sid=%s", call_sid)

    async def aclose(self) -> None:
        if self._own_client:
            await self._client.aclose()
