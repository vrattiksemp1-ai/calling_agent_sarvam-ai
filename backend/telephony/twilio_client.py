"""Twilio REST client for placing and ending outbound phone calls.

Uses plain httpx with HTTP Basic auth (Account SID : Auth Token) so there is no
extra runtime dependency beyond what the project already installs.

Call flow:
  1. The web app posts to POST /api/calls with the target number.
  2. This client creates a Twilio call whose TwiML opens a Media Stream
     WebSocket (wss://<public-base>/api/calls/stream) for two-way audio.
  3. When the call is answered, Twilio streams audio to the bridge and the
     agent talks back over the same socket.
"""

from __future__ import annotations

import base64

import httpx

from backend.config import Settings
from backend.errors import AppError, ProviderUnavailableError
from backend.utils.logging import get_logger

logger = get_logger(__name__)

TWILIO_API_BASE = "https://api.twilio.com/2010-04-01/Accounts/{sid}"


class CallNotConfiguredError(AppError):
    def __init__(self):
        super().__init__(
            code="CALL_NOT_CONFIGURED",
            message="Phone calling is not configured. Set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER in .env, then restart.",
            retryable=False,
            status_code=400,
        )


class TwilioClient:
    def __init__(self, settings: Settings, http_client: httpx.AsyncClient | None = None):
        self._settings = settings
        self._own_client = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=httpx.Timeout(settings.twilio_call_timeout))

    async def aclose(self) -> None:
        if self._own_client:
            await self._http.aclose()

    @property
    def configured(self) -> bool:
        s = self._settings
        return bool(s.twilio_account_sid and s.twilio_auth_token and s.twilio_from_number)

    def _auth(self) -> str:
        raw = f"{self._settings.twilio_account_sid}:{self._settings.twilio_auth_token}"
        return base64.b64encode(raw.encode("utf-8")).decode("ascii")

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self._auth()}", "Content-Type": "application/x-www-form-urlencoded"}

    def stream_twiml(self) -> str:
        base = self._settings.twilio_call_public_base_url.rstrip("/")
        if base.startswith("https://"):
            base = "wss://" + base[len("https://"):]
        elif base.startswith("http://"):
            base = "ws://" + base[len("http://"):]
        return (
            '<Response><Connect><Stream url="{url}">'
            "<Parameter name=\"role\" value=\"lead-agent\"/>"
            "</Stream></Connect></Response>"
        ).format(url=base + "/api/calls/stream")

    async def start_outbound_call(self, to_number: str) -> str:
        """Place the call and return the Twilio call SID."""
        s = self._settings
        if not self.configured:
            raise CallNotConfiguredError()
        if not s.twilio_call_public_base_url:
            raise AppError(
                code="CALL_NOT_CONFIGURED",
                message="TWILIO_CALL_PUBLIC_BASE_URL is empty. Point it at the public "
                "ngrok URL so Twilio can reach the audio WebSocket.",
                retryable=False,
                status_code=400,
            )
        base = TWILIO_API_BASE.format(sid=s.twilio_account_sid)
        data = {
            "To": to_number,
            "From": s.twilio_from_number,
            "Twiml": self.stream_twiml(),
        }
        try:
            resp = await self._http.post(
                f"{base}/Calls.json", headers=self._headers(), data=data
            )
            if resp.status_code >= 400:
                detail = resp.text[:500]
                logger.error("Twilio create call failed: %s", detail)
                raise ProviderUnavailableError(
                    "Twilio could not place the call. Check your trial credit, that "
                    "the destination number is verified, and the log for details.",
                    details=detail,
                )
            payload = resp.json()
            sid = payload.get("sid") or ""
            if not sid:
                raise ProviderUnavailableError(
                    "Twilio did not return a call SID.", details=resp.text[:500]
                )
            logger.info("Outbound call initiated: %s -> %s (%s)", s.twilio_from_number, to_number, sid)
            return sid
        except AppError:
            raise
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                "Twilio is not reachable.", details=str(exc)[:500]
            )

    async def complete_call(self, call_sid: str) -> None:
        """Hang up an in-progress call."""
        if not call_sid:
            return
        s = self._settings
        base = TWILIO_API_BASE.format(sid=s.twilio_account_sid)
        try:
            await self._http.post(
                f"{base}/Calls/{call_sid}.json",
                headers=self._headers(),
                data={"Status": "completed"},
            )
        except httpx.HTTPError as exc:
            logger.warning("Twilio complete call failed for %s: %s", call_sid, exc)
