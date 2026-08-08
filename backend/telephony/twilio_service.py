"""Twilio server-side integration built on the official `twilio` SDK.

Runs only in the backend. Credentials come from settings (never from the
browser) and are never logged or returned in API responses.

Responsibilities:
  * building the SDK client from environment variables,
  * starting an outbound call (with E.164 + trial-mode guards),
  * generating the TwiML that opens the Media Stream WebSocket,
  * listing the numbers verified in the Twilio Console (dropdown source),
  * verifying incoming Twilio webhook signatures,
  * converting Twilio errors into safe application errors.
"""

from __future__ import annotations

import asyncio
import secrets
import time
from dataclasses import dataclass

from backend.config import Settings
from backend.errors import AppError, ProviderUnavailableError
from backend.telephony.phone import normalize_e164
from backend.utils.logging import get_logger

logger = get_logger(__name__)

# Twilio webhook connection overrides (URL fragment): raise the TCP connect
# budget to its maximum and retry connect/5xx failures so a transient tunnel
# blip does not make Twilio play its "unable to reach the server" error.
# `rt` (read-timeout) retries are deliberately excluded: a read-timeout means
# the request may already have been processed, and retrying it would duplicate
# the turn. The 15 s hard cap on call-processing requests still applies.
WEBHOOK_CONNECTION_OVERRIDES = "#ct=10000&rt=15000&tt=15000&rc=3&rp=ct,5xx"

VALID_CALL_STATUSES = {
    "initiated",
    "ringing",
    "answered",
    "completed",
    "busy",
    "failed",
    "no-answer",
    "canceled",
}

# Twilio statuses we treat as "the call is over / unusable".
TERMINAL_STATUSES = {"busy", "failed", "no-answer", "canceled"}

VERIFIED_NUMBERS_CACHE_SECONDS = 30.0


class CallNotConfiguredError(AppError):
    def __init__(self):
        super().__init__(
            code="CALL_NOT_CONFIGURED",
            message="Phone calling is not configured. Set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER in .env, then restart.",
            retryable=False,
            status_code=400,
        )


class CallNotAllowedError(AppError):
    def __init__(self):
        super().__init__(
            code="CALL_NOT_ALLOWED",
            message="Trial mode allows calls only to numbers verified in the "
            "Twilio Console. Verify the destination number and try again.",
            retryable=False,
            status_code=400,
        )


class InvalidTwilioSignatureError(AppError):
    def __init__(self):
        super().__init__(
            code="INVALID_TWILIO_SIGNATURE",
            message="Request signature could not be verified as coming from Twilio.",
            retryable=False,
            status_code=403,
        )


@dataclass(frozen=True)
class OutboundCallResult:
    call_sid: str
    status: str
    to: str
    from_: str


class TwilioService:
    def __init__(self, settings: Settings, client=None) -> None:
        self._settings = settings
        self._client = client
        self._own_client = client is None
        self._verified_cache: tuple[float, list[str]] | None = None

        # RequestValidator never holds the token; it only stores it internally.
        self._validator = None
        from twilio.request_validator import RequestValidator

        if settings.twilio_auth_token:
            self._validator = RequestValidator(settings.twilio_auth_token)

    # ---------- setup ----------

    @property
    def configured(self) -> bool:
        s = self._settings
        return bool(
            s.twilio_account_sid and s.twilio_auth_token and s.twilio_from
        )

    @property
    def trial_mode(self) -> bool:
        return bool(self._settings.twilio_trial_mode)

    def _build_client(self):
        """Lazily create the sync SDK client (wrapped in threads from async code)."""
        from twilio.rest import Client
        from twilio.http.http_client import TwilioHttpClient

        s = self._settings
        http = TwilioHttpClient(timeout=s.twilio_call_timeout)
        return Client(s.twilio_account_sid, s.twilio_auth_token, http_client=http)

    def _get_client(self):
        if self._client is None:
            self._client = self._build_client()
        return self._client

    async def aclose(self) -> None:
        # The sync TwilioHttpClient keeps no long-lived sessions to close.
        self._client = None
        self._own_client = False

    # ---------- public URLs ----------

    def public_base(self) -> str:
        base = self._settings.public_base.rstrip("/")
        if not base:
            raise AppError(
                code="CALL_NOT_CONFIGURED",
                message="PUBLIC_BASE_URL is empty. Point it at the public ngrok "
                "URL so Twilio can reach the call endpoints.",
                retryable=False,
                status_code=400,
            )
        return base

    def voice_url(self) -> str:
        return f"{self.public_base()}/api/calls/twiml"

    def turn_url(self) -> str:
        """Webhook that receives each <Gather input="speech"> transcript.

        Twilio's speech-recognition <Gather> action callback is not always
        signed with X-Twilio-Signature (observed live on the trial pipeline),
        so the URL carries a shared-secret query token that the handler
        verifies whenever the signature header is absent.

        The URL fragment carries Twilio "webhook connection overrides"
        (https://www.twilio.com/docs/usage/webhooks/webhooks-connection-overrides):
        a longer connect timeout and extra retries on connect/read failures so
        a transient tunnel blip does not make Twilio play its "unable to reach
        the requested server" prompt (the 15 s total is still enforced by Twilio
        on call-processing requests).
        """
        url = f"{self.public_base()}/api/calls/turn"
        secret = self._settings.twilio_turn_webhook_secret
        if secret:
            url += f"?turn_token={secret}"
        return url + WEBHOOK_CONNECTION_OVERRIDES

    def turn_result_url(self, call_sid: str, pending_token: str) -> str:
        """Webhook Twilio Redirects to while a slow turn finishes in-process."""
        from urllib.parse import urlencode

        query = {"call_sid": call_sid, "pending": pending_token}
        secret = self._settings.twilio_turn_webhook_secret
        if secret:
            query["turn_token"] = secret
        return (
            f"{self.public_base()}/api/calls/turn-result?{urlencode(query)}"
            + WEBHOOK_CONNECTION_OVERRIDES
        )

    def audio_url(self, file_id: str) -> str:
        """Public URL for a hosted TTS WAV that <Play> can fetch."""
        return f"{self.public_base()}/api/calls/audio/{file_id}" + WEBHOOK_CONNECTION_OVERRIDES

    def status_callback_url(self) -> str:
        configured = self._settings.twilio_status_callback_url.strip()
        if configured:
            return configured
        return f"{self.public_base()}/api/calls/status"

    def stream_ws_url(self) -> str:
        """WebSocket URL for Twilio Media Streams (wss:// derived from public base)."""
        base = self.public_base()
        if base.startswith("https://"):
            return "wss://" + base[len("https://"):]
        if base.startswith("http://"):
            return "ws://" + base[len("http://"):]
        return "wss://" + base

    # ---------- TwiML ----------

    def stream_twiml(self) -> str:
        """TwiML for a true bidirectional Media Stream.

        ``<Connect><Stream>`` blocks the TwiML call flow while the WebSocket is
        active and permits server-to-caller ``media``, ``mark`` and ``clear``
        messages. ``<Start><Stream>`` is receive-only and cannot power the
        conversational bridge.
        """
        url = self.stream_ws_url() + "/api/calls/stream"
        status_cb = self.public_base() + "/api/calls/stream-status"
        return (
            '<?xml version="1.0" encoding="UTF-8"?>'
            "<Response><Connect><Stream url=\"{url}\" statusCallback=\"{cb}\">"
            '<Parameter name="role" value="lead-agent"/>'
            "</Stream></Connect></Response>"
        ).format(url=url, cb=status_cb)

    # ---------- verified numbers ----------

    def _configured_verified_numbers(self) -> list[str]:
        """Verified numbers from environment (fallback list), normalized."""
        raw_items = [self._settings.twilio_test_phone_number]
        raw_items += [
            item.strip()
            for item in self._settings.twilio_verified_numbers.split(",")
            if item.strip()
        ]
        numbers: list[str] = []
        for raw in raw_items:
            try:
                normalized = normalize_e164(raw)
            except AppError:
                continue
            if normalized not in numbers:
                numbers.append(normalized)
        return numbers

    async def verified_numbers(self) -> list[str]:
        """Numbers verified in the Twilio Console, plus the env fallback list."""
        if self._verified_cache is not None:
            fetched_at, cached = self._verified_cache
            if time.monotonic() - fetched_at < VERIFIED_NUMBERS_CACHE_SECONDS:
                return list(cached)

        numbers = self._configured_verified_numbers()
        # When .env already lists verified destinations, skip OutgoingCallerIds.
        # Some trial/API keys return 401 for that endpoint even though Calls work.
        if numbers:
            self._verified_cache = (time.monotonic(), list(numbers))
            return list(numbers)

        if self.configured and not getattr(self, "_verified_api_disabled", False):
            try:
                await asyncio.to_thread(self._fetch_verified_numbers, numbers)
            except Exception as exc:  # noqa: BLE001 - SDK raises several types
                status = getattr(exc, "status", None)
                code = getattr(exc, "code", None)
                if status == 401 or code == 20003 or "20003" in str(exc):
                    self._verified_api_disabled = True
                    logger.warning(
                        "Twilio OutgoingCallerIds unauthorized; "
                        "using TWILIO_VERIFIED_NUMBERS / TWILIO_TEST_PHONE_NUMBER only"
                    )
                else:
                    logger.warning("Could not refresh verified numbers from Twilio")

        self._verified_cache = (time.monotonic(), list(numbers))
        return list(numbers)

    def _fetch_verified_numbers(self, into: list[str]) -> None:
        client = self._get_client()
        for item in client.outgoing_caller_ids.list(limit=50):
            number = item.phone_number
            if number and number not in into:
                into.append(number)

    async def is_allowed_to_call(self, to_number: str) -> bool:
        if not self.trial_mode:
            return True
        verified = await self.verified_numbers()
        return to_number in verified

    # ---------- outbound calls ----------

    async def start_call(self, to_number: str) -> OutboundCallResult:
        """Place an outbound call and return a safe result."""
        if not self.configured:
            raise CallNotConfiguredError()

        to = normalize_e164(to_number)
        # Resolve public URLs early so misconfiguration surfaces before any
        # trial/verified-number lookups or API calls.
        voice_url = self.voice_url()
        status_callback_url = self.status_callback_url()
        if not await self.is_allowed_to_call(to):
            raise CallNotAllowedError()

        from_number = self._settings.twilio_from
        call = await asyncio.to_thread(
            self._create_call,
            to,
            from_number,
            voice_url,
            status_callback_url,
        )
        logger.info("Outbound call initiated: %s -> %s (%s)", from_number, to, call.sid)
        return OutboundCallResult(
            call_sid=call.sid,
            status=call.status or "queued",
            to=to,
            from_=from_number,
        )

    def _create_call(self, to: str, from_number: str, voice_url: str, status_callback_url: str):
        client = self._get_client()
        try:
            return client.calls.create(
                to=to,
                from_=from_number,
                url=voice_url,
                status_callback=status_callback_url,
                status_callback_event=[
                    "initiated",
                    "ringing",
                    "answered",
                    "completed",
                    "busy",
                    "failed",
                    "no-answer",
                    "canceled",
                ],
            )
        except Exception as exc:  # noqa: BLE001 - SDK raises several types
            raise self._to_safe_error(exc)

    async def complete_call(self, call_sid: str) -> None:
        """Hang up an in-progress call."""
        if not call_sid or not self.configured:
            return
        try:
            await asyncio.to_thread(
                lambda: self._get_client().calls(call_sid).update(status="completed")
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Twilio complete call failed for %s: %s", call_sid, safe_exc(exc))

    # ---------- webhook signature validation ----------

    def validate_signature(self, uri: str, params: dict, signature: str | None) -> bool:
        if self._validator is None or not signature:
            return False
        return self._validator.validate(uri, params, signature)

    def validate_turn_callback(
        self,
        uri: str,
        params: dict,
        signature: str | None,
        token: str | None,
    ) -> bool:
        """Validate a <Gather> turn webhook.

        When X-Twilio-Signature is present it is checked first. Redirect and
        speech-recognition callbacks are often unsigned or signed against a
        slightly different public URL behind ngrok, so a matching shared-secret
        turn_token is also accepted.
        """
        if signature:
            if self.validate_signature(uri, params, signature):
                return True
            # Present but invalid signature: allow only a matching turn_token
            # (ngrok/public URL mismatches). Never accept a bad signature alone.
            secret = self._settings.twilio_turn_webhook_secret
            return bool(secret) and secrets.compare_digest(token or "", secret)
        secret = self._settings.twilio_turn_webhook_secret
        if not secret:
            # Unsigned trial Gather callbacks with no shared secret configured.
            return True
        return secrets.compare_digest(token or "", secret)

    # ---------- error conversion ----------

    def _to_safe_error(self, exc: Exception) -> AppError:
        from twilio.base.exceptions import TwilioRestException

        if isinstance(exc, TwilioRestException):
            # Log the technical detail server-side only; never return it.
            logger.error(
                "Twilio API error: status=%s code=%s detail=%s",
                exc.status,
                exc.code,
                safe_exc(exc),
            )
            if exc.code in (21211, 13223, 13224):
                return AppError(
                    code="INVALID_PHONE_NUMBER",
                    message="The destination number is invalid or not routable. "
                    "Check the E.164 format and try again.",
                    retryable=False,
                    status_code=400,
                )
            if exc.code in (21408, 13222):
                return CallNotAllowedError()
            return ProviderUnavailableError(
                "Twilio could not place the call. Check your trial credit and "
                "that the destination number is verified (see the logs for details)."
            )
        logger.warning("Twilio call failed (non-API): %s", safe_exc(exc))
        return ProviderUnavailableError(
            "Twilio could not place the call. Please try again in a moment."
        )


def safe_exc(exc: Exception) -> str:
    """Format an exception for logs without ever including credentials."""
    text = str(exc)
    # Defensive redaction in case an SDK message echoes request headers.
    for marker in ("Basic ", "Authorization: ", "TWILIO_AUTH_TOKEN"):
        if marker in text:
            idx = text.find(marker)
            text = text[:idx] + marker + "<redacted>"
    return text[:500]
