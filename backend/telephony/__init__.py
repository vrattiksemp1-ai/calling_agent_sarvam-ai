"""Twilio Media Streams telephony bridge.

This package lets the web app place a real outbound phone call. Twilio connects
the call to a WebSocket (`/api/calls/stream`) and streams G.711 mu-law audio
both ways. The bridge runs speech detection (VAD) on the inbound audio, sends
each utterance through the existing STT -> LLM -> TTS pipeline, and streams the
synthesised reply back over the call.

Nothing here is required to run the web MVP - the feature is inert until
TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN / TWILIO_PHONE_NUMBER are configured.
"""

from backend.telephony.twilio_service import TwilioService

__all__ = ["TwilioService"]
