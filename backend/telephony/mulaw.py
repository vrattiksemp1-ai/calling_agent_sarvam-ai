"""G.711 mu-law codec and small WAV helpers used by the telephony bridge.

Twilio Media Streams carries audio as base64-encoded 8 kHz mu-law (PCMU).
Sarvam STT accepts a WAV file and Sarvam TTS returns a WAV, so we convert:

  mu-law bytes  --decode-->  int16 PCM  --wav header-->  STT file
  TTS WAV       --parse-->   int16 PCM  --downsample-->  mu-law bytes

mu-law encode/decode tables follow the classic CCITT algorithm.
"""

from __future__ import annotations

import base64
import struct

SAMPLE_RATE = 8000

# ---------- mu-law decode ----------

_DECODE_TABLE: list[int] = []


def _build_decode_table() -> list[int]:
    table = []
    for ulaw in range(256):
        inverted = (~ulaw) & 0xFF
        sign = inverted & 0x80
        exponent = (inverted >> 4) & 0x07
        mantissa = inverted & 0x0F
        sample = ((mantissa << 3) + 0x84) << exponent
        sample -= 0x84
        table.append(-sample if sign else sample)
    return table


_DECODE_TABLE = _build_decode_table()

# ---------- mu-law encode ----------

_ENCODE_CLAMP = 32767


def _build_encode_table() -> list[int]:
    """Map each non-negative PCM sample to the mu-law byte whose decoded value
    is closest to it. Built from the decoder so encode/decode are consistent."""
    import bisect

    positives = sorted(
        ((_DECODE_TABLE[b], b) for b in range(0x80, 0x100)),
        key=lambda t: t[0],
    )
    magnitudes = [t[0] for t in positives]
    table = []
    for m in range(32768):
        i = bisect.bisect_left(magnitudes, m)
        if i == 0:
            best = positives[0]
        elif i >= len(positives):
            best = positives[-1]
        else:
            lo, hi = positives[i - 1], positives[i]
            best = lo if (m - lo[0]) <= (hi[0] - m) else hi
        table.append(best[1])
    return table


_ENCODE_TABLE = _build_encode_table()


def encode_sample(sample: int) -> int:
    if sample > _ENCODE_CLAMP:
        sample = _ENCODE_CLAMP
    elif sample < -_ENCODE_CLAMP:
        sample = -_ENCODE_CLAMP
    code = _ENCODE_TABLE[sample if sample >= 0 else -sample]
    return code if sample >= 0 else (code & 0x7F)


def encode_mulaw(pcm16: bytes) -> bytes:
    """Encode little-endian int16 PCM to mu-law bytes."""
    count = len(pcm16) // 2
    values = struct.unpack(f"<{count}h", pcm16) if count else ()
    return bytes(encode_sample(s) for s in values)


def decode_mulaw(payload: bytes) -> bytes:
    """Decode mu-law bytes to little-endian int16 PCM."""
    if not payload:
        return b""
    return struct.pack(f"<{len(payload)}h", *(_DECODE_TABLE[b] for b in payload))


# ---------- WAV helpers ----------


def pcm16_to_wav(pcm16: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Wrap int16 PCM bytes in a canonical 16-bit mono WAV file."""
    data_size = len(pcm16)
    block_align = 2
    byte_rate = sample_rate * block_align
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", 36 + data_size),
            b"WAVE",
            b"fmt ",
            struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, block_align, 16),
            b"data",
            struct.pack("<I", data_size),
            pcm16,
        ]
    )


def wav_to_pcm16(wav_bytes: bytes) -> tuple[bytes, int]:
    """Extract int16 PCM and sample rate from a WAV file.

    Returns (pcm16, sample_rate). Supports 16-bit PCM and 8-bit unsigned
    (which is upconverted). Returns (b"", 8000) if no audio data is present.
    """
    if len(wav_bytes) < 44 or wav_bytes[:4] != b"RIFF":
        return b"", SAMPLE_RATE
    pos = 12
    data_start = None
    sample_rate = SAMPLE_RATE
    bits = 16
    channels = 1
    audio_format = 1
    fmt_seen = False
    while pos + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[pos:pos + 4]
        (chunk_size,) = struct.unpack_from("<I", wav_bytes, pos + 4)
        body = pos + 8
        if chunk_id == b"fmt " and body + chunk_size <= len(wav_bytes) and not fmt_seen:
            fmt_seen = True
            audio_format, channels = struct.unpack_from("<HH", wav_bytes, body)
            (sample_rate,) = struct.unpack_from("<I", wav_bytes, body + 4)
            (bits,) = struct.unpack_from("<H", wav_bytes, body + 14)
        elif chunk_id == b"data" and body <= len(wav_bytes):
            data_start = body
            data_end = min(body + chunk_size, len(wav_bytes))
            raw = wav_bytes[data_start:data_end]
            if audio_format == 1 and bits == 16:
                return raw, sample_rate or SAMPLE_RATE
            if audio_format == 1 and bits == 8:
                unsigned = list(raw)
                signed = struct.pack(f"<{len(unsigned)}h", *((b << 8) - 32768 for b in unsigned))
                return signed, sample_rate or SAMPLE_RATE
            return b"", sample_rate or SAMPLE_RATE
        pos = body + chunk_size + (chunk_size & 1)
    return b"", sample_rate or SAMPLE_RATE


def downsample_3x(pcm16: bytes) -> bytes:
    """Decimate 24 kHz int16 PCM to 8 kHz (every 3rd sample)."""
    count = len(pcm16) // 2
    if not count:
        return b""
    values = struct.unpack(f"<{count}h", pcm16)
    return struct.pack(f"<{(count + 2) // 3}h", *values[::3])


def tts_wav_to_mulaw(wav_bytes: bytes) -> bytes:
    """Convert a TTS WAV (any rate) to 8 kHz mu-law for the phone line."""
    pcm16, rate = wav_to_pcm16(wav_bytes)
    if not pcm16:
        return b""
    if rate > SAMPLE_RATE:
        pcm16 = downsample_3x(pcm16) if rate == 24000 else pcm16[:: (rate // SAMPLE_RATE)]
    return encode_mulaw(pcm16)


def base64_to_mulaw(payload: str) -> bytes:
    return decode_mulaw(base64.b64decode(payload or ""))


def mulaw_to_base64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")
