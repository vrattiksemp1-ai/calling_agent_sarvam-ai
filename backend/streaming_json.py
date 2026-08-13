"""Incremental extraction of a JSON ``assistant_message`` string.

The parser emits only decoded characters that are known to belong to the
top-level response field. The complete model output must still be validated by
``parse_structured_response`` before any conversation state is applied.
"""

from __future__ import annotations

import json
import re


_ASSISTANT_START = re.compile(r'"assistant_message"\s*:\s*"')
_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


class AssistantMessageStreamParser:
    """Decode one JSON string safely across arbitrary network chunk splits."""

    def __init__(self) -> None:
        self._search = ""
        self._value = ""
        self._started = False
        self._complete = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def complete(self) -> bool:
        return self._complete

    def feed(self, fragment: str) -> str:
        if self._complete or not fragment:
            return ""
        if not self._started:
            self._search += fragment
            match = _ASSISTANT_START.search(self._search)
            if match is None:
                # Keep enough overlap for a key split across SSE events while
                # preventing unbounded growth on malformed output.
                self._search = self._search[-256:]
                return ""
            self._started = True
            self._value = self._search[match.end() :]
            self._search = ""
        else:
            self._value += fragment
        return self._drain()

    def _drain(self) -> str:
        output: list[str] = []
        index = 0
        while index < len(self._value):
            char = self._value[index]
            if char == '"':
                self._complete = True
                index += 1
                break
            if char != "\\":
                output.append(char)
                index += 1
                continue

            if index + 1 >= len(self._value):
                break
            escape = self._value[index + 1]
            if escape in _SIMPLE_ESCAPES:
                output.append(_SIMPLE_ESCAPES[escape])
                index += 2
                continue
            if escape != "u":
                raise ValueError(f"Invalid JSON string escape: \\{escape}")
            if index + 6 > len(self._value):
                break
            first = self._value[index : index + 6]
            try:
                codepoint = int(first[2:], 16)
            except ValueError as exc:
                raise ValueError("Invalid JSON Unicode escape") from exc
            sequence = first
            consumed = 6
            if 0xD800 <= codepoint <= 0xDBFF:
                if index + 12 > len(self._value):
                    break
                second = self._value[index + 6 : index + 12]
                if not second.startswith("\\u"):
                    raise ValueError("Unpaired JSON high surrogate")
                sequence += second
                consumed = 12
            output.append(json.loads(f'"{sequence}"'))
            index += consumed

        self._value = self._value[index:]
        return "".join(output)

    def finish(self) -> None:
        """Raise when the model stream ended before closing the JSON string."""
        if not self._started:
            raise ValueError("assistant_message was not found in streamed JSON")
        if not self._complete:
            raise ValueError("assistant_message JSON string was incomplete")


class StreamingTextChunker:
    """Create natural TTS chunks without waiting for a complete response."""

    def __init__(self, minimum_chars: int = 24, maximum_chars: int = 120) -> None:
        self._buffer = ""
        self._minimum = minimum_chars
        self._maximum = maximum_chars

    def feed(self, text: str) -> list[str]:
        self._buffer += text
        chunks: list[str] = []
        while len(self._buffer) >= self._minimum:
            limit = min(len(self._buffer), self._maximum)
            split = max(
                self._buffer.rfind(mark, 0, limit + 1)
                for mark in (".", "?", "!", "।", ",", ";", "\n", " ")
            )
            if split < self._minimum and len(self._buffer) < self._maximum:
                break
            if split < self._minimum:
                split = self._maximum - 1
            chunk = self._buffer[: split + 1].strip()
            self._buffer = self._buffer[split + 1 :]
            if chunk:
                chunks.append(chunk)
        return chunks

    def flush(self) -> str:
        chunk, self._buffer = self._buffer.strip(), ""
        return chunk


class FirstSpeechChunkBuffer:
    """Emit the first complete sentence from an incremental text stream.

    A bounded word-boundary fallback prevents a punctuation-free model reply
    from delaying speech indefinitely.
    """

    _TERMINATORS = frozenset((".", "?", "!", "।", "॥", "\n"))

    def __init__(self, minimum_chars: int = 8, maximum_chars: int = 120) -> None:
        self._buffer = ""
        self._minimum = minimum_chars
        self._maximum = maximum_chars
        self._emitted = False

    def feed(self, text: str) -> str:
        if self._emitted or not text:
            return ""
        self._buffer += text

        for index, char in enumerate(self._buffer):
            end = index + 1
            if end >= self._minimum and char in self._TERMINATORS:
                return self._emit(end)

        if len(self._buffer) < self._maximum:
            return ""
        split = max(
            self._buffer.rfind(mark, 0, self._maximum + 1)
            for mark in (" ", "\t")
        )
        end = split if split >= self._minimum else self._maximum
        return self._emit(end)

    def _emit(self, end: int) -> str:
        chunk = self._buffer[:end].strip()
        if not chunk:
            return ""
        self._emitted = True
        self._buffer = self._buffer[end:]
        return chunk
