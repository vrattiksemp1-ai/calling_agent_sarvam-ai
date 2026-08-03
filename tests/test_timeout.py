"""Timeout handling tests for provider clients.

httpx.MockTransport bypasses the real timeout machinery, so these tests
spin up a local TCP server that accepts connections but never responds.
"""

import asyncio

import pytest

from backend.errors import ProviderUnavailableError
from backend.providers.llm_client import LlmClient
from backend.providers.sarvam_client import SarvamClient
from tests.conftest import make_settings


def _silence_sleep(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr("backend.providers.sarvam_client.asyncio.sleep", no_sleep)
    monkeypatch.setattr("backend.providers.llm_client.asyncio.sleep", no_sleep)


async def _start_hanging_server():
    """Start a server that accepts connections and never responds."""
    tasks = set()

    async def handler(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                pass
        finally:
            tasks.discard(task)
            try:
                writer.close()
            except Exception:
                pass

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    return server, port, tasks


async def _stop_hanging_server(server, tasks):
    server.close()
    for task in list(tasks):
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    await server.wait_closed()


@pytest.mark.asyncio
async def test_stt_timeout_raises_provider_unavailable(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    server, port, tasks = await _start_hanging_server()
    try:
        settings = make_settings(
            tmp_path,
            sarvam_base_url=f"http://127.0.0.1:{port}",
            sarvam_request_timeout=0.2,
        )
        client = SarvamClient(settings)
        path = tmp_path / "a.wav"
        path.write_bytes(b"data")
        with pytest.raises(ProviderUnavailableError):
            await client.transcribe(str(path), 500)
        await client.aclose()
    finally:
        await _stop_hanging_server(server, tasks)


@pytest.mark.asyncio
async def test_llm_timeout_raises_provider_unavailable(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    server, port, tasks = await _start_hanging_server()
    try:
        settings = make_settings(
            tmp_path,
            llm_provider="openai-compatible",
            llm_base_url=f"http://127.0.0.1:{port}",
            llm_timeout=0.2,
        )
        client = LlmClient(settings)
        with pytest.raises(ProviderUnavailableError):
            await client.generate([{"role": "user", "content": "hi"}])
        await client.aclose()
    finally:
        await _stop_hanging_server(server, tasks)


@pytest.mark.asyncio
async def test_health_check_timeout_returns_error_status(monkeypatch, tmp_path):
    _silence_sleep(monkeypatch)
    server, port, tasks = await _start_hanging_server()
    try:
        settings = make_settings(
            tmp_path,
            sarvam_base_url=f"http://127.0.0.1:{port}",
            sarvam_request_timeout=0.2,
            sarvam_api_key="test-key",
        )
        client = SarvamClient(settings)
        status = await client.health_check()
        await client.aclose()
        assert status.status == "error"
        assert status.latency_ms is not None
    finally:
        await _stop_hanging_server(server, tasks)
