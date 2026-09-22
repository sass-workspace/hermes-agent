"""The Agent Sessions API is chosen by SDK version, but entitlement is the APP's: when Slack
rejects ``agents.sessions.*`` for this app, the status line and thread title must not vanish
silently — the adapter falls back to the legacy ``assistant.threads.*`` methods, once, and stays
there for the life of the process."""
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter as slack_module
from plugins.platforms.slack.adapter import SlackAdapter


class _SlackApiError(Exception):
    def __init__(self, code):
        super().__init__(f"The request to the Slack API failed. (error: {code})")
        self.response = {"ok": False, "error": code}


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setattr(slack_module, "_AGENT_SESSIONS_SUPPORTED", True)
    monkeypatch.setattr(slack_module, "_AGENT_SESSIONS_REJECTED", False)
    a = SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake"))
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.agents_sessions_setStatus = AsyncMock()
    a._app.client.agents_sessions_rename = AsyncMock()
    a._app.client.assistant_threads_setStatus = AsyncMock()
    a._app.client.assistant_threads_setTitle = AsyncMock()
    return a


@pytest.mark.asyncio
async def test_an_entitled_app_never_touches_the_legacy_methods(adapter):
    await adapter.send_typing("C1", metadata={"thread_id": "1.0"})
    adapter._app.client.agents_sessions_setStatus.assert_awaited_once()
    adapter._app.client.assistant_threads_setStatus.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["not_allowed", "missing_scope", "unknown_method", "feature_not_enabled"])
async def test_a_rejected_status_call_falls_back_to_legacy_and_latches(adapter, caplog, code):
    client = adapter._app.client
    client.agents_sessions_setStatus = AsyncMock(side_effect=_SlackApiError(code))
    with caplog.at_level(logging.WARNING, logger=slack_module.logger.name):
        await adapter.send_typing("C1", metadata={"thread_id": "1.0"})
        await adapter.send_typing("C1", metadata={"thread_id": "2.0"})
    # first call: modern tried once, legacy took over; second call: straight to legacy
    assert client.agents_sessions_setStatus.await_count == 1
    assert client.assistant_threads_setStatus.await_count == 2
    assert client.assistant_threads_setStatus.await_args.kwargs["status"] == "is thinking..."
    assert sum("not entitled" in r.getMessage() for r in caplog.records) == 1
    assert slack_module._AGENT_SESSIONS_REJECTED is True


@pytest.mark.asyncio
async def test_the_latch_also_routes_thread_titles_to_legacy(adapter, monkeypatch):
    monkeypatch.setattr(slack_module, "_AGENT_SESSIONS_REJECTED", True)
    client = adapter._app.client
    assert slack_module._session_title_method(client) is client.assistant_threads_setTitle
    assert slack_module._session_status_method(client) is client.assistant_threads_setStatus


@pytest.mark.asyncio
async def test_a_transport_failure_is_not_an_entitlement_verdict(adapter):
    """A blip must not demote every later call to the deprecated API."""
    client = adapter._app.client
    client.agents_sessions_setStatus = AsyncMock(side_effect=_SlackApiError("ratelimited"))
    await adapter.send_typing("C1", metadata={"thread_id": "1.0"})
    client.assistant_threads_setStatus.assert_not_awaited()
    assert slack_module._AGENT_SESSIONS_REJECTED is False
    client.agents_sessions_setStatus = AsyncMock()
    await adapter.send_typing("C1", metadata={"thread_id": "2.0"})
    client.agents_sessions_setStatus.assert_awaited_once()
