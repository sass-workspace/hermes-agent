"""Slack-specific send_message delivery regressions.

Salvaged from #47547 and adapted to the post-#41112 plugin layout: the legacy
``_send_slack`` helper moved to ``plugins/platforms/slack/adapter.py::
_standalone_send`` and text sends now route through ``_send_via_adapter``
(live adapter first, registry standalone fallback).
"""

import asyncio
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.config import Platform
from tools.send_message_tool import _send_to_platform


def _ensure_slack_mock(monkeypatch):
    """Install lightweight Slack modules when optional Slack deps are absent."""
    if "slack_bolt" in sys.modules and hasattr(sys.modules["slack_bolt"], "__file__"):
        return

    slack_bolt = MagicMock()
    slack_bolt.async_app.AsyncApp = MagicMock
    slack_bolt.adapter.socket_mode.async_handler.AsyncSocketModeHandler = MagicMock

    slack_sdk = MagicMock()
    slack_sdk.web.async_client.AsyncWebClient = MagicMock

    for name, mod in [
        ("slack_bolt", slack_bolt),
        ("slack_bolt.async_app", slack_bolt.async_app),
        ("slack_bolt.adapter", slack_bolt.adapter),
        ("slack_bolt.adapter.socket_mode", slack_bolt.adapter.socket_mode),
        ("slack_bolt.adapter.socket_mode.async_handler", slack_bolt.adapter.socket_mode.async_handler),
        ("slack_sdk", slack_sdk),
        ("slack_sdk.web", slack_sdk.web),
        ("slack_sdk.web.async_client", slack_sdk.web.async_client),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


def test_slack_send_to_platform_routes_through_send_via_adapter(monkeypatch):
    """Slack text sends go through _send_via_adapter (live adapter first)."""
    _ensure_slack_mock(monkeypatch)

    live_send = AsyncMock(return_value={"success": True, "message_id": "live-ts"})

    with patch("tools.send_message_tool._send_via_adapter", live_send):
        result = asyncio.run(
            _send_to_platform(
                Platform.SLACK,
                SimpleNamespace(enabled=True, token="bad-token,good-token", extra={}),
                "C123",
                "**hello** from Hermes",
                thread_id="171.1",
            )
        )

    assert result == {"success": True, "message_id": "live-ts"}
    live_send.assert_awaited_once()
    call = live_send.await_args
    assert call.args[0] == Platform.SLACK
    assert call.args[2] == "C123"
    assert call.kwargs["thread_id"] == "171.1"


class _SlackResponse:
    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class _SlackPostContext:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _SlackSession:
    """Fake aiohttp session whose good-token posts succeed."""

    def __init__(self):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers, json, **kwargs):
        token = headers["Authorization"].removeprefix("Bearer ")
        self.calls.append((token, json))
        if token == "good-token":
            payload = {"ok": True, "ts": "171.123"}
        else:
            payload = {"ok": False, "error": "invalid_auth"}
        return _SlackPostContext(_SlackResponse(payload))


@pytest.fixture
def _standalone_send(monkeypatch):
    _ensure_slack_mock(monkeypatch)
    from plugins.platforms.slack import adapter as slack_adapter

    return slack_adapter._standalone_send


def test_standalone_send_stops_on_non_token_error(monkeypatch, _standalone_send):
    """Terminal errors (not token-scoped) must not burn the remaining tokens."""

    class _FatalSession(_SlackSession):
        def post(self, url, *, headers, json, **kwargs):
            token = headers["Authorization"].removeprefix("Bearer ")
            self.calls.append((token, json))
            return _SlackPostContext(
                _SlackResponse({"ok": False, "error": "msg_too_long"})
            )

    fake_session = _FatalSession()
    monkeypatch.setattr(
        "aiohttp.ClientSession", lambda *args, **kwargs: fake_session
    )

    pconfig = SimpleNamespace(enabled=True, token="tok-a,tok-b", extra={})
    result = asyncio.run(_standalone_send(pconfig, "C123", "hello"))

    assert result == {"error": "Slack API error: msg_too_long"}
    assert len(fake_session.calls) == 1


# ---------------------------------------------------------------------------
# Block Kit parity for the standalone lane (2026-08-27: a `hermes cron run`
# brief arrived as flat mrkdwn while the ticker's arrived with the layout).
# ---------------------------------------------------------------------------

_BRIEF_MD = "## Täglicher Exception-Brief · 27.08.2026\n\n**Turbogrün · Matien**\n\n" + "\n".join(
    f"- 🟡 **Freigabe nötig** · [TUR-{i}](https://elbdev.atlassian.net/browse/TUR-{i}) — Feedback prüfen."
    for i in range(6)
)


def test_standalone_send_renders_block_kit_when_rich_blocks_enabled(monkeypatch, _standalone_send):
    fake_session = _SlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: fake_session)
    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={"rich_blocks": True})
    result = asyncio.run(_standalone_send(pconfig, "C123", _BRIEF_MD, thread_id="1.2"))
    assert result["success"] is True
    assert len(fake_session.calls) == 1
    body = fake_session.calls[0][1]
    types = [b["type"] for b in body["blocks"]]
    assert types[0] == "header" and "rich_text" in types
    assert body["thread_ts"] == "1.2"
    # the text is the notification fallback: words, no heading marks
    assert "##" not in body["text"] and "Exception-Brief" in body["text"]


def test_standalone_send_without_rich_blocks_stays_plain(monkeypatch, _standalone_send):
    fake_session = _SlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: fake_session)
    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={})
    asyncio.run(_standalone_send(pconfig, "C123", _BRIEF_MD))
    assert "blocks" not in fake_session.calls[0][1]


def test_standalone_send_partitions_over_budget_into_follow_ups(monkeypatch, _standalone_send):
    from plugins.platforms.slack import block_kit
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1200)
    fake_session = _SlackSession()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: fake_session)
    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={"rich_blocks": True})
    long_md = "\n\n".join(
        f"**Kunde {i}**\n\n- 🟡 **Freigabe** · [TUR-{i}](https://elbdev.atlassian.net/browse/TUR-{i}) — " + "warum " * 20
        for i in range(8)
    )
    result = asyncio.run(_standalone_send(pconfig, "C123", long_md, thread_id="1.2"))
    assert result["success"] is True
    calls = fake_session.calls
    assert len(calls) >= 3
    assert all("blocks" in body and body["thread_ts"] == "1.2" for _, body in calls)
    assert result["message_id"] == "171.123"  # the first post's ts


def test_standalone_send_size_rejection_retries_flat(monkeypatch, _standalone_send):
    class _RejectBlocks(_SlackSession):
        def post(self, url, *, headers, json, **kwargs):
            self.calls.append((headers["Authorization"], json))
            if "blocks" in json:
                return _SlackPostContext(_SlackResponse({"ok": False, "error": "msg_blocks_too_long"}))
            return _SlackPostContext(_SlackResponse({"ok": True, "ts": "171.123"}))

    fake_session = _RejectBlocks()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: fake_session)
    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={"rich_blocks": True})
    result = asyncio.run(_standalone_send(pconfig, "C123", _BRIEF_MD))
    assert result["success"] is True
    assert len(fake_session.calls) == 2
    assert "blocks" not in fake_session.calls[1][1]
    assert "TUR-1" in fake_session.calls[1][1]["text"]  # the full formatted body, not the notification


def test_standalone_rejected_follow_up_retries_flat_with_full_text(monkeypatch, _standalone_send):
    from plugins.platforms.slack import block_kit
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1200)

    class _RejectSecond(_SlackSession):
        def post(self, url, *, headers, json, **kwargs):
            self.calls.append((headers["Authorization"], json))
            if len(self.calls) == 2 and "blocks" in json:
                return _SlackPostContext(_SlackResponse({"ok": False, "error": "msg_blocks_too_long"}))
            return _SlackPostContext(_SlackResponse({"ok": True, "ts": "171.123"}))

    fake_session = _RejectSecond()
    monkeypatch.setattr("aiohttp.ClientSession", lambda *args, **kwargs: fake_session)
    pconfig = SimpleNamespace(enabled=True, token="good-token", extra={"rich_blocks": True})
    long_md = "\n\n".join(
        f"**Kunde {i}**\n\n- 🟡 **Freigabe** · [TUR-{i}](https://elbdev.atlassian.net/browse/TUR-{i}) — " + "warum " * 20
        for i in range(8)
    )
    result = asyncio.run(_standalone_send(pconfig, "C123", long_md))
    assert result["success"] is True
    rejected, retry = fake_session.calls[1][1], fake_session.calls[2][1]
    assert "blocks" in rejected and "blocks" not in retry
    assert len(retry["text"]) > len(rejected["text"]) and "TUR-" in retry["text"]
