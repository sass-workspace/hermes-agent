"""Tests for the agent-authored reply button handler (``hermes_agent_reply``).

Contract under test (Phase 1b of the Block Kit directive feature): a click is
the authorized clicker's instruction dispatched into the thread's session via
the NORMAL inbound path (``handle_message``) — never a direct action — and a
double-click dispatches exactly once.
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from tests.gateway.test_slack_approval_buttons import _ensure_slack_mock

_ensure_slack_mock()

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter(authorized=True):
    config = PlatformConfig(enabled=True, token="xoxb-test-token")
    adapter = SlackAdapter(config)
    adapter._app = MagicMock()
    adapter._is_interactive_user_authorized = MagicMock(return_value=authorized)
    adapter._resolve_channel_name = AsyncMock(return_value="iris-ops")
    adapter._resolve_user_name = AsyncMock(return_value="Matien")
    adapter.handle_message = AsyncMock()
    client = AsyncMock()

    # Signature-aware double: the real _get_client REQUIRES a positional
    # chat_id (adapter.py) — a keyword-only call must fail here exactly as it
    # would in production, so the handler's call shape stays honest.
    def _get_client(chat_id, team_id=None):
        assert chat_id, "chat_id is required"
        return client

    adapter._get_client = MagicMock(side_effect=_get_client)
    return adapter, client


def _click_body(value="ja, TUR-445 freigeben", ts="111.222", thread_ts="100.000"):
    action = {
        "action_id": "hermes_agent_reply",
        "value": value,
        "action_ts": "999.111",
        "text": {"type": "plain_text", "text": "Freigeben"},
    }
    body = {
        "team": {"id": "T1"},
        "user": {"id": "U_MAT", "name": "matien"},
        "channel": {"id": "C1"},
        "message": {
            "ts": ts,
            "thread_ts": thread_ts,
            "text": "fallback",
            "blocks": [{"type": "card", "title": {"type": "mrkdwn", "text": "T"}}],
        },
    }
    return body, action


class TestAgentReplyButton:
    @pytest.mark.asyncio
    async def test_authorized_click_dispatches_instruction_into_thread_session(self):
        adapter, client = _make_adapter()
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)

        adapter.handle_message.assert_awaited_once()
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "ja, TUR-445 freigeben"
        assert event.source.thread_id == "100.000"
        assert event.source.chat_id == "C1"
        assert event.source.user_id == "U_MAT"
        # The dispatched event carries the RESOLVED display name, not the login.
        assert event.source.user_name == "Matien"
        # The card was updated in place with the actor (display name) before
        # dispatch.
        update_kwargs = client.chat_update.await_args.kwargs
        assert update_kwargs["ts"] == "111.222"
        marks = [b for b in update_kwargs["blocks"] if b["type"] == "context"]
        assert any("Freigeben" in str(b) and "Matien" in str(b) for b in marks)
        assert not any("matien\"" in str(b).lower() for b in marks)

    @pytest.mark.asyncio
    async def test_unauthorized_click_is_ignored(self):
        adapter, client = _make_adapter(authorized=False)
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()
        client.chat_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_double_click_dispatches_once(self):
        adapter, client = _make_adapter()
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        assert adapter.handle_message.await_count == 1
        assert client.chat_update.await_count == 1

    @pytest.mark.asyncio
    async def test_distinct_buttons_on_same_card_both_dispatch(self):
        adapter, client = _make_adapter()
        body, a1 = _click_body(value="ja, freigeben")
        _, a2 = _click_body(value="nein, ablehnen")
        await adapter._handle_agent_reply_action(AsyncMock(), body, a1)
        await adapter._handle_agent_reply_action(AsyncMock(), body, a2)
        assert adapter.handle_message.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_value_is_ignored(self):
        adapter, client = _make_adapter()
        body, action = _click_body(value="   ")
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_failed_card_update_still_dispatches(self):
        adapter, client = _make_adapter()
        client.chat_update = AsyncMock(side_effect=RuntimeError("update failed"))
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handler_registered_for_fixed_action_id(self):
        adapter, _ = _make_adapter()
        src = Path(_repo, "plugins", "platforms", "slack", "adapter.py").read_text()
        assert 'self._app.action("hermes_agent_reply")' in src
        assert 'self._app.action("hermes_agent_menu")' in src

    @pytest.mark.asyncio
    async def test_menu_selection_dispatches_selected_option_value(self):
        adapter, client = _make_adapter()
        body, action = _click_body(value="")
        action.pop("value", None)
        action.pop("text", None)
        action["action_id"] = "hermes_agent_menu"
        action["selected_option"] = {
            "text": {"type": "plain_text", "text": "In Prüfung"},
            "value": "TUR-445 in Prüfung setzen",
        }
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        event = adapter.handle_message.await_args.args[0]
        assert event.text == "TUR-445 in Prüfung setzen"
        update_kwargs = client.chat_update.await_args.kwargs
        assert any("In Prüfung" in str(b) for b in update_kwargs["blocks"])

    @pytest.mark.asyncio
    async def test_menu_same_option_double_select_dispatches_once(self):
        adapter, client = _make_adapter()
        body, action = _click_body(value="")
        action.pop("value", None)
        action["action_id"] = "hermes_agent_menu"
        action["selected_option"] = {
            "text": {"type": "plain_text", "text": "A"},
            "value": "tu a",
        }
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        assert adapter.handle_message.await_count == 1

    @pytest.mark.asyncio
    async def test_menu_distinct_options_both_dispatch(self):
        adapter, client = _make_adapter()
        body, a1 = _click_body(value="")
        a1.pop("value", None)
        a1["action_id"] = "hermes_agent_menu"
        a1["selected_option"] = {"text": {"type": "plain_text", "text": "A"}, "value": "tu a"}
        import copy

        a2 = copy.deepcopy(a1)
        a2["selected_option"]["value"] = "tu b"
        await adapter._handle_agent_reply_action(AsyncMock(), body, a1)
        await adapter._handle_agent_reply_action(AsyncMock(), body, a2)
        assert adapter.handle_message.await_count == 2

    @pytest.mark.asyncio
    async def test_unauthorized_menu_selection_ignored(self):
        adapter, client = _make_adapter(authorized=False)
        body, action = _click_body(value="")
        action.pop("value", None)
        action["action_id"] = "hermes_agent_menu"
        action["selected_option"] = {"text": {"type": "plain_text", "text": "A"}, "value": "tu a"}
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()
        client.chat_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_url_menu_option_dispatches_nothing(self):
        adapter, client = _make_adapter()
        body, action = _click_body(value="")
        action.pop("value", None)
        action["action_id"] = "hermes_agent_menu"
        action["selected_option"] = {
            "text": {"type": "plain_text", "text": "Jira öffnen"},
            "value": "url_0",
        }
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()
