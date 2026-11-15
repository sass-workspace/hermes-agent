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
    # conversations.info: an ordinary channel unless a test says otherwise (the click's conversation type).
    client.conversations_info = AsyncMock(return_value={"ok": True, "channel": {"is_mpim": False}})

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
        # No metadata: native admission refuses a Slack event that carries any.
        assert not event.metadata
        from gateway.session_envelope import _native_metadata
        assert _native_metadata(event) == {}
        # The card was updated in place with the actor (display name) before
        # dispatch.
        update_kwargs = client.chat_update.await_args.kwargs
        assert update_kwargs["ts"] == "111.222"
        marks = [b for b in update_kwargs["blocks"] if b["type"] == "context"]
        assert any("Freigeben" in str(b) and "Matien" in str(b) for b in marks)
        assert not any("matien\"" in str(b).lower() for b in marks)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reply_in_thread, expected_thread", [(True, "111.222"), (False, None)])
    async def test_a_top_level_card_joins_the_session_a_typed_reply_would_join(
            self, reply_in_thread, expected_thread):
        """Under ``reply_in_thread: false`` a top-level message lives in the whole-channel
        session: a click on a top-level card must not fork a session keyed on the card's ts
        (it would also slip past the busy-session queue of the active channel session)."""
        adapter, _client = _make_adapter()
        adapter.config.extra["reply_in_thread"] = reply_in_thread
        body, action = _click_body(thread_ts=None)
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)

        event = adapter.handle_message.await_args.args[0]
        assert event.source.thread_id == expected_thread
        assert event.reply_to_message_id is None

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


class TestAgentReplyButtonGates:
    """A click obeys the typed message path's channel gates and DM keying (Codex, M8). The
    conversation type comes from Slack's ``is_mpim`` (conversations.info), never from a name."""

    @staticmethod
    def _info(client, *, ok=True, is_mpim=False):
        client.conversations_info = AsyncMock(return_value={"ok": ok, "channel": {"is_mpim": is_mpim}})

    @pytest.mark.asyncio
    async def test_click_in_an_ignored_channel_dispatches_nothing(self, monkeypatch):
        adapter, client = _make_adapter()
        self._info(client)
        monkeypatch.setattr(adapter, "_slack_ignored_channels", lambda: {"C1"})
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()
        client.chat_update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_click_outside_allowed_channels_dispatches_nothing(self, monkeypatch):
        adapter, client = _make_adapter()
        self._info(client)
        monkeypatch.setattr(adapter, "_slack_allowed_channels", lambda: {"C9"})
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("channel_id, is_mpim", [("D1", False), ("G1", True)])
    async def test_click_in_a_dm_with_dms_disabled_dispatches_nothing(self, monkeypatch, channel_id, is_mpim):
        adapter, client = _make_adapter()
        self._info(client, is_mpim=is_mpim)
        monkeypatch.setattr(adapter, "_slack_disable_dms", lambda: True)
        body, action = _click_body()
        body["channel"] = {"id": channel_id}
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_group_dm_click_keys_the_shared_dm_session_and_checks_dm_rules(self):
        adapter, client = _make_adapter()
        self._info(client, is_mpim=True)
        body, action = _click_body()
        body["channel"] = {"id": "G1"}
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_type == "dm" and event.source.shared_session is True
        assert adapter._is_interactive_user_authorized.call_args.kwargs["chat_type"] == "dm"

    @pytest.mark.asyncio
    async def test_group_dm_click_refused_by_dm_rules_dispatches_nothing(self):
        adapter, client = _make_adapter()
        self._info(client, is_mpim=True)
        adapter._is_interactive_user_authorized = MagicMock(
            side_effect=lambda *a, **k: k.get("chat_type") != "dm")
        body, action = _click_body()
        body["channel"] = {"id": "G1"}
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_channel_named_like_a_group_dm_stays_a_group(self):
        adapter, client = _make_adapter()
        self._info(client, is_mpim=False)
        body, action = _click_body()
        body["channel"] = {"id": "C7", "name": "mpdm-ops"}
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        event = adapter.handle_message.await_args.args[0]
        assert event.source.chat_type == "group" and not event.source.shared_session

    @pytest.mark.asyncio
    async def test_unknown_conversation_type_dispatches_nothing(self):
        adapter, client = _make_adapter()
        self._info(client, ok=False)
        body, action = _click_body()
        await adapter._handle_agent_reply_action(AsyncMock(), body, action)
        adapter.handle_message.assert_not_awaited()
