"""Integration tests: SlackAdapter wiring of Block Kit into send paths.

Verifies the opt-in behaviour contract:
  * rich_blocks off (default)  => no ``blocks`` kwarg, plain ``text`` only
  * rich_blocks on             => ``blocks`` present AND ``text`` fallback set
  * edit_message: blocks only on finalize (streaming edits stay plain)
  * multi-chunk (>39k) messages fall back to plain text
"""

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter as slack_module
from plugins.platforms.slack.adapter import SlackAdapter


def _make_adapter(extra=None):
    config = PlatformConfig(enabled=True, token="xoxb-fake", extra=extra or {})
    a = SlackAdapter(config)
    a._app = MagicMock()
    client = AsyncMock()
    client.chat_postMessage = AsyncMock(return_value={"ts": "111.222"})
    client.chat_update = AsyncMock(return_value={"ts": "111.222"})
    a._get_client = MagicMock(return_value=client)
    a.stop_typing = AsyncMock()
    a._running = True
    return a, client


LONG_MD = "\n\n".join(
    f"**Kunde {i}**\n\n- 🟡 **Freigabe** · [TUR-{i}](https://elbdev.atlassian.net/browse/TUR-{i}) — " + "warum " * 20
    for i in range(8)
)
RICH_MD = "# Title\n\n- a\n  - nested\n\n---\n\nbody text"
RICH_TABLE_MD = (
    "| Item | Status | Note |\n"
    "|---|---:|---|\n"
    "| Hermes | ok | table |"
)


class SlackRejectedBlocks(Exception):
    def __init__(self, error="invalid_blocks"):
        super().__init__(f"Slack API rejected blocks: {error}")
        self.response = {"error": error}


def _slack_connection_key():
    from aiohttp.client_reqrep import ConnectionKey

    return ConnectionKey(
        host="slack.com",
        port=443,
        is_ssl=True,
        ssl=True,
        proxy=None,
        proxy_auth=None,
        proxy_headers_hash=None,
    )


class TestSendMessageBlocks:
    @pytest.mark.asyncio
    async def test_disabled_by_default_no_blocks(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]  # plain text still sent


    @pytest.mark.asyncio
    async def test_over_count_payload_is_partitioned_not_degraded(self):
        # 60 dividers used to make the renderer decline (flat text); now the
        # payload is split across consecutive posts, every one with blocks.
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", "\n\n".join(["---"] * 60))
        calls = client.chat_postMessage.await_args_list
        assert len(calls) >= 2
        assert all("blocks" in c.kwargs and len(c.kwargs["blocks"]) <= 50 for c in calls)
        # top-level first post -> top-level follow-ups (never buried in a thread)
        assert all("thread_ts" not in c.kwargs for c in calls)

    @pytest.mark.asyncio
    async def test_over_budget_follow_ups_inherit_thread_and_skip_broadcast(self, monkeypatch):
        from plugins.platforms.slack import block_kit
        monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
        adapter, client = _make_adapter({"rich_blocks": True, "feedback_buttons": True, "reply_broadcast": True})
        md = LONG_MD
        await adapter.send("C1", md, metadata={"thread_id": "999.111"})
        calls = client.chat_postMessage.await_args_list
        assert len(calls) >= 3
        assert all(c.kwargs["thread_ts"] == "999.111" for c in calls)
        assert calls[0].kwargs.get("reply_broadcast") is True
        assert all("reply_broadcast" not in c.kwargs for c in calls[1:])
        feedback = [c for c in calls if c.kwargs["blocks"][-1]["type"] == "context_actions"]
        assert len(feedback) == 1 and feedback[0] is calls[-1]
        assert all(c.kwargs["text"] and ":::" not in c.kwargs["text"] for c in calls[1:])

    @pytest.mark.asyncio
    async def test_size_rejection_after_partition_retries_flat_and_logs(self, caplog):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_postMessage = AsyncMock(
            side_effect=[SlackRejectedBlocks("msg_blocks_too_long"), {"ts": "111.222"}]
        )
        with caplog.at_level("WARNING"):
            await adapter.send("C1", RICH_MD)
        assert client.chat_postMessage.await_count == 2
        assert "blocks" not in client.chat_postMessage.await_args.kwargs
        assert "BLOCK_PAYLOAD_BUDGET" in caplog.text


    @pytest.mark.asyncio
    async def test_card_directive_renders_card_and_clean_text_fallback(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        md = (
            ":::card\n"
            "title: TUR-445 · Antwort\n"
            "Kurzer Entwurf.\n"
            "button: [Freigeben](reply: ja, TUR-445 freigeben)\n"
            ":::\n"
            "-# Quelle: Asana Story"
        )
        await adapter.send("C1", md)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert [b["type"] for b in kwargs["blocks"]] == ["card", "context"]
        assert kwargs["blocks"][0]["actions"][0]["action_id"] == "hermes_agent_reply"
        # The text fallback (notifications, old clients) sheds the scaffolding
        # and never carries the reply instruction.
        assert ":::" not in kwargs["text"]
        assert "title:" not in kwargs["text"]
        assert "freigeben" not in kwargs["text"]
        assert "TUR-445" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_directives_without_rich_blocks_send_clean_plain_text(self):
        adapter, client = _make_adapter()  # rich_blocks off
        await adapter.send("C1", ":::card\ntitle: T\nBody.\n:::")
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert ":::" not in kwargs["text"]
        assert "Body." in kwargs["text"]

    @pytest.mark.asyncio
    async def test_feedback_buttons_opt_in_appended_to_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True, "feedback_buttons": True})

        await adapter.send("C1", "final answer")

        blocks = client.chat_postMessage.await_args.kwargs["blocks"]
        feedback = blocks[-1]
        assert feedback["type"] == "context_actions"
        assert feedback["elements"][0]["type"] == "feedback_buttons"
        assert feedback["elements"][0]["action_id"] == "hermes_feedback"


class TestEditMessageBlocks:
    @pytest.mark.asyncio
    async def test_intermediate_edit_no_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=False)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"]

    @pytest.mark.asyncio
    async def test_finalize_edit_gets_blocks(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" in kwargs and kwargs["blocks"]
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_finalize_edit_posts_overflow_as_follow_ups(self, monkeypatch):
        from plugins.platforms.slack import block_kit
        monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", LONG_MD, finalize=True, metadata={"thread_id": "111.222"})
        client.chat_update.assert_awaited_once()
        first = client.chat_update.await_args.kwargs["blocks"]
        follow = client.chat_postMessage.await_args_list
        assert len(follow) >= 1
        assert all(c.kwargs["thread_ts"] == "111.222" for c in follow)
        assert all(c.kwargs["blocks"] != first for c in follow)  # overflow only

    @pytest.mark.asyncio
    async def test_block_rejection_retries_edit_without_blocks_using_workspace_client(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        client.chat_update = AsyncMock(
            side_effect=[SlackRejectedBlocks("invalid_blocks"), {"ts": "111.222"}]
        )

        result = await adapter.edit_message(
            "C1",
            "111.222",
            RICH_TABLE_MD,
            finalize=True,
            metadata={"team_id": "T_SECONDARY"},
        )

        assert result.success is True
        assert adapter._get_client.call_args_list == [
            call("C1", team_id="T_SECONDARY"),
            call("C1", team_id="T_SECONDARY"),
        ]
        assert client.chat_update.await_count == 2
        first = client.chat_update.await_args_list[0].kwargs
        second = client.chat_update.await_args_list[1].kwargs
        assert "blocks" in first and first["blocks"]
        assert second["blocks"] == []
        assert second["text"]

    @pytest.mark.asyncio
    async def test_timeout_error_on_edit_is_retryable_transient(self):
        adapter, client = _make_adapter()
        client.chat_update = AsyncMock(side_effect=TimeoutError("timed out"))

        result = await adapter.edit_message("C1", "111.222", RICH_MD, finalize=True)

        assert result.success is False
        assert result.retryable is True
        assert result.error_kind == "transient"


# ---------------------------------------------------------------------------
# markdown_blocks mode — Slack's native ``markdown`` Block Kit block (#8552)
# ---------------------------------------------------------------------------


class TestMarkdownBlockMode:
    """Opt-in ``markdown_blocks`` renders raw standard markdown via Slack's
    native ``markdown`` block, keeping the mrkdwn ``text`` fallback."""

    @pytest.mark.asyncio
    async def test_disabled_by_default(self):
        adapter, client = _make_adapter()
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs

    @pytest.mark.asyncio
    async def test_enabled_sends_markdown_block_with_raw_content(self):
        adapter, client = _make_adapter({"markdown_blocks": True})
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        blocks = kwargs["blocks"]
        assert blocks[0]["type"] == "markdown"
        # RAW standard markdown, not mrkdwn-converted — Slack translates it
        assert blocks[0]["text"] == RICH_TABLE_MD
        # mrkdwn fallback text is still present for notifications/search
        assert kwargs["text"]


    @pytest.mark.asyncio
    async def test_edit_finalize_uses_markdown_block(self):
        adapter, client = _make_adapter({"markdown_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_TABLE_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs["blocks"][0]["type"] == "markdown"
        assert kwargs["blocks"][0]["text"] == RICH_TABLE_MD


class TestNotificationFallback:
    """With blocks attached, ``text`` is the notification / screen-reader
    fallback rather than the body, so it carries readable prose instead of the
    whole converted message — whose first ~150 characters are layout markers
    and a column-padded table. Without blocks it IS the body and must stay
    byte-for-byte untouched.
    """

    @pytest.mark.asyncio
    async def test_blocks_present_text_is_cleaned_prose(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert kwargs["blocks"], "precondition: this content renders as blocks"
        text = kwargs["text"]
        assert text
        assert "|" not in text
        assert "```" not in text
        assert "---" not in text
        assert "Hermes" in text and "table" in text

    @pytest.mark.asyncio
    async def test_no_blocks_text_is_the_untouched_body(self):
        adapter, client = _make_adapter()  # rich_blocks off
        await adapter.send("C1", RICH_TABLE_MD)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"] == adapter.format_message(RICH_TABLE_MD)

    @pytest.mark.asyncio
    async def test_finalize_edit_also_gets_the_fallback(self):
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_TABLE_MD, finalize=True)
        kwargs = client.chat_update.await_args.kwargs
        assert kwargs["blocks"]
        assert "|" not in kwargs["text"]
        assert "Hermes" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_intermediate_edit_keeps_the_full_text(self):
        """Streaming flushes carry the body; only the final edit has blocks."""
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.edit_message("C1", "111.222", RICH_TABLE_MD, finalize=False)
        kwargs = client.chat_update.await_args.kwargs
        assert "blocks" not in kwargs
        assert kwargs["text"] == adapter.format_message(RICH_TABLE_MD)

    @pytest.mark.asyncio
    async def test_unsummarisable_content_keeps_the_converted_text(self):
        """Nothing readable survives -> never send an empty notification."""
        adapter, client = _make_adapter({"rich_blocks": True})
        fence_only = "```\nx = 1\n```"
        assert adapter.notification_text(fence_only) == ""
        await adapter.send("C1", fence_only)
        kwargs = client.chat_postMessage.await_args.kwargs
        assert kwargs["text"] == adapter.format_message(fence_only)




class TestFollowUpReviewFindings:
    @pytest.mark.asyncio
    async def test_rejected_follow_up_retries_with_its_full_text(self, monkeypatch):
        from plugins.platforms.slack import block_kit
        monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
        adapter, client = _make_adapter({"rich_blocks": True})
        calls = []

        async def post(**kwargs):
            calls.append(kwargs)
            if len(calls) == 2 and "blocks" in kwargs:
                raise SlackRejectedBlocks("msg_blocks_too_long")
            return {"ts": f"111.{len(calls)}"}

        client.chat_postMessage = AsyncMock(side_effect=post)
        await adapter.send("C1", LONG_MD)
        rejected = calls[1]
        retry = calls[2]
        assert "blocks" not in retry
        # the retry carries the group's whole content, not the 600-char notification
        assert len(retry["text"]) > len(rejected["text"])
        assert "TUR-" in retry["text"]

    @pytest.mark.asyncio
    async def test_editable_status_message_never_overflows(self, monkeypatch):
        from plugins.platforms.slack import block_kit
        monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
        adapter, client = _make_adapter({"rich_blocks": True})
        await adapter.send("C1", LONG_MD, metadata={"expect_edits": True})
        calls = client.chat_postMessage.await_args_list
        assert len(calls) == 1
        assert "blocks" not in calls[0].kwargs
