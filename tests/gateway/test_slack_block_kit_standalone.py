"""Block Kit in the standalone send lane (``_standalone_send``).

A cron job run from the CLI (``hermes cron run``) delivers through this lane, the same job from
the gateway's ticker through the live adapter: both must arrive with the same layout — same
renderer, same partitioning, same text fallback — and a rejected layout must still deliver.
"""
import asyncio
from types import SimpleNamespace

import pytest

from plugins.platforms.slack import adapter as slack_module
from plugins.platforms.slack import block_kit

RICH_MD = "# Tagesbrief\n\n- **Turbogrün** · wartet auf Freigabe\n\n---\n\nbody text"
LONG_MD = "\n\n".join(
    f"**Kunde {i}**\n\n- 🟡 **Freigabe** · [TUR-{i}](https://example.test/browse/TUR-{i}) — " + "warum " * 20
    for i in range(8))


class _Recorder:
    """Records every ``chat.postMessage`` payload; answers from ``script`` (default: ok)."""

    def __init__(self):
        self.payloads, self.script = [], []

    async def __call__(self, session, tok, method, payload, req_kw):
        assert method == "chat.postMessage"
        self.payloads.append(dict(payload))
        return self.script.pop(0) if self.script else {"ok": True, "ts": f"1.{len(self.payloads)}"}


@pytest.fixture
def posts(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr(slack_module, "_slack_json_post", recorder)
    return recorder


def _send(extra, message, token="xoxb-fake", **kw):
    pconfig = SimpleNamespace(enabled=True, token=token, extra=extra)
    return asyncio.run(slack_module._standalone_send(pconfig, "C1", message, **kw))


def test_rich_blocks_off_stays_plain_text(posts):
    assert _send({}, RICH_MD)["success"] is True
    (payload,) = posts.payloads
    assert "blocks" not in payload
    assert payload["text"] == slack_module._standalone_format_mrkdwn(RICH_MD)


def test_rich_blocks_on_sends_the_layout_with_a_readable_text_fallback(posts):
    assert _send({"rich_blocks": True}, RICH_MD, thread_id="171.1")["success"] is True
    (payload,) = posts.payloads
    assert [b["type"] for b in payload["blocks"]][0] == "header"
    assert payload["thread_ts"] == "171.1"
    # ``text`` is the notification / screen-reader fallback: words, no scaffolding.
    assert "Tagesbrief" in payload["text"] and "#" not in payload["text"] and "---" not in payload["text"]


def test_a_layout_past_one_payload_continues_in_follow_up_posts(posts, monkeypatch):
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
    assert _send({"rich_blocks": True}, LONG_MD, thread_id="171.1")["success"] is True
    assert len(posts.payloads) > 1
    assert all(p.get("blocks") for p in posts.payloads)
    assert all(p.get("thread_ts") == "171.1" for p in posts.payloads), "follow-ups stay in context"
    assert all("reply_broadcast" not in p for p in posts.payloads)


def test_a_rejected_layout_is_delivered_flat_with_the_whole_body(posts):
    posts.script[:] = [{"ok": False, "error": "invalid_blocks"}, {"ok": True, "ts": "1.2"}]
    result = _send({"rich_blocks": True}, RICH_MD)
    assert result["success"] is True and result["message_id"] == "1.2"
    first, retry = posts.payloads
    assert first["blocks"] and "blocks" not in retry
    assert retry["text"] == slack_module._standalone_format_mrkdwn(RICH_MD)


def test_a_rejected_first_post_sends_no_follow_ups(posts, monkeypatch):
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
    posts.script[:] = [{"ok": False, "error": "msg_blocks_too_long"}, {"ok": True, "ts": "1.2"}]
    assert _send({"rich_blocks": True}, LONG_MD)["success"] is True
    assert len(posts.payloads) == 2, "the layout is gone: the flat body is the whole delivery"


def test_a_failed_follow_up_never_fails_the_delivered_first_message(posts, monkeypatch):
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
    posts.script[:] = [{"ok": True, "ts": "1.1"}, {"ok": False, "error": "ratelimited"}]
    result = _send({"rich_blocks": True}, LONG_MD)
    assert result["success"] is True and result["message_id"] == "1.1"


def test_follow_ups_carry_the_profiles_unfurl_settings(posts, monkeypatch):
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
    assert _send({"rich_blocks": True, "unfurl_links": False}, LONG_MD)["success"] is True
    assert len(posts.payloads) > 1
    assert all(p.get("unfurl_links") is False for p in posts.payloads), "every post, not only the first"


def test_a_wrong_workspace_token_does_not_cost_the_next_token_its_layout(posts, monkeypatch):
    """Token A rejects the layout, then its flat retry answers channel_not_found (wrong workspace):
    token B must still get the block payload AND the follow-ups."""
    monkeypatch.setattr(block_kit, "BLOCK_PAYLOAD_BUDGET", 1500)
    posts.script[:] = [
        {"ok": False, "error": "invalid_blocks"},      # A: blocks
        {"ok": False, "error": "channel_not_found"},   # A: flat retry → next token
        {"ok": True, "ts": "2.1"},                     # B: blocks
    ]
    result = _send({"rich_blocks": True}, LONG_MD, token="xoxb-a,xoxb-b")
    assert result["success"] is True and result["message_id"] == "2.1"
    assert posts.payloads[2].get("blocks"), "token B was sent the flattened payload"
    assert len(posts.payloads) > 3, "token B's follow-ups were dropped"

