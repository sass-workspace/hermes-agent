"""Forwarded/shared Slack messages are first-class agent input.

Slack's "Forward message" composer delivers the forwarded source message in
the ``attachments`` array with BOTH ``is_share`` and ``is_msg_unfurl`` set.
The adapter's unfurl-echo skip used to drop it wholesale, so the agent
received only the outer instruction (e.g. "Handle this request:") and had to
answer that the request was missing.

The fixtures mirror the raw payload shape Slack sends (fields trimmed to what
the adapter reads). The suite pins:

  * outer instruction + forwarded source text both reach the agent
  * source author / channel / ts / permalink provenance stays separate from
    the forwarding user
  * forwarded files ride the normal download path (no re-upload)
  * ordering across multiple forwarded messages
  * precise diagnostic on a malformed share, no hallucinated request
  * no duplicate rendering (blocks vs attachments), and plain link unfurls
    with ``is_msg_unfurl`` are still skipped
  * the share pass runs after the unauthorized-user reject
"""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


SOURCE_TEXT = (
    "Hey <@UBOTUSER>\n"
    "could you change the cart upsell so that when the autumn fertilizer is "
    "in the cart it suggests the reseed pack instead of the spreader?\n"
    "I already adjusted the checkout upsell in the app, but how do I change "
    "the cart upsell? Can you handle that?\nThanks!"
)

PERMALINK = (
    "https://example-workspace.slack.com/archives/C0SOURCE01/"
    "p1787671472272789?thread_ts=1787671472.272789&cid=C0SOURCE01"
)


def _shared_file(file_id: str, name: str) -> dict:
    return {
        "id": file_id,
        "name": name,
        "title": name,
        "mimetype": "image/png",
        "filetype": "png",
        "size": 441899,
        "file_access": "visible",
        "url_private": f"https://files.slack.com/files-pri/T0RECEIVER-{file_id}/x.png",
        "url_private_download": (
            f"https://files.slack.com/files-pri/T0RECEIVER-{file_id}/download/x.png"
        ),
        "permalink": f"https://source-workspace.slack.com/files/U0SOURCEAUTH/{file_id}/x.png",
    }


def _share_attachment(text: str = SOURCE_TEXT, files: list | None = None) -> dict:
    """One forwarded-message attachment, shaped like the payload Slack sends."""
    att = {
        "id": 1,
        "is_share": True,
        "is_msg_unfurl": True,
        "is_thread_root_unfurl": True,
        "author_id": "U0SOURCEAUTH",
        "author_name": "Source Author",
        "author_subname": "Source Author",
        "channel_id": "C0SOURCE01",
        "channel_team": "T0RECEIVER",
        "footer": "Thread in Slack conversation",
        "fallback": "[August 25th, 2026] source: " + text[:80],
        "from_url": PERMALINK,
        "mrkdwn_in": ["text"],
        "text": text,
        "ts": "1787671472.272789",
        "blocks": [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": text}],
                    }
                ],
            }
        ],
    }
    if files is not None:
        att["files"] = files
    return att


def _event(
    text: str = "Handle this request:",
    attachments: list | None = None,
    blocks: list | None = None,
    ts: str = "1787671560.010779",
) -> dict:
    event = {
        "type": "message",
        "text": text,
        "user": "U_FORWARDER",
        "team": "T0RECEIVER",
        "channel": "D_BOT",
        "channel_type": "im",
        "ts": ts,
        "blocks": blocks
        if blocks is not None
        else [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": text}],
                    }
                ],
            }
        ],
    }
    if attachments is not None:
        event["attachments"] = attachments
    return event


@pytest.fixture()
def adapter():
    config = PlatformConfig(enabled=True, token="***")
    a = SlackAdapter(config)
    a._app = MagicMock()
    a._app.client = AsyncMock()
    a._app.client.users_info = AsyncMock(
        return_value={
            "user": {
                "is_bot": False,
                "profile": {"display_name": "Test User"},
                "real_name": "Test User",
            }
        }
    )
    a._app.client.conversations_info = AsyncMock(
        return_value={"ok": True, "channel": {"name": "client-project"}}
    )
    a._bot_user_id = "U_BOT"
    a._running = True
    a.handle_message = AsyncMock()
    return a


@pytest.fixture(autouse=True)
def _redirect_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "gateway.platforms.base.DOCUMENT_CACHE_DIR", tmp_path / "doc_cache"
    )
    monkeypatch.setattr(
        "gateway.platforms.base.VIDEO_CACHE_DIR", tmp_path / "video_cache"
    )


def _delivered_text(adapter) -> str:
    adapter.handle_message.assert_awaited()
    return adapter.handle_message.call_args[0][0].text


class TestForwardedMessageIngestion:
    @pytest.mark.asyncio
    async def test_plain_message_unchanged(self, adapter):
        await adapter._handle_slack_message(_event(text="hello there"))
        text = _delivered_text(adapter)
        assert "hello there" in text
        assert "Forwarded message" not in text

    @pytest.mark.asyncio
    async def test_forwarded_text_reaches_agent(self, adapter):
        """Outer instruction + full source request both arrive."""
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert "Handle this request:" in text
        assert "Forwarded message" in text
        assert "from: Source Author" in text
        assert "autumn fertilizer" in text
        assert "reseed pack instead of the spreader" in text
        # Source content is rendered as a quote, not as direct speech.
        assert "> Hey" in text

    @pytest.mark.asyncio
    async def test_provenance_stays_separate(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        msg_event = adapter.handle_message.call_args[0][0]
        # The forwarding user remains the message author…
        assert msg_event.source.user_id == "U_FORWARDER"
        # …while the source message keeps its own identity markers.
        assert "from: Source Author" in msg_event.text
        assert "ts: 1787671472.272789" in msg_event.text
        assert PERMALINK in msg_event.text

    @pytest.mark.asyncio
    async def test_source_channel_resolved(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        assert "#client-project (C0SOURCE01)" in _delivered_text(adapter)

    @pytest.mark.asyncio
    async def test_channel_resolution_failure_falls_back_to_id(self, adapter):
        adapter._app.client.conversations_info = AsyncMock(
            side_effect=Exception("channel_not_found")
        )
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert "channel: C0SOURCE01" in text
        assert "autumn fertilizer" in text

    @pytest.mark.asyncio
    async def test_forwarded_screenshots_downloaded(self, adapter, tmp_path):
        files = [
            _shared_file("F0SHARED001", "Screenshot 2026-08-25 at 17.22.40.png"),
            _shared_file("F0SHARED002", "Screenshot 2026-08-25 at 17.23.40.png"),
        ]
        cached = []

        async def _fake_download(url, ext, **kwargs):
            path = tmp_path / f"cached_{len(cached)}{ext}"
            path.write_bytes(b"\x89PNG fake")
            cached.append(str(path))
            return str(path)

        with patch.object(
            adapter, "_download_slack_file", side_effect=_fake_download
        ):
            await adapter._handle_slack_message(
                _event(attachments=[_share_attachment(files=files)])
            )

        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 2
        assert msg_event.media_types == ["image/png", "image/png"]
        assert all(os.path.exists(p) for p in msg_event.media_urls)
        assert "2 attached file(s)" in msg_event.text

    @pytest.mark.asyncio
    async def test_multiple_forwarded_messages_keep_order(self, adapter):
        first = _share_attachment(text="First message: upsell question")
        second = _share_attachment(text="Second message: screenshots follow")
        second["author_name"] = "Second Author"
        second["ts"] = "1787671499.000001"
        await adapter._handle_slack_message(
            _event(attachments=[first, second])
        )
        text = _delivered_text(adapter)
        assert text.index("First message") < text.index("Second message")
        assert "from: Source Author" in text
        assert "from: Second Author" in text

    @pytest.mark.asyncio
    async def test_empty_outer_text_share_still_usable(self, adapter):
        await adapter._handle_slack_message(
            _event(text="", blocks=[], attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert "Forwarded message" in text
        assert "autumn fertilizer" in text

    @pytest.mark.asyncio
    async def test_malformed_share_yields_diagnostic(self, adapter):
        broken = _share_attachment(text="")
        broken["blocks"] = []
        broken.pop("files", None)
        await adapter._handle_slack_message(_event(attachments=[broken]))
        text = _delivered_text(adapter)
        assert "could not be read" in text
        assert "no source text and no files" in text
        # Nothing hallucinated beyond the outer instruction + diagnostic.
        assert "autumn fertilizer" not in text

    @pytest.mark.asyncio
    async def test_no_duplicate_source_text(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert text.count("reseed pack instead of the spreader") == 1

    @pytest.mark.asyncio
    async def test_share_already_quoted_in_blocks_not_rendered_twice(self, adapter):
        # Composer quotes mirror the source into rich_text_quote blocks; when
        # Slack also delivers a share attachment with the same text, the
        # attachment renderer must yield nothing new.
        quote_blocks = [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [{"type": "text", "text": "Handle this request:"}],
                    },
                    {
                        "type": "rich_text_quote",
                        "elements": [{"type": "text", "text": SOURCE_TEXT}],
                    },
                ],
            }
        ]
        await adapter._handle_slack_message(
            _event(blocks=quote_blocks, attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert text.count("reseed pack instead of the spreader") == 1

    @pytest.mark.asyncio
    async def test_plain_link_unfurl_still_skipped(self, adapter):
        unfurl = {
            "id": 1,
            "is_msg_unfurl": True,
            "text": "Some bot echo content",
            "ts": "1787671000.000000",
        }
        await adapter._handle_slack_message(
            _event(text="look at this", attachments=[unfurl])
        )
        text = _delivered_text(adapter)
        assert "Some bot echo content" not in text
        assert "Forwarded message" not in text

    @pytest.mark.asyncio
    async def test_message_type_stays_text(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        msg_event = adapter.handle_message.call_args[0][0]
        assert msg_event.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_shared_files_deduped_against_outer_files(self, adapter, tmp_path):
        # If Slack ever mirrors a shared file into event.files as well, it
        # must download once, not twice.
        shared = _shared_file("F0SHARED001", "shot.png")
        cached = []

        async def _fake_download(url, ext, **kwargs):
            path = tmp_path / f"cached_{len(cached)}{ext}"
            path.write_bytes(b"\x89PNG fake")
            cached.append(str(path))
            return str(path)

        event = _event(attachments=[_share_attachment(files=[shared])])
        event["files"] = [dict(shared)]
        with patch.object(
            adapter, "_download_slack_file", side_effect=_fake_download
        ):
            await adapter._handle_slack_message(event)

        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 1

    @pytest.mark.asyncio
    async def test_shared_connect_stub_file_resolved_via_files_info(
        self, adapter, tmp_path
    ):
        # Slack Connect stubs (file_access="check_file_info", no URL fields)
        # inside a share must go through files.info like outer files do.
        stub = {"id": "F0SHARED001", "file_access": "check_file_info"}
        full = _shared_file("F0SHARED001", "shot.png")
        adapter._app.client.files_info = AsyncMock(
            return_value={"ok": True, "file": full}
        )

        async def _fake_download(url, ext, **kwargs):
            path = tmp_path / f"cached{ext}"
            path.write_bytes(b"\x89PNG fake")
            return str(path)

        with patch.object(
            adapter, "_download_slack_file", side_effect=_fake_download
        ):
            await adapter._handle_slack_message(
                _event(attachments=[_share_attachment(files=[stub])])
            )

        adapter._app.client.files_info.assert_awaited_once_with(file="F0SHARED001")
        msg_event = adapter.handle_message.call_args[0][0]
        assert len(msg_event.media_urls) == 1
        assert msg_event.media_types == ["image/png"]

    @pytest.mark.asyncio
    async def test_unauthorized_sender_triggers_no_name_resolution(self, adapter):
        # The early auth reject must fire before share rendering: an
        # unauthorized sender's forwarded message must not trigger
        # conversations.info (or any delivery).
        class FakeRunner:
            def _is_user_authorized(self, source):
                return False

            async def handle(self, event):  # pragma: no cover - never called
                raise AssertionError("unauthorized event must not be handled")

        adapter._message_handler = FakeRunner().handle
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        adapter._app.client.conversations_info.assert_not_awaited()
        adapter.handle_message.assert_not_awaited()


def test_thread_history_keeps_a_forwarded_message_and_still_skips_permalink_echoes():
    """The sibling read path: a thread whose root IS a forwarded message must not reach the agent
    as an empty root. A plain permalink unfurl still repeats a message the agent already reads."""
    from plugins.platforms.slack.adapter import _extract_text_from_slack_attachments

    share = {"is_share": True, "is_msg_unfurl": True, "text": "Please check the shipping costs"}
    echo = {"is_msg_unfurl": True, "text": "already read elsewhere"}
    out = _extract_text_from_slack_attachments([share, echo])
    assert "Please check the shipping costs" in out
    assert "already read elsewhere" not in out
