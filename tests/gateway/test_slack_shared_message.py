"""Forwarded/shared Slack messages are first-class agent input.

Regression suite for the real incident of 2026-08-25 (channel C0BS9K0DJ7N,
ts 1787671560.010779): Matien forwarded a client request by Hannah Mueller
from #elbdev-turbogruen into the Iris channel with the outer text
"Handle this request:". Slack delivers the forwarded source message in the
``attachments`` array with BOTH ``is_share`` and ``is_msg_unfurl`` set; the
adapter's unfurl-echo skip dropped it wholesale, so the agent received only
the outer instruction and answered "Anfrage fehlt".

The fixtures here mirror the captured raw payload shape exactly (fields
trimmed to what the adapter reads). The suite pins:

  * outer instruction + forwarded source text both reach the agent
  * source author / channel / ts / permalink provenance stays separate from
    the forwarding user
  * forwarded files ride the normal download path (no re-upload)
  * ordering across multiple forwarded messages
  * precise diagnostic on a malformed share, no hallucinated request
  * no duplicate rendering (blocks vs attachments), and plain link unfurls
    with ``is_msg_unfurl`` are still skipped
"""

import os

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.base import MessageType
from plugins.platforms.slack.adapter import SlackAdapter


HANNAH_TEXT = (
    "Hey <@UTR9UBKMZ>\n"
    "könntest du bitte beim Upsell, wenn man den Herbstdünger in den "
    "Warenkorb gelegt hat, die Nachsaat hinzufügen statt dem Turbostreuer?\n"
    "Ich habe gerade bereits in Candy Rack angepasst, dass das Upsell "
    "Nachsaat und Handstreuer zeigt, aber wie kann ich den Warenkorb Upsell "
    "anpassen? Geht das über dich?\nDanke!"
)

PERMALINK = (
    "https://elbdev-workspace.slack.com/archives/C07UURB4YE5/"
    "p1787671472272789?thread_ts=1787671472.272789&cid=C07UURB4YE5"
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
        "url_private": f"https://files.slack.com/files-pri/T01ADAB6BL6-{file_id}/x.png",
        "url_private_download": (
            f"https://files.slack.com/files-pri/T01ADAB6BL6-{file_id}/download/x.png"
        ),
        "permalink": f"https://turbogruen.slack.com/files/U0BQ21N9NMC/{file_id}/x.png",
    }


def _share_attachment(text: str = HANNAH_TEXT, files: list | None = None) -> dict:
    """One forwarded-message attachment, shaped like the captured payload."""
    att = {
        "id": 1,
        "is_share": True,
        "is_msg_unfurl": True,
        "is_thread_root_unfurl": True,
        "author_id": "U0BDEHGPX55",
        "author_name": "Hannah Mueller",
        "author_subname": "Hannah Mueller",
        "channel_id": "C07UURB4YE5",
        "channel_team": "TU2Q5VC7P",
        "footer": "Thread in Slack-Unterhaltung",
        "fallback": "[August 25th, 2026] hannah: " + text[:80],
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
        "user": "U_MATIEN",
        "team": "TU2Q5VC7P",
        "channel": "D_IRIS",
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
        return_value={"ok": True, "channel": {"name": "elbdev-turbogruen"}}
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
        """The real incident: outer instruction + full source request."""
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert "Handle this request:" in text
        assert "Forwarded message" in text
        assert "from: Hannah Mueller" in text
        assert "Herbstdünger" in text
        assert "Nachsaat hinzufügen statt dem Turbostreuer" in text
        # Source content is rendered as a quote, not as direct speech.
        assert "> Hey" in text

    @pytest.mark.asyncio
    async def test_provenance_stays_separate(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        msg_event = adapter.handle_message.call_args[0][0]
        # The forwarding user remains the message author…
        assert msg_event.source.user_id == "U_MATIEN"
        # …while the source message keeps its own identity markers.
        assert "from: Hannah Mueller" in msg_event.text
        assert "ts: 1787671472.272789" in msg_event.text
        assert PERMALINK in msg_event.text

    @pytest.mark.asyncio
    async def test_source_channel_resolved(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        assert "#elbdev-turbogruen (C07UURB4YE5)" in _delivered_text(adapter)

    @pytest.mark.asyncio
    async def test_channel_resolution_failure_falls_back_to_id(self, adapter):
        adapter._app.client.conversations_info = AsyncMock(
            side_effect=Exception("channel_not_found")
        )
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert "channel: C07UURB4YE5" in text
        assert "Herbstdünger" in text

    @pytest.mark.asyncio
    async def test_forwarded_screenshots_downloaded(self, adapter, tmp_path):
        files = [
            _shared_file("F0BS514RK0X", "Bildschirmfoto 2026-08-25 um 17.22.40.png"),
            _shared_file("F0BSGDXKEAH", "Bildschirmfoto 2026-08-25 um 17.23.40.png"),
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
        first = _share_attachment(text="Erste Nachricht: Upsell Frage")
        second = _share_attachment(text="Zweite Nachricht: Screenshots folgen")
        second["author_name"] = "Felix Kompenhans"
        second["ts"] = "1787671499.000001"
        await adapter._handle_slack_message(
            _event(attachments=[first, second])
        )
        text = _delivered_text(adapter)
        assert text.index("Erste Nachricht") < text.index("Zweite Nachricht")
        assert "from: Hannah Mueller" in text
        assert "from: Felix Kompenhans" in text

    @pytest.mark.asyncio
    async def test_empty_outer_text_share_still_usable(self, adapter):
        await adapter._handle_slack_message(
            _event(text="", blocks=[], attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert "Forwarded message" in text
        assert "Herbstdünger" in text

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
        assert "Herbstdünger" not in text

    @pytest.mark.asyncio
    async def test_no_duplicate_source_text(self, adapter):
        await adapter._handle_slack_message(
            _event(attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert text.count("Nachsaat hinzufügen statt dem Turbostreuer") == 1

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
                        "elements": [{"type": "text", "text": HANNAH_TEXT}],
                    },
                ],
            }
        ]
        await adapter._handle_slack_message(
            _event(blocks=quote_blocks, attachments=[_share_attachment()])
        )
        text = _delivered_text(adapter)
        assert text.count("Nachsaat hinzufügen statt dem Turbostreuer") == 1

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
        shared = _shared_file("F0BS514RK0X", "shot.png")
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
        stub = {"id": "F0BS514RK0X", "file_access": "check_file_info"}
        full = _shared_file("F0BS514RK0X", "shot.png")
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

        adapter._app.client.files_info.assert_awaited_once_with(file="F0BS514RK0X")
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
