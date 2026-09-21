"""``SlackAdapter.notification_text`` — the ``text`` fallback of a Block Kit payload.

Slack shows the ``text`` field — not the blocks — in push notifications, desktop banners and the
sidebar preview, and screen readers read it instead of block content. Two properties matter: the
preview opens on WORDS, never on layout scaffolding, and the full prose survives, because this
field is also the accessible reading of the message.
"""
import re

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


@pytest.fixture(scope="module")
def notify():
    return SlackAdapter(PlatformConfig(enabled=True, token="xoxb-fake", extra={})).notification_text


BRIEF = """## Tagesbrief · Mi 20.08.

3 Positionen brauchen dich · 2 warten auf Kunden

---

| Prio | Kunde | Was |
|---|---|---|
| P1 | Turbogrün | **[TUR-445](https://elbdev.atlassian.net/browse/TUR-445)** blockiert Launch |
| P1 | Johnny Urban | **[JU-266](https://elbdev.atlassian.net/browse/JU-266)** Antwort offen |

---

**Wartet auf Kunde**
- Larkson · KVA seit 12.08. ohne Rückmeldung
"""


class TestABriefReadsAsWords:
    @pytest.fixture(scope="class")
    def out(self, notify):
        return notify(BRIEF)

    def test_it_opens_on_words_not_on_a_marker(self, out):
        assert out[:1].isalnum(), out[:40]

    @pytest.mark.parametrize("scaffolding", ["##", "---", "```", "|", "](", "**"])
    def test_no_layout_scaffolding_survives(self, out, scaffolding):
        assert scaffolding not in out

    def test_table_cells_are_flattened_onto_one_line(self, out):
        assert "P1 · Turbogrün" in out

    def test_link_labels_survive_and_urls_are_dropped(self, out):
        assert "TUR-445" in out and "JU-266" in out
        assert "elbdev.atlassian.net" not in out

    def test_prose_beyond_the_first_line_survives(self, out):
        assert "Larkson" in out
        assert not re.search(r"(?m)^- ", out), "bullet markers are dropped"


def test_leading_state_emoji_are_stripped_and_their_words_survive(notify):
    out = notify("🔴 Turbogrün blockiert\n🟢 Inkster erledigt")
    assert "🔴" not in out and "🟢" not in out
    assert "blockiert" in out and "erledigt" in out


def test_a_date_token_speaks_through_its_fallback(notify):
    out = notify("Wartet seit <!date^1755388800^{ago}|17.08.> auf Annkatrin.")
    assert "17.08." in out and "<!date" not in out


@pytest.mark.parametrize("content", ["", "   \n\n ", "```\nx = 1\n```"])
def test_nothing_readable_yields_empty_so_the_caller_keeps_the_real_text(notify, content):
    assert notify(content) == ""


def test_an_over_long_message_is_capped_and_marked(notify):
    out = notify("Wort " * 400)
    assert len(out) <= SlackAdapter._NOTIFICATION_MAX + 1
    assert out.endswith("…")


def test_escaping_matches_the_normal_path(notify):
    out = notify("Vergleich a < b & c > d")
    assert "&lt;" in out and "&amp;" in out


def test_directive_scaffolding_never_reaches_the_preview(notify):
    """:::card / "-# " footers are Block-Kit-only: the preview shows their words, never their
    markers, and a reply button's instruction never leaks into it."""
    out = notify(
        ":::card\n"
        "title: TUR-445 · Antwort\n"
        "subtitle: Entwurf — nicht gesendet\n"
        "Kurzer Entwurfstext für den Kunden.\n"
        "button: [Jira öffnen](https://elbdev.atlassian.net/browse/TUR-445)\n"
        "button: [Freigeben](reply: ja, TUR-445 freigeben)\n"
        ":::\n"
        "-# Quelle: Asana Story")
    assert ":::" not in out and "title:" not in out and "button:" not in out
    assert "TUR-445 · Antwort" in out and "Entwurfstext" in out
    assert "Quelle: Asana Story" in out and "-#" not in out
    assert "freigeben" not in out
