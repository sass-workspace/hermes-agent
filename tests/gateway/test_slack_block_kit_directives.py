"""Unit tests for the structured-output directives in the Block Kit renderer.

Covers the ``:::card`` / ``:::carousel`` / ``:::report`` container fences and
the ``-# `` context-footer prefix, plus ``strip_directives`` (the plain-text /
notification fallback) and the ``sanitize_blocks`` clamps for the new types.
Contract under test: a directive may restyle content, never lose it — any
malformed or over-limit directive renders its inner content through the
normal pipeline.
"""

from plugins.platforms.slack.block_kit import (
    AGENT_REPLY_ACTION_ID,
    MAX_CARD_BODY,
    MAX_MARKDOWN_TEXT,
    MARKDOWN_SEGMENT_MAX,
    render_blocks,
    sanitize_blocks,
    strip_directives,
)


def _types(blocks):
    return [b["type"] for b in blocks]


CARD_MD = (
    ":::card\n"
    "title: TUR-445 · Render-Test\n"
    "subtitle: Entwurf — nicht gesendet\n"
    "Kurzer Body mit **bold**.\n"
    "button(primary): [Jira öffnen](https://elbdev.atlassian.net/browse/TUR-445)\n"
    "button: [Freigeben](reply: ja, TUR-445 freigeben)\n"
    ":::"
)


class TestCardDirective:
    def test_happy_path(self):
        blocks = render_blocks(CARD_MD)
        assert _types(blocks) == ["card"]
        card = blocks[0]
        assert card["title"]["text"] == "TUR-445 · Render-Test"
        assert card["subtitle"]["text"] == "Entwurf — nicht gesendet"
        assert "Kurzer Body" in card["body"]["text"]
        url_btn, reply_btn = card["actions"]
        assert url_btn["url"].startswith("https://elbdev.atlassian.net")
        assert url_btn["style"] == "primary"
        assert reply_btn["action_id"] == AGENT_REPLY_ACTION_ID
        assert reply_btn["value"] == "ja, TUR-445 freigeben"
        assert "url" not in reply_btn

    def test_body_only_card_is_valid(self):
        blocks = render_blocks(":::card\nnur ein Body\n:::")
        assert _types(blocks) == ["card"]
        assert "title" not in blocks[0]

    def test_empty_card_falls_back_to_nothing(self):
        # No title, no body, no parseable button -> not a card; inner content
        # (blank) renders to nothing rather than an invalid block.
        blocks = render_blocks("davor\n\n:::card\n:::\n\ndanach")
        assert all(b["type"] == "section" for b in blocks)

    def test_oversize_body_falls_back_to_normal_pipeline(self):
        long_body = "x" * (MAX_CARD_BODY + 100)
        blocks = render_blocks(f":::card\ntitle: T\n{long_body}\n:::")
        # Never truncate a draft: the content survives as normal blocks.
        assert "card" not in _types(blocks)
        assert long_body in str(blocks)

    def test_unclosed_fence_stays_literal_text(self):
        blocks = render_blocks(":::card\ntitle: T\nBody ohne Ende")
        assert "card" not in _types(blocks)
        assert ":::card" in str(blocks)

    def test_fenced_example_never_triggers(self):
        md = "```\n:::card\ntitle: Beispiel\n:::\n```"
        blocks = render_blocks(md)
        assert _types(blocks) == ["rich_text"]  # preformatted code, no card

    def test_four_buttons_fall_back(self):
        btns = "\n".join(
            f"button: [B{i}](https://example.com/{i})" for i in range(4)
        )
        blocks = render_blocks(f":::card\ntitle: T\nBody\n{btns}\n:::")
        assert "card" not in _types(blocks)

    def test_bracketed_label_rejected_as_button(self):
        # "[" in a link label kills the whole link in Slack; such a line is
        # body text, and the card still builds from title+body.
        blocks = render_blocks(
            ":::card\ntitle: T\nbutton: [[CP] Pflege](https://x.example)\n:::"
        )
        assert _types(blocks) == ["card"]
        assert "actions" not in blocks[0]


class TestCarouselDirective:
    def test_two_cards_build_carousel(self):
        md = (
            ":::carousel\n"
            ":::card\ntitle: K1\nBody 1\n:::\n"
            "\n"
            ":::card\ntitle: K2\nBody 2\n:::\n"
            ":::"
        )
        blocks = render_blocks(md)
        assert _types(blocks) == ["carousel"]
        assert [c["title"]["text"] for c in blocks[0]["elements"]] == ["K1", "K2"]

    def test_single_card_degrades_to_vertical_card(self):
        md = ":::carousel\n:::card\ntitle: K1\nBody\n:::\n:::"
        blocks = render_blocks(md)
        assert _types(blocks) == ["card"]

    def test_stray_prose_inside_carousel_falls_back(self):
        md = (
            ":::carousel\n"
            "loses Textfragment\n"
            ":::card\ntitle: K1\nBody\n:::\n"
            ":::card\ntitle: K2\nBody\n:::\n"
            ":::"
        )
        blocks = render_blocks(md)
        assert "carousel" not in _types(blocks)
        assert _types(blocks).count("card") == 2  # cards survive vertically
        assert "loses Textfragment" in str(blocks)


class TestReportDirective:
    def test_report_becomes_markdown_block_with_raw_text(self):
        inner = "## Titel\n\n- [x] done\n\n| a | b |\n|---|---|\n| 1 | 2 |"
        blocks = render_blocks(f":::report\n{inner}\n:::")
        assert _types(blocks) == ["markdown"]
        # RAW standard markdown — not mrkdwn-converted, not table-fenced.
        assert blocks[0]["text"] == inner

    def test_report_may_contain_code_fences(self):
        inner = "Text\n\n```python\nprint('x')\n```"
        blocks = render_blocks(f":::report\n{inner}\n:::")
        assert _types(blocks) == ["markdown"]
        assert "```python" in blocks[0]["text"]

    def test_oversize_report_falls_back_to_normal_pipeline(self):
        inner = "z" * (MARKDOWN_SEGMENT_MAX + 10)
        blocks = render_blocks(f":::report\n{inner}\n:::")
        assert "markdown" not in _types(blocks)
        assert blocks  # content survives as sections


class TestFooterDirective:
    def test_footer_line_becomes_context_block(self):
        blocks = render_blocks("Antwort.\n\n-# Quelle: Asana Story · geprüft 26.08.")
        assert _types(blocks) == ["section", "context"]
        assert "Quelle: Asana Story" in blocks[1]["elements"][0]["text"]

    def test_consecutive_footers_merge_into_one_context(self):
        blocks = render_blocks("-# Zeile 1\n-# Zeile 2")
        assert _types(blocks) == ["context"]
        assert blocks[0]["elements"][0]["text"] == "Zeile 1\nZeile 2"

    def test_plain_bullet_is_not_a_footer(self):
        blocks = render_blocks("- #iris-ops erwähnt")
        assert _types(blocks) == ["rich_text"]


class TestStripDirectives:
    def test_scaffolding_removed_content_kept(self):
        stripped = strip_directives(CARD_MD)
        assert ":::" not in stripped
        assert "title:" not in stripped
        assert "TUR-445 · Render-Test" in stripped
        assert "Kurzer Body" in stripped
        # Labels survive as prose; instructions and URLs never do.
        assert "Jira öffnen" in stripped
        assert "Freigeben" in stripped
        assert "ja, TUR-445 freigeben" not in stripped
        assert "elbdev.atlassian.net" not in stripped

    def test_footer_prefix_dropped_text_kept(self):
        assert strip_directives("-# Quelle: X") == "Quelle: X"

    def test_fenced_examples_untouched(self):
        md = "```\n:::card\ntitle: Beispiel\n:::\n```"
        assert strip_directives(md) == md

    def test_prose_key_lines_outside_directives_untouched(self):
        md = "title: das ist normale Prosa"
        assert strip_directives(md) == md

    def test_content_without_markers_returned_verbatim(self):
        md = "ganz normale Antwort"
        assert strip_directives(md) is md


class TestAdversarialInputs:
    """Regressions for the Phase 1 Codex review findings (all confirmed live)."""

    def test_fenced_button_inside_card_body_never_becomes_a_button(self):
        md = (
            ":::card\n"
            "title: T\n"
            "```\n"
            "button: [Run](reply: dangerous instruction)\n"
            "```\n"
            ":::"
        )
        blocks = render_blocks(md)
        assert "hermes_agent_reply" not in str(blocks)
        assert "dangerous instruction" in str(blocks)  # stays visible as text

    def test_fenced_directive_markers_do_not_alter_scan_depth(self):
        md = ":::report\nText\n```\n:::\n```\nafter\n:::"
        blocks = render_blocks(md)
        assert _types(blocks) == ["markdown"]
        assert "after" in blocks[0]["text"]

    def test_unrenderable_directive_content_survives_as_sections(self):
        inner = "\n\n".join(["---"] * 60)
        md = f"before\n\n:::card\n{inner}\n:::\n\nafter"
        blocks = render_blocks(md)
        blob = str(blocks)
        assert "before" in blob and "after" in blob
        assert "---" in blob  # the card's inner content did not vanish

    def test_oversize_reply_instruction_declines_never_truncates(self):
        big = "x" * 2100
        md = f":::card\ntitle: T\nBody\nbutton: [Run](reply: {big})\n:::"
        blocks = render_blocks(md)
        for b in blocks:
            for btn in b.get("actions", []) if isinstance(b, dict) else []:
                assert btn.get("action_id") != AGENT_REPLY_ACTION_ID
        # A truncated instruction is a different instruction — none dispatched.
        assert '"value"' not in str(blocks) or big in str(blocks)

    def test_malformed_reply_button_line_never_leaks_into_fallback(self):
        md = ":::card\ntitle: T\nBody\nbutton: [Freigeben](reply: ja) extra\n:::"
        stripped = strip_directives(md)
        assert "reply:" not in stripped
        assert "ja) extra" not in stripped

    def test_strip_handles_nested_longer_fences(self):
        md = "````\n```\n:::card\ntitle: Beispiel\n:::\n```\n````"
        assert strip_directives(md) == md

    def test_second_report_over_cumulative_budget_falls_back_not_truncated(self):
        # Slack caps ALL markdown blocks in one payload at 12k combined; two
        # individually valid reports must not let the sanitizer truncate the
        # tail — the second report degrades to sections with content intact.
        a = "a" * 7000
        b = "b" * 7000
        blocks = render_blocks(f":::report\n{a}\n:::\n\n:::report\n{b}\n:::")
        md_blocks = [x for x in blocks if x["type"] == "markdown"]
        assert len(md_blocks) == 1 and md_blocks[0]["text"] == a
        # Second report survived as plain sections (split at the 3000-char
        # section cap, so count characters rather than one contiguous string).
        blob = str(blocks)
        assert blob.count("b") >= 7000
        assert "…" not in blob

    def test_card_title_and_subtitle_are_mrkdwn_converted(self):
        # Authored **bold** must reach Slack as mrkdwn *bold* in card fields,
        # same as in the body — raw ** shows stray asterisks.
        def fake_mrkdwn(s):
            return s.replace("**", "*")

        blocks = render_blocks(
            ":::card\ntitle: **TUR-445** · Antwort\nsubtitle: **Entwurf**\nBody\n:::",
            mrkdwn_fn=fake_mrkdwn,
        )
        assert blocks[0]["title"]["text"] == "*TUR-445* · Antwort"
        assert blocks[0]["subtitle"]["text"] == "*Entwurf*"


class TestFieldsDirective:
    def test_label_value_lines_become_field_grid(self):
        blocks = render_blocks(
            ":::fields\nDatum: 26. August 2026\nBillable: vollständig\n:::"
        )
        assert _types(blocks) == ["section"]
        fields = blocks[0]["fields"]
        assert len(fields) == 2
        assert fields[0]["text"] == "*Datum*\n26. August 2026"

    def test_line_without_colon_used_verbatim(self):
        blocks = render_blocks(":::fields\nnur ein Wert\n:::")
        assert blocks[0]["fields"][0]["text"] == "nur ein Wert"

    def test_label_value_keeps_url_colons_intact(self):
        blocks = render_blocks(
            ":::fields\nTicket: https://elbdev.atlassian.net/browse/TUR-445\n:::"
        )
        assert "https://elbdev" in blocks[0]["fields"][0]["text"]

    def test_bare_url_line_stays_verbatim(self):
        def fake_mrkdwn(s):
            return s

        blocks = render_blocks(
            ":::fields\nhttps://example.com/x\n:::", mrkdwn_fn=fake_mrkdwn
        )
        assert blocks[0]["fields"][0]["text"] == "https://example.com/x"

    def test_fence_inside_fields_declines_to_literal(self):
        md = ":::fields\n```\nDatum: 26. August 2026\n```\n:::"
        blocks = render_blocks(md)
        assert not any(b.get("fields") for b in blocks)
        assert "Datum" in str(blocks)  # fence content survives literally

    def test_eleven_fields_fall_back(self):
        inner = "\n".join(f"K{i}: v" for i in range(11))
        blocks = render_blocks(f":::fields\n{inner}\n:::")
        assert "fields" not in str(_types(blocks)) or all(
            "fields" not in b for b in blocks
        )
        assert "K10" in str(blocks)  # content survives

    def test_empty_fields_render_nothing_invalid(self):
        blocks = render_blocks("davor\n\n:::fields\n:::")
        assert all(b["type"] == "section" and b.get("text") for b in blocks)


class TestMenuDirective:
    def test_menu_builds_overflow_accessory(self):
        md = (
            ":::menu\n"
            "**TUR-445** · wartet auf QA\n"
            "option: [In Prüfung](reply: TUR-445 in Prüfung setzen)\n"
            "option: [Jira öffnen](https://elbdev.atlassian.net/browse/TUR-445)\n"
            ":::"
        )
        blocks = render_blocks(md)
        assert _types(blocks) == ["section"]
        acc = blocks[0]["accessory"]
        assert acc["type"] == "overflow"
        assert acc["action_id"] == "hermes_agent_menu"
        reply_opt, url_opt = acc["options"]
        assert reply_opt["value"] == "TUR-445 in Prüfung setzen"
        assert url_opt["url"].startswith("https://")
        assert url_opt["value"] == "url_1"

    def test_six_options_fall_back(self):
        opts = "\n".join(f"option: [O{i}](reply: tu {i})" for i in range(6))
        blocks = render_blocks(f":::menu\nText\n{opts}\n:::")
        assert not any(b.get("accessory") for b in blocks)

    def test_menu_without_text_falls_back(self):
        blocks = render_blocks(":::menu\noption: [A](reply: b)\n:::")
        assert not any(b.get("accessory") for b in blocks)

    def test_oversize_menu_instruction_declines(self):
        big = "x" * 2100
        blocks = render_blocks(f":::menu\nText\noption: [A](reply: {big})\n:::")
        assert not any(b.get("accessory") for b in blocks)

    def test_oversize_option_label_invalidates_never_truncates(self):
        label = "L" * 80
        blocks = render_blocks(f":::menu\nText\noption: [{label}](reply: tu es)\n:::")
        assert not any(b.get("accessory") for b in blocks)
        assert label in str(blocks)  # survives as plain content

    def test_fenced_option_inside_menu_never_goes_live(self):
        md = ":::menu\nText\n```\noption: [Secret](reply: do secret)\n```\n:::"
        blocks = render_blocks(md)
        assert not any(b.get("accessory") for b in blocks)
        assert "do secret" in str(blocks)  # literal fenced example, not a control

    def test_menu_fallback_keeps_labels_in_stripped_text(self):
        # 6 options -> menu declines; the preview still names the choices.
        opts = "\n".join(f"option: [O{i}](reply: tu {i})" for i in range(6))
        stripped = strip_directives(f":::menu\nText\n{opts}\n:::")
        assert "O0" in stripped and "O5" in stripped
        assert "tu 0" not in stripped


class TestCardImage:
    def test_image_key_becomes_hero_image(self):
        md = (
            ":::card\n"
            "title: Deploy-Vorschau\n"
            "image: [Screenshot der Startseite](https://example.com/shot.png)\n"
            "Body.\n"
            ":::"
        )
        blocks = render_blocks(md)
        card = blocks[0]
        assert card["hero_image"]["image_url"] == "https://example.com/shot.png"
        assert card["hero_image"]["alt_text"] == "Screenshot der Startseite"

    def test_non_http_image_stays_body_text(self):
        md = ":::card\ntitle: T\nimage: [x](file:///etc/passwd)\n:::"
        blocks = render_blocks(md)
        assert "hero_image" not in blocks[0]

    def test_image_line_stripped_from_notification_fallback(self):
        md = ":::card\ntitle: T\nimage: [Alt](https://example.com/i.png)\nBody.\n:::"
        stripped = strip_directives(md)
        assert "example.com" not in stripped
        assert "Body." in stripped

    def test_refused_image_scheme_stays_prose_in_both_paths(self):
        # The rich path keeps a non-http image line as body text; the
        # stripped fallback must mirror that, not silently drop it.
        md = ":::card\ntitle: T\nimage: [x](file:///etc/passwd)\n:::"
        stripped = strip_directives(md)
        assert "[x](file:///etc/passwd)" in stripped


class TestSanitizeNewTypes:
    def test_markdown_cumulative_budget(self):
        big = {"type": "markdown", "text": "a" * (MAX_MARKDOWN_TEXT - 100)}
        second = {"type": "markdown", "text": "b" * 500}
        out = sanitize_blocks([big, second])
        total = sum(len(b["text"]) for b in out if b["type"] == "markdown")
        assert total <= MAX_MARKDOWN_TEXT

    def test_empty_card_dropped(self):
        out = sanitize_blocks(
            [{"type": "card", "title": {"type": "mrkdwn", "text": "  "}}]
        )
        assert out is None

    def test_card_fields_clamped(self):
        out = sanitize_blocks(
            [
                {
                    "type": "card",
                    "title": {"type": "mrkdwn", "text": "t" * 400},
                    "body": {"type": "mrkdwn", "text": "b" * 400},
                }
            ]
        )
        card = out[0]
        assert len(card["title"]["text"]) <= 150
        assert len(card["body"]["text"]) <= MAX_CARD_BODY

    def test_carousel_non_card_elements_dropped(self):
        out = sanitize_blocks(
            [
                {
                    "type": "carousel",
                    "elements": [
                        {"type": "card", "title": {"type": "mrkdwn", "text": "K"}},
                        {"type": "divider"},
                    ],
                }
            ]
        )
        assert [e["type"] for e in out[0]["elements"]] == ["card"]

    def test_carousel_without_cards_dropped(self):
        assert sanitize_blocks([{"type": "carousel", "elements": []}]) is None
