"""Unit tests for the Slack Block Kit renderer (pure function, no adapter)."""

from plugins.platforms.slack.block_kit import (
    MAX_BLOCKS,
    MAX_HEADER_TEXT,
    MAX_SECTION_TEXT,
    render_blocks,
    sanitize_blocks,
)


def _types(blocks):
    return [b["type"] for b in blocks]


class TestRenderBlocksBasics:
    def test_empty_returns_none(self):
        assert render_blocks("") is None
        assert render_blocks("   \n  ") is None


    def test_header_becomes_header_block(self):
        blocks = render_blocks("# Title")
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["type"] == "plain_text"
        assert blocks[0]["text"]["text"] == "Title"


class TestNestedLists:
    def test_nested_bullets_produce_increasing_indent(self):
        md = "- a\n  - b\n    - c"
        blocks = render_blocks(md)
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        indents = [e["indent"] for e in rich["elements"] if e["type"] == "rich_text_list"]
        # true nesting: indent levels must strictly increase across the run
        assert indents == sorted(indents)
        assert max(indents) >= 2
        assert min(indents) == 0


class TestInlineFormatting:
    def test_link_becomes_link_element(self):
        blocks = render_blocks("see [docs](https://example.com/x) now")
        # link lives in a section (paragraph) — but a bulleted link is a
        # rich_text link element; assert the URL survives somewhere.
        blob = str(blocks)
        assert "https://example.com/x" in blob


    @staticmethod
    def _bullet_elements(md):
        blocks = render_blocks(md)
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        lst = [e for e in rich["elements"] if e["type"] == "rich_text_list"][0]
        return lst["elements"][0]["elements"]

    def test_date_token_in_list_item_becomes_date_element(self):
        """A <!date…> token inside a bullet must be a native ``date`` element —
        as a ``text`` element it reaches the reader as literal characters."""
        els = self._bullet_elements(
            "- [TUR-1](https://x/1) **Freigabe** — wartet seit <!date^1789106400^{ago}|3 Tagen>")
        dates = [e for e in els if e["type"] == "date"]
        assert len(dates) == 1
        assert dates[0]["timestamp"] == 1789106400
        assert dates[0]["format"] == "{ago}"
        assert dates[0]["fallback"] == "3 Tagen"
        assert "url" not in dates[0]
        # no raw token survives as text, and the bold before it is intact
        assert not any("<!date" in e.get("text", "") for e in els if e["type"] == "text")
        assert any(e["type"] == "text" and e.get("style", {}).get("bold") for e in els)
        assert any(e["type"] == "link" for e in els)

    def test_date_token_with_url_carries_the_url(self):
        els = self._bullet_elements("- seit <!date^1789106400^{date_short}^https://x/1|01.09.>")
        d = [e for e in els if e["type"] == "date"][0]
        assert d["url"] == "https://x/1" and d["fallback"] == "01.09."

    def test_emoji_shortcode_in_list_item_becomes_emoji_element(self):
        els = self._bullet_elements("- :iris-followup: [TUR-1](https://x/1) **Aktion** — Kunde")
        assert els[0] == {"type": "emoji", "name": "iris-followup"}
        assert not any(":iris-followup:" in e.get("text", "") for e in els if e["type"] == "text")

    def test_time_and_ratio_are_not_emoji(self):
        els = self._bullet_elements("- Termin 10:30:45 und a:b:c bleiben Text")
        assert not any(e["type"] == "emoji" for e in els)
        assert "".join(e.get("text", "") for e in els) == "Termin 10:30:45 und a:b:c bleiben Text"

    def test_token_inside_code_span_stays_text(self):
        els = self._bullet_elements("- der Token `<!date^1^{ago}|x>` und `:iris-overdue:` als Code")
        assert not any(e["type"] in ("date", "emoji") for e in els)
        assert any(e.get("style", {}).get("code") and "<!date^1^{ago}|x>" == e["text"] for e in els)

    def test_user_mention_in_list_item_becomes_user_element(self):
        els = self._bullet_elements("- Rückfrage an <@U0123ABC> offen")
        assert {"type": "user", "user_id": "U0123ABC"} in els

    def test_table_cell_carries_a_date_element(self):
        # The token's own pipe would split the cell like any other pipe; inside
        # a table it is written escaped (``\|``), which the cell splitter
        # restores before the tokenizer sees it.
        md = "| Kunde | Alter |\n|---|---|\n| Turbogrün | <!date^1789106400^{ago}\\|3 T> |"
        blocks = render_blocks(md)
        table = [b for b in blocks if b["type"] == "table"][0]
        cell = table["rows"][1][1]
        els = cell["elements"][0]["elements"]
        assert any(e["type"] == "date" and e["fallback"] == "3 T" for e in els)

    def test_group_notification_text_uses_the_date_fallback_and_drops_emoji(self):
        from plugins.platforms.slack.block_kit import group_notification_text
        blocks = render_blocks(
            "- :iris-followup: [TUR-1](https://x/1) **Aktion** — seit <!date^1789106400^{ago}|3 Tagen>")
        text = group_notification_text(blocks, 0, 1)
        assert "3 Tagen" in text
        assert "{ago}" not in text and "1789106400" not in text
        assert "iris-followup" not in text

    def test_mrkdwn_flattening_round_trips_the_tokens(self):
        from plugins.platforms.slack.block_kit import group_mrkdwn_text
        blocks = render_blocks("- :iris-overdue: <@U0123ABC> seit <!date^1789106400^{ago}|3 T>")
        flat = group_mrkdwn_text(blocks)
        assert ":iris-overdue:" in flat and "<@U0123ABC>" in flat and "3 T" in flat


    def test_blank_line_separated_ordered_items_stay_in_one_list(self):
        """Regression: blank lines between ordered items must not reset numbering.

        Slack numbers each rich_text_list independently.  If blank lines break
        the list run, N items produce N separate lists each starting at 1.
        See: https://github.com/NousResearch/hermes-agent/issues/57076
        """
        md = "1. alpha\n\n1. beta\n\n1. gamma"
        blocks = render_blocks(md)
        rich = [b for b in blocks if b["type"] == "rich_text"][0]
        lists = [e for e in rich["elements"] if e["type"] == "rich_text_list"]
        # Must be ONE list with 3 items, not 3 separate single-item lists
        assert len(lists) == 1
        items = lists[0]["elements"]
        assert len(items) == 3


class TestTables:
    def test_pipe_table_renders_native_table_block(self):
        md = (
            "| Name | Status |\n"
            "|------|--------|\n"
            "| a | ok |\n"
            "| b | fail |"
        )
        blocks = render_blocks(md)
        assert len(blocks) == 1
        assert blocks[0]["type"] == "table"
        rows = blocks[0]["rows"]
        # header + 2 body rows, 2 columns each
        assert len(rows) == 3
        assert all(len(r) == 2 for r in rows)
        # cells are rich_text carrying the values
        assert str(rows[0]).count("Name") == 1
        assert "fail" in str(rows[2])


    def test_oversized_table_falls_back_to_monospace(self):
        # 120 rows > MAX_TABLE_ROWS -> monospace rich_text fallback, not a table
        big = "| a | b |\n|---|---|\n" + "\n".join(f"| x{i} | y |" for i in range(120))
        blocks = render_blocks(big)
        assert blocks[0]["type"] == "rich_text"  # preformatted fallback
        assert blocks[0]["elements"][0]["type"] == "rich_text_preformatted"


    def test_escaped_pipe_not_a_column_separator(self):
        md = (
            "| Expr | Meaning |\n"
            "|------|--------|\n"
            "| a \\| b | or |"
        )
        blocks = render_blocks(md)
        assert blocks[0]["type"] == "table"
        # the escaped-pipe cell stays a single cell containing a literal pipe
        body = blocks[0]["rows"][1]
        assert len(body) == 2
        assert "|" in str(body[0])


class TestLimits:

    def test_too_many_blocks_no_longer_declines(self):
        # 60 dividers => 60 blocks. The renderer used to return None (caller
        # fell back to flat text); the adapter now partitions instead.
        md = "\n\n".join(["---"] * (MAX_BLOCKS + 10))
        blocks = render_blocks(md)
        assert blocks is not None and len(blocks) == MAX_BLOCKS + 10


class TestPayloadPartition:
    """partition_blocks: <= budget and <= MAX_BLOCKS per group, never inside a block."""

    def _brief(self, clients=8, items=6, heading="**Kunde {c} · Matien**"):
        parts = ["## Täglicher Exception-Brief · 27. August 2026", ""]
        for c in range(clients):
            parts.append(heading.format(c=c))
            parts.append("")
            for i in range(items):
                k = 400 + c * 10 + i
                parts.append(
                    f"- 🟡 **Freigabe nötig** · [TUR-{k}](https://elbdev.atlassian.net/browse/TUR-{k}) "
                    "— Warenkorb-Rabatt ist veröffentlichungsbereit; Feedback zu Banner und Farbe muss geprüft werden."
                )
            parts.append("")
        return "\n".join(parts)

    def test_payload_size_counts_serialized_bytes(self):
        from plugins.platforms.slack.block_kit import payload_size
        assert payload_size(None) == 0
        assert 0 < payload_size(render_blocks("hallo")) < payload_size(render_blocks("hallo " * 500))

    def test_under_budget_stays_one_group(self):
        from plugins.platforms.slack.block_kit import partition_blocks
        blocks = render_blocks(self._brief(clients=2, items=3))
        assert partition_blocks(blocks) == [blocks]

    def test_over_budget_splits_and_respects_limits(self):
        from plugins.platforms.slack.block_kit import partition_blocks, payload_size
        blocks = render_blocks(self._brief(clients=8, items=6))
        groups = partition_blocks(blocks, budget=6000)
        assert len(groups) > 1
        assert [b for g in groups for b in g] == blocks  # order preserved, nothing lost
        for g in groups:
            assert payload_size(g) <= 6000 or len(g) == 1
            assert len(g) <= MAX_BLOCKS
            assert g[-1]["type"] != "divider"

    def test_block_count_alone_splits(self):
        from plugins.platforms.slack.block_kit import partition_blocks
        groups = partition_blocks(render_blocks("\n\n".join(["---"] * (MAX_BLOCKS + 10))))
        assert len(groups) >= 2 and all(len(g) <= MAX_BLOCKS for g in groups)

    def test_header_opens_the_next_group(self):
        from plugins.platforms.slack.block_kit import partition_blocks
        blocks = render_blocks(self._brief(clients=6, items=6, heading="## Kunde {c}"))
        groups = partition_blocks(blocks, budget=5000)
        assert len(groups) > 1
        assert all(g[0]["type"] == "header" for g in groups[1:])

    def test_markdown_group_uses_the_lower_budget(self):
        from plugins.platforms.slack.block_kit import partition_blocks, MARKDOWN_GROUP_BUDGET, payload_size
        report = ":::report\n" + "\n".join(f"- Zeile {i} " + "x" * 80 for i in range(120)) + "\n:::"
        md = "Vorab.\n\n" + report + "\n\n" + "\n".join(f"- Punkt {i} " + "y" * 200 for i in range(40))
        groups = partition_blocks(render_blocks(md))
        assert len(groups) >= 2
        for g in groups:
            if any(b["type"] == "markdown" for b in g):
                assert payload_size(g) <= MARKDOWN_GROUP_BUDGET or len(g) == 1

    def test_group_notification_text_is_plain_words(self):
        from plugins.platforms.slack.block_kit import group_notification_text
        blocks = render_blocks("**Kunde 1**\n\n- 🟡 **Freigabe** · [TUR-1](https://x/1) — warum")
        text = group_notification_text(blocks, 1, 2)
        assert "TUR-1" in text and "*" not in text and "https://" not in text
        assert group_notification_text([{"type": "divider"}], 1, 3) == "(Teil 2/3)"


class TestEmptyContentGuards:
    """Empty content must never produce a Slack-rejected (invalid_blocks) payload.

    Slack rejects a rich_text_section / rich_text_preformatted /
    rich_text_quote whose ``elements`` is empty or contains a zero-length
    ``text`` element, and a ``header`` whose plain_text is empty. Each guard
    below corresponds to a real chat.postMessage rejection observed in
    production ("missing element" / "must be more than 0 characters").
    """

    @staticmethod
    def _assert_schema_valid(blocks):
        def walk(o):
            if isinstance(o, dict):
                if o.get("type") in (
                    "rich_text_section", "rich_text_preformatted", "rich_text_quote"
                ):
                    assert o.get("elements"), f"empty {o['type']} elements"
                if o.get("type") == "text":
                    assert len(o.get("text", "")) > 0, "zero-length text element"
                if o.get("type") == "header":
                    assert o["text"]["text"], "empty plain_text header"
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(blocks)

    def test_ragged_and_empty_table_cells_are_schema_valid(self):
        # Blank middle cell + ragged short row (padded with "") must not emit
        # an empty section or a 0-char text element.
        md = (
            "| x | y | z |\n"
            "| --- | --- | --- |\n"
            "| 1 |  | 3 |\n"   # blank middle cell
            "| 4 |"           # ragged row -> padded with empty cells
        )
        blocks = render_blocks(md)
        assert blocks[0]["type"] == "table"
        self._assert_schema_valid(blocks)

    def test_empty_code_fence_quote_and_list_item_are_schema_valid(self):
        # Empty fenced code block (common around empty tool output), blank
        # quote line, and empty list item must all stay schema-valid.
        md = "```\n```\n\n> \n\n- \n- real item"
        blocks = render_blocks(md)
        assert blocks is not None
        self._assert_schema_valid(blocks)


class TestSanitizeBlocks:
    """Outbound boundary clamp: one bad block must never fail the whole call.

    Regression coverage for the invalid_blocks / msg_too_long bug class
    (#56615 null column_settings, #62054 / #53693 >3000-char sections on
    approval chat.update after HTML-escaping inflation).
    """


    def test_oversized_section_text_is_clamped(self):
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "x" * 3500}},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_SECTION_TEXT
        assert out[0]["text"]["text"].endswith("…")

    def test_html_escape_inflated_approval_update_is_clamped(self):
        # #53693 / #62054: send path budgeted the RAW text to <=3000, but the
        # interaction payload echoes it back HTML-escaped (& -> &amp;) so the
        # chat.update section exceeds the cap.
        inflated = "a" * 2990 + "&amp;" * 10  # 3040 chars
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": inflated}},
            {"type": "context", "elements": [{"type": "mrkdwn", "text": "✅ ok"}]},
        ]
        out = sanitize_blocks(blocks)
        assert len(out[0]["text"]["text"]) <= MAX_SECTION_TEXT
        # context block untouched
        assert out[1] == blocks[1]

    def test_null_column_settings_entries_are_fixed(self):
        # #56615: Slack rejects null entries in table column_settings.
        table = {
            "type": "table",
            "rows": [[{"type": "rich_text", "elements": []}]],
            "column_settings": [None, {"align": "center"}, None],
        }
        out = sanitize_blocks([table])
        cs = out[0]["column_settings"]
        assert cs == [{}, {"align": "center"}]
        assert all(isinstance(c, dict) for c in cs)

    def test_all_null_column_settings_are_dropped(self):
        table = {
            "type": "table",
            "rows": [[{"type": "rich_text", "elements": []}]],
            "column_settings": [None, None],
        }
        out = sanitize_blocks([table])
        assert "column_settings" not in out[0]


class TestSplitTextFenceBalanced:
    """_split_text closes/reopens ``` fences at section chunk boundaries."""

    def test_fenced_split_every_chunk_balanced(self):
        from plugins.platforms.slack.block_kit import _split_text

        text = "```\n" + "\n".join("y" * 20 for _ in range(30)) + "\n```"
        chunks = _split_text(text, 100)
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.count("```") % 2 == 0, (
                f"chunk {i} has unbalanced fences: {chunk[:60]!r}"
            )




class TestPartitionReviewFindings:
    """Codex review 2026-08-28: the carried tail must be re-checked."""

    def test_backed_off_tail_never_leaves_over_budget(self):
        from plugins.platforms.slack.block_kit import partition_blocks, payload_size
        items = "\n".join(f"- Punkt {i} " + "x" * 200 for i in range(50))
        md = f"p\n\n## Header\n\n{items}\n\n{items}"
        blocks = render_blocks(md)
        groups = partition_blocks(blocks)
        for g in groups:
            assert payload_size(g) <= 20000 or len(g) == 1, [payload_size(x) for x in groups]
        assert [b for g in groups for b in g] == blocks

    def test_group_mrkdwn_text_carries_the_whole_group(self):
        from plugins.platforms.slack.block_kit import group_mrkdwn_text
        md = "## Titel\n\n**Lead**\n\n- 🟡 **Freigabe** · [TUR-1](https://x/1) — warum\n- zweiter Punkt\n\n-# Quelle: Jira"
        text = group_mrkdwn_text(render_blocks(md))
        assert "*Titel*" in text and "<https://x/1|TUR-1>" in text and "zweiter Punkt" in text and "Quelle: Jira" in text
