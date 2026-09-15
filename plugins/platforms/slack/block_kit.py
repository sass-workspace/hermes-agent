"""Render agent markdown into Slack Block Kit blocks.

Opt-in (``slack.extra.rich_blocks: true``) alternative to the flat mrkdwn
``text`` payload produced by :meth:`SlackAdapter.format_message`.  Block Kit
gives us real structural primitives — section headers, dividers, and true
*nested* lists via ``rich_text`` — that plain mrkdwn can only approximate.

Design constraints (why this module is deliberately conservative):

* **Markdown pipe-tables render as native ``table`` blocks** — real grid
  cells with per-column alignment and inline-formatted ``rich_text`` content.
  A table that exceeds Slack's limits (100 rows / 20 cols / 10k aggregate
  cell chars) or won't parse falls back to aligned monospace
  ``rich_text_preformatted`` so a large table never breaks the message.
* **Slack caps a message at 50 blocks** and a ``section``/text object at 3000
  characters.  :func:`render_blocks` enforces both and, if the content simply
  cannot be expressed within them, returns ``None`` so the caller falls back
  to the plain-text path.  A rich render is a nice-to-have; it must never lose
  a message.
* **Every blocks payload MUST ship a ``text`` fallback.**  Slack uses it for
  notifications, screen readers, and old clients.  This module only builds the
  ``blocks`` list; the adapter pairs it with the existing mrkdwn string.

The renderer never raises: any unexpected input degrades to ``None`` (caller
uses plain text).  It is a pure function of its input — no Slack client, no
adapter state — so it is trivially unit-testable.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

# Slack Block Kit hard limits (https://docs.slack.dev/reference/block-kit/blocks)
MAX_BLOCKS = 50
MAX_SECTION_TEXT = 3000
MAX_HEADER_TEXT = 150
# Native table block limits (https://docs.slack.dev/reference/block-kit/blocks/table-block)
MAX_TABLE_ROWS = 100
MAX_TABLE_COLS = 20
MAX_TABLE_CHARS = 10000  # aggregate across all cells
# Card block limits (https://docs.slack.dev/reference/block-kit/blocks/card-block;
# body cap verified live: chat.postMessage rejects >200 with invalid_blocks)
MAX_CARD_TITLE = 150
MAX_CARD_BODY = 200
MAX_CARD_BUTTONS = 3
MAX_BUTTON_VALUE = 2000
# Carousel block limits (https://docs.slack.dev/reference/block-kit/blocks/carousel-block)
MAX_CAROUSEL_CARDS = 10
MIN_CAROUSEL_CARDS = 2  # ours, not Slack's: a one-card carousel is worse than a card
# markdown block: 12k cumulative across all markdown blocks in one payload
MAX_MARKDOWN_TEXT = 12000
# Renderer headroom below the cumulative cap (mirrors the adapter's margin)
MARKDOWN_SEGMENT_MAX = 11500
# Fixed action_id for agent-authored reply buttons; the adapter registers ONE
# Bolt handler for it and derives the session from the interaction payload.
AGENT_REPLY_ACTION_ID = "hermes_agent_reply"
# Same contract for overflow-menu selections (the handler reads
# ``selected_option.value`` instead of ``value``; semantics are identical).
AGENT_MENU_ACTION_ID = "hermes_agent_menu"
# Section-block structural limits (https://docs.slack.dev/reference/block-kit)
MAX_SECTION_FIELDS = 10
MAX_FIELD_TEXT = 2000
MAX_OVERFLOW_OPTIONS = 5
MAX_OPTION_LABEL = 75

# Payload budget — measured live 2026-08-28 (#iris-ops render-test thread,
# chat.postMessage and chat.update alike): a rich_text payload is accepted up
# to ≈24.5 kB of serialized JSON (``msg_blocks_too_long`` beyond), a
# ``markdown`` block up to ≈13 kB, section blocks past 30 kB. The budgets sit
# well under the measured limits; an over-budget payload is PARTITIONED into
# consecutive posts by the adapter, never degraded to flat text.
BLOCK_PAYLOAD_BUDGET = 20000
MARKDOWN_GROUP_BUDGET = 11500  # a group carrying a markdown block
SPLIT_LEAD_TYPES = ("header", "divider")  # preferred group openers

Block = Dict[str, Any]

# ----------------------------------------------------------------------------
# Line classification
# ----------------------------------------------------------------------------

_HR_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
_HEADER_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})(.*)$")
_ORDERED_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s{0,3}>\s?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{1,}:?\s*(\|\s*:?-{1,}:?\s*)+\|?\s*$")
# Structured-output directives (agent-authored, deliberately explicit):
#   :::card / :::report / :::carousel … :::   and   "-# " footer lines.
# A ``` code fence always wins over these — canon documents the syntax inside
# fences, and fenced examples must never trigger a live block.
_DIRECTIVE_OPEN_RE = re.compile(r"^\s{0,3}:::\s*(card|report|carousel|fields|menu)\s*$")
_DIRECTIVE_CLOSE_RE = re.compile(r"^\s{0,3}:::\s*$")
_FOOTER_RE = re.compile(r"^\s{0,3}-#\s+(.+)$")
_CARD_KEY_RE = re.compile(
    r"^(title|subtitle|button|image|option)(?:\((primary|danger)\))?:\s*(.+)$"
)
_BUTTON_RE = re.compile(r"^\[([^\]]+)\]\(\s*(?:reply:\s*(.+?)|(\S+))\s*\)$")


def _is_list_line(line: str) -> bool:
    """True if ``line`` is a markdown list item (bullet or ordered)."""
    return bool(_BULLET_RE.match(line) or _ORDERED_RE.match(line))


def _indent_level(spaces: str) -> int:
    """Map leading whitespace to a nesting level (2 spaces or 1 tab per level)."""
    width = 0
    for ch in spaces:
        width += 4 if ch == "\t" else 1
    return min(width // 2, 5)  # Slack rich_text_list supports up to indent 5


# ----------------------------------------------------------------------------
# Inline markdown → rich_text elements
# ----------------------------------------------------------------------------

# Order matters: code first (opaque), then links, then Slack tokens, then
# emphasis.
_INLINE_CODE_RE = re.compile(r"`([^`]+)`")
_LINK_RE = re.compile(r"(?<!!)\[([^\]]+)\]\(([^()\s]+(?:\([^()]*\)[^()\s]*)*)\)")
_BOLD_RE = re.compile(r"(?:\*\*|__)(.+?)(?:\*\*|__)")
_ITALIC_RE = re.compile(r"(?<![\*_])(?:\*|_)(?![\*_\s])(.+?)(?<![\*_\s])(?:\*|_)(?![\*_])")
_STRIKE_RE = re.compile(r"~~(.+?)~~")

# Slack's own inline tokens. mrkdwn text objects resolve these by themselves,
# but a rich_text list item or table cell carries typed ELEMENTS: a
# ``<!date^…>`` token or ``:shortcode:`` left inside a ``text`` element reaches
# the reader as literal characters. So the tokenizer emits the native
# ``date`` / ``emoji`` / ``user`` element types instead — which is what lets
# a live age or a workspace icon sit inside a bullet at all.
_DATE_TOKEN_RE = re.compile(r"<!date\^(\d+)\^([^|>^]+?)(?:\^(https?://[^|>]+))?\|([^>]*)>")
# An emoji shortcode: lowercase name, not glued to a word or another colon on
# either side, so ``10:30:45`` and ``a:b:c`` stay text.
_EMOJI_TOKEN_RE = re.compile(r"(?<![\w:/]):([a-z0-9_+\-]{2,}):(?![\w:])")
_USER_TOKEN_RE = re.compile(r"<@([UW][A-Z0-9]{2,})>")
_TOKEN_RE = re.compile(
    "|".join(f"(?:{rx.pattern})" for rx in (_DATE_TOKEN_RE, _EMOJI_TOKEN_RE, _USER_TOKEN_RE))
)


def _token_element(s: str) -> Optional[Dict[str, Any]]:
    """The rich_text element for one matched Slack token, or None."""
    d = _DATE_TOKEN_RE.fullmatch(s)
    if d:
        el: Dict[str, Any] = {
            "type": "date",
            "timestamp": int(d.group(1)),
            "format": d.group(2),
            "fallback": d.group(4) or d.group(2),
        }
        if d.group(3):
            el["url"] = d.group(3)
        return el
    u = _USER_TOKEN_RE.fullmatch(s)
    if u:
        return {"type": "user", "user_id": u.group(1)}
    e = _EMOJI_TOKEN_RE.fullmatch(s)
    if e:
        return {"type": "emoji", "name": e.group(1)}
    return None


def _inline_elements(text: str) -> List[Dict[str, Any]]:
    """Parse a run of inline markdown into rich_text section child elements.

    Produces ``text`` elements (optionally styled bold/italic/strike/code),
    ``link`` elements, and the native ``date`` / ``emoji`` / ``user`` elements
    for Slack's own inline tokens.  Unmatched markup is emitted verbatim as
    plain text, so this never loses characters.
    """
    elements: List[Dict[str, Any]] = []

    def emit_text(s: str, style: Optional[Dict[str, bool]] = None) -> None:
        if not s:
            return
        el: Dict[str, Any] = {"type": "text", "text": s}
        if style:
            el["style"] = style
        elements.append(el)

    # Tokenize by the highest-priority markers first using a single scan.
    # We recursively split on code, then links, then emphasis to keep spans
    # from overlapping incorrectly.
    def walk(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        # inline code is opaque — no nested styling
        for m in _INLINE_CODE_RE.finditer(s):
            _walk_links(s[pos:m.start()], style)
            code_style = dict(style)
            code_style["code"] = True
            emit_text(m.group(1), code_style or None)
            pos = m.end()
        _walk_links(s[pos:], style)

    def _walk_links(s: str, style: Dict[str, bool]) -> None:
        pos = 0
        for m in _LINK_RE.finditer(s):
            _walk_tokens(s[pos:m.start()], style)
            link_el: Dict[str, Any] = {"type": "link", "url": m.group(2), "text": m.group(1)}
            if style:
                link_el["style"] = dict(style)
            elements.append(link_el)
            pos = m.end()
        _walk_tokens(s[pos:], style)

    def _walk_tokens(s: str, style: Dict[str, bool]) -> None:
        # Slack tokens are atomic: no emphasis is applied to them (a ``date``
        # element carries no style field), and the text around them keeps
        # whatever style the enclosing span has.
        pos = 0
        for m in _TOKEN_RE.finditer(s):
            el = _token_element(m.group(0))
            if el is None:
                continue
            _walk_emphasis(s[pos:m.start()], style)
            elements.append(el)
            pos = m.end()
        _walk_emphasis(s[pos:], style)

    def _walk_emphasis(s: str, style: Dict[str, bool]) -> None:
        if not s:
            return
        # Try bold, then strike, then italic, recursing into the inner span.
        for rx, key in ((_BOLD_RE, "bold"), (_STRIKE_RE, "strike"), (_ITALIC_RE, "italic")):
            m = rx.search(s)
            if m:
                _walk_emphasis(s[:m.start()], style)
                inner_style = dict(style)
                inner_style[key] = True
                _walk_emphasis(m.group(1), inner_style)
                _walk_emphasis(s[m.end():], style)
                return
        emit_text(s, dict(style) if style else None)

    walk(text, {})
    return elements or [{"type": "text", "text": text}]


# ----------------------------------------------------------------------------
# Structural block builders
# ----------------------------------------------------------------------------


def _nonempty_elements(elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Make a rich_text child-element list safe for Slack.

    Slack rejects any ``rich_text_section`` / ``rich_text_preformatted`` /
    ``rich_text_quote`` whose ``elements`` list is empty or contains a ``text``
    element of zero length (``invalid_blocks``: "missing element" / "must be
    more than 0 characters"). Empty content is common — ragged table rows are
    padded with ``""``, agents emit empty code fences around empty tool output,
    blank quote lines and empty list items occur in the wild — so drop
    zero-length text elements and, if nothing remains, substitute a single
    space, which renders as blank yet stays schema-valid. Used by every
    rich_text builder so empty content can never poison the whole payload.
    """
    els = [e for e in elements if not (e.get("type") == "text" and not e.get("text"))]
    return els or [{"type": "text", "text": " "}]


def _header_block(text: str) -> Optional[Block]:
    # header blocks are plain_text only, 150 char cap. A markdown link would
    # reach the reader verbatim (observed live 2026-08-27), so it reduces to
    # its label — the URL belongs on the first body line by canon.
    clean = _LINK_RE.sub(lambda m: m.group(1), text)
    clean = re.sub(r"[*_~`]", "", clean).strip()
    if not clean:
        # Emphasis-/whitespace-only header (e.g. "# ***" or "#   ") reduces to
        # empty; Slack rejects an empty plain_text with invalid_blocks. Skip it
        # (caller drops None) rather than poison the whole payload.
        return None
    if len(clean) > MAX_HEADER_TEXT:
        clean = clean[: MAX_HEADER_TEXT - 1] + "…"
    return {"type": "header", "text": {"type": "plain_text", "text": clean, "emoji": True}}


def _divider_block() -> Block:
    return {"type": "divider"}


def _preformatted_block(text: str) -> Block:
    # rich_text_preformatted renders monospace; used for code fences + tables.
    return {
        "type": "rich_text",
        "elements": [
            {
                "type": "rich_text_preformatted",
                "elements": _nonempty_elements([{"type": "text", "text": text.rstrip("\n")}]),
            }
        ],
    }


def _quote_block(lines: List[str]) -> Block:
    section_children: List[Dict[str, Any]] = []
    for i, ln in enumerate(lines):
        if i:
            section_children.append({"type": "text", "text": "\n"})
        section_children.extend(_inline_elements(ln))
    return {
        "type": "rich_text",
        "elements": [{"type": "rich_text_quote", "elements": _nonempty_elements(section_children)}],
    }


def _list_block(items: List[Tuple[int, bool, str]]) -> Block:
    """Build ONE rich_text block from consecutive list items.

    ``items`` is a list of ``(indent, ordered, text)``.  Each contiguous run
    sharing the same (indent, ordered) becomes a ``rich_text_list`` element;
    indentation changes start a new element, which is how Slack renders true
    nesting.
    """
    elements: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    cur_key: Optional[Tuple[int, bool]] = None
    for indent, ordered, text in items:
        key = (indent, ordered)
        if key != cur_key:
            cur = {
                "type": "rich_text_list",
                "style": "ordered" if ordered else "bullet",
                "indent": indent,
                "elements": [],
            }
            elements.append(cur)
            cur_key = key
        if cur is None:
            # Defensive: should never happen (first iteration always enters
            # the ``if key != cur_key`` block above), but guard explicitly
            # so ``python -O`` doesn't silently drop the check.
            continue
        cur["elements"].append(
            {"type": "rich_text_section", "elements": _nonempty_elements(_inline_elements(text))}
        )
    return {"type": "rich_text", "elements": elements}


def _section_block(text: str) -> Block:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


# ----------------------------------------------------------------------------
# Table handling — native Block Kit ``table`` block, monospace fallback
# ----------------------------------------------------------------------------


def _parse_alignment(sep_line: str) -> List[str]:
    """Parse a markdown separator row (``|:--|:-:|--:|``) into column aligns.

    Returns a list of ``"left"``/``"center"``/``"right"`` per column.
    """
    aligns: List[str] = []
    for cell in sep_line.strip().strip("|").split("|"):
        c = cell.strip()
        left = c.startswith(":")
        right = c.endswith(":")
        if left and right:
            aligns.append("center")
        elif right:
            aligns.append("right")
        else:
            aligns.append("left")
    return aligns


def _split_row(row: str) -> List[str]:
    """Split a markdown table row into trimmed cell strings.

    Respects backslash-escaped pipes (``\\|``) so they aren't treated as
    column separators.
    """
    # Temporarily protect escaped pipes, split on real ones, then restore.
    protected = row.strip().strip("|").replace(r"\|", "\x00PIPE\x00")
    return [c.strip().replace("\x00PIPE\x00", "|") for c in protected.split("|")]


def _rich_text_cell(text: str) -> Dict[str, Any]:
    """A ``rich_text`` table cell carrying inline-formatted content.

    Empty cells are common (ragged rows are padded with ``""``); Slack rejects
    a cell whose section is empty or carries a zero-length text element, so the
    elements are routed through ``_nonempty_elements``.
    """
    return {
        "type": "rich_text",
        "elements": [
            {"type": "rich_text_section", "elements": _nonempty_elements(_inline_elements(text))}
        ],
    }


def _table_block(rows: List[str], sep_line: str) -> Optional[Block]:
    """Build a native Slack ``table`` block from markdown pipe-table rows.

    ``rows`` includes the header row (index 0) and body rows; ``sep_line`` is
    the ``|---|`` alignment row (already consumed by the caller).  Returns
    ``None`` when the table exceeds Slack's limits (100 rows / 20 cols /
    10,000 aggregate cell chars) or parses to nothing — the caller then falls
    back to the monospace preformatted rendering.
    """
    parsed = [_split_row(r) for r in rows if r.strip()]
    if not parsed:
        return None
    ncols = max(len(r) for r in parsed)
    # Reject rather than silently truncate beyond Slack's structural limits.
    if len(parsed) > MAX_TABLE_ROWS or ncols > MAX_TABLE_COLS:
        return None
    for r in parsed:
        r.extend([""] * (ncols - len(r)))

    total_chars = sum(len(c) for r in parsed for c in r)
    if total_chars > MAX_TABLE_CHARS:
        return None

    aligns = _parse_alignment(sep_line)
    # Slack requires every provided ``column_settings`` entry to be an object.
    # Missing trailing entries inherit defaults, so only emit settings through
    # the last non-default alignment. Earlier default-left placeholders still
    # need explicit valid objects to preserve positional alignment.
    last_non_default = -1
    for c in range(min(ncols, MAX_TABLE_COLS)):
        align = aligns[c] if c < len(aligns) else "left"
        if align != "left":
            last_non_default = c
    column_settings: List[Dict[str, Any]] = []
    for c in range(last_non_default + 1):
        align = aligns[c] if c < len(aligns) else "left"
        column_settings.append({"align": align})

    block: Block = {
        "type": "table",
        "rows": [[_rich_text_cell(cell) for cell in row] for row in parsed],
    }
    if column_settings:
        block["column_settings"] = column_settings
    return block


def _render_table(rows: List[str]) -> str:
    """Render markdown pipe-table rows as aligned monospace text (fallback)."""
    parsed: List[List[str]] = []
    for r in rows:
        cells = _split_row(r)
        parsed.append(cells)
    if not parsed:
        return "\n".join(rows)
    ncols = max(len(r) for r in parsed)
    for r in parsed:
        r.extend([""] * (ncols - len(r)))
    widths = [max(len(r[c]) for r in parsed) for c in range(ncols)]
    out_lines = []
    for ri, r in enumerate(parsed):
        line = " | ".join(r[c].ljust(widths[c]) for c in range(ncols))
        out_lines.append(line.rstrip())
        if ri == 0:  # header underline
            out_lines.append("-+-".join("-" * widths[c] for c in range(ncols)))
    return "\n".join(out_lines)


# ----------------------------------------------------------------------------
# Structured-output directives — card / carousel / report / "-#" footer
# ----------------------------------------------------------------------------


def _scan_directive(lines: List[str], start: int) -> Optional[Tuple[List[str], int]]:
    """Find the matching ``:::`` close for the directive opened at ``start``.

    Directives nest (a carousel contains cards): an inner ``:::name`` line
    increases depth, a bare ``:::`` closes the innermost. A ``` code fence
    inside the directive is opaque — fenced ``:::`` lines are literal example
    text and never alter the depth. Returns the inner lines and the index just
    past the closing fence, or ``None`` when the fence is never closed (caller
    then treats the marker as plain text).
    """
    depth = 1
    i = start + 1
    n = len(lines)
    code_marker: Optional[str] = None
    while i < n:
        fm = _FENCE_RE.match(lines[i])
        if code_marker is not None:
            if lines[i].lstrip().startswith(code_marker):
                code_marker = None
            i += 1
            continue
        if fm:
            code_marker = fm.group(1)
            i += 1
            continue
        if _DIRECTIVE_OPEN_RE.match(lines[i]):
            depth += 1
        elif _DIRECTIVE_CLOSE_RE.match(lines[i]):
            depth -= 1
            if depth == 0:
                return lines[start + 1 : i], i + 1
        i += 1
    return None


def _parse_button(spec: str, style: Optional[str], index: int) -> Optional[Dict[str, Any]]:
    """Parse one ``button:`` line body into a Block Kit button element.

    Two schemes: ``[Label](https://…)`` → URL button (no handler needed) and
    ``[Label](reply: <instruction>)`` → agent reply button whose value the
    gateway dispatches into the thread's session as the clicker's instruction.
    """
    m = _BUTTON_RE.match(spec)
    if not m:
        return None
    label = m.group(1).strip()
    reply_text = m.group(2)
    url = m.group(3)
    if not label or "[" in label or "]" in label:
        return None
    btn: Dict[str, Any] = {
        "type": "button",
        "text": {"type": "plain_text", "text": label[:75], "emoji": True},
    }
    if style in ("primary", "danger"):
        btn["style"] = style
    if reply_text is not None:
        text = reply_text.strip()
        # A truncated instruction is a DIFFERENT instruction — decline rather
        # than clamp, so the card falls back and nothing mutated is dispatched.
        if not text or len(text) > MAX_BUTTON_VALUE:
            return None
        btn["action_id"] = AGENT_REPLY_ACTION_ID
        btn["value"] = text
    elif url and url.lower().startswith(("http://", "https://")):
        btn["url"] = url
        btn["action_id"] = f"hermes_card_url_{index}"
    else:
        return None
    return btn


def _parse_card(inner: List[str], mrkdwn_fn) -> Optional[Block]:
    """Build a ``card`` block from the lines inside ``:::card … :::``.

    ``title:`` / ``subtitle:`` (first occurrence each) and ``button:`` /
    ``button(primary):`` / ``button(danger):`` are key lines; everything else
    is the body, converted to mrkdwn. Returns ``None`` when the content does
    not fit a card (no title AND no body, or any field over its documented
    cap — the body cap is 200 chars, verified live) so the caller renders the
    inner content through the normal pipeline instead of truncating it.
    """
    title: Optional[str] = None
    subtitle: Optional[str] = None
    hero_image: Optional[Dict[str, Any]] = None
    buttons: List[Dict[str, Any]] = []
    body_lines: List[str] = []
    code_marker: Optional[str] = None
    for ln in inner:
        # Key lines are only live OUTSIDE ``` fences — a fenced example of a
        # button line must never become a real button.
        if code_marker is not None:
            body_lines.append(ln)
            if ln.lstrip().startswith(code_marker):
                code_marker = None
            continue
        fmatch = _FENCE_RE.match(ln)
        if fmatch:
            code_marker = fmatch.group(1)
            body_lines.append(ln)
            continue
        m = _CARD_KEY_RE.match(ln.strip())
        if m:
            key, style, rest = m.group(1), m.group(2), m.group(3).strip()
            if key == "title" and title is None:
                title = rest
                continue
            if key == "subtitle" and subtitle is None:
                subtitle = rest
                continue
            if key == "image" and hero_image is None:
                im = _BUTTON_RE.match(rest)
                if im and im.group(3) and im.group(3).lower().startswith(
                    ("http://", "https://")
                ):
                    hero_image = {
                        "type": "image",
                        "image_url": im.group(3),
                        "alt_text": im.group(1).strip()[:2000] or "Bild",
                    }
                    continue
            if key == "button":
                btn = _parse_button(rest, style, len(buttons))
                if btn is not None:
                    buttons.append(btn)
                    continue
        body_lines.append(ln)
    body_md = "\n".join(body_lines).strip()
    body = mrkdwn_fn(body_md) if body_md else ""
    # Title/subtitle are mrkdwn text objects too — run them through the same
    # converter as the body so authored ``**bold**`` becomes mrkdwn ``*bold*``
    # instead of showing stray asterisks.
    title = mrkdwn_fn(title) if title else title
    subtitle = mrkdwn_fn(subtitle) if subtitle else subtitle
    if not title and not body:
        return None
    if len(buttons) > MAX_CARD_BUTTONS:
        return None
    for value, cap in ((title, MAX_CARD_TITLE), (subtitle, MAX_CARD_TITLE), (body, MAX_CARD_BODY)):
        if value and len(value) > cap:
            return None
    block: Block = {"type": "card"}
    if hero_image:
        block["hero_image"] = hero_image
    if title:
        block["title"] = {"type": "mrkdwn", "text": title, "verbatim": False}
    if subtitle:
        block["subtitle"] = {"type": "mrkdwn", "text": subtitle, "verbatim": False}
    if body:
        block["body"] = {"type": "mrkdwn", "text": body, "verbatim": False}
    if buttons:
        block["actions"] = buttons
    return block


def _parse_carousel(inner: List[str], mrkdwn_fn) -> Optional[Block]:
    """Build a ``carousel`` block from ``:::card`` fences inside a carousel.

    Only card directives (and blank lines) are allowed inside; any other
    content, an unparseable card, or a count outside [2, 10] returns ``None``
    and the caller renders the inner content normally (cards then appear
    vertically — never lost).
    """
    cards: List[Block] = []
    i = 0
    n = len(inner)
    while i < n:
        line = inner[i]
        if not line.strip():
            i += 1
            continue
        m = _DIRECTIVE_OPEN_RE.match(line)
        if not m or m.group(1) != "card":
            return None
        scan = _scan_directive(inner, i)
        if scan is None:
            return None
        card_lines, i = scan
        card = _parse_card(card_lines, mrkdwn_fn)
        if card is None:
            return None
        cards.append(card)
    if not (MIN_CAROUSEL_CARDS <= len(cards) <= MAX_CAROUSEL_CARDS):
        return None
    return {"type": "carousel", "elements": cards}


def _parse_fields(inner: List[str], mrkdwn_fn) -> Optional[Block]:
    """Build a two-column key-value ``section`` (fields grid) from ``:::fields``.

    Each non-empty line is one field. ``Label: value`` renders as a bold
    label over its value (the Block Kit template convention); a line without
    a colon is used verbatim. Declines (``None`` → normal pipeline) on zero
    or more than 10 fields, or any field over the 2000-char cap.
    """
    fields: List[Dict[str, Any]] = []
    for ln in inner:
        text = ln.strip()
        if not text:
            continue
        # A code fence inside a fields grid cannot be represented — decline
        # so the whole inner content renders through the normal pipeline
        # with its fence intact.
        if _FENCE_RE.match(ln):
            return None
        if ":" in text:
            label, _, value = text.partition(":")
            label, value = label.strip(), value.strip()
            # Not a label when the colon belongs to a URL scheme ("https://…"
            # as the whole line) — that line is verbatim content.
            if label and value and not value.startswith("//"):
                # Bold the label AFTER conversion: a literal ``*label*`` fed
                # to the markdown→mrkdwn converter is read as markdown italic
                # and comes out as ``_label_`` (observed live 2026-08-27).
                rendered = f"*{mrkdwn_fn(label)}*\n{mrkdwn_fn(value)}"
                if len(rendered) > MAX_FIELD_TEXT:
                    return None
                fields.append({"type": "mrkdwn", "text": rendered})
                continue
        rendered = mrkdwn_fn(text)
        if len(rendered) > MAX_FIELD_TEXT:
            return None
        fields.append({"type": "mrkdwn", "text": rendered})
    if not fields or len(fields) > MAX_SECTION_FIELDS:
        return None
    return {"type": "section", "fields": fields}


def _parse_menu(inner: List[str], mrkdwn_fn) -> Optional[Block]:
    """Build a ``section`` with an overflow-menu accessory from ``:::menu``.

    ``option:`` lines use the button grammar — ``[Label](reply: …)`` becomes a
    selectable instruction (same click-means-instruction contract as reply
    buttons, dispatched via ``AGENT_MENU_ACTION_ID``), ``[Label](https://…)``
    opens the URL. Every other line is the section text. Declines on zero
    text, zero valid options, more than 5 options, or an over-limit value —
    a truncated instruction would be a different instruction.
    """
    options: List[Dict[str, Any]] = []
    text_lines: List[str] = []
    for ln in inner:
        # Fenced content cannot be represented in a section+overflow shape,
        # and a fenced ``option:`` example must never become live — decline
        # and let the normal pipeline render everything literally.
        if _FENCE_RE.match(ln):
            return None
        m = _CARD_KEY_RE.match(ln.strip())
        if m and m.group(1) == "option":
            bm = _BUTTON_RE.match(m.group(3).strip())
            if bm is None:
                return None
            label = bm.group(1).strip()
            reply_text = bm.group(2)
            url = bm.group(3)
            # Over-limit invalidates, never truncates (canon): a shortened
            # label could misdescribe the instruction it triggers.
            if not label or "[" in label or "]" in label or len(label) > MAX_OPTION_LABEL:
                return None
            opt: Dict[str, Any] = {
                "text": {"type": "plain_text", "text": label}
            }
            if reply_text is not None:
                value = reply_text.strip()
                if not value or len(value) > MAX_BUTTON_VALUE:
                    return None
                opt["value"] = value
            elif url and url.lower().startswith(("http://", "https://")):
                opt["url"] = url
                opt["value"] = f"url_{len(options)}"
            else:
                return None
            options.append(opt)
        else:
            text_lines.append(ln)
    text_md = "\n".join(text_lines).strip()
    if not text_md or not options or len(options) > MAX_OVERFLOW_OPTIONS:
        return None
    return {
        "type": "section",
        "text": {"type": "mrkdwn", "text": mrkdwn_fn(text_md)},
        "accessory": {
            "type": "overflow",
            "action_id": AGENT_MENU_ACTION_ID,
            "options": options,
        },
    }


def _directive_fallback_markdown(name: str, inner: List[str]) -> str:
    """Markdown for a directive that could not become its block.

    A declined ``:::card`` (body over the 200-char cap, four buttons, …) used
    to re-render its raw lines, leaking ``title:`` / ``subtitle:`` /
    ``button:`` as prose (observed live 2026-08-27). Key lines are rewritten
    into their nearest plain-markdown form instead: title → bold line,
    subtitle → plain line, URL buttons/options → links, reply buttons → the
    label alone (the instruction is control data, never prose), images
    dropped. Fenced regions stay untouched; ``fields`` inner lines are already
    readable ``Label: Wert`` prose and a declined ``carousel`` keeps its
    ``:::card`` fences so the cards still render (vertically) — both pass
    through unchanged.
    """
    if name not in ("card", "menu"):
        return "\n".join(inner)
    out: List[str] = []
    code_marker: Optional[str] = None
    seen_title = seen_subtitle = seen_image = False
    for ln in inner:
        if code_marker is not None:
            out.append(ln)
            if ln.lstrip().startswith(code_marker):
                code_marker = None
            continue
        fmatch = _FENCE_RE.match(ln)
        if fmatch:
            code_marker = fmatch.group(1)
            out.append(ln)
            continue
        stripped = ln.strip()
        km = _CARD_KEY_RE.match(stripped)
        if not km:
            out.append(ln)
            continue
        key, rest = km.group(1), km.group(3).strip()
        # Mirror the parser exactly: only the FIRST title / subtitle / image
        # line is a key line; a later ``title: …`` is body text and stays.
        if key == "title":
            if seen_title:
                out.append(ln)
                continue
            seen_title = True
            out.append(f"**{rest}**")
        elif key == "subtitle":
            if seen_subtitle:
                out.append(ln)
                continue
            seen_subtitle = True
            out.append(rest)
        elif key in ("button", "option"):
            bm = _BUTTON_RE.match(rest)
            if bm and bm.group(3):
                out.append(f"[{bm.group(1).strip()}]({bm.group(3)})")
            elif bm:
                out.append(bm.group(1).strip())
            elif "reply:" not in rest:
                out.append(rest)
        elif key == "image":
            im = _BUTTON_RE.match(rest)
            is_image = bool(im and im.group(3) and im.group(3).lower().startswith(("http://", "https://")))
            if is_image and not seen_image:
                seen_image = True  # dropped, like the parser's hero image
            else:
                out.append(ln if seen_image or not is_image else rest)
    return "\n".join(out)


def _report_block(inner: List[str], remaining: int = MARKDOWN_SEGMENT_MAX) -> Optional[Block]:
    """Build a native ``markdown`` block from ``:::report`` content.

    The text is passed RAW (standard markdown, not mrkdwn) — Slack renders
    headings, tables, task lists and syntax-highlighted code natively.
    ``remaining`` is what is left of the per-payload cumulative markdown
    budget (Slack caps ALL markdown blocks in one message at 12k combined):
    declining against it here means a second over-budget report falls back to
    the normal pipeline with its content intact, instead of being truncated
    by the outbound sanitizer's last-resort clamp.
    """
    text = "\n".join(inner).strip()
    if not text or len(text) > min(MARKDOWN_SEGMENT_MAX, remaining):
        return None
    return {"type": "markdown", "text": text}


def strip_directives(content: str) -> str:
    """Remove directive scaffolding for plain-text / notification fallbacks.

    Marker lines (``:::card``, ``:::``) disappear, ``-# `` and card key
    prefixes are dropped while their content survives (a reply button keeps
    only its label — the ``reply:`` target is an instruction, not prose), and
    anything inside a ``` code fence is left untouched, so documented examples
    stay verbatim. Never raises; returns the input on any unexpected shape.
    """
    if not content or (":::" not in content and "-#" not in content):
        return content
    try:
        out: List[str] = []
        code_marker: Optional[str] = None
        depth = 0
        for line in content.splitlines():
            # Matched-fence tracking (not a blind toggle): a four-backtick
            # fence containing a three-backtick example must stay one opaque
            # region, closed only by its own marker.
            if code_marker is not None:
                out.append(line)
                if line.lstrip().startswith(code_marker):
                    code_marker = None
                continue
            fmatch = _FENCE_RE.match(line)
            if fmatch:
                code_marker = fmatch.group(1)
                out.append(line)
                continue
            if _DIRECTIVE_OPEN_RE.match(line):
                depth += 1
                continue
            if _DIRECTIVE_CLOSE_RE.match(line) and depth > 0:
                depth -= 1
                continue
            fm = _FOOTER_RE.match(line)
            if fm:
                out.append(fm.group(1))
                continue
            if depth > 0:
                km = _CARD_KEY_RE.match(line.strip())
                if km:
                    key, rest = km.group(1), km.group(3).strip()
                    if key in ("button", "option"):
                        # The LABEL is prose and survives; the reply
                        # instruction and the URL are control data and never
                        # reach the preview. A malformed reply line drops
                        # whole (its instruction cannot be separated safely).
                        bm = _BUTTON_RE.match(rest)
                        if bm:
                            out.append(bm.group(1).strip())
                            continue
                        if "reply:" in rest:
                            continue
                    if key == "image":
                        im = _BUTTON_RE.match(rest)
                        if im and im.group(3) and im.group(3).lower().startswith(
                            ("http://", "https://")
                        ):
                            continue  # a rendered image is not preview prose
                        # Non-image content (e.g. a refused scheme) stays
                        # body text in the rich path — mirror that here.
                    out.append(rest)
                    continue
            out.append(line)
        return "\n".join(out)
    except Exception:
        return content


# ----------------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------------


def render_blocks(
    markdown: str,
    mrkdwn_fn=None,
) -> Optional[List[Block]]:
    """Convert agent markdown to a Slack Block Kit ``blocks`` list.

    Args:
        markdown: The agent's response text (standard markdown).
        mrkdwn_fn: Optional callable converting a markdown paragraph to Slack
            mrkdwn for ``section`` blocks (the adapter passes
            ``format_message``).  When ``None``, the raw paragraph text is used.

    Returns:
        A list of Block Kit block dicts, or ``None`` when the content is empty,
        exceeds Slack's structural limits, or hits an unexpected shape — the
        caller then falls back to the flat ``text`` payload.  Never raises.
    """
    if not markdown or not markdown.strip():
        return None

    fmt = mrkdwn_fn or (lambda s: s)

    try:
        blocks: List[Block] = []
        lines = markdown.replace("\r\n", "\n").split("\n")
        i = 0
        n = len(lines)
        para: List[str] = []
        markdown_budget = MARKDOWN_SEGMENT_MAX  # cumulative across all reports

        def flush_para() -> None:
            if not para:
                return
            text = "\n".join(para).strip()
            para.clear()
            if not text:
                return
            rendered = fmt(text)
            # Split oversized sections on the 3000-char limit.
            for chunk in _split_text(rendered, MAX_SECTION_TEXT):
                blocks.append(_section_block(chunk))

        while i < n:
            line = lines[i]

            # Blank line: paragraph boundary
            if not line.strip():
                flush_para()
                i += 1
                continue

            # Fenced code block
            fence = _FENCE_RE.match(line)
            if fence:
                flush_para()
                marker = fence.group(1)
                body: List[str] = []
                i += 1
                while i < n and not lines[i].lstrip().startswith(marker):
                    body.append(lines[i])
                    i += 1
                i += 1  # consume closing fence
                blocks.append(_preformatted_block("\n".join(body)))
                continue

            # Structured-output directive (:::card / :::report / :::carousel).
            # Runs after the code-fence branch on purpose: a fenced example of
            # the syntax must stay literal. A directive that cannot be
            # expressed as its block (unclosed, over a hard cap, malformed)
            # falls back to rendering its inner content through this same
            # pipeline — a directive may restyle content, never lose it.
            dm = _DIRECTIVE_OPEN_RE.match(line)
            if dm:
                scan = _scan_directive(lines, i)
                if scan is None:
                    para.append(line)
                    i += 1
                    continue
                inner, i = scan
                flush_para()
                name = dm.group(1)
                produced: Optional[Block] = None
                if name == "card":
                    produced = _parse_card(inner, fmt)
                elif name == "carousel":
                    produced = _parse_carousel(inner, fmt)
                elif name == "fields":
                    produced = _parse_fields(inner, fmt)
                elif name == "menu":
                    produced = _parse_menu(inner, fmt)
                elif name == "report":
                    produced = _report_block(inner, remaining=markdown_budget)
                    if produced is not None:
                        markdown_budget -= len(produced["text"])
                if produced is not None:
                    blocks.append(produced)
                else:
                    inner_text = _directive_fallback_markdown(name, inner)
                    sub = render_blocks(inner_text, mrkdwn_fn=mrkdwn_fn)
                    if sub:
                        blocks.extend(sub)
                    elif inner_text.strip():
                        # Recursive rendering declined (e.g. too many blocks)
                        # while the outer message still renders — the inner
                        # content must survive as plain sections rather than
                        # vanish (a directive restyles, never loses).
                        for chunk in _split_text(fmt(inner_text), MAX_SECTION_TEXT):
                            blocks.append(_section_block(chunk))
                continue

            # "-# " footer line(s) → one context block (small gray metadata).
            fm = _FOOTER_RE.match(line)
            if fm:
                flush_para()
                footer_lines = [fm.group(1)]
                i += 1
                while i < n:
                    nxt = _FOOTER_RE.match(lines[i])
                    if not nxt:
                        break
                    footer_lines.append(nxt.group(1))
                    i += 1
                blocks.append(
                    {
                        "type": "context",
                        "elements": [{"type": "mrkdwn", "text": fmt("\n".join(footer_lines))}],
                    }
                )
                continue

            # Horizontal rule → divider
            if _HR_RE.match(line):
                flush_para()
                blocks.append(_divider_block())
                i += 1
                continue

            # ATX header
            hm = _HEADER_RE.match(line)
            if hm:
                flush_para()
                header = _header_block(hm.group(2))
                if header is not None:
                    blocks.append(header)
                i += 1
                continue

            # Pipe table: current line has a pipe AND next line is a separator
            if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
                flush_para()
                header_row = line
                sep_line = lines[i + 1]
                trows = [header_row]
                i += 2  # skip header + separator
                while i < n and "|" in lines[i] and lines[i].strip():
                    trows.append(lines[i])
                    i += 1
                # Prefer a native Block Kit table; fall back to aligned
                # monospace when it exceeds Slack's table limits or won't parse.
                table = _table_block(trows, sep_line)
                if table is not None:
                    blocks.append(table)
                else:
                    blocks.append(_preformatted_block(_render_table(trows)))
                continue

            # Blockquote group
            if _QUOTE_RE.match(line):
                flush_para()
                qlines: List[str] = []
                while i < n:
                    qm = _QUOTE_RE.match(lines[i])
                    if not qm:
                        break
                    qlines.append(qm.group(1))
                    i += 1
                blocks.append(_quote_block(qlines))
                continue

            # List group (bullets + ordered, with nesting)
            if _is_list_line(line):
                flush_para()
                items: List[Tuple[int, bool, str]] = []
                while i < n:
                    bm = _BULLET_RE.match(lines[i])
                    om = _ORDERED_RE.match(lines[i])
                    if bm:
                        items.append((_indent_level(bm.group(1)), False, bm.group(2)))
                        i += 1
                    elif om:
                        items.append((_indent_level(om.group(1)), True, om.group(3)))
                        i += 1
                    elif lines[i].strip() and lines[i].startswith((" ", "\t")) and items:
                        # continuation line of the previous item
                        indent, ordered, txt = items[-1]
                        items[-1] = (indent, ordered, txt + " " + lines[i].strip())
                        i += 1
                    elif not lines[i].strip() and items:
                        # Blank line inside a list run. LLM-authored ordered
                        # lists commonly separate items with a blank line; if
                        # the next non-blank line is another list item, treat
                        # the blank(s) as a soft separator and keep the run
                        # going so the items stay in one rich_text_list (Slack
                        # numbers each list independently, so splitting would
                        # restart every item at "1."). Otherwise the blank
                        # ends the list.
                        j = i + 1
                        while j < n and not lines[j].strip():
                            j += 1
                        if j < n and _is_list_line(lines[j]):
                            i = j
                        else:
                            break
                    else:
                        break
                blocks.append(_list_block(items))
                continue

            # Default: accumulate into a paragraph
            para.append(line)
            i += 1

        flush_para()

        if not blocks:
            return None
        # No MAX_BLOCKS decline here: the adapter partitions an over-budget or
        # over-count payload into consecutive posts (``partition_blocks``);
        # declining would degrade the whole message to flat text.
        return blocks
    except Exception:
        # Never let a rendering bug drop a message.
        return None


def _split_text(text: str, limit: int) -> List[str]:
    """Split ``text`` into <= ``limit``-char chunks on line, then hard, boundaries.

    Chunks are fence-balanced: when a split lands inside a ``` code span that
    survived into section text (the renderer normally routes fenced blocks to
    ``rich_text_preformatted``, but mrkdwn text can still carry fences), the
    fence is closed at the end of the chunk and reopened on the next so each
    section renders correctly on its own.
    """
    if len(text) <= limit:
        return [text]
    # Reserve headroom for the close/reopen markers the balancing pass adds.
    split_limit = max(limit - 8, limit // 2, 1) if "```" in text else limit
    out: List[str] = []
    remaining = text
    while len(remaining) > split_limit:
        cut = remaining.rfind("\n", 0, split_limit)
        if cut <= 0:
            cut = split_limit
        out.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        out.append(remaining)
    if len(out) > 1 and "```" in text:
        balanced: List[str] = []
        reopen = False
        for chunk in out:
            if reopen:
                chunk = "```\n" + chunk
            odd = chunk.count("```") % 2 == 1
            if odd:
                chunk += "\n```"
            reopen = odd
            balanced.append(chunk)
        out = balanced
    return out


# ----------------------------------------------------------------------------
# Outbound payload boundary — last-resort clamp before the Slack API
# ----------------------------------------------------------------------------


def _clamp_text_obj(text_obj: Dict[str, Any], limit: int) -> Dict[str, Any]:
    """Return ``text_obj`` with its ``text`` clamped to ``limit`` chars."""
    txt = text_obj.get("text") or ""
    if len(txt) <= limit:
        return text_obj
    clamped = dict(text_obj)
    clamped["text"] = txt[: limit - 1].rstrip() + "…"
    return clamped


def sanitize_blocks(blocks: Optional[List[Block]]) -> Optional[List[Block]]:
    """Clamp an outbound ``blocks`` payload to Slack's hard limits.

    Defensive boundary applied wherever the adapter attaches ``blocks`` to
    ``chat.postMessage`` / ``chat.update``.  One oversized or malformed block
    fails the WHOLE call with ``invalid_blocks`` — approval cards then never
    update and messages silently drop — so instead of trusting every builder,
    the payload is normalized just before the API call:

    * ``section`` / ``context`` text objects are truncated to the 3000-char
      cap with an ellipsis (Slack HTML-escapes ``< > &`` on storage, so text
      echoed back through interaction payloads can exceed the limit that the
      send path originally budgeted for — see #53693 / #62054).
    * ``header`` text is truncated to its 150-char cap.
    * Empty blocks (no text / no elements / no rows) are dropped — Slack
      rejects zero-length text objects and empty element lists.
    * ``table.column_settings`` entries must all be objects; ``null`` entries
      (emitted by older renderers, per the "use null to skip" misreading of
      the schema) are replaced with ``{}`` and default trailing entries are
      trimmed (#56615).
    * The payload is capped at Slack's 50-block maximum.

    Returns the sanitized list, or ``None`` when nothing valid remains — the
    caller then sends the plain ``text`` fallback alone.  Never raises.
    """
    if not blocks:
        return None
    try:
        out: List[Block] = []
        markdown_budget = MAX_MARKDOWN_TEXT
        for block in blocks:
            if not isinstance(block, dict) or not block.get("type"):
                continue
            btype = block["type"]

            if btype == "section":
                text_obj = block.get("text")
                has_body = bool(block.get("fields")) or bool(block.get("accessory"))
                if isinstance(text_obj, dict):
                    if not (text_obj.get("text") or "").strip() and not has_body:
                        continue
                    clamped = _clamp_text_obj(text_obj, MAX_SECTION_TEXT)
                    if clamped is not text_obj:
                        block = dict(block)
                        block["text"] = clamped
                elif not has_body:
                    continue

            elif btype == "header":
                text_obj = block.get("text")
                if not isinstance(text_obj, dict) or not (text_obj.get("text") or "").strip():
                    continue
                clamped = _clamp_text_obj(text_obj, MAX_HEADER_TEXT)
                if clamped is not text_obj:
                    block = dict(block)
                    block["text"] = clamped

            elif btype == "context":
                elements = block.get("elements") or []
                if not elements:
                    continue
                clamped_els = [
                    _clamp_text_obj(el, MAX_SECTION_TEXT)
                    if isinstance(el, dict) and el.get("type") in ("mrkdwn", "plain_text")
                    else el
                    for el in elements
                ]
                if any(c is not e for c, e in zip(clamped_els, elements)):
                    block = dict(block)
                    block["elements"] = clamped_els

            elif btype in ("rich_text", "actions", "context_actions"):
                if not block.get("elements"):
                    continue

            elif btype == "markdown":
                # 12k cap is CUMULATIVE across all markdown blocks per payload.
                txt = block.get("text") or ""
                if not txt.strip() or markdown_budget <= 0:
                    continue
                if len(txt) > markdown_budget:
                    block = dict(block)
                    block["text"] = txt[: markdown_budget - 1].rstrip() + "…"
                markdown_budget -= len(block.get("text") or "")

            elif btype == "card":
                # At least one of title/body/actions/hero_image must remain;
                # clamp the documented field caps (body 200 verified live).
                block = dict(block)
                for field, cap in (
                    ("title", MAX_CARD_TITLE),
                    ("subtitle", MAX_CARD_TITLE),
                    ("body", MAX_CARD_BODY),
                    ("subtext", MAX_CARD_BODY),
                ):
                    obj = block.get(field)
                    if isinstance(obj, dict):
                        if not (obj.get("text") or "").strip():
                            block.pop(field, None)
                        else:
                            block[field] = _clamp_text_obj(obj, cap)
                if isinstance(block.get("actions"), list):
                    block["actions"] = block["actions"][:MAX_CARD_BUTTONS]
                    if not block["actions"]:
                        block.pop("actions", None)
                if not any(
                    block.get(f) for f in ("title", "body", "actions", "hero_image")
                ):
                    continue

            elif btype == "carousel":
                elements = [
                    el
                    for el in (block.get("elements") or [])
                    if isinstance(el, dict) and el.get("type") == "card"
                ]
                if not elements:
                    continue
                block = dict(block)
                block["elements"] = elements[:MAX_CAROUSEL_CARDS]

            elif btype == "table":
                if not block.get("rows"):
                    continue
                settings = block.get("column_settings")
                if isinstance(settings, list) and any(
                    not isinstance(cs, dict) for cs in settings
                ):
                    fixed = [cs if isinstance(cs, dict) else {} for cs in settings]
                    while fixed and not fixed[-1]:
                        fixed.pop()
                    block = dict(block)
                    if fixed:
                        block["column_settings"] = fixed
                    else:
                        block.pop("column_settings", None)

            out.append(block)

        if not out:
            return None
        return out[:MAX_BLOCKS]
    except Exception:
        # A sanitizer bug must never take down the send path.
        return None


# ----------------------------------------------------------------------------
# Payload budget — partition an over-budget block list into consecutive posts
# ----------------------------------------------------------------------------

def payload_size(blocks: Optional[List[Block]]) -> int:
    """Serialized byte size of ``blocks`` as Slack receives them (0 on error)."""
    if not blocks:
        return 0
    try:
        return len(
            json.dumps(blocks, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
    except Exception:
        return 0


def _group_budget(group: List[Block]) -> int:
    return (
        MARKDOWN_GROUP_BUDGET
        if any(b.get("type") == "markdown" for b in group)
        else BLOCK_PAYLOAD_BUDGET
    )


def partition_blocks(
    blocks: Optional[List[Block]],
    budget: Optional[int] = None,
    max_blocks: Optional[int] = None,
) -> List[List[Block]]:
    """Split ``blocks`` into groups Slack accepts as one message each.

    Greedy: a group closes when adding the next block would exceed the byte
    budget (``MARKDOWN_GROUP_BUDGET`` once a group carries a markdown block)
    or ``max_blocks``. A closed group is trimmed back to its last header /
    divider so that lead-in opens the NEXT post rather than trailing this
    one; a trailing divider is dropped. A single block larger than the budget
    travels alone — never split inside a block. Always returns at least one
    group; on any unexpected shape returns ``[blocks]`` unchanged.
    """
    if not blocks:
        return []
    budget = BLOCK_PAYLOAD_BUDGET if budget is None else budget
    max_blocks = MAX_BLOCKS if max_blocks is None else max_blocks
    try:
        groups: List[List[Block]] = []
        cur: List[Block] = []
        for block in blocks:
            # Invariant: ``cur`` always fits. After a back-off the carried
            # tail plus the new block is re-checked through the same test,
            # so a group can never leave here over budget.
            while True:
                candidate = cur + [block]
                limit = min(budget, _group_budget(candidate))
                if not cur or (len(cur) < max_blocks and payload_size(candidate) <= limit):
                    cur = candidate
                    break
                # Prefer opening the next group with the last lead block.
                cut = len(cur)
                for i in range(len(cur) - 1, 0, -1):
                    if cur[i].get("type") in SPLIT_LEAD_TYPES:
                        cut = i
                        break
                head, tail = cur[:cut], cur[cut:]
                while head and head[-1].get("type") == "divider":
                    head.pop()
                if head and tail:
                    groups.append(head)
                    cur = tail  # re-evaluated with ``block`` on the next pass
                else:
                    groups.append(cur)
                    cur = []
        if cur:
            groups.append(cur)
        return groups or [list(blocks)]
    except Exception:
        return [list(blocks)]


def _inline_mrkdwn(elements: Any) -> str:
    out = ""
    for e in elements or []:
        if not isinstance(e, dict):
            continue
        t = e.get("type")
        if t == "text":
            s = str(e.get("text", ""))
            st = e.get("style") or {}
            if st.get("code"):
                s = f"`{s}`"
            if st.get("bold"):
                s = f"*{s}*"
            if st.get("italic"):
                s = f"_{s}_"
            if st.get("strike"):
                s = f"~{s}~"
            out += s
        elif t == "link":
            out += f"<{e.get('url', '')}|{e.get('text') or e.get('url', '')}>"
        elif t == "emoji":
            out += f":{e.get('name', '')}:"
        elif t == "user":
            out += f"<@{e.get('user_id', '')}>"
        elif t == "date":
            out += str(e.get("fallback") or "")
    return out


def group_mrkdwn_text(group: List[Block]) -> str:
    """The WHOLE content of a block group as flat mrkdwn — the body of a
    follow-up that Slack rejected with its blocks. Never truncates; a
    rejected overflow post must still carry its overflow."""
    lines: List[str] = []
    try:
        for b in group:
            t = b.get("type")
            if t == "header":
                lines.append(f"*{b['text']['text']}*")
            elif t == "section":
                if b.get("text"):
                    lines.append(str(b["text"].get("text", "")))
                for f in b.get("fields") or []:
                    lines.append(str(f.get("text", "")))
            elif t == "context":
                lines.append(" ".join(str(e.get("text", "")) for e in b.get("elements", []) if isinstance(e, dict)))
            elif t == "divider":
                lines.append("———")
            elif t == "markdown":
                lines.append(str(b.get("text", "")))
            elif t == "card":
                for k in ("title", "subtitle", "body"):
                    if b.get(k):
                        lines.append(str(b[k].get("text", "")))
            elif t == "rich_text":
                for el in b.get("elements", []):
                    et = el.get("type")
                    if et == "rich_text_list":
                        mark = "1. " if el.get("style") == "ordered" else "• "
                        for sec in el.get("elements", []):
                            lines.append("  " * int(el.get("indent", 0) or 0) + mark + _inline_mrkdwn(sec.get("elements")))
                    elif et == "rich_text_quote":
                        lines.append("> " + _inline_mrkdwn(el.get("elements")))
                    elif et == "rich_text_preformatted":
                        lines.append("```\n" + _inline_mrkdwn(el.get("elements")) + "\n```")
                    else:
                        lines.append(_inline_mrkdwn(el.get("elements")))
            elif t == "table":
                for row in b.get("rows", []):
                    lines.append(" · ".join(_inline_mrkdwn(c.get("elements")) if isinstance(c, dict) else str(c) for c in row))
            lines.append("")
    except Exception:
        pass
    text = "\n".join(lines).strip()
    return text or group_notification_text(group, 0, 1)


def group_notification_text(group: List[Block], index: int, total: int) -> str:
    """Plain words for the ``text`` fallback of a follow-up post."""
    words: List[str] = []

    def _walk(el: Any) -> None:
        if isinstance(el, dict):
            t = el.get("type")
            if t in ("plain_text", "mrkdwn") and isinstance(el.get("text"), str):
                words.append(el["text"])
                return
            if t == "text" and isinstance(el.get("text"), str):
                words.append(el["text"])
                return
            if t == "link":
                words.append(str(el.get("text") or el.get("url") or ""))
                return
            if t == "markdown" and isinstance(el.get("text"), str):
                words.append(el["text"])
                return
            # Native Slack tokens: a date shows its readable fallback, an
            # emoji or a user mention is not a word — without these branches
            # the generic descent below would leak the raw format string.
            if t == "date":
                words.append(str(el.get("fallback") or ""))
                return
            if t in ("emoji", "user"):
                return
            for v in el.values():
                _walk(v)
        elif isinstance(el, list):
            for v in el:
                _walk(v)

    try:
        _walk(group)
        text = " ".join(w.strip() for w in words if w and w.strip())
        text = re.sub(r"[*_~`#]+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
    except Exception:
        text = ""
    if not text:
        return f"(Teil {index + 1}/{total})"
    if len(text) > 600:
        text = text[:599].rstrip() + "…"
    return text
