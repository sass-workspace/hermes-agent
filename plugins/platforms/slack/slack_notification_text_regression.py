#!/usr/bin/env python3
"""Deterministic regression tests for SlackAdapter.notification_text().

Slack shows the ``text`` field — not the blocks — in push notifications,
desktop banners and the sidebar preview, and screen readers read it instead of
block content. These tests pin the two properties that matter:

  * the preview opens on WORDS, never on layout scaffolding, and
  * the full prose survives, because this field is also the accessible reading
    of the message.

Pure function, no network, no Slack client, no adapter state. Run directly:

    python3 slack_notification_text_regression.py
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys
import textwrap
import types
import unicodedata

ADAPTER = pathlib.Path(__file__).with_name("adapter.py")


def _load_functions():
    """Extract format_message + notification_text without importing the adapter.

    Importing the module would pull in slack_sdk, aiohttp and the whole gateway;
    these two methods are pure text transforms, so lifting them out keeps the
    test dependency-free and fast.
    """
    src = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    ns: dict = {"re": re, "unicodedata": unicodedata, "List": list}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.Assign)):
            try:
                exec(compile(ast.Module([node], []), "<adapter>", "exec"), ns)
            except Exception:
                pass  # module-level helpers with unmet deps are not needed here

    wanted = {"format_message", "notification_text"}
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            found[node.name] = textwrap.dedent(ast.get_source_segment(src, node))
    missing = wanted - found.keys()
    if missing:
        sys.exit("adapter.py no longer defines: %s" % ", ".join(sorted(missing)))
    for name in ("format_message", "notification_text"):
        exec(found[name], ns)
    return ns


NS = _load_functions()


class _Self:
    """Minimal stand-in carrying only the constants the method reads."""

    _NOTIFICATION_MAX = 600
    _NOTIFICATION_STATE_EMOJI = "🔴🟡🔵🟢"
    format_message = staticmethod(lambda text: NS["format_message"](_SELF, text))


_SELF = _Self()


def notify(content: str) -> str:
    return NS["notification_text"](_SELF, content)


# ---------------------------------------------------------------------------

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS  %s" % name)
    else:
        FAIL += 1
        print("FAIL  %s%s" % (name, ("  (%s)" % detail) if detail else ""))


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

out = notify(BRIEF)

check("opens on words, not on a marker", out[:1].isalnum(), repr(out[:40]))
check("no heading marker survives", "##" not in out, repr(out))
check("no divider rule survives", "---" not in out, repr(out))
check("no code fence survives", "```" not in out, repr(out))
check("no table pipes survive", "|" not in out, repr(out))
check("table cells are flattened onto one line", "P1 · Turbogrün" in out, repr(out))
check("link labels survive", "TUR-445" in out and "JU-266" in out, repr(out))
check("link URLs are dropped", "elbdev.atlassian.net" not in out, repr(out))
check("prose beyond the first line survives", "Larkson" in out, repr(out))
check("bullet markers are dropped", not re.search(r"(?m)^- ", out), repr(out))

# The two bugs that made the old preview unreadable.
check(
    "raw markdown links never reach the preview",
    "](" not in out,
    repr(out),
)
check("bold markers are gone", "**" not in out, repr(out))

# State emoji are redundant with a word by rule, and Slack shows them as
# :shortcode: in a preview.
emo = notify("🔴 Turbogrün blockiert\n🟢 Inkster erledigt")
check("leading state emoji are stripped", "🔴" not in emo and "🟢" not in emo, repr(emo))
check("the words they marked survive", "blockiert" in emo and "erledigt" in emo, repr(emo))

# Date tokens speak through their fallback text.
dt = notify("Wartet seit <!date^1755388800^{ago}|17.08.> auf Annkatrin.")
check("date token reduces to its fallback", "17.08." in dt, repr(dt))
check("date token syntax is gone", "<!date" not in dt, repr(dt))

# Guard rails.
check("empty input yields empty output", notify("") == "")
check("whitespace-only input yields empty output", notify("   \n\n ") == "")
check(
    "a fence-only message yields empty output so the caller keeps the real text",
    notify("```\nx = 1\n```") == "",
    repr(notify("```\nx = 1\n```")),
)

long_out = notify("Wort " * 400)
check("over-long input is capped", len(long_out) <= 601, str(len(long_out)))
check("a capped preview is marked as truncated", long_out.endswith("…"), repr(long_out[-20:]))

# Escaping must match the normal path — this string goes to Slack as mrkdwn.
esc = notify("Vergleich a < b & c > d")
check("control characters are escaped", "&lt;" in esc and "&amp;" in esc, repr(esc))

print()
print("%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
