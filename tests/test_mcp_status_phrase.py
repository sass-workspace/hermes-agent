"""MCP tool identifiers must never reach a user-facing status surface.

``build_status_phrase`` feeds Slack's ``assistant.threads.setStatus`` line,
which sits next to the bot's display name in the thread. Before this map, any
tool without a built-in verb fell through to ``f"is using {tool_name}"``, so an
Asana call surfaced as ``is using mcp__asana_abxpert__asana_get_task`` — an
internal connector name, and in this profile a partner-agency one. Slack's own
agent design guidance asks for plain language ("Looking up your calendar"),
never an API endpoint.

The invariant under test is deliberately broad: for ANY ``mcp__*`` tool name,
mapped or not, the phrase must be human-readable and must not contain the
transport identifier.
"""

from __future__ import annotations

import pytest

from agent.display import build_status_phrase, mcp_status_phrase


# Every tool exposed by the agency profile's 13 MCP servers, plus the shapes
# that must degrade gracefully.
MAPPED = [
    ("mcp__asana_abxpert__asana_get_task", "liest die Asana-Aufgabe"),
    ("mcp__asana__asana_search_tasks", "sucht in Asana"),
    ("mcp__asana_edubily__asana_get_stories_for_task", "liest den Asana-Verlauf"),
    ("mcp__asana_write__asana_assign_task", "weist die Asana-Aufgabe zu"),
    ("mcp__asana_write__asana_move_task_to_section", "verschiebt die Asana-Aufgabe"),
    ("mcp__jira__getJiraIssue", "liest das Jira-Ticket"),
    ("mcp__jira__searchJiraIssuesUsingJql", "sucht in Jira"),
    ("mcp__jira__transitionJiraIssue", "setzt den Jira-Status"),
    ("mcp__jira__addCommentToJiraIssue", "schreibt den Jira-Kommentar"),
    ("mcp__jira_backlog__jira_move_to_backlog", "verschiebt ins Jira-Backlog"),
    ("mcp__tempo__tempo_get_worklogs", "liest die Tempo-Zeiten"),
    ("mcp__tempo_write__tempo_create_worklog", "bucht die Zeit in Tempo"),
    ("mcp__billflow__get_invoice", "liest die Rechnung"),
    ("mcp__gcal__calendar_create_event", "legt den Termin an"),
    ("mcp__exec__exec_launch", "startet Claude Code"),
    ("mcp__case_memory__case_get", "liest den Fall"),
    ("mcp__higgsfield__creative_generate", "generiert das Motiv"),
]


@pytest.mark.parametrize("tool,expected", MAPPED)
def test_mapped_tools_get_their_phrase(tool: str, expected: str) -> None:
    assert mcp_status_phrase(tool) == expected


@pytest.mark.parametrize(
    "tool,expected",
    [
        # Unmapped tool on a known server falls back to the server phrase.
        ("mcp__asana__asana_some_future_tool", "arbeitet in Asana"),
        ("mcp__billflow__get_brand_new_thing", "arbeitet mit der Abrechnung"),
        ("mcp__case_memory__case_future", "arbeitet mit dem Fall"),
        # Unknown server stays generic rather than leaking anything.
        ("mcp__whatever__do_thing", "nutzt eine externe Anwendung"),
    ],
)
def test_unmapped_tools_degrade_without_leaking(tool: str, expected: str) -> None:
    assert mcp_status_phrase(tool) == expected


@pytest.mark.parametrize(
    "tool",
    [t for t, _ in MAPPED]
    + [
        "mcp__asana__asana_some_future_tool",
        "mcp__billflow__get_brand_new_thing",
        "mcp__whatever__do_thing",
    ],
)
def test_status_line_never_contains_the_transport_identifier(tool: str) -> None:
    phrase = build_status_phrase(tool, None)
    assert phrase, tool
    assert "mcp__" not in phrase, phrase
    assert not phrase.startswith("is using"), phrase


def test_non_mcp_tools_are_untouched() -> None:
    """The built-in English verbs keep their existing "is <verb>" form."""
    assert mcp_status_phrase("web_search") is None
    assert build_status_phrase("web_search", None) == "is searching the web…"
    assert build_status_phrase("terminal", None) == "is running…"


def test_unknown_non_mcp_tool_keeps_the_generic_fallback() -> None:
    """A plain plugin tool is not an MCP tool and keeps the old behaviour."""
    assert mcp_status_phrase("some_plugin_tool") is None
    assert build_status_phrase("some_plugin_tool", None) == "is using some_plugin_tool…"


def test_phrase_respects_the_slack_status_length_cap() -> None:
    """Slack truncates its status line near 50 characters."""
    for tool, _ in MAPPED:
        phrase = build_status_phrase(tool, None)
        assert len(phrase) <= 49, (tool, phrase, len(phrase))


def test_empty_and_pseudo_tools_return_none() -> None:
    assert mcp_status_phrase("") is None
    assert build_status_phrase("_thinking", None) is None
    assert build_status_phrase("", None) is None
