"""Status phrases for MCP tools — the transport identifier never reaches the user.

An MCP tool's name IS its transport identifier (``mcp__asana_abxpert__asana_get_task``). A status
line built from it names an internal connector at the user; ``agent.display.build_status_phrase``
asks this module first and falls back to a server-level phrase, so an unmapped or newly added
tool degrades to "arbeitet in Asana" instead of leaking the identifier. Kept out of
``agent/display.py`` so the deployment's phrase table never conflicts with display changes.
"""
from __future__ import annotations

import re

# MCP tools carry their transport identifier as the tool name
# (``mcp__asana_abxpert__asana_get_task``). Surfacing that raw in a status line
# names an internal connector at the user — Slack's own agent guidance asks for
# "Looking up your calendar", not an API endpoint — and for the agency profile
# it is also the partner-agency connector name. These phrases are complete and
# follow the bot's display name, so they are used verbatim rather than through
# the "is <verb>" construction above.
#
# Lookup is by full tool name first, then by the name with the
# ``mcp__<server>__`` prefix stripped; generic names (``search``, ``get_invoice``)
# are keyed fully so two servers cannot claim the same phrase.
_MCP_TOOL_STATUS: dict[str, str] = {
    # --- Asana (all three workspaces share these bare names) ---
    "asana_typeahead_search": "sucht in Asana",
    "asana_search_tasks": "sucht in Asana",
    "asana_get_task": "liest die Asana-Aufgabe",
    "asana_get_tasks": "liest die Asana-Aufgaben",
    "asana_get_stories_for_task": "liest den Asana-Verlauf",
    "asana_get_project": "liest das Asana-Projekt",
    "asana_get_projects": "liest die Asana-Projekte",
    "asana_get_projects_for_workspace": "liest die Asana-Projekte",
    "asana_get_projects_for_team": "liest die Asana-Projekte",
    "asana_get_project_sections": "liest die Asana-Spalten",
    "asana_get_project_status": "liest den Asana-Projektstatus",
    "asana_get_project_statuses": "liest den Asana-Projektstatus",
    "asana_list_workspaces": "liest die Asana-Workspaces",
    "asana_get_user": "sucht die Person in Asana",
    "asana_get_workspace_users": "sucht die Person in Asana",
    "asana_get_team_users": "sucht die Person in Asana",
    "asana_get_teams_for_workspace": "liest die Asana-Teams",
    "asana_get_teams_for_user": "liest die Asana-Teams",
    "asana_get_attachment": "liest den Asana-Anhang",
    "asana_get_attachments_for_object": "liest die Asana-Anhänge",
    "asana_create_task_story": "schreibt den Asana-Kommentar",
    "asana_assign_task": "weist die Asana-Aufgabe zu",
    "asana_move_task_to_section": "verschiebt die Asana-Aufgabe",
    # --- Jira / Atlassian ---
    "getJiraIssue": "liest das Jira-Ticket",
    "searchJiraIssuesUsingJql": "sucht in Jira",
    "getTransitionsForJiraIssue": "prüft die Jira-Status",
    "getJiraIssueRemoteIssueLinks": "liest die Jira-Verknüpfungen",
    "getVisibleJiraProjects": "liest die Jira-Projekte",
    "getJiraProjectIssueTypesMetadata": "liest die Jira-Metadaten",
    "getJiraIssueTypeMetaWithFields": "liest die Jira-Metadaten",
    "getIssueLinkTypes": "liest die Jira-Metadaten",
    "createJiraIssue": "erstellt das Jira-Ticket",
    "transitionJiraIssue": "setzt den Jira-Status",
    "editJiraIssue": "aktualisiert das Jira-Ticket",
    "addCommentToJiraIssue": "schreibt den Jira-Kommentar",
    "jira_move_to_backlog": "verschiebt ins Jira-Backlog",
    "atlassianUserInfo": "prüft den Jira-Zugang",
    "getAccessibleAtlassianResources": "prüft den Jira-Zugang",
    "lookupJiraAccountId": "sucht die Person in Jira",
    "searchConfluenceUsingCql": "sucht in Confluence",
    "getConfluencePage": "liest die Confluence-Seite",
    "getConfluenceSpaces": "liest die Confluence-Bereiche",
    "getPagesInConfluenceSpace": "liest die Confluence-Seiten",
    "getConfluencePageFooterComments": "liest die Confluence-Kommentare",
    "getConfluencePageInlineComments": "liest die Confluence-Kommentare",
    "getConfluenceCommentChildren": "liest die Confluence-Kommentare",
    "getConfluencePageDescendants": "liest die Confluence-Seiten",
    "getCompassComponents": "liest Compass",
    "getCompassComponent": "liest Compass",
    "getCompassCustomFieldDefinitions": "liest Compass",
    "getTeamworkGraphContext": "liest den Teamwork Graph",
    "getTeamworkGraphObject": "liest den Teamwork Graph",
    "mcp__jira__search": "sucht in Atlassian",
    "mcp__jira__fetch": "liest das Atlassian-Dokument",
    # --- Tempo ---
    "tempo_get_worklogs": "liest die Tempo-Zeiten",
    "tempo_get_worklogs_for_issue": "liest die Tempo-Zeiten",
    "tempo_get_accounts": "liest die Tempo-Stammdaten",
    "tempo_get_teams": "liest die Tempo-Stammdaten",
    "tempo_get_work_attributes": "liest die Tempo-Stammdaten",
    "tempo_create_worklog": "bucht die Zeit in Tempo",
    # --- BillFlow (generic bare names, keyed fully) ---
    "mcp__billflow__get_financial_summary": "prüft die Abrechnung",
    "mcp__billflow__get_client_revenue": "prüft den Kundenumsatz",
    "mcp__billflow__get_payment_behaviour": "prüft das Zahlungsverhalten",
    "mcp__billflow__get_utilization": "prüft die Auslastung",
    "mcp__billflow__get_billing_runs": "prüft die Abrechnungsläufe",
    "mcp__billflow__get_billed_worklogs": "prüft die abgerechneten Zeiten",
    "mcp__billflow__get_ledger_for_issue": "prüft das Ticket-Konto",
    "mcp__billflow__get_invoice": "liest die Rechnung",
    "mcp__billflow__get_billing_reconciliation": "gleicht die Abrechnung ab",
    # --- Calendar / contacts ---
    "contacts_search": "sucht den Kontakt",
    "calendar_list_events": "liest den Kalender",
    "calendar_free_busy": "prüft die Verfügbarkeit",
    "calendar_create_event": "legt den Termin an",
    # --- Claude Code execution ---
    "exec_prepare": "bereitet die Ausführung vor",
    "exec_launch": "startet Claude Code",
    "exec_result": "liest das Ausführungsergebnis",
    "exec_status": "prüft die Ausführung",
    # --- Case Memory ---
    "case_get": "liest den Fall",
    "case_search": "sucht in den Fällen",
    "case_upsert": "aktualisiert den Fall",
    "case_event": "schreibt ins Fallprotokoll",
    "case_evidence": "sichert den Befund",
    "case_mark_promoted": "markiert den Befund als übernommen",
    "case_ingest_execution": "übernimmt das Ausführungsergebnis",
    # --- Higgsfield ---
    "creative_plan": "plant die Generierung",
    "creative_generate": "generiert das Motiv",
    "creative_result": "liest das Generierungsergebnis",
}

# Fallback phrase per MCP server, for tools not named above. Keyed by the
# server segment of ``mcp__<server>__<tool>`` (hyphens become underscores).
# Complete phrases rather than bare labels, so German articles stay correct
# without the caller having to reason about grammar.
_MCP_SERVER_FALLBACKS: dict[str, str] = {
    "asana": "arbeitet in Asana",
    "asana_abxpert": "arbeitet in Asana",
    "asana_edubily": "arbeitet in Asana",
    "asana_write": "arbeitet in Asana",
    "jira": "arbeitet in Jira",
    "jira_backlog": "arbeitet in Jira",
    "tempo": "arbeitet in Tempo",
    "tempo_write": "arbeitet in Tempo",
    "billflow": "arbeitet mit der Abrechnung",
    "gcal": "arbeitet mit dem Kalender",
    "exec": "arbeitet mit Claude Code",
    "case_memory": "arbeitet mit dem Fall",
    "higgsfield": "arbeitet mit Higgsfield",
    "gcal_read": "arbeitet mit dem Kalender",
    "exec_launch": "arbeitet mit Claude Code",
    "exec_queue": "arbeitet mit Claude Code",
    "gmail": "arbeitet im Postfach",
    "gmail_send": "arbeitet im Postfach",
    "gmail_triage": "arbeitet im Postfach",
    "wispr": "liest die Meeting-Notizen",
    "sie": "arbeitet mit den Website-Daten",
    "ablyft": "prüft die Experimente",
    "ops_exceptions": "prüft die offenen Punkte",
    "context_read": "liest den Kundenkontext",
    "attachment_read": "liest den Anhang",
}

_MCP_PREFIX_RE = re.compile(r"^mcp__([a-z0-9_]+)__(.+)$")


def mcp_status_phrase(tool_name: str) -> str | None:
    """Return a complete German status phrase for an MCP tool, or None.

    Falls back to a server-labelled phrase for any tool not named explicitly,
    so a newly added MCP tool degrades to "arbeitet in Asana…" rather than
    leaking ``mcp__asana_abxpert__asana_get_task`` at the user.
    """
    if not tool_name:
        return None
    phrase = _MCP_TOOL_STATUS.get(tool_name)
    if phrase:
        return phrase
    m = _MCP_PREFIX_RE.match(tool_name)
    if not m:
        return None
    server, bare = m.group(1), m.group(2)
    phrase = _MCP_TOOL_STATUS.get(bare)
    if phrase:
        return phrase
    return _MCP_SERVER_FALLBACKS.get(server, "nutzt eine externe Anwendung")
