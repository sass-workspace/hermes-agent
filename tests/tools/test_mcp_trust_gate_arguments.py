"""The trust gate shows the approver the call's arguments (Iris fork, iris/api-additions)."""

import tools.approval_prompt as approval_prompt
from tools import mcp_tool_handlers as handlers
from tools import mcp_tool as core


def test_the_prompt_names_exactly_what_runs(monkeypatch):
    asked = []
    monkeypatch.setattr(approval_prompt, "request_elicitation_consent",
                        lambda message, description, **kw: asked.append(message) or "accept")
    monkeypatch.setattr(handlers, "_tool_is_read_only", lambda server, tool: False)
    key = "crm-write-ask"
    import tools.mcp_tool_scope as scope
    monkeypatch.setattr(scope, "_server_key", lambda name: name)
    monkeypatch.setitem(core._server_trust_levels, key, core._TRUST_UNTRUSTED)
    args = {"procedure": "angebot", "outreach_id": "0f1e", "params": {"package": "AUDIT", "amount_eur": 4900}}
    assert handlers._trust_gate_check(key, "crm_apply", args) is None
    assert '"procedure": "angebot"' in asked[0] and '"amount_eur": 4900' in asked[0]


def test_long_arguments_are_cut_and_say_so():
    text = handlers._gate_arguments({"note": "x" * 10_000})
    assert len(text) < 3200 and "cut; the full arguments run" in text
    assert handlers._gate_arguments(None) == ""
