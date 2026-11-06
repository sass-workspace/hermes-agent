"""The trust gate shows the approver the call's arguments (Iris pin: the CRM write-ask approvals depend on it;
upstream 2b6f15b4a1 delivers it, redacted and capped)."""

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
    assert handlers._trust_gate_check(key, "crm_apply", args=args) is None
    assert '"procedure": "angebot"' in asked[0] and '"amount_eur": 4900' in asked[0]

