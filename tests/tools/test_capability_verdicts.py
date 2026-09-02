"""A vanished MCP tool must not be reported as a missing capability.

When an MCP server parks (reconnect budget exhausted) or fails to connect,
``_deregister_tools()`` pulls its tools out of the registry. The model's
schema, however, is byte-stable for the life of a conversation — that is a
prompt-cache invariant, not an oversight — so it goes on calling them.

It used to get back ``{"error": "Unknown tool: mcp__asana__create_task"}``,
read that as proof the capability does not exist, tell the user Hermes cannot
do that thing, and never try again. A transport outage lasting seconds became
a permanent-looking loss of capability — and for OAuth servers, which had
effectively no self-recovery after parking, permanently permanent.

The verdict must answer a different question than "is this name in the
registry": WHICH of reconnecting / parked / in backoff / genuinely unknown is
true.
"""
from __future__ import annotations

import json

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


@pytest.fixture
def mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    mcp_tool._known_mcp_tool_owners.clear()
    yield mcp_tool
    mcp_tool._known_mcp_tool_owners.clear()
    mcp_tool._servers.pop("asana", None)
    mcp_tool._server_connect_retry_after.pop("asana", None)
    mcp_tool._server_connect_failures.pop("asana", None)


class _Server:
    def __init__(self, parked=False):
        self._was_parked = parked


# ---------------------------------------------------------------------------
# The registry hook
# ---------------------------------------------------------------------------


def test_a_genuinely_unknown_name_still_says_unknown():
    """The precise verdict must not blur the case it is not about."""
    from tools.registry import registry

    out = json.loads(registry.dispatch("no_such_tool_anywhere", {}))
    assert out["error"] == "Unknown tool: no_such_tool_anywhere"


def test_the_mcp_resolver_is_registered_at_import():
    import tools.mcp_tool as mcp_tool  # noqa: F401
    from tools.registry import registry

    assert mcp_tool._describe_unknown_mcp_tool in registry._unknown_tool_resolvers


def test_a_resolver_that_raises_falls_back_to_the_plain_verdict():
    """A broken resolver must never break dispatch."""
    from tools.registry import registry

    def _boom(name):
        raise RuntimeError("resolver exploded")

    registry.register_unknown_tool_resolver(_boom)
    try:
        out = json.loads(registry.dispatch("still_unknown_xyz", {}))
        assert out["error"] == "Unknown tool: still_unknown_xyz"
    finally:
        registry._unknown_tool_resolvers.remove(_boom)


def test_a_resolver_returning_none_defers_to_the_next():
    from tools.registry import registry

    def _defer(name):
        return None

    def _answer(name):
        return "precise verdict for " + name

    registry.register_unknown_tool_resolver(_defer)
    registry.register_unknown_tool_resolver(_answer)
    try:
        out = json.loads(registry.dispatch("deferred_tool_xyz", {}))
        assert out["error"] == "precise verdict for deferred_tool_xyz"
    finally:
        registry._unknown_tool_resolvers.remove(_defer)
        registry._unknown_tool_resolvers.remove(_answer)


def test_resolvers_are_not_registered_twice():
    from tools.registry import registry

    def _r(name):
        return None

    registry.register_unknown_tool_resolver(_r)
    registry.register_unknown_tool_resolver(_r)
    try:
        assert registry._unknown_tool_resolvers.count(_r) == 1
    finally:
        registry._unknown_tool_resolvers.remove(_r)


# ---------------------------------------------------------------------------
# Provenance survives deregistration
# ---------------------------------------------------------------------------


def test_provenance_outlives_deregistration(mcp):
    """The whole verdict rests on still knowing who owned the name."""
    mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
    mcp._forget_mcp_tool_server("mcp__asana__create_task")

    # The LIVE map is cleared...
    assert "mcp__asana__create_task" not in mcp._mcp_tool_server_names
    # ...but the association we need is kept.
    assert mcp._known_mcp_tool_owners["mcp__asana__create_task"] == "asana"


def test_the_owner_map_is_bounded(mcp):
    """A server with churning dynamic discovery must not grow it forever."""
    for i in range(mcp._MAX_REMEMBERED_TOOL_OWNERS + 50):
        mcp._track_mcp_tool_server(f"mcp__srv__tool_{i}", "srv")
    assert len(mcp._known_mcp_tool_owners) == mcp._MAX_REMEMBERED_TOOL_OWNERS


# ---------------------------------------------------------------------------
# The verdicts themselves
# ---------------------------------------------------------------------------


def _verdict(mcp, tool="mcp__asana__create_task"):
    return mcp._describe_unknown_mcp_tool(tool)


def test_an_unrelated_name_gets_no_mcp_verdict(mcp):
    assert _verdict(mcp, "read_file") is None


class TestParkedServer:
    @pytest.fixture(autouse=True)
    def _setup(self, mcp):
        mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._forget_mcp_tool_server("mcp__asana__create_task")
        mcp._servers["asana"] = _Server(parked=True)

    def test_it_names_the_server_and_the_state(self, mcp):
        v = _verdict(mcp)
        assert "asana" in v
        assert "parked" in v.lower()

    def test_it_gives_the_self_probe_interval(self, mcp):
        assert str(mcp._PARKED_RETRY_INTERVAL) in _verdict(mcp)

    def test_it_forbids_the_wrong_conclusion(self, mcp):
        """The regression, stated as plainly as the model needs it."""
        v = _verdict(mcp)
        assert "NOT a missing capability" in v
        assert "do NOT tell" in v

    def test_it_never_says_unknown_tool(self, mcp):
        assert "Unknown tool" not in _verdict(mcp)


class TestReconnectingServer:
    def test_a_live_server_reads_as_reconnecting(self, mcp):
        mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._servers["asana"] = _Server(parked=False)
        v = _verdict(mcp)
        assert "reconnect" in v.lower()
        assert "NOT a missing capability" in v
        assert "Retry the SAME call" in v


class TestConnectBackoff:
    def test_it_reports_the_remaining_cooldown(self, mcp):
        import time

        mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._servers["asana"] = _Server(parked=False)
        mcp._server_connect_retry_after["asana"] = time.monotonic() + 42.0
        v = _verdict(mcp)
        assert "backoff" in v.lower()
        assert "s" in v and ("~42" in v or "~41" in v)
        assert "NOT a missing capability" in v


class TestServerGoneFromConfig:
    def test_it_points_at_the_config_instead_of_denying_the_capability(self, mcp):
        mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
        mcp._servers.pop("asana", None)
        v = _verdict(mcp)
        assert "hermes mcp list" in v
        assert "assuming the capability does not exist" in v
        assert "Unknown tool" not in v


# ---------------------------------------------------------------------------
# End-to-end through dispatch
# ---------------------------------------------------------------------------


def test_dispatch_returns_the_precise_verdict_for_a_parked_tool(mcp):
    """What the model actually receives when it calls a parked server's tool."""
    from tools.registry import registry

    mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
    mcp._forget_mcp_tool_server("mcp__asana__create_task")
    mcp._servers["asana"] = _Server(parked=True)

    out = json.loads(registry.dispatch("mcp__asana__create_task", {}))
    assert "Unknown tool" not in out["error"]
    assert "asana" in out["error"]
    assert "NOT a missing capability" in out["error"]


def test_a_registered_tool_is_unaffected(mcp):
    """The resolver must only ever speak for names the registry lacks."""
    from tools.registry import registry, tool_result

    registry.register(
        name="mcp__asana__create_task",
        toolset="testing",
        schema={
            "name": "mcp__asana__create_task",
            "description": "test only",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=lambda args, **kw: tool_result(ok=True),
    )
    mcp._track_mcp_tool_server("mcp__asana__create_task", "asana")
    mcp._servers["asana"] = _Server(parked=True)
    try:
        out = json.loads(registry.dispatch("mcp__asana__create_task", {}))
        assert out.get("ok") is True
    finally:
        registry._tools.pop("mcp__asana__create_task", None)
