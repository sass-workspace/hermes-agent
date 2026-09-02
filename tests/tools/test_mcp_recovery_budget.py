"""Auth/reconnect waiting must be bounded per turn.

Each recovery wait was individually reasonable and collectively awful. One
failing MCP tool call could burn, in sequence:

    5s   waiting for a session to reappear
  + 10s  in the OAuth manager's handle_401
  + 15s  waiting on the reconnect that triggers
  + 15s  waiting on the session-expired reconnect when that path declined
  ------
    45s  of pure waiting, before the retry RPC even starts

per call, with nothing capping the total across several failing tools in one
turn. From the model's side that is indistinguishable from a hang.

The waits now share one per-turn budget. These tests pin the arithmetic, the
skip behavior once it is spent, and — importantly — that a spent budget stops
Hermes WAITING without stopping it RECOVERING: the reconnect is still
signalled so the server task rebuilds in the background.
"""
from __future__ import annotations

import json
import threading
import time
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


@pytest.fixture
def mcp(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_tool

    mcp_tool._reset_recovery_budget_for_tests()
    yield mcp_tool
    mcp_tool._reset_recovery_budget_for_tests()


@pytest.fixture
def in_turn(monkeypatch):
    """Bind a turn id, as the tool dispatcher does around every tool call."""
    from tools import approval

    monkeypatch.setattr(approval, "get_current_turn_id", lambda default="": "turn-1")
    return "turn-1"


# ---------------------------------------------------------------------------
# Budget arithmetic
# ---------------------------------------------------------------------------


def test_a_fresh_turn_has_the_full_budget(mcp, in_turn):
    assert mcp._recovery_budget_remaining() == mcp._TURN_RECOVERY_BUDGET_SEC


def test_spending_reduces_the_remaining_budget(mcp, in_turn):
    mcp._charge_recovery_budget(4.0)
    assert mcp._recovery_budget_remaining() == pytest.approx(
        mcp._TURN_RECOVERY_BUDGET_SEC - 4.0
    )
    mcp._charge_recovery_budget(4.0)
    assert mcp._recovery_budget_remaining() == pytest.approx(
        mcp._TURN_RECOVERY_BUDGET_SEC - 8.0
    )


def test_the_budget_floors_at_zero(mcp, in_turn):
    mcp._charge_recovery_budget(1000.0)
    assert mcp._recovery_budget_remaining() == 0.0


def test_the_total_across_many_waits_cannot_exceed_the_budget(mcp, in_turn):
    """The point of the whole change: a bound on the TURN, not per call."""
    granted = []
    for _ in range(20):
        with mcp._recovery_wait("probe", 15.0) as allowed:
            granted.append(allowed)
            mcp._charge_recovery_budget(allowed)
    assert sum(granted) <= mcp._TURN_RECOVERY_BUDGET_SEC + 0.5
    assert granted[0] == mcp._TURN_RECOVERY_BUDGET_SEC
    assert granted[-1] == 0.0


def test_a_wait_is_clamped_to_what_is_left(mcp, in_turn):
    mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC - 2.0)
    with mcp._recovery_wait("probe", 15.0) as allowed:
        assert allowed == pytest.approx(2.0)


def test_a_wait_never_gets_more_than_it_asked_for(mcp, in_turn):
    with mcp._recovery_wait("probe", 3.0) as allowed:
        assert allowed == 3.0


def test_elapsed_time_inside_the_block_is_charged(mcp, in_turn):
    before = mcp._recovery_budget_remaining()
    with mcp._recovery_wait("probe", 10.0):
        time.sleep(0.2)
    assert before - mcp._recovery_budget_remaining() >= 0.2


def test_a_raising_wait_still_charges_its_budget(mcp, in_turn):
    """A wait that blows up must not leak budget back to the turn."""
    before = mcp._recovery_budget_remaining()
    with pytest.raises(RuntimeError):
        with mcp._recovery_wait("probe", 10.0):
            time.sleep(0.15)
            raise RuntimeError("transport exploded mid-wait")
    assert before - mcp._recovery_budget_remaining() >= 0.15


# ---------------------------------------------------------------------------
# Scoping
# ---------------------------------------------------------------------------


def test_turns_have_independent_budgets(mcp, monkeypatch):
    from tools import approval

    current = {"id": "turn-a"}
    monkeypatch.setattr(
        approval, "get_current_turn_id", lambda default="": current["id"]
    )

    mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
    assert mcp._recovery_budget_remaining() == 0.0

    current["id"] = "turn-b"
    assert mcp._recovery_budget_remaining() == mcp._TURN_RECOVERY_BUDGET_SEC


def test_calls_outside_a_turn_are_never_starved(mcp, monkeypatch):
    """Startup discovery and CLI paths have no turn to protect."""
    from tools import approval

    monkeypatch.setattr(approval, "get_current_turn_id", lambda default="": "")
    mcp._charge_recovery_budget(1000.0)
    assert mcp._recovery_budget_remaining() == mcp._TURN_RECOVERY_BUDGET_SEC


def test_the_registry_is_bounded(mcp, monkeypatch):
    from tools import approval

    current = {"id": ""}
    monkeypatch.setattr(
        approval, "get_current_turn_id", lambda default="": current["id"]
    )
    for i in range(mcp._MAX_TRACKED_RECOVERY_TURNS + 30):
        current["id"] = f"turn-{i}"
        mcp._charge_recovery_budget(1.0)
    assert len(mcp._turn_recovery_spent) == mcp._MAX_TRACKED_RECOVERY_TURNS


def test_concurrent_charges_are_serialised(mcp, in_turn):
    def _worker():
        for _ in range(100):
            mcp._charge_recovery_budget(0.01)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    spent = mcp._TURN_RECOVERY_BUDGET_SEC - mcp._recovery_budget_remaining()
    assert spent == pytest.approx(8.0, abs=0.01)


# ---------------------------------------------------------------------------
# Behavior at the call sites
# ---------------------------------------------------------------------------


def test_a_spent_budget_skips_the_session_ready_wait(mcp, in_turn, monkeypatch):
    """The regression: the turn must not sit on a wait it cannot afford."""
    waited = {"n": 0}

    def _never_ready(*a, **kw):
        waited["n"] += 1
        time.sleep(kw.get("timeout", 0.0))
        return False

    monkeypatch.setattr(mcp, "_wait_for_server_session_ready", _never_ready)
    monkeypatch.setattr(mcp, "_trust_gate_check", lambda *a, **kw: None)

    server = MagicMock()
    server.name = "srv"
    server.session = None
    server._is_recycled_stdio.return_value = False
    mcp._servers["srv"] = server
    monkeypatch.setattr(mcp, "_signal_reconnect", lambda s: True)

    try:
        mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
        handler = mcp._make_tool_handler("srv", "op", 60.0)
        started = time.monotonic()
        out = json.loads(handler({}))
        elapsed = time.monotonic() - started

        assert waited["n"] == 0, "waited on a budget that was already spent"
        assert elapsed < 1.0
        assert "reconnect" in out["error"].lower()
    finally:
        mcp._servers.pop("srv", None)
        mcp._server_error_counts.pop("srv", None)
        mcp._server_breaker_opened_at.pop("srv", None)


def test_a_spent_budget_still_signals_the_reconnect(mcp, in_turn, monkeypatch):
    """Stop WAITING, not stop RECOVERING — the rebuild continues in the
    background, so the next turn finds a healthy server."""
    monkeypatch.setattr(
        mcp, "_wait_for_server_session_ready", lambda *a, **kw: False
    )
    monkeypatch.setattr(mcp, "_trust_gate_check", lambda *a, **kw: None)
    signalled = {"n": 0}

    def _signal(server):
        signalled["n"] += 1
        return True

    monkeypatch.setattr(mcp, "_signal_reconnect", _signal)

    server = MagicMock()
    server.name = "srv2"
    server.session = None
    server._is_recycled_stdio.return_value = False
    mcp._servers["srv2"] = server
    try:
        mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
        handler = mcp._make_tool_handler("srv2", "op", 60.0)
        handler({})
        assert signalled["n"] == 1
    finally:
        mcp._servers.pop("srv2", None)
        mcp._server_error_counts.pop("srv2", None)
        mcp._server_breaker_opened_at.pop("srv2", None)


def test_a_spent_budget_short_circuits_oauth_recovery(mcp, in_turn, monkeypatch):
    """handle_401 must not be entered with no time to wait on it."""
    monkeypatch.setattr(mcp, "_is_auth_error", lambda exc: True)
    called = {"n": 0}

    def _loop(*a, **kw):
        called["n"] += 1
        return True

    monkeypatch.setattr(mcp, "_run_on_mcp_loop", _loop)

    mcp._charge_recovery_budget(mcp._TURN_RECOVERY_BUDGET_SEC)
    out = mcp._handle_auth_error_and_retry(
        "srv3", Exception("401"), lambda: "{}", "tools/call op",
    )
    try:
        parsed = json.loads(out)
        assert parsed.get("needs_reauth") is True
        assert called["n"] == 0, "entered OAuth recovery with no budget"
    finally:
        mcp._server_error_counts.pop("srv3", None)
        mcp._server_breaker_opened_at.pop("srv3", None)


def test_recovery_still_runs_normally_with_budget_available(mcp, in_turn, monkeypatch):
    """The bound must not disable recovery on an ordinary first failure."""
    monkeypatch.setattr(mcp, "_is_auth_error", lambda exc: True)
    monkeypatch.setattr(mcp, "_run_on_mcp_loop", lambda *a, **kw: True)
    monkeypatch.setattr(mcp, "_signal_reconnect_and_wait", lambda *a, **kw: True)

    class _Manager:
        async def handle_401(self, *a, **kw):
            return True

    monkeypatch.setattr(
        "tools.mcp_oauth_manager.get_manager", lambda *a, **kw: _Manager()
    )
    server = MagicMock()
    server._reconnect_event = MagicMock()
    mcp._servers["srv4"] = server
    try:
        out = mcp._handle_auth_error_and_retry(
            "srv4", Exception("401"),
            lambda: json.dumps({"result": "ok"}), "tools/call op",
        )
        assert json.loads(out)["result"] == "ok"
    finally:
        mcp._servers.pop("srv4", None)
        mcp._server_error_counts.pop("srv4", None)


def test_worst_case_turn_impact_stays_within_the_budget(mcp, in_turn, monkeypatch):
    """Drive every recovery wait back to back and measure the total.

    Each site sleeps for whatever it is granted, so the elapsed wall clock is
    the turn impact this change exists to bound.
    """
    # Scale the real budget down rather than sleeping through 15s of it: the
    # mechanism under test is the shared cap, not its exact value.
    monkeypatch.setattr(mcp, "_TURN_RECOVERY_BUDGET_SEC", 2.0)

    def _sleepy_wait(srv, *, old_session=None, timeout=15.0):
        time.sleep(min(float(timeout), 20.0))
        return False

    monkeypatch.setattr(mcp, "_wait_for_server_session_ready", _sleepy_wait)

    started = time.monotonic()
    for op, requested in [
        ("session-ready", 5.0),
        ("oauth-recovery", 10.0),
        ("oauth-reconnect", 15.0),
        ("session-expired-reconnect", 15.0),
    ] * 3:
        with mcp._recovery_wait(op, requested) as allowed:
            if allowed > 0:
                time.sleep(min(allowed, 20.0))
    elapsed = time.monotonic() - started

    # Pre-fix this sequence would have slept 45s per round, 135s in total.
    assert elapsed <= mcp._TURN_RECOVERY_BUDGET_SEC + 1.0, (
        f"a turn spent {elapsed:.1f}s waiting on MCP recovery; the budget is "
        f"{mcp._TURN_RECOVERY_BUDGET_SEC:.0f}s"
    )


# ---------------------------------------------------------------------------
# The wiring that makes the budget real
# ---------------------------------------------------------------------------


def test_the_turn_key_reaches_a_tool_handler_for_real(mcp, tmp_path, monkeypatch):
    """Without this, the entire budget is a silent no-op.

    The turn key comes from a contextvar the tool dispatcher binds around
    dispatch. MCP tool handlers run inside that dispatch — but if the
    contextvar were not visible there (a different thread, a lost context),
    every budget lookup would fall into the "no turn to protect" branch, the
    cap would never apply, and nothing would fail: waits would simply go back
    to being unbounded. So exercise the REAL dispatcher, with no
    monkeypatching of the turn id.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from model_tools import handle_function_call
    from tools.registry import registry, tool_result

    seen = {}

    def _handler(args, **kw):
        seen["key"] = mcp._current_recovery_turn_key()
        seen["before"] = mcp._recovery_budget_remaining()
        mcp._charge_recovery_budget(5.0)
        seen["after"] = mcp._recovery_budget_remaining()
        return tool_result(ok=True)

    registry.register(
        name="_budget_wiring_probe",
        toolset="testing",
        schema={
            "name": "_budget_wiring_probe",
            "description": "test only",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=_handler,
    )
    try:
        handle_function_call("_budget_wiring_probe", {}, turn_id="TURN-XYZ")

        assert seen["key"] == "TURN-XYZ", (
            "the turn key is not visible inside a tool handler — the recovery "
            "budget would never apply to any MCP wait"
        )
        assert seen["before"] == mcp._TURN_RECOVERY_BUDGET_SEC
        # And the charge is scoped to that turn, not discarded.
        assert seen["after"] == pytest.approx(
            mcp._TURN_RECOVERY_BUDGET_SEC - 5.0
        )
    finally:
        registry._tools.pop("_budget_wiring_probe", None)
