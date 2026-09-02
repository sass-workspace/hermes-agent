"""Consecutive identical read-only tool calls must not be re-sent.

A turn that calls ``mcp__asana__get_task(gid="123")`` four times in a row
gets the same payload four times, and every copy stays in the conversation
for the rest of the session — inflating context and, with it, the latency of
every later request.

The dangerous version of this optimisation returns stale data, and unlike
``read_file`` this module has no freshness signal to check: a hosted MCP
server's data can change underneath us at any moment. So it only suppresses
a repeat of the IMMEDIATELY PRECEDING call — where nothing the agent did
could have changed anything, and the staleness window is one model
round-trip. These tests pin that, plus the rules that keep it honest:

  * only tools proven read-only by ``readOnlyHint: true``;
  * any other tool forgets the cache, with a generation bump so a write
    landing mid-call cannot leave a pre-write result behind;
  * errors are never suppressed.
"""
from __future__ import annotations

import json
import threading

import pytest

from tools import turn_read_cache


@pytest.fixture(autouse=True)
def _clean():
    turn_read_cache.reset_for_tests()
    yield
    turn_read_cache.reset_for_tests()


def _suppressed(turn, tool, args):
    """The replacement result, or None when the call should run."""
    return turn_read_cache.check(turn, tool, args)[0]


def _run(turn, tool, args):
    """Simulate a completed read: check, then record."""
    out, gen = turn_read_cache.check(turn, tool, args)
    if out is None:
        turn_read_cache.record(turn, tool, args, gen)
    return out


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_a_first_call_is_never_suppressed():
    assert _suppressed("t1", "get_task", {"gid": "1"}) is None


def test_an_immediate_repeat_is_suppressed():
    _run("t1", "get_task", {"gid": "1"})
    out = _suppressed("t1", "get_task", {"gid": "1"})
    assert out is not None
    parsed = json.loads(out)
    assert parsed["suppressed"] is True
    assert parsed["content_returned"] is False
    assert "directly above" in parsed["message"]


def test_a_repeat_after_another_read_is_NOT_suppressed():
    """The safety property the whole design turns on.

    A re-read after other work is a deliberate re-read — the model has a
    reason, and the world may have moved. Only a back-to-back repeat is
    provably useless.
    """
    _run("t1", "get_task", {"gid": "1"})
    _run("t1", "get_task", {"gid": "2"})
    assert _suppressed("t1", "get_task", {"gid": "1"}) is None


def test_different_arguments_are_not_suppressed():
    _run("t1", "get_task", {"gid": "1"})
    assert _suppressed("t1", "get_task", {"gid": "2"}) is None


def test_argument_order_does_not_defeat_the_match():
    _run("t1", "search", {"a": 1, "b": 2})
    assert _suppressed("t1", "search", {"b": 2, "a": 1}) is not None


def test_nested_argument_structures_match_by_value():
    _run("t1", "q", {"filter": {"x": [1, 2], "y": "z"}})
    assert _suppressed("t1", "q", {"filter": {"y": "z", "x": [1, 2]}}) is not None


def test_a_different_tool_with_the_same_arguments_is_not_suppressed():
    _run("t1", "get_task", {"gid": "1"})
    assert _suppressed("t1", "get_project", {"gid": "1"}) is None


def test_arguments_that_cannot_be_keyed_are_never_suppressed():
    """No stable key, no suppression — fall through and run the call."""
    class _Unkeyable:
        def __repr__(self):
            raise RuntimeError("cannot render")

    args = {"obj": _Unkeyable()}
    assert turn_read_cache._call_key("get_task", args) is None
    _run("t1", "get_task", args)
    assert _suppressed("t1", "get_task", args) is None


def test_arguments_are_matched_by_their_serialized_form():
    """Equal-by-value arguments match; unequal ones do not.

    Keys come from a canonical JSON encoding with ``default=str``, so the
    contract is "serializes the same" rather than "is the same object".
    """
    _run("t1", "get_task", {"gid": "1", "opt": None})
    assert _suppressed("t1", "get_task", {"opt": None, "gid": "1"}) is not None

    turn_read_cache.invalidate("t1")
    _run("t1", "get_task", {"gid": "1"})
    assert _suppressed("t1", "get_task", {"gid": 1}) is None


# ---------------------------------------------------------------------------
# Invalidation, including under concurrency
# ---------------------------------------------------------------------------


def test_invalidate_clears_the_last_call():
    """get -> update -> get must NOT answer the third call from the first."""
    _run("t1", "get_task", {"gid": "1"})
    assert _suppressed("t1", "get_task", {"gid": "1"}) is not None

    turn_read_cache.invalidate("t1")
    assert _suppressed("t1", "get_task", {"gid": "1"}) is None


def test_invalidate_does_not_touch_other_turns():
    _run("t1", "get_task", {"gid": "1"})
    _run("t2", "get_task", {"gid": "1"})
    turn_read_cache.invalidate("t1")
    assert _suppressed("t2", "get_task", {"gid": "1"}) is not None


def test_a_write_landing_mid_call_voids_the_recording():
    """Concurrent batches: the read started first but finished last.

    Invalidating on the write is not enough on its own — the in-flight read
    would otherwise record its pre-write result afterwards, and the next
    identical call would be answered from it.
    """
    _out, gen = turn_read_cache.check("t1", "get_task", {"gid": "1"})
    assert _out is None

    # A concurrent update_task lands while the read is still in flight.
    turn_read_cache.invalidate("t1")

    recorded = turn_read_cache.record("t1", "get_task", {"gid": "1"}, gen)
    assert recorded is False, "a pre-write result was cached after the write"
    assert _suppressed("t1", "get_task", {"gid": "1"}) is None


def test_a_read_with_no_intervening_write_records_normally():
    _out, gen = turn_read_cache.check("t1", "get_task", {"gid": "1"})
    assert turn_read_cache.record("t1", "get_task", {"gid": "1"}, gen) is True
    assert _suppressed("t1", "get_task", {"gid": "1"}) is not None


# ---------------------------------------------------------------------------
# Turn scoping
# ---------------------------------------------------------------------------


def test_turns_do_not_share_a_cache():
    _run("t1", "get_task", {"gid": "1"})
    assert _suppressed("t2", "get_task", {"gid": "1"}) is None


def test_an_empty_turn_id_never_suppresses():
    _run("", "get_task", {"gid": "1"})
    assert _suppressed("", "get_task", {"gid": "1"}) is None


def test_finish_turn_forgets_it():
    _run("t1", "get_task", {"gid": "1"})
    turn_read_cache.finish_turn("t1")
    assert _suppressed("t1", "get_task", {"gid": "1"}) is None


def test_the_turn_registry_is_bounded():
    for i in range(turn_read_cache._MAX_TRACKED_TURNS + 20):
        _run(f"turn-{i}", "get_task", {"gid": "1"})
    assert len(turn_read_cache._turns) == turn_read_cache._MAX_TRACKED_TURNS


def test_concurrent_use_does_not_corrupt_state():
    """Threads racing on one turn must leave it coherent, never wedged."""
    def _worker(n):
        for i in range(50):
            _run("t1", "get_task", {"gid": f"{n}-{i}"})
            turn_read_cache.invalidate("t1")

    threads = [threading.Thread(target=_worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Still usable, and the last write left nothing cached.
    assert _suppressed("t1", "get_task", {"gid": "0-0"}) is None
    _run("t1", "get_task", {"gid": "z"})
    assert _suppressed("t1", "get_task", {"gid": "z"}) is not None


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_repeated_ignoring_of_the_stub_escalates_to_a_hard_block():
    """A weak tool-follower must not burn its budget re-asking."""
    _run("t1", "get_task", {"gid": "1"})
    for _ in range(turn_read_cache._HARD_BLOCK_AFTER):
        out = json.loads(_suppressed("t1", "get_task", {"gid": "1"}))
        assert "error" not in out

    blocked = json.loads(_suppressed("t1", "get_task", {"gid": "1"}))
    assert "BLOCKED" in blocked["error"]
    assert blocked["suppressed"] is True


# ---------------------------------------------------------------------------
# Read-only classification
# ---------------------------------------------------------------------------


class TestReadOnlyClassification:
    @pytest.fixture
    def mcp(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools import mcp_tool

        yield mcp_tool
        mcp_tool._mcp_tool_server_names.pop("mcp__asana__get_task", None)
        mcp_tool._tool_read_only_hints.pop("asana", None)

    def _install(self, mcp, read_only):
        mcp._mcp_tool_server_names["mcp__asana__get_task"] = "asana"
        mcp._tool_read_only_hints["asana"] = {"get_task": read_only}

    def test_an_annotated_read_only_tool_qualifies(self, mcp):
        self._install(mcp, True)
        assert turn_read_cache.is_read_only_tool("mcp__asana__get_task") is True

    def test_a_write_capable_tool_does_not(self, mcp):
        self._install(mcp, False)
        assert turn_read_cache.is_read_only_tool("mcp__asana__get_task") is False

    def test_a_missing_annotation_fails_closed(self, mcp):
        """Same fail-closed rule the trust gate uses."""
        mcp._mcp_tool_server_names["mcp__asana__get_task"] = "asana"
        mcp._tool_read_only_hints["asana"] = {}
        assert turn_read_cache.is_read_only_tool("mcp__asana__get_task") is False

    def test_a_truthy_non_true_annotation_fails_closed(self, mcp):
        self._install(mcp, "yes")
        assert turn_read_cache.is_read_only_tool("mcp__asana__get_task") is False

    def test_an_unknown_tool_is_not_read_only(self, mcp):
        assert turn_read_cache.is_read_only_tool("some_random_tool") is False

    def test_a_deregistered_tool_fails_closed(self, mcp):
        """After a park, the live provenance map is gone — never guess."""
        self._install(mcp, True)
        mcp._forget_mcp_tool_server("mcp__asana__get_task")
        assert turn_read_cache.is_read_only_tool("mcp__asana__get_task") is False

    def test_core_file_tools_are_excluded(self, mcp):
        """read_file/search_files keep their own mtime-aware dedup."""
        assert turn_read_cache.is_read_only_tool("read_file") is False
        assert turn_read_cache.is_read_only_tool("search_files") is False


# ---------------------------------------------------------------------------
# End-to-end through the dispatcher
# ---------------------------------------------------------------------------


def _register(name, handler):
    from tools.registry import registry

    registry.register(
        name=name,
        toolset="testing",
        schema={
            "name": name,
            "description": "test only",
            "parameters": {"type": "object", "properties": {}},
        },
        handler=handler,
    )


def _drop(name):
    from tools.registry import registry

    registry._tools.pop(name, None)


@pytest.fixture
def read_only_probe(monkeypatch):
    """A registered tool the cache will classify as read-only."""
    calls = {"n": 0}

    from tools.registry import tool_result

    def _handler(args, **kw):
        calls["n"] += 1
        return tool_result(payload="x" * 500, call=calls["n"])

    _register("_ro_probe", _handler)
    monkeypatch.setattr(
        turn_read_cache, "is_read_only_tool", lambda name: name == "_ro_probe"
    )
    yield calls
    _drop("_ro_probe")


def test_dispatcher_suppresses_an_immediate_repeat(read_only_probe):
    from model_tools import handle_function_call

    first = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T"))
    assert first["call"] == 1

    second = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T"))
    assert second["suppressed"] is True
    assert read_only_probe["n"] == 1, "the tool ran a second time"
    # The payload is not re-sent — that is the whole point.
    assert "x" * 500 not in json.dumps(second)


def test_dispatcher_does_not_suppress_after_another_call(read_only_probe):
    from model_tools import handle_function_call

    handle_function_call("_ro_probe", {"q": 1}, turn_id="T")
    handle_function_call("_ro_probe", {"q": 2}, turn_id="T")
    out = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T"))
    assert out["call"] == 3
    assert read_only_probe["n"] == 3


def test_dispatcher_does_not_suppress_across_turns(read_only_probe):
    from model_tools import handle_function_call

    handle_function_call("_ro_probe", {"q": 1}, turn_id="T1")
    out = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T2"))
    assert out["call"] == 2


def test_a_write_between_reads_defeats_suppression(read_only_probe):
    """The staleness guard, end to end."""
    from model_tools import handle_function_call
    from tools.registry import tool_result

    _register("_rw_probe", lambda args, **kw: tool_result(ok=True))
    try:
        handle_function_call("_ro_probe", {"q": 1}, turn_id="T")
        handle_function_call("_rw_probe", {}, turn_id="T")
        out = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T"))
        assert out["call"] == 2, "answered a post-write read from a pre-write result"
        assert read_only_probe["n"] == 2
    finally:
        _drop("_rw_probe")


def test_a_failed_read_is_not_suppressed(monkeypatch):
    """A failed call has to stay retryable."""
    from model_tools import handle_function_call
    from tools.registry import tool_error, tool_result

    state = {"n": 0}

    def _flaky(args, **kw):
        state["n"] += 1
        if state["n"] == 1:
            return tool_error("upstream hiccup")
        return tool_result(ok=True)

    _register("_flaky_probe", _flaky)
    monkeypatch.setattr(
        turn_read_cache, "is_read_only_tool", lambda name: name == "_flaky_probe"
    )
    try:
        first = json.loads(handle_function_call("_flaky_probe", {}, turn_id="T"))
        assert "error" in first
        second = json.loads(handle_function_call("_flaky_probe", {}, turn_id="T"))
        assert second.get("ok") is True, "the retry was answered from the failure"
        assert state["n"] == 2
    finally:
        _drop("_flaky_probe")


def test_write_capable_tools_are_never_suppressed():
    from model_tools import handle_function_call
    from tools.registry import tool_result

    calls = {"n": 0}

    def _handler(args, **kw):
        calls["n"] += 1
        return tool_result(n=calls["n"])

    _register("_write_probe", _handler)
    try:
        handle_function_call("_write_probe", {"q": 1}, turn_id="T")
        out = json.loads(handle_function_call("_write_probe", {"q": 1}, turn_id="T"))
        assert out["n"] == 2
        assert calls["n"] == 2
    finally:
        _drop("_write_probe")


# ---------------------------------------------------------------------------
# "Immediately preceding" must hold even when the intervening call FAILS
# ---------------------------------------------------------------------------


def test_a_failed_intervening_read_breaks_the_chain():
    """A -> failed B -> A must run, not be answered from A's old result.

    Failed reads are deliberately never recorded, so the chain cannot be
    maintained by recording alone: `check` has to forget on a mismatch.
    """
    _run("t1", "get_task", {"gid": "1"})

    # B is attempted and fails, so nothing is recorded for it.
    out, _gen = turn_read_cache.check("t1", "get_task", {"gid": "2"})
    assert out is None

    assert _suppressed("t1", "get_task", {"gid": "1"}) is None, (
        "answered a read from a result that was no longer the previous call"
    )


def test_a_failed_intervening_read_also_resets_the_hit_count():
    """B's hits must not be carried into A and trip A's hard block early."""
    _run("t1", "get_task", {"gid": "1"})
    _suppressed("t1", "get_task", {"gid": "1"})  # 1 hit on A

    turn_read_cache.check("t1", "get_task", {"gid": "2"})  # B, fails

    _run("t1", "get_task", {"gid": "1"})
    first_repeat = json.loads(_suppressed("t1", "get_task", {"gid": "1"}))
    assert "error" not in first_repeat, "hard block fired on the first repeat"


def test_the_dispatcher_does_not_suppress_after_a_failed_read(monkeypatch):
    """The same property end to end."""
    from model_tools import handle_function_call
    from tools.registry import tool_error, tool_result

    calls = {"n": 0}

    def _handler(args, **kw):
        calls["n"] += 1
        if args.get("boom"):
            return tool_error("upstream hiccup")
        return tool_result(n=calls["n"])

    _register("_chain_probe", _handler)
    monkeypatch.setattr(
        turn_read_cache, "is_read_only_tool", lambda name: name == "_chain_probe"
    )
    try:
        handle_function_call("_chain_probe", {"q": 1}, turn_id="T")
        handle_function_call("_chain_probe", {"boom": True}, turn_id="T")
        out = json.loads(handle_function_call("_chain_probe", {"q": 1}, turn_id="T"))
        assert "suppressed" not in out
        assert calls["n"] == 3
    finally:
        _drop("_chain_probe")


# ---------------------------------------------------------------------------
# The concurrent execution path has its own direct-dispatch ladder
# ---------------------------------------------------------------------------


def test_the_concurrent_path_invalidates_for_a_direct_mutator(monkeypatch):
    """`invoke_tool` runs delegate_task and friends without ever reaching
    handle_function_call, so it must apply the invalidate-on-write rule too.

    Driven through the real `invoke_tool`, because the sequential executor's
    fix did nothing for this path.
    """
    from agent.agent_runtime_helpers import invoke_tool

    class _Agent:
        _current_turn_id = "T"
        _todo_store = None

        def _should_emit_quiet_tool_messages(self):
            return False

    turn_read_cache.record("T", "get_task", {"gid": "1"}, 0)
    assert _suppressed("T", "get_task", {"gid": "1"}) is not None

    # `tour` is one of the ladder's direct branches; any of them proves the
    # rule is applied before the ladder rather than per-tool.
    monkeypatch.setattr(
        turn_read_cache, "is_read_only_tool", lambda name: False
    )
    try:
        invoke_tool(_Agent(), "tour", {}, "task-1")
    except Exception:
        # The tool itself may fail in this bare harness; the invalidation
        # happens before the ladder, which is what is under test.
        pass

    assert _suppressed("T", "get_task", {"gid": "1"}) is None, (
        "the concurrent path ran a direct mutator without invalidating"
    )


def test_the_concurrent_path_does_not_invalidate_for_read_only_tools(monkeypatch):
    """Invalidating unconditionally there would disable the feature."""
    from agent.agent_runtime_helpers import invoke_tool
    from tools.registry import tool_result

    class _Agent:
        _current_turn_id = "T"
        valid_tool_names = {"_ro_conc_probe"}
        session_id = ""
        enabled_toolsets = None
        disabled_toolsets = None
        _current_api_request_id = ""

    _register("_ro_conc_probe", lambda args, **kw: tool_result(ok=True))
    monkeypatch.setattr(
        turn_read_cache,
        "is_read_only_tool",
        lambda name: name == "_ro_conc_probe",
    )
    try:
        turn_read_cache.record("T", "get_task", {"gid": "1"}, 0)
        try:
            invoke_tool(_Agent(), "_ro_conc_probe", {}, "task-1")
        except Exception:
            # This bare harness cannot satisfy everything the full dispatch
            # path wants. The invalidate decision is made before the ladder,
            # which is the whole of what this test asserts.
            pass
        assert _suppressed("T", "get_task", {"gid": "1"}) is not None, (
            "a read-only tool wiped the cache — suppression would never work"
        )
    finally:
        _drop("_ro_conc_probe")


# ---------------------------------------------------------------------------
# The shared policy helper, and the paths that reach the registry directly
# ---------------------------------------------------------------------------


def test_note_tool_dispatch_invalidates_for_a_write(monkeypatch):
    monkeypatch.setattr(turn_read_cache, "is_read_only_tool", lambda name: False)
    _run("T", "get_task", {"gid": "1"})
    turn_read_cache.note_tool_dispatch("update_task", "T")
    assert _suppressed("T", "get_task", {"gid": "1"}) is None


def test_note_tool_dispatch_leaves_read_only_tools_alone(monkeypatch):
    """Invalidating for reads too would disable suppression everywhere."""
    monkeypatch.setattr(turn_read_cache, "is_read_only_tool", lambda name: True)
    _run("T", "get_task", {"gid": "1"})
    turn_read_cache.note_tool_dispatch("mcp__asana__get_task", "T")
    assert _suppressed("T", "get_task", {"gid": "1"}) is not None


def test_note_tool_dispatch_without_a_turn_id_clears_every_turn(monkeypatch):
    """A caller that cannot identify its turn must still be safe.

    PluginContext.dispatch_tool is the real case: a plugin may dispatch from a
    hook that runs before the turn id is bound, or from background code with
    no turn at all.
    """
    monkeypatch.setattr(turn_read_cache, "is_read_only_tool", lambda name: False)
    _run("T1", "get_task", {"gid": "1"})
    _run("T2", "get_task", {"gid": "1"})

    turn_read_cache.note_tool_dispatch("delegate_task", "")

    assert _suppressed("T1", "get_task", {"gid": "1"}) is None
    assert _suppressed("T2", "get_task", {"gid": "1"}) is None


def test_invalidate_all_bumps_generations_so_in_flight_reads_are_voided(
    monkeypatch,
):
    _out, gen = turn_read_cache.check("T1", "get_task", {"gid": "1"})
    turn_read_cache.invalidate_all()
    assert turn_read_cache.record("T1", "get_task", {"gid": "1"}, gen) is False


def test_plugin_dispatch_invalidates_the_read_cache(monkeypatch):
    """The entry point round 6 found: PluginContext reaches the registry directly.

    Driven through the real `PluginContext.dispatch_tool`, because none of the
    other dispatch paths' fixes touch it.
    """
    from hermes_cli.plugins import PluginContext
    from tools.registry import registry, tool_result

    _register("_plugin_write_probe", lambda args, **kw: tool_result(ok=True))
    monkeypatch.setattr(turn_read_cache, "is_read_only_tool", lambda name: False)

    class _Manager:
        scope_key = None
        _cli_ref = None

    api = PluginContext.__new__(PluginContext)
    api._manager = _Manager()
    try:
        _run("T", "get_task", {"gid": "1"})
        assert _suppressed("T", "get_task", {"gid": "1"}) is not None

        api.dispatch_tool("_plugin_write_probe", {})

        assert _suppressed("T", "get_task", {"gid": "1"}) is None, (
            "a plugin mutated through the registry without invalidating"
        )
    finally:
        _drop("_plugin_write_probe")
