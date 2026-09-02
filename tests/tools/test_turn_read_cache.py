"""Identical read-only tool calls inside one turn must not be re-sent.

A turn that calls ``mcp__asana__get_task(gid="123")`` four times gets the same
payload four times, and every copy stays in the conversation for the rest of
the session — inflating context and, with it, the latency of every later
request. The model learned nothing on calls two through four.

The dangerous version of this optimisation returns stale data. These tests
pin the three rules that stop it:

  1. read-only only, proven by ``readOnlyHint: true`` and nothing weaker;
  2. any write invalidates the whole turn's cache;
  3. errors are never suppressed — a failed call stays retryable.
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


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------


def test_a_first_call_is_never_suppressed():
    assert turn_read_cache.check("t1", "get_task", {"gid": "1"}) is None


def test_an_identical_repeat_is_suppressed():
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    out = turn_read_cache.check("t1", "get_task", {"gid": "1"})
    assert out is not None
    parsed = json.loads(out)
    assert parsed["suppressed"] is True
    assert parsed["content_returned"] is False
    assert "refer to it" in parsed["message"]


def test_different_arguments_are_not_suppressed():
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    assert turn_read_cache.check("t1", "get_task", {"gid": "2"}) is None


def test_argument_order_does_not_defeat_the_match():
    turn_read_cache.record("t1", "search", {"a": 1, "b": 2})
    assert turn_read_cache.check("t1", "search", {"b": 2, "a": 1}) is not None


def test_nested_argument_structures_match_by_value():
    turn_read_cache.record("t1", "q", {"filter": {"x": [1, 2], "y": "z"}})
    assert turn_read_cache.check("t1", "q", {"filter": {"y": "z", "x": [1, 2]}}) is not None


def test_a_different_tool_with_the_same_arguments_is_not_suppressed():
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    assert turn_read_cache.check("t1", "get_project", {"gid": "1"}) is None


def test_unserializable_arguments_are_never_suppressed():
    """No key, no suppression — fall through and run the call."""
    class _Weird:
        pass

    args = {"obj": _Weird()}
    turn_read_cache.record("t1", "get_task", args)
    # default=str makes most things serializable; an object whose repr
    # differs per instance simply won't match, which is the safe direction.
    assert turn_read_cache.check("t1", "get_task", {"obj": _Weird()}) is None


# ---------------------------------------------------------------------------
# Rule 2: a write invalidates the turn
# ---------------------------------------------------------------------------


def test_invalidate_clears_the_turn():
    """get -> update -> get must NOT answer the third call from the first."""
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    assert turn_read_cache.check("t1", "get_task", {"gid": "1"}) is not None

    turn_read_cache.invalidate("t1")
    assert turn_read_cache.check("t1", "get_task", {"gid": "1"}) is None


def test_invalidate_does_not_touch_other_turns():
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    turn_read_cache.record("t2", "get_task", {"gid": "1"})
    turn_read_cache.invalidate("t1")
    assert turn_read_cache.check("t2", "get_task", {"gid": "1"}) is not None


# ---------------------------------------------------------------------------
# Turn scoping
# ---------------------------------------------------------------------------


def test_turns_do_not_share_a_cache():
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    assert turn_read_cache.check("t2", "get_task", {"gid": "1"}) is None


def test_an_empty_turn_id_never_suppresses():
    turn_read_cache.record("", "get_task", {"gid": "1"})
    assert turn_read_cache.check("", "get_task", {"gid": "1"}) is None


def test_finish_turn_forgets_it():
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    turn_read_cache.finish_turn("t1")
    assert turn_read_cache.check("t1", "get_task", {"gid": "1"}) is None


def test_the_turn_registry_is_bounded():
    for i in range(turn_read_cache._MAX_TRACKED_TURNS + 20):
        turn_read_cache.record(f"turn-{i}", "get_task", {"gid": "1"})
    assert len(turn_read_cache._turn_reads) == turn_read_cache._MAX_TRACKED_TURNS


def test_the_per_turn_key_count_is_bounded():
    for i in range(turn_read_cache._MAX_KEYS_PER_TURN + 20):
        turn_read_cache.record("t1", "get_task", {"gid": str(i)})
    assert len(turn_read_cache._turn_reads["t1"]) == turn_read_cache._MAX_KEYS_PER_TURN


def test_concurrent_use_is_serialised():
    """Every distinct key survives; none is lost to a torn update.

    Stays under _MAX_KEYS_PER_TURN so eviction cannot mask a lost write.
    """
    per_thread = 30
    threads_n = 6
    assert per_thread * threads_n <= turn_read_cache._MAX_KEYS_PER_TURN

    def _worker(n):
        for i in range(per_thread):
            turn_read_cache.record("t1", "get_task", {"gid": f"{n}-{i}"})

    threads = [threading.Thread(target=_worker, args=(n,)) for n in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(turn_read_cache._turn_reads["t1"]) == per_thread * threads_n


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------


def test_repeated_ignoring_of_the_stub_escalates_to_a_hard_block():
    """A weak tool-follower must not burn its budget re-asking."""
    turn_read_cache.record("t1", "get_task", {"gid": "1"})
    for _ in range(turn_read_cache._HARD_BLOCK_AFTER):
        out = json.loads(turn_read_cache.check("t1", "get_task", {"gid": "1"}))
        assert "error" not in out

    blocked = json.loads(turn_read_cache.check("t1", "get_task", {"gid": "1"}))
    assert "BLOCKED" in blocked["error"]
    assert blocked["suppressed"] is True


# ---------------------------------------------------------------------------
# Rule 1: read-only classification
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


def test_dispatcher_suppresses_the_second_identical_read(read_only_probe):
    from model_tools import handle_function_call

    first = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T"))
    assert first["call"] == 1

    second = json.loads(handle_function_call("_ro_probe", {"q": 1}, turn_id="T"))
    assert second["suppressed"] is True
    assert read_only_probe["n"] == 1, "the tool ran a second time"
    # The payload is not re-sent — that is the whole point.
    assert "x" * 500 not in json.dumps(second)


def test_dispatcher_does_not_suppress_different_arguments(read_only_probe):
    from model_tools import handle_function_call

    handle_function_call("_ro_probe", {"q": 1}, turn_id="T")
    out = json.loads(handle_function_call("_ro_probe", {"q": 2}, turn_id="T"))
    assert out["call"] == 2
    assert read_only_probe["n"] == 2


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


def test_write_capable_tools_are_never_suppressed(monkeypatch):
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
# The executor's direct-dispatch bypass
# ---------------------------------------------------------------------------


def test_every_direct_dispatch_tool_is_listed():
    """The bypass list must cover every branch of the executor's ladder.

    Those tools never reach handle_function_call, so they never hit the
    invalidate-on-write rule there. If a new branch is added to the ladder
    without adding its name here, a write would silently stop invalidating
    and a cached read could go stale for the rest of the turn.
    """
    import inspect
    import re

    from agent import tool_executor

    src = inspect.getsource(tool_executor)
    ladder = src[src.index('if function_name == "todo":'):]
    ladder = ladder[:ladder.index("\n        else:")]
    branched = set(re.findall(r'function_name == "([a-z_]+)"', ladder))
    assert branched, "could not find the executor's direct-dispatch ladder"

    missing = branched - set(tool_executor._EXECUTOR_DIRECT_DISPATCH_TOOLS)
    assert not missing, (
        f"executor branches on {sorted(missing)} without listing them in "
        "_EXECUTOR_DIRECT_DISPATCH_TOOLS — those writes would not invalidate "
        "the per-turn read cache"
    )


def test_the_bypass_list_names_only_real_tools():
    """A stale name here would silently invalidate nothing."""
    import inspect

    from agent import tool_executor

    src = inspect.getsource(tool_executor)
    for name in tool_executor._EXECUTOR_DIRECT_DISPATCH_TOOLS:
        assert f'function_name == "{name}"' in src, (
            f"{name} is listed as direct-dispatch but the executor has no "
            "branch for it"
        )


def test_a_direct_dispatch_tool_invalidates_the_turn_cache():
    """message_agent can drive another agent into mutating the read's subject."""
    from agent import tool_executor

    turn_read_cache.record("T", "get_task", {"gid": "1"})
    assert turn_read_cache.check("T", "get_task", {"gid": "1"}) is not None

    assert "message_agent" in tool_executor._EXECUTOR_DIRECT_DISPATCH_TOOLS
    # Mirror what the executor does for a direct-dispatch tool.
    turn_read_cache.invalidate("T")

    assert turn_read_cache.check("T", "get_task", {"gid": "1"}) is None
