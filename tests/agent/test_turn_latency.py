"""Per-turn latency accounting and the approval-wait split.

The point of this module is diagnostic honesty: when a turn takes 40
seconds, the summary must say which bucket ate it. Two failure modes make
that useless, and both are locked down here:

  * human think-time counted as tool latency, which makes a 200ms tool look
    pathological and sends the investigation to the wrong place;
  * provider time landing in the unattributed remainder, which is the exact
    case that keeps getting misdiagnosed ("the tool is slow" when the tool
    is fast and the model call is not).
"""
import json
import threading
import time
from unittest.mock import patch

import pytest

from agent import turn_latency


@pytest.fixture(autouse=True)
def _clean_registry():
    turn_latency.reset_for_tests()
    yield
    turn_latency.reset_for_tests()


# ---------------------------------------------------------------------------
# Bucket accounting
# ---------------------------------------------------------------------------


def test_buckets_split_model_tool_and_approval():
    turn_latency.start_turn("t1", session_id="s1", platform="cli")
    turn_latency.record_model_call("t1", 2.0)
    turn_latency.record_model_call("t1", 1.0)
    turn_latency.record_tool_call("t1", "read_file", 500.0)
    turn_latency.record_tool_call("t1", "terminal", 1000.0, approval_wait_ms=800.0)

    summary = turn_latency.finish_turn("t1", log=False)
    assert summary["api_calls"] == 2
    assert summary["model_ms"] == 3000.0
    assert summary["tool_calls"] == 2
    # 500 + (1000 - 800): the approval wait is NOT charged to the tool.
    assert summary["tool_ms"] == 700.0
    assert summary["approval_wait_ms"] == 800.0
    assert summary["session_id"] == "s1"
    assert summary["platform"] == "cli"


def test_approval_wait_is_capped_at_the_call_duration():
    """A clock skew must never produce negative executed time."""
    turn_latency.start_turn("t1")
    turn_latency.record_tool_call("t1", "terminal", 100.0, approval_wait_ms=5000.0)
    summary = turn_latency.finish_turn("t1", log=False)
    assert summary["tool_ms"] == 0.0
    assert summary["approval_wait_ms"] == 100.0


def test_other_bucket_is_the_unattributed_remainder():
    turn_latency.start_turn("t1")
    time.sleep(0.05)
    summary = turn_latency.finish_turn("t1", log=False)
    assert summary["wall_ms"] >= 50.0
    # Nothing was attributed, so it all lands in `other`.
    assert summary["other_ms"] == pytest.approx(summary["wall_ms"], rel=0.02)


def test_other_never_goes_negative_under_concurrent_tools():
    """Concurrent tools can account more time than the wall clock."""
    turn_latency.start_turn("t1")
    turn_latency.record_tool_call("t1", "a", 60_000.0)
    turn_latency.record_tool_call("t1", "b", 60_000.0)
    summary = turn_latency.finish_turn("t1", log=False)
    assert summary["other_ms"] == 0.0


def test_per_tool_breakdown_is_ranked_by_time():
    turn_latency.start_turn("t1")
    turn_latency.record_tool_call("t1", "fast", 10.0)
    turn_latency.record_tool_call("t1", "fast", 10.0)
    turn_latency.record_tool_call("t1", "slow", 900.0)
    summary = turn_latency.finish_turn("t1", log=False)
    assert [t["name"] for t in summary["tools"]] == ["slow", "fast"]
    assert summary["tools"][1]["calls"] == 2


# ---------------------------------------------------------------------------
# Lifecycle / robustness
# ---------------------------------------------------------------------------


def test_recording_against_an_unknown_turn_is_a_noop():
    """A tool call outside any tracked turn must not raise."""
    turn_latency.record_model_call("nope", 1.0)
    turn_latency.record_tool_call("nope", "x", 1.0)
    assert turn_latency.summarize("nope") is None
    assert turn_latency.finish_turn("nope", log=False) is None


def test_empty_turn_id_is_ignored():
    turn_latency.start_turn("")
    assert turn_latency.finish_turn("", log=False) is None


def test_finish_turn_unregisters():
    turn_latency.start_turn("t1")
    assert turn_latency.finish_turn("t1", log=False) is not None
    assert turn_latency.finish_turn("t1", log=False) is None


def test_summarize_does_not_unregister():
    turn_latency.start_turn("t1")
    turn_latency.record_model_call("t1", 1.0)
    assert turn_latency.summarize("t1")["model_ms"] == 1000.0
    assert turn_latency.finish_turn("t1", log=False) is not None


def test_registry_is_bounded_against_leaked_turns():
    """A turn that never finishes must not grow the registry forever."""
    for i in range(turn_latency._MAX_TRACKED_TURNS + 25):
        turn_latency.start_turn(f"turn-{i}")
    assert len(turn_latency._records) == turn_latency._MAX_TRACKED_TURNS
    # The oldest were evicted; the newest survive.
    assert turn_latency.summarize("turn-0") is None
    assert turn_latency.summarize(
        f"turn-{turn_latency._MAX_TRACKED_TURNS + 24}"
    ) is not None


def test_concurrent_tool_recording_is_serialised():
    turn_latency.start_turn("t1")

    def _worker():
        for _ in range(200):
            turn_latency.record_tool_call("t1", "read_file", 1.0)

    threads = [threading.Thread(target=_worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = turn_latency.finish_turn("t1", log=False)
    assert summary["tool_calls"] == 1600
    assert summary["tool_ms"] == 1600.0


# ---------------------------------------------------------------------------
# Approval-wait counter
# ---------------------------------------------------------------------------


def test_counter_accumulates_and_reads_as_a_delta():
    before = turn_latency.approval_wait_total()
    turn_latency.record_approval_wait(1.5)
    turn_latency.record_approval_wait(0.5)
    assert turn_latency.approval_wait_total() - before == 2.0


def test_nested_regions_roll_up_into_the_outer_delta():
    """An outer dispatch really did sit through the inner one's wait.

    `delegate` driving a subagent on the same thread has the child's approval
    wait inside its own duration_ms, so it must be able to subtract it.
    """
    outer_mark = turn_latency.approval_wait_total()
    turn_latency.record_approval_wait(1.0)
    inner_mark = turn_latency.approval_wait_total()
    turn_latency.record_approval_wait(5.0)
    inner = turn_latency.approval_wait_total() - inner_mark
    outer = turn_latency.approval_wait_total() - outer_mark
    assert inner == 5.0
    assert outer == 6.0


def test_counter_is_thread_local():
    """Concurrent tool calls must never pollute each other's accounting."""
    seen = {}
    barrier = threading.Barrier(2)

    def _worker(name, amount):
        mark = turn_latency.approval_wait_total()
        barrier.wait(timeout=5)
        turn_latency.record_approval_wait(amount)
        time.sleep(0.02)
        seen[name] = turn_latency.approval_wait_total() - mark

    threads = [
        threading.Thread(target=_worker, args=("a", 1.0)),
        threading.Thread(target=_worker, args=("b", 9.0)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen == {"a": 1.0, "b": 9.0}


def test_a_fresh_thread_starts_at_zero():
    turn_latency.record_approval_wait(42.0)
    seen = {}

    def _worker():
        seen["total"] = turn_latency.approval_wait_total()

    t = threading.Thread(target=_worker)
    t.start()
    t.join()
    assert seen["total"] == 0.0


def test_approval_wait_timer_charges_elapsed_time():
    mark = turn_latency.approval_wait_total()
    with turn_latency.approval_wait_timer():
        time.sleep(0.05)
    assert turn_latency.approval_wait_total() - mark >= 0.05


def test_approval_wait_timer_charges_even_when_the_prompt_raises():
    mark = turn_latency.approval_wait_total()
    with pytest.raises(RuntimeError):
        with turn_latency.approval_wait_timer():
            time.sleep(0.02)
            raise RuntimeError("user surface blew up")
    assert turn_latency.approval_wait_total() - mark >= 0.02


# ---------------------------------------------------------------------------
# The diagnostic the whole module exists for
# ---------------------------------------------------------------------------


def test_summary_names_the_model_when_the_tool_is_fast():
    """The recurring misdiagnosis: a fast tool blamed for a slow turn."""
    turn_latency.start_turn("t1")
    turn_latency.record_model_call("t1", 18.0)
    turn_latency.record_tool_call("t1", "exec_prepare", 120.0)
    summary = turn_latency.finish_turn("t1", log=False)

    assert summary["model_ms"] > summary["tool_ms"] * 100
    line = turn_latency.format_summary(summary)
    assert "model=18.0s" in line
    assert "exec_prepare 0.1s x1" in line


def test_format_summary_marks_approval_wait_on_the_tool():
    turn_latency.start_turn("t1")
    turn_latency.record_tool_call("t1", "terminal", 9000.0, approval_wait_ms=8000.0)
    line = turn_latency.format_summary(turn_latency.finish_turn("t1", log=False))
    assert "approval_wait=8.0s" in line
    assert "(+8.0s approval)" in line


# ---------------------------------------------------------------------------
# Integration with the tool dispatcher
# ---------------------------------------------------------------------------


def _register_probe(name, handler):
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


def _drop_probe(name):
    from tools.registry import registry

    registry._tools.pop(name, None)


def test_dispatcher_splits_approval_wait_out_of_tool_duration():
    """End-to-end: a tool that blocks on approval reports both numbers."""
    from model_tools import handle_function_call
    from tools.registry import tool_result

    def _handler(args, **kwargs):
        # Stand in for a guard that prompts the user mid-dispatch.
        with turn_latency.approval_wait_timer():
            time.sleep(0.15)
        return tool_result(ok=True)

    _register_probe("_latency_probe_tool", _handler)
    try:
        turn_latency.start_turn("turn-x")
        out = json.loads(
            handle_function_call("_latency_probe_tool", {}, turn_id="turn-x")
        )
        assert out.get("ok") is True

        summary = turn_latency.finish_turn("turn-x", log=False)
        assert summary["tool_calls"] == 1
        # The wait was recognised...
        assert summary["approval_wait_ms"] >= 150
        # ...and subtracted, so the tool itself reads as fast.
        assert summary["tool_ms"] < 100
    finally:
        _drop_probe("_latency_probe_tool")


def test_dispatcher_reports_approval_wait_to_the_post_tool_call_hook():
    from model_tools import handle_function_call
    from tools.registry import tool_result

    def _handler(args, **kwargs):
        with turn_latency.approval_wait_timer():
            time.sleep(0.12)
        return tool_result(ok=True)

    _register_probe("_latency_hook_probe", _handler)
    seen = {}

    def _invoke_hook(name, **kwargs):
        if name == "post_tool_call":
            seen.update(kwargs)
        return []

    try:
        with patch("hermes_cli.lifecycle.has_hook", lambda name: name == "post_tool_call"), \
             patch("hermes_cli.lifecycle.invoke_hook", _invoke_hook):
            handle_function_call("_latency_hook_probe", {}, turn_id="turn-y")
        assert seen.get("approval_wait_ms", 0) >= 120
        assert seen["duration_ms"] >= seen["approval_wait_ms"]
    finally:
        _drop_probe("_latency_hook_probe")


def test_dispatcher_reports_zero_approval_wait_for_unattended_tools():
    from model_tools import handle_function_call
    from tools.registry import tool_result

    _register_probe("_latency_quiet_probe", lambda args, **kw: tool_result(ok=True))
    try:
        turn_latency.start_turn("turn-z")
        handle_function_call("_latency_quiet_probe", {}, turn_id="turn-z")
        summary = turn_latency.finish_turn("turn-z", log=False)
        assert summary["approval_wait_ms"] == 0.0
        assert summary["tool_calls"] == 1
    finally:
        _drop_probe("_latency_quiet_probe")


def test_a_raising_tool_is_still_recorded_with_its_approval_wait():
    """A handler that blows up is exactly the call worth accounting for.

    Recording after the dispatch block dropped these entirely — both the
    latency and any approval wait already paid for.
    """
    from model_tools import handle_function_call

    def _handler(args, **kwargs):
        with turn_latency.approval_wait_timer():
            time.sleep(0.12)
        raise RuntimeError("handler exploded after the human answered")

    _register_probe("_latency_boom_probe", _handler)
    try:
        turn_latency.start_turn("turn-boom")
        out = json.loads(
            handle_function_call("_latency_boom_probe", {}, turn_id="turn-boom")
        )
        assert "error" in out
        summary = turn_latency.finish_turn("turn-boom", log=False)
        assert summary["tool_calls"] == 1, "a failed dispatch was not recorded"
        assert summary["approval_wait_ms"] >= 120
    finally:
        _drop_probe("_latency_boom_probe")


def test_a_raising_tool_is_recorded_exactly_once():
    """The dispatch `finally` and the outer handler must not double count."""
    from model_tools import handle_function_call

    def _handler(args, **kwargs):
        raise RuntimeError("boom")

    _register_probe("_latency_once_probe", _handler)
    try:
        turn_latency.start_turn("turn-once")
        handle_function_call("_latency_once_probe", {}, turn_id="turn-once")
        summary = turn_latency.finish_turn("turn-once", log=False)
        assert summary["tool_calls"] == 1
    finally:
        _drop_probe("_latency_once_probe")


def test_the_dispatcher_publishes_its_measured_split_for_another_thread():
    """The executor emits the hook from a DIFFERENT thread than the tool ran on.

    The approval counter is thread-local, so measuring in the executor would
    always read zero — silently reporting approval_wait_ms=0 for exactly the
    calls that block on a human longest (dangerous terminal commands). The
    dispatcher publishes what it measured, keyed by tool_call_id.
    """
    from agent.tool_executor import _measured_approval_wait_ms
    from model_tools import handle_function_call
    from tools.registry import tool_result

    def _handler(args, **kwargs):
        with turn_latency.approval_wait_timer():
            time.sleep(0.12)
        return tool_result(ok=True)

    _register_probe("_latency_cross_thread_probe", _handler)
    seen = {}

    def _run_on_worker():
        handle_function_call(
            "_latency_cross_thread_probe", {},
            turn_id="turn-ct", tool_call_id="call-ct",
        )

    try:
        worker = threading.Thread(target=_run_on_worker)
        worker.start()
        worker.join()

        # Read it from THIS thread, whose own counter never moved.
        assert turn_latency.approval_wait_total() == 0.0
        seen["ms"] = _measured_approval_wait_ms("call-ct")
        assert seen["ms"] >= 120, (
            "the executor's thread could not see the wait the worker measured"
        )
        # Taking it consumes it, so a later call cannot reuse a stale value.
        assert _measured_approval_wait_ms("call-ct") == 0
    finally:
        _drop_probe("_latency_cross_thread_probe")


def test_published_splits_do_not_cross_between_concurrent_calls():
    turn_latency.publish_call_approval_wait("call-a", 111)
    turn_latency.publish_call_approval_wait("call-b", 222)
    assert turn_latency.take_call_approval_wait("call-b") == 222
    assert turn_latency.take_call_approval_wait("call-a") == 111
    assert turn_latency.take_call_approval_wait("call-a") == 0


def test_the_published_map_is_bounded():
    for i in range(turn_latency._MAX_PUBLISHED_CALLS + 20):
        turn_latency.publish_call_approval_wait(f"call-{i}", i)
    assert len(turn_latency._published_call_waits) == turn_latency._MAX_PUBLISHED_CALLS


def test_a_gate_rejection_before_dispatch_is_still_recorded():
    """A blocked call burned wall clock, sometimes a human's. Account for it."""
    from model_tools import handle_function_call

    turn_latency.start_turn("turn-gate")
    # An unknown tool short-circuits before any dispatch.
    handle_function_call("_no_such_tool_at_all", {}, turn_id="turn-gate")
    summary = turn_latency.finish_turn("turn-gate", log=False)
    # Registry dispatch handles unknown names, so this still records once.
    assert summary["tool_calls"] >= 1
