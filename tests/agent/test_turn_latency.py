"""Per-turn latency accounting: wall = model + tools + approval_wait + other.

A slow turn is almost never slow for the reason anyone guesses, and nothing measured where the
wall clock went. These tests pin the arithmetic of the split, that the approval wait is read from
the runtime's own human-wait accounting (so a fast tool that sat on a human does not look
pathological), that recording survives failure, and the wiring that makes the ledger real: the
dispatcher's one terminal ``post_tool_call`` funnel and the turn-identity binding.
"""
from __future__ import annotations

import logging
import time
from types import SimpleNamespace

import pytest

from agent import turn_latency as tl

LOGGER = "agent.turn_latency"


@pytest.fixture(autouse=True)
def _clean():
    tl.reset_for_tests()
    yield
    tl.reset_for_tests()


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


def test_the_buckets_are_reported_and_other_is_the_remainder():
    tl.start_turn("t1")
    tl.record_model_call("t1", 0.30)
    tl.record_model_call("t1", 0.20)
    tl.record_tool_call("t1", "terminal", 120)
    tl.record_tool_call("t1", "read_file", 30)
    s = tl.summarize("t1")
    assert s["model_ms"] == pytest.approx(500.0)
    assert s["api_calls"] == 2
    assert s["tool_ms"] == pytest.approx(150.0)
    assert s["tool_calls"] == 2
    assert s["approval_wait_ms"] == 0.0
    assert s["other_ms"] == pytest.approx(max(0.0, s["wall_ms"] - 650.0), abs=0.2)
    assert [t["name"] for t in s["tools"]] == ["terminal", "read_file"], "ranked by time spent"


def test_concurrent_tools_never_produce_a_negative_remainder():
    tl.start_turn("t1")
    for _ in range(8):
        tl.record_tool_call("t1", "web_search", 5000)  # 40 s of tool time in a ~0 s turn
    assert tl.summarize("t1")["other_ms"] == 0.0


def test_a_human_wait_is_split_out_of_the_tool_that_sat_on_it():
    """The split that makes the other three honest, read from the runtime's own accounting."""
    from tools.approval_human_wait import human_wait_window

    tl.start_turn("t1")
    started = time.monotonic()
    with human_wait_window():
        time.sleep(0.25)
    tl.record_tool_call("t1", "terminal", (time.monotonic() - started) * 1000.0 + 40.0)
    s = tl.summarize("t1")
    assert s["approval_wait_ms"] >= 250.0
    assert s["tool_ms"] < 250.0, "the wait must not be charged to the tool"
    assert s["tool_ms"] == pytest.approx(40.0, abs=30.0)


def test_a_wait_before_the_turn_started_is_not_charged_to_it():
    from tools.approval_human_wait import human_wait_window

    with human_wait_window():
        time.sleep(0.15)
    tl.start_turn("t1")
    assert tl.summarize("t1")["approval_wait_ms"] == 0.0


def test_another_sessions_wait_is_not_charged_to_this_turn():
    from tools.approval_context import reset_current_session_key, set_current_session_key
    from tools.approval_human_wait import human_wait_window

    token = set_current_session_key("session-a")
    try:
        tl.start_turn("t1")
    finally:
        reset_current_session_key(token)
    with human_wait_window("session-b"):
        time.sleep(0.15)
    assert tl.summarize("t1")["approval_wait_ms"] == 0.0


# ---------------------------------------------------------------------------
# Registry behaviour
# ---------------------------------------------------------------------------


def test_a_second_start_keeps_the_existing_ledger():
    tl.start_turn("t1")
    tl.record_model_call("t1", 1.0)
    tl.start_turn("t1")
    assert tl.summarize("t1")["api_calls"] == 1


def test_recording_against_an_unknown_turn_is_a_no_op():
    tl.record_model_call("nope", 1.0)
    tl.record_tool_call("nope", "terminal", 10)
    tl.record_tool_call("", "terminal", 10)
    assert tl.summarize("nope") is None
    assert tl.finish_turn("nope") is None


def test_the_registry_is_bounded():
    for i in range(tl._MAX_TRACKED_TURNS + 20):
        tl.start_turn(f"t{i}")
    assert len(tl._records) == tl._MAX_TRACKED_TURNS
    assert tl.summarize("t0") is None, "oldest first out"


def test_bad_values_never_raise():
    tl.start_turn("t1")
    tl.record_model_call("t1", "not-a-number")  # type: ignore[arg-type]
    tl.record_tool_call("t1", "terminal", None)  # type: ignore[arg-type]
    tl.record_tool_call("t1", "terminal", -5)
    assert tl.summarize("t1")["tool_ms"] == 0.0


def test_finish_logs_one_line_and_unregisters(caplog):
    tl.start_turn("t1")
    tl.record_model_call("t1", 1.5)
    tl.record_tool_call("t1", "mcp__asana__create_task", 700)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        summary = tl.finish_turn("t1")
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(lines) == 1
    assert lines[0].startswith("turn latency turn=t1 wall=")
    for part in ("model=1.5s(1 calls)", "tools=0.7s(1 calls)", "approval_wait=0.0s", "other=",
                 "top: mcp__asana__create_task 0.7s x1"):
        assert part in lines[0]
    assert summary["turn_id"] == "t1"
    assert tl.summarize("t1") is None


def test_a_model_attempt_that_raises_still_costs_its_time():
    tl.start_turn("t1")
    with pytest.raises(RuntimeError):
        with tl.model_call_timer("t1"):
            time.sleep(0.05)
            raise RuntimeError("provider exploded")
    s = tl.summarize("t1")
    assert s["api_calls"] == 1 and s["model_ms"] >= 50.0


# ---------------------------------------------------------------------------
# The wiring that makes the ledger real
# ---------------------------------------------------------------------------


@pytest.fixture
def probe_tool():
    from tools.registry import registry, tool_result

    def _handler(args, **kw):
        time.sleep(0.05)
        if args.get("boom"):
            raise RuntimeError("handler exploded")
        return tool_result(ok=True)

    registry.register(
        name="_latency_probe", toolset="testing",
        schema={"name": "_latency_probe", "description": "test only",
                "parameters": {"type": "object", "properties": {}}},
        handler=_handler)
    yield "_latency_probe"
    registry._tools.pop("_latency_probe", None)


def test_a_real_dispatch_lands_in_the_turns_ledger_exactly_once(probe_tool):
    from model_tools import handle_function_call

    tl.start_turn("TURN-1")
    handle_function_call(probe_tool, {}, turn_id="TURN-1", tool_call_id="call-1")
    s = tl.summarize("TURN-1")
    assert s["tool_calls"] == 1
    assert s["tools"][0]["name"] == probe_tool
    assert s["tool_ms"] >= 50.0


def test_a_dispatch_that_raises_is_still_recorded(probe_tool):
    from model_tools import handle_function_call

    tl.start_turn("TURN-1")
    handle_function_call(probe_tool, {"boom": True}, turn_id="TURN-1", tool_call_id="call-1")
    assert tl.summarize("TURN-1")["tool_calls"] == 1


def test_an_executor_owned_call_is_recorded_by_its_terminal_hook_not_twice(probe_tool):
    """The tool executor suppresses the dispatcher's hook and emits the one terminal
    ``post_tool_call`` itself: the ledger must follow that exactly-once contract."""
    import model_tools
    from agent.inline_tool_executors import emit_terminal_post_tool_call

    tl.start_turn("TURN-1")
    with model_tools.suppress_post_tool_call_hook():
        model_tools.handle_function_call(probe_tool, {}, turn_id="TURN-1", tool_call_id="call-1")
    assert tl.summarize("TURN-1")["tool_calls"] == 0

    agent = SimpleNamespace(session_id="s", _current_turn_id="TURN-1", _current_api_request_id="")
    emit_terminal_post_tool_call(agent, function_name=probe_tool, function_args={}, result="{}",
                                 effective_task_id="task", tool_call_id="call-1", duration_ms=80)
    s = tl.summarize("TURN-1")
    assert s["tool_calls"] == 1 and s["tool_ms"] == pytest.approx(80.0)


def test_binding_a_turns_identity_opens_its_ledger():
    from agent.turn_context import _bind_turn_identity

    agent = SimpleNamespace(session_id="sess", _relay_pending_turn_id="sess:task:abcd1234")
    _task_id, turn_id = _bind_turn_identity(agent, "task", None, None, None, None)
    assert turn_id == "sess:task:abcd1234"
    assert tl.summarize(turn_id) is not None


# ---------------------------------------------------------------------------
# End to end: a real turn logs where its wall clock went
# ---------------------------------------------------------------------------


def _chat_response(content, tool_calls, finish_reason):
    message = SimpleNamespace(content=content, reasoning=None, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=None)


class _FakeCompletions:
    def __init__(self, tool_name):
        self.calls, self.tool_name = 0, tool_name

    def create(self, **kwargs):
        self.calls += 1
        time.sleep(0.05)
        if self.calls == 1:
            call = SimpleNamespace(id="call_1", type="function",
                                   function=SimpleNamespace(name=self.tool_name, arguments="{}"))
            return _chat_response(None, [call], "tool_calls")
        return _chat_response("done", [], "stop")


def test_a_real_turn_logs_its_split_from_the_facade(monkeypatch, caplog, probe_tool):
    from run_agent import AIAgent

    client = SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions(probe_tool)))
    monkeypatch.setattr("agent.process_bootstrap.OpenAI", lambda **kwargs: client)
    monkeypatch.setattr("model_tools.get_tool_definitions",
                        lambda *args, **kwargs: [{"function": {"name": probe_tool}}])
    agent = AIAgent(model="test-model", api_key="test-key", base_url="http://localhost:8080/v1",
                    platform="cli", max_iterations=5, quiet_mode=True, skip_memory=True)
    agent._disable_streaming = True

    with caplog.at_level(logging.INFO, logger=LOGGER):
        result = agent.run_conversation("run the probe")

    assert result["final_response"].startswith("done")
    lines = [r.getMessage() for r in caplog.records if r.name == LOGGER and "turn latency" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "model=" in lines[0] and "(2 calls)" in lines[0]
    assert "tools=" in lines[0] and "(1 calls)" in lines[0]
    assert f"top: {probe_tool} " in lines[0]
    assert not tl._records, "the ledger must be closed when the turn ends"
