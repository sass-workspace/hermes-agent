"""Regression: useless no-arg ``kanban_show()`` calls.

Two defects put an orchestrator profile into a loop of calls that could
never succeed:

1. ``agent/agent_init.py`` gated the ~6k-char WORKER lifecycle protocol on
   ``"kanban_show" in agent.valid_tool_names``, on the stated premise that
   "kanban_show tool is present iff HERMES_KANBAN_TASK is set". That is not
   true — ``_check_kanban_mode()`` also grants the kanban tools to any
   orchestrator profile that enabled the toolset. Those profiles got a
   system prompt whose first instruction is *"Call ``kanban_show()`` first
   (no args — it defaults to your task)"*, with no task to default to.

2. The resulting error, "task_id is required (or set HERMES_KANBAN_TASK in
   the env)", read as a retryable hint — so the model retried the identical
   no-arg call. It named a remedy the agent cannot perform mid-turn and
   never said the same call could not work.

These tests pin both halves, plus the sibling handlers that shared the same
vague message.
"""
from __future__ import annotations

import json

import pytest


ALL_DEFAULTING_HANDLERS = [
    ("_handle_show", "kanban_show"),
    ("_handle_complete", "kanban_complete"),
    ("_handle_block", "kanban_block"),
    ("_handle_request_review", "kanban_request_review"),
    ("_handle_request_changes", "kanban_request_changes"),
    ("_handle_heartbeat", "kanban_heartbeat"),
    ("_handle_attach", "kanban_attach"),
    ("_handle_attach_url", "kanban_attach_url"),
    ("_handle_attachments", "kanban_attachments"),
]


@pytest.fixture
def orchestrator_env(monkeypatch, tmp_path):
    """An orchestrator profile: kanban toolset enabled, no assigned task."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_PROFILE", "test-orchestrator")
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    from pathlib import Path as _Path
    monkeypatch.setattr(_Path, "home", lambda: tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# The worker-protocol gate
# ---------------------------------------------------------------------------


def test_worker_predicate_is_false_without_an_assigned_task(orchestrator_env):
    from tools.kanban_tools import is_dispatcher_spawned_task_worker

    assert is_dispatcher_spawned_task_worker() is False


def test_worker_predicate_is_true_for_a_dispatcher_spawned_worker(
    monkeypatch, orchestrator_env
):
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    monkeypatch.setattr(kt, "_is_dispatcher_owned_worker", lambda: True)
    monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: False)
    assert kt.is_dispatcher_spawned_task_worker() is True


def test_worker_predicate_is_false_for_a_delegated_child(
    monkeypatch, orchestrator_env
):
    """A delegate_task child inherits HERMES_KANBAN_* but owns no task."""
    from tools import kanban_tools as kt

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task-123")
    monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: True)
    assert kt.is_dispatcher_spawned_task_worker() is False


def test_worker_predicate_matches_the_no_arg_default_resolution(
    monkeypatch, orchestrator_env
):
    """The prompt gate and the tool's own default must never drift apart.

    The guidance says "call it with no args"; the predicate that gates the
    guidance must be true exactly when that call actually resolves an id.
    """
    from tools import kanban_tools as kt

    for env_task, owned, delegated in [
        (None, True, False),
        ("t1", True, False),
        ("t1", False, False),
        ("t1", True, True),
    ]:
        if env_task is None:
            monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        else:
            monkeypatch.setenv("HERMES_KANBAN_TASK", env_task)
        monkeypatch.setattr(kt, "_is_dispatcher_owned_worker", lambda: owned)
        monkeypatch.setattr(kt, "_is_delegated_child_context", lambda: delegated)

        resolves = kt._default_task_id(None) is not None
        assert kt.is_dispatcher_spawned_task_worker() is resolves, (
            f"predicate/default disagree for env={env_task} owned={owned} "
            f"delegated={delegated}"
        )


def test_guidance_is_not_injected_for_an_orchestrator(monkeypatch, orchestrator_env):
    """The regression itself: an orchestrator must not be told to call
    kanban_show() with no args, because that call can never work for it."""
    import agent.agent_init as agent_init
    from agent.prompt_builder import KANBAN_GUIDANCE

    class _Agent:
        valid_tool_names = {"kanban_show", "kanban_list"}

    a = _Agent()
    monkeypatch.setattr(
        "tools.kanban_tools.is_dispatcher_spawned_task_worker", lambda: False
    )
    _resolve_kanban_guidance(agent_init, a)
    assert a._kanban_worker_guidance == ""
    assert "no args" not in a._kanban_worker_guidance
    # Sanity: the guidance really does carry that instruction.
    assert "no args" in KANBAN_GUIDANCE


def test_guidance_is_still_injected_for_a_real_worker(monkeypatch, orchestrator_env):
    """The fix must not strip the protocol from the agents that need it."""
    import agent.agent_init as agent_init
    from agent.prompt_builder import KANBAN_GUIDANCE

    class _Agent:
        valid_tool_names = {"kanban_show", "kanban_complete"}

    a = _Agent()
    monkeypatch.setattr(
        "tools.kanban_tools.is_dispatcher_spawned_task_worker", lambda: True
    )
    _resolve_kanban_guidance(agent_init, a)
    assert a._kanban_worker_guidance == KANBAN_GUIDANCE


def test_guidance_absent_when_the_tool_is_absent(monkeypatch, orchestrator_env):
    import agent.agent_init as agent_init

    class _Agent:
        valid_tool_names = {"read_file"}

    a = _Agent()
    monkeypatch.setattr(
        "tools.kanban_tools.is_dispatcher_spawned_task_worker", lambda: True
    )
    _resolve_kanban_guidance(agent_init, a)
    assert a._kanban_worker_guidance == ""


def _resolve_kanban_guidance(agent_init, agent) -> None:
    """Invoke the REAL resolver agent_init uses, against a stub agent."""
    agent._kanban_worker_guidance = agent_init.resolve_kanban_worker_guidance(agent)


def test_agent_init_uses_the_shared_resolver():
    """The init path must go through the function these tests exercise."""
    import inspect

    import agent.agent_init as agent_init

    assert "resolve_kanban_worker_guidance(agent)" in inspect.getsource(
        agent_init.initialize_tools
        if hasattr(agent_init, "initialize_tools")
        else agent_init
    )


def test_the_resolver_tolerates_an_agent_without_tool_names():
    """Never raise into agent init over a missing attribute."""
    import agent.agent_init as agent_init

    class _Bare:
        pass

    assert agent_init.resolve_kanban_worker_guidance(_Bare()) == ""


# ---------------------------------------------------------------------------
# The verdict itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "handler_name,tool_name", ALL_DEFAULTING_HANDLERS,
    ids=[t for _h, t in ALL_DEFAULTING_HANDLERS],
)
def test_no_arg_call_returns_a_terminal_actionable_verdict(
    orchestrator_env, handler_name, tool_name
):
    from tools import kanban_tools as kt

    out = json.loads(getattr(kt, handler_name)({}))
    error = out["error"]

    # Names itself, so the model knows which call it must not repeat.
    assert tool_name in error
    # Says why THIS session has no default.
    assert "not" in error and "dispatcher" in error
    # Says plainly that repeating the identical call cannot work.
    assert "again with no task_id will fail the same way" in error
    # Points at a route that can work.
    assert "task_id=" in error
    # Machine-readable marker for surfaces that want to branch on it.
    assert out.get("needs_task_id") is True


@pytest.mark.parametrize(
    "handler_name,tool_name", ALL_DEFAULTING_HANDLERS,
    ids=[t for _h, t in ALL_DEFAULTING_HANDLERS],
)
def test_no_arg_verdict_no_longer_suggests_setting_an_env_var(
    orchestrator_env, handler_name, tool_name
):
    """The old wording named a remedy the agent cannot perform mid-turn."""
    from tools import kanban_tools as kt

    error = json.loads(getattr(kt, handler_name)({}))["error"]
    assert "HERMES_KANBAN_TASK" not in error
    assert "task_id is required" not in error


def test_the_vague_message_is_gone_from_the_module():
    """No sibling handler may keep the retry-inviting wording."""
    from pathlib import Path

    src = Path("tools/kanban_tools.py").read_text()
    assert "task_id is required (or set HERMES_KANBAN_TASK in the env)" not in src


def test_an_explicit_task_id_still_works_for_an_orchestrator(orchestrator_env):
    """The fix must not block the route it recommends."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    kb._INITIALIZED_PATHS.clear()
    kb.init_db()
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="routed", assignee="factory")
    finally:
        conn.close()

    out = json.loads(kt._handle_show({"task_id": tid}))
    assert out["task"]["id"] == tid
