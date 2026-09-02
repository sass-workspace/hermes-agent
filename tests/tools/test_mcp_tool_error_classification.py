"""Application-level MCP tool errors must not trip the transport breaker.

The circuit breaker in ``tools/mcp_tool.py`` is a *reachability* breaker:
when it opens it tells the model the server is unreachable and blacks out
every one of that server's tools for the cooldown. Before this fix, any
``isError: true`` tool result counted as a strike — so three consecutive
schema-validation rejections from one hosted server (a payload bug on our
side, answered perfectly by a perfectly healthy transport) manufactured a
fake outage.

These tests drive the real handler against the fixture payloads in
``tests/fixtures/mcp_tool_result_errors.json`` and assert:

  * application errors leave the breaker closed and reach the model verbatim;
  * transport failures tunnelled through ``isError`` still strike the breaker;
  * sibling servers are never affected either way;
  * the auth/session retry helpers return a real tool error instead of
    masking it as ``needs_reauth``.
"""
import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest


pytest.importorskip("mcp.client.auth.oauth2")


FIXTURES = json.loads(
    (
        Path(__file__).resolve().parents[1] / "fixtures" / "mcp_tool_result_errors.json"
    ).read_text()
)
APPLICATION_CASES = [(c["id"], c["text"]) for c in FIXTURES["application"]]
TRANSPORT_CASES = [(c["id"], c["text"]) for c in FIXTURES["transport"]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_result(text: str):
    """Build an MCP ``CallToolResult``-shaped object with ``isError: true``."""
    block = MagicMock()
    block.text = text
    block.resource = None
    result = MagicMock()
    result.is_error = True
    result.isError = True
    result.content = [block]
    return result


def _ok_result(text: str = "fine"):
    block = MagicMock()
    block.text = text
    result = MagicMock()
    result.is_error = False
    result.isError = False
    result.content = [block]
    result.structured_content = None
    result.structuredContent = None
    result.meta = None
    return result


def _install_stub_server(mcp_tool, name: str, call_tool_impl):
    """Install a fake connected MCP server (mirrors the breaker suite's stub)."""
    server = MagicMock()
    server.name = name
    session = MagicMock()
    session.call_tool = call_tool_impl
    server.session = session

    ready_flag = threading.Event()
    ready_flag.set()

    class _ReadyAdapter:
        def is_set(self):
            return ready_flag.is_set()

        def clear(self):
            ready_flag.clear()

        def set(self):
            ready_flag.set()

    server._ready = _ReadyAdapter()
    server._reconnect_event = MagicMock()
    server._is_recycled_stdio.return_value = False
    # Real servers expose these as plain non-callables/False; a bare MagicMock
    # would return truthy Mocks and divert the call into the fast-fail path.
    server._stdio_children_dead = lambda: False
    server._watch_stdio_children = None
    server._mark_session_proven = MagicMock()

    mcp_tool._servers[name] = server
    mcp_tool._server_error_counts.pop(name, None)
    mcp_tool._server_breaker_opened_at.pop(name, None)
    return server


def _cleanup(mcp_tool, *names: str) -> None:
    for name in names:
        mcp_tool._servers.pop(name, None)
        mcp_tool._server_error_counts.pop(name, None)
        mcp_tool._server_breaker_opened_at.pop(name, None)


@pytest.fixture
def mcp_tool(monkeypatch, tmp_path):
    """The module with its background loop stubbed to run coroutines inline."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    import asyncio

    from tools import mcp_tool as module

    def _run_inline(coro_or_factory, timeout=30):
        coro = coro_or_factory() if callable(coro_or_factory) else coro_or_factory
        return asyncio.run(coro) if asyncio.iscoroutine(coro) else coro

    monkeypatch.setattr(module, "_run_on_mcp_loop", _run_inline)
    monkeypatch.setattr(module, "_trust_gate_check", lambda *a, **kw: None)
    return module


# ---------------------------------------------------------------------------
# Unit: the classifier itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_id,text", APPLICATION_CASES, ids=[c[0] for c in APPLICATION_CASES])
def test_application_payloads_classify_as_application(mcp_tool, case_id, text):
    assert mcp_tool._classify_tool_result_error(text) == "application"


@pytest.mark.parametrize("case_id,text", TRANSPORT_CASES, ids=[c[0] for c in TRANSPORT_CASES])
def test_transport_payloads_classify_as_transport(mcp_tool, case_id, text):
    assert mcp_tool._classify_tool_result_error(text) == "transport"


def test_non_string_error_payload_is_application(mcp_tool):
    """An unreadable error is not evidence that the server is unreachable."""
    for payload in (None, 42, {"nested": "dict"}, ["list"]):
        assert mcp_tool._classify_tool_result_error(payload) == "application"


# ---------------------------------------------------------------------------
# E2E through the real tool handler
# ---------------------------------------------------------------------------


def test_repeated_validation_errors_never_open_the_breaker(mcp_tool):
    """The fake-outage regression: N consecutive validation rejections.

    Runs well past ``_CIRCUIT_BREAKER_THRESHOLD`` and asserts every single
    call still reached the server and returned the server's own message.
    """
    texts = [text for _id, text in APPLICATION_CASES if text]
    calls = {"n": 0}

    async def _call_tool(*a, **kw):
        calls["n"] += 1
        return _error_result(texts[(calls["n"] - 1) % len(texts)])

    _install_stub_server(mcp_tool, "asana", _call_tool)
    try:
        handler = mcp_tool._make_tool_handler("asana", "create_task", 5.0)
        rounds = mcp_tool._CIRCUIT_BREAKER_THRESHOLD * 3
        for i in range(rounds):
            out = json.loads(handler({"name": "x"}))
            assert "error" in out
            # The server's own message, not a breaker blackout message.
            assert "unreachable" not in out["error"]
            assert "consecutive" not in out["error"]
            assert mcp_tool._server_error_counts.get("asana", 0) == 0, (
                f"breaker took a strike on application error #{i + 1}"
            )
        assert calls["n"] == rounds, "a call was short-circuited by the breaker"
        assert "asana" not in mcp_tool._server_breaker_opened_at
    finally:
        _cleanup(mcp_tool, "asana")


def test_tunnelled_transport_error_still_opens_the_breaker(mcp_tool):
    """Preserved behavior: a real 502/401 in an isError payload is a strike."""
    async def _call_tool(*a, **kw):
        return _error_result("502 Bad Gateway: upstream service did not respond.")

    _install_stub_server(mcp_tool, "flaky", _call_tool)
    try:
        handler = mcp_tool._make_tool_handler("flaky", "fetch", 5.0)
        for i in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD):
            handler({})
            # Exactly one strike per failed call — no double counting.
            assert mcp_tool._server_error_counts["flaky"] == i + 1
        assert (
            mcp_tool._server_error_counts["flaky"]
            == mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        )
        # Next call is short-circuited by the open breaker.
        blocked = json.loads(handler({}))
        assert "unreachable" in blocked["error"]
    finally:
        _cleanup(mcp_tool, "flaky")


def test_application_error_closes_a_partially_open_breaker(mcp_tool):
    """A completed round-trip proves reachability, exactly as a success does."""
    mode = {"transport": True}

    async def _call_tool(*a, **kw):
        if mode["transport"]:
            return _error_result("503 Service Unavailable")
        return _error_result("Validation failed: `html_notes` is not valid XML.")

    _install_stub_server(mcp_tool, "mixed", _call_tool)
    try:
        handler = mcp_tool._make_tool_handler("mixed", "op", 5.0)
        handler({})
        handler({})
        assert mcp_tool._server_error_counts["mixed"] == 2
        mode["transport"] = False
        handler({})
        assert mcp_tool._server_error_counts["mixed"] == 0
    finally:
        _cleanup(mcp_tool, "mixed")


def test_sibling_servers_are_isolated(mcp_tool):
    """Neither verdict may leak across servers."""
    async def _app_error(*a, **kw):
        return _error_result("Not Found: task with gid '1' does not exist.")

    async def _transport_error(*a, **kw):
        return _error_result("Connection refused (ECONNREFUSED).")

    _install_stub_server(mcp_tool, "srv_app", _app_error)
    _install_stub_server(mcp_tool, "srv_net", _transport_error)
    try:
        app_handler = mcp_tool._make_tool_handler("srv_app", "op", 5.0)
        net_handler = mcp_tool._make_tool_handler("srv_net", "op", 5.0)
        for _ in range(mcp_tool._CIRCUIT_BREAKER_THRESHOLD + 1):
            app_handler({})
            net_handler({})
        assert mcp_tool._server_error_counts.get("srv_app", 0) == 0
        assert (
            mcp_tool._server_error_counts["srv_net"]
            >= mcp_tool._CIRCUIT_BREAKER_THRESHOLD
        )
        assert "srv_app" not in mcp_tool._server_breaker_opened_at
    finally:
        _cleanup(mcp_tool, "srv_app", "srv_net")


def test_successful_call_still_resets(mcp_tool):
    """Guard the untouched path: a plain success closes the breaker."""
    mode = {"fail": True}

    async def _call_tool(*a, **kw):
        return _error_result("503 Service Unavailable") if mode["fail"] else _ok_result()

    _install_stub_server(mcp_tool, "ok", _call_tool)
    try:
        handler = mcp_tool._make_tool_handler("ok", "op", 5.0)
        handler({})
        assert mcp_tool._server_error_counts["ok"] == 1
        mode["fail"] = False
        assert "result" in json.loads(handler({}))
        assert mcp_tool._server_error_counts["ok"] == 0
    finally:
        _cleanup(mcp_tool, "ok")


# ---------------------------------------------------------------------------
# Sibling call path: the auth / session-expired retry helpers
# ---------------------------------------------------------------------------


def _force_auth_recovery(monkeypatch, mcp_tool):
    """Make ``_handle_auth_error_and_retry`` believe recovery succeeded."""
    monkeypatch.setattr(mcp_tool, "_is_auth_error", lambda exc: True)
    monkeypatch.setattr(
        mcp_tool, "_signal_reconnect_and_wait", lambda *a, **kw: True
    )

    class _Manager:
        async def handle_401(self, *a, **kw):
            return True

    monkeypatch.setattr(
        "tools.mcp_oauth_manager.get_manager", lambda *a, **kw: _Manager()
    )


def test_auth_retry_returns_application_error_not_needs_reauth(
    monkeypatch, mcp_tool
):
    """The retry proved the credentials work — don't cry re-auth."""
    _force_auth_recovery(monkeypatch, mcp_tool)
    _install_stub_server(mcp_tool, "authy", None)
    try:
        app_error = mcp_tool.tool_error(
            "Validation failed: `html_notes` is not valid XML."
        )
        out = mcp_tool._handle_auth_error_and_retry(
            "authy", Exception("401"), lambda: app_error, "tools/call create_task",
        )
        assert out == app_error
        parsed = json.loads(out)
        assert "needs_reauth" not in parsed
        assert mcp_tool._server_error_counts.get("authy", 0) == 0
    finally:
        _cleanup(mcp_tool, "authy")


def test_auth_retry_still_reports_needs_reauth_on_transport_error(
    monkeypatch, mcp_tool
):
    """Preserved behavior: a still-401 retry keeps the re-auth guidance."""
    _force_auth_recovery(monkeypatch, mcp_tool)
    _install_stub_server(mcp_tool, "authy2", None)
    try:
        out = mcp_tool._handle_auth_error_and_retry(
            "authy2",
            Exception("401"),
            lambda: mcp_tool.tool_error("401 Unauthorized: token is invalid"),
            "tools/call create_task",
        )
        parsed = json.loads(out)
        assert parsed.get("needs_reauth") is True
        # EXACTLY one strike. The classification pass must not bump on top of
        # the needs_reauth return's own bump — "consecutive failures" is the
        # quantity the threshold is calibrated against.
        assert mcp_tool._server_error_counts.get("authy2", 0) == 1
    finally:
        _cleanup(mcp_tool, "authy2")


def test_session_expired_transport_retry_costs_exactly_one_strike(
    monkeypatch, mcp_tool
):
    """The helper classifies; the caller's generic path owns the strike."""
    monkeypatch.setattr(mcp_tool, "_is_session_expired_error", lambda exc: True)
    monkeypatch.setattr(
        mcp_tool, "_signal_reconnect_and_wait", lambda *a, **kw: True
    )
    monkeypatch.setattr(mcp_tool, "_mcp_loop", MagicMock(is_running=lambda: True))
    _install_stub_server(mcp_tool, "expired2", None)
    try:
        out = mcp_tool._handle_session_expired_and_retry(
            "expired2",
            Exception("Session not found"),
            lambda: mcp_tool.tool_error("502 Bad Gateway"),
            "tools/call get_task",
        )
        # Falls through so the caller can run its own error path.
        assert out is None
        # The helper itself recorded nothing.
        assert mcp_tool._server_error_counts.get("expired2", 0) == 0
    finally:
        _cleanup(mcp_tool, "expired2")


def test_session_expired_retry_returns_application_error(monkeypatch, mcp_tool):
    """The rebuilt session carried the call — surface the tool's own answer."""
    monkeypatch.setattr(mcp_tool, "_is_session_expired_error", lambda exc: True)
    monkeypatch.setattr(
        mcp_tool, "_signal_reconnect_and_wait", lambda *a, **kw: True
    )
    monkeypatch.setattr(mcp_tool, "_mcp_loop", MagicMock(is_running=lambda: True))
    _install_stub_server(mcp_tool, "expired", None)
    try:
        app_error = mcp_tool.tool_error("Not Found: task '1' does not exist.")
        out = mcp_tool._handle_session_expired_and_retry(
            "expired", Exception("Session not found"), lambda: app_error,
            "tools/call get_task",
        )
        assert out == app_error
        assert mcp_tool._server_error_counts.get("expired", 0) == 0
    finally:
        _cleanup(mcp_tool, "expired")
