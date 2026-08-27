"""Regression: a live stdio MCP child must never be classified as dead.

Upstream commit 786f37071 ("psutil.pid_exists for stdio children liveness —
Windows footgun") inverted the loop in ``MCPServerTask._stdio_children_dead``:
an ALIVE tracked pid returned ``True`` ("every child has exited"), so the
fast-fail added in #81995 fired on every call of every stdio server whose
child pids had been captured at spawn. In production this surfaced as

    TimeoutError: MCP stdio subprocess for '<server>' has exited; failing
    the call fast instead of waiting 300s

for subprocesses that were demonstrably alive (case-memory, context-read,
exec, tempo, asana-write, sie — 2026-08-26).

These tests use REAL pids on purpose — the test process itself (alive) and a
subprocess that has been waited on (gone) — so the core cases exercise the
genuine psutil path with no mocks. Accepted, documented limits of the probe
(unchanged from #81995): a recycled pid or an unreaped zombie counts as alive,
which is the fail-SAFE direction (the call waits out the tool timeout instead
of being failed fast) and never the inverted failure this file guards against.
"""
import asyncio
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from tools.mcp_tool import MCPServerTask


def _server(pids, http=False):
    server = MCPServerTask("srv")
    server._config = {"url": "https://example.invalid/mcp"} if http else {"command": "true"}
    server._stdio_child_pids = set(pids)
    return server


@pytest.fixture
def gone_pid():
    """A pid that certainly does not exist any more: a child that ran and was reaped."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


LIVE_PID = os.getpid()


# --- the regression -----------------------------------------------------------

def test_all_children_alive_is_not_dead():
    """The production symptom: one live tracked child → must NOT be 'dead'.
    Fails on 786f37071 (returned True)."""
    assert _server([LIVE_PID])._stdio_children_dead() is False


def test_one_dead_one_alive_is_not_dead(gone_pid):
    """A helper child that exited while the server child lives is not 'every child has exited'.
    Fails on 786f37071 (returned True as soon as the live pid was probed)."""
    assert _server([gone_pid, LIVE_PID])._stdio_children_dead() is False
    assert _server([LIVE_PID, gone_pid])._stdio_children_dead() is False


def test_all_children_gone_is_dead(gone_pid):
    assert _server([gone_pid])._stdio_children_dead() is True


def test_no_tracked_pids_is_unknown_not_dead():
    assert _server([])._stdio_children_dead() is False
    server = MCPServerTask("srv")
    server._config = {"command": "true"}
    if hasattr(server, "_stdio_child_pids"):
        del server._stdio_child_pids
    assert server._stdio_children_dead() is False


def test_http_transport_is_never_dead(gone_pid):
    assert _server([gone_pid], http=True)._stdio_children_dead() is False


# --- psutil-unavailable fallback (the os.kill probe, #81995 semantics) ----------

def _without_psutil():
    return patch.dict(sys.modules, {"psutil": None})


def test_fallback_alive_pid_is_not_dead():
    with _without_psutil():
        assert _server([LIVE_PID])._stdio_children_dead() is False


def test_fallback_gone_pid_is_dead(gone_pid):
    with _without_psutil():
        assert _server([gone_pid])._stdio_children_dead() is True


def test_fallback_permission_error_counts_as_alive():
    """A pid we exist-but-cannot-signal is treated alive (never fail fast on uncertainty)."""
    with _without_psutil(), patch("tools.mcp_tool.os.kill", side_effect=PermissionError):
        assert _server([12345])._stdio_children_dead() is False


def test_fallback_process_lookup_error_counts_as_dead():
    with _without_psutil(), patch("tools.mcp_tool.os.kill", side_effect=ProcessLookupError):
        assert _server([12345])._stdio_children_dead() is True


def test_fallback_generic_os_error_counts_as_alive():
    """Any other OSError is uncertainty, and uncertainty never reads as dead."""
    with _without_psutil(), patch("tools.mcp_tool.os.kill", side_effect=OSError):
        assert _server([12345])._stdio_children_dead() is False


# --- the in-flight watcher (#81995) — the other production consumer ----------------

def test_watcher_does_not_resolve_while_a_child_lives():
    """With a live tracked child the watcher must keep polling — it never cancels an in-flight
    RPC. On 786f37071 it resolved immediately, cancelling every stdio call."""
    server = _server([LIVE_PID])

    async def _run():
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(server._watch_stdio_children(), timeout=0.6)

    asyncio.run(_run())


def test_watcher_resolves_once_the_last_child_is_gone(gone_pid):
    server = _server([LIVE_PID, gone_pid])

    async def _run():
        watcher = asyncio.create_task(server._watch_stdio_children())
        await asyncio.sleep(0.4)
        assert not watcher.done(), "watcher resolved while a tracked child was still alive"
        server._stdio_child_pids = {gone_pid}  # the live one exits between polls
        await asyncio.wait_for(watcher, timeout=3.0)

    asyncio.run(_run())
