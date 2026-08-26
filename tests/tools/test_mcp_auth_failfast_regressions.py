"""Regressions pinning the auth-failure fail-fast chain end to end.

Incident shape (2026-08-26 gateway outage class): a hosted MCP server's
OAuth token becomes unusable while the connection is parked. On the next
connect the SDK falls through toward browser re-authorization, which a
non-interactive gateway can never complete. The chain that must hold:

  redirect boundary raises OAuthNonInteractiveError fast (#57836)
    -> _classify_mcp_failure says "permanent"
      -> run() parks after ONE attempt
        -> the park warning names `hermes mcp login <server>`

These tests pin each link plus the deliberate NON-escalations: a plain
timeout and a WAF-style 405 stay "transient" (bounded retry ladder, then
parked self-probe), so ordinary network weather never masquerades as an
auth failure.
"""

import asyncio
import logging

import httpx
import pytest

from tools.mcp_oauth import OAuthNonInteractiveError
from tools.mcp_tool import MCPServerTask, _classify_mcp_failure


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://mcp.example.com/v1/mcp/authv2")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError(str(status), request=request, response=response)


# ── classification links ─────────────────────────────────────────────────────

def test_non_interactive_oauth_error_is_permanent():
    """The redirect-boundary raise must classify as permanent so the park
    happens after ONE attempt — never a 300s browser-flow wait per retry."""
    exc = OAuthNonInteractiveError("browser authorization required")
    assert _classify_mcp_failure(exc) == "permanent"


def test_plain_timeout_stays_transient():
    """The pre-fix incident symptom (bare TimeoutError) is network weather:
    it must keep the bounded retry ladder, not park as auth."""
    assert _classify_mcp_failure(asyncio.TimeoutError()) == "transient"


def test_waf_style_405_stays_transient():
    """A 405 (observed WAF/Human-Verification challenge shape) is NOT an
    auth failure: 401/403 are permanent, everything else keeps the retry
    ladder. Pinned deliberately — a WAF-specific verdict is a separate,
    additive change and must not silently repurpose the auth path."""
    assert _classify_mcp_failure(_http_error(405)) == "transient"


def test_401_and_403_are_permanent():
    assert _classify_mcp_failure(_http_error(401)) == "permanent"
    assert _classify_mcp_failure(_http_error(403)) == "permanent"


# ── park behavior + actionable remedy ────────────────────────────────────────

@pytest.mark.no_isolate
def test_auth_park_is_immediate_and_names_relogin(monkeypatch, tmp_path, caplog):
    """OAuthNonInteractiveError on connect parks after ONE transport attempt
    and the single warning carries the `hermes mcp login <server>` remedy —
    the actionable line the ops watchdog forwards verbatim."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from tools import mcp_tool

    _real_sleep = asyncio.sleep

    async def _fast_sleep(_delay, *a, **kw):
        await _real_sleep(0)

    monkeypatch.setattr(mcp_tool.asyncio, "sleep", _fast_sleep)

    state = {"transport_calls": 0, "parked": False}

    async def _scenario():
        class _Task(MCPServerTask):
            def _is_http(self):
                return False

            def _deregister_tools(self):
                state["parked"] = True
                self._registered_tool_names = []

            async def _run_stdio(self, config):
                state["transport_calls"] += 1
                raise OAuthNonInteractiveError(
                    "MCP OAuth requires browser authorization but no "
                    "interactive session is available"
                )

        task = _Task("jira")

        with caplog.at_level(logging.DEBUG, logger="tools.mcp_tool"):
            run_task = asyncio.ensure_future(task.run({"command": "x"}))
            for _ in range(500):
                await _real_sleep(0)
                if state["parked"]:
                    break

        assert state["parked"], "auth failure never parked"
        assert state["transport_calls"] == 1, (
            f"auth failure burned {state['transport_calls']} attempts — "
            "must park after the first"
        )

        task._shutdown_event.set()
        task._reconnect_event.set()
        try:
            await asyncio.wait_for(run_task, timeout=15)
        except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
            run_task.cancel()

    asyncio.run(_scenario())

    auth_warnings = [
        r for r in caplog.records
        if r.levelno == logging.WARNING
        and "failed initial authentication" in r.getMessage()
    ]
    assert len(auth_warnings) == 1, (
        f"expected exactly one auth park warning, got "
        f"{[r.getMessage() for r in auth_warnings]}"
    )
    message = auth_warnings[0].getMessage()
    assert "hermes mcp login jira" in message, message
    assert "OAuthNonInteractiveError" in message, message
