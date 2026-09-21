"""An OAuth flow abandoned mid-handshake must not wedge its provider.

One provider serialises every request of an MCP server behind one lock. HTTPX may close the auth
generator from a DIFFERENT task than the one that entered it (a cancelled request, a transport
teardown). With a task-owned lock that close cannot release it: the provider stays locked, every
later request of that server blocks forever, and the server can never connect again until the
process restarts. The lock must therefore carry no task ownership: whoever abandons the flow,
the next one acquires.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp.client.auth.oauth2", reason="MCP SDK OAuth module required")


async def _noop_redirect(_url: str) -> None:
    return None


async def _noop_callback():
    return "code", None


async def _provider(tmp_path, monkeypatch):
    from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
    from pydantic import AnyUrl

    from tools.mcp_oauth import HermesTokenStorage
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS, reset_manager_for_tests

    assert _HERMES_PROVIDER_CLS is not None
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    reset_manager_for_tests()
    storage = HermesTokenStorage("srv")
    await storage.set_tokens(OAuthToken(
        access_token="old_access", token_type="Bearer", expires_in=3600, refresh_token="old_refresh"))
    redirect = AnyUrl("http://127.0.0.1:12345/callback")
    await storage.set_client_info(OAuthClientInformationFull(
        client_id="test-client", redirect_uris=[redirect],
        grant_types=["authorization_code", "refresh_token"], response_types=["code"],
        token_endpoint_auth_method="none"))
    return _HERMES_PROVIDER_CLS(
        server_name="srv", server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(redirect_uris=[redirect], client_name="Hermes Agent"),
        storage=storage, redirect_handler=_noop_redirect, callback_handler=_noop_callback)


@pytest.mark.asyncio
async def test_a_flow_abandoned_from_another_task_does_not_wedge_the_provider(tmp_path, monkeypatch):
    from tools.mcp_tool import sdk_httpx

    httpx = sdk_httpx()
    provider = await _provider(tmp_path, monkeypatch)
    request = httpx.Request("POST", "https://example.com/mcp")

    # Task A enters the flow and answers the resource request with a 401: the SDK starts an OAuth
    # transition and suspends on ITS OWN request — mid-handshake, with the provider lock held.
    flow = provider.async_auth_flow(request)
    holder: dict = {}

    async def _enter():
        outbound = await flow.__anext__()
        holder["handshake_request"] = await flow.asend(httpx.Response(401, request=outbound))

    await asyncio.wait_for(asyncio.ensure_future(_enter()), timeout=5)
    assert holder["handshake_request"] is not request, "the flow must be inside the OAuth transition"

    # Task B — not the task that entered the flow — abandons it.
    await asyncio.wait_for(asyncio.ensure_future(flow.aclose()), timeout=5)

    # A second flow on the same provider must get the lock: it reaches its first request.
    second = provider.async_auth_flow(httpx.Request("POST", "https://example.com/mcp"))
    try:
        outbound = await asyncio.wait_for(second.__anext__(), timeout=2)
    finally:
        await second.aclose()
    assert outbound.url.host == "example.com"
