"""The Slack SDK must stop reconnecting a Socket Mode client that was closed.

Until slack-sdk 3.44.1 ``SocketModeClient.connect()`` looped ``while True``: a client closed
during teardown whose websocket connect kept failing retried forever ("Failed to connect …
Retrying…" every ping interval, for the life of the process — the closed-session zombie loop).
The adapter carries no workaround for that; it relies on the SDK's fix (slackapi#1956). These
tests fail when the installed SDK would bring the loop back.
"""
import asyncio
from importlib.metadata import version

import pytest

pytest.importorskip("slack_sdk.socket_mode.aiohttp")


def _version_tuple(text: str) -> tuple:
    return tuple(int(part) for part in text.split(".")[:3] if part.isdigit())


def test_the_installed_sdk_carries_the_closed_client_fix():
    assert _version_tuple(version("slack-sdk")) >= (3, 44, 1), (
        "slack-sdk < 3.44.1 reconnects a closed Socket Mode client forever; upgrade the SDK or "
        "restore the adapter-side teardown quiesce")


def test_a_closed_client_stops_connecting_instead_of_retrying_forever():
    from slack_sdk.socket_mode.aiohttp import SocketModeClient

    async def scenario():
        client = SocketModeClient(app_token="xapp-test", ping_interval=0.01)
        attempts = {"n": 0}

        async def _refused(*args, **kwargs):
            attempts["n"] += 1
            if attempts["n"] >= 2:
                client.closed = True  # closed by a teardown while connect() is still retrying
            raise ConnectionError("connect refused")

        client.wss_uri = "wss://example.test/link"
        client.aiohttp_client_session.ws_connect = _refused
        try:
            await asyncio.wait_for(client.connect(), timeout=5)
        finally:
            client.closed = True
            await client.aiohttp_client_session.close()
        return attempts["n"]

    assert asyncio.run(scenario()) == 2, "connect() must return once the client is closed"
