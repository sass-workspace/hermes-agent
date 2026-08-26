"""Regressions for the Socket Mode closed-session zombie loop.

Incident shape (observed live): after a teardown, an ORPHANED
SocketModeClient kept retrying ``connect()`` against a permanently closed
aiohttp session — ``slack_bolt.AsyncApp: Failed to connect (error:
Session is closed); Retrying...`` every ~ping_interval, thousands of
times over one night, healed only by a process restart. Two mechanisms
end it:

  * teardown quiesces to a FIXED POINT: cancel rounds re-snapshot the
    client's task attrs, beating the monitor->connect() rebind race
    (slackapi/python-slack-sdk#1913);
  * the watchdog sweeps spawned clients and reaps any orphan that still
    runs tasks while no longer being the current handler's client.
"""

import asyncio

import pytest

from plugins.platforms.slack.adapter import (
    SlackAdapter,
    _quiesce_socket_client,
    _socket_client_tasks,
)


class _RebindingClient:
    """Fake SocketModeClient whose monitor recreates itself when cancelled —
    the moving target a single snapshot-cancel misses."""

    def __init__(self, generations: int):
        self._generations = generations
        self.current_session_monitor = None
        self.message_processor = None
        self.message_receiver = None

    def spawn(self):
        self.current_session_monitor = asyncio.ensure_future(self._monitor())
        self.message_receiver = asyncio.ensure_future(self._sleeper())

    def _respawn_if_budget(self):
        # connect() rebinds task attributes: on cancellation the next
        # generation appears in our attrs, exactly like the SDK's
        # monitor_current_session() kicking off a fresh connect().
        if self._generations > 0:
            self._generations -= 1
            loop = asyncio.get_running_loop()
            self.message_receiver = loop.create_task(self._sleeper())

    async def _monitor(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self._respawn_if_budget()
            raise

    async def _sleeper(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self._respawn_if_budget()
            raise


def test_quiesce_beats_the_rebind_race():
    """Three self-recreating generations must still end fully quiet."""

    async def _scenario():
        client = _RebindingClient(generations=3)
        client.spawn()
        assert _socket_client_tasks(client)
        assert await _quiesce_socket_client(client) is True
        assert _socket_client_tasks(client) == []

    asyncio.run(_scenario())


class _EternalClient:
    """Pathological client: every snapshot of its attrs sees a FRESH live
    task, no matter how many were cancelled — the theoretical worst case of
    the rebind race. Models "recreates faster than we cancel"."""

    current_session_monitor = None
    message_processor = None

    def __init__(self):
        self.made: list = []

    @property
    def message_receiver(self):
        task = asyncio.ensure_future(asyncio.sleep(3600))
        self.made.append(task)
        return task


def test_quiesce_gives_up_boundedly_on_pathological_client():
    """A client that recreates forever must not hang shutdown — the bound
    returns False instead of looping."""

    async def _scenario():
        client = _EternalClient()
        assert await _quiesce_socket_client(client, rounds=3) is False
        for task in client.made:  # cleanup for the test loop
            task.cancel()
        await asyncio.gather(*client.made, return_exceptions=True)

    asyncio.run(_scenario())


def _bare_adapter() -> SlackAdapter:
    adapter = SlackAdapter.__new__(SlackAdapter)
    adapter._handler = None
    adapter._spawned_socket_clients = __import__("weakref").WeakSet()
    return adapter


class _Handler:
    def __init__(self, client):
        self.client = client


def test_sweep_reaps_orphan_but_spares_current():
    """The sweep must end the orphan's tasks and leave the current client's
    tasks alone (no duplicate processing: exactly one live receiver after)."""

    async def _scenario():
        adapter = _bare_adapter()

        orphan = _RebindingClient(generations=0)
        orphan.spawn()
        current = _RebindingClient(generations=0)
        current.spawn()

        adapter._spawned_socket_clients.add(orphan)
        adapter._spawned_socket_clients.add(current)
        adapter._handler = _Handler(current)

        await adapter._sweep_orphaned_socket_clients()

        assert _socket_client_tasks(orphan) == [], "orphan kept retrying"
        live = _socket_client_tasks(current)
        assert live, "current client's tasks must survive the sweep"
        assert orphan not in adapter._spawned_socket_clients

        # Cleanup.
        await _quiesce_socket_client(current)

    asyncio.run(_scenario())


def test_sweep_with_no_orphans_is_a_no_op():
    async def _scenario():
        adapter = _bare_adapter()
        current = _RebindingClient(generations=0)
        current.spawn()
        adapter._spawned_socket_clients.add(current)
        adapter._handler = _Handler(current)

        await adapter._sweep_orphaned_socket_clients()
        assert _socket_client_tasks(current), "healthy current client touched"
        await _quiesce_socket_client(current)

    asyncio.run(_scenario())
