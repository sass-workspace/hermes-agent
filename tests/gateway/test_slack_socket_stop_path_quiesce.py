"""The SHIPPED stop path must quiesce a monitor that recreates itself.

Complements test_slack_socket_zombie_recovery.py (which exercises the
helpers): here `_stop_socket_mode_handler` itself is driven with a
handler whose `close_async()` behaves like slack_sdk's — it closes the
session and cancels the snapshot it knows about — while the client's
monitor rebinds a fresh receiver on cancellation (the #1913 race). The
adapter's post-close quiesce is what turns that into silence.
"""

import asyncio
import weakref

from plugins.platforms.slack.adapter import SlackAdapter, _socket_client_tasks


class _RaceyClient:
    def __init__(self):
        self.current_session_monitor = None
        self.message_processor = None
        self.message_receiver = None
        self.rebinds = 0

    def spawn(self):
        self.current_session_monitor = asyncio.ensure_future(self._monitor())
        self.message_receiver = asyncio.ensure_future(asyncio.sleep(3600))

    async def _monitor(self):
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            # SDK behavior: the monitor kicks off connect(), which rebinds a
            # NEW receiver task — the one a single snapshot-cancel misses.
            self.rebinds += 1
            loop = asyncio.get_running_loop()
            self.message_receiver = loop.create_task(asyncio.sleep(3600))
            raise


class _Handler:
    """Mimics AsyncSocketModeHandler.close_async(): closes the session and
    cancels the tasks it can see at that moment."""

    def __init__(self, client):
        self.client = client
        self.closed = False

    async def close_async(self):
        self.closed = True
        for task in _socket_client_tasks(self.client):
            task.cancel()
        await asyncio.sleep(0)


def test_stop_path_quiesces_rebinding_client():
    async def _scenario():
        adapter = SlackAdapter.__new__(SlackAdapter)
        client = _RaceyClient()
        client.spawn()
        # Let the tasks actually START (enter their try blocks): a task
        # cancelled before its first step never runs its except handler,
        # which would make the race — and this test — vacuous.
        await asyncio.sleep(0)
        adapter._handler = _Handler(client)
        adapter._socket_mode_task = asyncio.ensure_future(asyncio.sleep(3600))
        adapter._spawned_socket_clients = weakref.WeakSet([client])

        await adapter._stop_socket_mode_handler()

        assert adapter._handler is None
        assert adapter._socket_mode_task is None
        assert client.rebinds >= 1, "the race never happened — test is vacuous"
        assert _socket_client_tasks(client) == [], (
            "a rebound receiver survived the shipped stop path"
        )

    asyncio.run(_scenario())
