"""A poisoned OAuth provider must be evicted and rebuilt, not reused.

Hosted OAuth MCP servers effectively never recovered on their own once
parked. The mechanism, confirmed against the installed SDK:

``OAuthClientProvider.async_auth_flow`` is an async GENERATOR that acquires
``self.context.lock`` — an ``anyio.Lock`` — and holds it across every yield
for the whole flow. ``anyio.Lock.release()`` is owner-checked: it raises
``RuntimeError("The current task is not holding this lock")`` unless the
releasing task is the one that acquired it.

So when a connect attempt is cancelled mid-flow (connect timeout, transport
TaskGroup drop, park), the generator is finalized from a different task than
the one holding the lock — or dropped and never finalized — and the lock
stays held forever. Hermes deliberately reuses one provider instance across
reconnects, so every later reconnect blocked on a lock whose owner task no
longer existed.

The first test class pins the SDK behavior these tests rest on, so an SDK
upgrade that changes it fails loudly here rather than silently disarming the
fix. The rest drive Hermes' detection and eviction.

Nothing here touches the network or any real credential.
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import MagicMock

import pytest


anyio = pytest.importorskip("anyio")
pytest.importorskip("mcp.client.auth.oauth2")


# ---------------------------------------------------------------------------
# The SDK premise this fix rests on
# ---------------------------------------------------------------------------


class TestSdkPremise:
    def test_auth_flow_is_an_async_generator_over_an_anyio_lock(self):
        """Runtime shape, not source text: a generator, and an anyio lock.

        Both properties are what make poisoning possible at all. The lock
        being HELD across yields is proven behaviourally by
        ``test_abandoned_flow_latches_the_poison_flag`` below.
        """
        import dataclasses

        import mcp.client.auth.oauth2 as sdk

        assert inspect.isasyncgenfunction(sdk.OAuthClientProvider.async_auth_flow)

        lock_field = next(
            f for f in dataclasses.fields(sdk.OAuthContext) if f.name == "lock"
        )
        assert lock_field.default_factory is anyio.Lock, (
            "OAuthContext.lock is no longer an anyio.Lock — re-verify whether "
            "provider poisoning is still possible before trusting eviction"
        )

    def test_anyio_lock_release_is_owner_checked(self):
        """The property that turns an abandoned flow into a permanent lock."""
        async def _main():
            lock = anyio.Lock()

            async def _flow():
                async with lock:
                    yield "request"

            agen = _flow()

            async def _task_a():
                await agen.asend(None)  # acquires as THIS task

            async with anyio.create_task_group() as tg:
                tg.start_soon(_task_a)

            # The owning task is gone and the lock is still held.
            assert lock.locked()

            with pytest.raises(RuntimeError, match="not holding this lock"):
                await agen.aclose()

            # ...and it stays held, so any later acquire waits forever.
            assert lock.locked()

        asyncio.run(_main())

    def test_a_rebuilt_lock_is_immediately_acquirable(self):
        """The fix's premise: rebuilding the provider clears the deadlock."""
        async def _main():
            async def _try(lock):
                with anyio.move_on_after(0.3):
                    async with lock:
                        return "acquired"
                return "deadlocked"

            poisoned = anyio.Lock()

            async def _flow():
                async with poisoned:
                    yield "request"

            agen = _flow()
            await agen.asend(None)
            assert poisoned.locked()

            assert await _try(anyio.Lock()) == "acquired"

        asyncio.run(_main())


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _provider_stub(*, locked: bool, depth: int, latched: bool = False):
    """A stand-in carrying only what the poison check reads."""
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS

    provider = object.__new__(_HERMES_PROVIDER_CLS)
    provider._hermes_server_name = "probe"
    provider._hermes_flow_depth = depth
    provider._hermes_lock_poisoned = latched
    ctx = MagicMock()
    # A real bool, as anyio.Lock.locked() returns — a bare MagicMock would
    # answer with a truthy Mock and make every provider look poisoned.
    lock = MagicMock()
    lock.locked.return_value = bool(locked)
    ctx.lock = lock
    provider.context = ctx
    return provider


class TestPoisonDetection:
    def test_lock_held_with_no_flow_running_is_poisoned(self):
        assert _provider_stub(locked=True, depth=0)._hermes_lock_is_poisoned()

    def test_lock_held_during_a_live_flow_is_not_poisoned(self):
        """A flow in progress legitimately owns the lock."""
        assert not _provider_stub(locked=True, depth=1)._hermes_lock_is_poisoned()

    def test_free_lock_is_not_poisoned(self):
        assert not _provider_stub(locked=False, depth=0)._hermes_lock_is_poisoned()

    def test_the_latched_flag_alone_is_enough(self):
        """We watched the release fail; trust that over the lock's own state."""
        assert _provider_stub(
            locked=False, depth=0, latched=True
        )._hermes_lock_is_poisoned()

    def test_a_provider_without_a_usable_lock_is_not_poisoned(self):
        """Never fail closed on an SDK shape we don't recognise."""
        from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS

        provider = object.__new__(_HERMES_PROVIDER_CLS)
        provider._hermes_server_name = "probe"
        provider._hermes_flow_depth = 0
        provider._hermes_lock_poisoned = False
        provider.context = MagicMock(lock=None)
        assert not provider._hermes_lock_is_poisoned()


# ---------------------------------------------------------------------------
# Eviction on reconnect
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_oauth_manager as mod

    mod.reset_manager_for_tests()
    yield mod.get_manager()
    mod.reset_manager_for_tests()


def _install_entry(manager, name, provider, url="https://example.test/mcp"):
    from tools.mcp_oauth_manager import _ProviderEntry

    entry = _ProviderEntry(server_url=url, oauth_config=None)
    entry.provider = provider
    manager._entries[manager._key(name)] = entry
    return entry


class TestEvictionOnReconnect:
    def test_a_healthy_provider_is_reused(self, manager, monkeypatch):
        """Reuse across reconnects is deliberate and must be preserved."""
        provider = _provider_stub(locked=False, depth=0)
        _install_entry(manager, "srv", provider)
        monkeypatch.setattr(
            manager, "_build_provider",
            lambda *a, **kw: pytest.fail("rebuilt a healthy provider"),
        )
        assert manager.get_or_build_provider(
            "srv", "https://example.test/mcp", None
        ) is provider

    def test_a_poisoned_provider_is_evicted_and_rebuilt(self, manager, monkeypatch):
        """The regression: the reconnect that used to deadlock forever."""
        poisoned = _provider_stub(locked=True, depth=0)
        _install_entry(manager, "srv", poisoned)

        rebuilt = _provider_stub(locked=False, depth=0)
        monkeypatch.setattr(manager, "_build_provider", lambda *a, **kw: rebuilt)

        got = manager.get_or_build_provider("srv", "https://example.test/mcp", None)
        assert got is rebuilt
        assert got is not poisoned

    def test_a_latched_poisoned_provider_is_evicted(self, manager, monkeypatch):
        latched = _provider_stub(locked=False, depth=0, latched=True)
        _install_entry(manager, "srv", latched)
        rebuilt = _provider_stub(locked=False, depth=0)
        monkeypatch.setattr(manager, "_build_provider", lambda *a, **kw: rebuilt)
        assert manager.get_or_build_provider(
            "srv", "https://example.test/mcp", None
        ) is rebuilt

    def test_eviction_does_not_touch_persisted_tokens(self, manager, monkeypatch, tmp_path):
        """Rebuilding must never cost the user their credentials."""
        tokens = tmp_path / "mcp-tokens"
        tokens.mkdir()
        token_file = tokens / "srv.json"
        token_file.write_text('{"access_token": "kept"}')

        _install_entry(manager, "srv", _provider_stub(locked=True, depth=0))
        monkeypatch.setattr(
            manager, "_build_provider",
            lambda *a, **kw: _provider_stub(locked=False, depth=0),
        )
        manager.get_or_build_provider("srv", "https://example.test/mcp", None)

        assert token_file.exists()
        assert token_file.read_text() == '{"access_token": "kept"}'

    def test_sibling_servers_are_untouched(self, manager, monkeypatch):
        """One server's poisoned provider must not evict anyone else's."""
        poisoned = _provider_stub(locked=True, depth=0)
        healthy = _provider_stub(locked=False, depth=0)
        _install_entry(manager, "broken", poisoned)
        _install_entry(manager, "fine", healthy, url="https://other.test/mcp")

        monkeypatch.setattr(
            manager, "_build_provider",
            lambda *a, **kw: _provider_stub(locked=False, depth=0),
        )
        manager.get_or_build_provider("broken", "https://example.test/mcp", None)

        assert manager._entries[manager._key("fine")].provider is healthy

    def test_eviction_survives_a_provider_that_raises_on_inspection(
        self, manager, monkeypatch
    ):
        """A broken poison check must not break the connect path."""
        provider = _provider_stub(locked=False, depth=0)
        provider._hermes_lock_is_poisoned = MagicMock(side_effect=RuntimeError("nope"))
        _install_entry(manager, "srv", provider)
        assert manager.get_or_build_provider(
            "srv", "https://example.test/mcp", None
        ) is provider

    def test_no_entry_is_a_noop(self, manager):
        assert manager._evict_if_poisoned("never-seen") is False


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# End-to-end: the real bridge against a real anyio.Lock
# ---------------------------------------------------------------------------


def _bare_provider(monkeypatch, tmp_path, server_name="srv"):
    """A real HermesMCPOAuthProvider with only the bridge's collaborators."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from tools import mcp_oauth_manager as mod

    mod.reset_manager_for_tests()
    provider = object.__new__(mod._HERMES_PROVIDER_CLS)
    provider._hermes_server_name = server_name
    provider._hermes_home = str(tmp_path)
    provider._hermes_preregistered = True
    provider._hermes_flow_depth = 0
    provider._hermes_lock_poisoned = False
    provider.context = MagicMock()

    async def _noop_disk_watch(*a, **kw):
        return False

    monkeypatch.setattr(
        mod.MCPOAuthManager, "invalidate_if_disk_changed", _noop_disk_watch
    )

    async def _noop_poison(self, response):
        return None

    monkeypatch.setattr(mod._HERMES_PROVIDER_CLS, "_maybe_flag_poisoned_client", _noop_poison)
    monkeypatch.setattr(
        mod._HERMES_PROVIDER_CLS, "_persist_oauth_metadata_if_changed",
        lambda self: None,
    )
    return provider


def _patch_base_flow_holding(monkeypatch, provider, lock):
    """Replace the SDK's flow with one that holds ``lock`` across its yield.

    The same lock is published on ``provider.context.lock``, because that is
    where the poison check reads it — the SDK's real flow locks exactly that
    object.
    """
    from mcp.client.auth.oauth2 import OAuthClientProvider

    provider.context.lock = lock

    async def _base(self, request):
        async with lock:
            response = yield request

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", _base)


def test_abandoned_flow_latches_the_poison_flag(monkeypatch, tmp_path):
    """The regression, end to end on the real bridge.

    A flow started in one task and abandoned when that task ends leaves the
    SDK's ``context.lock`` held. Closing the bridge afterwards must observe
    the owner-check RuntimeError and latch the provider as poisoned, rather
    than letting it surface at loop shutdown long after the provider has been
    reused and deadlocked.
    """
    provider = _bare_provider(monkeypatch, tmp_path)

    async def _main():
        lock = anyio.Lock()
        _patch_base_flow_holding(monkeypatch, provider, lock)
        gen = provider.async_auth_flow(object())

        async def _task_a():
            await gen.__anext__()          # acquires the lock as THIS task

        async with anyio.create_task_group() as tg:
            tg.start_soon(_task_a)

        assert lock.locked(), "precondition: the abandoned flow holds the lock"
        assert provider._hermes_flow_depth == 1

        # The connect path tears the flow down from a different task.
        await gen.aclose()

    asyncio.run(_main())

    assert provider._hermes_lock_poisoned is True
    assert provider._hermes_flow_depth == 0
    assert provider._hermes_lock_is_poisoned() is True


def test_a_normal_flow_leaves_the_provider_clean(monkeypatch, tmp_path):
    """The healthy path must not be marked poisoned or leak flow depth."""
    provider = _bare_provider(monkeypatch, tmp_path)

    async def _main():
        lock = anyio.Lock()
        _patch_base_flow_holding(monkeypatch, provider, lock)
        request = object()
        gen = provider.async_auth_flow(request)
        assert await gen.__anext__() is request
        assert provider._hermes_flow_depth == 1
        with pytest.raises(StopAsyncIteration):
            await gen.asend(MagicMock())
        assert not lock.locked()

    asyncio.run(_main())

    assert provider._hermes_lock_poisoned is False
    assert provider._hermes_flow_depth == 0
    assert provider._hermes_lock_is_poisoned() is False


def test_same_task_teardown_is_not_treated_as_poisoning(monkeypatch, tmp_path):
    """An ordinary abort by the owning task releases the lock cleanly."""
    provider = _bare_provider(monkeypatch, tmp_path)

    async def _main():
        lock = anyio.Lock()
        _patch_base_flow_holding(monkeypatch, provider, lock)
        gen = provider.async_auth_flow(object())
        await gen.__anext__()
        await gen.aclose()          # same task that acquired it
        assert not lock.locked()

    asyncio.run(_main())

    assert provider._hermes_lock_poisoned is False
    assert provider._hermes_lock_is_poisoned() is False


def test_flow_depth_is_restored_when_the_bridge_raises(monkeypatch, tmp_path):
    """A leaked depth counter would hide a later real poisoning."""
    provider = _bare_provider(monkeypatch, tmp_path)

    from mcp.client.auth.oauth2 import OAuthClientProvider

    async def _base(self, request):
        raise RuntimeError("discovery exploded")
        yield  # pragma: no cover — makes this an async generator

    monkeypatch.setattr(OAuthClientProvider, "async_auth_flow", _base)

    async def _main():
        gen = provider.async_auth_flow(object())
        with pytest.raises(RuntimeError, match="discovery exploded"):
            await gen.__anext__()

    asyncio.run(_main())
    assert provider._hermes_flow_depth == 0


def test_a_truthy_non_bool_locked_is_not_treated_as_poisoned():
    """Only an unambiguous True counts.

    Guards a hazard this repo has been bitten by: every attribute of a
    MagicMock answers with a truthy Mock, so a lenient ``bool(...)`` here
    would report a healthy provider as poisoned and rebuild it on every
    single reconnect.
    """
    from tools.mcp_oauth_manager import _HERMES_PROVIDER_CLS

    provider = object.__new__(_HERMES_PROVIDER_CLS)
    provider._hermes_server_name = "probe"
    provider._hermes_flow_depth = 0
    provider._hermes_lock_poisoned = False
    provider.context = MagicMock()  # context.lock.locked() -> truthy Mock
    assert provider._hermes_lock_is_poisoned() is False

    # ...but a latched observation still wins.
    provider._hermes_lock_poisoned = True
    assert provider._hermes_lock_is_poisoned() is True


# ---------------------------------------------------------------------------
# A stalled in-flight flow is poison too
# ---------------------------------------------------------------------------


class TestStalledFlowDetection:
    """Depth alone goes blind exactly when it matters.

    Once the lock is poisoned, the NEXT flow blocks inside the SDK while
    holding a depth slot. Depth stays pinned above zero, so a depth-only
    check reports "healthy" forever — on a provider that can never connect.
    """

    def _stalled(self, mcp_mod, *, age):
        provider = _provider_stub(locked=True, depth=1)
        provider._hermes_flow_started_mono = (
            __import__("time").monotonic() - age
        )
        return provider

    def test_a_long_stalled_flow_is_poison(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools import mcp_oauth_manager as mod

        provider = self._stalled(mod, age=mod._STALLED_AUTH_FLOW_SEC + 60)
        assert provider._hermes_lock_is_poisoned() is True

    def test_a_flow_still_within_the_window_is_not(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from tools import mcp_oauth_manager as mod

        provider = self._stalled(mod, age=5.0)
        assert provider._hermes_lock_is_poisoned() is False

    def test_an_in_flight_flow_with_no_start_stamp_is_not_poison(self):
        """Never invent a verdict from missing evidence."""
        provider = _provider_stub(locked=True, depth=1)
        provider._hermes_flow_started_mono = None
        assert provider._hermes_lock_is_poisoned() is False


# ---------------------------------------------------------------------------
# Eviction must not discard a concurrently-rebuilt provider
# ---------------------------------------------------------------------------


def test_eviction_is_identity_guarded(manager, monkeypatch):
    """Another thread may rebuild between the check and the pop.

    Popping blindly would throw away a fresh, healthy provider and make both
    callers redo the work.
    """
    poisoned = _provider_stub(locked=True, depth=0)
    _install_entry(manager, "srv", poisoned)

    replacement = _provider_stub(locked=False, depth=0)
    real_evict_if_poisoned = manager._evict_if_poisoned

    # Simulate the race: a rebuild lands while _evict_if_poisoned is deciding.
    original_lock_check = type(poisoned)._hermes_lock_is_poisoned

    def _swap_then_answer(self):
        manager._entries[manager._key("srv")].provider = replacement
        return original_lock_check(self)

    # A scoped context, not undo(): undo() would also revert the fixture's
    # HERMES_HOME, and _key() resolves the home at call time.
    with monkeypatch.context() as m:
        m.setattr(type(poisoned), "_hermes_lock_is_poisoned", _swap_then_answer)
        evicted = real_evict_if_poisoned("srv")

    assert evicted is False, "evicted a provider that had already been replaced"
    assert manager._entries[manager._key("srv")].provider is replacement


def test_flow_depth_is_released_only_after_the_lock_is(monkeypatch, tmp_path):
    """Decrementing before aclose() would let a racing reconnect see
    "lock held, no flow running" during a perfectly normal teardown."""
    provider = _bare_provider(monkeypatch, tmp_path)
    observed = {}

    class _WatchedLock:
        """Delegates to a real anyio.Lock, noting the depth at release."""

        def __init__(self, on_release):
            self._lock = anyio.Lock()
            self._on_release = on_release

        async def __aenter__(self):
            return await self._lock.__aenter__()

        async def __aexit__(self, *exc):
            self._on_release()
            return await self._lock.__aexit__(*exc)

        def locked(self):
            return self._lock.locked()

    async def _main():
        lock = _WatchedLock(
            lambda: observed.setdefault(
                "depth_at_release", provider._hermes_flow_depth
            )
        )
        _patch_base_flow_holding(monkeypatch, provider, lock)
        gen = provider.async_auth_flow(object())
        await gen.__anext__()
        await gen.aclose()

    asyncio.run(_main())

    assert observed.get("depth_at_release") == 1, (
        "the depth slot was released before the lock — a reconnect racing "
        "here would rebuild a healthy provider"
    )
    assert provider._hermes_flow_depth == 0


def test_the_stall_threshold_never_undercuts_the_configured_browser_wait():
    """The browser login is awaited INSIDE the flow.

    `oauth.timeout` is operator-configurable above its 300s default, so a
    fixed bound could judge a provider poisoned while a human was still
    legitimately logging in.
    """
    from tools.mcp_oauth_manager import _STALLED_AUTH_FLOW_SEC

    provider = _provider_stub(locked=True, depth=1)

    provider._hermes_callback_timeout = 0.0
    assert provider._hermes_stall_threshold() == _STALLED_AUTH_FLOW_SEC

    provider._hermes_callback_timeout = 300.0
    assert provider._hermes_stall_threshold() == _STALLED_AUTH_FLOW_SEC

    # A long configured wait raises the bar above it, with headroom.
    provider._hermes_callback_timeout = 3600.0
    assert provider._hermes_stall_threshold() == 7200.0

    provider._hermes_callback_timeout = "not-a-number"
    assert provider._hermes_stall_threshold() == _STALLED_AUTH_FLOW_SEC


def test_a_flow_inside_a_long_configured_wait_is_not_poison():
    import time as _time

    provider = _provider_stub(locked=True, depth=1)
    provider._hermes_callback_timeout = 3600.0
    provider._hermes_flow_started_mono = _time.monotonic() - 2000.0
    assert provider._hermes_lock_is_poisoned() is False, (
        "evicted a provider that was still within its configured browser wait"
    )
