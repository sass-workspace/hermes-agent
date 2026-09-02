"""Suppress *consecutive* identical read-only tool calls within one turn.

A turn that calls ``mcp__asana__get_task(gid="123")`` four times in a row
gets the same payload four times. Each copy is appended to the conversation
permanently — so it inflates context for the rest of the session, and
inflates the latency of every subsequent request along with it. The model is
not learning anything on calls two through four; the answer is one message
up.

``read_file`` has solved this for files since forever. But it can do
something this module cannot: it has a **freshness signal** (the file's
mtime), so it knows whether the content actually changed. For a hosted MCP
server there is no such signal. Anything cached here could have been changed
underneath us — by the user in their browser, by a cron job, by another
agent, or by the service itself for a time-dependent read.

That asymmetry sets the whole design. With no way to *verify* freshness, the
only safe thing to suppress is a repeat that cannot have gone stale for a
reason we could have known about. So:

**Only immediately-consecutive repeats are suppressed.** If the model called
this exact tool with these exact arguments as its previous tool call, and
nothing else has happened since, a second call cannot tell it anything new —
that is the pathological loop this module exists for. The moment ANY other
tool runs, the cache forgets: a re-read after other work is a deliberate
re-read, and it runs.

That is a deliberately smaller optimisation than "remember every read for
the whole turn". It still catches the case that actually hurts (a model
re-asking the same question in a loop), and it shrinks the staleness window
to a single model round-trip — the same window any single tool call already
has, and therefore not a new risk.

Three further rules:

1. **Read-only only.** A tool qualifies when its MCP discovery annotations
   carry ``readOnlyHint: true`` exactly — the same fail-closed rule the trust
   gate uses. Missing or malformed annotations mean write-capable.
2. **Any write invalidates, and does so with a generation bump.** Tool calls
   can run CONCURRENTLY, so "the write invalidated the cache" is not enough
   on its own: a read that started before the write could otherwise record
   its pre-write result afterwards. Each call carries the generation it
   observed and is dropped on record if a write has intervened.
3. **Errors are never suppressed.** A failed call stays retryable.

Scoped to the turn and dropped when it ends.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Bound the registry; turns normally end by being dropped, but a crashed turn
# must not leak.
_MAX_TRACKED_TURNS = 32
# After this many suppressed repeats of one call, stop being polite about it.
# Mirrors read_file's escalation: a weak tool-follower that ignores the stub
# would otherwise burn its whole iteration budget re-asking.
_HARD_BLOCK_AFTER = 2

SUPPRESSED_MESSAGE = (
    "Identical to your previous tool call, and nothing has run in between. "
    "The result from that call is directly above in this conversation and is "
    "still current — read it instead of calling again."
)


@dataclass
class _TurnState:
    # Key of the most recently completed read-only call, or None when the
    # last thing that happened was anything else.
    last_key: Optional[str] = None
    # Consecutive suppressed repeats of last_key.
    hits: int = 0
    # Bumped by every write. A read records only if the generation it
    # observed is still current — see the module docstring, rule 2.
    generation: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


_turns: "OrderedDict[str, _TurnState]" = OrderedDict()
_registry_lock = threading.Lock()


def _call_key(tool_name: str, args: Any) -> Optional[str]:
    """Stable key for (tool, arguments), or None if the args won't serialize.

    ``sort_keys`` so argument order cannot make two identical calls look
    different, and a hash so a large argument payload is not held for the
    life of the turn.
    """
    try:
        encoded = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        # Broad on purpose: `default=str` runs arbitrary __str__/__repr__ code,
        # which can raise anything at all. An argument we cannot key is simply
        # never suppressed — the safe direction — and must never propagate out
        # of an optimisation into the tool call itself.
        logger.debug("turn read cache: arguments could not be keyed", exc_info=True)
        return None
    digest = hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()[:32]
    return f"{tool_name}:{digest}"


def _state(turn_id: str, create: bool) -> Optional[_TurnState]:
    with _registry_lock:
        state = _turns.get(turn_id)
        if state is None:
            if not create:
                return None
            state = _turns[turn_id] = _TurnState()
            while len(_turns) > _MAX_TRACKED_TURNS:
                _turns.popitem(last=False)
        _turns.move_to_end(turn_id)
        return state


def check(turn_id: str, tool_name: str, args: Any) -> Tuple[Optional[str], int]:
    """Decide whether this call repeats the immediately preceding one.

    Returns ``(replacement_result_or_None, generation)``. The generation must
    be handed back to :func:`record` so a write that lands mid-call can void
    the recording.
    """
    if not turn_id:
        return None, 0
    key = _call_key(tool_name, args)
    # create=True: the turn must be TRACKED from its first read, not from its
    # first recording. An untracked turn is invisible to invalidate_all(), so
    # a read that started before a plugin's write would record afterwards with
    # a generation nobody had bumped — the exact stale result the generation
    # guard exists to prevent.
    state = _state(turn_id, create=True)
    if state is None:  # pragma: no cover — create=True always returns one
        return None, 0
    with state.lock:
        generation = state.generation
        if key is None or state.last_key != key:
            # This call is NOT a repeat, so whatever was remembered is no
            # longer the immediately preceding one. Forget it here rather
            # than relying on the caller to record a replacement: a read
            # that ERRORS is deliberately never recorded, so leaving the old
            # key in place would let A -> failed B -> A be answered from A's
            # stale result, and would carry B's hit count into it.
            state.last_key = None
            state.hits = 0
            return None, generation
        state.hits += 1
        hits = state.hits

    if hits > _HARD_BLOCK_AFTER:
        from tools.registry import tool_error

        return (
            tool_error(
                f"BLOCKED: you have called '{tool_name}' with these exact "
                f"arguments {hits + 1} times in a row with nothing in "
                f"between. The result is already in your context. STOP "
                f"repeating this call and continue with the task.",
                repeated_calls=hits + 1,
                suppressed=True,
            ),
            generation,
        )

    return (
        json.dumps(
            {
                "status": "unchanged",
                "message": SUPPRESSED_MESSAGE,
                "tool": tool_name,
                "suppressed": True,
                "content_returned": False,
            },
            ensure_ascii=False,
        ),
        generation,
    )


def record(turn_id: str, tool_name: str, args: Any, generation: int) -> bool:
    """Remember this read as the turn's most recent call. True if recorded.

    Dropped when a write bumped the generation while this call was in
    flight: concurrent batches mean "the write invalidated first" is not
    enough on its own to keep a pre-write result out of the cache.
    """
    if not turn_id:
        return False
    key = _call_key(tool_name, args)
    if key is None:
        return False
    # create=False: if this turn was evicted from the capped registry while
    # the read was in flight, recreating it here would resurrect it at
    # generation 0 — losing any invalidate_all() that happened in between and
    # caching a pre-write result. `check()` already created the turn, so a
    # missing one means it was evicted, and dropping the record is correct.
    state = _state(turn_id, create=False)
    if state is None:
        logger.debug(
            "turn read cache: turn %s was evicted mid-call; not recording",
            turn_id,
        )
        return False
    with state.lock:
        if state.generation != generation:
            logger.debug(
                "turn read cache: dropping %s — a write landed mid-call",
                tool_name,
            )
            return False
        state.last_key = key
        state.hits = 0
        return True


def invalidate(turn_id: str) -> None:
    """Forget the last call and void any read still in flight.

    Called whenever anything that is not a proven read-only tool runs. The
    generation bump is what makes it safe under concurrency.
    """
    if not turn_id:
        return
    state = _state(turn_id, create=True)
    if state is None:  # pragma: no cover
        return
    with state.lock:
        state.generation += 1
        state.last_key = None
        state.hits = 0


def invalidate_all() -> None:
    """Forget every turn's last call.

    For dispatch paths that can mutate state but cannot identify the turn
    they belong to — notably ``PluginAPI.dispatch_tool``, which plugins may
    call from a hook that runs before the turn id is bound, or from
    background code with no turn at all.

    Blunt on purpose. The cache holds one key per turn and only ever
    suppresses a back-to-back repeat, so the entire cost of being wrong here
    is one extra read; the cost of being wrong the other way is a stale
    answer.
    """
    with _registry_lock:
        states = list(_turns.values())
    for state in states:
        with state.lock:
            state.generation += 1
            state.last_key = None
            state.hits = 0


def note_tool_dispatch(tool_name: str, turn_id: str) -> None:
    """Apply the invalidate-on-write rule for one about-to-run tool.

    THE single expression of that rule, for every dispatch path that does
    not go through ``model_tools.handle_function_call``. Anything not
    provably a read-only MCP tool invalidates; read-only MCP tools are left
    alone so the dispatcher can do its own check-and-record.

    It exists as a named function because the rule kept being missed: it was
    added at one call site at a time, and each miss was a silently reachable
    stale read rather than a visible failure. A grep for this name now finds
    every path that has considered the question.
    """
    try:
        if not is_read_only_tool(tool_name):
            if turn_id:
                invalidate(turn_id)
            else:
                invalidate_all()
    except Exception:
        logger.debug(
            "turn read cache: dispatch note failed for %s", tool_name,
            exc_info=True,
        )


def finish_turn(turn_id: str) -> None:
    """Drop a turn's state once it ends."""
    if not turn_id:
        return
    with _registry_lock:
        _turns.pop(turn_id, None)


def reset_for_tests() -> None:
    """Drop every tracked turn. Test-support only."""
    with _registry_lock:
        _turns.clear()


# Every MCP tool is registered as ``mcp__<server>__<tool>``. Checked here as a
# literal so this module never imports tools.mcp_tool just to learn the
# prefix — see is_read_only_tool.
_MCP_REGISTRY_PREFIX = "mcp__"


def is_read_only_tool(tool_name: str) -> bool:
    """True only for a tool we can positively prove is a read.

    Currently: MCP tools whose discovery annotations carry
    ``readOnlyHint: true`` exactly. Everything else — core tools, plugins,
    MCP tools with missing or malformed annotations — is treated as
    write-capable and never suppressed.

    ``read_file`` and ``search_files`` are deliberately excluded even though
    they are reads: they have their own mtime-aware dedup in
    ``tools/file_tools.py``, which can tell "unchanged" from "not re-read"
    and is strictly better than this one. Two layers would fight.

    The name-shape check comes FIRST and matters for cold start: this runs on
    every tool dispatch, and ``tools/mcp_tool.py`` is deliberately excluded
    from built-in tool discovery (``tools/registry.py``) because importing it
    is expensive. Asking the question through an import would have dragged
    that module into every process that ever calls any tool, MCP configured
    or not. A name that cannot be an MCP tool is answered without it.
    """
    if not tool_name or not tool_name.startswith(_MCP_REGISTRY_PREFIX):
        return False
    try:
        from tools.mcp_tool import is_read_only_mcp_tool

        return bool(is_read_only_mcp_tool(tool_name))
    except Exception:
        logger.debug("read-only lookup failed for %s", tool_name, exc_info=True)
        return False
