"""Suppress identical read-only tool calls within one turn.

A turn that calls ``mcp__asana__get_task(gid="123")`` four times gets the
same payload four times. Each copy is permanently added to the conversation
— so it inflates context for the rest of the session, and inflates every
subsequent request's latency along with it. The model is not learning
anything on calls two through four; it already has the answer, two messages
up.

``read_file`` has solved this for files since forever, with an mtime-keyed
dedup that returns a short stub instead of the content. This is the same
idea for the tools that had no equivalent — most importantly read-only MCP
tools, where the payload is largest and the round-trip is remote.

Three rules keep it from ever returning something stale:

1. **Read-only only.** A tool qualifies when its MCP discovery annotations
   carry ``readOnlyHint: true`` exactly — the same fail-closed rule the trust
   gate uses, so missing or malformed annotations mean "write-capable" and
   are never suppressed. (``readOnlyHint`` is supplied by the server and a
   hostile server can lie; the worst a lie buys here is one stale read
   inside a single turn, and only of the server's own data.)
2. **Any write invalidates everything.** The moment a non-read-only tool
   runs, the whole turn's cache is dropped. Otherwise
   ``get_task`` → ``update_task`` → ``get_task`` would answer the third call
   with the pre-update state. This is deliberately blunt: we do not try to
   reason about which reads a given write could have affected.
3. **Errors are never suppressed.** A failed call must stay retryable.

Scoped to the turn, so a later turn always re-reads: the world moves between
turns, and the model's context has been compacted or extended in ways this
module has no view of.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Bound the registry; turns normally end by being evicted here, but a crashed
# turn must not leak.
_MAX_TRACKED_TURNS = 32
# Distinct read calls remembered per turn.
_MAX_KEYS_PER_TURN = 256
# After this many suppressed repeats of one call, stop being polite about it.
# Mirrors read_file's escalation: a weak tool-follower that ignores the stub
# would otherwise burn its whole iteration budget re-asking.
_HARD_BLOCK_AFTER = 2

_lock = threading.Lock()
# turn_id -> {call_key: hits}
_turn_reads: "OrderedDict[str, OrderedDict[str, int]]" = OrderedDict()

SUPPRESSED_MESSAGE = (
    "Already called in this turn with identical arguments, and nothing has "
    "written since. The result from that earlier call in this conversation "
    "is still current — refer to it instead of calling again."
)


def _call_key(tool_name: str, args: Any) -> Optional[str]:
    """Stable key for (tool, arguments), or None if the args won't serialize.

    ``sort_keys`` so argument order cannot make two identical calls look
    different, and a hash so a large argument payload does not get held for
    the life of the turn.
    """
    try:
        encoded = json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None
    digest = hashlib.sha256(encoded.encode("utf-8", "replace")).hexdigest()[:32]
    return f"{tool_name}:{digest}"


def _turn_bucket(turn_id: str, create: bool) -> Optional["OrderedDict[str, int]"]:
    """Return this turn's key->hits map. Caller must hold ``_lock``."""
    bucket = _turn_reads.get(turn_id)
    if bucket is None:
        if not create:
            return None
        bucket = _turn_reads[turn_id] = OrderedDict()
        while len(_turn_reads) > _MAX_TRACKED_TURNS:
            _turn_reads.popitem(last=False)
    _turn_reads.move_to_end(turn_id)
    return bucket


def check(turn_id: str, tool_name: str, args: Any) -> Optional[str]:
    """Return a replacement result when this exact call already ran.

    None means "go ahead and run it". A returned string is the tool result
    the model should see instead of a second identical payload.
    """
    if not turn_id:
        return None
    key = _call_key(tool_name, args)
    if key is None:
        return None
    with _lock:
        bucket = _turn_bucket(turn_id, create=False)
        if bucket is None or key not in bucket:
            return None
        hits = bucket[key] + 1
        bucket[key] = hits
        bucket.move_to_end(key)

    if hits > _HARD_BLOCK_AFTER:
        from tools.registry import tool_error

        return tool_error(
            f"BLOCKED: you have called '{tool_name}' with these exact "
            f"arguments {hits + 1} times in this turn and nothing has "
            f"changed in between. The result is in your context already. "
            f"STOP repeating this call and continue with the task.",
            repeated_calls=hits + 1,
            suppressed=True,
        )

    return json.dumps(
        {
            "status": "unchanged",
            "message": SUPPRESSED_MESSAGE,
            "tool": tool_name,
            "suppressed": True,
            "content_returned": False,
        },
        ensure_ascii=False,
    )


def record(turn_id: str, tool_name: str, args: Any) -> None:
    """Remember that this read call ran and produced a usable result."""
    if not turn_id:
        return
    key = _call_key(tool_name, args)
    if key is None:
        return
    with _lock:
        bucket = _turn_bucket(turn_id, create=True)
        if bucket is None:  # pragma: no cover — create=True always returns one
            return
        bucket.setdefault(key, 0)
        bucket.move_to_end(key)
        while len(bucket) > _MAX_KEYS_PER_TURN:
            bucket.popitem(last=False)


def invalidate(turn_id: str) -> None:
    """Drop everything remembered for this turn.

    Called when a write-capable tool runs: any cached read may now be stale,
    and we do not try to guess which.
    """
    if not turn_id:
        return
    with _lock:
        _turn_reads.pop(turn_id, None)


def finish_turn(turn_id: str) -> None:
    """Forget a turn once it ends."""
    invalidate(turn_id)


def reset_for_tests() -> None:
    """Drop every tracked turn. Test-support only."""
    with _lock:
        _turn_reads.clear()


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
    """
    try:
        from tools.mcp_tool import is_read_only_mcp_tool

        return bool(is_read_only_mcp_tool(tool_name))
    except Exception:
        logger.debug("read-only lookup failed for %s", tool_name, exc_info=True)
        return False


def stats() -> Dict[str, Tuple[int, int]]:
    """Diagnostics: {turn_id: (distinct_calls, suppressed_repeats)}."""
    with _lock:
        return {
            turn: (len(bucket), sum(bucket.values()))
            for turn, bucket in _turn_reads.items()
        }
