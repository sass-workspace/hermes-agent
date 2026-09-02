"""Per-turn latency accounting — where did the wall clock actually go?

A slow turn is almost never slow for the reason the user guesses. The
recurring pattern: a tool gets blamed for a turn that took 40 seconds, an
investigation measures the tool, finds it takes 200ms, and the real cost —
provider round-trips, or the control flow between them — is never named
because nothing measures it.

This module closes that gap by splitting one turn's wall clock into four
buckets that sum to it:

    wall = model + tool + approval_wait + other

* ``model``  — time inside provider calls (``api_duration`` per API call).
* ``tool``   — time executing tools, **excluding** any approval wait.
* ``approval_wait`` — time blocked on a human answering an approval or
  elicitation prompt. This is the split that makes the rest honest: it is
  not latency the agent can do anything about, and folded into ``tool`` it
  makes a fast tool look pathological.
* ``other``  — the remainder: compression, persistence, hooks, message
  assembly, and everything else between the calls.

Nothing here leaves the machine. It is one INFO line per turn in
``agent.log`` plus an additive ``approval_wait_ms`` field on the existing
``post_tool_call`` hook payload — no outbound telemetry, no new model tool,
no prompt or toolset change, so per-conversation prompt caching is
untouched.

Every entry point is fail-open: latency accounting must never be the reason
a turn breaks.
"""
from __future__ import annotations

import contextlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Bound the registry. Turns normally unregister themselves in finish_turn(),
# but a crashed turn (or a caller that never finishes) must not leak: the
# oldest record is evicted once the cap is crossed.
_MAX_TRACKED_TURNS = 64

# Tools listed individually in the summary line, ranked by time spent.
_TOP_TOOLS_IN_SUMMARY = 4


@dataclass
class _ToolStat:
    calls: int = 0
    ms: float = 0.0
    approval_ms: float = 0.0


@dataclass
class _TurnRecord:
    turn_id: str
    session_id: str = ""
    platform: str = ""
    started_mono: float = field(default_factory=time.monotonic)
    api_calls: int = 0
    model_ms: float = 0.0
    tool_calls: int = 0
    tool_ms: float = 0.0
    approval_wait_ms: float = 0.0
    tools: Dict[str, _ToolStat] = field(default_factory=dict)
    # Tool calls can run concurrently (_execute_tool_calls_concurrent), so
    # every mutation of this record is serialised.
    lock: threading.Lock = field(default_factory=threading.Lock)


_records: "OrderedDict[str, _TurnRecord]" = OrderedDict()
_registry_lock = threading.Lock()

# Approval wait is accumulated as a MONOTONIC PER-THREAD COUNTER, and callers
# read it as a delta across the region they care about:
#
#     before = approval_wait_total()
#     ...run the tool, which may prompt a human...
#     waited = approval_wait_total() - before
#
# A scoped context manager would be the obvious shape, and was the first
# one tried — but the tool dispatcher has early returns between the first
# blocking surface (ACP edit approval) and the dispatch, and any scope
# spanning them leaks its state on those paths. A counter has no scope to
# leak: it cannot be left open, cannot be reset out of order, and a delta is
# correct no matter which path the caller returns through.
#
# Thread-local because tool calls run concurrently on pool workers; a
# nested dispatch on one thread rolls up naturally, since the outer delta
# spans the inner one — which is what we want, the outer call really did sit
# through the inner wait.
_approval_thread_state = threading.local()


def _get(turn_id: str) -> Optional[_TurnRecord]:
    if not turn_id:
        return None
    with _registry_lock:
        return _records.get(turn_id)


def start_turn(turn_id: str, session_id: str = "", platform: str = "") -> None:
    """Begin accounting for ``turn_id``.

    Idempotent: a second start for a live turn id keeps the existing record
    rather than resetting it. Silently discarding a turn's accumulated
    buckets would make the summary lie about exactly the turn someone is
    investigating.
    """
    if not turn_id:
        return
    try:
        with _registry_lock:
            if turn_id in _records:
                logger.debug(
                    "turn latency: turn %s already being tracked — keeping "
                    "the existing ledger", turn_id,
                )
                _records.move_to_end(turn_id)
                return
            _records[turn_id] = _TurnRecord(
                turn_id=turn_id,
                session_id=session_id or "",
                platform=platform or "",
            )
            _records.move_to_end(turn_id)
            while len(_records) > _MAX_TRACKED_TURNS:
                _records.popitem(last=False)
    except Exception:
        logger.debug("turn latency: start_turn failed", exc_info=True)


def record_model_call(turn_id: str, seconds: float) -> None:
    """Record one provider round-trip's duration."""
    record = _get(turn_id)
    if record is None:
        return
    try:
        with record.lock:
            record.api_calls += 1
            record.model_ms += max(0.0, float(seconds)) * 1000.0
    except Exception:
        logger.debug("turn latency: record_model_call failed", exc_info=True)


def record_tool_call(
    turn_id: str,
    tool_name: str,
    duration_ms: float,
    approval_wait_ms: float = 0.0,
) -> None:
    """Record one tool dispatch, splitting out its approval wait.

    ``duration_ms`` is the full dispatch duration as the tool dispatcher
    measures it; ``approval_wait_ms`` is however much of that was spent
    blocked on a human. Only the difference is charged to the tool.
    """
    record = _get(turn_id)
    if record is None:
        return
    try:
        total = max(0.0, float(duration_ms))
        waited = min(max(0.0, float(approval_wait_ms)), total)
        executed = total - waited
        with record.lock:
            record.tool_calls += 1
            record.tool_ms += executed
            record.approval_wait_ms += waited
            stat = record.tools.get(tool_name)
            if stat is None:
                stat = record.tools[tool_name] = _ToolStat()
            stat.calls += 1
            stat.ms += executed
            stat.approval_ms += waited
    except Exception:
        logger.debug("turn latency: record_tool_call failed", exc_info=True)


# ---------------------------------------------------------------------------
# Approval-wait split
# ---------------------------------------------------------------------------


def approval_wait_total() -> float:
    """Seconds this thread has spent blocked on humans, since it started.

    Only differences between two readings are meaningful.
    """
    return float(getattr(_approval_thread_state, "total", 0.0))


def record_approval_wait(seconds: float) -> None:
    """Add ``seconds`` of human-blocked time to this thread's counter."""
    try:
        _approval_thread_state.total = approval_wait_total() + max(
            0.0, float(seconds)
        )
    except Exception:
        logger.debug("turn latency: record_approval_wait failed", exc_info=True)


@contextlib.contextmanager
def approval_wait_timer():
    """Time a blocking approval prompt and add it to this thread's counter."""
    started = time.monotonic()
    try:
        yield
    finally:
        record_approval_wait(time.monotonic() - started)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def summarize(turn_id: str) -> Optional[Dict[str, Any]]:
    """Return the current buckets for ``turn_id`` without unregistering it."""
    record = _get(turn_id)
    if record is None:
        return None
    return _summarize_record(record)


def _summarize_record(record: _TurnRecord) -> Dict[str, Any]:
    with record.lock:
        wall_ms = max(0.0, (time.monotonic() - record.started_mono) * 1000.0)
        model_ms = record.model_ms
        tool_ms = record.tool_ms
        approval_ms = record.approval_wait_ms
        tools = [
            {
                "name": name,
                "calls": stat.calls,
                "ms": round(stat.ms, 1),
                "approval_ms": round(stat.approval_ms, 1),
            }
            for name, stat in record.tools.items()
        ]
        api_calls = record.api_calls
        tool_calls = record.tool_calls
        session_id = record.session_id
        platform = record.platform
    tools.sort(key=lambda t: t["ms"], reverse=True)
    # Accounted time can exceed the wall clock when tools run concurrently —
    # report the remainder as zero rather than a negative "other".
    other_ms = max(0.0, wall_ms - model_ms - tool_ms - approval_ms)
    return {
        "turn_id": record.turn_id,
        "session_id": session_id,
        "platform": platform,
        "wall_ms": round(wall_ms, 1),
        "model_ms": round(model_ms, 1),
        "api_calls": api_calls,
        "tool_ms": round(tool_ms, 1),
        "tool_calls": tool_calls,
        "approval_wait_ms": round(approval_ms, 1),
        "other_ms": round(other_ms, 1),
        "tools": tools,
    }


def format_summary(summary: Dict[str, Any]) -> str:
    """Render one summary as a single log line."""
    def _s(key: str) -> str:
        return f"{summary.get(key, 0.0) / 1000.0:.1f}s"

    line = (
        f"turn latency turn={summary.get('turn_id', '')} "
        f"wall={_s('wall_ms')} "
        f"model={_s('model_ms')}({summary.get('api_calls', 0)} calls) "
        f"tools={_s('tool_ms')}({summary.get('tool_calls', 0)} calls) "
        f"approval_wait={_s('approval_wait_ms')} "
        f"other={_s('other_ms')}"
    )
    top = summary.get("tools") or []
    if top:
        parts = [
            f"{t['name']} {t['ms'] / 1000.0:.1f}s x{t['calls']}"
            + (
                f" (+{t['approval_ms'] / 1000.0:.1f}s approval)"
                if t.get("approval_ms")
                else ""
            )
            for t in top[:_TOP_TOOLS_IN_SUMMARY]
        ]
        line += " | top: " + ", ".join(parts)
    return line


def finish_turn(turn_id: str, *, log: bool = True) -> Optional[Dict[str, Any]]:
    """Unregister ``turn_id`` and return (and optionally log) its summary."""
    if not turn_id:
        return None
    try:
        with _registry_lock:
            record = _records.pop(turn_id, None)
        if record is None:
            return None
        summary = _summarize_record(record)
        if log:
            logger.info("%s", format_summary(summary))
        return summary
    except Exception:
        logger.debug("turn latency: finish_turn failed", exc_info=True)
        return None


def reset_for_tests() -> None:
    """Drop all tracked turns and this thread's approval counter. Tests only."""
    with _registry_lock:
        _records.clear()
    _approval_thread_state.total = 0.0
