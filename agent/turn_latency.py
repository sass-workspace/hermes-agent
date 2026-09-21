"""Per-turn latency accounting — where did the wall clock actually go?

A slow turn is almost never slow for the reason the user guesses: a tool gets blamed for a 40 s
turn, someone measures the tool at 200 ms, and the real cost — provider round-trips, or the control
flow between them — is never named because nothing measures it. One turn's wall clock is split
into four buckets that sum to it:

    wall = model + tools + approval_wait + other

* ``model`` — time inside provider calls, per ATTEMPT (a failed attempt still cost its time).
* ``tools`` — time executing tools, EXCLUDING approval wait.
* ``approval_wait`` — time blocked on a human answering an approval prompt, read as the turn's
  delta of ``tools.approval_human_wait.human_wait_seconds`` (measured at the source, per session).
  Folded into ``tools`` it makes a fast tool look pathological.
* ``other`` — the remainder: compression, persistence, message assembly, and the pre-dispatch
  gates of every call (request middleware, ``pre_tool_call`` hooks, guardrails): the dispatcher's
  clock starts after them, and a call they block reports ``duration_ms=0`` by contract.

Local diagnosis only: one INFO line per turn in ``agent.log``. No outbound telemetry, no model
tool, no prompt or toolset change. Every entry point is fail-open — latency accounting must never
be the reason a turn breaks.
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

# A crashed turn (or a caller that never finishes) must not leak: oldest record out past the cap.
_MAX_TRACKED_TURNS = 64
# Tools listed individually in the summary line, ranked by time spent.
_TOP_TOOLS_IN_SUMMARY = 4


@dataclass
class _ToolStat:
    calls: int = 0
    ms: float = 0.0


@dataclass
class _TurnRecord:
    turn_id: str
    wait_key: Optional[str] = None
    wait_baseline_s: float = 0.0
    started_mono: float = field(default_factory=time.monotonic)
    api_calls: int = 0
    model_ms: float = 0.0
    tool_calls: int = 0
    tool_ms: float = 0.0
    tools: Dict[str, _ToolStat] = field(default_factory=dict)
    # Tool calls run concurrently on pool workers: every mutation is serialised.
    lock: threading.Lock = field(default_factory=threading.Lock)


_records: "OrderedDict[str, _TurnRecord]" = OrderedDict()
_registry_lock = threading.Lock()


def _human_wait_seconds(wait_key: Optional[str]) -> float:
    try:
        from tools.approval_human_wait import human_wait_seconds
        return float(human_wait_seconds(wait_key))
    except Exception:
        return 0.0


def _get(turn_id: str) -> Optional[_TurnRecord]:
    if not turn_id:
        return None
    with _registry_lock:
        return _records.get(turn_id)


def start_turn(turn_id: str) -> None:
    """Begin accounting for ``turn_id``. Idempotent: a second start for a live turn keeps the
    existing ledger — silently discarding accumulated buckets would make the summary lie about
    exactly the turn someone is investigating. The human-wait session key is snapshotted HERE, on
    the thread that runs the turn; the summary may be logged from another context."""
    if not turn_id:
        return
    try:
        with _registry_lock:
            if turn_id in _records:
                _records.move_to_end(turn_id)
                return
        try:
            from tools.approval_context import get_current_session_key
            wait_key: Optional[str] = get_current_session_key()
        except Exception:
            wait_key = None
        record = _TurnRecord(turn_id=turn_id, wait_key=wait_key, wait_baseline_s=_human_wait_seconds(wait_key))
        with _registry_lock:
            _records.setdefault(turn_id, record)
            _records.move_to_end(turn_id)
            while len(_records) > _MAX_TRACKED_TURNS:
                _records.popitem(last=False)
    except Exception:
        logger.debug("turn latency: start_turn failed", exc_info=True)


def record_model_call(turn_id: str, seconds: float) -> None:
    """Record one provider attempt's duration."""
    record = _get(turn_id)
    if record is None:
        return
    try:
        with record.lock:
            record.api_calls += 1
            record.model_ms += max(0.0, float(seconds)) * 1000.0
    except Exception:
        logger.debug("turn latency: record_model_call failed", exc_info=True)


@contextlib.contextmanager
def model_call_timer(turn_id: str):
    """Charge the enclosed provider attempt to the turn's ``model`` bucket — in a ``finally``, so
    an attempt that raises still costs its time (the slow turn that crashed is the one worth
    accounting for)."""
    started = time.monotonic()
    try:
        yield
    finally:
        record_model_call(turn_id, time.monotonic() - started)


def record_tool_call(turn_id: str, tool_name: str, duration_ms: float) -> None:
    """Record one finished tool call (its full duration; the turn's approval wait is split out
    of the ``tools`` bucket in the summary)."""
    record = _get(turn_id)
    if record is None:
        return
    try:
        spent = max(0.0, float(duration_ms or 0.0))
        with record.lock:
            record.tool_calls += 1
            record.tool_ms += spent
            stat = record.tools.setdefault(tool_name or "?", _ToolStat())
            stat.calls += 1
            stat.ms += spent
    except Exception:
        logger.debug("turn latency: record_tool_call failed", exc_info=True)


def _summarize_record(record: _TurnRecord) -> Dict[str, Any]:
    approval_ms = max(0.0, _human_wait_seconds(record.wait_key) - record.wait_baseline_s) * 1000.0
    with record.lock:
        wall_ms = max(0.0, (time.monotonic() - record.started_mono) * 1000.0)
        model_ms, api_calls, tool_calls = record.model_ms, record.api_calls, record.tool_calls
        raw_tool_ms = record.tool_ms
        tools = [{"name": name, "calls": stat.calls, "ms": round(stat.ms, 1)} for name, stat in record.tools.items()]
    tools.sort(key=lambda t: t["ms"], reverse=True)
    # A human answers INSIDE a tool call, so the wait is part of the raw tool time. A wait can
    # never exceed the turn; concurrent tools can push accounted time past the wall clock, so
    # the remainder is floored at zero rather than reported negative.
    approval_ms = min(approval_ms, wall_ms)
    tool_ms = max(0.0, raw_tool_ms - approval_ms)
    other_ms = max(0.0, wall_ms - model_ms - tool_ms - approval_ms)
    return {
        "turn_id": record.turn_id, "wall_ms": round(wall_ms, 1), "model_ms": round(model_ms, 1),
        "api_calls": api_calls, "tool_ms": round(tool_ms, 1), "tool_calls": tool_calls,
        "approval_wait_ms": round(approval_ms, 1), "other_ms": round(other_ms, 1), "tools": tools,
    }


def summarize(turn_id: str) -> Optional[Dict[str, Any]]:
    """The current buckets for ``turn_id`` without unregistering it."""
    record = _get(turn_id)
    return None if record is None else _summarize_record(record)


def format_summary(summary: Dict[str, Any]) -> str:
    """Render one summary as a single log line."""
    def _s(key: str) -> str:
        return f"{summary.get(key, 0.0) / 1000.0:.1f}s"

    line = (f"turn latency turn={summary.get('turn_id', '')} wall={_s('wall_ms')} "
            f"model={_s('model_ms')}({summary.get('api_calls', 0)} calls) "
            f"tools={_s('tool_ms')}({summary.get('tool_calls', 0)} calls) "
            f"approval_wait={_s('approval_wait_ms')} other={_s('other_ms')}")
    top = (summary.get("tools") or [])[:_TOP_TOOLS_IN_SUMMARY]
    if top:
        line += " | top: " + ", ".join(f"{t['name']} {t['ms'] / 1000.0:.1f}s x{t['calls']}" for t in top)
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
    with _registry_lock:
        _records.clear()
