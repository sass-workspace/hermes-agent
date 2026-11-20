"""Durable cron execution-ledger behavior."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path


def _point_ledger(monkeypatch, tmp_path):
    from cron import executions

    monkeypatch.setattr(executions, "EXECUTIONS_FILE", tmp_path / "cron" / "executions.db")
    return executions


def test_execution_transitions_are_durable(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    claimed = executions.create_execution("job-1", source="builtin")
    assert claimed["status"] == "claimed"
    assert claimed["claimed_at"]
    assert claimed["started_at"] is None
    assert claimed["finished_at"] is None

    running = executions.mark_execution_running(claimed["id"])
    assert running["status"] == "running"
    assert running["started_at"]

    completed = executions.finish_execution(claimed["id"], success=True)
    assert completed["status"] == "completed"
    assert completed["finished_at"]
    assert completed["error"] is None

    persisted = executions.list_executions(job_id="job-1")
    assert persisted == [completed]


def test_execution_can_be_loaded_by_exact_attempt_id(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    first = executions.create_execution("same-job", source="builtin")
    second = executions.create_execution("same-job", source="builtin")

    assert executions.get_execution(first["id"]) == first
    assert executions.get_execution(second["id"]) == second
    assert executions.get_execution("missing") is None


def test_fresh_external_handoff_is_not_recovered_before_worker_adopts(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("handoff-job", source="builtin")
    assert executions.mark_execution_handoff_pending(record["id"]) is not None

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)

    assert executions.recover_interrupted_executions() == 0
    assert executions.get_execution(record["id"])["status"] == "claimed"
    adopted = executions.adopt_claimed_execution(record["id"])
    assert adopted["status"] == "running"
    assert adopted["handoff_pending"] == 0


def test_stale_external_handoff_is_recovered_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("handoff-job", source="builtin")
    pending = executions.mark_execution_handoff_pending(record["id"])

    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-gateway")
    monkeypatch.setattr(executions, "_owner_is_live", lambda _pid, _started: False)
    monkeypatch.setattr(
        executions.time,
        "time",
        lambda: pending["handoff_started_at"]
        + executions.HANDOFF_ADOPTION_GRACE_SECONDS
        + 1,
    )

    assert executions.recover_interrupted_executions() == 1
    recovered = executions.get_execution(record["id"])
    assert recovered["status"] == "unknown"
    assert recovered["handoff_pending"] == 0


def test_recovery_does_not_overwrite_concurrent_worker_adoption(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("adoption-race", source="builtin")
    pending = executions.mark_execution_handoff_pending(record["id"])
    assert pending is not None
    monkeypatch.setattr(executions, "_PROCESS_ID", "replacement-scheduler")
    monkeypatch.setattr(
        executions.time,
        "time",
        lambda: pending["handoff_started_at"]
        + executions.HANDOFF_ADOPTION_GRACE_SECONDS
        + 1,
    )

    def adopt_while_liveness_is_checked(_pid, _started_at):
        monkeypatch.setattr(executions, "_PROCESS_ID", "external-worker")
        monkeypatch.setattr(executions.os, "getpid", lambda: 4242)
        monkeypatch.setattr(executions, "_process_start_time", lambda _pid: 9876)
        assert executions.adopt_claimed_execution(record["id"]) is not None
        return False

    monkeypatch.setattr(executions, "_owner_is_live", adopt_while_liveness_is_checked)

    assert executions.recover_interrupted_executions() == 0
    current = executions.get_execution(record["id"])
    assert current is not None
    assert current["status"] == "running"
    assert current["process_id"] == "external-worker"
    assert current["pid"] == 4242


def test_foreign_process_cannot_start_or_finish_execution(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("owner-fence", source="builtin")
    original_process_id = executions._PROCESS_ID
    original_pid = record["pid"]

    monkeypatch.setattr(executions, "_PROCESS_ID", "foreign-process")
    monkeypatch.setattr(executions.os, "getpid", lambda: original_pid + 1)
    assert executions.mark_execution_running(record["id"]) is None
    assert executions.finish_execution(record["id"], success=True) is None

    monkeypatch.setattr(executions, "_PROCESS_ID", original_process_id)
    monkeypatch.setattr(executions.os, "getpid", lambda: original_pid)
    assert executions.mark_execution_running(record["id"]) is not None
    assert executions.finish_execution(record["id"], success=True) is not None


def test_execution_ledger_follows_the_current_profile_home(monkeypatch, tmp_path):
    from cron import executions

    current_home = {"path": tmp_path / "default"}
    monkeypatch.setattr(executions, "EXECUTIONS_FILE", None)
    monkeypatch.setattr(executions, "get_hermes_home", lambda: current_home["path"])

    default_row = executions.create_execution("default-job", source="builtin")
    current_home["path"] = tmp_path / "worker"
    worker_row = executions.create_execution("worker-job", source="builtin")

    assert executions.list_executions() == [worker_row]
    current_home["path"] = tmp_path / "default"
    assert executions.list_executions() == [default_row]
    assert (tmp_path / "default" / "cron" / "executions.db").is_file()
    assert (tmp_path / "worker" / "cron" / "executions.db").is_file()


def test_terminal_execution_cannot_be_rewritten(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("immutable", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)

    assert executions.finish_execution(
        record["id"], success=False, error="late writer"
    ) is None
    assert executions.latest_execution("immutable")["status"] == "completed"


def test_retention_bounds_terminal_history_but_preserves_inflight(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 3)
    inflight = executions.create_execution("live", source="builtin")
    executions.mark_execution_running(inflight["id"])
    for index in range(8):
        row = executions.create_execution(f"done-{index}", source="builtin")
        executions.finish_execution(row["id"], success=True)

    records = executions.list_executions(limit=100)
    assert len([row for row in records if row["status"] == "completed"]) == 3
    assert executions.latest_execution("live")["status"] == "running"


def test_recently_finished_long_running_execution_survives_retention(
    monkeypatch, tmp_path
):
    executions = _point_ledger(monkeypatch, tmp_path)
    monkeypatch.setattr(executions, "MAX_TERMINAL_EXECUTIONS", 1)
    long_running = executions.create_execution("long-running", source="builtin")
    assert executions.mark_execution_running(long_running["id"]) is not None
    newer = executions.create_execution("newer", source="builtin")
    assert executions.finish_execution(newer["id"], success=True) is not None

    finished = executions.finish_execution(long_running["id"], success=True)

    assert finished is not None
    assert finished["status"] == "completed"
    assert executions.get_execution(long_running["id"])["status"] == "completed"
    assert executions.get_execution(newer["id"]) is None


def test_corrupt_store_fails_closed_without_overwrite(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    executions.EXECUTIONS_FILE.parent.mkdir(parents=True)
    executions.EXECUTIONS_FILE.write_bytes(b"not a sqlite database")

    with __import__("pytest").raises(sqlite3.DatabaseError):
        executions.create_execution("new", source="builtin")
    assert executions.EXECUTIONS_FILE.read_bytes() == b"not a sqlite database"


def test_cron_runs_cli_prints_execution_history(monkeypatch, tmp_path, capsys):
    executions = _point_ledger(monkeypatch, tmp_path)
    row = executions.create_execution("cli-job", source="builtin")
    executions.finish_execution(row["id"], success=False, error="boom")
    from hermes_cli.cron import cron_runs

    cron_runs("cli-job", limit=10)

    output = capsys.readouterr().out
    assert row["id"] in output
    assert "failed" in output
    assert "boom" in output


def test_quick_backup_includes_execution_ledger():
    from hermes_cli.backup import _QUICK_STATE_FILES

    assert "cron/executions.db" in _QUICK_STATE_FILES


def test_failed_execution_keeps_error(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    record = executions.create_execution("job-2", source="external")
    failed = executions.finish_execution(record["id"], success=False, error="provider exploded")

    assert failed["status"] == "failed"
    assert failed["error"] == "provider exploded"


def test_recovery_does_not_mark_live_process_execution_unknown(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("still-live", source="builtin")
    executions.mark_execution_running(record["id"])

    assert executions.recover_interrupted_executions() == 0
    assert executions.latest_execution("still-live")["status"] == "running"


def test_restart_marks_interrupted_execution_unknown_without_requeue(tmp_path):
    """Real temp-HERMES_HOME subprocess restart: in-flight is audit-only unknown."""
    home = tmp_path / "home"
    repo = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(repo)

    create = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cron.executions import create_execution, mark_execution_running; "
            "r=create_execution('restart-job', source='builtin'); "
            "mark_execution_running(r['id']); print(r['id'])",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    execution_id = create.stdout.strip()

    recover = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from cron.executions import recover_interrupted_executions, list_executions; "
            "print(recover_interrupted_executions()); "
            "print(json.dumps(list_executions(job_id='restart-job'))) ",
        ],
        cwd=repo,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    lines = recover.stdout.strip().splitlines()
    assert lines[0] == "1"
    records = json.loads(lines[1])
    assert len(records) == 1
    assert records[0]["id"] == execution_id
    assert records[0]["status"] == "unknown"
    assert records[0]["finished_at"]
    assert "restart" in records[0]["error"].lower()
    # Recovery only classifies the old attempt. It must not manufacture a new
    # claimed record (which would imply an automatic retry).
    assert [r["status"] for r in records] == ["unknown"]


def test_generic_submit_failure_finishes_attempt_and_releases_guard(monkeypatch):
    from cron import scheduler

    class BrokenPool:
        def submit(self, _callable):
            raise ValueError("executor rejected")

    finished = []
    monkeypatch.setattr(
        scheduler, "create_execution",
        lambda *_args, **_kwargs: {"id": "exec-submit-fail"},
    )
    monkeypatch.setattr(
        scheduler, "finish_execution",
        lambda execution_id, **kwargs: finished.append((execution_id, kwargs)),
    )
    monkeypatch.setattr(scheduler, "get_due_jobs", lambda: [{"id": "submit-fail"}])
    monkeypatch.setattr(scheduler, "claim_job_for_fire", lambda _job_id: True)
    monkeypatch.setattr(scheduler, "_get_parallel_pool", lambda _workers: BrokenPool())

    assert scheduler.tick(verbose=False, sync=False) == 0
    assert finished == [
        ("exec-submit-fail", {
            "success": False,
            "error": "Executor dispatch failed: executor rejected",
        })
    ]
    assert "submit-fail" not in scheduler.get_running_job_ids()


def test_run_one_job_records_running_then_terminal(monkeypatch):
    from cron import scheduler

    events = []
    run_execution_ids = []
    monkeypatch.setattr(
        scheduler,
        "mark_execution_running",
        lambda execution_id: events.append(("running", execution_id)) or {},
        raising=False,
    )
    monkeypatch.setattr(
        scheduler,
        "finish_execution",
        lambda execution_id, **kwargs: events.append(("finish", execution_id, kwargs)),
        raising=False,
    )
    monkeypatch.setattr(scheduler, "claim_dispatch", lambda _job_id: True)

    def fake_run_job(job, *, defer_agent_teardown=None, execution_id=None, **_kw):
        run_execution_ids.append(execution_id)
        return True, "output", "response", None

    monkeypatch.setattr(scheduler, "run_job", fake_run_job)
    monkeypatch.setattr(scheduler, "save_job_output", lambda *_args: None)
    monkeypatch.setattr(scheduler, "_deliver_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(scheduler, "mark_job_run", lambda *_args, **_kwargs: None)

    assert scheduler.run_one_job({"id": "job-3", "execution_id": "exec-3"}) is True
    assert run_execution_ids == ["exec-3"]
    assert events[0] == ("running", "exec-3")
    assert events[-1][0:2] == ("finish", "exec-3")
    assert events[-1][2]["success"] is True


def test_provider_start_recovers_interrupted_records_before_tick(monkeypatch):
    import cron.scheduler_provider as provider

    events = []
    stop = __import__("threading").Event()
    stop.set()
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
        raising=False,
    )
    monkeypatch.setattr("cron.jobs.record_ticker_heartbeat", lambda **_kwargs: events.append("heartbeat"))

    provider.InProcessCronScheduler().start(stop, interval=1)

    assert events[:2] == ["recover", "heartbeat"]


def test_external_provider_start_recovers_interrupted_records(monkeypatch):
    from plugins.cron_providers.chronos import ChronosCronScheduler

    provider = ChronosCronScheduler()
    provider._client = type("Client", (), {"arm": lambda self, **kwargs: None})()
    events = []
    monkeypatch.setattr(
        "cron.executions.recover_interrupted_executions",
        lambda: events.append("recover") or 0,
    )
    monkeypatch.setattr(provider, "reconcile", lambda: events.append("reconcile"))

    provider.start(__import__("threading").Event())

    assert events == ["recover", "reconcile"]


class _TrackingConnection:
    """Delegates to a real sqlite3.Connection while recording close() calls.

    sqlite3.Connection is a static C type: it has no per-instance __dict__
    and its class methods can't be monkeypatched, so open/close tracking is
    done via a delegating wrapper returned in place of the real connection.
    """

    def __init__(self, real, closed_ids):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed_ids", closed_ids)

    def close(self):
        self._closed_ids.append(id(self._real))
        self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _count_open_connections(executions, monkeypatch):
    """Wrap sqlite3.connect to track open/close balance for the ledger module."""
    opened_ids = []
    closed_ids = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened_ids.append(id(conn))
        return _TrackingConnection(conn, closed_ids)

    monkeypatch.setattr(executions.sqlite3, "connect", tracking_connect)
    return opened_ids, closed_ids


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Regression for #69567: every ledger call must close its connection
    deterministically instead of relying on garbage collection."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    record = executions.create_execution("leak-check", source="builtin")
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)
    executions.list_executions(job_id="leak-check")
    executions.latest_executions(["leak-check"])
    executions.recover_interrupted_executions()

    assert opened
    assert sorted(opened) == sorted(closed)


def test_early_return_still_closes_connection(monkeypatch, tmp_path):
    """mark_execution_running returns None mid-block on a bad transition;
    the connection must still be closed rather than leaked."""
    executions = _point_ledger(monkeypatch, tmp_path)
    opened, closed = _count_open_connections(executions, monkeypatch)

    assert executions.mark_execution_running("does-not-exist") is None

    assert len(opened) == 1
    assert len(closed) == 1






def test_job_listing_exposes_latest_execution(monkeypatch, tmp_path):
    from cron import jobs

    monkeypatch.setattr(jobs, "CRON_DIR", tmp_path / "cron")
    monkeypatch.setattr(jobs, "JOBS_FILE", tmp_path / "cron" / "jobs.json")
    monkeypatch.setattr(jobs, "OUTPUT_DIR", tmp_path / "cron" / "output")
    executions = _point_ledger(monkeypatch, tmp_path)

    job = jobs.create_job(prompt="audit me", schedule="every 1h", name="audit")
    record = executions.create_execution(job["id"], source="builtin")
    executions.mark_execution_running(record["id"])

    listed = jobs.list_jobs(include_disabled=True)
    assert listed[0]["latest_execution"]["id"] == record["id"]
    assert listed[0]["latest_execution"]["status"] == "running"


def test_history_orders_by_instant_across_dst_fall_back(monkeypatch, tmp_path):
    """01:10-05:00 is 20 minutes after 01:50-04:00 but sorts first as text."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    executions = _point_ledger(monkeypatch, tmp_path)
    new_york = ZoneInfo("America/New_York")
    first_pass = datetime(2026, 11, 1, 1, 50, tzinfo=new_york, fold=0)
    second_pass = datetime(2026, 11, 1, 1, 10, tzinfo=new_york, fold=1)
    monkeypatch.setattr(executions, "_hermes_now", lambda: first_pass)
    earlier = executions.create_execution("dst-job", source="builtin")
    monkeypatch.setattr(executions, "_hermes_now", lambda: second_pass)
    later = executions.create_execution("dst-job", source="builtin")
    assert later["claimed_at"] < earlier["claimed_at"]

    assert executions.latest_execution("dst-job")["id"] == later["id"]
    assert executions.latest_executions(["dst-job"])["dst-job"]["id"] == later["id"]
    assert [r["id"] for r in executions.list_executions(job_id="dst-job")] == [
        later["id"], earlier["id"],
    ]
    page = executions.list_executions(job_id="dst-job", before_claimed_at=later["claimed_at"])
    assert [r["id"] for r in page] == [earlier["id"]]


# --- session recorded per attempt (Iris: routine results by execution, not by title) -----------

def test_execution_records_its_session(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("job-s", source="cron")
    assert record["session_id"] is None

    assert executions.set_execution_session(record["id"], "cron-sess-1") is True
    assert executions.get_execution(record["id"])["session_id"] == "cron-sess-1"
    # Same value again is still a success (idempotent), and a missing row is not.
    assert executions.set_execution_session(record["id"], "cron-sess-1") is True
    assert executions.set_execution_session("missing", "x") is False
    assert executions.set_execution_session(record["id"], "") is False

    # Metadata only: a terminal row still moves to its compression tip.
    executions.mark_execution_running(record["id"])
    executions.finish_execution(record["id"], success=True)
    assert executions.set_execution_session(record["id"], "cron-sess-tip") is True
    [listed] = executions.list_executions(job_id="job-s")
    assert listed["status"] == "completed"
    assert listed["session_id"] == "cron-sess-tip"


def test_record_execution_session_never_raises(monkeypatch, tmp_path):
    executions = _point_ledger(monkeypatch, tmp_path)

    def boom(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(executions, "set_execution_session", boom)
    executions.record_execution_session("exec", "sess")  # no exception
    executions.record_execution_session(None, "sess")


def test_ledger_without_session_column_gains_it(monkeypatch, tmp_path):
    """A ledger written by an older build opens with the new column, rows intact."""
    path = tmp_path / "cron" / "executions.db"
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """CREATE TABLE executions (
             id TEXT PRIMARY KEY, job_id TEXT NOT NULL, source TEXT NOT NULL,
             process_id TEXT NOT NULL, pid INTEGER NOT NULL, process_started_at INTEGER,
             status TEXT NOT NULL CHECK(status IN
               ('claimed','running','completed','failed','unknown','cancelled','missed')),
             handoff_pending INTEGER NOT NULL DEFAULT 0, handoff_started_at REAL,
             claimed_at TEXT NOT NULL, started_at TEXT, finished_at TEXT, error TEXT,
             delivery_outcome TEXT, scheduled_instant TEXT, progress_at TEXT)"""
    )
    conn.execute(
        "INSERT INTO executions (id, job_id, source, process_id, pid, status, claimed_at) "
        "VALUES ('old', 'job-o', 'cron', 'p', 1, 'completed', '2026-11-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()
    executions = _point_ledger(monkeypatch, tmp_path)

    old = executions.get_execution("old")
    assert old["session_id"] is None
    assert executions.set_execution_session("old", "sess-old") is True
    assert executions.get_execution("old")["session_id"] == "sess-old"


def test_finalize_moves_the_row_to_the_compression_tip(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from cron import scheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("job-c", source="cron")
    executions.set_execution_session(record["id"], "cron-root")
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda sid: "cron-root-tip"
    fake_db.session_lifecycle_statuses.return_value = {}
    monkeypatch.setattr(scheduler, "_set_cron_session_title", lambda *_a, **_k: True)

    scheduler._finalize_cron_session(fake_db, None, "job-c", "Job C", "cron-root",
                                     execution_id=record["id"])

    assert executions.get_execution(record["id"])["session_id"] == "cron-root-tip"
    fake_db.end_session.assert_called_once_with("cron-root-tip", "cron_complete")


def test_finalize_without_compression_keeps_the_session(monkeypatch, tmp_path):
    from unittest.mock import MagicMock

    from cron import scheduler

    executions = _point_ledger(monkeypatch, tmp_path)
    record = executions.create_execution("job-n", source="cron")
    executions.set_execution_session(record["id"], "cron-root")
    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda sid: sid
    fake_db.session_lifecycle_statuses.return_value = {}
    monkeypatch.setattr(scheduler, "_set_cron_session_title", lambda *_a, **_k: True)

    scheduler._finalize_cron_session(fake_db, None, "job-n", "Job N", "cron-root",
                                     execution_id=record["id"])

    assert executions.get_execution(record["id"])["session_id"] == "cron-root"


def test_canonical_fire_records_the_owner_session_at_the_receipt(tmp_path, monkeypatch):
    """The firer journals the owner's receipt ``{session_id, admission_id}``; the ledger row carries
    that session id, the ``cron:<job>:<execution>`` chat the run's transcript lives in."""
    import asyncio
    import threading
    from types import SimpleNamespace

    from cron import executions as executions_mod
    from cron import jobs, scheduler
    from cron.executions import create_execution, get_execution, mark_execution_running
    from cron.scheduler_authority import run_canonical_job
    from gateway import run, session_cron
    from gateway.config import GatewayConfig, Platform
    from gateway.session import SessionStore
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionRef
    from gateway.session_local_recovery import local_adapter_map
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch, claim_session_input, settle_session_input

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(executions_mod, 'EXECUTIONS_FILE', tmp_path / 'cron' / 'executions.db')
    monkeypatch.setattr(jobs, 'get_job', lambda jid: {'id': jid, 'prompt': 'p', 'model': 'm'})
    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'model': {}, 'platform_toolsets': {'cli': []}})
    execution = create_execution('job', source='cron')
    mark_execution_running(execution['id'])
    db = SessionDB(tmp_path / 'state.db')
    runner = SimpleNamespace(_draining=False, session_store=SessionStore(tmp_path / 'sessions', GatewayConfig()),
                             adapters={})
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='test', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='test'))
    runner.session_authority = authority
    admitted = threading.Event()
    monkeypatch.setattr(authority, '_schedule', lambda ref: admitted.set())
    monkeypatch.setattr(scheduler, 'run_job',
                        lambda job, *, extra_prompt, execution_id, cancel_event: (True, 'doc', 'answer', None))

    async def probe():
        session_cron.bind_owner(authority)
        try:
            firer = asyncio.create_task(asyncio.to_thread(run_canonical_job, {'id': 'job'},
                                                          execution_id=execution['id']))
            assert await asyncio.to_thread(admitted.wait, 10), firer
            sid = next(iter(authority.sessions))
            row = claim_session_input(db, epoch=authority.epoch, session_id=sid)
            policy = local_adapter_map(authority)[Platform.LOCAL].policies[sid]
            await session_cron.execute(authority, SessionRef(authority.profile_id, sid), row, policy)
            settle_session_input(db, epoch=authority.epoch, admission_id=row['admission_id'],
                                 generation=row['generation'], outcome='completed',
                                 result={'result': {'cron_result': [True, 'doc', 'answer', None]}, 'usage': {}})
            await asyncio.wait_for(firer, 10)
            return sid
        finally:
            session_cron.unbind_owner(authority)

    try:
        sid = asyncio.run(probe())
    finally:
        db.close()
    assert sid
    assert get_execution(execution['id'])['session_id'] == sid


def test_run_job_records_the_session_it_runs_in(tmp_path, monkeypatch):
    """The owner-side ``run_job`` books its session id on the execution row before the agent
    starts (it also covers runs whose firer never saw a receipt)."""
    from unittest.mock import MagicMock, patch

    from cron import executions as executions_mod
    from cron.executions import _transaction, create_execution, get_execution
    from tests.cron.test_scheduler import _run_owned_job

    monkeypatch.setattr(executions_mod, 'EXECUTIONS_FILE', tmp_path / 'cron' / 'executions.db')
    execution = create_execution('test-job', source='cron')
    # The owner bridge admits under request id ``cron-owner-request``: that IS the row id.
    with _transaction() as conn:
        conn.execute("UPDATE executions SET id='cron-owner-request' WHERE id=?", (execution['id'],))
    seen = {}

    def construct(*_a, session_id, **_k):
        seen['at_construct'] = get_execution('cron-owner-request')['session_id']
        agent = MagicMock()
        agent.run_conversation.return_value = {"final_response": "ok"}
        return agent

    fake_db = MagicMock()
    fake_db.get_compression_tip.side_effect = lambda sid: sid
    with patch("cron.scheduler._hermes_home", tmp_path), \
         patch("cron.scheduler_delivery._resolve_origin", return_value=None), \
         patch("hermes_cli.env_loader.load_hermes_dotenv"), \
         patch("hermes_cli.env_loader.reset_secret_source_cache"), \
         patch("hermes_state_registry.acquire", return_value=fake_db), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value={
             "api_key": "k", "base_url": "https://example.invalid/v1",
             "provider": "openrouter", "api_mode": "chat_completions"}), \
         patch("run_agent.AIAgent", side_effect=construct):
        ok, _doc, final, _err = _run_owned_job({"id": "test-job", "name": "t", "prompt": "hello"},
                                               tmp_path, fake_db)
    assert ok and final == "ok"
    assert seen['at_construct'] == 'canonical-cron-session'
    assert get_execution('cron-owner-request')['session_id'] == 'canonical-cron-session'
