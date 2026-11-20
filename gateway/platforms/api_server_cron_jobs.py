"""Cron job executions door of the API server: ``GET /api/jobs/{job_id}/executions``.

Split out of ``gateway/platforms/api_server.py`` (Iris seam on the bot host line)."""
import logging

from aiohttp import web

logger = logging.getLogger("gateway.platforms.api_server")


EXECUTIONS_DEFAULT_LIMIT = 20
EXECUTIONS_MAX_LIMIT = 100
_EXECUTION_FIELDS = ("id", "job_id", "status", "started_at", "finished_at", "session_id")


async def handle_list_executions(adapter, request: web.Request) -> web.Response:
    """``GET /api/jobs/{job_id}/executions?limit=N`` — the job's attempts, newest first, each with
    the session its agent turn ran in (``session_id``; null before the run admitted, for no-agent
    jobs, and for rows written before the column existed). History outlives the job: a removed
    job's rows are still listed until the ledger prunes them."""
    job_id, err = adapter._cron_request_guard(request, need_job_id=True)
    if err:
        return err
    raw = request.query.get("limit")
    limit = EXECUTIONS_DEFAULT_LIMIT
    if raw is not None:
        try:
            limit = int(raw)
        except ValueError:
            return web.json_response({"error": "limit must be an integer"}, status=400)
        limit = max(1, min(limit, EXECUTIONS_MAX_LIMIT))
    try:
        import asyncio

        from cron.executions import list_executions
        # Off the event loop: the ledger lock and SQLite busy waits must not stall the gateway.
        # to_thread copies the context, so the /p/<profile>/ home scope still applies.
        rows = await asyncio.to_thread(list_executions, job_id=job_id, limit=limit)
    except Exception as e:  # the HTTP boundary: a redacted 500, the traceback stays in the log
        logger.warning("Cron executions API failed for job %s", job_id, exc_info=True)
        return adapter._cron_error_response(e)
    return web.json_response(
        {"executions": [{key: row.get(key) for key in _EXECUTION_FIELDS} for row in rows]})
