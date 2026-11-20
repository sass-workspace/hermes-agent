"""Cron job edit door of the API server: ``PATCH /api/jobs/{job_id}`` and the shared job response.

Split out of ``gateway/platforms/api_server.py``. Store callables and helpers are read from the
facade at call time, so ``patch("gateway.platforms.api_server._cron_update")`` keeps intercepting.
"""
import logging

from aiohttp import web

logger = logging.getLogger("gateway.platforms.api_server")


def job_response(adapter, fn, job_id: str, *, notify: bool, invalid_input=()) -> web.Response:
    """Run ``fn(job_id)``: 404 when falsy, else ``{"job": ...}``; an ``invalid_input`` exception
    -> 400 (the client's value was rejected), any other -> 500."""
    from gateway.platforms import api_server as facade
    try:
        job = fn(job_id)
        if not job:
            return web.json_response({"error": "Job not found"}, status=404)
        if notify:
            facade._notify_cron_provider_jobs_changed()
        return web.json_response({"job": job})
    except invalid_input as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:  # the HTTP boundary: a redacted 500, the traceback stays in the log
        logger.warning("Cron jobs API store call failed for job %s", job_id, exc_info=True)
        return adapter._cron_error_response(e)


def _invalid_update(adapter, sanitized: dict):
    """The 400 for a PATCH value the handler can judge itself, else None."""
    if not sanitized:
        return web.json_response({"error": "No valid fields to update"}, status=400)
    for key in ("name", "prompt", "schedule"):
        if key in sanitized and not isinstance(sanitized[key], str):
            return web.json_response({"error": f"{key.capitalize()} must be a string"}, status=400)
    if "name" in sanitized and len(sanitized["name"]) > adapter._MAX_NAME_LENGTH:
        return web.json_response({"error": f"Name must be ≤ {adapter._MAX_NAME_LENGTH} characters"}, status=400)
    return adapter._validate_cron_prompt(sanitized["prompt"]) if "prompt" in sanitized else None


async def handle_update_job(adapter, request: web.Request) -> web.Response:
    """PATCH /api/jobs/{job_id}: a rejected value is a 400 ``{"error": ...}`` as on POST, never a 500."""
    from gateway.platforms import api_server as facade
    job_id, err = adapter._cron_request_guard(request, need_job_id=True)
    if err:
        return err
    try:
        body = await request.json()
    except ValueError as e:  # malformed JSON, as on POST
        return web.json_response({"error": str(e)}, status=400)
    if not isinstance(body, dict):
        return web.json_response({"error": "Request body must be a JSON object"}, status=400)
    # Whitelist allowed fields to prevent arbitrary key injection
    sanitized = {k: v for k, v in body.items() if k in adapter._UPDATE_ALLOWED_FIELDS}
    invalid = _invalid_update(adapter, sanitized)
    if invalid is not None:
        return invalid
    # The store's ValueError is a rejected value (zone, schedule, repeat): 400 like POST.
    return job_response(adapter, lambda jid: facade._cron_update(jid, sanitized), job_id, notify=True,
                        invalid_input=ValueError)


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
