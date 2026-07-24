from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from core import state
from core.config import WEB_RESULT_RETENTION_SECONDS
from core.policy import is_web_mode
from models.job import JobStatus
from services.search_artifacts import cleanup_web_search_artifacts


_cleanup_task: asyncio.Task | None = None
logger = logging.getLogger(__name__)
_TERMINAL = {
    JobStatus.completed,
    JobStatus.completed_with_warnings,
    JobStatus.failed,
    JobStatus.cancelled,
    JobStatus.interrupted,
}


async def cleanup_expired_web_results() -> int:
    if not is_web_mode():
        return 0
    now = datetime.now(timezone.utc)
    expired = []
    for job in state.get_all_jobs():
        if (
            job.owner_session_hash is None
            or job.job_type != "search"
            or job.status not in _TERMINAL
            or job.finished_at is None
        ):
            continue
        if (now - job.finished_at).total_seconds() >= WEB_RESULT_RETENTION_SECONDS:
            expired.append(job)

    for job in expired:
        try:
            await asyncio.to_thread(
                cleanup_web_search_artifacts,
                job,
                remove_results=True,
            )
            await state.delete_job(job.job_id)
        except OSError:
            logger.exception("Could not remove expired public result %s", job.job_id)
    return len(expired)


async def _cleanup_loop() -> None:
    while True:
        try:
            await cleanup_expired_web_results()
        except Exception:
            logger.exception("Unexpected public result cleanup failure")
        await asyncio.sleep(60)


async def start_web_cleanup() -> None:
    global _cleanup_task
    if not is_web_mode():
        return
    await cleanup_expired_web_results()
    if _cleanup_task is None or _cleanup_task.done():
        _cleanup_task = asyncio.create_task(_cleanup_loop(), name="ligq-web-cleanup")


async def stop_web_cleanup() -> None:
    global _cleanup_task
    if _cleanup_task is None:
        return
    _cleanup_task.cancel()
    try:
        await _cleanup_task
    except asyncio.CancelledError:
        pass
    _cleanup_task = None
