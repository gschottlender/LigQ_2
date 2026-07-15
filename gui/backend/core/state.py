from __future__ import annotations

import asyncio
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from core.config import JOB_DB_PATH
from models.job import Job, JobFailure, JobStatus


logger = logging.getLogger(__name__)

jobs: dict[str, Job] = {}
processes: dict[str, asyncio.subprocess.Process] = {}
_lock = asyncio.Lock()
_initialized = False

_UNFINISHED_STATUSES = {
    JobStatus.queued,
    JobStatus.running,
    JobStatus.partial_results,
}


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(JOB_DB_PATH, timeout=30)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _write_job(job: Job) -> None:
    if not _initialized:
        return
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO jobs (job_id, created_at, payload)
            VALUES (?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                created_at = excluded.created_at,
                payload = excluded.payload
            """,
            (job.job_id, job.created_at.isoformat(), job.model_dump_json()),
        )


def _delete_persisted_job(job_id: str) -> None:
    if not _initialized:
        return
    with _connect() as connection:
        connection.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))


async def initialize() -> None:
    """Load persisted jobs and mark prior unfinished work as interrupted."""
    global _initialized

    async with _lock:
        JOB_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            rows = connection.execute(
                "SELECT job_id, payload FROM jobs ORDER BY created_at DESC"
            ).fetchall()

        jobs.clear()
        for job_id, payload in rows:
            try:
                jobs[job_id] = Job.model_validate_json(payload)
            except Exception as exc:
                logger.warning("Ignoring invalid persisted job %s: %s", job_id, exc)

        _initialized = True
        now = datetime.now(timezone.utc)
        for job_id, job in list(jobs.items()):
            if job.status not in _UNFINISHED_STATUSES:
                continue
            message = "The backend stopped before this job finished."
            interrupted = job.model_copy(
                update={
                    "status": JobStatus.interrupted,
                    "finished_at": now,
                    "elapsed_seconds": (
                        (now - job.started_at).total_seconds() if job.started_at else None
                    ),
                    "error": message,
                    "failure": JobFailure(label="Interrupted", message=message),
                }
            )
            jobs[job_id] = interrupted
            _write_job(interrupted)


async def get_job(job_id: str) -> Optional[Job]:
    async with _lock:
        return jobs.get(job_id)


async def set_job(job: Job) -> None:
    async with _lock:
        jobs[job.job_id] = job
        _write_job(job)


async def update_job(job_id: str, **kwargs) -> Optional[Job]:
    async with _lock:
        job = jobs.get(job_id)
        if job is None:
            return None
        updated = job.model_copy(update=kwargs)
        jobs[job_id] = updated
        _write_job(updated)
        return updated


async def delete_job(job_id: str) -> bool:
    """Permanently remove a job record. Runtime cancellation uses update_job."""
    async with _lock:
        if job_id not in jobs:
            return False
        del jobs[job_id]
        processes.pop(job_id, None)
        _delete_persisted_job(job_id)
        return True


def get_all_jobs() -> list[Job]:
    return sorted(jobs.values(), key=lambda job: job.created_at, reverse=True)


async def get_latest_job_by_type(job_type: str) -> Optional[Job]:
    async with _lock:
        matching = [job for job in jobs.values() if job.job_type == job_type]
        return max(matching, key=lambda job: job.created_at, default=None)


async def mark_unfinished_interrupted(message: str) -> None:
    now = datetime.now(timezone.utc)
    async with _lock:
        for job_id, job in list(jobs.items()):
            if job.status not in _UNFINISHED_STATUSES:
                continue
            interrupted = job.model_copy(
                update={
                    "status": JobStatus.interrupted,
                    "finished_at": now,
                    "elapsed_seconds": (
                        (now - job.started_at).total_seconds() if job.started_at else None
                    ),
                    "error": message,
                    "failure": JobFailure(label="Interrupted", message=message),
                }
            )
            jobs[job_id] = interrupted
            _write_job(interrupted)
