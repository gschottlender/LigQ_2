import asyncio
from typing import Optional
from models.job import Job, JobStatus

jobs: dict[str, Job] = {}
processes: dict[str, asyncio.subprocess.Process] = {}
_lock = asyncio.Lock()


async def get_job(job_id: str) -> Optional[Job]:
    async with _lock:
        return jobs.get(job_id)


async def set_job(job: Job) -> None:
    async with _lock:
        jobs[job.job_id] = job


async def update_job(job_id: str, **kwargs) -> Optional[Job]:
    async with _lock:
        job = jobs.get(job_id)
        if job is None:
            return None
        updated = job.model_copy(update=kwargs)
        jobs[job_id] = updated
        return updated


async def delete_job(job_id: str) -> bool:
    async with _lock:
        if job_id not in jobs:
            return False
        del jobs[job_id]
        processes.pop(job_id, None)
        return True


def get_all_jobs() -> list[Job]:
    return sorted(jobs.values(), key=lambda j: j.created_at, reverse=True)


async def get_latest_job_by_type(job_type: str) -> Optional[Job]:
    async with _lock:
        matching = [job for job in jobs.values() if job.job_type == job_type]
        return max(matching, key=lambda job: job.created_at, default=None)
