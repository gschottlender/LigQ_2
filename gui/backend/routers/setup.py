from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from core import state
from models.job import Job, JobStatus
from services.job_runner import enqueue_job
from services.setup_service import inspect_setup_status, is_default_setup_ready, setup_job_args


router = APIRouter(prefix="/api/setup", tags=["setup"])
ACTIVE_STATUSES = {JobStatus.queued, JobStatus.running, JobStatus.partial_results}


async def _latest_setup_job() -> Job | None:
    return await state.get_latest_job_by_type("setup")


@router.get("/status")
async def setup_status():
    latest = await _latest_setup_job()
    active_job = latest if latest and latest.status in ACTIVE_STATUSES else None
    status = await inspect_setup_status(active=active_job is not None)
    status["job_id"] = active_job.job_id if active_job else None
    status["job_status"] = active_job.status.value if active_job else None
    return status


@router.post("/download", status_code=201)
async def start_setup_download():
    latest = await _latest_setup_job()
    if latest and latest.status in ACTIVE_STATUSES:
        return JSONResponse(
            status_code=200,
            content={"job_id": latest.job_id, "status": latest.status.value},
        )

    if is_default_setup_ready():
        return JSONResponse(
            status_code=409,
            content={
                "error": "setup_already_complete",
                "message": "Default LigQ 2 data is already installed.",
                "details": None,
            },
        )

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        job_type="setup",
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
    )
    await state.set_job(job)
    await enqueue_job(job_id, setup_job_args())
    return {"job_id": job_id, "status": JobStatus.queued.value}
