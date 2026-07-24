from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core import state
from models.job import Job, JobStatus
from services.job_runner import enqueue_job
from services.setup_service import inspect_setup_status, is_default_setup_ready, setup_job_args
from core.policy import is_web_mode
from services.web_access import require_resource_management
from services.web_readiness import inspect_web_readiness


router = APIRouter(prefix="/api/setup", tags=["setup"])
ACTIVE_STATUSES = {JobStatus.queued, JobStatus.running, JobStatus.partial_results}


class SetupDownloadRequest(BaseModel):
    include_ecfp_cache: bool = True
    include_fcfp_cache: bool = False


async def _latest_setup_job() -> Job | None:
    return await state.get_latest_job_by_type("setup")


@router.get("/status")
async def setup_status():
    if is_web_mode():
        readiness = await inspect_web_readiness()
        return {
            "ready": bool(readiness.get("ready")),
            "state": "ready" if readiness.get("ready") else "maintenance",
            "message": (
                "Public search data is ready."
                if readiness.get("ready")
                else "The public service data is being maintained."
            ),
            "packages": [],
            "job_id": None,
            "job_status": None,
        }
    latest = await _latest_setup_job()
    active_job = latest if latest and latest.status in ACTIVE_STATUSES else None
    status = await inspect_setup_status(active=active_job is not None)
    status["job_id"] = active_job.job_id if active_job else None
    status["job_status"] = active_job.status.value if active_job else None
    return status


@router.post("/download", status_code=201)
async def start_setup_download(options: SetupDownloadRequest | None = None):
    require_resource_management()
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
    selection = options or SetupDownloadRequest()
    await enqueue_job(
        job_id,
        setup_job_args(
            include_ecfp_cache=selection.include_ecfp_cache,
            include_fcfp_cache=selection.include_fcfp_cache,
        ),
    )
    return {"job_id": job_id, "status": JobStatus.queued.value}
