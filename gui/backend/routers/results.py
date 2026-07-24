import asyncio
import io
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from core import state
from core.config import RESULTS_DIR
from models.job import JobStatus
from services.tsv_reader import read_summary, read_tsv_paginated
from core.policy import is_web_mode
from services.search_artifacts import cleanup_web_search_artifacts
from services.web_access import require_job_access, session_hash

router = APIRouter(prefix="/api/jobs", tags=["results"])
history_router = APIRouter(prefix="/api/results", tags=["history"])

ACTIVE_RESULT_STATUSES = {
    JobStatus.queued,
    JobStatus.running,
    JobStatus.partial_results,
}


async def _resolve_output_dir(request: Request, job_id: str) -> Path:
    """Return the output directory for a job.

    Checks in-memory state first; falls back to RESULTS_DIR/{job_id} on disk
    so that past runs can be browsed without an active job record.
    """
    job = await state.get_job(job_id)
    if job is not None:
        require_job_access(request, job)
        if not job.output_dir:
            raise HTTPException(
                404,
                detail={"error": "no_output", "message": "Job has no output directory.", "details": None},
            )
        return Path(job.output_dir)

    if is_web_mode():
        require_job_access(request, None)
    candidate = RESULTS_DIR / job_id
    if candidate.is_dir():
        return candidate

    raise HTTPException(
        404,
        detail={"error": "job_not_found", "message": f"Job '{job_id}' not found.", "details": None},
    )


def _query_result_dir(output_dir: Path, query_id: str) -> Path:
    root = (output_dir / "search_results").resolve()
    candidate = (root / query_id).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            404,
            detail={
                "error": "query_not_found",
                "message": "The requested query was not found.",
                "details": None,
            },
        ) from exc
    return candidate


# ─── History ──────────────────────────────────────────────────────────────────


@history_router.get("")
async def list_results(request: Request):
    """Scan the results/ directory and return all past search runs."""
    if is_web_mode():
        owner = session_hash(request)
        entries = []
        for job in state.get_all_jobs():
            if (
                job.job_type != "search"
                or job.owner_session_hash != owner
                or not job.output_dir
                or job.status not in {
                    JobStatus.completed,
                    JobStatus.completed_with_warnings,
                }
            ):
                continue
            output_dir = Path(job.output_dir)
            if not output_dir.exists():
                continue
            entries.append({
                "result_id": job.job_id,
                "result_label": output_dir.name,
                "created_at": job.created_at.isoformat(),
                "n_queries": job.n_queries or len(job.all_queries),
                "queries": list(job.all_queries),
                "status": "completed",
                "search_mode": job.search_mode or "zinc",
            })
        return {"results": entries}

    if not RESULTS_DIR.exists():
        return {"results": []}

    entries = []
    for folder in RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue
        sr_dir = folder / "search_results"
        queries = (
            sorted(d.name for d in sr_dir.iterdir() if d.is_dir())
            if sr_dir.exists()
            else []
        )
        status = "completed" if (folder / "search_results_summary.tsv").exists() else "partial"
        created_at = datetime.fromtimestamp(
            folder.stat().st_mtime, tz=timezone.utc
        ).isoformat()
        entries.append({
            "result_id": folder.name,
            "created_at": created_at,
            "n_queries": len(queries),
            "queries": queries,
            "status": status,
        })

    entries.sort(key=lambda e: e["created_at"], reverse=True)
    return {"results": entries}


@history_router.delete("")
async def clear_results_history(request: Request):
    """Delete stored search results while preserving active search outputs."""
    if is_web_mode():
        owner = session_hash(request)
        deleted_results: list[str] = []
        skipped_active: list[str] = []
        failed_results: list[str] = []
        for job in state.get_all_jobs():
            if job.job_type != "search" or job.owner_session_hash != owner:
                continue
            if job.status in ACTIVE_RESULT_STATUSES:
                skipped_active.append(job.job_id)
                continue
            try:
                await asyncio.to_thread(
                    cleanup_web_search_artifacts,
                    job,
                    remove_results=True,
                )
                await state.delete_job(job.job_id)
                deleted_results.append(job.job_id)
            except OSError:
                failed_results.append(job.job_id)
        return {
            "deleted_count": len(deleted_results),
            "deleted_results": deleted_results,
            "skipped_active": skipped_active,
            "failed_count": len(failed_results),
            "failed_results": failed_results,
        }

    if not RESULTS_DIR.exists():
        return {
            "deleted_count": 0,
            "deleted_results": [],
            "skipped_active": [],
            "failed_results": [],
        }

    active_output_dirs = {
        Path(job.output_dir).resolve()
        for job in state.get_all_jobs()
        if (
            job.job_type == "search"
            and job.status in ACTIVE_RESULT_STATUSES
            and job.output_dir
        )
    }

    deleted_results: list[str] = []
    skipped_active: list[str] = []
    failed_results: list[str] = []

    for folder in sorted(RESULTS_DIR.iterdir(), key=lambda path: path.name):
        if not folder.is_dir() or folder.is_symlink():
            continue
        if folder.resolve() in active_output_dirs:
            skipped_active.append(folder.name)
            continue
        try:
            shutil.rmtree(folder)
            deleted_results.append(folder.name)
        except OSError:
            failed_results.append(folder.name)

    return {
        "deleted_count": len(deleted_results),
        "deleted_results": deleted_results,
        "skipped_active": skipped_active,
        "failed_results": failed_results,
    }


# ─── Summary ──────────────────────────────────────────────────────────────────


@router.get("/{job_id}/summary")
async def get_summary(request: Request, job_id: str):
    output_dir = await _resolve_output_dir(request, job_id)
    queries = read_summary(
        output_dir / "search_results_summary.tsv",
        output_dir / "search_results",
    )
    return {"queries": queries}


# ─── Protein ranking ──────────────────────────────────────────────────────────


@router.get("/{job_id}/queries/{query_id}/protein-ranking")
async def get_protein_ranking(
    request: Request,
    job_id: str,
    query_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=900000),
    search_type: str = "all",
    sort_by: str = "protein_rank",
    sort_dir: str = "asc",
):
    output_dir = await _resolve_output_dir(request, job_id)
    path = _query_result_dir(output_dir, query_id) / "protein_ranking.tsv"
    filters = {"search_type": search_type} if search_type != "all" else None
    return read_tsv_paginated(path, page, per_page, filters, sort_by, sort_dir)


# ─── Known ligands ────────────────────────────────────────────────────────────


@router.get("/{job_id}/queries/{query_id}/known-ligands")
async def get_known_ligands(
    request: Request,
    job_id: str,
    query_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=900000),
    search_type: str = "all",
    source: str = "all",
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
):
    output_dir = await _resolve_output_dir(request, job_id)
    path = _query_result_dir(output_dir, query_id) / "known_ligands.tsv"
    filters: dict = {}
    if search_type != "all":
        filters["search_type"] = search_type
    if source != "all":
        filters["source"] = source
    return read_tsv_paginated(path, page, per_page, filters or None, sort_by, sort_dir)


# ─── Predicted ligands ────────────────────────────────────────────────────────


@router.get("/{job_id}/queries/{query_id}/predicted-ligands")
async def get_predicted_ligands(
    request: Request,
    job_id: str,
    query_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=900000),
    search_type: str = "all",
    sort_by: str = "tanimoto",
    sort_dir: str = "desc",
):
    output_dir = await _resolve_output_dir(request, job_id)
    path = _query_result_dir(output_dir, query_id) / "predicted_ligands.tsv"
    filters = {"search_type": search_type} if search_type != "all" else None
    return read_tsv_paginated(path, page, per_page, filters, sort_by, sort_dir)


# ─── Downloads ────────────────────────────────────────────────────────────────


def _build_zip(output_dir: Path, job_id: str, query_id: Optional[str] = None) -> io.BytesIO:
    buf = io.BytesIO()
    base = Path(job_id)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if query_id is None:
            summary = output_dir / "search_results_summary.tsv"
            if summary.exists():
                zf.write(summary, base / "search_results_summary.tsv")
            sr_dir = output_dir / "search_results"
            if sr_dir.exists():
                for qdir in sr_dir.iterdir():
                    if qdir.is_dir():
                        for f in qdir.iterdir():
                            if f.is_file():
                                zf.write(f, base / "search_results" / qdir.name / f.name)
        else:
            qdir = _query_result_dir(output_dir, query_id)
            if qdir.exists():
                for f in qdir.iterdir():
                    if f.is_file():
                        zf.write(f, base / f.name)
    buf.seek(0)
    return buf


@router.get("/{job_id}/download")
async def download_job(request: Request, job_id: str):
    output_dir = await _resolve_output_dir(request, job_id)
    buf = _build_zip(output_dir, job_id)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}.zip"},
    )


@router.get("/{job_id}/queries/{query_id}/download")
async def download_query(request: Request, job_id: str, query_id: str):
    output_dir = await _resolve_output_dir(request, job_id)
    buf = _build_zip(output_dir, job_id, query_id)
    filename = f"{job_id}_{query_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
