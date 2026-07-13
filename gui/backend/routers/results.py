import io
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from core import state
from core.config import RESULTS_DIR
from services.tsv_reader import read_summary, read_tsv_paginated

router = APIRouter(prefix="/api/jobs", tags=["results"])
history_router = APIRouter(prefix="/api/results", tags=["history"])


async def _resolve_output_dir(job_id: str) -> Path:
    """Return the output directory for a job.

    Checks in-memory state first; falls back to RESULTS_DIR/{job_id} on disk
    so that past runs can be browsed without an active job record.
    """
    job = await state.get_job(job_id)
    if job is not None:
        if not job.output_dir:
            raise HTTPException(
                404,
                detail={"error": "no_output", "message": "Job has no output directory.", "details": None},
            )
        return Path(job.output_dir)

    candidate = RESULTS_DIR / job_id
    if candidate.is_dir():
        return candidate

    raise HTTPException(
        404,
        detail={"error": "job_not_found", "message": f"Job '{job_id}' not found.", "details": None},
    )


# ─── History ──────────────────────────────────────────────────────────────────


@history_router.get("")
async def list_results():
    """Scan the results/ directory and return all past search runs."""
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


# ─── Summary ──────────────────────────────────────────────────────────────────


@router.get("/{job_id}/summary")
async def get_summary(job_id: str):
    output_dir = await _resolve_output_dir(job_id)
    queries = read_summary(
        output_dir / "search_results_summary.tsv",
        output_dir / "search_results",
    )
    return {"queries": queries}


# ─── Protein ranking ──────────────────────────────────────────────────────────


@router.get("/{job_id}/queries/{query_id}/protein-ranking")
async def get_protein_ranking(
    job_id: str,
    query_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=900000),
    search_type: str = "all",
    sort_by: str = "protein_rank",
    sort_dir: str = "asc",
):
    output_dir = await _resolve_output_dir(job_id)
    path = output_dir / "search_results" / query_id / "protein_ranking.tsv"
    filters = {"search_type": search_type} if search_type != "all" else None
    return read_tsv_paginated(path, page, per_page, filters, sort_by, sort_dir)


# ─── Known ligands ────────────────────────────────────────────────────────────


@router.get("/{job_id}/queries/{query_id}/known-ligands")
async def get_known_ligands(
    job_id: str,
    query_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=900000),
    search_type: str = "all",
    source: str = "all",
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
):
    output_dir = await _resolve_output_dir(job_id)
    path = output_dir / "search_results" / query_id / "known_ligands.tsv"
    filters: dict = {}
    if search_type != "all":
        filters["search_type"] = search_type
    if source != "all":
        filters["source"] = source
    return read_tsv_paginated(path, page, per_page, filters or None, sort_by, sort_dir)


# ─── Predicted ligands ────────────────────────────────────────────────────────


@router.get("/{job_id}/queries/{query_id}/predicted-ligands")
async def get_predicted_ligands(
    job_id: str,
    query_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=900000),
    search_type: str = "all",
    sort_by: str = "tanimoto",
    sort_dir: str = "desc",
):
    output_dir = await _resolve_output_dir(job_id)
    path = output_dir / "search_results" / query_id / "predicted_ligands.tsv"
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
            qdir = output_dir / "search_results" / query_id
            if qdir.exists():
                for f in qdir.iterdir():
                    if f.is_file():
                        zf.write(f, base / f.name)
    buf.seek(0)
    return buf


@router.get("/{job_id}/download")
async def download_job(job_id: str):
    output_dir = await _resolve_output_dir(job_id)
    buf = _build_zip(output_dir, job_id)
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}.zip"},
    )


@router.get("/{job_id}/queries/{query_id}/download")
async def download_query(job_id: str, query_id: str):
    output_dir = await _resolve_output_dir(job_id)
    buf = _build_zip(output_dir, job_id, query_id)
    filename = f"{job_id}_{query_id}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )