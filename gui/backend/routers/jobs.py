from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.config import DATABASES_DIR, PIPELINE_ROOT, RESULTS_DIR, TEMP_RESULTS_DIR, UPLOADS_DIR
from core import state
from models.job import AddRepresentationRequest, Job, JobFailure, JobStatus
from services.fs_inspector import (
    database_exists,
    representation_is_search_ready,
)
from services.job_runner import enqueue_job, terminate_job_process
from services.uploads import UploadTooLargeError, inspect_fasta, save_upload_stream

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

BSI_REPRESENTATION = "morgan_1024_r2"
BSI_CLI_METRIC = "tanimoto"
BSI_DEFAULT_THRESHOLD = 0.98


def _err(code: str, message: str, status_code: int = 400, details=None):
    return JSONResponse(
        status_code=status_code,
        content={"error": code, "message": message, "details": details},
    )


def _valid_fasta(content: bytes) -> bool:
    try:
        text = content.decode("utf-8", errors="replace")
    except Exception:
        return False
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    has_header = any(l.startswith(">") for l in lines)
    has_seq = any(not l.startswith(">") for l in lines)
    return has_header and has_seq


def _get_fasta_query_ids(content: bytes) -> list[str]:
    ids = []
    for line in content.decode("utf-8", errors="replace").splitlines():
        if line.startswith(">"):
            header = line[1:].strip()
            if header:
                ids.append(header.split()[0])
    return ids


def _build_search_args(
    *,
    fasta_path: Path,
    output_dir: Path,
    ligand_provider: str,
    search_representation: str,
    search_metric: str,
    search_threshold: Optional[float],
    search_threshold_max: Optional[float],
    use_sequence: bool,
    use_nearest_k: bool,
    nearest_k: int,
    use_domains: bool,
    known_only: bool,
    use_bsi: bool,
    bsi_threshold: float,
) -> list[str]:
    effective_representation = BSI_REPRESENTATION if use_bsi else search_representation
    effective_metric = BSI_CLI_METRIC if use_bsi else search_metric
    args: list[str] = [
        str(PIPELINE_ROOT / "run_ligq_2.py"),
        "--input-fasta", str(fasta_path),
        "--output-dir", str(output_dir),
        "--ligand-provider", ligand_provider,
        "--search-representation", effective_representation,
        "--search-metric", effective_metric,
        "--data-dir", str(DATABASES_DIR),
        "--temp-results-dir", str(TEMP_RESULTS_DIR / fasta_path.stem),
        "--progress-json",
    ]
    if use_bsi:
        args += ["--bsi", "--bsi-threshold", str(bsi_threshold)]
    else:
        if search_threshold is not None:
            args += ["--search-threshold", str(search_threshold)]
        if search_threshold_max is not None:
            args += ["--search-threshold-max", str(search_threshold_max)]
    if use_sequence:
        args.append("--sequence")
    if use_nearest_k:
        args += ["--nearest_k", "--nearest-k", str(nearest_k)]
    if use_domains:
        args.append("--domains")
    if known_only:
        args.append("--known-only")
    return args


# ─── Search ───────────────────────────────────────────────────────────────────


@router.post("/search", status_code=201)
async def start_search(
    fasta_file: UploadFile = File(...),
    ligand_provider: str = Form(...),
    search_representation: str = Form(...),
    search_metric: str = Form(...),
    search_threshold: Optional[float] = Form(None),
    search_threshold_max: Optional[float] = Form(None),
    use_sequence: bool = Form(True),
    use_nearest_k: bool = Form(True),
    nearest_k: int = Form(5),
    use_domains: bool = Form(False),
    known_only: bool = Form(False),
    use_bsi: bool = Form(False),
    bsi_threshold: float = Form(BSI_DEFAULT_THRESHOLD),
):
    if not database_exists(ligand_provider):
        return _err("database_not_found", f"Ligand provider '{ligand_provider}' not found.")

    effective_representation = BSI_REPRESENTATION if use_bsi else search_representation
    if not representation_is_search_ready(ligand_provider, effective_representation):
        return _err(
            "representation_not_ready",
            (
                f"Representation '{effective_representation}' is incomplete. Both its .dat and "
                f".meta.json files must exist for '{ligand_provider}' and 'pdb_chembl'. "
                "Add the representation again before searching."
            ),
        )

    if use_bsi:
        if not (0.0 <= bsi_threshold <= 1.0):
            return _err("invalid_bsi_threshold", "bsi_threshold must be between 0.0 and 1.0.")
        if known_only:
            return _err(
                "incompatible_search_modes",
                "BSI and known-only mode cannot be enabled together.",
            )
    else:
        if search_threshold is not None and not (0.0 <= search_threshold <= 1.0):
            return _err("invalid_threshold", "search_threshold must be between 0.0 and 1.0.")
        if search_threshold_max is not None and not (0.0 <= search_threshold_max <= 1.0):
            return _err(
                "invalid_threshold_max",
                "search_threshold_max must be between 0.0 and 1.0.",
            )
        if (
            search_threshold is not None
            and search_threshold_max is not None
            and search_threshold_max < search_threshold
        ):
            return _err(
                "invalid_threshold_range",
                "search_threshold_max must be greater than or equal to search_threshold.",
            )

    if use_nearest_k and nearest_k < 1:
        return _err("invalid_nearest_k", "nearest_k must be >= 1 when use_nearest_k is true.")

    job_id = str(uuid.uuid4())
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    fasta_path = UPLOADS_DIR / f"{job_id}.fasta"
    try:
        uploaded_bytes = await save_upload_stream(fasta_file, fasta_path)
    except UploadTooLargeError as exc:
        return _err("file_too_large", str(exc), status_code=413)
    if uploaded_bytes == 0:
        fasta_path.unlink(missing_ok=True)
        return _err("invalid_fasta", "FASTA file is empty or contains no valid sequences.")

    valid_fasta, all_queries = inspect_fasta(fasta_path)
    if not valid_fasta:
        fasta_path.unlink(missing_ok=True)
        return _err("invalid_fasta", "FASTA file is empty or contains no valid sequences.")
    n_queries = len(all_queries)

    stem = re.sub(r"[^\w-]", "_", Path(fasta_file.filename or "queries").stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_name = f"{stem}_{timestamp}"
    output_dir_rel = f"results/{result_name}"
    output_dir = RESULTS_DIR / result_name

    args = _build_search_args(
        fasta_path=fasta_path,
        output_dir=output_dir,
        ligand_provider=ligand_provider,
        search_representation=search_representation,
        search_metric=search_metric,
        search_threshold=search_threshold,
        search_threshold_max=search_threshold_max,
        use_sequence=use_sequence,
        use_nearest_k=use_nearest_k,
        nearest_k=nearest_k,
        use_domains=use_domains,
        known_only=known_only,
        use_bsi=use_bsi,
        bsi_threshold=bsi_threshold,
    )

    job = Job(
        job_id=job_id,
        job_type="search",
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        output_dir=str(output_dir),
        all_queries=all_queries,
        n_queries=n_queries,
    )
    await state.set_job(job)
    await enqueue_job(job_id, args, output_dir, n_queries)

    return {"job_id": job_id, "status": "queued", "output_dir": output_dir_rel}


# ─── Build database ────────────────────────────────────────────────────────────


@router.post("/build-database", status_code=201)
async def build_database(
    input_file: UploadFile = File(...),
    base_name: str = Form(...),
    id_column: str = Form(""),
    smiles_column: str = Form(""),
):
    if not re.match(r"^[a-zA-Z0-9_]+$", base_name):
        return _err(
            "invalid_name",
            "base_name must contain only alphanumeric characters and underscores.",
        )

    if database_exists(base_name):
        return _err("conflict", f"Database '{base_name}' already exists.", status_code=409)

    suffix = Path(input_file.filename or "").suffix.lower()
    if suffix not in (".smi", ".csv", ".tsv", ".parquet"):
        return _err(
            "invalid_file_type",
            "Supported file types: .smi, .csv, .tsv, .parquet",
        )

    if suffix != ".smi" and (not id_column or not smiles_column):
        return _err(
            "missing_columns",
            "id_column and smiles_column are required for CSV/TSV/Parquet files.",
        )

    job_id = str(uuid.uuid4())
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOADS_DIR / f"{job_id}{suffix}"
    try:
        uploaded_bytes = await save_upload_stream(input_file, upload_path)
    except UploadTooLargeError as exc:
        return _err("file_too_large", str(exc), status_code=413)
    if uploaded_bytes == 0:
        upload_path.unlink(missing_ok=True)
        return _err("empty_file", "The uploaded compound file is empty.")

    args: list[str] = [
        str(PIPELINE_ROOT / "build_compound_database.py"),
        "--input-file", str(upload_path),
        "--output-dir", str(DATABASES_DIR),
        "--base-name", base_name,
        "--progress-json",
    ]
    if suffix != ".smi":
        args += ["--id-column", id_column, "--smiles-column", smiles_column]

    job = Job(
        job_id=job_id,
        job_type="build_database",
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
    )
    await state.set_job(job)
    await enqueue_job(job_id, args)

    return {"job_id": job_id, "status": "queued"}


# ─── Add representation ────────────────────────────────────────────────────────


@router.post("/add-representation", status_code=201)
async def add_representation(
    body: AddRepresentationRequest,
):
    if not database_exists(body.base_name):
        return _err("database_not_found", f"Database '{body.base_name}' not found.")

    if representation_is_search_ready(body.base_name, body.rep_name):
        return _err(
            "conflict",
            f"Representation '{body.rep_name}' already exists for '{body.base_name}'.",
            status_code=409,
        )

    if body.representation_type == "rdkit" and not body.rdkit_fp_kind:
        return _err("missing_field", "rdkit_fp_kind is required for rdkit representation.")

    if body.representation_type == "huggingface" and not body.model_id:
        return _err("missing_field", "model_id is required for huggingface representation.")

    args: list[str] = [
        str(PIPELINE_ROOT / "add_new_representation.py"),
        "--output-dir", str(DATABASES_DIR),
        "--representation-type", body.representation_type,
        "--n-bits", str(body.n_bits),
        "--rep-name", body.rep_name,
        "--progress-json",
    ]
    if body.representation_type == "rdkit" and body.rdkit_fp_kind:
        args += ["--rdkit-fp-kind", body.rdkit_fp_kind]
    if body.representation_type == "huggingface" and body.model_id:
        args += ["--model-id", body.model_id]
    if body.batch_size is not None:
        args += ["--batch-size", str(body.batch_size)]
    if body.n_jobs is not None:
        args += ["--n-jobs", str(body.n_jobs)]

    if body.base_name in ("zinc", "local"):
        args += ["--base", body.base_name]
    else:
        args += ["--base-name", body.base_name]
        args += ["--ensure-local-compatible"]

    job_id = str(uuid.uuid4())
    job = Job(
        job_id=job_id,
        job_type="add_representation",
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
    )
    await state.set_job(job)
    await enqueue_job(job_id, args)

    return {"job_id": job_id, "status": "queued"}


# ─── List / Get / Delete ───────────────────────────────────────────────────────


@router.get("")
async def list_jobs(limit: int = 20, job_type: Optional[str] = None):
    all_jobs = state.get_all_jobs()
    if job_type:
        all_jobs = [j for j in all_jobs if j.job_type == job_type]
    return {"jobs": [j.model_dump() for j in all_jobs[:limit]]}


@router.get("/{job_id}")
async def get_job(job_id: str):
    job = await state.get_job(job_id)
    if job is None:
        raise HTTPException(
            404,
            detail={"error": "job_not_found", "message": f"Job '{job_id}' not found.", "details": None},
        )
    return job.model_dump()


@router.delete("/{job_id}", status_code=204)
async def cancel_job(job_id: str):
    job = await state.get_job(job_id)
    if job is None:
        raise HTTPException(
            404,
            detail={"error": "job_not_found", "message": f"Job '{job_id}' not found.", "details": None},
        )
    if job.status in {
        JobStatus.completed,
        JobStatus.completed_with_warnings,
        JobStatus.failed,
        JobStatus.cancelled,
        JobStatus.interrupted,
    }:
        return

    now = datetime.now(timezone.utc)
    message = "The job was cancelled by the user."
    await state.update_job(
        job_id,
        status=JobStatus.cancelled,
        finished_at=now,
        elapsed_seconds=(
            (now - job.started_at).total_seconds() if job.started_at else None
        ),
        error=message,
        failure=JobFailure(label="Cancelled", message=message),
    )
    await terminate_job_process(job_id)
