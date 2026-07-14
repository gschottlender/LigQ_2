import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from core.config import PIPELINE_ROOT, UPLOADS_DIR
from core import state
from models.job import AddRepresentationRequest, Job, JobStatus
from services.fs_inspector import database_exists, list_databases, representation_exists
from services.job_runner import run_job

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


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


# ─── Search ───────────────────────────────────────────────────────────────────


@router.post("/search", status_code=201)
async def start_search(
    background_tasks: BackgroundTasks,
    fasta_file: UploadFile = File(...),
    ligand_provider: str = Form(...),
    search_representation: str = Form(...),
    search_metric: str = Form(...),
    search_threshold: float | None = Form(None),
    search_threshold_max: float | None = Form(None),
    use_sequence: bool = Form(True),
    use_nearest_k: bool = Form(True),
    nearest_k: int = Form(5),
    use_domains: bool = Form(False),
    known_only: bool = Form(False),
):
    content = await fasta_file.read()
    if not content or not _valid_fasta(content):
        return _err("invalid_fasta", "FASTA file is empty or contains no valid sequences.")
    all_queries = _get_fasta_query_ids(content)
    n_queries = len(all_queries)

    if not database_exists(ligand_provider):
        return _err("database_not_found", f"Ligand provider '{ligand_provider}' not found.")

    if not representation_exists(ligand_provider, search_representation):
        return _err(
            "representation_not_found",
            f"Representation '{search_representation}' not found for '{ligand_provider}'.",
        )

    if search_threshold is not None and not (0.0 <= search_threshold <= 1.0):
        return _err("invalid_threshold", "search_threshold must be between 0.0 and 1.0.")
    if search_threshold_max is not None and not (0.0 <= search_threshold_max <= 1.0):
        return _err("invalid_threshold_max", "search_threshold_max must be between 0.0 and 1.0.")
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
    fasta_path.write_bytes(content)

    stem = re.sub(r"[^\w-]", "_", Path(fasta_file.filename or "queries").stem)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_rel = f"results/{stem}_{timestamp}"
    output_dir = PIPELINE_ROOT / output_dir_rel

    args: list[str] = [
        str(PIPELINE_ROOT / "run_ligq_2.py"),
        "--input-fasta", str(fasta_path),
        "--output-dir", str(output_dir),
        "--ligand-provider", ligand_provider,
        "--search-representation", search_representation,
        "--search-metric", search_metric,
        "--progress-json",
    ]
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
    background_tasks.add_task(run_job, job_id, args, output_dir, n_queries)

    return {"job_id": job_id, "status": "queued", "output_dir": output_dir_rel}


# ─── Build database ────────────────────────────────────────────────────────────


@router.post("/build-database", status_code=201)
async def build_database(
    background_tasks: BackgroundTasks,
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
    upload_path.write_bytes(await input_file.read())

    args: list[str] = [
        str(PIPELINE_ROOT / "build_compound_database.py"),
        "--input-file", str(upload_path),
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
    background_tasks.add_task(run_job, job_id, args)

    return {"job_id": job_id, "status": "queued"}


# ─── Add representation ────────────────────────────────────────────────────────


@router.post("/add-representation", status_code=201)
async def add_representation(
    background_tasks: BackgroundTasks,
    body: AddRepresentationRequest,
):
    if not database_exists(body.base_name):
        return _err("database_not_found", f"Database '{body.base_name}' not found.")

    if representation_exists(body.base_name, body.rep_name):
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
        "--output-dir", "databases",
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
    background_tasks.add_task(run_job, job_id, args)

    return {"job_id": job_id, "status": "queued"}


# ─── List / Get / Delete ───────────────────────────────────────────────────────


@router.get("")
async def list_jobs(limit: int = 20, job_type: str | None = None):
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
    async with state._lock:
        proc = state.processes.get(job_id)
        if proc:
            proc.terminate()
    await state.delete_job(job_id)
