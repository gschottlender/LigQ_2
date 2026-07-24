from __future__ import annotations

import asyncio
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from core.config import (
    DATABASES_DIR,
    MAX_UPLOAD_BYTES,
    PIPELINE_ROOT,
    RESULTS_DIR,
    TEMP_RESULTS_DIR,
    UPLOADS_DIR,
    WEB_MAX_FASTA_BYTES,
    WEB_MAX_FASTA_RESIDUES,
    WEB_MAX_FASTA_SEQUENCES,
    WEB_RATE_LIMIT_COUNT,
    WEB_SEARCH_TIMEOUT_SECONDS,
)
from core import state
from core.policy import is_web_mode, web_representation_policy
from models.job import AddRepresentationRequest, Job, JobFailure, JobStatus
from services.fs_inspector import (
    database_exists,
    representation_is_search_ready,
)
from services.hardware import get_hardware_capabilities
from services.job_runner import enqueue_job, terminate_job_process
from services.resource_artifacts import RESOURCE_JOB_TYPES, cleanup_resource_job_artifacts
from services.uploads import (
    UploadTooLargeError,
    inspect_fasta,
    inspect_fasta_details,
    save_upload_stream,
)
from services.web_access import (
    client_ip_hash,
    require_job_access,
    require_resource_management,
    session_hash,
)
from services.web_limits import rate_limit_status, record_accepted_search
from services.web_readiness import inspect_web_readiness
from services.search_artifacts import cleanup_search_artifacts, cleanup_web_search_artifacts

router = APIRouter(prefix="/api/jobs", tags=["jobs"])

BSI_REPRESENTATION = "morgan_1024_r2"
BSI_CLI_METRIC = "tanimoto"
BSI_DEFAULT_THRESHOLD = 0.98
_ACTIVE_JOB_STATUSES = {JobStatus.queued, JobStatus.running, JobStatus.partial_results}


def _err(
    code: str,
    message: str,
    status_code: int = 400,
    details=None,
    headers: dict[str, str] | None = None,
):
    return JSONResponse(
        status_code=status_code,
        content={"error": code, "message": message, "details": details},
        headers=headers,
    )


def _job_payload(job: Job) -> dict:
    if is_web_mode():
        payload = job.model_dump(exclude={"owner_session_hash", "input_path"})
        if job.output_dir:
            payload["output_dir"] = Path(job.output_dir).name
        return payload
    return job.model_dump()


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
    immutable_web_data: bool = False,
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
    if immutable_web_data:
        args += ["--data-read-only", "--predicted-cache-read-only"]
    return args


# ─── Search ───────────────────────────────────────────────────────────────────


@router.post("/search", status_code=201)
async def start_search(
    request: Request,
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
    web_mode = is_web_mode()
    ip_hash = client_ip_hash(request) if web_mode else ""
    if web_mode:
        readiness = await inspect_web_readiness()
        if not readiness.get("ready"):
            return _err(
                "service_unavailable",
                "The public search data or precomputed caches are unavailable.",
                status_code=503,
            )
        allowed, retry_after = await rate_limit_status(ip_hash)
        if not allowed:
            return _err(
                "rate_limit_exceeded",
                (
                    "This address has reached the limit of "
                    f"{WEB_RATE_LIMIT_COUNT} accepted searches per hour."
                ),
                status_code=429,
                headers={"Retry-After": str(retry_after)},
            )
        if use_bsi:
            return _err(
                "search_policy_violation",
                "BSI is not available on the public web service.",
                status_code=422,
                details={"field": "use_bsi"},
            )
        if ligand_provider != "zinc":
            return _err(
                "search_policy_violation",
                "The public web service supports only the ZINC provider.",
                status_code=422,
                details={"field": "ligand_provider"},
            )
        if not (use_sequence or use_nearest_k or use_domains):
            return _err(
                "search_policy_violation",
                "Select at least one protein recovery method.",
                status_code=422,
                details={"field": "methods"},
            )
        if use_nearest_k and not 1 <= nearest_k <= 15:
            return _err(
                "search_policy_violation",
                "nearest_k must be between 1 and 15.",
                status_code=422,
                details={"field": "nearest_k"},
            )
        if not known_only:
            representation_policy = web_representation_policy(search_representation)
            if representation_policy is None:
                return _err(
                    "search_policy_violation",
                    "The selected representation is not available on the public web service.",
                    status_code=422,
                    details={"field": "search_representation"},
                )
            if search_metric != representation_policy.metric:
                return _err(
                    "search_policy_violation",
                    "The selected metric is not allowed for this representation.",
                    status_code=422,
                    details={"field": "search_metric"},
                )
            effective_min = (
                representation_policy.default_threshold
                if search_threshold is None
                else search_threshold
            )
            if not representation_policy.cache_threshold_min <= effective_min <= 1.0:
                return _err(
                    "search_policy_violation",
                    (
                        f"The minimum cutoff for {representation_policy.label} must be "
                        f"between {representation_policy.cache_threshold_min} and 1.0."
                    ),
                    status_code=422,
                    details={"field": "search_threshold"},
                )
            search_threshold = effective_min
            if search_threshold_max is None:
                search_threshold_max = 1.0
        elif use_bsi:
            return _err(
                "search_policy_violation",
                "Known-only mode cannot be combined with BSI.",
                status_code=422,
            )

    if use_bsi:
        capabilities = await asyncio.to_thread(get_hardware_capabilities)
        if not capabilities.cuda_available:
            return _err(
                "gpu_required",
                (
                    "A CUDA-capable GPU is required to run BSI through the graphical interface. "
                    "Command-line BSI remains available for administrative runs."
                ),
                status_code=422,
            )

    if not known_only and not database_exists(ligand_provider):
        return _err("database_not_found", f"Ligand provider '{ligand_provider}' not found.")

    effective_representation = BSI_REPRESENTATION if use_bsi else search_representation
    if (
        not known_only
        and not representation_is_search_ready(ligand_provider, effective_representation)
    ):
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
        uploaded_bytes = await save_upload_stream(
            fasta_file,
            fasta_path,
            max_bytes=WEB_MAX_FASTA_BYTES if web_mode else MAX_UPLOAD_BYTES,
        )
    except UploadTooLargeError as exc:
        return _err("file_too_large", str(exc), status_code=413)
    if uploaded_bytes == 0:
        fasta_path.unlink(missing_ok=True)
        return _err("invalid_fasta", "FASTA file is empty or contains no valid sequences.")

    if web_mode:
        fasta_inspection = inspect_fasta_details(fasta_path)
        valid_fasta = fasta_inspection.valid
        all_queries = fasta_inspection.query_ids
    else:
        valid_fasta, all_queries = inspect_fasta(fasta_path)
    if not valid_fasta:
        fasta_path.unlink(missing_ok=True)
        return _err("invalid_fasta", "FASTA file is empty or contains no valid sequences.")
    if web_mode:
        if fasta_inspection.sequence_count > WEB_MAX_FASTA_SEQUENCES:
            fasta_path.unlink(missing_ok=True)
            return _err(
                "fasta_limit",
                f"Public searches accept at most {WEB_MAX_FASTA_SEQUENCES} FASTA records.",
                status_code=422,
                details={"field": "sequence_count", "actual": fasta_inspection.sequence_count},
            )
        if fasta_inspection.total_residues > WEB_MAX_FASTA_RESIDUES:
            fasta_path.unlink(missing_ok=True)
            return _err(
                "fasta_limit",
                f"Public searches accept at most {WEB_MAX_FASTA_RESIDUES} total residues.",
                status_code=422,
                details={"field": "total_residues", "actual": fasta_inspection.total_residues},
            )
        if fasta_inspection.duplicate_ids:
            fasta_path.unlink(missing_ok=True)
            return _err(
                "fasta_limit",
                "FASTA identifiers must be unique on the public web service.",
                status_code=422,
                details={"field": "identifiers", "duplicates": fasta_inspection.duplicate_ids[:10]},
            )
        unsafe_ids = [
            query_id
            for query_id in fasta_inspection.query_ids
            if query_id in {".", ".."}
            or "/" in query_id
            or "\\" in query_id
            or not query_id.isprintable()
        ]
        if unsafe_ids:
            fasta_path.unlink(missing_ok=True)
            return _err(
                "invalid_fasta_identifier",
                "FASTA identifiers cannot contain path separators or control characters.",
                status_code=422,
                details={"field": "identifiers", "invalid": unsafe_ids[:10]},
            )
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
        immutable_web_data=web_mode,
    )

    job = Job(
        job_id=job_id,
        job_type="search",
        status=JobStatus.queued,
        created_at=datetime.now(timezone.utc),
        output_dir=str(output_dir),
        all_queries=all_queries,
        n_queries=n_queries,
        owner_session_hash=session_hash(request) if web_mode else None,
        search_mode="known_only" if known_only else "zinc",
        input_path=str(fasta_path),
    )
    if web_mode:
        admitted = await state.try_set_exclusive_web_search(job)
        if not admitted:
            fasta_path.unlink(missing_ok=True)
            return _err(
                "server_busy",
                "The public server is currently processing another search. Please try again later.",
                status_code=503,
                headers={"Retry-After": "30"},
            )
    else:
        await state.set_job(job)
    try:
        await enqueue_job(
            job_id,
            args,
            output_dir,
            n_queries,
            timeout_seconds=WEB_SEARCH_TIMEOUT_SECONDS if web_mode else None,
        )
    except Exception:
        if web_mode:
            await asyncio.to_thread(
                cleanup_web_search_artifacts,
                job,
                remove_results=True,
            )
            await state.delete_job(job_id)
            return _err(
                "service_unavailable",
                "The public search could not be accepted. Please try again.",
                status_code=503,
            )
        raise
    if web_mode:
        await record_accepted_search(ip_hash)

    return {"job_id": job_id, "status": "queued", "output_dir": output_dir_rel}


# ─── Build database ────────────────────────────────────────────────────────────


@router.post("/build-database", status_code=201)
async def build_database(
    input_file: UploadFile = File(...),
    base_name: str = Form(...),
    id_column: str = Form(""),
    smiles_column: str = Form(""),
):
    require_resource_management()
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
        "--staging-token", job_id,
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
    require_resource_management()
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$", body.rep_name):
        return _err(
            "invalid_name",
            "rep_name must start with an alphanumeric character and contain only letters, digits, dots, underscores, and hyphens.",
        )

    if body.representation_type == "huggingface":
        if not body.model_id:
            return _err("missing_field", "model_id is required for huggingface representation.")
        capabilities = await asyncio.to_thread(get_hardware_capabilities)
        if not capabilities.cuda_available:
            return _err(
                "gpu_required",
                (
                    "A CUDA-capable GPU is required to generate HuggingFace embeddings "
                    "through the graphical interface. Command-line generation remains unrestricted."
                ),
                status_code=422,
            )

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

    job_id = str(uuid.uuid4())
    args: list[str] = [
        str(PIPELINE_ROOT / "add_new_representation.py"),
        "--output-dir", str(DATABASES_DIR),
        "--representation-type", body.representation_type,
        "--n-bits", str(body.n_bits),
        "--rep-name", body.rep_name,
        "--progress-json",
        "--staging-token", job_id,
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
async def list_jobs(request: Request, limit: int = 20, job_type: Optional[str] = None):
    all_jobs = state.get_all_jobs()
    if is_web_mode():
        owner = session_hash(request)
        all_jobs = [job for job in all_jobs if job.owner_session_hash == owner]
    if job_type:
        all_jobs = [j for j in all_jobs if j.job_type == job_type]
    return {"jobs": [_job_payload(job) for job in all_jobs[:limit]]}


@router.get("/{job_id}")
async def get_job(request: Request, job_id: str):
    job = require_job_access(request, await state.get_job(job_id))
    return _job_payload(job)


@router.delete("/{job_id}", status_code=204)
async def cancel_job(request: Request, job_id: str):
    job = require_job_access(request, await state.get_job(job_id))
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
    cancelled_job = await state.update_job_if_status(
        job_id,
        _ACTIVE_JOB_STATUSES,
        status=JobStatus.cancelled,
        finished_at=now,
        elapsed_seconds=(
            (now - job.started_at).total_seconds() if job.started_at else None
        ),
        error=message,
        failure=JobFailure(label="Cancelled", message=message),
    )
    if cancelled_job is None:
        return
    await terminate_job_process(job_id)
    if cancelled_job.job_type in RESOURCE_JOB_TYPES:
        await asyncio.to_thread(cleanup_resource_job_artifacts, job_id)
    if cancelled_job.job_type == "search":
        await asyncio.to_thread(
            cleanup_search_artifacts,
            cancelled_job,
            remove_results=True,
        )
