from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from core.config import DATABASES_DIR, PIPELINE_ROOT


SETUP_SCRIPT = PIPELINE_ROOT / "prepare_ligq_2_data.py"
HF_REPO_ID = "gschottlender/LigQ_2"
HF_REVISION = "main"
FALLBACK_TOTAL_REQUIRED_BYTES = 6_945_959_057
FALLBACK_TOTAL_FILE_COUNT = 63

BSI_FAMILIES = (
    "PF00001", "PF00002", "PF00026", "PF00067", "PF00069", "PF00089",
    "PF00104", "PF00112", "PF00135", "PF00194", "PF00209", "PF00233",
    "PF00413", "PF00520", "PF00850", "PF01094", "PF07714",
)

DEFAULT_PREDICTED_CACHE_NAMESPACE = (
    "results_databases/predicted_bindings/zinc/"
    "search_representation=morgan_1024_r2__search_metric=tanimoto__cache_threshold_min=0.4"
)
DEFAULT_PREDICTED_CACHE_FILES = (
    "manifest.json",
    "predicted_binding_data.parquet",
    "predicted_binding_progress.json",
    "cached_proteins.json",
    "predicted_binding_rowgroup_index.json",
)

# Match the runtime readiness boundary with concrete leaf files. This avoids
# treating a partially created directory as ready while a setup job is being
# resumed. The default predicted cache prevents an expensive first search, and
# BSI is included because it is a first-class GUI search mode.
FAST_READY_PATHS = (
    "sequences/target_sequences.fasta",
    "results_databases/known_binding_data.parquet",
    "results_databases/protein_domains.parquet",
    "compound_data/pdb_chembl/ligands.parquet",
    "compound_data/pdb_chembl/reps/morgan_1024_r2.dat",
    "compound_data/pdb_chembl/reps/morgan_1024_r2.meta.json",
    "complementary_databases/blast/target_sequences.pdb",
    "complementary_databases/blast/target_sequences.phr",
    "complementary_databases/blast/target_sequences.pin",
    "complementary_databases/blast/target_sequences.pjs",
    "complementary_databases/blast/target_sequences.pot",
    "complementary_databases/blast/target_sequences.psq",
    "complementary_databases/blast/target_sequences.ptf",
    "complementary_databases/blast/target_sequences.pto",
    "complementary_databases/pfam/Pfam-A.hmm",
    "complementary_databases/pfam/Pfam-A.hmm.h3f",
    "complementary_databases/pfam/Pfam-A.hmm.h3i",
    "complementary_databases/pfam/Pfam-A.hmm.h3m",
    "complementary_databases/pfam/Pfam-A.hmm.h3p",
    "compound_data/zinc/ligands.parquet",
    "compound_data/zinc/reps/morgan_1024_r2.dat",
    "compound_data/zinc/reps/morgan_1024_r2.meta.json",
    *(
        f"{DEFAULT_PREDICTED_CACHE_NAMESPACE}/{filename}"
        for filename in DEFAULT_PREDICTED_CACHE_FILES
    ),
    "bsi_models/mpg_1024/manifest.json",
    "bsi_models/mpg_1024/summary.csv",
    *(f"bsi_models/mpg_1024/{family}/model.pth" for family in BSI_FAMILIES),
    *(f"bsi_models/mpg_1024/{family}/params.json" for family in BSI_FAMILIES),
)


def is_default_setup_ready(data_dir: Path = DATABASES_DIR) -> bool:
    return all((data_dir / relative_path).exists() for relative_path in FAST_READY_PATHS)


def _available_bytes(data_dir: Path = DATABASES_DIR) -> int:
    probe = data_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def fallback_setup_status(*, active: bool = False) -> dict[str, Any]:
    installed = is_default_setup_ready()
    ready = installed and not active
    required_bytes = 0 if installed else FALLBACK_TOTAL_REQUIRED_BYTES
    available = _available_bytes()
    return {
        "ready": ready,
        "state": "ready" if ready else ("downloading" if active else "required"),
        "repo_id": HF_REPO_ID,
        "revision": HF_REVISION,
        "required_download_bytes": required_bytes,
        "total_required_bytes": FALLBACK_TOTAL_REQUIRED_BYTES,
        "available_bytes": available,
        "enough_space": required_bytes <= available,
        "required_file_count": 0 if installed else FALLBACK_TOTAL_FILE_COUNT,
        "total_file_count": FALLBACK_TOTAL_FILE_COUNT,
        "missing_paths": [],
        "size_source": "repository_snapshot",
        "metadata_error": None,
    }


async def inspect_setup_status(*, active: bool = False) -> dict[str, Any]:
    if is_default_setup_ready() or active:
        return fallback_setup_status(active=active)

    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(SETUP_SCRIPT),
        "--data-dir",
        str(DATABASES_DIR),
        "--status-json",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(PIPELINE_ROOT),
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        status = fallback_setup_status()
        status["metadata_error"] = "Timed out while reading Hugging Face dataset metadata."
        return status

    if process.returncode != 0:
        status = fallback_setup_status()
        detail = stderr.decode("utf-8", errors="replace").strip()
        status["metadata_error"] = detail or "Could not inspect Hugging Face dataset metadata."
        return status

    try:
        lines = [line for line in stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
        return json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError, TypeError):
        status = fallback_setup_status()
        status["metadata_error"] = "The setup metadata response was invalid."
        return status


def setup_job_args() -> list[str]:
    return [
        str(SETUP_SCRIPT),
        "--data-dir",
        str(DATABASES_DIR),
        "--repo-id",
        HF_REPO_ID,
        "--revision",
        HF_REVISION,
        "--progress-json",
    ]
