from __future__ import annotations

import shutil
from pathlib import Path

from core.config import RESULTS_DIR, TEMP_RESULTS_DIR, UPLOADS_DIR
from models.job import Job


def _contained(path: Path, root: Path) -> Path | None:
    try:
        resolved = path.resolve()
        resolved.relative_to(root.resolve())
        return resolved
    except (OSError, ValueError):
        return None


def _remove_path(path: Path, root: Path) -> None:
    target = _contained(path, root)
    if target is None or not target.exists():
        return
    if target.is_dir() and not target.is_symlink():
        shutil.rmtree(target)
    else:
        target.unlink(missing_ok=True)


def cleanup_search_artifacts(job: Job, *, remove_results: bool) -> None:
    if job.job_type != "search":
        return
    if job.input_path:
        input_path = Path(job.input_path)
        _remove_path(input_path, UPLOADS_DIR)
        _remove_path(TEMP_RESULTS_DIR / input_path.stem, TEMP_RESULTS_DIR)
    if remove_results and job.output_dir:
        _remove_path(Path(job.output_dir), RESULTS_DIR)


def cleanup_web_search_artifacts(job: Job, *, remove_results: bool) -> None:
    if job.owner_session_hash is None:
        return
    cleanup_search_artifacts(job, remove_results=remove_results)
