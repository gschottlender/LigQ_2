from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from core.config import COMPOUND_DATA_DIR, UPLOADS_DIR


RESOURCE_JOB_TYPES = {"build_database", "add_representation"}
_JOB_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_BUILD_JOB_MARKER = ".ligq_build_job"


def _validate_job_id(job_id: str) -> None:
    if not _JOB_TOKEN_RE.fullmatch(job_id):
        raise ValueError("Invalid resource job identifier.")


def _representation_dirs() -> list[Path]:
    if not COMPOUND_DATA_DIR.exists():
        return []
    return [root / "reps" for root in COMPOUND_DATA_DIR.iterdir() if root.is_dir()]


def _remove_publishing_marker(marker: Path) -> list[Path]:
    removed: list[Path] = []
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {}

    for field in ("data", "meta"):
        value = payload.get(field)
        if not isinstance(value, str) or Path(value).name != value:
            continue
        target = marker.parent / value
        if target.is_file():
            target.unlink()
            removed.append(target)
    marker.unlink(missing_ok=True)
    removed.append(marker)
    return removed


def cleanup_resource_job_artifacts(job_id: str) -> list[Path]:
    """Remove only incomplete artifacts owned by one GUI resource job."""
    _validate_job_id(job_id)
    removed: list[Path] = []

    for reps_dir in _representation_dirs():
        if not reps_dir.exists():
            continue
        for partial in reps_dir.glob(f"*.partial.{job_id}"):
            if partial.is_file():
                partial.unlink()
                removed.append(partial)
        for marker in reps_dir.glob(f"*.publishing.{job_id}"):
            if marker.is_file():
                removed.extend(_remove_publishing_marker(marker))

    if COMPOUND_DATA_DIR.exists():
        for staging_root in COMPOUND_DATA_DIR.glob(f"*.building.{job_id}"):
            if staging_root.is_dir():
                shutil.rmtree(staging_root)
                removed.append(staging_root)

        # A database may have been atomically promoted just before cancellation.
        # Its marker proves that it still belongs to this uncommitted job.
        for root in list(COMPOUND_DATA_DIR.iterdir()):
            if not root.is_dir():
                continue
            marker = root / _BUILD_JOB_MARKER
            try:
                marker_job_id = marker.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if marker_job_id == job_id:
                shutil.rmtree(root)
                removed.append(root)

    if UPLOADS_DIR.exists():
        for upload in UPLOADS_DIR.glob(f"{job_id}.*"):
            if upload.is_file():
                upload.unlink()
                removed.append(upload)

    return removed


def finalize_resource_job_artifacts(job_id: str) -> list[Path]:
    """Commit successful resource outputs and remove job-scoped input files."""
    _validate_job_id(job_id)
    removed: list[Path] = []

    if COMPOUND_DATA_DIR.exists():
        for root in COMPOUND_DATA_DIR.iterdir():
            if not root.is_dir():
                continue
            marker = root / _BUILD_JOB_MARKER
            try:
                marker_job_id = marker.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if marker_job_id == job_id:
                marker.unlink()
                removed.append(marker)

    # Successful representation jobs should leave no partials. Removing an
    # exact job-token match is safe and keeps finalization idempotent.
    for reps_dir in _representation_dirs():
        if not reps_dir.exists():
            continue
        for partial in reps_dir.glob(f"*.partial.{job_id}"):
            if partial.is_file():
                partial.unlink()
                removed.append(partial)
        for marker in reps_dir.glob(f"*.publishing.{job_id}"):
            if marker.is_file():
                removed.extend(_remove_publishing_marker(marker))

    if UPLOADS_DIR.exists():
        for upload in UPLOADS_DIR.glob(f"{job_id}.*"):
            if upload.is_file():
                upload.unlink()
                removed.append(upload)

    return removed
