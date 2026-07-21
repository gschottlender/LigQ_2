from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest


BACKEND_ROOT = Path(__file__).resolve().parent
PIPELINE_ROOT = BACKEND_ROOT.parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))

import build_compound_database as database_builder
from core import state
from models.job import AddRepresentationRequest, Job, JobStatus
from routers import jobs as jobs_router
from services import resource_artifacts


def _configure_roots(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    compound_data = tmp_path / "databases" / "compound_data"
    uploads = tmp_path / "uploads"
    compound_data.mkdir(parents=True)
    uploads.mkdir(parents=True)
    monkeypatch.setattr(resource_artifacts, "COMPOUND_DATA_DIR", compound_data)
    monkeypatch.setattr(resource_artifacts, "UPLOADS_DIR", uploads)
    return compound_data, uploads


def test_cancel_cleanup_removes_partial_representation_but_keeps_completed_phase(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compound_data, _ = _configure_roots(tmp_path, monkeypatch)
    job_id = "job-123"
    reps = compound_data / "custom" / "reps"
    reps.mkdir(parents=True)

    # This phase completed before cancellation and must remain reusable.
    (reps / "complete.dat").write_bytes(b"complete")
    (reps / "complete.meta.json").write_text("{}")

    (reps / f".partial.dat.partial.{job_id}").write_bytes(b"partial")
    (reps / f".partial.meta.json.partial.{job_id}").write_text("{}")

    # Simulate cancellation between the two atomic publication renames.
    (reps / "publishing.dat").write_bytes(b"not-committed")
    publishing_marker = reps / f".publishing.publishing.{job_id}"
    publishing_marker.write_text(
        json.dumps({"data": "publishing.dat", "meta": "publishing.meta.json"})
    )

    resource_artifacts.cleanup_resource_job_artifacts(job_id)

    assert (reps / "complete.dat").is_file()
    assert (reps / "complete.meta.json").is_file()
    assert not (reps / f".partial.dat.partial.{job_id}").exists()
    assert not (reps / f".partial.meta.json.partial.{job_id}").exists()
    assert not (reps / "publishing.dat").exists()
    assert not publishing_marker.exists()


def test_cancel_cleanup_removes_database_staging_promoted_marker_and_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compound_data, uploads = _configure_roots(tmp_path, monkeypatch)
    job_id = "job-456"

    staging = compound_data / f".vendor.building.{job_id}"
    staging.mkdir()
    (staging / "partial.bin").write_bytes(b"partial")

    promoted = compound_data / "promoted_vendor"
    promoted.mkdir()
    (promoted / ".ligq_build_job").write_text(job_id)
    (promoted / "ligands.parquet").write_bytes(b"uncommitted")

    unrelated = compound_data / "existing"
    unrelated.mkdir()
    (unrelated / "ligands.parquet").write_bytes(b"keep")
    upload = uploads / f"{job_id}.csv"
    upload.write_bytes(b"source")

    resource_artifacts.cleanup_resource_job_artifacts(job_id)

    assert not staging.exists()
    assert not promoted.exists()
    assert (unrelated / "ligands.parquet").is_file()
    assert not upload.exists()


def test_success_finalization_commits_database_and_removes_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compound_data, uploads = _configure_roots(tmp_path, monkeypatch)
    job_id = "job-789"
    database = compound_data / "vendor"
    database.mkdir()
    marker = database / ".ligq_build_job"
    marker.write_text(job_id)
    (database / "ligands.parquet").write_bytes(b"complete")
    upload = uploads / f"{job_id}.smi"
    upload.write_bytes(b"source")

    resource_artifacts.finalize_resource_job_artifacts(job_id)

    assert database.is_dir()
    assert (database / "ligands.parquet").is_file()
    assert not marker.exists()
    assert not upload.exists()


def test_database_builder_publishes_staging_directory_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_file = tmp_path / "vendor.csv"
    input_file.write_text("compound_id,smiles\nA,CCO\n")

    def fake_ligand_index(*, final_ligs, root, progress_callback=None):
        root.mkdir(parents=True, exist_ok=True)
        output = root / "ligands.parquet"
        output.write_bytes(f"rows={len(final_ligs)}".encode())
        if progress_callback:
            progress_callback(len(final_ligs), len(final_ligs))
        return output

    def fake_representation(*, root, name, progress_callback=None, **_kwargs):
        reps = Path(root) / "reps"
        reps.mkdir(parents=True, exist_ok=True)
        (reps / f"{name}.dat").write_bytes(b"complete")
        (reps / f"{name}.meta.json").write_text("{}")
        if progress_callback:
            progress_callback(1, 1)

    monkeypatch.setattr(database_builder, "build_ligand_index", fake_ligand_index)
    monkeypatch.setattr(database_builder, "build_morgan_representation", fake_representation)

    final_root = database_builder.build_compound_database(
        input_file=input_file,
        output_dir=tmp_path / "databases",
        base_name="vendor",
        id_column="compound_id",
        smiles_column="smiles",
        staging_token="atomic-job",
    )

    assert final_root == tmp_path / "databases" / "compound_data" / "vendor"
    assert (final_root / "ligands.parquet").is_file()
    assert (final_root / "reps" / "morgan_1024_r2.dat").is_file()
    assert (final_root / ".ligq_build_job").read_text() == "atomic-job"
    assert not (final_root.parent / ".vendor.building.atomic-job").exists()


def test_cleanup_rejects_unsafe_job_identifier(tmp_path: Path, monkeypatch) -> None:
    _configure_roots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Invalid resource job identifier"):
        resource_artifacts.cleanup_resource_job_artifacts("../../outside")


def test_add_representation_rejects_unsafe_representation_name() -> None:
    response = asyncio.run(
        jobs_router.add_representation(
            AddRepresentationRequest(
                base_name="zinc",
                representation_type="rdkit",
                rep_name="../../outside",
                n_bits=1024,
                rdkit_fp_kind="morgan",
            )
        )
    )

    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_name"


def test_cancel_endpoint_waits_for_termination_and_cleanup(monkeypatch) -> None:
    async def scenario() -> None:
        job_id = "cancel-endpoint-job"
        calls: list[str] = []
        state.jobs.clear()
        state._initialized = False
        await state.set_job(
            Job(
                job_id=job_id,
                job_type="add_representation",
                status=JobStatus.running,
                created_at=datetime.now(timezone.utc),
                started_at=datetime.now(timezone.utc),
            )
        )

        async def fake_terminate(received_job_id: str) -> None:
            assert received_job_id == job_id
            calls.append("terminated")

        def fake_cleanup(received_job_id: str) -> list[Path]:
            assert received_job_id == job_id
            calls.append("cleaned")
            return []

        monkeypatch.setattr(jobs_router, "terminate_job_process", fake_terminate)
        monkeypatch.setattr(jobs_router, "cleanup_resource_job_artifacts", fake_cleanup)

        await jobs_router.cancel_job(job_id)
        cancelled = await state.get_job(job_id)

        assert calls == ["terminated", "cleaned"]
        assert cancelled is not None
        assert cancelled.status == JobStatus.cancelled
        await state.delete_job(job_id)

    try:
        asyncio.run(scenario())
    finally:
        state.jobs.clear()
        state.processes.clear()
        state._initialized = False
