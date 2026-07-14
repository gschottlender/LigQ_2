from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "gui" / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.job import Job, JobProgress, JobStatus
from core import state
from services import fs_inspector
from services.fs_inspector import (
    get_metric_from_manifest,
    load_search_threshold_defaults,
    representation_files_complete,
)
from services.job_runner import _build_job_failure, _parse_progress_event, _subprocess_env, run_job
from services.tsv_reader import _parse_list_value, read_tsv_paginated

from progress_reporting import PROGRESS_PREFIX, ProgressEmitter


EXPECTED_THRESHOLDS = {
    "chemberta_zinc_base_768": 0.936140,
    "rdkit_1024": 0.930324,
    "maccs": 0.831169,
    "ap_rdkit": 0.767087,
    "morgan_feature_1024_r2": 0.509451,
    "topological_torsion_rdkit_1024": 0.502932,
    "morgan_1024_r2": 0.415094,
}


def test_progress_emitter_produces_backend_job_progress(capsys) -> None:
    emitter = ProgressEmitter(enabled=True)
    emitter.emit(
        step="building_fingerprints",
        label="Computing Morgan fingerprints",
        step_index=3,
        step_count=4,
        percent=52,
        current=25,
        total=100,
        unit="compounds",
        context="test_database",
        eta_seconds=15,
    )

    line = capsys.readouterr().out.strip()
    assert line.startswith(PROGRESS_PREFIX)
    progress = _parse_progress_event(line)
    assert progress is not None
    assert progress.step == "building_fingerprints"
    assert progress.label == "Computing Morgan fingerprints"
    assert progress.step_index == 3
    assert progress.step_count == 4
    assert progress.percent == 52
    assert progress.current == 25
    assert progress.total == 100
    assert progress.unit == "compounds"
    assert progress.context == "test_database"
    assert progress.eta_seconds == 15


def test_progress_parser_rejects_non_events_and_invalid_json() -> None:
    assert _parse_progress_event("regular pipeline output") is None
    assert _parse_progress_event(f"{PROGRESS_PREFIX}not-json") is None


def test_gui_threshold_defaults_match_pipeline_defaults_file() -> None:
    assert load_search_threshold_defaults() == EXPECTED_THRESHOLDS
    assert json.loads((ROOT / "search_threshold_defaults.json").read_text()) == EXPECTED_THRESHOLDS


def test_metric_detection_uses_representation_metadata(tmp_path: Path) -> None:
    rep_path = tmp_path / "representation.dat"
    rep_path.write_bytes(b"representation")
    meta_path = rep_path.with_suffix(".meta.json")

    for fingerprint_type in (
        "morgan",
        "morgan_feature",
        "ap",
        "topological_torsion",
        "rdkit",
        "maccs",
    ):
        meta_path.write_text(json.dumps({"fingerprint_type": fingerprint_type}))
        assert get_metric_from_manifest(rep_path) == "tanimoto"

    meta_path.write_text(json.dumps({"packed_bits": True, "dtype": "uint8"}))
    assert get_metric_from_manifest(rep_path) == "tanimoto"

    meta_path.write_text(json.dumps({"model_id": "seyonec/ChemBERTa-zinc-base-v1"}))
    assert get_metric_from_manifest(rep_path) == "cosine"

    meta_path.write_text(json.dumps({"search_metric": "tanimoto", "model_id": "ignored"}))
    assert get_metric_from_manifest(rep_path) == "tanimoto"


def _write_representation(root: Path, name: str, *, data: bool = True, meta: bool = True) -> None:
    reps_dir = root / "reps"
    reps_dir.mkdir(parents=True, exist_ok=True)
    if data:
        (reps_dir / f"{name}.dat").write_bytes(b"representation")
    if meta:
        (reps_dir / f"{name}.meta.json").write_text(
            json.dumps({"fingerprint_type": "morgan"})
        )


def test_representation_files_complete_requires_data_and_metadata(tmp_path: Path) -> None:
    root = tmp_path / "database"
    _write_representation(root, "complete")
    _write_representation(root, "missing_meta", meta=False)
    _write_representation(root, "missing_data", data=False)

    assert representation_files_complete(root, "complete")
    assert not representation_files_complete(root, "missing_meta")
    assert not representation_files_complete(root, "missing_data")


def test_list_representations_only_returns_representations_ready_in_both_databases(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compound_data = tmp_path / "compound_data"
    target = compound_data / "custom"
    local = compound_data / "pdb_chembl"

    _write_representation(target, "ready")
    _write_representation(local, "ready")
    _write_representation(target, "target_only")
    _write_representation(target, "missing_target_meta", meta=False)
    _write_representation(local, "missing_target_meta")
    _write_representation(target, "missing_local_meta")
    _write_representation(local, "missing_local_meta", meta=False)

    monkeypatch.setattr(fs_inspector, "COMPOUND_DATA_DIR", compound_data)

    assert [rep["name"] for rep in fs_inspector.list_representations("custom")] == ["ready"]


def test_job_failure_identifies_the_active_step() -> None:
    job = Job(
        job_id="test-job",
        job_type="add_representation",
        status=JobStatus.running,
        created_at="2026-07-14T12:00:00Z",
        progress_message="Computing fingerprints",
        progress=JobProgress(
            step="building_custom",
            label="Computing fingerprints for custom",
            step_index=2,
            step_count=4,
            percent=35,
        ),
    )

    failure, error = _build_job_failure(job, "RuntimeError: representation build stopped")

    assert failure.step == "building_custom"
    assert failure.step_index == 2
    assert failure.step_count == 4
    assert failure.message == "RuntimeError: representation build stopped"
    assert error.startswith("Failed at step 2/4: Computing fingerprints for custom.")


def test_subprocess_env_prefers_conda_libraries(tmp_path: Path, monkeypatch) -> None:
    conda_prefix = tmp_path / "conda-env"
    conda_lib = conda_prefix / "lib"
    conda_lib.mkdir(parents=True)
    monkeypatch.setenv("CONDA_PREFIX", str(conda_prefix))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")

    env = _subprocess_env()

    assert env["LD_LIBRARY_PATH"] == f"{conda_lib}:/existing/lib"


def test_failed_process_captures_stderr_for_the_active_step() -> None:
    async def scenario() -> None:
        job_id = "failure-capture-test"
        await state.set_job(
            Job(
                job_id=job_id,
                job_type="add_representation",
                status=JobStatus.queued,
                created_at=datetime.now(timezone.utc),
            )
        )
        payload = json.dumps(
            {
                "step": "building_pdb_chembl",
                "label": "Computing fingerprints for pdb_chembl",
                "step_index": 3,
                "step_count": 4,
                "percent": 67,
            }
        )
        code = (
            f'print("LIGQ_PROGRESS " + {payload!r}, flush=True); '
            'raise RuntimeError("metadata write failed")'
        )

        await run_job(job_id, ["-c", code])
        job = await state.get_job(job_id)

        assert job is not None
        assert job.status == JobStatus.failed
        assert job.failure is not None
        assert job.failure.step == "building_pdb_chembl"
        assert job.failure.step_index == 3
        assert job.failure.message == "RuntimeError: metadata write failed"
        await state.delete_job(job_id)

    asyncio.run(scenario())


def test_historical_list_formats_are_normalized() -> None:
    expected = ["PF00089", "PF14670"]
    assert _parse_list_value("['PF00089' 'PF14670']") == expected
    assert _parse_list_value("[PF00089 PF14670]") == expected
    assert _parse_list_value("['PF00089', 'PF14670']") == expected
    assert _parse_list_value("PF00089;PF14670") == expected
    assert _parse_list_value("PF00089,PF14670") == expected


def test_paginated_results_return_binding_sites_as_lists(tmp_path: Path) -> None:
    path = tmp_path / "known_ligands.tsv"
    pd.DataFrame(
        [
            {
                "chem_comp_id": "LIG",
                "binding_sites": "['PF00089' 'PF14670']",
                "pdb_ids": "1ABC;2DEF",
            }
        ]
    ).to_csv(path, sep="\t", index=False)

    result = read_tsv_paginated(path)
    assert result["data"][0]["binding_sites"] == ["PF00089", "PF14670"]
    assert result["data"][0]["pdb_ids"] == ["1ABC", "2DEF"]
