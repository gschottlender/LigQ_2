from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "gui" / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from services.fs_inspector import load_search_threshold_defaults
from services.job_runner import _parse_progress_event
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
