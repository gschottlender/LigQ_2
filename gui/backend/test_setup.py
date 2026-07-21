from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = ROOT / "gui" / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import prepare_ligq_2_data
from progress_reporting import PROGRESS_PREFIX, ProgressEmitter
from services.job_runner import _parse_progress_event
from services.setup_service import FAST_READY_PATHS, is_default_setup_ready, setup_job_args


def test_default_setup_readiness_requires_every_sentinel(tmp_path: Path) -> None:
    assert not is_default_setup_ready(tmp_path)

    for relative_path in FAST_READY_PATHS:
        path = tmp_path / relative_path
        if path.suffix:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        else:
            path.mkdir(parents=True, exist_ok=True)

    assert is_default_setup_ready(tmp_path)

    (tmp_path / "compound_data/zinc/ligands.parquet").unlink()
    assert not is_default_setup_ready(tmp_path)


def test_setup_job_uses_dedicated_script_and_structured_progress() -> None:
    args = setup_job_args()

    assert args[0].endswith("prepare_ligq_2_data.py")
    assert "--data-dir" in args
    assert "--progress-json" in args


def test_backend_readiness_manifest_matches_setup_script() -> None:
    assert set(FAST_READY_PATHS) == set(prepare_ligq_2_data.REQUIRED_DATA_PATHS)


class SetupDownloadTrackerTests(unittest.TestCase):
    def test_reports_bytes_and_completed_files(self) -> None:
        files = [
            prepare_ligq_2_data.RequiredRepoFile("already-present.bin", 100),
            prepare_ligq_2_data.RequiredRepoFile("downloading.bin", 300),
        ]
        tracker = prepare_ligq_2_data.SetupDownloadTracker(
            progress=ProgressEmitter(enabled=True),
            files=files,
            missing=[files[1]],
            context="0.00 GB from test/repo",
            emit_interval_seconds=0,
        )
        tracker.set_fallback_tqdm(lambda *args, **kwargs: None)

        output = io.StringIO()
        with redirect_stdout(output):
            tracker.emit(force=True)
            tracker.begin_file(files[1])
            with tracker.tqdm(total=300, initial=0) as byte_progress:
                byte_progress.update(120)
            tracker.end_file()
            tracker.finish_file(files[1])

        lines = [
            line for line in output.getvalue().splitlines()
            if line.startswith(PROGRESS_PREFIX)
        ]
        events = [json.loads(line[len(PROGRESS_PREFIX):]) for line in lines]
        parsed = [_parse_progress_event(line) for line in lines]

        self.assertEqual(events[0]["downloaded_bytes"], 100)
        self.assertEqual(events[0]["completed_files"], 1)
        self.assertTrue(any(
            event["downloaded_bytes"] == 220 and event["completed_files"] == 1
            for event in events
        ))
        self.assertEqual(events[-1]["downloaded_bytes"], 400)
        self.assertEqual(events[-1]["download_total_bytes"], 400)
        self.assertEqual(events[-1]["completed_files"], 2)
        self.assertEqual(events[-1]["total_files"], 2)
        self.assertTrue(all(event is not None for event in parsed))
        self.assertEqual(parsed[-1].downloaded_bytes, 400)
        self.assertEqual(parsed[-1].total_files, 2)

    def test_download_helper_tracks_hugging_face_byte_callbacks(self) -> None:
        files = [
            prepare_ligq_2_data.RequiredRepoFile("first.bin", 200),
            prepare_ligq_2_data.RequiredRepoFile("nested/second.bin", 300),
        ]
        sizes = {item.path: item.size for item in files}

        def fake_hf_download(*, filename: str, local_dir: Path, **kwargs) -> str:
            size = sizes[filename]
            with prepare_ligq_2_data.hf_file_download.tqdm(
                total=size,
                initial=0,
                desc=filename,
            ) as byte_progress:
                byte_progress.update(size // 2)
                byte_progress.update(size - size // 2)
            destination = Path(local_dir) / filename
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"downloaded")
            return str(destination)

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            prepare_ligq_2_data,
            "hf_hub_download",
            side_effect=fake_hf_download,
        ), redirect_stdout(output):
            prepare_ligq_2_data._download_required_files(
                data_dir=Path(temp_dir),
                files=files,
                missing=files,
                progress=ProgressEmitter(enabled=True),
                repo_id="test/repo",
                revision="main",
                max_workers=1,
            )
            self.assertTrue((Path(temp_dir) / "first.bin").is_file())
            self.assertTrue((Path(temp_dir) / "nested/second.bin").is_file())

        events = [
            json.loads(line[len(PROGRESS_PREFIX):])
            for line in output.getvalue().splitlines()
            if line.startswith(PROGRESS_PREFIX)
        ]
        self.assertTrue(any(event["downloaded_bytes"] > 0 for event in events))
        self.assertEqual(events[-1]["downloaded_bytes"], 500)
        self.assertEqual(events[-1]["download_total_bytes"], 500)
        self.assertEqual(events[-1]["completed_files"], 2)
        self.assertEqual(events[-1]["total_files"], 2)
