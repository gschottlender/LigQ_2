from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import prepare_ligq_2_data as setup
from progress_reporting import ProgressEmitter


class FakeApi:
    def __init__(self, entries=None, error: Exception | None = None):
        self.entries = entries or []
        self.error = error

    def list_repo_tree(self, *args, **kwargs):
        if self.error:
            raise self.error
        return self.entries


def test_required_repo_files_are_filtered_and_sized() -> None:
    cache_manifest = f"{setup.DEFAULT_PREDICTED_CACHE_NAMESPACE}/manifest.json"
    api = FakeApi(
        [
            SimpleNamespace(path="sequences/target_sequences.fasta", size=120),
            SimpleNamespace(path="compound_data/zinc/ligands.parquet", size=350),
            SimpleNamespace(path=cache_manifest, size=249),
            SimpleNamespace(path="unrelated/file.bin", size=999),
            SimpleNamespace(path="sequences", size=None),
        ]
    )

    files = setup.list_required_repo_files(api=api)

    assert [(item.path, item.size) for item in files] == [
        ("compound_data/zinc/ligands.parquet", 350),
        (cache_manifest, 249),
        ("sequences/target_sequences.fasta", 120),
    ]


def test_default_predicted_cache_is_part_of_gui_setup() -> None:
    expected = {
        f"{setup.DEFAULT_PREDICTED_CACHE_NAMESPACE}/{filename}"
        for filename in setup.DEFAULT_PREDICTED_CACHE_FILES
    }

    assert setup.DEFAULT_PREDICTED_CACHE_NAMESPACE.endswith("cache_threshold_min=0.4")
    assert expected <= set(setup.REQUIRED_DATA_PATHS)
    assert len(expected) == 5


def test_status_counts_only_missing_files(tmp_path: Path) -> None:
    existing = tmp_path / "sequences" / "target_sequences.fasta"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"local data")
    api = FakeApi(
        [
            SimpleNamespace(path="sequences/target_sequences.fasta", size=120),
            SimpleNamespace(path="compound_data/zinc/ligands.parquet", size=350),
        ]
    )

    status = setup.inspect_default_data(tmp_path, api=api)

    assert status["ready"] is False
    assert status["required_download_bytes"] == 350
    assert status["total_required_bytes"] == 470
    assert status["required_file_count"] == 1
    assert status["missing_paths"] == ["compound_data/zinc/ligands.parquet"]
    assert status["size_source"] == "huggingface"


def test_status_uses_repository_snapshot_when_metadata_is_unavailable(tmp_path: Path) -> None:
    status = setup.inspect_default_data(
        tmp_path,
        api=FakeApi(error=RuntimeError("offline")),
    )

    assert status["ready"] is False
    assert status["required_download_bytes"] == setup.FALLBACK_TOTAL_REQUIRED_BYTES
    assert status["size_source"] == "repository_snapshot"
    assert status["metadata_error"] == "offline"


def test_prepare_downloads_only_missing_files(tmp_path: Path, monkeypatch, capsys) -> None:
    files = [
        setup.RequiredRepoFile("sequences/target_sequences.fasta", 120),
        setup.RequiredRepoFile("compound_data/zinc/ligands.parquet", 350),
    ]
    existing = tmp_path / files[0].path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"already installed")
    downloaded: dict[str, object] = {}

    monkeypatch.setattr(setup, "list_required_repo_files", lambda **kwargs: files)

    def fake_hf_hub_download(**kwargs):
        downloaded[kwargs["filename"]] = kwargs
        target = Path(kwargs["local_dir"]) / kwargs["filename"]
        target.parent.mkdir(parents=True)
        target.write_bytes(b"downloaded")
        return str(tmp_path)

    monkeypatch.setattr(setup, "hf_hub_download", fake_hf_hub_download)

    setup.prepare_default_data(tmp_path, progress=ProgressEmitter(enabled=True))

    assert set(downloaded) == {files[1].path}
    assert downloaded[files[1].path]["local_dir"] == tmp_path
    output = capsys.readouterr().out
    assert '"step":"downloading"' in output
    assert '"label":"Default data ready"' in output
