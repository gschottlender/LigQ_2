import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_ligq_2 import (
    DEFAULT_CACHE_NAMESPACE,
    HF_REQUIRED_RELATIVE_PATHS,
    HF_CORE_REQUIRED_RELATIVE_PATHS,
    HF_OPTIONAL_CACHE_PATH_GROUPS,
    LEGACY_DEFAULT_CACHE_NAMESPACE,
    _missing_required_base_paths,
    ensure_base_data_from_hf,
)


def _write_placeholder(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("placeholder", encoding="utf-8")


def _materialize_required_layout(root: Path) -> None:
    for rel_path in HF_REQUIRED_RELATIVE_PATHS:
        path = root / rel_path
        if path.suffix:
            _write_placeholder(path)
        else:
            path.mkdir(parents=True, exist_ok=True)


def _materialize_optional_cache_group(root: Path, cache_namespace: str) -> None:
    for suffix in ("manifest.json", "predicted_binding_data.parquet", "predicted_binding_progress.json"):
        _write_placeholder(root / "results_databases" / cache_namespace / suffix)


class TestRunLigQHFLayout(unittest.TestCase):
    def test_missing_required_base_paths_empty_data_dir(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            missing = _missing_required_base_paths(data_dir)
            self.assertEqual(len(missing), len(HF_REQUIRED_RELATIVE_PATHS))

    def test_ensure_base_data_from_hf_skips_when_default_ready(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            _materialize_required_layout(data_dir)
            _materialize_optional_cache_group(data_dir, DEFAULT_CACHE_NAMESPACE)

            with patch("run_ligq_2.snapshot_download") as mock_snapshot:
                ensure_base_data_from_hf(data_dir=data_dir, repo_id="dummy/repo")
                mock_snapshot.assert_not_called()

    def test_missing_required_base_paths_for_custom_provider_excludes_zinc_assets(self):
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            missing = _missing_required_base_paths(data_dir, provider_name="vendor")
            self.assertEqual(len(missing), len(HF_CORE_REQUIRED_RELATIVE_PATHS))

    def test_ensure_base_data_from_hf_copies_required_paths_only(self):
        with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as dst_td:
            src_root = Path(src_td)
            data_dir = Path(dst_td)
            _materialize_required_layout(src_root)
            _materialize_optional_cache_group(src_root, DEFAULT_CACHE_NAMESPACE)

            extra_file = src_root / "results_databases" / "extra_unused_file.txt"
            _write_placeholder(extra_file)

            with patch("run_ligq_2.snapshot_download", return_value=str(src_root)):
                ensure_base_data_from_hf(data_dir=data_dir, repo_id="dummy/repo")

            self.assertEqual(_missing_required_base_paths(data_dir), [])
            self.assertTrue((data_dir / "results_databases" / DEFAULT_CACHE_NAMESPACE / "predicted_binding_data.parquet").exists())
            self.assertFalse((data_dir / "results_databases" / "extra_unused_file.txt").exists())

    def test_ensure_base_data_from_hf_accepts_legacy_cache_namespace(self):
        with tempfile.TemporaryDirectory() as src_td, tempfile.TemporaryDirectory() as dst_td:
            src_root = Path(src_td)
            data_dir = Path(dst_td)
            _materialize_required_layout(src_root)
            _materialize_optional_cache_group(src_root, LEGACY_DEFAULT_CACHE_NAMESPACE)

            with patch("run_ligq_2.snapshot_download", return_value=str(src_root)):
                ensure_base_data_from_hf(data_dir=data_dir, repo_id="dummy/repo")

            self.assertEqual(_missing_required_base_paths(data_dir), [])
            self.assertTrue(
                (
                    data_dir
                    / "results_databases"
                    / LEGACY_DEFAULT_CACHE_NAMESPACE
                    / "predicted_binding_data.parquet"
                ).exists()
            )

    def test_optional_cache_groups_are_well_formed(self):
        for rel_group in HF_OPTIONAL_CACHE_PATH_GROUPS:
            self.assertEqual(len(rel_group), 3)


if __name__ == "__main__":
    unittest.main()
