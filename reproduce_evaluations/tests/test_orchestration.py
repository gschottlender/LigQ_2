import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from package.common import contained, copy_verified, require_revision
from package.resources import benchmark_records, download, verify_files
from package.workflow import calculation_config, commands, ordered_stages


class OrchestrationTests(unittest.TestCase):
    def test_dependencies_and_no_hardware_benchmark(self):
        self.assertEqual(ordered_stages("preprocessing"), ["representations", "preprocessing"])
        args = SimpleNamespace(data_dir=Path("data"), output_dir=Path("out"),
                               python=Path("main_python"), chemistry_python=Path("chemistry_python"))
        steps = commands(args)
        self.assertEqual(set(steps), set(ordered_stages("all")))
        self.assertEqual(steps["preprocessing"][0][0], "chemistry_python")
        self.assertEqual(steps["representations"][0][0], "main_python")
        self.assertIn("2-15", steps["neighbors"][1])
        self.assertIn("--include-nonretrieved-background", steps["distributions"][0])
        self.assertFalse(any("performance" in " ".join(c) for values in steps.values() for c in values))

    def test_fixed_revision_and_path_traversal(self):
        for value in (None, "main", "v1.0.0", "a" * 39):
            with self.assertRaises(ValueError):
                require_revision(value)
        self.assertEqual(require_revision("a" * 40), "a" * 40)
        with self.assertRaises(ValueError):
            contained(Path("data"), "../../escape")

    def test_verified_import_rejects_corrupt_source_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            destination = Path(directory) / "destination"
            source.write_bytes(b"historical")
            digest = hashlib.sha256(b"historical").hexdigest()
            copy_verified(source, destination, digest)
            verify_files(directory, [{"path":"destination", "size_bytes":10, "sha256":digest}])
            destination.write_bytes(b"different")
            with self.assertRaises(FileExistsError):
                copy_verified(source, destination, digest)
            with self.assertRaises(ValueError):
                copy_verified(source, destination, "0"*64)

    def test_no_application_paths_in_generated_config(self):
        cfg = calculation_config(Path("reproduction_data"), Path("reproduction_results"))
        self.assertIn("/frozen/", cfg["paths"]["binding_data"])
        self.assertIn("/derived/", cfg["paths"]["blast_db_prefix"])
        self.assertNotIn("databases_backup", str(cfg))

    def test_download_rejects_partial_explicit_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "both"):
                download(Path(directory), artifact_repository="owner/dataset")
            with self.assertRaisesRegex(ValueError, "both"):
                download(Path(directory), artifact_revision="a" * 40)

    def test_publication_lock_matches_resource_inventory(self):
        from package.common import ROOT, read_json, sha256
        if not (ROOT / "artifact_lock.json").exists():
            self.skipTest("Publication has not yet been made")
        record = read_json(ROOT / "artifact_lock.json")
        require_revision(record["revision"])
        self.assertEqual(record["repository"], "gschottlender/LigQ_2_evaluations")
        self.assertEqual(record["resource_lock_sha256"], sha256(ROOT / "resource_lock.json"))

    def test_bounded_download_selects_real_cells_and_keeps_global_metadata(self):
        from package.common import ROOT, read_json
        lock = read_json(ROOT / "resource_lock.json")
        rows = benchmark_records(lock, ["ampc"], [42])
        cells = [r for r in rows if len(r["path"].split("/")) == 4]
        self.assertEqual(len(cells), 4)
        self.assertTrue(all(r["path"].startswith("benchmark/ampc/random_state_42/") for r in cells))
        self.assertTrue(any(r["path"] == "benchmark/manifest.csv" for r in rows))
        self.assertEqual(benchmark_records(lock), lock["benchmark_files"])
        with self.assertRaises(ValueError):
            benchmark_records(lock, ["not_a_target"], [42])
        with self.assertRaises(ValueError):
            benchmark_records(lock, ["ampc"], [99])


if __name__ == "__main__":
    unittest.main()
