import tempfile
import unittest
from pathlib import Path

from compound_processing.huggingmolecules_microenv import _detect_install_dir


class TestHuggingMoleculesMicroenv(unittest.TestCase):
    def test_detect_install_dir_in_repo_root(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            (repo / "pyproject.toml").write_text("[build-system]\nrequires=[]\n")

            out = _detect_install_dir(repo, None)
            self.assertEqual(out, repo)

    def test_detect_install_dir_with_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            pkg = repo / "python"
            pkg.mkdir(parents=True)
            (pkg / "setup.py").write_text("from setuptools import setup\nsetup()\n")

            out = _detect_install_dir(repo, "python")
            self.assertEqual(out, pkg.resolve())

    def test_detect_install_dir_raises_when_missing(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            repo.mkdir()
            with self.assertRaises(RuntimeError):
                _detect_install_dir(repo, None)


if __name__ == "__main__":
    unittest.main()
