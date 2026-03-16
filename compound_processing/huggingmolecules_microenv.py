from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


_DEFAULT_REQS = (
    "numpy",
    "pandas",
    "pyarrow",
    "torch",
    "transformers",
    "tqdm",
)


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _microenv_python(microenv_dir: Path) -> Path:
    return microenv_dir / "bin" / "python"


def _repo_basename(repo_url: str) -> str:
    name = repo_url.rstrip("/").rsplit("/", 1)[-1]
    if name.endswith(".git"):
        name = name[:-4]
    return name or "repo"


def _clone_repo(repo_url: str, repo_ref: Optional[str], dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    _run(["git", "clone", "--depth", "1", repo_url, str(dst)])
    if repo_ref:
        _run(["git", "fetch", "--depth", "1", "origin", str(repo_ref)], cwd=dst)
        _run(["git", "checkout", str(repo_ref)], cwd=dst)


def _is_python_project(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() or (path / "setup.py").is_file()


def _detect_install_dir(repo_dir: Path, repo_subdir: Optional[str]) -> Path:
    if repo_subdir:
        candidate = (repo_dir / repo_subdir).resolve()
        if not candidate.exists() or not _is_python_project(candidate):
            raise RuntimeError(
                f"HuggingMolecules repo_subdir '{repo_subdir}' is not installable. "
                f"Expected setup.py or pyproject.toml under: {candidate}"
            )
        return candidate

    if _is_python_project(repo_dir):
        return repo_dir

    # HuggingMolecules upstream commonly keeps Python package files under ./src
    src_dir = repo_dir / "src"
    if src_dir.is_dir() and _is_python_project(src_dir):
        return src_dir

    level1 = [p for p in repo_dir.iterdir() if p.is_dir()]
    for p in level1:
        if _is_python_project(p):
            return p

    for p in level1:
        for c in p.iterdir() if p.exists() else []:
            if c.is_dir() and _is_python_project(c):
                return c

    raise RuntimeError(
        "Could not find an installable Python project in cloned HuggingMolecules repository. "
        "Pass --hm-repo-subdir (default is 'src') pointing to the folder containing setup.py or pyproject.toml."
    )




def _try_clean_huggingmolecules_cache(py_bin: Path) -> None:
    """Best-effort cache cleanup to avoid stale config/weights incompatibilities."""
    try:
        subprocess.run([str(py_bin), "-m", "src.clean_cache", "--all"], check=False)
    except Exception:
        # Intentionally ignore: cache cleaner module may not exist for all refs/layouts.
        pass


def ensure_huggingmolecules_microenv(
    microenv_dir: Path,
    *,
    hm_repo_url: str,
    hm_repo_ref: Optional[str],
    hm_repo_subdir: Optional[str],
    force_install: bool,
) -> Path:
    microenv_dir = Path(microenv_dir)
    py = _microenv_python(microenv_dir)

    if force_install and microenv_dir.exists():
        stamp = microenv_dir / "install_stamp.json"
        if stamp.exists():
            stamp.unlink()

    if not py.exists():
        microenv_dir.mkdir(parents=True, exist_ok=True)
        _run([sys.executable, "-m", "venv", str(microenv_dir)])

    stamp_path = microenv_dir / "install_stamp.json"
    expected = {
        "repo_url": hm_repo_url,
        "repo_ref": hm_repo_ref or "",
        "repo_subdir": hm_repo_subdir or "",
    }

    needs_install = force_install or not stamp_path.exists()
    if not needs_install:
        with stamp_path.open() as f:
            current = json.load(f)
        needs_install = current != expected

    if needs_install:
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        _run([str(py), "-m", "pip", "install", *_DEFAULT_REQS])

        src_root = microenv_dir / "_src"
        repo_dir = src_root / _repo_basename(hm_repo_url)
        _clone_repo(hm_repo_url, hm_repo_ref, repo_dir)
        install_dir = _detect_install_dir(repo_dir, hm_repo_subdir)
        _run([str(py), "-m", "pip", "install", str(install_dir)])
        _try_clean_huggingmolecules_cache(py)

        with stamp_path.open("w") as f:
            json.dump(expected, f, indent=2)

    return py


def build_huggingmolecules_with_microenv(
    *,
    root: Path,
    rep_name: str,
    n_bits: Optional[int],
    batch_size: int,
    max_length: Optional[int],
    model_type: str,
    model_id: str,
    microenv_dir: Path,
    hm_repo_url: str,
    hm_repo_ref: Optional[str],
    hm_repo_subdir: Optional[str],
    force_install: bool,
) -> None:
    py = ensure_huggingmolecules_microenv(
        microenv_dir,
        hm_repo_url=hm_repo_url,
        hm_repo_ref=hm_repo_ref,
        hm_repo_subdir=hm_repo_subdir,
        force_install=force_install,
    )

    worker = Path(__file__).resolve().parent.parent / "microservices" / "huggingmolecules_worker.py"
    cmd = [
        str(py),
        str(worker),
        "--root",
        str(root),
        "--rep-name",
        rep_name,
        "--batch-size",
        str(batch_size),
        "--model-type",
        model_type,
        "--model-id",
        model_id,
    ]

    if n_bits is not None:
        cmd.extend(["--n-bits", str(int(n_bits))])
    if max_length is not None:
        cmd.extend(["--max-length", str(int(max_length))])

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path(__file__).resolve().parent.parent))
    subprocess.run(cmd, check=True, env=env)
