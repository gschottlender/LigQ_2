from __future__ import annotations

import json
import os
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


def _repo_spec(repo_url: str, repo_ref: Optional[str]) -> str:
    repo_ref = (repo_ref or "").strip()
    return f"git+{repo_url}@{repo_ref}" if repo_ref else f"git+{repo_url}"


def _run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def _microenv_python(microenv_dir: Path) -> Path:
    return microenv_dir / "bin" / "python"


def ensure_huggingmolecules_microenv(
    microenv_dir: Path,
    *,
    hm_repo_url: str,
    hm_repo_ref: Optional[str],
    force_install: bool,
) -> Path:
    microenv_dir = Path(microenv_dir)
    py = _microenv_python(microenv_dir)

    if force_install and microenv_dir.exists():
        # keep previous environment around for debugging; only reset stamp
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
    }

    needs_install = force_install or not stamp_path.exists()
    if not needs_install:
        with stamp_path.open() as f:
            current = json.load(f)
        needs_install = current != expected

    if needs_install:
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
        _run([str(py), "-m", "pip", "install", *_DEFAULT_REQS])
        _run([str(py), "-m", "pip", "install", _repo_spec(hm_repo_url, hm_repo_ref)])
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
    force_install: bool,
) -> None:
    py = ensure_huggingmolecules_microenv(
        microenv_dir,
        hm_repo_url=hm_repo_url,
        hm_repo_ref=hm_repo_ref,
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
