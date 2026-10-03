from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
SEEDS = (42, 10, 27, 3, 8)
METHODS = ("morgan_1024_r2", "ap_rdkit", "chemberta_zinc_base_768", "maccs",
           "rdkit_1024", "topological_torsion_rdkit_1024", "morgan_feature_1024_r2")
SENSITIVITY_TARGETS = ("cp2c9", "cp3a4", "aa2ar", "drd3", "plk1", "mk01",
                       "esr1", "andr", "ampc", "cah2", "bace1", "fa10")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Path escapes its resource directory: {relative}")
    return candidate


def require_revision(revision):
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use a full 40-character immutable Hugging Face commit, not main or a tag.")
    return revision


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if not source.is_file() or sha256(source) != expected:
        raise ValueError(f"Missing or changed historical resource: {source}")
    if destination.exists():
        if sha256(destination) != expected:
            raise FileExistsError(f"Refusing to replace a different file: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, temporary)
    if sha256(temporary) != expected:
        raise ValueError(f"Copy verification failed: {destination}")
    temporary.replace(destination)


def command(script, *arguments, python=None):
    return [str(python or sys.executable), str(ENGINE / script), *map(str, arguments)]


def execute(argv, *, pythonpath=True):
    import os
    environment = dict(os.environ)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(ROOT / ".mplconfig")
    if pythonpath:
        environment["PYTHONPATH"] = os.pathsep.join((str(ENGINE), str(ENGINE / "dependencies/evaluation_core"),
            str(ENGINE / "dependencies/ligq_core"), environment.get("PYTHONPATH", "")))
    print("Running:", " ".join(map(str, argv)), flush=True)
    subprocess.run(argv, check=True, cwd=ROOT, env=environment)


def tree_signature():
    files = sorted([*ENGINE.rglob("*.py"), *(ROOT / "package").glob("*.py"),
                    ROOT / "reproduce.py", ROOT / "resource_lock.json"])
    payload = {str(p.relative_to(ROOT)): sha256(p) for p in files}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
