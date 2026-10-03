from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, with_hash: bool = False) -> dict:
    record = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        stat = path.stat()
        record.update({"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        if with_hash:
            record["sha256"] = sha256_file(path)
    return record


def representation_files(store_root: Path, names: list[str]) -> list[Path]:
    paths = [store_root / "ligands.parquet"]
    for name in names:
        meta_path = store_root / "reps" / f"{name}.meta.json"
        paths.append(meta_path)
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            paths.append(store_root / "reps" / meta["file"])
    return paths


def write_input_manifest(path: Path, files: list[Path], with_hash: bool) -> list[dict]:
    records = [file_record(file, with_hash=with_hash) for file in files]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records
