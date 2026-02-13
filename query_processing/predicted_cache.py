from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from query_processing.results_tables import build_predicted_binding_data_incremental


def _cache_namespace(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


@contextmanager
def file_lock(lock_path: Path, timeout_s: int = 900, poll_interval_s: float = 0.2):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fd = None
    while True:
        try:
            fd = os.open(lock_path.as_posix(), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            break
        except FileExistsError:
            if time.time() - start > timeout_s:
                raise TimeoutError(f"Timeout while waiting lock: {lock_path}")
            time.sleep(poll_interval_s)
    try:
        yield
    finally:
        try:
            if fd is not None:
                os.close(fd)
        finally:
            if lock_path.exists():
                lock_path.unlink()


def ensure_provider_cache(
    data_dir: Path,
    provider,
    known_binding: pd.DataFrame,
    proteins_needed: set[str],
    force_rebuild_cache: bool = False,
) -> pd.DataFrame:
    cache_root = Path(data_dir) / "results_databases" / "predicted_bindings"
    method_key = "__".join([f"{k}={_cache_namespace(str(v))}" for k, v in provider.method_signature().items() if k != "provider"])
    cache_dir = cache_root / provider.provider_name / method_key
    cache_dir.mkdir(parents=True, exist_ok=True)

    predicted_path = cache_dir / "predicted_binding_data.parquet"
    progress_path = cache_dir / "predicted_binding_progress.json"
    manifest_path = cache_dir / "manifest.json"
    lock_path = cache_dir / ".cache.lock"

    expected_manifest = dict(provider.method_signature())
    expected_manifest["db_fingerprint"] = provider.database_fingerprint(data_dir)

    with file_lock(lock_path):
        regenerate_cache = force_rebuild_cache
        if manifest_path.is_file() and not regenerate_cache:
            with open(manifest_path, "r") as f:
                current_manifest = json.load(f)
            if current_manifest != expected_manifest:
                regenerate_cache = True

        if regenerate_cache:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        if progress_path.exists():
            with open(progress_path, "r") as f:
                processed = set(json.load(f))
        else:
            processed = set()

        if predicted_path.exists():
            cached = pd.read_parquet(predicted_path)
            if "uniprot_id" in cached.columns:
                processed |= set(cached["uniprot_id"].astype(str).unique())

        proteins_to_compute = sorted(set(proteins_needed) - processed)

        if proteins_to_compute:
            known_subset = known_binding[known_binding["uniprot_id"].astype(str).isin(proteins_to_compute)].copy()
            build_predicted_binding_data_incremental(
                proteins_to_process=proteins_to_compute,
                cache_dir=cache_dir,
                provider=provider,
                known_binding=known_subset,
                resume=not regenerate_cache,
            )

        with open(manifest_path, "w") as f:
            json.dump(expected_manifest, f, indent=2)

        if predicted_path.exists():
            return pd.read_parquet(predicted_path)

        return pd.DataFrame(columns=["uniprot_id"])
