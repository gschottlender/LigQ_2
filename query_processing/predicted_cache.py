from __future__ import annotations

import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from math import inf
from typing import Callable

import pandas as pd
import pyarrow.parquet as pq

from query_processing.results_tables import build_predicted_binding_data_incremental


_HF_DOWNLOAD_METADATA_ROOT = Path(".cache") / "huggingface" / "download"
_CACHE_ARTIFACT_FILENAMES = (
    "manifest.json",
    "predicted_binding_data.parquet",
    "predicted_binding_progress.json",
    "cached_proteins.json",
    "predicted_binding_rowgroup_index.json",
)
_MIGRATE_DB_FINGERPRINT_KEY = "_migrate_db_fingerprint"


def _cache_namespace(value: str) -> str:
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _cache_method_signature(provider) -> dict:
    if hasattr(provider, "cache_method_signature"):
        return dict(provider.cache_method_signature())
    return dict(provider.method_signature())


def _requested_cache_coverage(provider) -> tuple[float | None, float | None]:
    if hasattr(provider, "cache_coverage"):
        threshold_min, threshold_max = provider.cache_coverage()
        return (
            None if threshold_min is None else float(threshold_min),
            None if threshold_max is None else float(threshold_max),
        )
    return None, None


def _provider_with_cache_coverage(provider, threshold_min: float | None, threshold_max: float | None):
    if hasattr(provider, "with_cache_coverage"):
        return provider.with_cache_coverage(threshold_min, threshold_max)
    return provider


def _provider_filter_cached_results(provider, df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(provider, "filter_cached_results"):
        return provider.filter_cached_results(df)
    return df


def _provider_database_fingerprint_version(provider) -> int:
    if hasattr(provider, "database_fingerprint_version"):
        return int(provider.database_fingerprint_version())
    return 1


def _provider_database_fingerprint_files(provider, data_dir: Path) -> list[Path]:
    if hasattr(provider, "database_fingerprint_files"):
        return [Path(path) for path in provider.database_fingerprint_files(data_dir)]
    return []


def _provider_supports_hf_legacy_cache_migration(provider) -> bool:
    if hasattr(provider, "supports_hf_legacy_cache_migration"):
        return bool(provider.supports_hf_legacy_cache_migration())
    return False


def remove_predicted_cache_dirs(
    data_dir: Path,
    provider_names: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[Path]:
    """Remove predicted-ligand cache directories and return removed paths."""
    cache_root = Path(data_dir) / "results_databases" / "predicted_bindings"
    removed: list[Path] = []

    if provider_names is None:
        if cache_root.exists():
            shutil.rmtree(cache_root)
            removed.append(cache_root)
        return removed

    for provider_name in sorted({str(name) for name in provider_names}):
        cache_dir = cache_root / provider_name
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            removed.append(cache_dir)

    return removed


def _cache_dir_name(method_signature: dict, threshold_min: float | None, threshold_max: float | None) -> str:
    parts = [
        f"{k}={_cache_namespace(str(v))}"
        for k, v in method_signature.items()
        if k != "provider"
    ]
    if threshold_min is not None:
        parts.append(f"cache_threshold_min={_cache_namespace(str(threshold_min))}")
    if threshold_max is not None:
        parts.append(f"cache_threshold_max={_cache_namespace(str(threshold_max))}")
    return "__".join(parts)


def _normalize_manifest(manifest: dict) -> dict:
    normalized = dict(manifest)
    if "cache_threshold_min" not in normalized:
        if "zinc_search_threshold" in normalized:
            normalized["cache_threshold_min"] = float(normalized["zinc_search_threshold"])
        else:
            normalized["cache_threshold_min"] = None
    elif normalized["cache_threshold_min"] is not None:
        normalized["cache_threshold_min"] = float(normalized["cache_threshold_min"])

    if "cache_threshold_max" not in normalized:
        if "zinc_search_threshold_max" in normalized:
            value = normalized["zinc_search_threshold_max"]
            normalized["cache_threshold_max"] = None if value is None else float(value)
        else:
            normalized["cache_threshold_max"] = None
    elif normalized["cache_threshold_max"] is not None:
        normalized["cache_threshold_max"] = float(normalized["cache_threshold_max"])

    try:
        normalized["db_fingerprint_version"] = int(normalized.get("db_fingerprint_version", 1))
    except (TypeError, ValueError):
        normalized["db_fingerprint_version"] = 1

    return normalized


def _manifest_matches_method_signature(manifest: dict, method_signature: dict) -> bool:
    manifest = _normalize_manifest(manifest)
    for key, value in method_signature.items():
        if manifest.get(key) != value:
            return False
    return True


def _manifest_matches_method(
    manifest: dict,
    method_signature: dict,
    db_fingerprint: str,
    db_fingerprint_version: int = 1,
) -> bool:
    manifest = _normalize_manifest(manifest)
    return (
        _manifest_matches_method_signature(manifest, method_signature)
        and manifest.get("db_fingerprint") == db_fingerprint
        and manifest.get("db_fingerprint_version") == int(db_fingerprint_version)
    )


def _hf_metadata_path(path: Path, data_dir: Path) -> Path | None:
    try:
        relative_path = path.relative_to(data_dir)
    except ValueError:
        return None
    return data_dir / _HF_DOWNLOAD_METADATA_ROOT / Path(f"{relative_path}.metadata")


def _hf_download_revision(path: Path, data_dir: Path) -> str | None:
    metadata_path = _hf_metadata_path(path, data_dir)
    if metadata_path is None or not path.is_file() or not metadata_path.is_file():
        return None
    try:
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        revision = lines[0].strip()
        downloaded_at = float(lines[2])
    except (OSError, IndexError, TypeError, ValueError):
        return None
    if not revision or path.stat().st_mtime > downloaded_at + 1.0:
        return None
    return revision


def _can_migrate_legacy_cache(
    *,
    cache_dir: Path,
    provider,
    data_dir: Path,
    manifest: dict,
    current_fingerprint_version: int,
) -> bool:
    manifest = _normalize_manifest(manifest)
    if manifest.get("db_fingerprint_version", 1) >= int(current_fingerprint_version):
        return False
    if hasattr(provider, "legacy_database_fingerprint"):
        legacy_fingerprint = provider.legacy_database_fingerprint(data_dir)
        if legacy_fingerprint and manifest.get("db_fingerprint") == legacy_fingerprint:
            return True
    if not _provider_supports_hf_legacy_cache_migration(provider):
        return False

    database_files = _provider_database_fingerprint_files(provider, data_dir)
    cache_files = [cache_dir / filename for filename in _CACHE_ARTIFACT_FILENAMES]
    if not database_files or not all(path.is_file() for path in cache_files):
        return False

    revisions = [
        _hf_download_revision(path, data_dir)
        for path in (*database_files, *cache_files)
    ]
    return all(revision is not None for revision in revisions) and len(set(revisions)) == 1


def _cache_covers_request(
    manifest: dict,
    requested_threshold_min: float | None,
    requested_threshold_max: float | None,
) -> bool:
    manifest = _normalize_manifest(manifest)
    cache_threshold_min = manifest.get("cache_threshold_min")
    cache_threshold_max = manifest.get("cache_threshold_max")

    if requested_threshold_min is not None:
        if cache_threshold_min is None or cache_threshold_min > requested_threshold_min:
            return False
    if requested_threshold_max is None:
        if cache_threshold_max is not None:
            return False
    else:
        if cache_threshold_max is not None and cache_threshold_max < requested_threshold_max:
            return False
    return True


def _candidate_sort_key(manifest: dict) -> tuple[float, float]:
    manifest = _normalize_manifest(manifest)
    cache_threshold_min = manifest.get("cache_threshold_min")
    cache_threshold_max = manifest.get("cache_threshold_max")
    min_key = -(cache_threshold_min if cache_threshold_min is not None else -inf)
    max_key = cache_threshold_max if cache_threshold_max is not None else inf
    return min_key, max_key


def _discover_compatible_cache(
    cache_root: Path,
    provider,
    data_dir: Path,
    requested_threshold_min: float | None,
    requested_threshold_max: float | None,
    proteins_needed: set[str] | None = None,
) -> tuple[Path | None, dict | None]:
    provider_root = cache_root / provider.provider_name
    if not provider_root.exists():
        return None, None

    method_signature = _cache_method_signature(provider)
    db_fingerprint = provider.database_fingerprint(data_dir)
    db_fingerprint_version = _provider_database_fingerprint_version(provider)
    requested_proteins = (
        {str(protein) for protein in proteins_needed}
        if proteins_needed is not None
        else None
    )
    candidates: list[tuple[Path, dict, int]] = []
    for cache_dir in sorted(provider_root.iterdir()):
        if not cache_dir.is_dir():
            continue
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not _manifest_matches_method_signature(manifest, method_signature):
            continue
        if not _cache_covers_request(manifest, requested_threshold_min, requested_threshold_max):
            continue

        normalized_manifest = _normalize_manifest(manifest)
        if not _manifest_matches_method(
            normalized_manifest,
            method_signature,
            db_fingerprint,
            db_fingerprint_version,
        ):
            if not _can_migrate_legacy_cache(
                cache_dir=cache_dir,
                provider=provider,
                data_dir=data_dir,
                manifest=normalized_manifest,
                current_fingerprint_version=db_fingerprint_version,
            ):
                continue
            normalized_manifest[_MIGRATE_DB_FINGERPRINT_KEY] = True

        cached_requested = 0
        if requested_proteins is not None:
            try:
                cached_proteins = _read_cached_protein_index(
                    cache_dir / "cached_proteins.json"
                )
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                cached_proteins = None
            if cached_proteins is not None:
                cached_requested = len(requested_proteins & cached_proteins)

        candidates.append((cache_dir, normalized_manifest, cached_requested))

    if not candidates:
        return None, None

    best_dir, best_manifest, _ = sorted(
        candidates,
        key=lambda item: (-item[2], *_candidate_sort_key(item[1])),
    )[0]
    return best_dir, best_manifest


def _read_lock_pid(lock_path: Path) -> int | None:
    try:
        return int(lock_path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _try_read_predicted_parquet(
    predicted_path: Path,
    load_dataframe: bool = True,
) -> tuple[pd.DataFrame | None, bool]:
    if not predicted_path.exists():
        return None, False

    try:
        pq.ParquetFile(predicted_path)
        if not load_dataframe:
            return None, False
        return pd.read_parquet(predicted_path), False
    except Exception as exc:
        print(
            "[WARN] Predicted cache parquet is unreadable/corrupt. "
            f"Will rebuild cache from scratch: {predicted_path} ({exc})"
        )
        return None, True


def _read_cached_uniprot_ids(predicted_path: Path, batch_size: int = 65536) -> set[str]:
    if not predicted_path.exists():
        return set()

    parquet_file = pq.ParquetFile(predicted_path)
    if "uniprot_id" not in parquet_file.schema_arrow.names:
        return set()

    processed: set[str] = set()
    for batch in parquet_file.iter_batches(
        batch_size=max(int(batch_size), 1),
        columns=["uniprot_id"],
    ):
        processed.update(str(value) for value in batch.column(0).to_pylist() if value is not None)
    return processed


def _read_cached_protein_index(index_path: Path) -> set[str] | None:
    if not index_path.exists():
        return None
    with open(index_path, "r") as f:
        return {str(value) for value in json.load(f)}


def _write_cached_protein_index(index_path: Path, proteins: set[str]) -> None:
    with open(index_path, "w") as f:
        json.dump(sorted(proteins), f)


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
            lock_pid = _read_lock_pid(lock_path)
            if lock_pid is not None and not _process_is_alive(lock_pid):
                print(f"[WARN] Removing stale cache lock owned by dead PID {lock_pid}: {lock_path}")
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
                continue
            if time.time() - start > timeout_s:
                owner = _read_lock_pid(lock_path)
                owner_msg = f" (owner_pid={owner})" if owner is not None else ""
                raise TimeoutError(f"Timeout while waiting lock: {lock_path}{owner_msg}")
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
    load_dataframe: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> pd.DataFrame | Path:
    cache_root = Path(data_dir) / "results_databases" / "predicted_bindings"
    requested_threshold_min, requested_threshold_max = _requested_cache_coverage(provider)
    method_signature = _cache_method_signature(provider)

    if force_rebuild_cache:
        cache_dir = cache_root / provider.provider_name / _cache_dir_name(
            method_signature=method_signature,
            threshold_min=requested_threshold_min,
            threshold_max=requested_threshold_max,
        )
        selected_manifest = None
    else:
        cache_dir, selected_manifest = _discover_compatible_cache(
            cache_root=cache_root,
            provider=provider,
            data_dir=Path(data_dir),
            requested_threshold_min=requested_threshold_min,
            requested_threshold_max=requested_threshold_max,
            proteins_needed=set(proteins_needed),
        )
        if cache_dir is None:
            cache_dir = cache_root / provider.provider_name / _cache_dir_name(
                method_signature=method_signature,
                threshold_min=requested_threshold_min,
                threshold_max=requested_threshold_max,
            )

    cache_threshold_min = requested_threshold_min
    cache_threshold_max = requested_threshold_max
    migrate_db_fingerprint = False
    if selected_manifest is not None:
        selected_manifest = dict(selected_manifest)
        migrate_db_fingerprint = bool(
            selected_manifest.pop(_MIGRATE_DB_FINGERPRINT_KEY, False)
        )
        cache_threshold_min = selected_manifest.get("cache_threshold_min")
        cache_threshold_max = selected_manifest.get("cache_threshold_max")

    cache_provider = _provider_with_cache_coverage(provider, cache_threshold_min, cache_threshold_max)
    cache_dir.mkdir(parents=True, exist_ok=True)

    predicted_path = cache_dir / "predicted_binding_data.parquet"
    progress_path = cache_dir / "predicted_binding_progress.json"
    protein_index_path = cache_dir / "cached_proteins.json"
    row_group_index_path = cache_dir / "predicted_binding_rowgroup_index.json"
    manifest_path = cache_dir / "manifest.json"
    lock_path = cache_dir / ".cache.lock"

    expected_manifest = dict(method_signature)
    expected_manifest["cache_threshold_min"] = cache_threshold_min
    expected_manifest["cache_threshold_max"] = cache_threshold_max
    expected_manifest["db_fingerprint"] = cache_provider.database_fingerprint(data_dir)
    expected_manifest["db_fingerprint_version"] = _provider_database_fingerprint_version(
        cache_provider
    )

    with file_lock(lock_path):
        regenerate_cache = force_rebuild_cache
        if manifest_path.is_file() and not regenerate_cache:
            with open(manifest_path, "r") as f:
                current_manifest = _normalize_manifest(json.load(f))
            if current_manifest != expected_manifest:
                can_migrate = (
                    migrate_db_fingerprint
                    and _manifest_matches_method_signature(current_manifest, method_signature)
                    and _cache_covers_request(
                        current_manifest,
                        cache_threshold_min,
                        cache_threshold_max,
                    )
                    and _can_migrate_legacy_cache(
                        cache_dir=cache_dir,
                        provider=cache_provider,
                        data_dir=Path(data_dir),
                        manifest=current_manifest,
                        current_fingerprint_version=expected_manifest["db_fingerprint_version"],
                    )
                )
                if can_migrate:
                    print(
                        "[INFO] Migrating downloaded predicted cache to portable "
                        f"database fingerprint: {cache_dir}"
                    )
                    with open(manifest_path, "w") as f:
                        json.dump(expected_manifest, f, indent=2)
                else:
                    regenerate_cache = True

        if regenerate_cache:
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        cached, parquet_corrupted = _try_read_predicted_parquet(
            predicted_path,
            load_dataframe=load_dataframe,
        )
        if parquet_corrupted:
            regenerate_cache = True
            for path in (predicted_path, progress_path, protein_index_path, row_group_index_path, manifest_path):
                if path.exists():
                    path.unlink()

        processed_from_index = False
        processed_from_progress = False
        indexed = _read_cached_protein_index(protein_index_path) if not regenerate_cache else None
        if indexed is not None:
            processed = indexed
            processed_from_index = True
        elif progress_path.exists() and not regenerate_cache:
            with open(progress_path, "r") as f:
                processed = {str(value) for value in json.load(f)}
            processed_from_progress = True
            _write_cached_protein_index(protein_index_path, processed)
        else:
            processed = set()

        if cached is not None and "uniprot_id" in cached.columns:
            processed |= set(cached["uniprot_id"].astype(str).unique())
            _write_cached_protein_index(protein_index_path, processed)
        elif (
            not processed_from_index
            and not processed_from_progress
            and predicted_path.exists()
            and not regenerate_cache
        ):
            processed |= _read_cached_uniprot_ids(predicted_path)
            _write_cached_protein_index(protein_index_path, processed)

        requested_total = len(set(proteins_needed))
        already_cached = len(set(proteins_needed) & processed)
        proteins_to_compute = sorted(set(proteins_needed) - processed)
        print(
            "[INFO] Requested LigQ proteins: "
            f"total={requested_total}, cached={already_cached}, pending={len(proteins_to_compute)}"
        )
        if progress_callback is not None:
            progress_callback(already_cached, requested_total)

        if proteins_to_compute:
            known_subset = known_binding[known_binding["uniprot_id"].astype(str).isin(proteins_to_compute)].copy()

            def report_pending_progress(current: int, _pending_total: int) -> None:
                if progress_callback is not None:
                    progress_callback(
                        min(requested_total, already_cached + current),
                        requested_total,
                    )

            build_predicted_binding_data_incremental(
                proteins_to_process=proteins_to_compute,
                cache_dir=cache_dir,
                provider=cache_provider,
                known_binding=known_subset,
                resume=not regenerate_cache,
                progress_callback=report_pending_progress,
            )

        with open(manifest_path, "w") as f:
            json.dump(expected_manifest, f, indent=2)

        if not load_dataframe:
            if predicted_path.exists():
                return predicted_path
            return pd.DataFrame(columns=["uniprot_id"])

        refreshed_cached, parquet_corrupted = _try_read_predicted_parquet(predicted_path)
        if parquet_corrupted:
            raise RuntimeError(
                "Predicted cache parquet is corrupt after rebuild attempt. "
                f"Please remove cache directory and rerun: {cache_dir}"
            )

        if refreshed_cached is not None:
            return _provider_filter_cached_results(provider, refreshed_cached)

        return pd.DataFrame(columns=["uniprot_id"])
