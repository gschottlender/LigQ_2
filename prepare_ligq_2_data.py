from __future__ import annotations

import argparse
import json
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import file_download as hf_file_download

from progress_reporting import ProgressEmitter


HF_REPO_ID = "gschottlender/LigQ_2"
HF_REVISION = "main"

BSI_FAMILIES = (
    "PF00001", "PF00002", "PF00026", "PF00067", "PF00069", "PF00089",
    "PF00104", "PF00112", "PF00135", "PF00194", "PF00209", "PF00233",
    "PF00413", "PF00520", "PF00850", "PF01094", "PF07714",
)

DEFAULT_PREDICTED_CACHE_NAMESPACE = (
    "results_databases/predicted_bindings/zinc/"
    "search_representation=morgan_1024_r2__search_metric=tanimoto__cache_threshold_min=0.4"
)
DEFAULT_PREDICTED_CACHE_FILES = (
    "manifest.json",
    "predicted_binding_data.parquet",
    "predicted_binding_progress.json",
    "cached_proteins.json",
    "predicted_binding_rowgroup_index.json",
)

FEATURE_PREDICTED_CACHE_NAMESPACE = (
    "results_databases/predicted_bindings/zinc/"
    "search_representation=morgan_feature_1024_r2__search_metric=tanimoto__cache_threshold_min=0.5"
)

# Core GUI runtime installation. Exact paths intentionally exclude the
# redundant compressed Pfam archive and BSI training diagnostics.
CORE_DATA_PATHS = (
    "sequences/target_sequences.fasta",
    "results_databases/known_binding_data.parquet",
    "results_databases/protein_domains.parquet",
    "compound_data/pdb_chembl/ligands.parquet",
    "compound_data/pdb_chembl/reps/morgan_1024_r2.dat",
    "compound_data/pdb_chembl/reps/morgan_1024_r2.meta.json",
    "complementary_databases/blast/target_sequences.pdb",
    "complementary_databases/blast/target_sequences.phr",
    "complementary_databases/blast/target_sequences.pin",
    "complementary_databases/blast/target_sequences.pjs",
    "complementary_databases/blast/target_sequences.pot",
    "complementary_databases/blast/target_sequences.psq",
    "complementary_databases/blast/target_sequences.ptf",
    "complementary_databases/blast/target_sequences.pto",
    "complementary_databases/pfam/Pfam-A.hmm",
    "complementary_databases/pfam/Pfam-A.hmm.h3f",
    "complementary_databases/pfam/Pfam-A.hmm.h3i",
    "complementary_databases/pfam/Pfam-A.hmm.h3m",
    "complementary_databases/pfam/Pfam-A.hmm.h3p",
    "compound_data/zinc/ligands.parquet",
    "compound_data/zinc/reps/morgan_1024_r2.dat",
    "compound_data/zinc/reps/morgan_1024_r2.meta.json",
    "bsi_models/mpg_1024/manifest.json",
    "bsi_models/mpg_1024/summary.csv",
    *(f"bsi_models/mpg_1024/{family}/model.pth" for family in BSI_FAMILIES),
    *(f"bsi_models/mpg_1024/{family}/params.json" for family in BSI_FAMILIES),
)

ECFP_CACHE_PATHS = tuple(
    f"{DEFAULT_PREDICTED_CACHE_NAMESPACE}/{filename}"
    for filename in DEFAULT_PREDICTED_CACHE_FILES
)

FCFP_CACHE_PATHS = (
    "compound_data/pdb_chembl/reps/morgan_feature_1024_r2.dat",
    "compound_data/pdb_chembl/reps/morgan_feature_1024_r2.meta.json",
    "compound_data/zinc/reps/morgan_feature_1024_r2.dat",
    "compound_data/zinc/reps/morgan_feature_1024_r2.meta.json",
    *(
        f"{FEATURE_PREDICTED_CACHE_NAMESPACE}/{filename}"
        for filename in DEFAULT_PREDICTED_CACHE_FILES
    ),
)

SETUP_PACKAGE_PATHS = {
    "core": CORE_DATA_PATHS,
    "ecfp_cache": ECFP_CACHE_PATHS,
    "fcfp_cache": FCFP_CACHE_PATHS,
}
ALL_SETUP_DATA_PATHS = tuple(
    path
    for package_paths in SETUP_PACKAGE_PATHS.values()
    for path in package_paths
)

# Legacy alias: command-line callers that do not choose packages retain the
# previous core + default ECFP cache behavior.
REQUIRED_DATA_PATHS = CORE_DATA_PATHS + ECFP_CACHE_PATHS

# Metadata snapshot from the official repository on 2026-07-24. Live Hub
# metadata remains the source of truth.
FALLBACK_PACKAGE_METADATA = {
    "core": {"total_bytes": 5_934_643_866, "total_file_count": 58},
    "ecfp_cache": {"total_bytes": 679_398_692, "total_file_count": 5},
    "fcfp_cache": {"total_bytes": 2_024_098_043, "total_file_count": 9},
}
FALLBACK_TOTAL_REQUIRED_BYTES = (
    FALLBACK_PACKAGE_METADATA["core"]["total_bytes"]
    + FALLBACK_PACKAGE_METADATA["ecfp_cache"]["total_bytes"]
)
FALLBACK_TOTAL_FILE_COUNT = (
    FALLBACK_PACKAGE_METADATA["core"]["total_file_count"]
    + FALLBACK_PACKAGE_METADATA["ecfp_cache"]["total_file_count"]
)


@dataclass(frozen=True)
class RequiredRepoFile:
    path: str
    size: int


def _available_bytes(data_dir: Path) -> int:
    probe = data_dir
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def list_required_repo_files(
    *,
    api: Any | None = None,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    required_paths: Iterable[str] = REQUIRED_DATA_PATHS,
) -> list[RequiredRepoFile]:
    required_paths = tuple(required_paths)
    required_path_set = set(required_paths)
    validate_manifest = api is None
    api = api or HfApi()
    entries = api.list_repo_tree(
        repo_id,
        repo_type="dataset",
        revision=revision,
        recursive=True,
        expand=True,
    )
    files = [
        RequiredRepoFile(path=entry.path, size=int(entry.size or 0))
        for entry in entries
        if getattr(entry, "size", None) is not None and entry.path in required_path_set
    ]
    if not files:
        raise RuntimeError(f"No required files were found in Hugging Face dataset '{repo_id}'.")
    if validate_manifest:
        remote_paths = {item.path for item in files}
        unavailable = sorted(required_path_set - remote_paths)
        if unavailable:
            preview = ", ".join(unavailable[:5])
            suffix = "" if len(unavailable) <= 5 else f" and {len(unavailable) - 5} more"
            raise RuntimeError(f"Required files are unavailable in '{repo_id}': {preview}{suffix}.")
    return sorted(files, key=lambda item: item.path)


def _missing_files(data_dir: Path, files: Iterable[RequiredRepoFile]) -> list[RequiredRepoFile]:
    return [item for item in files if not (data_dir / item.path).is_file()]


def _package_status(
    package_id: str,
    data_dir: Path,
    files_by_path: dict[str, RequiredRepoFile],
) -> dict[str, Any]:
    paths = SETUP_PACKAGE_PATHS[package_id]
    files = [files_by_path[path] for path in paths]
    missing = _missing_files(data_dir, files)
    return {
        "id": package_id,
        "required": package_id == "core",
        "default_selected": package_id != "fcfp_cache",
        "installed": not missing,
        "required_download_bytes": sum(item.size for item in missing),
        "total_bytes": sum(item.size for item in files),
        "required_file_count": len(missing),
        "total_file_count": len(files),
        "missing_paths": [item.path for item in missing],
    }


def _fallback_package_status(package_id: str, data_dir: Path) -> dict[str, Any]:
    paths = SETUP_PACKAGE_PATHS[package_id]
    missing_paths = [path for path in paths if not (data_dir / path).is_file()]
    snapshot = FALLBACK_PACKAGE_METADATA[package_id]
    return {
        "id": package_id,
        "required": package_id == "core",
        "default_selected": package_id != "fcfp_cache",
        "installed": not missing_paths,
        "required_download_bytes": 0 if not missing_paths else snapshot["total_bytes"],
        "total_bytes": snapshot["total_bytes"],
        "required_file_count": len(missing_paths),
        "total_file_count": snapshot["total_file_count"],
        "missing_paths": missing_paths,
    }


def _build_setup_status(
    *,
    packages: list[dict[str, Any]],
    available_bytes: int,
    repo_id: str,
    revision: str,
    size_source: str,
    metadata_error: str | None,
) -> dict[str, Any]:
    packages_by_id = {package["id"]: package for package in packages}
    core = packages_by_id["core"]
    default_packages = [
        packages_by_id["core"],
        packages_by_id["ecfp_cache"],
    ]
    required_bytes = sum(package["required_download_bytes"] for package in default_packages)
    total_bytes = sum(package["total_bytes"] for package in default_packages)
    missing_paths = [
        path
        for package in default_packages
        for path in package["missing_paths"]
    ]
    ready = bool(core["installed"])
    return {
        "ready": ready,
        "state": "ready" if ready else "required",
        "repo_id": repo_id,
        "revision": revision,
        "required_download_bytes": required_bytes,
        "total_required_bytes": total_bytes,
        "available_bytes": available_bytes,
        "enough_space": required_bytes <= available_bytes,
        "required_file_count": sum(
            package["required_file_count"] for package in default_packages
        ),
        "total_file_count": sum(
            package["total_file_count"] for package in default_packages
        ),
        "missing_paths": missing_paths,
        "size_source": size_source,
        "metadata_error": metadata_error,
        "packages": packages,
    }


def inspect_default_data(
    data_dir: Path,
    *,
    api: Any | None = None,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
) -> dict[str, Any]:
    data_dir = Path(data_dir)
    available = _available_bytes(data_dir)
    try:
        files = list_required_repo_files(
            api=api,
            repo_id=repo_id,
            revision=revision,
            required_paths=ALL_SETUP_DATA_PATHS,
        )
        files_by_path = {item.path: item for item in files}
        packages = [
            _package_status(package_id, data_dir, files_by_path)
            for package_id in SETUP_PACKAGE_PATHS
        ]
        return _build_setup_status(
            packages=packages,
            available_bytes=available,
            repo_id=repo_id,
            revision=revision,
            size_source="huggingface",
            metadata_error=None,
        )
    except Exception as exc:
        packages = [
            _fallback_package_status(package_id, data_dir)
            for package_id in SETUP_PACKAGE_PATHS
        ]
        return _build_setup_status(
            packages=packages,
            available_bytes=available,
            repo_id=repo_id,
            revision=revision,
            size_source="repository_snapshot",
            metadata_error=str(exc),
        )


class _TrackedByteProgressBar:
    """Minimal tqdm-compatible object used by Hugging Face file downloads."""

    def __init__(
        self,
        tracker: "SetupDownloadTracker",
        item: RequiredRepoFile,
        *,
        initial: int = 0,
    ) -> None:
        self._tracker = tracker
        self._item = item
        self._tracker.set_file_bytes(item, initial)

    def __enter__(self) -> "_TrackedByteProgressBar":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        return None

    def update(self, amount: int | float = 1) -> bool:
        self._tracker.add_file_bytes(self._item, amount)
        return True


class SetupDownloadTracker:
    """Aggregate concurrent Hugging Face byte and file progress."""

    def __init__(
        self,
        *,
        progress: ProgressEmitter,
        files: Iterable[RequiredRepoFile],
        missing: Iterable[RequiredRepoFile],
        context: str,
        emit_interval_seconds: float = 0.25,
    ) -> None:
        all_files = list(files)
        missing_paths = {item.path for item in missing}
        self._progress = progress
        self._files = {item.path: item for item in all_files}
        self._file_bytes = {
            item.path: 0 if item.path in missing_paths else item.size
            for item in all_files
        }
        self._completed = {
            item.path for item in all_files if item.path not in missing_paths
        }
        self._context = context
        self._total_bytes = sum(item.size for item in all_files)
        self._total_files = len(all_files)
        self._session_start_bytes = sum(self._file_bytes.values())
        self._started_at = time.monotonic()
        self._last_emit_at = 0.0
        self._emit_interval_seconds = emit_interval_seconds
        self._lock = threading.Lock()
        self._thread_local = threading.local()
        self._fallback_tqdm: Any = None

    def set_fallback_tqdm(self, tqdm_factory: Any) -> None:
        self._fallback_tqdm = tqdm_factory

    def begin_file(self, item: RequiredRepoFile) -> None:
        self._thread_local.item = item

    def end_file(self) -> None:
        self._thread_local.item = None

    def tqdm(self, *args: Any, **kwargs: Any) -> Any:
        item = getattr(self._thread_local, "item", None)
        if item is None:
            return self._fallback_tqdm(*args, **kwargs)
        return _TrackedByteProgressBar(
            self,
            item,
            initial=max(0, int(kwargs.get("initial") or 0)),
        )

    def set_file_bytes(self, item: RequiredRepoFile, amount: int | float) -> None:
        with self._lock:
            current = self._file_bytes.get(item.path, 0)
            self._file_bytes[item.path] = min(item.size, max(current, int(amount)))
            self._emit_locked()

    def add_file_bytes(self, item: RequiredRepoFile, amount: int | float) -> None:
        increment = max(0, int(amount))
        if increment == 0:
            return
        with self._lock:
            current = self._file_bytes.get(item.path, 0)
            self._file_bytes[item.path] = min(item.size, current + increment)
            self._emit_locked()

    def finish_file(self, item: RequiredRepoFile) -> None:
        with self._lock:
            self._file_bytes[item.path] = item.size
            self._completed.add(item.path)
            self._emit_locked(force=True)

    def emit(self, *, force: bool = False) -> None:
        with self._lock:
            self._emit_locked(force=force)

    def _emit_locked(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_emit_at < self._emit_interval_seconds:
            return
        self._last_emit_at = now

        downloaded_bytes = min(self._total_bytes, sum(self._file_bytes.values()))
        completed_files = min(self._total_files, len(self._completed))
        fraction = downloaded_bytes / self._total_bytes if self._total_bytes else 1.0
        percent = 8 + round(fraction * 80)

        session_bytes = downloaded_bytes - self._session_start_bytes
        elapsed = max(0.001, now - self._started_at)
        if session_bytes > 0 and downloaded_bytes < self._total_bytes:
            eta_seconds = round((self._total_bytes - downloaded_bytes) / (session_bytes / elapsed))
        else:
            eta_seconds = 0

        self._progress.emit(
            step="downloading",
            label="Downloading default data",
            step_index=2,
            step_count=3,
            percent=percent,
            current=downloaded_bytes,
            total=self._total_bytes,
            unit="bytes",
            context=self._context,
            eta_seconds=eta_seconds,
            downloaded_bytes=downloaded_bytes,
            download_total_bytes=self._total_bytes,
            completed_files=completed_files,
            total_files=self._total_files,
        )


def _download_required_files(
    *,
    data_dir: Path,
    files: list[RequiredRepoFile],
    missing: list[RequiredRepoFile],
    progress: ProgressEmitter,
    repo_id: str,
    revision: str,
    max_workers: int = 8,
) -> None:
    context = f"{sum(item.size for item in files) / 1_000_000_000:.2f} GB from {repo_id}"
    tracker = SetupDownloadTracker(
        progress=progress,
        files=files,
        missing=missing,
        context=context,
    )
    tracker.emit(force=True)

    original_tqdm = hf_file_download.tqdm
    tracker.set_fallback_tqdm(original_tqdm)
    hf_file_download.tqdm = tracker.tqdm

    def download_one(item: RequiredRepoFile) -> None:
        tracker.begin_file(item)
        try:
            hf_hub_download(
                repo_id=repo_id,
                filename=item.path,
                repo_type="dataset",
                revision=revision,
                local_dir=data_dir,
            )
            tracker.finish_file(item)
        finally:
            tracker.end_file()

    try:
        with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(missing)))) as executor:
            futures = [executor.submit(download_one, item) for item in missing]
            for future in as_completed(futures):
                future.result()
    finally:
        hf_file_download.tqdm = original_tqdm

    tracker.emit(force=True)


def prepare_default_data(
    data_dir: Path,
    *,
    progress: ProgressEmitter,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    include_ecfp_cache: bool = True,
    include_fcfp_cache: bool = False,
) -> None:
    data_dir = Path(data_dir)
    progress.emit(
        step="inspecting",
        label="Inspecting required data",
        step_index=1,
        step_count=3,
        percent=2,
        context=repo_id,
    )

    try:
        selected_paths = list(CORE_DATA_PATHS)
        if include_ecfp_cache:
            selected_paths.extend(ECFP_CACHE_PATHS)
        if include_fcfp_cache:
            selected_paths.extend(FCFP_CACHE_PATHS)
        files = list_required_repo_files(
            repo_id=repo_id,
            revision=revision,
            required_paths=selected_paths,
        )
    except Exception as exc:
        raise RuntimeError(f"Could not inspect Hugging Face dataset '{repo_id}': {exc}") from exc

    missing = _missing_files(data_dir, files)
    required_bytes = sum(item.size for item in missing)
    available = _available_bytes(data_dir)
    if required_bytes > available:
        raise RuntimeError(
            "Insufficient disk space for initial setup: "
            f"{required_bytes} bytes required, {available} bytes available."
        )

    if missing:
        data_dir.mkdir(parents=True, exist_ok=True)
        _download_required_files(
            data_dir=data_dir,
            files=files,
            missing=missing,
            progress=progress,
            repo_id=repo_id,
            revision=revision,
            max_workers=8,
        )

    progress.emit(
        step="verifying",
        label="Verifying installed data",
        step_index=3,
        step_count=3,
        percent=92,
        context=str(data_dir),
    )
    still_missing = _missing_files(data_dir, files)
    if still_missing:
        preview = ", ".join(item.path for item in still_missing[:5])
        suffix = "" if len(still_missing) <= 5 else f" and {len(still_missing) - 5} more"
        raise RuntimeError(f"Initial setup is incomplete. Missing: {preview}{suffix}.")

    progress.emit(
        step="verifying",
        label="Default data ready",
        step_index=3,
        step_count=3,
        percent=99,
        current=len(files),
        total=len(files),
        unit="files",
        context=str(data_dir),
    )
    print(f"[INFO] Default LigQ 2 data is ready in: {data_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the default LigQ 2 GUI data.")
    parser.add_argument("--data-dir", default="databases")
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--revision", default=HF_REVISION)
    parser.add_argument("--status-json", action="store_true")
    parser.add_argument("--progress-json", action="store_true")
    parser.add_argument(
        "--skip-ecfp-cache",
        action="store_true",
        help="Install the required base data without the default ECFP cache.",
    )
    parser.add_argument(
        "--include-fcfp-cache",
        action="store_true",
        help=(
            "Also install the Morgan Feature FCFP representations and its "
            "precomputed ZINC cache."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    if args.status_json:
        print(
            json.dumps(
                inspect_default_data(
                    data_dir,
                    repo_id=args.repo_id,
                    revision=args.revision,
                ),
                separators=(",", ":"),
            )
        )
        return

    prepare_default_data(
        data_dir,
        progress=ProgressEmitter(enabled=args.progress_json),
        repo_id=args.repo_id,
        revision=args.revision,
        include_ecfp_cache=not args.skip_ecfp_cache,
        include_fcfp_cache=args.include_fcfp_cache,
    )


if __name__ == "__main__":
    main()
