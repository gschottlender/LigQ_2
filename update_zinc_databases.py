#!/usr/bin/env python
"""
update_zinc_database.py

Command-line entry point to prepare the local ZINC database used by the project.

This script:

  1) Ensures that the ZINC URL list file 'ZINC-downloader-2D-smi.uri'
     is present under <output_dir>/zinc. If it is missing, it is downloaded
     from the 'zinc' folder of the Hugging Face repository
     'gschottlender/LigQ_2'.

  2) Calls `generate_zinc_database(...)` from `generate_databases.zinc_db`
     with consistent defaults to:
       - download all ZINC .smi chunks listed in the URL file,
       - build a unified ligands_smiles.parquet,
       - move previous representation files into
         compound_data/zinc/old_reps_backup,
       - build the ZINC ligand index and Morgan fingerprints.
"""

import argparse
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from generate_databases.zinc_db import generate_zinc_database

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


def backup_zinc_predicted_cache(
    output_dir: Path,
    backup_dirname: str = "old_predicted_bindings_backup",
) -> Optional[Path]:
    """
    Move the existing ZINC predicted-binding cache to a timestamped backup.

    ZINC rebuilds change the ligand universe and invalidate any per-protein
    predicted cache computed against the previous base. The runtime cache also
    validates database fingerprints, but moving the old cache keeps the
    results_databases tree unambiguous after an update.
    """
    cache_dir = output_dir / "results_databases" / "predicted_bindings" / "zinc"
    if not cache_dir.exists():
        return None

    existing_entries = sorted(cache_dir.iterdir())
    if not existing_entries:
        return None

    backup_root = output_dir / "results_databases" / backup_dirname / "zinc"
    backup_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / timestamp
    suffix = 1
    while backup_dir.exists():
        suffix += 1
        backup_dir = backup_root / f"{timestamp}_{suffix}"

    shutil.move(str(cache_dir), str(backup_dir))
    print(f"[INFO] Moved existing ZINC predicted cache to backup: {backup_dir}")
    return backup_dir


# ---------------------------------------------------------------------------
# Helper: download the ZINC URL file from Hugging Face if missing
# ---------------------------------------------------------------------------

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

def download_zinc_uri_file_if_missing(
    zinc_data_dir: Path,
    hf_repo_id: str = "gschottlender/LigQ_2",
    hf_subpath: str = "zinc/ZINC-downloader-2D-smi.uri",
    filename: str = "ZINC-downloader-2D-smi.uri",
) -> Path:
    """
    Ensure that the ZINC URL file exists under `zinc_data_dir`.
    If not, download it from a Hugging Face *dataset* repository.

    Parameters
    ----------
    zinc_data_dir : Path
        Directory where the ZINC-related files live (e.g. <output_dir>/zinc).
    hf_repo_id : str
        Hugging Face dataset repository ID that contains the URL file.
    hf_subpath : str
        Relative path to the URL file inside the dataset repo
        (e.g. 'zinc/ZINC-downloader-2D-smi.uri').
    filename : str
        Local filename to use under `zinc_data_dir`
        (e.g. 'ZINC-downloader-2D-smi.uri').

    Returns
    -------
    Path
        Path to the local URL file: zinc_data_dir / filename
    """
    zinc_data_dir.mkdir(parents=True, exist_ok=True)
    local_path = zinc_data_dir / filename 

    if local_path.exists():
        print(f"[INFO] ZINC URL file already present: {local_path}")
        return local_path

    print(
        "[INFO] ZINC URL file not found. "
        f"Downloading from Hugging Face dataset '{hf_repo_id}' "
        f"subpath '{hf_subpath}'..."
    )

    try:
        repo_file_path = hf_hub_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            filename=hf_subpath,          # p.ej. 'zinc/ZINC-downloader-2D-smi.uri'
            local_dir=str(zinc_data_dir), # p.ej. '<output_dir>/zinc'
        )
    except RepositoryNotFoundError as e:
        print(
            "[ERROR] Could not access the Hugging Face dataset "
            f"'{hf_repo_id}'. Double-check that it exists and is public.\n"
            "URL should look like:\n"
            f"  https://huggingface.co/datasets/{hf_repo_id}\n"
        )
        raise
    except HfHubHTTPError as e:
        print(
            "[ERROR] Failed to download the ZINC URL file from Hugging Face.\n"
            f"Details: {e}\n"
            "You can also create the file manually at:\n"
            f"  {local_path}\n"
        )
        raise

    repo_file_path = Path(repo_file_path)

    if repo_file_path != local_path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        repo_file_path.replace(local_path)

    print(f"[INFO] ZINC URL file downloaded to: {local_path}")
    return local_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    args : list of str, optional
        Arguments to parse (mainly for testing). If None, sys.argv[1:] is used.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Build the local ZINC database. "
            "If needed, download the 'ZINC-downloader-2D-smi.uri' file from "
            "the Hugging Face repository 'gschottlender/LigQ_2' into "
            "<output-dir>/zinc, then run generate_zinc_database()."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="databases",
        help=(
            "Root directory where databases will be stored. "
            "ZINC-specific files will be under <output-dir>/zinc and "
            "<output-dir>/compound_data/zinc. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--temp-data-dir",
        type=str,
        default="temp_data",
        help=(
            "Root directory for temporary data (e.g. raw ZINC .smi chunks). "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="gschottlender/LigQ_2",
        help=(
            "Hugging Face repository ID that contains the file "
            "'zinc/ZINC-downloader-2D-smi.uri'. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--hf-subpath",
        type=str,
        default="zinc/ZINC-downloader-2D-smi.uri",
        help=(
            "Relative path to the URL file inside the Hugging Face repository. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--chemberta-rep",
        action="store_true",
        help="Generate (or update) the ChemBERTa compound embeddings database."
    )

    parser.add_argument(
        "--keep-existing-reps",
        action="store_true",
        help=(
            "Keep the current files under <output-dir>/compound_data/zinc/reps "
            "instead of moving them to old_reps_backup before rebuilding."
        ),
    )

    parser.add_argument(
        "--keep-existing-predicted-cache",
        action="store_true",
        help=(
            "Keep the current files under "
            "<output-dir>/results_databases/predicted_bindings/zinc instead "
            "of moving them to old_predicted_bindings_backup before rebuilding."
        ),
    )

    parser.add_argument(
        "--download-workers",
        type=int,
        default=4,
        help=(
            "Number of parallel workers for ZINC chunk download. "
            "Lower values are slower but more robust against rate limits. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--download-retries-per-scheme",
        type=int,
        default=4,
        help=(
            "Retries per URL scheme (https first, then http fallback). "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--download-retry-wait-seconds",
        type=float,
        default=2.0,
        help=(
            "Base wait in seconds between retries; backoff is linear "
            "(base * attempt). Default: %(default)s"
        ),
    )

    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    temp_data_dir = Path(args.temp_data_dir).resolve()
    generate_chemberta = args.chemberta_rep
    backup_old_reps = not args.keep_existing_reps
    backup_predicted_cache = not args.keep_existing_predicted_cache

    print(f"[INFO] Output directory       : {output_dir}")
    print(f"[INFO] Temporary data directory: {temp_data_dir}")
    print(f"[INFO] Download workers        : {args.download_workers}")
    print(f"[INFO] Retries per scheme      : {args.download_retries_per_scheme}")
    print(f"[INFO] Retry base wait (s)     : {args.download_retry_wait_seconds}")
    print(f"[INFO] Backup old reps         : {backup_old_reps}")
    print(f"[INFO] Backup predicted cache  : {backup_predicted_cache}")

    if backup_predicted_cache:
        backup_zinc_predicted_cache(output_dir=output_dir)

    # ------------------------------------------------------------------
    # 1) Ensure ZINC URL file exists under <output_dir>/zinc
    # ------------------------------------------------------------------
    zinc_data_dir = output_dir / "zinc"
    url_file_path = download_zinc_uri_file_if_missing(
        zinc_data_dir=zinc_data_dir,
        hf_repo_id=args.hf_repo_id,
        hf_subpath=args.hf_subpath,
        filename="ZINC-downloader-2D-smi.uri",
    )

    # ------------------------------------------------------------------
    # 2) Run the ZINC database generation pipeline
    # ------------------------------------------------------------------
    #
    # We pass the paths and fixed parameters exactly as requested:
    #   - urls_filename: "ZINC-downloader-2D-smi.uri"
    #   - ligands_smiles_filename: "ligands_smiles.parquet"
    #   - compound_root: <output_dir>/compound_data/zinc
    #   - n_bits: 1024
    #   - radius: 2
    #   - batch_size: 10_000
    #   - rep_name: "morgan_1024_r2"
    #
    # The function generate_zinc_database(...) is expected to:
    #   - read the URL file from `zinc_data_dir / urls_filename`,
    #   - download the raw .smi chunks into `zinc_temp_dir`,
    #   - build the ligands_smiles parquet,
    #   - move old reps to compound_root/old_reps_backup,
    #   - build the ligand index + Morgan representation under compound_root.
    # ------------------------------------------------------------------
    zinc_temp_dir = temp_data_dir / "zinc_db"
    compound_root = output_dir / "compound_data" / "zinc"

    print("[INFO] Starting ZINC database generation via generate_zinc_database(...)")
    generate_zinc_database(
        zinc_data_dir=zinc_data_dir,
        zinc_temp_dir=zinc_temp_dir,
        urls_filename="ZINC-downloader-2D-smi.uri",      # do not change
        ligands_smiles_filename="ligands_smiles.parquet",  # do not change
        compound_root=compound_root,
        n_bits=1024,
        radius=2,
        batch_size=10_000,
        rep_name="morgan_1024_r2",
        download_workers=args.download_workers,
        download_retries_per_scheme=args.download_retries_per_scheme,
        download_retry_wait_seconds=args.download_retry_wait_seconds,
        chemberta_rep=generate_chemberta,
        backup_old_reps=backup_old_reps,
    )
    print("[INFO] ZINC database generation completed.")


if __name__ == "__main__":
    main()
