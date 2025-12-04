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
       - build the ZINC ligand index and Morgan fingerprints.
"""

import argparse
from pathlib import Path
from typing import Optional

from generate_databases.zinc_db import generate_zinc_database


# ---------------------------------------------------------------------------
# Helper: download the ZINC URL file from Hugging Face if missing
# ---------------------------------------------------------------------------

def download_zinc_uri_file_if_missing(
    zinc_data_dir: Path,
    hf_repo_id: str = "gschottlender/LigQ_2",
    hf_subpath: str = "zinc/ZINC-downloader-2D-smi.uri",
    filename: str = "ZINC-downloader-2D-smi.uri",
) -> Path:
    """
    Ensure that the ZINC URL file exists under `zinc_data_dir`.
    If not, download it from a Hugging Face repository.

    Parameters
    ----------
    zinc_data_dir : Path
        Directory where the ZINC-related files live (e.g. <output_dir>/zinc).
    hf_repo_id : str
        Hugging Face repository ID that contains the URL file.
    hf_subpath : str
        Relative path to the URL file inside the repository.
    filename : str
        Name of the local file (by default 'ZINC-downloader-2D-smi.uri').

    Returns
    -------
    Path
        Path to the local URL file.
    """
    zinc_data_dir.mkdir(parents=True, exist_ok=True)
    local_path = zinc_data_dir / filename

    if local_path.exists():
        print(f"[INFO] ZINC URL file already present: {local_path}")
        return local_path

    print(
        "[INFO] ZINC URL file not found. "
        f"Downloading from Hugging Face repo '{hf_repo_id}' "
        f"subpath '{hf_subpath}'..."
    )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download the ZINC URL file but is "
            "not installed. Please install it with:\n\n"
            "    pip install huggingface_hub\n"
        )

    # Download the file from the specified repo/subpath
    repo_file_path = hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_subpath,
        local_dir=zinc_data_dir,
        local_dir_use_symlinks=False,
    )

    # If hf_hub_download stored it with an extra path, rename/move if needed
    repo_file_path = Path(repo_file_path)
    if repo_file_path.name != filename:
        # Move/rename to the expected filename
        repo_file_path.rename(local_path)
    else:
        local_path = repo_file_path

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

    return parser.parse_args(args)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    temp_data_dir = Path(args.temp_data_dir).resolve()

    print(f"[INFO] Output directory       : {output_dir}")
    print(f"[INFO] Temporary data directory: {temp_data_dir}")

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
    )
    print("[INFO] ZINC database generation completed.")


if __name__ == "__main__":
    main()
