"""
zinc_db.py

Utilities to:
  1) Download the ZINC SMILES database from a list of URLs.
  2) Build a unified ligands_smiles.parquet file.
  3) Build a ZINC compound database (ligands.parquet + Morgan fingerprints memmap).

This module is designed to be imported and its functions called from other scripts.
"""

import subprocess
from pathlib import Path
from typing import Dict, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from compound_processing.compound_helpers import (
    build_ligand_index,
    build_morgan_representation,
)


def download_zinc_database(
    data_dir: Union[str, Path] = "databases/zinc",
    temp_data_dir: Union[str, Path] = "temp_data/zinc_db",
    urls_filename: str = "ZINC-downloader-2D-smi.uri",
) -> None:
    """
    Download the ZINC database from a list of URLs using wget.

    The function is intentionally quiet:
      - shows only a compact single-line progress indicator
      - hides normal wget output
      - reports only failed downloads and a final summary

    Parameters
    ----------
    data_dir : str or Path
        Directory where the URL list file is located.
    temp_data_dir : str or Path
        Temporary directory where the raw .smi files will be downloaded.
    urls_filename : str
        Name of the file containing the ZINC URLs (one per line).
    """
    data_dir = Path(data_dir)
    temp_data_dir = Path(temp_data_dir)
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    urls_path = data_dir / urls_filename

    if not urls_path.exists():
        raise FileNotFoundError(f"URL file not found: {urls_path}")

    # Read and clean URLs (ignore comments and empty lines, force https)
    with urls_path.open("r") as f:
        urls = [
            line.strip().replace("http://", "https://")
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    if not urls:
        print("[ERROR] No valid URLs found in the URL file.")
        return

    total = len(urls)
    failed = []

    for i, url in enumerate(urls, start=1):
        filename = url.split("/")[-1]
        output_path = temp_data_dir / filename

        # Compact progress line (overwritten on each iteration)
        print(f"\r[INFO] {i}/{total} Downloading {filename}...", end="", flush=True)

        wget_cmd = [
            "wget",
            "--tries=10",
            "--retry-connrefused",
            "--waitretry=5",
            "--timeout=30",
            "--read-timeout=30",
            "--continue",
            "--no-dns-cache",
            "--quiet",  # silence wget normal output
            "-O",
            str(output_path),
            url,
        ]

        result = subprocess.run(
            wget_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            failed.append(filename)

    # New line after finishing the loop
    print()

    if failed:
        print("[ERROR] The following downloads failed:")
        for f in failed:
            print(f"   - {f}")
    else:
        print("[INFO] All downloads completed successfully.")

    print("[INFO] ZINC download finished.")


def build_zinc_ligands_smiles_parquet(
    zinc_root: Union[str, Path],
    output_parquet: Union[str, Path],
) -> Path:
    """
    Traverse the ZINC directory tree recursively (looking for *.smi files) and
    build a single ligands_smiles.parquet file with:

        - chem_comp_id : ZINC ID (second column in each .smi line)
        - smiles       : SMILES string (first column in each .smi line)

    The function:
      - is quiet except for a compact progress line
      - warns about empty / invalid files
      - writes incrementally using a ParquetWriter

    Parameters
    ----------
    zinc_root : str or Path
        Root directory containing the downloaded ZINC .smi files (e.g., BA, BB, BC...).
    output_parquet : str or Path
        Path to the output Parquet file to be created.

    Returns
    -------
    Path
        Path to the generated Parquet file.
    """
    zinc_root = Path(zinc_root)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    smi_files = sorted(zinc_root.rglob("*.smi"))
    if not smi_files:
        raise FileNotFoundError(f"No .smi files found under {zinc_root}")

    total_files = len(smi_files)
    writer: pq.ParquetWriter | None = None
    n_total = 0

    for idx, smi_path in enumerate(smi_files, start=1):
        smiles_list = []
        ids_list = []

        # Read file and parse SMILES + ZINC ID
        with smi_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles_list.append(parts[0])
                    ids_list.append(parts[1])

        if not smiles_list:
            # Warn only once per empty / invalid file
            print(f"\n[WARN] {smi_path} is empty or has no valid lines.")
            continue

        df_chunk = pd.DataFrame(
            {
                "chem_comp_id": ids_list,
                "smiles": smiles_list,
            }
        )

        table = pa.Table.from_pandas(df_chunk, preserve_index=False)

        if writer is None:
            writer = pq.ParquetWriter(
                output_parquet,
                table.schema,
                compression="snappy",
            )

        writer.write_table(table)
        n_total += len(df_chunk)

        # Compact progress (single line updated in place)
        print(
            f"\r[INFO] Processing files: {idx}/{total_files} "
            f"({n_total} ligands accumulated)",
            end="",
            flush=True,
        )

    # Final newline after progress
    print()

    if writer is not None:
        writer.close()

    print(f"[INFO] Done. Total ligands processed: {n_total}")
    print(f"[INFO] Output Parquet written to: {output_parquet}")

    return output_parquet


def build_zinc_compound_database(
    ligands_smiles_parquet: Union[str, Path] = "databases/zinc/ligands_smiles.parquet",
    root: Union[str, Path] = "databases/compound_data/zinc",
    n_bits: int = 1024,
    radius: int = 2,
    batch_size: int = 10_000,
    rep_name: str = "morgan_1024_r2",
) -> Dict[str, Path]:
    """
    End-to-end pipeline to build the ZINC compound database:

      1) Read ligands_smiles.parquet (chem_comp_id, smiles).
      2) Build ligands.parquet with:
           - dense integer index: lig_idx
           - chem_comp_id and SMILES
           - InChIKey (via build_ligand_index).
      3) Build a Morgan fingerprint representation as a memmap
         (rep_name.dat + rep_name.meta.json).

    Parameters
    ----------
    ligands_smiles_parquet : str or Path
        Parquet file with columns ['chem_comp_id', 'smiles'] obtained from ZINC.
    root : str or Path
        Root directory where ligands.parquet and reps/ will be written.
    n_bits : int
        Number of bits for the Morgan fingerprint.
    radius : int
        Radius for the Morgan fingerprint.
    batch_size : int
        Batch size for fingerprint calculation.
    rep_name : str
        Name of the representation (e.g. 'morgan_1024_r2').

    Returns
    -------
    dict
        Paths to:
          - 'ligands'   : ligands.parquet
          - 'rep_data'  : memmap .dat file with fingerprints
          - 'rep_meta'  : .meta.json file with metadata about the representation
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    ligands_smiles_parquet = Path(ligands_smiles_parquet)
    if not ligands_smiles_parquet.exists():
        raise FileNotFoundError(f"File not found: {ligands_smiles_parquet}")

    # ------------------------------------------------------------------
    # 1) Read ligands_smiles.parquet
    # ------------------------------------------------------------------
    df = pd.read_parquet(ligands_smiles_parquet, columns=["chem_comp_id", "smiles"])
    n = len(df)
    if n == 0:
        raise ValueError("ligands_smiles.parquet is empty.")

    print(f"[INFO] ZINC ligands: {n} compounds read from {ligands_smiles_parquet}")

    # ------------------------------------------------------------------
    # 2) Build ligands.parquet with dense index and InChIKey
    # ------------------------------------------------------------------
    ligands_path = build_ligand_index(final_ligs=df, root=root)
    print(f"[INFO] ligands.parquet written to: {ligands_path}")

    # ------------------------------------------------------------------
    # 3) Build Morgan representation as a memmap
    # ------------------------------------------------------------------
    build_morgan_representation(
        root=root,
        n_bits=n_bits,
        radius=radius,
        batch_size=batch_size,
        name=rep_name,
    )

    reps_dir = root / "reps"
    data_path = reps_dir / f"{rep_name}.dat"
    meta_path = reps_dir / f"{rep_name}.meta.json"

    print(f"[INFO] Representation '{rep_name}' generated at:")
    print(f"       - data : {data_path}")
    print(f"       - meta : {meta_path}")

    return {
        "ligands": ligands_path,
        "rep_data": data_path,
        "rep_meta": meta_path,
    }


def generate_zinc_database(
    zinc_data_dir: Union[str, Path] = "databases/zinc",
    zinc_temp_dir: Union[str, Path] = "temp_data/zinc_db",
    urls_filename: str = "ZINC-downloader-2D-smi.uri",
    ligands_smiles_filename: str = "ligands_smiles.parquet",
    compound_root: Union[str, Path] = "databases/compound_data/zinc",
    n_bits: int = 1024,
    radius: int = 2,
    batch_size: int = 10_000,
    rep_name: str = "morgan_1024_r2",
) -> Dict[str, Path]:
    """
    High-level helper to generate the full ZINC compound database in one call.

    Steps:
      1) Download ZINC .smi chunks (if not already present) into `zinc_temp_dir`.
      2) Build a unified ligands_smiles.parquet in `zinc_data_dir`.
      3) Build the ZINC compound database under `compound_root`:
           - ligands.parquet
           - Morgan fingerprints memmap (rep_name.dat + rep_name.meta.json)

    This function is intended to be called from another script
    (no __main__ logic needed here).

    Parameters
    ----------
    zinc_data_dir : str or Path
        Directory where the URL file lives and where ligands_smiles.parquet will be written.
    zinc_temp_dir : str or Path
        Temporary directory where the raw ZINC .smi files are downloaded.
    urls_filename : str
        Name of the URL list file inside `zinc_data_dir`.
    ligands_smiles_filename : str
        Name of the output Parquet file for ligands+SMILES within `zinc_data_dir`.
    compound_root : str or Path
        Root directory for the final ZINC compound database (ligands.parquet + reps/).
    n_bits : int
        Number of bits for the Morgan fingerprints.
    radius : int
        Radius for the Morgan fingerprints.
    batch_size : int
        Batch size for fingerprint computation.
    rep_name : str
        Name of the representation to be created (e.g. 'morgan_1024_r2').

    Returns
    -------
    dict
        The same dictionary returned by `build_zinc_compound_database`, with paths to:
          - 'ligands'
          - 'rep_data'
          - 'rep_meta'
    """
    zinc_data_dir = Path(zinc_data_dir)
    zinc_temp_dir = Path(zinc_temp_dir)

    # 1) Download ZINC SMILES files
    download_zinc_database(
        data_dir=zinc_data_dir,
        temp_data_dir=zinc_temp_dir,
        urls_filename=urls_filename,
    )

    # 2) Build unified ligands_smiles.parquet
    ligands_smiles_path = zinc_data_dir / ligands_smiles_filename
    build_zinc_ligands_smiles_parquet(
        zinc_root=zinc_temp_dir,
        output_parquet=ligands_smiles_path,
    )

    # 3) Build the ZINC compound database (ligands + fingerprints)
    result_paths = build_zinc_compound_database(
        ligands_smiles_parquet=ligands_smiles_path,
        root=compound_root,
        n_bits=n_bits,
        radius=radius,
        batch_size=batch_size,
        rep_name=rep_name,
    )

    return result_paths
