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
from typing import Union, Tuple, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from compound_processing.compound_helpers import (
    build_ligand_index,
    build_morgan_representation,
    build_huggingface_representation
)


def download_zinc_database(
    data_dir: Union[str, Path] = "databases/zinc",
    temp_data_dir: Union[str, Path] = "temp_data/zinc_db",
    urls_filename: str = "ZINC-downloader-2D-smi.uri",
) -> None:
    data_dir = Path(data_dir)
    temp_data_dir = Path(temp_data_dir)
    temp_data_dir.mkdir(parents=True, exist_ok=True)

    urls_path = data_dir / urls_filename
    if not urls_path.exists():
        raise FileNotFoundError(f"URL file not found: {urls_path}")

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

    def _download_one(url: str) -> Tuple[str, bool]:
        filename = url.split("/")[-1]
        output_path = temp_data_dir / filename

        wget_cmd = [
            "wget",
            "--tries=10",
            "--retry-connrefused",
            "--waitretry=5",
            "--timeout=30",
            "--read-timeout=30",
            "--continue",
            "--no-dns-cache",
            "--quiet",
            "-O",
            str(output_path),
            url,
        ]

        result = subprocess.run(
            wget_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return filename, (result.returncode == 0)

    # "máximo workers" razonable para I/O:
    # - si devolvés os.cpu_count() suele estar bien
    # - pero para descargas, podés ir más alto sin romper nada.
    # Para no sorpresarte, uso min(32, os.cpu_count()+4) estilo stdlib.
    max_workers = min(32, (os.cpu_count() or 1) + 4)

    failed: List[str] = []
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_download_one, url) for url in urls]

        for fut in as_completed(futures):
            filename, ok = fut.result()
            done += 1

            # Compact progress (single line)
            print(f"\r[INFO] {done}/{total} Downloading {filename}...", end="", flush=True)

            if not ok:
                failed.append(filename)

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
    zinc_root = Path(zinc_root)
    output_parquet = Path(output_parquet)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    smi_files = sorted(zinc_root.rglob("*.smi"))
    if not smi_files:
        raise FileNotFoundError(f"No .smi files found under {zinc_root}")

    total_files = len(smi_files)
    writer: pq.ParquetWriter | None = None
    n_total = 0

    def _parse_one(idx: int, smi_path: Path) -> Tuple[int, Path, List[str], List[str]]:
        smiles_list: List[str] = []
        ids_list: List[str] = []

        with smi_path.open("r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    smiles_list.append(parts[0])
                    ids_list.append(parts[1])

        return idx, smi_path, smiles_list, ids_list

    # Lectura/parseo es I/O + un poco CPU; threads suele andar bien.
    # Si tus .smi están en SSD y el parseo domina, ProcessPool también sirve,
    # pero threads es más simple y estable con PyArrow en el main thread.
    max_workers = os.cpu_count() or 1  # "máximo" disponible

    pending: Dict[int, Tuple[Path, List[str], List[str]]] = {}
    next_to_write = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_parse_one, idx, smi_path)
            for idx, smi_path in enumerate(smi_files)
        ]

        for fut in as_completed(futures):
            idx, smi_path, smiles_list, ids_list = fut.result()
            pending[idx] = (smi_path, smiles_list, ids_list)

            # Escribimos todo lo consecutivo disponible en orden
            while next_to_write in pending:
                smi_path_w, smiles_w, ids_w = pending.pop(next_to_write)

                if not smiles_w:
                    # Warning en el mismo orden que el serial
                    print(f"\n[WARN] {smi_path_w} is empty or has no valid lines.")
                    next_to_write += 1
                    continue

                df_chunk = pd.DataFrame({"chem_comp_id": ids_w, "smiles": smiles_w})
                table = pa.Table.from_pandas(df_chunk, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_parquet,
                        table.schema,
                        compression="snappy",
                    )

                writer.write_table(table)
                n_total += len(df_chunk)

                # Progress igual que antes (idx/total_files y acumulado)
                print(
                    f"\r[INFO] Processing files: {next_to_write+1}/{total_files} "
                    f"({n_total} ligands accumulated)",
                    end="",
                    flush=True,
                )

                next_to_write += 1

    print()
    if writer is not None:
        writer.close()

    print(f"[INFO] Done. Total ligands processed: {n_total}")
    print(f"[INFO] Output Parquet written to: {output_parquet}")
    return output_parquet

# Agregar posibilidad de agregar chemberta (funcion clonada pero con chemberta)
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

def build_zinc_chemberta_compound_database(
    root: Union[str, Path] = "databases/compound_data/zinc",
    batch_size: int = 14,
    rep_name: str = "chemberta_zinc_base_768",
) -> Dict[str, Path]:
    """
    End-to-end pipeline to build the ZINC ChemBERTa representation.

      1) Asume que bajo `root` ya existe un ligands.parquet con:
           - lig_idx (índice denso)
           - chem_comp_id
           - smiles
           - InChIKey
      2) Construye la representación ChemBERTa como memmap
         (rep_name.dat + rep_name.meta.json) usando esos ligandos.

    Parameters
    ----------
    root : str or Path
        Root directory where ligands.parquet and reps/ live.
    batch_size : int
        Batch size for ChemBERTa embedding calculation.
    rep_name : str
        Name of the representation (e.g. 'chemberta_zinc_base_768').

    Returns
    -------
    dict
        Paths to:
          - 'ligands'   : ligands.parquet (ya existente)
          - 'rep_data'  : memmap .dat file with embeddings
          - 'rep_meta'  : .meta.json file with metadata about the representation
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    # Esperamos que ligands.parquet ya exista (creado por build_zinc_compound_database)
    ligands_path = root / "ligands.parquet"
    if not ligands_path.exists():
        raise FileNotFoundError(
            f"Expected ligands.parquet at {ligands_path} – "
            "run build_zinc_compound_database (Morgan) first."
        )

    # ------------------------------------------------------------------
    # 1) Build ChemBERTa representation as a memmap
    # ------------------------------------------------------------------
    build_huggingface_representation(
        root=root,
        n_bits=768,         # o el nombre real del parámetro dim/size en tu función
        batch_size=batch_size,
        name=rep_name,
    )

    reps_dir = root / "reps"
    data_path = reps_dir / f"{rep_name}.dat"
    meta_path = reps_dir / f"{rep_name}.meta.json"

    print(f"[INFO] ChemBERTa representation '{rep_name}' generated at:")
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
    chemberta_rep: bool = True,
) -> Dict[str, Path]:
    """
    High-level helper to generate the full ZINC compound database in one call.

    Steps:
      1) Download ZINC .smi chunks (if not already present) into `zinc_temp_dir`.
      2) Build a unified ligands_smiles.parquet in `zinc_data_dir`.
      3) Build the ZINC compound database under `compound_root`:
           - ligands.parquet
           - Morgan fingerprints memmap (rep_name.dat + rep_name.meta.json)
      4) Optionally build ChemBERTa embeddings memmap under the same root.

    Returns
    -------
    dict
        At least the dictionary returned by `build_zinc_compound_database`:
          - 'ligands'
          - 'rep_data'
          - 'rep_meta'
        And, if chemberta_rep is True, you can extend it with ChemBERTa paths.
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

    # 3) Build the ZINC compound database (ligands + Morgan fingerprints)
    print('Building ZINC Morgan compound database')
    result_paths = build_zinc_compound_database(
        ligands_smiles_parquet=ligands_smiles_path,
        root=compound_root,
        n_bits=n_bits,
        radius=radius,
        batch_size=batch_size,
        rep_name=rep_name,
    )

    # 4) Build ChemBERTa compound database (re-using the same ligands.parquet)
    if chemberta_rep:
        print('Building ZINC ChemBERTa compound database')
        chemberta_paths = build_zinc_chemberta_compound_database(
            root=compound_root,
            batch_size=14,
            rep_name="chemberta_zinc_base_768",
        )
        # opcional: extender el dict de salida con info de ChemBERTa
        result_paths["chemberta_rep_data"] = chemberta_paths["rep_data"]
        result_paths["chemberta_rep_meta"] = chemberta_paths["rep_meta"]

    return result_paths

