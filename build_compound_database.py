#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

import pandas as pd

from compound_processing.compound_helpers import (
    build_ligand_index,
    build_morgan_representation,
)
from progress_reporting import ProgressEmitter


COMMON_ID_COLUMNS = [
    "chem_comp_id",
    "compound",
    "compound_id",
    "id",
]

COMMON_SMILES_COLUMNS = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "canonical_SMILES",
]

_STAGING_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_BUILD_JOB_MARKER = ".ligq_build_job"


def _infer_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".tsv", ".txt"}:
        return "tsv"
    if suffix == ".smi":
        return "smi"
    if suffix == ".parquet":
        return "parquet"
    raise ValueError(
        f"Could not infer file format from extension '{path.suffix}'. "
        "Use --file-format explicitly."
    )


def _read_table(input_file: Path, file_format: str | None, delimiter: str | None) -> pd.DataFrame:
    resolved_format = (file_format or _infer_format(input_file)).strip().lower()
    if resolved_format == "csv":
        return pd.read_csv(input_file, sep="," if delimiter is None else delimiter)
    if resolved_format == "tsv":
        return pd.read_csv(input_file, sep="\t" if delimiter is None else delimiter)
    if resolved_format == "smi":
        if delimiter is None:
            df = pd.read_csv(
                input_file,
                sep=r"\s+",
                header=None,
                usecols=[0, 1],
                names=["smiles", "chem_comp_id"],
                engine="python",
            )
        else:
            df = pd.read_csv(
                input_file,
                sep=delimiter,
                header=None,
                usecols=[0, 1],
                names=["smiles", "chem_comp_id"],
            )
        if df.shape[1] < 2:
            raise ValueError(
                ".smi inputs must contain at least two columns: SMILES and compound_id."
            )
        return df
    if resolved_format == "parquet":
        return pd.read_parquet(input_file)
    raise ValueError(f"Unsupported file format: {resolved_format}")


def _infer_column(columns: list[str], candidates: list[str], label: str) -> str:
    lookup = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    raise ValueError(
        f"Could not infer {label} column from input columns: {columns}. "
        f"Pass --{label}-column explicitly."
    )


def normalize_compound_table(
    df: pd.DataFrame,
    id_column: str | None,
    smiles_column: str | None,
) -> pd.DataFrame:
    columns = list(df.columns)
    resolved_id_column = id_column or _infer_column(columns, COMMON_ID_COLUMNS, "id")
    resolved_smiles_column = smiles_column or _infer_column(columns, COMMON_SMILES_COLUMNS, "smiles")

    if resolved_id_column not in df.columns:
        raise ValueError(f"ID column '{resolved_id_column}' not found in input table.")
    if resolved_smiles_column not in df.columns:
        raise ValueError(f"SMILES column '{resolved_smiles_column}' not found in input table.")

    out = df[[resolved_id_column, resolved_smiles_column]].copy()
    out = out.rename(columns={resolved_id_column: "chem_comp_id", resolved_smiles_column: "smiles"})
    out = out.dropna(subset=["chem_comp_id", "smiles"]).copy()
    out["chem_comp_id"] = out["chem_comp_id"].astype(str).str.strip()
    out["smiles"] = out["smiles"].astype(str).str.strip()
    out = out[(out["chem_comp_id"] != "") & (out["smiles"] != "")]
    out = out.drop_duplicates(subset=["chem_comp_id"], keep="first").reset_index(drop=True)

    if out.empty:
        raise ValueError("No valid compounds remained after filtering null/empty rows.")
    return out


def build_compound_database(
    input_file: str | Path,
    output_dir: str | Path = "databases",
    base_name: str = "custom",
    file_format: str | None = None,
    id_column: str | None = None,
    smiles_column: str | None = None,
    delimiter: str | None = None,
    default_rep_batch_size: int = 10000,
    default_rep_n_jobs: int | None = None,
    default_rep_chunksize: int = 500,
    progress: ProgressEmitter | None = None,
    staging_token: str | None = None,
) -> Path:
    input_path = Path(input_file)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input compound table not found: {input_path}")

    final_root = Path(output_dir) / "compound_data" / base_name
    root = final_root
    if staging_token is not None:
        if not _STAGING_TOKEN_RE.fullmatch(staging_token):
            raise ValueError(
                "staging_token must contain only letters, digits, underscores, and hyphens."
            )
        root = final_root.parent / f".{base_name}.building.{staging_token}"
        if root.exists():
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    if staging_token is not None:
        (root / _BUILD_JOB_MARKER).write_text(staging_token, encoding="utf-8")

    if progress:
        progress.emit(
            step="reading_input",
            label="Reading compound file",
            step_index=1,
            step_count=4,
            percent=1,
            context=base_name,
        )
    raw_df = _read_table(input_file=input_path, file_format=file_format, delimiter=delimiter)
    final_ligs = normalize_compound_table(
        df=raw_df,
        id_column=id_column,
        smiles_column=smiles_column,
    )

    n_ligands = len(final_ligs)
    if progress:
        progress.emit(
            step="indexing_compounds",
            label="Generating InChIKeys and ligand index",
            step_index=2,
            step_count=4,
            percent=5,
            current=0,
            total=n_ligands,
            unit="compounds",
            context=base_name,
        )

    def index_progress(current: int, total: int) -> None:
        if progress:
            progress.emit(
                step="indexing_compounds",
                label="Generating InChIKeys and ligand index",
                step_index=2,
                step_count=4,
                percent=5 + round(current / total * 45),
                current=current,
                total=total,
                unit="compounds",
                context=base_name,
            )

    build_ligand_index(
        final_ligs=final_ligs,
        root=root,
        progress_callback=index_progress,
    )

    if progress:
        progress.emit(
            step="building_fingerprints",
            label="Computing Morgan fingerprints",
            step_index=3,
            step_count=4,
            percent=52,
            current=0,
            total=n_ligands,
            unit="compounds",
            context=base_name,
        )

    def fingerprint_progress(current: int, total: int) -> None:
        if progress:
            progress.emit(
                step="building_fingerprints",
                label="Computing Morgan fingerprints",
                step_index=3,
                step_count=4,
                percent=52 + round(current / total * 45),
                current=current,
                total=total,
                unit="compounds",
                context=base_name,
            )

    build_morgan_representation(
        root=root,
        n_bits=1024,
        radius=2,
        batch_size=default_rep_batch_size,
        name="morgan_1024_r2",
        n_jobs=default_rep_n_jobs,
        chunksize=default_rep_chunksize,
        progress_callback=fingerprint_progress,
    )
    if progress:
        progress.emit(
            step="finalizing",
            label="Finalizing compound database",
            step_index=4,
            step_count=4,
            percent=99,
            context=base_name,
        )
    if staging_token is not None:
        if final_root.exists():
            raise FileExistsError(
                f"Cannot publish database '{base_name}': {final_root} already exists."
            )
        os.replace(root, final_root)
        return final_root
    return root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import a compound table into the LigQ_2 internal compound_data format "
            "and build the default Morgan representation."
        )
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Input CSV/TSV/SMI/Parquet file.",
    )
    parser.add_argument(
        "--output-dir",
        default="databases",
        help="Root output directory where compound_data/<base-name>/ will be created.",
    )
    parser.add_argument(
        "--base-name",
        required=True,
        help="Logical base name. Output will be written under compound_data/<base-name>/.",
    )
    parser.add_argument(
        "--file-format",
        choices=["csv", "tsv", "smi", "parquet"],
        default=None,
        help="Optional input file format. If omitted, inferred from the file extension.",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Column name containing the compound identifier.",
    )
    parser.add_argument(
        "--smiles-column",
        default=None,
        help="Column name containing the SMILES string.",
    )
    parser.add_argument(
        "--delimiter",
        default=None,
        help="Optional delimiter override for CSV/TSV text inputs.",
    )
    parser.add_argument(
        "--default-rep-batch-size",
        type=int,
        default=10000,
        help="Batch size for the default Morgan representation build.",
    )
    parser.add_argument(
        "--default-rep-n-jobs",
        type=int,
        default=None,
        help="Optional CPU worker count for the default Morgan representation build.",
    )
    parser.add_argument(
        "--default-rep-chunksize",
        type=int,
        default=500,
        help="Chunk size for multiprocessing imap during the default Morgan build.",
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--staging-token",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress = ProgressEmitter(enabled=args.progress_json)
    root = build_compound_database(
        input_file=args.input_file,
        output_dir=args.output_dir,
        base_name=args.base_name,
        file_format=args.file_format,
        id_column=args.id_column,
        smiles_column=args.smiles_column,
        delimiter=args.delimiter,
        default_rep_batch_size=args.default_rep_batch_size,
        default_rep_n_jobs=args.default_rep_n_jobs,
        default_rep_chunksize=args.default_rep_chunksize,
        progress=progress,
        staging_token=args.staging_token,
    )
    print(f"[INFO] Compound database created at: {root}")


if __name__ == "__main__":
    main()
