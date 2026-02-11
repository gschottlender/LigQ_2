from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional
import argparse
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from compound_processing.zinc_search import get_zinc_ligands
from compound_processing.compound_helpers import LigandStore  # kept for type/context


# ---------------------------------------------------------------------------
# 1) Protein-domain mapping
# ---------------------------------------------------------------------------


def build_protein_domains_table(
    data_dir: str | Path = "databases",
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    """
    Build the protein-domain mapping table from the merged binding dataset.

    Input file (fixed location):
        <data_dir>/merged_databases/binding_data_merged.parquet

    Output file (fixed name):
        <results_dir>/protein_domains.parquet

    Parameters
    ----------
    data_dir : str or Path
        Directory containing the merged_databases/ subfolder.
    results_dir : str or Path, default "results"
        Directory where the final protein_domains.parquet will be written.

    Returns
    -------
    pd.DataFrame
        DataFrame with unique (uniprot_id, pfam_id) pairs.
    """
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    input_path = data_dir / "merged_databases" / "binding_data_merged.parquet"
    output_path = results_dir / "protein_domains.parquet"

    results_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read merged binding data
    df = pd.read_parquet(input_path)

    # 2) Keep only uniprot_id and pfam_id
    required_cols = ["uniprot_id", "pfam_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in binding data: {missing}")

    df_dom = df[required_cols].copy()

    # 3) Clean rows and types
    df_dom = df_dom.dropna(subset=["uniprot_id", "pfam_id"])
    df_dom["uniprot_id"] = df_dom["uniprot_id"].astype(str)
    df_dom["pfam_id"] = df_dom["pfam_id"].astype(str)

    # 4) Drop duplicated pairs
    df_dom = df_dom.drop_duplicates(subset=["uniprot_id", "pfam_id"])

    # 5) Sort (optional but tidy)
    df_dom = df_dom.sort_values(["uniprot_id", "pfam_id"]).reset_index(drop=True)

    # 6) Save the final mapping table
    df_dom.to_parquet(output_path, index=False)

    return df_dom


# ---------------------------------------------------------------------------
# 2) Known-binding aggregation
# ---------------------------------------------------------------------------


def _first_non_null(values: pd.Series) -> Any:
    """
    Return the first non-null value in the series.
    If all values are null, return None.
    """
    for v in values:
        if v is None:
            continue
        # Handle pandas NA / NaN
        if isinstance(v, float) and np.isnan(v):
            continue
        return v
    return None


def collapse_by_domain(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse rows so that each (uniprot_id, chem_comp_id, pfam_id, source)
    appears once, aggregating pdb_id into a list and merging curated fields.

    Result:
      - One row per (uniprot_id, chem_comp_id, pfam_id, source).
      - pdb_ids: list of unique PDB IDs.
      - pchembl: max value for the group.
      - mechanism, activity_comment, curation_method: first non-null scalar.
    """

    def _unique_pdb_list(values: pd.Series) -> List[str]:
        uniq: List[str] = []
        seen: set[str] = set()
        for v in values:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if v not in seen:
                seen.add(v)
                uniq.append(v)
        return uniq

    grouped = (
        df.groupby(["uniprot_id", "chem_comp_id", "pfam_id", "source"], as_index=False)
        .agg(
            pdb_ids=("pdb_id", _unique_pdb_list),
            pchembl=("pchembl", "max"),
            mechanism=("mechanism", _first_non_null),
            activity_comment=("activity_comment", _first_non_null),
            curation_method=("curation_method", _first_non_null),
        )
    )

    return grouped


def collapse_binding_sites(df_domain_collapsed: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse domain-level rows into one row per (uniprot_id, chem_comp_id, source),
    aggregating:
      - binding_sites: list of Pfam IDs where the ligand binds,
      - pdb_ids: merged list of all PDB IDs for that protein-ligand-source,
      - pchembl: max value,
      - mechanism, activity_comment, curation_method: first non-null scalar.
    """

    def _merge_pdb_lists(series: pd.Series) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for v in series:
            if v is None:
                continue
            # v can be list/tuple/set/ndarray or scalar
            if isinstance(v, (list, tuple, set, np.ndarray)):
                iterable = list(v)
            else:
                iterable = [v]
            for x in iterable:
                if x is None:
                    continue
                if isinstance(x, float) and np.isnan(x):
                    continue
                if x not in seen:
                    seen.add(x)
                    merged.append(x)
        return merged

    def _binding_sites_list(s: pd.Series) -> List[str]:
        sites: List[str] = []
        seen: set[str] = set()
        for v in s:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if v not in seen:
                seen.add(v)
                sites.append(v)
        # Sort for deterministic output
        return sorted(sites)

    grouped = (
        df_domain_collapsed.groupby(["uniprot_id", "chem_comp_id", "source"], as_index=False)
        .agg(
            binding_sites=("pfam_id", _binding_sites_list),
            pdb_ids=("pdb_ids", _merge_pdb_lists),
            pchembl=("pchembl", "max"),
            mechanism=("mechanism", _first_non_null),
            activity_comment=("activity_comment", _first_non_null),
            curation_method=("curation_method", _first_non_null),
        )
    )

    return grouped


def build_known_binding_data(df_binding: pd.DataFrame) -> pd.DataFrame:
    """
    Build a curated known-binding table with one row per
    (uniprot_id, chem_comp_id, source), including:
      - binding_sites: list of Pfam domains where the ligand binds,
      - pdb_ids: list of PDB IDs supporting the interaction,
      - curated fields aggregated across sources.

    Parameters
    ----------
    df_binding : pd.DataFrame
        Raw binding table with columns including:
            - uniprot_id
            - chem_comp_id
            - pfam_id
            - source
            - pdb_id
            - pchembl
            - mechanism
            - activity_comment
            - curation_method

    Returns
    -------
    pd.DataFrame
        Aggregated known-binding table with domain and PDB information collapsed.
    """
    # 1) Collapse by (uniprot_id, chem_comp_id, pfam_id, source)
    #    → produces lists of pdb_ids and merges curated fields
    df_by_domain = collapse_by_domain(df_binding)

    # 2) Collapse by (uniprot_id, chem_comp_id, source)
    #    → produces binding_sites list and merges everything
    df_known = collapse_binding_sites(df_by_domain)

    return df_known


# ---------------------------------------------------------------------------
# 3) Add SMILES and save known-binding table
# ---------------------------------------------------------------------------


def add_smiles_to_known_binding(
    known_binding: pd.DataFrame,
    pdb_chembl_smiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add a 'smiles' column to the known_binding table by mapping chem_comp_id
    to SMILES from the pdb_chembl_smiles table.

    This function is designed to be memory-efficient:
      - builds a minimal lookup (chem_comp_id -> smiles)
      - uses Series.map to add a single column
      - does not perform a full DataFrame merge with extra columns.

    Any ligands without SMILES are dropped (they are not usable for similarity).

    Parameters
    ----------
    known_binding : pd.DataFrame
        Curated binding table with at least a 'chem_comp_id' column.
    pdb_chembl_smiles : pd.DataFrame
        Table with at least:
            - 'chem_comp_id'
            - 'smiles'

    Returns
    -------
    pd.DataFrame
        Copy of known_binding with an additional 'smiles' column and
        rows without SMILES removed.
    """
    kb = known_binding.copy()

    # 1) Build a minimal lookup: chem_comp_id -> smiles
    smiles_lookup = (
        pdb_chembl_smiles[["chem_comp_id", "smiles"]]
        .dropna(subset=["chem_comp_id", "smiles"])
        .drop_duplicates(subset=["chem_comp_id"])
    )

    # Normalize types to string to avoid mismatches
    smiles_lookup["chem_comp_id"] = smiles_lookup["chem_comp_id"].astype(str)
    kb["chem_comp_id"] = kb["chem_comp_id"].astype(str)

    smiles_lookup = smiles_lookup.set_index("chem_comp_id")["smiles"]

    # 2) Map chem_comp_id -> smiles into known_binding
    kb["smiles"] = kb["chem_comp_id"].map(smiles_lookup)

    # 3) Drop ligands without SMILES (not usable for similarity search)
    kb = kb.dropna(subset=["smiles"]).reset_index(drop=True)

    return kb


def save_known_binding_table(
    known_binding: pd.DataFrame,
    results_dir: str | Path = "results",
) -> Path:
    """
    Save the final curated known-binding table (already enriched with SMILES)
    to <results_dir>/known_binding_data.parquet.

    Parameters
    ----------
    known_binding : pd.DataFrame
        Final curated table containing:
            - uniprot_id
            - chem_comp_id
            - binding_sites
            - pdb_ids
            - source
            - pchembl
            - mechanism
            - activity_comment
            - curation_method
            - smiles
    results_dir : str or Path, default "results"
        Output directory where the Parquet will be written.

    Returns
    -------
    Path
        Full path to the saved Parquet file.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    output_path = results_dir / "known_binding_data.parquet"
    known_binding.to_parquet(output_path, index=False)

    return output_path


# ---------------------------------------------------------------------------
# 4) Predicted ZINC binding table (incremental, memory-efficient)
# ---------------------------------------------------------------------------


def build_predicted_zinc_binding_data(
    known_binding: pd.DataFrame,
    store_pdb_chembl: LigandStore,
    rep_pdb_chembl,
    store_zinc: LigandStore,
    rep_zinc,
    search_rep_ref=None,
    search_rep_zinc=None,
    search_metric: str = "tanimoto",
    results_dir: str | Path = "results",
    resume: bool = True,
    max_proteins: Optional[int] = None,
) -> None:
    """
    Build the global predicted ZINC binding table in an incremental,
    memory-efficient way.

    For each protein (uniprot_id) present in `known_binding`:
      - run get_zinc_ligands(...) to get ZINC hits
      - add a 'uniprot_id' column
      - write the result as a new row group in
        <results_dir>/predicted_zinc_binding_data.parquet

    'known_binding' plays two roles:
      - defines which proteins to process,
      - acts as a curated binding table (pdb_chembl_binding_data) filtered
        to ligands with SMILES, avoiding unusable compounds.

    The function also maintains a small JSON file with processed proteins
    so that the pipeline can be resumed if interrupted.

    Parameters
    ----------
    known_binding : pd.DataFrame
        Curated known-binding table, must contain at least 'uniprot_id'.
    store_pdb_chembl : LigandStore
        Ligand store for PDB+ChEMBL ligands.
    rep_pdb_chembl : Representation
        Fingerprint/embedding representation for PDB+ChEMBL ligands.
    store_zinc : LigandStore
        Ligand store for ZINC ligands.
    rep_zinc : Representation
        Fingerprint/embedding representation for ZINC ligands.
    search_rep_ref : Representation, optional
        Optional representation to use specifically for similarity search queries.
        If None, `rep_pdb_chembl` is used.
    search_rep_zinc : Representation, optional
        Optional representation to use specifically for ZINC similarity search.
        If None, `rep_zinc` is used.
    search_metric : str, default "tanimoto"
        Similarity metric for the ZINC search backend ("tanimoto" or "cosine").
    results_dir : str or Path, default "results"
        Directory where 'predicted_zinc_binding_data.parquet' and the progress
        JSON file will be written.
    resume : bool, default True
        If True, skip proteins that are already listed as processed in the
        progress file and preserve existing Parquet content.
    max_proteins : int, optional
        If given, process at most this number of proteins (useful for tests).

    Returns
    -------
    None
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = results_dir / "predicted_zinc_binding_data.parquet"
    progress_path = results_dir / "predicted_zinc_binding_progress.json"

    # 1) Proteins to process (preserve first-appearance order)
    all_uniprot = known_binding["uniprot_id"].astype(str)
    uniprot_order = list(dict.fromkeys(all_uniprot))

    if max_proteins is not None:
        uniprot_order = uniprot_order[:max_proteins]

    # 2) Already processed proteins (for resume)
    if resume and progress_path.exists():
        with open(progress_path, "r") as f:
            processed_proteins = set(json.load(f))
    else:
        processed_proteins = set()

    # 3) Prepare ParquetWriter (incremental writes)
    writer: Optional[pq.ParquetWriter] = None
    temp_path: Optional[Path] = None

    if parquet_path.exists() and resume:
        # There is an existing Parquet file and we want to resume.
        # We copy its row groups into a temporary file, then append new
        # results, and finally replace the original file.
        pf = pq.ParquetFile(parquet_path)
        existing_schema = pf.schema_arrow

        temp_path = parquet_path.with_suffix(".tmp.parquet")
        writer = pq.ParquetWriter(temp_path.as_posix(), existing_schema)

        # Copy existing row groups one by one (streaming, no full file in RAM).
        for rg_idx in range(pf.num_row_groups):
            table_rg = pf.read_row_group(rg_idx)
            writer.write_table(table_rg)

    # 4) Main loop over proteins
    for i, prot in enumerate(uniprot_order, start=1):
        print(
            f"\r[INFO] Processing protein {i}/{len(uniprot_order)}: {prot}          ",
            end="",
            flush=True,
        )

        # Skip if already processed
        if prot in processed_proteins:
            continue

        # 4.1) Get ZINC ligands for this protein
        zinc_ligands = get_zinc_ligands(
            prot=prot,
            pdb_chembl_binding_data=known_binding,  # curated table used as binding data
            store_pdb_chembl=store_pdb_chembl,
            rep_pdb_chembl=rep_pdb_chembl,
            store_zinc=store_zinc,
            rep_zinc=rep_zinc,
            search_rep_ref=search_rep_ref,
            search_rep_zinc=search_rep_zinc,
            search_metric=search_metric,
        )

        # No hits: still mark protein as processed
        if zinc_ligands is None or zinc_ligands.empty:
            processed_proteins.add(prot)
            with open(progress_path, "w") as f:
                json.dump(sorted(processed_proteins), f)
            continue

        # 4.2) Add uniprot_id column
        zinc_ligands = zinc_ligands.copy()
        zinc_ligands.insert(0, "uniprot_id", prot)

        # 4.3) Convert to pyarrow.Table and write as a row group
        table = pa.Table.from_pandas(zinc_ligands, preserve_index=False)

        if writer is None:
            # No existing Parquet (or resume=False) → create a new writer
            writer = pq.ParquetWriter(parquet_path.as_posix(), table.schema)
        else:
            # Align schema if necessary (same column order)
            schema_names = writer.schema.names
            if table.schema.names != schema_names:
                table = table.select(schema_names)

        writer.write_table(table)

        # 4.4) Update progress
        processed_proteins.add(prot)
        with open(progress_path, "w") as f:
            json.dump(sorted(processed_proteins), f)

    # 5) Close writer and, if needed, replace original file with temporary
    if writer is not None:
        writer.close()
        if temp_path is not None:
            os.replace(temp_path, parquet_path)

    print()
    print("[INFO] Finished writing predicted_zinc_binding_data.")


# ---------------------------------------------------------------------------
# 5) Command-line interface (CLI)
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Arguments
    ---------
    --data-dir / -d : str
        Root directory containing all input databases.
        Default: 'databases'

    --output-dir / -o : str | None
        Directory where result tables are written.
        Default: <data_dir>/results_databases

    --regenerate : flag
        Forces full regeneration of the selected results directory
        (either --output-dir or <data_dir>/results_databases).
        If it already exists, it is moved into a backup folder
        (<results_dir>_backup, <results_dir>_backup1, ...)

    --max-proteins : int
        If given, process at most this number of proteins in the ZINC search.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the protein-domain mapping table, the known_binding table, "
            "and the predicted_zinc_binding_data table using preprocessed databases."
        )
    )

    parser.add_argument(
        "-d", "--data-dir",
        default="databases",
        help="Root directory containing input databases (default: %(default)s)",
    )

    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help=(
            "Directory where result tables will be written. "
            "Default is <data_dir>/results_databases."
        ),
    )

    parser.add_argument(
        "--regenerate",
        action="store_true",
        help=(
            "Rebuild all results from scratch. If <data_dir>/results_databases "
            "already exists, it will be moved into a backup directory "
            "before regenerating."
        ),
    )

    parser.add_argument(
        "--max-proteins",
        type=int,
        default=None,
        help=(
            "If given, process at most this number of proteins in the ZINC search "
            "(useful for quick tests)."
        ),
    )

    parser.add_argument(
        "--search-representation",
        default="morgan_1024_r2",
        help=(
            "Representation name to use for ZINC similarity search. "
            "Default keeps legacy behavior with Morgan fingerprints "
            "('morgan_1024_r2'). Example for embeddings: "
            "'chemberta_zinc_base_768'."
        ),
    )

    parser.add_argument(
        "--search-metric",
        choices=["tanimoto", "cosine"],
        default="tanimoto",
        help=(
            "Similarity metric used for ZINC search. "
            "Default is 'tanimoto'. For embedding representations, use 'cosine'."
        ),
    )

    return parser.parse_args()


def _backup_results_dir(results_dir: Path) -> None:
    """
    If results_dir exists, move it to a backup directory.
    Backup naming scheme:
        results_dir_backup
        results_dir_backup1
        results_dir_backup2
        ...
    Never overwrites an existing backup.
    """
    if not results_dir.exists():
        return

    base = results_dir
    backup = base.with_name(base.name + "_backup")

    # If backup already exists, add numeric suffixes
    counter = 1
    while backup.exists():
        backup = base.with_name(f"{base.name}_backup{counter}")
        counter += 1

    print(f"[INFO] Backing up existing results directory: {results_dir} -> {backup}")
    shutil.move(str(results_dir), str(backup))


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    if args.output_dir is None:
        results_dir = data_dir / "results_databases"
    else:
        results_dir = Path(args.output_dir)

    # ----------------------------------------------------------------------
    # 0) Handle --regenerate (backup + clean start)
    # ----------------------------------------------------------------------
    if args.regenerate:
        _backup_results_dir(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        resume = False  # Force full regeneration of ZINC prediction tables
        print("[INFO] Regenerating all results from scratch (resume = False).")
    else:
        # Normal mode: create results_dir if missing but do not delete anything
        results_dir.mkdir(parents=True, exist_ok=True)
        resume = True
        print("[INFO] Normal mode: continuing or creating results (resume = True).")

    # ----------------------------------------------------------------------
    # 1) Load merged binding and SMILES tables
    # ----------------------------------------------------------------------
    binding_path = data_dir / "merged_databases" / "binding_data_merged.parquet"
    smiles_path = data_dir / "merged_databases" / "ligs_smiles_merged.parquet"

    if not binding_path.exists():
        raise FileNotFoundError(f"Binding data not found: {binding_path}")
    if not smiles_path.exists():
        raise FileNotFoundError(f"SMILES table not found: {smiles_path}")

    print(f"[INFO] Loading binding data from {binding_path}")
    pdb_chembl_binding_data = pd.read_parquet(binding_path)

    print(f"[INFO] Loading SMILES data from {smiles_path}")
    pdb_chembl_smiles = pd.read_parquet(smiles_path)

    # ----------------------------------------------------------------------
    # 2) Initialize LigandStore objects and load representations
    # ----------------------------------------------------------------------
    PDB_CHEMBL_ROOT = data_dir / "compound_data" / "pdb_chembl"
    ZINC_ROOT = data_dir / "compound_data" / "zinc"

    print(f"[INFO] Initializing LigandStore for PDB+ChEMBL at {PDB_CHEMBL_ROOT}")
    store_pdb_chembl = LigandStore(PDB_CHEMBL_ROOT)

    print(f"[INFO] Initializing LigandStore for ZINC at {ZINC_ROOT}")
    store_zinc = LigandStore(ZINC_ROOT)

    print("[INFO] Loading 'morgan_1024_r2' representation for PDB+ChEMBL")
    rep_pdb_chembl = store_pdb_chembl.load_representation("morgan_1024_r2")

    print("[INFO] Loading 'morgan_1024_r2' representation for ZINC")
    rep_zinc = store_zinc.load_representation("morgan_1024_r2")

    # Optional search representation/metric (defaults preserve legacy behavior)
    search_rep_name = args.search_representation
    search_metric = args.search_metric

    if search_rep_name == "morgan_1024_r2":
        search_rep_ref = rep_pdb_chembl
        search_rep_zinc = rep_zinc
    else:
        print(f"[INFO] Loading '{search_rep_name}' representation for PDB+ChEMBL")
        search_rep_ref = store_pdb_chembl.load_representation(search_rep_name)
        print(f"[INFO] Loading '{search_rep_name}' representation for ZINC")
        search_rep_zinc = store_zinc.load_representation(search_rep_name)

    print(
        "[INFO] ZINC search configuration -> "
        f"representation='{search_rep_name}', metric='{search_metric}'"
    )

    # ----------------------------------------------------------------------
    # 3) Build protein-domain mapping table
    # ----------------------------------------------------------------------
    print("[INFO] Building protein-domain mapping table...")
    df_dom = build_protein_domains_table(
        data_dir=data_dir,
        results_dir=results_dir,
    )
    prot_list = df_dom["uniprot_id"].unique()
    print(f"[INFO] Protein-domain table built ({len(prot_list)} unique proteins).")

    # ----------------------------------------------------------------------
    # 4) Build and save known_binding table
    # ----------------------------------------------------------------------
    print("[INFO] Building known_binding table...")
    known_binding = build_known_binding_data(pdb_chembl_binding_data)
    known_binding = add_smiles_to_known_binding(known_binding, pdb_chembl_smiles)

    print("[INFO] Saving known_binding_data.parquet")
    save_known_binding_table(known_binding, results_dir=results_dir)

    # ----------------------------------------------------------------------
    # 5) Build predicted_zinc_binding_data (incremental or full regen)
    # ----------------------------------------------------------------------
    print("[INFO] Building predicted_zinc_binding_data.parquet ...")
    build_predicted_zinc_binding_data(
        known_binding=known_binding,
        store_pdb_chembl=store_pdb_chembl,
        rep_pdb_chembl=rep_pdb_chembl,
        store_zinc=store_zinc,
        rep_zinc=rep_zinc,
        search_rep_ref=search_rep_ref,
        search_rep_zinc=search_rep_zinc,
        search_metric=search_metric,
        results_dir=results_dir,
        resume=resume,  # True = append; False = regenerate from scratch
        max_proteins=args.max_proteins,
    )

    print("[INFO] Finished.")


if __name__ == "__main__":
    main()
