from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from compound_processing.compound_helpers import LigandStore
from compound_processing.zinc_search import get_zinc_ligands


def build_protein_domains_table(
    data_dir: str | Path = "databases",
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    results_dir = Path(results_dir)

    input_path = data_dir / "merged_databases" / "binding_data_merged.parquet"
    output_path = results_dir / "protein_domains.parquet"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(input_path)
    required_cols = ["uniprot_id", "pfam_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in binding data: {missing}")

    df_dom = df[required_cols].copy()
    df_dom = df_dom.dropna(subset=["uniprot_id", "pfam_id"])
    df_dom["uniprot_id"] = df_dom["uniprot_id"].astype(str)
    df_dom["pfam_id"] = df_dom["pfam_id"].astype(str)
    df_dom = df_dom.drop_duplicates(subset=["uniprot_id", "pfam_id"])
    df_dom = df_dom.sort_values(["uniprot_id", "pfam_id"]).reset_index(drop=True)
    df_dom.to_parquet(output_path, index=False)
    return df_dom


def _first_non_null(values: pd.Series) -> Any:
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        return v
    return None


def collapse_by_domain(df: pd.DataFrame) -> pd.DataFrame:
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

    return (
        df.groupby(["uniprot_id", "chem_comp_id", "pfam_id", "source"], as_index=False)
        .agg(
            pdb_ids=("pdb_id", _unique_pdb_list),
            pchembl=("pchembl", "max"),
            mechanism=("mechanism", _first_non_null),
            activity_comment=("activity_comment", _first_non_null),
            curation_method=("curation_method", _first_non_null),
        )
    )


def collapse_binding_sites(df_domain_collapsed: pd.DataFrame) -> pd.DataFrame:
    def _merge_pdb_lists(series: pd.Series) -> List[str]:
        merged: List[str] = []
        seen: set[str] = set()
        for v in series:
            if v is None:
                continue
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
        return sorted(sites)

    return (
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


def build_known_binding_data(df_binding: pd.DataFrame) -> pd.DataFrame:
    return collapse_binding_sites(collapse_by_domain(df_binding))


def add_smiles_to_known_binding(known_binding: pd.DataFrame, pdb_chembl_smiles: pd.DataFrame) -> pd.DataFrame:
    kb = known_binding.copy()
    smiles_lookup = (
        pdb_chembl_smiles[["chem_comp_id", "smiles"]]
        .dropna(subset=["chem_comp_id", "smiles"])
        .drop_duplicates(subset=["chem_comp_id"])
    )
    smiles_lookup["chem_comp_id"] = smiles_lookup["chem_comp_id"].astype(str)
    kb["chem_comp_id"] = kb["chem_comp_id"].astype(str)
    smiles_lookup = smiles_lookup.set_index("chem_comp_id")["smiles"]
    kb["smiles"] = kb["chem_comp_id"].map(smiles_lookup)
    kb = kb.dropna(subset=["smiles"]).reset_index(drop=True)
    return kb


def save_known_binding_table(known_binding: pd.DataFrame, results_dir: str | Path = "results") -> Path:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "known_binding_data.parquet"
    known_binding.to_parquet(output_path, index=False)
    return output_path


def build_predicted_binding_data_incremental(
    proteins_to_process: list[str],
    cache_dir: str | Path,
    provider,
    known_binding: pd.DataFrame,
    resume: bool = True,
) -> None:
    """Provider-agnostic incremental writer for per-protein predicted ligands."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = cache_dir / "predicted_binding_data.parquet"
    progress_path = cache_dir / "predicted_binding_progress.json"

    if resume and progress_path.exists():
        with open(progress_path, "r") as f:
            processed_proteins = set(json.load(f))
    else:
        processed_proteins = set()

    writer: Optional[pq.ParquetWriter] = None
    temp_path: Optional[Path] = None

    if parquet_path.exists() and resume:
        pf = pq.ParquetFile(parquet_path)
        existing_schema = pf.schema_arrow
        temp_path = parquet_path.with_suffix(".tmp.parquet")
        writer = pq.ParquetWriter(temp_path.as_posix(), existing_schema)
        for rg_idx in range(pf.num_row_groups):
            writer.write_table(pf.read_row_group(rg_idx))

    total = len(proteins_to_process)
    with tqdm(total=total, desc="Predicted proteins", unit="protein") as pbar:
        for i, prot in enumerate(proteins_to_process, start=1):
            if prot in processed_proteins:
                pbar.update(1)
                pbar.set_postfix(completadas=i, faltan=max(total - i, 0))
                continue

            ligands = provider.compute_for_protein(prot=prot, known_binding=known_binding)
            if ligands is None or ligands.empty:
                processed_proteins.add(prot)
                with open(progress_path, "w") as f:
                    json.dump(sorted(processed_proteins), f)
                pbar.update(1)
                pbar.set_postfix(completadas=i, faltan=max(total - i, 0))
                continue

            ligands = ligands.copy()
            ligands.insert(0, "uniprot_id", prot)
            table = pa.Table.from_pandas(ligands, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(parquet_path.as_posix(), table.schema)
            else:
                schema_names = writer.schema.names
                if table.schema.names != schema_names:
                    table = table.select(schema_names)

            writer.write_table(table)
            processed_proteins.add(prot)
            with open(progress_path, "w") as f:
                json.dump(sorted(processed_proteins), f)

            pbar.update(1)
            pbar.set_postfix(completadas=i, faltan=max(total - i, 0))

    if writer is not None:
        writer.close()
        if temp_path is not None:
            os.replace(temp_path, parquet_path)

    print()
    print("[INFO] Finished writing predicted_binding_data.")


class ZincProviderAdapter:
    """Compatibility adapter using existing get_zinc_ligands implementation."""

    def __init__(
        self,
        store_pdb_chembl: LigandStore,
        rep_pdb_chembl,
        store_zinc: LigandStore,
        rep_zinc,
        search_rep_ref=None,
        search_rep_zinc=None,
        search_metric: str = "tanimoto",
        zinc_search_threshold: float = 0.5,
        cluster_threshold: float = 0.8,
    ):
        self.store_pdb_chembl = store_pdb_chembl
        self.rep_pdb_chembl = rep_pdb_chembl
        self.store_zinc = store_zinc
        self.rep_zinc = rep_zinc
        self.search_rep_ref = search_rep_ref
        self.search_rep_zinc = search_rep_zinc
        self.search_metric = search_metric
        self.zinc_search_threshold = zinc_search_threshold
        self.cluster_threshold = cluster_threshold

    def compute_for_protein(self, prot: str, known_binding: pd.DataFrame) -> pd.DataFrame:
        return get_zinc_ligands(
            prot=prot,
            pdb_chembl_binding_data=known_binding,
            store_pdb_chembl=self.store_pdb_chembl,
            rep_pdb_chembl=self.rep_pdb_chembl,
            store_zinc=self.store_zinc,
            rep_zinc=self.rep_zinc,
            search_rep_ref=self.search_rep_ref,
            search_rep_zinc=self.search_rep_zinc,
            search_metric=self.search_metric,
            zinc_search_threshold=self.zinc_search_threshold,
            cluster_threshold=self.cluster_threshold,
        )
