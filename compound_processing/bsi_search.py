from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from compound_processing.bsi_group_models import (
    BSIGroupMLP,
    ensure_dense_fp_matrix,
    load_group_model,
    resolve_torch_device,
    score_query_against_dense_chunk,
)
from compound_processing.compound_database_search import (
    cluster_ligands_by_tanimoto,
    max_pairwise_tanimoto_from_ids,
    select_ligands_maxmin_by_tanimoto,
)
from compound_processing.compound_helpers import LigandStore, Representation


BSI_MODEL_SUBDIR = Path("bsi_models") / "mpg_1024"
BSI_REPRESENTATION = "morgan_1024_r2"
BSI_DEFAULT_MAX_KNOWN_LIGANDS = 10


class BSIModelRegistry:
    def __init__(self, models_root: str | Path):
        self.models_root = Path(models_root)
        self.summary_path = self.models_root / "summary.csv"
        self.manifest_path = self.models_root / "manifest.json"
        if not self.summary_path.is_file():
            raise FileNotFoundError(f"BSI model summary not found: {self.summary_path}")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"BSI model manifest not found: {self.manifest_path}")

        self.summary = pd.read_csv(self.summary_path)
        if "pfam_id" not in self.summary.columns:
            raise ValueError(f"BSI summary must contain 'pfam_id': {self.summary_path}")
        if "val_pr_auc" not in self.summary.columns:
            self.summary["val_pr_auc"] = np.nan
        if "status" in self.summary.columns:
            self.summary = self.summary[self.summary["status"].astype(str) == "trained"].copy()

        self.summary["pfam_id"] = self.summary["pfam_id"].astype(str)
        self.supported_pfams = set(self.summary["pfam_id"])
        self._model_cache: dict[str, tuple[BSIGroupMLP, dict]] = {}

    def select_best_pfam(self, pfam_ids: Sequence[str]) -> str | None:
        candidates = self.summary[self.summary["pfam_id"].isin([str(p) for p in pfam_ids])]
        if candidates.empty:
            return None
        candidates = candidates.copy()
        candidates["val_pr_auc"] = pd.to_numeric(candidates["val_pr_auc"], errors="coerce")
        candidates = candidates.sort_values(
            ["val_pr_auc", "pfam_id"],
            ascending=[False, True],
            na_position="last",
        )
        return str(candidates.iloc[0]["pfam_id"])

    def load(self, pfam_id: str, device: str | torch.device = "auto") -> tuple[BSIGroupMLP, dict]:
        pfam_id = str(pfam_id)
        resolved = resolve_torch_device(device)
        cache_key = f"{pfam_id}:{resolved}"
        if cache_key not in self._model_cache:
            model_path = self.models_root / pfam_id / "model.pth"
            params_path = self.models_root / pfam_id / "params.json"
            if not model_path.is_file():
                raise FileNotFoundError(f"BSI model not found for {pfam_id}: {model_path}")
            if not params_path.is_file():
                raise FileNotFoundError(f"BSI params not found for {pfam_id}: {params_path}")
            self._model_cache[cache_key] = load_group_model(
                model_path=model_path,
                params_path=params_path,
                device=resolved,
            )
        return self._model_cache[cache_key]

    def fingerprint(self) -> str:
        files = [self.summary_path, self.manifest_path]
        for pfam_id in sorted(self.supported_pfams):
            files.append(self.models_root / pfam_id / "params.json")
            files.append(self.models_root / pfam_id / "model.pth")

        payload = []
        for path in files:
            if not path.is_file():
                continue
            st = path.stat()
            payload.append(
                {
                    "path": str(path.relative_to(self.models_root)),
                    "size": int(st.st_size),
                    "mtime_ns": int(st.st_mtime_ns),
                }
            )
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _resolve_bsi_chunk_size(
    *,
    device: str | torch.device,
    target_chunk_size: int | None,
) -> int:
    if target_chunk_size is not None:
        return int(target_chunk_size)
    resolved = resolve_torch_device(device)
    return 100_000 if resolved.type == "cuda" else 50_000


def _select_representative_ligands(
    prot: str,
    known_binding: pd.DataFrame,
    rep_pdb_chembl: Representation,
    *,
    max_queries: int,
    cluster_threshold: float,
) -> list[str]:
    df_prot = known_binding.loc[
        known_binding["uniprot_id"].astype(str) == str(prot),
        ["chem_comp_id", "pchembl"],
    ]
    chem_ids = (
        df_prot["chem_comp_id"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    chem_ids = [cid for cid in chem_ids if cid in rep_pdb_chembl.id_to_idx]
    if not chem_ids:
        return []

    pchembl_map = (
        df_prot.dropna(subset=["pchembl"])
        .assign(chem_comp_id=lambda df: df["chem_comp_id"].astype(str))
        .groupby("chem_comp_id")["pchembl"]
        .max()
        .to_dict()
    )

    if len(chem_ids) <= int(max_queries):
        representatives, _clusters, _assignment = cluster_ligands_by_tanimoto(
            chem_comp_ids=chem_ids,
            rep=rep_pdb_chembl,
            threshold=cluster_threshold,
            pchembl_map=pchembl_map,
        )
        return representatives

    representatives_mm, _clusters_mm, _assignment_mm = select_ligands_maxmin_by_tanimoto(
        chem_comp_ids=chem_ids,
        rep=rep_pdb_chembl,
        n_select=int(max_queries),
        pchembl_map=pchembl_map,
    )

    ti_max = max_pairwise_tanimoto_from_ids(representatives_mm, rep_pdb_chembl)
    if ti_max >= cluster_threshold:
        representatives, _clusters, _assignment = cluster_ligands_by_tanimoto(
            chem_comp_ids=representatives_mm,
            rep=rep_pdb_chembl,
            threshold=cluster_threshold,
            pchembl_map=pchembl_map,
        )
        return representatives
    return representatives_mm


def _top_indices_above_threshold(scores: np.ndarray, threshold: float, topk: int | None) -> np.ndarray:
    keep = np.flatnonzero(scores >= float(threshold))
    if keep.size == 0 or topk is None or keep.size <= int(topk):
        return keep
    vals = scores[keep]
    local = np.argpartition(vals, -int(topk))[-int(topk):]
    local = local[np.argsort(vals[local])[::-1]]
    return keep[local]


def search_bsi_against_target(
    query_ids: Sequence[str],
    *,
    model: BSIGroupMLP,
    fp_bits: int,
    store_ref: LigandStore,
    rep_ref: Representation,
    store_target: LigandStore,
    rep_target: Representation,
    threshold: float,
    device: str | torch.device = "auto",
    target_chunk_size: int | None = None,
    model_batch_size: int = 65536,
    per_chunk_topk: int | None = 1000,
    global_topk: int | None = 50000,
    compound_prefix: str = "",
    target_limit: int | None = None,
) -> pd.DataFrame:
    query_ids = [str(q) for q in query_ids if str(q) in rep_ref.id_to_idx]
    if not query_ids:
        return pd.DataFrame(columns=["chem_comp_id", "query_id", "bsi_score", "smiles"])

    chunk_size = _resolve_bsi_chunk_size(device=device, target_chunk_size=target_chunk_size)
    query_fps = ensure_dense_fp_matrix(rep_ref.get_raw_by_ids(query_ids), fp_bits)
    ligands_by_idx = store_target.ligands.set_index("lig_idx")
    rows: list[pd.DataFrame] = []

    for query_id, query_fp in zip(query_ids, query_fps):
        q_target_idx: list[np.ndarray] = []
        q_scores: list[np.ndarray] = []

        for start, _end, raw_chunk in rep_target.iter_raw_chunks(chunk_size, end=target_limit):
            target_fps = ensure_dense_fp_matrix(raw_chunk, fp_bits)
            scores = score_query_against_dense_chunk(
                model=model,
                query_fp=query_fp,
                candidate_fps=target_fps,
                batch_size=model_batch_size,
            )
            keep = _top_indices_above_threshold(scores, threshold=threshold, topk=per_chunk_topk)
            if keep.size == 0:
                continue
            q_target_idx.append((keep + start).astype(np.int64, copy=False))
            q_scores.append(scores[keep].astype(np.float32, copy=False))

        if not q_target_idx:
            continue

        target_idx = np.concatenate(q_target_idx)
        bsi_scores = np.concatenate(q_scores)
        if global_topk is not None and target_idx.size > int(global_topk):
            top = np.argpartition(bsi_scores, -int(global_topk))[-int(global_topk):]
            top = top[np.argsort(bsi_scores[top])[::-1]]
            target_idx = target_idx[top]
            bsi_scores = bsi_scores[top]
        else:
            order = np.argsort(bsi_scores)[::-1]
            target_idx = target_idx[order]
            bsi_scores = bsi_scores[order]

        meta = ligands_by_idx.loc[target_idx, [c for c in ("chem_comp_id", "smiles") if c in ligands_by_idx.columns]]
        out = meta.reset_index(drop=True).copy()
        out.insert(0, "query_id", query_id)
        out["bsi_score"] = bsi_scores
        rows.append(out)

    if not rows:
        return pd.DataFrame(columns=["chem_comp_id", "query_id", "bsi_score", "smiles"])

    hits = pd.concat(rows, ignore_index=True)
    if compound_prefix:
        hits["chem_comp_id"] = compound_prefix + hits["chem_comp_id"].astype(str)
    else:
        hits["chem_comp_id"] = hits["chem_comp_id"].astype(str)
    hits = (
        hits.sort_values("bsi_score", ascending=False)
        .drop_duplicates(subset=["chem_comp_id", "smiles"], keep="first")
        .reset_index(drop=True)
    )
    if global_topk is not None:
        hits = hits.head(int(global_topk)).reset_index(drop=True)
    return hits[["chem_comp_id", "query_id", "bsi_score", "smiles"]]


def get_bsi_ligands(
    prot: str,
    known_binding: pd.DataFrame,
    protein_domains: pd.DataFrame,
    model_registry: BSIModelRegistry,
    store_pdb_chembl: LigandStore,
    rep_pdb_chembl: Representation,
    store_target: LigandStore,
    rep_target: Representation,
    *,
    max_queries: int = BSI_DEFAULT_MAX_KNOWN_LIGANDS,
    cluster_threshold: float = 0.8,
    bsi_threshold: float = 0.5,
    search_device: str | torch.device = "auto",
    search_target_chunk_size: int | None = None,
    bsi_model_batch_size: int = 65536,
    search_per_iteration_topk: int = 1000,
    search_global_topk: int = 50000,
    compound_prefix: str = "",
) -> pd.DataFrame:
    domains = protein_domains.loc[
        protein_domains["uniprot_id"].astype(str) == str(prot),
        "pfam_id",
    ].dropna().astype(str).tolist()
    pfam_id = model_registry.select_best_pfam(domains)
    if pfam_id is None:
        return pd.DataFrame(columns=["chem_comp_id", "query_id", "bsi_score", "pfam_id", "smiles"])

    representatives = _select_representative_ligands(
        prot=prot,
        known_binding=known_binding,
        rep_pdb_chembl=rep_pdb_chembl,
        max_queries=max_queries,
        cluster_threshold=cluster_threshold,
    )
    if not representatives:
        return pd.DataFrame(columns=["chem_comp_id", "query_id", "bsi_score", "pfam_id", "smiles"])

    model, params = model_registry.load(pfam_id, device=search_device)
    fp_bits = int(params.get("fp_bits", 1024))
    hits = search_bsi_against_target(
        query_ids=representatives,
        model=model,
        fp_bits=fp_bits,
        store_ref=store_pdb_chembl,
        rep_ref=rep_pdb_chembl,
        store_target=store_target,
        rep_target=rep_target,
        threshold=bsi_threshold,
        device=search_device,
        target_chunk_size=search_target_chunk_size,
        model_batch_size=bsi_model_batch_size,
        per_chunk_topk=search_per_iteration_topk,
        global_topk=search_global_topk,
        compound_prefix=compound_prefix,
    )
    if hits.empty:
        return pd.DataFrame(columns=["chem_comp_id", "query_id", "bsi_score", "pfam_id", "smiles"])
    hits.insert(3, "pfam_id", pfam_id)
    return hits
