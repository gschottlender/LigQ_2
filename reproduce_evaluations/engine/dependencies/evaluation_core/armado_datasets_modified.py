
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict, Optional, Tuple
import random
import json
import ast

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina
from pathlib import Path
import sys
from typing import Any, Sequence
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent

# Resolve only the adjacent frozen kernels, never a live LigQ2 checkout.
sys.path.insert(0, str(CURRENT_DIR.parent / "ligq_core"))

from compound_helpers import Representation, LigandStore
import metrics

@dataclass
class SplitFixed:
    known_ids: List
    test_active_ids: List
    removed_test_too_similar: List
    diagnostics: Dict


def _mol(smi: str):
    try:
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def _inchikey(mol):
    try:
        return Chem.inchi.MolToInchiKey(mol)
    except Exception:
        return None


def _morgan_fp(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def _scaffold_smi(mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaf, isomericSmiles=False) if scaf is not None else ""
    except Exception:
        return ""


def prepare_actives_df(activos: Iterable, smiles_df: pd.DataFrame,
                       id_col="chem_comp_id", smiles_col="smiles") -> pd.DataFrame:
    """Build an active-compound table with molecules, InChIKeys, and scaffolds; deduplicate by InChIKey."""
    activos_set = set(activos)
    df = smiles_df.loc[smiles_df[id_col].isin(activos_set), [id_col, smiles_col]].copy()
    df = df.dropna(subset=[smiles_col]).drop_duplicates(subset=[id_col])

    rows = []
    for _, r in df.iterrows():
        m = _mol(r[smiles_col])
        if m is None:
            continue
        ik = _inchikey(m)
        if ik is None:
            continue
        rows.append({
            id_col: r[id_col],
            smiles_col: r[smiles_col],
            "mol": m,
            "inchikey": ik,
            "scaffold": _scaffold_smi(m),
        })

    out = pd.DataFrame(rows)
    out = out.drop_duplicates(subset=["inchikey"]).reset_index(drop=True)
    return out


def _sample_ids(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    rng = random.Random(random_state)
    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    return df.iloc[idxs[:min(n, len(df))]].reset_index(drop=True)


def _sample_test_stratified_by_scaffold(df: pd.DataFrame, n: int, random_state: int,
                                        max_per_scaffold: int = 3) -> pd.DataFrame:
    """Diverse test sampling with at most X compounds per scaffold."""
    rng = random.Random(random_state)

    groups: Dict[str, List[int]] = {}
    for i, scaf in enumerate(df["scaffold"].tolist()):
        groups.setdefault(scaf, []).append(i)

    for scaf in groups:
        rng.shuffle(groups[scaf])

    scaffolds = list(groups.keys())
    rng.shuffle(scaffolds)

    chosen = []
    used = {s: 0 for s in scaffolds}

    while len(chosen) < min(n, len(df)):
        progressed = False
        for scaf in scaffolds:
            if len(chosen) >= n:
                break
            if used[scaf] >= max_per_scaffold:
                continue
            if not groups[scaf]:
                continue
            chosen.append(groups[scaf].pop())
            used[scaf] += 1
            progressed = True
        if not progressed:
            break

    return df.iloc[chosen].reset_index(drop=True)

def choose_split_sizes(
    n_actives: int,
    *,
    known_frac: float = 0.10,     # 10% seeds
    test_to_known_ratio: float = 5.0,  # test = 5x known
    known_min: int = 30,
    known_max: int = 200,
    test_min: int = 100,
    test_max: int = 1000,
) -> tuple[int, int]:
    """
    Return proportional, bounded (n_known, n_test) sizes.
    Always attempt to keep n_known + n_test <= n_actives.
    """
    if n_actives <= 0:
        return 0, 0

    n_known = int(round(n_actives * known_frac))
    n_known = max(known_min, min(known_max, n_known))

    # Handle small targets
    n_known = min(n_known, max(1, n_actives - 1))

    n_test = int(round(test_to_known_ratio * n_known))
    n_test = max(test_min, min(test_max, n_test))

    # Ensure that it fits
    n_test = min(n_test, n_actives - n_known)

    # If the test set is too small, reduce known to make room for test
    if n_test < min(test_min, n_actives - n_known) and n_actives > 1:
        # Try to reduce known if it is large
        while n_test < test_min and n_known > 1 and (n_actives - n_known) > n_test:
            n_known -= 1
            n_test = min(int(round(test_to_known_ratio * n_known)), test_max, n_actives - n_known)

    return n_known, n_test

def cap_actives_diverse_by_scaffold(df: pd.DataFrame, cap: int, random_state: int, id_col="chem_comp_id") -> pd.DataFrame:
    if cap is None or len(df) <= cap:
        return df.reset_index(drop=True)

    rng = random.Random(random_state)

    # Group indices by scaffold
    groups = {}
    for i, scaf in enumerate(df["scaffold"].tolist()):
        groups.setdefault(scaf, []).append(i)
    for scaf in groups:
        rng.shuffle(groups[scaf])

    scaffolds = list(groups.keys())
    rng.shuffle(scaffolds)

    chosen = []
    while len(chosen) < cap:
        progressed = False
        for scaf in scaffolds:
            if len(chosen) >= cap:
                break
            if groups[scaf]:
                chosen.append(groups[scaf].pop())
                progressed = True
        if not progressed:
            break

    out = df.iloc[chosen].copy()
    out = out.drop_duplicates(subset=[id_col]).reset_index(drop=True)
    return out


def split_hit_expansion_fixed_sizes(
    activos: Iterable,
    smiles_df: pd.DataFrame,
    n_known: int | None = None,
    n_test: int | None = None,
    *,
    # Sizing policy (when n_known or n_test is None)
    known_frac: float = 0.10,
    test_to_known_ratio: float = 5.0,
    known_min: int = 30,
    known_max: int = 200,
    test_min: int = 100,
    test_max: int = 1000,
    max_total_actives: int | None = 2000,   # Total cap per target after deduplication
    cap_diverse: bool = True,

    id_col="chem_comp_id",
    smiles_col="smiles",
    nontrivial_tanimoto_cutoff: float = 0.85,
    tanimoto_radius: int = 2,
    tanimoto_nbits: int = 1024,
    stratify_test_by_scaffold: bool = True,
    max_per_scaffold_test: int = 3,
    random_state: int = 0,
    fallback_if_short: str = "relax_cutoff",  # "allow_short" | "relax_cutoff"
    relax_steps: Tuple[float, ...] = (0.90, 0.92, 0.95),
) -> SplitFixed:

    df = prepare_actives_df(activos, smiles_df, id_col=id_col, smiles_col=smiles_col)

    diag = {
        "n_actives_total_dedup": len(df),
        "max_total_actives": max_total_actives,
    }

    # Cap total actives after deduplication for very large targets
    if max_total_actives is not None and len(df) > max_total_actives:
        if cap_diverse:
            df = cap_actives_diverse_by_scaffold(df, max_total_actives, random_state, id_col=id_col)
            diag["cap_mode"] = "diverse_by_scaffold"
        else:
            df = _sample_ids(df, max_total_actives, random_state)
            diag["cap_mode"] = "random"
        diag["n_actives_after_cap"] = len(df)
    else:
        diag["n_actives_after_cap"] = len(df)

    if len(df) < 2:
        diag["status"] = "too_few_actives"
        return SplitFixed([], [], [], diag)

    # Choose sizes unless fixed values were supplied
    if n_known is None or n_test is None:
        nk, nt = choose_split_sizes(
            len(df),
            known_frac=known_frac,
            test_to_known_ratio=test_to_known_ratio,
            known_min=known_min,
            known_max=known_max,
            test_min=test_min,
            test_max=test_max,
        )
        if n_known is None:
            n_known = nk
        if n_test is None:
            n_test = nt

    diag.update({"n_known_req": n_known, "n_test_req": n_test})

    # If there are still too few compounds, degrade gracefully
    if len(df) < (n_known + 1):
        known_df = _sample_ids(df, min(n_known, len(df) - 1), random_state)
        rest_df = df[~df[id_col].isin(set(known_df[id_col]))].reset_index(drop=True)
        test_df = _sample_ids(rest_df, min(n_test, len(rest_df)), random_state + 1)
        diag["status"] = "too_few_actives_for_requested_sizes"
        diag["final_n_known"] = len(known_df)
        diag["final_n_test"] = len(test_df)
        return SplitFixed(
            known_ids=known_df[id_col].tolist(),
            test_active_ids=test_df[id_col].tolist(),
            removed_test_too_similar=[],
            diagnostics=diag
        )

    # 1) Choose known actives
    known_df = _sample_ids(df, n_known, random_state)

    # 2) Remaining
    remaining_df = df[~df[id_col].isin(set(known_df[id_col]))].reset_index(drop=True)

    # 3) anti-trivial filter
    known_fps = [_morgan_fp(m, radius=tanimoto_radius, nbits=tanimoto_nbits) for m in known_df["mol"]]

    def filter_candidates(cutoff: float):
        kept_rows = []
        removed_ids = []
        for i, row in remaining_df.iterrows():
            fp = _morgan_fp(row["mol"], radius=tanimoto_radius, nbits=tanimoto_nbits)
            mx = max(DataStructs.BulkTanimotoSimilarity(fp, known_fps)) if known_fps else 0.0
            if mx > cutoff:
                removed_ids.append(row[id_col])
            else:
                kept_rows.append(i)
        return remaining_df.iloc[kept_rows].reset_index(drop=True), removed_ids

    cutoff_used = nontrivial_tanimoto_cutoff
    candidates, removed_ids = filter_candidates(cutoff_used)

    if len(candidates) < n_test and fallback_if_short == "relax_cutoff":
        for c in relax_steps:
            cutoff_used = c
            candidates, removed_ids = filter_candidates(cutoff_used)
            if len(candidates) >= n_test:
                break

    # 4) sample test
    if len(candidates) >= n_test:
        if stratify_test_by_scaffold:
            test_df = _sample_test_stratified_by_scaffold(
                candidates, n_test, random_state + 2, max_per_scaffold=max_per_scaffold_test
            )
        else:
            test_df = _sample_ids(candidates, n_test, random_state + 2)
        status = "ok"
    else:
        test_df = candidates
        status = "short_test_set" if fallback_if_short == "allow_short" else "short_even_after_relax"

    diag.update({
        "cutoff_used": cutoff_used,
        "n_candidates_after_filter": len(candidates),
        "n_removed_too_similar": len(removed_ids),
        "final_n_known": len(known_df),
        "final_n_test": len(test_df),
        "status": status
    })

    return SplitFixed(
        known_ids=known_df[id_col].tolist(),
        test_active_ids=test_df[id_col].tolist(),
        removed_test_too_similar=removed_ids,
        diagnostics=diag
    )


def split_scaffold_fixed_sizes(
    activos: Iterable,
    smiles_df: pd.DataFrame,
    n_known: int | None = None,
    n_test: int | None = None,
    *,
    # Sizing policy when values are None
    known_frac: float = 0.10,
    test_to_known_ratio: float = 5.0,
    known_min: int = 30,
    known_max: int = 200,
    test_min: int = 100,
    test_max: int = 1000,
    max_total_actives: int | None = 2000,
    cap_diverse: bool = True,

    id_col="chem_comp_id",
    smiles_col="smiles",
    scaffold_train_frac: float = 0.70,
    stratify_test_by_scaffold: bool = True,
    max_per_scaffold_test: int = 3,
    random_state: int = 0,
) -> SplitFixed:

    df = prepare_actives_df(activos, smiles_df, id_col=id_col, smiles_col=smiles_col)
    diag = {
        "n_actives_total_dedup": len(df),
        "max_total_actives": max_total_actives,
    }

    # Cap after deduplication
    if max_total_actives is not None and len(df) > max_total_actives:
        if cap_diverse:
            df = cap_actives_diverse_by_scaffold(df, max_total_actives, random_state, id_col=id_col)
            diag["cap_mode"] = "diverse_by_scaffold"
        else:
            df = _sample_ids(df, max_total_actives, random_state)
            diag["cap_mode"] = "random"
        diag["n_actives_after_cap"] = len(df)
    else:
        diag["n_actives_after_cap"] = len(df)

    if len(df) < 2:
        diag["status"] = "too_few_actives"
        return SplitFixed([], [], [], diag)

    # Proportional sizes when needed
    if n_known is None or n_test is None:
        nk, nt = choose_split_sizes(
            len(df),
            known_frac=known_frac,
            test_to_known_ratio=test_to_known_ratio,
            known_min=known_min,
            known_max=known_max,
            test_min=test_min,
            test_max=test_max,
        )
        if n_known is None:
            n_known = nk
        if n_test is None:
            n_test = nt

    diag.update({"n_known_req": n_known, "n_test_req": n_test})

    # scaffolds -> indices
    scaffold_to_idxs: Dict[str, List[int]] = {}
    for i, scaf in enumerate(df["scaffold"].tolist()):
        scaffold_to_idxs.setdefault(scaf, []).append(i)

    scaffolds = list(scaffold_to_idxs.keys())
    rng = random.Random(random_state)
    rng.shuffle(scaffolds)

    n_scaf = len(scaffolds)
    n_train_scaf = int(round(scaffold_train_frac * n_scaf))
    n_train_scaf = max(1, min(n_train_scaf, n_scaf - 1))

    train_scaf = set(scaffolds[:n_train_scaf])
    test_scaf = set(scaffolds[n_train_scaf:])

    train_df = df[df["scaffold"].isin(train_scaf)].reset_index(drop=True)
    test_df_all = df[df["scaffold"].isin(test_scaf)].reset_index(drop=True)

    # Choose known actives from the training side
    if len(train_df) >= n_known:
        known_df = _sample_ids(train_df, n_known, random_state + 1)
    else:
        known_df = train_df

    # Choose test actives from the test side
    if len(test_df_all) >= n_test:
        if stratify_test_by_scaffold:
            test_df = _sample_test_stratified_by_scaffold(
                test_df_all, n_test, random_state + 2, max_per_scaffold=max_per_scaffold_test
            )
        else:
            test_df = _sample_ids(test_df_all, n_test, random_state + 2)
        status = "ok"
    else:
        test_df = test_df_all
        status = "short_test_set"

    diag.update({
        "n_scaffolds_total": n_scaf,
        "n_train_scaffolds": len(train_scaf),
        "n_test_scaffolds": len(test_scaf),
        "train_actives": len(train_df),
        "test_actives_candidates": len(test_df_all),
        "final_n_known": len(known_df),
        "final_n_test": len(test_df),
        "status": status
    })

    return SplitFixed(
        known_ids=known_df[id_col].tolist(),
        test_active_ids=test_df[id_col].tolist(),
        removed_test_too_similar=[],
        diagnostics=diag
    )


def split_hit_expansion_fixed_sizes_clustering(
    activos: Iterable,
    smiles_df: pd.DataFrame,
    n_known: int | None = None,
    n_test: int | None = None,
    *,
    known_frac: float = 0.10,
    test_to_known_ratio: float = 5.0,
    known_min: int = 30,
    known_max: int = 200,
    test_min: int = 100,
    test_max: int = 1000,
    max_total_actives: int | None = 2000,
    cap_diverse: bool = True,
    id_col="chem_comp_id",
    smiles_col="smiles",
    butina_cutoff: float | None = None,
    nontrivial_tanimoto_cutoff: float = 0.85,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
    random_state: int = 0,
    **_ignored_kwargs,
) -> SplitFixed:
    """
    Split using representatives from Butina/Tanimoto clusters.

    First cluster all actives and retain one medoid-like representative per
    cluster. Then construct known/unknown sets using the same sizing policy as
    split_hit_expansion_fixed_sizes.
    """
    if butina_cutoff is None:
        butina_cutoff = nontrivial_tanimoto_cutoff

    if butina_cutoff <= 0 or butina_cutoff > 1:
        raise ValueError("butina_cutoff must be in (0, 1].")

    df = prepare_actives_df(activos, smiles_df, id_col=id_col, smiles_col=smiles_col)
    diag = {
        "split_mode": "butina_representatives",
        "n_actives_total_dedup": len(df),
        "max_total_actives": max_total_actives,
        "butina_cutoff": float(butina_cutoff),
        "butina_radius": int(butina_radius),
        "butina_nbits": int(butina_nbits),
    }

    if max_total_actives is not None and len(df) > max_total_actives:
        if cap_diverse:
            df = cap_actives_diverse_by_scaffold(df, max_total_actives, random_state, id_col=id_col)
            diag["cap_mode"] = "diverse_by_scaffold"
        else:
            df = _sample_ids(df, max_total_actives, random_state)
            diag["cap_mode"] = "random"
        diag["n_actives_after_cap"] = len(df)
    else:
        diag["n_actives_after_cap"] = len(df)

    if len(df) < 2:
        diag["status"] = "too_few_actives"
        return SplitFixed([], [], [], diag)

    fps = [_morgan_fp(mol, radius=butina_radius, nbits=butina_nbits) for mol in df["mol"]]
    n_fps = len(fps)
    if n_fps == 1:
        clusters = ((0,),)
    else:
        dists = []
        for i in range(1, n_fps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
            dists.extend([1.0 - float(x) for x in sims])
        clusters = Butina.ClusterData(
            dists,
            n_fps,
            1.0 - float(butina_cutoff),
            isDistData=True,
            reordering=True,
        )

    cluster_idxs = [tuple(int(i) for i in cluster) for cluster in clusters]
    representative_indices = [
        _choose_butina_cluster_representative(cluster, fps)
        for cluster in cluster_idxs
    ]
    rep_df = df.iloc[representative_indices].reset_index(drop=True)

    diag.update({
        "n_clusters_total": int(len(cluster_idxs)),
        "n_representatives": int(len(rep_df)),
        "cluster_sizes": [int(len(c)) for c in cluster_idxs],
    })

    if len(rep_df) < 2:
        diag["status"] = "too_few_representatives"
        return SplitFixed([], [], [], diag)

    if n_known is None or n_test is None:
        nk, nt = choose_split_sizes(
            len(rep_df),
            known_frac=known_frac,
            test_to_known_ratio=test_to_known_ratio,
            known_min=known_min,
            known_max=known_max,
            test_min=test_min,
            test_max=test_max,
        )
        if n_known is None:
            n_known = nk
        if n_test is None:
            n_test = nt

    diag.update({"n_known_req": int(n_known), "n_test_req": int(n_test)})

    rng = random.Random(random_state)
    rep_indices = list(range(len(rep_df)))
    rng.shuffle(rep_indices)

    n_known_eff = min(int(n_known), max(len(rep_indices) - 1, 0))
    known_rep_indices = rep_indices[:n_known_eff]
    remaining_rep_indices = rep_indices[n_known_eff:]
    n_test_eff = min(int(n_test), len(remaining_rep_indices))
    test_rep_indices = remaining_rep_indices[:n_test_eff]

    known_df = rep_df.iloc[known_rep_indices].reset_index(drop=True)
    test_df = rep_df.iloc[test_rep_indices].reset_index(drop=True)

    status = "ok"
    if len(known_df) < n_known:
        status = "short_known_set"
    elif len(test_df) < n_test:
        status = "short_test_set"

    diag.update({
        "final_n_known": int(len(known_df)),
        "final_n_test": int(len(test_df)),
        "n_unused_representatives": int(max(len(rep_df) - len(known_df) - len(test_df), 0)),
        "status": status,
    })

    return SplitFixed(
        known_ids=known_df[id_col].astype(str).tolist(),
        test_active_ids=test_df[id_col].astype(str).tolist(),
        removed_test_too_similar=[],
        diagnostics=diag,
    )


def get_target_ligand_sets(
    *,
    target_name: str,
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Reproduce the base logic used by run_ef_eval_target_vs_neighbors for one target:
      - obtain the target's UniProt IDs
      - define true actives
      - define the family blacklist
      - construct the inactive universe
    """
    targets_dude_s = targets_dude.copy()
    binding_data_s = binding_data.copy()
    smiles_s = smiles.copy()

    targets_dude_s["target"] = targets_dude_s["target"].astype(str)
    if "uniprot_id" in targets_dude_s.columns:
        targets_dude_s["uniprot_id"] = targets_dude_s["uniprot_id"].astype(str)

    binding_data_s["uniprot_id"] = binding_data_s["uniprot_id"].astype(str)
    binding_data_s["chem_comp_id"] = binding_data_s["chem_comp_id"].astype(str)
    binding_data_s["pfam_id"] = binding_data_s["pfam_id"].astype(str)
    smiles_s["chem_comp_id"] = smiles_s["chem_comp_id"].astype(str)

    target = str(target_name)

    uniprots = (
        targets_dude_s.loc[targets_dude_s["target"] == target, "uniprot_id"]
        .dropna().astype(str).unique().tolist()
    )

    activos = (
        binding_data_s.loc[binding_data_s["uniprot_id"].isin(uniprots), "chem_comp_id"]
        .dropna().astype(str).unique().tolist()
    )

    fams_target = (
        binding_data_s.loc[binding_data_s["uniprot_id"].isin(uniprots), "pfam_id"]
        .dropna().astype(str).unique().tolist()
    )

    lista_negra = (
        binding_data_s.loc[binding_data_s["pfam_id"].isin(fams_target), "chem_comp_id"]
        .dropna().astype(str).tolist()
    )

    inactivos = (
        smiles_s.loc[~smiles_s["chem_comp_id"].isin(lista_negra), "chem_comp_id"]
        .dropna().astype(str).tolist()
    )
    inactivos = stable_set_difference_preserve_order(inactivos, activos)

    return {
        "target": target,
        "uniprot_ids": unique_preserve_order(uniprots),
        "pfam_ids": unique_preserve_order(fams_target),
        "active_ids": unique_preserve_order(activos),
        "inactive_ids": unique_preserve_order(inactivos),
        "blacklist_ids": unique_preserve_order(lista_negra),
    }


def _choose_butina_cluster_representative(cluster_indices: Sequence[int], fps: Sequence) -> int:
    """
    Choose a medoid-like representative within the cluster by maximizing mean
    similarity to the other members, with ties broken by original index.
    """
    if len(cluster_indices) == 1:
        return int(cluster_indices[0])

    best_idx = None
    best_score = None
    cluster_indices = [int(i) for i in cluster_indices]

    for idx in cluster_indices:
        others = [j for j in cluster_indices if j != idx]
        sims = DataStructs.BulkTanimotoSimilarity(fps[idx], [fps[j] for j in others])
        mean_sim = float(np.mean(sims)) if sims else 1.0
        score = (mean_sim, -idx)
        if best_score is None or score > best_score:
            best_idx = idx
            best_score = score

    return int(best_idx)


def cluster_actives_by_butina(
    active_ids: Iterable,
    smiles_df: pd.DataFrame,
    *,
    cutoff: float = 0.4,
    radius: int = 2,
    nbits: int = 1024,
    id_col: str = "chem_comp_id",
    smiles_col: str = "smiles",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cluster actives with Butina using Morgan/Tanimoto.

    cutoff:
      - minimum similarity for grouping (for example, 0.4 or 0.3)

    Return:
      - df_debug with cluster assignments and representative flags
      - meta with representative lists and cluster sizes
    """
    if cutoff <= 0 or cutoff > 1:
        raise ValueError("cutoff must be in (0, 1]. Expected a Tanimoto value such as 0.4 or 0.3.")

    df = prepare_actives_df(active_ids, smiles_df, id_col=id_col, smiles_col=smiles_col).copy()

    if df.empty:
        empty_cols = [
            id_col, smiles_col, "inchikey", "scaffold",
            "cluster_id", "cluster_size", "is_representative", "cluster_representative_id",
        ]
        return pd.DataFrame(columns=empty_cols), {
            "cutoff": float(cutoff),
            "radius": int(radius),
            "nbits": int(nbits),
            "n_input_actives": 0,
            "n_dedup_actives": 0,
            "n_clusters": 0,
            "n_representatives": 0,
            "representative_ids": [],
            "cluster_sizes": [],
        }

    fps = [_morgan_fp(mol, radius=radius, nbits=nbits) for mol in df["mol"]]
    n_fps = len(fps)

    if n_fps == 1:
        df["cluster_id"] = 0
        df["cluster_size"] = 1
        df["is_representative"] = True
        df["cluster_representative_id"] = df[id_col].astype(str)
        out = df.drop(columns=["mol"]).reset_index(drop=True)
        return out, {
            "cutoff": float(cutoff),
            "radius": int(radius),
            "nbits": int(nbits),
            "n_input_actives": len(unique_preserve_order(active_ids)),
            "n_dedup_actives": 1,
            "n_clusters": 1,
            "n_representatives": 1,
            "representative_ids": out.loc[out["is_representative"], id_col].astype(str).tolist(),
            "cluster_sizes": [1],
        }

    dists = []
    for i in range(1, n_fps):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - float(x) for x in sims])

    dist_thresh = 1.0 - float(cutoff)
    clusters = Butina.ClusterData(dists, n_fps, dist_thresh, isDistData=True, reordering=True)

    cluster_rows = []
    representative_ids = []
    cluster_sizes = []

    for cluster_id, cluster in enumerate(clusters):
        cluster = tuple(int(i) for i in cluster)
        rep_idx = _choose_butina_cluster_representative(cluster, fps)
        rep_id = str(df.iloc[rep_idx][id_col])
        cluster_sizes.append(len(cluster))
        representative_ids.append(rep_id)

        for idx in cluster:
            cluster_rows.append({
                "__row_idx": int(idx),
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(cluster)),
                "is_representative": bool(idx == rep_idx),
                "cluster_representative_id": rep_id,
            })

    cluster_df = pd.DataFrame(cluster_rows)
    out = (
        df.reset_index(drop=True)
        .reset_index()
        .rename(columns={"index": "__row_idx"})
        .merge(cluster_df, on="__row_idx", how="left", validate="one_to_one")
        .drop(columns=["__row_idx", "mol"])
        .sort_values(["cluster_id", "is_representative"], ascending=[True, False])
        .reset_index(drop=True)
    )

    meta = {
        "cutoff": float(cutoff),
        "radius": int(radius),
        "nbits": int(nbits),
        "n_input_actives": len(unique_preserve_order(active_ids)),
        "n_dedup_actives": int(len(df)),
        "n_clusters": int(len(clusters)),
        "n_representatives": int(len(representative_ids)),
        "representative_ids": unique_preserve_order(representative_ids),
        "cluster_sizes": [int(x) for x in cluster_sizes],
    }
    return out, meta


def build_target_dataset_payload(
    *,
    target_name: str,
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
    split_kwargs: Optional[dict] = None,
    pool_kwargs: Optional[dict] = None,
    use_butina_actives: bool = False,
    butina_cutoff: float = 0.4,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Build the complete dataset for one target and return:
      - a JSON-serializable payload containing all relevant lists
      - an active-clustering debug DataFrame (empty when Butina is not used)
    """
    if split_kwargs is None:
        split_kwargs = dict(
            max_total_actives=1000,
            known_frac=0.20,
            known_min=10,
            known_max=200,
            test_to_known_ratio=5,
            test_min=50,
            test_max=1000,
            nontrivial_tanimoto_cutoff=0.85,
            fallback_if_short="allow_short",
            random_state=42,
        )

    if pool_kwargs is None:
        pool_kwargs = dict(
            ratio_neg_to_pos=200,
            n_bg_cap=None,
            random_state=0,
        )

    target_sets = get_target_ligand_sets(
        target_name=target_name,
        targets_dude=targets_dude,
        binding_data=binding_data,
        smiles=smiles,
    )

    active_ids_original = target_sets["active_ids"]
    clustering_df = pd.DataFrame()
    clustering_meta = None

    if use_butina_actives:
        clustering_df, clustering_meta = cluster_actives_by_butina(
            active_ids_original,
            smiles,
            cutoff=butina_cutoff,
            radius=butina_radius,
            nbits=butina_nbits,
        )
        active_ids_for_split = (
            clustering_df.loc[clustering_df["is_representative"], "chem_comp_id"]
            .astype(str).tolist()
        )
    else:
        active_ids_for_split = list(active_ids_original)

    split_res = split_hit_expansion_fixed_sizes(
        activos=active_ids_for_split,
        smiles_df=smiles,
        **split_kwargs,
    )

    eval_pool = build_eval_pool_ids(
        test_active_ids=split_res.test_active_ids,
        background_ids=target_sets["inactive_ids"],
        **pool_kwargs,
    )

    payload = {
        "target": str(target_sets["target"]),
        "uniprot_ids": target_sets["uniprot_ids"],
        "pfam_ids": target_sets["pfam_ids"],
        "active_ids_original": unique_preserve_order(active_ids_original),
        "active_ids_for_split": unique_preserve_order(active_ids_for_split),
        "known_active_ids": unique_preserve_order(split_res.known_ids),
        "test_active_ids": unique_preserve_order(split_res.test_active_ids),
        "removed_test_too_similar_ids": unique_preserve_order(split_res.removed_test_too_similar),
        "inactive_ids": target_sets["inactive_ids"],
        "eval_pool_ids": unique_preserve_order(eval_pool["pool_ids"]),
        "eval_positive_ids": unique_preserve_order(eval_pool["pos_ids"]),
        "eval_negative_ids": unique_preserve_order(eval_pool["neg_ids"]),
        "split_diagnostics": split_res.diagnostics,
        "pool_diagnostics": {
            "n_pos": int(eval_pool["n_pos"]),
            "n_neg": int(eval_pool["n_neg"]),
            "ratio_neg_to_pos_effective": eval_pool["ratio_neg_to_pos_effective"],
        },
        "butina_clustering": clustering_meta,
    }
    return payload, clustering_df


def run_build_and_save_target_datasets(
    *,
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
    output_dir: str | Path,
    split_kwargs: Optional[dict] = None,
    pool_kwargs: Optional[dict] = None,
    min_known_for_eval: Optional[int] = None,
    use_butina_actives: bool = False,
    butina_cutoff: float = 0.4,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
    save_cluster_csv: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Iterate over all targets and save, for each target:
      - `dataset.json` containing active/inactive lists and the split
      - `active_clustering.csv` for debugging when Butina is used

    Return a summary DataFrame for the run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_dude_s = targets_dude.copy()
    targets_dude_s["target"] = targets_dude_s["target"].astype(str)
    target_list = targets_dude_s["target"].dropna().astype(str).unique().tolist()

    summary_rows = []

    for i, target in enumerate(target_list, 1):
        if verbose:
            print(f"\n[{i}/{len(target_list)}] Building dataset for target: {target}")

        payload, clustering_df = build_target_dataset_payload(
            target_name=target,
            targets_dude=targets_dude,
            binding_data=binding_data,
            smiles=smiles,
            split_kwargs=split_kwargs,
            pool_kwargs=pool_kwargs,
            use_butina_actives=use_butina_actives,
            butina_cutoff=butina_cutoff,
            butina_radius=butina_radius,
            butina_nbits=butina_nbits,
        )

        n_known = len(payload["known_active_ids"])
        if min_known_for_eval is not None and n_known < int(min_known_for_eval):
            if verbose:
                print(f"  Skip save (n_known={n_known} < {min_known_for_eval})")
            summary_rows.append({
                "target": target,
                "saved": False,
                "skip_reason": f"n_known<{int(min_known_for_eval)}",
                "n_active_original": len(payload["active_ids_original"]),
                "n_active_for_split": len(payload["active_ids_for_split"]),
                "n_known": n_known,
                "n_test": len(payload["test_active_ids"]),
                "n_inactive": len(payload["inactive_ids"]),
                "dataset_json": None,
                "cluster_csv": None,
            })
            continue

        safe_target = str(target).replace("/", "_")
        target_dir = output_dir / safe_target
        target_dir.mkdir(parents=True, exist_ok=True)

        dataset_path = target_dir / "dataset.json"
        with dataset_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        cluster_csv_path = None
        if use_butina_actives and save_cluster_csv and not clustering_df.empty:
            cluster_csv_path = target_dir / "active_clustering.csv"
            clustering_df.to_csv(cluster_csv_path, index=False)

        summary_rows.append({
            "target": target,
            "saved": True,
            "skip_reason": None,
            "n_active_original": len(payload["active_ids_original"]),
            "n_active_for_split": len(payload["active_ids_for_split"]),
            "n_known": n_known,
            "n_test": len(payload["test_active_ids"]),
            "n_inactive": len(payload["inactive_ids"]),
            "dataset_json": str(dataset_path),
            "cluster_csv": str(cluster_csv_path) if cluster_csv_path is not None else None,
        })

    return pd.DataFrame(summary_rows)


DEFAULT_MPGS = [
    "PF00001",
    "PF00002",
    "PF00026",
    "PF00067",
    "PF00069",
    "PF00089",
    "PF00104",
    "PF00112",
    "PF00135",
    "PF00194",
    "PF00209",
    "PF00233",
    "PF00413",
    "PF00520",
    "PF00850",
    "PF01094",
    "PF07714",
]


def _normalize_targets_pfam_df(targets_pfam_df: pd.DataFrame) -> pd.DataFrame:
    df = targets_pfam_df.copy()
    rename_map = {}
    if "pfam" not in df.columns and "pfam_id" in df.columns:
        rename_map["pfam_id"] = "pfam"
    if rename_map:
        df = df.rename(columns=rename_map)

    missing = {"target", "pfam"} - set(df.columns)
    if missing:
        raise ValueError(f"targets_pfam_df must contain target and pfam columns. Missing: {sorted(missing)}")

    df["target"] = df["target"].astype(str)
    df["pfam"] = df["pfam"].astype(str)
    return df.loc[:, ["target", "pfam"]].drop_duplicates().reset_index(drop=True)


def build_family_group_map(
    targets_pfam_df: pd.DataFrame,
    *,
    mpgs: Optional[Sequence[str]] = None,
    kinase_pfams: Sequence[str] = ("PF00069", "PF07714"),
    kinase_family_name: str = "kinase",
) -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """
    Return:
      - target_to_family: target -> family_id
      - family_to_targets: family_id -> ordered unique targets

    Rules:
      - group PF00069 and PF07714 as `kinase`
      - retain each other Pfam ID as its family_id
      - raise an error if a target belongs to more than one non-kinase family
    """
    df = _normalize_targets_pfam_df(targets_pfam_df)

    if mpgs is None:
        mpgs = DEFAULT_MPGS

    mpgs_set = {str(x) for x in mpgs}
    kinase_pfams_set = {str(x) for x in kinase_pfams}
    df = df[df["pfam"].isin(mpgs_set)].copy()

    def _family_from_pfam(pfam: str) -> str:
        if pfam in kinase_pfams_set:
            return str(kinase_family_name)
        return str(pfam)

    df["family_id"] = df["pfam"].map(_family_from_pfam)

    per_target = df.groupby("target")["family_id"].agg(lambda s: unique_preserve_order(s.tolist()))

    target_to_family = {}
    for target, families in per_target.items():
        families = [str(x) for x in families]
        if len(families) > 1:
            raise ValueError(
                f"Target {target} belongs to multiple incompatible families: {families}. "
                "Check the Pfam annotations."
            )
        target_to_family[str(target)] = families[0]

    family_to_targets: Dict[str, List[str]] = {}
    for target, family_id in target_to_family.items():
        family_to_targets.setdefault(str(family_id), []).append(str(target))

    for family_id in list(family_to_targets):
        family_to_targets[family_id] = unique_preserve_order(family_to_targets[family_id])

    return target_to_family, family_to_targets


def load_saved_target_dataset_payloads(dataset_output_dir: str | Path) -> pd.DataFrame:
    """
    Load target-level `dataset.json` files and return a DataFrame containing
    the complete payload and selected derived counts.
    """
    dataset_output_dir = Path(dataset_output_dir)
    if not dataset_output_dir.exists():
        raise FileNotFoundError(f"dataset_output_dir does not exist: {dataset_output_dir}")

    rows = []
    for dataset_path in sorted(dataset_output_dir.glob("*/dataset.json")):
        with dataset_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        target = str(payload.get("target", dataset_path.parent.name))
        row = {
            "target": target,
            "dataset_json": str(dataset_path),
            "target_dir": str(dataset_path.parent),
            "payload": payload,
            "pfam_ids": unique_preserve_order(payload.get("pfam_ids", [])),
            "active_ids_original": unique_preserve_order(payload.get("active_ids_original", [])),
            "active_ids_for_split": unique_preserve_order(payload.get("active_ids_for_split", [])),
            "known_active_ids": unique_preserve_order(payload.get("known_active_ids", [])),
            "test_active_ids": unique_preserve_order(payload.get("test_active_ids", [])),
            "inactive_ids": unique_preserve_order(payload.get("inactive_ids", [])),
            "eval_negative_ids": unique_preserve_order(payload.get("eval_negative_ids", [])),
        }
        row["n_active_original"] = len(row["active_ids_original"])
        row["n_active_for_split"] = len(row["active_ids_for_split"])
        row["n_known"] = len(row["known_active_ids"])
        row["n_test"] = len(row["test_active_ids"])
        row["n_inactive"] = len(row["inactive_ids"])
        rows.append(row)

    return pd.DataFrame(rows)


def _stable_intersection_preserve_order(list_of_id_lists: Sequence[Sequence[str]]) -> List[str]:
    if not list_of_id_lists:
        return []
    normalized = [unique_preserve_order(ids) for ids in list_of_id_lists]
    common = set(normalized[0])
    for ids in normalized[1:]:
        common &= set(ids)
    return [x for x in normalized[0] if x in common]


def _sample_preserve_uniques(ids: Sequence[str], n: int, random_state: int) -> List[str]:
    ids_u = unique_preserve_order(ids)
    if n >= len(ids_u):
        return ids_u
    rng = random.Random(random_state)
    idxs = list(range(len(ids_u)))
    rng.shuffle(idxs)
    chosen = [ids_u[i] for i in idxs[:n]]
    return unique_preserve_order(chosen)


def build_family_forbidden_ligand_payloads(
    *,
    dataset_output_dir: str | Path,
    targets_pfam_df: pd.DataFrame,
    inactive_multiplier: int = 100,
    random_state: int = 42,
    mpgs: Optional[Sequence[str]] = None,
    kinase_pfams: Sequence[str] = ("PF00069", "PF07714"),
    kinase_family_name: str = "kinase",
) -> Dict[str, Dict[str, Any]]:
    """
    Build family-level payloads from previously saved `dataset.json` files.

    Sample `forbidden_inactive_ids` from the intersection of inactive sets for
    all targets in the family so it can be reused in any dataset or evaluation
    for that family.
    """
    if inactive_multiplier <= 0:
        raise ValueError("inactive_multiplier must be a positive integer.")

    df_payloads = load_saved_target_dataset_payloads(dataset_output_dir)
    if df_payloads.empty:
        return {}

    target_to_family, family_to_targets = build_family_group_map(
        targets_pfam_df,
        mpgs=mpgs,
        kinase_pfams=kinase_pfams,
        kinase_family_name=kinase_family_name,
    )

    available_targets = set(df_payloads["target"].astype(str).tolist())
    family_payloads: Dict[str, Dict[str, Any]] = {}

    for family_id, family_targets_all in family_to_targets.items():
        family_targets = [t for t in family_targets_all if t in available_targets]
        if not family_targets:
            continue

        family_df = (
            df_payloads[df_payloads["target"].isin(family_targets)]
            .copy()
            .sort_values("target")
            .reset_index(drop=True)
        )

        test_ids = unique_preserve_order([
            lig
            for ligs in family_df["test_active_ids"].tolist()
            for lig in ligs
        ])
        known_ids = unique_preserve_order([
            lig
            for ligs in family_df["known_active_ids"].tolist()
            for lig in ligs
        ])
        overlap_known_test = [lig for lig in test_ids if lig in set(known_ids)]

        inactive_intersection = _stable_intersection_preserve_order(
            family_df["inactive_ids"].tolist()
        )
        per_target_n_valid_actives = {
            str(row["target"]): int(row["n_active_for_split"])
            for _, row in family_df.iterrows()
        }
        family_max_valid_actives = max(per_target_n_valid_actives.values()) if per_target_n_valid_actives else 0
        n_inactive_requested = int(inactive_multiplier) * int(family_max_valid_actives)

        forbidden_inactive_ids = _sample_preserve_uniques(
            inactive_intersection,
            n=n_inactive_requested,
            random_state=random_state,
        )

        per_target_summary = []
        for _, row in family_df.iterrows():
            payload = row["payload"]
            pfam_ids = unique_preserve_order(payload.get("pfam_ids", []))
            family_pfams = [pf for pf in pfam_ids if target_to_family.get(str(row["target"])) == str(family_id) or pf in kinase_pfams]
            per_target_summary.append({
                "target": str(row["target"]),
                "pfam_ids": pfam_ids,
                "n_active_original": int(row["n_active_original"]),
                "n_active_for_split": int(row["n_active_for_split"]),
                "n_known_active_ids": int(row["n_known"]),
                "n_test_active_ids": int(row["n_test"]),
                "n_inactive_ids": int(row["n_inactive"]),
                "dataset_json": str(row["dataset_json"]),
                "family_id": str(family_id),
                "family_pfams": unique_preserve_order(family_pfams),
            })

        forbidden_all_ids = unique_preserve_order(test_ids + forbidden_inactive_ids)
        family_pfams = unique_preserve_order([
            pf
            for pfams in family_df["pfam_ids"].tolist()
            for pf in pfams
            if (pf in set(kinase_pfams) and str(family_id) == str(kinase_family_name)) or pf == str(family_id)
        ])

        family_payloads[str(family_id)] = {
            "family_id": str(family_id),
            "targets": unique_preserve_order(family_targets),
            "pfams": family_pfams,
            "forbidden_test_ids": test_ids,
            "forbidden_inactive_ids": forbidden_inactive_ids,
            "forbidden_all_ids": forbidden_all_ids,
            "family_max_valid_actives": int(family_max_valid_actives),
            "inactive_multiplier": int(inactive_multiplier),
            "inactive_sampling_source": "intersection",
            "n_targets": int(len(family_targets)),
            "n_test_forbidden": int(len(test_ids)),
            "n_inactive_forbidden_requested": int(n_inactive_requested),
            "n_inactive_forbidden_selected": int(len(forbidden_inactive_ids)),
            "n_inactive_intersection_available": int(len(inactive_intersection)),
            "per_target_n_valid_actives": per_target_n_valid_actives,
            "per_target_summary": per_target_summary,
            "known_test_overlap_ids": overlap_known_test,
            "sampling_random_state": int(random_state),
        }

    return family_payloads


def run_build_and_save_family_forbidden_ligands(
    *,
    dataset_output_dir: str | Path,
    family_output_dir: str | Path,
    targets_pfam_df: pd.DataFrame,
    inactive_multiplier: int = 100,
    random_state: int = 42,
    mpgs: Optional[Sequence[str]] = None,
    kinase_pfams: Sequence[str] = ("PF00069", "PF07714"),
    kinase_family_name: str = "kinase",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Save for each family:
      - forbidden_ligands.json
      - target_summary.csv
    """
    family_output_dir = Path(family_output_dir)
    family_output_dir.mkdir(parents=True, exist_ok=True)

    family_payloads = build_family_forbidden_ligand_payloads(
        dataset_output_dir=dataset_output_dir,
        targets_pfam_df=targets_pfam_df,
        inactive_multiplier=inactive_multiplier,
        random_state=random_state,
        mpgs=mpgs,
        kinase_pfams=kinase_pfams,
        kinase_family_name=kinase_family_name,
    )

    summary_rows = []
    for family_id, payload in sorted(family_payloads.items()):
        safe_family = str(family_id).replace("/", "_")
        family_dir = family_output_dir / safe_family
        family_dir.mkdir(parents=True, exist_ok=True)

        json_path = family_dir / "forbidden_ligands.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        target_summary_path = family_dir / "target_summary.csv"
        pd.DataFrame(payload["per_target_summary"]).to_csv(target_summary_path, index=False)

        if verbose:
            print(
                f"[family={family_id}] targets={payload['n_targets']} "
                f"test_forbidden={payload['n_test_forbidden']} "
                f"inactive_forbidden={payload['n_inactive_forbidden_selected']}/"
                f"{payload['n_inactive_forbidden_requested']}"
            )

        summary_rows.append({
            "family_id": str(family_id),
            "n_targets": int(payload["n_targets"]),
            "n_test_forbidden": int(payload["n_test_forbidden"]),
            "n_inactive_forbidden_requested": int(payload["n_inactive_forbidden_requested"]),
            "n_inactive_forbidden_selected": int(payload["n_inactive_forbidden_selected"]),
            "n_inactive_intersection_available": int(payload["n_inactive_intersection_available"]),
            "family_max_valid_actives": int(payload["family_max_valid_actives"]),
            "forbidden_json": str(json_path),
            "target_summary_csv": str(target_summary_path),
        })

    return pd.DataFrame(summary_rows)


def load_saved_family_forbidden_payloads(family_output_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """
    Load family-level `forbidden_ligands.json` files and return a mapping from
    family_id to payload.
    """
    family_output_dir = Path(family_output_dir)
    if not family_output_dir.exists():
        raise FileNotFoundError(f"family_output_dir does not exist: {family_output_dir}")

    payloads: Dict[str, Dict[str, Any]] = {}
    for json_path in sorted(family_output_dir.glob("*/forbidden_ligands.json")):
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        family_id = str(payload.get("family_id", json_path.parent.name))
        payload["_forbidden_json_path"] = str(json_path)
        payloads[family_id] = payload
    return payloads


def build_target_final_eval_dataset_payload(
    *,
    target_payload: Dict[str, Any],
    family_forbidden_payload: Dict[str, Any],
    inactive_multiplier: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Build the final evaluation dataset for one target:
      - known_active_ids: known target ligands
      - unknown_active_ids: actives to retrieve (test_active_ids)
      - inactive_ids: sample from the family's forbidden_inactive_ids
    """
    if inactive_multiplier <= 0:
        raise ValueError("inactive_multiplier must be a positive integer.")

    target = str(target_payload["target"])
    family_id = str(family_forbidden_payload["family_id"])

    known_active_ids = unique_preserve_order(target_payload.get("known_active_ids", []))
    unknown_active_ids = unique_preserve_order(target_payload.get("test_active_ids", []))
    family_forbidden_inactive_ids = unique_preserve_order(
        family_forbidden_payload.get("forbidden_inactive_ids", [])
    )

    n_unknown = len(unknown_active_ids)
    n_inactive_requested = int(inactive_multiplier) * int(n_unknown)
    inactive_ids = _sample_preserve_uniques(
        family_forbidden_inactive_ids,
        n=n_inactive_requested,
        random_state=random_state,
    )

    eval_dataset_ids = unique_preserve_order(unknown_active_ids + inactive_ids)

    return {
        "target": target,
        "family_id": family_id,
        "pfam_ids": unique_preserve_order(target_payload.get("pfam_ids", [])),
        "known_active_ids": known_active_ids,
        "unknown_active_ids": unknown_active_ids,
        "inactive_ids": inactive_ids,
        "eval_dataset_ids": eval_dataset_ids,
        "inactive_sampling_source": "family_forbidden_inactives",
        "inactive_multiplier": int(inactive_multiplier),
        "n_known": int(len(known_active_ids)),
        "n_unknown": int(n_unknown),
        "n_inactive_requested": int(n_inactive_requested),
        "n_inactive_selected": int(len(inactive_ids)),
        "family_forbidden_inactive_total": int(len(family_forbidden_inactive_ids)),
        "sampling_random_state": int(random_state),
        "target_dataset_json": target_payload.get("_dataset_json_path"),
        "family_forbidden_json": family_forbidden_payload.get("_forbidden_json_path"),
    }


def run_build_and_save_final_eval_datasets(
    *,
    dataset_output_dir: str | Path,
    family_output_dir: str | Path,
    targets_pfam_df: pd.DataFrame,
    final_output_dir: str | Path,
    inactive_multiplier: int = 100,
    random_state: int = 42,
    mpgs: Optional[Sequence[str]] = None,
    kinase_pfams: Sequence[str] = ("PF00069", "PF07714"),
    kinase_family_name: str = "kinase",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Save a final evaluation dataset for each target containing:
      - known ligands
      - unknown ligands
      - inactives sampled from the family's forbidden list
    """
    final_output_dir = Path(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    df_target_payloads = load_saved_target_dataset_payloads(dataset_output_dir)
    family_payloads = load_saved_family_forbidden_payloads(family_output_dir)
    target_to_family, _ = build_family_group_map(
        targets_pfam_df,
        mpgs=mpgs,
        kinase_pfams=kinase_pfams,
        kinase_family_name=kinase_family_name,
    )

    summary_rows = []
    for _, row in df_target_payloads.sort_values("target").iterrows():
        target = str(row["target"])
        family_id = target_to_family.get(target)
        if family_id is None:
            if verbose:
                print(f"[target={target}] skip: no family assigned")
            summary_rows.append({
                "target": target,
                "family_id": None,
                "n_known": int(row["n_known"]),
                "n_unknown": int(row["n_test"]),
                "n_inactive_requested": None,
                "n_inactive_selected": None,
                "dataset_json": str(row["dataset_json"]),
                "family_forbidden_json": None,
                "final_eval_json": None,
                "skip_reason": "missing_family_mapping",
            })
            continue

        family_payload = family_payloads.get(str(family_id))
        if family_payload is None:
            if verbose:
                print(f"[target={target}] skip: forbidden_ligands does not exist for family {family_id}")
            summary_rows.append({
                "target": target,
                "family_id": str(family_id),
                "n_known": int(row["n_known"]),
                "n_unknown": int(row["n_test"]),
                "n_inactive_requested": None,
                "n_inactive_selected": None,
                "dataset_json": str(row["dataset_json"]),
                "family_forbidden_json": None,
                "final_eval_json": None,
                "skip_reason": "missing_family_forbidden_payload",
            })
            continue

        target_payload = dict(row["payload"])
        target_payload["_dataset_json_path"] = str(row["dataset_json"])

        final_payload = build_target_final_eval_dataset_payload(
            target_payload=target_payload,
            family_forbidden_payload=family_payload,
            inactive_multiplier=inactive_multiplier,
            random_state=random_state,
        )

        target_dir = final_output_dir / str(target).replace("/", "_")
        target_dir.mkdir(parents=True, exist_ok=True)
        final_json_path = target_dir / "final_eval_dataset.json"
        with final_json_path.open("w", encoding="utf-8") as f:
            json.dump(final_payload, f, ensure_ascii=False, indent=2)

        if verbose:
            print(
                f"[target={target}] family={family_id} "
                f"known={final_payload['n_known']} unknown={final_payload['n_unknown']} "
                f"inactive={final_payload['n_inactive_selected']}/{final_payload['n_inactive_requested']}"
            )

        summary_rows.append({
            "target": target,
            "family_id": str(family_id),
            "n_known": int(final_payload["n_known"]),
            "n_unknown": int(final_payload["n_unknown"]),
            "n_inactive_requested": int(final_payload["n_inactive_requested"]),
            "n_inactive_selected": int(final_payload["n_inactive_selected"]),
            "dataset_json": str(row["dataset_json"]),
            "family_forbidden_json": str(family_payload.get("_forbidden_json_path")),
            "final_eval_json": str(final_json_path),
            "skip_reason": None,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = final_output_dir / "final_eval_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    return summary_df

def build_eval_pool_ids(
    test_active_ids: List,
    background_ids: List,
    ratio_neg_to_pos: int = 200,
    n_bg_cap: Optional[int] = None,
    random_state: int = 0,
) -> Dict[str, List]:
    """
    Build an evaluation pool with a fixed negative-to-positive ratio.
    Return:
      - pool_ids: shuffled list
      - pos_ids: test positives
      - neg_ids: background sample
    """
    rng = random.Random(random_state)
    pos_ids = list(dict.fromkeys(test_active_ids))  # Deduplicate while preserving order
    n_pos = len(pos_ids)

    n_bg = ratio_neg_to_pos * n_pos
    if n_bg_cap is not None:
        n_bg = min(n_bg, n_bg_cap)

    # Remove positive IDs from the background
    pos_set = set(pos_ids)
    bg_candidates = [x for x in background_ids if x not in pos_set]

    if len(bg_candidates) < n_bg:
        # If fewer are available, use all of them
        neg_ids = bg_candidates
    else:
        idxs = list(range(len(bg_candidates)))
        rng.shuffle(idxs)
        neg_ids = [bg_candidates[i] for i in idxs[:n_bg]]

    pool = pos_ids + neg_ids
    rng.shuffle(pool)

    return {
        "pool_ids": pool,
        "pos_ids": pos_ids,
        "neg_ids": neg_ids,
        "n_pos": n_pos,
        "n_neg": len(neg_ids),
        "ratio_neg_to_pos_effective": (len(neg_ids) / n_pos) if n_pos else None
    }

def max_score_per_target(
    query_ids,
    rep_queries,
    rep_targets,
    metric,
    *,
    device="cpu",                 # "cpu", "cuda", or externally resolved "auto"
    q_batch_size=256,
    target_chunk_size=200_000,
    assume_normalized=None,       # For cosine; may be None when metadata says normalized
    clamp_max=None,               # For example, 0.7 to cap the score
):
    """
    Return a scores_max vector of length N_targets where:
        scores_max[j] = max_{q in query_ids} sim(q, target_j)

    - Do not store the complete QxX matrix; reduce by maximum within each chunk.
    - Support metric="tanimoto" (packed Morgan bits) or metric="cosine"
      (floating-point embeddings), following validate_metric/score_block rules.
    """

    n_targets = rep_targets.n_ligands
    out = np.full((n_targets,), -np.inf, dtype=np.float32)

    # Batch queries to avoid loading thousands of seeds at once
    qids = list(query_ids)
    if len(qids) == 0 or n_targets == 0:
        return np.zeros((n_targets,), dtype=np.float32)

    for q0 in range(0, len(qids), q_batch_size):
        q1 = min(q0 + q_batch_size, len(qids))
        batch_ids = qids[q0:q1]
        Q = rep_queries.get_raw_by_ids(batch_ids)  # Raw packed uint8 or float values
        if Q.shape[0] != len(batch_ids):
            raise ValueError("Query vectors do not match query_ids; IDs are missing from the store/representation.")

        # GPU cosine fast path avoids repeatedly uploading Q
        precomputed_q = None
        use_gpu_fast_cosine = (str(device).startswith("cuda") and metric == "cosine")
        if use_gpu_fast_cosine:
            import torch
            precomputed_q = metrics.prepare_cosine_queries_torch(
                Q, device=torch.device(device), assume_normalized=assume_normalized, q_meta=rep_queries.meta
            )

        # Scan E in chunks
        for start, end, X in rep_targets.iter_raw_chunks(target_chunk_size):
            if X.size == 0:
                continue

            if precomputed_q is not None:
                # Return torch.Tensor with shape (Q, chunk)
                scores_t = metrics.cosine_torch_tensor_prepared_q(
                    precomputed_q, X, assume_normalized=assume_normalized, q_meta=rep_queries.meta
                )
                scores = scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
            else:
                scores = metrics.score_block(
                    metric, Q, X,
                    q_meta=rep_queries.meta,
                    x_meta=rep_targets.meta,
                    device=device,
                    assume_normalized=assume_normalized,
                    return_torch=False,
                ).astype(np.float32, copy=False)

            # Reduce: maximum across queries => (chunk,)
            chunk_max = scores.max(axis=0)

            if clamp_max is not None:
                # Cap scores above the limit without removing them
                chunk_max = np.minimum(chunk_max, float(clamp_max))

            # Merge with the global maximum
            out[start:end] = np.maximum(out[start:end], chunk_max)

    # Replace -inf with zero in the edge case where it remains
    out[~np.isfinite(out)] = 0.0
    return out

def percentile_bins(scores, percentiles=(99, 95, 90, 80, 50)):
    """
    Return:
      - cuts: {p: threshold} mapping with actual thresholds
      - bin_id: integer array of assigned bins (0 = >=p_max, 1 = >=p_next, etc.)
    """
    s = np.asarray(scores, dtype=np.float32)
    ps = list(percentiles)
    ps_sorted = sorted(ps, reverse=True)

    cuts = {p: float(np.percentile(s, p)) for p in ps_sorted}

    # Build bins; example percentiles: [99, 95, 90]
    # bin 0: score >= cut99
    # bin 1: cut95 <= score < cut99
    # bin 2: cut90 <= score < cut95
    # bin 3: score < cut90  (remainder)
    thresholds = [cuts[p] for p in ps_sorted]
    bin_id = np.full_like(s, fill_value=len(thresholds), dtype=np.int32)  # Default: remainder

    prev = np.inf
    for i, thr in enumerate(thresholds):
        mask = (s >= thr) & (s < prev)
        bin_id[mask] = i
        prev = thr

    return cuts, bin_id

def build_eval_table(
    scores_max,
    store_targets,
    *,
    percentiles=(99, 95, 90, 80, 50),
    active_ids=None,            # Set/list of active chem_comp_id values in E
    id_field="chem_comp_id",
    extra_fields=("smiles",),
):
    """
    Return a DataFrame with:
      lig_idx, chem_comp_id, smiles..., score, bin, is_active
    """
    cuts, bin_id = percentile_bins(scores_max, percentiles=percentiles)

    ligs = store_targets.ligands.copy()
    if "lig_idx" not in ligs.columns:
        raise ValueError("store_targets.ligands must contain a 'lig_idx' column.")

    df = pd.DataFrame({
        "lig_idx": np.asarray(ligs["lig_idx"].values, dtype=np.int64),
        "score": np.asarray(scores_max, dtype=np.float32),
        "bin": np.asarray(bin_id, dtype=np.int32),
    })

    # Merge basic metadata
    cols = [id_field] + [c for c in extra_fields if c in ligs.columns]
    df = df.merge(ligs[["lig_idx"] + cols], on="lig_idx", how="left")

    if active_ids is not None:
        active_set = set(active_ids)
        df["is_active"] = df[id_field].astype(str).isin(active_set)
    else:
        df["is_active"] = False

    return df, cuts

def enrichment_factor_by_bin(df, *, bin_col="bin", label_col="is_active"):
    """
    EF(bin) = (actives_in_bin / size_bin) / (total_actives / total_size)

    Preserve the original behavior: one row per observed bin, with the classic
    n / actives / actives_rate / EF columns.
    """
    total = len(df)
    total_actives = int(df[label_col].sum())
    if total == 0 or total_actives == 0:
        raise ValueError("No data or active compounds are available for EF calculation.")

    base_rate = total_actives / total

    rows = []
    for b, g in df.groupby(bin_col, sort=True):
        n = len(g)
        a = int(g[label_col].sum())
        rate = (a / n) if n > 0 else 0.0
        ef = (rate / base_rate) if base_rate > 0 else np.nan
        rows.append({"bin": int(b), "n": n, "actives": a, "actives_rate": rate, "EF": ef})

    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)


def normalize_ef_mode(ef_mode: str = "both") -> str:
    """
    Normalize the EF calculation mode.

    Valid values:
      - "band": EF over disjoint bands
      - "cumulative": classic cumulative EF (top compounds through that percentile)
      - "both": compute both modes and return them in separate columns
    """
    ef_mode = str(ef_mode).strip().lower()
    aliases = {
        "bands": "band",
        "bin": "band",
        "bins": "band",
        "disjoint": "band",
        "accum": "cumulative",
        "accumulated": "cumulative",
        "classic": "cumulative",
        "all": "both",
    }
    ef_mode = aliases.get(ef_mode, ef_mode)

    valid = {"band", "cumulative", "both"}
    if ef_mode not in valid:
        raise ValueError(f"Invalid ef_mode: {ef_mode}. Use one of {sorted(valid)}")
    return ef_mode


def enrichment_factor_by_percentile_mode(
    df,
    *,
    percentiles=(99.5, 99, 98.5, 98, 95, 90, 80, 50),
    bin_col="bin",
    label_col="is_active",
    ef_mode: str = "both",
):
    """
    Compute percentile EF in wide format, with separate columns for band and
    cumulative modes.

    For each percentile p:
      - band: use only the bin corresponding to p
      - cumulative: use score >= cut(p), equivalent to accumulating bins from the top

    Always return one row per percentile and distinct columns for each mode:
      n_band, actives_band, actives_rate_band, EF_band
      n_cumulative, actives_cumulative, actives_rate_cumulative, EF_cumulative

    If ef_mode requests only one mode, leave the other as NaN.
    """
    ef_mode = normalize_ef_mode(ef_mode)

    total = len(df)
    total_actives = int(df[label_col].sum())
    if total == 0 or total_actives == 0:
        raise ValueError("No data or active compounds are available for EF calculation.")

    base_rate = total_actives / total
    ps_sorted = sorted([float(p) for p in percentiles], reverse=True)

    rows = []
    for bin_id, percentile in enumerate(ps_sorted):
        row = {
            "bin": int(bin_id),
            "percentile": float(percentile),
        }

        if ef_mode in {"band", "both"}:
            g_band = df.loc[df[bin_col] == bin_id]
            n_band = int(len(g_band))
            a_band = int(g_band[label_col].sum()) if n_band > 0 else 0
            rate_band = (a_band / n_band) if n_band > 0 else np.nan
            ef_band = (rate_band / base_rate) if (n_band > 0 and base_rate > 0) else np.nan
            row.update({
                "n_band": n_band,
                "actives_band": a_band,
                "actives_rate_band": rate_band,
                "EF_band": ef_band,
            })
        else:
            row.update({
                "n_band": np.nan,
                "actives_band": np.nan,
                "actives_rate_band": np.nan,
                "EF_band": np.nan,
            })

        if ef_mode in {"cumulative", "both"}:
            g_cum = df.loc[df[bin_col] <= bin_id]
            n_cum = int(len(g_cum))
            a_cum = int(g_cum[label_col].sum()) if n_cum > 0 else 0
            rate_cum = (a_cum / n_cum) if n_cum > 0 else np.nan
            ef_cum = (rate_cum / base_rate) if (n_cum > 0 and base_rate > 0) else np.nan
            row.update({
                "n_cumulative": n_cum,
                "actives_cumulative": a_cum,
                "actives_rate_cumulative": rate_cum,
                "EF_cumulative": ef_cum,
            })
        else:
            row.update({
                "n_cumulative": np.nan,
                "actives_cumulative": np.nan,
                "actives_rate_cumulative": np.nan,
                "EF_cumulative": np.nan,
            })

        rows.append(row)

    return pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)

def max_score_for_pool_ids(
    seed_ids,
    pool_ids,
    *,
    rep_queries,
    rep_targets,
    metric,
    device="cpu",
    q_batch_size=256,
    pool_batch_size=50_000,
    assume_normalized=None,
    clamp_max=None,
):
    """
    Return scores aligned one-to-one with pool_ids:
        scores[i] = max_{q in seed_ids} sim(q, pool_ids[i])

    - Do not scan the entire database; load only pool_ids vectors in batches.
    - Use metrics.score_block() (Tanimoto for Morgan, cosine for embeddings).
    """
    seed_ids = list(seed_ids)
    pool_ids = list(pool_ids)

    if len(seed_ids) == 0 or len(pool_ids) == 0:
        return np.zeros((len(pool_ids),), dtype=np.float32)

    out = np.full((len(pool_ids),), -np.inf, dtype=np.float32)

    # Optionally precompute the GPU cosine fast path
    precomputed_q_batches = None
    use_gpu_fast_cosine = (str(device).startswith("cuda") and metric == "cosine")
    if use_gpu_fast_cosine:
        import torch
        precomputed_q_batches = []
        for q0 in range(0, len(seed_ids), q_batch_size):
            q1 = min(q0 + q_batch_size, len(seed_ids))
            Q = rep_queries.get_raw_by_ids(seed_ids[q0:q1])
            precomputed_q_batches.append(
                metrics.prepare_cosine_queries_torch(
                    Q,
                    device=torch.device(device),
                    assume_normalized=assume_normalized,
                    q_meta=rep_queries.meta,
                )
            )

    for p0 in range(0, len(pool_ids), pool_batch_size):
        p1 = min(p0 + pool_batch_size, len(pool_ids))
        pool_chunk_ids = pool_ids[p0:p1]
        X = rep_targets.get_raw_by_ids(pool_chunk_ids)

        # Chunk accumulator: maximum across seeds
        chunk_best = np.full((len(pool_chunk_ids),), -np.inf, dtype=np.float32)

        if precomputed_q_batches is not None:
            # cosine GPU fast path
            for q_pre in precomputed_q_batches:
                scores_t = metrics.cosine_torch_tensor_prepared_q(
                    q_pre,
                    X,
                    assume_normalized=assume_normalized,
                    q_meta=rep_queries.meta,
                )
                scores = scores_t.detach().cpu().numpy().astype(np.float32, copy=False)
                chunk_best = np.maximum(chunk_best, scores.max(axis=0))
        else:
            # Generic path: query batches
            for q0 in range(0, len(seed_ids), q_batch_size):
                q1 = min(q0 + q_batch_size, len(seed_ids))
                Q = rep_queries.get_raw_by_ids(seed_ids[q0:q1])

                scores = metrics.score_block(
                    metric,
                    Q,
                    X,
                    q_meta=rep_queries.meta,
                    x_meta=rep_targets.meta,
                    device=device,
                    assume_normalized=assume_normalized,
                    return_torch=False,
                ).astype(np.float32, copy=False)

                chunk_best = np.maximum(chunk_best, scores.max(axis=0))

        if clamp_max is not None:
            chunk_best = np.minimum(chunk_best, float(clamp_max))

        out[p0:p1] = chunk_best

    out[~np.isfinite(out)] = 0.0
    return out

def build_eval_table_from_pool_ids(
    pool_ids,
    scores,
    *,
    store,                      # LigandStore used to retrieve SMILES/metadata
    percentiles=(99, 95, 90, 80, 50),
    active_ids=None,            # Set/list of positive chem_comp_id values in the pool
    id_field="chem_comp_id",
    extra_fields=("smiles",),
):
    """
    Build a DataFrame for the pool only:
      chem_comp_id, smiles..., score, bin, is_active
    Return (df, cuts), where cuts[p] is the actual score threshold at that percentile.
    """
    pool_ids = [str(x) for x in pool_ids]
    scores = np.asarray(scores, dtype=np.float32)
    if len(pool_ids) != len(scores):
        raise ValueError("pool_ids and scores must have the same length.")

    # Percentiles over the pool
    ps_sorted = sorted(list(percentiles), reverse=True)
    cuts = {p: float(np.percentile(scores, p)) for p in ps_sorted}

    thresholds = [cuts[p] for p in ps_sorted]
    bin_id = np.full((len(scores),), fill_value=len(thresholds), dtype=np.int32)
    prev = np.inf
    for i, thr in enumerate(thresholds):
        mask = (scores >= thr) & (scores < prev)
        bin_id[mask] = i
        prev = thr

    df = pd.DataFrame({
        id_field: pool_ids,
        "score": scores,
        "bin": bin_id,
    })

    # Merge metadata from store.ligands by chem_comp_id
    ligs = store.ligands.copy()
    ligs[id_field] = ligs[id_field].astype(str)

    cols = [id_field] + [c for c in extra_fields if c in ligs.columns]
    df = df.merge(ligs[cols], on=id_field, how="left")

    if active_ids is not None:
        active_set = set(map(str, active_ids))
        df["is_active"] = df[id_field].isin(active_set)
    else:
        df["is_active"] = False

    return df, cuts

PERCENTILES = (99.5, 99, 98.5, 98, 95, 90, 80, 50)
TRACKED_RETRIEVED_PERCENTILES = (99.5, 99.0, 98.0)

def ef_results_to_long(df_ef: pd.DataFrame, cuts: dict, target: str,
                       method: str = "morgan_tanimoto", percentiles=PERCENTILES) -> pd.DataFrame:
    """
    Convert a per-target result (df_ef + cuts) to long format for aggregate plotting.

    df_ef: expected columns -> bin, n, actives, actives_rate, EF
    cuts: {percentile_int: threshold_float} mapping returned by build_eval_table_from_pool_ids
    """
    ps_sorted = sorted(list(percentiles), reverse=True)
    bin_to_percentile = {i: ps_sorted[i] for i in range(len(ps_sorted))}

    out = df_ef.copy()
    out["target"] = target
    out["method"] = method
    out["percentile"] = out["bin"].map(bin_to_percentile)

    def _cut_lookup(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return np.nan
        return cuts.get(int(p), np.nan)

    out["score_cut"] = out["percentile"].map(_cut_lookup)

    # Drop the remainder bin (bin == len(percentiles)), which has no assigned percentile
    out = out.dropna(subset=["percentile"]).copy()
    #out["percentile"] = out["percentile"].astype(int)

    return out[["target", "method", "percentile", "score_cut", "n", "actives", "actives_rate", "EF"]]


def plot_ef_boxplot_with_scorecuts(df_long: pd.DataFrame, *,
                                   title: str = "EF by percentile (across targets)",
                                   logy: bool = True,
                                   show_cut_text: bool = True,
                                   percentiles=PERCENTILES):
    """
    EF box plot (distribution across targets) for each percentile.

    X: percentile (plus median score cutoff)
    Y: EF (log scale recommended)
    """
    df = df_long.dropna(subset=["percentile", "EF"]).copy()
    #df["percentile"] = df["percentile"].astype(int)

    ps_sorted = sorted(list(percentiles), reverse=True)

    data = []
    labels = []
    for p in ps_sorted:
        vals = df.loc[df["percentile"] == p, "EF"].astype(float).values
        vals = vals[np.isfinite(vals) & (vals > 0)]
        data.append(vals)

        if show_cut_text:
            med_cut = np.nanmedian(df.loc[df["percentile"] == p, "score_cut"].astype(float).values)
            labels.append(f"{p}\ncut~{med_cut:.3f}")
        else:
            labels.append(str(p))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xlabel("Percentile bin (median score cut)")
    ax.set_ylabel("Enrichment Factor (EF)")
    ax.set_title(title)

    if logy:
        ax.set_yscale("log")
    ax.axhline(1.0, linewidth=1)  # baseline EF=1

    plt.tight_layout()
    return fig, ax

# ------------------------------------------------------------
# General utilities
# ------------------------------------------------------------

def unique_preserve_order(ids: Sequence) -> List[str]:
    """Deduplicate while preserving order, convert values to strings, and ignore NaN."""
    seen = set()
    out = []
    for x in ids:
        if pd.isna(x):
            continue
        sx = str(x)
        if sx not in seen:
            seen.add(sx)
            out.append(sx)
    return out


def stable_set_difference_preserve_order(ids: Sequence, excluded_ids: Sequence) -> List[str]:
    """Stable set difference with deterministic order and the contents of set(ids) - set(excluded_ids)."""
    excluded = set(unique_preserve_order(excluded_ids))
    return [x for x in unique_preserve_order(ids) if x not in excluded]


def ef_results_to_long_fixed(
    df_ef: pd.DataFrame,
    cuts: dict,
    target: str,
    method: str = "morgan_tanimoto",
    percentiles=(99.5, 99, 98.5, 98, 95, 90, 80, 50),
) -> pd.DataFrame:
    """
    Corrected version for floating-point percentiles (for example, 99.5 and 98.5).

    Accept both the historical output (n / actives / actives_rate / EF) and the
    new output with separate columns by mode:
      - n_band / actives_band / actives_rate_band / EF_band
      - n_cumulative / actives_cumulative / actives_rate_cumulative / EF_cumulative
    """
    ps_sorted = sorted(list(percentiles), reverse=True)
    bin_to_percentile = {i: ps_sorted[i] for i in range(len(ps_sorted))}

    out = df_ef.copy()
    out["target"] = target
    out["method"] = method

    if "percentile" not in out.columns:
        out["percentile"] = out["bin"].map(bin_to_percentile)

    def _cut_lookup(p):
        if p is None or (isinstance(p, float) and np.isnan(p)):
            return np.nan
        return cuts.get(p, np.nan)  # Do not cast to int

    out["score_cut"] = out["percentile"].map(_cut_lookup)

    # Drop the remainder bin
    out = out.dropna(subset=["percentile"]).copy()

    preferred_cols = [
        "target", "method", "percentile", "score_cut",
        "n", "actives", "actives_rate", "EF",
        "n_band", "actives_band", "actives_rate_band", "EF_band",
        "n_cumulative", "actives_cumulative", "actives_rate_cumulative", "EF_cumulative",
    ]
    keep_cols = [c for c in preferred_cols if c in out.columns]
    return out[keep_cols]

def filter_ids_present_in_rep(ids, rep):
    """
    Return (present_ids, missing_ids) according to rep.id_to_idx.
    Convert all values to strings while preserving order.
    """
    ids_u = unique_preserve_order(ids)
    id_set = rep.id_to_idx  # dict chem_comp_id -> idx
    present, missing = [], []
    for cid in ids_u:
        if cid in id_set:
            present.append(cid)
        else:
            missing.append(cid)
    return present, missing

def filter_eval_ids_by_rep(pool_ids, pos_ids, seed_ids, rep):
    pool_ids = [str(x) for x in pool_ids]
    pos_ids = [str(x) for x in pos_ids]
    seed_ids = [str(x) for x in seed_ids]

    pool_repr, pool_missing = filter_ids_present_in_rep(pool_ids, rep)
    pool_repr_set = set(pool_repr)

    # Positives only within the representable pool
    pos_ids_repr = [x for x in pos_ids if x in pool_repr_set]

    seed_repr, seed_missing = filter_ids_present_in_rep(seed_ids, rep)

    meta = {
        "n_pool_missing_in_rep": len(pool_missing),
        "n_seed_missing_in_rep": len(seed_missing),
        "n_pool_repr": len(pool_repr),
        "n_seed_repr": len(seed_repr),
        "n_pos_in_pool_repr": len(pos_ids_repr),
    }
    return pool_repr, pos_ids_repr, seed_repr, meta


def _validate_eval_table_integrity(
    df_eval: pd.DataFrame,
    *,
    expected_pool_ids: Sequence[str],
    expected_pos_ids: Sequence[str],
    id_field: str = "chem_comp_id",
) -> None:
    """
    Validate that the evaluation DataFrame exactly preserves the representable
    pool IDs and expected positive actives.
    """
    if id_field not in df_eval.columns:
        raise ValueError(f"df_eval does not contain the required column {id_field!r}.")

    if df_eval[id_field].isna().any():
        raise ValueError("df_eval contains null ligand IDs after the metadata merge.")

    observed_pool_ids = df_eval[id_field].astype(str).tolist()
    if len(observed_pool_ids) != len(expected_pool_ids):
        raise ValueError(
            "The length of df_eval does not match the evaluated pool; ligands may have been duplicated or lost."
        )

    if len(set(observed_pool_ids)) != len(observed_pool_ids):
        raise ValueError("df_eval contains duplicate ligand IDs, preventing reliable set tracking.")

    if set(observed_pool_ids) != set(map(str, expected_pool_ids)):
        raise ValueError("df_eval does not exactly preserve the evaluated pool IDs.")

    observed_pos_ids = set(df_eval.loc[df_eval["is_active"], id_field].astype(str).tolist())
    expected_pos_ids_set = set(map(str, expected_pos_ids))
    if observed_pos_ids != expected_pos_ids_set:
        raise ValueError("The positive actives in df_eval do not match the expected positive IDs.")


def _build_retrieved_active_set_rows(
    *,
    df_eval: pd.DataFrame,
    target_id: str,
    method_label: str,
    percentiles: Sequence[float],
    tracked_percentiles: Sequence[float] = TRACKED_RETRIEVED_PERCENTILES,
    id_field: str = "chem_comp_id",
) -> List[Dict[str, Any]]:
    """
    Extract actives retrieved at each percentile using the bins already built
    for EF. Do not redefine ranking or cutoff logic.
    """
    available_percentiles = {float(p) for p in percentiles}
    missing_percentiles = [float(p) for p in tracked_percentiles if float(p) not in available_percentiles]
    if missing_percentiles:
        raise ValueError(
            f"Required percentiles are absent from the evaluation: {missing_percentiles}. "
            f"Available percentiles: {sorted(available_percentiles, reverse=True)}"
        )

    ps_sorted = sorted([float(p) for p in percentiles], reverse=True)
    percentile_to_bin = {p: i for i, p in enumerate(ps_sorted)}

    rows = []
    for percentile in tracked_percentiles:
        percentile = float(percentile)
        if percentile not in percentile_to_bin:
            raise ValueError(f"Could not map percentile {percentile} to an evaluation bin.")

        bin_id = percentile_to_bin[percentile]
        retrieved_active_ids = unique_preserve_order(
            df_eval.loc[(df_eval["bin"] == bin_id) & (df_eval["is_active"]), id_field].astype(str).tolist()
        )
        retrieved_inactive_ids = unique_preserve_order(
            df_eval.loc[(df_eval["bin"] == bin_id) & (~df_eval["is_active"]), id_field].astype(str).tolist()
        )

        if len(retrieved_active_ids) != len(set(retrieved_active_ids)):
            raise ValueError(
                f"Duplicate active IDs detected for target={target_id}, method={method_label}, percentile={percentile}."
            )
        if len(retrieved_inactive_ids) != len(set(retrieved_inactive_ids)):
            raise ValueError(
                f"Duplicate inactive IDs detected for target={target_id}, method={method_label}, percentile={percentile}."
            )

        rows.append({
            "target_id": str(target_id),
            "method_label": str(method_label),
            "percentile": percentile,
            "retrieved_active_ids": retrieved_active_ids,
            "n_retrieved_actives": len(retrieved_active_ids),
            "retrieved_inactive_ids": retrieved_inactive_ids,
            "n_retrieved_inactives": len(retrieved_inactive_ids),
        })

    return rows


def _build_known_active_set_row(
    *,
    target_id: str,
    method_label: str,
    known_active_ids: Sequence[str],
) -> Dict[str, Any]:
    known_ids = unique_preserve_order(known_active_ids)
    if len(known_ids) == 0:
        raise ValueError(f"Target {target_id} has no known_active_ids; this should not occur for evaluated targets.")

    return {
        "target_id": str(target_id),
        "method_label": str(method_label),
        "known_active_ids": known_ids,
        "n_known_actives": len(known_ids),
    }


def _build_known_consistency_check(
    df_known_active_sets: pd.DataFrame,
    *,
    strict_known_consistency: bool = False,
) -> pd.DataFrame:
    rows = []

    if df_known_active_sets.empty:
        return pd.DataFrame(columns=[
            "target_id", "all_methods_same_known_set", "n_methods_compared",
            "known_set_size", "details_if_mismatch",
        ])

    for target_id, g in df_known_active_sets.groupby("target_id", sort=True):
        method_to_known = {
            str(method): set(unique_preserve_order(ids))
            for method, ids in zip(g["method_label"], g["known_active_ids"])
        }

        reference_method = next(iter(method_to_known))
        reference_set = method_to_known[reference_method]
        mismatches = []

        for method, known_set in method_to_known.items():
            if known_set != reference_set:
                missing_vs_ref = sorted(reference_set - known_set)
                extra_vs_ref = sorted(known_set - reference_set)
                mismatches.append(
                    f"{method}: missing_vs_{reference_method}={missing_vs_ref[:10]}, extra_vs_{reference_method}={extra_vs_ref[:10]}"
                )

        all_same = len(mismatches) == 0
        rows.append({
            "target_id": str(target_id),
            "all_methods_same_known_set": all_same,
            "n_methods_compared": int(len(method_to_known)),
            "known_set_size": int(len(reference_set)),
            "details_if_mismatch": None if all_same else " | ".join(mismatches),
        })

    out = pd.DataFrame(rows)
    if strict_known_consistency and not out["all_methods_same_known_set"].all():
        mismatched_targets = out.loc[~out["all_methods_same_known_set"], "target_id"].astype(str).tolist()
        raise ValueError(f"Inconsistent known_active_ids across methods for targets: {mismatched_targets}")

    return out
# ------------------------------------------------------------
# Build seed candidates from neighboring proteins
# ------------------------------------------------------------

def collect_neighbor_candidate_ligands(
    target_name: str,
    neighbor_ranked_ids_by_target: Dict[str, Sequence[str]],
    binding_data_str: pd.DataFrame,
    top_n_neighbors: int,
    *,
    uniprot_col: str = "uniprot_id",
    lig_col: str = "chem_comp_id",
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Collect active ligands from a target's top-N neighboring proteins.
    Preserve neighbor-ranking order, followed by local order of appearance.
    Do not apply anti-trivial cleanup here; only collect and deduplicate.
    """
    ranked_neighbors = [str(x) for x in neighbor_ranked_ids_by_target.get(target_name, [])]
    selected_neighbors = ranked_neighbors[:top_n_neighbors]

    if len(selected_neighbors) == 0:
        return [], {
            "selected_neighbors": [],
            "n_neighbors_with_ligands": 0,
            "raw_candidate_count": 0,
            "unique_candidate_count": 0,
            "per_neighbor_ligand_counts": {},
        }

    cands = []
    per_neighbor_counts = {}

    for uid in selected_neighbors:
        ids = (
            binding_data_str.loc[binding_data_str[uniprot_col] == uid, lig_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        per_neighbor_counts[uid] = len(ids)
        cands.extend(ids)

    cands_unique = unique_preserve_order(cands)

    meta = {
        "selected_neighbors": selected_neighbors,
        "n_neighbors_with_ligands": int(sum(v > 0 for v in per_neighbor_counts.values())),
        "raw_candidate_count": len(cands),
        "unique_candidate_count": len(cands_unique),
        "per_neighbor_ligand_counts": per_neighbor_counts,
    }
    return cands_unique, meta


def filter_neighbor_candidates_nontrivial(
    candidate_ids: Sequence[str],
    target_active_ids: Sequence[str],
    rep_morgan,
    *,
    tanimoto_cutoff: float = 0.85,
    device: str = "cpu",
    q_batch_size: int = 256,
    pool_batch_size: int = 50_000,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Clean neighbor candidates against the true target:
      1) remove exact ID overlap with target actives
      2) remove candidates with maximum Tanimoto > cutoff against target actives

    Also filter IDs that cannot be represented by rep_morgan to avoid KeyError.
    """
    candidates = unique_preserve_order(candidate_ids)
    target_actives = unique_preserve_order(target_active_ids)
    target_set = set(target_actives)

    # 1) Exact overlap by ID
    before_exact = len(candidates)
    candidates_wo_exact = [cid for cid in candidates if cid not in target_set]
    n_removed_exact = before_exact - len(candidates_wo_exact)

    # 1b) Filter candidates and target actives by Morgan representability
    candidates_repr, missing_candidates_repr = filter_ids_present_in_rep(candidates_wo_exact, rep_morgan)
    target_actives_repr, missing_target_actives_repr = filter_ids_present_in_rep(target_actives, rep_morgan)

    if len(candidates_repr) == 0:
        return [], {
            "n_removed_exact_overlap": int(n_removed_exact),
            "n_removed_tanimoto_trivial": 0,
            "n_after_clean": 0,
            "n_candidates_not_in_rep": int(len(missing_candidates_repr)),
            "n_target_actives_not_in_rep": int(len(missing_target_actives_repr)),
            "n_target_actives_repr": int(len(target_actives_repr)),
        }

    # Tanimoto filtering is impossible without representable target actives
    if len(target_actives_repr) == 0:
        return candidates_repr, {
            "n_removed_exact_overlap": int(n_removed_exact),
            "n_removed_tanimoto_trivial": 0,
            "n_after_clean": int(len(candidates_repr)),
            "n_candidates_not_in_rep": int(len(missing_candidates_repr)),
            "n_target_actives_not_in_rep": int(len(missing_target_actives_repr)),
            "n_target_actives_repr": 0,
        }

    # 2) Anti-trivial Tanimoto filtering against representable target actives
    sims_to_target = max_score_for_pool_ids(
        seed_ids=target_actives_repr,
        pool_ids=candidates_repr,
        rep_queries=rep_morgan,
        rep_targets=rep_morgan,
        metric="tanimoto",
        device=device,
        q_batch_size=q_batch_size,
        pool_batch_size=pool_batch_size,
        clamp_max=None,
    )

    keep_mask = sims_to_target <= float(tanimoto_cutoff)
    cleaned = [cid for cid, keep in zip(candidates_repr, keep_mask) if keep]

    meta = {
        "n_removed_exact_overlap": int(n_removed_exact),
        "n_removed_tanimoto_trivial": int((~keep_mask).sum()),
        "n_after_clean": int(len(cleaned)),
        "n_candidates_not_in_rep": int(len(missing_candidates_repr)),
        "n_target_actives_not_in_rep": int(len(missing_target_actives_repr)),
        "n_target_actives_repr": int(len(target_actives_repr)),
    }
    return cleaned, meta


# ------------------------------------------------------------
# Global MaxMin chemical diversity over neighbor candidates
# ------------------------------------------------------------

def maxmin_select_global_by_tanimoto(
    candidate_ids: Sequence[str],
    k: int,
    rep_morgan,  # Morgan representation for Tanimoto
    *,
    init_index: int = 0,     # Deterministic
    device: str = "cpu",
    pool_batch_size: int = 50_000,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Greedy MaxMin selection (farthest-point sampling) using Tanimoto.
    - GLOBAL MaxMin with no cap per neighboring protein.
    - Return up to k seeds.
    - Use max_score_for_pool_ids to reuse the similarity backend.
    """
    cands = unique_preserve_order(candidate_ids)
    n = len(cands)
    k_eff = min(int(k), n)

    if k_eff <= 0 or n == 0:
        return [], {"k_requested": int(k), "k_selected": 0, "n_candidates": n}

    selected = []
    selected_mask = np.zeros(n, dtype=bool)

    # Maximum similarity of each candidate to the selected set
    max_sim_to_selected = np.full(n, -np.inf, dtype=np.float32)

    idx = int(init_index) % n

    for step in range(k_eff):
        if step == 0:
            # First seed: deterministic
            while selected_mask[idx]:
                idx = (idx + 1) % n
        else:
            # Choose the candidate farthest from the current set (minimum maximum similarity)
            score_for_pick = max_sim_to_selected.copy()
            score_for_pick[selected_mask] = np.inf
            idx = int(np.argmin(score_for_pick))

            if not np.isfinite(score_for_pick[idx]):
                break

        picked_id = cands[idx]
        selected.append(picked_id)
        selected_mask[idx] = True

        if len(selected) >= k_eff:
            break

        # Update maximum similarity to the selected set with the new seed
        sims = max_score_for_pool_ids(
            seed_ids=[picked_id],
            pool_ids=cands,
            rep_queries=rep_morgan,
            rep_targets=rep_morgan,
            metric="tanimoto",
            device=device,
            q_batch_size=1,
            pool_batch_size=pool_batch_size,
            clamp_max=None,
        )

        max_sim_to_selected = np.maximum(
            max_sim_to_selected,
            sims.astype(np.float32, copy=False),
        )

    meta = {
        "k_requested": int(k),
        "k_selected": int(len(selected)),
        "n_candidates": int(n),
    }
    return selected, meta


def build_neighbor_seed_ids_matched_k(
    *,
    target_name: str,
    target_active_ids: Sequence[str],            # All true target actives
    k_requested: int,                            # = len(res.known_ids) from the original split
    neighbor_ranked_ids_by_target: Dict[str, Sequence[str]],
    binding_data_str: pd.DataFrame,              # binding_data with string columns
    top_n_neighbors: int,
    rep_morgan,
    tanimoto_cleanup_cutoff: float = 0.85,
    device_seed_selection: str = "cpu",
    q_batch_size_filter: int = 256,
    pool_batch_size_filter: int = 50_000,
    pool_batch_size_maxmin: int = 50_000,
    maxmin_init_index: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    """
    Complete pipeline for building neighbor seeds:
      top-N neighbors -> ligands -> anti-trivial cleanup -> global MaxMin -> <= K
    """
    cands_raw, meta_collect = collect_neighbor_candidate_ligands(
        target_name=target_name,
        neighbor_ranked_ids_by_target=neighbor_ranked_ids_by_target,
        binding_data_str=binding_data_str,
        top_n_neighbors=top_n_neighbors,
        uniprot_col="uniprot_id",
        lig_col="chem_comp_id",
    )

    cands_clean, meta_clean = filter_neighbor_candidates_nontrivial(
        candidate_ids=cands_raw,
        target_active_ids=target_active_ids,
        rep_morgan=rep_morgan,
        tanimoto_cutoff=tanimoto_cleanup_cutoff,
        device=device_seed_selection,
        q_batch_size=q_batch_size_filter,
        pool_batch_size=pool_batch_size_filter,
    )

    seed_ids, meta_maxmin = maxmin_select_global_by_tanimoto(
        candidate_ids=cands_clean,
        k=k_requested,
        rep_morgan=rep_morgan,
        init_index=maxmin_init_index,
        device=device_seed_selection,
        pool_batch_size=pool_batch_size_maxmin,
    )

    meta = {
        "target": target_name,
        "top_n_neighbors_requested": int(top_n_neighbors),
        "top_n_neighbors_available": int(len(neighbor_ranked_ids_by_target.get(target_name, []))),
        "top_n_neighbors_considered": int(len(meta_collect["selected_neighbors"])),
        "n_neighbors_with_ligands": int(meta_collect["n_neighbors_with_ligands"]),
        "neighbor_ids_considered": meta_collect["selected_neighbors"],  # Useful debug list
        "n_neighbor_candidates_raw": int(meta_collect["raw_candidate_count"]),
        "n_neighbor_candidates_unique": int(meta_collect["unique_candidate_count"]),
        "n_removed_exact_overlap": int(meta_clean["n_removed_exact_overlap"]),
        "n_removed_tanimoto_trivial": int(meta_clean["n_removed_tanimoto_trivial"]),
        "n_neighbor_candidates_clean": int(meta_clean["n_after_clean"]),
        "k_requested": int(k_requested),
        "k_effective": int(len(seed_ids)),
    }
    return seed_ids, meta

def run_ef_eval_target_vs_neighbors(
    *,
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
    store,                                   # LigandStore
    neighbor_ranked_ids_by_target: Dict[str, Sequence[str]],

    # Evaluation (Morgan/Tanimoto or ChemBERTa/cosine)
    rep_eval,
    metric_eval: str,                        # "tanimoto" or "cosine"
    method_label: str,                       # For example, "morgan_tanimoto" / "chemberta_cosine"
    device_eval: str = "cpu",
    assume_normalized_eval=None,
    clamp_max_eval=None,
    q_batch_size_eval: int = 256,
    pool_batch_size_eval: int = 50_000,

    # Neighbor seed selection (always Morgan/Tanimoto)
    rep_morgan_for_seed_selection=None,      # REQUIRED
    top_n_neighbors_for_seeds: int = 10,
    tanimoto_cleanup_cutoff: float = 0.85,
    device_seed_selection: str = "cpu",
    q_batch_size_seed_filter: int = 256,
    pool_batch_size_seed_filter: int = 50_000,
    pool_batch_size_maxmin: int = 50_000,
    maxmin_init_index: int = 0,

    # Split/evaluation
    percentiles=(99.5, 99, 98.5, 98, 95, 90, 80, 50),
    ef_mode: str = "both",
    min_known_for_eval: int = 50,
    split_kwargs: Optional[dict] = None,
    verbose: bool = True,

    # Policy when there are fewer than K neighbor seeds
    neighbor_short_policy: str = "allow_short",  # "allow_short" | "skip"
    strict_known_consistency: bool = False,
    use_target_seeds: bool = True,
    use_neighbor_seeds: bool = True,
    split_function=None,
    target_score_sink=None,
):
    """
    Return:
      - df_long_target
      - df_long_neighbors
      - df_seed_meta (debug / trazabilidad)
      - df_retrieved_active_sets
      - df_known_active_sets
      - df_known_consistency_check

    EF output uses long format by target/method/percentile, with separate
    columns by mode:
      - bands:      n_band / actives_band / actives_rate_band / EF_band
      - cumulative: n_cumulative / actives_cumulative / actives_rate_cumulative / EF_cumulative

    ef_mode:
      - "band": bands only
      - "cumulative": cumulative only
      - "both": both (default)

    Practical flags:
      - use_target_seeds: run the condition with target seeds
      - use_neighbor_seeds: run the condition with neighbor seeds
    """

    if not use_target_seeds and not use_neighbor_seeds:
        raise ValueError("At least one of use_target_seeds or use_neighbor_seeds must be True.")

    if rep_morgan_for_seed_selection is None and use_neighbor_seeds:
        raise ValueError("rep_morgan_for_seed_selection is required for MaxMin plus Tanimoto cleanup.")

    if split_kwargs is None:
        split_kwargs = dict(
            max_total_actives=1000,
            known_frac=0.20,
            known_min=10,
            known_max=200,
            test_to_known_ratio=5,
            test_min=50,
            test_max=1000,
            nontrivial_tanimoto_cutoff=0.85,
            fallback_if_short="allow_short",
            random_state=42,
        )

    ef_mode = normalize_ef_mode(ef_mode)
    if split_function is None:
        split_function = split_hit_expansion_fixed_sizes

    # Normalize types once
    targets_dude_s = targets_dude.copy()
    binding_data_s = binding_data.copy()
    smiles_s = smiles.copy()

    targets_dude_s["target"] = targets_dude_s["target"].astype(str)
    if "uniprot_id" in targets_dude_s.columns:
        targets_dude_s["uniprot_id"] = targets_dude_s["uniprot_id"].astype(str)

    binding_data_s["uniprot_id"] = binding_data_s["uniprot_id"].astype(str)
    binding_data_s["chem_comp_id"] = binding_data_s["chem_comp_id"].astype(str)
    binding_data_s["pfam_id"] = binding_data_s["pfam_id"].astype(str)

    smiles_s["chem_comp_id"] = smiles_s["chem_comp_id"].astype(str)

    all_rows_target = []
    all_rows_neighbors = []
    seed_meta_rows = []
    retrieved_active_set_rows = []
    known_active_set_rows = []

    target_list = targets_dude_s["target"].dropna().astype(str).unique().tolist()

    for i, target in enumerate(target_list, 1):
        if verbose:
            print(f"\n[{i}/{len(target_list)}] Target: {target}")

        # ---------------------------------------------------------
        # Target actives and inactives (same logic as the original loop)
        # ---------------------------------------------------------
        uniprots = (
            targets_dude_s.loc[targets_dude_s["target"] == target, "uniprot_id"]
            .dropna().astype(str).unique().tolist()
        )

        activos = (
            binding_data_s.loc[binding_data_s["uniprot_id"].isin(uniprots), "chem_comp_id"]
            .dropna().astype(str).unique().tolist()
        )

        fams_target = (
            binding_data_s.loc[binding_data_s["uniprot_id"].isin(uniprots), "pfam_id"]
            .dropna().astype(str).unique().tolist()
        )

        lista_negra = (
            binding_data_s.loc[binding_data_s["pfam_id"].isin(fams_target), "chem_comp_id"]
            .dropna().astype(str).tolist()
        )

        inactivos = (
            smiles_s.loc[~smiles_s["chem_comp_id"].isin(lista_negra), "chem_comp_id"]
            .dropna().astype(str).tolist()
        )

        # Additional safeguard: same contents with deterministic order
        inactivos = stable_set_difference_preserve_order(inactivos, activos)

        if verbose:
            print("  actives:", len(activos))
            print("  inactives:", len(inactivos))

        # ---------------------------------------------------------
        # ORIGINAL target split, which defines test/background and K
        # ---------------------------------------------------------
        res = split_function(
            activos=activos,
            smiles_df=smiles_s,
            **split_kwargs,
        )

        if len(res.known_ids) < min_known_for_eval:
            if verbose:
                print(f"  Skip entire target (n_known={len(res.known_ids)} < {min_known_for_eval})")
            continue

        dataset = build_eval_pool_ids(res.test_active_ids, inactivos)

        seed_ids_target = [str(x) for x in res.known_ids]
        pool_ids_raw = [str(x) for x in dataset["pool_ids"]]
        pos_ids_raw = [str(x) for x in dataset["pos_ids"]]

        k_requested = len(seed_ids_target)

        # ---------------------------------------------------------
        # Filter IDs by representability in rep_eval: pool, positives, and target seeds
        # ---------------------------------------------------------
        pool_ids_eval, pos_ids_eval, seed_ids_target_eval, eval_rep_meta_target = filter_eval_ids_by_rep(
            pool_ids=pool_ids_raw,
            pos_ids=pos_ids_raw,
            seed_ids=seed_ids_target,
            rep=rep_eval,
        )

        if verbose:
            print(
                "  rep_eval coverage (target seeds): "
                f"pool {eval_rep_meta_target['n_pool_repr']}/{len(pool_ids_raw)} "
                f"(missing={eval_rep_meta_target['n_pool_missing_in_rep']}), "
                f"seed {eval_rep_meta_target['n_seed_repr']}/{len(seed_ids_target)} "
                f"(missing={eval_rep_meta_target['n_seed_missing_in_rep']}), "
                f"pos_in_pool_repr={eval_rep_meta_target['n_pos_in_pool_repr']}/{len(pos_ids_raw)}"
            )

        # Basic safeguards: skip the target if an essential set is empty
        if len(pool_ids_eval) == 0:
            if verbose:
                print("  Skip entire target (empty pool after rep_eval filtering)")
            continue
        if len(pos_ids_eval) == 0:
            if verbose:
                print("  Skip entire target (no positives in the pool representable by rep_eval)")
            continue
        if use_target_seeds and len(seed_ids_target_eval) == 0:
            if verbose:
                print("  Skip entire target (no target seeds representable by rep_eval)")
            continue

        target_method_name = f"{method_label}__target_seeds"
        neighbor_method_name = f"{method_label}__neighbor_seeds_top{top_n_neighbors_for_seeds}"

        if use_target_seeds:
            known_active_set_rows.append(_build_known_active_set_row(
                target_id=target,
                method_label=target_method_name,
                known_active_ids=seed_ids_target,
            ))

            # ---------------------------------------------------------
            # A) ORIGINAL evaluation with target seeds using filtered IDs
            # ---------------------------------------------------------
            scores_pool_target = max_score_for_pool_ids(
                seed_ids_target_eval,
                pool_ids_eval,
                rep_queries=rep_eval,
                rep_targets=rep_eval,
                metric=metric_eval,
                device=device_eval,
                q_batch_size=q_batch_size_eval,
                pool_batch_size=pool_batch_size_eval,
                assume_normalized=assume_normalized_eval,
                clamp_max=clamp_max_eval,
            )

            df_eval_t, cuts_t = build_eval_table_from_pool_ids(
                pool_ids_eval,
                scores_pool_target,
                store=store,
                percentiles=percentiles,
                active_ids=pos_ids_eval,
                extra_fields=("smiles",),
            )
            _validate_eval_table_integrity(
                df_eval_t,
                expected_pool_ids=pool_ids_eval,
                expected_pos_ids=pos_ids_eval,
            )

            # Optional observation point for fixed-budget analyses. The
            # original percentile/EF calculation and return contract below
            # remain unchanged when no sink is supplied.
            if target_score_sink is not None:
                target_score_sink(
                    target=target,
                    method=method_label,
                    pool_ids=pool_ids_eval,
                    positive_ids=pos_ids_eval,
                    seed_ids=seed_ids_target_eval,
                    scores=scores_pool_target,
                    score_cut_995=cuts_t[99.5],
                )

            df_ef_t = enrichment_factor_by_percentile_mode(
                df_eval_t,
                percentiles=percentiles,
                ef_mode=ef_mode,
            )

            df_long_t = ef_results_to_long_fixed(
                df_ef_t, cuts_t, target,
                method=target_method_name,
                percentiles=percentiles,
            )
            df_long_t["seed_source"] = "target"
            df_long_t["n_seed_requested"] = k_requested
            df_long_t["n_seed_effective"] = len(seed_ids_target_eval)  # Representable in rep_eval
            df_long_t["top_n_neighbors_for_seeds"] = np.nan
            df_long_t["n_pool_eval"] = len(pool_ids_eval)
            df_long_t["n_pos_eval"] = len(pos_ids_eval)
            all_rows_target.append(df_long_t)
            retrieved_active_set_rows.extend(_build_retrieved_active_set_rows(
                df_eval=df_eval_t,
                target_id=target,
                method_label=target_method_name,
                percentiles=percentiles,
            ))

        if use_neighbor_seeds:
            # ---------------------------------------------------------
            # B) Build seeds from neighbors (cleanup plus MaxMin with Morgan)
            # ---------------------------------------------------------
            neighbor_seed_ids, nb_meta = build_neighbor_seed_ids_matched_k(
                target_name=target,
                target_active_ids=activos,  # All true target actives
                k_requested=k_requested,    # Same K as the original split
                neighbor_ranked_ids_by_target=neighbor_ranked_ids_by_target,
                binding_data_str=binding_data_s,
                top_n_neighbors=top_n_neighbors_for_seeds,
                rep_morgan=rep_morgan_for_seed_selection,
                tanimoto_cleanup_cutoff=tanimoto_cleanup_cutoff,
                device_seed_selection=device_seed_selection,
                q_batch_size_filter=q_batch_size_seed_filter,
                pool_batch_size_filter=pool_batch_size_seed_filter,
                pool_batch_size_maxmin=pool_batch_size_maxmin,
                maxmin_init_index=maxmin_init_index,
            )

            # Filter neighbor seeds by representability in rep_eval
            neighbor_seed_ids_eval, neighbor_seed_ids_missing_eval = filter_ids_present_in_rep(neighbor_seed_ids, rep_eval)

            nb_meta["target"] = target
            nb_meta["method_eval"] = method_label
            nb_meta["n_target_actives_total"] = len(activos)
            nb_meta["n_target_known_original"] = len(seed_ids_target)
            nb_meta["n_target_known_eval_repr"] = len(seed_ids_target_eval)
            nb_meta["n_target_test_original"] = len(res.test_active_ids)
            nb_meta["n_pool_raw"] = len(pool_ids_raw)
            nb_meta["n_pool_eval_repr"] = len(pool_ids_eval)
            nb_meta["n_pos_eval_repr"] = len(pos_ids_eval)
            nb_meta["n_pool_missing_in_rep_eval"] = eval_rep_meta_target["n_pool_missing_in_rep"]
            nb_meta["n_target_seed_missing_in_rep_eval"] = eval_rep_meta_target["n_seed_missing_in_rep"]
            nb_meta["n_neighbor_seed_missing_in_rep_eval"] = len(neighbor_seed_ids_missing_eval)
            nb_meta["n_neighbor_seed_eval_repr"] = len(neighbor_seed_ids_eval)

            seed_meta_rows.append(nb_meta)

            if verbose:
                print(
                    f"  Neighbor seeds (pre-eval filter): requested K={k_requested}, effective K={len(neighbor_seed_ids)} | "
                    f"raw={nb_meta['n_neighbor_candidates_raw']} unique={nb_meta['n_neighbor_candidates_unique']} "
                    f"clean={nb_meta['n_neighbor_candidates_clean']}"
                )
                print(
                    f"  Neighbor seeds in rep_eval: {len(neighbor_seed_ids_eval)}/{len(neighbor_seed_ids)} "
                    f"(missing={len(neighbor_seed_ids_missing_eval)})"
                )

            if len(neighbor_seed_ids_eval) == 0:
                if verbose:
                    print("  Skip neighbor condition (0 seeds after rep_eval filtering)")
                continue

            if neighbor_short_policy == "skip" and len(neighbor_seed_ids_eval) < k_requested:
                if verbose:
                    print(f"  Skip neighbor condition (short K after rep_eval: {len(neighbor_seed_ids_eval)} < {k_requested})")
                continue

            # ---------------------------------------------------------
            # B2) Evaluate with neighbor seeds using the SAME filtered pool/positives
            # ---------------------------------------------------------
            scores_pool_neighbors = max_score_for_pool_ids(
                neighbor_seed_ids_eval,
                pool_ids_eval,
                rep_queries=rep_eval,
                rep_targets=rep_eval,
                metric=metric_eval,
                device=device_eval,
                q_batch_size=q_batch_size_eval,
                pool_batch_size=pool_batch_size_eval,
                assume_normalized=assume_normalized_eval,
                clamp_max=clamp_max_eval,
            )

            df_eval_n, cuts_n = build_eval_table_from_pool_ids(
                pool_ids_eval,
                scores_pool_neighbors,
                store=store,
                percentiles=percentiles,
                active_ids=pos_ids_eval,
                extra_fields=("smiles",),
            )
            _validate_eval_table_integrity(
                df_eval_n,
                expected_pool_ids=pool_ids_eval,
                expected_pos_ids=pos_ids_eval,
            )

            df_ef_n = enrichment_factor_by_percentile_mode(
                df_eval_n,
                percentiles=percentiles,
                ef_mode=ef_mode,
            )

            df_long_n = ef_results_to_long_fixed(
                df_ef_n, cuts_n, target,
                method=neighbor_method_name,
                percentiles=percentiles,
            )
            df_long_n["seed_source"] = "neighbors"
            df_long_n["n_seed_requested"] = k_requested
            df_long_n["n_seed_effective"] = len(neighbor_seed_ids_eval)   # Representable in rep_eval
            df_long_n["top_n_neighbors_for_seeds"] = top_n_neighbors_for_seeds
            df_long_n["n_pool_eval"] = len(pool_ids_eval)
            df_long_n["n_pos_eval"] = len(pos_ids_eval)
            all_rows_neighbors.append(df_long_n)
            known_active_set_rows.append(_build_known_active_set_row(
                target_id=target,
                method_label=neighbor_method_name,
                known_active_ids=seed_ids_target,
            ))
            retrieved_active_set_rows.extend(_build_retrieved_active_set_rows(
                df_eval=df_eval_n,
                target_id=target,
                method_label=neighbor_method_name,
                percentiles=percentiles,
            ))

    df_long_target = pd.concat(all_rows_target, ignore_index=True) if all_rows_target else pd.DataFrame()
    df_long_neighbors = pd.concat(all_rows_neighbors, ignore_index=True) if all_rows_neighbors else pd.DataFrame()
    df_seed_meta = pd.DataFrame(seed_meta_rows)
    df_retrieved_active_sets = pd.DataFrame(retrieved_active_set_rows)
    df_known_active_sets = pd.DataFrame(known_active_set_rows)
    df_known_consistency_check = _build_known_consistency_check(
        df_known_active_sets,
        strict_known_consistency=strict_known_consistency,
    )

    return (
        df_long_target,
        df_long_neighbors,
        df_seed_meta,
        df_retrieved_active_sets,
        df_known_active_sets,
        df_known_consistency_check,
    )


def run_ef_eval_target_vs_neighbors_clustering(
    *,
    butina_cutoff: float | None = None,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
    split_kwargs: Optional[dict] = None,
    **kwargs,
):
    """
    Variant of run_ef_eval_target_vs_neighbors using a split over Butina representatives.

    Preserve the same output contract, but first reduce actives to one
    representative per cluster and then define known/unknown. The rest of the
    EF evaluation is unchanged.
    """
    split_kwargs_clustering = dict(split_kwargs or {})
    if butina_cutoff is not None:
        split_kwargs_clustering["butina_cutoff"] = float(butina_cutoff)
    split_kwargs_clustering.update({
        "butina_radius": int(butina_radius),
        "butina_nbits": int(butina_nbits),
    })
    return run_ef_eval_target_vs_neighbors(
        split_kwargs=split_kwargs_clustering,
        split_function=split_hit_expansion_fixed_sizes_clustering,
        **kwargs,
    )


def run_ef_eval_neighbors_sweep_single_method(
    *,
    neighbor_counts: Sequence[int],
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
    store,
    neighbor_ranked_ids_by_target: Dict[str, Sequence[str]],
    rep_eval,
    metric_eval: str,
    method_label: str,
    device_eval: str = "cpu",
    assume_normalized_eval=None,
    clamp_max_eval=None,
    q_batch_size_eval: int = 256,
    pool_batch_size_eval: int = 50_000,
    rep_morgan_for_seed_selection=None,
    tanimoto_cleanup_cutoff: float = 0.85,
    device_seed_selection: str = "cpu",
    q_batch_size_seed_filter: int = 256,
    pool_batch_size_seed_filter: int = 50_000,
    pool_batch_size_maxmin: int = 50_000,
    maxmin_init_index: int = 0,
    percentiles=(99.5, 99, 98.5, 98, 95, 90, 80, 50),
    ef_mode: str = "both",
    min_known_for_eval: int = 50,
    split_kwargs: Optional[dict] = None,
    verbose: bool = True,
    neighbor_short_policy: str = "allow_short",
    strict_known_consistency: bool = False,
    split_function=None,
) -> Dict[str, pd.DataFrame]:
    """
    Sweep top_n_neighbors_for_seeds values for one evaluation method and
    consolidate the results.

    Keep all other settings fixed. Comparisons between runs are valid when the
    split is deterministic, for example with a fixed split_kwargs["random_state"].

    Return a dictionary with:
      - df_long_neighbors_all
      - df_seed_meta_all
      - df_retrieved_active_sets_all
      - df_known_active_sets_all
      - df_known_consistency_checks_all
      - df_target_inclusion_summary
    """
    neighbor_counts = [int(x) for x in neighbor_counts]
    if len(neighbor_counts) == 0:
        raise ValueError("neighbor_counts cannot be empty.")
    if any(x <= 0 for x in neighbor_counts):
        raise ValueError("All neighbor_counts values must be positive integers.")

    if rep_morgan_for_seed_selection is None:
        raise ValueError("rep_morgan_for_seed_selection is required to select neighbor seeds.")

    all_long_neighbors = []
    all_seed_meta = []
    all_retrieved_sets = []
    all_known_sets = []
    all_consistency_checks = []

    for top_n_neighbors in neighbor_counts:
        if verbose:
            print(f"\n================ Sweep neighbors: top_n_neighbors_for_seeds={top_n_neighbors} ================")

        (
            _df_long_target,
            df_long_neighbors,
            df_seed_meta,
            df_retrieved_active_sets,
            df_known_active_sets,
            df_known_consistency_check,
        ) = run_ef_eval_target_vs_neighbors(
            targets_dude=targets_dude,
            binding_data=binding_data,
            smiles=smiles,
            store=store,
            neighbor_ranked_ids_by_target=neighbor_ranked_ids_by_target,
            rep_eval=rep_eval,
            metric_eval=metric_eval,
            method_label=method_label,
            device_eval=device_eval,
            assume_normalized_eval=assume_normalized_eval,
            clamp_max_eval=clamp_max_eval,
            q_batch_size_eval=q_batch_size_eval,
            pool_batch_size_eval=pool_batch_size_eval,
            rep_morgan_for_seed_selection=rep_morgan_for_seed_selection,
            top_n_neighbors_for_seeds=top_n_neighbors,
            tanimoto_cleanup_cutoff=tanimoto_cleanup_cutoff,
            device_seed_selection=device_seed_selection,
            q_batch_size_seed_filter=q_batch_size_seed_filter,
            pool_batch_size_seed_filter=pool_batch_size_seed_filter,
            pool_batch_size_maxmin=pool_batch_size_maxmin,
            maxmin_init_index=maxmin_init_index,
            percentiles=percentiles,
            ef_mode=ef_mode,
            min_known_for_eval=min_known_for_eval,
            split_kwargs=split_kwargs,
            verbose=verbose,
            neighbor_short_policy=neighbor_short_policy,
            strict_known_consistency=strict_known_consistency,
            use_target_seeds=False,
            use_neighbor_seeds=True,
            split_function=split_function,
        )

        if not df_long_neighbors.empty:
            tmp = df_long_neighbors.copy()
            tmp["neighbor_count_sweep"] = int(top_n_neighbors)
            all_long_neighbors.append(tmp)

        if not df_seed_meta.empty:
            tmp = df_seed_meta.copy()
            tmp["neighbor_count_sweep"] = int(top_n_neighbors)
            all_seed_meta.append(tmp)

        if not df_retrieved_active_sets.empty:
            tmp = df_retrieved_active_sets.copy()
            tmp["neighbor_count_sweep"] = int(top_n_neighbors)
            all_retrieved_sets.append(tmp)

        if not df_known_active_sets.empty:
            tmp = df_known_active_sets.copy()
            tmp["neighbor_count_sweep"] = int(top_n_neighbors)
            all_known_sets.append(tmp)

        if not df_known_consistency_check.empty:
            tmp = df_known_consistency_check.copy()
            tmp["neighbor_count_sweep"] = int(top_n_neighbors)
            all_consistency_checks.append(tmp)

    df_long_neighbors_all = (
        pd.concat(all_long_neighbors, ignore_index=True)
        if all_long_neighbors else pd.DataFrame()
    )
    df_seed_meta_all = (
        pd.concat(all_seed_meta, ignore_index=True)
        if all_seed_meta else pd.DataFrame()
    )
    df_retrieved_active_sets_all = (
        pd.concat(all_retrieved_sets, ignore_index=True)
        if all_retrieved_sets else pd.DataFrame()
    )
    df_known_active_sets_all = (
        pd.concat(all_known_sets, ignore_index=True)
        if all_known_sets else pd.DataFrame()
    )
    df_known_consistency_checks_all = (
        pd.concat(all_consistency_checks, ignore_index=True)
        if all_consistency_checks else pd.DataFrame()
    )

    if df_seed_meta_all.empty:
        df_target_inclusion_summary = pd.DataFrame()
    else:
        expected_counts = len(set(neighbor_counts))
        df_target_inclusion_summary = (
            df_seed_meta_all
            .assign(
                target=lambda d: d["target"].astype(str),
                top_n_neighbors_for_seeds=lambda d: d["top_n_neighbors_requested"].astype(int),
                applies_to_target=lambda d: (
                    (d["top_n_neighbors_available"] >= d["top_n_neighbors_requested"]) &
                    (d["k_effective"] == d["k_requested"])
                ),
            )
            .groupby("target", as_index=False)
            .agg(
                n_neighbor_settings_evaluated=("neighbor_count_sweep", "nunique"),
                n_neighbor_settings_applicable=("applies_to_target", "sum"),
                min_neighbors_available=("top_n_neighbors_available", "min"),
                min_k_effective=("k_effective", "min"),
                max_k_requested=("k_requested", "max"),
            )
        )
        df_target_inclusion_summary["applicable_to_all_neighbor_counts"] = (
            (df_target_inclusion_summary["n_neighbor_settings_evaluated"] == expected_counts) &
            (df_target_inclusion_summary["n_neighbor_settings_applicable"] == expected_counts)
        )

    return {
        "df_long_neighbors_all": df_long_neighbors_all,
        "df_seed_meta_all": df_seed_meta_all,
        "df_retrieved_active_sets_all": df_retrieved_active_sets_all,
        "df_known_active_sets_all": df_known_active_sets_all,
        "df_known_consistency_checks_all": df_known_consistency_checks_all,
        "df_target_inclusion_summary": df_target_inclusion_summary,
    }


def run_ef_eval_neighbors_sweep_single_method_clustering(
    *,
    butina_cutoff: float | None = None,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
    split_kwargs: Optional[dict] = None,
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Clustering variant of run_ef_eval_neighbors_sweep_single_method.

    Use the same evaluation set as run_ef_eval_target_vs_neighbors_clustering:
    actives reduced to one Butina representative per cluster, known/unknown
    split over those representatives, and neighbor seeds selected with K equal
    to the size of the target known set.
    """
    split_kwargs_clustering = dict(split_kwargs or {})
    if butina_cutoff is not None:
        split_kwargs_clustering["butina_cutoff"] = float(butina_cutoff)
    split_kwargs_clustering.update({
        "butina_radius": int(butina_radius),
        "butina_nbits": int(butina_nbits),
    })
    return run_ef_eval_neighbors_sweep_single_method(
        split_kwargs=split_kwargs_clustering,
        split_function=split_hit_expansion_fixed_sizes_clustering,
        **kwargs,
    )


def _import_bsi_group_model_utils(bsi_repo_dir: str | Path):
    bsi_repo_dir = Path(bsi_repo_dir).resolve()
    if str(bsi_repo_dir) not in sys.path:
        sys.path.insert(0, str(bsi_repo_dir))
    from src.bsi_group_models import ensure_dense_fp_matrix, load_group_model, score_query_against_dense_chunk
    return ensure_dense_fp_matrix, load_group_model, score_query_against_dense_chunk


def _resolve_bsi_repo_dir_from_models_dir(bsi_models_dir: str | Path) -> Path:
    bsi_models_dir = Path(bsi_models_dir).resolve()
    if bsi_models_dir.parent.name == "out":
        return bsi_models_dir.parent.parent
    return Path("/home/gustavo/disco_2/BSI_2/repo")


def _discover_target_bsi_model_paths(bsi_models_dir: str | Path) -> Dict[str, Path]:
    bsi_models_dir = Path(bsi_models_dir).resolve()
    if not bsi_models_dir.exists():
        raise FileNotFoundError(f"bsi_models_dir does not exist: {bsi_models_dir}")

    model_paths: Dict[str, Path] = {}
    summary_path = bsi_models_dir / "summary.csv"
    bsi_repo_dir = _resolve_bsi_repo_dir_from_models_dir(bsi_models_dir)

    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if "target_name" in summary.columns and "model_path" in summary.columns:
            if "status" in summary.columns:
                summary = summary.loc[summary["status"].astype(str) == "trained"].copy()
            for _, row in summary.iterrows():
                target = str(row["target_name"])
                raw_path = Path(str(row["model_path"]))
                candidates = [
                    raw_path if raw_path.is_absolute() else bsi_repo_dir / raw_path,
                    bsi_models_dir / target / "model.pth",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        model_paths[target] = candidate.resolve()
                        break

    for model_path in bsi_models_dir.glob("*/model.pth"):
        model_paths.setdefault(model_path.parent.name, model_path.resolve())

    return model_paths


def _max_bsi_score_for_pool_ids(
    *,
    seed_ids: Sequence[str],
    pool_ids: Sequence[str],
    rep_morgan,
    model,
    fp_bits: int,
    ensure_dense_fp_matrix,
    score_query_against_dense_chunk,
    pool_batch_size: int = 10_000,
    pair_batch_size: int = 65_536,
) -> np.ndarray:
    seed_ids = [str(x) for x in seed_ids]
    pool_ids = [str(x) for x in pool_ids]
    if len(seed_ids) == 0 or len(pool_ids) == 0:
        return np.zeros((len(pool_ids),), dtype=np.float32)

    seed_fps = ensure_dense_fp_matrix(rep_morgan.get_raw_by_ids(seed_ids), fp_bits)
    out = np.full((len(pool_ids),), -np.inf, dtype=np.float32)

    for p0 in range(0, len(pool_ids), int(pool_batch_size)):
        p1 = min(p0 + int(pool_batch_size), len(pool_ids))
        pool_chunk_ids = pool_ids[p0:p1]
        pool_fps = ensure_dense_fp_matrix(rep_morgan.get_raw_by_ids(pool_chunk_ids), fp_bits)

        chunk_best = np.full((len(pool_chunk_ids),), -np.inf, dtype=np.float32)
        for seed_fp in seed_fps:
            scores = score_query_against_dense_chunk(
                model,
                seed_fp,
                pool_fps,
                batch_size=pair_batch_size,
            ).astype(np.float32, copy=False)
            chunk_best = np.maximum(chunk_best, scores)

        out[p0:p1] = chunk_best

    out[~np.isfinite(out)] = 0.0
    return out


def run_ef_eval_bsi_target_exclusion_clustering(
    *,
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
    store,
    rep_morgan,
    bsi_models_dir: str | Path = "/home/gustavo/disco_2/BSI_2/repo/out/pf00069_target_exclusion_models_1024",
    bsi_repo_dir: str | Path | None = None,
    method_label: str = "bsi_pf00069_target_exclusion_1024",
    device: str = "auto",
    percentiles=(99.5, 99, 98.5, 98, 95, 90, 80, 50),
    ef_mode: str = "both",
    min_known_for_eval: int = 50,
    split_kwargs: Optional[dict] = None,
    butina_cutoff: float | None = None,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
    pool_batch_size_bsi: int = 10_000,
    pair_batch_size_bsi: int = 65_536,
    verbose: bool = True,
    strict_known_consistency: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Evaluate target-exclusion BSI models using the same EF format as other methods.

    For each target with an available model:
      - construct the same Butina-representative split as the clustering evaluation
      - use the target's known_active_ids as seeds
      - rank the pool by max_seed BSI(seed, candidate)
      - return tables compatible with run_ef_eval_target_vs_neighbors_clustering
    """
    if split_kwargs is None:
        split_kwargs = dict(
            max_total_actives=1000,
            known_frac=0.20,
            known_min=10,
            known_max=200,
            test_to_known_ratio=5,
            test_min=50,
            test_max=1000,
            nontrivial_tanimoto_cutoff=0.85,
            fallback_if_short="allow_short",
            random_state=42,
        )

    bsi_models_dir = Path(bsi_models_dir).resolve()
    if bsi_repo_dir is None:
        bsi_repo_dir = _resolve_bsi_repo_dir_from_models_dir(bsi_models_dir)
    bsi_repo_dir = Path(bsi_repo_dir).resolve()

    ensure_dense_fp_matrix, load_group_model, score_query_against_dense_chunk = _import_bsi_group_model_utils(bsi_repo_dir)
    model_paths_by_target = _discover_target_bsi_model_paths(bsi_models_dir)
    if not model_paths_by_target:
        raise FileNotFoundError(f"No BSI models were found in {bsi_models_dir}")

    split_kwargs_clustering = dict(split_kwargs or {})
    if butina_cutoff is not None:
        split_kwargs_clustering["butina_cutoff"] = float(butina_cutoff)
    split_kwargs_clustering.update({
        "butina_radius": int(butina_radius),
        "butina_nbits": int(butina_nbits),
    })

    ef_mode = normalize_ef_mode(ef_mode)

    targets_dude_s = targets_dude.copy()
    binding_data_s = binding_data.copy()
    smiles_s = smiles.copy()

    targets_dude_s["target"] = targets_dude_s["target"].astype(str)
    if "uniprot_id" in targets_dude_s.columns:
        targets_dude_s["uniprot_id"] = targets_dude_s["uniprot_id"].astype(str)
    binding_data_s["uniprot_id"] = binding_data_s["uniprot_id"].astype(str)
    binding_data_s["chem_comp_id"] = binding_data_s["chem_comp_id"].astype(str)
    binding_data_s["pfam_id"] = binding_data_s["pfam_id"].astype(str)
    smiles_s["chem_comp_id"] = smiles_s["chem_comp_id"].astype(str)

    all_rows = []
    meta_rows = []
    retrieved_active_set_rows = []
    known_active_set_rows = []

    target_list_all = targets_dude_s["target"].dropna().astype(str).unique().tolist()
    target_list = [t for t in target_list_all if t in model_paths_by_target]

    if verbose:
        skipped = sorted(set(target_list_all) - set(target_list))
        print(f"BSI targets with a model and present in targets_dude: {len(target_list)}")
        if skipped:
            print(f"Targets without a BSI model (skipped): {len(skipped)}")

    loaded_models: Dict[str, Tuple[Any, Dict[str, Any], int]] = {}

    for i, target in enumerate(target_list, 1):
        if verbose:
            print(f"\n[{i}/{len(target_list)}] Target BSI: {target}")

        model_path = model_paths_by_target[target]
        model, params = load_group_model(model_path, device=device)
        fp_bits = int(params["fp_bits"])
        if fp_bits != int(rep_morgan.dim):
            raise ValueError(
                f"BSI model {target} expects fp_bits={fp_bits}, but rep_morgan.dim={rep_morgan.dim}."
            )
        loaded_models[target] = (model, params, fp_bits)

        uniprots = (
            targets_dude_s.loc[targets_dude_s["target"] == target, "uniprot_id"]
            .dropna().astype(str).unique().tolist()
        )

        activos = (
            binding_data_s.loc[binding_data_s["uniprot_id"].isin(uniprots), "chem_comp_id"]
            .dropna().astype(str).unique().tolist()
        )

        fams_target = (
            binding_data_s.loc[binding_data_s["uniprot_id"].isin(uniprots), "pfam_id"]
            .dropna().astype(str).unique().tolist()
        )

        lista_negra = (
            binding_data_s.loc[binding_data_s["pfam_id"].isin(fams_target), "chem_comp_id"]
            .dropna().astype(str).tolist()
        )

        inactivos = (
            smiles_s.loc[~smiles_s["chem_comp_id"].isin(lista_negra), "chem_comp_id"]
            .dropna().astype(str).tolist()
        )
        inactivos = stable_set_difference_preserve_order(inactivos, activos)

        if verbose:
            print("  actives:", len(activos))
            print("  inactives:", len(inactivos))
            print("  model:", model_path)

        res = split_hit_expansion_fixed_sizes_clustering(
            activos=activos,
            smiles_df=smiles_s,
            **split_kwargs_clustering,
        )

        if len(res.known_ids) < min_known_for_eval:
            if verbose:
                print(f"  Skip target BSI (n_known={len(res.known_ids)} < {min_known_for_eval})")
            continue

        dataset = build_eval_pool_ids(res.test_active_ids, inactivos)
        seed_ids_target = [str(x) for x in res.known_ids]
        pool_ids_raw = [str(x) for x in dataset["pool_ids"]]
        pos_ids_raw = [str(x) for x in dataset["pos_ids"]]

        pool_ids_eval, pos_ids_eval, seed_ids_eval, eval_rep_meta = filter_eval_ids_by_rep(
            pool_ids=pool_ids_raw,
            pos_ids=pos_ids_raw,
            seed_ids=seed_ids_target,
            rep=rep_morgan,
        )

        if verbose:
            print(
                "  rep_morgan coverage: "
                f"pool {eval_rep_meta['n_pool_repr']}/{len(pool_ids_raw)} "
                f"(missing={eval_rep_meta['n_pool_missing_in_rep']}), "
                f"seed {eval_rep_meta['n_seed_repr']}/{len(seed_ids_target)} "
                f"(missing={eval_rep_meta['n_seed_missing_in_rep']}), "
                f"pos_in_pool_repr={eval_rep_meta['n_pos_in_pool_repr']}/{len(pos_ids_raw)}"
            )

        if len(pool_ids_eval) == 0 or len(pos_ids_eval) == 0 or len(seed_ids_eval) == 0:
            if verbose:
                print("  Skip BSI target (empty pool/positives/seeds after representation filtering)")
            continue

        model, params, fp_bits = loaded_models[target]
        scores_pool = _max_bsi_score_for_pool_ids(
            seed_ids=seed_ids_eval,
            pool_ids=pool_ids_eval,
            rep_morgan=rep_morgan,
            model=model,
            fp_bits=fp_bits,
            ensure_dense_fp_matrix=ensure_dense_fp_matrix,
            score_query_against_dense_chunk=score_query_against_dense_chunk,
            pool_batch_size=pool_batch_size_bsi,
            pair_batch_size=pair_batch_size_bsi,
        )

        df_eval, cuts = build_eval_table_from_pool_ids(
            pool_ids_eval,
            scores_pool,
            store=store,
            percentiles=percentiles,
            active_ids=pos_ids_eval,
            extra_fields=("smiles",),
        )
        _validate_eval_table_integrity(
            df_eval,
            expected_pool_ids=pool_ids_eval,
            expected_pos_ids=pos_ids_eval,
        )

        df_ef = enrichment_factor_by_percentile_mode(
            df_eval,
            percentiles=percentiles,
            ef_mode=ef_mode,
        )

        bsi_method_name = f"{method_label}__target_seeds"
        df_long = ef_results_to_long_fixed(
            df_ef,
            cuts,
            target,
            method=bsi_method_name,
            percentiles=percentiles,
        )
        df_long["seed_source"] = "target"
        df_long["n_seed_requested"] = len(seed_ids_target)
        df_long["n_seed_effective"] = len(seed_ids_eval)
        df_long["top_n_neighbors_for_seeds"] = np.nan
        df_long["n_pool_eval"] = len(pool_ids_eval)
        df_long["n_pos_eval"] = len(pos_ids_eval)
        df_long["method_label"] = method_label
        all_rows.append(df_long)

        known_active_set_rows.append(_build_known_active_set_row(
            target_id=target,
            method_label=bsi_method_name,
            known_active_ids=seed_ids_target,
        ))
        retrieved_active_set_rows.extend(_build_retrieved_active_set_rows(
            df_eval=df_eval,
            target_id=target,
            method_label=bsi_method_name,
            percentiles=percentiles,
        ))

        meta = {
            "target": target,
            "method_label": method_label,
            "method": bsi_method_name,
            "model_path": str(model_path),
            "fp_bits": int(fp_bits),
            "pfam_id": params.get("pfam_id", ""),
            "holdout_proteins": params.get("holdout_proteins", []),
            "n_target_actives_total": len(activos),
            "n_target_known_original": len(seed_ids_target),
            "n_target_known_eval_repr": len(seed_ids_eval),
            "n_target_test_original": len(res.test_active_ids),
            "n_pool_raw": len(pool_ids_raw),
            "n_pool_eval_repr": len(pool_ids_eval),
            "n_pos_eval_repr": len(pos_ids_eval),
            "n_pool_missing_in_rep_eval": eval_rep_meta["n_pool_missing_in_rep"],
            "n_seed_missing_in_rep_eval": eval_rep_meta["n_seed_missing_in_rep"],
            "split_status": res.diagnostics.get("status", ""),
            "split_mode": res.diagnostics.get("split_mode", ""),
            "n_representatives": res.diagnostics.get("n_representatives", np.nan),
            "butina_cutoff": res.diagnostics.get("butina_cutoff", np.nan),
        }
        meta_rows.append(meta)

    df_long_bsi = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    df_bsi_meta = pd.DataFrame(meta_rows)
    df_retrieved_active_sets = pd.DataFrame(retrieved_active_set_rows)
    df_known_active_sets = pd.DataFrame(known_active_set_rows)
    df_known_consistency_check = _build_known_consistency_check(
        df_known_active_sets,
        strict_known_consistency=strict_known_consistency,
    )

    return (
        df_long_bsi,
        df_bsi_meta,
        df_retrieved_active_sets,
        df_known_active_sets,
        df_known_consistency_check,
    )

def combine_df_family_summary_repetitions(
    df_family_summary_list,
    repetition_names=None,
    group_cols=None,
    median_cols=None,
    keep_all_repetitions=False,
):
    """
    Combine several df_family_summary tables from distinct repetitions and
    return one df_family_summary with the original structure, where the
    specified numeric columns contain medians across repetitions.

    Parameters
    ----------
    df_family_summary_list : list[pd.DataFrame]
        List of df_family_summary DataFrames, one per repetition.

    repetition_names : list[str] or None
        Optional names for the repetitions.

    group_cols : list[str] or None
        Columns defining the same logical cell across repetitions. If None,
        use the expected df_family_summary structure.

    median_cols : list[str] or None
        Columns over which to compute medians across repetitions. If None, use
        the expected numeric columns.

    keep_all_repetitions : bool
        If True, also return the concatenated table with a 'repetition' column.

    Returns
    -------
    df_family_summary : pd.DataFrame
        Table with the same format as before and medians across repetitions.

    df_family_summary_all : pd.DataFrame, optional
        Only when keep_all_repetitions=True.
    """

    if repetition_names is None:
        repetition_names = [f"rep_{i+1}" for i in range(len(df_family_summary_list))]

    if len(repetition_names) != len(df_family_summary_list):
        raise ValueError("repetition_names must have the same length as df_family_summary_list.")

    if group_cols is None:
        group_cols = [
            "method",
            "percentile",
            "familia",
            "method_label",
            "metric_eval",
        ]

    if median_cols is None:
        median_cols = [
            "EF_group_band",
            "n_targets_band",
            "EF_group_cumulative",
            "n_targets_cumulative",
        ]

    dfs = []

    for rep_name, df in zip(repetition_names, df_family_summary_list):
        tmp = df.copy()
        tmp["repetition"] = rep_name

        # Ensure numeric types where appropriate
        for col in median_cols:
            if col in tmp.columns:
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")

        tmp["percentile"] = pd.to_numeric(tmp["percentile"], errors="coerce")

        dfs.append(tmp)

    df_family_summary_all = pd.concat(dfs, ignore_index=True)

    # Validate columns
    missing_group_cols = [c for c in group_cols if c not in df_family_summary_all.columns]
    missing_median_cols = [c for c in median_cols if c not in df_family_summary_all.columns]

    if missing_group_cols:
        raise ValueError(f"Grouping columns are missing: {missing_group_cols}")

    if missing_median_cols:
        raise ValueError(f"Numeric columns required for median calculation are missing: {missing_median_cols}")

    # Median across repetitions while preserving the original structure
    df_family_summary = (
        df_family_summary_all
        .groupby(group_cols, as_index=False, dropna=False)
        .agg({col: "median" for col in median_cols})
    )

    # Reorder columns to match the original format
    original_order = [
        "method",
        "percentile",
        "familia",
        "EF_group_band",
        "n_targets_band",
        "EF_group_cumulative",
        "n_targets_cumulative",
        "method_label",
        "metric_eval",
    ]

    existing_original_order = [c for c in original_order if c in df_family_summary.columns]
    remaining_cols = [c for c in df_family_summary.columns if c not in existing_original_order]

    df_family_summary = df_family_summary[existing_original_order + remaining_cols]

    # Sort for readability
    df_family_summary = df_family_summary.sort_values(
        ["method", "percentile", "familia"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    if keep_all_repetitions:
        return df_family_summary, df_family_summary_all

    return df_family_summary

def parse_id_list(x):
    """
    Convert a cell containing a list object, list string, or NaN into a Python
    list without duplicates while preserving order.
    """
    if isinstance(x, list):
        vals = x
    elif isinstance(x, tuple):
        vals = list(x)
    elif isinstance(x, set):
        vals = list(x)
    elif pd.isna(x):
        return []
    elif isinstance(x, str):
        x = x.strip()
        if not x:
            return []
        vals = ast.literal_eval(x)
        if not isinstance(vals, (list, tuple, set, np.ndarray)):
            vals = [vals]
        else:
            vals = list(vals)
    else:
        vals = [x]

    return list(dict.fromkeys(vals))


def tanimoto_max_to_refs(query_fps, ref_fps):
    """
    For each fingerprint in query_fps, compute its maximum Tanimoto similarity
    against all fingerprints in ref_fps.

    query_fps: binary array with shape (n_query, n_bits)
    ref_fps:   binary array with shape (n_ref, n_bits)

    Return:
        max_sims: array shape (n_query,)
    """
    if query_fps is None or len(query_fps) == 0:
        return np.array([], dtype=np.float32)

    if ref_fps is None or len(ref_fps) == 0:
        return np.full(len(query_fps), np.nan, dtype=np.float32)

    A = query_fps.astype(np.int32, copy=False)
    B = ref_fps.astype(np.int32, copy=False)

    inter = A @ B.T
    a_sum = A.sum(axis=1, dtype=np.int32)[:, None]
    b_sum = B.sum(axis=1, dtype=np.int32)[None, :]
    union = a_sum + b_sum - inter

    sims = np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0
    )

    return sims.max(axis=1).astype(np.float32)

def compute_df_novedad_single_repetition(
    activos,
    recuperados,
    morgan_rep,
    percentil_ref=99.5,
    morgan_label="morgan_1024_r2__target_seeds",
):
    """
    Compute df_novedad for one repetition.

    Require these functions to already exist:
        - parse_id_list
        - tanimoto_max_to_refs

    Return:
        df_novedad with one row per target_id x method_label.
    """

    # =========================
    # Preprocess retrieved actives
    # =========================

    rec_ref = recuperados.loc[
        recuperados["percentile"].astype(float) >= float(percentil_ref),
        ["target_id", "method_label", "percentile", "retrieved_active_ids"]
    ].copy()

    rec_ref["percentile"] = rec_ref["percentile"].astype(float)
    rec_ref["retrieved_active_ids"] = rec_ref["retrieved_active_ids"].apply(parse_id_list)

    # Sort from highest to lowest percentile to preserve logical order
    rec_ref = rec_ref.sort_values(
        ["target_id", "method_label", "percentile"],
        ascending=[True, True, False]
    )

    # Merge cumulative percentile bands
    rec_ref = (
        rec_ref
        .groupby(["target_id", "method_label"], as_index=False)["retrieved_active_ids"]
        .agg(lambda series: list(dict.fromkeys([x for lst in series for x in lst])))
    )

    # =========================
    # Preprocess known actives
    # =========================

    act_known = activos.loc[:, ["target_id", "known_active_ids"]].copy()
    act_known["known_active_ids"] = act_known["known_active_ids"].apply(parse_id_list)

    act_known = (
        act_known
        .groupby("target_id", as_index=False)["known_active_ids"]
        .agg(lambda series: list(dict.fromkeys([x for lst in series for x in lst])))
    )

    # =========================
    # Fast lookup maps
    # =========================

    retrieved_map = rec_ref.set_index(["target_id", "method_label"])["retrieved_active_ids"].to_dict()
    known_map = act_known.set_index("target_id")["known_active_ids"].to_dict()

    targets = sorted(set(act_known["target_id"]) & set(rec_ref["target_id"]))

    methods_to_compare = sorted(
        m for m in rec_ref["method_label"].unique()
        if m != morgan_label
    )

    # =========================
    # Main loop
    # =========================

    rows = []

    for target in targets:
        known_ids = known_map.get(target, [])
        morgan_ids = retrieved_map.get((target, morgan_label), [])

        known_set = set(known_ids)
        morgan_set = set(morgan_ids)

        known_fps = morgan_rep.get_by_ids(known_ids) if len(known_ids) > 0 else None

        # This metric is meaningful only if known_ids represents the correct universe.
        n_morgan_missed_true_actives = max(len(known_set) - len(morgan_set), 0)

        for method in methods_to_compare:
            method_ids = retrieved_map.get((target, method), [])
            method_set = set(method_ids)

            # Actives retrieved by the method but missed by Morgan
            new_ids = [x for x in method_ids if x not in morgan_set]
            new_ids_unique = list(dict.fromkeys(new_ids))

            n_method = len(method_set)
            n_new = len(new_ids_unique)

            pct_new = n_new / n_method if n_method > 0 else np.nan

            frac_morgan_missed_recovered = (
                n_new / n_morgan_missed_true_actives
                if n_morgan_missed_true_actives > 0 else np.nan
            )

            if n_new > 0 and known_fps is not None and len(known_ids) > 0:
                new_fps = morgan_rep.get_by_ids(new_ids_unique)
                max_sims = tanimoto_max_to_refs(new_fps, known_fps)

                similaridad_nuevos_mediana = (
                    float(np.median(max_sims)) if len(max_sims) > 0 else np.nan
                )
                similaridad_nuevos_p10 = (
                    float(np.percentile(max_sims, 10)) if len(max_sims) > 0 else np.nan
                )
            else:
                similaridad_nuevos_mediana = np.nan
                similaridad_nuevos_p10 = np.nan

            rows.append({
                "target_id": target,
                "method_label": method,
                "n_hits_method": n_method,
                "n_hits_morgan": len(morgan_set),
                "n_known_actives": len(known_set),
                "n_morgan_missed_true_actives": n_morgan_missed_true_actives,
                "n_nuevos": n_new,
                "pct_nuevos": pct_new,
                "frac_morgan_missed_recovered": frac_morgan_missed_recovered,
                "similaridad_nuevos_mediana": similaridad_nuevos_mediana,
                "similaridad_nuevos_p10": similaridad_nuevos_p10,
            })

    df_novedad = pd.DataFrame(rows)

    return df_novedad

def compute_df_novedad_across_repetitions(
    seeds,
    known_path_template,
    retrieved_path_template,
    morgan_rep,
    percentil_ref=99.5,
    morgan_label="morgan_1024_r2__target_seeds",
    keep_all_repetitions=False,
):
    """
    Compute df_novedad for multiple repetitions and return a final df_novedad
    containing medians across repetitions.

    Parameters
    ----------
    seeds : list
        List of seeds/repetitions.

    known_path_template : str
        Path template for known_active_sets_all_methods.csv. Must contain {seed}.

    retrieved_path_template : str
        Path template for retrieved_active_sets_all_methods.csv. Must contain {seed}.

    morgan_rep : Representation object
        Morgan representation used to calculate Tanimoto.

    percentil_ref : float
        Cumulative reference percentile, for example 99.5, 99, or 98.

    keep_all_repetitions : bool
        If True, also return the long table containing all repetitions.

    Returns
    -------
    df_novedad : pd.DataFrame
        Same structure as before, with medians across repetitions.

    df_novedad_all : pd.DataFrame, optional
        Only when keep_all_repetitions=True.
    """

    all_dfs = []

    for seed in seeds:
        print(f"Processing seed {seed}...")

        activos = pd.read_csv(known_path_template.format(seed=seed))
        recuperados = pd.read_csv(retrieved_path_template.format(seed=seed))

        df_seed = compute_df_novedad_single_repetition(
            activos=activos,
            recuperados=recuperados,
            morgan_rep=morgan_rep,
            percentil_ref=percentil_ref,
            morgan_label=morgan_label,
        )

        df_seed["seed"] = seed
        all_dfs.append(df_seed)

    df_novedad_all = pd.concat(all_dfs, ignore_index=True)

    # Identifier columns: preserve one row per target x method
    group_cols = [
        "target_id",
        "method_label",
    ]

    # Numeric columns for which medians across repetitions are required
    median_cols = [
        "n_hits_method",
        "n_hits_morgan",
        "n_known_actives",
        "n_morgan_missed_true_actives",
        "n_nuevos",
        "pct_nuevos",
        "frac_morgan_missed_recovered",
        "similaridad_nuevos_mediana",
        "similaridad_nuevos_p10",
    ]

    # Ensure numeric types
    for col in median_cols:
        df_novedad_all[col] = pd.to_numeric(df_novedad_all[col], errors="coerce")

    df_novedad = (
        df_novedad_all
        .groupby(group_cols, as_index=False, dropna=False)
        .agg({col: "median" for col in median_cols})
    )

    # Preserve the original column order
    final_cols = [
        "target_id",
        "method_label",
        "n_hits_method",
        "n_hits_morgan",
        "n_known_actives",
        "n_morgan_missed_true_actives",
        "n_nuevos",
        "pct_nuevos",
        "frac_morgan_missed_recovered",
        "similaridad_nuevos_mediana",
        "similaridad_nuevos_p10",
    ]

    df_novedad = df_novedad[final_cols]

    df_novedad = df_novedad.sort_values(
        ["target_id", "method_label"]
    ).reset_index(drop=True)

    if keep_all_repetitions:
        return df_novedad, df_novedad_all

    return df_novedad
