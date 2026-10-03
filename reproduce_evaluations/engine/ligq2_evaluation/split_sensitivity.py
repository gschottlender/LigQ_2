from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

from armado_datasets_modified import (
    SplitFixed,
    _choose_butina_cluster_representative,
    cap_actives_diverse_by_scaffold,
    choose_split_sizes,
    prepare_actives_df,
)


PROTOCOL_ECFP4 = "ecfp4_butina_0.8"
PROTOCOL_FCFP4 = "fcfp4_butina_0.8"
PROTOCOL_BM = "bemis_murcko_representatives"
PROTOCOLS = (PROTOCOL_ECFP4, PROTOCOL_FCFP4, PROTOCOL_BM)


def _prepare_capped_actives(
    activos: Iterable,
    smiles_df: pd.DataFrame,
    *,
    prepared_actives_df: pd.DataFrame | None,
    max_total_actives: int | None,
    cap_diverse: bool,
    random_state: int,
    id_col: str,
    smiles_col: str,
) -> tuple[pd.DataFrame, dict]:
    frame = (
        prepared_actives_df.copy().reset_index(drop=True)
        if prepared_actives_df is not None
        else prepare_actives_df(activos, smiles_df, id_col=id_col, smiles_col=smiles_col)
    )
    diagnostics = {
        "n_actives_total_dedup": int(len(frame)),
        "max_total_actives": max_total_actives,
    }
    if max_total_actives is not None and len(frame) > max_total_actives:
        if cap_diverse:
            frame = cap_actives_diverse_by_scaffold(
                frame, max_total_actives, random_state, id_col=id_col
            )
            diagnostics["cap_mode"] = "diverse_by_scaffold"
        else:
            frame = frame.sample(
                n=int(max_total_actives), random_state=int(random_state), replace=False
            ).reset_index(drop=True)
            diagnostics["cap_mode"] = "random"
    else:
        frame = frame.reset_index(drop=True)
        diagnostics["cap_mode"] = "none"
    diagnostics["n_actives_after_cap"] = int(len(frame))
    return frame, diagnostics


def _finalize_representatives(
    representatives: pd.DataFrame,
    diagnostics: dict,
    *,
    n_known: int | None,
    n_test: int | None,
    known_frac: float,
    test_to_known_ratio: float,
    known_min: int,
    known_max: int,
    test_min: int,
    test_max: int,
    random_state: int,
    id_col: str,
) -> SplitFixed:
    diagnostics = dict(diagnostics)
    diagnostics["n_representatives"] = int(len(representatives))
    if len(representatives) < 2:
        diagnostics["status"] = "too_few_representatives"
        return SplitFixed([], [], [], diagnostics)

    if n_known is None or n_test is None:
        selected_known, selected_test = choose_split_sizes(
            len(representatives),
            known_frac=known_frac,
            test_to_known_ratio=test_to_known_ratio,
            known_min=known_min,
            known_max=known_max,
            test_min=test_min,
            test_max=test_max,
        )
        if n_known is None:
            n_known = selected_known
        if n_test is None:
            n_test = selected_test

    diagnostics.update({"n_known_req": int(n_known), "n_test_req": int(n_test)})
    indices = list(range(len(representatives)))
    random.Random(int(random_state)).shuffle(indices)
    n_known_eff = min(int(n_known), max(len(indices) - 1, 0))
    known_indices = indices[:n_known_eff]
    remaining = indices[n_known_eff:]
    n_test_eff = min(int(n_test), len(remaining))
    test_indices = remaining[:n_test_eff]

    status = "ok"
    if n_known_eff < int(n_known):
        status = "short_known_set"
    elif n_test_eff < int(n_test):
        status = "short_test_set"
    diagnostics.update(
        {
            "final_n_known": int(n_known_eff),
            "final_n_test": int(n_test_eff),
            "n_unused_representatives": int(
                max(len(representatives) - n_known_eff - n_test_eff, 0)
            ),
            "status": status,
        }
    )
    return SplitFixed(
        known_ids=representatives.iloc[known_indices][id_col].astype(str).tolist(),
        test_active_ids=representatives.iloc[test_indices][id_col].astype(str).tolist(),
        removed_test_too_similar=[],
        diagnostics=diagnostics,
    )


def split_fcfp4_butina_representatives(
    activos: Iterable,
    smiles_df: pd.DataFrame,
    n_known: int | None = None,
    n_test: int | None = None,
    *,
    known_frac: float = 0.10,
    test_to_known_ratio: float = 10.0,
    known_min: int = 10,
    known_max: int = 200,
    test_min: int = 50,
    test_max: int = 1000,
    max_total_actives: int | None = 1000,
    cap_diverse: bool = True,
    id_col: str = "chem_comp_id",
    smiles_col: str = "smiles",
    butina_cutoff: float | None = None,
    nontrivial_tanimoto_cutoff: float = 0.8,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
    random_state: int = 0,
    prepared_actives_df: pd.DataFrame | None = None,
    **_ignored_kwargs,
) -> SplitFixed:
    """Mirror the published Butina split, replacing ECFP4 with FCFP4."""
    cutoff = float(
        nontrivial_tanimoto_cutoff if butina_cutoff is None else butina_cutoff
    )
    if not 0 < cutoff <= 1:
        raise ValueError("butina_cutoff must be in (0, 1].")
    frame, diagnostics = _prepare_capped_actives(
        activos,
        smiles_df,
        prepared_actives_df=prepared_actives_df,
        max_total_actives=max_total_actives,
        cap_diverse=cap_diverse,
        random_state=random_state,
        id_col=id_col,
        smiles_col=smiles_col,
    )
    diagnostics.update(
        {
            "split_mode": PROTOCOL_FCFP4,
            "fingerprint": "Morgan feature invariants (FCFP4)",
            "butina_cutoff": cutoff,
            "butina_radius": int(butina_radius),
            "butina_nbits": int(butina_nbits),
        }
    )
    if len(frame) < 2:
        diagnostics["status"] = "too_few_actives"
        return SplitFixed([], [], [], diagnostics)

    fingerprints = [
        AllChem.GetMorganFingerprintAsBitVect(
            mol,
            int(butina_radius),
            nBits=int(butina_nbits),
            useFeatures=True,
        )
        for mol in frame["mol"]
    ]
    distances: list[float] = []
    for index in range(1, len(fingerprints)):
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprints[index], fingerprints[:index]
        )
        distances.extend(1.0 - float(value) for value in similarities)
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        1.0 - cutoff,
        isDistData=True,
        reordering=True,
    )
    cluster_indices = [tuple(int(i) for i in cluster) for cluster in clusters]
    representative_indices = [
        _choose_butina_cluster_representative(cluster, fingerprints)
        for cluster in cluster_indices
    ]
    representatives = frame.iloc[representative_indices].reset_index(drop=True)
    diagnostics.update(
        {
            "n_clusters_total": int(len(cluster_indices)),
            "cluster_sizes": [int(len(cluster)) for cluster in cluster_indices],
        }
    )
    return _finalize_representatives(
        representatives,
        diagnostics,
        n_known=n_known,
        n_test=n_test,
        known_frac=known_frac,
        test_to_known_ratio=test_to_known_ratio,
        known_min=known_min,
        known_max=known_max,
        test_min=test_min,
        test_max=test_max,
        random_state=random_state,
        id_col=id_col,
    )


def split_bemis_murcko_representatives(
    activos: Iterable,
    smiles_df: pd.DataFrame,
    n_known: int | None = None,
    n_test: int | None = None,
    *,
    known_frac: float = 0.10,
    test_to_known_ratio: float = 10.0,
    known_min: int = 10,
    known_max: int = 200,
    test_min: int = 50,
    test_max: int = 1000,
    max_total_actives: int | None = 1000,
    cap_diverse: bool = True,
    id_col: str = "chem_comp_id",
    smiles_col: str = "smiles",
    random_state: int = 0,
    prepared_actives_df: pd.DataFrame | None = None,
    **_ignored_kwargs,
) -> SplitFixed:
    """Select one seeded representative per Bemis-Murcko scaffold.

    Acyclic molecules have an empty Murcko scaffold. They are deliberately
    assigned structure-specific keys instead of being collapsed into one
    artificial all-acyclic cluster.
    """
    frame, diagnostics = _prepare_capped_actives(
        activos,
        smiles_df,
        prepared_actives_df=prepared_actives_df,
        max_total_actives=max_total_actives,
        cap_diverse=cap_diverse,
        random_state=random_state,
        id_col=id_col,
        smiles_col=smiles_col,
    )
    diagnostics.update(
        {
            "split_mode": PROTOCOL_BM,
            "acyclic_policy": "one structure-specific group per InChIKey",
        }
    )
    if len(frame) < 2:
        diagnostics["status"] = "too_few_actives"
        return SplitFixed([], [], [], diagnostics)

    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in frame.iterrows():
        scaffold = str(row["scaffold"] or "")
        key = scaffold if scaffold else f"__acyclic__:{row['inchikey']}"
        groups[key].append(int(index))

    rng = random.Random(int(random_state) + 104729)
    representative_indices: list[int] = []
    cluster_sizes: list[int] = []
    for key in sorted(groups):
        candidates = groups[key]
        representative_indices.append(candidates[rng.randrange(len(candidates))])
        cluster_sizes.append(len(candidates))
    representatives = frame.iloc[representative_indices].reset_index(drop=True)
    diagnostics.update(
        {
            "n_clusters_total": int(len(groups)),
            "n_scaffolds_nonempty": int(sum(not key.startswith("__acyclic__:") for key in groups)),
            "n_acyclic_structure_groups": int(sum(key.startswith("__acyclic__:") for key in groups)),
            "cluster_sizes": [int(size) for size in cluster_sizes],
        }
    )
    return _finalize_representatives(
        representatives,
        diagnostics,
        n_known=n_known,
        n_test=n_test,
        known_frac=known_frac,
        test_to_known_ratio=test_to_known_ratio,
        known_min=known_min,
        known_max=known_max,
        test_min=test_min,
        test_max=test_max,
        random_state=random_state,
        id_col=id_col,
    )


def splitter_for_protocol(protocol: str):
    if protocol == PROTOCOL_FCFP4:
        return split_fcfp4_butina_representatives
    if protocol == PROTOCOL_BM:
        return split_bemis_murcko_representatives
    raise ValueError(f"No new split must be calculated for protocol: {protocol}")
