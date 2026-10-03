from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd


import armado_datasets_modified as adm


def _as_json_list(values: Iterable) -> str:
    return json.dumps([str(x) for x in values], ensure_ascii=False)


def save_unknown_active_sets_by_seed(
    *,
    targets_dude: pd.DataFrame,
    binding_data: pd.DataFrame,
    smiles: pd.DataFrame,
    seeds=(8,),
    base_out: str = "resultados_EF_repeticiones_con_inactivos",
    min_known_for_eval: int = 50,
    split_kwargs_base: dict | None = None,
    use_clustering_split: bool = False,
    butina_cutoff: float | None = None,
    butina_radius: int = 2,
    butina_nbits: int = 1024,
) -> dict[int, pd.DataFrame]:
    """
    Reconstruct only the known/unknown/inactive split for each target and seed.

    This function does not compute evaluation similarities or run methods. It
    follows the same logic as run_ef_eval_target_vs_neighbors up to construction
    of the evaluation pool.
    """
    if split_kwargs_base is None:
        split_kwargs_base = dict(
            max_total_actives=1000,
            known_frac=0.20,
            known_min=10,
            known_max=200,
            test_to_known_ratio=5,
            test_min=50,
            test_max=1000,
            nontrivial_tanimoto_cutoff=0.85,
            fallback_if_short="allow_short",
        )

    targets_dude_s = targets_dude.copy()
    binding_data_s = binding_data.copy()
    smiles_s = smiles.copy()

    targets_dude_s["target"] = targets_dude_s["target"].astype(str)
    if "uniprot_id" not in targets_dude_s.columns:
        raise ValueError("targets_dude must contain the 'uniprot_id' column.")
    targets_dude_s["uniprot_id"] = targets_dude_s["uniprot_id"].astype(str)

    binding_data_s["uniprot_id"] = binding_data_s["uniprot_id"].astype(str)
    binding_data_s["chem_comp_id"] = binding_data_s["chem_comp_id"].astype(str)
    binding_data_s["pfam_id"] = binding_data_s["pfam_id"].astype(str)
    smiles_s["chem_comp_id"] = smiles_s["chem_comp_id"].astype(str)

    outputs: dict[int, pd.DataFrame] = {}

    for random_state in seeds:
        base_path = Path(base_out)
        if base_path.name == f"seed_{random_state}":
            outdir = base_path
        else:
            outdir = base_path / f"seed_{random_state}"
        outdir.mkdir(parents=True, exist_ok=True)

        split_kwargs = dict(split_kwargs_base)
        split_kwargs["random_state"] = int(random_state)
        if use_clustering_split:
            if butina_cutoff is None:
                butina_cutoff_seed = split_kwargs.get("nontrivial_tanimoto_cutoff", 0.85)
            else:
                butina_cutoff_seed = butina_cutoff
            split_kwargs.update({
                "butina_cutoff": float(butina_cutoff_seed),
                "butina_radius": int(butina_radius),
                "butina_nbits": int(butina_nbits),
            })

        split_func = (
            adm.split_hit_expansion_fixed_sizes_clustering
            if use_clustering_split
            else adm.split_hit_expansion_fixed_sizes
        )

        rows = []
        target_list = targets_dude_s["target"].dropna().astype(str).unique().tolist()

        for target in target_list:
            uniprots = (
                targets_dude_s.loc[targets_dude_s["target"] == target, "uniprot_id"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            activos = (
                binding_data_s.loc[
                    binding_data_s["uniprot_id"].isin(uniprots),
                    "chem_comp_id",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            fams_target = (
                binding_data_s.loc[
                    binding_data_s["uniprot_id"].isin(uniprots),
                    "pfam_id",
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            lista_negra = (
                binding_data_s.loc[
                    binding_data_s["pfam_id"].isin(fams_target),
                    "chem_comp_id",
                ]
                .dropna()
                .astype(str)
                .tolist()
            )

            inactivos = (
                smiles_s.loc[
                    ~smiles_s["chem_comp_id"].isin(lista_negra),
                    "chem_comp_id",
                ]
                .dropna()
                .astype(str)
                .tolist()
            )
            inactivos = adm.stable_set_difference_preserve_order(inactivos, activos)

            res = split_func(
                activos=activos,
                smiles_df=smiles_s,
                **split_kwargs,
            )

            skip_reason = ""
            if len(res.known_ids) < min_known_for_eval:
                skip_reason = f"n_known<{min_known_for_eval}"

            dataset = adm.build_eval_pool_ids(res.test_active_ids, inactivos)

            rows.append(
                {
                    "seed": int(random_state),
                    "target_id": str(target),
                    "split_mode": "butina_representatives" if use_clustering_split else "post_split_tanimoto_filter",
                    "butina_cutoff": float(split_kwargs["butina_cutoff"]) if use_clustering_split else "",
                    "butina_radius": int(butina_radius) if use_clustering_split else "",
                    "butina_nbits": int(butina_nbits) if use_clustering_split else "",
                    "skipped_by_eval": bool(skip_reason),
                    "skip_reason": skip_reason,
                    "uniprot_ids": _as_json_list(uniprots),
                    "pfam_ids": _as_json_list(fams_target),
                    "known_active_ids": _as_json_list(res.known_ids),
                    "unknown_active_ids": _as_json_list(res.test_active_ids),
                    "inactive_ids": _as_json_list(inactivos),
                    "pool_ids": _as_json_list(dataset["pool_ids"]),
                    "n_active_total_raw": int(len(activos)),
                    "n_inactive_total_raw": int(len(inactivos)),
                    "n_known_actives": int(len(res.known_ids)),
                    "n_unknown_actives": int(len(res.test_active_ids)),
                    "n_pool_total": int(len(dataset["pool_ids"])),
                    "n_pool_unknown_actives": int(len(dataset["pos_ids"])),
                    "n_pool_inactives": int(len(dataset["neg_ids"])),
                }
            )

        df_unknown_by_target = pd.DataFrame(rows)

        full_path = outdir / "unknown_active_sets_by_target.csv"
        counts_path = outdir / "target_total_counts.csv"

        df_unknown_by_target.to_csv(full_path, index=False)
        df_unknown_by_target[
            [
                "seed",
                "target_id",
                "split_mode",
                "butina_cutoff",
                "skipped_by_eval",
                "skip_reason",
                "n_active_total_raw",
                "n_known_actives",
                "n_unknown_actives",
                "n_inactive_total_raw",
                "n_pool_total",
                "n_pool_unknown_actives",
                "n_pool_inactives",
            ]
        ].to_csv(counts_path, index=False)

        print(f"[seed={random_state}] saved: {full_path}")
        print(f"[seed={random_state}] saved: {counts_path}")
        outputs[int(random_state)] = df_unknown_by_target

    return outputs


if __name__ == "__main__":
    required = ("targets_dude", "binding_data", "smiles")
    missing = [name for name in required if name not in globals()]
    if missing:
        raise RuntimeError(
            "Required in-memory variables are missing: "
            + ", ".join(missing)
            + ". Run it from the notebook with: %run -i save_unknown_actives_by_seed.py"
        )

    if "OUTDIR" in globals():
        base_out = OUTDIR
    elif "outdir" in globals():
        base_out = outdir
    else:
        base_out = "resultados_EF_repeticiones_con_inactivos"

    if "SEEDS" in globals():
        seeds = SEEDS
    elif "random_state" in globals():
        seeds = [random_state]
    else:
        seeds = [8]

    use_clustering_split = bool(globals().get("USE_CLUSTERING_SPLIT", False))
    butina_cutoff = globals().get("BUTINA_CUTOFF", None)
    if butina_cutoff is not None:
        butina_cutoff = float(butina_cutoff)
    butina_radius = int(globals().get("BUTINA_RADIUS", 2))
    butina_nbits = int(globals().get("BUTINA_NBITS", 1024))

    if "SPLIT_KWARGS_BASE" in globals():
        split_kwargs_base = SPLIT_KWARGS_BASE
    elif "split_kwargs" in globals():
        split_kwargs_base = {
            key: value
            for key, value in split_kwargs.items()
            if key != "random_state"
        }
    else:
        split_kwargs_base = None

    unknown_active_outputs = save_unknown_active_sets_by_seed(
        targets_dude=targets_dude,
        binding_data=binding_data,
        smiles=smiles,
        seeds=seeds,
        base_out=base_out,
        min_known_for_eval=50,
        split_kwargs_base=split_kwargs_base,
        use_clustering_split=use_clustering_split,
        butina_cutoff=butina_cutoff,
        butina_radius=butina_radius,
        butina_nbits=butina_nbits,
    )

    df_unknown_by_target = unknown_active_outputs[int(list(seeds)[-1])]
