#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ligq2_evaluation.config import load_config, output_path, parse_csv_arg, split_kwargs
from ligq2_evaluation.constants import RAW_NEIGHBOR_COUNTS, SEEDS
from ligq2_evaluation.runtime import load_context, prepare_output, write_run_manifest


METHOD = "morgan_1024_r2"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final nearest-neighbor seed benchmark.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yml")))
    parser.add_argument("--seeds")
    parser.add_argument("--targets")
    parser.add_argument("--neighbor-counts", help="Default preserves the notebook values, including auxiliary k=20.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run(cfg, seeds, targets=None, neighbor_counts=None, force=False, resume=False):
    from armado_datasets_modified import run_ef_eval_neighbors_sweep_single_method_clustering

    counts = list(neighbor_counts or RAW_NEIGHBOR_COUNTS)
    out_root = output_path(cfg, "neighbors")
    prepare_output(out_root, force=force, resume=resume)
    context = load_context(cfg, [METHOD], targets, with_neighbors=True)

    for seed in seeds:
        outdir = out_root / f"sweep_neighbors_seed_{seed}"
        prefix = f"sweep_neighbors_{METHOD}"
        sentinel = outdir / f"{prefix}_all_targets.csv"
        if resume and sentinel.exists():
            print(f"[seed={seed}] complete output exists; skipping")
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        result = run_ef_eval_neighbors_sweep_single_method_clustering(
            neighbor_counts=counts,
            targets_dude=context.targets,
            binding_data=context.binding,
            smiles=context.smiles,
            store=context.store,
            neighbor_ranked_ids_by_target=context.neighbor_rankings,
            rep_eval=context.representations[METHOD],
            metric_eval="tanimoto",
            method_label=METHOD,
            rep_morgan_for_seed_selection=context.representations[METHOD],
            split_kwargs=split_kwargs(seed),
        )
        all_rows = result["df_long_neighbors_all"]
        seed_meta = result["df_seed_meta_all"]
        inclusion = result["df_target_inclusion_summary"]
        valid_targets = set(inclusion.loc[inclusion["applicable_to_all_neighbor_counts"], "target"].astype(str))
        valid_rows = all_rows[all_rows["target"].astype(str).isin(valid_targets)].copy().reset_index(drop=True)
        valid_meta = seed_meta[seed_meta["target"].astype(str).isin(valid_targets)].copy().reset_index(drop=True)
        summary = (valid_rows.groupby(["neighbor_count_sweep", "percentile"], as_index=False)
                   .agg(n_targets=("target", "nunique"), median_EF_band=("EF_band", "median"),
                        mean_EF_band=("EF_band", "mean"), median_EF_cumulative=("EF_cumulative", "median"),
                        mean_EF_cumulative=("EF_cumulative", "mean"))
                   .sort_values(["neighbor_count_sweep", "percentile"], ascending=[True, False]))
        report_columns = ["target", "neighbor_count_sweep", "top_n_neighbors_requested", "top_n_neighbors_available",
                          "top_n_neighbors_considered", "neighbor_ids_considered", "n_neighbor_candidates_raw",
                          "n_neighbor_candidates_unique", "n_neighbor_candidates_clean", "k_requested", "k_effective"]
        report = valid_meta[report_columns].sort_values(["neighbor_count_sweep", "target"])

        inclusion.to_csv(outdir / f"{prefix}_target_inclusion_summary.csv", index=False)
        all_rows.to_csv(sentinel, index=False)
        valid_rows.to_csv(outdir / f"{prefix}_valid_targets_only.csv", index=False)
        summary.to_csv(outdir / f"{prefix}_summary_by_percentile.csv", index=False)
        report.to_csv(outdir / f"{prefix}_neighbor_source_report.csv", index=False)
        seed_meta.to_csv(outdir / f"{prefix}_seed_meta_all.csv", index=False)
        valid_meta.to_csv(outdir / f"{prefix}_seed_meta_valid_targets_only.csv", index=False)
        result["df_retrieved_active_sets_all"].to_csv(outdir / f"{prefix}_retrieved_active_sets_all.csv", index=False)
        result["df_known_active_sets_all"].to_csv(outdir / f"{prefix}_known_active_sets_all.csv", index=False)
        result["df_known_consistency_checks_all"].to_csv(outdir / f"{prefix}_known_consistency_checks_all.csv", index=False)
        print(f"[seed={seed}] valid targets: {len(valid_targets)}")

    write_run_manifest(out_root / "run_manifest.json", cfg, "neighbor_benchmark", {
        "seeds": list(seeds), "targets": targets or "all", "neighbor_counts_computed": counts,
        "neighbor_counts_published": [1, 2, 3, 4, 5, 10, 15],
    })


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg, parse_csv_arg(args.seeds, int) or list(SEEDS), parse_csv_arg(args.targets),
        parse_csv_arg(args.neighbor_counts, int), args.force, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
