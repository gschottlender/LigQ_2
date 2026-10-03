#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import pandas as pd

from ligq2_evaluation.config import load_config, output_path, parse_csv_arg, split_kwargs
from ligq2_evaluation.constants import FAMILIES, METHODS, RAW_PERCENTILES, SEEDS
from ligq2_evaluation.families import compute_family_balanced_ef_both
from ligq2_evaluation.runtime import load_context, prepare_output, write_run_manifest
from ligq2_evaluation.fixed_budget import (
    POLICIES, RankingCache, analyze_cache, cache_signature, targets_from_legacy,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run the final S1 molecular-representation benchmark.")
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yml")))
    parser.add_argument("--seeds", help="Comma-separated subset; default is the five published seeds.")
    parser.add_argument("--targets", help="Comma-separated target subset for smoke tests.")
    parser.add_argument("--methods", help="Comma-separated method subset for smoke tests.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixed-budget-combinations", action="store_true",
                        help="Cache the original scores and evaluate all pairs/triples at an exact top-0.5%% budget")
    parser.add_argument("--fusion-policy", choices=(*POLICIES, "all"), default="best",
                        help="Combination rule; only used with --fixed-budget-combinations")
    parser.add_argument("--reference-results-dir", type=Path,
                        help="Optional historical seed_<N> directory tree to verify unchanged EF and retrieved sets")
    return parser.parse_args()


def _evaluate_method(context, target_frame, method, seed, score_sink=None):
    from armado_datasets_modified import run_ef_eval_target_vs_neighbors_clustering

    return run_ef_eval_target_vs_neighbors_clustering(
        targets_dude=target_frame,
        binding_data=context.binding,
        smiles=context.smiles,
        store=context.store,
        neighbor_ranked_ids_by_target=context.neighbor_rankings,
        rep_eval=context.representations[method],
        metric_eval=METHODS[method],
        method_label=method,
        device_eval="cpu",
        assume_normalized_eval=None,
        clamp_max_eval=None,
        rep_morgan_for_seed_selection=context.representations["morgan_1024_r2"],
        top_n_neighbors_for_seeds=5,
        tanimoto_cleanup_cutoff=0.85,
        device_seed_selection="cpu",
        percentiles=RAW_PERCENTILES,
        min_known_for_eval=50,
        split_kwargs=split_kwargs(seed),
        verbose=True,
        neighbor_short_policy="allow_short",
        strict_known_consistency=False,
        use_neighbor_seeds=False,
        ef_mode="both",
        target_score_sink=score_sink,
    )


def _assert_historical_unchanged(seed_dir: Path, method: str, targets: list[str],
                                 new_long: pd.DataFrame, new_retrieved: pd.DataFrame) -> None:
    """Fail rather than silently combine a rerun with different historical data."""
    previous = pd.read_csv(seed_dir / "df_long_target_all_methods.csv")
    previous = previous[(previous["method_label"] == method) & previous["target"].isin(targets)]
    current = new_long.assign(method_label=method)
    columns = current.columns.tolist()
    sort_by = ["target", "percentile"]
    pd.testing.assert_frame_equal(
        current[columns].sort_values(sort_by).reset_index(drop=True),
        previous[columns].sort_values(sort_by).reset_index(drop=True),
        check_dtype=False, check_exact=False, atol=1e-8, rtol=1e-8,
        obj=f"Historical EF output for {method}",
    )
    prior_sets = pd.read_csv(seed_dir / "retrieved_active_sets_all_methods.csv")
    label = f"{method}__target_seeds"
    prior_sets = prior_sets[(prior_sets["method_label"] == label) & prior_sets["target_id"].isin(targets)]

    def normalized_sets(frame):
        result = {}
        for _, row in frame.iterrows():
            key = (str(row["target_id"]), float(row["percentile"]))
            result[key] = (
                set(ast.literal_eval(row["retrieved_active_ids"]) if isinstance(row["retrieved_active_ids"], str)
                    else row["retrieved_active_ids"]),
                set(ast.literal_eval(row["retrieved_inactive_ids"]) if isinstance(row["retrieved_inactive_ids"], str)
                    else row["retrieved_inactive_ids"]),
            )
        return result

    if normalized_sets(prior_sets) != normalized_sets(new_retrieved):
        raise AssertionError(f"Historical retrieved sets changed for {method}, targets={targets}")


def run(cfg, seeds, targets=None, methods=None, force=False, resume=False,
        fixed_budget_combinations=False, fusion_policy="best", reference_results_dir=None):
    from unknown_active_sets import save_unknown_active_sets_by_seed

    selected_methods = list(methods or METHODS)
    unknown = sorted(set(selected_methods) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    if fixed_budget_combinations and len(selected_methods) < 2:
        raise ValueError("Fixed-budget combinations require at least two methods")
    if fixed_budget_combinations and len(set(selected_methods)) != len(selected_methods):
        raise ValueError("Fixed-budget methods must be distinct")
    if fusion_policy != "best" and not fixed_budget_combinations:
        raise ValueError("--fusion-policy requires --fixed-budget-combinations")
    out_root = output_path(cfg, "representations")
    prepare_output(out_root, force=force, resume=resume)
    cache = RankingCache(out_root, cache_signature(cfg, selected_methods)) if fixed_budget_combinations else None
    policies = list(POLICIES) if fusion_policy == "all" else [fusion_policy]

    # Recombining an already complete cache never loads the molecular store or
    # reruns BLAST/similarity. Historical outputs remain untouched.
    if cache is not None and resume and all(
        (out_root / f"seed_{seed}" / "df_long_target_all_methods.csv").is_file() for seed in seeds
    ):
        historical_targets = targets_from_legacy(out_root, list(seeds), selected_methods, targets)
        if all(cache.has(seed, target, method) for seed in seeds
               for target in historical_targets for method in selected_methods):
            analyze_cache(cache, out_root / "fixed_budget_complementarity", seeds=list(seeds),
                          targets=historical_targets, methods=selected_methods, policies=policies)
            return

    context = load_context(cfg, selected_methods, targets, with_neighbors=True)
    wrote_legacy = False

    for seed in seeds:
        outdir = out_root / f"seed_{seed}"
        sentinel = outdir / "df_long_target_all_methods.csv"
        if resume and sentinel.exists():
            if cache is not None:
                expected_targets = targets_from_legacy(out_root, [seed], selected_methods, targets)
                for method in selected_methods:
                    missing = [target for target in expected_targets if not cache.has(seed, target, method)]
                    if not missing:
                        continue
                    print(f"[seed={seed}, method={method}] generating missing ranking cache for {len(missing)} targets")
                    captured = []
                    subset = context.targets[context.targets["target"].astype(str).isin(missing)]
                    values = _evaluate_method(context, subset, method, seed,
                                              score_sink=lambda **row: captured.append(row))
                    new_long, _, _, new_retrieved, _, _ = values
                    _assert_historical_unchanged(outdir, method, missing, new_long, new_retrieved)
                    if {row["target"] for row in captured} != set(missing):
                        raise ValueError(f"Could not regenerate every historical ranking: {missing}")
                    for row in captured:
                        cache.save(seed=seed, **row)
            else:
                print(f"[seed={seed}] complete output exists; skipping")
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        results = {}
        summaries, group_summaries = [], []
        retrieved, known, checks = [], [], []
        for method in selected_methods:
            print(f"\n========== method={method} seed={seed} ==========")
            sink = (lambda **row: cache.save(seed=seed, **row)) if cache is not None else None
            values = _evaluate_method(context, context.targets, method, seed, score_sink=sink)
            df_long, df_neighbors, df_meta, df_retrieved, df_known, df_check = values
            if reference_results_dir is not None:
                reference_seed = Path(reference_results_dir).expanduser().resolve() / f"seed_{seed}"
                _assert_historical_unchanged(reference_seed, method,
                                             df_long["target"].astype(str).unique().tolist(),
                                             df_long, df_retrieved)
            balanced, by_family, annotated = compute_family_balanced_ef_both(df_long, FAMILIES)
            results[method] = {"df_long": df_long, "neighbors": df_neighbors, "meta": df_meta, "annotated": annotated}
            summary = balanced.assign(method_label=method, metric_eval=METHODS[method])
            summaries.append(summary[["method_label", "metric_eval", "method", "percentile", "EF_balanced_band", "EF_balanced_cumulative", "n_groups_band", "n_groups_cumulative"]])
            group_summaries.append(by_family.assign(method_label=method, metric_eval=METHODS[method]))
            retrieved.append(df_retrieved.copy())
            known.append(df_known.copy())
            checks.append(df_check.copy())

        family_summary = pd.concat(summaries, ignore_index=True).sort_values(["percentile", "EF_balanced_cumulative"], ascending=[False, False])
        family_groups = pd.concat(group_summaries, ignore_index=True)
        all_retrieved = pd.concat(retrieved, ignore_index=True)
        all_known = pd.concat(known, ignore_index=True)
        all_checks = pd.concat(checks, ignore_index=True)
        all_long = pd.concat([entry["df_long"].assign(method_label=method) for method, entry in results.items()], ignore_index=True)
        score_cuts = all_long[["target", "method_label", "method", "percentile", "score_cut", "n_pool_eval", "n_pos_eval"]].drop_duplicates()

        family_summary.to_csv(outdir / "resultados_metodos_EF_family_balanced.csv", index=False)
        family_groups.to_csv(outdir / "resultados_metodos_EF_por_familia.csv", index=False)
        all_retrieved.to_csv(outdir / "retrieved_active_sets_all_methods.csv", index=False)
        all_known.to_csv(outdir / "known_active_sets_all_methods.csv", index=False)
        all_checks.to_csv(outdir / "known_consistency_checks_all_methods.csv", index=False)
        all_long.to_csv(sentinel, index=False)
        score_cuts.to_csv(outdir / "score_cuts_by_target_method_percentile.csv", index=False)
        wrote_legacy = True

    if wrote_legacy or cache is None:
        save_unknown_active_sets_by_seed(
            targets_dude=context.targets,
            binding_data=context.binding,
            smiles=context.smiles,
            seeds=seeds,
            base_out=str(out_root),
            min_known_for_eval=50,
            split_kwargs_base={key: value for key, value in split_kwargs(seeds[0]).items() if key != "random_state"},
            use_clustering_split=True,
            butina_cutoff=0.8,
            butina_radius=2,
            butina_nbits=1024,
        )
        write_run_manifest(out_root / "run_manifest.json", cfg, "representation_benchmark", {
            "seeds": list(seeds), "targets": targets or "all", "methods": selected_methods,
            "percentiles": list(RAW_PERCENTILES), "published_percentiles": list(RAW_PERCENTILES[:6]),
        })
    if cache is not None:
        expected_targets = targets_from_legacy(out_root, list(seeds), selected_methods, targets)
        missing_cache = [(seed, target, method) for seed in seeds for target in expected_targets
                         for method in selected_methods if not cache.has(seed, target, method)]
        if missing_cache:
            raise ValueError(f"Incomplete ranking cache after evaluation: {missing_cache[:5]}")
        analyze_cache(cache, out_root / "fixed_budget_complementarity", seeds=list(seeds),
                      targets=expected_targets, methods=selected_methods, policies=policies)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    run(cfg, parse_csv_arg(args.seeds, int) or list(SEEDS), parse_csv_arg(args.targets),
        parse_csv_arg(args.methods), args.force, args.resume,
        args.fixed_budget_combinations, args.fusion_policy, args.reference_results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
