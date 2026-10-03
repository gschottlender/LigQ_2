#!/usr/bin/env python3
"""Bounded real validation: one target/partition, all evaluation families, no full certification."""
from __future__ import annotations

import argparse
import ast
import datetime as dt
import importlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from package.common import ENGINE, ROOT, command, read_json, write_json

TARGET, SEED = "ampc", 42
MORGAN = "morgan_1024_r2"


def bootstrap():
    from package.figures import bootstrap as install_paths
    install_paths()


def check_representations(args):
    import pandas as pd
    from package.validation import verify_representation_splits, verify_retrieved_sets
    results = args.output_dir / "calculations"
    path = results / "representations/seed_42"
    actual = pd.read_csv(path / "df_long_target_all_methods.csv")
    expected = pd.read_csv(ROOT / "reference_data/representations/seed_42/df_long_target_all_methods.csv")
    expected = expected.loc[expected.target.eq(TARGET)]
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(actual[columns].sort_values(columns).reset_index(drop=True),
        expected.sort_values(columns).reset_index(drop=True), check_dtype=False,
        check_exact=False, atol=1e-8, rtol=1e-8)
    if actual.method_label.nunique() != 7 or len(actual) != 56:
        raise ValueError("All seven representations and eight percentiles are required")
    verify_representation_splits(results, args.data_dir / "benchmark", seeds=[SEED])
    verify_retrieved_sets(pd.read_csv(path / "retrieved_active_sets_all_methods.csv"), SEED,
        pd.read_csv(ROOT / "reference_data/cohorts/retrieved_set_digests.csv"))
    fixed = pd.read_csv(results / "representations/fixed_budget_complementarity/target_partition_budget_metrics.csv")
    if len(fixed) != 63 or not fixed.policy.eq("mean").all() or fixed.budget.nunique() != 1:
        raise ValueError("Expected 63 mean-rank one/two/three-method combinations with an exact common budget")
    write_json(args.output_dir / "representation_check.json", {
        "target":TARGET, "seed":SEED, "methods":7, "EF_rows_match":56,
        "published_known_positive_background_IDs_match":True,
        "historical_retrieved_ID_sets_match":True, "fixed_budget_combinations":63})


def check_domain_adaptive_union(args):
    bootstrap()
    import pandas as pd
    from ligq2_evaluation.config import load_config
    from ligq2_evaluation.runtime import load_context
    from ligq2_evaluation.full_domain import build_full_domain_sources
    from notebook_analysis import run_union_analysis_for_seed
    results = args.output_dir / "calculations"
    cfg = load_config(results / "calculation_config.json")
    # Retain ALL benchmark targets when defining the domain-source exclusion universe.
    context = load_context(cfg, [MORGAN], with_neighbors=False)
    source = build_full_domain_sources(context.targets, context.binding, {TARGET})[TARGET]
    functions = importlib.import_module("run_full_domain_benchmark")
    prefix = results / "neighbors/sweep_neighbors_seed_42" / f"sweep_neighbors_{MORGAN}"
    rows = pd.read_csv(prefix.with_name(prefix.name + "_all_targets.csv"))
    known = pd.read_csv(prefix.with_name(prefix.name + "_known_active_sets_all.csv"))
    record = rows.iloc[0]
    expected = {"n_pool_eval":int(record.n_pool_eval), "n_pos_eval":int(record.n_pos_eval),
                "n_seed_requested":int(record.n_seed_requested),
                "known_active_ids":ast.literal_eval(known.iloc[0].known_active_ids)}
    status = functions._evaluate_cell(cfg, context, source, SEED, expected,
        functions._cell_paths(results / "full_domain_bounded", SEED, TARGET), "bounded-validation")
    if not status["eligible"]:
        raise ValueError(f"Domain cell was not eligible: {status['reason']}")
    adaptive = importlib.import_module("15_analyze_adaptive_k")
    metrics, metadata = adaptive.load_complete_sweep(results / "neighbor_k_complete",
        neighbor_counts=list(range(2, 16)), expected_seeds=(SEED,))
    blast = adaptive.load_blast_hits(args.data_dir / "frozen/neighbor_blast_hits.csv", {TARGET})
    covariates = adaptive.build_k_covariates(metadata, blast)
    choices = adaptive.build_policy_choices(covariates, blast, min_k=2, max_k=15,
        identity_floors=[50., 55., 60.], ligand_budgets=[50, 100], include_combined=False)
    selected = adaptive.attach_selected_results(choices, metrics)
    if selected.empty or not selected.selected_k.between(2, 15).all():
        raise ValueError("Adaptive choices are empty or outside K=2..15")
    selected.to_csv(results / "bounded_adaptive_choices.csv", index=False)
    rescue = importlib.import_module("analyze_identity_ligand_rescue_policy")
    candidate_matrix = covariates.pivot(index=["target", "seed"], columns="neighbor_count_sweep",
                                      values="n_neighbor_candidates_clean")
    ef_matrix = metrics.loc[metrics.percentile.eq(99.5)].pivot(
        index=["target", "seed"], columns="neighbor_count_sweep", values="EF_cumulative")
    identity = rescue.load_ranked_identities(args.data_dir / "frozen/neighbor_blast_hits.csv", {TARGET})
    combined, _, _, _ = rescue.evaluate_grid(candidate_matrix, ef_matrix, identity,
        identity_floors=[50., 55., 60.], ligand_budgets=[50, 100], min_k=2, max_k=15)
    if len(combined) != 6 or not combined.selected_k.between(2, 15).all():
        raise ValueError("Combined identity/rescue grid was not complete")
    combined.to_csv(results / "bounded_identity_rescue_choices.csv", index=False)
    families = read_json(ROOT / "configs/historical_union_categories.json")["historical_families"]
    union, _, _ = run_union_analysis_for_seed(results / "representations", SEED, families, percentile=99.5,
                                             denominator_col="n_pool_unknown_actives")
    if union.empty:
        raise ValueError("Original expanded-union analysis produced no target results")
    union.to_csv(results / "bounded_expanded_union.csv", index=False)
    write_json(args.output_dir / "domain_adaptive_union_check.json", {
        "target":TARGET, "seed":SEED, "domain_matching_split_verified":True,
        "domain_sources":status["n_source_proteins_with_ligands"],
        "complete_real_K_values":list(range(2,16)), "identity_and_budget_policies":True,
        "combined_policy_cells":len(combined), "expanded_union_rows":len(union)})


def check_preprocessing(args):
    import pandas as pd
    actual = pd.read_csv(args.output_dir / "calculations/preprocessing/target_partition_metrics.csv")
    expected = pd.read_csv(ROOT / "reference_data/preprocessing/target_partition_metrics.csv")
    expected = expected.loc[expected.target.eq(TARGET) & expected.random_state.eq(SEED)]
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(actual[columns].sort_values(columns).reset_index(drop=True),
        expected.sort_values(columns).reset_index(drop=True), check_dtype=False,
        check_exact=False, atol=1e-8, rtol=1e-8)
    if len(actual) != 96:
        raise ValueError("Expected three preprocessing protocols, four methods and eight percentiles")
    write_json(args.output_dir / "preprocessing_check.json", {
        "target":TARGET, "seed":SEED, "rows_match_historical":96,
        "protocols":"ECFP4/Butina, FCFP4/Butina, Bemis-Murcko", "retrieval_methods":4})


def check_distributions(args):
    import numpy as np
    import pandas as pd
    path = args.output_dir / "calculations/distributions"
    metadata = read_json(path / "run_metadata.json")
    counts = pd.read_csv(path / "hit_counts_by_partition.csv")
    histograms = pd.read_csv(path / "histograms_category_balanced.csv")
    fractions = histograms.groupby(["property", "group"]).fraction.sum()
    if len(counts) != 1 or metadata["n_targets"] != 1 or metadata["n_partitions"] != 1:
        raise ValueError("Distribution analysis exceeded its bounded scope")
    if set(histograms.group) != {"TP", "putative_FP", "putative_TN"} or len(fractions) != 12:
        raise ValueError("Expected four properties/similarities and three groups")
    if not np.allclose(fractions.to_numpy(), 1., atol=1e-12):
        raise ValueError("Normalized distributions do not sum to one")
    write_json(args.output_dir / "distribution_check.json", {
        "target":TARGET, "seed":SEED, "groups":3, "normalized_distributions":12,
        "inclusive_P99_hit_IDs_verified_by_engine":True})


def run(args):
    from package.workflow import calculation_config, verify_runtime
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    results = output / "calculations"
    cfg_path = results / "calculation_config.json"
    write_json(cfg_path, calculation_config(args.data_dir, results))
    versions = {"main":verify_runtime(args.python), "chemistry":verify_runtime(args.chemistry_python, chemistry=True)}
    report_path = output / "bounded_validation_report.json"
    report = {"status":"running", "target":TARGET, "seed":SEED, "versions":versions,
              "completed_steps":[], "full_recalculation_certified":False,
              "maximum_minutes":args.max_minutes, "started_utc":dt.datetime.now(dt.timezone.utc).isoformat()}
    write_json(report_path, report)
    deadline = time.monotonic() + args.max_minutes*60
    environment = dict(os.environ, MPLBACKEND="Agg", MPLCONFIGDIR=str(output / "mpl_cache"),
        PYTHONUNBUFFERED="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    environment["PYTHONPATH"] = os.pathsep.join((str(ENGINE), str(ENGINE / "dependencies/evaluation_core"),
                                               str(ENGINE / "dependencies/ligq_core")))
    if args.cache_dir:
        environment.update(HF_HOME=str(args.cache_dir), HF_HUB_CACHE=str(args.cache_dir / "hub"),
                           HF_XET_CACHE=str(args.cache_dir / "xet"))
    def cli(python, *argv):
        return [str(python), str(ROOT / "reproduce.py"), *map(str, argv)]
    def phase(name):
        return [str(args.python), str(Path(__file__).resolve()), "--phase", name,
                "--data-dir", str(args.data_dir), "--output-dir", str(output)]
    steps = [
        ("bounded_download", cli(args.python, "download", "--targets", TARGET, "--seeds", SEED, "--data-dir", args.data_dir)),
        ("unit_tests", cli(args.chemistry_python, "smoke-test", "--output-dir", output / "unit_tests")),
        ("seven_representations_and_fixed_budget", command("02_run_representation_benchmark.py", "--config", cfg_path,
             "--targets", TARGET, "--seeds", SEED, "--fixed-budget-combinations", "--fusion-policy", "mean", "--resume", python=args.python)),
        ("exact_representation_regression", phase("representations")),
        ("original_neighbor_K_values", command("03_run_neighbor_benchmark.py", "--config", cfg_path,
             "--targets", TARGET, "--seeds", SEED, "--neighbor-counts", "2,3,4,5,10,15", "--resume", python=args.python)),
        ("complete_missing_K_values", command("14_complete_neighbor_k_sweep.py", "--config", cfg_path,
             "--targets", TARGET, "--seeds", SEED, "--retained-root", results / "neighbors", "--blast-hits", args.data_dir / "frozen/neighbor_blast_hits.csv",
             "--matched-targets", ROOT / "reference_data/cohorts/neighbor_targets.csv", "--output-dir", results / "neighbor_k_complete",
             "--neighbor-counts", "2-15", "--resume", python=args.python)),
        ("full_domain_adaptive_and_expanded_union", phase("domain_adaptive_union")),
        ("preprocessing_sensitivity", command("18_run_active_preprocessing_sensitivity.py", "--config", cfg_path,
             "--targets", TARGET, "--seeds", SEED, "--butina-cutoff", ".8", "--baseline-results-dir", results / "representations",
             "--benchmark-dir", args.data_dir / "benchmark", "--output-dir", results / "preprocessing", "--resume", "--device", "auto", python=args.chemistry_python)),
        ("exact_preprocessing_regression", phase("preprocessing")),
        ("three_group_distributions", command("19_characterize_retrieved_hits.py", "--config", cfg_path,
             "--reference-dir", results / "representations", "--targets", TARGET, "--seeds", SEED,
             "--output-dir", results / "distributions", "--percentile", "99", "--include-nonretrieved-background",
             "--descriptor-cache", results / "descriptor_cache", "--descriptor-workers", "8", "--skip-individual", "--resume", python=args.chemistry_python)),
        ("distribution_normalization_checks", phase("distributions")),
        ("all_historical_figures", cli(args.chemistry_python, "plot", "--source", "historical", "--output-dir", output / "historical_figures")),
        ("all_historical_pixel_checks", cli(args.chemistry_python, "validate", "--figures-only", "--output-dir", output / "historical_figures")),
    ]
    write_json(output / "command_plan.json", [{"step":name, "command":argv} for name, argv in steps])
    try:
        for name, argv in steps:
            report["current_step"] = name
            write_json(report_path, report)
            print(f"Bounded validation: {name}", flush=True)
            remaining = deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Bounded validation reached its total time limit")
            started = time.monotonic()
            with (output / f"{name}.log").open("a", encoding="utf-8") as log:
                child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=remaining)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            if code:
                raise RuntimeError(f"{name} exited with {code}; inspect {output / (name+'.log')}")
            report["completed_steps"].append({"step":name, "elapsed_seconds":time.monotonic()-started})
            write_json(report_path, report)
        report.update(status="passed", finished_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                      scope="Bounded functional/regression checks, not a complete 61-target recalculation")
        write_json(report_path, report)
        print(f"Bounded validation PASSED: {report_path}", flush=True)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                      finished_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        write_json(report_path, report)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/bounded_publication")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--chemistry-python", type=Path)
    parser.add_argument("--max-minutes", type=float, default=15)
    parser.add_argument("--phase", choices=("representations", "domain_adaptive_union", "preprocessing", "distributions"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.phase:
        {"representations":check_representations, "domain_adaptive_union":check_domain_adaptive_union,
         "preprocessing":check_preprocessing, "distributions":check_distributions}[args.phase](args)
    else:
        if args.chemistry_python is None or not 0 < args.max_minutes <= 30:
            parser.error("Provide --chemistry-python and a time bound of at most 30 minutes")
        run(args)
