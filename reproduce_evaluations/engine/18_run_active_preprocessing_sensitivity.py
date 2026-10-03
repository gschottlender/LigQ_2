#!/usr/bin/env python3
"""Sensitivity of representation rankings to active-set preprocessing.

The target panel is selected without using enrichment results. Selection is
stratified by protein family and favors targets retaining substantial known
and evaluation sets under all three preprocessing protocols:

* published ECFP4/Butina at Tanimoto 0.8 (reused, never recalculated);
* FCFP4/Butina at Tanimoto 0.8;
* one seeded representative per Bemis-Murcko scaffold.

The expensive molecular searches are limited to ECFP4, FCFP4, Topological
Torsion, and ChemBERTa. Run ``--prepare-only`` first to inspect the panel.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit

from ligq2_evaluation.config import load_config, parse_csv_arg, resolve_path, split_kwargs
from ligq2_evaluation.constants import FAMILIES, FAMILY_LABELS, METHODS, PRETTY_METHODS, RAW_PERCENTILES, SEEDS
from ligq2_evaluation.runtime import load_context
from ligq2_evaluation.split_sensitivity import (
    PROTOCOL_BM,
    PROTOCOL_ECFP4,
    PROTOCOL_FCFP4,
    PROTOCOLS,
    splitter_for_protocol,
)


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = SCRIPT_DIR.parent
DEFAULT_OUTPUT = SCRIPT_DIR / "reproduction_output" / "active_preprocessing_sensitivity"
DEFAULT_BASELINE_RESULTS = SCRIPT_DIR / "reproduction_output" / "representation_benchmark"
DEFAULT_BENCHMARK = WORKSPACE / "benchmark_hf_10_90_butina080_uniform"
DEFAULT_METHODS = (
    "morgan_1024_r2",
    "morgan_feature_1024_r2",
    "topological_torsion_rdkit_1024",
    "chemberta_zinc_base_768",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yml"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-results-dir", type=Path, default=DEFAULT_BASELINE_RESULTS)
    parser.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--seeds", help="Comma-separated seeds; defaults to the five published partitions")
    parser.add_argument("--methods", help="Comma-separated retrieval methods")
    parser.add_argument("--targets", help="Explicit comma-separated panel; bypasses automatic selection")
    parser.add_argument("--targets-per-family", type=int, default=2)
    parser.add_argument("--min-known", type=int, default=50)
    parser.add_argument("--butina-cutoff", type=float, default=0.8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="Similarity backend; auto uses CUDA when available")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Build split census and select the panel without molecular searches")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def _family_map() -> dict[str, str]:
    return {
        target: FAMILY_LABELS[family]
        for family, subfamilies in FAMILIES.items()
        for targets in subfamilies.values()
        for target in targets
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _split_path(output: Path, protocol: str, seed: int, target: str) -> Path:
    return output / "split_manifests" / protocol / f"seed_{seed}" / f"{target}.json"


def _serialize_split(path: Path, target: str, seed: int, protocol: str, split) -> None:
    _write_json(
        path,
        {
            "target": target,
            "random_state": int(seed),
            "protocol": protocol,
            "known_ids": [str(value) for value in split.known_ids],
            "evaluation_active_ids": [str(value) for value in split.test_active_ids],
            "diagnostics": split.diagnostics,
        },
    )


def _load_split(path: Path):
    from armado_datasets_modified import SplitFixed

    payload = json.loads(path.read_text(encoding="utf-8"))
    return SplitFixed(
        known_ids=payload["known_ids"],
        test_active_ids=payload["evaluation_active_ids"],
        removed_test_too_similar=[],
        diagnostics=payload["diagnostics"],
    )


def _active_ids_by_target(targets: pd.DataFrame, binding: pd.DataFrame) -> dict[str, list[str]]:
    binding = binding.copy()
    binding["uniprot_id"] = binding["uniprot_id"].astype(str)
    binding["chem_comp_id"] = binding["chem_comp_id"].astype(str)
    result: dict[str, list[str]] = {}
    for target, group in targets.groupby("target", sort=True):
        uniprots = group["uniprot_id"].dropna().astype(str).unique().tolist()
        result[str(target)] = (
            binding.loc[binding["uniprot_id"].isin(uniprots), "chem_comp_id"]
            .dropna().astype(str).unique().tolist()
        )
    return result


def _baseline_census(benchmark_dir: Path, seeds: list[int]) -> pd.DataFrame:
    manifest_path = benchmark_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Published benchmark manifest not found: {manifest_path}")
    frame = pd.read_csv(manifest_path)
    frame = frame[frame["random_state"].astype(int).isin(seeds)].copy()
    frame = frame.rename(
        columns={
            "target_id": "target",
            "n_active_total_raw": "n_active_total_raw",
            "n_known_actives": "n_known",
            "n_evaluation_actives": "n_evaluation",
        }
    )
    frame["protocol"] = PROTOCOL_ECFP4
    frame["n_representatives"] = frame["n_known"] + frame["n_evaluation"]
    frame["status"] = "published"
    return frame[
        ["target", "random_state", "protocol", "n_active_total_raw", "n_representatives",
         "n_known", "n_evaluation", "status"]
    ]


def prepare_splits_and_panel(
    cfg: dict,
    output: Path,
    benchmark_dir: Path,
    seeds: list[int],
    *,
    targets_per_family: int,
    min_known: int,
    cutoff: float,
    explicit_targets: list[str] | None,
    resume: bool,
    quiet: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    targets = pd.read_csv(resolve_path(cfg, "paths", "targets_csv"))
    targets["target"] = targets["target"].astype(str)
    targets["uniprot_id"] = targets["uniprot_id"].astype(str)
    binding = pd.read_parquet(
        resolve_path(cfg, "paths", "binding_data"),
        columns=["uniprot_id", "chem_comp_id"],
    )
    smiles = pd.read_parquet(
        resolve_path(cfg, "paths", "smiles_data"),
        columns=["chem_comp_id", "smiles"],
    )
    smiles["chem_comp_id"] = smiles["chem_comp_id"].astype(str)

    baseline = _baseline_census(benchmark_dir, seeds)
    eligible_baseline = set(
        baseline.groupby("target")["random_state"].nunique().loc[lambda value: value == len(seeds)].index
    )
    if explicit_targets:
        wanted = {value.lower() for value in explicit_targets}
        missing = sorted(wanted - eligible_baseline)
        if missing:
            raise ValueError(f"Targets absent from all published partitions: {', '.join(missing)}")
        candidate_targets = sorted(wanted)
    else:
        candidate_targets = sorted(eligible_baseline)

    target_frame = targets[targets["target"].isin(candidate_targets)].copy()
    active_ids = _active_ids_by_target(target_frame, binding)
    from armado_datasets_modified import prepare_actives_df

    prepared_actives = {
        target: prepare_actives_df(ids, smiles)
        for target, ids in active_ids.items()
    }
    kwargs_base = {key: value for key, value in split_kwargs(seeds[0]).items() if key != "random_state"}
    kwargs_base["butina_cutoff"] = float(cutoff)
    census_rows = baseline[baseline["target"].isin(candidate_targets)].to_dict("records")

    for protocol in (PROTOCOL_FCFP4, PROTOCOL_BM):
        splitter = splitter_for_protocol(protocol)
        for target in candidate_targets:
            if not quiet:
                print(f"[prepare] {protocol}: {target}", flush=True)
            for seed in seeds:
                path = _split_path(output, protocol, seed, target)
                if resume and path.is_file():
                    split = _load_split(path)
                else:
                    split = splitter(
                        activos=active_ids[target],
                        smiles_df=smiles,
                        prepared_actives_df=prepared_actives[target],
                        random_state=seed,
                        **kwargs_base,
                    )
                    _serialize_split(path, target, seed, protocol, split)
                diagnostics = split.diagnostics
                census_rows.append(
                    {
                        "target": target,
                        "random_state": int(seed),
                        "protocol": protocol,
                        "n_active_total_raw": int(len(active_ids[target])),
                        "n_representatives": int(diagnostics.get("n_representatives", 0)),
                        "n_known": int(len(split.known_ids)),
                        "n_evaluation": int(len(split.test_active_ids)),
                        "status": str(diagnostics.get("status", "")),
                    }
                )

    family_by_target = _family_map()
    census = pd.DataFrame(census_rows)
    census["protein_family"] = census["target"].map(family_by_target)
    census["eligible"] = (
        (census["n_known"] >= int(min_known)) & (census["n_evaluation"] > 0)
    )
    census = census.sort_values(["protein_family", "target", "protocol", "random_state"])
    output.mkdir(parents=True, exist_ok=True)
    census.to_csv(output / "split_census.csv", index=False)

    target_summary = (
        census.groupby(["target", "protein_family"], as_index=False)
        .agg(
            n_protocol_partitions=("eligible", "size"),
            n_eligible_protocol_partitions=("eligible", "sum"),
            minimum_known=("n_known", "min"),
            minimum_evaluation=("n_evaluation", "min"),
            minimum_representatives=("n_representatives", "min"),
        )
    )
    expected = len(PROTOCOLS) * len(seeds)
    target_summary["eligible_all_protocols_and_partitions"] = (
        (target_summary["n_protocol_partitions"] == expected)
        & (target_summary["n_eligible_protocol_partitions"] == expected)
    )
    target_summary["selected"] = False

    if explicit_targets:
        chosen = set(candidate_targets)
        invalid = target_summary.loc[
            target_summary["target"].isin(chosen)
            & ~target_summary["eligible_all_protocols_and_partitions"],
            "target",
        ].tolist()
        if invalid:
            raise ValueError(
                "Explicit targets fail the minimum-data criterion under at least one protocol: "
                + ", ".join(invalid)
            )
    else:
        chosen: set[str] = set()
        eligible = target_summary[target_summary["eligible_all_protocols_and_partitions"]].copy()
        for _, family_frame in eligible.groupby("protein_family", sort=True):
            ranked = family_frame.sort_values(
                ["minimum_known", "minimum_evaluation", "minimum_representatives", "target"],
                ascending=[False, False, False, True],
            )
            chosen.update(ranked.head(int(targets_per_family))["target"].tolist())
    target_summary.loc[target_summary["target"].isin(chosen), "selected"] = True
    target_summary = target_summary.sort_values(
        ["protein_family", "selected", "minimum_known", "target"],
        ascending=[True, False, False, True],
    )
    target_summary.to_csv(output / "target_availability_and_selection.csv", index=False)
    panel = target_summary[target_summary["selected"]].copy()
    panel.to_csv(output / "selected_target_panel.csv", index=False)
    if panel.empty:
        raise ValueError("No targets satisfy the requested minimum-data criterion")
    return census, panel


def _cell_path(output: Path, protocol: str, seed: int, method: str, target: str) -> Path:
    return output / "cells" / protocol / f"seed_{seed}" / method / f"{target}.csv"


def run_new_protocols(
    cfg: dict,
    output: Path,
    panel: pd.DataFrame,
    seeds: list[int],
    methods: list[str],
    *,
    min_known: int,
    resume: bool,
    force: bool,
    quiet: bool,
    device: str,
) -> None:
    from armado_datasets_modified import run_ef_eval_target_vs_neighbors

    target_names = panel["target"].astype(str).tolist()
    context = load_context(cfg, methods, target_names, with_neighbors=False)
    family_by_target = _family_map()

    for protocol in (PROTOCOL_FCFP4, PROTOCOL_BM):
        for seed in seeds:
            for target in target_names:
                split = _load_split(_split_path(output, protocol, seed, target))
                if len(split.known_ids) < min_known:
                    raise ValueError(f"Prepared split is no longer eligible: {protocol}/{seed}/{target}")
                target_frame = context.targets[context.targets["target"].astype(str) == target]

                def fixed_split(**_kwargs):
                    return split

                for method in methods:
                    cell = _cell_path(output, protocol, seed, method, target)
                    if resume and cell.is_file() and not force:
                        continue
                    if cell.exists() and not force:
                        raise FileExistsError(f"Cell already exists: {cell}; use --resume or --force")
                    if not quiet:
                        print(f"[evaluate] {protocol} seed={seed} target={target} method={method}", flush=True)
                    values = run_ef_eval_target_vs_neighbors(
                        targets_dude=target_frame,
                        binding_data=context.binding,
                        smiles=context.smiles,
                        store=context.store,
                        neighbor_ranked_ids_by_target={},
                        rep_eval=context.representations[method],
                        metric_eval=METHODS[method],
                        method_label=method,
                        device_eval=device,
                        percentiles=RAW_PERCENTILES,
                        min_known_for_eval=min_known,
                        split_kwargs={"random_state": int(seed)},
                        verbose=not quiet,
                        use_target_seeds=True,
                        use_neighbor_seeds=False,
                        split_function=fixed_split,
                        ef_mode="both",
                    )
                    long_frame = values[0].copy()
                    if long_frame.empty:
                        raise RuntimeError(f"No result for {protocol}/{seed}/{target}/{method}")
                    long_frame["protocol"] = protocol
                    long_frame["random_state"] = int(seed)
                    long_frame["method_label"] = method
                    long_frame["protein_family"] = family_by_target.get(target, "")
                    cell.parent.mkdir(parents=True, exist_ok=True)
                    long_frame.to_csv(cell, index=False)


def _load_baseline(
    baseline_results: Path,
    targets: list[str],
    seeds: list[int],
    methods: list[str],
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = baseline_results / f"seed_{seed}" / "df_long_target_all_methods.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Baseline result missing: {path}")
        frame = pd.read_csv(path)
        frame = frame[
            frame["target"].astype(str).isin(targets)
            & frame["method_label"].astype(str).isin(methods)
        ].copy()
        frame["protocol"] = PROTOCOL_ECFP4
        frame["random_state"] = int(seed)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    expected = len(targets) * len(seeds) * len(methods) * len(RAW_PERCENTILES)
    if len(result) != expected:
        raise ValueError(f"Incomplete baseline subset: expected {expected} rows, found {len(result)}")
    result["protein_family"] = result["target"].map(_family_map())
    return result


def _load_new_cells(output: Path, targets: list[str], seeds: list[int], methods: list[str]) -> pd.DataFrame:
    frames = []
    missing = []
    for protocol in (PROTOCOL_FCFP4, PROTOCOL_BM):
        for seed in seeds:
            for target in targets:
                for method in methods:
                    path = _cell_path(output, protocol, seed, method, target)
                    if not path.is_file():
                        missing.append(str(path))
                    else:
                        frames.append(pd.read_csv(path))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} evaluation cells; first: {missing[0]}")
    return pd.concat(frames, ignore_index=True)


def summarize_and_plot(
    output: Path,
    baseline_results: Path,
    panel: pd.DataFrame,
    seeds: list[int],
    methods: list[str],
) -> None:
    import matplotlib.pyplot as plt

    targets = panel["target"].astype(str).tolist()
    baseline = _load_baseline(baseline_results, targets, seeds, methods)
    combined = pd.concat([baseline, _load_new_cells(output, targets, seeds, methods)], ignore_index=True)
    combined.to_csv(output / "target_partition_metrics.csv", index=False)

    target_medians = (
        combined.groupby(
            ["protocol", "method_label", "target", "protein_family", "percentile"],
            as_index=False,
        )
        .agg(
            EF_cumulative=("EF_cumulative", "median"),
            EF_band=("EF_band", "median"),
            n_partitions=("random_state", "nunique"),
        )
    )
    target_medians.to_csv(output / "target_medians_across_partitions.csv", index=False)
    family_medians = (
        target_medians.groupby(
            ["protocol", "method_label", "protein_family", "percentile"], as_index=False
        )
        .agg(
            EF_family_cumulative=("EF_cumulative", "median"),
            EF_family_band=("EF_band", "median"),
            n_targets=("target", "nunique"),
        )
    )
    family_medians.to_csv(output / "family_medians.csv", index=False)
    summary = (
        family_medians.groupby(["protocol", "method_label", "percentile"], as_index=False)
        .agg(
            EF_category_balanced_cumulative=("EF_family_cumulative", "median"),
            EF_category_q25_cumulative=("EF_family_cumulative", lambda value: value.quantile(0.25)),
            EF_category_q75_cumulative=("EF_family_cumulative", lambda value: value.quantile(0.75)),
            EF_category_balanced_band=("EF_family_band", "median"),
            n_categories=("protein_family", "nunique"),
        )
    )
    summary["rank_cumulative"] = summary.groupby(["protocol", "percentile"])[
        "EF_category_balanced_cumulative"
    ].rank(method="min", ascending=False)
    summary.to_csv(output / "sensitivity_summary.csv", index=False)

    rank_rows = []
    for percentile in sorted(summary["percentile"].unique(), reverse=True):
        pivot = summary[summary["percentile"] == percentile].pivot(
            index="method_label", columns="protocol", values="rank_cumulative"
        )
        if PROTOCOL_ECFP4 not in pivot:
            continue
        for protocol in (PROTOCOL_FCFP4, PROTOCOL_BM):
            paired = pivot[[PROTOCOL_ECFP4, protocol]].dropna()
            correlation = paired[PROTOCOL_ECFP4].corr(paired[protocol], method="spearman")
            rank_rows.append(
                {
                    "percentile": float(percentile),
                    "protocol": protocol,
                    "spearman_rank_correlation_vs_ecfp4_preprocessing": float(correlation),
                    "n_methods": int(len(paired)),
                }
            )
    pd.DataFrame(rank_rows).to_csv(output / "method_rank_stability.csv", index=False)

    plot_summary(summary, output, methods)


def plot_summary(summary: pd.DataFrame, output: Path, methods=None) -> None:
    """Render the original sensitivity panel independently of molecular searches."""
    import matplotlib.pyplot as plt

    methods = list(methods or DEFAULT_METHODS)
    plot = summary[np.isclose(summary["percentile"], 99.5)].copy()
    matrix = plot.pivot(
        index="protocol", columns="method_label", values="EF_category_balanced_cumulative"
    ).reindex(index=list(PROTOCOLS), columns=methods)
    figure, axis = plt.subplots(figsize=(9.2, 3.8))
    image = axis.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    axis.set_xticks(range(len(methods)), [PRETTY_METHODS[value] for value in methods], rotation=25, ha="right")
    axis.set_yticks(range(len(PROTOCOLS)), [
        "ECFP4/Butina 0.8\n(published)", "FCFP4/Butina 0.8", "Bemis-Murcko\nrepresentatives"
    ])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            if pd.notna(value):
                axis.text(column, row, f"{value:.2f}", ha="center", va="center", color="white")
    axis.set_title("Active-set preprocessing sensitivity at the top 0.5%")
    axis.set_xlabel("Retrieval representation")
    axis.set_ylabel("Active-set preprocessing")
    figure.colorbar(image, ax=axis, label="Category-balanced cumulative EF")
    figure.tight_layout()
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / f"active_preprocessing_sensitivity_EF0.5.{extension}", dpi=300)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    output = args.output_dir.expanduser().resolve()
    baseline_results = args.baseline_results_dir.expanduser().resolve()
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    seeds = parse_csv_arg(args.seeds, int) or list(SEEDS)
    methods = parse_csv_arg(args.methods) or list(DEFAULT_METHODS)
    unknown_methods = sorted(set(methods) - set(METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown methods: {', '.join(unknown_methods)}")
    explicit_targets = parse_csv_arg(args.targets)
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    output.mkdir(parents=True, exist_ok=True)
    _, panel = prepare_splits_and_panel(
        cfg,
        output,
        benchmark_dir,
        seeds,
        targets_per_family=args.targets_per_family,
        min_known=args.min_known,
        cutoff=args.butina_cutoff,
        explicit_targets=explicit_targets,
        resume=args.resume,
        quiet=args.quiet,
    )
    print("\nSelected target panel:")
    print(panel[["target", "protein_family", "minimum_known", "minimum_evaluation"]].to_string(index=False))
    if args.prepare_only:
        print(f"\nPreparation complete. Inspect: {output / 'selected_target_panel.csv'}")
        return 0

    run_new_protocols(
        cfg,
        output,
        panel,
        seeds,
        methods,
        min_known=args.min_known,
        resume=args.resume,
        force=args.force,
        quiet=args.quiet,
        device=device,
    )
    summarize_and_plot(output, baseline_results, panel, seeds, methods)
    _write_json(
        output / "run_metadata.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "rdkit": rdkit.__version__,
            "protocols": list(PROTOCOLS),
            "methods": methods,
            "seeds": seeds,
            "targets": panel["target"].astype(str).tolist(),
            "selection_rule": (
                "Within each protein family, rank targets eligible in every protocol/partition "
                "by minimum known, evaluation, and representative counts; select the first N."
            ),
            "min_known": int(args.min_known),
            "butina_cutoff": float(args.butina_cutoff),
            "device": device,
            "baseline_results_dir": str(baseline_results),
            "benchmark_dir": str(benchmark_dir),
        },
    )
    print(f"\nComplete results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
