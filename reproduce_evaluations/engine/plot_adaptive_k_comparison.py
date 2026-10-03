#!/usr/bin/env python3
"""Compare fixed K, Full domain, and best adaptive-policy EF curves.

The best identity and ligand-evidence policies are selected post hoc by the
largest category-balanced median cumulative EF at the requested percentile
(EF0.5% by default), then their complete publication-percentile curves are
plotted. This script only aggregates existing results; it performs no
molecular search.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ligq2_evaluation.constants import FAMILIES, FAMILY_LABELS, FAMILY_ORDER, SEEDS
from ligq2_evaluation.runtime import prepare_output


FIXED_POLICIES = ("fixed_k_2", "fixed_k_5", "fixed_k_15")
PLOT_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0)
DISPLAY_ORDER = (
    "fixed_k_2", "fixed_k_5", "fixed_k_15", "full_domain",
    "best_ligand_budget", "best_identity_floor",
)
def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adaptive-dir", type=Path,
        default=script_dir / "reproduction_output/adaptive_k",
    )
    parser.add_argument(
        "--full-domain-input", type=Path,
        default=(
            script_dir / "reproduction_output/full_domain_benchmark"
            / "neighbor_vs_full_domain_all_seed_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=script_dir / "reproduction_output/adaptive_k_comparison",
    )
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def family_mapping() -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    order = []
    for family in FAMILY_ORDER:
        label = FAMILY_LABELS.get(family, family)
        order.append(label)
        for targets in FAMILIES[family].values():
            for target in targets:
                if target in mapping and mapping[target] != label:
                    raise ValueError(f"Target {target!r} belongs to multiple families")
                mapping[target] = label
    return mapping, order


def extract_policy_value(policy: str, policy_type: str) -> float:
    prefix = {
        "identity_floor": "identity_floor_",
        "ligand_budget": "ligand_budget_",
    }[policy_type]
    if not policy.startswith(prefix):
        raise ValueError(f"Unexpected {policy_type} policy name: {policy}")
    try:
        return float(policy.removeprefix(prefix))
    except ValueError as exc:
        raise ValueError(f"Cannot parse policy value from {policy}") from exc


def select_best_policy(
    summary: pd.DataFrame,
    *,
    policy_type: str,
    percentile: float,
) -> pd.Series:
    required = {
        "policy", "policy_type", "percentile",
        "category_balanced_median_EF_cumulative",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Adaptive summary is missing columns: {missing}")
    rows = summary[
        summary["policy_type"].eq(policy_type)
        & np.isclose(summary["percentile"].astype(float), percentile)
    ].copy()
    if rows.empty:
        raise ValueError(f"No {policy_type} policies found at percentile {percentile}")
    rows["policy_value"] = rows["policy"].map(
        lambda value: extract_policy_value(str(value), policy_type)
    )
    # More stringent identity and smaller ligand budget are deterministic,
    # parsimonious tie-breaks after maximizing the plotted endpoint.
    tie_ascending = policy_type == "ligand_budget"
    rows = rows.sort_values(
        ["category_balanced_median_EF_cumulative", "policy_value"],
        ascending=[False, tie_ascending],
        kind="stable",
    )
    return rows.iloc[0]


def load_adaptive_target_rows(
    adaptive_dir: Path,
    *,
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = adaptive_dir / "adaptive_category_balanced_summary.csv"
    targets_path = adaptive_dir / "adaptive_target_medians.csv"
    summary = pd.read_csv(summary_path)
    targets = pd.read_csv(targets_path)
    best_identity = select_best_policy(
        summary, policy_type="identity_floor", percentile=percentile
    )
    best_ligand = select_best_policy(
        summary, policy_type="ligand_budget", percentile=percentile
    )
    policy_map = {
        "fixed_k_2": "fixed_k_2",
        "fixed_k_5": "fixed_k_5",
        "fixed_k_15": "fixed_k_15",
        str(best_ligand["policy"]): "best_ligand_budget",
        str(best_identity["policy"]): "best_identity_floor",
    }
    selected = targets[
        targets["policy"].isin(policy_map)
        & targets["percentile"].astype(float).isin(PLOT_PERCENTILES)
    ].copy()
    selected["strategy"] = selected["policy"].map(policy_map)
    selected = selected.rename(
        columns={"median_partition_EF_cumulative": "target_median_EF_cumulative"}
    )
    keep = [
        "strategy", "policy", "policy_type", "target", "protein_family",
        "percentile", "target_median_EF_cumulative", "n_partitions",
    ]
    selected = selected[keep]
    selected_policies = pd.DataFrame([
        {
            "strategy": "best_ligand_budget",
            "selected_policy": best_ligand["policy"],
            "criterion": "cleaned candidate-ligand threshold before MaxMin",
            "selected_value": best_ligand["policy_value"],
            "selection_endpoint": "category-balanced median cumulative EF",
            "percentile": percentile,
            "selection_is_post_hoc": True,
        },
        {
            "strategy": "best_identity_floor",
            "selected_policy": best_identity["policy"],
            "criterion": "minimum BLAST percentage identity for the next neighbor",
            "selected_value": best_identity["policy_value"],
            "selection_endpoint": "category-balanced median cumulative EF",
            "percentile": percentile,
            "selection_is_post_hoc": True,
        },
    ])
    return selected, selected_policies


def load_full_domain_target_rows(
    path: Path,
    *,
    percentile: float,
    targets: set[str],
) -> pd.DataFrame:
    data = pd.read_csv(path)
    required = {"target", "seed", "strategy", "percentile", "EF_cumulative"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Full-domain results are missing columns: {missing}")
    data["target"] = data["target"].astype(str).str.lower()
    selected = data[
        data["strategy"].eq("Full domain")
        & data["percentile"].astype(float).isin(PLOT_PERCENTILES)
        & data["target"].isin(targets)
    ].copy()
    if selected.duplicated(["target", "seed", "percentile"]).any():
        raise ValueError("Duplicate Full domain target/partition/percentile rows")
    counts = selected.groupby(["target", "percentile"])["seed"].nunique()
    expected_cells = len(targets) * len(PLOT_PERCENTILES)
    if len(counts) != expected_cells or not counts.eq(len(SEEDS)).all():
        raise ValueError("Full domain lacks a complete target/partition/percentile grid")
    mapping, _ = family_mapping()
    result = (
        selected.groupby(["target", "percentile"], as_index=False)
        .agg(
            target_median_EF_cumulative=("EF_cumulative", "median"),
            n_partitions=("seed", "nunique"),
        )
    )
    result["strategy"] = "full_domain"
    result["policy"] = "Full domain"
    result["policy_type"] = "full_domain"
    result["protein_family"] = result["target"].map(mapping)
    if result["protein_family"].isna().any():
        raise ValueError("A Full domain target lacks a protein-family assignment")
    return result[[
        "strategy", "policy", "policy_type", "target", "protein_family",
        "percentile", "target_median_EF_cumulative", "n_partitions",
    ]]


def validate_strategy_grid(target_rows: pd.DataFrame) -> None:
    if target_rows.duplicated(["strategy", "target", "percentile"]).any():
        raise ValueError("Duplicate strategy/target/percentile rows")
    expected_targets = None
    for strategy, rows in target_rows.groupby("strategy"):
        observed = set(rows["target"])
        if expected_targets is None:
            expected_targets = observed
        elif observed != expected_targets:
            raise ValueError(f"Strategy {strategy} has a different target set")
        if not rows["n_partitions"].eq(len(SEEDS)).all():
            raise ValueError(f"Strategy {strategy} does not summarize all five partitions")
        cells = rows.groupby("target")["percentile"].agg(set)
        incomplete = cells[cells.map(lambda values: values != set(PLOT_PERCENTILES))]
        if not incomplete.empty:
            raise ValueError(f"Strategy {strategy} lacks plotted percentiles for some targets")
    observed_strategies = set(target_rows["strategy"])
    if observed_strategies != set(DISPLAY_ORDER):
        raise ValueError(
            f"Unexpected strategy set: {sorted(observed_strategies)}; "
            f"expected {list(DISPLAY_ORDER)}"
        )


def aggregate_strategies(
    target_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    family = (
        target_rows.groupby(["strategy", "percentile", "protein_family"], as_index=False)
        .agg(
            family_median_EF_cumulative=("target_median_EF_cumulative", "median"),
            n_targets=("target", "nunique"),
        )
    )
    summary = (
        family.groupby(["strategy", "percentile"], as_index=False)
        .agg(
            category_balanced_median_EF_cumulative=("family_median_EF_cumulative", "median"),
            category_q25_EF_cumulative=("family_median_EF_cumulative", lambda x: x.quantile(0.25)),
            category_q75_EF_cumulative=("family_median_EF_cumulative", lambda x: x.quantile(0.75)),
            category_min_EF_cumulative=("family_median_EF_cumulative", "min"),
            category_max_EF_cumulative=("family_median_EF_cumulative", "max"),
            n_categories=("protein_family", "nunique"),
        )
    )
    return family, summary


def display_labels(selected_policies: pd.DataFrame) -> dict[str, str]:
    values = selected_policies.set_index("strategy")["selected_value"]
    ligand = int(values["best_ligand_budget"])
    identity = values["best_identity_floor"]
    return {
        "fixed_k_2": "K=2",
        "fixed_k_5": "K=5",
        "fixed_k_15": "K=15",
        "full_domain": "Full domain",
        "best_ligand_budget": f"Best adaptive ligand\nbudget (≥{ligand})",
        "best_identity_floor": f"Best adaptive identity\nfloor (≥{identity:g}%)",
    }


def plot_comparison(
    family: pd.DataFrame,
    summary: pd.DataFrame,
    selected_policies: pd.DataFrame,
    *,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    labels = display_labels(selected_policies)
    figure, axis = plt.subplots(figsize=(10.8, 6.5))
    viridis = plt.get_cmap("viridis")
    palette = {
        "fixed_k_2": viridis(0.0),
        "fixed_k_5": viridis(0.42),
        "fixed_k_15": viridis(0.85),
        "full_domain": "#d62728",
        "best_ligand_budget": "#ff7f0e",
        "best_identity_floor": "#2ca02c",
    }
    linestyles = {
        "fixed_k_2": "-", "fixed_k_5": "-", "fixed_k_15": "-",
        "full_domain": "--", "best_ligand_budget": "-.",
        "best_identity_floor": ":",
    }
    x = np.arange(len(PLOT_PERCENTILES), dtype=float)
    for strategy in DISPLAY_ORDER:
        rows = summary[summary["strategy"].eq(strategy)].set_index("percentile")
        if not set(PLOT_PERCENTILES).issubset(rows.index):
            raise ValueError(f"Missing plotted percentiles for {strategy}")
        rows = rows.loc[list(PLOT_PERCENTILES)]
        color = palette[strategy]
        axis.plot(
            x, rows["category_balanced_median_EF_cumulative"].to_numpy(float),
            marker="o", markersize=5.5, linewidth=2.4,
            linestyle=linestyles[strategy], color=color, label=labels[strategy],
        )
        axis.fill_between(
            x, rows["category_q25_EF_cumulative"].to_numpy(float),
            rows["category_q75_EF_cumulative"].to_numpy(float),
            color=color, alpha=0.14, linewidth=0,
        )
    axis.set_xticks(x, [f"{value:g}" for value in PLOT_PERCENTILES])
    axis.set_xlabel("Percentile threshold")
    axis.set_ylabel("Category-balanced median cumulative enrichment factor")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axis.get_legend_handles_labels()
    from matplotlib.patches import Patch
    handles.append(Patch(facecolor="0.45", alpha=0.14, edgecolor="none"))
    legend_labels.append("Category IQR")
    axis.legend(handles, legend_labels, title="Seed source", frameon=False)
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"adaptive_vs_fixed_and_domain_cumulative_EF_publication.{suffix}"
        figure.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)
    adaptive_targets, selected_policies = load_adaptive_target_rows(
        args.adaptive_dir.expanduser().resolve(), percentile=args.percentile
    )
    matched_targets = set(adaptive_targets["target"])
    full_domain = load_full_domain_target_rows(
        args.full_domain_input.expanduser().resolve(), percentile=args.percentile,
        targets=matched_targets,
    )
    target_rows = pd.concat([adaptive_targets, full_domain], ignore_index=True)
    validate_strategy_grid(target_rows)
    family, summary = aggregate_strategies(target_rows)
    labels = display_labels(selected_policies)
    summary["display_label"] = summary["strategy"].map(labels)
    summary["display_order"] = summary["strategy"].map(
        {strategy: index for index, strategy in enumerate(DISPLAY_ORDER)}
    )
    summary["percentile_order"] = summary["percentile"].map(
        {value: index for index, value in enumerate(PLOT_PERCENTILES)}
    )
    summary = summary.sort_values(["display_order", "percentile_order"])
    figure_paths = plot_comparison(
        family, summary, selected_policies, output_dir=output_dir, dpi=args.dpi
    )

    selected_policies.to_csv(output_dir / "selected_adaptive_policies.csv", index=False)
    target_rows.to_csv(output_dir / "strategy_target_medians.csv", index=False)
    family.to_csv(output_dir / "strategy_family_medians.csv", index=False)
    summary.to_csv(output_dir / "strategy_category_balanced_summary.csv", index=False)
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "adaptive_dir": str(args.adaptive_dir.expanduser().resolve()),
        "full_domain_input": str(args.full_domain_input.expanduser().resolve()),
        "percentile": args.percentile,
        "selection_endpoint": "cumulative EF at top 0.5% screened",
        "plotted_percentiles": list(PLOT_PERCENTILES),
        "n_targets": int(target_rows["target"].nunique()),
        "partition_seeds": list(SEEDS),
        "aggregation_order": [
            "median across five partitions within target",
            "median across targets within protein category",
            "median and IQR across protein-category medians",
        ],
        "adaptive_selection": (
            "Post hoc maximum category-balanced median cumulative EF at the selected percentile; "
            "identity ties favor the stricter floor and ligand-budget ties favor the smaller budget."
        ),
        "selected_adaptive_policies": selected_policies.to_dict("records"),
        "figures": [str(path) for path in figure_paths],
    }, indent=2) + "\n", encoding="utf-8")
    print(selected_policies.to_string(index=False))
    endpoint_summary = summary[np.isclose(summary["percentile"], args.percentile)]
    print(endpoint_summary[[
        "display_label", "category_balanced_median_EF_cumulative",
        "category_q25_EF_cumulative", "category_q75_EF_cumulative",
    ]].to_string(index=False))
    print(f"Comparison complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
