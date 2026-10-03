#!/usr/bin/env python3
"""Plot fixed K, Full domain, and identity-plus-ligand rescue EF curves.

The adaptive K choices are read from the completed identity/ligand rescue
grid and joined to the retained K=2..15 results at every plotted percentile.
No molecular search or reranking is performed.
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
from matplotlib.patches import Patch

from ligq2_evaluation.constants import FAMILIES, FAMILY_LABELS, FAMILY_ORDER, SEEDS
from ligq2_evaluation.runtime import prepare_output
from ligq2_evaluation.publication_style import readable_labels


PLOT_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0)
DISPLAY_ORDER = ("fixed_k_5", "fixed_k_15", "full_domain", "identity_ligand_rescue")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=script_dir / "reproduction_output/neighbor_k_complete",
    )
    parser.add_argument(
        "--rescue-dir",
        type=Path,
        default=script_dir / "reproduction_output/identity_ligand_rescue",
    )
    parser.add_argument(
        "--full-domain-input",
        type=Path,
        default=(
            script_dir / "reproduction_output/full_domain_benchmark"
            / "neighbor_vs_full_domain_all_seed_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "reproduction_output/identity_ligand_rescue_comparison",
    )
    parser.add_argument("--identity-floor", type=float, default=55.0)
    parser.add_argument("--ligand-budget", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def family_mapping() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for family in FAMILY_ORDER:
        label = FAMILY_LABELS.get(family, family)
        for targets in FAMILIES[family].values():
            for target in targets:
                if target in mapping and mapping[target] != label:
                    raise ValueError(f"Target {target!r} belongs to multiple families")
                mapping[target] = label
    return mapping


def load_sweep_metrics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "target", "seed", "neighbor_count_sweep", "percentile", "EF_cumulative"
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Complete K sweep is missing columns: {missing}")
    frame["target"] = frame["target"].astype(str).str.lower()
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["neighbor_count_sweep"] = pd.to_numeric(
        frame["neighbor_count_sweep"], errors="raise"
    ).astype(int)
    frame = frame[frame["percentile"].astype(float).isin(PLOT_PERCENTILES)].copy()
    if frame.duplicated(
        ["target", "seed", "neighbor_count_sweep", "percentile"]
    ).any():
        raise ValueError("Duplicate target/partition/K/percentile sweep rows")
    return frame


def target_medians_from_partition_rows(
    rows: pd.DataFrame,
    *,
    strategy: str,
    policy: str,
) -> pd.DataFrame:
    if rows.duplicated(["target", "seed", "percentile"]).any():
        raise ValueError(f"Duplicate partition rows for {strategy}")
    result = (
        rows.groupby(["target", "percentile"], as_index=False)
        .agg(
            target_median_EF_cumulative=("EF_cumulative", "median"),
            n_partitions=("seed", "nunique"),
        )
    )
    result["strategy"] = strategy
    result["policy"] = policy
    result["protein_family"] = result["target"].map(family_mapping())
    if result["protein_family"].isna().any():
        missing = sorted(result.loc[result["protein_family"].isna(), "target"].unique())
        raise ValueError(f"Targets lack family assignments: {missing}")
    return result[[
        "strategy", "policy", "target", "protein_family", "percentile",
        "target_median_EF_cumulative", "n_partitions",
    ]]


def load_fixed_target_rows(metrics: pd.DataFrame, k: int) -> pd.DataFrame:
    selected = metrics[metrics["neighbor_count_sweep"].eq(k)].copy()
    return target_medians_from_partition_rows(
        selected,
        strategy=f"fixed_k_{k}",
        policy=f"K={k}",
    )


def load_rescue_target_rows(
    choices_path: Path,
    metrics: pd.DataFrame,
    *,
    identity_floor: float,
    ligand_budget: int,
) -> pd.DataFrame:
    choices = pd.read_csv(choices_path)
    required = {
        "target", "seed", "identity_floor", "ligand_budget", "selected_k"
    }
    missing = sorted(required - set(choices.columns))
    if missing:
        raise ValueError(f"Rescue choices are missing columns: {missing}")
    choices["target"] = choices["target"].astype(str).str.lower()
    choices["seed"] = pd.to_numeric(choices["seed"], errors="raise").astype(int)
    choices["selected_k"] = pd.to_numeric(
        choices["selected_k"], errors="raise"
    ).astype(int)
    selected = choices[
        np.isclose(choices["identity_floor"].astype(float), identity_floor)
        & choices["ligand_budget"].astype(int).eq(ligand_budget)
    ][["target", "seed", "selected_k"]].copy()
    if selected.empty:
        raise ValueError(
            f"No rescue choices for identity={identity_floor:g}, budget={ligand_budget}"
        )
    if selected.duplicated(["target", "seed"]).any():
        raise ValueError("Duplicate target/partition rescue choices")
    merged = selected.merge(
        metrics,
        left_on=["target", "seed", "selected_k"],
        right_on=["target", "seed", "neighbor_count_sweep"],
        how="left",
        validate="one_to_many",
    )
    expected = len(selected) * len(PLOT_PERCENTILES)
    if len(merged) != expected or merged["EF_cumulative"].isna().any():
        raise ValueError("Rescue choices lack a complete percentile grid")
    policy = f"Identity ≥{identity_floor:g}% + ligand rescue ≥{ligand_budget}"
    return target_medians_from_partition_rows(
        merged,
        strategy="identity_ligand_rescue",
        policy=policy,
    )


def load_full_domain_target_rows(path: Path, targets: set[str]) -> pd.DataFrame:
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
    return target_medians_from_partition_rows(
        selected,
        strategy="full_domain",
        policy="Full domain",
    )


def validate_strategy_grid(target_rows: pd.DataFrame) -> None:
    if target_rows.duplicated(["strategy", "target", "percentile"]).any():
        raise ValueError("Duplicate strategy/target/percentile rows")
    expected_targets: set[str] | None = None
    for strategy, rows in target_rows.groupby("strategy"):
        observed_targets = set(rows["target"])
        if expected_targets is None:
            expected_targets = observed_targets
        elif observed_targets != expected_targets:
            raise ValueError(f"Strategy {strategy} has a different target set")
        if not rows["n_partitions"].eq(len(SEEDS)).all():
            raise ValueError(f"Strategy {strategy} does not contain all five partitions")
        cells = rows.groupby("target")["percentile"].agg(set)
        incomplete = cells[cells.map(lambda values: values != set(PLOT_PERCENTILES))]
        if not incomplete.empty:
            raise ValueError(f"Strategy {strategy} has incomplete percentile curves")
    if set(target_rows["strategy"]) != set(DISPLAY_ORDER):
        raise ValueError("Unexpected comparison strategy set")


def aggregate_strategies(
    target_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    families = (
        target_rows.groupby(["strategy", "percentile", "protein_family"], as_index=False)
        .agg(
            family_median_EF_cumulative=("target_median_EF_cumulative", "median"),
            n_targets=("target", "nunique"),
        )
    )
    summary = (
        families.groupby(["strategy", "percentile"], as_index=False)
        .agg(
            category_balanced_median_EF_cumulative=(
                "family_median_EF_cumulative", "median"
            ),
            category_q25_EF_cumulative=(
                "family_median_EF_cumulative", lambda values: values.quantile(0.25)
            ),
            category_q75_EF_cumulative=(
                "family_median_EF_cumulative", lambda values: values.quantile(0.75)
            ),
            n_categories=("protein_family", "nunique"),
        )
    )
    return families, summary


@readable_labels
def plot_comparison(
    summary: pd.DataFrame,
    *,
    identity_floor: float,
    ligand_budget: int,
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    labels = {
        "fixed_k_5": "K=5",
        "fixed_k_15": "K=15",
        "full_domain": "Full domain",
        "identity_ligand_rescue": (
            f"Identity ≥{identity_floor:g}% + ligand rescue ≥{ligand_budget}"
        ),
    }
    colors = {
        "fixed_k_5": "#1f77b4",
        "fixed_k_15": "#9467bd",
        "full_domain": "#d62728",
        "identity_ligand_rescue": "#2ca02c",
    }
    linestyles = {
        "fixed_k_5": "-",
        "fixed_k_15": "-",
        "full_domain": "--",
        "identity_ligand_rescue": "-.",
    }
    figure, axis = plt.subplots(figsize=(10.8, 6.5))
    x = np.arange(len(PLOT_PERCENTILES), dtype=float)
    for strategy in DISPLAY_ORDER:
        rows = summary[summary["strategy"].eq(strategy)].set_index("percentile")
        rows = rows.loc[list(PLOT_PERCENTILES)]
        color = colors[strategy]
        axis.plot(
            x,
            rows["category_balanced_median_EF_cumulative"].to_numpy(float),
            marker="o",
            markersize=5.5,
            linewidth=2.4,
            linestyle=linestyles[strategy],
            color=color,
            label=labels[strategy],
        )
        axis.fill_between(
            x,
            rows["category_q25_EF_cumulative"].to_numpy(float),
            rows["category_q75_EF_cumulative"].to_numpy(float),
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    axis.set_xticks(x, [f"{value:g}" for value in PLOT_PERCENTILES])
    axis.set_xlabel("Percentile threshold")
    axis.set_ylabel("Category-balanced median\ncumulative enrichment factor")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.14, edgecolor="none"))
    legend_labels.append("Category IQR")
    axis.legend(handles, legend_labels, title="Seed source", frameon=False)
    figure.tight_layout()
    paths = []
    stem = "fixed_k_domain_and_identity_ligand_rescue_cumulative_EF_publication"
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)
    sweep_path = args.sweep_dir.expanduser().resolve() / "all_seeds_valid_rows.csv"
    metrics = load_sweep_metrics(sweep_path)
    fixed_k5 = load_fixed_target_rows(metrics, 5)
    fixed_k15 = load_fixed_target_rows(metrics, 15)
    rescue = load_rescue_target_rows(
        args.rescue_dir.expanduser().resolve() / "identity_ligand_rescue_choices.csv",
        metrics,
        identity_floor=args.identity_floor,
        ligand_budget=args.ligand_budget,
    )
    matched_targets = set(fixed_k5["target"])
    full_domain = load_full_domain_target_rows(
        args.full_domain_input.expanduser().resolve(), matched_targets
    )
    target_rows = pd.concat(
        [fixed_k5, fixed_k15, full_domain, rescue], ignore_index=True
    )
    validate_strategy_grid(target_rows)
    families, summary = aggregate_strategies(target_rows)
    display_order = {strategy: index for index, strategy in enumerate(DISPLAY_ORDER)}
    percentile_order = {value: index for index, value in enumerate(PLOT_PERCENTILES)}
    summary["display_order"] = summary["strategy"].map(display_order)
    summary["percentile_order"] = summary["percentile"].map(percentile_order)
    summary = summary.sort_values(["display_order", "percentile_order"])
    figure_paths = plot_comparison(
        summary,
        identity_floor=args.identity_floor,
        ligand_budget=args.ligand_budget,
        output_dir=output_dir,
        dpi=args.dpi,
    )
    target_rows.to_csv(output_dir / "strategy_target_medians.csv", index=False)
    families.to_csv(output_dir / "strategy_family_medians.csv", index=False)
    summary.to_csv(output_dir / "strategy_category_balanced_summary.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_input": str(sweep_path),
        "rescue_input": str(args.rescue_dir.expanduser().resolve()),
        "full_domain_input": str(args.full_domain_input.expanduser().resolve()),
        "identity_floor_percent": args.identity_floor,
        "ligand_candidate_threshold": args.ligand_budget,
        "plotted_percentiles": list(PLOT_PERCENTILES),
        "strategies": list(DISPLAY_ORDER),
        "n_targets": int(target_rows["target"].nunique()),
        "partition_seeds": list(SEEDS),
        "aggregation_order": [
            "median across five partitions within target",
            "median across targets within protein category",
            "median and IQR across protein-category medians",
        ],
        "adaptive_policy_selection_is_post_hoc": True,
        "figures": [str(path) for path in figure_paths],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    endpoint = summary[np.isclose(summary["percentile"], 99.5)]
    print(endpoint[[
        "strategy", "category_balanced_median_EF_cumulative",
        "category_q25_EF_cumulative", "category_q75_EF_cumulative",
    ]].to_string(index=False))
    print(f"Comparison complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
