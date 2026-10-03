#!/usr/bin/env python3
"""Compare the three retained adaptive neighbor policies with fixed K=5.

Reuses previously generated target medians and selected policy definitions.
No new policy tuning, molecular searches or ranking calculations are run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from ligq2_evaluation.constants import SEEDS
from plot_adaptive_k_comparison import PLOT_PERCENTILES, aggregate_strategies


DISPLAY_ORDER = (
    "fixed_k_5", "best_identity_floor", "best_ligand_budget", "identity_ligand_rescue",
)
TARGET_COLUMNS = (
    "target", "protein_family", "percentile", "target_median_EF_cumulative", "n_partitions",
)


def combine_target_rows(adaptive: pd.DataFrame, rescue: pd.DataFrame) -> pd.DataFrame:
    """Require matching baselines and complete, identically annotated grids."""
    for name, frame in (("adaptive", adaptive), ("rescue", rescue)):
        missing = {"strategy", "policy", *TARGET_COLUMNS} - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame.duplicated(["strategy", "target", "percentile"]).any():
            raise ValueError(f"Duplicate strategy/target/percentile rows in {name} input.")
    adaptive_k5 = adaptive.loc[adaptive.strategy.eq("fixed_k_5"), list(TARGET_COLUMNS)]
    rescue_k5 = rescue.loc[rescue.strategy.eq("fixed_k_5"), list(TARGET_COLUMNS)]
    def ordered(frame):
        return frame.sort_values(["target", "percentile"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered(adaptive_k5), ordered(rescue_k5))
    rows = pd.concat([
        adaptive.loc[adaptive.strategy.isin(DISPLAY_ORDER[:3]), ["strategy", "policy", *TARGET_COLUMNS]],
        rescue.loc[rescue.strategy.eq(DISPLAY_ORDER[3]), ["strategy", "policy", *TARGET_COLUMNS]],
    ], ignore_index=True)
    if rows.isna().any().any():
        raise ValueError("Missing values in the selected target rows.")
    if rows.duplicated(["strategy", "target", "percentile"]).any():
        raise ValueError("Duplicate strategy/target/percentile rows.")
    if set(rows.strategy) != set(DISPLAY_ORDER):
        raise ValueError("One or more required strategies are absent.")
    if not np.isfinite(rows.target_median_EF_cumulative).all():
        raise ValueError("Non-finite EF values.")
    baseline = ordered(adaptive_k5)[["target", "protein_family", "percentile", "n_partitions"]]
    if baseline.empty or not baseline.n_partitions.eq(len(SEEDS)).all():
        raise ValueError("Baseline lacks five-partition target summaries.")
    for strategy, frame in rows.groupby("strategy"):
        pd.testing.assert_frame_equal(
            ordered(frame)[baseline.columns], baseline,
            obj=f"Matched target/category/partition grid for {strategy}",
        )
        for percentiles in frame.groupby("target").percentile.agg(set):
            if percentiles != set(PLOT_PERCENTILES):
                raise ValueError(f"Incomplete percentile grid for {strategy}.")
    return rows


def plot_comparison(summary: pd.DataFrame, labels: dict, output: Path, dpi: int) -> None:
    colors = {
        "fixed_k_5": "#1f77b4", "best_identity_floor": "#9467bd",
        "best_ligand_budget": "#ff7f0e", "identity_ligand_rescue": "#2ca02c",
    }
    styles = dict(zip(DISPLAY_ORDER, ("-", ":", "--", "-.")))
    figure, axis = plt.subplots(figsize=(10.8, 6.5))
    x = np.arange(len(PLOT_PERCENTILES))
    for strategy in DISPLAY_ORDER:
        frame = summary.loc[summary.strategy.eq(strategy)].set_index("percentile").loc[list(PLOT_PERCENTILES)]
        axis.plot(
            x, frame.category_balanced_median_EF_cumulative.to_numpy(float),
            color=colors[strategy], linestyle=styles[strategy], marker="o",
            markersize=5.5, linewidth=2.4, label=labels[strategy],
        )
        axis.fill_between(
            x, frame.category_q25_EF_cumulative.to_numpy(float),
            frame.category_q75_EF_cumulative.to_numpy(float),
            color=colors[strategy], alpha=0.14, linewidth=0,
        )
    axis.set_xticks(x, [f"{p:g}" for p in PLOT_PERCENTILES])
    axis.set_xlabel("Percentile threshold")
    axis.set_ylabel("Category-balanced median\ncumulative enrichment factor")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    axis.spines[["top", "right"]].set_visible(False)
    handles, legend_labels = axis.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.14, edgecolor="none"))
    legend_labels.append("Category IQR")
    axis.legend(handles, legend_labels, title="Neighbor selection", frameon=False)
    figure.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"three_adaptive_vs_k5_cumulative_EF.{suffix}",
            dpi=dpi if suffix == "png" else None, bbox_inches="tight",
        )
    plt.close(figure)


def main() -> int:
    root = Path(__file__).resolve().parent / "reproduction_output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-comparison-dir", type=Path, default=root / "adaptive_k_comparison")
    parser.add_argument("--rescue-comparison-dir", type=Path, default=root / "identity_ligand_rescue_comparison")
    parser.add_argument("--output-dir", type=Path, default=root / "three_adaptive_vs_k5")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--font-size", type=float, default=16, help="Base font size in points (default: 16).")
    args = parser.parse_args()
    if args.font_size < 6:
        parser.error("--font-size must be at least 6 points.")
    adaptive = args.adaptive_comparison_dir.expanduser().resolve()
    rescue = args.rescue_comparison_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output in (adaptive, rescue):
        parser.error("Use a separate output directory to preserve the previous comparisons.")
    inputs = [
        adaptive / "strategy_target_medians.csv", rescue / "strategy_target_medians.csv",
        adaptive / "selected_adaptive_policies.csv", rescue / "run_metadata.json",
    ]
    rows = combine_target_rows(pd.read_csv(inputs[0]), pd.read_csv(inputs[1]))
    policies = pd.read_csv(inputs[2])
    if policies.strategy.duplicated().any():
        raise ValueError("Duplicate selected policy definitions.")
    values = policies.set_index("strategy").selected_value
    identity = float(values["best_identity_floor"])
    budget = int(values["best_ligand_budget"])
    rescue_meta = json.loads(inputs[3].read_text())
    rescue_identity = rescue_meta["identity_floor_percent"]
    rescue_budget = rescue_meta["ligand_candidate_threshold"]
    labels = {
        "fixed_k_5": "K=5",
        "best_identity_floor": f"Adaptive identity (≥{identity:g}%)",
        "best_ligand_budget": f"Adaptive ligand threshold (≥{budget})",
        "identity_ligand_rescue": f"Identity ≥{rescue_identity:g}% + ligand rescue ≥{rescue_budget}",
    }
    families, summary = aggregate_strategies(rows)
    summary["display_label"] = summary.strategy.map(labels)
    output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({
        "font.size": args.font_size, "axes.labelsize": args.font_size + 2,
        "xtick.labelsize": args.font_size, "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size, "legend.title_fontsize": args.font_size,
    }):
        plot_comparison(summary, labels, output, args.dpi)
    rows.to_csv(output / "strategy_target_medians.csv", index=False)
    families.to_csv(output / "strategy_family_medians.csv", index=False)
    summary.to_csv(output / "strategy_category_balanced_summary.csv", index=False)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        "strategies": labels, "n_targets": int(rows.target.nunique()),
        "partition_seeds": list(SEEDS), "percentiles": list(PLOT_PERCENTILES),
        "aggregation": "Median partitions within target; median targets within category; median and IQR across categories.",
        "policy_selection": "Reuse previously selected post hoc policies; no new optimization.",
        "ligand_threshold_definition": "Cleaned candidate ligands before MaxMin, not a new selected-seed budget.",
        "spread": "Category Q25-Q75, not confidence intervals.",
        "font_size_points": args.font_size,
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.loc[summary.percentile.eq(99.5), ["display_label", "category_balanced_median_EF_cumulative"]].to_string(index=False))
    print(f"Wrote comparison for {rows.target.nunique()} targets to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
