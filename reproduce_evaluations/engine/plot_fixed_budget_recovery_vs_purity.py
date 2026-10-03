#!/usr/bin/env python3
"""Plot fixed-budget recovery versus purity for the best method combinations.

The input is ``method_combination_summary.csv`` produced by
``10_run_fixed_budget_complementarity.py``. One publication-style scatter
plot is generated per fusion policy. Strategies are ranked without using any
additional molecular calculations: descending category-balanced recall,
then descending category-balanced precision, then combination name.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from ligq2_evaluation.constants import PRETTY_METHODS
from ligq2_evaluation.fixed_budget import POLICIES


REQUIRED_COLUMNS = {
    "policy",
    "method_combination",
    "n_methods",
    "recall_at_n",
    "precision_at_n",
    "ef_at_n",
    "n_targets",
}

POLICY_TITLES = {
    "best": "Best-rank fusion",
    "mean": "Mean-rank fusion",
    "balanced": "Balanced allocation",
}

COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
MARKERS = ("o", "s", "^", "D")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_root = script_dir / "reproduction_output/representation_benchmark/fixed_budget_complementarity"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=default_root / "method_combination_summary.csv",
        help="Fixed-budget method-combination summary CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_root / "recovery_vs_purity",
        help="Directory for figures, selected-strategy table, and metadata",
    )
    parser.add_argument(
        "--policies",
        default="all",
        help="Comma-separated policies (best,mean,balanced) or 'all'",
    )
    parser.add_argument("--top-n", type=int, default=4, help="Strategies shown per policy")
    parser.add_argument("--dpi", type=int, default=600, help="PNG resolution")
    return parser.parse_args()


def parse_policies(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(POLICIES)
    policies = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"Unknown fusion policies: {unknown}")
    if not policies or len(policies) != len(set(policies)):
        raise ValueError("Choose one or more distinct fusion policies")
    return policies


def validate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required summary columns: {missing}")
    result = frame.copy()
    if result.duplicated(["policy", "method_combination"]).any():
        raise ValueError("Duplicate policy/method-combination rows in summary")
    if not set(result["policy"]).issubset(POLICIES):
        raise ValueError("Summary contains an unknown fusion policy")
    for column in ("recall_at_n", "precision_at_n", "ef_at_n"):
        result[column] = pd.to_numeric(result[column], errors="raise")
        if not np.isfinite(result[column]).all():
            raise ValueError(f"Column {column} contains non-finite values")
    for column in ("recall_at_n", "precision_at_n"):
        if not result[column].between(0.0, 1.0).all():
            raise ValueError(f"Column {column} must contain fractions between zero and one")
    result["n_methods"] = pd.to_numeric(result["n_methods"], errors="raise").astype(int)
    if (result["n_methods"] < 1).any():
        raise ValueError("Every strategy must contain at least one method")
    return result


def select_top_combinations(frame: pd.DataFrame, policy: str, top_n: int) -> pd.DataFrame:
    """Return a deterministic recovery-first ranking for one fusion policy."""
    if top_n < 1:
        raise ValueError("--top-n must be positive")
    subset = frame.loc[frame["policy"] == policy].copy()
    if len(subset) < top_n:
        raise ValueError(f"Policy {policy!r} has only {len(subset)} strategies, not {top_n}")
    subset = subset.sort_values(
        ["recall_at_n", "precision_at_n", "method_combination"],
        ascending=[False, False, True],
        kind="stable",
    ).head(top_n)
    subset.insert(0, "plot_rank", range(1, len(subset) + 1))
    return subset.reset_index(drop=True)


def pretty_combination(value: str) -> str:
    labels = []
    for raw_method in value.split(" + "):
        method = raw_method.strip()
        if method.endswith("__target_seeds"):
            method = method[: -len("__target_seeds")]
        label = PRETTY_METHODS.get(method, method)
        label = label.replace("ECFP4 (1024 bits)", "ECFP4")
        label = label.replace("FCFP4 (1024 bits)", "FCFP4")
        label = label.replace("Topological Torsion", "TT")
        labels.append(label)
    return " + ".join(labels)


def plot_policy(rows: pd.DataFrame, policy: str, output_dir: Path, dpi: int) -> list[Path]:
    plot_rows = rows.copy()
    plot_rows["recovery_percent"] = 100.0 * plot_rows["recall_at_n"]
    plot_rows["purity_percent"] = 100.0 * plot_rows["precision_at_n"]
    plot_rows["label"] = plot_rows["method_combination"].map(pretty_combination)

    with plt.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "legend.fontsize": 9.2,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig, axis = plt.subplots(figsize=(7.0, 5.0), dpi=dpi)
        handles = []
        for position, row in plot_rows.reset_index(drop=True).iterrows():
            color = COLORS[position % len(COLORS)]
            marker = MARKERS[position % len(MARKERS)]
            axis.scatter(
                row["recovery_percent"], row["purity_percent"],
                s=105, marker=marker, facecolor=color, edgecolor="black",
                linewidth=0.85, alpha=0.96, zorder=4,
            )
            handles.append(Line2D(
                [0], [0], marker=marker, linestyle="None", label=row["label"],
                markerfacecolor=color, markeredgecolor="black",
                markeredgewidth=0.85, markersize=8.5,
            ))

        axis.set_title(POLICY_TITLES[policy], loc="left", pad=10)
        axis.set_xlabel("Category-balanced median recall@N (%)", labelpad=8)
        axis.set_ylabel("Category-balanced median precision@N (%)", labelpad=8)

        x = plot_rows["recovery_percent"].to_numpy(dtype=float)
        y = plot_rows["purity_percent"].to_numpy(dtype=float)
        x_span = float(np.ptp(x))
        y_span = float(np.ptp(y))
        x_margin = max(x_span * 0.30, 0.9)
        y_margin = max(y_span * 0.45, 1.1)
        axis.set_xlim(float(x.min()) - x_margin, float(x.max()) + x_margin)
        axis.set_ylim(float(y.min()) - y_margin, float(y.max()) + y_margin)
        axis.xaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=6))
        axis.grid(True, which="major", color="0.7", linestyle="-", linewidth=0.45, alpha=0.25)
        axis.set_axisbelow(True)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="both", which="major", direction="out", length=4.2, width=0.9, pad=4)

        legend = axis.legend(
            handles=handles, loc="lower right", frameon=True, borderpad=0.7,
            labelspacing=0.65, handletextpad=0.65, borderaxespad=0.65,
            handlelength=1.2,
        )
        legend.get_frame().set_edgecolor("0.75")
        legend.get_frame().set_linewidth(0.85)
        legend.get_frame().set_alpha(0.96)
        legend.get_frame().set_facecolor("white")
        fig.tight_layout(pad=0.7)

        stem = output_dir / f"fixed_budget_recovery_vs_purity_{policy}"
        paths = [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]
        fig.savefig(paths[0], bbox_inches="tight", pad_inches=0.05, dpi=dpi, facecolor="white")
        fig.savefig(paths[1], bbox_inches="tight", pad_inches=0.05, facecolor="white")
        fig.savefig(paths[2], bbox_inches="tight", pad_inches=0.05, facecolor="white")
        plt.close(fig)
    return paths


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Fixed-budget summary not found: {args.input}")
    policies = parse_policies(args.policies)
    frame = validate_summary(pd.read_csv(args.input))
    missing_policies = [policy for policy in policies if policy not in set(frame["policy"])]
    if missing_policies:
        raise ValueError(f"Requested policies are absent from the summary: {missing_policies}")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = []
    generated = []
    for policy in policies:
        rows = select_top_combinations(frame, policy, args.top_n)
        selected.append(rows)
        generated.extend(plot_policy(rows, policy, output_dir, args.dpi))

    selected_frame = pd.concat(selected, ignore_index=True)
    selected_path = output_dir / "top_fixed_budget_combinations.csv"
    selected_frame.to_csv(selected_path, index=False)
    metadata_path = output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input.expanduser().resolve()),
        "policies": policies,
        "top_n": args.top_n,
        "ranking_rule": "recall_at_n descending, precision_at_n descending, method_combination ascending",
        "x_axis": "category-balanced median recall@N (%) = 100 * recall_at_n",
        "y_axis": "category-balanced median precision@N (%) = 100 * precision_at_n",
        "aggregation": "inherited from method_combination_summary.csv: median partitions within target, targets within category, then categories",
        "generated_figures": [str(path) for path in generated],
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Selected combinations: {selected_path}")
    for path in generated:
        print(f"Figure: {path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
