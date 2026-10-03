#!/usr/bin/env python3
"""Preview category-balanced EF figures with category IQR and range."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ligq2_evaluation.constants import FAMILIES, METHODS
from ligq2_evaluation.families import add_family_annotations


SEEDS = (42, 10, 27, 3, 8)
PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0, 90.0)
METHOD_ORDER = (
    "ap_rdkit",
    "chemberta_zinc_base_768",
    "maccs",
    "morgan_1024_r2",
    "morgan_feature_1024_r2",
    "rdkit_1024",
    "topological_torsion_rdkit_1024",
)
METHOD_LABELS = {
    "ap_rdkit": "Atom Pair",
    "chemberta_zinc_base_768": "ChemBERTa",
    "maccs": "MACCS",
    "morgan_1024_r2": "ECFP4 (1024 bits)",
    "morgan_feature_1024_r2": "FCFP4 (1024 bits)",
    "rdkit_1024": "RDKit Path",
    "topological_torsion_rdkit_1024": "Topological Torsion",
}
METHOD_COLORS = dict(zip(METHOD_ORDER, plt.get_cmap("tab10").colors[: len(METHOD_ORDER)]))


def parse_args() -> argparse.Namespace:
    script_root = Path(__file__).resolve().parent
    workspace = script_root.parent
    parser = argparse.ArgumentParser(
        description="Generate category-balanced EF previews with IQR and range."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=workspace / "resultados_EF_repeticiones_con_inactivos_10_90",
        help="Directory containing seed_<seed>/resultados_metodos_EF_por_familia.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace / "reviewer_preview_category_spread",
    )
    return parser.parse_args()


def load_category_summaries(input_dir: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = input_dir / f"seed_{seed}" / "df_long_target_all_methods.csv"
        target_results = pd.read_csv(path)
        target_results = target_results[
            target_results["percentile"].isin(PERCENTILES)
        ].copy()
        annotated = add_family_annotations(target_results, FAMILIES)
        annotated = annotated[
            annotated["familia"].notna() & annotated["EF_cumulative"].notna()
        ].copy()
        frame = (
            annotated.groupby(
                ["method", "method_label", "percentile", "familia"],
                as_index=False,
                dropna=False,
            )["EF_cumulative"]
            .median()
            .rename(columns={"EF_cumulative": "EF_group_cumulative"})
        )
        frame["metric_eval"] = frame["method_label"].map(METHODS)
        frame["seed"] = seed
        frames.append(frame)
    all_partitions = pd.concat(frames, ignore_index=True)

    # Preserve the published aggregation order: first take the median across
    # the five partitions within each protein category, then summarize the
    # resulting category medians.
    category_medians = (
        all_partitions.groupby(
            ["method", "method_label", "metric_eval", "percentile", "familia"],
            as_index=False,
            dropna=False,
        )["EF_group_cumulative"]
        .median()
        .rename(columns={"EF_group_cumulative": "category_median_EF"})
    )
    return category_medians


def summarize_spread(category_medians: pd.DataFrame) -> pd.DataFrame:
    summary = (
        category_medians.groupby(
            ["method", "method_label", "metric_eval", "percentile"],
            as_index=False,
            dropna=False,
        )["category_median_EF"]
        .agg(
            median_EF="median",
            q25_EF=lambda values: values.quantile(0.25),
            q75_EF=lambda values: values.quantile(0.75),
            min_EF="min",
            max_EF="max",
            n_categories="count",
        )
    )
    for source in ("median_EF", "q25_EF", "q75_EF", "min_EF", "max_EF"):
        if (summary[source] <= 0).any():
            raise ValueError(f"Cannot log-transform non-positive values in {source}.")
        summary[f"log10_{source}"] = np.log10(summary[source])
    return summary


def ordered_method_data(summary: pd.DataFrame, method: str) -> pd.DataFrame:
    data = summary[summary["method_label"] == method].copy()
    positions = {value: index for index, value in enumerate(PERCENTILES)}
    data["x"] = data["percentile"].map(positions)
    return data.sort_values("x")


def style_axis(ax: plt.Axes) -> None:
    positions = np.arange(len(PERCENTILES))
    labels = [str(int(value)) if float(value).is_integer() else str(value) for value in PERCENTILES]
    ax.set_xticks(positions, labels, rotation=25, ha="right")
    ax.axhline(0.0, color="#999999", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")


def plot_overlay(
    summary: pd.DataFrame,
    output_dir: Path,
    methods: tuple[str, ...] = METHOD_ORDER,
) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    for method in methods:
        data = ordered_method_data(summary, method)
        color = METHOD_COLORS[method]
        x = data["x"].to_numpy(dtype=float)
        median = data["log10_median_EF"].to_numpy(dtype=float)
        q25 = data["log10_q25_EF"].to_numpy(dtype=float)
        q75 = data["log10_q75_EF"].to_numpy(dtype=float)
        ax.fill_between(x, q25, q75, color=color, alpha=0.11, linewidth=0)
        ax.plot(
            x,
            median,
            color=color,
            marker="o",
            linewidth=2.3,
            markersize=5.5,
            label=METHOD_LABELS[method],
        )

    style_axis(ax)
    ax.set_xlabel("Percentile threshold")
    ax.set_ylabel(r"Category-balanced median $\log_{10}(\mathrm{EF})$")
    ax.legend(ncol=2, frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir / "category_balanced_ef_iqr_overlay")
    plt.close(fig)


def plot_facets(
    summary: pd.DataFrame,
    output_dir: Path,
    methods: tuple[str, ...] = METHOD_ORDER,
    font_size: float | None = None,
) -> None:
    columns = 4 if len(methods) > 4 else 2
    rows = int(np.ceil(len(methods) / columns))
    large_four_panel = font_size is not None and len(methods) <= 4
    fig, axes = plt.subplots(
        rows, columns, figsize=(10, 8.8) if large_four_panel else (3.5 * columns, 3.8 * rows),
        sharex=True, sharey=True, squeeze=False,
    )
    flat_axes = axes.ravel()
    for ax, method in zip(flat_axes, methods):
        data = ordered_method_data(summary, method)
        color = METHOD_COLORS[method]
        x = data["x"].to_numpy(dtype=float)
        minimum = data["log10_min_EF"].to_numpy(dtype=float)
        q25 = data["log10_q25_EF"].to_numpy(dtype=float)
        median = data["log10_median_EF"].to_numpy(dtype=float)
        q75 = data["log10_q75_EF"].to_numpy(dtype=float)
        maximum = data["log10_max_EF"].to_numpy(dtype=float)
        ax.fill_between(x, minimum, maximum, color=color, alpha=0.10, linewidth=0)
        ax.fill_between(x, q25, q75, color=color, alpha=0.30, linewidth=0)
        ax.plot(x, median, color=color, marker="o", linewidth=2.2, markersize=5)
        ax.set_title(METHOD_LABELS[method], fontsize=11 if font_size is None else font_size, fontweight="semibold")
        style_axis(ax)

    for ax in flat_axes[len(methods):]:
        ax.axis("off")
    fig.supxlabel("Percentile threshold", y=0.15 if large_four_panel else (0.055 if len(methods) > 4 else 0.07))
    fig.supylabel(r"$\log_{10}(\mathrm{EF})$", x=0.025)
    legend_handles = [
        Line2D([0], [0], color="#333333", marker="o", linewidth=2.2, label="Category-balanced median"),
        Patch(facecolor="#777777", alpha=0.30, label="Category IQR"),
        Patch(facecolor="#777777", alpha=0.10, label="Category min–max range"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.76, 0.025) if len(methods) > 4 else (0.5, 0.005),
        frameon=False,
        ncol=1 if len(methods) > 4 or large_four_panel else 3,
        fontsize=9 if font_size is None else font_size - 2,
    )
    bottom = 0.22 if large_four_panel else (0.09 if len(methods) > 4 else 0.12)
    fig.tight_layout(rect=(0.035, bottom, 1, 1))
    save_figure(fig, output_dir / "category_balanced_ef_iqr_range_facets")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    category_medians = load_category_summaries(input_dir)
    summary = summarize_spread(category_medians)
    category_medians.to_csv(output_dir / "category_medians_across_partitions.csv", index=False)
    summary.to_csv(output_dir / "category_balanced_cumulative_spread.csv", index=False)
    plot_overlay(summary, output_dir)
    plot_facets(summary, output_dir)
    print(f"Wrote category-spread previews to: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
