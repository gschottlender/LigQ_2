#!/usr/bin/env python3
"""Pack seven-category supplementary figures into three single-page panels.

This is plot-only: reuse frozen category EF medians and normalized similarity
histograms. No searches, ranking changes, new uncertainty bands, or chemical
calculations are performed. Existing single-category figures are preserved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from ligq2_evaluation.constants import FAMILIES, FAMILY_LABELS, FAMILY_ORDER, SEEDS
from ligq2_evaluation.families import add_family_annotations
from preview_category_balanced_spread import METHOD_ORDER, METHOD_LABELS, METHOD_COLORS, PERCENTILES


CATEGORY_ORDER = tuple(FAMILY_LABELS[f] for f in FAMILY_ORDER if f != "Canales iónicos")
NEIGHBOR_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0)
NEIGHBOR_STRATEGIES = ("K=2", "K=3", "K=5", "K=10", "K=15", "Full domain")
GROUPS = ("TP", "putative_FP", "putative_TN")
GROUP_COLORS = {"TP": "#2378b5", "putative_FP": "#df7c31", "putative_TN": "#707070"}
GROUP_LABELS = {
    "TP": "Retrieved actives (TP; ≥P99 cutoff)",
    "putative_FP": "Retrieved background (putative FP; ≥P99 cutoff)",
    "putative_TN": "Non-retrieved background (putative TN; <P99 cutoff)",
}
STEMS = (
    "A1_representations_by_family_panel",
    "A2_neighbors_and_full_domain_by_family_panel",
    "A3_similarity_p99_tp_fp_tn_by_family_panel",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_categories(table, column):
    table = table.copy()
    table["protein_category"] = table[column].map(lambda value: FAMILY_LABELS.get(value, value))
    observed = set(table.protein_category)
    if observed != set(CATEGORY_ORDER):
        raise ValueError(f"Expected seven categories; missing={set(CATEGORY_ORDER)-observed}, extra={observed-set(CATEGORY_ORDER)}")
    return table


def validate_curves(table, series, value, order, percentiles):
    required = {"protein_category", series, value, "percentile"}
    if missing := required - set(table):
        raise ValueError(f"Missing curve columns: {sorted(missing)}")
    table = table[table.percentile.isin(percentiles)].copy()
    if table.duplicated(["protein_category", series, "percentile"]).any():
        raise ValueError("Duplicate category/series/percentile rows")
    if not np.isfinite(table[value].to_numpy(float)).all():
        raise ValueError("Non-finite enrichment values")
    for category in CATEGORY_ORDER:
        rows = table[table.protein_category.eq(category)]
        if set(rows[series]) != set(order):
            raise ValueError(f"Incomplete strategy set for {category}")
        for label in order:
            actual = set(rows.loc[rows[series].eq(label), "percentile"])
            if actual != set(percentiles):
                raise ValueError(f"Incomplete percentile curve for {category}/{label}")
    return table


def representation_counts(base: Path):
    """Read original target membership for titles; do not reaggregate EF."""
    paths, previous = [], None
    for seed in SEEDS:
        path = base / f"seed_{seed}" / "df_long_target_all_methods.csv"
        paths.append(path)
        frame = pd.read_csv(path)
        frame = frame[frame.method_label.isin(METHOD_ORDER) & frame.percentile.isin(PERCENTILES)]
        frame = frame[frame.EF_cumulative.notna()]
        frame = add_family_annotations(frame, FAMILIES).dropna(subset=["familia"])
        frame = normalize_categories(frame, "familia")
        membership = {
            (category, method): frozenset(frame.loc[
                frame.protein_category.eq(category) & frame.method_label.eq(method), "target"
            ]) for category in CATEGORY_ORDER for method in METHOD_ORDER
        }
        if previous is not None and membership != previous:
            raise ValueError("Target membership differs between representation partitions")
        previous = membership
    counts = {}
    for category in CATEGORY_ORDER:
        groups = [previous[(category, method)] for method in METHOD_ORDER]
        if any(group != groups[0] for group in groups):
            raise ValueError(f"Representation target membership differs between methods for {category}")
        counts[category] = len(groups[0])
    return counts, paths


def make_canvas(title, xlabel, ylabel, note):
    fig, axes = plt.subplots(3, 3, figsize=(16.4, 11.8))
    fig.subplots_adjust(left=.074, right=.985, bottom=.11, top=.90, wspace=.27, hspace=.42)
    fig.suptitle(title, fontsize=20, y=.972)
    fig.supxlabel(xlabel, fontsize=16, y=.062)
    fig.supylabel(ylabel, fontsize=16, x=.012)
    fig.text(.5, .022, note, ha="center", va="bottom", fontsize=11)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=.7, alpha=.25)
        ax.tick_params(labelsize=12)
    for ax in axes.flat[7:]:
        ax.set_axis_off()
    return fig, list(axes.flat[:7])


def category_title(ax, category, n_targets, index):
    ax.set_title(f"{chr(65+index)}  {category} (n={n_targets})", loc="left", fontsize=15, fontweight="bold", pad=10)


def plot_ef(table, *, series, value, order, percentiles, colors, labels, counts,
            title, note, log10=False, shared_y=False):
    ylabel = r"Median $\log_{10}(\mathrm{EF})$" if log10 else "Median cumulative enrichment factor"
    fig, axes = make_canvas(title, "Percentile threshold", ylabel, note)
    positions = np.arange(len(percentiles))
    plot_values = table[value].to_numpy(float)
    if log10:
        if np.any(plot_values <= 0):
            raise ValueError("Non-positive EF cannot be represented as log10")
        plot_values = np.log10(plot_values)
    shared_limits = (min(0, plot_values.min())-.05, plot_values.max()+.1)
    for index, (category, ax) in enumerate(zip(CATEGORY_ORDER, axes)):
        frame = table[table.protein_category.eq(category)]
        for strategy in order:
            rows = frame[frame[series].eq(strategy)].set_index("percentile").loc[list(percentiles)]
            values = rows[value].to_numpy(float)
            if log10:
                values = np.log10(values)
            ax.plot(positions, values, color=colors[strategy], marker="o", ms=4.5,
                    lw=2, linestyle="--" if strategy == "Full domain" else "-", label=labels[strategy])
        ax.axhline(0 if log10 else 1, color=".6", linestyle=":", linewidth=.9, zorder=0)
        ax.set_xticks(positions, [f"{p:g}" for p in percentiles], rotation=25, ha="right")
        if shared_y:
            ax.set_ylim(*shared_limits)
        else:
            ax.set_ylim(bottom=0)
        category_title(ax, category, counts[category], index)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, title="Molecular representation" if log10 else "Seed source",
               loc="center", bbox_to_anchor=(.67, .223), ncol=2, frameon=False,
               fontsize=13, title_fontsize=14, columnspacing=1.8, handlelength=2.4)
    return fig


def validate_histograms(table):
    table = table[table.property.eq("similarity")].copy()
    table = normalize_categories(table, "protein_family")
    if table.duplicated(["protein_category", "group", "bin"]).any():
        raise ValueError("Duplicate category/group/bin histogram rows")
    for category in CATEGORY_ORDER:
        rows = table[table.protein_category.eq(category)]
        if set(rows.group) != set(GROUPS):
            raise ValueError(f"Missing TP/FP/TN distributions for {category}")
        if rows.n_targets.nunique() != 1:
            raise ValueError(f"Histogram group target counts differ for {category}")
        for group in GROUPS:
            cells = rows[rows.group.eq(group)].sort_values("bin")
            if len(cells) != 50 or not np.allclose(cells.bin_left, np.linspace(0, .98, 50)) or not np.allclose(cells.bin_right, np.linspace(.02, 1, 50)):
                raise ValueError(f"Unexpected similarity bins for {category}/{group}")
            if not np.isfinite(cells.fraction).all() or (cells.fraction < 0).any() or not np.isclose(cells.fraction.sum(), 1):
                raise ValueError(f"Non-normalized histogram for {category}/{group}")
    return table


def plot_similarity(table):
    fig, axes = make_canvas(
        "Chemical similarity of retrieved and non-retrieved compounds",
        "Maximum ECFP4/Tanimoto similarity to known actives", "Fraction within group",
        "P99 is defined separately in each target/partition; bin width = 0.02.\n"
        "Each group is normalized separately; partitions and targets contribute equally within each category.",
    )
    ymax = table.fraction.max()*1.12
    for index, (category, ax) in enumerate(zip(CATEGORY_ORDER, axes)):
        frame = table[table.protein_category.eq(category)]
        for group in GROUPS:
            rows = frame[frame.group.eq(group)].sort_values("bin")
            bins = np.r_[rows.bin_left.to_numpy(), rows.bin_right.iloc[-1]]
            ax.stairs(rows.fraction.to_numpy(), bins, color=GROUP_COLORS[group], linewidth=2,
                      linestyle="--" if group == "putative_TN" else "-", label=GROUP_LABELS[group])
            if group != "putative_TN":
                ax.stairs(rows.fraction.to_numpy(), bins, color=GROUP_COLORS[group], fill=True, alpha=.08)
        ax.set(xlim=(0, 1), ylim=(0, ymax))
        ax.set_xticks(np.linspace(0, 1, 6))
        category_title(ax, category, int(frame.n_targets.iloc[0]), index)
    handles = [Line2D([], [], color=GROUP_COLORS[g], lw=2,
                     linestyle="--" if g == "putative_TN" else "-") for g in GROUPS]
    fig.legend(handles, [GROUP_LABELS[g] for g in GROUPS], loc="center", bbox_to_anchor=(.67, .223),
               ncol=1, frameon=False, fontsize=13, handlelength=2.8)
    return fig


def main(argv=None):
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-summary", type=Path, default=workspace/"reviewer_preview_category_spread/category_medians_across_partitions.csv")
    parser.add_argument("--representation-results-dir", type=Path, default=workspace/"resultados_EF_repeticiones_con_inactivos_10_90")
    parser.add_argument("--neighbor-summary", type=Path, default=workspace/"evaluation_scripts/reproduction_output/full_domain_benchmark/neighbor_vs_full_domain_category_medians.csv")
    parser.add_argument("--similarity-summary", type=Path, default=workspace/"reviewer_retrieved_hits_with_tn_p99/figures/histograms_by_family.csv")
    parser.add_argument("--output-dir", type=Path, default=workspace/"publication_appendices")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)
    if args.dpi < 100:
        parser.error("--dpi must be at least 100")
    out = args.output_dir.resolve()
    if out == args.representation_summary.parent.resolve() or out == args.neighbor_summary.parent.resolve() or out == args.similarity_summary.parent.resolve():
        parser.error("Use a separate output directory to preserve existing figures")
    counts, count_paths = representation_counts(args.representation_results_dir)
    source_paths = [args.representation_summary, args.neighbor_summary, args.similarity_summary, *count_paths]
    hashes = {str(p.resolve()): digest(p) for p in source_paths}
    rep = validate_curves(normalize_categories(pd.read_csv(args.representation_summary), "familia"),
                          "method_label", "category_median_EF", METHOD_ORDER, PERCENTILES)
    neighbors = validate_curves(normalize_categories(pd.read_csv(args.neighbor_summary), "familia"),
                                "strategy", "category_median_EF_cumulative", NEIGHBOR_STRATEGIES, NEIGHBOR_PERCENTILES)
    neighbor_counts = {}
    for family in CATEGORY_ORDER:
        rows = neighbors[neighbors.protein_category.eq(family)]
        if rows.n_targets.nunique() != 1:
            raise ValueError(f"Neighbor target counts differ for {family}")
        neighbor_counts[family] = int(rows.n_targets.iloc[0])
    histograms = validate_histograms(pd.read_csv(args.similarity_summary))
    palette = plt.get_cmap("viridis")
    neighbor_colors = {strategy: palette(i/4) for i, strategy in enumerate(NEIGHBOR_STRATEGIES[:-1])}
    neighbor_colors["Full domain"] = "#d62728"
    out.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out/"A1_plot_data.csv", index=False)
    neighbors.to_csv(out/"A2_plot_data.csv", index=False)
    histograms.to_csv(out/"A3_plot_data.csv", index=False)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 13,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}):
        figures = [
            plot_ef(rep, series="method_label", value="category_median_EF", order=METHOD_ORDER,
                    percentiles=PERCENTILES, colors=METHOD_COLORS, labels=METHOD_LABELS, counts=counts,
                    title="Molecular representation performance by protein category", log10=True, shared_y=True,
                    note="Within each category: median target EF per partition, then median across five partitions.\n"
                    "Category-level medians are plotted; no between-category IQR is defined within an individual category."),
            plot_ef(neighbors, series="strategy", value="category_median_EF_cumulative", order=NEIGHBOR_STRATEGIES,
                    percentiles=NEIGHBOR_PERCENTILES, colors=neighbor_colors, labels={s:s for s in NEIGHBOR_STRATEGIES},
                    counts=neighbor_counts, title="Nearest-neighbor and full-domain performance by protein category",
                    note="Median across five partitions within target, then median across targets within category.\n"
                    "Panels use independent y-axis scales; the original retained K sweep and Full domain medians are unchanged."),
            plot_similarity(histograms),
        ]
        with PdfPages(out/"supplementary_family_appendices.pdf", metadata={
            "Title": "LigQ2 category-level supplementary panels", "Author": "LigQ2",
            "Subject": "Three pages, each containing all seven protein categories",
        }) as pdf:
            for stem, fig in zip(STEMS, figures):
                for extension in ("png", "pdf", "svg"):
                    fig.savefig(out/f"{stem}.{extension}", dpi=args.dpi)
                pdf.savefig(fig)
                plt.close(fig)
    for path in source_paths:
        if digest(path) != hashes[str(path.resolve())]:
            raise RuntimeError(f"Source file changed during plotting: {path}")
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "script_sha256": digest(Path(__file__)),
        "source_sha256": hashes, "source_data_unchanged": True,
        "layout": "three pages; one 3x3 grid per analysis; seven data subpanels and shared legend",
        "category_order": list(CATEGORY_ORDER), "figure_size_inches": [16.4, 11.8],
        "panel_title_font_size": 15, "tick_font_size": 12, "legend_font_size": 13, "dpi": args.dpi,
        "representation_counts": counts, "neighbor_counts": neighbor_counts,
        "representation_percentiles": list(PERCENTILES), "neighbor_percentiles": list(NEIGHBOR_PERCENTILES),
        "neighbor_strategies": list(NEIGHBOR_STRATEGIES), "similarity_groups": list(GROUPS),
        "similarity_bin_width": .02, "threshold_percentile": 99,
        "representation_aggregation": "median targets within category and partition, then median partitions",
        "neighbor_aggregation": "median partitions within target, then median targets within category",
        "similarity_aggregation": "normalize each group histogram per target/partition; mean partitions within target; mean targets within category",
        "uncertainty": "No new bands: Category IQR describes between-category variation and cannot be drawn within one category",
        "outputs": [f"{stem}.{fmt}" for stem in STEMS for fmt in ("png", "pdf", "svg")]+["supplementary_family_appendices.pdf"],
    }
    (out/"run_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    captions = """# Supplementary category-level figure panels

Each appendix is one page containing all seven protein categories (A-G).
The combined PDF contains exactly three pages. These are plot-only derivatives
of retained results, not new benchmark runs. Original individual figures are
unchanged. All labels and captions are in English.

## A1. Molecular representation comparison

Cumulative enrichment factors for seven molecular representations, shown
separately for each protein category. For each method and percentile, the
median across targets was calculated within each category and partition,
followed by the median across the five partitions. The plotted ordinate is
log10 of that category median EF. The six displayed percentiles reproduce the
retained main/supplementary summary. Titles give the category target counts.
Panels share their y-axis scale. No between-category IQR is shown within an
individual category; these curves are descriptive medians, not confidence
intervals.

## A2. Nearest-neighbor versus full-domain evidence

Cumulative enrichment factors for the retained K=2, 3, 5, 10 and 15 settings
and Full domain, separated by protein category. Each target contributes its
median across five partitions; curves are the median across those target
values within each category. The 56 matched targets and the five displayed
percentiles are unchanged from the retained family plots. Y-axis scales are
independent between panels for readability. The original requested maximum
seed budget is shared; realized seed counts can differ between strategies.
No new searches, MaxMin selections, or uncertainty bands were calculated.

## A3. Chemical similarity above and below the percentile-99 cutoff

Distributions of maximum ECFP4/Tanimoto similarity to the known-active seeds
for held-out actives retrieved at or above the full-pool percentile-99 cutoff
(TP), background compounds retrieved at or above the same cutoff (putative
FP), and background compounds below the cutoff (putative TN). The cutoff is
defined independently for each target and partition and retains tied scores;
it is not a fixed Tanimoto value of 0.99. Histograms use a bin width of 0.02,
with each group normalized separately within target/partition. Bin fractions
are averaged over the five partitions within each target and then equally
over targets within each category. The panels cover 61 targets and 305
partitions, use shared axes, and show similarity only, without the additional
physicochemical-property panels. Background compounds are not experimentally
confirmed inactives; FP and TN are therefore putative labels. Non-retrieved
actives (FN) are not included in the TN curve.

## Regeneration

From the evaluation workspace root, using a pandas/numpy/matplotlib environment:

```bash
python evaluation_scripts/plot_family_appendices.py
```

Input paths, source SHA256 hashes, target counts and exact displayed data are
retained in `run_metadata.json` and `A1_plot_data.csv` through `A3_plot_data.csv`.
"""
    (out/"APPENDIX_CAPTIONS.md").write_text(captions)
    print(f"Wrote three seven-category panels and a three-page PDF to {out}")
    print(f"Target counts: representations={sum(counts.values())}, neighbors={sum(neighbor_counts.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
