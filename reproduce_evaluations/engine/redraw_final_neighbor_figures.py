#!/usr/bin/env python3
"""Redraw three final neighbor/transfer comparisons with readable labels.

Uses only frozen summary CSVs. Existing general and full-domain family
figures are updated, while the other two comparisons gain family-specific
panels from their retained family medians. No EF calculation is repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import plot_ecfp4_transfer_strategy_comparison as transfer
import plot_identity_ligand_rescue_comparison as rescue
import run_full_domain_benchmark as domain
from ligq2_evaluation.constants import FAMILY_LABELS, FAMILY_ORDER
from ligq2_evaluation.publication_style import readable_labels


@readable_labels
def plot_family_curves(table, *, family_column, value_column, strategies, labels,
                       colors, styles, output_dir, stem):
    """Plot existing family medians without inventing category IQR bands."""
    required = {family_column, value_column, "strategy", "percentile", "n_targets"}
    if required - set(table.columns):
        raise ValueError(f"Missing family-summary columns: {sorted(required - set(table.columns))}")
    order = [FAMILY_LABELS[family] for family in FAMILY_ORDER]
    normalized = table.copy()
    normalized["family_label"] = normalized[family_column].map(lambda value: FAMILY_LABELS.get(value, value))
    unknown = set(normalized.family_label) - set(order)
    if unknown:
        raise ValueError(f"Unexpected protein categories: {sorted(unknown)}")
    normalized = normalized[normalized.percentile.isin(rescue.PLOT_PERCENTILES)].copy()
    if normalized.duplicated(["family_label", "strategy", "percentile"]).any():
        raise ValueError("Duplicate family/strategy/percentile rows.")
    figures = output_dir / "family_figures"
    figures.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_dir / f"{stem}_by_protein_group.pdf") as pdf:
        for family in order:
            frame = normalized[normalized.family_label.eq(family)]
            if frame.empty:
                continue
            counts = frame.n_targets.unique()
            if len(counts) != 1:
                raise ValueError(f"Inconsistent target counts for {family}.")
            fig, ax = plt.subplots(figsize=(10.8, 6.5))
            x = np.arange(len(rescue.PLOT_PERCENTILES))
            for strategy in strategies:
                rows = frame[frame.strategy.eq(strategy)].set_index("percentile")
                if set(rows.index) != set(rescue.PLOT_PERCENTILES):
                    raise ValueError(f"Incomplete family curve for {family}, {strategy}.")
                rows = rows.loc[list(rescue.PLOT_PERCENTILES)]
                ax.plot(x, rows[value_column].to_numpy(float), marker="o",
                        markersize=5.5, linewidth=2.4, color=colors[strategy],
                        linestyle=styles[strategy], label=labels[strategy])
            ax.set_xticks(x, [f"{p:g}" for p in rescue.PLOT_PERCENTILES])
            ax.set_xlabel("Percentile threshold")
            ax.set_ylabel("Median cumulative\nenrichment factor")
            ax.set_title(f"{family} (n={int(counts[0])} targets)")
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(title="Ligand evidence strategy", frameon=False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            slug = re.sub(r"[^a-z0-9]+", "_", family.lower()).strip("_")
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(figures / f"{stem}_{slug}.{suffix}",
                            dpi=600 if suffix == "png" else None, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path,
                        default=Path(__file__).resolve().parent / "reproduction_output")
    parser.add_argument("--font-size", type=float, default=16)
    args = parser.parse_args()
    root = args.results_root.expanduser().resolve()
    rescue_dir = root / "identity_ligand_rescue_comparison"
    domain_dir = root / "full_domain_benchmark"
    transfer_dir = root / "ecfp4_transfer_strategy_comparison"
    paths = [
        rescue_dir / "strategy_category_balanced_summary.csv",
        rescue_dir / "strategy_family_medians.csv",
        rescue_dir / "run_metadata.json",
        domain_dir / "neighbor_vs_full_domain_category_balanced_spread.csv",
        domain_dir / "neighbor_vs_full_domain_category_medians.csv",
        transfer_dir / "ecfp4_transfer_strategy_category_balanced_spread.csv",
        transfer_dir / "ecfp4_transfer_strategy_family_medians.csv",
    ]
    digests = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    meta = json.loads(paths[2].read_text())
    identity, budget = meta["identity_floor_percent"], meta["ligand_candidate_threshold"]
    rescue.plot_comparison(pd.read_csv(paths[0]), identity_floor=identity,
                           ligand_budget=budget, output_dir=rescue_dir,
                           dpi=600, font_size=args.font_size)
    rescue_labels = {
        "fixed_k_5": "K=5", "fixed_k_15": "K=15", "full_domain": "Full domain",
        "identity_ligand_rescue": f"Identity ≥{identity:g}% + ligand rescue ≥{budget}",
    }
    plot_family_curves(
        pd.read_csv(paths[1]), family_column="protein_family",
        value_column="family_median_EF_cumulative", strategies=rescue.DISPLAY_ORDER,
        labels=rescue_labels,
        colors=dict(zip(rescue.DISPLAY_ORDER, ("#1f77b4", "#9467bd", "#d62728", "#2ca02c"))),
        styles=dict(zip(rescue.DISPLAY_ORDER, ("-", "-", "--", "-."))),
        output_dir=rescue_dir, stem="fixed_k_domain_and_identity_ligand_rescue",
        font_size=args.font_size,
    )
    print("Updated preferred neighbor/adaptive comparison and family figures.", flush=True)
    domain._plot_comparison(pd.read_csv(paths[3]), domain_dir, font_size=args.font_size)
    domain._plot_family_comparisons(pd.read_csv(paths[4]), domain_dir, font_size=args.font_size)
    print("Updated retained K sweep versus Full domain and family figures.", flush=True)
    transfer.plot_comparison(pd.read_csv(paths[5]), transfer_dir / "ecfp4_transfer_strategy_comparison",
                             font_size=args.font_size)
    plot_family_curves(
        pd.read_csv(paths[6]), family_column="familia", value_column="family_median_EF_cumulative",
        strategies=transfer.STRATEGIES, labels={value: value for value in transfer.STRATEGIES},
        colors=dict(zip(transfer.STRATEGIES, ("#4477AA", "#228833", "#CC3311"))),
        styles=dict(zip(transfer.STRATEGIES, ("-", "-", "--"))),
        output_dir=transfer_dir, stem="ecfp4_transfer_strategy_comparison",
        font_size=args.font_size,
    )
    for path in paths:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digests[str(path)]:
            raise RuntimeError(f"Source data changed while redrawing: {path}")
    for output in (rescue_dir, domain_dir, transfer_dir):
        (output / "figure_style_metadata.json").write_text(json.dumps({
            "font_size_points": args.font_size, "axis_label_size_points": args.font_size + 2,
            "source_sha256": digests, "source_data_unchanged": True,
            "general_bands": "Retained Category IQR, unchanged.",
            "family_panels": "Retained family medians, without between-category IQR bands.",
        }, indent=2) + "\n")
    print("Updated all three comparisons and family panels; every input hash is unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
