#!/usr/bin/env python3
"""Plot four representations using the frozen supplementary EF summaries.

No molecular searches or statistical reaggregation are performed. The IQR
overlay preserves the original colors, axes and values. A companion facet
figure also displays the category minimum-to-maximum range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from preview_category_balanced_spread import (
    PERCENTILES,
    plot_facets,
    plot_overlay,
)


MAIN_METHODS = (
    "morgan_1024_r2",
    "morgan_feature_1024_r2",
    "chemberta_zinc_base_768",
    "rdkit_1024",
)


def select_main_methods(summary: pd.DataFrame) -> pd.DataFrame:
    """Select existing rows without changing any statistic or log transform."""
    required = {
        "method_label", "percentile", "median_EF", "q25_EF", "q75_EF",
        "min_EF", "max_EF", "n_categories", "log10_median_EF",
        "log10_q25_EF", "log10_q75_EF", "log10_min_EF", "log10_max_EF",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"Missing summary columns: {sorted(missing)}")
    selected = summary.loc[summary["method_label"].isin(MAIN_METHODS)].copy()
    if selected.duplicated(["method_label", "percentile"]).any():
        raise ValueError("Duplicate method/percentile rows in the source summary.")
    for method in MAIN_METHODS:
        observed = set(selected.loc[selected["method_label"].eq(method), "percentile"])
        if observed != set(PERCENTILES):
            raise ValueError(f"Incomplete or unexpected percentiles for {method}: {sorted(observed)}")
    if selected[list(required)].isna().any().any():
        raise ValueError("Missing values in the selected summary.")
    return selected


def main() -> int:
    workspace = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=workspace / "reviewer_preview_category_spread",
        help="Directory containing category_balanced_cumulative_spread.csv.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=workspace / "reviewer_main_representation_comparison",
    )
    parser.add_argument("--font-size", type=float, default=16, help="Base font size in points (default: 16).")
    args = parser.parse_args()
    if args.font_size < 6:
        parser.error("--font-size must be at least 6 points.")
    source = args.input_dir.expanduser().resolve() / "category_balanced_cumulative_spread.csv"
    output = args.output_dir.expanduser().resolve()
    if source.parent == output:
        parser.error("Use a separate output directory to preserve the supplementary figure.")
    summary = select_main_methods(pd.read_csv(source))
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / source.name, index=False)
    with plt.rc_context({
        "font.size": args.font_size, "axes.labelsize": args.font_size + 2,
        "xtick.labelsize": args.font_size, "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size, "legend.title_fontsize": args.font_size,
    }):
        plot_overlay(summary, output, methods=MAIN_METHODS)
        plot_facets(summary, output, methods=MAIN_METHODS, font_size=args.font_size)
    metadata = {
        "source_summary": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "methods": list(MAIN_METHODS),
        "percentiles": list(PERCENTILES),
        "aggregation": "Unchanged from the source supplementary summary.",
        "spread": "Q25-Q75 across category medians; facets also show min-max.",
        "interpretation": "Descriptive category variation, not confidence intervals or significance tests.",
        "font_size_points": args.font_size,
    }
    (output / "plot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote main four-representation figures and exact source statistics to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
