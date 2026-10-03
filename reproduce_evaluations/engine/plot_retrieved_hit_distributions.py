#!/usr/bin/env python
"""Readable selected-hit histograms with complete data and explicit tail labels.

Rebin the saved hit observations, without changing selection or aggregation.
Display windows are a visualization choice, not chemical exclusion criteria.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT = Path(__file__).with_name("19_characterize_retrieved_hits.py")
SPEC = importlib.util.spec_from_file_location("retrieved_hits_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)

WINDOWS = {"similarity": (0, 1), "molecular_weight": (0, 1200),
           "logp": (-5, 10), "hba_minus_hbd": (-5.5, 20.5)}
WIDTHS = {"similarity": 0.02, "molecular_weight": 25, "logp": 0.25,
          "hba_minus_hbd": 1}
GROUP_ORDER = ("TP", "putative_FP", "putative_TN")
COLORS = {**analysis.COLORS, "putative_TN": "#707070"}
LABELS = {**analysis.LABELS, "putative_TN": "Non-retrieved background (putative TN)"}


def fixed_width_edges(hits):
    edges = {"similarity": np.linspace(0, 1, 51)}
    for prop in ("molecular_weight", "logp"):
        width = WIDTHS[prop]
        lower = np.floor(hits[prop].min() / width) * width
        upper = np.ceil(hits[prop].max() / width) * width
        if lower == upper:
            upper += width
        edges[prop] = np.arange(lower, upper + width / 2, width)
    edges["hba_minus_hbd"] = np.arange(hits.hba_minus_hbd.min() - 0.5,
                                       hits.hba_minus_hbd.max() + 1.5)
    return edges


def outside_fraction(table, limits):
    """Exact histogram mass outside a display window aligned with bin edges."""
    left, right = limits
    # All display windows align with bin edges, so no fractional bin clipping.
    return float(table.loc[table.bin_right.le(left) | table.bin_left.ge(right), "fraction"].sum())


def panel(histograms, output, title, subtitle, formats, full_range=False):
    groups = tuple(group for group in GROUP_ORDER if group in set(histograms.group))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    for letter, (ax, (prop, label)) in zip("ABCD", zip(axes.flat, analysis.PROPERTIES.items())):
        tails = {}
        for group in groups:
            table = histograms.loc[histograms.property.eq(prop) & histograms.group.eq(group)].sort_values("bin")
            if table.empty:
                continue
            bins = np.r_[table.bin_left.to_numpy(), table.bin_right.iloc[-1]]
            ax.stairs(table.fraction.to_numpy(), bins, color=COLORS[group], lw=1.8,
                      linestyle="--" if group == "putative_TN" else "-", label=LABELS[group])
            if group != "putative_TN":
                ax.stairs(table.fraction.to_numpy(), bins, color=COLORS[group], fill=True, alpha=0.12)
            tails[group] = outside_fraction(table, WINDOWS[prop])
        ax.set(xlabel=label, ylabel="Fraction within group")
        ax.set_title(letter, loc="left", fontweight="bold")
        if not full_range:
            ax.set_xlim(*WINDOWS[prop])
            if prop != "similarity" and tails:
                text = f"Outside view: TP {100*tails.get('TP', 0):.2f}%\nputative FP {100*tails.get('putative_FP', 0):.2f}%"
                if "putative_TN" in tails:
                    text += f"\nputative TN {100*tails['putative_TN']:.2f}%"
                ax.text(0.985, 0.97, text, transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.935),
               ncol=1 if len(groups) == 3 else 2, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=13, y=0.99)
    fig.text(0.5, 0.014, subtitle + ("\nAll observations retained; tails outside display windows are reported."
                                   if not full_range else "\nComplete observed ranges."),
             ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.055, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output.with_suffix("." + fmt), dpi=160)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--skip-individual", action="store_true")
    parser.add_argument("--individual-formats", default="png", help="Comma-separated png,pdf,svg")
    parser.add_argument("--full-range", action="store_true")
    args = parser.parse_args(argv)
    formats = tuple(args.individual_formats.split(","))
    if not set(formats) <= {"png", "pdf", "svg"}:
        parser.error("Formats must be png, pdf, or svg.")
    source = args.input_dir.resolve()
    out = (args.output_dir or source / "figures").resolve()
    out.mkdir(parents=True, exist_ok=True)
    hits_path = source / "retrieved_hits.parquet"
    counts = pd.read_csv(source / "hit_counts_by_partition.csv")
    if hits_path.exists():
        hits = pd.read_parquet(hits_path)
        edges = fixed_width_edges(hits)
        histograms, exclusions = analysis.build_histograms(hits, edges)
    else:
        hits_path = source / "histograms_by_partition.parquet"
        histograms = pd.read_parquet(hits_path)
        run = json.loads((source / "run_metadata.json").read_text())
        edges = {key: np.asarray(values) for key, values in run["bins"].items()}
        exclusions = run["empty_distribution_pairs"]
    targets, families, overall = analysis.aggregate_histograms(histograms)
    for name, table in (("by_partition", histograms), ("by_target", targets),
                        ("by_family", families), ("category_balanced", overall)):
        table.to_csv(out / f"histograms_{name}.csv", index=False)
    percentile = float(counts.percentile.iloc[0])
    title = f"ECFP4 recovered hits | percentile {percentile:g}"
    if "putative_TN" in set(histograms.group):
        title = f"ECFP4 hits and non-retrieved background | percentile {percentile:g}"
    panel(overall, out / "retrieved_hits_distributions", title,
          "Equal category weights; equal target weights within category; equal partition weights within target.",
          ("png", "pdf", "svg"), args.full_range)
    panel(overall, out / "retrieved_hits_distributions_full_range", title,
          "Equal category, target and partition weights; same bins as the focused figure.",
          ("png", "pdf", "svg"), True)
    tail_rows = []
    for (prop, group), table in overall.groupby(["property", "group"]):
        tail_rows.append({"property": prop, "group": group, "display_min": WINDOWS[prop][0],
                          "display_max": WINDOWS[prop][1],
                          "fraction_outside_view": outside_fraction(table, WINDOWS[prop])})
    pd.DataFrame(tail_rows).to_csv(out / "display_tail_fractions.csv", index=False)
    for family, table in families.groupby("protein_family"):
        folder = out / "by_family" / family.lower().replace(" ", "_")
        panel(table, folder / "distributions", title + " | " + family,
              "Equal target weights; equal partition weights within target.", ("png", "pdf"), args.full_range)
    if not args.skip_individual:
        for target, table in targets.groupby("target"):
            folder = out / "individual_distributions/by_target" / target
            panel(table, folder / "distributions", title + " | " + target.upper(),
                  "Mean normalized distributions over the five partitions.", formats, args.full_range)
            table.to_csv(folder / "histograms.csv", index=False)
        cells = histograms.groupby(["target", "partition"])
        for index, ((target, seed), table) in enumerate(cells, 1):
            folder = out / "individual_distributions/by_partition" / target / f"seed_{seed}"
            count = counts.loc[counts.target.eq(target) & counts.partition.eq(seed)].iloc[0]
            count_text = f"Groups normalized separately. TP: {count.n_TP:,}; putative FP: {count.n_putative_FP:,}."
            if "n_putative_TN" in counts.columns:
                count_text += f" Putative TN: {count.n_putative_TN:,}."
            panel(table, folder / "distributions", f"{title} | {target.upper()} | seed {seed}",
                  count_text,
                  formats, args.full_range)
            table.to_csv(folder / "histograms.csv", index=False)
            if index % 25 == 0:
                print(f"Refined partition panels: {index}/{len(cells)}", flush=True)
    metadata = {"source_hit_file": str(hits_path), "source_hit_sha256": analysis.sha256_file(hits_path),
                "plot_script_sha256": analysis.sha256_file(Path(__file__)),
                "rdkit_version": analysis.rdBase.rdkitVersion, "bin_widths": WIDTHS,
                "bins": {key: values.tolist() for key, values in edges.items()},
                "display_windows": WINDOWS, "all_observations_retained": True,
                "full_range": args.full_range, "empty_distribution_pairs": exclusions,
                "n_targets": counts.target.nunique(), "n_partitions": len(counts),
                "groups": [group for group in GROUP_ORDER if group in set(histograms.group)],
                "aggregation": "Mean partition proportions within target, mean targets within family, mean families",
                "individual_formats": formats, "individual_generated": not args.skip_individual}
    (out / "figure_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Figures and exact histogram tables: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
