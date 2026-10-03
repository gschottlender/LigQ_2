#!/usr/bin/env python3
"""Compare ECFP4 target, nearest-K, and full-domain ligand evidence.

This is a derived analysis: it reads retained benchmark results and never
recalculates molecular similarities.  The comparison is restricted to targets
with complete results for all three strategies in all five published
partitions.  For every curve, target values are first summarized across
partitions, targets are then summarized within protein families, and the
plotted value and ribbon are the median and IQR across family medians.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ligq2_evaluation.category_spread import summarize_category_spread
from ligq2_evaluation.constants import FAMILIES, RAW_PERCENTILES, SEEDS
from ligq2_evaluation.families import add_family_annotations
from ligq2_evaluation.runtime import prepare_output
from ligq2_evaluation.publication_style import readable_labels


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
METHOD = "morgan_1024_r2"
SEQUENCE = "Sequence (target ligands)"
NEAREST_K5 = "Nearest K=5"
FULL_DOMAIN = "Full domain"
STRATEGIES = (SEQUENCE, NEAREST_K5, FULL_DOMAIN)
PLOT_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--representation-dir",
        type=Path,
        default=WORKSPACE_DIR / "resultados_EF_repeticiones_con_inactivos_10_90",
        help="Directory containing seed_<N>/df_long_target_all_methods.csv.",
    )
    parser.add_argument(
        "--neighbors-dir",
        type=Path,
        default=WORKSPACE_DIR / "resultados_EF_vecinos_repeticiones_10_90",
        help="Retained nearest-neighbor result directory (used for split validation).",
    )
    parser.add_argument(
        "--full-domain-dir",
        type=Path,
        default=SCRIPT_DIR / "reproduction_output/full_domain_benchmark",
        help="Completed full-domain benchmark directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "reproduction_output/ecfp4_transfer_strategy_comparison",
    )
    parser.add_argument("--force", action="store_true", help="Allow replacing derived outputs.")
    parser.add_argument("--resume", action="store_true", help="Allow regenerating derived outputs.")
    return parser.parse_args()


def _require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing columns in {source}: {missing}")


def load_sequence_rows(representation_dir: Path) -> pd.DataFrame:
    frames = []
    for seed in SEEDS:
        path = representation_dir / f"seed_{seed}" / "df_long_target_all_methods.csv"
        frame = pd.read_csv(path)
        _require_columns(
            frame,
            {"target", "method_label", "percentile", "EF_cumulative", "n_pool_eval", "n_pos_eval"},
            path,
        )
        frame = frame[frame["method_label"].eq(METHOD)].copy()
        frame["seed"] = seed
        frame["strategy"] = SEQUENCE
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_transferred_rows(full_domain_dir: Path) -> pd.DataFrame:
    path = full_domain_dir / "neighbor_vs_full_domain_all_seed_rows.csv"
    frame = pd.read_csv(path)
    _require_columns(
        frame,
        {"target", "seed", "strategy", "percentile", "EF_cumulative", "n_pool_eval", "n_pos_eval"},
        path,
    )
    frame = frame[frame["strategy"].isin(("K=5", FULL_DOMAIN))].copy()
    frame["strategy"] = frame["strategy"].replace({"K=5": NEAREST_K5})
    return frame


def _complete_targets(frame: pd.DataFrame, strategy: str) -> set[str]:
    subset = frame[frame["strategy"].eq(strategy)]
    expected = len(SEEDS) * len(RAW_PERCENTILES)
    counts = subset.groupby("target")[["seed", "percentile"]].apply(
        lambda values: len(values.drop_duplicates())
    )
    return set(counts[counts.eq(expected)].index.astype(str))


def align_and_validate_rows(sequence: pd.DataFrame, transferred: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    combined = pd.concat([sequence, transferred], ignore_index=True, sort=False)
    combined["target"] = combined["target"].astype(str)
    combined["seed"] = pd.to_numeric(combined["seed"], errors="raise").astype(int)
    combined["percentile"] = pd.to_numeric(combined["percentile"], errors="raise").astype(float)
    combined["EF_cumulative"] = pd.to_numeric(combined["EF_cumulative"], errors="raise")
    keys = ["target", "seed", "percentile", "strategy"]
    duplicated = combined.duplicated(keys, keep=False)
    if duplicated.any():
        raise ValueError(f"Duplicate strategy cells: {combined.loc[duplicated, keys].head().to_dict('records')}")

    complete = [_complete_targets(combined, strategy) for strategy in STRATEGIES]
    common_targets = sorted(set.intersection(*complete))
    if not common_targets:
        raise ValueError("No target has complete results for all three strategies")
    combined = combined[combined["target"].isin(common_targets)].copy()

    counts = combined.groupby(["target", "strategy", "percentile"])["seed"].nunique()
    expected_cells = len(common_targets) * len(STRATEGIES) * len(RAW_PERCENTILES)
    if len(counts) != expected_cells or not counts.eq(len(SEEDS)).all():
        raise ValueError("At least one common target/strategy/percentile lacks five partitions")

    pool_check = combined.pivot(
        index=["target", "seed", "percentile"], columns="strategy", values="n_pool_eval"
    )
    positive_check = combined.pivot(
        index=["target", "seed", "percentile"], columns="strategy", values="n_pos_eval"
    )
    if not pool_check.nunique(axis=1).eq(1).all():
        raise ValueError("Strategies do not share identical evaluation-pool sizes")
    if not positive_check.nunique(axis=1).eq(1).all():
        raise ValueError("Strategies do not share identical positive counts")
    return combined, common_targets


def _known_sets(path: Path, *, method_label: str | None = None, neighbor_count: int | None = None) -> dict[str, tuple[str, ...]]:
    frame = pd.read_csv(path)
    _require_columns(frame, {"target_id", "known_active_ids"}, path)
    if method_label is not None:
        frame = frame[frame["method_label"].eq(method_label)]
    if neighbor_count is not None:
        frame = frame[frame["neighbor_count_sweep"].eq(neighbor_count)]
    result = {}
    for row in frame.itertuples(index=False):
        values = ast.literal_eval(row.known_active_ids) if isinstance(row.known_active_ids, str) else row.known_active_ids
        result[str(row.target_id)] = tuple(str(value) for value in values)
    return result


def validate_known_active_splits(
    representation_dir: Path,
    neighbors_dir: Path,
    full_domain_dir: Path,
    targets: list[str],
) -> None:
    for seed in SEEDS:
        sequence = _known_sets(
            representation_dir / f"seed_{seed}" / "known_active_sets_all_methods.csv",
            method_label=f"{METHOD}__target_seeds",
        )
        nearest = _known_sets(
            neighbors_dir / f"sweep_neighbors_seed_{seed}" / f"sweep_neighbors_{METHOD}_known_active_sets_all.csv",
            neighbor_count=5,
        )
        for target in targets:
            domain = _known_sets(full_domain_dir / f"seed_{seed}" / target / "known_active_sets.csv")
            values = (sequence.get(target), nearest.get(target), domain.get(target))
            if any(value is None for value in values):
                raise ValueError(f"Missing known-active split for target={target}, seed={seed}")
            if values[0] != values[1] or values[0] != values[2]:
                raise ValueError(f"Known-active split mismatch for target={target}, seed={seed}")


def summarize(combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_medians = (
        combined.groupby(["target", "strategy", "percentile"], as_index=False, observed=True)
        .agg(
            median_partition_EF_cumulative=("EF_cumulative", "median"),
            n_partitions=("seed", "nunique"),
        )
    )
    target_medians = add_family_annotations(target_medians, FAMILIES)
    if target_medians["familia"].isna().any():
        unknown = sorted(target_medians.loc[target_medians["familia"].isna(), "target"].unique())
        raise ValueError(f"Targets without a protein-family assignment: {unknown}")
    family_medians = (
        target_medians.groupby(["strategy", "percentile", "familia"], as_index=False, observed=True)
        .agg(
            family_median_EF_cumulative=("median_partition_EF_cumulative", "median"),
            n_targets=("target", "nunique"),
        )
    )
    _, spread = summarize_category_spread(
        family_medians,
        group_columns=["strategy", "percentile"],
        value_column="family_median_EF_cumulative",
    )
    return target_medians, family_medians, spread


@readable_labels
def plot_comparison(spread: pd.DataFrame, output_stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {SEQUENCE: "#4477AA", NEAREST_K5: "#228833", FULL_DOMAIN: "#CC3311"}
    line_styles = {SEQUENCE: "-", NEAREST_K5: "-", FULL_DOMAIN: "--"}
    x = np.arange(len(PLOT_PERCENTILES))
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    for strategy in STRATEGIES:
        rows = spread[spread["strategy"].eq(strategy)].set_index("percentile")
        missing = [value for value in PLOT_PERCENTILES if value not in rows.index]
        if missing:
            raise ValueError(f"Missing plotted percentiles for {strategy}: {missing}")
        rows = rows.loc[list(PLOT_PERCENTILES)]
        median = rows["category_balanced_median"].to_numpy(dtype=float)
        q25 = rows["category_q25"].to_numpy(dtype=float)
        q75 = rows["category_q75"].to_numpy(dtype=float)
        ax.plot(
            x,
            median,
            marker="o",
            markersize=6,
            linewidth=2.5,
            linestyle=line_styles[strategy],
            color=colors[strategy],
            label=strategy,
        )
        ax.fill_between(x, q25, q75, color=colors[strategy], alpha=0.15, linewidth=0)
        ef2_index = PLOT_PERCENTILES.index(98.0)
        ax.scatter(
            [ef2_index], [median[ef2_index]], s=75, color=colors[strategy],
            edgecolor="white", linewidth=1.1, zorder=4,
        )

    screened = [100.0 - value for value in PLOT_PERCENTILES]
    tick_labels = [f"{value:g}\n({fraction:g}%)" for value, fraction in zip(PLOT_PERCENTILES, screened)]
    ax.set_xticks(x, tick_labels)
    ax.set_xlabel("Percentile threshold (screened fraction)")
    ax.set_ylabel("Category-balanced median\ncumulative enrichment factor")
    ax.axvline(PLOT_PERCENTILES.index(98.0), color="0.45", linestyle=":", linewidth=1.0, alpha=0.75)
    ax.text(
        PLOT_PERCENTILES.index(98.0), 1.02, "EF2%", transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", color="0.35", fontsize=plt.rcParams["font.size"],
    )
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.15, edgecolor="none"))
    labels.append("Category IQR")
    ax.legend(handles, labels, title="Ligand evidence strategy", frameon=False)
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    representation_dir = args.representation_dir.expanduser().resolve()
    neighbors_dir = args.neighbors_dir.expanduser().resolve()
    full_domain_dir = args.full_domain_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)

    sequence = load_sequence_rows(representation_dir)
    transferred = load_transferred_rows(full_domain_dir)
    combined, common_targets = align_and_validate_rows(sequence, transferred)
    validate_known_active_splits(representation_dir, neighbors_dir, full_domain_dir, common_targets)
    target_medians, family_medians, spread = summarize(combined)

    combined.to_csv(output_dir / "ecfp4_transfer_strategy_all_partition_rows.csv", index=False)
    target_medians.to_csv(output_dir / "ecfp4_transfer_strategy_target_medians.csv", index=False)
    family_medians.to_csv(output_dir / "ecfp4_transfer_strategy_family_medians.csv", index=False)
    spread.to_csv(output_dir / "ecfp4_transfer_strategy_category_balanced_spread.csv", index=False)
    target_medians[["target", "familia", "subfamilia"]].drop_duplicates().sort_values(
        ["familia", "target"]
    ).to_csv(output_dir / "common_targets.csv", index=False)
    plot_comparison(spread, output_dir / "ecfp4_transfer_strategy_comparison")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "representation": "ECFP4/Morgan radius 2, 1024 bits",
        "metric": "Tanimoto",
        "ef": "cumulative",
        "strategies": list(STRATEGIES),
        "seeds": list(SEEDS),
        "plot_percentiles": list(PLOT_PERCENTILES),
        "common_target_count": len(common_targets),
        "protein_family_count": int(family_medians["familia"].nunique()),
        "aggregation": "median across partitions within target; median across targets within family; median and IQR across family medians",
        "seed_budget": "same requested maximum: number of target known actives; transferred strategies may use fewer when eligible ligands are insufficient",
        "inputs": {
            "representation_dir": str(representation_dir),
            "neighbors_dir": str(neighbors_dir),
            "full_domain_dir": str(full_domain_dir),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"Generated ECFP4 strategy comparison for {len(common_targets)} targets, "
        f"{len(SEEDS)} partitions, and {metadata['protein_family_count']} protein families: {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
