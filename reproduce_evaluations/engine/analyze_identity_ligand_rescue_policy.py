#!/usr/bin/env python3
"""Tune an identity cutoff with a ligand-evidence rescue rule.

The identity cutoff first determines the biologically supported neighbor
prefix. If that prefix does not provide the requested number of cleaned
candidate ligands, progressively more neighbors are included until the
ligand threshold is reached. K=15 is used when the threshold is never met.

This derived analysis reuses the completed K=2..15 results and does not rerun
molecular searches or use enrichment factors to choose K for a target.
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

from analyze_min_k_ligand_budget_sensitivity import (
    family_mapping,
    load_matrices,
    parse_int_list,
)
from ligq2_evaluation.constants import SEEDS
from ligq2_evaluation.runtime import prepare_output


DEFAULT_IDENTITY_FLOORS = tuple(range(10, 71, 5))
DEFAULT_LIGAND_BUDGETS = tuple(range(50, 501, 50))


def parse_float_list(value: str, label: str) -> list[float]:
    values = sorted({float(token.strip()) for token in value.split(",") if token.strip()})
    if not values or any(item <= 0 or item > 100 for item in values):
        raise ValueError(f"{label} must contain values in (0,100]")
    return values


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=script_dir / "reproduction_output/neighbor_k_complete",
    )
    parser.add_argument(
        "--adaptive-dir",
        type=Path,
        default=script_dir / "reproduction_output/adaptive_k",
    )
    parser.add_argument(
        "--blast-hits",
        type=Path,
        default=script_dir / "reproduction_output/k_covariates/neighbor_blast_hits.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "reproduction_output/identity_ligand_rescue",
    )
    parser.add_argument(
        "--identity-floors",
        default=",".join(map(str, DEFAULT_IDENTITY_FLOORS)),
    )
    parser.add_argument(
        "--ligand-budgets",
        default=",".join(map(str, DEFAULT_LIGAND_BUDGETS)),
    )
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_ranked_identities(path: Path, targets: set[str]) -> dict[str, list[float]]:
    frame = pd.read_csv(path)
    required = {"target_name", "neighbor_rank", "pident"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BLAST hit table is missing columns: {missing}")
    frame["target"] = frame["target_name"].astype(str).str.lower()
    frame = frame[frame["target"].isin(targets)].copy()
    frame["neighbor_rank"] = pd.to_numeric(
        frame["neighbor_rank"], errors="raise"
    ).astype(int)
    frame["pident"] = pd.to_numeric(frame["pident"], errors="raise")
    if frame.duplicated(["target", "neighbor_rank"]).any():
        raise ValueError("Duplicate target/rank cells in BLAST hit table")
    absent = sorted(targets - set(frame["target"]))
    if absent:
        raise ValueError(f"Targets absent from BLAST hit table: {absent[:10]}")
    return {
        target: rows.sort_values("neighbor_rank")["pident"].astype(float).tolist()
        for target, rows in frame.groupby("target", sort=False)
    }


def identity_prefix_k(
    ranked_identities: list[float],
    *,
    floor: float,
    min_k: int,
    max_k: int,
) -> int:
    """Return the retained identity-supported prefix, bounded below by min_k."""
    available = min(len(ranked_identities), max_k)
    if available <= min_k:
        return min_k
    selected = min_k
    for next_rank in range(min_k + 1, available + 1):
        if float(ranked_identities[next_rank - 1]) < floor:
            return selected
        selected = next_rank
    return selected


def ligand_budget_k(
    candidate_values: np.ndarray,
    neighbor_counts: np.ndarray,
    *,
    budget: int,
    min_k: int,
    max_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = (neighbor_counts >= min_k) & (neighbor_counts <= max_k)
    eligible = (candidate_values >= budget) & allowed[None, :]
    reached = eligible.any(axis=1)
    first_positions = np.argmax(eligible, axis=1)
    selected = np.where(reached, neighbor_counts[first_positions], max_k).astype(int)
    return selected, reached


def combine_identity_and_budget_k(
    identity_k: np.ndarray,
    budget_k: np.ndarray,
) -> np.ndarray:
    """Retain the identity prefix and extend it when ligand evidence is sparse."""
    return np.maximum(identity_k, budget_k).astype(int)


def evaluate_grid(
    candidate_matrix: pd.DataFrame,
    ef_matrix: pd.DataFrame,
    identities: dict[str, list[float]],
    *,
    identity_floors: list[float],
    ligand_budgets: list[int],
    min_k: int,
    max_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping, _ = family_mapping()
    neighbor_counts = candidate_matrix.columns.to_numpy(dtype=int)
    candidates = candidate_matrix.to_numpy(dtype=float)
    ef_values = ef_matrix.to_numpy(dtype=float)
    targets_array = candidate_matrix.index.get_level_values("target").to_numpy()
    seeds_array = candidate_matrix.index.get_level_values("seed").to_numpy(dtype=int)
    unique_targets = sorted(set(targets_array))

    choice_frames = []
    target_frames = []
    family_frames = []
    summary_rows = []
    for floor in identity_floors:
        identity_by_target = {
            target: identity_prefix_k(
                identities[target], floor=floor, min_k=min_k, max_k=max_k
            )
            for target in unique_targets
        }
        identity_k_values = np.array(
            [identity_by_target[target] for target in targets_array], dtype=int
        )
        for budget in ligand_budgets:
            budget_k_values, reached = ligand_budget_k(
                candidates,
                neighbor_counts,
                budget=budget,
                min_k=min_k,
                max_k=max_k,
            )
            selected_k = combine_identity_and_budget_k(
                identity_k_values, budget_k_values
            )
            column_lookup = {int(k): pos for pos, k in enumerate(neighbor_counts)}
            positions = np.array([column_lookup[int(k)] for k in selected_k], dtype=int)
            selected_ef = ef_values[np.arange(len(ef_values)), positions]
            choices = pd.DataFrame({
                "target": targets_array,
                "seed": seeds_array,
                "identity_floor": floor,
                "ligand_budget": budget,
                "identity_prefix_k": identity_k_values,
                "ligand_budget_k": budget_k_values,
                "selected_k": selected_k,
                "budget_reached": reached,
                "EF_cumulative": selected_ef,
            })
            targets = (
                choices.groupby("target", as_index=False)
                .agg(
                    median_partition_EF_cumulative=("EF_cumulative", "median"),
                    median_identity_prefix_k=("identity_prefix_k", "median"),
                    median_ligand_budget_k=("ligand_budget_k", "median"),
                    median_selected_k=("selected_k", "median"),
                    min_selected_k=("selected_k", "min"),
                    max_selected_k=("selected_k", "max"),
                    budget_reached_fraction=("budget_reached", "mean"),
                    n_partitions=("seed", "nunique"),
                )
            )
            targets["protein_family"] = targets["target"].map(mapping)
            if targets["protein_family"].isna().any():
                missing = sorted(targets.loc[targets["protein_family"].isna(), "target"])
                raise ValueError(f"Targets lack family assignments: {missing}")
            targets["identity_floor"] = floor
            targets["ligand_budget"] = budget
            families = (
                targets.groupby("protein_family", as_index=False)
                .agg(
                    family_median_EF_cumulative=(
                        "median_partition_EF_cumulative", "median"
                    ),
                    n_targets=("target", "nunique"),
                )
            )
            families["identity_floor"] = floor
            families["ligand_budget"] = budget
            family_ef = families["family_median_EF_cumulative"]
            summary_rows.append({
                "identity_floor": floor,
                "ligand_budget": budget,
                "category_balanced_median_EF_cumulative": float(family_ef.median()),
                "category_q25_EF_cumulative": float(family_ef.quantile(0.25)),
                "category_q75_EF_cumulative": float(family_ef.quantile(0.75)),
                "all_target_median_EF_cumulative": float(
                    targets["median_partition_EF_cumulative"].median()
                ),
                "median_selected_k": float(targets["median_selected_k"].median()),
                "budget_reached_fraction": float(choices["budget_reached"].mean()),
                "n_targets": int(targets["target"].nunique()),
                "n_categories": int(families["protein_family"].nunique()),
            })
            choice_frames.append(choices)
            target_frames.append(targets)
            family_frames.append(families)
    return (
        pd.concat(choice_frames, ignore_index=True),
        pd.concat(target_frames, ignore_index=True),
        pd.concat(family_frames, ignore_index=True),
        pd.DataFrame(summary_rows),
    )


def plot_heatmap(summary: pd.DataFrame, output_dir: Path, dpi: int) -> list[Path]:
    matrix = summary.pivot(
        index="identity_floor",
        columns="ligand_budget",
        values="category_balanced_median_EF_cumulative",
    ).sort_index(ascending=False)
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    values = matrix.to_numpy()
    image = axis.imshow(values, aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    axis.set_yticks(range(len(matrix.index)), labels=[f"{x:g}" for x in matrix.index])
    axis.set_xlabel("Cleaned candidate-ligand threshold")
    axis.set_ylabel("BLAST identity floor (%)")
    axis.set_title("Identity cutoff with ligand-evidence rescue at EF0.5%")
    median_value = np.nanmedian(values)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            color = "black" if values[row, column] > median_value else "white"
            axis.text(
                column, row, f"{values[row, column]:.1f}",
                ha="center", va="center", color=color, fontsize=7,
            )
    best_position = np.unravel_index(np.nanargmax(values), values.shape)
    axis.scatter(
        [best_position[1]], [best_position[0]], marker="s", s=600,
        facecolors="none", edgecolors="#E15759", linewidths=2.2,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Category-balanced median cumulative EF")
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"identity_ligand_rescue_EF0.5_heatmap.{suffix}"
        figure.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    floors = parse_float_list(args.identity_floors, "Identity floors")
    budgets = parse_int_list(args.ligand_budgets, "Ligand budgets")
    if not 2 <= args.min_k <= args.max_k <= 15:
        raise ValueError("Require 2 <= min-k <= max-k <= 15")
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)
    candidates, ef = load_matrices(
        args.sweep_dir.expanduser().resolve(),
        args.adaptive_dir.expanduser().resolve(),
        percentile=args.percentile,
        max_k=args.max_k,
    )
    identities = load_ranked_identities(
        args.blast_hits.expanduser().resolve(),
        set(candidates.index.get_level_values("target")),
    )
    choices, targets, families, summary = evaluate_grid(
        candidates,
        ef,
        identities,
        identity_floors=floors,
        ligand_budgets=budgets,
        min_k=args.min_k,
        max_k=args.max_k,
    )
    ranking = summary.sort_values(
        ["category_balanced_median_EF_cumulative", "identity_floor", "ligand_budget"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    choices.to_csv(output_dir / "identity_ligand_rescue_choices.csv", index=False)
    targets.to_csv(output_dir / "identity_ligand_rescue_target_summary.csv", index=False)
    families.to_csv(output_dir / "identity_ligand_rescue_family_summary.csv", index=False)
    summary.to_csv(output_dir / "identity_ligand_rescue_grid_summary.csv", index=False)
    ranking.to_csv(output_dir / "identity_ligand_rescue_ranking.csv", index=False)
    figure_paths = plot_heatmap(summary, output_dir, args.dpi)
    best = ranking.iloc[0].to_dict()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_percentile": args.percentile,
        "screened_fraction_percent": 100.0 - args.percentile,
        "minimum_k": args.min_k,
        "maximum_k": args.max_k,
        "identity_floors_percent": floors,
        "ligand_candidate_budgets": budgets,
        "partition_seeds": list(SEEDS),
        "selection_uses_EF": False,
        "grid_maximum_selected_post_hoc_using_EF": True,
        "selection_rule": (
            "Retain the consecutive neighbor prefix supported by the identity floor. "
            "If its cleaned candidate pool is below the ligand threshold, extend K "
            "until the threshold is reached; use K=15 if it is never reached."
        ),
        "best_grid_cell": best,
        "figures": [str(path) for path in figure_paths],
        "interpretation_warning": (
            "The best grid cell is a retrospective benchmark optimum and must not be "
            "presented as an externally validated universal default."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(
        "Best identity-plus-ligand rescue policy: "
        f"identity floor={best['identity_floor']:g}%, "
        f"ligand budget={int(best['ligand_budget'])}, "
        "category-balanced median cumulative "
        f"EF{100.0 - args.percentile:g}%="
        f"{best['category_balanced_median_EF_cumulative']:.6f}"
    )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
