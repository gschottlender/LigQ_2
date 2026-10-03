#!/usr/bin/env python3
"""Evaluate ligand-budget policies across a grid of minimum K values.

This is a derived analysis: it reuses the completed K=2..15 enrichment-factor
results and the cleaned ligand-candidate counts. It does not rerun molecular
similarity searches or use enrichment factors when selecting K for a target.
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

from ligq2_evaluation.constants import FAMILIES, FAMILY_LABELS, FAMILY_ORDER, SEEDS
from ligq2_evaluation.runtime import prepare_output


DEFAULT_MIN_K_VALUES = tuple(range(2, 16))
DEFAULT_LIGAND_BUDGETS = tuple(range(50, 501, 50))


def parse_int_list(value: str, label: str) -> list[int]:
    values = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not values or any(item <= 0 for item in values):
        raise ValueError(f"{label} must contain positive integers")
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
        help="Directory containing target_partition_k_covariates.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "reproduction_output/min_k_ligand_budget_sensitivity",
    )
    parser.add_argument(
        "--min-k-values",
        default=",".join(map(str, DEFAULT_MIN_K_VALUES)),
    )
    parser.add_argument(
        "--ligand-budgets",
        default=",".join(map(str, DEFAULT_LIGAND_BUDGETS)),
        help="Cleaned candidate-ligand stopping thresholds",
    )
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def family_mapping() -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    order: list[str] = []
    for family in FAMILY_ORDER:
        label = FAMILY_LABELS.get(family, family)
        order.append(label)
        for targets in FAMILIES[family].values():
            for target in targets:
                if target in mapping and mapping[target] != label:
                    raise ValueError(f"Target {target!r} belongs to multiple families")
                mapping[target] = label
    return mapping, order


def load_matrices(
    sweep_dir: Path,
    adaptive_dir: Path,
    *,
    percentile: float,
    max_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    covariate_path = adaptive_dir / "target_partition_k_covariates.csv"
    metric_path = sweep_dir / "all_seeds_valid_rows.csv"
    if not covariate_path.is_file():
        raise FileNotFoundError(f"Missing adaptive covariates: {covariate_path}")
    if not metric_path.is_file():
        raise FileNotFoundError(f"Missing complete K sweep: {metric_path}")

    covariates = pd.read_csv(covariate_path)
    metrics = pd.read_csv(metric_path)
    required_covariates = {
        "target", "seed", "neighbor_count_sweep", "n_neighbor_candidates_clean"
    }
    required_metrics = {
        "target", "seed", "neighbor_count_sweep", "percentile", "EF_cumulative"
    }
    for frame, required, label in (
        (covariates, required_covariates, "covariates"),
        (metrics, required_metrics, "metrics"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Missing {label} columns: {missing}")
        frame["target"] = frame["target"].astype(str).str.lower()
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        frame["neighbor_count_sweep"] = pd.to_numeric(
            frame["neighbor_count_sweep"], errors="raise"
        ).astype(int)

    counts = list(range(2, max_k + 1))
    metrics = metrics[np.isclose(metrics["percentile"], percentile)].copy()
    index_columns = ["target", "seed"]
    candidate_matrix = (
        covariates.pivot(
            index=index_columns,
            columns="neighbor_count_sweep",
            values="n_neighbor_candidates_clean",
        )
        .sort_index()
        .reindex(columns=counts)
    )
    ef_matrix = (
        metrics.pivot(
            index=index_columns,
            columns="neighbor_count_sweep",
            values="EF_cumulative",
        )
        .sort_index()
        .reindex(columns=counts)
    )
    if candidate_matrix.isna().any().any() or ef_matrix.isna().any().any():
        raise ValueError("The target/partition/K grid is incomplete")
    if not candidate_matrix.index.equals(ef_matrix.index):
        raise ValueError("Covariate and EF target/partition indices differ")
    observed_seeds = set(candidate_matrix.index.get_level_values("seed"))
    if observed_seeds != set(SEEDS):
        raise ValueError(f"Unexpected partition seeds: {sorted(observed_seeds)}")
    return candidate_matrix, ef_matrix


def choose_k(
    candidate_values: np.ndarray,
    neighbor_counts: np.ndarray,
    *,
    min_k: int,
    max_k: int,
    budget: int,
) -> tuple[np.ndarray, np.ndarray]:
    allowed = (neighbor_counts >= min_k) & (neighbor_counts <= max_k)
    eligible = (candidate_values >= budget) & allowed[None, :]
    reached = eligible.any(axis=1)
    first_positions = np.argmax(eligible, axis=1)
    selected = np.where(reached, neighbor_counts[first_positions], max_k).astype(int)
    return selected, reached


def evaluate_grid(
    candidate_matrix: pd.DataFrame,
    ef_matrix: pd.DataFrame,
    *,
    min_k_values: list[int],
    ligand_budgets: list[int],
    max_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping, _ = family_mapping()
    neighbor_counts = candidate_matrix.columns.to_numpy(dtype=int)
    candidates = candidate_matrix.to_numpy(dtype=float)
    ef_values = ef_matrix.to_numpy(dtype=float)
    target_values = candidate_matrix.index.get_level_values("target").to_numpy()
    seed_values = candidate_matrix.index.get_level_values("seed").to_numpy(dtype=int)

    choice_frames = []
    target_frames = []
    family_frames = []
    summary_rows = []
    for min_k in min_k_values:
        for budget in ligand_budgets:
            selected_k, reached = choose_k(
                candidates,
                neighbor_counts,
                min_k=min_k,
                max_k=max_k,
                budget=budget,
            )
            column_positions = selected_k - int(neighbor_counts.min())
            selected_ef = ef_values[np.arange(len(ef_values)), column_positions]
            choices = pd.DataFrame({
                "target": target_values,
                "seed": seed_values,
                "min_k": min_k,
                "max_k": max_k,
                "ligand_budget": budget,
                "selected_k": selected_k,
                "budget_reached": reached,
                "EF_cumulative": selected_ef,
            })
            targets = (
                choices.groupby("target", as_index=False)
                .agg(
                    median_partition_EF_cumulative=("EF_cumulative", "median"),
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
            targets["min_k"] = min_k
            targets["max_k"] = max_k
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
            families["min_k"] = min_k
            families["max_k"] = max_k
            families["ligand_budget"] = budget
            family_ef = families["family_median_EF_cumulative"]
            summary_rows.append({
                "min_k": min_k,
                "max_k": max_k,
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
        index="min_k",
        columns="ligand_budget",
        values="category_balanced_median_EF_cumulative",
    ).sort_index()
    figure, axis = plt.subplots(figsize=(10.5, 6.5))
    image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis")
    axis.set_xticks(range(len(matrix.columns)), labels=matrix.columns)
    axis.set_yticks(range(len(matrix.index)), labels=matrix.index)
    axis.set_xlabel("Cleaned candidate-ligand threshold")
    axis.set_ylabel("Minimum K")
    axis.set_title("Ligand-budget policy sensitivity at EF0.5%")
    best_position = np.unravel_index(np.nanargmax(matrix.to_numpy()), matrix.shape)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iloc[row, column]
            color = "black" if value > np.nanmedian(matrix.to_numpy()) else "white"
            axis.text(column, row, f"{value:.1f}", ha="center", va="center", color=color, fontsize=7)
    axis.scatter(
        [best_position[1]], [best_position[0]], marker="s", s=600,
        facecolors="none", edgecolors="#E15759", linewidths=2.2,
    )
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Category-balanced median cumulative EF")
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"min_k_ligand_budget_EF0.5_heatmap.{suffix}"
        figure.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    min_k_values = parse_int_list(args.min_k_values, "Minimum K values")
    ligand_budgets = parse_int_list(args.ligand_budgets, "Ligand budgets")
    if min(min_k_values) < 2 or max(min_k_values) > args.max_k:
        raise ValueError("Require 2 <= every minimum K <= max-k")
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)
    candidates, ef = load_matrices(
        args.sweep_dir.expanduser().resolve(),
        args.adaptive_dir.expanduser().resolve(),
        percentile=args.percentile,
        max_k=args.max_k,
    )
    choices, targets, families, summary = evaluate_grid(
        candidates,
        ef,
        min_k_values=min_k_values,
        ligand_budgets=ligand_budgets,
        max_k=args.max_k,
    )
    ranking = summary.sort_values(
        ["category_balanced_median_EF_cumulative", "min_k", "ligand_budget"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    choices.to_csv(output_dir / "min_k_ligand_budget_choices.csv", index=False)
    targets.to_csv(output_dir / "min_k_ligand_budget_target_summary.csv", index=False)
    families.to_csv(output_dir / "min_k_ligand_budget_family_summary.csv", index=False)
    summary.to_csv(output_dir / "min_k_ligand_budget_grid_summary.csv", index=False)
    ranking.to_csv(output_dir / "min_k_ligand_budget_ranking.csv", index=False)
    figure_paths = plot_heatmap(summary, output_dir, args.dpi)
    best = ranking.iloc[0].to_dict()
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "endpoint_percentile": args.percentile,
        "screened_fraction_percent": 100.0 - args.percentile,
        "min_k_values": min_k_values,
        "max_k": args.max_k,
        "ligand_candidate_budgets": ligand_budgets,
        "partition_seeds": list(SEEDS),
        "selection_uses_EF": False,
        "grid_maximum_selected_post_hoc_using_EF": True,
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
        "Best ligand-budget policy: "
        f"min-K={int(best['min_k'])}, budget={int(best['ligand_budget'])}, "
        "category-balanced median cumulative "
        f"EF{100.0 - args.percentile:g}%="
        f"{best['category_balanced_median_EF_cumulative']:.6f}"
    )
    print(f"Outputs: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
