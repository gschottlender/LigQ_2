#!/usr/bin/env python3
"""Evaluate adaptive nearest-neighbor K policies from a completed K=2..15 sweep.

The policy is selected without using enrichment-factor values. Sequence
policies stop before adding a neighbor below a BLAST identity floor. Ligand
evidence policies select the smallest K whose cleaned candidate pool reaches
the requested threshold and fall back to K=15 when it never does.
"""

from __future__ import annotations

import argparse
import ast
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


DEFAULT_K = tuple(range(2, 16))
DEFAULT_IDENTITY_FLOORS = (40.0, 45.0, 50.0, 55.0, 60.0)
DEFAULT_LIGAND_BUDGETS = (50, 75, 100)
METRIC_KEYS = ("target", "seed", "neighbor_count_sweep", "percentile")
META_KEYS = ("target", "seed", "neighbor_count_sweep")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir", type=Path,
        default=script_dir / "reproduction_output/neighbor_k_complete",
        help="Output directory produced by 14_complete_neighbor_k_sweep.py",
    )
    parser.add_argument(
        "--blast-hits", type=Path,
        default=script_dir / "reproduction_output/k_covariates/neighbor_blast_hits.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=script_dir / "reproduction_output/adaptive_k",
    )
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=15)
    parser.add_argument(
        "--identity-floors", default=",".join(map(str, DEFAULT_IDENTITY_FLOORS)),
        help="Comma-separated BLAST percentage-identity floors",
    )
    parser.add_argument(
        "--ligand-budgets", default=",".join(map(str, DEFAULT_LIGAND_BUDGETS)),
        help="Comma-separated cleaned candidate-ligand thresholds",
    )
    parser.add_argument("--percentile", type=float, default=99.5)
    parser.add_argument(
        "--include-combined-policies", action="store_true",
        help="Also evaluate policies requiring a ligand budget subject to an identity floor",
    )
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_numbers(value: str, cast, label: str) -> list:
    numbers = sorted({cast(token.strip()) for token in value.split(",") if token.strip()})
    if not numbers or any(number <= 0 for number in numbers):
        raise ValueError(f"{label} must contain positive values")
    return numbers


def family_mapping() -> tuple[dict[str, str], list[str]]:
    mapping: dict[str, str] = {}
    order = []
    for family in FAMILY_ORDER:
        label = FAMILY_LABELS.get(family, family)
        order.append(label)
        for targets in FAMILIES[family].values():
            for target in targets:
                if target in mapping and mapping[target] != label:
                    raise ValueError(f"Target {target!r} belongs to multiple families")
                mapping[target] = label
    return mapping, order


def parse_neighbor_ids(value) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if pd.isna(value):
        return []
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(f"Neighbor IDs are not a list: {value!r}")
    return [str(item) for item in parsed]


def load_complete_sweep(
    sweep_dir: Path,
    *,
    neighbor_counts: list[int],
    expected_seeds: tuple[int, ...] = SEEDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_path = sweep_dir / "all_seeds_valid_rows.csv"
    metadata_path = sweep_dir / "all_seeds_seed_metadata.csv"
    if not metrics_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Complete sweep outputs are missing. Run 14_complete_neighbor_k_sweep.py first."
        )
    metrics = pd.read_csv(metrics_path)
    metadata = pd.read_csv(metadata_path)
    metric_required = {
        *METRIC_KEYS, "EF_band", "EF_cumulative", "n_seed_requested",
        "n_seed_effective", "n_pool_eval", "n_pos_eval",
    }
    meta_required = {
        *META_KEYS, "neighbor_ids_considered", "top_n_neighbors_considered",
        "top_n_neighbors_available", "n_neighbor_candidates_clean",
        "k_requested", "k_effective",
    }
    for frame, required, label in (
        (metrics, metric_required, "metrics"),
        (metadata, meta_required, "seed metadata"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"Complete-sweep {label} is missing columns: {missing}")
        frame["target"] = frame["target"].astype(str).str.lower()
        frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
        frame["neighbor_count_sweep"] = pd.to_numeric(
            frame["neighbor_count_sweep"], errors="raise"
        ).astype(int)
        frame.drop(frame[~frame["neighbor_count_sweep"].isin(neighbor_counts)].index, inplace=True)
    if metrics.duplicated(list(METRIC_KEYS)).any():
        raise ValueError("Duplicate target/partition/K/percentile rows in complete sweep")
    if metadata.duplicated(list(META_KEYS)).any():
        raise ValueError("Duplicate target/partition/K rows in complete-sweep metadata")

    expected_k = set(neighbor_counts)
    expected_seed_set = set(expected_seeds)
    for label, frame in (("metrics", metrics), ("metadata", metadata)):
        grouped = frame.groupby("target").agg(
            seeds=("seed", lambda values: set(map(int, values))),
            counts=("neighbor_count_sweep", lambda values: set(map(int, values))),
        )
        bad = grouped[
            grouped["seeds"].map(lambda values: values != expected_seed_set)
            | grouped["counts"].map(lambda values: values != expected_k)
        ]
        if not bad.empty:
            raise ValueError(f"Incomplete target grid in {label}: {bad.index.tolist()[:10]}")
    metric_targets = set(metrics["target"])
    meta_targets = set(metadata["target"])
    if metric_targets != meta_targets:
        raise ValueError("Metric and metadata target sets differ")
    return metrics.reset_index(drop=True), metadata.reset_index(drop=True)


def load_blast_hits(path: Path, targets: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"target_name", "sseqid", "neighbor_rank", "pident", "qcovs", "evalue", "bitscore"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BLAST hit table is missing columns: {missing}")
    frame["target"] = frame["target_name"].astype(str).str.lower()
    frame = frame[frame["target"].isin(targets)].copy()
    frame["neighbor_rank"] = pd.to_numeric(frame["neighbor_rank"], errors="raise").astype(int)
    frame["pident"] = pd.to_numeric(frame["pident"], errors="raise")
    if frame.duplicated(["target", "neighbor_rank"]).any():
        raise ValueError("Duplicate target/rank cells in BLAST hit table")
    absent = sorted(targets - set(frame["target"]))
    if absent:
        raise ValueError(f"Targets absent from BLAST hit table: {absent[:10]}")
    return frame.sort_values(["target", "neighbor_rank"]).reset_index(drop=True)


def build_k_covariates(
    metadata: pd.DataFrame,
    blast_hits: pd.DataFrame,
) -> pd.DataFrame:
    identity_lookup = {
        target: rows.set_index("neighbor_rank")["pident"].to_dict()
        for target, rows in blast_hits.groupby("target", sort=False)
    }
    hit_id_lookup = {
        target: rows.set_index("neighbor_rank")["sseqid"].astype(str).to_dict()
        for target, rows in blast_hits.groupby("target", sort=False)
    }
    rows = []
    mismatches = []
    for record in metadata.to_dict("records"):
        target = str(record["target"])
        count = int(record["neighbor_count_sweep"])
        ids = parse_neighbor_ids(record["neighbor_ids_considered"])
        expected_ids = [
            hit_id_lookup[target][rank]
            for rank in range(1, min(count, len(hit_id_lookup[target])) + 1)
            if rank in hit_id_lookup[target]
        ]
        if ids != expected_ids[:len(ids)]:
            mismatches.append({
                "target": target,
                "seed": int(record["seed"]),
                "neighbor_count_sweep": count,
                "metadata_ids": ";".join(ids),
                "blast_prefix_ids": ";".join(expected_ids[:len(ids)]),
            })
        actual_count = len(ids)
        identities = [identity_lookup[target].get(rank) for rank in range(1, actual_count + 1)]
        if any(value is None for value in identities):
            raise ValueError(f"Missing BLAST identity within retained prefix for {target}, K={count}")
        rows.append({
            **record,
            "actual_neighbor_count": actual_count,
            "actual_boundary_pident": identities[-1] if identities else np.nan,
            "minimum_pident_first_actual_neighbors": min(identities) if identities else np.nan,
            "nominal_k_boundary_pident": identity_lookup[target].get(count, np.nan),
            "reached_nominal_k": actual_count >= count,
        })
    if mismatches:
        example = mismatches[:3]
        raise ValueError(f"Retained neighbor IDs do not match BLAST ranking: {example}")
    return pd.DataFrame(rows)


def choose_identity_k(
    ranked_identities: list[float],
    *,
    floor: float,
    min_k: int,
    max_k: int,
) -> tuple[int, str, float]:
    """Choose the largest admissible K, retaining min_k as the lower bound."""
    available = min(len(ranked_identities), max_k)
    if available <= min_k:
        return min_k, "all_available_neighbors_used", np.nan
    selected = min_k
    for next_rank in range(min_k + 1, available + 1):
        next_identity = float(ranked_identities[next_rank - 1])
        if next_identity < floor:
            return selected, "next_neighbor_below_identity_floor", next_identity
        selected = next_rank
    if available < max_k:
        return selected, "all_available_neighbors_used", np.nan
    return max_k, "maximum_k_reached", np.nan


def choose_ligand_budget_k(
    rows: pd.DataFrame,
    *,
    budget: int,
    min_k: int,
    max_k: int,
) -> tuple[int, str, bool]:
    eligible = rows[
        rows["neighbor_count_sweep"].between(min_k, max_k)
        & rows["n_neighbor_candidates_clean"].ge(budget)
    ].sort_values("neighbor_count_sweep")
    if not eligible.empty:
        return int(eligible.iloc[0]["neighbor_count_sweep"]), "ligand_budget_reached", True
    return max_k, "maximum_k_fallback_budget_not_reached", False


def choose_combined_k(
    rows: pd.DataFrame,
    ranked_identities: list[float],
    *,
    budget: int,
    floor: float,
    min_k: int,
    max_k: int,
) -> tuple[int, str, bool, float]:
    indexed = rows.set_index("neighbor_count_sweep")
    for count in range(min_k, max_k + 1):
        if float(indexed.loc[count, "n_neighbor_candidates_clean"]) >= budget:
            return count, "ligand_budget_reached", True, np.nan
        next_rank = count + 1
        if next_rank <= min(len(ranked_identities), max_k):
            next_identity = float(ranked_identities[next_rank - 1])
            if next_identity < floor:
                return count, "identity_floor_reached_before_budget", False, next_identity
    return max_k, "maximum_k_fallback_budget_not_reached", False, np.nan


def build_policy_choices(
    metadata: pd.DataFrame,
    blast_hits: pd.DataFrame,
    *,
    min_k: int,
    max_k: int,
    identity_floors: list[float],
    ligand_budgets: list[int],
    include_combined: bool,
) -> pd.DataFrame:
    identities_by_target = {
        target: rows.sort_values("neighbor_rank")["pident"].astype(float).tolist()
        for target, rows in blast_hits.groupby("target", sort=False)
    }
    records = []
    for (target, seed), rows in metadata.groupby(["target", "seed"], sort=True):
        rows = rows.sort_values("neighbor_count_sweep")
        indexed = rows.set_index("neighbor_count_sweep")
        if set(range(min_k, max_k + 1)) - set(indexed.index.astype(int)):
            raise ValueError(f"Missing adaptive K cells for {target}, seed={seed}")

        def add_choice(
            policy: str,
            policy_type: str,
            selected_k: int,
            reason: str,
            *,
            identity_floor=np.nan,
            ligand_budget=np.nan,
            budget_reached=np.nan,
            next_excluded_pident=np.nan,
        ) -> None:
            selected = indexed.loc[int(selected_k)]
            records.append({
                "target": target,
                "seed": int(seed),
                "policy": policy,
                "policy_type": policy_type,
                "selected_k": int(selected_k),
                "selection_reason": reason,
                "identity_floor": identity_floor,
                "ligand_budget": ligand_budget,
                "budget_reached": budget_reached,
                "next_excluded_pident": next_excluded_pident,
                "actual_neighbor_count": int(selected["actual_neighbor_count"]),
                "actual_boundary_pident": selected["actual_boundary_pident"],
                "nominal_k_boundary_pident": selected["nominal_k_boundary_pident"],
                "n_neighbor_candidates_clean": int(selected["n_neighbor_candidates_clean"]),
                "requested_seed_budget": int(selected["k_requested"]),
                "effective_seed_count": int(selected["k_effective"]),
            })

        for count in range(2, max_k + 1):
            add_choice(f"fixed_k_{count}", "fixed_k", count, "fixed_k")

        identities = identities_by_target[target]
        for floor in identity_floors:
            count, reason, excluded_identity = choose_identity_k(
                identities, floor=floor, min_k=min_k, max_k=max_k
            )
            add_choice(
                f"identity_floor_{floor:g}", "identity_floor", count, reason,
                identity_floor=floor, next_excluded_pident=excluded_identity,
            )

        for budget in ligand_budgets:
            count, reason, reached = choose_ligand_budget_k(
                rows, budget=budget, min_k=min_k, max_k=max_k
            )
            add_choice(
                f"ligand_budget_{budget}", "ligand_budget", count, reason,
                ligand_budget=budget, budget_reached=reached,
            )

        matched_budget = int(indexed.loc[min_k, "k_requested"])
        count, reason, reached = choose_ligand_budget_k(
            rows, budget=matched_budget, min_k=min_k, max_k=max_k
        )
        add_choice(
            "matched_target_seed_budget", "matched_ligand_budget", count, reason,
            ligand_budget=matched_budget, budget_reached=reached,
        )

        if include_combined:
            for floor in identity_floors:
                for budget in ligand_budgets:
                    count, reason, reached, excluded_identity = choose_combined_k(
                        rows, identities, budget=budget, floor=floor,
                        min_k=min_k, max_k=max_k,
                    )
                    add_choice(
                        f"identity_floor_{floor:g}__ligand_budget_{budget}",
                        "combined", count, reason, identity_floor=floor,
                        ligand_budget=budget, budget_reached=reached,
                        next_excluded_pident=excluded_identity,
                    )
    choices = pd.DataFrame(records)
    if choices.duplicated(["target", "seed", "policy"]).any():
        raise ValueError("Duplicate adaptive policy choices")
    return choices


def attach_selected_results(choices: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if choices.duplicated(["target", "seed", "policy"]).any():
        raise ValueError("Duplicate target/partition/policy choices before EF merge")
    if metrics.duplicated(["target", "seed", "neighbor_count_sweep", "percentile"]).any():
        raise ValueError("Duplicate target/partition/K/percentile EF rows before policy merge")
    selected = choices.merge(
        metrics,
        left_on=["target", "seed", "selected_k"],
        right_on=["target", "seed", "neighbor_count_sweep"],
        how="left",
        # Several policies may legitimately select the same K. Metrics then
        # contribute one row per percentile to every matching policy.
        validate="many_to_many",
        suffixes=("", "_metric"),
    )
    if selected["percentile"].isna().any():
        raise ValueError("At least one adaptive choice has no matching EF result")
    if selected.duplicated(["target", "seed", "policy", "percentile"]).any():
        raise ValueError("Duplicate target/partition/policy/percentile rows after EF merge")
    percentile_count = metrics["percentile"].nunique()
    if len(selected) != len(choices) * percentile_count:
        raise ValueError(
            f"Policy/EF merge produced {len(selected)} rows; expected "
            f"{len(choices) * percentile_count}"
        )
    return selected


def summarize_selected_results(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mapping, _ = family_mapping()
    selected = selected.copy()
    selected["protein_family"] = selected["target"].map(mapping)
    missing = sorted(selected.loc[selected["protein_family"].isna(), "target"].unique())
    if missing:
        raise ValueError(f"Targets lack protein-family assignments: {missing}")
    target = (
        selected.groupby(
            ["policy", "policy_type", "target", "protein_family", "percentile"],
            as_index=False,
        )
        .agg(
            median_partition_EF_band=("EF_band", "median"),
            median_partition_EF_cumulative=("EF_cumulative", "median"),
            median_selected_k=("selected_k", "median"),
            min_selected_k=("selected_k", "min"),
            max_selected_k=("selected_k", "max"),
            n_partitions=("seed", "nunique"),
        )
    )
    family = (
        target.groupby(
            ["policy", "policy_type", "protein_family", "percentile"],
            as_index=False,
        )
        .agg(
            family_median_EF_band=("median_partition_EF_band", "median"),
            family_median_EF_cumulative=("median_partition_EF_cumulative", "median"),
            n_targets=("target", "nunique"),
        )
    )
    balanced = (
        family.groupby(["policy", "policy_type", "percentile"], as_index=False)
        .agg(
            category_balanced_median_EF_band=("family_median_EF_band", "median"),
            category_q25_EF_band=("family_median_EF_band", lambda x: x.quantile(0.25)),
            category_q75_EF_band=("family_median_EF_band", lambda x: x.quantile(0.75)),
            category_balanced_median_EF_cumulative=("family_median_EF_cumulative", "median"),
            category_q25_EF_cumulative=("family_median_EF_cumulative", lambda x: x.quantile(0.25)),
            category_q75_EF_cumulative=("family_median_EF_cumulative", lambda x: x.quantile(0.75)),
            n_categories=("protein_family", "nunique"),
        )
    )
    return target, family, balanced


def policy_diagnostics(
    choices: pd.DataFrame,
    target_summary: pd.DataFrame,
    balanced: pd.DataFrame,
    *,
    percentile: float,
) -> pd.DataFrame:
    choice_target = (
        choices.groupby(["policy", "policy_type", "target"], as_index=False)
        .agg(
            median_selected_k=("selected_k", "median"),
            median_actual_neighbor_count=("actual_neighbor_count", "median"),
            median_boundary_pident=("actual_boundary_pident", "median"),
            median_clean_candidate_ligands=("n_neighbor_candidates_clean", "median"),
            budget_reached_fraction=("budget_reached", "mean"),
            n_partitions=("seed", "nunique"),
        )
    )
    summary = (
        choice_target.groupby(["policy", "policy_type"], as_index=False)
        .agg(
            n_targets=("target", "nunique"),
            selected_k_q25=("median_selected_k", lambda x: x.quantile(0.25)),
            selected_k_median=("median_selected_k", "median"),
            selected_k_q75=("median_selected_k", lambda x: x.quantile(0.75)),
            median_actual_neighbor_count=("median_actual_neighbor_count", "median"),
            median_boundary_pident=("median_boundary_pident", "median"),
            median_clean_candidate_ligands=("median_clean_candidate_ligands", "median"),
            budget_reached_fraction=("budget_reached_fraction", "mean"),
        )
    )
    endpoint_targets = target_summary[np.isclose(target_summary["percentile"], percentile)]
    endpoint_target_ef = (
        endpoint_targets.groupby(["policy", "policy_type"], as_index=False)
        .agg(target_median_EF_cumulative=("median_partition_EF_cumulative", "median"))
    )
    endpoint_balanced = balanced[np.isclose(balanced["percentile"], percentile)][[
        "policy", "policy_type", "category_balanced_median_EF_cumulative",
        "category_q25_EF_cumulative", "category_q75_EF_cumulative", "n_categories",
    ]]
    return (
        summary.merge(endpoint_target_ef, on=["policy", "policy_type"], validate="one_to_one")
        .merge(endpoint_balanced, on=["policy", "policy_type"], validate="one_to_one")
        .sort_values(["policy_type", "policy"])
        .reset_index(drop=True)
    )


def compare_with_fixed_k5(target_summary: pd.DataFrame, *, percentile: float) -> pd.DataFrame:
    endpoint = target_summary[np.isclose(target_summary["percentile"], percentile)]
    pivot = endpoint.pivot(
        index="target", columns="policy", values="median_partition_EF_cumulative"
    )
    if "fixed_k_5" not in pivot.columns:
        raise ValueError("fixed_k_5 baseline is absent")
    records = []
    baseline = pivot["fixed_k_5"]
    for policy in pivot.columns:
        difference = pivot[policy] - baseline
        tied = np.isclose(difference, 0.0, rtol=1e-12, atol=1e-12)
        records.append({
            "policy": policy,
            "reference_policy": "fixed_k_5",
            "n_targets": int(difference.notna().sum()),
            "median_paired_delta_EF_cumulative": float(difference.median()),
            "mean_paired_delta_EF_cumulative": float(difference.mean()),
            "wins": int(((difference > 0) & ~tied).sum()),
            "ties": int(tied.sum()),
            "losses": int(((difference < 0) & ~tied).sum()),
        })
    return pd.DataFrame(records).sort_values("policy").reset_index(drop=True)


def k_selection_distribution(choices: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        choices.groupby(["policy", "policy_type", "selected_k"], as_index=False)
        .agg(n_target_partitions=("target", "size"), n_targets=("target", "nunique"))
    )
    totals = choices.groupby("policy").size().rename("total_target_partitions")
    grouped = grouped.merge(totals, on="policy", validate="many_to_one")
    grouped["fraction_target_partitions"] = (
        grouped["n_target_partitions"] / grouped["total_target_partitions"]
    )
    return grouped


def plot_fixed_k_tradeoff(
    selected: pd.DataFrame,
    choices: pd.DataFrame,
    balanced: pd.DataFrame,
    *,
    percentile: float,
    ligand_budgets: list[int],
    output_dir: Path,
    dpi: int,
) -> list[Path]:
    fixed_choices = choices[choices["policy_type"].eq("fixed_k")].copy()
    fixed_choices["K"] = fixed_choices["selected_k"].astype(int)
    target_covariates = (
        fixed_choices.groupby(["target", "K"], as_index=False)
        .agg(
            boundary_pident=("nominal_k_boundary_pident", "median"),
            clean_ligands=("n_neighbor_candidates_clean", "median"),
        )
    )
    identity = target_covariates.groupby("K")["boundary_pident"].agg(
        median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75)
    )
    endpoint = balanced[
        balanced["policy_type"].eq("fixed_k")
        & np.isclose(balanced["percentile"], percentile)
    ].copy()
    endpoint["K"] = endpoint["policy"].str.replace("fixed_k_", "", regex=False).astype(int)
    endpoint = endpoint.sort_values("K")

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    axes[0].plot(identity.index, identity["median"], marker="o", color="#4E79A7")
    axes[0].fill_between(
        identity.index.to_numpy(float), identity["q25"].to_numpy(float),
        identity["q75"].to_numpy(float), color="#4E79A7", alpha=0.20,
    )
    axes[0].set_ylabel("BLAST identity of neighbor at rank K (%)")
    axes[0].set_title("Biological proximity")

    for budget in ligand_budgets:
        coverage = (
            target_covariates.assign(reached=target_covariates["clean_ligands"].ge(budget))
            .groupby("K")["reached"].mean()
        )
        axes[1].plot(coverage.index, coverage.values, marker="o", label=f"≥{budget} ligands")
    axes[1].set_ylim(0, 1.03)
    axes[1].set_ylabel("Fraction of targets reaching threshold")
    axes[1].set_title("Ligand-evidence coverage")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(
        endpoint["K"], endpoint["category_balanced_median_EF_cumulative"],
        marker="o", color="#E15759",
    )
    axes[2].fill_between(
        endpoint["K"].to_numpy(float), endpoint["category_q25_EF_cumulative"].to_numpy(float),
        endpoint["category_q75_EF_cumulative"].to_numpy(float), color="#E15759", alpha=0.20,
    )
    axes[2].set_ylabel(f"Category-balanced cumulative EF ({100 - percentile:g}% screened)")
    axes[2].set_title("Retrieval performance")

    for axis in axes:
        axis.set_xlabel("Maximum number of neighbors (K)")
        axis.set_xticks(range(2, 16))
        axis.grid(alpha=0.18)
    figure.suptitle("Nearest-neighbor K trade-off", y=1.02)
    figure.tight_layout()
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = output_dir / f"fixed_k_identity_evidence_performance_tradeoff.{suffix}"
        figure.savefig(path, dpi=dpi if suffix == "png" else None, bbox_inches="tight")
        paths.append(path)
    plt.close(figure)
    return paths


def main() -> int:
    args = parse_args()
    if not 2 <= args.min_k <= args.max_k <= 15:
        raise ValueError("Require 2 <= min-k <= max-k <= 15")
    identity_floors = parse_numbers(args.identity_floors, float, "Identity floors")
    ligand_budgets = parse_numbers(args.ligand_budgets, int, "Ligand budgets")
    neighbor_counts = list(DEFAULT_K)
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)

    metrics, metadata = load_complete_sweep(
        args.sweep_dir.expanduser().resolve(), neighbor_counts=neighbor_counts
    )
    blast = load_blast_hits(
        args.blast_hits.expanduser().resolve(), set(metadata["target"])
    )
    metadata_covariates = build_k_covariates(metadata, blast)
    choices = build_policy_choices(
        metadata_covariates, blast, min_k=args.min_k, max_k=args.max_k,
        identity_floors=identity_floors, ligand_budgets=ligand_budgets,
        include_combined=args.include_combined_policies,
    )
    selected = attach_selected_results(choices, metrics)
    target, family, balanced = summarize_selected_results(selected)
    diagnostics = policy_diagnostics(
        choices, target, balanced, percentile=args.percentile
    )
    paired = compare_with_fixed_k5(target, percentile=args.percentile)
    distribution = k_selection_distribution(choices)

    metadata_covariates.to_csv(output_dir / "target_partition_k_covariates.csv", index=False)
    choices.to_csv(output_dir / "adaptive_policy_choices.csv", index=False)
    selected.to_csv(output_dir / "adaptive_selected_results_all_percentiles.csv", index=False)
    target.to_csv(output_dir / "adaptive_target_medians.csv", index=False)
    family.to_csv(output_dir / "adaptive_family_medians.csv", index=False)
    balanced.to_csv(output_dir / "adaptive_category_balanced_summary.csv", index=False)
    diagnostics.to_csv(output_dir / "adaptive_policy_diagnostics.csv", index=False)
    paired.to_csv(output_dir / "adaptive_policy_vs_fixed_k5.csv", index=False)
    distribution.to_csv(output_dir / "adaptive_k_selection_distribution.csv", index=False)
    figure_paths = plot_fixed_k_tradeoff(
        selected, choices, balanced, percentile=args.percentile,
        ligand_budgets=ligand_budgets, output_dir=output_dir, dpi=args.dpi,
    )

    metadata_payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sweep_dir": str(args.sweep_dir.expanduser().resolve()),
        "blast_hits": str(args.blast_hits.expanduser().resolve()),
        "neighbor_counts_available": neighbor_counts,
        "adaptive_min_k": args.min_k,
        "adaptive_max_k": args.max_k,
        "identity_floors_percent": identity_floors,
        "ligand_candidate_budgets": ligand_budgets,
        "endpoint_percentile": args.percentile,
        "partition_seeds": list(SEEDS),
        "n_targets": int(metrics["target"].nunique()),
        "selection_uses_EF": False,
        "identity_policy": (
            "Begin at min_k and add ranked neighbors while the next neighbor's BLAST "
            "percentage identity is at least the floor; never exceed max_k."
        ),
        "ligand_policy": (
            "Select the smallest K in [min_k,max_k] with at least the requested number "
            "of cleaned candidate ligands before MaxMin; use max_k if never reached."
        ),
        "important_budget_note": (
            "The 50/75/100 values are stopping thresholds for candidate evidence, not "
            "new fixed MaxMin seed counts. EF values retain the original target-specific "
            "requested seed budget and are reused without molecular reranking."
        ),
        "figures": [str(path) for path in figure_paths],
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata_payload, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Adaptive K analysis complete: {metrics['target'].nunique()} targets, "
        f"{len(choices)} target/partition/policy choices -> {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
