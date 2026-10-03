#!/usr/bin/env python3
"""Complete and consolidate the ECFP4 nearest-neighbor sweep for K=2..15.

Existing retained K values are reused without recomputation. Only missing K
values are evaluated. New calculations are checkpointed separately
for every partition seed and K, so interrupted runs can be resumed safely.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ligq2_evaluation.config import load_config, parse_csv_arg, split_kwargs
from ligq2_evaluation.constants import SEEDS
from ligq2_evaluation.runtime import (
    bootstrap_dependencies,
    build_neighbor_rankings,
    filter_targets,
    load_context,
    prepare_output,
)


METHOD = "morgan_1024_r2"
DEFAULT_COUNTS = tuple(range(2, 16))
COMPONENTS = {
    "metrics": "df_long_neighbors_all",
    "seed_metadata": "df_seed_meta_all",
    "retrieved_active_sets": "df_retrieved_active_sets_all",
    "known_active_sets": "df_known_active_sets_all",
    "known_consistency_checks": "df_known_consistency_checks_all",
}
RETAINED_SUFFIXES = {
    "metrics": "all_targets.csv",
    "seed_metadata": "seed_meta_all.csv",
    "retrieved_active_sets": "retrieved_active_sets_all.csv",
    "known_active_sets": "known_active_sets_all.csv",
    "known_consistency_checks": "known_consistency_checks_all.csv",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=script_dir / "config.yml")
    parser.add_argument(
        "--retained-root", type=Path,
        default=script_dir.parent / "resultados_EF_vecinos_repeticiones_10_90",
    )
    parser.add_argument(
        "--blast-hits", type=Path,
        default=script_dir / "reproduction_output/k_covariates/neighbor_blast_hits.csv",
        help="Cached complete BLAST ranking from script 13; rebuilt if absent",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=script_dir / "reproduction_output/neighbor_k_complete",
    )
    parser.add_argument(
        "--matched-targets", type=Path,
        default=script_dir / "reproduction_output/k_stability/k_stability_by_target.csv",
        help=(
            "Target table defining the historical matched comparison set. "
            "Ignored with --include-all-complete-targets"
        ),
    )
    parser.add_argument(
        "--include-all-complete-targets", action="store_true",
        help="Keep every target with a complete K/partition grid instead of the historical matched set",
    )
    parser.add_argument("--neighbor-counts", default="2-15")
    parser.add_argument("--seeds", help="Comma-separated partition seeds")
    parser.add_argument("--targets", help="Optional comma-separated smoke-test targets")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_counts(value: str) -> list[int]:
    counts: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending K range: {token}")
            counts.extend(range(start, end + 1))
        else:
            counts.append(int(token))
    counts = sorted(set(counts))
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Neighbor counts must be positive integers")
    return counts


def retained_path(root: Path, seed: int, component: str) -> Path:
    return (
        root / f"sweep_neighbors_seed_{seed}"
        / f"sweep_neighbors_{METHOD}_{RETAINED_SUFFIXES[component]}"
    )


def load_retained_components(root: Path, seed: int) -> dict[str, pd.DataFrame]:
    frames = {}
    for component in COMPONENTS:
        path = retained_path(root, seed, component)
        if not path.is_file():
            raise FileNotFoundError(f"Retained neighbor result not found: {path}")
        frame = pd.read_csv(path)
        if "neighbor_count_sweep" not in frame.columns:
            raise ValueError(f"Retained file lacks neighbor_count_sweep: {path}")
        frame["neighbor_count_sweep"] = frame["neighbor_count_sweep"].astype(int)
        frames[component] = frame
    return frames


def checkpoint_dir(output_dir: Path, seed: int, count: int) -> Path:
    return output_dir / "checkpoints" / f"seed_{seed}" / f"k_{count}"


def checkpoint_complete(path: Path) -> bool:
    return all((path / f"{component}.csv").is_file() for component in COMPONENTS)


def save_checkpoint(path: Path, result: dict[str, pd.DataFrame], seed: int, count: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for component, result_key in COMPONENTS.items():
        frame = result[result_key].copy()
        if not frame.empty and "neighbor_count_sweep" not in frame.columns:
            frame["neighbor_count_sweep"] = int(count)
        frame.to_csv(path / f"{component}.csv", index=False)
    (path / "metadata.json").write_text(json.dumps({
        "partition_seed": int(seed),
        "neighbor_count": int(count),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")


def load_checkpoint(path: Path) -> dict[str, pd.DataFrame]:
    if not checkpoint_complete(path):
        raise FileNotFoundError(f"Incomplete checkpoint: {path}")
    return {
        component: pd.read_csv(path / f"{component}.csv")
        for component in COMPONENTS
    }


def rankings_from_cache(path: Path) -> dict[str, list[str]]:
    frame = pd.read_csv(path)
    required = {"target_name", "sseqid", "neighbor_rank"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"BLAST cache is missing columns: {missing}")
    frame["target_name"] = frame["target_name"].astype(str).str.lower()
    frame["sseqid"] = frame["sseqid"].astype(str)
    frame["neighbor_rank"] = pd.to_numeric(frame["neighbor_rank"], errors="raise")
    return {
        target: rows.sort_values("neighbor_rank")["sseqid"].tolist()
        for target, rows in frame.groupby("target_name", sort=False)
    }


def filter_component(
    frame: pd.DataFrame,
    *,
    count: int,
    selected_targets: set[str] | None,
) -> pd.DataFrame:
    result = frame[frame["neighbor_count_sweep"].astype(int).eq(count)].copy()
    if selected_targets is not None and "target" in result.columns:
        result = result[result["target"].astype(str).str.lower().isin(selected_targets)]
    if not result.empty:
        result["neighbor_count_sweep"] = int(count)
    return result.reset_index(drop=True)


def combine_seed_components(
    retained: dict[str, pd.DataFrame],
    *,
    output_dir: Path,
    seed: int,
    neighbor_counts: list[int],
    selected_targets: set[str] | None,
) -> dict[str, pd.DataFrame]:
    combined: dict[str, pd.DataFrame] = {}
    retained_counts = set(retained["metrics"]["neighbor_count_sweep"].astype(int))
    for component in COMPONENTS:
        pieces = []
        for count in neighbor_counts:
            if count in retained_counts:
                piece = filter_component(
                    retained[component], count=count, selected_targets=selected_targets
                )
            else:
                checkpoint = load_checkpoint(checkpoint_dir(output_dir, seed, count))
                piece = filter_component(
                    checkpoint[component], count=count, selected_targets=selected_targets
                )
            if not piece.empty:
                pieces.append(piece)
        combined[component] = (
            pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        )
    return combined


def validate_seed_grid(
    metrics: pd.DataFrame,
    seed_metadata: pd.DataFrame,
    neighbor_counts: list[int],
) -> set[str]:
    if metrics.empty or seed_metadata.empty:
        raise ValueError("Consolidated metrics or seed metadata are empty")
    metric_counts = metrics.groupby("target")["neighbor_count_sweep"].nunique()
    meta_counts = seed_metadata.groupby("target")["neighbor_count_sweep"].nunique()
    required = len(neighbor_counts)
    valid = set(metric_counts[metric_counts.eq(required)].index.astype(str))
    valid &= set(meta_counts[meta_counts.eq(required)].index.astype(str))
    if not valid:
        raise ValueError("No target contains the complete requested K grid")
    duplicate_keys = ["target", "neighbor_count_sweep", "percentile"]
    if metrics.duplicated(duplicate_keys).any():
        raise ValueError("Duplicate target/K/percentile rows after consolidation")
    if seed_metadata.duplicated(["target", "neighbor_count_sweep"]).any():
        raise ValueError("Duplicate target/K rows in consolidated seed metadata")
    return valid


def write_seed_outputs(
    output_dir: Path,
    seed: int,
    combined: dict[str, pd.DataFrame],
    valid_targets: set[str],
) -> None:
    destination = output_dir / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    for component, frame in combined.items():
        frame.to_csv(destination / f"{component}_all.csv", index=False)
        if "target" in frame.columns:
            valid = frame[frame["target"].astype(str).isin(valid_targets)].copy()
            valid.to_csv(destination / f"{component}_valid_targets.csv", index=False)


def aggregate_across_seeds(
    output_dir: Path,
    seeds: list[int],
    neighbor_counts: list[int],
    matched_targets: set[str] | None = None,
) -> set[str]:
    metrics_frames = []
    metadata_frames = []
    valid_by_seed = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        metrics = pd.read_csv(seed_dir / "metrics_all.csv")
        metadata = pd.read_csv(seed_dir / "seed_metadata_all.csv")
        valid = validate_seed_grid(metrics, metadata, neighbor_counts)
        valid_by_seed.append(valid)
        metrics["seed"] = int(seed)
        metadata["seed"] = int(seed)
        metrics_frames.append(metrics)
        metadata_frames.append(metadata)
    common = set.intersection(*valid_by_seed)
    if matched_targets is not None:
        missing = sorted(matched_targets - common)
        if missing:
            raise ValueError(
                "Historical matched targets lack a complete K grid: "
                + ", ".join(missing[:10])
            )
        common &= matched_targets
    if not common:
        raise ValueError("No targets have a complete K grid in every partition")
    metrics = pd.concat(metrics_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    metrics = metrics[metrics["target"].astype(str).isin(common)].copy()
    metadata = metadata[metadata["target"].astype(str).isin(common)].copy()
    metrics.to_csv(output_dir / "all_seeds_valid_rows.csv", index=False)
    metadata.to_csv(output_dir / "all_seeds_seed_metadata.csv", index=False)

    check = (
        metrics.groupby(["target", "neighbor_count_sweep", "percentile"], as_index=False)
        .agg(n_seeds=("seed", "nunique"))
    )
    if not check["n_seeds"].eq(len(seeds)).all():
        raise ValueError("At least one target/K/percentile cell lacks a partition")
    check.to_csv(output_dir / "seed_count_check.csv", index=False)
    target_median = (
        metrics.groupby(["target", "neighbor_count_sweep", "percentile"], as_index=False)
        .agg(
            median_seed_EF_band=("EF_band", "median"),
            median_seed_EF_cumulative=("EF_cumulative", "median"),
            mean_seed_EF_band=("EF_band", "mean"),
            mean_seed_EF_cumulative=("EF_cumulative", "mean"),
            std_seed_EF_band=("EF_band", "std"),
            std_seed_EF_cumulative=("EF_cumulative", "std"),
            n_seeds=("seed", "nunique"),
        )
    )
    target_median.to_csv(output_dir / "target_median_across_seeds.csv", index=False)
    (output_dir / "valid_targets.txt").write_text(
        "\n".join(sorted(common)) + "\n", encoding="utf-8"
    )
    return common


def load_matched_targets(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Matched-target table not found: {path}")
    frame = pd.read_csv(path, usecols=["target"])
    targets = set(frame["target"].dropna().astype(str).str.lower())
    if not targets:
        raise ValueError(f"Matched-target table contains no targets: {path}")
    return targets


def main() -> int:
    args = parse_args()
    counts = parse_counts(args.neighbor_counts)
    seeds = parse_csv_arg(args.seeds, int) or list(SEEDS)
    selected_target_list = parse_csv_arg(args.targets)
    selected_targets = (
        {target.lower() for target in selected_target_list}
        if selected_target_list else None
    )
    matched_targets = (
        None if args.include_all_complete_targets
        else load_matched_targets(args.matched_targets)
    )
    if matched_targets is not None and selected_targets is not None:
        outside = sorted(selected_targets - matched_targets)
        if outside:
            raise ValueError(
                "Requested smoke-test targets are outside the historical matched set: "
                + ", ".join(outside)
                + ". Use --include-all-complete-targets to evaluate them explicitly."
            )
        matched_targets &= selected_targets
    retained_by_seed = {
        seed: load_retained_components(args.retained_root, seed) for seed in seeds
    }
    missing_by_seed = {
        seed: [
            count for count in counts
            if count not in set(frames["metrics"]["neighbor_count_sweep"].astype(int))
        ]
        for seed, frames in retained_by_seed.items()
    }
    for seed in seeds:
        existing = sorted(
            set(counts) - set(missing_by_seed[seed])
        )
        print(f"seed={seed}: reuse K={existing}; calculate K={missing_by_seed[seed]}")
    planned_targets = selected_targets or matched_targets
    print(
        "target scope: "
        + (f"{len(planned_targets)} targets" if planned_targets is not None else "all configured targets")
    )
    if args.plan_only:
        return 0

    output_dir = args.output_dir.expanduser().resolve()
    prepare_output(output_dir, force=args.force, resume=args.resume)
    cfg = load_config(args.config)
    context = load_context(cfg, [METHOD], selected_targets=None, with_neighbors=False)
    calculation_targets = (
        selected_target_list
        if selected_target_list
        else (sorted(matched_targets) if matched_targets is not None else None)
    )
    targets_for_run = filter_targets(context.targets, calculation_targets)
    if args.blast_hits.is_file():
        neighbor_rankings = rankings_from_cache(args.blast_hits)
        print(f"Reusing BLAST rankings: {args.blast_hits}")
    else:
        neighbor_rankings = build_neighbor_rankings(cfg, context.targets)

    bootstrap_dependencies()
    from armado_datasets_modified import run_ef_eval_neighbors_sweep_single_method_clustering

    for seed in seeds:
        for count in missing_by_seed[seed]:
            checkpoint = checkpoint_dir(output_dir, seed, count)
            if args.resume and checkpoint_complete(checkpoint):
                print(f"[seed={seed} K={count}] checkpoint exists; skipping")
                continue
            print(f"\n========== seed={seed} K={count} ==========")
            result = run_ef_eval_neighbors_sweep_single_method_clustering(
                neighbor_counts=[count],
                targets_dude=targets_for_run,
                binding_data=context.binding,
                smiles=context.smiles,
                store=context.store,
                neighbor_ranked_ids_by_target=neighbor_rankings,
                rep_eval=context.representations[METHOD],
                metric_eval="tanimoto",
                method_label=METHOD,
                rep_morgan_for_seed_selection=context.representations[METHOD],
                split_kwargs=split_kwargs(seed),
                device_eval=args.device,
                device_seed_selection=args.device,
            )
            save_checkpoint(checkpoint, result, seed, count)

        combined = combine_seed_components(
            retained_by_seed[seed], output_dir=output_dir, seed=seed,
            neighbor_counts=counts, selected_targets=selected_targets,
        )
        valid = validate_seed_grid(
            combined["metrics"], combined["seed_metadata"], counts
        )
        write_seed_outputs(output_dir, seed, combined, valid)
        print(f"[seed={seed}] complete-grid targets: {len(valid)}")

    common_targets = aggregate_across_seeds(
        output_dir, seeds, counts, matched_targets=matched_targets
    )
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(args.config.expanduser().resolve()),
        "retained_root": str(args.retained_root.expanduser().resolve()),
        "blast_hits": str(args.blast_hits.expanduser().resolve()),
        "neighbor_counts": counts,
        "partition_seeds": seeds,
        "selected_targets": sorted(selected_targets) if selected_targets else "all",
        "calculation_target_count": len(targets_for_run),
        "matched_targets_source": (
            None if args.include_all_complete_targets
            else str(args.matched_targets.expanduser().resolve())
        ),
        "n_consolidated_targets": len(common_targets),
        "method": METHOD,
        "metric": "tanimoto",
        "representation": "ECFP4/Morgan radius 2, 1024 bits",
        "existing_results_reused": True,
        "new_results_checkpointed_by_seed_and_k": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Complete K sweep: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
