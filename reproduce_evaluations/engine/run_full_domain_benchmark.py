#!/usr/bin/env python3
"""Evaluate full shared-Pfam seed sources beside the retained neighbor sweep."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ligq2_evaluation.category_spread import summarize_category_spread
from ligq2_evaluation.config import load_config, output_path, parse_csv_arg, resolve_path, split_kwargs
from ligq2_evaluation.constants import (
    FAMILIES, FAMILY_LABELS, FAMILY_ORDER, PLOTTED_NEIGHBOR_COUNTS, RAW_PERCENTILES, SEEDS,
)
from ligq2_evaluation.families import add_family_annotations
from ligq2_evaluation.full_domain import build_full_domain_sources, slice_binding_for_target
from ligq2_evaluation.runtime import load_context, prepare_output, write_run_manifest
from ligq2_evaluation.publication_style import readable_labels


METHOD = "morgan_1024_r2"
PLOT_PERCENTILES = (99.5, 99.0, 98.5, 98.0, 95.0)
DOMAIN_LABEL = "Full domain"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate only the full shared-Pfam condition and combine it with "
            "the existing five-partition nearest-neighbor results."
        )
    )
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yml")))
    parser.add_argument(
        "--neighbors-dir",
        type=Path,
        help="Directory containing the existing sweep_neighbors_seed_<seed> outputs.",
    )
    parser.add_argument("--output-dir", type=Path, help="Default: <outputs.root>/full_domain_benchmark")
    parser.add_argument("--targets", help="Comma-separated target names for a bounded run")
    parser.add_argument("--seeds", help="Comma-separated partition seeds for a bounded run")
    parser.add_argument("--dry-run", action="store_true", help="Report domain-source counts without molecular searches")
    parser.add_argument("--plot-only", action="store_true", help="Combine completed outputs without molecular searches")
    parser.add_argument("--resume", action="store_true", help="Skip completed target/seed cells")
    parser.add_argument("--force", action="store_true", help="Recompute target/seed cells in the output directory")
    return parser.parse_args()


def _reference_directory(cfg: dict, override: Path | None) -> Path:
    if override is not None:
        base = override.expanduser().resolve()
    else:
        configured = output_path(cfg, "neighbors")
        retained = Path(__file__).resolve().parent.parent / "resultados_EF_vecinos_repeticiones_10_90"
        base = configured if configured.exists() else retained
    if not (base / f"sweep_neighbors_seed_{SEEDS[0]}").is_dir():
        raise FileNotFoundError(
            f"Existing neighbor results were not found at {base}. Supply --neighbors-dir."
        )
    return base


def _reference_rows(base: Path) -> tuple[pd.DataFrame, set[str], dict[tuple[int, str], dict]]:
    frames: list[pd.DataFrame] = []
    valid_by_seed: list[set[str]] = []
    checks: dict[tuple[int, str], dict] = {}
    for seed in SEEDS:
        prefix = base / f"sweep_neighbors_seed_{seed}" / f"sweep_neighbors_{METHOD}"
        rows = pd.read_csv(prefix.with_name(prefix.name + "_all_targets.csv"))
        rows = rows[rows["neighbor_count_sweep"].isin(PLOTTED_NEIGHBOR_COUNTS)].copy()
        complete = rows.groupby("target")["neighbor_count_sweep"].nunique()
        valid_by_seed.append(set(complete[complete.eq(len(PLOTTED_NEIGHBOR_COUNTS))].index.astype(str)))
        rows["seed"] = seed
        frames.append(rows)

        known = pd.read_csv(prefix.with_name(prefix.name + "_known_active_sets_all.csv"))
        known = known[known["neighbor_count_sweep"].eq(PLOTTED_NEIGHBOR_COUNTS[0])]
        reference_cells = rows[
            rows["neighbor_count_sweep"].eq(PLOTTED_NEIGHBOR_COUNTS[0])
            & rows["percentile"].eq(PLOT_PERCENTILES[0])
        ]
        for row in reference_cells.itertuples(index=False):
            checks[(seed, str(row.target))] = {
                "n_pool_eval": int(row.n_pool_eval),
                "n_pos_eval": int(row.n_pos_eval),
                "n_seed_requested": int(row.n_seed_requested),
            }
        for row in known.itertuples(index=False):
            key = (seed, str(row.target_id))
            if key in checks:
                checks[key]["known_active_ids"] = list(ast.literal_eval(row.known_active_ids))

    common = set.intersection(*valid_by_seed)
    if not common:
        raise ValueError("No targets are shared by all five neighbor partitions and plotted K values")
    for seed in SEEDS:
        for target in common:
            if "known_active_ids" not in checks.get((seed, target), {}):
                raise ValueError(f"Missing retained split for seed={seed}, target={target}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["target"].astype(str).isin(common)].copy()
    return combined, common, checks


def _file_stamp(path: Path) -> dict:
    info = path.stat()
    return {"path": str(path), "size": info.st_size, "mtime_ns": info.st_mtime_ns}


def _signature(cfg: dict, reference_dir: Path, source, seed: int) -> str:
    store_root = resolve_path(cfg, "paths", "ligand_store")
    data = {
        "protocol_version": 2,
        "target": source.target,
        "seed": seed,
        "split_kwargs": split_kwargs(seed),
        "candidate_cleanup_tanimoto_cutoff": 0.85,
        "maxmin_initial_index": 0,
        "short_seed_policy": "allow_short",
        "source_uniprot_ids": source.source_uniprot_ids,
        "pfam_ids": source.pfam_ids,
        "binding": _file_stamp(resolve_path(cfg, "paths", "binding_data")),
        "smiles": _file_stamp(resolve_path(cfg, "paths", "smiles_data")),
        "targets": _file_stamp(resolve_path(cfg, "paths", "targets_csv")),
        "ligand_store": _file_stamp(store_root / "ligands.parquet"),
        "representation_data": _file_stamp(store_root / "reps" / f"{METHOD}.dat"),
        "representation_metadata": _file_stamp(store_root / "reps" / f"{METHOD}.meta.json"),
        "reference": _file_stamp(
            reference_dir / f"sweep_neighbors_seed_{seed}" / f"sweep_neighbors_{METHOD}_all_targets.csv"
        ),
        "reference_known_sets": _file_stamp(
            reference_dir / f"sweep_neighbors_seed_{seed}" / f"sweep_neighbors_{METHOD}_known_active_sets_all.csv"
        ),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def _cell_paths(out: Path, seed: int, target: str) -> dict[str, Path]:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
        raise ValueError(f"Unsafe target identifier: {target!r}")
    base = out / f"seed_{seed}" / target
    return {
        "base": base,
        "metrics": base / "metrics.csv",
        "metadata": base / "seed_metadata.csv",
        "known": base / "known_active_sets.csv",
        "retrieved": base / "retrieved_active_sets.csv",
        "status": base / "status.json",
    }


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _write_json(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _verify_matching_split(result: dict, expected: dict, *, seed: int, target: str) -> None:
    metadata = result["df_seed_meta_all"]
    if metadata.empty or len(metadata) != 1:
        raise ValueError(f"Missing or duplicate domain seed metadata for seed={seed}, target={target}")
    row = metadata.iloc[0]
    for actual_name, expected_name in (
        ("n_pool_eval_repr", "n_pool_eval"),
        ("n_pos_eval_repr", "n_pos_eval"),
        ("k_requested", "n_seed_requested"),
    ):
        if int(row[actual_name]) != expected[expected_name]:
            raise ValueError(
                f"Retained K data and full-domain {actual_name} disagree for "
                f"seed={seed}, target={target}: {row[actual_name]} != {expected[expected_name]}"
            )
    known = result["df_known_active_sets_all"]
    if not known.empty:
        actual = list(known.iloc[0]["known_active_ids"])
        if actual != expected["known_active_ids"]:
            raise ValueError(f"Known-active split differs from retained K data for seed={seed}, target={target}")


def _evaluate_cell(cfg: dict, context, source, seed: int, expected: dict, paths: dict[str, Path], signature: str) -> dict:
    from armado_datasets_modified import run_ef_eval_neighbors_sweep_single_method_clustering

    source_count = len(source.source_uniprot_ids)
    status = {
        "seed": seed,
        "target": source.target,
        "signature": signature,
        "pfam_ids": list(source.pfam_ids),
        "n_domain_proteins": source.n_domain_proteins,
        "n_excluded_benchmark_proteins": source.n_excluded_benchmark_proteins,
        "n_source_proteins_with_ligands": source_count,
        "source_uniprot_ids": list(source.source_uniprot_ids),
    }
    paths["base"].mkdir(parents=True, exist_ok=True)
    if source_count == 0:
        status.update(eligible=False, full_seed_budget=False, reason="no_ligand_bearing_domain_proteins")
        _write_json(status, paths["status"])
        return status

    binding_subset = slice_binding_for_target(context.binding, source)
    target_rows = context.targets[context.targets["target"].astype(str).eq(source.target)].copy()
    result = run_ef_eval_neighbors_sweep_single_method_clustering(
        neighbor_counts=[source_count],
        targets_dude=target_rows,
        binding_data=binding_subset,
        smiles=context.smiles,
        store=context.store,
        neighbor_ranked_ids_by_target={source.target: list(source.source_uniprot_ids)},
        rep_eval=context.representations[METHOD],
        metric_eval="tanimoto",
        method_label=METHOD,
        rep_morgan_for_seed_selection=context.representations[METHOD],
        tanimoto_cleanup_cutoff=0.85,
        maxmin_init_index=0,
        percentiles=RAW_PERCENTILES,
        min_known_for_eval=50,
        neighbor_short_policy="allow_short",
        split_kwargs=split_kwargs(seed),
        verbose=False,
    )
    _verify_matching_split(result, expected, seed=seed, target=source.target)

    metrics = result["df_long_neighbors_all"].copy()
    metadata = result["df_seed_meta_all"].copy()
    known = result["df_known_active_sets_all"].copy()
    retrieved = result["df_retrieved_active_sets_all"].copy()
    if not metrics.empty:
        metrics["method"] = f"{METHOD}__full_domain"
        metrics["seed_source"] = "full_domain"
        metrics["strategy"] = DOMAIN_LABEL
        metrics["seed"] = seed
        metrics["n_source_proteins"] = source_count
    if not metadata.empty:
        metadata["strategy"] = DOMAIN_LABEL
        metadata["seed"] = seed
    if not known.empty:
        known["method_label"] = f"{METHOD}__full_domain"
    if not retrieved.empty:
        retrieved["method_label"] = f"{METHOD}__full_domain"

    for key, frame in (("metrics", metrics), ("metadata", metadata), ("known", known), ("retrieved", retrieved)):
        _write_csv(frame, paths[key])
    full_seed_budget = bool(
        not metadata.empty and int(metadata.iloc[0]["k_effective"]) == int(metadata.iloc[0]["k_requested"])
    )
    # The retained K sweep used neighbor_short_policy="allow_short". Keep that
    # rule for the full-domain condition and report the effective budget.
    status["full_seed_budget"] = full_seed_budget
    status["eligible"] = bool(len(metrics) == len(RAW_PERCENTILES))
    status["reason"] = (
        "full_seed_budget" if status["eligible"] and full_seed_budget
        else "short_seed_budget" if status["eligible"]
        else "missing_metrics"
    )
    status["n_selected_seeds"] = int(metadata.iloc[0]["k_effective"])
    status["n_requested_seeds"] = int(metadata.iloc[0]["k_requested"])
    _write_json(status, paths["status"])
    return status


def _load_statuses(out: Path, targets: set[str]) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    statuses = []
    missing = []
    for seed in SEEDS:
        for target in sorted(targets):
            path = _cell_paths(out, seed, target)["status"]
            if not path.exists():
                missing.append((seed, target))
            else:
                statuses.append(json.loads(path.read_text(encoding="utf-8")))
    return pd.DataFrame(statuses), missing


@readable_labels
def _plot_comparison(spread: pd.DataFrame, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    strategies = [*[f"K={value}" for value in PLOTTED_NEIGHBOR_COUNTS], DOMAIN_LABEL]
    colors = plt.get_cmap("viridis")
    palette = {
        f"K={count}": colors(index / (len(PLOTTED_NEIGHBOR_COUNTS) - 1))
        for index, count in enumerate(PLOTTED_NEIGHBOR_COUNTS)
    }
    palette[DOMAIN_LABEL] = "#d62728"
    fig, ax = plt.subplots(figsize=(10.8, 6.5))
    x = np.arange(len(PLOT_PERCENTILES))
    for strategy in strategies:
        rows = spread[spread["strategy"].eq(strategy)].set_index("percentile")
        if not set(PLOT_PERCENTILES).issubset(rows.index):
            raise ValueError(f"Missing plotted percentiles for {strategy}")
        rows = rows.loc[list(PLOT_PERCENTILES)]
        color = palette[strategy]
        ax.plot(
            x,
            rows["category_balanced_median"].to_numpy(dtype=float),
            marker="o", markersize=5.5, linewidth=2.4,
            linestyle="--" if strategy == DOMAIN_LABEL else "-",
            color=color, label=strategy,
        )
        ax.fill_between(
            x,
            rows["category_q25"].to_numpy(dtype=float),
            rows["category_q75"].to_numpy(dtype=float),
            color=color, alpha=0.14, linewidth=0,
        )
    ax.set_xticks(x, [f"{value:g}" for value in PLOT_PERCENTILES])
    ax.set_xlabel("Percentile threshold")
    ax.set_ylabel("Category-balanced median\ncumulative enrichment factor")
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="0.45", alpha=0.14, edgecolor="none"))
    labels.append("Category IQR")
    ax.legend(handles, labels, title="Seed source", frameon=False)
    fig.tight_layout()
    stem = out / "nearest_neighbor_vs_full_domain_cumulative_EF_publication"
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
    plt.close(fig)


@readable_labels
def _plot_family_comparisons(category_medians: pd.DataFrame, out: Path) -> None:
    """Reproduce the neighbor-sweep family panels with Full domain added."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    strategies = [*[f"K={value}" for value in PLOTTED_NEIGHBOR_COUNTS], DOMAIN_LABEL]
    colors = plt.get_cmap("viridis")
    palette = {
        f"K={count}": colors(index / (len(PLOTTED_NEIGHBOR_COUNTS) - 1))
        for index, count in enumerate(PLOTTED_NEIGHBOR_COUNTS)
    }
    palette[DOMAIN_LABEL] = "#d62728"
    figure_dir = out / "family_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(out / "nearest_neighbor_vs_full_domain_EF_by_protein_group.pdf") as pdf:
        for family in FAMILY_ORDER:
            family_rows = category_medians[category_medians["familia"].eq(family)]
            if family_rows.empty:
                continue
            counts = family_rows["n_targets"].unique()
            if len(counts) != 1:
                raise ValueError(f"Inconsistent target counts for family {family}: {counts}")
            family_label = FAMILY_LABELS[family]
            fig, ax = plt.subplots(figsize=(8.5, 5.5))
            x = np.arange(len(PLOT_PERCENTILES))
            for strategy in strategies:
                rows = family_rows[family_rows["strategy"].eq(strategy)].set_index("percentile")
                if not set(PLOT_PERCENTILES).issubset(rows.index):
                    raise ValueError(f"Missing plotted percentiles for {family}, {strategy}")
                rows = rows.loc[list(PLOT_PERCENTILES)]
                ax.plot(
                    x, rows["category_median_EF_cumulative"].to_numpy(dtype=float),
                    marker="o", markersize=5, linewidth=2.2,
                    linestyle="--" if strategy == DOMAIN_LABEL else "-",
                    color=palette[strategy], label=strategy,
                )
            ax.set_xticks(x, [f"{value:g}" for value in PLOT_PERCENTILES])
            ax.set_xlabel("Percentile threshold")
            ax.set_ylabel("Median cumulative\nenrichment factor")
            ax.set_title(f"{family_label} (n={int(counts[0])} targets)")
            ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.3)
            ax.legend(title="Seed source", frameon=False)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            slug = re.sub(r"[^a-z0-9]+", "_", family_label.lower()).strip("_")
            stem = figure_dir / f"nearest_neighbor_vs_full_domain_{slug}"
            for suffix in ("png", "pdf", "svg"):
                fig.savefig(stem.with_suffix(f".{suffix}"), dpi=600, bbox_inches="tight")
            plt.close(fig)


def _combine_and_plot(reference: pd.DataFrame, targets: set[str], out: Path) -> int:
    statuses, missing = _load_statuses(out, targets)
    if missing:
        print(f"{len(missing)} target/seed cells remain. Resume with --resume; no comparison figure was written.")
        return 0
    statuses.to_csv(out / "full_domain_target_seed_coverage.csv", index=False)
    eligible = set(statuses.groupby("target")["eligible"].all().loc[lambda values: values].index)
    if not eligible:
        raise ValueError("No target has a full domain seed budget in all five partitions")

    reference = reference[reference["target"].astype(str).isin(eligible)].copy()
    reference["strategy"] = reference["neighbor_count_sweep"].map(lambda count: f"K={int(count)}")
    domain_frames = []
    metadata_frames = []
    for seed in SEEDS:
        for target in sorted(eligible):
            paths = _cell_paths(out, seed, target)
            domain_frames.append(pd.read_csv(paths["metrics"]))
            metadata_frames.append(pd.read_csv(paths["metadata"]))
    domain = pd.concat(domain_frames, ignore_index=True)
    metadata = pd.concat(metadata_frames, ignore_index=True)
    reference["seed"] = reference["seed"].astype(int)
    combined = pd.concat([reference, domain], ignore_index=True)
    counts = combined.groupby(["target", "strategy", "percentile"])["seed"].nunique()
    expected_cells = len(eligible) * (len(PLOTTED_NEIGHBOR_COUNTS) + 1) * len(RAW_PERCENTILES)
    if len(counts) != expected_cells or not counts.eq(len(SEEDS)).all():
        raise ValueError("The combined comparison has a target/strategy/percentile without five partitions")
    target_medians = (
        combined.groupby(["target", "strategy", "percentile"], as_index=False)
        .agg(median_seed_EF_cumulative=("EF_cumulative", "median"), n_seeds=("seed", "nunique"))
    )
    target_medians = add_family_annotations(target_medians, FAMILIES)
    if target_medians["familia"].isna().any():
        unknown = sorted(target_medians.loc[target_medians["familia"].isna(), "target"].unique())
        raise ValueError(f"Unclassified comparison targets: {unknown}")
    category_medians = (
        target_medians.groupby(["strategy", "percentile", "familia"], as_index=False)
        .agg(category_median_EF_cumulative=("median_seed_EF_cumulative", "median"),
             n_targets=("target", "nunique"))
    )
    _, spread = summarize_category_spread(
        category_medians,
        group_columns=["strategy", "percentile"],
        value_column="category_median_EF_cumulative",
    )
    combined.to_csv(out / "neighbor_vs_full_domain_all_seed_rows.csv", index=False)
    seed_budgets = (
        combined.groupby(["target", "seed", "strategy"], as_index=False)
        .agg(n_seed_requested=("n_seed_requested", "first"),
             n_seed_effective=("n_seed_effective", "first"))
    )
    seed_budgets.to_csv(out / "neighbor_vs_full_domain_seed_budgets.csv", index=False)
    metadata.to_csv(out / "full_domain_seed_metadata_all.csv", index=False)
    target_medians.to_csv(out / "neighbor_vs_full_domain_target_medians.csv", index=False)
    category_medians.to_csv(out / "neighbor_vs_full_domain_category_medians.csv", index=False)
    spread.to_csv(out / "neighbor_vs_full_domain_category_balanced_spread.csv", index=False)
    _plot_comparison(spread, out)
    _plot_family_comparisons(category_medians, out)
    full_budget = int(statuses.groupby("target")["full_seed_budget"].all().sum())
    print(
        f"Comparison complete: {len(eligible)}/{len(targets)} targets have results "
        f"in all five partitions; {full_budget} have the full domain seed budget."
    )
    return len(eligible)


def main() -> int:
    args = parse_args()
    if args.dry_run and args.plot_only:
        raise ValueError("--dry-run and --plot-only cannot be combined")
    cfg = load_config(args.config)
    reference_dir = _reference_directory(cfg, args.neighbors_dir)
    reference, common_targets, checks = _reference_rows(reference_dir)
    out = args.output_dir.expanduser().resolve() if args.output_dir else output_path(cfg, "root") / "full_domain_benchmark"

    if args.plot_only:
        if not out.is_dir():
            raise FileNotFoundError(f"Full-domain output directory does not exist: {out}")
        _combine_and_plot(reference, common_targets, out)
        return 0

    selected = set(parse_csv_arg(args.targets) or sorted(common_targets))
    if not selected.issubset(common_targets):
        raise ValueError(f"Targets outside the retained K comparison: {sorted(selected - common_targets)}")
    seeds = parse_csv_arg(args.seeds, int) or list(SEEDS)
    if not set(seeds).issubset(SEEDS):
        raise ValueError(f"Only the original partition seeds are supported: {SEEDS}")

    if args.dry_run:
        targets = pd.read_csv(resolve_path(cfg, "paths", "targets_csv"))
        binding = pd.read_parquet(
            resolve_path(cfg, "paths", "binding_data"),
            columns=["uniprot_id", "pfam_id", "chem_comp_id"],
        )
        sources = build_full_domain_sources(targets, binding, selected)
        sizes = pd.Series([len(source.source_uniprot_ids) for source in sources.values()])
        print(
            f"Full domain: {len(sources)} targets; source proteins "
            f"min={sizes.min()}, median={sizes.median():g}, max={sizes.max()}; "
            f"{int(sizes.gt(15).sum())} targets have more than 15 sources."
        )
        return 0

    prepare_output(out, force=args.force, resume=args.resume)
    # Keep the entire benchmark target table available: the historical BLAST
    # neighbor run excluded every benchmark-target UniProt ID, not just the
    # 56 targets plotted in its main figure.
    context = load_context(cfg, [METHOD], with_neighbors=False)
    sources = build_full_domain_sources(context.targets, context.binding, selected)
    for seed in seeds:
        for index, target in enumerate(sorted(selected), start=1):
            source = sources[target]
            paths = _cell_paths(out, seed, target)
            signature = _signature(cfg, reference_dir, source, seed)
            if args.resume and paths["status"].exists():
                old = json.loads(paths["status"].read_text(encoding="utf-8"))
                if old.get("signature") != signature:
                    raise ValueError(
                        f"Inputs changed for seed={seed}, target={target}. "
                        "Use a new output directory or --force to recompute."
                    )
                print(f"[seed={seed} {index}/{len(selected)}] {target}: already complete")
                continue
            print(f"[seed={seed} {index}/{len(selected)}] {target}: {len(source.source_uniprot_ids)} sources", flush=True)
            status = _evaluate_cell(cfg, context, source, seed, checks[(seed, target)], paths, signature)
            print(f"  {status['reason']}", flush=True)

    matched = _combine_and_plot(reference, common_targets, out)
    if matched:
        write_run_manifest(out / "run_manifest.json", cfg, "full_domain_benchmark", {
            "retained_neighbor_results": str(reference_dir),
            "seeds": list(SEEDS),
            "fixed_k_values": list(PLOTTED_NEIGHBOR_COUNTS),
            "full_domain_rule": "all ligand-bearing proteins sharing any query Pfam, excluding all benchmark-target proteins",
            "protein_source_order": "ascending UniProt ID",
            "seed_budget": "same number of target known actives as each retained target/partition",
            "short_seed_policy": "allow_short, matching the retained neighbor sweep",
            "candidate_cleanup_tanimoto_cutoff": 0.85,
            "maxmin_initial_index": 0,
            "matched_target_count": matched,
        })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
