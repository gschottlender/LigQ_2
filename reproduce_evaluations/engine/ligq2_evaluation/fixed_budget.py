"""Cached, label-blind fixed-budget rankings for representation combinations."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import resolve_path, split_kwargs
from .constants import FAMILIES, METHODS, RAW_PERCENTILES
from .provenance import file_record, representation_files, sha256_file


PERCENTILE = 99.5
POLICIES = ("best", "mean", "balanced")
HIGHLIGHTED = (
    ("morgan_1024_r2",),
    ("morgan_1024_r2", "morgan_feature_1024_r2"),
    ("morgan_1024_r2", "topological_torsion_rdkit_1024"),
    ("morgan_1024_r2", "morgan_feature_1024_r2", "topological_torsion_rdkit_1024"),
)


def _digest_strings(values) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def cache_signature(cfg: dict, methods: list[str]) -> str:
    """Fingerprint the exact inputs/settings that determine evaluation scores."""
    store = resolve_path(cfg, "paths", "ligand_store")
    paths = [resolve_path(cfg, "paths", name) for name in
             ("binding_data", "smiles_data", "targets_csv")]
    paths.extend(representation_files(store, methods))
    dependencies = Path(__file__).resolve().parents[1] / "dependencies"
    source_files = (
        dependencies / "evaluation_core/armado_datasets_modified.py",
        dependencies / "ligq_core/metrics.py",
        dependencies / "ligq_core/compound_helpers.py",
        dependencies / "SOURCES.json",
    )
    records = [file_record(path, with_hash=path.suffix in {".json", ".py"})
               for path in dict.fromkeys(paths)]
    if not all(record["exists"] for record in records):
        missing = [record["path"] for record in records if not record["exists"]]
        raise FileNotFoundError(f"Ranking input files are missing: {missing}")
    payload = {
        "schema": 1,
        "files": records,
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "methods": [(name, METHODS[name]) for name in methods],
        "split": split_kwargs(0),
        "percentiles": list(RAW_PERCENTILES),
        "numpy_version": np.__version__,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


class RankingCache:
    """One common pool plus one score/rank vector per method and partition."""

    def __init__(self, base_dir: Path, signature: str):
        self.signature = signature
        self.root = Path(base_dir) / "ranking_cache" / signature[:16]

    def _partition_dir(self, seed: int, target: str) -> Path:
        if not target or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                             for char in target) or target in {".", ".."}:
            raise ValueError(f"Unsafe target identifier: {target!r}")
        return self.root / f"seed_{int(seed)}" / target

    def _pool_paths(self, seed: int, target: str) -> tuple[Path, Path]:
        directory = self._partition_dir(seed, target)
        return directory / "pool.parquet", directory / "pool.json"

    def _method_paths(self, seed: int, target: str, method: str) -> tuple[Path, Path]:
        if method not in METHODS:
            raise ValueError(f"Unknown method: {method}")
        directory = self._partition_dir(seed, target)
        return directory / f"{method}.npz", directory / f"{method}.json"

    def has(self, seed: int, target: str, method: str) -> bool:
        pool_path, pool_meta_path = self._pool_paths(seed, target)
        array_path, meta_path = self._method_paths(seed, target, method)
        if not all(path.is_file() for path in (pool_path, pool_meta_path, array_path, meta_path)):
            return False
        try:
            pool_meta = json.loads(pool_meta_path.read_text(encoding="utf-8"))
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return (
                pool_meta.get("signature") == self.signature
                and meta.get("signature") == self.signature
                and meta.get("method") == method
                and meta.get("pool_sha256") == sha256_file(pool_path)
                and meta.get("array_sha256") == sha256_file(array_path)
            )
        except (OSError, ValueError, KeyError):
            return False

    def save(self, *, seed: int, target: str, method: str, pool_ids,
             positive_ids, seed_ids, scores, score_cut_995: float) -> None:
        pool_ids = np.asarray([str(value) for value in pool_ids], dtype=str)
        scores = np.asarray(scores, dtype=np.float32)
        if len(pool_ids) == 0 or len(pool_ids) != len(scores):
            raise ValueError("Pool IDs and scores must be nonempty and aligned")
        if len(set(pool_ids.tolist())) != len(pool_ids) or not np.isfinite(scores).all():
            raise ValueError("Pool IDs must be unique and scores must be finite")
        order = np.argsort(pool_ids, kind="stable")
        ids = pool_ids[order]
        values = scores[order]
        positives = {str(value) for value in positive_ids}
        labels = np.isin(ids, list(positives))
        if int(labels.sum()) != len(positives):
            raise ValueError("Positive IDs are not all present in the evaluated pool")

        pool_path, pool_meta_path = self._pool_paths(seed, target)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        if pool_path.is_file() and pool_meta_path.is_file():
            pool_meta = json.loads(pool_meta_path.read_text(encoding="utf-8"))
            existing = pd.read_parquet(pool_path)
            if (pool_meta.get("signature") != self.signature
                    or existing["compound_id"].astype(str).tolist() != ids.tolist()
                    or existing["is_active"].astype(bool).tolist() != labels.tolist()):
                raise ValueError(f"Methods do not share an identical evaluation pool: {target}, seed {seed}")
        else:
            pool_tmp = pool_path.with_suffix(".parquet.tmp")
            pd.DataFrame({"compound_id": ids, "is_active": labels}).to_parquet(pool_tmp, index=False)
            pool_tmp.replace(pool_path)
            pool_meta_path.write_text(json.dumps({
                "signature": self.signature, "n_pool": len(ids), "n_positives": len(positives),
                "pool_sha256": sha256_file(pool_path),
            }, indent=2) + "\n", encoding="utf-8")

        rank_order = np.lexsort((ids, -values))
        ranks = np.empty(len(ids), dtype=np.int32)
        ranks[rank_order] = np.arange(1, len(ids) + 1, dtype=np.int32)
        array_path, meta_path = self._method_paths(seed, target, method)
        array_tmp = array_path.with_suffix(".npz.tmp")
        with array_tmp.open("wb") as handle:
            np.savez_compressed(handle, scores=values, ranks=ranks)
        array_tmp.replace(array_path)
        meta_path.write_text(json.dumps({
            "signature": self.signature,
            "method": method,
            "metric": METHODS[method],
            "score_cut_995": float(score_cut_995),
            "n_pool": len(ids),
            "seed_ids_sha256": _digest_strings(seed_ids),
            "pool_sha256": sha256_file(pool_path),
            "array_sha256": sha256_file(array_path),
        }, indent=2) + "\n", encoding="utf-8")

    def load(self, seed: int, target: str, methods: list[str]):
        pool_path, _ = self._pool_paths(seed, target)
        pool = pd.read_parquet(pool_path)
        if pool["compound_id"].duplicated().any():
            raise ValueError(f"Duplicate compounds in ranking cache: {target}, seed {seed}")
        rankings = {}
        for method in methods:
            if not self.has(seed, target, method):
                raise ValueError(f"Missing or stale ranking cache: {target}, seed {seed}, {method}")
            array_path, meta_path = self._method_paths(seed, target, method)
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            with np.load(array_path, allow_pickle=False) as arrays:
                scores = arrays["scores"].copy()
                ranks = arrays["ranks"].copy()
            if len(scores) != len(pool) or len(ranks) != len(pool):
                raise ValueError(f"Corrupt ranking cache: {array_path}")
            rankings[method] = {"scores": scores, "ranks": ranks,
                                "cut": float(meta["score_cut_995"])}
        return pool, rankings


def select_compounds(ids: np.ndarray, rankings: dict, methods: tuple[str, ...],
                     policy: str, budget: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exactly `budget` unique indices, fusion values, and source indices.

    No activity labels are accepted here. Candidate compounds must be in the
    historical top-0.5%-by-score set of at least one selected method.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown fusion policy: {policy}")
    masks = [rankings[method]["scores"] >= rankings[method]["cut"] for method in methods]
    candidate = np.flatnonzero(np.logical_or.reduce(masks))
    if len(candidate) < budget:
        raise ValueError("Top-99.5-percentile union is smaller than the fixed budget")
    ranks = np.stack([rankings[method]["ranks"][candidate] for method in methods])
    best = ranks.min(axis=0)
    mean = ranks.mean(axis=0)
    if len(methods) == 1 or policy == "best":
        order = np.lexsort((ids[candidate], mean, best))[:budget]
        return candidate[order], best[order].astype(float), np.full(budget, -1, dtype=int)
    if policy == "mean":
        order = np.lexsort((ids[candidate], best, mean))[:budget]
        return candidate[order], mean[order], np.full(budget, -1, dtype=int)

    # Equal per-method allocation, with at most one extra slot for the first
    # methods in the stable METHODS order. Skip already selected IDs and refill
    # from the next candidate of that same method.
    quotas = [budget // len(methods) + (i < budget % len(methods)) for i in range(len(methods))]
    lists = [np.flatnonzero(mask)[np.argsort(rankings[method]["ranks"][mask], kind="stable")]
             for method, mask in zip(methods, masks)]
    pointers = [0] * len(methods)
    assigned = [0] * len(methods)
    chosen, sources, seen = [], [], set()
    while len(chosen) < budget:
        for position, indices in enumerate(lists):
            if assigned[position] >= quotas[position]:
                continue
            while pointers[position] < len(indices) and int(indices[pointers[position]]) in seen:
                pointers[position] += 1
            if pointers[position] >= len(indices):
                raise ValueError("Cannot fill balanced quota inside the top-99.5-percentile sets")
            selected = int(indices[pointers[position]])
            pointers[position] += 1
            seen.add(selected)
            chosen.append(selected)
            sources.append(position)
            assigned[position] += 1
    selected = np.asarray(chosen, dtype=int)
    return selected, np.arange(1, budget + 1, dtype=float), np.asarray(sources, dtype=int)


def _family_map() -> dict[str, str]:
    result = {}
    for family, subfamilies in FAMILIES.items():
        for targets in subfamilies.values():
            result.update({target: family for target in targets})
    return result


def targets_from_legacy(base_dir: Path, seeds: list[int], methods: list[str],
                        selected_targets: list[str] | None = None) -> list[str]:
    """Discover targets actually evaluated in every requested partition/method."""
    wanted = {str(target).lower() for target in selected_targets} if selected_targets else None
    target_lists = []
    for seed in seeds:
        path = Path(base_dir) / f"seed_{seed}" / "df_long_target_all_methods.csv"
        frame = pd.read_csv(path, usecols=["target", "method_label"])
        method_sets = []
        for method in methods:
            found = set(frame.loc[frame["method_label"] == method, "target"].astype(str))
            if wanted is not None:
                found &= wanted
            if not found:
                raise ValueError(f"No historical results for {method}, seed {seed}")
            method_sets.append(found)
        if any(found != method_sets[0] for found in method_sets[1:]):
            raise ValueError(f"Methods have different target coverage in seed {seed}")
        target_lists.append(method_sets[0])
    if any(found != target_lists[0] for found in target_lists[1:]):
        raise ValueError("Requested partitions have different target coverage")
    if wanted is not None and target_lists[0] != wanted:
        raise ValueError(f"Requested targets lack historical results: {sorted(wanted - target_lists[0])}")
    return sorted(target_lists[0])


def analyze_cache(cache: RankingCache, out_dir: Path, *, seeds: list[int],
                  targets: list[str], methods: list[str], policies: list[str]) -> Path:
    """Build all one-, two-, and three-method results from cached scores."""
    if len(methods) < 2:
        raise ValueError("At least two methods are required for combinations")
    if any(policy not in POLICIES for policy in policies):
        raise ValueError(f"Policies must be in {POLICIES}")
    strategies = [combination for count in range(1, min(3, len(methods)) + 1)
                  for combination in itertools.combinations(methods, count)]
    family_by_target = _family_map()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    for seed in seeds:
        for target in targets:
            pool, rankings = cache.load(seed, target, methods)
            ids = pool["compound_id"].astype(str).to_numpy()
            labels = pool["is_active"].astype(bool).to_numpy()
            n_pool, n_pos = len(pool), int(labels.sum())
            if not n_pool or not n_pos:
                raise ValueError(f"Empty evaluation pool or positive set: {target}, seed {seed}")
            budget = math.ceil(0.005 * n_pool)
            for policy in policies:
                selected_rows = []
                for combination in strategies:
                    indices, fusion_values, sources = select_compounds(
                        ids, rankings, combination, policy, budget)
                    if len(indices) != budget or len(set(ids[indices])) != budget:
                        raise AssertionError("Fixed-budget selection is not unique and exact")
                    n_active = int(labels[indices].sum())
                    name = " + ".join(f"{method}__target_seeds" for method in combination)
                    metrics.append({
                        "policy": policy, "seed": seed, "target": target,
                        "protein_family": family_by_target.get(target, "Unclassified"),
                        "method_combination": name, "n_methods": len(combination),
                        "n_pool": n_pool, "n_pool_actives": n_pos, "budget": budget,
                        "n_recovered_actives": n_active,
                        "recall_at_n": n_active / n_pos,
                        "precision_at_n": n_active / budget,
                        "ef_at_n": (n_active / budget) / (n_pos / n_pool),
                    })
                    for position, index in enumerate(indices):
                        selected_rows.append({
                            "policy": policy, "seed": seed, "target": target,
                            "method_combination": name, "rank": position + 1,
                            "compound_id": ids[index], "is_active": bool(labels[index]),
                            "fusion_value": float(fusion_values[position]),
                            "allocated_method": combination[sources[position]] if sources[position] >= 0 else "",
                            "component_ranks": json.dumps([int(rankings[method]["ranks"][index])
                                                            for method in combination]),
                            "component_scores": json.dumps([float(rankings[method]["scores"][index])
                                                             for method in combination]),
                        })
                path = out_dir / "selected_rankings" / policy / f"seed_{seed}" / f"{target}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(selected_rows).to_parquet(path, index=False)

    metrics_path = out_dir / "target_partition_budget_metrics.csv"
    metadata_path = out_dir / "run_metadata.json"
    metrics_df = pd.DataFrame(metrics)
    if metrics_path.is_file() and metadata_path.is_file():
        previous_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        compatible = (
            previous_meta.get("cache_signature") == cache.signature
            and previous_meta.get("methods") == methods
            and previous_meta.get("seeds") == seeds
            and previous_meta.get("targets") == targets
        )
        if not compatible:
            raise ValueError("Existing fixed-budget outputs use different inputs; choose a separate output directory")
        previous = pd.read_csv(metrics_path)
        previous = previous[~previous["policy"].isin(policies)]
        metrics_df = pd.concat([previous, metrics_df], ignore_index=True)
    metrics_df.to_csv(metrics_path, index=False)
    keys = ["policy", "method_combination", "n_methods", "target", "protein_family"]
    measures = ["n_recovered_actives", "recall_at_n", "precision_at_n", "ef_at_n"]
    target_df = metrics_df.groupby(keys, as_index=False)[measures].median()
    target_df.to_csv(out_dir / "target_median_across_partitions.csv", index=False)
    family_keys = ["policy", "method_combination", "n_methods", "protein_family"]
    family_df = target_df.groupby(family_keys, as_index=False)[measures].median()
    family_df.to_csv(out_dir / "family_medians.csv", index=False)
    summary_keys = ["policy", "method_combination", "n_methods"]
    summary_df = family_df.groupby(summary_keys, as_index=False)[measures].median()
    counts = target_df.groupby(summary_keys, as_index=False).agg(n_targets=("target", "nunique"))
    summary_df = summary_df.merge(counts, on=summary_keys, validate="one_to_one")
    summary_df.to_csv(out_dir / "method_combination_summary.csv", index=False)
    _plot_highlights(summary_df, out_dir, methods)
    metadata_path.write_text(json.dumps({
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cache_signature": cache.signature,
        "methods": methods, "policies": sorted(metrics_df["policy"].unique().tolist()),
        "seeds": seeds, "targets": targets,
        "budget": "ceil(0.005 * n_pool)",
        "candidate_rule": "score >= each method's original 99.5th-percentile cutoff; union within combination",
        "ranking_ties": "descending score, then ascending compound ID",
        "mean_rank_scope": "all component methods over the complete evaluation pool",
        "balanced_remainder": "first methods in the configured method order",
        "summary_aggregation": "median across partitions within target, median across targets within family, median across families",
    }, indent=2) + "\n", encoding="utf-8")
    return metrics_path


def _plot_highlights(summary: pd.DataFrame, out_dir: Path, methods: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["ECFP4", "ECFP4 + FCFP4", "ECFP4 + torsion", "ECFP4 + FCFP4 + torsion"]
    method_order = {method: position for position, method in enumerate(methods)}
    highlighted = [(combination, label) for combination, label in zip(HIGHLIGHTED, names)
                   if set(combination).issubset(method_order)]
    strategy_names = [" + ".join(f"{method}__target_seeds" for method in
                                 sorted(combination, key=method_order.__getitem__))
                      for combination, _ in highlighted]
    available = [policy for policy in POLICIES if policy in summary["policy"].unique()]
    fig, axes = plt.subplots(1, len(available), figsize=(5.2 * len(available), 4.5), squeeze=False)
    for axis, policy in zip(axes[0], available):
        subset = summary[summary["policy"] == policy].set_index("method_combination")
        present = [(name, label) for name, (_, label) in zip(strategy_names, highlighted)
                   if name in subset.index]
        values = [100 * float(subset.loc[name, "recall_at_n"]) for name, _ in present]
        axis.bar(range(len(present)), values, color="#4477AA")
        axis.set_xticks(range(len(present)), [label.replace(" + ", "\n+ ") for _, label in present], rotation=0)
        axis.set_ylabel("Category-balanced median recall@N (%)")
        axis.set_title(policy)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_dir / "fixed_budget_highlighted_recall.png", dpi=300)
    fig.savefig(out_dir / "fixed_budget_highlighted_recall.pdf")
    plt.close(fig)
