#!/usr/bin/env python
"""Characterize ECFP4 percentile hits without recomputing or changing retrieval.

Compare held-out actives (TP) and background hits (putative FP). Histogram
aggregation uses equal weights for partitions, targets, and protein categories.
The distributions describe selected hits, not the complete benchmark.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from ligq2_evaluation.config import load_config, resolve_path
from ligq2_evaluation.constants import FAMILIES, FAMILY_LABELS, SEEDS
from ligq2_evaluation.fixed_budget import RankingCache
from ligq2_evaluation.provenance import sha256_file

METHOD = "morgan_1024_r2"
GROUPS = ("TP", "putative_FP")
COLORS = {"TP": "#2378b5", "putative_FP": "#df7c31"}
LABELS = {"TP": "True positives", "putative_FP": "Putative false positives"}
PROPERTIES = {
    "similarity": "Maximum Tanimoto to known actives",
    "molecular_weight": "Molecular weight (Da)",
    "logp": "RDKit cLogP",
    "hba_minus_hbd": "HBA − HBD",
}


def family_map():
    return {target: FAMILY_LABELS[family] for family, subgroups in FAMILIES.items()
            for targets in subgroups.values() for target in targets}


def select_hits(scores, percentile):
    """Legacy inclusive percentile cutoff: retain ties, never clip to a fixed N."""
    scores = np.asarray(scores)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("An evaluation pool must contain finite scores.")
    cutoff = float(np.percentile(scores, percentile))
    return scores >= cutoff, cutoff


def molecular_properties(smiles):
    """Use the stored structure as-is; canonicalization is for QC only."""
    if not isinstance(smiles, str) or not smiles.strip():
        return {"valid_structure": False, "issue": "missing_smiles"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid_structure": False, "issue": "invalid_smiles"}
    hba, hbd = rdMolDescriptors.CalcNumHBA(mol), rdMolDescriptors.CalcNumHBD(mol)
    result = {"canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
              "molecular_weight": Descriptors.MolWt(mol), "logp": Crippen.MolLogP(mol),
              "hba": hba, "hbd": hbd, "hba_minus_hbd": hba - hbd,
              "valid_structure": True, "issue": ""}
    if not all(np.isfinite(result[key]) for key in PROPERTIES if key != "similarity"):
        return {"valid_structure": False, "issue": "nonfinite_descriptor"}
    return result


def build_histograms(hits, edges, groups=GROUPS):
    """Paired normalized group histograms, excluding empty group pairs explicitly."""
    rows, exclusions = [], []
    for (target, family, seed), cell in hits.groupby(["target", "protein_family", "partition"]):
        for prop, bins in edges.items():
            arrays = {group: cell.loc[cell.group.eq(group), prop].dropna().to_numpy()
                      for group in groups}
            if any(len(values) == 0 for values in arrays.values()):
                exclusions.append({"target": target, "protein_family": family,
                                   "partition": seed, "property": prop,
                                   "issue": "empty_group_distribution",
                                   "detail": json.dumps({g: len(a) for g, a in arrays.items()})})
                continue
            for group, values in arrays.items():
                counts = np.histogram(values, bins=bins)[0]
                if counts.sum() != len(values):
                    raise ValueError(f"Histogram bins omit values: {target}/{seed}/{prop}")
                for index, count in enumerate(counts):
                    rows.append({"target": target, "protein_family": family, "partition": seed,
                                 "property": prop, "group": group, "bin": index,
                                 "bin_left": bins[index], "bin_right": bins[index + 1],
                                 "count": int(count), "n_valid": len(values),
                                 "fraction": count / len(values)})
    return pd.DataFrame(rows), exclusions


def aggregate_histograms(partitions):
    """Arithmetic means of normalized histograms; no pooling of molecule counts."""
    bins = ["property", "group", "bin", "bin_left", "bin_right"]
    targets = partitions.groupby(["target", "protein_family"] + bins, as_index=False).agg(
        fraction=("fraction", "mean"), n_partitions=("partition", "nunique"))
    families = targets.groupby(["protein_family"] + bins, as_index=False).agg(
        fraction=("fraction", "mean"), n_targets=("target", "nunique"))
    overall = families.groupby(bins, as_index=False).agg(
        fraction=("fraction", "mean"), n_families=("protein_family", "nunique"))
    return targets, families, overall


def plot_panel(histograms, output, title, subtitle, formats=("png", "pdf", "svg")):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))
    for letter, (ax, (prop, label)) in zip("ABCD", zip(axes.flat, PROPERTIES.items())):
        subset = histograms.loc[histograms.property.eq(prop)]
        for group in GROUPS:
            values = subset.loc[subset.group.eq(group)].sort_values("bin")
            if values.empty:
                continue
            bins = np.r_[values.bin_left.to_numpy(), values.bin_right.iloc[-1]]
            ax.stairs(values.fraction.to_numpy(), bins, color=COLORS[group], lw=1.8,
                      label=LABELS[group])
            ax.stairs(values.fraction.to_numpy(), bins, color=COLORS[group],
                      fill=True, alpha=0.12)
        ax.set(xlabel=label, ylabel="Fraction within group")
        ax.set_title(letter, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(bottom=0)
        if prop == "similarity":
            ax.set_xlim(0, 1)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.925),
               ncol=2, frameon=False)
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.text(0.5, 0.017, subtitle, ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.045, 1, 0.88))
    output.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output.with_suffix("." + fmt), dpi=180)
    plt.close(fig)


def make_edges(hits):
    edges = {"similarity": np.linspace(0, 1, 51)}
    for prop in ("molecular_weight", "logp"):
        values = hits[prop].dropna()
        lower, upper = float(values.min()), float(values.max())
        if lower == upper:
            lower, upper = lower - 0.5, upper + 0.5
        edges[prop] = np.linspace(lower, upper, 41)
    lower, upper = hits.hba_minus_hbd.min(), hits.hba_minus_hbd.max()
    edges["hba_minus_hbd"] = np.arange(lower - 0.5, upper + 1.5)
    return edges


def legacy_rows(reference, seed, percentile):
    folder = reference / f"seed_{seed}"
    all_stats = pd.read_csv(folder / "df_long_target_all_methods.csv")
    all_stats = all_stats.loc[all_stats.method_label.eq(METHOD)]
    stats = all_stats.loc[np.isclose(all_stats.percentile, percentile)]
    if stats.target.duplicated().any():
        raise ValueError("Duplicate legacy target/method/percentile rows.")
    stats = stats.set_index("target")
    retrieved = pd.read_csv(folder / "retrieved_active_sets_all_methods.csv")
    # The original ID lists are DISJOINT PERCENTILE BANDS, not cumulative sets.
    # For top 1%, unite the 99.5 and 99 bands. Do not claim ID validation when
    # an intervening band was never exported (e.g. 98.5 in a top-2% request).
    retrieved = retrieved.loc[retrieved.method_label.isin([METHOD, METHOD + "__target_seeds"]) &
                              retrieved.percentile.ge(percentile)]
    if retrieved.duplicated(["target_id", "percentile"]).any():
        raise ValueError("Duplicate legacy retrieved-ID target/method/percentile rows.")
    cumulative = []
    for target, bands in retrieved.groupby("target_id"):
        expected = set(all_stats.loc[all_stats.target.eq(target) &
                                     all_stats.percentile.ge(percentile), "percentile"])
        if expected != set(bands.percentile):
            continue
        row = {"target_id": target}
        for column in ("retrieved_active_ids", "retrieved_inactive_ids"):
            ids = set()
            for value in bands[column]:
                ids.update(map(str, ast.literal_eval(value)))
            row[column] = repr(sorted(ids))
        cumulative.append(row)
    cumulative = pd.DataFrame(cumulative, columns=["target_id", "retrieved_active_ids",
                                                 "retrieved_inactive_ids"]).set_index("target_id")
    return stats, cumulative


def collect_hits(cache_root, reference, percentile, targets, seeds):
    cells = sorted(cache_root.glob("seed_*/*/" + METHOD + ".json"))
    if not cells:
        raise FileNotFoundError(f"No ECFP4 ranking cache at {cache_root}")
    signature = json.loads(cells[0].read_text())["signature"]
    if cache_root.name != signature[:16]:
        raise ValueError("Cache root does not match its recorded signature.")
    cache = RankingCache(cache_root.parent.parent, signature)
    expected_targets = {path.parent.name for path in cells}
    requested = set(targets) if targets else expected_targets
    if not requested <= expected_targets:
        raise ValueError(f"Missing requested targets: {sorted(requested - expected_targets)}")
    families = family_map()
    frames, counts, provenance = [], [], []
    for seed in seeds:
        stats, retrieved = legacy_rows(reference, seed, percentile)
        for target in sorted(requested):
            if target not in families:
                raise ValueError(f"Target without a curated family: {target}")
            if not cache.has(seed, target, METHOD):
                raise ValueError(f"Missing/corrupt ECFP4 ranking cache: {target}/{seed}")
            pool, methods = cache.load(seed, target, [METHOD])
            scores = methods[METHOD]["scores"]
            mask, cutoff = select_hits(scores, percentile)
            chosen = pool.loc[mask].copy()
            chosen["similarity"] = scores[mask]
            chosen["group"] = np.where(chosen.is_active, "TP", "putative_FP")
            chosen["target"], chosen["partition"] = target, seed
            chosen["protein_family"] = families[target]
            old = stats.loc[target]
            if (int(mask.sum()) != int(old.n_cumulative) or
                    int(chosen.is_active.sum()) != int(old.actives_cumulative) or
                    not np.isclose(cutoff, old.score_cut, rtol=0, atol=1e-5)):
                raise ValueError(f"Retrieved hits differ from legacy counts/cutoff: {target}/{seed}")
            if target in retrieved.index:
                previous = retrieved.loc[target]
                for group, column in (("TP", "retrieved_active_ids"),
                                      ("putative_FP", "retrieved_inactive_ids")):
                    ids = set(map(str, ast.literal_eval(previous[column])))
                    if set(chosen.loc[chosen.group.eq(group), "compound_id"].astype(str)) != ids:
                        raise ValueError(f"Retrieved IDs differ from legacy lists: {target}/{seed}/{group}")
            frames.append(chosen)
            counts.append({"target": target, "protein_family": families[target], "partition": seed,
                           "percentile": percentile, "score_cut": cutoff, "n_pool": len(pool),
                           "legacy_score_cut": float(old.score_cut),
                           "score_cut_absolute_difference": abs(cutoff - float(old.score_cut)),
                           "n_selected": len(chosen), "n_TP": int(chosen.is_active.sum()),
                           "n_putative_FP": int((~chosen.is_active).sum()),
                           "precision": float(chosen.is_active.mean()),
                           "legacy_counts_validated": True,
                           "legacy_ID_sets_validated": target in retrieved.index})
            provenance.append({"target": target, "partition": seed,
                               "metadata_sha256": sha256_file(cache._partition_dir(seed, target) /
                                                              (METHOD + ".json"))})
        print(f"seed={seed}: reused and validated {len(requested)} ECFP4 rankings", flush=True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(counts), provenance, signature


def add_descriptors(hits, smiles_path):
    source = pd.read_parquet(smiles_path, columns=["chem_comp_id", "smiles"])
    source["chem_comp_id"] = source.chem_comp_id.astype(str)
    source = source.loc[source.chem_comp_id.isin(hits.compound_id.astype(str))].copy()
    if source.chem_comp_id.duplicated().any():
        # Ambiguous lookup must not silently choose a structure.
        if source.groupby("chem_comp_id").smiles.nunique(dropna=False).gt(1).any():
            raise ValueError("Conflicting SMILES for a selected compound ID.")
        source = source.drop_duplicates("chem_comp_id")
    descriptors = {}
    valid_smiles = source.smiles.dropna().unique()
    for index, smiles in enumerate(valid_smiles, 1):
        descriptors[smiles] = molecular_properties(smiles)
        if index % 10000 == 0:
            print(f"Descriptors: {index}/{len(valid_smiles)} unique stored SMILES", flush=True)
    records = [{"compound_id": row.chem_comp_id, "smiles": row.smiles,
                **descriptors.get(row.smiles, molecular_properties(None))}
               for row in source.itertuples()]
    properties = pd.DataFrame(records)
    result = hits.merge(properties, on="compound_id", how="left", validate="many_to_one")
    result["valid_structure"] = result.valid_structure.fillna(False).astype(bool)
    result["issue"] = result.issue.fillna("missing_compound_id")
    issues = []
    for row in result.loc[~result.valid_structure].itertuples():
        issues.append({"target": row.target, "protein_family": row.protein_family,
                       "partition": row.partition, "group": row.group,
                       "compound_id": row.compound_id, "issue": row.issue})
    valid = result.loc[result.valid_structure]
    # Preserve aliases as benchmark observations; report rather than collapse them.
    for (target, seed, canonical), cell in valid.groupby(["target", "partition", "canonical_smiles"]):
        if len(cell) > 1:
            issues.append({"target": target, "partition": seed,
                           "issue": "cross_group_exact_structure" if cell.group.nunique() > 1
                           else "duplicate_structure_IDs_retained",
                           "detail": json.dumps(cell.compound_id.astype(str).tolist())})
    return result, issues, len(descriptors)


def summaries(hits):
    rows = []
    for (target, family, seed, group), cell in hits.groupby(
            ["target", "protein_family", "partition", "group"]):
        for prop in PROPERTIES:
            values = cell[prop].dropna()
            q = values.quantile([0.25, 0.5, 0.75])
            rows.append({"target": target, "protein_family": family, "partition": seed,
                         "group": group, "property": prop, "n_selected": len(cell),
                         "n_valid": len(values), "q25": q.iloc[0], "median": q.iloc[1],
                         "q75": q.iloc[2], "mean": values.mean()})
    partitions = pd.DataFrame(rows)
    target = partitions.groupby(["target", "protein_family", "group", "property"], as_index=False).agg(
        n_partitions=("partition", "nunique"), n_selected_median=("n_selected", "median"),
        n_valid_median=("n_valid", "median"), q25=("q25", "median"),
        median=("median", "median"), q75=("q75", "median"), mean=("mean", "median"))
    return partitions, target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yml"))
    parser.add_argument("--reference-dir", type=Path,
                        default=Path(__file__).parent / "reproduction_output/representation_benchmark")
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--percentile", type=float, default=99)
    parser.add_argument("--targets", help="Optional comma-separated smoke-test targets")
    parser.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revalidate-checkpoint", action="store_true",
                        help="Reuse descriptors after a script-only change, checking every cached hit again")
    parser.add_argument("--skip-individual", action="store_true")
    parser.add_argument("--include-nonretrieved-background", action="store_true",
                        help="Add background below the cutoff as putative TN; keep FN out of the curves")
    parser.add_argument("--hits-dir", type=Path,
                        help="Previous two-group analysis, used only to reuse and validate descriptors")
    parser.add_argument("--descriptor-cache", type=Path,
                        help="Shared resumable descriptor cache for the three-group analysis")
    parser.add_argument("--descriptor-workers", type=int, default=8)
    args = parser.parse_args(argv)
    if not 0 < args.percentile < 100:
        parser.error("Percentile must be between 0 and 100.")
    seeds = [int(value) for value in args.seeds.split(",")]
    if len(set(seeds)) != len(seeds) or not set(seeds) <= set(SEEDS):
        parser.error("Seeds must be unique original benchmark partitions.")
    if args.descriptor_workers < 1:
        parser.error("Descriptor worker count must be positive.")
    if args.include_nonretrieved_background:
        from ligq2_evaluation.retrieved_background import run
        return run(args, sys.modules[__name__], seeds)
    cfg = load_config(args.config)
    smiles_path = resolve_path(cfg, "paths", "smiles_data")
    reference = args.reference_dir.resolve()
    roots = sorted((reference / "ranking_cache").glob("*"))
    if args.cache_root is None and len(roots) != 1:
        parser.error("Provide --cache-root when the reference has multiple/no cache signatures.")
    cache_root = (args.cache_root or roots[0]).resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    targets = sorted(args.targets.split(",")) if args.targets else None
    # Resume is guarded by hashes of all score metadata and legacy tables, plus the
    # exact SMILES source. RankingCache verifies score/pool hashes on first use.
    fingerprint = {"schema_version": 1, "script_sha256": sha256_file(Path(__file__)),
                   "percentile": args.percentile, "targets": targets, "seeds": seeds,
                   "rdkit_version": rdBase.rdkitVersion,
                   "smiles_path": str(smiles_path), "smiles_sha256": sha256_file(smiles_path),
                   "cache_root": str(cache_root),
                   "cache_metadata": {str(p.relative_to(cache_root)): sha256_file(p)
                                      for p in sorted(cache_root.glob("seed_*/*/*.json"))},
                   "legacy_tables": {str(p): sha256_file(p) for seed in seeds for p in
                                     [reference / f"seed_{seed}" / name for name in
                                      ("df_long_target_all_methods.csv", "retrieved_active_sets_all_methods.csv")]}}
    checkpoint = out / "analysis_checkpoint.json"
    if args.resume and checkpoint.exists():
        saved = json.loads(checkpoint.read_text())
        script_only_change = ({k: v for k, v in saved["fingerprint"].items() if k != "script_sha256"} ==
                              {k: v for k, v in fingerprint.items() if k != "script_sha256"})
        if saved["fingerprint"] != fingerprint and not (
                args.revalidate_checkpoint and script_only_change):
            raise ValueError("Resume inputs/settings changed; use a new output directory. "
                             "A script-only change may use --revalidate-checkpoint.")
        hits = pd.read_parquet(out / "retrieved_hits.parquet")
        counts = pd.read_csv(out / "hit_counts_by_partition.csv")
        issues, provenance, signature = saved["issues"], saved["cache_provenance"], saved["cache_signature"]
        n_structures = saved["n_unique_smiles"]
        if args.revalidate_checkpoint:
            current, counts, provenance, signature = collect_hits(
                cache_root, reference, args.percentile, targets, seeds)
            keys = ["target", "partition", "compound_id"]
            cols = keys + ["is_active", "similarity", "group", "protein_family"]
            left = current[cols].sort_values(keys).reset_index(drop=True)
            right = hits[cols].sort_values(keys).reset_index(drop=True)
            pd.testing.assert_frame_equal(left, right, check_exact=True)
            counts.to_csv(out / "hit_counts_by_partition.csv", index=False)
            saved.update(fingerprint=fingerprint, cache_provenance=provenance, cache_signature=signature)
            checkpoint.write_text(json.dumps(saved, indent=2) + "\n")
            print("Every saved hit, score and label revalidated; descriptors unchanged", flush=True)
        print("Reusing validated selected hits and descriptors", flush=True)
    else:
        hits, counts, provenance, signature = collect_hits(
            cache_root, reference, args.percentile, targets, seeds)
        hits, issues, n_structures = add_descriptors(hits, smiles_path)
        hits.to_parquet(out / "retrieved_hits.parquet", index=False)
        counts.to_csv(out / "hit_counts_by_partition.csv", index=False)
        checkpoint.write_text(json.dumps({"fingerprint": fingerprint, "issues": issues,
                                         "cache_provenance": provenance, "cache_signature": signature,
                                         "n_unique_smiles": n_structures}, indent=2) + "\n")
    edges = make_edges(hits)
    histograms, exclusions = build_histograms(hits, edges)
    if histograms.empty:
        raise ValueError("No paired TP/putative-FP distributions.")
    target_histograms, family_histograms, overall = aggregate_histograms(histograms)
    for name, table in (("histograms_by_partition", histograms),
                        ("histograms_by_target", target_histograms),
                        ("histograms_by_family", family_histograms),
                        ("histograms_category_balanced", overall)):
        table.to_csv(out / (name + ".csv"), index=False)
    partition_summary, target_summary = summaries(hits)
    partition_summary.to_csv(out / "properties_by_partition.csv", index=False)
    target_summary.to_csv(out / "properties_by_target.csv", index=False)
    counts.groupby(["target", "protein_family"], as_index=False).agg(
        n_partitions=("partition", "nunique"), n_TP_median=("n_TP", "median"),
        n_putative_FP_median=("n_putative_FP", "median"),
        precision_median=("precision", "median"), score_cut_median=("score_cut", "median")
    ).to_csv(out / "hit_counts_by_target.csv", index=False)
    pd.DataFrame(issues + exclusions, columns=["target", "protein_family", "partition",
                                             "group", "compound_id", "property", "issue", "detail"]
                 ).to_csv(out / "data_quality_report.csv", index=False)
    label = f"ECFP4 hits at percentile {args.percentile:g} (inclusive cutoff)"
    plot_panel(overall, out / "retrieved_hits_distributions", label,
               "Equal category weights; within each category, equal target and partition weights.")
    for family, table in family_histograms.groupby("protein_family"):
        plot_panel(table, out / "by_family" / family.lower().replace(" ", "_") /
                   "distributions", label + " | " + family,
                   "Mean normalized distributions: equal target and partition weights.", ("png", "pdf"))
    if not args.skip_individual:
        for target, table in target_histograms.groupby("target"):
            family = table.protein_family.iloc[0]
            plot_panel(table, out / "individual_distributions/by_target" / target /
                       "distributions", f"{label} | {target.upper()} ({family})",
                       "Mean of normalized partition distributions; each partition weighs equally.",
                       ("png", "pdf"))
        cells = histograms.groupby(["target", "partition"])
        for index, ((target, seed), table) in enumerate(cells, 1):
            cell = counts.loc[counts.target.eq(target) & counts.partition.eq(seed)].iloc[0]
            plot_panel(table, out / "individual_distributions/by_partition" / target /
                       f"seed_{seed}" / "distributions", f"{label} | {target.upper()} | seed {seed}",
                       f"TP: {cell.n_TP:,}; putative FP: {cell.n_putative_FP:,}. "
                       "Each group normalized separately.", ("png", "pdf"))
            if index % 25 == 0:
                print(f"Individual partition panels: {index}/{len(cells)}", flush=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint, "cache_signature": signature,
                "cache_provenance": provenance,
                "method": "ECFP4, radius 2, 1024 bits; cached maximum Tanimoto to known seeds",
                "selection": "score >= numpy.percentile(full evaluation pool, percentile); ties retained",
                "descriptor_functions": {"MW": "Descriptors.MolWt", "logP": "Crippen.MolLogP",
                                         "HBA": "CalcNumHBA", "HBD": "CalcNumHBD"},
                "structure_policy": "Stored SMILES as-is; no structure deduplication, salt removal, "
                                    "neutralization, or tautomer transformation; canonicalization for QC only",
                "aggregation": "Normalize each TP/FP histogram separately; arithmetic mean of partitions "
                               "within target, targets within family, then equally weighted families",
                "empty_group_policy": "Exclude that property/partition from BOTH group histograms and report it",
                "quartile_summary": "Per-group quartiles in each partition, then median of each statistic over partitions",
                "bins": {key: value.tolist() for key, value in edges.items()},
                "n_targets": hits.target.nunique(), "n_partitions": len(counts),
                "n_categories": hits.protein_family.nunique(), "n_selected_observations": len(hits),
                "n_unique_selected_ids": hits.compound_id.nunique(), "n_unique_smiles": n_structures,
                "n_TP_observations": int(hits.group.eq("TP").sum()),
                "n_putative_FP_observations": int(hits.group.eq("putative_FP").sum()),
                "n_invalid_structure_observations": int((~hits.valid_structure).sum()),
                "n_quality_issues": len(issues), "n_empty_distribution_pairs": len(exclusions),
                "individual_figures_generated": not args.skip_individual or (
                    len(list((out / "individual_distributions/by_target").glob("*/distributions.png"))) ==
                    hits.target.nunique() and
                    len(list((out / "individual_distributions/by_partition").glob("*/seed_*/distributions.png"))) ==
                    len(counts)),
                "interpretation": "Conditional distributions among selected hits; FP are putative, "
                                  "not experimentally confirmed inactives; shape is independent of hit counts."}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = (f"{metadata['n_targets']} targets; {metadata['n_partitions']} target-partitions; "
               f"{metadata['n_categories']} categories\n"
               f"Selected observations: {len(hits):,} (TP {metadata['n_TP_observations']:,}; "
               f"putative FP {metadata['n_putative_FP_observations']:,})\n"
               f"Invalid structure observations: {metadata['n_invalid_structure_observations']}; "
               f"empty distribution pairs: {len(exclusions)}; QC records: {len(issues)}\n"
               "Legacy counts, thresholds and available compound-ID sets validated.\n"
               "Normalized distributions describe selected hits, not all actives/background.\n"
               f"Output directory: {out}\nMain panel: {out / 'retrieved_hits_distributions.png'}\n"
               f"Individual panels: {out / 'individual_distributions'}\n")
    (out / "summary.txt").write_text(summary)
    print(summary, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
