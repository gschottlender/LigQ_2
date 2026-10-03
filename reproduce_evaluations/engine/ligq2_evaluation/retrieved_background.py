"""Stream TP, putative FP and putative TN characterization from cached rankings.

Descriptor computations are cached by stored SMILES, but benchmark observations
remain compound IDs. Full candidate tables are never pooled across targets.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import inspect
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase

from .config import load_config, resolve_path
from .fixed_budget import RankingCache
from .provenance import sha256_file

GROUPS = ("TP", "putative_FP", "putative_TN")
DESCRIPTOR_COLUMNS = ("canonical_smiles", "molecular_weight", "logp", "hba", "hbd",
                      "hba_minus_hbd", "valid_structure", "issue")


def classify_candidates(is_active, scores, cutoff):
    """The cutoff is inclusive for hits; ties can never become TN or FN."""
    active = np.asarray(is_active, dtype=bool)
    selected = np.asarray(scores) >= cutoff
    return np.where(active, np.where(selected, "TP", "FN"),
                    np.where(selected, "putative_FP", "putative_TN"))


def quiet_worker():
    # RDKit errors may include structures. Save failures to local QC tables only.
    rdBase.DisableLog("rdApp.error")
    rdBase.DisableLog("rdApp.warning")


def descriptor_cache(source, folder, analysis, source_hash, workers, hits_dir=None):
    """Resumable content-keyed chunks; reuse old hit descriptors only after checks."""
    folder.mkdir(parents=True, exist_ok=True)
    signature = {"schema": 1, "smiles_sha256": source_hash,
                 "rdkit_version": rdBase.rdkitVersion,
                 "descriptor_function_sha256": hashlib.sha256(
                     inspect.getsource(analysis.molecular_properties).encode()).hexdigest()}
    manifest = folder / "input_signature.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Descriptor cache input/chemistry changed; use a different cache directory.")
    manifest.write_text(json.dumps(signature, indent=2) + "\n")
    frames = [pd.read_parquet(p) for p in sorted(folder.glob("chunk_*.parquet"))]
    reused = 0
    if hits_dir is not None:
        old_metadata = json.loads((hits_dir / "run_metadata.json").read_text())
        old_parameters = old_metadata["fingerprint"]
        if (old_parameters["rdkit_version"] != rdBase.rdkitVersion or
                old_parameters["smiles_sha256"] != source_hash):
            raise ValueError("Previous hit descriptors use a different structure source or RDKit version.")
        old = pd.read_parquet(hits_dir / "retrieved_hits.parquet")
        lookup = source.set_index("chem_comp_id").smiles
        matching = old.loc[old.compound_id.isin(lookup.index)]
        if not matching.smiles.reset_index(drop=True).equals(
                matching.compound_id.map(lookup).reset_index(drop=True)):
            raise ValueError("Previous hit SMILES do not match the current compound source.")
        raw = old[["smiles", *DESCRIPTOR_COLUMNS]].drop_duplicates()
        raw = raw.loc[raw.smiles.notna()]
        if raw.smiles.duplicated().any():
            raise ValueError("Conflicting previous descriptors for the same stored SMILES.")
        # Verify the implemented descriptor contract against representative cached records.
        with rdBase.BlockLogs():
            for record in raw.head(100).to_dict("records"):
                recalculated = analysis.molecular_properties(record["smiles"])
                for key in ("canonical_smiles", "molecular_weight", "logp", "hba", "hbd", "hba_minus_hbd"):
                    if record["valid_structure"] and recalculated[key] != record[key]:
                        raise ValueError("Previous descriptors disagree with the current implementation.")
        frames.append(raw)
        reused = len(raw)
    cached = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["smiles", *DESCRIPTOR_COLUMNS])
    cached = cached.drop_duplicates()
    if cached.smiles.duplicated().any():
        raise ValueError("Conflicting descriptors in the shared cache.")
    required = source.smiles.dropna().unique().tolist()
    completed = set(cached.smiles)
    pending = [smiles for smiles in required if smiles not in completed]
    print(f"Descriptors: {len(required):,} distinct stored SMILES; "
          f"{len(cached):,} reused ({reused:,} from previous hits); {len(pending):,} to compute", flush=True)
    new_frames = []
    context = multiprocessing.get_context("fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=quiet_worker) as executor:
        for start in range(0, len(pending), 10000):
            batch = pending[start:start + 10000]
            values = list(executor.map(analysis.molecular_properties, batch, chunksize=250))
            table = pd.DataFrame([{"smiles": smiles, **properties}
                                  for smiles, properties in zip(batch, values)]).reindex(
                                      columns=["smiles", *DESCRIPTOR_COLUMNS])
            digest = hashlib.sha256("\0".join(batch).encode()).hexdigest()[:20]
            path = folder / f"chunk_{digest}.parquet"
            temporary = path.with_suffix(".parquet.tmp")
            table.to_parquet(temporary, index=False)
            temporary.replace(path)
            new_frames.append(table)
            print(f"New descriptors: {min(start + len(batch), len(pending)):,}/{len(pending):,}", flush=True)
    if new_frames:
        cached = pd.concat([cached, *new_frames], ignore_index=True)
    # Store reused records too, so this cache becomes independently reusable.
    cached.to_parquet(folder / "chunk_reused.parquet", index=False)
    properties = source.rename(columns={"chem_comp_id": "compound_id"}).merge(
        cached, on="smiles", how="left", validate="many_to_one")
    properties["valid_structure"] = properties.valid_structure.fillna(False).astype(bool)
    properties["issue"] = properties.issue.fillna("missing_smiles")
    properties.to_parquet(folder / "compound_descriptors.parquet", index=False)
    return properties, {"signature": signature, "n_required_unique_smiles": len(required),
                        "n_computed": len(pending), "n_reused": len(cached) - len(pending),
                        "n_invalid_compound_IDs": int((~properties.valid_structure).sum())}


def cell_quality(cell):
    rows = []
    for row in cell.loc[~cell.valid_structure].itertuples():
        rows.append({"issue": row.issue, "group": row.group, "compound_id": row.compound_id})
    valid = cell.loc[cell.valid_structure]
    duplicates = valid.loc[valid.canonical_smiles.duplicated(keep=False)]
    for _, aliases in duplicates.groupby("canonical_smiles"):
        rows.append({"issue": "cross_group_exact_structure" if aliases.group.nunique() > 1
                     else "duplicate_structure_IDs_retained",
                     "detail": json.dumps(aliases.compound_id.astype(str).tolist())})
    return rows


def run(args, analysis, seeds):
    import plot_retrieved_hit_distributions as plotting

    cfg = load_config(args.config)
    smiles_path = resolve_path(cfg, "paths", "smiles_data")
    source_hash = sha256_file(smiles_path)
    reference = args.reference_dir.resolve()
    roots = sorted((reference / "ranking_cache").glob("*"))
    if args.cache_root is None and len(roots) != 1:
        raise ValueError("Provide --cache-root when multiple/no cache signatures exist.")
    cache_root = (args.cache_root or roots[0]).resolve()
    out = args.output_dir.resolve()
    hits_dir = args.hits_dir.resolve() if args.hits_dir else None
    if hits_dir == out or (out / "retrieved_hits.parquet").exists():
        raise ValueError("Use a separate output directory; the previous two-group analysis is preserved.")
    out.mkdir(parents=True, exist_ok=True)
    targets = sorted(value.strip() for value in args.targets.split(",")) if args.targets else None
    # Reconstruct and validate the original selected compounds before adding any TN.
    original_hits, original_counts, provenance, signature = analysis.collect_hits(
        cache_root, reference, args.percentile, targets, seeds)
    cache = RankingCache(cache_root.parent.parent, signature)
    families = analysis.family_map()
    source = pd.read_parquet(smiles_path, columns=["chem_comp_id", "smiles"])
    source["chem_comp_id"] = source.chem_comp_id.astype(str)
    if source.chem_comp_id.duplicated().any():
        raise ValueError("Compound source must map each ID unambiguously.")
    if targets:
        required_ids = set(original_hits.compound_id)
        for row in original_counts.itertuples():
            pool_path, _ = cache._pool_paths(row.partition, row.target)
            required_ids.update(pd.read_parquet(pool_path, columns=["compound_id"]).compound_id)
        source = source.loc[source.chem_comp_id.isin(required_ids)].copy()
    descriptor_folder = (args.descriptor_cache or out / "descriptor_cache").resolve()
    properties, descriptor_info = descriptor_cache(
        source, descriptor_folder, analysis, source_hash, args.descriptor_workers, hits_dir)
    edges = plotting.fixed_width_edges(properties)
    indexed = properties.set_index("compound_id")
    fingerprint = {"schema": 1, "module_sha256": sha256_file(Path(__file__)),
                   "primary_script_sha256": sha256_file(Path(analysis.__file__)),
                   "source_sha256": source_hash, "rdkit_version": rdBase.rdkitVersion,
                   "percentile": args.percentile, "targets": targets, "seeds": seeds,
                   "cache_root": str(cache_root), "cache_signature": signature,
                   "validated_cache_provenance": provenance,
                   "bins": {key: value.tolist() for key, value in edges.items()},
                   "groups": GROUPS}
    # Normalize tuple-valued fields before equality checks against JSON.
    fingerprint = json.loads(json.dumps(fingerprint))
    manifest = out / "analysis_inputs.json"
    if manifest.exists() and json.loads(manifest.read_text()) != fingerprint:
        raise ValueError("Analysis inputs/settings changed; use another output directory.")
    manifest.write_text(json.dumps(fingerprint, indent=2) + "\n")
    hist_frames, summary_frames, count_rows, issues, empty_pairs = [], [], [], [], []
    for index, old in enumerate(original_counts.itertuples(), 1):
        target, seed = old.target, old.partition
        folder = out / "checkpoints" / target / f"seed_{seed}"
        done = folder / "status.json"
        if args.resume and done.exists():
            status = json.loads(done.read_text())
            for name, digest in status["file_hashes"].items():
                if sha256_file(folder / name) != digest:
                    raise ValueError(f"Corrupt analysis checkpoint: {target}/{seed}/{name}")
            hist_frames.append(pd.read_parquet(folder / "histograms.parquet"))
            summary_frames.append(pd.read_csv(folder / "properties.csv"))
            count_rows.append(status["counts"])
            issues.extend(status["quality"])
            empty_pairs.extend(status["empty_pairs"])
            continue
        pool, rankings = cache.load(seed, target, [analysis.METHOD])
        scores = rankings[analysis.METHOD]["scores"]
        mask, cutoff = analysis.select_hits(scores, args.percentile)
        if cutoff != old.score_cut:
            raise ValueError(f"Cutoff changed during analysis: {target}/{seed}")
        groups = classify_candidates(pool.is_active, scores, cutoff)
        # FN are counted, not mixed into the putative TN curve.
        include = groups != "FN"
        cell = pool.loc[include, ["compound_id"]].copy()
        cell["similarity"], cell["group"] = scores[include], groups[include]
        cell = cell.join(indexed, on="compound_id", validate="many_to_one")
        cell["valid_structure"] = cell.valid_structure.fillna(False).astype(bool)
        cell["issue"] = cell.issue.fillna("missing_compound_id")
        cell["target"], cell["protein_family"], cell["partition"] = target, families[target], seed
        selected = cell.loc[cell.group.isin(analysis.GROUPS), ["compound_id", "similarity", "group"]]
        previous = original_hits.loc[original_hits.target.eq(target) & original_hits.partition.eq(seed),
                                     ["compound_id", "similarity", "group"]]
        pd.testing.assert_frame_equal(selected.sort_values("compound_id").reset_index(drop=True),
                                      previous.sort_values("compound_id").reset_index(drop=True), check_exact=True)
        hist, exclusions = analysis.build_histograms(cell, edges, groups=GROUPS)
        partition_summary, _ = analysis.summaries(cell)
        partition_summary = partition_summary.rename(columns={"n_selected": "n_observations"})
        qc = cell_quality(cell)
        for issue in qc:
            issue.update(target=target, protein_family=families[target], partition=seed)
        counts = {key: value for key, value in old._asdict().items() if key != "Index"}
        counts.update(n_putative_TN=int(np.sum(groups == "putative_TN")),
                      n_FN_not_plotted=int(np.sum(groups == "FN")),
                      n_background_total=int((~pool.is_active).sum()),
                      n_invalid_structure_observations=int((~cell.valid_structure).sum()),
                      selected_hits_unchanged=True)
        if (counts["n_putative_FP"] + counts["n_putative_TN"] != counts["n_background_total"] or
                counts["n_TP"] + counts["n_putative_FP"] + counts["n_putative_TN"] +
                counts["n_FN_not_plotted"] != counts["n_pool"]):
            raise ValueError("Candidate classification does not exhaust the evaluation pool.")
        folder.mkdir(parents=True, exist_ok=True)
        hist.to_parquet(folder / "histograms.parquet", index=False)
        partition_summary.to_csv(folder / "properties.csv", index=False)
        status = {"counts": counts, "quality": qc, "empty_pairs": exclusions,
                  "file_hashes": {name: sha256_file(folder / name) for name in
                                  ("histograms.parquet", "properties.csv")}}
        done.write_text(json.dumps(status, indent=2) + "\n")
        hist_frames.append(hist)
        summary_frames.append(partition_summary)
        count_rows.append(counts)
        issues.extend(qc)
        empty_pairs.extend(exclusions)
        if index % 20 == 0:
            print(f"Three-group distributions: {index}/{len(original_counts)} target-partitions", flush=True)
    histograms = pd.concat(hist_frames, ignore_index=True)
    histograms.to_parquet(out / "histograms_by_partition.parquet", index=False)
    target_hist, family_hist, overall = analysis.aggregate_histograms(histograms)
    for name, table in (("by_partition", histograms), ("by_target", target_hist),
                        ("by_family", family_hist), ("category_balanced", overall)):
        table.to_csv(out / f"histograms_{name}.csv", index=False)
    summaries = pd.concat(summary_frames, ignore_index=True)
    summaries.to_csv(out / "properties_by_partition.csv", index=False)
    summaries.groupby(["target", "protein_family", "group", "property"], as_index=False).agg(
        n_partitions=("partition", "nunique"), n_observations_median=("n_observations", "median"),
        n_valid_median=("n_valid", "median"), q25=("q25", "median"), median=("median", "median"),
        q75=("q75", "median"), mean=("mean", "median")
    ).to_csv(out / "properties_by_target.csv", index=False)
    counts = pd.DataFrame(count_rows)
    counts.to_csv(out / "hit_counts_by_partition.csv", index=False)
    counts.groupby(["target", "protein_family"], as_index=False).agg(
        n_partitions=("partition", "nunique"), n_TP_median=("n_TP", "median"),
        n_putative_FP_median=("n_putative_FP", "median"), n_putative_TN_median=("n_putative_TN", "median"),
        n_FN_not_plotted_median=("n_FN_not_plotted", "median"), precision_median=("precision", "median")
    ).to_csv(out / "hit_counts_by_target.csv", index=False)
    pd.DataFrame(issues + empty_pairs, columns=["target", "protein_family", "partition", "group",
                                             "compound_id", "property", "issue", "detail"]
                 ).to_csv(out / "data_quality_report.csv", index=False)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "fingerprint": fingerprint,
                "bins": {key: values.tolist() for key, values in edges.items()},
                "descriptor_cache": str(descriptor_folder), "descriptor_info": descriptor_info,
                "groups": GROUPS, "n_targets": counts.target.nunique(), "n_partitions": len(counts),
                "n_categories": counts.protein_family.nunique(),
                "n_TP_observations": int(counts.n_TP.sum()),
                "n_putative_FP_observations": int(counts.n_putative_FP.sum()),
                "n_putative_TN_observations": int(counts.n_putative_TN.sum()),
                "n_FN_not_plotted_observations": int(counts.n_FN_not_plotted.sum()),
                "n_invalid_structure_observations": int(counts.n_invalid_structure_observations.sum()),
                "n_quality_records": len(issues), "empty_distribution_pairs": empty_pairs,
                "selection": "TP/putative FP: score >= full-pool percentile; putative TN: background score < cutoff",
                "structure_policy": "Same stored SMILES and descriptors as two-group analysis; canonicalization "
                                    "for QC only; compound-ID observations retained, without structure deduplication",
                "aggregation": "Separate normalized group histograms; mean partitions within target; "
                               "mean targets within category; equal category weights",
                "membership_storage": "Reconstruct all group memberships from hashed cached pool/scores and cutoff; "
                                      "no giant pooled per-compound TN table",
                "interpretation": "All evaluated background represented by FP plus TN; held-out actives below "
                                  "cutoff are FN and not plotted. TN/FP are putative, not confirmed biological labels. "
                                  "Within each target/partition, similarity separation FP/TN follows from the cutoff."}
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = (f"{len(counts)} target-partitions; {counts.target.nunique()} targets\n"
               f"TP {counts.n_TP.sum():,}; putative FP {counts.n_putative_FP.sum():,}; "
               f"putative TN {counts.n_putative_TN.sum():,}; FN not plotted {counts.n_FN_not_plotted.sum():,}\n"
               f"Invalid structure observations: {counts.n_invalid_structure_observations.sum():,}; "
               f"QC records: {len(issues):,}; empty distribution pairs: {len(empty_pairs)}\n"
               "Original selected compound IDs/scores/labels unchanged.\n"
               f"Outputs: {out}\n")
    (out / "summary.txt").write_text(summary)
    print(summary, flush=True)
    arguments = ["--input-dir", str(out)]
    if args.skip_individual:
        arguments.append("--skip-individual")
    return plotting.main(arguments)
