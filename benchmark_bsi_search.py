#!/usr/bin/env python

from __future__ import annotations

import argparse
import itertools
import resource
import time
from pathlib import Path

import pandas as pd
import torch

from compound_processing.bsi_search import (
    BSI_MODEL_SUBDIR,
    BSI_REPRESENTATION,
    BSIModelRegistry,
    _select_representative_ligands,
    search_bsi_against_target,
)
from compound_processing.compound_helpers import LigandStore


def _parse_csv_ints(value: str) -> list[int]:
    out = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out or any(x <= 0 for x in out):
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers.")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BSI search over a LigQ_2 compound provider.")
    parser.add_argument("--data-dir", type=Path, default=Path("databases"))
    parser.add_argument("--ligand-provider", default="zinc")
    parser.add_argument("--protein-id", default=None, help="Reference protein to use as seed source.")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--bsi-threshold", type=float, default=0.5)
    parser.add_argument("--target-limit", type=int, default=100000, help="Limit target ligands for benchmark; use 0 for full base.")
    parser.add_argument("--seed-counts", type=_parse_csv_ints, default=[1, 5, 10])
    parser.add_argument("--target-chunk-sizes", type=_parse_csv_ints, default=[25000, 50000])
    parser.add_argument("--model-batch-sizes", type=_parse_csv_ints, default=[32768, 65536])
    parser.add_argument("--per-chunk-topk", type=int, default=1000)
    parser.add_argument("--global-topk", type=int, default=50000)
    parser.add_argument("--output-tsv", type=Path, default=Path("results_benchmarks/bsi_search_benchmark.tsv"))
    return parser.parse_args()


def _choose_protein(known: pd.DataFrame, domains: pd.DataFrame, registry: BSIModelRegistry) -> str:
    supported = domains[domains["pfam_id"].astype(str).isin(registry.supported_pfams)]
    candidates = supported["uniprot_id"].astype(str).drop_duplicates()
    known_prots = set(known["uniprot_id"].astype(str).unique())
    for prot in candidates:
        if prot in known_prots:
            return str(prot)
    raise RuntimeError("No protein with known ligands and supported BSI PFAM was found.")


def _rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir
    target_limit = None if int(args.target_limit) <= 0 else int(args.target_limit)

    registry = BSIModelRegistry(data_dir / BSI_MODEL_SUBDIR)
    known = pd.read_parquet(data_dir / "results_databases" / "known_binding_data.parquet")
    domains = pd.read_parquet(data_dir / "results_databases" / "protein_domains.parquet")
    protein_id = str(args.protein_id) if args.protein_id else _choose_protein(known, domains, registry)
    pfam_id = registry.select_best_pfam(
        domains.loc[domains["uniprot_id"].astype(str) == protein_id, "pfam_id"].astype(str).tolist()
    )
    if pfam_id is None:
        raise RuntimeError(f"Protein {protein_id} has no supported BSI PFAM.")

    store_ref = LigandStore(data_dir / "compound_data" / "pdb_chembl")
    store_target = LigandStore(data_dir / "compound_data" / args.ligand_provider)
    rep_ref = store_ref.load_representation(BSI_REPRESENTATION)
    rep_target = store_target.load_representation(BSI_REPRESENTATION)
    model, params = registry.load(pfam_id, device=args.device)
    fp_bits = int(params.get("fp_bits", 1024))
    representatives = _select_representative_ligands(
        prot=protein_id,
        known_binding=known,
        rep_pdb_chembl=rep_ref,
        max_queries=max(args.seed_counts),
        cluster_threshold=0.8,
    )
    if not representatives:
        raise RuntimeError(f"No BSI seed representatives found for protein {protein_id}.")

    rows: list[dict] = []
    for n_seeds, chunk_size, batch_size in itertools.product(
        args.seed_counts,
        args.target_chunk_sizes,
        args.model_batch_sizes,
    ):
        query_ids = representatives[: min(int(n_seeds), len(representatives))]
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        hits = search_bsi_against_target(
            query_ids=query_ids,
            model=model,
            fp_bits=fp_bits,
            store_ref=store_ref,
            rep_ref=rep_ref,
            store_target=store_target,
            rep_target=rep_target,
            threshold=args.bsi_threshold,
            device=args.device,
            target_chunk_size=int(chunk_size),
            model_batch_size=int(batch_size),
            per_chunk_topk=args.per_chunk_topk,
            global_topk=args.global_topk,
            compound_prefix="ZINC" if args.ligand_provider == "zinc" else "",
            target_limit=target_limit,
        )
        elapsed = time.perf_counter() - t0
        n_target = rep_target.n_ligands if target_limit is None else min(target_limit, rep_target.n_ligands)
        rows.append(
            {
                "protein_id": protein_id,
                "pfam_id": pfam_id,
                "provider": args.ligand_provider,
                "device": args.device,
                "n_target_ligands": n_target,
                "n_seeds": len(query_ids),
                "target_chunk_size": int(chunk_size),
                "model_batch_size": int(batch_size),
                "bsi_threshold": float(args.bsi_threshold),
                "elapsed_s": elapsed,
                "ligands_per_s": (n_target * len(query_ids)) / elapsed if elapsed > 0 else 0.0,
                "hits": len(hits),
                "peak_rss_mb": _rss_mb(),
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated() / (1024**2)
                    if args.device == "cuda" and torch.cuda.is_available()
                    else 0.0
                ),
            }
        )
        print(rows[-1])

    out = pd.DataFrame(rows)
    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_tsv, sep="\t", index=False)
    print(f"[INFO] Wrote benchmark results to: {args.output_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
