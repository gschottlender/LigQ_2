#!/usr/bin/env python

"""Precompute a predicted-ligand cache without producing query result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from compound_processing.device_utils import resolve_torch_device
from query_processing.ligand_providers import build_provider
from query_processing.predicted_cache import (
    ensure_provider_cache,
    find_compatible_cache_dir,
)
from run_ligq_2 import (
    DEFAULT_SEARCH_THRESHOLDS_BY_REPRESENTATION,
    ensure_base_data_from_hf,
    ensure_known_binding_table,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Precompute predicted ligands for every protein in the installed "
            "LigQ target FASTA. This updates only the shared predicted cache; "
            "it does not run BLAST/HMMER or create per-query results."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--data-dir",
        type=Path,
        default=Path("databases"),
        help="LigQ data root.",
    )
    parser.add_argument(
        "--protein-fasta",
        type=Path,
        default=None,
        help=(
            "FASTA whose unique header IDs define the proteins to cache. "
            "Defaults to <data-dir>/sequences/target_sequences.fasta."
        ),
    )
    parser.add_argument(
        "--ligand-provider",
        default="zinc",
        help="Target provider under compound_data/.",
    )
    parser.add_argument(
        "--search-representation",
        default="morgan_1024_r2",
        help=(
            "Installed representation name. It must exist for both the target "
            "provider and compound_data/pdb_chembl."
        ),
    )
    parser.add_argument(
        "--search-metric",
        choices=["tanimoto", "cosine"],
        default="tanimoto",
        help="Similarity metric associated with the representation.",
    )
    parser.add_argument(
        "--search-threshold",
        type=float,
        default=None,
        help=(
            "Minimum score stored in the cache. When omitted, use the same "
            "representation-specific default as run_ligq_2.py."
        ),
    )
    parser.add_argument(
        "--search-threshold-max",
        type=float,
        default=None,
        help="Optional inclusive maximum score stored in the cache.",
    )
    parser.add_argument("--cluster-threshold", type=float, default=0.8)
    parser.add_argument(
        "--search-device",
        default="auto",
        help="Search backend device: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument(
        "--search-query-batch-size",
        type=int,
        default=None,
        help="Optional query batch size override for compound searches.",
    )
    parser.add_argument(
        "--search-target-chunk-size",
        type=int,
        default=None,
        help="Optional target database chunk size override.",
    )
    parser.add_argument(
        "--search-per-iteration-topk",
        type=int,
        default=1000,
        help="Maximum predicted hits retained per iteration/chunk.",
    )
    parser.add_argument(
        "--search-global-topk",
        type=int,
        default=10000,
        help="Maximum global predicted hits retained per protein.",
    )
    parser.add_argument(
        "--force-rebuild-known-binding",
        action="store_true",
        help="Rebuild known_binding_data.parquet from merged local sources.",
    )
    parser.add_argument(
        "--force-rebuild-predicted-cache",
        action="store_true",
        help=(
            "Discard the exact requested cache namespace before computing. "
            "Without this flag, compatible work is resumed and reused."
        ),
    )
    parser.add_argument(
        "--hf-repo-id",
        default="gschottlender/LigQ_2",
        help="Hugging Face dataset used when required base data is missing.",
    )
    parser.add_argument(
        "--skip-hf-predicted-cache",
        action="store_true",
        help=(
            "When base data must be downloaded, exclude the optional published "
            "predicted cache."
        ),
    )
    return parser.parse_args(argv)


def read_fasta_protein_ids(path: Path) -> set[str]:
    """Return unique first-token FASTA header IDs, matching LigQ query ID rules."""
    if not path.is_file():
        raise FileNotFoundError(f"Protein FASTA not found: {path}")

    protein_ids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            if header:
                protein_ids.add(header.split()[0])

    if not protein_ids:
        raise ValueError(f"No protein IDs were found in FASTA headers: {path}")
    return protein_ids


def resolve_search_threshold(args: argparse.Namespace) -> float:
    if args.search_threshold is not None:
        return float(args.search_threshold)

    default_threshold = DEFAULT_SEARCH_THRESHOLDS_BY_REPRESENTATION.get(
        args.search_representation
    )
    if default_threshold is None:
        known_reps = ", ".join(sorted(DEFAULT_SEARCH_THRESHOLDS_BY_REPRESENTATION))
        raise ValueError(
            "No default --search-threshold is defined for representation "
            f"'{args.search_representation}'. Pass --search-threshold explicitly. "
            f"Known default representations: {known_reps}."
        )
    return float(default_threshold)


def validate_args(args: argparse.Namespace, threshold: float) -> None:
    if args.search_threshold_max is not None and args.search_threshold_max < threshold:
        raise ValueError("--search-threshold-max must be >= --search-threshold.")
    if args.search_query_batch_size is not None and args.search_query_batch_size <= 0:
        raise ValueError("--search-query-batch-size must be > 0.")
    if args.search_target_chunk_size is not None and args.search_target_chunk_size <= 0:
        raise ValueError("--search-target-chunk-size must be > 0.")
    if args.search_per_iteration_topk <= 0:
        raise ValueError("--search-per-iteration-topk must be > 0.")
    if args.search_global_topk <= 0:
        raise ValueError("--search-global-topk must be > 0.")


def _read_cached_protein_ids(cache_dir: Path) -> set[str]:
    index_path = cache_dir / "cached_proteins.json"
    if not index_path.is_file():
        raise RuntimeError(f"Completed cache is missing its protein index: {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list):
        raise RuntimeError(f"Invalid protein index in cache: {index_path}")
    return {str(value) for value in values}


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir.expanduser().resolve()
    threshold = resolve_search_threshold(args)
    validate_args(args, threshold)

    ensure_base_data_from_hf(
        data_dir=data_dir,
        repo_id=args.hf_repo_id,
        provider_name=args.ligand_provider,
        download_optional_predicted_cache=not args.skip_hf_predicted_cache,
    )

    protein_fasta = (
        args.protein_fasta.expanduser().resolve()
        if args.protein_fasta is not None
        else data_dir / "sequences" / "target_sequences.fasta"
    )
    proteins_needed = read_fasta_protein_ids(protein_fasta)
    known_binding = ensure_known_binding_table(
        data_dir=data_dir,
        force_rebuild=args.force_rebuild_known_binding,
    )

    resolved_device = resolve_torch_device(args.search_device)
    print(f"[INFO] Protein universe: {protein_fasta}")
    print(f"[INFO] Unique proteins to cache: {len(proteins_needed)}")
    print(f"[INFO] Ligand provider: {args.ligand_provider}")
    print(f"[INFO] Search representation: {args.search_representation}")
    print(f"[INFO] Search metric: {args.search_metric}")
    print(
        "[INFO] Cache score coverage: "
        f"{threshold} to {args.search_threshold_max if args.search_threshold_max is not None else 'unbounded'}"
    )
    print(f"[INFO] Effective search device: {resolved_device}")
    if str(args.search_device).lower().startswith("cuda") and resolved_device.type != "cuda":
        print(
            "[WARN] CUDA was requested but is not usable. LigQ will fall back to CPU; "
            "stop now if this was not intentional."
        )

    provider = build_provider(
        provider_name=args.ligand_provider,
        data_dir=data_dir,
        search_representation=args.search_representation,
        search_metric=args.search_metric,
        search_threshold=threshold,
        search_threshold_max=args.search_threshold_max,
        cluster_threshold=args.cluster_threshold,
        search_per_iteration_topk=args.search_per_iteration_topk,
        search_global_topk=args.search_global_topk,
        search_device=args.search_device,
        search_q_batch_size=args.search_query_batch_size,
        search_target_chunk_size=args.search_target_chunk_size,
    )

    ensure_provider_cache(
        data_dir=data_dir,
        provider=provider,
        known_binding=known_binding,
        proteins_needed=proteins_needed,
        force_rebuild_cache=args.force_rebuild_predicted_cache,
        load_dataframe=False,
    )

    cache_dir = find_compatible_cache_dir(
        data_dir=data_dir,
        provider=provider,
        proteins_needed=proteins_needed,
    )
    if cache_dir is None:
        raise RuntimeError("Cache generation finished but no compatible cache could be found.")

    cached_proteins = _read_cached_protein_ids(cache_dir)
    missing = proteins_needed - cached_proteins
    if missing:
        examples = ", ".join(sorted(missing)[:10])
        raise RuntimeError(
            "Predicted cache is incomplete after generation: "
            f"{len(missing)} of {len(proteins_needed)} proteins are missing. "
            f"Examples: {examples}"
        )

    print(f"[INFO] Compatible cache: {cache_dir}")
    print(f"[INFO] Verified cached protein coverage: {len(proteins_needed)}/{len(proteins_needed)}")
    print("[INFO] Predicted cache precomputation completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
