#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd

from compound_processing.bsi_search import BSI_DEFAULT_MAX_KNOWN_LIGANDS, BSI_MODEL_SUBDIR

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None

from query_processing.query_processing_functions import (
    prepare_complementary_databases,
    run_blast_sequence_search,
    build_nearest_k_candidates_from_blast,
    _load_ranked_blast_hits,
    filter_nearest_k_candidates_by_query_domains,
    run_hmmer_domain_search,
    map_pfam_hits_to_candidate_proteins,
    combine_sequence_and_domain_candidates,
    add_shared_domain_counts_to_candidates,
    parse_query_fasta,
    build_query_ligand_results_parallel,
)
from query_processing.results_tables import (
    build_protein_domains_table,
    build_known_binding_data,
    add_smiles_to_known_binding,
    save_known_binding_table,
)
from query_processing.ligand_providers import build_provider
from query_processing.predicted_cache import ensure_provider_cache


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


DEFAULT_CACHE_NAMESPACE = (
    "predicted_bindings/zinc/"
    "search_representation=morgan_1024_r2__search_metric=tanimoto__cache_threshold_min=0.4"
)

HF_CORE_REQUIRED_RELATIVE_PATHS = [
    "sequences",
    "results_databases/known_binding_data.parquet",
    "results_databases/protein_domains.parquet",
    "compound_data/pdb_chembl/ligands.parquet",
    "compound_data/pdb_chembl/reps/morgan_1024_r2.dat",
    "compound_data/pdb_chembl/reps/morgan_1024_r2.meta.json",
    "complementary_databases/blast",
    "complementary_databases/pfam",
]

HF_DEFAULT_PROVIDER_REQUIRED_RELATIVE_PATHS = [
    "compound_data/zinc/ligands.parquet",
    "compound_data/zinc/reps/morgan_1024_r2.dat",
    "compound_data/zinc/reps/morgan_1024_r2.meta.json",
]
HF_BSI_REQUIRED_RELATIVE_PATHS = [
    BSI_MODEL_SUBDIR.as_posix(),
]
HF_REQUIRED_RELATIVE_PATHS = (
    HF_CORE_REQUIRED_RELATIVE_PATHS + HF_DEFAULT_PROVIDER_REQUIRED_RELATIVE_PATHS
)

# Legacy internal alias kept to avoid breaking historical references.
HF_ZINC_REQUIRED_RELATIVE_PATHS = HF_DEFAULT_PROVIDER_REQUIRED_RELATIVE_PATHS

HF_OPTIONAL_CACHE_PATH_GROUPS = [
    [
        f"results_databases/{DEFAULT_CACHE_NAMESPACE}/manifest.json",
        f"results_databases/{DEFAULT_CACHE_NAMESPACE}/predicted_binding_data.parquet",
        f"results_databases/{DEFAULT_CACHE_NAMESPACE}/predicted_binding_progress.json",
        f"results_databases/{DEFAULT_CACHE_NAMESPACE}/cached_proteins.json",
        f"results_databases/{DEFAULT_CACHE_NAMESPACE}/predicted_binding_rowgroup_index.json",
    ],
]

DEFAULT_SEARCH_THRESHOLDS_BY_REPRESENTATION = {
    "chemberta_zinc_base_768": 0.936140,
    "rdkit_1024": 0.930324,
    "maccs": 0.831169,
    "ap_rdkit": 0.767087,
    "morgan_feature_1024_r2": 0.509451,
    "topological_torsion_rdkit_1024": 0.502932,
    "morgan_1024_r2": 0.415094,
}


def _required_base_paths(provider_name: str, include_bsi_models: bool = False) -> list[str]:
    required = list(HF_CORE_REQUIRED_RELATIVE_PATHS)
    if provider_name == "zinc":
        required.extend(HF_DEFAULT_PROVIDER_REQUIRED_RELATIVE_PATHS)
    if include_bsi_models:
        required.extend(HF_BSI_REQUIRED_RELATIVE_PATHS)
    return required


def _missing_required_base_paths(
    data_dir: Path,
    provider_name: str = "zinc",
    include_bsi_models: bool = False,
) -> list[Path]:
    missing: list[Path] = []
    for rel_path in _required_base_paths(provider_name, include_bsi_models=include_bsi_models):
        candidate = data_dir / rel_path
        if not candidate.exists():
            missing.append(candidate)
    return missing


def _has_any_optional_cache_group(root: Path) -> bool:
    for rel_group in HF_OPTIONAL_CACHE_PATH_GROUPS:
        if all((root / rel_path).exists() for rel_path in rel_group):
            return True
    return False


def _hf_allow_patterns(
    required_rel_paths: list[str],
    download_optional_predicted_cache: bool,
) -> list[str]:
    patterns: list[str] = []
    for rel_path in required_rel_paths:
        patterns.append(rel_path)
        patterns.append(f"{rel_path}/**")

    if download_optional_predicted_cache:
        for rel_group in HF_OPTIONAL_CACHE_PATH_GROUPS:
            patterns.extend(rel_group)

    return patterns


def _copy_path_if_missing(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            _copy_path_if_missing(child, dst / child.name)
    else:
        if dst.exists():
            if dst.stat().st_size == src.stat().st_size:
                return
            dst.unlink()
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def ensure_base_data_from_hf(
    data_dir: Path,
    repo_id: str = "gschottlender/LigQ_2",
    provider_name: str = "zinc",
    download_optional_predicted_cache: bool = True,
    include_bsi_models: bool = False,
) -> None:
    data_dir = Path(data_dir)
    required_rel_paths = _required_base_paths(provider_name, include_bsi_models=include_bsi_models)
    missing_paths = _missing_required_base_paths(
        data_dir,
        provider_name=provider_name,
        include_bsi_models=include_bsi_models,
    )
    if not missing_paths:
        print("[INFO] Found default-ready base data in data_dir. Skipping HF download.")
        return

    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is not installed, but base data is missing.\n"
            "Install it with: pip install huggingface_hub\n"
            "or manually place the default-ready dataset structure under data_dir."
        )

    print(f"[INFO] Downloading base data from Hugging Face dataset '{repo_id}'...")
    local_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=_hf_allow_patterns(
                required_rel_paths=required_rel_paths,
                download_optional_predicted_cache=(
                    provider_name == "zinc" and download_optional_predicted_cache
                ),
            ),
        )
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in required_rel_paths:
        src = local_dir / rel_path
        dst = data_dir / rel_path
        if not src.exists():
            raise FileNotFoundError(
                f"Expected required path inside HF dataset at: {src}"
            )
        _copy_path_if_missing(src, dst)

    copied_optional_cache = False
    if provider_name == "zinc" and download_optional_predicted_cache:
        for rel_group in HF_OPTIONAL_CACHE_PATH_GROUPS:
            if all((local_dir / rel_path).exists() for rel_path in rel_group):
                for rel_path in rel_group:
                    _copy_path_if_missing(local_dir / rel_path, data_dir / rel_path)
                copied_optional_cache = True
                break

    if provider_name == "zinc" and download_optional_predicted_cache and not copied_optional_cache:
        print("[INFO] No optional default predicted-cache namespace found in HF dataset. Continuing without it.")

    still_missing = _missing_required_base_paths(
        data_dir,
        provider_name=provider_name,
        include_bsi_models=include_bsi_models,
    )
    if still_missing:
        missing_str = "\n".join(f"  - {path}" for path in still_missing)
        raise FileNotFoundError(
            "Default-ready base data is still incomplete after HF download. "
            f"Missing paths:\n{missing_str}"
        )


def ensure_known_binding_table(data_dir: Path, force_rebuild: bool = False) -> pd.DataFrame:
    results_db_dir = ensure_dir(data_dir / "results_databases")
    known_path = results_db_dir / "known_binding_data.parquet"

    if known_path.is_file() and not force_rebuild:
        return pd.read_parquet(known_path)

    binding_path = data_dir / "merged_databases" / "binding_data_merged.parquet"
    smiles_path = data_dir / "merged_databases" / "ligs_smiles_merged.parquet"

    if not binding_path.is_file():
        raise FileNotFoundError(f"Merged binding data not found: {binding_path}")
    if not smiles_path.is_file():
        raise FileNotFoundError(f"Merged smiles data not found: {smiles_path}")

    binding_df = pd.read_parquet(binding_path)
    smiles_df = pd.read_parquet(smiles_path)
    known_binding = build_known_binding_data(binding_df)
    known_binding = add_smiles_to_known_binding(known_binding, smiles_df)
    save_known_binding_table(known_binding, results_dir=results_db_dir)
    return known_binding


def ensure_protein_domains_table(data_dir: Path, force_rebuild: bool = False) -> None:
    results_db_dir = ensure_dir(data_dir / "results_databases")
    dom_path = results_db_dir / "protein_domains.parquet"
    if dom_path.is_file() and not force_rebuild:
        return
    build_protein_domains_table(data_dir=data_dir, results_dir=results_db_dir)


def resolve_search_threshold(args: argparse.Namespace) -> None:
    if args.known_only or args.bsi or args.search_threshold is not None:
        return

    default_threshold = DEFAULT_SEARCH_THRESHOLDS_BY_REPRESENTATION.get(
        args.search_representation
    )
    if default_threshold is None:
        known_reps = ", ".join(sorted(DEFAULT_SEARCH_THRESHOLDS_BY_REPRESENTATION))
        raise ValueError(
            "No default --search-threshold is defined for search representation "
            f"'{args.search_representation}'. Pass --search-threshold explicitly. "
            f"Known default representations: {known_reps}."
        )

    args.search_threshold = default_threshold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LigQ_2 ligand search pipeline for a protein FASTA.")
    parser.add_argument("-i", "--input-fasta", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("-d", "--data-dir", default="databases")
    parser.add_argument("-t", "--temp-results-dir", default="temp_results")
    parser.add_argument("-j", "--n-workers", type=int, default=4)

    parser.add_argument("--min-identity", type=float, default=0.9)
    parser.add_argument("--min-query-coverage", type=float, default=0.9)
    parser.add_argument("--min-subject-coverage", type=float, default=0.7)
    parser.add_argument("--blast-evalue-max", type=float, default=1e-5)
    parser.add_argument("--max-hits", type=int, default=150)
    parser.add_argument("--hmmer-evalue-max", type=float, default=1e-5)
    parser.add_argument("--nearest-k", dest="nearest_k_count", type=int, default=5)
    parser.add_argument("--max-domain-candidates-per-domain", type=int, default=20)

    parser.add_argument("--sequence", dest="use_sequence_flag", action="store_true")
    parser.add_argument("--nearest_k", dest="use_nearest_k_flag", action="store_true")
    parser.add_argument("--domains", dest="use_domains_flag", action="store_true")

    parser.add_argument("--keep-repeated-ligands", action="store_true")
    parser.add_argument("--hf-repo-id", default="gschottlender/LigQ_2")
    parser.add_argument(
        "--skip-hf-predicted-cache",
        action="store_true",
        help=(
            "When base data must be downloaded from Hugging Face, download only "
            "the required base files and exclude the optional predicted_bindings cache."
        ),
    )
    parser.add_argument(
        "--known-only",
        action="store_true",
        help=(
            "Return only curated PDB/ChEMBL ligands for recovered proteins. "
            "Skips provider setup, predicted-ligand cache generation, and "
            "searches against ZINC or custom compound databases."
        ),
    )
    parser.add_argument(
        "--bsi",
        action="store_true",
        help="Use BSI instead of Tanimoto/Cosine for predicted-ligand search.",
    )
    parser.add_argument(
        "--bsi-threshold",
        type=float,
        default=0.5,
        help="Minimum BSI score to report when --bsi is enabled (default: 0.5).",
    )

    parser.add_argument("--ligand-provider", default="zinc", help="Predicted-ligand provider (default: zinc).")
    parser.add_argument("--search-representation", default="morgan_1024_r2")
    parser.add_argument("--search-metric", choices=["tanimoto", "cosine"], default="tanimoto")
    parser.add_argument(
        "--search-device",
        default="auto",
        help="Search backend device: auto, cpu, cuda, or cuda:<index> (default: auto).",
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
        help="Optional target database chunk size override for compound searches.",
    )
    parser.add_argument(
        "--bsi-model-batch-size",
        type=int,
        default=65536,
        help="Batch size used inside the BSI neural model for one seed against one target chunk.",
    )
    parser.add_argument(
        "--bsi-max-known-ligands",
        dest="bsi_max_known_ligands",
        type=int,
        default=BSI_DEFAULT_MAX_KNOWN_LIGANDS,
        help="Maximum representative known ligands used for BSI search per protein (default: 10).",
    )
    parser.add_argument(
        "--search-threshold",
        dest="search_threshold",
        type=float,
        default=None,
        help=(
            "Minimum score for predicted hits. If omitted, LigQ_2 uses the "
            "representation-specific percentile-99.5 default when available."
        ),
    )
    parser.add_argument("--zinc-search-threshold", dest="search_threshold", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--search-threshold-max",
        dest="search_threshold_max",
        type=float,
        default=None,
        help="Optional maximum similarity threshold (inclusive) for predicted hits.",
    )
    parser.add_argument(
        "--zinc-search-threshold-max",
        dest="search_threshold_max",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--cluster-threshold", type=float, default=0.8)
    parser.add_argument(
        "--search-per-iteration-topk",
        dest="search_per_iteration_topk",
        type=int,
        default=1000,
        help="Maximum predicted hits retained per iteration/chunk during search.",
    )
    parser.add_argument(
        "--zinc-per-iteration-topk",
        dest="search_per_iteration_topk",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--search-global-topk",
        dest="search_global_topk",
        type=int,
        default=10000,
        help="Maximum number of global predicted hits retained per protein.",
    )
    parser.add_argument(
        "--zinc-global-topk",
        dest="search_global_topk",
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument("--force-rebuild-known-binding", action="store_true")
    parser.add_argument("--force-rebuild-protein-domains", action="store_true")
    parser.add_argument("--force-rebuild-predicted-cache", action="store_true")
    parser.add_argument("--block3-query-chunk-size", type=int, default=100)
    parser.add_argument(
        "--block3-predicted-filter-batch-size",
        dest="block3_predicted_filter_batch_size",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--block3-zinc-filter-batch-size",
        dest="block3_predicted_filter_batch_size",
        type=int,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolve_search_threshold(args)

    if args.bsi_threshold < 0.0 or args.bsi_threshold > 1.0:
        raise ValueError("--bsi-threshold must be between 0 and 1.")
    if args.search_query_batch_size is not None and args.search_query_batch_size <= 0:
        raise ValueError("--search-query-batch-size must be > 0.")
    if args.search_target_chunk_size is not None and args.search_target_chunk_size <= 0:
        raise ValueError("--search-target-chunk-size must be > 0.")
    if args.bsi_model_batch_size <= 0:
        raise ValueError("--bsi-model-batch-size must be > 0.")
    if args.bsi_max_known_ligands <= 0:
        raise ValueError("--bsi-max-known-ligands must be > 0.")
    if args.max_domain_candidates_per_domain <= 0:
        raise ValueError("--max-domain-candidates-per-domain must be > 0.")
    if (
        not args.known_only
        and not args.bsi
        and args.search_threshold_max is not None
        and args.search_threshold_max < args.search_threshold
    ):
        raise ValueError(
            "--search-threshold-max must be >= --search-threshold."
        )
    if not args.known_only and args.search_per_iteration_topk <= 0:
        raise ValueError("--search-per-iteration-topk must be > 0.")
    if not args.known_only and args.search_global_topk <= 0:
        raise ValueError("--search-global-topk must be > 0.")

    input_fasta = Path(args.input_fasta)
    data_dir = Path(args.data_dir)
    temp_results_dir = Path(args.temp_results_dir)
    output_dir = Path(args.output_dir)

    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    # Method selection defaults:
    # - if no explicit method flags are passed, use sequence + nearest_k
    # - domains remains optional by default
    methods_explicit = args.use_sequence_flag or args.use_nearest_k_flag or args.use_domains_flag
    use_sequence = args.use_sequence_flag or not methods_explicit
    use_nearest_k = args.use_nearest_k_flag or not methods_explicit
    use_domains = args.use_domains_flag

    ensure_base_data_from_hf(
        data_dir=data_dir,
        repo_id=args.hf_repo_id,
        provider_name="" if args.known_only else args.ligand_provider,
        download_optional_predicted_cache=(
            not args.known_only and not args.bsi and not args.skip_hf_predicted_cache
        ),
        include_bsi_models=(not args.known_only and args.bsi),
    )
    if use_domains or use_nearest_k or (not args.known_only and args.bsi):
        ensure_protein_domains_table(data_dir=data_dir, force_rebuild=args.force_rebuild_protein_domains)

    print("[INFO] Preparing complementary databases...")
    prepare_complementary_databases(
        data_dir=data_dir,
        prepare_pfam=(use_domains or use_nearest_k),
        prepare_blast=(use_sequence or use_nearest_k or use_domains),
    )

    if temp_results_dir.exists():
        shutil.rmtree(temp_results_dir)
    temp_results_dir = ensure_dir(temp_results_dir)
    output_dir = ensure_dir(output_dir)

    df_queries = parse_query_fasta(input_fasta)

    df_candidates_seq = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    df_candidates_nearest_k = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    df_candidates_nearest_k_ranked = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    df_blast_ranked = pd.DataFrame()
    df_candidates_dom = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    df_hmmer = pd.DataFrame()

    if use_sequence or use_nearest_k or use_domains:
        blast_max_target_seqs = args.max_hits
        if use_nearest_k or use_domains:
            blast_max_target_seqs = max(
                args.max_hits,
                args.nearest_k_count * 200 if use_nearest_k else 0,
                args.max_domain_candidates_per_domain * 200 if use_domains else 0,
                5000,
            )
        df_candidates_seq = run_blast_sequence_search(
            query_fasta=input_fasta,
            data_dir=data_dir,
            temp_results_dir=temp_results_dir,
            min_identity=args.min_identity,
            min_query_coverage=args.min_query_coverage,
            min_subject_coverage=args.min_subject_coverage,
            evalue_max=args.blast_evalue_max,
            max_hits=args.max_hits,
            blast_max_target_seqs=blast_max_target_seqs,
            n_workers=args.n_workers,
        )
        if not use_sequence:
            df_candidates_seq = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])

    if use_domains:
        df_blast_ranked = _load_ranked_blast_hits(
            temp_results_dir=temp_results_dir,
            exclude_sequence_hits=False,
        )

    if use_nearest_k:
        df_candidates_nearest_k_ranked = build_nearest_k_candidates_from_blast(
            temp_results_dir=temp_results_dir,
            df_candidates_seq=df_candidates_seq,
            save_candidates=True,
            candidates_filename="candidate_proteins_nearest_k_ranked.tsv",
        )
    if use_domains or use_nearest_k:
        df_hmmer = run_hmmer_domain_search(
            query_fasta=input_fasta,
            data_dir=data_dir,
            temp_results_dir=temp_results_dir,
            evalue_max=args.hmmer_evalue_max,
            n_workers=args.n_workers,
        )

    if use_nearest_k:
        df_candidates_nearest_k = filter_nearest_k_candidates_by_query_domains(
            df_candidates_nearest_k=df_candidates_nearest_k_ranked,
            df_hmmer=df_hmmer,
            data_dir=data_dir,
            nearest_k=args.nearest_k_count,
            temp_results_dir=temp_results_dir,
            save_candidates=True,
        )

    if use_domains:
        df_candidates_dom = map_pfam_hits_to_candidate_proteins(
            df_hmmer=df_hmmer,
            data_dir=data_dir,
            temp_results_dir=temp_results_dir,
            max_candidates_per_domain=args.max_domain_candidates_per_domain,
            df_blast_ranked=df_blast_ranked,
        )

    df_candidates_all = combine_sequence_and_domain_candidates(
        df_candidates_seq=df_candidates_seq,
        df_candidates_nearest_k=df_candidates_nearest_k,
        df_candidates_domain=df_candidates_dom,
    )
    df_candidates_all = add_shared_domain_counts_to_candidates(
        df_candidates=df_candidates_all,
        df_hmmer=df_hmmer,
        data_dir=data_dir,
    )

    known_db = ensure_known_binding_table(
        data_dir=data_dir,
        force_rebuild=args.force_rebuild_known_binding,
    )

    proteins_needed = set(df_candidates_all["sseqid"].astype(str).unique()) if not df_candidates_all.empty else set()

    predicted_score_col = None
    if args.known_only:
        print("[INFO] Known-only mode enabled. Skipping predicted-ligand provider and cache.")
        predicted_db = pd.DataFrame(
            columns=[
                "uniprot_id",
                "chem_comp_id",
                "possible_binding_sites",
                "query_id",
                "score",
                "smiles",
            ]
        )
    else:
        provider = build_provider(
            provider_name=args.ligand_provider,
            data_dir=data_dir,
            search_representation=args.search_representation,
            search_metric=args.search_metric,
            search_threshold=args.search_threshold,
            search_threshold_max=args.search_threshold_max,
            cluster_threshold=args.cluster_threshold,
            search_per_iteration_topk=args.search_per_iteration_topk,
            search_global_topk=args.search_global_topk,
            use_bsi=args.bsi,
            bsi_threshold=args.bsi_threshold,
            search_device=args.search_device,
            search_q_batch_size=args.search_query_batch_size,
            search_target_chunk_size=args.search_target_chunk_size,
            bsi_model_batch_size=args.bsi_model_batch_size,
            bsi_max_known_ligands=args.bsi_max_known_ligands,
        )

        predicted_db = ensure_provider_cache(
            data_dir=data_dir,
            provider=provider,
            known_binding=known_db,
            proteins_needed=proteins_needed,
            force_rebuild_cache=args.force_rebuild_predicted_cache,
            load_dataframe=False,
        )
        predicted_score_col = getattr(provider, "score_column", None)

    predicted_threshold_min = args.bsi_threshold if args.bsi else args.search_threshold
    predicted_threshold_max = None if args.bsi else args.search_threshold_max
    summary_df = build_query_ligand_results_parallel(
        df_queries=df_queries,
        df_candidates_all=df_candidates_all,
        known_db=known_db,
        predicted_db=predicted_db,
        output_dir=output_dir,
        search_results_subdir="search_results",
        save_per_query=True,
        save_summary=True,
        njobs=args.n_workers,
        chunk_size_queries=args.block3_query_chunk_size,
        drop_duplicates=not args.keep_repeated_ligands,
        predicted_filter_batch_size=args.block3_predicted_filter_batch_size,
        predicted_score_col=predicted_score_col,
        predicted_threshold_min=predicted_threshold_min,
        predicted_threshold_max=predicted_threshold_max,
    )

    print("[INFO] Pipeline completed successfully.")
    print(f"[INFO] Global summary shape: {summary_df.shape}")
    print(f"[INFO] Results written under: {output_dir}")


if __name__ == "__main__":
    main()
