#!/usr/bin/env python

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None

from query_processing.query_processing_functions import (
    prepare_complementary_databases,
    run_blast_sequence_search,
    run_hmmer_domain_search,
    map_pfam_hits_to_candidate_proteins,
    combine_sequence_and_domain_candidates,
    parse_query_fasta,
    build_query_ligand_results_parallel,
)
from generate_results_tables import (
    build_known_binding_data,
    add_smiles_to_known_binding,
    save_known_binding_table,
    build_predicted_zinc_binding_data,
)
from compound_processing.compound_helpers import LigandStore


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_base_data_from_hf(data_dir: Path, repo_id: str = "gschottlender/LigQ_2") -> None:
    data_dir = Path(data_dir)
    sequences_dir = data_dir / "sequences"
    results_db_dir = data_dir / "results_databases"

    if sequences_dir.exists() and results_db_dir.exists():
        print("[INFO] Found 'sequences' and 'results_databases' in data_dir. Skipping HF download.")
        return

    if snapshot_download is None:
        raise ImportError(
            "huggingface_hub is not installed, but base data is missing.\n"
            "Install it with: pip install huggingface_hub\n"
            "or manually place 'sequences' and 'results_databases' under data_dir."
        )

    print(f"[INFO] Downloading base data from Hugging Face dataset '{repo_id}'...")
    local_dir = Path(snapshot_download(repo_id=repo_id, repo_type="dataset"))
    src_sequences = local_dir / "sequences"
    src_results = local_dir / "results_databases"

    if not src_sequences.is_dir():
        raise FileNotFoundError(f"Expected 'sequences' directory inside HF dataset at: {src_sequences}")
    if not src_results.is_dir():
        raise FileNotFoundError(f"Expected 'results_databases' directory inside HF dataset at: {src_results}")

    data_dir.mkdir(parents=True, exist_ok=True)
    if not sequences_dir.exists():
        shutil.copytree(src_sequences, sequences_dir)
    if not results_db_dir.exists():
        shutil.copytree(src_results, results_db_dir)


def _compute_file_fingerprint(path: Path) -> dict:
    """
    Return a lightweight fingerprint dictionary for a file.
    """
    st = path.stat()
    return {
        "path": str(path),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _cache_namespace(value: str) -> str:
    """
    Build a filesystem-safe namespace token.
    """
    return value.replace("/", "_").replace(":", "_").replace(" ", "_")


def _compute_zinc_db_fingerprint(data_dir: Path, search_representation: str) -> str:
    """
    Build a deterministic fingerprint for the local ZINC searchable content.

    This fingerprint is used to invalidate method-specific cache tables when
    the underlying ZINC local database changes.
    """
    zinc_root = data_dir / "compound_data" / "zinc"
    reps_root = zinc_root / "reps"
    meta_path = reps_root / f"{search_representation}.meta.json"

    if not meta_path.is_file():
        raise FileNotFoundError(
            f"Representation metadata not found for '{search_representation}' at: {meta_path}"
        )

    with open(meta_path, "r") as f:
        meta = json.load(f)

    rep_data_path = reps_root / meta["file"]
    ligands_path = zinc_root / "ligands.parquet"
    zinc_smiles_path = data_dir / "zinc" / "ligands_smiles.parquet"

    files = [meta_path, rep_data_path, ligands_path]
    if zinc_smiles_path.is_file():
        files.append(zinc_smiles_path)

    parts = {
        "search_representation": search_representation,
        "files": [_compute_file_fingerprint(p) for p in files],
    }
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _ensure_known_binding_table(data_dir: Path, force_rebuild: bool = False) -> pd.DataFrame:
    """
    Ensure known_binding_data.parquet exists under results_databases.

    If missing (or force_rebuild=True), rebuild from merged_databases inputs.
    """
    results_db_dir = ensure_dir(data_dir / "results_databases")
    known_path = results_db_dir / "known_binding_data.parquet"

    if known_path.is_file() and not force_rebuild:
        print(f"[INFO] Loading known binding table from cache: {known_path}")
        return pd.read_parquet(known_path)

    binding_path = data_dir / "merged_databases" / "binding_data_merged.parquet"
    smiles_path = data_dir / "merged_databases" / "ligs_smiles_merged.parquet"

    if not binding_path.is_file():
        raise FileNotFoundError(f"Merged binding data not found: {binding_path}")
    if not smiles_path.is_file():
        raise FileNotFoundError(f"Merged smiles data not found: {smiles_path}")

    print("[INFO] Building known binding table from merged databases...")
    binding_df = pd.read_parquet(binding_path)
    smiles_df = pd.read_parquet(smiles_path)

    known_binding = build_known_binding_data(binding_df)
    known_binding = add_smiles_to_known_binding(known_binding, smiles_df)
    save_known_binding_table(known_binding, results_dir=results_db_dir)

    print(f"[INFO] Known binding table saved to: {known_path}")
    return known_binding


def _prepare_zinc_cache_table(
    data_dir: Path,
    known_binding: pd.DataFrame,
    proteins_needed: set[str],
    search_representation: str,
    search_metric: str,
    zinc_search_threshold: float,
    cluster_threshold: float,
    force_rebuild_cache: bool = False,
) -> pd.DataFrame:
    """
    Ensure method-specific ZINC cache is present and updated for needed proteins.

    Returns a DataFrame compatible with Block 3 (must contain 'uniprot_id').
    """
    cache_root = ensure_dir(data_dir / "results_databases" / "predicted_bindings")
    method_key = f"rep={_cache_namespace(search_representation)}__metric={_cache_namespace(search_metric)}"
    cache_dir = ensure_dir(cache_root / "zinc" / method_key)

    predicted_path = cache_dir / "predicted_zinc_binding_data.parquet"
    manifest_path = cache_dir / "manifest.json"

    zinc_fp = _compute_zinc_db_fingerprint(data_dir, search_representation)
    expected_manifest = {
        "provider": "zinc",
        "search_representation": search_representation,
        "search_metric": search_metric,
        "zinc_search_threshold": float(zinc_search_threshold),
        "cluster_threshold": float(cluster_threshold),
        "zinc_db_fingerprint": zinc_fp,
    }

    regenerate_cache = force_rebuild_cache
    if manifest_path.is_file() and not regenerate_cache:
        with open(manifest_path, "r") as f:
            current_manifest = json.load(f)
        if current_manifest != expected_manifest:
            print("[INFO] Cache manifest mismatch detected. Rebuilding method-specific cache.")
            regenerate_cache = True

    if regenerate_cache:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        predicted_path = cache_dir / "predicted_zinc_binding_data.parquet"

    if proteins_needed:
        if predicted_path.is_file():
            cached_df = pd.read_parquet(predicted_path)
            cached_proteins = set(cached_df["uniprot_id"].astype(str).unique()) if "uniprot_id" in cached_df.columns else set()
        else:
            cached_proteins = set()

        proteins_to_compute = sorted(proteins_needed - cached_proteins)
    else:
        proteins_to_compute = []

    if proteins_to_compute:
        print(
            "[INFO] On-demand ZINC search for missing proteins: "
            f"{len(proteins_to_compute)} protein(s)."
        )

        pdb_chembl_root = data_dir / "compound_data" / "pdb_chembl"
        zinc_root = data_dir / "compound_data" / "zinc"

        store_pdb_chembl = LigandStore(pdb_chembl_root)
        store_zinc = LigandStore(zinc_root)

        rep_pdb_chembl = store_pdb_chembl.load_representation("morgan_1024_r2")
        rep_zinc = store_zinc.load_representation("morgan_1024_r2")

        if search_representation == "morgan_1024_r2":
            search_rep_ref = rep_pdb_chembl
            search_rep_zinc = rep_zinc
        else:
            search_rep_ref = store_pdb_chembl.load_representation(search_representation)
            search_rep_zinc = store_zinc.load_representation(search_representation)

        known_subset = known_binding[known_binding["uniprot_id"].astype(str).isin(proteins_to_compute)].copy()

        build_predicted_zinc_binding_data(
            known_binding=known_subset,
            store_pdb_chembl=store_pdb_chembl,
            rep_pdb_chembl=rep_pdb_chembl,
            store_zinc=store_zinc,
            rep_zinc=rep_zinc,
            search_rep_ref=search_rep_ref,
            search_rep_zinc=search_rep_zinc,
            search_metric=search_metric,
            zinc_search_threshold=zinc_search_threshold,
            cluster_threshold=cluster_threshold,
            results_dir=cache_dir,
            resume=not regenerate_cache,
            max_proteins=None,
        )
    else:
        print("[INFO] All required proteins are already available in the local ZINC cache.")

    with open(manifest_path, "w") as f:
        json.dump(expected_manifest, f, indent=2)

    if predicted_path.is_file():
        return pd.read_parquet(predicted_path)

    return pd.DataFrame(columns=["uniprot_id", "chem_comp_id", "query_id", "tanimoto", "smiles"])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the LigQ_2 pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Run LigQ_2 ligand search pipeline for a protein FASTA."
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--input-fasta",
        required=True,
        help="Path to multi-FASTA file with query protein sequences.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Output directory where final results will be written.",
    )

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

    parser.add_argument(
        "--search-representation",
        default="morgan_1024_r2",
        help="Representation used for on-demand ZINC similarity search.",
    )
    parser.add_argument(
        "--search-metric",
        choices=["tanimoto", "cosine"],
        default="tanimoto",
        help="Similarity metric used for on-demand ZINC search.",
    )
    parser.add_argument(
        "--zinc-search-threshold",
        type=float,
        default=0.5,
        help="Similarity threshold for ZINC hits.",
    )
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.8,
        help="Clustering threshold used for representative query ligand selection.",
    )
    parser.add_argument(
        "--force-rebuild-known-binding",
        action="store_true",
        help="Force rebuild of known_binding_data.parquet from merged_databases.",
    )
    parser.add_argument(
        "--force-rebuild-zinc-cache",
        action="store_true",
        help=(
            "Force rebuild of the method-specific on-demand ZINC cache for the "
            "selected representation/metric."
        ),
    )


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

    parser.add_argument("--keep-repeated-ligands", action="store_true")
    parser.add_argument("--hf-repo-id", default="gschottlender/LigQ_2")

    parser.add_argument("--ligand-provider", default="zinc", help="Predicted-ligand provider (default: zinc).")
    parser.add_argument("--search-representation", default="morgan_1024_r2")
    parser.add_argument("--search-metric", choices=["tanimoto", "cosine"], default="tanimoto")
    parser.add_argument("--zinc-search-threshold", type=float, default=0.5)
    parser.add_argument("--cluster-threshold", type=float, default=0.8)

    parser.add_argument("--force-rebuild-known-binding", action="store_true")
    parser.add_argument("--force-rebuild-protein-domains", action="store_true")
    parser.add_argument("--force-rebuild-predicted-cache", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_fasta = Path(args.input_fasta)
    data_dir = Path(args.data_dir)
    temp_results_dir = Path(args.temp_results_dir)
    output_dir = Path(args.output_dir)

    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    ensure_base_data_from_hf(data_dir=data_dir, repo_id=args.hf_repo_id)
    ensure_protein_domains_table(data_dir=data_dir, force_rebuild=args.force_rebuild_protein_domains)

    print("[INFO] Preparing complementary databases (Pfam, BLAST, etc.)...")
    prepare_complementary_databases(data_dir=data_dir)

    if temp_results_dir.exists():
        shutil.rmtree(temp_results_dir)
    temp_results_dir = ensure_dir(temp_results_dir)
    output_dir = ensure_dir(output_dir)

    df_queries = parse_query_fasta(input_fasta)

    df_candidates_seq = run_blast_sequence_search(
        query_fasta=input_fasta,
        data_dir=data_dir,
        temp_results_dir=temp_results_dir,
        min_identity=args.min_identity,
        min_query_coverage=args.min_query_coverage,
        min_subject_coverage=args.min_subject_coverage,
        evalue_max=args.blast_evalue_max,
        max_hits=args.max_hits,
        n_workers=args.n_workers,
    )

    df_hmmer = run_hmmer_domain_search(
        query_fasta=input_fasta,
        data_dir=data_dir,
        temp_results_dir=temp_results_dir,
        evalue_max=args.hmmer_evalue_max,
        n_workers=args.n_workers,
    )

    df_candidates_dom = map_pfam_hits_to_candidate_proteins(df_hmmer=df_hmmer, data_dir=data_dir)
    df_candidates_all = combine_sequence_and_domain_candidates(
        df_candidates_seq=df_candidates_seq,
        df_candidates_domain=df_candidates_dom,
    )

    # 7) Load/prepare ligand databases (known + on-demand ZINC cache)
    print("[INFO] Preparing known-binding and predicted-ligand databases...")
    known_db = _ensure_known_binding_table(
        data_dir=data_dir,
        force_rebuild=args.force_rebuild_known_binding,
    )

    proteins_needed = set(df_candidates_all["sseqid"].astype(str).unique()) if not df_candidates_all.empty else set()
    zinc_db = _prepare_zinc_cache_table(
        data_dir=data_dir,
        known_binding=known_db,
        proteins_needed=proteins_needed,
        search_representation=args.search_representation,
        search_metric=args.search_metric,
        zinc_search_threshold=args.zinc_search_threshold,
        cluster_threshold=args.cluster_threshold,
        force_rebuild_cache=args.force_rebuild_zinc_cache,
    )

    summary_df = build_query_ligand_results_parallel(
        df_queries=df_queries,
        df_candidates_all=df_candidates_all,
        known_db=known_db,
        zinc_db=predicted_db,
        output_dir=output_dir,
        search_results_subdir="search_results",
        save_per_query=True,
        save_summary=True,
        njobs=args.n_workers,
        chunk_size_queries=100,
        drop_duplicates=not args.keep_repeated_ligands,
    )

    print("[INFO] Pipeline completed successfully.")
    print(f"[INFO] Global summary shape: {summary_df.shape}")
    print(f"[INFO] Results written under: {output_dir}")


if __name__ == "__main__":
    main()
