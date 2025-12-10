#!/usr/bin/env python

"""
run_ligq_2.py

Entry point for the LigQ_2 pipeline.

Given a query FASTA with protein sequences, this script:

  1) Ensures that base data (sequences + results_databases) is present
     under --data-dir (downloading it from Hugging Face if needed).

  2) Prepares complementary databases in data_dir/complementary_databases/
     (Pfam HMMs, BLAST DB, etc.) via prepare_complementary_databases(...) .

  3) Runs the sequence-based search (Block 1) using BLAST.

  4) Runs the domain-based search (Block 2) using HMMER (hmmscan + Pfam).

  5) Maps Pfam domain hits to candidate proteins and combines sequence
     and domain candidates.

  6) Joins candidates with known ligands and ZINC hits, and builds
     per-query result folders plus a global summary table (Block 3),
     using a parallel implementation.

The script is designed to be called from the command line, e.g.:

    python run_ligq_2.py \
        -i queries.fasta \
        -o results \
        --data-dir databases \
        -j 4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    # Used to download base data (sequences + results_databases) from HF
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


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist and return it as a Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_base_data_from_hf(data_dir: Path, repo_id: str = "gschottlender/LigQ_2") -> None:
    """
    Ensure that 'sequences' and 'results_databases' exist under data_dir.

    If either directory is missing, download the dataset from Hugging Face
    (repo_id) using snapshot_download and copy the relevant folders.

    Parameters
    ----------
    data_dir : Path
        Base data directory where 'sequences' and 'results_databases'
        should reside.
    repo_id : str, default 'gschottlender/LigQ_2'
        Hugging Face dataset repository ID.
    """
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

    # Expected layout inside the dataset
    src_sequences = local_dir / "sequences"
    src_results = local_dir / "results_databases"

    if not src_sequences.is_dir():
        raise FileNotFoundError(
            f"Expected 'sequences' directory inside HF dataset at: {src_sequences}"
        )
    if not src_results.is_dir():
        raise FileNotFoundError(
            f"Expected 'results_databases' directory inside HF dataset at: {src_results}"
        )

    data_dir.mkdir(parents=True, exist_ok=True)

    # Copy sequences
    if not sequences_dir.exists():
        print(f"[INFO] Copying 'sequences' to: {sequences_dir}")
        shutil.copytree(src_sequences, sequences_dir)
    else:
        print(f"[INFO] 'sequences' already exists at: {sequences_dir} (not overwritten).")

    # Copy results_databases
    if not results_db_dir.exists():
        print(f"[INFO] Copying 'results_databases' to: {results_db_dir}")
        shutil.copytree(src_results, results_db_dir)
    else:
        print(f"[INFO] 'results_databases' already exists at: {results_db_dir} (not overwritten).")


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

    # General paths
    parser.add_argument(
        "-d",
        "--data-dir",
        default="databases",
        help=(
            "Base data directory. It must contain 'sequences' and "
            "'results_databases' (downloaded from Hugging Face if missing), "
            "and will also hold 'complementary_databases/'."
        ),
    )
    parser.add_argument(
        "-t",
        "--temp-results-dir",
        default="temp_results",
        help="Directory for intermediate results (BLAST/HMMER outputs).",
    )

    # Parallelism
    parser.add_argument(
        "-j",
        "--n-workers",
        type=int,
        default=4,
        help="Number of workers / threads used by BLAST, HMMER and Block 3.",
    )

    # BLAST (Block 1) parameters
    parser.add_argument(
        "--min-identity",
        type=float,
        default=0.9,
        help="Minimum sequence identity (0–1) for BLAST hits.",
    )
    parser.add_argument(
        "--min-query-coverage",
        type=float,
        default=0.9,
        help="Minimum query coverage (0–1) for BLAST hits.",
    )
    parser.add_argument(
        "--min-subject-coverage",
        type=float,
        default=0.7,
        help="Minimum subject coverage (0–1) for BLAST hits.",
    )
    parser.add_argument(
        "--blast-evalue-max",
        type=float,
        default=1e-5,
        help="Maximum e-value for BLAST hits.",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=150,
        help="Maximum number of BLAST hits to keep per query.",
    )

    # HMMER (Block 2) parameters
    parser.add_argument(
        "--hmmer-evalue-max",
        type=float,
        default=1e-5,
        help="Maximum domain i-Evalue for HMMER (hmmscan) hits.",
    )

    # Hugging Face dataset control
    parser.add_argument(
        "--hf-repo-id",
        default="gschottlender/LigQ_2",
        help="Hugging Face dataset ID for base data download.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run the full LigQ_2 pipeline:

      - Ensure base data in data_dir (from Hugging Face if needed).
      - Prepare complementary databases (Pfam, BLAST).
      - Run BLAST and HMMER searches.
      - Combine candidates.
      - Build per-query ligand results and a global summary.
    """
    args = parse_args()

    input_fasta = Path(args.input_fasta)
    data_dir = Path(args.data_dir)
    temp_results_dir = Path(args.temp_results_dir)
    output_dir = Path(args.output_dir)

    if not input_fasta.is_file():
        raise FileNotFoundError(f"Input FASTA not found: {input_fasta}")

    # 0) Ensure base data (sequences + results_databases) is present
    ensure_base_data_from_hf(data_dir=data_dir, repo_id=args.hf_repo_id)

    # 1) Prepare complementary databases (Pfam, BLAST, etc.)
    #    This function should be idempotent and only do work if needed.
    print("[INFO] Preparing complementary databases (Pfam, BLAST, etc.)...")
    prepare_complementary_databases(data_dir=data_dir)

    # 2) Clean temp_results_dir to avoid mixing runs
    if temp_results_dir.exists():
        print(f"[INFO] Removing existing temporary results directory: {temp_results_dir}")
        shutil.rmtree(temp_results_dir)
    temp_results_dir = ensure_dir(temp_results_dir)
    output_dir = ensure_dir(output_dir)

    # 3) Read query IDs from FASTA (Block 0 utility)
    print(f"[INFO] Parsing query FASTA: {input_fasta}")
    df_queries = parse_query_fasta(input_fasta)
    n_queries = df_queries["qseqid"].nunique()
    print(f"[INFO] Total queries parsed: {n_queries}")

    # 4) Block 1: Sequence-based search with BLAST
    print("[INFO] Starting Block 1 (BLAST sequence search)...")
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

    # 5) Block 2: Domain-based search with HMMER / Pfam
    print("[INFO] Starting Block 2 (HMMER Pfam domain search)...")
    df_hmmer = run_hmmer_domain_search(
        query_fasta=input_fasta,
        data_dir=data_dir,
        temp_results_dir=temp_results_dir,
        evalue_max=args.hmmer_evalue_max,
        n_workers=args.n_workers,
    )

    # Map Pfam hits to candidate proteins
    print("[INFO] Mapping Pfam domain hits to candidate proteins...")
    df_candidates_dom = map_pfam_hits_to_candidate_proteins(df_hmmer=df_hmmer)

    # 6) Combine sequence and domain candidates, enforcing sequence > domain priority
    print("[INFO] Combining sequence-based and domain-based candidate proteins...")
    df_candidates_all = combine_sequence_and_domain_candidates(
        df_candidates_seq=df_candidates_seq,
        df_candidates_domain=df_candidates_dom,
    )

    # 7) Load ligand databases (known + ZINC)
    print("[INFO] Loading ligand databases from results_databases/...")
    known_db_path = data_dir / "results_databases" / "known_binding_data.parquet"
    zinc_db_path = data_dir / "results_databases" / "predicted_zinc_binding_data.parquet"

    if not known_db_path.is_file():
        raise FileNotFoundError(f"known_binding_data.parquet not found at: {known_db_path}")
    if not zinc_db_path.is_file():
        raise FileNotFoundError(f"predicted_zinc_binding_data.parquet not found at: {zinc_db_path}")

    known_db = pd.read_parquet(known_db_path)
    zinc_db = pd.read_parquet(zinc_db_path)

    # 8) Block 3: Build per-query ligand results (parallel)
    print("[INFO] Starting Block 3 (per-query ligand results, parallel)...")
    summary_df = build_query_ligand_results_parallel(
        df_queries=df_queries,
        df_candidates_all=df_candidates_all,
        known_db=known_db,
        zinc_db=zinc_db,
        output_dir=output_dir,
        search_results_subdir="search_results",
        save_per_query=True,
        save_summary=True,
        njobs=args.n_workers,
    )

    print("[INFO] Pipeline completed successfully.")
    print(f"[INFO] Global summary shape: {summary_df.shape}")
    print(f"[INFO] Results written under: {output_dir}")


if __name__ == "__main__":
    main()
