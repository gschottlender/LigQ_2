from __future__ import annotations

import gzip
import json
import shutil
import shutil as pyshutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Callable, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from query_processing.predicted_rowgroup_index import load_or_build_row_group_index


# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------

PFAM_A_URL = (
    "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
)


# ----------------------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------------------


def _read_parquet_rows_for_uniprot_ids(
    parquet_path: str | Path,
    uniprot_ids: list[str],
    batch_size: int = 2000,
) -> pd.DataFrame:
    """Read parquet rows filtered by uniprot_id using row-group lookup when available."""
    parquet_path = Path(parquet_path)
    parquet_file = pq.ParquetFile(parquet_path)

    def _empty_with_schema() -> pd.DataFrame:
        schema = parquet_file.schema_arrow
        arrays = [pa.array([], type=field.type) for field in schema]
        return pa.Table.from_arrays(arrays, schema=schema).to_pandas()

    if not uniprot_ids:
        return _empty_with_schema()
    if "uniprot_id" not in parquet_file.schema_arrow.names:
        raise ValueError(f"Parquet file must contain 'uniprot_id': {parquet_path}")

    wanted = pa.array([str(value) for value in set(uniprot_ids)])
    wanted_values = {str(value) for value in set(uniprot_ids)}

    try:
        row_group_index = load_or_build_row_group_index(parquet_path)
    except (OSError, ValueError, json.JSONDecodeError):
        row_group_index = None

    if row_group_index is not None:
        row_groups = sorted(
            {
                row_group
                for uniprot_id in wanted_values
                for row_group in row_group_index.get(uniprot_id, [])
            }
        )
        if not row_groups:
            return _empty_with_schema()

        table = parquet_file.read_row_groups(row_groups)
        uniprot_col = table["uniprot_id"].cast(pa.string())
        mask = pc.is_in(uniprot_col, value_set=wanted)
        if not pc.any(mask).as_py():
            return _empty_with_schema()
        return table.filter(mask).to_pandas()

    parts: list[pd.DataFrame] = []
    for record_batch in parquet_file.iter_batches(batch_size=max(int(batch_size), 1)):
        uniprot_col = record_batch.column(record_batch.schema.get_field_index("uniprot_id")).cast(pa.string())
        mask = pc.is_in(uniprot_col, value_set=wanted)
        if pc.any(mask).as_py():
            filtered = record_batch.filter(mask)
            parts.append(pa.Table.from_batches([filtered]).to_pandas())

    if not parts:
        return _empty_with_schema()
    if len(parts) == 1:
        return parts[0]
    return pd.concat(parts, ignore_index=True)


def run_command(cmd: list[str], cwd: Path | None = None) -> None:
    """
    Run an external command and raise a RuntimeError if it fails.
    """
    print(f"[INFO] Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def check_dependency(name: str) -> None:
    """
    Ensure an executable is available in PATH.
    """
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required program '{name}' not found in PATH. "
            f"Install it in your conda env (e.g. 'conda install -c bioconda {name}') "
            "and try again."
        )


def ensure_dir(path: str | Path) -> Path:
    """
    Create directory if it does not exist and return it as Path.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_list(series: pd.Series):
    """
    Return a list of unique values preserving the first-seen order.

    Intended for scalar columns (uniprot_id, query_id, etc.).
    """
    seen = set()
    out = []
    for v in series.dropna():
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _unique_list_flat(series: pd.Series):
    """
    Return a flattened list of unique values preserving order.

    If a value is a list/tuple/array, it is expanded; otherwise it is
    treated as a single element. Intended for binding_sites, pdb_ids, etc.
    """
    seen = set()
    out = []
    for v in series.dropna():
        if isinstance(v, (list, tuple, np.ndarray)):
            vals = list(v)
        else:
            vals = [v]
        for x in vals:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
    return out


# ----------------------------------------------------------------------
# Block 0 – Complementary databases: Pfam + BLAST
# ----------------------------------------------------------------------

def download_and_prepare_pfam_a(pfam_dir: Path, url: str = PFAM_A_URL) -> Path:
    """
    Download Pfam-A.hmm.gz (if needed), decompress it and run hmmpress.

    Parameters
    ----------
    pfam_dir : Path
        Directory where Pfam HMM files will be stored.
    url : str
        URL for the Pfam-A HMM archive.

    Returns
    -------
    Path
        Path to Pfam-A.hmm (uncompressed).
    """
    pfam_dir.mkdir(parents=True, exist_ok=True)

    gz_path = pfam_dir / "Pfam-A.hmm.gz"
    hmm_path = pfam_dir / "Pfam-A.hmm"

    # 1) Download if missing
    if not gz_path.exists() and not hmm_path.exists():
        print(f"[INFO] Downloading Pfam-A from {url}")
        urllib.request.urlretrieve(url, gz_path)
        print(f"[INFO] Downloaded to {gz_path}")
    else:
        print(
            "[INFO] Pfam-A compressed or uncompressed file already present, "
            "skipping download."
        )

    # 2) Decompress if needed
    if not hmm_path.exists():
        print(f"[INFO] Decompressing {gz_path} -> {hmm_path}")
        with gzip.open(gz_path, "rb") as f_in, open(hmm_path, "wb") as f_out:
            pyshutil.copyfileobj(f_in, f_out)
        print("[INFO] Decompression finished.")
    else:
        print("[INFO] Pfam-A.hmm already decompressed.")

    # 3) Run hmmpress if index files are missing
    h3_files = list(pfam_dir.glob("Pfam-A.hmm.h3*"))
    if h3_files:
        print(
            "[INFO] Pfam-A HMM already indexed (h3 files found), "
            "skipping hmmpress."
        )
    else:
        check_dependency("hmmpress")
        print("[INFO] Indexing Pfam-A with hmmpress...")
        run_command(["hmmpress", hmm_path.name], cwd=pfam_dir)
        print("[INFO] hmmpress finished.")

    return hmm_path


def prepare_blast_db(
    data_dir: Path,
    complementary_dir: Path,
    fasta_rel_path: str = "sequences/target_sequences.fasta",
    db_name: str = "target_sequences",
    force: bool = False,
) -> Path:
    """
    Create a BLAST protein database from target_sequences.fasta.

    Parameters
    ----------
    data_dir : Path
        Base data directory (e.g. Path("databases")).
    complementary_dir : Path
        Path to complementary_databases directory (inside data_dir).
    fasta_rel_path : str
        Relative path (from data_dir) to the FASTA file with target sequences.
    db_name : str
        Name prefix for the BLAST database files.

    Returns
    -------
    Path
        Directory where the BLAST DB files were created.
    """
    check_dependency("makeblastdb")

    blast_dir = complementary_dir / "blast"
    blast_dir.mkdir(parents=True, exist_ok=True)

    fasta_path = data_dir / fasta_rel_path
    if not fasta_path.exists():
        raise FileNotFoundError(
            f"FASTA file not found: {fasta_path}. "
            "You must provide target_sequences.fasta beforehand."
        )

    # BLAST DB target prefix (files: .pin/.psq/.phr, etc.)
    db_prefix = blast_dir / db_name

    db_files = sorted(blast_dir.glob(f"{db_name}.*"))
    required_db_files = [
        blast_dir / f"{db_name}.phr",
        blast_dir / f"{db_name}.pin",
        blast_dir / f"{db_name}.psq",
    ]
    db_complete = all(path.exists() for path in required_db_files)
    db_current = (
        db_complete
        and min(path.stat().st_mtime_ns for path in required_db_files)
        >= fasta_path.stat().st_mtime_ns
    )

    if db_current and not force:
        print("[INFO] BLAST DB already present and current, skipping makeblastdb.")
        return blast_dir

    if db_files:
        reason = "forced rebuild" if force else "stale or incomplete BLAST DB"
        print(f"[INFO] Removing existing BLAST DB files ({reason})...")
        for path in db_files:
            path.unlink()

    print(f"[INFO] Creating BLAST protein DB from {fasta_path}")
    cmd = [
        "makeblastdb",
        "-in",
        str(fasta_path),
        "-dbtype",
        "prot",
        "-out",
        str(db_prefix),
    ]
    run_command(cmd)

    print("[INFO] BLAST DB created in:", blast_dir)
    return blast_dir


def prepare_complementary_databases(
    data_dir: str | Path = "databases",
    prepare_pfam: bool = True,
    prepare_blast: bool = True,
) -> None:
    """
    High-level wrapper for the first block of the master script.

    - Ensure complementary_databases/ structure exists.
    - Download & index Pfam-A (HMMER).
    - Build BLAST DB from target_sequences.fasta.

    Parameters
    ----------
    data_dir : str or Path
        Base directory where your project stores databases.
    """
    data_dir = Path(data_dir)
    complementary_dir = data_dir / "complementary_databases"

    print(f"[INFO] Using data_dir = {data_dir}")
    complementary_dir.mkdir(parents=True, exist_ok=True)

    # 1) Pfam-A + hmmpress
    if prepare_pfam:
        pfam_dir = complementary_dir / "pfam"
        download_and_prepare_pfam_a(pfam_dir)

    # 2) BLAST DB from target_sequences.fasta
    if prepare_blast:
        prepare_blast_db(data_dir=data_dir, complementary_dir=complementary_dir)

    print("[INFO] Complementary databases are ready.")


# ----------------------------------------------------------------------
# Query parsing
# ----------------------------------------------------------------------

def parse_query_fasta(
    fasta_path: str | Path,
) -> pd.DataFrame:
    """
    Parse a FASTA file and return the list of query sequence IDs (qseqid).

    The qseqid is defined as the FASTA header up to the first whitespace,
    i.e. the first token after '>'.

    Parameters
    ----------
    fasta_path : str or Path
        Path to the input FASTA file.

    Returns
    -------
    pd.DataFrame
        DataFrame with one column:
          - qseqid : query sequence IDs
        One row per sequence in the FASTA.
    """
    fasta_path = Path(fasta_path)

    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    qseqids: list[str] = []

    with fasta_path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                # Take everything after '>' up to first whitespace
                qseqid = line[1:].strip().split()[0]
                qseqids.append(qseqid)

    if not qseqids:
        raise ValueError(f"No FASTA headers found in file: {fasta_path}")

    df_queries = pd.DataFrame({"qseqid": qseqids})

    return df_queries


# ----------------------------------------------------------------------
# Block 1 – Sequence-based search (BLAST)
# ----------------------------------------------------------------------

def run_blast_sequence_search(
    query_fasta: str | Path,
    data_dir: str | Path = "databases",
    temp_results_dir: str | Path = "temp_results",
    min_identity: float = 0.9,
    min_query_coverage: float = 0.9,
    min_subject_coverage: float = 0.7,
    evalue_max: float = 1e-5,
    max_hits: int = 150,
    blast_max_target_seqs: int | None = None,
    blast_program: str = "blastp",
    blast_db_name: str = "target_sequences",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Block 1: sequence-based search against local BLAST DB.

    Parameters
    ----------
    query_fasta : str or Path
        Multi-FASTA file with input sequences.
    data_dir : str or Path, default 'databases'
        Base data directory (where complementary_databases/ lives).
    temp_results_dir : str or Path, default 'temp_results'
        Directory where intermediate BLAST outputs are stored.
        This function will write:
            - blast_sequence_search.tsv           (raw BLAST tabular output)
            - blast_sequence_candidates.tsv       (filtered detailed candidates)
            - candidate_proteins_sequence.tsv     (qseqid / sseqid / search_type)
    min_identity : float, default 0.9
        Minimum sequence identity (0–1).
    min_query_coverage : float, default 0.9
        Minimum coverage of the query (0–1).
    min_subject_coverage : float, default 0.7
        Minimum coverage of the subject (0–1).
    evalue_max : float, default 1e-5
        Maximum e-value accepted.
    max_hits : int, default 150
        Maximum number of BLAST hits to keep per query.
    blast_max_target_seqs : int or None, default None
        Maximum number of raw BLAST hits to request per query. If None,
        `max_hits` is reused. This can be set larger than `max_hits` to keep
        a broader raw pool for downstream methods such as nearest-k.
    blast_program : str, default 'blastp'
        BLAST program (blastp, blastx, etc.).
    blast_db_name : str, default 'target_sequences'
        Name of the BLAST DB (without extensions).
    n_workers : int, default 4
        Number of CPU threads to pass to BLAST (-num_threads).

    Returns
    -------
    pd.DataFrame
        Candidate proteins by sequence, with filters already applied.
        Columns include:
            qseqid, sseqid, pident, qcov, scov, evalue, bitscore, qlen, slen
    """
    query_fasta = Path(query_fasta)
    data_dir = Path(data_dir)
    temp_results_dir = ensure_dir(temp_results_dir)

    # Path to BLAST DB
    blast_db = data_dir / "complementary_databases" / "blast" / blast_db_name

    if not query_fasta.is_file():
        raise FileNotFoundError(f"Query FASTA not found: {query_fasta}")

    # Raw BLAST output file
    blast_out = temp_results_dir / "blast_sequence_search.tsv"
    if blast_max_target_seqs is None:
        blast_max_target_seqs = max_hits

    # Extended tabular BLAST output format, including qlen and slen
    outfmt = (
        "6 "
        "qseqid sseqid pident length mismatch gapopen "
        "qstart qend sstart send evalue bitscore qlen slen"
    )

    cmd = [
        blast_program,
        "-query",
        str(query_fasta),
        "-db",
        str(blast_db),
        "-outfmt",
        outfmt,
        "-evalue",
        str(evalue_max),
        "-max_target_seqs",
        str(blast_max_target_seqs),
        "-num_threads",
        str(n_workers),   # BLAST uses n_workers threads
        "-out",
        str(blast_out),   # BLAST writes to this file
    ]

    print(f"[INFO] Running BLAST sequence search (Block 1) with {n_workers} threads...")
    run_command(cmd)

    if not blast_out.is_file():
        raise RuntimeError(
            f"BLAST output file not found: {blast_out}. "
            "Check that BLAST executed correctly and that -out was set."
        )

    print(f"[INFO] Parsing BLAST output: {blast_out}")
    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
    ]
    df = pd.read_csv(blast_out, sep="\t", header=None, names=cols)

    if df.empty:
        print("[INFO] No BLAST hits found after initial search.")
        detailed_path = temp_results_dir / "blast_sequence_candidates.tsv"
        df.to_csv(detailed_path, sep="\t", index=False)
        mapping_path = temp_results_dir / "candidate_proteins_sequence.tsv"
        pd.DataFrame(columns=["qseqid", "sseqid", "search_type"]).to_csv(
            mapping_path, sep="\t", index=False
        )
        return df

    # Coverage calculations
    df["qcov"] = df["length"] / df["qlen"]
    df["scov"] = df["length"] / df["slen"]
    df["pident_frac"] = df["pident"] / 100.0

    # Apply conservative filters
    mask = (
        (df["pident_frac"] >= min_identity)
        & (df["qcov"] >= min_query_coverage)
        & (df["scov"] >= min_subject_coverage)
        & (df["evalue"] <= evalue_max)
    )
    df_filt = df.loc[mask].copy()

    if df_filt.empty:
        print("[INFO] No BLAST hits passed the filtering thresholds.")
        detailed_path = temp_results_dir / "blast_sequence_candidates.tsv"
        df_filt.to_csv(detailed_path, sep="\t", index=False)
        mapping_path = temp_results_dir / "candidate_proteins_sequence.tsv"
        pd.DataFrame(columns=["qseqid", "sseqid", "search_type"]).to_csv(
            mapping_path, sep="\t", index=False
        )
        return df_filt

    # Sort by strongest BLAST evidence first.
    df_filt = df_filt.sort_values(
        by=["qseqid", "bitscore", "evalue", "pident", "qcov", "scov", "sseqid"],
        ascending=[True, False, True, False, False, False, True],
    )

    # Limit number of hits per query if requested
    if max_hits is not None:
        df_filt = (
            df_filt.groupby("qseqid", group_keys=False)
            .head(max_hits)
            .reset_index(drop=True)
        )

    print(
        f"[INFO] Sequence candidates after filtering: {len(df_filt)} rows "
        f"({df_filt['qseqid'].nunique()} queries)"
    )

    # 1) Save filtered detailed table
    detailed_path = temp_results_dir / "blast_sequence_candidates.tsv"
    df_filt.to_csv(detailed_path, sep="\t", index=False)
    print(f"[INFO] Saved detailed BLAST candidates to: {detailed_path}")

    # 2) Compact mapping qseqid / sseqid / search_type='sequence'
    mapping = (
        df_filt[["qseqid", "sseqid", "bitscore", "evalue", "pident", "qcov", "scov"]]
        .drop_duplicates(subset=["qseqid", "sseqid"], keep="first")
        .assign(search_type="sequence")
        .reset_index(drop=True)
    )
    mapping_path = temp_results_dir / "candidate_proteins_sequence.tsv"
    mapping[["qseqid", "sseqid", "search_type"]].to_csv(mapping_path, sep="\t", index=False)
    print(f"[INFO] Saved sequence-based candidate mapping to: {mapping_path}")

    return mapping


def build_nearest_k_candidates_from_blast(
    temp_results_dir: str | Path = "temp_results",
    df_candidates_seq: pd.DataFrame | None = None,
    save_candidates: bool = True,
    candidates_filename: str = "candidate_proteins_nearest_k.tsv",
) -> pd.DataFrame:
    """
    Build ranked nearest-k BLAST candidates per query from raw BLAST output.

    Rules
    -----
    - Uses `blast_sequence_search.tsv` written by `run_blast_sequence_search`.
    - Ranks hits by:
        1) bitscore descending
        2) evalue ascending
    - Excludes (qseqid, sseqid) already present in `df_candidates_seq`.
    - Does not apply hard quality cutoffs; callers can filter/truncate later.

    Returns
    -------
    pd.DataFrame
        Columns:
          - qseqid
          - sseqid
          - search_type ('nearest_k')
    """
    temp_results_dir = Path(temp_results_dir)
    blast_out = temp_results_dir / "blast_sequence_search.tsv"

    if not blast_out.is_file():
        raise FileNotFoundError(
            f"Raw BLAST output not found: {blast_out}. "
            "Run run_blast_sequence_search first."
        )

    ranked = _load_ranked_blast_hits(
        temp_results_dir=temp_results_dir,
        df_candidates_seq=df_candidates_seq,
        exclude_sequence_hits=True,
    )

    if ranked.empty:
        nearest = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    else:
        keep_cols = [
            c
            for c in ["qseqid", "sseqid", "bitscore", "evalue", "pident", "qcov", "scov"]
            if c in ranked.columns
        ]
        nearest = (
            ranked[keep_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        nearest["search_type"] = "nearest_k"

    if save_candidates:
        nearest_path = temp_results_dir / candidates_filename
        nearest[["qseqid", "sseqid", "search_type"]].to_csv(nearest_path, sep="\t", index=False)
        print(f"[INFO] Saved nearest-k candidate mapping to: {nearest_path}")

    return nearest


def _load_ranked_blast_hits(
    temp_results_dir: str | Path = "temp_results",
    df_candidates_seq: pd.DataFrame | None = None,
    exclude_sequence_hits: bool = False,
) -> pd.DataFrame:
    """
    Load raw BLAST hits and rank one best hit per (query, subject).
    """
    temp_results_dir = Path(temp_results_dir)
    blast_out = temp_results_dir / "blast_sequence_search.tsv"

    if not blast_out.is_file():
        raise FileNotFoundError(
            f"Raw BLAST output not found: {blast_out}. "
            "Run run_blast_sequence_search first."
        )

    cols = [
        "qseqid",
        "sseqid",
        "pident",
        "length",
        "mismatch",
        "gapopen",
        "qstart",
        "qend",
        "sstart",
        "send",
        "evalue",
        "bitscore",
        "qlen",
        "slen",
    ]
    df = pd.read_csv(blast_out, sep="\t", header=None, names=cols)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "qseqid",
                "sseqid",
                "pident",
                "length",
                "evalue",
                "bitscore",
                "qlen",
                "slen",
                "qcov",
                "scov",
            ]
        )

    df["qcov"] = df["length"] / df["qlen"]
    df["scov"] = df["length"] / df["slen"]
    df = (
        df.sort_values(
            by=["qseqid", "sseqid", "bitscore", "evalue", "pident", "qcov", "scov"],
            ascending=[True, True, False, True, False, False, False],
        )
        .drop_duplicates(subset=["qseqid", "sseqid"], keep="first")
        .reset_index(drop=True)
    )

    if exclude_sequence_hits:
        if df_candidates_seq is None:
            df_candidates_seq = pd.DataFrame(columns=["qseqid", "sseqid"])

        seq_pairs = df_candidates_seq[["qseqid", "sseqid"]].drop_duplicates()
        if not seq_pairs.empty:
            df = df.merge(
                seq_pairs,
                on=["qseqid", "sseqid"],
                how="left",
                indicator=True,
            )
            df = df[df["_merge"] == "left_only"].drop(columns=["_merge"])

    return (
        df.sort_values(
            by=["qseqid", "bitscore", "evalue", "pident", "qcov", "scov", "sseqid"],
            ascending=[True, False, True, False, False, False, True],
        )
        .reset_index(drop=True)
    )


def filter_nearest_k_candidates_by_query_domains(
    df_candidates_nearest_k: pd.DataFrame,
    df_hmmer: pd.DataFrame,
    data_dir: str | Path = "databases",
    nearest_k: int | None = 5,
    temp_results_dir: str | Path = "temp_results",
    save_candidates: bool = True,
    candidates_filename: str = "candidate_proteins_nearest_k_domain_filtered.tsv",
) -> pd.DataFrame:
    """
    Restrict ranked nearest-k candidates to proteins sharing at least one
    Pfam domain with the query, then keep up to `nearest_k` proteins per query.
    """
    expected_cols = {"qseqid", "sseqid"}
    if df_candidates_nearest_k is None or df_candidates_nearest_k.empty:
        filtered = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    elif nearest_k is not None and nearest_k <= 0:
        filtered = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    elif df_hmmer is None or df_hmmer.empty:
        filtered = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    else:
        missing = expected_cols - set(df_candidates_nearest_k.columns)
        if missing:
            raise ValueError(
                "df_candidates_nearest_k is missing required columns: "
                f"{missing}"
            )

        if "pfam_id" not in df_hmmer.columns:
            if "pfam_acc" not in df_hmmer.columns:
                raise ValueError(
                    "df_hmmer must contain either 'pfam_id' or 'pfam_acc' column."
                )
            df_hmmer = df_hmmer.copy()
            df_hmmer["pfam_id"] = df_hmmer["pfam_acc"].astype(str).str.split(".").str[0]

        data_dir = Path(data_dir)
        protein_domains_path = data_dir / "results_databases" / "protein_domains.parquet"
        if not protein_domains_path.is_file():
            raise FileNotFoundError(
                f"protein_domains.parquet not found at: {protein_domains_path}"
            )

        df_query_domains = df_hmmer[["qseqid", "pfam_id"]].dropna().drop_duplicates()
        if df_query_domains.empty:
            filtered = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
        else:
            df_domains = pd.read_parquet(
                protein_domains_path,
                columns=["uniprot_id", "pfam_id"],
            ).dropna(subset=["uniprot_id", "pfam_id"])
            df_domains["uniprot_id"] = df_domains["uniprot_id"].astype(str)
            df_domains["pfam_id"] = df_domains["pfam_id"].astype(str)
            df_domains = df_domains.drop_duplicates(subset=["uniprot_id", "pfam_id"])

            ranked = df_candidates_nearest_k.copy()
            if "search_type" not in ranked.columns:
                ranked["search_type"] = "nearest_k"

            ranked["qseqid"] = ranked["qseqid"].astype(str)
            ranked["sseqid"] = ranked["sseqid"].astype(str)
            df_query_domains["qseqid"] = df_query_domains["qseqid"].astype(str)

            matched = (
                ranked
                .merge(
                    df_domains.rename(columns={"uniprot_id": "sseqid"}),
                    on="sseqid",
                    how="inner",
                )
                .merge(df_query_domains, on=["qseqid", "pfam_id"], how="inner")
            )

            filtered = (
                matched.drop(columns=["pfam_id"])
                .drop_duplicates(subset=["qseqid", "sseqid"], keep="first")
                .reset_index(drop=True)
            )

            if nearest_k is not None:
                filtered = (
                    filtered.groupby("qseqid", group_keys=False)
                    .head(int(nearest_k))
                    .reset_index(drop=True)
                )

    if save_candidates:
        temp_results_dir = Path(temp_results_dir)
        temp_results_dir.mkdir(parents=True, exist_ok=True)
        filtered_path = temp_results_dir / candidates_filename
        filtered[["qseqid", "sseqid", "search_type"]].to_csv(filtered_path, sep="\t", index=False)
        print(f"[INFO] Saved nearest-k domain-filtered mapping to: {filtered_path}")

    return filtered


# ----------------------------------------------------------------------
# Block 2 – Domain-based search (HMMER + Pfam)
# ----------------------------------------------------------------------

def run_hmmer_domain_search(
    query_fasta: str | Path,
    data_dir: str | Path = "databases",
    temp_results_dir: str | Path = "temp_results",
    pfam_hmm_name: str = "Pfam-A.hmm",
    evalue_max: float = 1e-5,
    use_ga_cutoff: bool = True,
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Block 2: domain-based search using HMMER (hmmscan) against Pfam-A.

    Parameters
    ----------
    query_fasta : str or Path
        Multi-FASTA file with input protein sequences.
    data_dir : str or Path, default 'databases'
        Base data directory (where complementary_databases/ lives).
    temp_results_dir : str or Path, default 'temp_results'
        Directory where intermediate HMMER outputs are stored.
    pfam_hmm_name : str, default 'Pfam-A.hmm'
        Name of the Pfam HMM file (inside complementary_databases/pfam/).
    evalue_max : float, default 1e-5
        Maximum domain i-Evalue accepted (in addition to GA cutoff).
    use_ga_cutoff : bool, default True
        If True, adds '--cut_ga' to hmmscan to use Pfam's GA thresholds.
    n_workers : int, default 4
        Number of CPU threads to pass to HMMER (--cpu).

    Returns
    -------
    pd.DataFrame
        Domain hits passing GA cutoff (if enabled) and e-value filter.
        Columns include, among others:
            qseqid, pfam_id, pfam_acc, full_evalue, dom_i_evalue, dom_score,
            hmm_from, hmm_to, ali_from, ali_to, env_from, env_to, acc
        where:
            - qseqid = query (sequence) id from the FASTA
            - pfam_id = Pfam accession without version (e.g. 'PF00069')
            - pfam_acc = full Pfam accession with version (e.g. 'PF00069.20')
    """
    query_fasta = Path(query_fasta)
    data_dir = Path(data_dir)
    temp_results_dir = ensure_dir(temp_results_dir)

    if not query_fasta.is_file():
        raise FileNotFoundError(f"Query FASTA not found: {query_fasta}")

    # Path to Pfam-A HMM database
    pfam_hmm = data_dir / "complementary_databases" / "pfam" / pfam_hmm_name
    if not pfam_hmm.is_file():
        raise FileNotFoundError(
            f"Pfam HMM file not found: {pfam_hmm}. "
            "Make sure you ran the Pfam preparation step (download + hmmpress)."
        )

    # Output file in temp_results
    domtblout_path = temp_results_dir / "hmmer_pfam_domtblout.txt"

    cmd = [
        "hmmscan",
        "--cpu",
        str(n_workers),   # HMMER uses n_workers threads
        "--noali",        # do not output alignments (lighter output)
        "--domtblout",
        str(domtblout_path),
    ]
    if use_ga_cutoff:
        cmd.append("--cut_ga")

    cmd.extend([str(pfam_hmm), str(query_fasta)])

    print(f"[INFO] Running HMMER Pfam domain search (Block 2) with {n_workers} threads...")
    run_command(cmd)

    if not domtblout_path.is_file():
        raise RuntimeError(
            f"HMMER domtblout file not found: {domtblout_path}. "
            "Check that hmmscan executed correctly."
        )

    print(f"[INFO] Parsing HMMER domtblout: {domtblout_path}")

    rows: list[dict] = []
    with domtblout_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            # domtblout columns (HMMER 3.x):
            #  0: target name
            #  1: target accession
            #  2: tlen
            #  3: query name
            #  4: query accession
            #  5: qlen
            #  6: full sequence E-value
            #  7: full sequence score
            #  8: full sequence bias
            #  9: # (domain number)
            # 10: of (total domains)
            # 11: c-Evalue (domain)
            # 12: i-Evalue (domain)
            # 13: score (domain)
            # 14: bias (domain)
            # 15: hmm from
            # 16: hmm to
            # 17: ali from
            # 18: ali to
            # 19: env from
            # 20: env to
            # 21: acc (posterior probability)
            # 22+: description of target (optional, free text)

            target_name = parts[0]
            target_acc = parts[1]
            tlen = int(parts[2])
            query_name = parts[3]
            query_acc = parts[4]
            qlen = int(parts[5])

            full_evalue = float(parts[6])
            full_score = float(parts[7])
            full_bias = float(parts[8])

            dom_index = int(parts[9])
            dom_of = int(parts[10])

            dom_c_evalue = float(parts[11])
            dom_i_evalue = float(parts[12])
            dom_score = float(parts[13])
            dom_bias = float(parts[14])

            hmm_from = int(parts[15])
            hmm_to = int(parts[16])
            ali_from = int(parts[17])
            ali_to = int(parts[18])
            env_from = int(parts[19])
            env_to = int(parts[20])
            acc = float(parts[21])

            # Pfam accession without version (PF00069.20 -> PF00069)
            pfam_id = target_acc.split(".")[0]

            rows.append(
                {
                    "qseqid": query_name,
                    "qacc": query_acc,
                    "qlen": qlen,
                    "target_name": target_name,
                    "pfam_acc": target_acc,
                    "pfam_id": pfam_id,
                    "tlen": tlen,
                    "full_evalue": full_evalue,
                    "full_score": full_score,
                    "full_bias": full_bias,
                    "dom_index": dom_index,
                    "dom_of": dom_of,
                    "dom_c_evalue": dom_c_evalue,
                    "dom_i_evalue": dom_i_evalue,
                    "dom_score": dom_score,
                    "dom_bias": dom_bias,
                    "hmm_from": hmm_from,
                    "hmm_to": hmm_to,
                    "ali_from": ali_from,
                    "ali_to": ali_to,
                    "env_from": env_from,
                    "env_to": env_to,
                    "acc": acc,
                }
            )

    if not rows:
        print("[INFO] No HMMER domain hits found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Additional filter by domain i-Evalue
    mask = df["dom_i_evalue"] <= evalue_max
    df_filt = df.loc[mask].copy()

    if df_filt.empty:
        print(
            "[INFO] No domain hits passed the i-Evalue threshold "
            f"({evalue_max})."
        )
        return df_filt

    # Sort by query, then by i-Evalue and domain score
    df_filt = df_filt.sort_values(
        by=["qseqid", "dom_i_evalue", "dom_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    # Save filtered table to temp_results (detailed Block 2 output)
    filtered_path = temp_results_dir / "hmmer_pfam_hits_filtered.tsv"
    df_filt.to_csv(filtered_path, sep="\t", index=False)
    print(
        f"[INFO] Domain candidates after filtering: {len(df_filt)} rows "
        f"({df_filt['qseqid'].nunique()} queries). "
        f"Saved to: {filtered_path}"
    )

    return df_filt


# ----------------------------------------------------------------------
# Map Pfam hits → candidate proteins
# ----------------------------------------------------------------------

def map_pfam_hits_to_candidate_proteins(
    df_hmmer: pd.DataFrame,
    data_dir: str | Path = "databases",
    temp_results_dir: str | Path = "temp_results",
    save_full_hits: bool = True,
    save_candidates: bool = True,
    max_candidates_per_domain: int | None = None,
    df_blast_ranked: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Map Pfam domain hits (from HMMER) to candidate proteins using
    the protein_domains.parquet table.

    Parameters
    ----------
    df_hmmer : pd.DataFrame
        HMMER hits table after filtering (GA cutoff + e-value, etc.).
        Must contain at least:
          - qseqid   : query protein ID (from input FASTA)
          - pfam_acc : Pfam accession with version, e.g. 'PF00025.28'
        Optionally can already have:
          - pfam_id  : base Pfam ID without version, e.g. 'PF00025'
    data_dir : str or Path, default 'databases'
        Base data directory.
        Protein domains are expected at:
            <data_dir>/results_databases/protein_domains.parquet
    temp_results_dir : str or Path, default 'temp_results'
        Directory where intermediate results are stored.
    save_full_hits : bool, default True
        If True, save the full HMMER hits table to
        temp_results/hmmer_domain_hits.tsv
    save_candidates : bool, default True
        If True, save the slim candidate table to
        temp_results/candidate_proteins_domain.tsv
    max_candidates_per_domain : int, optional
        If provided, keep at most this many proteins per (query, Pfam).
    df_blast_ranked : pd.DataFrame, optional
        Raw BLAST ranking with qseqid/sseqid and score columns. Used to rank
        proteins inside each query Pfam before applying max_candidates_per_domain.

    Returns
    -------
    pd.DataFrame
        Candidate proteins by domain, with columns:
          - qseqid
          - sseqid       (mapped uniprot_id)
          - search_type  ('domain')
    """
    temp_results_dir = Path(temp_results_dir)
    temp_results_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir)

    # Fixed path to protein–domain mapping table
    protein_domains_path = (
        data_dir / "results_databases" / "protein_domains.parquet"
    )

    if not protein_domains_path.is_file():
        raise FileNotFoundError(
            f"protein_domains.parquet not found at: {protein_domains_path}"
        )

    # 1) Save full HMMER hits (for debugging / inspection)
    if save_full_hits:
        hmmer_hits_path = temp_results_dir / "hmmer_domain_hits.tsv"
        df_hmmer.to_csv(hmmer_hits_path, sep="\t", index=False)
        print(f"[INFO] Full HMMER hits saved to: {hmmer_hits_path}")

    if df_hmmer.empty:
        print("[INFO] No HMMER hits provided. Returning empty candidates table.")
        return pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])

    # 2) Ensure pfam_id without version (PF00025.28 -> PF00025)
    if "pfam_id" not in df_hmmer.columns:
        if "pfam_acc" not in df_hmmer.columns:
            raise ValueError(
                "df_hmmer must contain either 'pfam_id' or 'pfam_acc' column."
            )
        df_hmmer = df_hmmer.copy()
        df_hmmer["pfam_id"] = df_hmmer["pfam_acc"].astype(str).str.split(".").str[0]

    # Unique (query, Pfam) pairs with the strongest HMMER evidence per Pfam.
    domain_score_cols = ["qseqid", "pfam_id"]
    for col in ["dom_score", "dom_i_evalue"]:
        if col in df_hmmer.columns:
            domain_score_cols.append(col)
    df_qpfam = df_hmmer[domain_score_cols].drop_duplicates()
    agg_spec = {}
    if "dom_score" in df_qpfam.columns:
        agg_spec["best_domain_score"] = ("dom_score", "max")
    if "dom_i_evalue" in df_qpfam.columns:
        agg_spec["best_domain_evalue"] = ("dom_i_evalue", "min")
    if agg_spec:
        df_qpfam = (
            df_qpfam.groupby(["qseqid", "pfam_id"], as_index=False)
            .agg(**agg_spec)
        )
    else:
        df_qpfam = df_qpfam[["qseqid", "pfam_id"]].drop_duplicates()
    print(
        f"[INFO] Unique (query, Pfam) pairs from HMMER: "
        f"{len(df_qpfam)} rows, {df_qpfam['qseqid'].nunique()} queries"
    )

    # 3) Load protein–domain table
    df_domains = pd.read_parquet(protein_domains_path)

    expected_cols = {"uniprot_id", "pfam_id"}
    missing = expected_cols - set(df_domains.columns)
    if missing:
        raise ValueError(
            f"protein_domains.parquet is missing columns: {missing}. "
            f"Expected at least: {expected_cols}"
        )

    # 4) Map Pfam → uniprot_id
    df_map = df_qpfam.merge(
        df_domains[["uniprot_id", "pfam_id"]],
        on="pfam_id",
        how="left",
    )

    n_missing = df_map["uniprot_id"].isna().sum()
    if n_missing > 0:
        print(
            f"[INFO] {n_missing} (query, Pfam) pairs have no matching uniprot_id "
            f"in protein_domains and will be dropped."
        )
        df_map = df_map.dropna(subset=["uniprot_id"])

    if df_map.empty:
        print("[INFO] No Pfam hits could be mapped to known proteins.")
        df_candidates = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])
    else:
        df_map = (
            df_map[
                [
                    c
                    for c in [
                        "qseqid",
                        "pfam_id",
                        "uniprot_id",
                        "best_domain_score",
                        "best_domain_evalue",
                    ]
                    if c in df_map.columns
                ]
            ]
            .drop_duplicates()
            .rename(columns={"uniprot_id": "sseqid"})
            .reset_index(drop=True)
        )

        if max_candidates_per_domain is not None:
            if max_candidates_per_domain <= 0:
                raise ValueError("max_candidates_per_domain must be > 0.")

            if df_blast_ranked is None:
                df_blast_ranked = _load_ranked_blast_hits(
                    temp_results_dir=temp_results_dir,
                    exclude_sequence_hits=False,
                )

            rank_cols = ["qseqid", "sseqid", "bitscore", "evalue", "pident", "qcov", "scov"]
            missing_rank_cols = set(rank_cols) - set(df_blast_ranked.columns)
            if missing_rank_cols:
                raise ValueError(
                    "df_blast_ranked is missing required columns: "
                    f"{missing_rank_cols}"
                )

            ranking = df_blast_ranked[rank_cols].copy()
            ranking["qseqid"] = ranking["qseqid"].astype(str)
            ranking["sseqid"] = ranking["sseqid"].astype(str)
            df_map["qseqid"] = df_map["qseqid"].astype(str)
            df_map["sseqid"] = df_map["sseqid"].astype(str)

            df_map = df_map.merge(
                ranking,
                on=["qseqid", "sseqid"],
                how="left",
            )
            df_map["has_blast_hit"] = df_map["bitscore"].notna()
            df_map = (
                df_map.sort_values(
                    by=[
                        "qseqid",
                        "pfam_id",
                        "has_blast_hit",
                        "bitscore",
                        "evalue",
                        "pident",
                        "qcov",
                        "scov",
                        "best_domain_score",
                        "best_domain_evalue",
                        "sseqid",
                    ],
                    ascending=[True, True, False, False, True, False, False, False, False, True, True],
                    na_position="last",
                )
                .groupby(["qseqid", "pfam_id"], group_keys=False)
                .head(int(max_candidates_per_domain))
                .reset_index(drop=True)
            )

        # 5) Canonical candidate table by domain
        if "has_blast_hit" not in df_map.columns:
            df_map["has_blast_hit"] = False
        grouped_aggs = {
            "n_shared_domains": ("pfam_id", "nunique"),
            "has_blast_hit": ("has_blast_hit", "max"),
        }
        if "bitscore" in df_map.columns:
            grouped_aggs["bitscore"] = ("bitscore", "max")
        if "evalue" in df_map.columns:
            grouped_aggs["evalue"] = ("evalue", "min")
        if "pident" in df_map.columns:
            grouped_aggs["pident"] = ("pident", "max")
        if "qcov" in df_map.columns:
            grouped_aggs["qcov"] = ("qcov", "max")
        if "scov" in df_map.columns:
            grouped_aggs["scov"] = ("scov", "max")
        if "best_domain_score" in df_map.columns:
            grouped_aggs["best_domain_score"] = ("best_domain_score", "max")
        if "best_domain_evalue" in df_map.columns:
            grouped_aggs["best_domain_evalue"] = ("best_domain_evalue", "min")

        df_candidates = (
            df_map.groupby(["qseqid", "sseqid"], as_index=False)
            .agg(**grouped_aggs)
            .reset_index(drop=True)
        )
        df_candidates["search_type"] = "domain"

        print(
            f"[INFO] Domain-based candidate proteins: {len(df_candidates)} rows "
            f"({df_candidates['qseqid'].nunique()} queries, "
            f"{df_candidates['sseqid'].nunique()} unique proteins)"
        )

    # 6) Save reduced candidate table
    if save_candidates and not df_candidates.empty:
        candidates_path = temp_results_dir / "candidate_proteins_domain.tsv"
        df_candidates[["qseqid", "sseqid", "search_type"]].to_csv(candidates_path, sep="\t", index=False)
        print(f"[INFO] Domain candidates saved to: {candidates_path}")

    return df_candidates


# ----------------------------------------------------------------------
# Combined candidate mapping (sequence + domain)
# ----------------------------------------------------------------------

def combine_sequence_and_domain_candidates(
    df_candidates_seq: pd.DataFrame | None,
    df_candidates_nearest_k: pd.DataFrame | None = None,
    df_candidates_domain: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Combine candidate proteins from sequence- and domain-based searches.

    Rule:
      - For each qseqid, proteins found by sequence are excluded
        from the domain-based candidates.

    Parameters
    ----------
    df_candidates_seq : pd.DataFrame or None
        Must contain columns:
          - qseqid
          - sseqid
        search_type is assumed to be 'sequence' (or will be set).
    df_candidates_domain : pd.DataFrame or None
        Must contain columns:
          - qseqid
          - sseqid
        search_type is assumed to be 'domain' (or will be set).

    Returns
    -------
    pd.DataFrame
        Combined candidates with columns:
          - qseqid
          - sseqid
          - search_type
    """
    # Normalize None → empty DataFrames
    if df_candidates_seq is None:
        df_candidates_seq = pd.DataFrame(columns=["qseqid", "sseqid"])
    if df_candidates_nearest_k is None:
        df_candidates_nearest_k = pd.DataFrame(columns=["qseqid", "sseqid"])
    if df_candidates_domain is None:
        df_candidates_domain = pd.DataFrame(columns=["qseqid", "sseqid"])

    if "search_type" not in df_candidates_seq.columns:
        df_candidates_seq = df_candidates_seq.copy()
        df_candidates_seq["search_type"] = "sequence"

    if "search_type" not in df_candidates_nearest_k.columns:
        df_candidates_nearest_k = df_candidates_nearest_k.copy()
        df_candidates_nearest_k["search_type"] = "nearest_k"

    if "search_type" not in df_candidates_domain.columns:
        df_candidates_domain = df_candidates_domain.copy()
        df_candidates_domain["search_type"] = "domain"

    frames = []
    if not df_candidates_seq.empty:
        frames.append(df_candidates_seq.copy())
    if not df_candidates_nearest_k.empty:
        frames.append(df_candidates_nearest_k.copy())
    if not df_candidates_domain.empty:
        frames.append(df_candidates_domain.copy())

    if not frames:
        return pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])

    priority = pd.CategoricalDtype(["sequence", "nearest_k", "domain"], ordered=True)
    df_all = pd.concat(frames, ignore_index=True).drop_duplicates()
    df_all["search_type"] = df_all["search_type"].astype(priority)
    df_all = (
        df_all.sort_values(["qseqid", "sseqid", "search_type"])
        .drop_duplicates(subset=["qseqid", "sseqid"], keep="first")
        .reset_index(drop=True)
    )
    df_all["search_type"] = df_all["search_type"].astype(str)
    return df_all


def add_shared_domain_counts_to_candidates(
    df_candidates: pd.DataFrame,
    df_hmmer: pd.DataFrame,
    data_dir: str | Path = "databases",
) -> pd.DataFrame:
    """Add n_shared_domains for candidate proteins without changing candidate selection."""
    if df_candidates is None or df_candidates.empty:
        return df_candidates
    if df_hmmer is None or df_hmmer.empty:
        return df_candidates

    if "pfam_id" not in df_hmmer.columns:
        if "pfam_acc" not in df_hmmer.columns:
            return df_candidates
        df_hmmer = df_hmmer.copy()
        df_hmmer["pfam_id"] = df_hmmer["pfam_acc"].astype(str).str.split(".").str[0]

    protein_domains_path = Path(data_dir) / "results_databases" / "protein_domains.parquet"
    if not protein_domains_path.is_file():
        return df_candidates

    df_query_domains = df_hmmer[["qseqid", "pfam_id"]].dropna().drop_duplicates()
    if df_query_domains.empty:
        return df_candidates

    df_domains = pd.read_parquet(
        protein_domains_path,
        columns=["uniprot_id", "pfam_id"],
    ).dropna(subset=["uniprot_id", "pfam_id"])
    df_domains["uniprot_id"] = df_domains["uniprot_id"].astype(str)
    df_domains["pfam_id"] = df_domains["pfam_id"].astype(str)
    df_domains = df_domains.drop_duplicates(subset=["uniprot_id", "pfam_id"])

    candidates = df_candidates.copy()
    candidates["qseqid"] = candidates["qseqid"].astype(str)
    candidates["sseqid"] = candidates["sseqid"].astype(str)
    df_query_domains["qseqid"] = df_query_domains["qseqid"].astype(str)

    shared_counts = (
        candidates[["qseqid", "sseqid"]]
        .drop_duplicates()
        .merge(
            df_domains.rename(columns={"uniprot_id": "sseqid"}),
            on="sseqid",
            how="inner",
        )
        .merge(df_query_domains, on=["qseqid", "pfam_id"], how="inner")
        .groupby(["qseqid", "sseqid"], as_index=False)["pfam_id"]
        .nunique()
        .rename(columns={"pfam_id": "computed_n_shared_domains"})
    )

    candidates = candidates.merge(shared_counts, on=["qseqid", "sseqid"], how="left")
    if "n_shared_domains" in candidates.columns:
        candidates["n_shared_domains"] = candidates["n_shared_domains"].combine_first(
            candidates["computed_n_shared_domains"]
        )
    else:
        candidates["n_shared_domains"] = candidates["computed_n_shared_domains"]
    candidates = candidates.drop(columns=["computed_n_shared_domains"])
    candidates["n_shared_domains"] = candidates["n_shared_domains"].fillna(0).astype(int)
    return candidates


# ----------------------------------------------------------------------
# Block 3 – Query → ligand results (chunked, simple per-query collapse)
# ----------------------------------------------------------------------

def _build_protein_ranking_for_query(df_cand_q: pd.DataFrame) -> pd.DataFrame:
    """Build an explanatory per-query protein ranking from candidate metadata."""
    output_cols = [
        "protein_rank",
        "qseqid",
        "sseqid",
        "search_type",
        "ranking_source",
        "blast_bitscore",
        "blast_evalue",
        "blast_pident",
        "blast_qcov",
        "blast_scov",
        "best_domain_score",
        "best_domain_evalue",
        "n_shared_domains",
    ]
    if df_cand_q is None or df_cand_q.empty:
        return pd.DataFrame(columns=output_cols)

    ranking = df_cand_q.copy()
    for col in ["bitscore", "evalue", "pident", "qcov", "scov", "best_domain_score", "best_domain_evalue"]:
        if col not in ranking.columns:
            ranking[col] = np.nan
    if "n_shared_domains" not in ranking.columns:
        ranking["n_shared_domains"] = np.nan

    priority = pd.CategoricalDtype(["sequence", "nearest_k", "domain"], ordered=True)
    ranking["search_type"] = ranking["search_type"].astype(priority)
    ranking["_has_blast"] = ranking["bitscore"].notna()
    ranking["_method_rank"] = ranking["search_type"].cat.codes

    ranking = ranking.sort_values(
        by=[
            "_method_rank",
            "_has_blast",
            "bitscore",
            "evalue",
            "pident",
            "qcov",
            "scov",
            "best_domain_score",
            "best_domain_evalue",
            "n_shared_domains",
            "sseqid",
        ],
        ascending=[True, False, False, True, False, False, False, False, True, False, True],
        na_position="last",
    ).reset_index(drop=True)

    ranking["protein_rank"] = np.arange(1, len(ranking) + 1)
    ranking["ranking_source"] = np.where(
        ranking["_has_blast"],
        "blast",
        np.where(ranking["best_domain_score"].notna() | ranking["best_domain_evalue"].notna(), "domain", ""),
    )
    ranking["search_type"] = ranking["search_type"].astype(str)
    ranking = ranking.rename(
        columns={
            "bitscore": "blast_bitscore",
            "evalue": "blast_evalue",
            "pident": "blast_pident",
            "qcov": "blast_qcov",
            "scov": "blast_scov",
        }
    )
    return ranking[output_cols]


def _filter_protein_ranking_to_ligand_contributors(
    protein_ranking: pd.DataFrame,
    known_ligands: pd.DataFrame,
    predicted_ligands: pd.DataFrame,
) -> pd.DataFrame:
    """Keep proteins represented in at least one final per-query ligand table."""
    if protein_ranking is None or protein_ranking.empty:
        return protein_ranking

    contributing_proteins: set[str] = set()
    for ligand_table in (known_ligands, predicted_ligands):
        if ligand_table is None or ligand_table.empty:
            continue
        protein_col = next(
            (col for col in ("uniprot_id", "sseqid") if col in ligand_table.columns),
            None,
        )
        if protein_col is None:
            continue
        contributing_proteins.update(
            ligand_table[protein_col].dropna().astype(str).tolist()
        )

    filtered = protein_ranking[
        protein_ranking["sseqid"].astype(str).isin(contributing_proteins)
    ].copy()
    filtered = filtered.reset_index(drop=True)
    filtered["protein_rank"] = np.arange(1, len(filtered) + 1)
    return filtered


def _process_single_query(
    qseqid: str,
    df_cand_q: pd.DataFrame,
    known_q: pd.DataFrame,
    predicted_q: pd.DataFrame,
    known_db_cols: list[str],
    known_ligand_col: str | None,
    predicted_ligand_col: str | None,
    search_results_dir: Path,
    save_per_query: bool = True,
    drop_duplicates: bool = True,
) -> dict:
    """
    Process a single query (qseqid).

    For a given query:
      - Rank proteins that contribute at least one retained known or predicted
        ligand, split by search_type (sequence / nearest_k / domain).
      - Optionally collapse ligands to one row per ligand ID
        (chem_comp_id / predicted_chem_comp_id), prioritizing sequence hits
        over domain hits.
      - Write per-query TSV files (known_ligands.tsv, predicted_ligands.tsv).
      - Compute lightweight summary counts for the global summary table.

    IMPORTANT:
    - `known_q` and `predicted_q` are NOT globally collapsed; they come
      directly from per-chunk merges.
    """
    # Ensure DataFrames
    if known_q is None:
        known_q = pd.DataFrame()
    if predicted_q is None:
        predicted_q = pd.DataFrame()

    protein_ranking = _build_protein_ranking_for_query(df_cand_q)
    protein_rank_map: dict[str, int] = {}
    if not protein_ranking.empty:
        protein_rank_map = dict(
            zip(
                protein_ranking["sseqid"].astype(str),
                protein_ranking["protein_rank"].astype(int),
            )
        )

    # -----------------------------
    # 1) Optional ligand collapse
    # -----------------------------
    def _sort_and_collapse(df: pd.DataFrame, ligand_col: str | None) -> pd.DataFrame:
        """
        Sort by search_type to prioritize sequence hits and optionally
        collapse to one row per ligand, keeping the first occurrence.
        """
        if df.empty or ligand_col is None or ligand_col not in df.columns:
            return df

        df = df.copy()

        # Prioritize protein ranking first; keep stable ligand ordering within a protein.
        if "uniprot_id" in df.columns:
            df["_protein_rank_sort"] = (
                df["uniprot_id"].astype(str).map(protein_rank_map).fillna(len(protein_rank_map) + 1)
            )
        elif "sseqid" in df.columns:
            df["_protein_rank_sort"] = (
                df["sseqid"].astype(str).map(protein_rank_map).fillna(len(protein_rank_map) + 1)
            )
        else:
            df["_protein_rank_sort"] = len(protein_rank_map) + 1

        # Prioritize "sequence" over "domain" if search_type is present.
        if "search_type" in df.columns:
            # Use an ordered categorical so sequence < domain.
            cat = pd.CategoricalDtype(["sequence", "nearest_k", "domain"], ordered=True)
            df["search_type"] = df["search_type"].astype(cat)
            df = df.sort_values(["_protein_rank_sort", "search_type"], kind="mergesort")
        else:
            df = df.sort_values("_protein_rank_sort", kind="mergesort")

        if drop_duplicates:
            # One row per ligand (ligand_col), keeping the first row,
            # which will correspond to the best-ranked protein when present.
            df = df.drop_duplicates(subset=[ligand_col], keep="first")

        df = df.drop(columns=["_protein_rank_sort"])
        return df

    known_q_final = _sort_and_collapse(known_q, known_ligand_col)
    predicted_q_final = _sort_and_collapse(predicted_q, predicted_ligand_col)
    protein_ranking = _filter_protein_ranking_to_ligand_contributors(
        protein_ranking,
        known_q_final,
        predicted_q_final,
    )

    # Protein summary counts match the filtered ranking exposed to users.
    if protein_ranking.empty:
        n_prot_seq = 0
        n_prot_nk = 0
        n_prot_dom = 0
    else:
        mask_seq = protein_ranking["search_type"] == "sequence"
        mask_nk = protein_ranking["search_type"] == "nearest_k"
        mask_dom = protein_ranking["search_type"] == "domain"
        n_prot_seq = protein_ranking.loc[mask_seq, "sseqid"].nunique()
        n_prot_nk = protein_ranking.loc[mask_nk, "sseqid"].nunique()
        n_prot_dom = protein_ranking.loc[mask_dom, "sseqid"].nunique()

    # -----------------------------
    # 2) Write per-query TSV files
    # -----------------------------
    if save_per_query and (
        not df_cand_q.empty or not known_q_final.empty or not predicted_q_final.empty
    ):
        q_dir = ensure_dir(search_results_dir / qseqid)

        protein_ranking.to_csv(q_dir / "protein_ranking.tsv", sep="\t", index=False)

        # --- Known ligands ---
        if not known_q_final.empty:
            # Keep: search_type + all known_db columns that are present.
            cols_known = ["search_type"] + list(known_db_cols)
            cols_known = [c for c in cols_known if c in known_q_final.columns]
            df_known_out = known_q_final[cols_known].copy()

            known_path = q_dir / "known_ligands.tsv"
            df_known_out.to_csv(known_path, sep="\t", index=False)

        # --- Predicted ligands ---
        if not predicted_q_final.empty:
            df_predicted_out = predicted_q_final.copy()

            # Normalize ligand column name to "chem_comp_id" when possible.
            ligand_col_out = "chem_comp_id"
            if predicted_ligand_col is not None:
                if (
                    predicted_ligand_col in df_predicted_out.columns
                    and ligand_col_out not in df_predicted_out.columns
                ):
                    df_predicted_out = df_predicted_out.rename(
                        columns={predicted_ligand_col: ligand_col_out}
                    )
                elif predicted_ligand_col == ligand_col_out:
                    ligand_col_out = predicted_ligand_col
                else:
                    # Fallback: if chem_comp_id already exists, use it.
                    if ligand_col_out not in df_predicted_out.columns:
                        if "chem_comp_id" in df_predicted_out.columns:
                            ligand_col_out = "chem_comp_id"

            preferred_order = [
                "search_type",
                "uniprot_id",
                ligand_col_out,
                "possible_binding_sites",
                "query_id",
                "bsi_score",
                "pfam_id",
                "tanimoto",
                "similarity",
                "smiles",
            ]
            cols_predicted = [c for c in preferred_order if c in df_predicted_out.columns]
            extra_cols = [c for c in df_predicted_out.columns if c not in cols_predicted]
            cols_predicted = cols_predicted + extra_cols

            df_predicted_out = df_predicted_out[cols_predicted]

            predicted_path = q_dir / "predicted_ligands.tsv"
            df_predicted_out.to_csv(predicted_path, sep="\t", index=False)

    # -----------------------------
    # 3) Ligand counts for summary
    # -----------------------------
    # Known ligands
    if (
        known_q_final.empty
        or known_ligand_col is None
        or known_ligand_col not in known_q_final.columns
    ):
        n_known_seq = 0
        n_known_nk = 0
        n_known_dom = 0
    else:
        df_known_seq = known_q_final[known_q_final["search_type"] == "sequence"]
        df_known_dom = known_q_final[known_q_final["search_type"] == "domain"]
        df_known_nk = known_q_final[known_q_final["search_type"] == "nearest_k"]
        n_known_seq = df_known_seq[known_ligand_col].nunique()
        n_known_nk = df_known_nk[known_ligand_col].nunique()
        n_known_dom = df_known_dom[known_ligand_col].nunique()

    # Predicted ligands
    if (
        predicted_q_final.empty
        or predicted_ligand_col is None
        or predicted_ligand_col not in predicted_q_final.columns
    ):
        n_predicted_seq = 0
        n_predicted_nk = 0
        n_predicted_dom = 0
    else:
        df_predicted_seq = predicted_q_final[predicted_q_final["search_type"] == "sequence"]
        df_predicted_dom = predicted_q_final[predicted_q_final["search_type"] == "domain"]
        df_predicted_nk = predicted_q_final[predicted_q_final["search_type"] == "nearest_k"]
        n_predicted_seq = df_predicted_seq[predicted_ligand_col].nunique()
        n_predicted_nk = df_predicted_nk[predicted_ligand_col].nunique()
        n_predicted_dom = df_predicted_dom[predicted_ligand_col].nunique()

    return {
        "qseqid": qseqid,
        "n_proteins_sequence": n_prot_seq,
        "n_proteins_nearest_k": n_prot_nk,
        "n_proteins_domain": n_prot_dom,
        "n_known_ligands_sequence": n_known_seq,
        "n_known_ligands_nearest_k": n_known_nk,
        "n_known_ligands_domain": n_known_dom,
        "n_predicted_ligands_sequence": n_predicted_seq,
        "n_predicted_ligands_nearest_k": n_predicted_nk,
        "n_predicted_ligands_domain": n_predicted_dom,
    }


def build_query_ligand_results(
    df_queries: pd.DataFrame,
    df_candidates_all: pd.DataFrame,
    known_db: pd.DataFrame,
    predicted_db: pd.DataFrame,
    output_dir: str | Path = "results",
    search_results_subdir: str = "search_results",
    save_per_query: bool = True,
    save_summary: bool = True,
    drop_duplicates: bool = True,
    predicted_score_col: str | None = None,
    predicted_threshold_min: float | None = None,
    predicted_threshold_max: float | None = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Sequential wrapper for Block 3.

    Internally delegates to the "parallel" implementation with:
      - njobs = 1
      - executor = "thread" (ignored)
      - chunk_size_queries = len(df_queries)

    This keeps a single entry point for Block 3 logic.
    """
    return build_query_ligand_results_parallel(
        df_queries=df_queries,
        df_candidates_all=df_candidates_all,
        known_db=known_db,
        predicted_db=predicted_db,
        output_dir=output_dir,
        search_results_subdir=search_results_subdir,
        save_per_query=save_per_query,
        save_summary=save_summary,
        njobs=1,
        executor="thread",
        chunk_size_queries=len(df_queries) if len(df_queries) > 0 else 1,
        drop_duplicates=drop_duplicates,
        predicted_score_col=predicted_score_col,
        predicted_threshold_min=predicted_threshold_min,
        predicted_threshold_max=predicted_threshold_max,
        progress_callback=progress_callback,
    )


def build_query_ligand_results_parallel(
    df_queries: pd.DataFrame,
    df_candidates_all: pd.DataFrame,
    known_db: pd.DataFrame,
    predicted_db: pd.DataFrame | str | Path,
    output_dir: str | Path = "results",
    search_results_subdir: str = "search_results",
    save_per_query: bool = True,
    save_summary: bool = True,
    njobs: int = 4,
    executor: str = "process",  # kept for compatibility, currently ignored
    chunk_size_queries: int | None = 100,
    drop_duplicates: bool = True,
    predicted_filter_batch_size: int = 2000,
    predicted_score_col: str | None = None,
    predicted_threshold_min: float | None = None,
    predicted_threshold_max: float | None = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """
    Parallel Block 3 with query chunking (per-query ligand collapse).

    Strategy:
      - Build a list of qseqid from df_queries.
      - Work in chunks of queries (chunk_size_queries) to limit
        merge size and RAM usage.
      - For each chunk:
          * Filter df_candidates_all to that subset of qseqid.
          * Reduce known_db and predicted_db to the proteins present
            in the chunk.
          * Merge per chunk:
              - df_cand_chunk × known_db_chunk
              - df_cand_chunk × predicted_db_chunk
          * Split per qseqid (groupby).
          * Process each query sequentially to keep RAM bounded.
            using `_process_single_query`, which:
              - counts ligand-contributing proteins,
              - sorts by search_type,
              - optionally collapses to one row per ligand ID,
              - writes per-query TSVs,
              - returns one summary row.

    Notes:
      - The `executor` parameter is kept for API compatibility and is ignored.
      - Queries are processed sequentially to avoid peak-memory spikes from
        large intermediate merged DataFrames.
      - `drop_duplicates=True` collapses to one row per ligand, giving
        priority to ligands found by sequence search.
    """
    if njobs is None or njobs < 1:
        njobs = 1

    output_dir = ensure_dir(output_dir)
    search_results_dir = ensure_dir(Path(output_dir) / search_results_subdir)

    # Basic column validations
    if "qseqid" not in df_queries.columns:
        raise ValueError("df_queries must contain a 'qseqid' column.")
    if "qseqid" not in df_candidates_all.columns:
        raise ValueError("df_candidates_all must contain a 'qseqid' column.")
    if "sseqid" not in df_candidates_all.columns:
        raise ValueError(
            "df_candidates_all must contain a 'sseqid' column (subject protein id)."
        )
    if "search_type" not in df_candidates_all.columns:
        raise ValueError(
            "df_candidates_all must contain a 'search_type' column "
            "('sequence'/'nearest_k'/'domain')."
        )

    qseqids_all = list(df_queries["qseqid"])

    # No queries → return empty summary
    if len(qseqids_all) == 0:
        empty_cols = [
            "qseqid",
            "n_proteins_sequence",
            "n_proteins_nearest_k",
            "n_proteins_domain",
            "n_known_ligands_sequence",
            "n_known_ligands_nearest_k",
            "n_known_ligands_domain",
            "n_predicted_ligands_sequence",
            "n_predicted_ligands_nearest_k",
            "n_predicted_ligands_domain",
        ]
        summary_df = pd.DataFrame(columns=empty_cols)
        if save_summary:
            summary_path = output_dir / "search_results_summary.tsv"
            summary_df.to_csv(summary_path, sep="\t", index=False)
            print(f"[INFO] Global summary saved to: {summary_path}")
        return summary_df

    # Detect ligand columns
    known_ligand_col = "chem_comp_id" if "chem_comp_id" in known_db.columns else None

    predicted_is_path = isinstance(predicted_db, (str, Path))
    predicted_path = Path(predicted_db) if predicted_is_path else None
    if predicted_is_path:
        if predicted_path is None or not predicted_path.exists():
            raise ValueError(f"predicted_db parquet path does not exist: {predicted_db}")
        predicted_columns = pq.ParquetFile(predicted_path).schema_arrow.names
    else:
        predicted_columns = list(predicted_db.columns)

    if "predicted_chem_comp_id" in predicted_columns:
        predicted_ligand_col = "predicted_chem_comp_id"
    elif "zinc_chem_comp_id" in predicted_columns:
        predicted_ligand_col = "zinc_chem_comp_id"
    elif "chem_comp_id" in predicted_columns:
        predicted_ligand_col = "chem_comp_id"
    else:
        predicted_ligand_col = None

    known_db_cols = list(known_db.columns)

    # Pre-reduce known_db and predicted_db to proteins present in df_candidates_all
    if df_candidates_all.empty:
        prots_all = np.array([], dtype=object)
    else:
        prots_all = df_candidates_all["sseqid"].unique()

    if "uniprot_id" not in known_db.columns:
        raise ValueError("known_db must contain an 'uniprot_id' column.")
    if "uniprot_id" not in predicted_columns:
        raise ValueError("predicted_db must contain an 'uniprot_id' column.")

    known_db_small = known_db[known_db["uniprot_id"].isin(prots_all)].copy()
    predicted_db_small = None if predicted_is_path else predicted_db[predicted_db["uniprot_id"].isin(prots_all)].copy()

    # Normalize chunk size
    n_total = len(qseqids_all)
    if chunk_size_queries is None or chunk_size_queries <= 0:
        chunk_size_queries = n_total

    summary_rows: list[dict] = []

    # Process queries in chunks
    for start in range(0, n_total, chunk_size_queries):
        end = min(start + chunk_size_queries, n_total)
        q_chunk = qseqids_all[start:end]

        # Filter candidates to current chunk
        if df_candidates_all.empty:
            df_cand_chunk = df_candidates_all
        else:
            df_cand_chunk = df_candidates_all[
                df_candidates_all["qseqid"].isin(q_chunk)
            ]

        if df_cand_chunk.empty:
            cand_by_q: dict[str, pd.DataFrame] = {}
            known_db_chunk = pd.DataFrame(columns=known_db.columns)
            predicted_db_chunk = pd.DataFrame(columns=predicted_columns)
        else:
            cand_by_q = {
                q: subdf for q, subdf in df_cand_chunk.groupby("qseqid", sort=False)
            }
            prots_chunk = df_cand_chunk["sseqid"].unique()
            known_db_chunk = known_db_small[
                known_db_small["uniprot_id"].isin(prots_chunk)
            ]

            if predicted_is_path:
                predicted_db_chunk = _read_parquet_rows_for_uniprot_ids(
                    parquet_path=predicted_path,
                    uniprot_ids=[str(p) for p in prots_chunk],
                    batch_size=predicted_filter_batch_size,
                )
            else:
                predicted_db_chunk = predicted_db_small[
                    predicted_db_small["uniprot_id"].isin(prots_chunk)
                ]

            if predicted_score_col is not None and predicted_score_col in predicted_db_chunk.columns:
                if predicted_threshold_min is not None:
                    predicted_db_chunk = predicted_db_chunk[predicted_db_chunk[predicted_score_col] >= float(predicted_threshold_min)]
                if predicted_threshold_max is not None:
                    predicted_db_chunk = predicted_db_chunk[predicted_db_chunk[predicted_score_col] <= float(predicted_threshold_max)]
                predicted_db_chunk = predicted_db_chunk.reset_index(drop=True)

        # Process queries in the chunk.
        # NOTE: we intentionally avoid building chunk-level merged DataFrames
        # (df_cand_chunk × known_db / predicted_db), because those can explode in RAM.
        for qseqid in q_chunk:
            df_cand_q = cand_by_q.get(
                qseqid,
                pd.DataFrame(columns=["qseqid", "sseqid", "search_type"]),
            )

            if df_cand_q.empty:
                known_q = pd.DataFrame()
                predicted_q = pd.DataFrame()
            else:
                prots_q = df_cand_q["sseqid"].unique()
                df_cand_q_for_ligands = df_cand_q[["qseqid", "sseqid", "search_type"]].copy()

                known_q = df_cand_q_for_ligands.merge(
                    known_db_chunk[known_db_chunk["uniprot_id"].isin(prots_q)],
                    left_on="sseqid",
                    right_on="uniprot_id",
                    how="inner",
                )

                predicted_db_q = predicted_db_chunk[predicted_db_chunk["uniprot_id"].isin(prots_q)]
                predicted_q = df_cand_q_for_ligands.merge(
                    predicted_db_q,
                    left_on="sseqid",
                    right_on="uniprot_id",
                    how="inner",
                )

            summary_row = _process_single_query(
                qseqid=qseqid,
                df_cand_q=df_cand_q,
                known_q=known_q,
                predicted_q=predicted_q,
                known_db_cols=known_db_cols,
                known_ligand_col=known_ligand_col,
                predicted_ligand_col=predicted_ligand_col,
                search_results_dir=search_results_dir,
                save_per_query=save_per_query,
                drop_duplicates=drop_duplicates,
            )
            summary_rows.append(summary_row)
            if progress_callback:
                progress_callback(len(summary_rows), n_total)

        print(f"[INFO] Block 3: processed queries {start + 1}-{end} / {n_total}")

    # Build global summary
    summary_df = (
        pd.DataFrame(summary_rows).sort_values("qseqid").reset_index(drop=True)
    )

    if save_summary:
        summary_path = output_dir / "search_results_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"[INFO] Global summary saved to: {summary_path}")

    return summary_df
