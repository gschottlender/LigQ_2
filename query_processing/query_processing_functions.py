from __future__ import annotations

import gzip
import shutil
import shutil as pyshutil
import subprocess
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------

PFAM_A_URL = (
    "ftp://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
)


# ----------------------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------------------

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

    # If DB seems already created, skip
    if (blast_dir / f"{db_name}.pin").exists() or (blast_dir / f"{db_name}.psq").exists():
        print("[INFO] BLAST DB already present, skipping makeblastdb.")
        return blast_dir

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
    pfam_dir = complementary_dir / "pfam"
    download_and_prepare_pfam_a(pfam_dir)

    # 2) BLAST DB from target_sequences.fasta
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
        str(max_hits),
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

    # Sort by identity (desc) and evalue (asc)
    df_filt = df_filt.sort_values(
        by=["qseqid", "pident", "evalue"],
        ascending=[True, False, True],
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
        df_filt[["qseqid", "sseqid"]]
        .drop_duplicates()
        .assign(search_type="sequence")
        .reset_index(drop=True)
    )
    mapping_path = temp_results_dir / "candidate_proteins_sequence.tsv"
    mapping.to_csv(mapping_path, sep="\t", index=False)
    print(f"[INFO] Saved sequence-based candidate mapping to: {mapping_path}")

    return mapping


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

    # Unique (query, Pfam) pairs
    df_qpfam = df_hmmer[["qseqid", "pfam_id"]].drop_duplicates()
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
        # 5) Canonical candidate table by domain
        df_candidates = (
            df_map[["qseqid", "uniprot_id"]]
            .drop_duplicates()
            .rename(columns={"uniprot_id": "sseqid"})
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
        df_candidates.to_csv(candidates_path, sep="\t", index=False)
        print(f"[INFO] Domain candidates saved to: {candidates_path}")

    return df_candidates


# ----------------------------------------------------------------------
# Combined candidate mapping (sequence + domain)
# ----------------------------------------------------------------------

def combine_sequence_and_domain_candidates(
    df_candidates_seq: pd.DataFrame | None,
    df_candidates_domain: pd.DataFrame | None,
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
    if df_candidates_domain is None:
        df_candidates_domain = pd.DataFrame(columns=["qseqid", "sseqid"])

    if "search_type" not in df_candidates_seq.columns:
        df_candidates_seq = df_candidates_seq.copy()
        df_candidates_seq["search_type"] = "sequence"

    if "search_type" not in df_candidates_domain.columns:
        df_candidates_domain = df_candidates_domain.copy()
        df_candidates_domain["search_type"] = "domain"

    # If there are no sequence-based candidates, keep all domain-based ones
    if df_candidates_seq.empty:
        return (
            df_candidates_domain[["qseqid", "sseqid", "search_type"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

    # Pairs found by sequence
    seq_pairs = df_candidates_seq[["qseqid", "sseqid"]].drop_duplicates()

    # Exclude from domain any (qseqid, sseqid) already found by sequence
    df_domain_filtered = df_candidates_domain.merge(
        seq_pairs,
        on=["qseqid", "sseqid"],
        how="left",
        indicator=True,
    )
    df_domain_filtered = df_domain_filtered[df_domain_filtered["_merge"] == "left_only"]
    df_domain_filtered = df_domain_filtered.drop(columns=["_merge"])

    # Concatenate sequence and filtered domain candidates
    df_all = pd.concat(
        [
            df_candidates_seq[["qseqid", "sseqid", "search_type"]],
            df_domain_filtered[["qseqid", "sseqid", "search_type"]],
        ],
        ignore_index=True,
    ).drop_duplicates()

    return df_all.reset_index(drop=True)


# ----------------------------------------------------------------------
# Block 3 – Query → ligand results (sequential + parallel versions)
# ----------------------------------------------------------------------

def _process_single_query(
    qseqid: str,
    df_candidates_all: pd.DataFrame,
    known_merged: pd.DataFrame,
    zinc_merged: pd.DataFrame,
    known_db: pd.DataFrame,
    zinc_db: pd.DataFrame,
    known_ligand_col: str | None,
    zinc_ligand_col: str | None,
    search_results_dir: Path,
    save_per_query: bool = True,
) -> dict:
    """
    Process a single query (qseqid): collapse ligands, write per-query
    TSV files, and compute summary counts.
    """
    # --- Candidate proteins ---
    if not df_candidates_all.empty:
        df_cand_q = df_candidates_all[df_candidates_all["qseqid"] == qseqid]
    else:
        df_cand_q = pd.DataFrame(columns=["qseqid", "sseqid", "search_type"])

    if df_cand_q.empty:
        n_prot_seq = 0
        n_prot_dom = 0
    else:
        n_prot_seq = df_cand_q[df_cand_q["search_type"] == "sequence"]["sseqid"].nunique()
        n_prot_dom = df_cand_q[df_cand_q["search_type"] == "domain"]["sseqid"].nunique()

    # --- Subtables of ligands for this query (before collapsing) ---
    known_q = known_merged[known_merged["qseqid"] == qseqid].copy()
    zinc_q = zinc_merged[zinc_merged["qseqid"] == qseqid].copy()

    # --- 1) Collapse ligands in known_q with priority sequence > domain ---
    if not known_q.empty and known_ligand_col is not None:
        known_q["priority"] = np.where(known_q["search_type"] == "sequence", 0, 1)
        known_q = known_q.sort_values([known_ligand_col, "priority"])

        group_cols = ["qseqid", known_ligand_col]
        agg_dict: dict[str, object] = {
            "search_type": "first",  # respect priority sequence > domain
        }

        # Special aggregators if columns exist
        if "uniprot_id" in known_q.columns:
            agg_dict["uniprot_id"] = _unique_list
        if "binding_sites" in known_q.columns:
            agg_dict["binding_sites"] = _unique_list_flat
        if "pdb_ids" in known_q.columns:
            agg_dict["pdb_ids"] = _unique_list_flat

        # For all remaining columns, take the first value
        for col in known_q.columns:
            if col in group_cols or col in agg_dict or col == "priority":
                continue
            agg_dict[col] = "first"

        known_q_collapsed = (
            known_q.groupby(group_cols, as_index=False)
            .agg(agg_dict)
            .drop(columns=["priority"], errors="ignore")
        )
    else:
        known_q_collapsed = known_q  # empty or no ligand ID column

    # --- 2) Build possible_binding_sites for ZINC, if possible ---
    if (
        not zinc_q.empty
        and zinc_ligand_col is not None
        and not known_q_collapsed.empty
        and "binding_sites" in known_q_collapsed.columns
        and "query_id" in zinc_q.columns
    ):
        # Map: chem_comp_id (known) -> binding_sites (list) for this query
        bs_map = (
            known_q_collapsed.groupby(known_ligand_col)["binding_sites"]
            .apply(_unique_list_flat)
            .to_dict()
        )

        def _map_possible_bs(row: pd.Series):
            qid = row.get("query_id", None)
            if qid in bs_map:
                return bs_map[qid]
            return np.nan

        zinc_q["possible_binding_sites"] = zinc_q.apply(_map_possible_bs, axis=1)
    elif not zinc_q.empty:
        # No binding_sites or query_id → keep column as NaN
        if "possible_binding_sites" not in zinc_q.columns:
            zinc_q["possible_binding_sites"] = np.nan

    # --- 3) Collapse ZINC ligands with priority sequence > domain ---
    if not zinc_q.empty and zinc_ligand_col is not None:
        zinc_q["priority"] = np.where(zinc_q["search_type"] == "sequence", 0, 1)
        zinc_q = zinc_q.sort_values([zinc_ligand_col, "priority"])

        group_cols_z = ["qseqid", zinc_ligand_col]
        agg_dict_z: dict[str, object] = {
            "search_type": "first",
        }

        if "uniprot_id" in zinc_q.columns:
            agg_dict_z["uniprot_id"] = _unique_list
        if "possible_binding_sites" in zinc_q.columns:
            agg_dict_z["possible_binding_sites"] = _unique_list_flat

        # For remaining columns, take the first value
        for col in zinc_q.columns:
            if col in group_cols_z or col in agg_dict_z or col == "priority":
                continue
            agg_dict_z[col] = "first"

        zinc_q_collapsed = (
            zinc_q.groupby(group_cols_z, as_index=False)
            .agg(agg_dict_z)
            .drop(columns=["priority"], errors="ignore")
        )
    else:
        zinc_q_collapsed = zinc_q

    # --- 4) Write per-query TSV files (collapsed tables) ---
    if save_per_query and (not known_q_collapsed.empty or not zinc_q_collapsed.empty):
        q_dir = ensure_dir(search_results_dir / qseqid)

        # Known ligands
        if not known_q_collapsed.empty:
            cols_known = ["search_type"] + list(known_db.columns)
            cols_known = [c for c in cols_known if c in known_q_collapsed.columns]
            df_known_out = known_q_collapsed[cols_known].copy()
            known_path = q_dir / "known_ligands.tsv"
            df_known_out.to_csv(known_path, sep="\t", index=False)

        # ZINC ligands
        if not zinc_q_collapsed.empty:
            df_zinc_out = zinc_q_collapsed.copy()

            # Normalize ligand ID column name to 'chem_comp_id'
            ligand_col_out = "chem_comp_id"
            if zinc_ligand_col is not None and zinc_ligand_col != ligand_col_out:
                if (
                    zinc_ligand_col in df_zinc_out.columns
                    and ligand_col_out not in df_zinc_out.columns
                ):
                    df_zinc_out = df_zinc_out.rename(
                        columns={zinc_ligand_col: ligand_col_out}
                    )
            elif zinc_ligand_col is not None:
                ligand_col_out = zinc_ligand_col

            # Preferred column order for ZINC TSV
            preferred_order = [
                "search_type",
                "uniprot_id",
                ligand_col_out,
                "possible_binding_sites",
                "query_id",
                "tanimoto",
                "smiles",
            ]
            # Columns that exist, in that order
            cols_zinc = [c for c in preferred_order if c in df_zinc_out.columns]
            # Add any remaining columns at the end
            extra_cols = [c for c in df_zinc_out.columns if c not in cols_zinc]
            cols_zinc = cols_zinc + extra_cols

            df_zinc_out = df_zinc_out[cols_zinc]
            zinc_path = q_dir / "zinc_ligands.tsv"
            df_zinc_out.to_csv(zinc_path, sep="\t", index=False)

    # --- 5) Ligand counts (unique IDs) for summary ---
    # Known
    if known_q_collapsed.empty or known_ligand_col is None:
        n_known_seq = 0
        n_known_dom = 0
    else:
        df_known_seq = known_q_collapsed[known_q_collapsed["search_type"] == "sequence"]
        df_known_dom = known_q_collapsed[known_q_collapsed["search_type"] == "domain"]
        n_known_seq = df_known_seq[known_ligand_col].nunique()
        n_known_dom = df_known_dom[known_ligand_col].nunique()

    # ZINC
    if zinc_q_collapsed.empty or zinc_ligand_col is None:
        n_zinc_seq = 0
        n_zinc_dom = 0
    else:
        df_zinc_seq = zinc_q_collapsed[zinc_q_collapsed["search_type"] == "sequence"]
        df_zinc_dom = zinc_q_collapsed[zinc_q_collapsed["search_type"] == "domain"]
        n_zinc_seq = df_zinc_seq[zinc_ligand_col].nunique()
        n_zinc_dom = df_zinc_dom[zinc_ligand_col].nunique()

    return {
        "qseqid": qseqid,
        "n_proteins_sequence": n_prot_seq,
        "n_proteins_domain": n_prot_dom,
        "n_known_ligands_sequence": n_known_seq,
        "n_known_ligands_domain": n_known_dom,
        "n_zinc_ligands_sequence": n_zinc_seq,
        "n_zinc_ligands_domain": n_zinc_dom,
    }


def build_query_ligand_results(
    df_queries: pd.DataFrame,
    df_candidates_all: pd.DataFrame,
    known_db: pd.DataFrame,
    zinc_db: pd.DataFrame,
    output_dir: str | Path = "results",
    search_results_subdir: str = "search_results",
    save_per_query: bool = True,
    save_summary: bool = True,
) -> pd.DataFrame:
    """
    Block 3 (sequential version): join candidate proteins with known and
    ZINC ligands, generate per-query result files (with deduplicated ligands)
    and a global summary table.
    """
    output_dir = ensure_dir(output_dir)
    search_results_dir = ensure_dir(output_dir / search_results_subdir)

    # --- 1) Prepare candidate table, ensuring 'uniprot_id' column ---
    if df_candidates_all.empty:
        print(
            "[INFO] df_candidates_all is empty. Only summary (all zeros) "
            "will be returned."
        )
    else:
        expected_cols = {"qseqid", "sseqid", "search_type"}
        missing = expected_cols - set(df_candidates_all.columns)
        if missing:
            raise ValueError(
                f"df_candidates_all is missing columns: {missing}. "
                f"Expected at least: {expected_cols}"
            )

    if not df_candidates_all.empty:
        df_cand = df_candidates_all.copy()
        df_cand = df_cand.rename(columns={"sseqid": "uniprot_id"})
    else:
        df_cand = df_candidates_all

    # --- 2) Join with known and ZINC ligand databases ---
    # 2.1 known_db
    if not df_cand.empty:
        if "uniprot_id" not in known_db.columns:
            raise ValueError("known_db must contain 'uniprot_id' column.")

        known_merged = df_cand.merge(
            known_db,
            on="uniprot_id",
            how="inner",
        )
    else:
        known_merged = pd.DataFrame(
            columns=["qseqid", "search_type"] + list(known_db.columns)
        )

    # 2.2 zinc_db
    if not df_cand.empty:
        if "uniprot_id" not in zinc_db.columns:
            raise ValueError("zinc_db must contain 'uniprot_id' column.")

        zinc_merged = df_cand.merge(
            zinc_db,
            on="uniprot_id",
            how="inner",
        )
    else:
        zinc_merged = pd.DataFrame(
            columns=["qseqid", "search_type"] + list(zinc_db.columns)
        )

    # Ensure minimal columns
    for df_tmp in (known_merged, zinc_merged):
        if "qseqid" not in df_tmp.columns:
            df_tmp["qseqid"] = pd.Series(dtype=str)
        if "search_type" not in df_tmp.columns:
            df_tmp["search_type"] = pd.Series(dtype=str)

    # --- 3) Identify ligand ID columns ---
    # Known
    known_ligand_col = "chem_comp_id" if "chem_comp_id" in known_db.columns else None

    # ZINC: can be 'zinc_chem_comp_id' or 'chem_comp_id'
    if "zinc_chem_comp_id" in zinc_db.columns:
        zinc_ligand_col = "zinc_chem_comp_id"
    elif "chem_comp_id" in zinc_db.columns:
        zinc_ligand_col = "chem_comp_id"
    else:
        zinc_ligand_col = None

    # --- 4) Sequential loop over queries ---
    summary_rows: list[dict] = []
    for qseqid in df_queries["qseqid"]:
        summary_row = _process_single_query(
            qseqid=qseqid,
            df_candidates_all=df_candidates_all,
            known_merged=known_merged,
            zinc_merged=zinc_merged,
            known_db=known_db,
            zinc_db=zinc_db,
            known_ligand_col=known_ligand_col,
            zinc_ligand_col=zinc_ligand_col,
            search_results_dir=search_results_dir,
            save_per_query=save_per_query,
        )
        summary_rows.append(summary_row)

    summary_df = pd.DataFrame(summary_rows).sort_values("qseqid").reset_index(drop=True)

    if save_summary:
        summary_path = output_dir / "search_results_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"[INFO] Global summary saved to: {summary_path}")

    return summary_df


# ----------------------------------------------------------------------
# Parallel version of Block 3
# ----------------------------------------------------------------------

_GLOBAL_PARALLEL_STATE: dict[str, object] = {}


def _init_parallel_worker(
    df_candidates_all: pd.DataFrame,
    known_merged: pd.DataFrame,
    zinc_merged: pd.DataFrame,
    known_db: pd.DataFrame,
    zinc_db: pd.DataFrame,
    known_ligand_col: str | None,
    zinc_ligand_col: str | None,
    search_results_dir: str | Path,
    save_per_query: bool,
) -> None:
    """
    Initializer for worker processes: store large shared objects in a
    global dictionary so they are available to _process_single_query_parallel.
    """
    _GLOBAL_PARALLEL_STATE["df_candidates_all"] = df_candidates_all
    _GLOBAL_PARALLEL_STATE["known_merged"] = known_merged
    _GLOBAL_PARALLEL_STATE["zinc_merged"] = zinc_merged
    _GLOBAL_PARALLEL_STATE["known_db"] = known_db
    _GLOBAL_PARALLEL_STATE["zinc_db"] = zinc_db
    _GLOBAL_PARALLEL_STATE["known_ligand_col"] = known_ligand_col
    _GLOBAL_PARALLEL_STATE["zinc_ligand_col"] = zinc_ligand_col
    _GLOBAL_PARALLEL_STATE["search_results_dir"] = Path(search_results_dir)
    _GLOBAL_PARALLEL_STATE["save_per_query"] = save_per_query


def _process_single_query_parallel(qseqid: str) -> dict:
    """
    Wrapper that retrieves shared state from _GLOBAL_PARALLEL_STATE and
    delegates to _process_single_query.
    """
    st = _GLOBAL_PARALLEL_STATE
    return _process_single_query(
        qseqid=qseqid,
        df_candidates_all=st["df_candidates_all"],
        known_merged=st["known_merged"],
        zinc_merged=st["zinc_merged"],
        known_db=st["known_db"],
        zinc_db=st["zinc_db"],
        known_ligand_col=st["known_ligand_col"],
        zinc_ligand_col=st["zinc_ligand_col"],
        search_results_dir=st["search_results_dir"],
        save_per_query=st["save_per_query"],
    )


def build_query_ligand_results_parallel(
    df_queries: pd.DataFrame,
    df_candidates_all: pd.DataFrame,
    known_db: pd.DataFrame,
    zinc_db: pd.DataFrame,
    output_dir: str | Path = "results",
    search_results_subdir: str = "search_results",
    save_per_query: bool = True,
    save_summary: bool = True,
    njobs: int = 4,
) -> pd.DataFrame:
    """
    Parallel version of build_query_ligand_results.

    Large tables are prepared once in the parent process and then shared
    with worker processes via a global state dictionary. Each worker
    processes one or more queries independently.

    Parameters
    ----------
    df_queries : pd.DataFrame
        DataFrame with one column 'qseqid' listing all queries.
    df_candidates_all : pd.DataFrame
        Combined candidate table (sequence + domain), with columns:
          - qseqid
          - sseqid
          - search_type
    known_db : pd.DataFrame
        Known ligand binding table with at least 'uniprot_id'.
    zinc_db : pd.DataFrame
        ZINC hits table with at least 'uniprot_id'.
    output_dir : str or Path, default 'results'
        Base directory where result files are written.
    search_results_subdir : str, default 'search_results'
        Subdirectory for per-query results.
    save_per_query : bool, default True
        If True, write per-query known_ligands.tsv and zinc_ligands.tsv.
    save_summary : bool, default True
        If True, write search_results_summary.tsv.
    njobs : int, default 4
        Number of worker processes.

    Returns
    -------
    pd.DataFrame
        Summary table (one row per query) with counts of proteins and ligands.
    """
    if njobs is None or njobs < 1:
        njobs = 1

    output_dir = ensure_dir(output_dir)
    search_results_dir = ensure_dir(output_dir / search_results_subdir)

    # --- 1) Prepare candidate table, ensuring 'uniprot_id' column ---
    if df_candidates_all.empty:
        print(
            "[INFO] df_candidates_all is empty. Only summary (all zeros) "
            "will be returned."
        )
    else:
        expected_cols = {"qseqid", "sseqid", "search_type"}
        missing = expected_cols - set(df_candidates_all.columns)
        if missing:
            raise ValueError(
                f"df_candidates_all is missing columns: {missing}. "
                f"Expected at least: {expected_cols}"
            )

    if not df_candidates_all.empty:
        df_cand = df_candidates_all.copy()
        df_cand = df_cand.rename(columns={"sseqid": "uniprot_id"})
    else:
        df_cand = df_candidates_all

    # --- 2) Join with known and ZINC ligand databases (once) ---
    if not df_cand.empty:
        if "uniprot_id" not in known_db.columns:
            raise ValueError("known_db must contain 'uniprot_id' column.")
        known_merged = df_cand.merge(known_db, on="uniprot_id", how="inner")
    else:
        known_merged = pd.DataFrame(
            columns=["qseqid", "search_type"] + list(known_db.columns)
        )

    if not df_cand.empty:
        if "uniprot_id" not in zinc_db.columns:
            raise ValueError("zinc_db must contain 'uniprot_id' column.")
        zinc_merged = df_cand.merge(zinc_db, on="uniprot_id", how="inner")
    else:
        zinc_merged = pd.DataFrame(
            columns=["qseqid", "search_type"] + list(zinc_db.columns)
        )

    for df_tmp in (known_merged, zinc_merged):
        if "qseqid" not in df_tmp.columns:
            df_tmp["qseqid"] = pd.Series(dtype=str)
        if "search_type" not in df_tmp.columns:
            df_tmp["search_type"] = pd.Series(dtype=str)

    # --- 3) Identify ligand ID columns ---
    known_ligand_col = "chem_comp_id" if "chem_comp_id" in known_db.columns else None

    if "zinc_chem_comp_id" in zinc_db.columns:
        zinc_ligand_col = "zinc_chem_comp_id"
    elif "chem_comp_id" in zinc_db.columns:
        zinc_ligand_col = "chem_comp_id"
    else:
        zinc_ligand_col = None

    qseqids = list(df_queries["qseqid"])

    # If only one job or one query, fall back to sequential implementation
    if njobs == 1 or len(qseqids) <= 1:
        return build_query_ligand_results(
            df_queries=df_queries,
            df_candidates_all=df_candidates_all,
            known_db=known_db,
            zinc_db=zinc_db,
            output_dir=output_dir,
            search_results_subdir=search_results_subdir,
            save_per_query=save_per_query,
            save_summary=save_summary,
        )

    # --- 4) Parallel execution ---
    summary_rows: list[dict] = []
    with ProcessPoolExecutor(
        max_workers=njobs,
        initializer=_init_parallel_worker,
        initargs=(
            df_candidates_all,
            known_merged,
            zinc_merged,
            known_db,
            zinc_db,
            known_ligand_col,
            zinc_ligand_col,
            search_results_dir,
            save_per_query,
        ),
    ) as ex:
        futures = {ex.submit(_process_single_query_parallel, q): q for q in qseqids}
        for fut in as_completed(futures):
            res = fut.result()
            summary_rows.append(res)

    summary_df = pd.DataFrame(summary_rows).sort_values("qseqid").reset_index(drop=True)

    if save_summary:
        summary_path = output_dir / "search_results_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        print(f"[INFO] Global summary saved to: {summary_path}")

    return summary_df
