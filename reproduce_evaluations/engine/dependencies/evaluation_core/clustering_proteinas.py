import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd


# ----------------------------
# FASTA utilities (without Biopython)
# ----------------------------

def parse_fasta_to_dict(fasta_path: str) -> Dict[str, str]:
    """
    Read a FASTA file and return a mapping: {seq_id: sequence}.
    Use the first header token as the ID (before the first space).
    """
    seqs: Dict[str, str] = {}
    current_id = None
    chunks: List[str] = []
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(chunks)
                header = line[1:].strip()
                current_id = header.split()[0]
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            seqs[current_id] = "".join(chunks)
    return seqs


def write_fasta(seqs: Dict[str, str], out_fasta: str) -> None:
    with open(out_fasta, "w") as f:
        for sid, s in seqs.items():
            f.write(f">{sid}\n")
            for i in range(0, len(s), 60):
                f.write(s[i:i + 60] + "\n")


# ----------------------------
# BLAST utils
# ----------------------------

def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            f"  {' '.join(cmd)}\n\n"
            f"STDOUT:\n{p.stdout}\n\n"
            f"STDERR:\n{p.stderr}\n"
        )


def ensure_blast_db(fasta_path: str, db_prefix: str) -> str:
    fasta_path = str(Path(fasta_path))
    db_prefix = str(Path(db_prefix))

    # BLAST creates .pin/.psq/.phr files for proteins
    if all(Path(db_prefix + ext).exists() for ext in [".pin", ".psq", ".phr"]):
        return db_prefix

    _run(["makeblastdb", "-in", fasta_path, "-dbtype", "prot", "-out", db_prefix])
    return db_prefix


def run_blastp_tabular(
    query_fasta: str,
    db_prefix: str,
    max_targets: int = 5000,
    num_threads: int = 8,
) -> pd.DataFrame:
    outfmt_fields = [
        "qseqid", "sseqid",
        "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send",
        "evalue", "bitscore",
        "qlen", "slen",
        "qcovs",
    ]
    outfmt = "6 " + " ".join(outfmt_fields)

    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False) as tmp:
        out_path = tmp.name

    try:
        _run([
            "blastp",
            "-query", query_fasta,
            "-db", db_prefix,
            "-out", out_path,
            "-outfmt", outfmt,
            "-max_target_seqs", str(max_targets),
            "-num_threads", str(num_threads),
        ])
        df = pd.read_csv(out_path, sep="\t", header=None, names=outfmt_fields)
        return df
    finally:
        try:
            os.remove(out_path)
        except OSError:
            pass


# ----------------------------
# Seed selection logic
# ----------------------------

@dataclass
class BlastSeedConfig:
    n_neighbors: int = 10
    max_targets_per_query: int = 5000
    num_threads: int = 8

    # Reasonable-score filters
    max_evalue: float = 1e-5
    min_bitscore: float = 60.0
    min_pident: float = 25.0
    min_qcovs: float = 50.0
    min_align_len: int = 0  # 0 = no filtra


def build_exclusion_set(target_to_uniprot: Dict[str, Sequence[str]]) -> Set[str]:
    excl: Set[str] = set()
    for ids in target_to_uniprot.values():
        excl.update(ids)
    return excl


def pick_representative_longest(
    uniprot_ids: Sequence[str],
    seqs: Dict[str, str],
) -> str:
    """
    Choose the protein with the longest sequence among the supplied IDs.
    - IDs that are absent from seqs are ignored by the length criterion.
    - Raise an error if none of the IDs are present.
    """
    available = [(uid, len(seqs[uid])) for uid in uniprot_ids if uid in seqs]
    if not available:
        raise ValueError(f"None of the provided Uniprot IDs are present in the FASTA DB: {list(uniprot_ids)}")
    # Deterministic tie-break: descending length, then ascending UID
    available.sort(key=lambda x: (-x[1], x[0]))
    return available[0][0]


def select_top_neighbors_by_blast(
    target_name: str,
    target_uniprot_ids: Sequence[str],
    fasta_db_path: str,
    db_prefix: str,
    target_to_uniprot: Dict[str, Sequence[str]],
    config: Optional[BlastSeedConfig] = None,
) -> Tuple[List[str], pd.DataFrame, str]:
    """
    MODIFIED:
      - Use ONLY one query: the representative (longest sequence) among target_uniprot_ids.
      - Still exclude ALL target-associated IDs (including those for the current target) from subjects.

    Return:
      - top_ids: N most similar proteins (UniProt IDs), excluding target-associated IDs
      - hits_df: table of the best hits aggregated by sseqid
      - representative_query_id: UniProt ID selected as the representative (the longest)
    """
    if config is None:
        config = BlastSeedConfig()

    # 1) Load sequences
    seqs = parse_fasta_to_dict(fasta_db_path)

    # 2) Choose the representative (longest) and build a query FASTA with ONLY that sequence
    rep_id = pick_representative_longest(target_uniprot_ids, seqs)

    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as tmp:
        query_fasta = tmp.name
    try:
        write_fasta({rep_id: seqs[rep_id]}, query_fasta)

        # 3) Ensure the BLAST database exists
        ensure_blast_db(fasta_db_path, db_prefix)

        # 4) Run BLAST
        raw = run_blastp_tabular(
            query_fasta=query_fasta,
            db_prefix=db_prefix,
            max_targets=config.max_targets_per_query,
            num_threads=config.num_threads,
        )
    finally:
        try:
            os.remove(query_fasta)
        except OSError:
            pass

    if raw.empty:
        return [], pd.DataFrame(), rep_id

    # 5) Exclude ALL target-associated IDs and ALL IDs for this target (even non-query IDs)
    exclude = build_exclusion_set(target_to_uniprot)
    exclude.update(target_uniprot_ids)

    # 6) Quality filters
    df = raw.copy()
    for c in ["pident", "length", "evalue", "bitscore", "qcovs", "qlen", "slen"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["evalue", "bitscore", "pident", "qcovs", "length"])

    df = df[df["evalue"] <= config.max_evalue]
    df = df[df["bitscore"] >= config.min_bitscore]
    df = df[df["pident"] >= config.min_pident]
    df = df[df["qcovs"] >= config.min_qcovs]
    if config.min_align_len > 0:
        df = df[df["length"] >= config.min_align_len]

    # 7) Exclude target-associated IDs (and the target itself)
    df = df[~df["sseqid"].isin(exclude)]

    if df.empty:
        return [], pd.DataFrame(), rep_id

    # 8) With one query, multi-query aggregation is unnecessary, but retain
    #    "best per sseqid" for robustness if BLAST returns multiple HSPs
    df_sorted = df.sort_values(["sseqid", "bitscore", "pident", "qcovs"], ascending=[True, False, False, False])
    best = df_sorted.groupby("sseqid", as_index=False).first()

    # 9) Final ranking
    best = best.sort_values(["bitscore", "pident", "qcovs", "evalue"], ascending=[False, False, False, True]).reset_index(drop=True)

    top = best.head(config.n_neighbors)
    top_ids = top["sseqid"].tolist()

    return top_ids, best, rep_id



def build_neighbor_rankings_for_all_targets(
    target_to_uniprot: Dict[str, Sequence[str]],
    fasta_db_path: str,
    db_prefix: str,
    config: Optional[BlastSeedConfig] = None,
    keep_only_top_n: Optional[int] = None,  # None => save the complete filtered ranking
    verbose: bool = True,
) -> Tuple[
    Dict[str, List[str]],        # neighbor_ranked_ids_by_target
    Dict[str, str],              # representative query id by target
    Dict[str, pd.DataFrame],     # hits_by_target (ranked table)
    pd.DataFrame,                # consolidated all_hits_long table
]:
    """
    Iterate over all targets and obtain, for each one:
      - ranked list of nearby proteins (UniProt IDs)
      - representative query used
      - filtered/ranked BLAST hit table

    keep_only_top_n:
      - None: save the entire filtered ranking (complete hits_df)
      - int: truncate both the list and table to the top N
    """
    if config is None:
        config = BlastSeedConfig()

    # Create/check the BLAST database once (rather than once per target)
    ensure_blast_db(fasta_db_path, db_prefix)

    neighbor_ranked_ids_by_target: Dict[str, List[str]] = {}
    rep_query_by_target: Dict[str, str] = {}
    hits_by_target: Dict[str, pd.DataFrame] = {}
    all_rows = []

    targets = list(target_to_uniprot.keys())

    for i, target_name in enumerate(targets, 1):
        if verbose:
            print(f"[{i}/{len(targets)}] Target: {target_name}")

        target_uniprot_ids = target_to_uniprot[target_name]

        try:
            top_ids, hits_df, rep_id = select_top_neighbors_by_blast(
                target_name=target_name,
                target_uniprot_ids=target_uniprot_ids,
                fasta_db_path=fasta_db_path,
                db_prefix=db_prefix,
                target_to_uniprot=target_to_uniprot,
                config=config,
            )
        except Exception as e:
            # Log and continue when desired
            if verbose:
                print(f"  ERROR for {target_name}: {e}")
            neighbor_ranked_ids_by_target[target_name] = []
            rep_query_by_target[target_name] = None
            hits_by_target[target_name] = pd.DataFrame()
            continue

        rep_query_by_target[target_name] = rep_id

        # hits_df is already filtered and ranked (best hit per sseqid)
        if hits_df is None or hits_df.empty:
            neighbor_ranked_ids_by_target[target_name] = []
            hits_by_target[target_name] = pd.DataFrame()
            if verbose:
                print("  No hits remained after filtering.")
            continue

        # Complete ranking by sseqid (or truncated if requested)
        ranked_ids = hits_df["sseqid"].tolist()

        if keep_only_top_n is not None:
            ranked_ids = ranked_ids[:keep_only_top_n]
            hits_df_use = hits_df.head(keep_only_top_n).copy()
        else:
            hits_df_use = hits_df.copy()

        neighbor_ranked_ids_by_target[target_name] = ranked_ids
        hits_by_target[target_name] = hits_df_use

        # Long-form consolidated table with useful metadata
        tmp = hits_df_use.copy().reset_index(drop=True)
        tmp.insert(0, "neighbor_rank", range(1, len(tmp) + 1))
        tmp.insert(0, "representative_query_id", rep_id)
        tmp.insert(0, "target_name", target_name)

        # Optionally also save the target's original IDs (joined as a string)
        tmp["target_uniprot_ids"] = ",".join(map(str, target_uniprot_ids))

        all_rows.append(tmp)

        if verbose:
            print(f"  Representative query: {rep_id}")
            print(f"  #neighbors (saved ranking): {len(ranked_ids)}")
            if len(ranked_ids) > 0:
                print(f"  Top 5: {ranked_ids[:5]}")

    all_hits_long = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()

    return neighbor_ranked_ids_by_target, rep_query_by_target, hits_by_target, all_hits_long
