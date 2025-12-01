import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests


BASE_URL_ACCESSIONS = "https://rest.uniprot.org/uniprotkb/accessions"


def _parse_uniprot_multi_fasta(text: str) -> Dict[str, str]:
    """
    Parse a UniProt multi-FASTA text and return a dict:

        {uniprot_id -> sequence}

    Expected UniProt-style headers, for example:
        >sp|P12345|PROT_HUMAN ...
        >tr|Q8ABC1|PROT_MOUSE ...

    If the header does not follow the sp|ACC| pattern, the function
    falls back to using the first token after '>' as the key.
    """
    seqs: Dict[str, str] = {}

    current_id: str | None = None
    chunks: List[str] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith(">"):
            # Close previous sequence (if any)
            if current_id is not None and chunks:
                seqs[current_id] = "".join(chunks)

            header = line[1:]  # remove '>'
            parts = header.split("|")

            if len(parts) >= 3 and parts[0] in {"sp", "tr"}:
                # Typical UniProt header, e.g. sp|P12345|PROT_HUMAN
                current_id = parts[1]
            else:
                # Fallback: use the first whitespace-separated token
                current_id = header.split()[0]

            chunks = []
        else:
            chunks.append(line)

    # Last sequence
    if current_id is not None and chunks:
        seqs[current_id] = "".join(chunks)

    return seqs


def fetch_uniprot_fasta_by_accessions(
    ids: Iterable[str],
    chunk_size: int = 200,
    timeout: int = 60,
) -> Tuple[Dict[str, str | None], List[str]]:
    """
    Given an iterable of UniProt accessions, call the endpoint:

        /uniprotkb/accessions?accessions=ID1,ID2,...&format=fasta

    in chunks of size `chunk_size` and return:

      - results: dict {uniprot_id -> sequence (str) or None if not returned}
      - failed:  list of IDs that could not be retrieved

    This function does NOT use multithreading; it reduces the number
    of requests compared to calling /{acc}.fasta for each ID.
    """
    # Normalize and deduplicate IDs, preserving order
    seen = set()
    unique_ids: List[str] = []
    for x in ids:
        if x is None:
            continue
        acc = str(x).strip()
        if not acc or acc.lower() == "nan":
            continue
        if acc not in seen:
            seen.add(acc)
            unique_ids.append(acc)

    # Initialize result dict with all IDs set to None
    results: Dict[str, str | None] = {acc: None for acc in unique_ids}
    failed: List[str] = []

    # Helper to yield chunks
    def chunked(seq: List[str], size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    for chunk in chunked(unique_ids, chunk_size):
        acc_str = ",".join(chunk)

        params = {
            "accessions": acc_str,
            "format": "fasta",
        }

        try:
            resp = requests.get(
                BASE_URL_ACCESSIONS,
                params=params,
                timeout=timeout,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[WARN] Network error for chunk {chunk[0]}..{chunk[-1]}: {e}")
            # Mark the entire chunk as failed
            failed.extend(chunk)
            continue

        fasta_text = resp.text
        parsed = _parse_uniprot_multi_fasta(fasta_text)

        # IDs returned in this chunk
        returned = set(parsed.keys())

        # Fill results for this chunk
        for acc in chunk:
            if acc in parsed:
                results[acc] = parsed[acc]

        # Any ID from the chunk not present in the parsed dict is considered failed
        for acc in chunk:
            if acc not in returned:
                failed.append(acc)

    return results, failed


def retry_uniprot_failed_ids(
    results: Dict[str, str | None],
    failed: List[str],
    retry_chunk_size: int = 50,
    timeout: int = 60,
    max_retries: int = 5,
) -> Tuple[Dict[str, str | None], List[str]]:
    """
    Retry fetching sequences ONLY for the IDs in `failed`, calling
    `fetch_uniprot_fasta_by_accessions` with a smaller `chunk_size`
    (default 50), iteratively.

    In each iteration:
      * Calls `fetch_uniprot_fasta_by_accessions` on the failed IDs.
      * Updates `results` whenever new sequences are obtained.
      * Updates `failed` with IDs that are still missing.
    The process stops when:
      * The number of failed IDs does not decrease between iterations, or
      * `max_retries` is reached.

    Returns:
      - results: updated dict {uniprot_id -> sequence or None}
      - failed:  final list of IDs that could not be retrieved
    """
    # Normalize and deduplicate initial failed IDs
    failed = sorted(set(failed))

    if not failed:
        return results, failed

    attempt = 0
    while failed and attempt < max_retries:
        attempt += 1
        prev_failed = failed[:]  # copy for comparison

        print(
            f"[INFO] Retry {attempt}/{max_retries}: "
            f"{len(prev_failed)} IDs still failed (chunk_size={retry_chunk_size})"
        )

        # Retry only on the previously failed IDs
        retry_results, retry_failed = fetch_uniprot_fasta_by_accessions(
            prev_failed,
            chunk_size=retry_chunk_size,
            timeout=timeout,
        )

        # Count how many sequences were recovered in this round
        recovered_this_round = 0
        for acc, seq in retry_results.items():
            if seq is not None and results.get(acc) is None:
                results[acc] = seq
                recovered_this_round += 1

        # Update the list of failed IDs with this round's failures
        failed = sorted(set(retry_failed))

        print(
            f"[INFO] Retry {attempt}: recovered {recovered_this_round} IDs, "
            f"{len(failed)} IDs still failed."
        )

        # Stop condition: number of failed IDs did not change
        if len(failed) == len(prev_failed):
            print("[INFO] Number of failed IDs did not decrease. Stopping retries.")
            break

    return results, failed


def update_target_sequences_pickle(
    binding_data_merged: pd.DataFrame,
    output_dir: str | Path,
    uniprot_col: str = "uniprot_id",
    first_chunk_size: int = 200,
    retry_chunk_size: int = 50,
    timeout: int = 60,
    max_retries: int = 5,
) -> Dict[str, str | None]:
    """
    Update the sequence dictionary stored in `target_sequences.pkl`
    inside `output_dir`, using the unique UniProt protein IDs found in
    `binding_data_merged[uniprot_col]`.

    Logic:
      1. Load existing dictionary from `target_sequences.pkl` (if it does
         not exist, start from an empty dict).
      2. Get the unique UniProt IDs from `binding_data_merged`.
      3. Compute:
           - new_ids: present in the current binding data but NOT in the dict
                      -> need to be downloaded from UniProt.
           - obsolete_ids: present in the dict but NOT in the binding data
                           -> should be removed from the dict.
      4. Download sequences for `new_ids` using
         `fetch_uniprot_fasta_by_accessions`, then retry failures using
         `retry_uniprot_failed_ids`.
      5. Remove `obsolete_ids` from the dictionary.
      6. Save the updated dictionary back to `target_sequences.pkl`.
      7. Return the updated dictionary in memory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = output_dir / "target_sequences.pkl"

    # 1) Load existing dictionary, or start empty
    if pkl_path.exists():
        with pkl_path.open("rb") as f:
            seq_dict: Dict[str, str | None] = pickle.load(f)
    else:
        seq_dict = {}

    # 2) Get unique UniProt IDs from the binding data
    raw_ids = binding_data_merged[uniprot_col].unique()

    seen = set()
    current_ids: List[str] = []
    for x in raw_ids:
        if x is None:
            continue
        acc = str(x).strip()
        if not acc or acc.lower() == "nan":
            continue
        if acc not in seen:
            seen.add(acc)
            current_ids.append(acc)

    current_set = set(current_ids)
    existing_set = set(seq_dict.keys())

    # 3) Determine new and obsolete IDs
    new_ids = sorted(current_set - existing_set)
    obsolete_ids = sorted(existing_set - current_set)

    print(f"[INFO] Current binding UniProt IDs: {len(current_set)}")
    print(f"[INFO] IDs in previous dictionary: {len(existing_set)}")
    print(f"[INFO] New IDs to download: {len(new_ids)}")
    print(f"[INFO] Obsolete IDs to remove from dictionary: {len(obsolete_ids)}")

    # 4) Download sequences for new IDs, with retries
    if new_ids:
        # First pass
        new_results, failed = fetch_uniprot_fasta_by_accessions(
            new_ids,
            chunk_size=first_chunk_size,
            timeout=timeout,
        )

        # Merge initial results into seq_dict
        for acc, seq in new_results.items():
            seq_dict[acc] = seq

        # Retry on failed IDs
        if failed:
            seq_dict, failed = retry_uniprot_failed_ids(
                results=seq_dict,
                failed=failed,
                retry_chunk_size=retry_chunk_size,
                timeout=timeout,
                max_retries=max_retries,
            )

            if failed:
                print(
                    f"[WARN] After retries, there are still "
                    f"{len(failed)} IDs without sequences."
                )

    # 5) Remove obsolete IDs from the dictionary
    for acc in obsolete_ids:
        seq_dict.pop(acc, None)

    # Optional: small summary of None entries
    n_none = sum(1 for v in seq_dict.values() if v is None)
    print(f"[INFO] Total IDs in updated dictionary: {len(seq_dict)}")
    print(f"[INFO] Entries with None sequence: {n_none}")

    # 6) Save updated dictionary
    with pkl_path.open("wb") as f:
        pickle.dump(seq_dict, f)

    print(f"[INFO] Sequence dictionary saved to {pkl_path}")

    return seq_dict

def uniprot_dict_to_fasta(
    output_dir: str | Path,
    pkl_name: str = "target_sequences.pkl",
    fasta_name: str = "target_sequences.fasta",
) -> Path:
    """
    Load a UniProt sequence dictionary `{uniprot_id: sequence}` from a pickle file
    and export it to a FASTA file.

    Parameters
    ----------
    output_dir : str | Path
        Directory containing the pickle file.
    pkl_name : str, optional
        Name of the pickle file containing the dictionary.
    fasta_name : str, optional
        Name of the output FASTA file.

    Returns
    -------
    Path
        Path to the generated FASTA file.
    """
    output_dir = Path(output_dir)
    pkl_path = output_dir / pkl_name
    fasta_path = output_dir / fasta_name

    if not pkl_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {pkl_path}")

    # Load the dictionary
    with pkl_path.open("rb") as f:
        seq_dict: Dict[str, Optional[str]] = pickle.load(f)

    # Write FASTA
    with fasta_path.open("w") as f_out:
        for uniprot_id, seq in seq_dict.items():
            if seq is None:
                # You can log or skip silently
                # print(f"[WARN] Missing sequence for {uniprot_id}, skipping.")
                continue

            # Standard FASTA header: >uniprot_id
            f_out.write(f">{uniprot_id}\n")

            # Wrap sequence in 60-char lines (FASTA convention)
            for i in range(0, len(seq), 60):
                f_out.write(seq[i:i+60] + "\n")

    print(f"[INFO] FASTA written to {fasta_path}")
    return fasta_path