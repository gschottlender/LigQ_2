from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Mapping, Optional, List, Dict, Tuple, Union

import numpy as np
import pandas as pd
import torch

from compound_processing.compound_helpers import (
    LigandStore,
    Representation,
    build_morgan_representation,
    build_ligand_index,
)
from compound_processing import backend, metrics


def promote_representatives_by_pchembl(
    representatives: List[str],
    clusters: Dict[str, List[str]],
    assignment: Dict[str, str],
    pchembl_map: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """
    Given a clustering (representatives, clusters, assignment) and a mapping
    chem_comp_id -> pchembl, replace each cluster representative with the ligand
    that has the highest pchembl within that cluster (if available).

    If no member of a cluster has a valid pchembl, the original representative
    is kept.

    Parameters
    ----------
    representatives : list of str
        Current cluster representatives.
    clusters : dict
        Mapping rep_id -> list of cluster members.
    assignment : dict
        Mapping lig_id -> rep_id.
    pchembl_map : mapping, optional
        Mapping chem_comp_id -> pchembl (float). Can be a dict or a pandas
        Series converted with .to_dict().

    Returns
    -------
    new_reps : list of str
    new_clusters : dict
    new_assignment : dict
    """
    if pchembl_map is None:
        return representatives, clusters, assignment

    new_reps: List[str] = []
    new_clusters: Dict[str, List[str]] = {}
    new_assignment: Dict[str, str] = {}

    for old_rep in representatives:
        members = clusters[old_rep]

        # Look for the member with highest pchembl
        best_rep = old_rep
        best_val = -math.inf
        any_valid = False

        for lig_id in members:
            val = pchembl_map.get(lig_id, None)
            if val is None:
                continue

            # Skip NaN values
            try:
                if math.isnan(val):
                    continue
            except TypeError:
                # In case val is not a float
                pass

            if (not any_valid) or (val > best_val):
                best_val = val
                best_rep = lig_id
                any_valid = True

        # Use best_rep as the new cluster key
        new_reps.append(best_rep)
        new_clusters[best_rep] = members

        for lig_id in members:
            new_assignment[lig_id] = best_rep

    return new_reps, new_clusters, new_assignment


def cluster_ligands_by_tanimoto(
    chem_comp_ids: Sequence[str],
    rep: Representation,
    threshold: float = 0.75,
    pchembl_map: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """
    Cluster ligands by Tanimoto similarity using a simple greedy strategy.

    If pchembl_map is provided, after clustering, each cluster representative
    is replaced by the member with the highest pchembl (when available).

    Parameters
    ----------
    chem_comp_ids : sequence of str
        Ligand identifiers to cluster.
    rep : Representation
        Representation object with .get_by_ids(ids, as_float=False) returning
        binary fingerprints.
    threshold : float, default 0.75
        Minimum Tanimoto similarity to assign a ligand to an existing cluster.
    pchembl_map : mapping, optional
        Mapping chem_comp_id -> pchembl used to promote high-pchembl members
        to representatives.

    Returns
    -------
    representatives : list of str
        Final cluster representatives.
    clusters : dict
        Mapping rep_id -> list of cluster members.
    assignment : dict
        Mapping lig_id -> rep_id.
    """
    # Filter duplicates while preserving order
    seen = set()
    unique_ids: List[str] = []
    for cid in chem_comp_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)

    if not unique_ids:
        return [], {}, {}

    # Get binary fingerprints (0/1) for all ligands
    fps = rep.get_by_ids(unique_ids, as_float=False)  # shape: (N, dim), uint8 0/1
    bit_counts = fps.sum(axis=1).astype(np.int32)      # shape: (N,)

    representatives_idx: List[int] = []
    representatives: List[str] = []
    clusters: Dict[str, List[str]] = {}
    assignment: Dict[str, str] = {}

    for i, lig_id in enumerate(unique_ids):
        fp_i = fps[i]
        bits_i = bit_counts[i]

        if not representatives_idx:
            # First ligand → first cluster
            representatives_idx.append(i)
            rep_id = lig_id
            representatives.append(rep_id)
            clusters[rep_id] = [lig_id]
            assignment[lig_id] = rep_id
            continue

        reps_idx_arr = np.array(representatives_idx, dtype=np.int32)
        reps_fps = fps[reps_idx_arr]           # (R, dim)
        reps_bits = bit_counts[reps_idx_arr]   # (R,)

        inter = np.bitwise_and(reps_fps, fp_i).sum(axis=1)  # (R,)
        denom = reps_bits + bits_i - inter                  # (R,)
        valid = denom > 0
        ti = np.zeros_like(denom, dtype=float)
        ti[valid] = inter[valid] / denom[valid]

        hits = np.where(ti >= threshold)[0]

        if hits.size > 0:
            rep_idx_global = reps_idx_arr[hits[0]]
            rep_id = unique_ids[rep_idx_global]
            clusters[rep_id].append(lig_id)
            assignment[lig_id] = rep_id
        else:
            representatives_idx.append(i)
            rep_id = lig_id
            representatives.append(rep_id)
            clusters[rep_id] = [lig_id]
            assignment[lig_id] = rep_id

    # Post-process representatives according to pchembl, if provided
    representatives, clusters, assignment = promote_representatives_by_pchembl(
        representatives=representatives,
        clusters=clusters,
        assignment=assignment,
        pchembl_map=pchembl_map,
    )

    return representatives, clusters, assignment


def select_ligands_maxmin_by_tanimoto(
    chem_comp_ids: Sequence[str],
    rep: Representation,
    n_select: int = 50,
    pchembl_map: Optional[Mapping[str, float]] = None,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, str]]:
    """
    MaxMin selection of ligands by Tanimoto distance.

    After building the MaxMin set and assigning each ligand to its closest
    representative, if pchembl_map is provided, the representative of each
    cluster is replaced by the member with the highest pchembl (when available).

    Parameters
    ----------
    chem_comp_ids : sequence of str
        Ligand identifiers to select from.
    rep : Representation
        Representation object with .get_by_ids(ids, as_float=False) returning
        binary fingerprints.
    n_select : int, default 50
        Maximum number of representatives to select.
    pchembl_map : mapping, optional
        Mapping chem_comp_id -> pchembl used to promote high-pchembl members
        to representatives.

    Returns
    -------
    representatives : list of str
        Final representatives (possibly promoted by pchembl).
    clusters : dict
        Mapping rep_id -> list of cluster members.
    assignment : dict
        Mapping lig_id -> rep_id.
    """
    # Filter duplicates while preserving order
    seen = set()
    unique_ids: List[str] = []
    for cid in chem_comp_ids:
        if cid not in seen:
            seen.add(cid)
            unique_ids.append(cid)

    if not unique_ids:
        return [], {}, {}

    fps = rep.get_by_ids(unique_ids, as_float=False)   # (N, dim)
    if fps.shape[0] != len(unique_ids):
        raise ValueError("Number of fingerprints does not match unique_ids length.")

    bit_counts = fps.sum(axis=1).astype(np.int32)
    N = fps.shape[0]
    n_select_eff = min(n_select, N)

    # Initialize with the first ligand
    selected_idx: List[int] = [0]
    is_selected = np.zeros(N, dtype=bool)
    is_selected[0] = True

    fp0 = fps[0]
    bits0 = bit_counts[0]

    inter0 = np.bitwise_and(fps, fp0).sum(axis=1)
    denom0 = bit_counts + bits0 - inter0
    valid0 = denom0 > 0
    ti0 = np.zeros_like(denom0, dtype=float)
    ti0[valid0] = inter0[valid0] / denom0[valid0]
    dist0 = 1.0 - ti0

    min_dist = dist0.copy()
    min_dist[0] = 0.0

    # Greedy MaxMin loop
    while len(selected_idx) < n_select_eff:
        candidates = np.where(~is_selected)[0]
        if candidates.size == 0:
            break

        best_local = int(np.argmax(min_dist[candidates]))
        best_idx = int(candidates[best_local])

        selected_idx.append(best_idx)
        is_selected[best_idx] = True

        fp_j = fps[best_idx]
        bits_j = bit_counts[best_idx]

        inter = np.bitwise_and(fps, fp_j).sum(axis=1)
        denom = bit_counts + bits_j - inter
        valid = denom > 0
        ti_j = np.zeros_like(denom, dtype=float)
        ti_j[valid] = inter[valid] / denom[valid]
        dist_j = 1.0 - ti_j

        min_dist = np.minimum(min_dist, dist_j)

    representatives_idx = selected_idx
    representatives: List[str] = [unique_ids[i] for i in representatives_idx]

    # Assign each ligand to its closest representative
    reps_idx_arr = np.array(representatives_idx, dtype=np.int32)
    reps_fps = fps[reps_idx_arr]
    reps_bits = bit_counts[reps_idx_arr]

    clusters: Dict[str, List[str]] = {rep_id: [] for rep_id in representatives}
    assignment: Dict[str, str] = {}

    for i, lig_id in enumerate(unique_ids):
        fp_i = fps[i]
        bits_i = bit_counts[i]

        inter = np.bitwise_and(reps_fps, fp_i).sum(axis=1)
        denom = reps_bits + bits_i - inter
        valid = denom > 0
        ti = np.zeros_like(denom, dtype=float)
        ti[valid] = inter[valid] / denom[valid]

        best_rep_local = int(np.argmax(ti))
        rep_idx_global = reps_idx_arr[best_rep_local]
        rep_id = unique_ids[rep_idx_global]

        clusters[rep_id].append(lig_id)
        assignment[lig_id] = rep_id

    # Post-process representatives according to pchembl
    representatives, clusters, assignment = promote_representatives_by_pchembl(
        representatives=representatives,
        clusters=clusters,
        assignment=assignment,
        pchembl_map=pchembl_map,
    )

    return representatives, clusters, assignment


def max_pairwise_tanimoto_from_ids(
    representatives: Sequence[str],
    rep: Representation,
) -> float:
    """
    Compute the maximum pairwise Tanimoto similarity among the given ligands,
    using the binary representation from `rep`.

    Parameters
    ----------
    representatives : list of str
        List of chem_comp_id acting as representatives.
    rep : Representation
        Representation object that provides fingerprints via
        rep.get_by_ids(ids, as_float=False).

    Returns
    -------
    float
        Maximum Tanimoto coefficient among all pairs of representatives.
    """
    if len(representatives) < 2:
        return 0.0

    # Obtain binary fingerprints (uint8 0/1)
    fps = rep.get_by_ids(representatives, as_float=False)   # shape: (N, dim)
    N = fps.shape[0]
    bit_counts = fps.sum(axis=1).astype(np.int32)

    max_ti = 0.0

    # Compare all pairs (OK because we have at most ~50 reps)
    for i in range(N):
        fp_i = fps[i]
        bits_i = bit_counts[i]
        for j in range(i + 1, N):
            fp_j = fps[j]
            bits_j = bit_counts[j]

            inter = np.bitwise_and(fp_i, fp_j).sum()
            denom = bits_i + bits_j - inter
            if denom > 0:
                ti = inter / denom
            else:
                ti = 0.0

            if ti > max_ti:
                max_ti = ti

    return float(max_ti)


# ---------------------------------------------------------------------
# Global helpers for multiprocessing workers (lazy initialization)
# ---------------------------------------------------------------------

_ZINC_MEMMAP: Optional[np.memmap] = None
_ZINC_DIM: Optional[int] = None


def _unpack_bits(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Unpack bits from uint8 array of shape (N, n_bytes) to (N, n_bits) with 0/1
    values. Uses np.unpackbits and then truncates to n_bits as a safety check.
    """
    arr = np.unpackbits(packed, axis=1)
    if arr.shape[1] > n_bits:
        arr = arr[:, :n_bits]
    return arr


def _ensure_zinc_memmap(
    memmap_path: str,
    n_ligands: int,
    packed_dim: int,
    dtype_str: str,
    dim: int,
) -> None:
    """
    Initialize the ZINC memmap in the worker process if it is not already
    initialized. Stores the memmap and dimensionality in module-level globals.
    """
    global _ZINC_MEMMAP, _ZINC_DIM

    if _ZINC_MEMMAP is not None:
        # Already initialized in this process
        return

    dtype = np.dtype(dtype_str)
    _ZINC_MEMMAP = np.memmap(
        memmap_path,
        mode="r",
        dtype=dtype,
        shape=(n_ligands, packed_dim),
    )
    _ZINC_DIM = dim


def _search_zinc_chunk(
    memmap_path: str,
    n_ligands: int,
    packed_dim: int,
    dtype_str: str,
    dim: int,
    chunk_start: int,
    chunk_end: int,
    batch_query_ids: Sequence[str],
    batch_fps: np.ndarray,
    batch_bitcounts: np.ndarray,
    tanimoto_threshold: float,
) -> List[Tuple[str, int, float]]:
    """
    Worker function that:
      - Ensures the ZINC memmap is initialized in this process.
      - Reads a range [chunk_start:chunk_end) of ZINC indices.
      - Unpacks fingerprints for that chunk.
      - Computes Tanimoto similarity against all query fingerprints in the batch.
      - Returns a list of (query_id, lig_idx_zinc, tanimoto) for hits above
        the given threshold.
    """
    global _ZINC_MEMMAP, _ZINC_DIM

    # Lazy initialization of the memmap in this worker process
    _ensure_zinc_memmap(
        memmap_path=memmap_path,
        n_ligands=n_ligands,
        packed_dim=packed_dim,
        dtype_str=dtype_str,
        dim=dim,
    )

    if _ZINC_MEMMAP is None or _ZINC_DIM is None:
        raise RuntimeError("ZINC memmap could not be initialized in worker.")

    # 1) Read packed block
    packed_block = _ZINC_MEMMAP[chunk_start:chunk_end]   # (B, packed_dim)
    if packed_block.size == 0:
        return []

    # 2) Unpack to 0/1 bits
    block_bits = _unpack_bits(packed_block, _ZINC_DIM)   # (B, dim)
    block_bitcounts = block_bits.sum(axis=1).astype(np.int32)  # (B,)

    # 3) Compute intersections as dot products (bitwise AND implicit with 0/1)
    # batch_fps: (Q, dim)  with 0/1
    # block_bits: (B, dim) with 0/1
    # inter: (Q, B) number of shared bits
    inter = batch_fps @ block_bits.T   # np.dot with uint8 → intersections

    # 4) Compute Tanimoto denominator
    denom = batch_bitcounts[:, None] + block_bitcounts[None, :] - inter  # (Q, B)

    # Avoid division by zero
    valid = denom > 0
    ti = np.zeros_like(inter, dtype=np.float32)
    ti[valid] = inter[valid] / denom[valid]

    # 5) Apply threshold
    mask = ti >= tanimoto_threshold
    q_idx, b_idx = np.where(mask)

    hits: List[Tuple[str, int, float]] = []
    if len(q_idx) == 0:
        return hits

    for k in range(len(q_idx)):
        qi = q_idx[k]
        bi = b_idx[k]
        lig_idx_global = chunk_start + bi
        hits.append((batch_query_ids[qi], int(lig_idx_global), float(ti[qi, bi])))

    return hits

# GPU version of zinc search functions

def search_similar_in_zinc(
    query_ids: Sequence[str],
    store_ref: LigandStore,
    rep_ref: Representation,
    store_zinc: LigandStore,
    rep_zinc: Representation,
    tanimoto_threshold: float = 0.5,
    q_batch_size: int = 50,
    zinc_chunk_size: int = 200_000,
    n_jobs: int = 4,
    max_hits_per_query: Optional[int] = None,
) -> pd.DataFrame:
    """
    Search for similar compounds in the ZINC database for a set of ligands
    (query_ids) defined in the reference database (store_ref / rep_ref).

    Strategy
    --------
      - Pre-compute 0/1 fingerprints for all query_ids with rep_ref.
      - Iterate over ZINC in chunks (zinc_chunk_size) from rep_zinc.memmap.
      - For each batch of queries (q_batch_size), spawn N workers:
            each worker processes one ZINC chunk via _search_zinc_chunk.
      - Collect (query_id, lig_idx_zinc, Tanimoto) pairs above
        tanimoto_threshold.
      - Map lig_idx_zinc -> chem_comp_id, smiles using store_zinc.

    Parameters
    ----------
    query_ids : sequence of str
        Ligand identifiers to be used as queries.
    store_ref : LigandStore
        Reference ligand store (PDB+ChEMBL).
    rep_ref : Representation
        Representation for reference ligands.
    store_zinc : LigandStore
        Ligand store for ZINC.
    rep_zinc : Representation
        Representation for ZINC ligands, using a packed memmap.
    tanimoto_threshold : float, default 0.5
        Minimum Tanimoto similarity to report a hit.
    q_batch_size : int, default 50
        Number of queries in each batch.
    zinc_chunk_size : int, default 200_000
        Number of ZINC ligands per chunk.
    n_jobs : int, default 4
        Number of worker processes to use. If 1, a sequential path is used.
    max_hits_per_query : int, optional
        If not None, limit the number of hits per query to this value
        (highest Tanimoto first).

    Returns
    -------
    pd.DataFrame
        Columns:
          - query_id
          - lig_idx_zinc
          - chem_comp_id
          - smiles
          - tanimoto
    """
    if not query_ids:
        return pd.DataFrame(
            columns=["query_id", "lig_idx_zinc", "chem_comp_id", "smiles", "tanimoto"]
        )

    # ------------------------------------------------------------------
    # 1) Pre-compute query fingerprints (0/1)
    # ------------------------------------------------------------------
    query_ids = list(query_ids)
    fps_queries = rep_ref.get_by_ids(query_ids, as_float=False)  # (Q, dim)
    if fps_queries.shape[0] != len(query_ids):
        raise ValueError(
            "Number of fingerprints does not match number of query_ids."
        )

    bitcounts_queries = fps_queries.sum(axis=1).astype(np.int32)  # (Q,)
    dim = rep_ref.dim

    # ------------------------------------------------------------------
    # 2) Memmap info for ZINC (to pass into workers)
    # ------------------------------------------------------------------
    memmap = rep_zinc.memmap
    n_ligands_zinc = memmap.shape[0]
    packed_dim = memmap.shape[1]
    dtype_str = str(memmap.dtype)
    memmap_path = memmap.filename  # path to underlying file
    if memmap_path is None:
        raise RuntimeError(
            "ZINC memmap does not have a filename; recreate it with an explicit path."
        )

    # ------------------------------------------------------------------
    # 3) Define ZINC chunks
    # ------------------------------------------------------------------
    chunk_ranges: List[Tuple[int, int]] = []
    for start in range(0, n_ligands_zinc, zinc_chunk_size):
        end = min(start + zinc_chunk_size, n_ligands_zinc)
        chunk_ranges.append((start, end))

    # ------------------------------------------------------------------
    # 4) Prepare ZINC ligands indexed by lig_idx
    # ------------------------------------------------------------------
    ligs_zinc = store_zinc.ligands.copy()
    if "lig_idx" not in ligs_zinc.columns:
        raise ValueError("ZINC ligands parquet must contain column 'lig_idx'.")

    ligs_zinc_by_idx = ligs_zinc.set_index("lig_idx")

    # ------------------------------------------------------------------
    # 5) Loop over query batches + multiprocessing over ZINC chunks
    # ------------------------------------------------------------------
    all_hits: List[Tuple[str, int, float]] = []

    # Sequential path when n_jobs == 1
    if n_jobs == 1:
        for q_start in range(0, len(query_ids), q_batch_size):
            q_end = min(q_start + q_batch_size, len(query_ids))
            batch_ids = query_ids[q_start:q_end]
            batch_fps = fps_queries[q_start:q_end]
            batch_bitcounts = bitcounts_queries[q_start:q_end]

            for (cs, ce) in chunk_ranges:
                hits_chunk = _search_zinc_chunk(
                    memmap_path,
                    n_ligands_zinc,
                    packed_dim,
                    dtype_str,
                    dim,
                    cs,
                    ce,
                    batch_ids,
                    batch_fps,
                    batch_bitcounts,
                    tanimoto_threshold,
                )
                if hits_chunk:
                    all_hits.extend(hits_chunk)

    else:
        # Parallel path using 'fork' context (avoids pickling issues in notebooks)
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=n_jobs, mp_context=ctx) as executor:
            n_queries = len(query_ids)
            for q_start in range(0, n_queries, q_batch_size):
                q_end = min(q_start + q_batch_size, n_queries)
                batch_ids = query_ids[q_start:q_end]
                batch_fps = fps_queries[q_start:q_end]              # (Qb, dim)
                batch_bitcounts = bitcounts_queries[q_start:q_end]  # (Qb,)

                futures = []
                for (cs, ce) in chunk_ranges:
                    futures.append(
                        executor.submit(
                            _search_zinc_chunk,
                            memmap_path,
                            n_ligands_zinc,
                            packed_dim,
                            dtype_str,
                            dim,
                            cs,
                            ce,
                            batch_ids,
                            batch_fps,
                            batch_bitcounts,
                            tanimoto_threshold,
                        )
                    )

                for fut in as_completed(futures):
                    hits_chunk = fut.result()
                    if hits_chunk:
                        all_hits.extend(hits_chunk)

    # ------------------------------------------------------------------
    # 6) Convert hits to DataFrame and map chem_comp_id / smiles
    # ------------------------------------------------------------------
    if not all_hits:
        return pd.DataFrame(
            columns=["query_id", "lig_idx_zinc", "chem_comp_id", "smiles", "tanimoto"]
        )

    hits_df = pd.DataFrame(all_hits, columns=["query_id", "lig_idx_zinc", "tanimoto"])

    hits_df = hits_df.merge(
        ligs_zinc_by_idx[["chem_comp_id", "smiles"]],
        left_on="lig_idx_zinc",
        right_index=True,
        how="left",
    )

    # ------------------------------------------------------------------
    # 7) Limit number of hits per query if requested
    # ------------------------------------------------------------------
    if max_hits_per_query is not None:
        hits_df = (
            hits_df.sort_values(["query_id", "tanimoto"], ascending=[True, False])
            .groupby("query_id", as_index=False)
            .head(max_hits_per_query)
            .reset_index(drop=True)
        )

    return hits_df[["query_id", "lig_idx_zinc", "chem_comp_id", "smiles", "tanimoto"]]

# GPU Version of Zinc ligand searches


# =============================================================================
# Bit-unpacking helpers (packed uint8 -> dense 0/1 bits) with a cached LUT
# =============================================================================

# Cache a per-device lookup table (LUT) used to unpack bits fast on GPU/CPU.
# Key: torch.device, Value: tensor of shape (256, 8) with uint8 bits (0/1).
_UNPACK_LUT_CACHE: dict[torch.device, torch.Tensor] = {}


def _get_unpack_lut(device: torch.device) -> torch.Tensor:
    """
    Return a (256, 8) lookup table mapping each uint8 value [0..255] to its
    8 bits (MSB -> LSB), stored on the requested device.

    The LUT is cached per device to avoid rebuilding and re-transferring it
    for every chunk.
    """
    lut = _UNPACK_LUT_CACHE.get(device)
    if lut is not None:
        return lut

    # vals: 0..255 (int16 to be safe for shifts)
    vals = torch.arange(256, dtype=torch.int16)

    # shifts: 7..0 so that we extract bits from MSB to LSB
    shifts = torch.arange(7, -1, -1, dtype=torch.int16)  # MSB -> LSB

    # bits: (256, 8) uint8 with 0/1 values
    bits = ((vals[:, None] >> shifts[None, :]) & 1).to(torch.uint8)

    # Move LUT to device once, then cache it
    lut = bits.to(device=device, non_blocking=True)
    _UNPACK_LUT_CACHE[device] = lut
    return lut


def _unpack_bits_torch_from_packed(
    packed_u8: np.ndarray,
    n_bits: int,
    device: torch.device,
    force_copy: bool = True,
) -> torch.Tensor:
    """
    Unpack bit-packed fingerprints stored as uint8 bytes into dense 0/1 bits.

    Parameters
    ----------
    packed_u8
        NumPy array with shape (B, packed_dim) and dtype uint8, representing
        B fingerprints in packed-bit form. Each byte stores 8 bits.
        This is often a view into an np.memmap slice.
    n_bits
        Target number of bits to keep after unpacking (e.g., dim=1024).
        Safety: we truncate if the unpacked length is larger.
    device
        torch.device where the resulting bit matrix will live.
        Typically CUDA for GPU search.
    force_copy
        If True, copy the NumPy input to ensure it is writable and contiguous.
        This avoids the PyTorch warning:
          "The given NumPy array is not writable..."
        which is common for memmap slices.

    Returns
    -------
    torch.Tensor
        A uint8 tensor of shape (B, n_bits) on `device`, containing 0/1 values.

    Notes
    -----
    This method expands packed bytes into dense bits, which can increase memory
    usage by ~8x. It is simple and fast to implement, but may become memory-bound
    for very large chunks.
    """
    # Ensure dtype is uint8, since the LUT expects bytes [0..255]
    if packed_u8.dtype != np.uint8:
        packed_u8 = packed_u8.astype(np.uint8, copy=False)

    # IMPORTANT:
    # np.memmap slices are often read-only. torch.from_numpy(...) will warn
    # because PyTorch does not support non-writable tensors. Copying removes
    # the warning and avoids undefined behavior if any accidental write occurs.
    if force_copy:
        packed_u8 = np.array(packed_u8, copy=True)  # writable + contiguous

    # Create CPU tensor that shares memory with the NumPy array, then move to GPU
    t = torch.from_numpy(packed_u8).to(device=device, non_blocking=True)  # uint8

    # LUT-based unpacking: for each byte, we obtain 8 bits (MSB->LSB)
    lut = _get_unpack_lut(device)
    bits = lut[t.long()]  # (B, packed_dim, 8), uint8

    # Flatten bytes*8 -> bits dimension
    bits = bits.reshape(bits.shape[0], bits.shape[1] * 8)  # (B, packed_dim*8)

    # Safety: truncate to exactly n_bits if unpacked length exceeds n_bits
    if bits.shape[1] > n_bits:
        bits = bits[:, :n_bits]

    return bits  # uint8 0/1 on GPU


def _tanimoto_from_binary_mats_torch(
    q_bits_u8: torch.Tensor,  # (Q, dim) uint8 0/1 on GPU
    q_counts: torch.Tensor,   # (Q,) int32 or float32 on GPU
    b_bits_u8: torch.Tensor,  # (B, dim) uint8 0/1 on GPU
    b_counts: torch.Tensor,   # (B,) int32 or float32 on GPU
) -> torch.Tensor:
    """
    Compute the Tanimoto similarity matrix between two sets of binary vectors.

    Given:
      - Q query fingerprints in q_bits_u8 (0/1)
      - B block fingerprints in b_bits_u8 (0/1)

    Tanimoto is:
      ti = inter / (|q| + |b| - inter),
    where inter is the number of shared 1-bits.

    Implementation details
    ----------------------
    On CUDA, integer matmul (int32 addmm) is not generally supported. Therefore
    we compute intersections using float32 matmul, which remains exact for typical
    fingerprint dimensions (e.g., 1024/2048) because the sums are small integers
    that float32 can represent exactly.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape (Q, B) with Tanimoto similarities.
    """
    # Convert to float32 for CUDA-friendly matmul.
    # Intersections remain exact for typical bitvector dimensions.
    q_f = q_bits_u8.to(torch.float32)
    b_f = b_bits_u8.to(torch.float32)

    # inter: (Q, B) number of shared bits
    inter = q_f @ b_f.T

    # Convert bitcounts to float32
    q_c = q_counts.to(torch.float32)
    b_c = b_counts.to(torch.float32)

    # denom: (Q, B)
    denom = q_c[:, None] + b_c[None, :] - inter

    # Avoid division by zero
    ti = torch.zeros_like(denom, dtype=torch.float32)
    valid = denom > 0
    ti[valid] = inter[valid] / denom[valid]
    return ti


# =============================================================================
# 1) Max pairwise Tanimoto among representatives (GPU, vectorized)
# =============================================================================

def max_pairwise_tanimoto_from_ids_torch_gpu(
    representatives: Sequence[str],
    rep: "Representation",
    device: Union[str, torch.device] = "cuda",
) -> float:
    """
    Compute the maximum pairwise Tanimoto similarity among the given ligands
    (representatives) using GPU acceleration.

    This replaces the O(N^2) Python double-loop with a vectorized approach:
      - Fetch binary 0/1 fingerprints (N, dim) as uint8 on CPU
      - Transfer to GPU
      - Compute the full intersection matrix via matrix multiplication
      - Convert intersections to Tanimoto for all pairs
      - Return the maximum over the upper triangle (excluding the diagonal)

    Parameters
    ----------
    representatives
        List/sequence of chem_comp_id strings.
    rep
        Representation object providing fingerprints via:
            rep.get_by_ids(ids, as_float=False)
        Expected output is 0/1 uint8 matrix.
    device
        CUDA device (default "cuda"). You can pass "cuda:0", etc.

    Returns
    -------
    float
        Maximum Tanimoto coefficient among all representative pairs.
    """
    reps = list(representatives)
    if len(reps) < 2:
        return 0.0

    # Fetch fingerprints as 0/1 uint8 on CPU
    fps = rep.get_by_ids(reps, as_float=False)  # (N, dim)
    if fps.dtype != np.uint8:
        fps = fps.astype(np.uint8, copy=False)

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    # Move to GPU
    t_u8 = torch.from_numpy(fps).to(dev, non_blocking=True)  # uint8
    counts = t_u8.sum(dim=1).to(torch.int32)                 # (N,) int32

    # Intersections:
    # Use float32 matmul to avoid int32 matmul limitations on CUDA.
    t_f = t_u8.to(torch.float32)
    inter = t_f @ t_f.T  # (N, N) float32, exact for typical dims

    # Denominator for Tanimoto: |a| + |b| - inter
    counts_f = counts.to(torch.float32)
    denom = counts_f[:, None] + counts_f[None, :] - inter

    # Compute Tanimoto safely
    ti = torch.zeros_like(denom, dtype=torch.float32)
    valid = denom > 0
    ti[valid] = inter[valid] / denom[valid]

    # Upper triangle without diagonal contains unique pairs
    triu = torch.triu(ti, diagonal=1)
    max_ti = triu.max().item()
    return float(max_ti)


# =============================================================================
# 2) ZINC chunk search: GPU compute (single-process)
# =============================================================================

def _search_zinc_chunk_torch_gpu(
    memmap: np.memmap,
    chunk_start: int,
    chunk_end: int,
    batch_query_ids: Sequence[str],
    batch_fps_u8: np.ndarray,         # (Q, dim) uint8 0/1 on CPU
    batch_bitcounts_i32: np.ndarray,  # (Q,) int32 on CPU
    dim: int,
    tanimoto_threshold: float,
    device: Union[str, torch.device] = "cuda",
    return_topk: Optional[int] = None,
    force_copy_packed: bool = True,
) -> List[Tuple[str, int, float]]:
    """
    Search a single ZINC chunk against a batch of query fingerprints on GPU.

    Parameters
    ----------
    memmap
        ZINC packed fingerprint memmap of shape (n_ligands, packed_dim), dtype uint8.
        Each row is a packed-bit fingerprint (packed_dim bytes = dim/8).
    chunk_start, chunk_end
        Index range [chunk_start, chunk_end) selecting a contiguous block from ZINC.
    batch_query_ids
        Query IDs corresponding to rows in batch_fps_u8 / batch_bitcounts_i32.
    batch_fps_u8
        Dense 0/1 query fingerprints, shape (Q, dim), uint8.
    batch_bitcounts_i32
        Precomputed bitcounts for each query fingerprint, shape (Q,), int32.
    dim
        Number of bits in the dense representation (e.g., 1024).
    tanimoto_threshold
        Minimum similarity required to report a hit.
    device
        Target device ("cuda" by default).
    return_topk
        If None: return all hits >= threshold for this chunk.
        If int K: return up to top-K hits per query for this chunk (and stop early
        per query once values drop below threshold).
    force_copy_packed
        If True: copy the packed memmap slice before torch.from_numpy to avoid
        "not writable" warnings.

    Returns
    -------
    List[Tuple[str, int, float]]
        A list of (query_id, lig_idx_global, tanimoto) for hits found in this chunk.
        lig_idx_global is the ZINC ligand index in the full memmap.
    """
    # Read packed block from memmap (CPU)
    packed_block = memmap[chunk_start:chunk_end]
    if packed_block.size == 0:
        return []

    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    # -------------------------------------------------------------------------
    # 1) Transfer query batch to GPU
    # -------------------------------------------------------------------------
    if batch_fps_u8.dtype != np.uint8:
        batch_fps_u8 = batch_fps_u8.astype(np.uint8, copy=False)

    q_bits = torch.from_numpy(batch_fps_u8).to(dev, non_blocking=True)  # uint8 0/1
    q_counts = torch.from_numpy(batch_bitcounts_i32).to(dev, non_blocking=True).to(torch.int32)

    # -------------------------------------------------------------------------
    # 2) Unpack packed ZINC block to dense bits on GPU
    # -------------------------------------------------------------------------
    b_bits = _unpack_bits_torch_from_packed(
        packed_block,
        n_bits=dim,
        device=dev,
        force_copy=force_copy_packed,
    )
    b_counts = b_bits.sum(dim=1).to(torch.int32)

    # -------------------------------------------------------------------------
    # 3) Compute Tanimoto similarities on GPU
    # -------------------------------------------------------------------------
    ti = _tanimoto_from_binary_mats_torch(q_bits, q_counts, b_bits, b_counts)  # (Q, B)

    hits: List[Tuple[str, int, float]] = []

    # -------------------------------------------------------------------------
    # 4) Collect hits: either all above threshold or top-k per query
    # -------------------------------------------------------------------------
    if return_topk is None:
        # Find all entries above threshold
        mask = ti >= float(tanimoto_threshold)
        idx = torch.nonzero(mask, as_tuple=False)  # (H, 2) where H is number of hits
        if idx.numel() == 0:
            return hits

        # Gather only the hit values (avoid copying full ti back to CPU)
        vals = ti[idx[:, 0], idx[:, 1]]

        # Move indices and values to CPU for Python-side list building
        idx_cpu = idx.detach().cpu().numpy()
        vals_cpu = vals.detach().cpu().numpy()

        for k in range(idx_cpu.shape[0]):
            qi = int(idx_cpu[k, 0])              # query row index
            bi = int(idx_cpu[k, 1])              # block row index (within chunk)
            lig_idx_global = chunk_start + bi    # global ZINC index
            hits.append((batch_query_ids[qi], lig_idx_global, float(vals_cpu[k])))

        return hits

    # Top-k mode: return up to K best hits per query for this chunk
    k = int(return_topk)
    if k <= 0:
        return hits

    # topv/topi: (Q, k) each row contains the best candidates within the chunk
    topv, topi = torch.topk(ti, k=min(k, ti.shape[1]), dim=1)
    topv_cpu = topv.detach().cpu().numpy()
    topi_cpu = topi.detach().cpu().numpy()

    for qi in range(topv_cpu.shape[0]):
        qid = batch_query_ids[qi]
        for kk in range(topv_cpu.shape[1]):
            val = float(topv_cpu[qi, kk])
            if val < tanimoto_threshold:
                # Because topk is sorted desc, we can break early for this query
                break
            bi = int(topi_cpu[qi, kk])
            lig_idx_global = chunk_start + bi
            hits.append((qid, lig_idx_global, val))

    return hits


def search_similar_in_zinc_torch_gpu(
    query_ids: Sequence[str],
    store_ref: "LigandStore",
    rep_ref: "Representation",
    store_zinc: "LigandStore",
    rep_zinc: "Representation",
    tanimoto_threshold: float = 0.5,
    q_batch_size: int = 200,
    zinc_chunk_size: int = 200_000,
    device: Union[str, torch.device] = "cuda",
    max_hits_per_query: Optional[int] = None,
    per_chunk_topk_hint: Optional[int] = None,
    force_copy_packed: bool = True,
) -> pd.DataFrame:
    """
    GPU-based similarity search in ZINC using Tanimoto over binary fingerprints.

    High-level algorithm
    --------------------
    1) Fetch query fingerprints from rep_ref as dense 0/1 uint8 matrix.
    2) Iterate over ZINC fingerprints in chunks from rep_zinc.memmap (packed bytes).
    3) For each query batch and ZINC chunk:
         - unpack packed ZINC fingerprints to dense bits on GPU
         - compute Tanimoto matrix on GPU
         - extract hits >= threshold (or top-k per query per chunk)
    4) Map ZINC indices to chem_comp_id and smiles via store_zinc.
    5) Optionally cap number of hits per query globally.

    Parameters
    ----------
    query_ids
        Ligand identifiers to use as queries (from the reference store/rep).
    store_ref
        Reference ligand store (unused here except for symmetry with CPU function).
    rep_ref
        Reference representation used to fetch query fingerprints (dense 0/1).
    store_zinc
        ZINC ligand store containing at least columns: lig_idx, chem_comp_id, smiles.
    rep_zinc
        ZINC representation containing a packed-byte memmap of fingerprints.
    tanimoto_threshold
        Minimum Tanimoto similarity to report a hit.
    q_batch_size
        Number of query ligands to process per batch.
    zinc_chunk_size
        Number of ZINC ligands to process per chunk.
    device
        Torch device string or torch.device for GPU compute.
    max_hits_per_query
        If not None, keep only the top-N hits per query globally (after merge).
    per_chunk_topk_hint
        If not None, limit results to top-K per query *per chunk* (can reduce
        memory/CPU overhead if threshold generates many hits).
    force_copy_packed
        Copy packed memmap chunks to avoid "not writable" warnings.

    Returns
    -------
    pd.DataFrame
        Columns: query_id, lig_idx_zinc, chem_comp_id, smiles, tanimoto
    """
    if not query_ids:
        return pd.DataFrame(columns=["query_id", "lig_idx_zinc", "chem_comp_id", "smiles", "tanimoto"])

    qids = list(query_ids)

    # -------------------------------------------------------------------------
    # 1) Fetch query fingerprints on CPU (dense 0/1 uint8)
    # -------------------------------------------------------------------------
    fps_queries = rep_ref.get_by_ids(qids, as_float=False)  # (Q, dim)
    if fps_queries.shape[0] != len(qids):
        raise ValueError("Number of fingerprints does not match number of query_ids.")
    if fps_queries.dtype != np.uint8:
        fps_queries = fps_queries.astype(np.uint8, copy=False)

    bitcounts_queries = fps_queries.sum(axis=1).astype(np.int32)
    dim = rep_ref.dim

    # -------------------------------------------------------------------------
    # 2) ZINC packed memmap and chunking
    # -------------------------------------------------------------------------
    memmap = rep_zinc.memmap
    n_ligands_zinc = memmap.shape[0]

    chunk_ranges: List[Tuple[int, int]] = []
    for start in range(0, n_ligands_zinc, zinc_chunk_size):
        end = min(start + zinc_chunk_size, n_ligands_zinc)
        chunk_ranges.append((start, end))

    # -------------------------------------------------------------------------
    # 3) Prepare mapping from ZINC internal index -> chem_comp_id / smiles
    # -------------------------------------------------------------------------
    ligs_zinc = store_zinc.ligands.copy()
    if "lig_idx" not in ligs_zinc.columns:
        raise ValueError("ZINC ligands parquet must contain column 'lig_idx'.")
    ligs_zinc_by_idx = ligs_zinc.set_index("lig_idx")

    # -------------------------------------------------------------------------
    # 4) Main loop: query batches x ZINC chunks
    # -------------------------------------------------------------------------
    all_hits: List[Tuple[str, int, float]] = []

    for q_start in range(0, len(qids), q_batch_size):
        q_end = min(q_start + q_batch_size, len(qids))

        batch_ids = qids[q_start:q_end]
        batch_fps = fps_queries[q_start:q_end]
        batch_counts = bitcounts_queries[q_start:q_end]

        for cs, ce in chunk_ranges:
            hits_chunk = _search_zinc_chunk_torch_gpu(
                memmap=memmap,
                chunk_start=cs,
                chunk_end=ce,
                batch_query_ids=batch_ids,
                batch_fps_u8=batch_fps,
                batch_bitcounts_i32=batch_counts,
                dim=dim,
                tanimoto_threshold=tanimoto_threshold,
                device=device,
                return_topk=per_chunk_topk_hint,
                force_copy_packed=force_copy_packed,
            )
            if hits_chunk:
                all_hits.extend(hits_chunk)

    # -------------------------------------------------------------------------
    # 5) Convert hits to DataFrame + merge annotations
    # -------------------------------------------------------------------------
    if not all_hits:
        return pd.DataFrame(columns=["query_id", "lig_idx_zinc", "chem_comp_id", "smiles", "tanimoto"])

    hits_df = pd.DataFrame(all_hits, columns=["query_id", "lig_idx_zinc", "tanimoto"])

    hits_df = hits_df.merge(
        ligs_zinc_by_idx[["chem_comp_id", "smiles"]],
        left_on="lig_idx_zinc",
        right_index=True,
        how="left",
    )

    # -------------------------------------------------------------------------
    # 6) Optionally limit the number of hits per query globally
    # -------------------------------------------------------------------------
    if max_hits_per_query is not None:
        hits_df = (
            hits_df.sort_values(["query_id", "tanimoto"], ascending=[True, False])
            .groupby("query_id", as_index=False)
            .head(max_hits_per_query)
            .reset_index(drop=True)
        )

    return hits_df[["query_id", "lig_idx_zinc", "chem_comp_id", "smiles", "tanimoto"]]


def _resolve_search_device(
    device: Optional[Union[str, torch.device]] = None,
) -> torch.device:
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def search_similar_in_zinc_custom(
    query_ids: Sequence[str],
    store_ref: LigandStore,
    rep_ref: Representation,
    store_zinc: LigandStore,
    rep_zinc: Representation,
    *,
    metric: metrics.MetricName = "tanimoto",
    mode: str = "threshold",
    threshold: Optional[float] = 0.5,
    topk: Optional[int] = None,
    q_batch_size: int = 200,
    zinc_chunk_size: int = 200_000,
    device: Optional[Union[str, torch.device]] = "auto",
    n_jobs: int = 4,
    max_hits_per_query: Optional[int] = None,
    assume_normalized: Optional[bool] = None,
    per_chunk_topk_hint: Optional[int] = None,
    force_copy_packed_gpu: bool = True,
) -> pd.DataFrame:
    """
    Customizable ZINC search that supports multiple representations and metrics.

    This function preserves the optimized Tanimoto pathways for packed fingerprints
    and falls back to a generic block-scoring loop for other metrics (e.g., cosine).

    Parameters
    ----------
    query_ids
        Ligand identifiers to query.
    store_ref, rep_ref
        Reference ligand store and representation.
    store_zinc, rep_zinc
        ZINC ligand store and representation.
    metric
        Similarity metric to use ("tanimoto" or "cosine").
    mode
        "threshold" to return all hits >= threshold, or "topk" to return top-K per query.
    threshold
        Minimum similarity score to report a hit (required for mode="threshold").
        When mode="topk", acts as an optional floor to filter low scores.
    topk
        Required when mode="topk".
    q_batch_size
        Query batch size.
    zinc_chunk_size
        ZINC chunk size.
    device
        "cpu", "cuda", or "auto" (default auto).
    n_jobs
        Number of CPU workers for the packed Tanimoto CPU backend.
    max_hits_per_query
        Cap results per query globally.
    assume_normalized
        For cosine similarity, whether vectors are already normalized.
    per_chunk_topk_hint
        Optional per-chunk top-k hint for the GPU Tanimoto backend.
    force_copy_packed_gpu
        Copy packed memmap chunks on GPU to avoid non-writable warnings.
    """
    if mode not in ("threshold", "topk"):
        raise ValueError("mode must be 'threshold' or 'topk'.")
    if mode == "threshold" and threshold is None:
        raise ValueError("threshold must be provided for mode='threshold'.")
    if mode == "topk" and (topk is None or topk <= 0):
        raise ValueError("topk must be provided and > 0 for mode='topk'.")

    metrics.validate_metric(metric, rep_ref.meta, rep_zinc.meta)
    resolved_device = _resolve_search_device(device)
    device_str = "cuda" if resolved_device.type == "cuda" else "cpu"

    if (
        mode == "threshold"
        and metric == "tanimoto"
        and rep_ref.packed_bits
        and rep_zinc.packed_bits
    ):
        if resolved_device.type == "cuda":
            return search_similar_in_zinc_torch_gpu(
                query_ids=query_ids,
                store_ref=store_ref,
                rep_ref=rep_ref,
                store_zinc=store_zinc,
                rep_zinc=rep_zinc,
                tanimoto_threshold=float(threshold),
                q_batch_size=q_batch_size,
                zinc_chunk_size=zinc_chunk_size,
                device=resolved_device,
                max_hits_per_query=max_hits_per_query,
                per_chunk_topk_hint=per_chunk_topk_hint,
                force_copy_packed=force_copy_packed_gpu,
            )

        return search_similar_in_zinc(
            query_ids=query_ids,
            store_ref=store_ref,
            rep_ref=rep_ref,
            store_zinc=store_zinc,
            rep_zinc=rep_zinc,
            tanimoto_threshold=float(threshold),
            q_batch_size=q_batch_size,
            zinc_chunk_size=zinc_chunk_size,
            n_jobs=n_jobs,
            max_hits_per_query=max_hits_per_query,
        )

    request = backend.SearchRequest(
        query_ids=query_ids,
        store_ref=store_ref,
        rep_ref=rep_ref,
        store_target=store_zinc,
        rep_target=rep_zinc,
        metric=metric,
        mode=mode,
        threshold=threshold,
        topk=topk,
        device=resolved_device,
        q_batch_size=q_batch_size,
        target_chunk_size=zinc_chunk_size,
        max_hits_per_query=max_hits_per_query,
        assume_normalized=assume_normalized,
        return_fields=("chem_comp_id", "smiles"),
        id_field="chem_comp_id",
    )
    hits_df = backend.search(request)
    if "target_idx" in hits_df.columns:
        hits_df = hits_df.rename(columns={"target_idx": "lig_idx_zinc"})
    return hits_df

def get_zinc_ligands(
    prot: str,
    pdb_chembl_binding_data: pd.DataFrame,
    store_pdb_chembl: "LigandStore",
    rep_pdb_chembl: "Representation",
    store_zinc: "LigandStore",
    rep_zinc: "Representation",
    max_queries: int = 50,
    cluster_threshold: float = 0.8,
    zinc_search_threshold: float = 0.5,
    # ------------------------------------------------------------------
    # Search customization:
    # - search_rep_ref/search_rep_zinc: representation to use for ZINC search
    # - search_metric: "tanimoto" or "cosine" (see compound_processing.metrics)
    # - search_device: "cpu" / "cuda" / None (auto)
    # ------------------------------------------------------------------
    search_rep_ref: Optional["Representation"] = None,
    search_rep_zinc: Optional["Representation"] = None,
    search_metric: metrics.MetricName = "tanimoto",
    search_mode: str = "threshold",
    search_device: Optional[Union[str, torch.device]] = "auto",
    search_q_batch_size: Optional[int] = None,
    search_zinc_chunk_size: Optional[int] = None,
    search_n_jobs: int = 4,
    search_assume_normalized: Optional[bool] = None,
    search_topk: Optional[int] = None,
) -> pd.DataFrame:
    """
    For a protein (uniprot_id = prot), select a set of representative ligands
    (up to max_queries) and run similarity search in ZINC.

    This function is a single entry-point that supports configurable ZINC
    searches across multiple representations and metrics. For packed Tanimoto
    fingerprints, it preserves the existing optimized CPU/GPU backends; for
    other metrics (e.g., cosine over embeddings), it falls back to a generic
    block-scoring loop that still honors chunking and batching.

    IMPORTANT: The representative selection logic and post-processing are
    unchanged relative to the original CPU-only implementation. Only the ZINC
    search backend is switched automatically.

    Parameters
    ----------
    prot : str
        Protein identifier (uniprot_id).
    pdb_chembl_binding_data : pd.DataFrame
        Binding data including at least columns: uniprot_id, chem_comp_id, pchembl.
    store_pdb_chembl : LigandStore
        Reference ligand store (PDB+ChEMBL).
    rep_pdb_chembl : Representation
        Representation for reference ligands.
    store_zinc : LigandStore
        Ligand store for ZINC.
    rep_zinc : Representation
        Representation for ZINC ligands (packed memmap).
    max_queries : int, default 50
        Maximum number of representative ligands to query in ZINC.
    cluster_threshold : float, default 0.8
        Threshold used for clustering and redundancy checks.
    zinc_search_threshold : float, default 0.5
        Similarity threshold for reporting ZINC hits.
    search_rep_ref, search_rep_zinc : Optional[Representation]
        Representations used for the ZINC search. Defaults to rep_pdb_chembl and rep_zinc.
    search_metric : {"tanimoto", "cosine"}
        Similarity metric for ZINC search.
    search_mode : {"threshold", "topk"}
        Mode for ZINC search (threshold or top-K).
    search_device : Optional[Union[str, torch.device]], default "auto"
        Backend selection for ZINC search:
          - None: auto (CUDA if available, else CPU)
          - "cpu": force CPU backend
          - "cuda" / "cuda:0": prefer CUDA backend (falls back to CPU if unavailable)
          - "auto": same as None
    search_q_batch_size : Optional[int]
        Batch size for ZINC search (auto-tuned defaults if None).
    search_zinc_chunk_size : Optional[int]
        ZINC chunk size for search (auto-tuned defaults if None).
    search_n_jobs : int, default 4
        CPU worker count for the packed Tanimoto backend.
    search_assume_normalized : Optional[bool]
        For cosine similarity, whether vectors are already normalized.
    search_topk : Optional[int]
        If set and search_mode="topk", return only the top-K hits per query.

    Returns
    -------
    pd.DataFrame
        Columns:
          - chem_comp_id  (prefixed with "ZINC")
          - query_id
          - similarity score column ("tanimoto" or "similarity")
          - smiles
        Internal column lig_idx_zinc is removed before returning.
    """
    # ------------------------------------------------------------------
    # 0) Resolve device choice (auto / forced) for ZINC search
    # ------------------------------------------------------------------
    resolved_device = _resolve_search_device(search_device)
    search_rep_ref = rep_pdb_chembl if search_rep_ref is None else search_rep_ref
    search_rep_zinc = rep_zinc if search_rep_zinc is None else search_rep_zinc
    score_col = "tanimoto" if search_metric == "tanimoto" else "similarity"

    if search_q_batch_size is None or search_zinc_chunk_size is None:
        if search_metric == "tanimoto" and search_rep_ref.packed_bits and search_rep_zinc.packed_bits:
            if resolved_device.type == "cuda":
                search_q_batch_size = 200 if search_q_batch_size is None else search_q_batch_size
                search_zinc_chunk_size = 200_000 if search_zinc_chunk_size is None else search_zinc_chunk_size
            else:
                search_q_batch_size = 100 if search_q_batch_size is None else search_q_batch_size
                search_zinc_chunk_size = 50_000 if search_zinc_chunk_size is None else search_zinc_chunk_size
        else:
            search_q_batch_size = 200 if search_q_batch_size is None else search_q_batch_size
            search_zinc_chunk_size = 200_000 if search_zinc_chunk_size is None else search_zinc_chunk_size

    # ------------------------------------------------------------------
    # 1) Subset by protein and extract chem_ids + pchembl_map
    # ------------------------------------------------------------------
    df_prot = pdb_chembl_binding_data.loc[
        pdb_chembl_binding_data["uniprot_id"] == prot,
        ["chem_comp_id", "pchembl"],
    ]

    chem_ids = (
        df_prot["chem_comp_id"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )

    if not chem_ids:
        return pd.DataFrame(columns=["query_id", "chem_comp_id", "smiles", score_col])

    pchembl_map = (
        df_prot
        .dropna(subset=["pchembl"])
        .groupby("chem_comp_id")["pchembl"]
        .max()
        .to_dict()
    )

    # ------------------------------------------------------------------
    # 2) Representative selection
    # ------------------------------------------------------------------
    n_ligs = len(chem_ids)

    if n_ligs <= max_queries:
        # Cluster-based selection
        representatives, clusters, assignment = cluster_ligands_by_tanimoto(
            chem_comp_ids=chem_ids,
            rep=rep_pdb_chembl,
            threshold=cluster_threshold,
            pchembl_map=pchembl_map,
        )
    else:
        # MaxMin selection
        representatives_mm, clusters_mm, assignment_mm = select_ligands_maxmin_by_tanimoto(
            chem_comp_ids=chem_ids,
            rep=rep_pdb_chembl,
            n_select=max_queries,
            pchembl_map=pchembl_map,
        )

        # Check redundancy among MaxMin representatives
        # Keep the original CPU implementation here to preserve behavior.
        ti_max = max_pairwise_tanimoto_from_ids(representatives_mm, rep_pdb_chembl)

        if ti_max >= cluster_threshold:
            # Re-cluster MaxMin reps to reduce redundancy
            representatives, clusters, assignment = cluster_ligands_by_tanimoto(
                chem_comp_ids=representatives_mm,
                rep=rep_pdb_chembl,
                threshold=cluster_threshold,
                pchembl_map=pchembl_map,
            )
        else:
            # Keep MaxMin representatives directly
            representatives = representatives_mm

    # ------------------------------------------------------------------
    # 3) Search in ZINC (customizable backend)
    # ------------------------------------------------------------------
    zinc_ligands = search_similar_in_zinc_custom(
        query_ids=representatives,
        store_ref=store_pdb_chembl,
        rep_ref=search_rep_ref,
        store_zinc=store_zinc,
        rep_zinc=search_rep_zinc,
        metric=search_metric,
        mode=search_mode,
        threshold=zinc_search_threshold,
        topk=search_topk,
        q_batch_size=search_q_batch_size,
        zinc_chunk_size=search_zinc_chunk_size,
        device=resolved_device,
        n_jobs=search_n_jobs,
        max_hits_per_query=None,
        assume_normalized=search_assume_normalized,
        per_chunk_topk_hint=None,
    )

    # If there are no hits, return an empty DataFrame with the expected columns
    if zinc_ligands.empty:
        return pd.DataFrame(columns=["query_id", "chem_comp_id", "smiles", score_col])

    # ------------------------------------------------------------------
    # 4) Remove internal column "lig_idx_zinc"
    # ------------------------------------------------------------------
    if "lig_idx_zinc" in zinc_ligands.columns:
        zinc_ligands = zinc_ligands.drop(columns=["lig_idx_zinc"])

    # ------------------------------------------------------------------
    # 5) Add "ZINC" prefix and remove duplicate ZINC compounds
    # ------------------------------------------------------------------
    zinc_ligands["chem_comp_id"] = "ZINC" + zinc_ligands["chem_comp_id"].astype(str)

    zinc_ligands = zinc_ligands.sort_values(score_col, ascending=False)

    zinc_ligands = zinc_ligands.drop_duplicates(
        subset=["chem_comp_id", "smiles"],
        keep="first",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6) Return only the public columns
    # ------------------------------------------------------------------
    return zinc_ligands[["chem_comp_id", "query_id", score_col, "smiles"]]
