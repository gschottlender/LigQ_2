from __future__ import annotations

import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence, Mapping, Optional, List, Dict, Tuple

import numpy as np
import pandas as pd

from compound_processing.compound_helpers import (
    LigandStore,
    Representation,
    build_morgan_representation,
    build_ligand_index,
)


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


def get_zinc_ligands(
    prot: str,
    pdb_chembl_binding_data: pd.DataFrame,
    store_pdb_chembl: LigandStore,
    rep_pdb_chembl: Representation,
    store_zinc: LigandStore,
    rep_zinc: Representation,
    max_queries: int = 50,
    cluster_threshold: float = 0.8,
    zinc_search_threshold: float = 0.5,
) -> pd.DataFrame:
    """
    For a protein (uniprot_id = prot), select a set of representative ligands
    (up to max_queries) and run similarity search in ZINC.

    Strategy
    --------
      - Extract chem_comp_id values for `prot` in pdb_chembl_binding_data.
      - Build a chem_comp_id → pchembl map (maximum pchembl per ligand).
      - If n_ligands <= max_queries:
            * cluster_ligands_by_tanimoto with threshold = cluster_threshold,
              promoting highest-pchembl ligand within each cluster.
      - If n_ligands > max_queries:
            * MaxMin selection to max_queries (select_ligands_maxmin_by_tanimoto)
              using pchembl_map to promote higher pchembl ligands.
            * Compute max pairwise Tanimoto among selected reps.
            * If ti_max >= cluster_threshold:
                  cluster these reps again with cluster_threshold.
              Else:
                  keep MaxMin reps directly.
      - Query ZINC using search_similar_in_zinc with the final representatives.
      - Post-process ZINC hits:
            * remove internal index column,
            * prefix chem_comp_id with "ZINC",
            * sort by Tanimoto descending,
            * drop duplicate ZINC compounds.

    Returns
    -------
    pd.DataFrame
        Columns:
          - query_id
          - chem_comp_id  (prefixed with "ZINC")
          - smiles
          - tanimoto
        (lig_idx_zinc is removed before returning)
    """
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
        return pd.DataFrame(
            columns=["query_id", "chem_comp_id", "smiles", "tanimoto"]
        )

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
    # 3) Search in ZINC
    # ------------------------------------------------------------------
    zinc_ligands = search_similar_in_zinc(
        query_ids=representatives,
        store_ref=store_pdb_chembl,
        rep_ref=rep_pdb_chembl,
        store_zinc=store_zinc,
        rep_zinc=rep_zinc,
        tanimoto_threshold=zinc_search_threshold,
        q_batch_size=100,
        zinc_chunk_size=50_000,
        n_jobs=4,
        max_hits_per_query=None,
    )

    # If there are no hits, return an empty DataFrame with the expected columns
    if zinc_ligands.empty:
        return pd.DataFrame(
            columns=["query_id", "chem_comp_id", "smiles", "tanimoto"]
        )

    # ------------------------------------------------------------------
    # 4) Remove internal column "lig_idx_zinc"
    # ------------------------------------------------------------------
    if "lig_idx_zinc" in zinc_ligands.columns:
        zinc_ligands = zinc_ligands.drop(columns=["lig_idx_zinc"])

    # ------------------------------------------------------------------
    # 5) Add "ZINC" prefix and remove duplicate ZINC compounds
    # ------------------------------------------------------------------
    # Prefix ZINC chem_comp_id to clearly distinguish them from PDB/ChEMBL IDs
    zinc_ligands["chem_comp_id"] = "ZINC" + zinc_ligands["chem_comp_id"].astype(str)

    # Sort by Tanimoto descending so we keep the strongest hit per compound
    zinc_ligands = zinc_ligands.sort_values("tanimoto", ascending=False)

    # Drop duplicates by ZINC compound (and smiles for safety), keeping best Tanimoto
    zinc_ligands = zinc_ligands.drop_duplicates(
        subset=["chem_comp_id", "smiles"],
        keep="first",
    ).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 6) Return only the public columns
    # ------------------------------------------------------------------
    return zinc_ligands[["chem_comp_id", "query_id", "tanimoto", "smiles"]]
