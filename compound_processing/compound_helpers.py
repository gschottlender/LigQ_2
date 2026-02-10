"""
Helpers for compound unification and numerical representations.

This module provides:

- Unification of PDB and ChEMBL ligand tables using InChIKey
  while preserving all rows (no drops for missing InChIKey).
- Construction of a ligand index (ligands.parquet) with a dense
  integer index (lig_idx) for efficient array storage.
- Efficient computation and storage of Morgan fingerprints
  (packed bits in a memmap on disk).
- A small API to retrieve representations (e.g. Morgan) by comp_id
  without loading the full matrix into RAM.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, List, Tuple

import os
import multiprocessing as mp
import json
import time
from contextlib import nullcontext
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import inchi, AllChem, DataStructs
from rdkit import RDLogger
import torch
from transformers import AutoModel, AutoTokenizer

# Silence RDKit warnings (invalid SMILES, sanitization issues, etc.)
RDLogger.DisableLog("rdApp.*")
# Morgan globals
_MORGAN_WORKER_CFG: Dict[str, int] = {"n_bits": 1024, "radius": 2, "n_bytes": 128}

# ---------------------------------------------------------------------------
# 1. Basic utilities: InChIKey, Morgan fingerprints, bit packing
# ---------------------------------------------------------------------------

def smiles_to_inchikey(smiles: str) -> Optional[str]:
    """
    Convert a SMILES string to an InChIKey. Returns None if it fails.

    This is helpful to structurally unify compounds coming from different
    sources (PDB, ChEMBL, etc.) even when SMILES differ in notation.
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def morgan_fp_bits(
    smiles: str,
    n_bits: int = 1024,
    radius: int = 2,
) -> Optional[np.ndarray]:
    """
    Compute a Morgan fingerprint as a 0/1 numpy array of shape (n_bits,).

    Returns None if the SMILES cannot be parsed or the fingerprint
    cannot be generated.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    n_bits : int
        Fingerprint length in bits (default: 1024).
    radius : int
        Morgan fingerprint radius (default: 2).
    """
    if pd.isna(smiles):
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None


def pack_bits(arr: np.ndarray) -> np.ndarray:
    """
    Pack a 0/1 array of bits along the last axis using numpy.packbits.

    Parameters
    ----------
    arr : np.ndarray
        Array of shape (N, n_bits) with values 0/1.

    Returns
    -------
    np.ndarray
        Packed array of shape (N, n_bits / 8), dtype=uint8.
    """
    return np.packbits(arr, axis=-1)


def unpack_bits(packed: np.ndarray, n_bits: int) -> np.ndarray:
    """
    Unpack bits from the last axis back to an array of length n_bits.

    Parameters
    ----------
    packed : np.ndarray
        Packed array of shape (N, n_bytes).
    n_bits : int
        Desired number of bits in the output.

    Returns
    -------
    np.ndarray
        Unpacked array of shape (N, n_bits), dtype=uint8 with values 0/1.
    """
    unpacked = np.unpackbits(packed, axis=-1)
    # In case there are extra bits, truncate to the desired length
    if unpacked.shape[-1] > n_bits:
        unpacked = unpacked[..., :n_bits]
    return unpacked

def _init_morgan_worker(n_bits: int, radius: int) -> None:
    _MORGAN_WORKER_CFG["n_bits"] = int(n_bits)
    _MORGAN_WORKER_CFG["radius"] = int(radius)
    _MORGAN_WORKER_CFG["n_bytes"] = int(n_bits) // 8

def _morgan_fp_bits_or_zero(smiles: str) -> np.ndarray:
    """
    Worker: calculate fp bits (n_bits,) uint8, or vector 0 if it fails.
    Use global config set to _init_morgan_worker.
    """
    n_bits = _MORGAN_WORKER_CFG["n_bits"]
    radius = _MORGAN_WORKER_CFG["radius"]

    arr = morgan_fp_bits(smiles, n_bits=n_bits, radius=radius)
    if arr is None:
        return np.zeros((n_bits,), dtype=np.uint8)

    # ensure correct dtype/shape
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape != (n_bits,):
        return np.zeros((n_bits,), dtype=np.uint8)
    return arr


def _morgan_fp_packed_or_zero(smiles: str) -> Tuple[np.ndarray, bool]:
    """
    Worker: calculate packed fp bytes (n_bits/8,) uint8 and a success flag.

    Returning packed bytes from workers reduces IPC payload significantly versus
    sending full bit vectors.
    """
    n_bits = _MORGAN_WORKER_CFG["n_bits"]
    n_bytes = _MORGAN_WORKER_CFG["n_bytes"]
    radius = _MORGAN_WORKER_CFG["radius"]

    arr = morgan_fp_bits(smiles, n_bits=n_bits, radius=radius)
    if arr is None:
        return np.zeros((n_bytes,), dtype=np.uint8), False

    arr = np.asarray(arr, dtype=np.uint8)
    if arr.shape != (n_bits,):
        return np.zeros((n_bytes,), dtype=np.uint8), False

    packed = np.packbits(arr)
    if packed.shape != (n_bytes,):
        return np.zeros((n_bytes,), dtype=np.uint8), False
    return packed, True

# ---------------------------------------------------------------------------
# 2. Unify PDB and ChEMBL ligand tables
# ---------------------------------------------------------------------------


def unify_pdb_chembl(
    ligs_smiles_pdb: pd.DataFrame,
    ligs_smiles_chembl: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Unify PDB and ChEMBL ligands using InChIKey and build an ID mapping.

    Both input tables are expected to use the same ID column name:
      - 'chem_comp_id' : ligand identifier (PDB 3-letter codes, CHEMBL IDs, etc.)
      - 'smiles'       : canonical SMILES string

    The function performs a structure-based unification:

      - For each non-null InChIKey, all ligands (from PDB and/or ChEMBL)
        sharing that InChIKey are grouped together.
      - A single canonical ID is chosen for the group:
          * Prefer a PDB ID (source == 'pdb') if present.
          * Otherwise, choose one of the ChEMBL IDs (lexicographically smallest).
      - All original IDs in the group are mapped to this canonical ID.

      - For rows with null InChIKey, no structure-based unification is
        possible. Each chem_comp_id becomes canonical for itself.

    Returns
    -------
    final_ligs : pd.DataFrame
        DataFrame with one row per canonical ligand, columns:
          - 'chem_comp_id' : canonical ID
          - 'smiles'       : canonical SMILES (taken from one representative)
        This is the table that should be used to build ligands.parquet.

    id_mapping : dict
        Dictionary mapping ANY original ID (from ligs_smiles_pdb or
        ligs_smiles_chembl) to its canonical ID in final_ligs:
          { original_chem_comp_id -> canonical_chem_comp_id }

        - For canonical IDs themselves, the mapping will simply be
          id_mapping[canonical_id] == canonical_id.
        - For structurally duplicated IDs (same InChIKey), all will map
          to the same canonical ID.
    """

    pdb_df = ligs_smiles_pdb.copy()
    chembl_df = ligs_smiles_chembl.copy()

    # Basic column checks
    for name, df in [("ligs_smiles_pdb", pdb_df), ("ligs_smiles_chembl", chembl_df)]:
        if "chem_comp_id" not in df.columns:
            raise ValueError(f"{name} must have a 'chem_comp_id' column.")
        if "smiles" not in df.columns:
            raise ValueError(f"{name} must have a 'smiles' column.")

    # Compute InChIKey for both tables
    pdb_df["inchikey"] = pdb_df["smiles"].map(smiles_to_inchikey)
    chembl_df["inchikey"] = chembl_df["smiles"].map(smiles_to_inchikey)

    # Tag source
    pdb_df["source"] = "pdb"
    chembl_df["source"] = "chembl"

    # Keep only the relevant columns
    pdb_df = pdb_df[["chem_comp_id", "smiles", "inchikey", "source"]]
    chembl_df = chembl_df[["chem_comp_id", "smiles", "inchikey", "source"]]

    # Concatenate both tables
    combined = pd.concat([pdb_df, chembl_df], ignore_index=True)

    # Split into non-null InChIKey (can be unified) and null (cannot)
    non_null = combined[combined["inchikey"].notna()].copy()
    null_rows = combined[combined["inchikey"].isna()].copy()

    # Mapping from original ID -> canonical ID
    id_mapping: Dict[str, str] = {}
    canonical_rows = []

    # ------------------------------------------------------------------
    # 1. Handle non-null InChIKey groups (structure-based unification)
    # ------------------------------------------------------------------
    if not non_null.empty:
        for inchikey, grp in non_null.groupby("inchikey", sort=False):
            # Prefer PDB ligands as canonical, if available
            pdb_grp = grp[grp["source"] == "pdb"]
            if not pdb_grp.empty:
                # Choose one PDB ligand as canonical (e.g. lexicographically smallest ID)
                canon_row = pdb_grp.sort_values("chem_comp_id").iloc[0]
            else:
                # No PDB ligand: choose one ChEMBL ligand as canonical
                canon_row = grp.sort_values("chem_comp_id").iloc[0]

            canon_id = canon_row["chem_comp_id"]
            canonical_rows.append(canon_row)

            # Map all IDs in this group to the canonical ID
            for cid in grp["chem_comp_id"].unique():
                id_mapping[cid] = canon_id

    # ------------------------------------------------------------------
    # 2. Handle null InChIKey rows (no structural unification possible)
    # ------------------------------------------------------------------
    # For these, each chem_comp_id is its own canonical ID, unless
    # it was already assigned in the previous step.
    if not null_rows.empty:
        # Drop exact duplicates to avoid adding the same canonical row twice
        null_rows = null_rows.drop_duplicates(subset=["chem_comp_id", "smiles", "inchikey", "source"])

        for _, row in null_rows.iterrows():
            cid = row["chem_comp_id"]
            if cid not in id_mapping:
                # This ID was not part of any non-null InChIKey group
                id_mapping[cid] = cid
                canonical_rows.append(row)

    # ------------------------------------------------------------------
    # 3. Build final_ligs from canonical rows
    # ------------------------------------------------------------------
    canonical_df = pd.DataFrame(canonical_rows)

    # Ensure uniqueness by canonical ID (in case of any accidental duplicates)
    canonical_df = canonical_df.sort_values("chem_comp_id").drop_duplicates(
        subset=["chem_comp_id"],
        keep="first",
    )

    final_ligs = canonical_df[["chem_comp_id", "smiles"]].reset_index(drop=True)

    return final_ligs, id_mapping


# ---------------------------------------------------------------------------
# 3. Build ligand index and Morgan representation
# ---------------------------------------------------------------------------

def build_ligand_index(
    final_ligs: pd.DataFrame,
    root: str | Path,
) -> Path:
    """
    Build the ligand index table (ligands.parquet) with a dense integer index.

    The resulting table contains:
      - chem_comp_id   : final unified ligand ID (PDB or ChEMBL)
      - smiles    : canonical SMILES used downstream
      - inchikey  : structure-based key (may be null)
      - lig_idx   : dense integer index [0..N-1] used to index arrays on disk

    Parameters
    ----------
    final_ligs : pd.DataFrame
        Unified ligand table with at least ['chem_comp_id', 'smiles'].
    root : str or Path
        Directory where ligands.parquet will be written.

    Returns
    -------
    Path
        Path to the written ligands.parquet file.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    df = final_ligs.copy()
    if "inchikey" not in df.columns:
        df["inchikey"] = df["smiles"].map(smiles_to_inchikey)

    df = df.reset_index(drop=True)
    df["lig_idx"] = np.arange(len(df), dtype=np.int64)

    out_path = root / "ligands.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def build_morgan_representation(
    root: str | Path,
    n_bits: int = 1024,
    radius: int = 2,
    batch_size: int = 10000,
    name: str = "morgan_1024_r2",
    n_jobs: Optional[int] = None,
    chunksize: int = 500,
) -> None:
    """
    Compute Morgan fingerprints for all ligands in ligands.parquet and
    store them as a packed bit matrix in a memmap on disk.

    Parallel version:
      - Uses multiprocessing to compute fingerprints in parallel within each batch.
      - Writes the memmap slices from the main process only (safe).

    Parameters
    ----------
    n_jobs : int, optional
        Number of worker processes. Default: cpu_count().
    chunksize : int
        Chunk size passed to pool.imap for better throughput.
    """
    root = Path(root)
    reps_dir = root / "reps"
    reps_dir.mkdir(exist_ok=True, parents=True)

    ligs_path = root / "ligands.parquet"
    ligs = pd.read_parquet(ligs_path, columns=["smiles"])
    n = len(ligs)

    if n == 0:
        raise ValueError("ligands.parquet is empty, nothing to process.")
    if n_bits % 8 != 0:
        raise ValueError("n_bits must be divisible by 8 for packed storage.")

    # Default workers: use all available CPUs unless user overrides.
    cpu_avail = os.cpu_count() or 1
    if n_jobs is None:
        n_jobs = cpu_avail
    else:
        n_jobs = max(1, min(int(n_jobs), cpu_avail))

    n_bytes = n_bits // 8
    data_path = reps_dir / f"{name}.dat"

    # Create an empty memmap on disk
    mm = np.memmap(
        data_path,
        mode="w+",
        dtype=np.uint8,
        shape=(n, n_bytes),
    )

    # Multiprocessing context: fork is best on Linux (esp. notebook)
    ctx = mp.get_context("fork")
    t0 = time.perf_counter()
    failed_smiles = 0

    # Create pool ONCE (important: avoid overhead per batch)
    with ctx.Pool(
        processes=n_jobs,
        initializer=_init_morgan_worker,
        initargs=(n_bits, radius),
    ) as pool:
        # Process ligands batch by batch
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            smiles_list = ligs.iloc[start:end]["smiles"].tolist()

            # Parallel compute: workers return packed bytes directly to reduce IPC overhead.
            packed_and_status = list(pool.imap(_morgan_fp_packed_or_zero, smiles_list, chunksize=chunksize))
            fps_packed = np.stack([x[0] for x in packed_and_status], axis=0)  # (B, n_bytes)

            # Count true parser/fingerprint failures (avoid inferring from all-zero rows).
            failed_smiles += int(sum(0 if ok else 1 for _, ok in packed_and_status))

            mm[start:end, :] = fps_packed

    mm.flush()

    elapsed_s = float(time.perf_counter() - t0)
    ligands_per_s = float(n / elapsed_s) if elapsed_s > 0 else 0.0

    # Metadata describing this representation
    meta = {
        "name": name,
        "file": f"{name}.dat",
        "dtype": "uint8",
        "dim": int(n_bits),
        "radius": int(radius),
        "packed_bits": True,
        "packed_dim": int(n_bytes),
        "n_ligands": int(n),
        "n_jobs": int(n_jobs),  # opcional, pero útil para trazabilidad
        "elapsed_seconds": elapsed_s,
        "ligands_per_second": ligands_per_s,
        "failed_smiles": int(failed_smiles),
        "failed_smiles_fraction": float(failed_smiles / n),
    }
    meta_path = reps_dir / f"{name}.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def build_huggingface_representation(
    root: str | Path,
    n_bits: Optional[int] = 768,
    batch_size: int = 14,
    name: str = "chemberta_zinc_base_768",
    tokenizer=None,
    model=None,
    device: Optional[torch.device] = None,
    model_id: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_length: Optional[int] = None,
    pooling: str = "mean_attention_mask",
) -> None:
    root = Path(root)
    reps_dir = root / "reps"
    reps_dir.mkdir(exist_ok=True, parents=True)

    ligs_path = root / "ligands.parquet"
    ligs = pd.read_parquet(ligs_path, columns=["smiles"])
    n = len(ligs)
    if n == 0:
        raise ValueError("ligands.parquet is empty, nothing to process.")

    # -----------------------------
    # Load / reuse HuggingFace model
    # -----------------------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    if model is None:
        model = AutoModel.from_pretrained(model_id).to(device)
        model.eval()
    else:
        # ensure eval + device
        model.eval()
        model = model.to(device)

    hidden_size = int(getattr(model.config, "hidden_size", 0))
    if hidden_size <= 0:
        raise ValueError("Could not infer hidden_size from model.config.")
    if n_bits is not None and int(n_bits) != hidden_size:
        raise ValueError(
            f"n_bits={n_bits} does not match model hidden_size={hidden_size}. "
            f"Use n_bits={hidden_size} or switch model."
        )

    dim = hidden_size
    data_path = reps_dir / f"{name}.dat"

    mm = np.memmap(
        data_path,
        mode="w+",
        dtype=np.float16,
        shape=(n, dim),
    )

    smiles_all = ligs["smiles"].tolist()
    t0 = time.perf_counter()
    invalid_smiles = 0
    embed_failures = 0

    # -----------------------------
    # Embedding helper (mean pooling)
    # -----------------------------
    def _embed_batch(smiles_list: list[str]) -> np.ndarray:
        enc = tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        amp_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )

        with torch.inference_mode(), amp_ctx:
            out = model(**enc)
            last = out.last_hidden_state  # (B, T, H)
            attn = enc.get("attention_mask", None)

            if pooling == "cls":
                pooled = last[:, 0, :]
            elif attn is None:
                pooled = last.mean(dim=1)
            elif pooling == "mean_attention_mask":
                attn_f = attn.unsqueeze(-1).to(last.dtype)
                summed = (last * attn_f).sum(dim=1)
                denom = attn_f.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom
            else:
                raise ValueError(
                    "Unsupported pooling strategy. Use 'mean_attention_mask' or 'cls'."
                )

        return pooled.detach().to(torch.float16).cpu().numpy()

    # -----------------------------
    # Process ligands batch by batch
    # -----------------------------
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_smiles = smiles_all[start:end]
        batch_n = end - start

        emb_out = np.zeros((batch_n, dim), dtype=np.float16)
        valid_idx = []
        valid_smiles = []

        for idx, smi in enumerate(batch_smiles):
            if pd.isna(smi):
                invalid_smiles += 1
                continue
            smi_str = str(smi).strip()
            if not smi_str:
                invalid_smiles += 1
                continue
            valid_idx.append(idx)
            valid_smiles.append(smi_str)

        if not valid_smiles:
            mm[start:end, :] = emb_out
            continue

        try:
            emb = _embed_batch(valid_smiles)
            if emb.shape != (len(valid_smiles), dim):
                raise RuntimeError(f"Unexpected embedding shape {emb.shape} vs {(len(valid_smiles), dim)}")
            emb_out[np.asarray(valid_idx)] = emb.astype(np.float16, copy=False)
        except Exception:
            # fallback per-smiles (slow but robust)
            for idx, smi in zip(valid_idx, valid_smiles):
                try:
                    emb_out[idx] = _embed_batch([smi])[0].astype(np.float16, copy=False)
                except Exception:
                    embed_failures += 1

        mm[start:end, :] = emb_out

    mm.flush()
    elapsed_s = float(time.perf_counter() - t0)
    ligands_per_s = float(n / elapsed_s) if elapsed_s > 0 else 0.0

    meta: Dict = {
        "name": name,
        "file": f"{name}.dat",
        "dtype": "float16",
        "dim": int(dim),
        "packed_bits": False,
        "packed_dim": None,
        "n_ligands": int(n),
        "model_id": model_id,
        "pooling": pooling,
        "max_length": max_length,
        "elapsed_seconds": elapsed_s,
        "ligands_per_second": ligands_per_s,
        "invalid_smiles": int(invalid_smiles),
        "invalid_smiles_fraction": float(invalid_smiles / n),
        "embed_failures": int(embed_failures),
    }
    meta_path = reps_dir / f"{name}.meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def build_chemberta_representation(
    root: str | Path,
    n_bits: Optional[int] = 768,
    batch_size: int = 14,
    name: str = "chemberta_zinc_base_768",
    tokenizer=None,
    model=None,
    device: Optional[torch.device] = None,
    model_id: str = "seyonec/ChemBERTa-zinc-base-v1",
    max_length: Optional[int] = None,
) -> None:
    """Backward-compatible wrapper for the old API name."""
    build_huggingface_representation(
        root=root,
        n_bits=n_bits,
        batch_size=batch_size,
        name=name,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_id=model_id,
        max_length=max_length,
    )

# ---------------------------------------------------------------------------
# 4. Access representations by chem_comp_id: LigandStore & Representation
# ---------------------------------------------------------------------------

class Representation:
    """
    Numerical representation of ligands stored as a memmap on disk.

    Two access patterns:
    1) get_by_ids(): user/ML-friendly (may unpack bits to 0/1)
    2) raw access (new): backend-friendly (returns memmap rows as stored on disk)
       - packed_bits=True  -> returns uint8 packed bytes, shape (N, packed_dim)
       - packed_bits=False -> returns float16/float32/etc, shape (N, dim)
    """

    def __init__(
        self,
        name: str,
        memmap: np.memmap,
        meta: Dict,
        id_to_idx: Dict[str, int],
    ):
        self.name = name
        self.memmap = memmap
        self.meta = meta
        self.id_to_idx = id_to_idx

    # -----------------------------
    # Contract / metadata helpers
    # -----------------------------
    @property
    def dim(self) -> int:
        """Logical dimensionality (e.g. n_bits for Morgan, embedding dim for ChemBERTa)."""
        return int(self.meta["dim"])

    @property
    def packed_bits(self) -> bool:
        """Whether this representation is stored as packed bits on disk."""
        return bool(self.meta.get("packed_bits", False))

    @property
    def packed_dim(self) -> Optional[int]:
        """
        Physical dimensionality on disk for packed representations (bytes per vector).
        None for non-packed representations.
        """
        pdim = self.meta.get("packed_dim", None)
        return None if pdim is None else int(pdim)

    @property
    def dtype(self) -> np.dtype:
        """Numpy dtype of the underlying memmap."""
        return np.dtype(self.meta["dtype"])

    @property
    def n_ligands(self) -> int:
        """Number of ligands (rows) in the memmap."""
        return int(self.meta["n_ligands"])

    @property
    def raw_dim(self) -> int:
        """
        Physical last-dimension size in the memmap:
        - packed: packed_dim
        - non-packed: dim
        """
        return int(self.packed_dim) if self.packed_bits else int(self.dim)

    # -----------------------------
    # ID <-> index mapping
    # -----------------------------
    def indices_from_ids(self, comp_ids: List[str]) -> np.ndarray:
        """
        Convert a list of chem_comp_id strings into an array of integer indices (lig_idx).
        Raises KeyError if any chem_comp_id is not found.
        """
        idxs = []
        for cid in comp_ids:
            try:
                idxs.append(self.id_to_idx[cid])
            except KeyError:
                raise KeyError(f"chem_comp_id '{cid}' not found in ligand index.")
        return np.array(idxs, dtype=np.int64)

    def _indices_from_ids(self, comp_ids: List[str]) -> np.ndarray:
        # Backward compatibility: keep internal name used by existing code
        return self.indices_from_ids(comp_ids)

    # -----------------------------
    # RAW access (new)
    # -----------------------------
    def get_raw_by_indices(self, idxs: np.ndarray) -> np.ndarray:
        """
        Return rows exactly as stored on disk (no unpacking, no dtype conversion).
        Shape:
          - packed: (N, packed_dim) uint8
          - non-packed: (N, dim) float16/float32/...
        """
        idxs = np.asarray(idxs, dtype=np.int64)
        if idxs.size == 0:
            return np.zeros((0, self.raw_dim), dtype=self.dtype)
        return self.memmap[idxs]

    def get_raw_by_ids(self, comp_ids: List[str]) -> np.ndarray:
        """Same as get_raw_by_indices, but maps chem_comp_id -> lig_idx first."""
        if len(comp_ids) == 0:
            return np.zeros((0, self.raw_dim), dtype=self.dtype)
        idxs = self.indices_from_ids(comp_ids)
        return self.get_raw_by_indices(idxs)

    def iter_raw_chunks(
        self,
        chunk_size: int,
        start: int = 0,
        end: Optional[int] = None,
    ):
        """
        Iterate over the memmap in contiguous chunks, yielding:
            (start_idx, end_idx, raw_block)

        raw_block is a memmap slice with shape:
          - packed: (B, packed_dim)
          - non-packed: (B, dim)
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        n = self.n_ligands
        s = max(0, int(start))
        e = n if end is None else min(n, int(end))
        if s >= e:
            return
        for i in range(s, e, chunk_size):
            j = min(i + chunk_size, e)
            yield i, j, self.memmap[i:j]

    # -----------------------------
    # Existing high-level access (unchanged behavior)
    # -----------------------------
    def get_by_ids(
        self,
        comp_ids: List[str],
        as_float: bool = False,
    ) -> np.ndarray:
        """
        Retrieve the representation vectors for a list of comp_ids.

        For bit-packed representations (e.g. Morgan), this:
          - reads the packed rows from the memmap
          - unpacks them to 0/1 arrays of shape (n_ids, dim)

        Parameters
        ----------
        comp_ids : list of str
            Ligand IDs (final comp_id) to fetch.
        as_float : bool
            If True, convert the result to float32 (useful for ML models).
            If False, keep the native dtype (e.g. uint8 0/1 for bits).

        Returns
        -------
        np.ndarray
            Array of shape (len(comp_ids), dim).
        """
        if len(comp_ids) == 0:
            return np.zeros((0, self.dim), dtype=np.float32 if as_float else np.uint8)

        idxs = self._indices_from_ids(comp_ids)
        raw = self.memmap[idxs]  # (n_ids, dim_packed) or (n_ids, dim)

        if self.packed_bits:
            arr = unpack_bits(raw, self.dim)  # (n_ids, dim)
        else:
            arr = np.asarray(raw)

        if as_float:
            return arr.astype(np.float32)
        return arr


class LigandStore:
    """
    Simple manager for ligands and their numerical representations.

    Expected directory structure under `root`:

      root/
        ligands.parquet
        reps/
          <name>.dat
          <name>.meta.json

    The ligands.parquet file must contain:
      - 'chem_comp_id'
      - 'lig_idx'
    """

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root

        ligs_path = root / "ligands.parquet"
        if not ligs_path.exists():
            raise FileNotFoundError(f"ligands.parquet not found at {ligs_path}")

        self.ligands = pd.read_parquet(ligs_path)

        if "chem_comp_id" not in self.ligands.columns or "lig_idx" not in self.ligands.columns:
            raise ValueError("ligands.parquet must contain 'chem_comp_id' and 'lig_idx' columns.")

        # Map comp_id -> lig_idx for fast lookup
        self.id_to_idx: Dict[str, int] = dict(
            zip(self.ligands["chem_comp_id"], self.ligands["lig_idx"])
        )

    def load_representation(self, name: str) -> Representation:
        """
        Load a representation stored under root / 'reps'.

        The corresponding meta file (<name>.meta.json) defines:
          - dtype
          - dim
          - packed_bits
          - packed_dim (if packed_bits=True)
          - n_ligands

        Parameters
        ----------
        name : str
            Representation name (e.g. 'morgan_1024_r2').

        Returns
        -------
        Representation
            A Representation object bound to this LigandStore.
        """
        reps_dir = self.root / "reps"
        meta_path = reps_dir / f"{name}.meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata for representation '{name}' not found at {meta_path}"
            )

        with meta_path.open() as f:
            meta = json.load(f)

        data_path = reps_dir / meta["file"]
        dtype = np.dtype(meta["dtype"])

        if meta.get("packed_bits", False):
            shape = (meta["n_ligands"], meta["packed_dim"])
        else:
            shape = (meta["n_ligands"], meta["dim"])

        mm = np.memmap(
            data_path,
            mode="r",
            dtype=dtype,
            shape=shape,
        )

        return Representation(
            name=name,
            memmap=mm,
            meta=meta,
            id_to_idx=self.id_to_idx,
        )
