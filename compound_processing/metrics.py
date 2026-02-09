"""
compound_helpers.metrics

Small, self-contained similarity "kernels" that are reusable from zinc_search.py (or any other
search loop). This module does NOT implement chunking, multiprocessing, hit collection, etc.
It only computes score matrices between a query block Q and a target block X.

Supported metrics:
- "tanimoto": for binary fingerprints (Morgan). Works with:
    * packed bits (uint8, shape (N, packed_dim))  [preferred for large DB scans]
    * dense bits  (uint8/bool 0/1, shape (N, dim)) [debug / legacy]
- "cosine": for float embeddings (ChemBERTa / HuggingFace). Works with float16/float32 arrays.

Device:
- device="cpu": NumPy implementation
- device="cuda" (or torch.device("cuda")): Torch implementation (requires torch installed)


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


MetricName = Literal["tanimoto", "cosine"]


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------

class MetricValidationError(ValueError):
    """Raised when a metric is requested for incompatible representations."""


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _as_device(device: Union[str, "torch.device"]) -> str:
    if isinstance(device, str):
        d = device.lower()
        if d.startswith("cuda"):
            return "cuda"
        return "cpu"
    # torch.device
    if str(device).startswith("cuda"):
        return "cuda"
    return "cpu"


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for device='cuda' paths, but torch is not available.")


def _meta_get(meta: Optional[Dict[str, Any]], key: str, default: Any = None) -> Any:
    if meta is None:
        return default
    return meta.get(key, default)


# -----------------------------------------------------------------------------
# Validation (meta as source of truth)
# -----------------------------------------------------------------------------

def validate_metric(metric: MetricName, q_meta: Dict[str, Any], x_meta: Dict[str, Any]) -> None:
    """
    Validate that a metric can be computed between a query representation and a target representation.

    Parameters
    ----------
    metric
        "tanimoto" or "cosine".
    q_meta, x_meta
        Representation meta dicts (the ones saved to <name>.meta.json).

    Raises
    ------
    MetricValidationError
        If reps are incompatible for the requested metric.
    """
    if metric not in ("tanimoto", "cosine"):
        raise MetricValidationError(f"Unknown metric '{metric}'")

    q_packed = bool(q_meta.get("packed_bits", False))
    x_packed = bool(x_meta.get("packed_bits", False))

    if metric == "tanimoto":
        # We only support tanimoto for binary representations.
        # Preferred: packed bytes (uint8). Dense 0/1 is allowed as a fallback.
        if q_packed != x_packed:
            raise MetricValidationError(
                "Tanimoto requires both query and target to have the same packed_bits setting "
                f"(query packed_bits={q_packed}, target packed_bits={x_packed})."
            )

        q_dim = int(q_meta.get("dim"))
        x_dim = int(x_meta.get("dim"))
        if q_dim != x_dim:
            raise MetricValidationError(f"Tanimoto requires same dim: query dim={q_dim}, target dim={x_dim}.")

        if q_packed:
            q_pd = int(q_meta.get("packed_dim"))
            x_pd = int(x_meta.get("packed_dim"))
            if q_pd != x_pd:
                raise MetricValidationError(
                    f"Tanimoto packed requires same packed_dim: query {q_pd}, target {x_pd}."
                )
            # dtype should be uint8 for packed
            q_dtype = str(q_meta.get("dtype"))
            x_dtype = str(x_meta.get("dtype"))
            if q_dtype != "uint8" or x_dtype != "uint8":
                raise MetricValidationError(
                    f"Tanimoto packed expects dtype 'uint8'. Got query={q_dtype}, target={x_dtype}."
                )

    elif metric == "cosine":
        # Cosine expects float vectors, not packed bits
        if q_packed or x_packed:
            raise MetricValidationError(
                "Cosine requires float embeddings (packed_bits=False). "
                f"Got query packed_bits={q_packed}, target packed_bits={x_packed}."
            )
        q_dim = int(q_meta.get("dim"))
        x_dim = int(x_meta.get("dim"))
        if q_dim != x_dim:
            raise MetricValidationError(f"Cosine requires same dim: query dim={q_dim}, target dim={x_dim}.")
        q_dtype = str(q_meta.get("dtype"))
        x_dtype = str(x_meta.get("dtype"))
        allowed = {"float16", "float32"}
        if q_dtype not in allowed or x_dtype not in allowed:
            raise MetricValidationError(
                f"Cosine expects dtype in {sorted(allowed)}. Got query={q_dtype}, target={x_dtype}."
            )


# -----------------------------------------------------------------------------
# CPU implementations
# -----------------------------------------------------------------------------

# Global popcount LUT for bytes [0..255]
_POPCOUNT_LUT = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)


def popcount_packed_u8_cpu(packed: np.ndarray) -> np.ndarray:
    """
    Count set bits for each row of a packed uint8 matrix.

    Parameters
    ----------
    packed : np.ndarray
        shape (N, packed_dim), dtype uint8.

    Returns
    -------
    np.ndarray
        shape (N,), dtype int32.
    """
    if packed.dtype != np.uint8:
        packed = packed.astype(np.uint8, copy=False)
    # LUT indexing returns uint8; sum -> int64, cast down
    return _POPCOUNT_LUT[packed].sum(axis=1).astype(np.int32)


def tanimoto_packed_cpu(
    q_packed: np.ndarray,
    x_packed: np.ndarray,
    *,
    q_counts: Optional[np.ndarray] = None,
    x_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute Tanimoto between query packed fingerprints and target packed fingerprints on CPU.

    Returns a (Q, X) float32 matrix.

    Notes:
    - This avoids unpacking to dense (0/1).
    - Implementation loops over queries to avoid allocating a huge (Q, X, packed_dim) array.
    """
    if q_packed.dtype != np.uint8:
        q_packed = q_packed.astype(np.uint8, copy=False)
    if x_packed.dtype != np.uint8:
        x_packed = x_packed.astype(np.uint8, copy=False)

    Q = q_packed.shape[0]
    X = x_packed.shape[0]
    if Q == 0 or X == 0:
        return np.zeros((Q, X), dtype=np.float32)

    if q_counts is None:
        q_counts = popcount_packed_u8_cpu(q_packed)
    else:
        q_counts = q_counts.astype(np.int32, copy=False)

    if x_counts is None:
        x_counts = popcount_packed_u8_cpu(x_packed)
    else:
        x_counts = x_counts.astype(np.int32, copy=False)

    out = np.zeros((Q, X), dtype=np.float32)

    # For each query, compute intersection popcount with all X rows:
    # inter[j] = popcount( q & x_j )
    for qi in range(Q):
        q_row = q_packed[qi]  # (packed_dim,)
        inter = _POPCOUNT_LUT[np.bitwise_and(x_packed, q_row)].sum(axis=1).astype(np.int32)  # (X,)
        denom = q_counts[qi] + x_counts - inter
        valid = denom > 0
        # float division
        out_q = out[qi]
        out_q[valid] = inter[valid].astype(np.float32) / denom[valid].astype(np.float32)
        # else remains 0.0
    return out


def tanimoto_dense_cpu(
    q_bits: np.ndarray,
    x_bits: np.ndarray,
    *,
    q_counts: Optional[np.ndarray] = None,
    x_counts: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    CPU Tanimoto for dense 0/1 fingerprints (legacy/debug path).
    q_bits: (Q, dim), uint8/bool 0/1
    x_bits: (X, dim), uint8/bool 0/1
    Returns (Q, X) float32
    """
    if q_bits.size == 0 or x_bits.size == 0:
        return np.zeros((q_bits.shape[0], x_bits.shape[0]), dtype=np.float32)

    q_u8 = q_bits.astype(np.uint8, copy=False)
    x_u8 = x_bits.astype(np.uint8, copy=False)

    if q_counts is None:
        q_counts = q_u8.sum(axis=1).astype(np.int32)
    else:
        q_counts = q_counts.astype(np.int32, copy=False)

    if x_counts is None:
        x_counts = x_u8.sum(axis=1).astype(np.int32)
    else:
        x_counts = x_counts.astype(np.int32, copy=False)

    # intersections: (Q, X)
    inter = q_u8 @ x_u8.T  # uint8 dot -> intersections
    denom = q_counts[:, None] + x_counts[None, :] - inter
    valid = denom > 0
    out = np.zeros_like(inter, dtype=np.float32)
    out[valid] = inter[valid].astype(np.float32) / denom[valid].astype(np.float32)
    return out


def cosine_cpu(
    q: np.ndarray,
    x: np.ndarray,
    *,
    assume_normalized: Optional[bool] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Cosine similarity on CPU.

    If assume_normalized is True, computes dot-product only.
    If False, normalizes q and x on the fly.
    If None, tries to infer from dtype and typical meta usage (but you should pass it from meta).
    """
    if q.size == 0 or x.size == 0:
        return np.zeros((q.shape[0], x.shape[0]), dtype=np.float32)

    qf = q.astype(np.float32, copy=False)
    xf = x.astype(np.float32, copy=False)

    if assume_normalized is False or assume_normalized is None:
        # Normalize rows
        qn = np.linalg.norm(qf, axis=1, keepdims=True)
        xn = np.linalg.norm(xf, axis=1, keepdims=True)
        qf = qf / np.clip(qn, eps, None)
        xf = xf / np.clip(xn, eps, None)

    # dot product
    return (qf @ xf.T).astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# GPU (torch) implementations
# -----------------------------------------------------------------------------

# Cache unpack LUT per device (copied from zinc_search.py)
_UNPACK_LUT_CACHE: Dict["torch.device", "torch.Tensor"] = {}


def _get_unpack_lut(device: "torch.device") -> "torch.Tensor":
    _require_torch()
    lut = _UNPACK_LUT_CACHE.get(device)
    if lut is not None:
        return lut
    vals = torch.arange(256, dtype=torch.int16)
    shifts = torch.arange(7, -1, -1, dtype=torch.int16)
    bits = ((vals[:, None] >> shifts[None, :]) & 1).to(torch.uint8)
    lut = bits.to(device=device, non_blocking=True)
    _UNPACK_LUT_CACHE[device] = lut
    return lut


def unpack_bits_torch_from_packed(
    packed_u8: np.ndarray,
    n_bits: int,
    device: "torch.device",
    *,
    force_copy: bool = True,
) -> "torch.Tensor":
    """
    Packed uint8 (B, packed_dim) -> dense bits (B, n_bits) uint8 on device.
    Copied from zinc_search.py with minimal changes.
    """
    _require_torch()
    if packed_u8.dtype != np.uint8:
        packed_u8 = packed_u8.astype(np.uint8, copy=False)
    if force_copy:
        packed_u8 = np.array(packed_u8, copy=True, order="C")
    t = torch.from_numpy(packed_u8).to(device=device, non_blocking=True)  # (B, packed_dim) uint8
    lut = _get_unpack_lut(device)  # (256, 8) uint8
    bits = lut[t.long()]           # (B, packed_dim, 8)
    bits = bits.reshape(bits.shape[0], -1)  # (B, packed_dim*8)
    if bits.shape[1] > n_bits:
        bits = bits[:, :n_bits]
    return bits


def tanimoto_dense_torch(
    q_bits: "torch.Tensor",
    x_bits: "torch.Tensor",
    q_counts: "torch.Tensor",
    x_counts: "torch.Tensor",
) -> "torch.Tensor":
    """
    Compute Tanimoto on GPU for dense 0/1 bit matrices.
    q_bits: (Q, dim) uint8
    x_bits: (X, dim) uint8
    q_counts: (Q,) int32
    x_counts: (X,) int32
    returns (Q, X) float32
    """
    _require_torch()
    # matmul in float32 for stability; bits as float32 for intersection
    inter = torch.matmul(q_bits.to(torch.float32), x_bits.to(torch.float32).T)  # (Q, X)
    denom = q_counts[:, None].to(torch.float32) + x_counts[None, :].to(torch.float32) - inter
    # avoid 0 division
    ti = torch.zeros_like(inter, dtype=torch.float32)
    mask = denom > 0
    ti[mask] = inter[mask] / denom[mask]
    return ti


def tanimoto_packed_torch(
    q_packed: np.ndarray,
    x_packed: np.ndarray,
    *,
    n_bits: int,
    device: Union[str, "torch.device"] = "cuda",
    force_copy: bool = True,
) -> np.ndarray:
    """
    GPU Tanimoto for packed fingerprints using "unpack + matmul" strategy (same as zinc_search.py).
    Returns numpy float32 (Q, X).
    """
    _require_torch()
    dev = torch.device(device)
    q_bits = unpack_bits_torch_from_packed(q_packed, n_bits=n_bits, device=dev, force_copy=force_copy)
    x_bits = unpack_bits_torch_from_packed(x_packed, n_bits=n_bits, device=dev, force_copy=force_copy)
    q_counts = q_bits.sum(dim=1).to(torch.int32)
    x_counts = x_bits.sum(dim=1).to(torch.int32)
    ti = tanimoto_dense_torch(q_bits, x_bits, q_counts, x_counts)
    return ti.detach().cpu().numpy().astype(np.float32, copy=False)


def tanimoto_packed_torch_tensor(
    q_packed: np.ndarray,
    x_packed: np.ndarray,
    *,
    n_bits: int,
    device: Union[str, "torch.device"] = "cuda",
    force_copy: bool = True,
) -> "torch.Tensor":
    """
    GPU Tanimoto for packed fingerprints, returning a torch.Tensor on device.
    """
    _require_torch()
    dev = torch.device(device)
    q_bits = unpack_bits_torch_from_packed(q_packed, n_bits=n_bits, device=dev, force_copy=force_copy)
    x_bits = unpack_bits_torch_from_packed(x_packed, n_bits=n_bits, device=dev, force_copy=force_copy)
    q_counts = q_bits.sum(dim=1).to(torch.int32)
    x_counts = x_bits.sum(dim=1).to(torch.int32)
    return tanimoto_dense_torch(q_bits, x_bits, q_counts, x_counts)


def cosine_torch(
    q: np.ndarray,
    x: np.ndarray,
    *,
    device: Union[str, "torch.device"] = "cuda",
    assume_normalized: Optional[bool] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    GPU cosine similarity using torch matmul. Returns numpy float32 (Q, X).
    """
    _require_torch()
    dev = torch.device(device)
    qt = torch.as_tensor(q, device=dev)
    xt = torch.as_tensor(x, device=dev)

    qt = qt.to(torch.float32)
    xt = xt.to(torch.float32)

    if assume_normalized is False or assume_normalized is None:
        qn = torch.linalg.norm(qt, dim=1, keepdim=True).clamp_min(eps)
        xn = torch.linalg.norm(xt, dim=1, keepdim=True).clamp_min(eps)
        qt = qt / qn
        xt = xt / xn

    sim = torch.matmul(qt, xt.T).to(torch.float32)
    return sim.detach().cpu().numpy().astype(np.float32, copy=False)


def cosine_torch_tensor(
    q: np.ndarray,
    x: np.ndarray,
    *,
    device: Union[str, "torch.device"] = "cuda",
    assume_normalized: Optional[bool] = None,
    eps: float = 1e-12,
) -> "torch.Tensor":
    """
    GPU cosine similarity using torch matmul. Returns torch.Tensor on device.
    """
    _require_torch()
    dev = torch.device(device)
    qt = torch.as_tensor(q, device=dev).to(torch.float32)
    xt = torch.as_tensor(x, device=dev).to(torch.float32)

    if assume_normalized is False or assume_normalized is None:
        qn = torch.linalg.norm(qt, dim=1, keepdim=True).clamp_min(eps)
        xn = torch.linalg.norm(xt, dim=1, keepdim=True).clamp_min(eps)
        qt = qt / qn
        xt = xt / xn

    return torch.matmul(qt, xt.T).to(torch.float32)


# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------

def score_block(
    metric: MetricName,
    Q: np.ndarray,
    X: np.ndarray,
    *,
    q_meta: Optional[Dict[str, Any]] = None,
    x_meta: Optional[Dict[str, Any]] = None,
    device: Union[str, "torch.device"] = "cpu",
    assume_normalized: Optional[bool] = None,
    force_copy_packed_gpu: bool = True,
    return_torch: bool = False,
) -> np.ndarray:
    """
    Compute a score matrix between a query block Q and target block X.

    Parameters
    ----------
    metric
        "tanimoto" or "cosine"
    Q, X
        Arrays containing either:
          - packed uint8 fingerprints (preferred for tanimoto)
          - dense 0/1 fingerprints (legacy tanimoto)
          - float embeddings (cosine)
    q_meta, x_meta
        Optional meta dicts. If provided, used for validation + picking packed vs dense logic.
    device
        "cpu" or "cuda" (or torch.device)
    assume_normalized
        Only for cosine: if True, dot-product only; if False normalize on the fly.
        If None, and meta provided, we will attempt to read meta["normalized"] (default False).
    force_copy_packed_gpu
        When computing packed tanimoto on GPU via unpack-LUT, memmap slices are often non-writable.
        For compatibility, we can copy to contiguous arrays before torch.from_numpy.

    Returns
    -------
    np.ndarray
        Score matrix of shape (Q, X) float32.

    Notes
    -----
    This function is intentionally "pure" with respect to searching strategy.
    It does not apply thresholds, topk, etc.
    """
    dev = _as_device(device)

    if q_meta is not None and x_meta is not None:
        validate_metric(metric, q_meta, x_meta)

    if metric == "tanimoto":
        # Decide packed vs dense
        packed = bool(_meta_get(q_meta, "packed_bits", False)) if q_meta is not None else (Q.dtype == np.uint8 and X.dtype == np.uint8 and Q.ndim == 2 and X.ndim == 2 and Q.shape[1] < _meta_get(q_meta, "dim", 10**9))
        # In practice: if meta says packed_bits=True, use packed path.
        if q_meta is not None:
            packed = bool(q_meta.get("packed_bits", False))

        if dev == "cpu":
            if packed:
                return tanimoto_packed_cpu(Q, X)
            return tanimoto_dense_cpu(Q, X)

        # GPU path
        _require_torch()
        n_bits = int(_meta_get(q_meta, "dim", None))
        if n_bits is None:
            raise ValueError("GPU tanimoto requires q_meta['dim'] to be provided (number of bits).")
        if packed:
            if return_torch:
                return tanimoto_packed_torch_tensor(
                    Q, X, n_bits=n_bits, device=device, force_copy=force_copy_packed_gpu
                )
            return tanimoto_packed_torch(Q, X, n_bits=n_bits, device=device, force_copy=force_copy_packed_gpu)

        # Dense bits on GPU: accept numpy and move to torch inside
        dev_t = torch.device(device)
        q_bits = torch.as_tensor(Q.astype(np.uint8, copy=False), device=dev_t)
        x_bits = torch.as_tensor(X.astype(np.uint8, copy=False), device=dev_t)
        q_counts = q_bits.sum(dim=1).to(torch.int32)
        x_counts = x_bits.sum(dim=1).to(torch.int32)
        ti = tanimoto_dense_torch(q_bits, x_bits, q_counts, x_counts)
        if return_torch:
            return ti
        return ti.detach().cpu().numpy().astype(np.float32, copy=False)

    # cosine
    if assume_normalized is None and q_meta is not None:
        assume_normalized = bool(q_meta.get("normalized", False))

    if dev == "cpu":
        return cosine_cpu(Q, X, assume_normalized=assume_normalized)

    _require_torch()
    if return_torch:
        return cosine_torch_tensor(Q, X, device=device, assume_normalized=assume_normalized)
    return cosine_torch(Q, X, device=device, assume_normalized=assume_normalized)


def select_hits(
    scores: Union[np.ndarray, "torch.Tensor"],
    *,
    mode: Literal["threshold", "topk"],
    threshold: Optional[float] = None,
    topk: Optional[int] = None,
) -> Tuple[Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"], Union[np.ndarray, "torch.Tensor"]]:
    """
    Select hits from a score matrix using either thresholding or top-k per query.
    Returns (q_idx, x_idx, vals) on the same device/type as scores.
    """
    if mode not in ("threshold", "topk"):
        raise ValueError(f"Unknown mode '{mode}'.")

    if isinstance(scores, np.ndarray):
        if mode == "threshold":
            if threshold is None:
                raise ValueError("threshold must be provided for mode='threshold'.")
            mask = scores >= float(threshold)
            q_idx, x_idx = np.where(mask)
            return q_idx, x_idx, scores[q_idx, x_idx]

        if topk is None or topk <= 0:
            raise ValueError("topk must be provided and > 0 for mode='topk'.")
        if scores.size == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        k = min(topk, scores.shape[1])
        idx = np.argpartition(scores, -k, axis=1)[:, -k:]
        row_scores = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(row_scores, axis=1)[:, ::-1]
        idx = np.take_along_axis(idx, order, axis=1)
        row_scores = np.take_along_axis(row_scores, order, axis=1)
        if threshold is not None:
            keep = row_scores >= float(threshold)
            q_idx, k_idx = np.where(keep)
            return q_idx, idx[q_idx, k_idx], row_scores[q_idx, k_idx]
        q_idx, k_idx = np.where(np.ones_like(row_scores, dtype=bool))
        return q_idx, idx[q_idx, k_idx], row_scores[q_idx, k_idx]

    # torch tensor
    _require_torch()
    if mode == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided for mode='threshold'.")
        mask = scores >= float(threshold)
        idx = torch.nonzero(mask, as_tuple=False)
        if idx.numel() == 0:
            empty = scores.new_empty((0,), dtype=torch.int64)
            return empty, empty, scores.new_empty((0,), dtype=scores.dtype)
        vals = scores[idx[:, 0], idx[:, 1]]
        return idx[:, 0], idx[:, 1], vals

    if topk is None or topk <= 0:
        raise ValueError("topk must be provided and > 0 for mode='topk'.")
    k = min(topk, scores.shape[1])
    topv, topi = torch.topk(scores, k=k, dim=1)
    if threshold is not None:
        keep = topv >= float(threshold)
        idx = torch.nonzero(keep, as_tuple=False)
        if idx.numel() == 0:
            empty = scores.new_empty((0,), dtype=torch.int64)
            return empty, empty, scores.new_empty((0,), dtype=scores.dtype)
        vals = topv[idx[:, 0], idx[:, 1]]
        return idx[:, 0], topi[idx[:, 0], idx[:, 1]], vals
    q_idx = torch.arange(scores.shape[0], device=scores.device).unsqueeze(1).expand_as(topi).reshape(-1)
    x_idx = topi.reshape(-1)
    vals = topv.reshape(-1)
    return q_idx, x_idx, vals
