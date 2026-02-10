from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple, Union, List, Dict

import numpy as np
import pandas as pd
import torch

from compound_processing.compound_helpers import LigandStore, Representation
from compound_processing import metrics


@dataclass(frozen=True)
class SearchRequest:
    query_ids: Sequence[str]
    store_ref: LigandStore
    rep_ref: Representation
    store_target: LigandStore
    rep_target: Representation
    metric: metrics.MetricName
    mode: str = "threshold"
    threshold: Optional[float] = 0.5
    topk: Optional[int] = None
    device: Union[str, torch.device] = "auto"
    q_batch_size: int = 200
    target_chunk_size: int = 200_000
    max_hits_per_query: Optional[int] = None
    assume_normalized: Optional[bool] = None
    return_fields: Sequence[str] = field(default_factory=lambda: ("chem_comp_id", "smiles"))
    id_field: str = "chem_comp_id"
    prefix: Optional[str] = None


class TargetStoreAdapter:
    def __init__(
        self,
        store: LigandStore,
        *,
        id_field: str,
        return_fields: Sequence[str],
        prefix: Optional[str] = None,
    ) -> None:
        self.store = store
        self.id_field = id_field
        self.return_fields = tuple(return_fields)
        self.prefix = prefix

        ligs = store.ligands.copy()
        if "lig_idx" not in ligs.columns:
            raise ValueError("Target ligands parquet must contain column 'lig_idx'.")
        self._ligs_by_idx = ligs.set_index("lig_idx")

    def merge_metadata(self, hits: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in self.return_fields if c in self._ligs_by_idx.columns]
        if not cols:
            return hits
        merged = hits.merge(
            self._ligs_by_idx[cols],
            left_on="target_idx",
            right_index=True,
            how="left",
        )
        if self.prefix and self.id_field in merged.columns:
            merged[self.id_field] = self.prefix + merged[self.id_field].astype(str)
        return merged


class MetricKernel:
    def __init__(
        self,
        metric: metrics.MetricName,
        *,
        q_meta: Dict,
        x_meta: Dict,
        device: Union[str, torch.device],
        assume_normalized: Optional[bool],
        force_copy_packed_gpu: bool = True,
    ) -> None:
        metrics.validate_metric(metric, q_meta, x_meta)
        self.metric = metric
        self.q_meta = q_meta
        self.x_meta = x_meta
        self.device = device
        self.assume_normalized = assume_normalized
        self.force_copy_packed_gpu = force_copy_packed_gpu

    def score_block(
        self,
        q_block: np.ndarray,
        x_block: np.ndarray,
        *,
        return_torch: bool,
    ) -> Union[np.ndarray, torch.Tensor]:
        return metrics.score_block(
            self.metric,
            q_block,
            x_block,
            q_meta=self.q_meta,
            x_meta=self.x_meta,
            device=self.device,
            assume_normalized=self.assume_normalized,
            force_copy_packed_gpu=self.force_copy_packed_gpu,
            return_torch=return_torch,
        )

    def select_hits(
        self,
        scores: Union[np.ndarray, torch.Tensor],
        *,
        mode: str,
        threshold: Optional[float],
        topk: Optional[int],
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor]]:
        return metrics.select_hits(scores, mode=mode, threshold=threshold, topk=topk)


def _resolve_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, torch.device):
        resolved = device
    else:
        resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def _merge_topk(
    num_queries: int,
    q_idx: np.ndarray,
    x_idx: np.ndarray,
    scores: np.ndarray,
    *,
    topk: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if q_idx.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )
    best_scores = np.full((num_queries, topk), -np.inf, dtype=np.float32)
    best_idx = np.full((num_queries, topk), -1, dtype=np.int64)

    for qi in np.unique(q_idx):
        mask = q_idx == qi
        vals = scores[mask]
        idxs = x_idx[mask]
        if vals.size == 0:
            continue
        merged_scores = np.concatenate([best_scores[qi], vals])
        merged_indices = np.concatenate([best_idx[qi], idxs])
        if merged_scores.size <= topk:
            order = np.argsort(merged_scores)[::-1]
        else:
            best = np.argpartition(merged_scores, -topk)[-topk:]
            order = best[np.argsort(merged_scores[best])[::-1]]
        best_scores[qi] = merged_scores[order][:topk]
        best_idx[qi] = merged_indices[order][:topk]

    out_q: List[np.ndarray] = []
    out_x: List[np.ndarray] = []
    out_s: List[np.ndarray] = []
    for qi in range(num_queries):
        valid = best_idx[qi] >= 0
        if not np.any(valid):
            continue
        out_q.append(np.repeat(qi, valid.sum()))
        out_x.append(best_idx[qi][valid])
        out_s.append(best_scores[qi][valid])
    if not out_q:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float32),
        )
    return (
        np.concatenate(out_q),
        np.concatenate(out_x),
        np.concatenate(out_s),
    )


def search(request: SearchRequest) -> pd.DataFrame:
    if request.mode not in ("threshold", "topk"):
        raise ValueError("mode must be 'threshold' or 'topk'.")
    if request.mode == "threshold" and request.threshold is None:
        raise ValueError("threshold must be provided for mode='threshold'.")
    if request.mode == "topk" and (request.topk is None or request.topk <= 0):
        raise ValueError("topk must be provided and > 0 for mode='topk'.")

    score_col = "tanimoto" if request.metric == "tanimoto" else "similarity"
    if not request.query_ids:
        return pd.DataFrame(columns=["query_id", "target_idx", score_col, *request.return_fields])

    resolved_device = _resolve_device(request.device)
    device_str = "cuda" if resolved_device.type == "cuda" else "cpu"

    qids = list(request.query_ids)
    q_raw = request.rep_ref.get_raw_by_ids(qids)
    if q_raw.shape[0] != len(qids):
        raise ValueError("Number of query vectors does not match number of query_ids.")

    adapter = TargetStoreAdapter(
        request.store_target,
        id_field=request.id_field,
        return_fields=request.return_fields,
        prefix=request.prefix,
    )

    kernel = MetricKernel(
        request.metric,
        q_meta=request.rep_ref.meta,
        x_meta=request.rep_target.meta,
        device=device_str,
        assume_normalized=request.assume_normalized,
    )

    query_id_out: List[np.ndarray] = []
    target_idx_out: List[np.ndarray] = []
    score_out: List[np.ndarray] = []

    for q_start in range(0, len(qids), request.q_batch_size):
        q_end = min(q_start + request.q_batch_size, len(qids))
        batch_q = q_raw[q_start:q_end]
        batch_ids = qids[q_start:q_end]

        batch_q_idx: List[np.ndarray] = []
        batch_x_idx: List[np.ndarray] = []
        batch_scores: List[np.ndarray] = []

        for chunk_start, _chunk_end, x_raw in request.rep_target.iter_raw_chunks(request.target_chunk_size):
            if x_raw.size == 0:
                continue

            scores = kernel.score_block(batch_q, x_raw, return_torch=(resolved_device.type == "cuda"))
            q_idx, x_idx, vals = kernel.select_hits(
                scores,
                mode=request.mode,
                threshold=request.threshold,
                topk=request.topk,
            )

            if resolved_device.type == "cuda":
                q_idx = q_idx.detach().cpu().numpy()
                x_idx = x_idx.detach().cpu().numpy()
                vals = vals.detach().cpu().numpy()

            if q_idx.size == 0:
                continue

            batch_q_idx.append(q_idx.astype(np.int64, copy=False))
            batch_x_idx.append((x_idx + chunk_start).astype(np.int64, copy=False))
            batch_scores.append(vals.astype(np.float32, copy=False))

        if not batch_q_idx:
            continue

        q_idx_all = np.concatenate(batch_q_idx)
        x_idx_all = np.concatenate(batch_x_idx)
        scores_all = np.concatenate(batch_scores)

        if request.mode == "topk" and request.topk is not None:
            q_idx_all, x_idx_all, scores_all = _merge_topk(
                len(batch_ids),
                q_idx_all,
                x_idx_all,
                scores_all,
                topk=request.topk,
            )

        if q_idx_all.size == 0:
            continue

        query_id_out.append(np.asarray(batch_ids, dtype=object)[q_idx_all])
        target_idx_out.append(x_idx_all)
        score_out.append(scores_all)

    if not query_id_out:
        return pd.DataFrame(columns=["query_id", "target_idx", score_col, *request.return_fields])

    hits_df = pd.DataFrame(
        {
            "query_id": np.concatenate(query_id_out),
            "target_idx": np.concatenate(target_idx_out),
            score_col: np.concatenate(score_out),
        }
    )

    hits_df = adapter.merge_metadata(hits_df)

    if request.max_hits_per_query is not None:
        hits_df = (
            hits_df.sort_values(["query_id", score_col], ascending=[True, False])
            .groupby("query_id", as_index=False)
            .head(request.max_hits_per_query)
            .reset_index(drop=True)
        )

    return hits_df
