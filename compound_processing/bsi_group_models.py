from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


class BSIGroupMLP(nn.Module):
    """BSI group model over the symmetric fp1 + fp2 input transform."""

    def __init__(self, input_size: int, hidden_layers: Sequence[int], dropout: float) -> None:
        super().__init__()
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer")

        layers: list[nn.Module] = []
        prev = int(input_size)
        for hidden in hidden_layers:
            hidden = int(hidden)
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(float(dropout)))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(-1)


def build_group_mlp_from_params(params: dict) -> BSIGroupMLP:
    return BSIGroupMLP(
        input_size=int(params["fp_bits"]),
        hidden_layers=[int(x) for x in params["hidden_layers"]],
        dropout=float(params["dropout"]),
    )


def resolve_torch_device(device: str | torch.device = "auto") -> torch.device:
    if isinstance(device, str) and device.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return resolved


def load_group_model(
    model_path: str | Path,
    params_path: str | Path | None = None,
    device: str | torch.device = "auto",
) -> tuple[BSIGroupMLP, dict]:
    model_path = Path(model_path)
    if params_path is None:
        params_path = model_path.with_name("params.json")
        if not Path(params_path).exists():
            params_path = model_path.with_suffix(".params.json")
    params_path = Path(params_path)

    with open(params_path, "r") as handle:
        params = json.load(handle)

    resolved_device = resolve_torch_device(device)
    model = build_group_mlp_from_params(params)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(resolved_device)
    model.eval()
    return model, params


def ensure_dense_fp_matrix(fps: np.ndarray, fp_bits: int) -> np.ndarray:
    arr = np.asarray(fps)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] == fp_bits:
        return arr.astype(np.float32, copy=False)
    packed_dim = (int(fp_bits) + 7) // 8
    if arr.dtype == np.uint8 and arr.shape[1] == packed_dim:
        return np.unpackbits(arr, axis=-1)[:, :fp_bits].astype(np.float32, copy=False)
    raise ValueError(
        f"Expected dense shape (*, {fp_bits}) or packed uint8 shape (*, {packed_dim}); "
        f"got {arr.shape} {arr.dtype}"
    )


@torch.inference_mode()
def score_query_against_dense_chunk(
    model: BSIGroupMLP,
    query_fp: np.ndarray,
    candidate_fps: np.ndarray,
    batch_size: int = 65536,
    device: str | torch.device | None = None,
) -> np.ndarray:
    candidate_fps = np.asarray(candidate_fps)
    if candidate_fps.ndim != 2:
        raise ValueError("candidate_fps must be a 2D array")
    query = np.asarray(query_fp, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != candidate_fps.shape[1]:
        raise ValueError(f"query dim {query.shape[1]} does not match candidate dim {candidate_fps.shape[1]}")

    resolved_device = next(model.parameters()).device if device is None else torch.device(device)
    scores: list[np.ndarray] = []
    model.eval()
    for start in range(0, candidate_fps.shape[0], int(batch_size)):
        end = min(start + int(batch_size), candidate_fps.shape[0])
        x_np = candidate_fps[start:end].astype(np.float32, copy=False) + query
        x = torch.from_numpy(np.ascontiguousarray(x_np)).to(resolved_device)
        scores.append(torch.sigmoid(model(x)).detach().cpu().numpy().astype(np.float32, copy=False))
    if not scores:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(scores)
