from __future__ import annotations

from functools import lru_cache
from typing import Union

import torch


@lru_cache(maxsize=16)
def cuda_device_is_usable(device: str = "cuda") -> bool:
    """Return True only if CUDA is visible and can run a trivial torch op."""
    try:
        if not torch.cuda.is_available():
            return False
        dev = torch.device(device)
        if dev.type != "cuda":
            return False
        _ = torch.cuda.get_device_properties(dev)
        probe = torch.empty((1,), device=dev)
        probe.add_(1)
        torch.cuda.synchronize(dev)
        return True
    except Exception:
        return False


def resolve_torch_device(device: Union[str, torch.device, None] = "auto") -> torch.device:
    """Resolve auto/cpu/cuda requests without selecting an unusable CUDA device."""
    if device is None or (isinstance(device, str) and device.lower() == "auto"):
        return torch.device("cuda" if cuda_device_is_usable("cuda") else "cpu")

    resolved = device if isinstance(device, torch.device) else torch.device(device)
    if resolved.type == "cuda" and not cuda_device_is_usable(str(resolved)):
        return torch.device("cpu")
    return resolved
