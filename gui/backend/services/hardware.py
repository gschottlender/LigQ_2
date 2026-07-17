from __future__ import annotations

import json
import subprocess
import sys
from functools import lru_cache
from typing import Optional

from pydantic import BaseModel


class HardwareCapabilities(BaseModel):
    cuda_available: bool
    cuda_device_name: Optional[str]


_CUDA_PROBE = r"""
import json
import torch

result = {"cuda_available": False, "cuda_device_name": None}
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        try:
            device = torch.device(f"cuda:{index}")
            probe = torch.empty((1,), device=device)
            probe.add_(1)
            torch.cuda.synchronize(device)
            result = {
                "cuda_available": True,
                "cuda_device_name": str(torch.cuda.get_device_name(device)),
            }
            break
        except Exception:
            continue
print(json.dumps(result))
"""


@lru_cache(maxsize=1)
def get_hardware_capabilities() -> HardwareCapabilities:
    """Probe CUDA in a short-lived process so the API keeps no GPU context."""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _CUDA_PROBE],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        payload = json.loads(completed.stdout.strip().splitlines()[-1])
        return HardwareCapabilities.model_validate(payload)
    except (IndexError, OSError, json.JSONDecodeError, subprocess.SubprocessError, ValueError):
        return HardwareCapabilities(cuda_available=False, cuda_device_name=None)
