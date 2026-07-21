from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from models.job import AddRepresentationRequest
from routers import jobs as jobs_router
from routers import system as system_router
from services import hardware
from services.hardware import HardwareCapabilities


def test_hardware_probe_parses_usable_cuda_device(monkeypatch) -> None:
    class CompletedProbe:
        stdout = '{"cuda_available": true, "cuda_device_name": "Test GPU"}\n'

    monkeypatch.setattr(hardware.subprocess, "run", lambda *_args, **_kwargs: CompletedProbe())
    hardware.get_hardware_capabilities.cache_clear()

    capabilities = hardware.get_hardware_capabilities()

    assert capabilities.cuda_available is True
    assert capabilities.cuda_device_name == "Test GPU"
    hardware.get_hardware_capabilities.cache_clear()


def test_capabilities_endpoint_reports_backend_probe(monkeypatch) -> None:
    expected = HardwareCapabilities(cuda_available=True, cuda_device_name="Test GPU")
    monkeypatch.setattr(system_router, "get_hardware_capabilities", lambda: expected)

    response = system_router.system_capabilities()

    assert response == expected


def test_gui_rejects_huggingface_generation_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        jobs_router,
        "get_hardware_capabilities",
        lambda: HardwareCapabilities(cuda_available=False, cuda_device_name=None),
    )

    response = asyncio.run(
        jobs_router.add_representation(
            AddRepresentationRequest(
                base_name="zinc",
                representation_type="huggingface",
                rep_name="chemberta_zinc_base_768",
                n_bits=768,
                model_id="seyonec/ChemBERTa-zinc-base-v1",
                batch_size=14,
            )
        )
    )

    assert response.status_code == 422
    assert json.loads(response.body)["error"] == "gpu_required"
