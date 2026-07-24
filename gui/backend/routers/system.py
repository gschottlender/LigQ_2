from fastapi import APIRouter

from core.policy import policy_payload
from services.hardware import HardwareCapabilities, get_hardware_capabilities
from services.web_readiness import inspect_web_readiness


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/capabilities")
def system_capabilities() -> HardwareCapabilities:
    return get_hardware_capabilities()


@router.get("/policy")
def system_policy() -> dict:
    return policy_payload()


@router.get("/readiness")
async def system_readiness() -> dict:
    return await inspect_web_readiness()
