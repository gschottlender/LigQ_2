from fastapi import APIRouter

from services.hardware import HardwareCapabilities, get_hardware_capabilities


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/capabilities")
def system_capabilities() -> HardwareCapabilities:
    return get_hardware_capabilities()
