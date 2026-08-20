from __future__ import annotations

import asyncio
import time
from typing import Any

from core.config import DATABASES_DIR
from core.policy import is_web_mode
from ligq_support.web_validation_receipt import inspect_web_validation_receipt


_cache_lock = asyncio.Lock()
_cached_at = 0.0
_cached_status: dict[str, Any] | None = None
_READY_CACHE_SECONDS = 30.0
_NOT_READY_CACHE_SECONDS = 30.0


async def inspect_web_readiness(*, force: bool = False) -> dict[str, Any]:
    global _cached_at, _cached_status
    if not is_web_mode():
        return {
            "ready": True,
            "mode": "local",
            "checks": {},
            "errors": [],
        }

    async with _cache_lock:
        now = time.monotonic()
        cache_seconds = (
            _READY_CACHE_SECONDS
            if _cached_status and _cached_status.get("ready")
            else _NOT_READY_CACHE_SECONDS
        )
        if (
            not force
            and _cached_status is not None
            and now - _cached_at < cache_seconds
        ):
            return dict(_cached_status)

        try:
            status = inspect_web_validation_receipt(DATABASES_DIR)
        except Exception as exc:
            detail = str(exc).strip() or type(exc).__name__
            status = {
                "ready": False,
                "mode": "web",
                "checks": {},
                "errors": [f"Web data receipt inspection failed: {detail}"],
            }

        _cached_status = status
        _cached_at = time.monotonic()
        return dict(status)


def clear_web_readiness_cache() -> None:
    global _cached_at, _cached_status
    _cached_at = 0.0
    _cached_status = None
