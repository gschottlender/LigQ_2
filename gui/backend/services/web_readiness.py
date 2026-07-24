from __future__ import annotations

import asyncio
import json
import sys
import time
from typing import Any

from core.config import DATABASES_DIR, PIPELINE_ROOT
from core.policy import is_web_mode


_cache_lock = asyncio.Lock()
_cached_at = 0.0
_cached_status: dict[str, Any] | None = None
_READY_CACHE_SECONDS = 3600.0
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

        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "ligq_support.validate_web_data",
            "--data-dir",
            str(DATABASES_DIR),
            "--json",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PIPELINE_ROOT),
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=120)
            lines = [
                line
                for line in stdout.decode("utf-8", errors="replace").splitlines()
                if line.strip()
            ]
            status = json.loads(lines[-1])
            if process.returncode not in {0, 1}:
                raise RuntimeError(
                    stderr.decode("utf-8", errors="replace").strip()
                    or f"Validator exited with code {process.returncode}."
                )
        except Exception as exc:
            if process.returncode is None:
                process.kill()
                await process.wait()
            status = {
                "ready": False,
                "mode": "web",
                "checks": {},
                "errors": [f"Web data validation failed: {exc}"],
            }

        _cached_status = status
        _cached_at = time.monotonic()
        return dict(status)


def clear_web_readiness_cache() -> None:
    global _cached_at, _cached_status
    _cached_at = 0.0
    _cached_status = None
