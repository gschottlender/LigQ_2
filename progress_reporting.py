from __future__ import annotations

import json
import time
from typing import Any


PROGRESS_PREFIX = "LIGQ_PROGRESS "


class ProgressEmitter:
    """Emit newline-delimited progress events for GUI subprocess consumers."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        self._step = ""
        self._step_started_at = time.monotonic()

    def emit(
        self,
        *,
        step: str,
        label: str,
        step_index: int,
        step_count: int,
        percent: int,
        current: int | None = None,
        total: int | None = None,
        unit: str | None = None,
        context: str | None = None,
        eta_seconds: int | None = None,
    ) -> None:
        if not self.enabled:
            return

        if step != self._step:
            self._step = step
            self._step_started_at = time.monotonic()

        if eta_seconds is None and current and total and current < total:
            elapsed = time.monotonic() - self._step_started_at
            eta_seconds = max(0, round(elapsed * (total - current) / current))

        event: dict[str, Any] = {
            "step": step,
            "label": label,
            "step_index": max(1, int(step_index)),
            "step_count": max(1, int(step_count)),
            "percent": max(0, min(99, int(percent))),
            "current": None if current is None else int(current),
            "total": None if total is None else int(total),
            "unit": unit,
            "context": context,
            "eta_seconds": eta_seconds,
        }
        print(f"{PROGRESS_PREFIX}{json.dumps(event, separators=(',', ':'))}", flush=True)
