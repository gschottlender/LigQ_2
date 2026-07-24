from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque

from core.config import WEB_RATE_LIMIT_COUNT, WEB_RATE_LIMIT_WINDOW_SECONDS


_rate_lock = asyncio.Lock()
_accepted_by_ip: dict[str, deque[float]] = defaultdict(deque)


async def rate_limit_status(ip_hash: str) -> tuple[bool, int]:
    now = time.monotonic()
    cutoff = now - WEB_RATE_LIMIT_WINDOW_SECONDS
    async with _rate_lock:
        entries = _accepted_by_ip[ip_hash]
        while entries and entries[0] <= cutoff:
            entries.popleft()
        if len(entries) >= WEB_RATE_LIMIT_COUNT:
            retry_after = max(1, round(entries[0] + WEB_RATE_LIMIT_WINDOW_SECONDS - now))
            return False, retry_after
        return True, 0


async def record_accepted_search(ip_hash: str) -> None:
    async with _rate_lock:
        _accepted_by_ip[ip_hash].append(time.monotonic())
