"""Concurrency, rate limiting, retries."""
from __future__ import annotations

import functools
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm


class RateLimiter:
    def __init__(self, per_sec: float):
        self.interval, self.next, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            wait = self.next - now
            self.next = max(now, self.next) + self.interval
        if wait > 0:
            time.sleep(wait)


def run_parallel(fn, items, workers: int = 1, rate_per_sec: float | None = None, desc: str | None = None):
    """Ordered map. Note: per-query latency is inflated when workers > 1."""
    items = list(items)
    limiter = RateLimiter(rate_per_sec) if rate_per_sec else None

    def wrapped(x):
        if limiter:
            limiter.acquire()
        return fn(x)

    if workers <= 1:
        return [wrapped(x) for x in tqdm(items, desc=desc, leave=False)]
    with ThreadPoolExecutor(workers) as ex:
        return list(tqdm(ex.map(wrapped, items), total=len(items), desc=desc, leave=False))


def retry(attempts: int = 5, base: float = 1.0):
    def deco(fn):
        @functools.wraps(fn)
        def inner(*a, **kw):
            for i in range(attempts):
                try:
                    return fn(*a, **kw)
                except Exception as e:  # noqa: BLE001
                    name = type(e).__name__.lower()
                    if i == attempts - 1 or "auth" in name or "permission" in name:
                        raise
                    time.sleep(base * 2 ** i + random.random())
        return inner
    return deco