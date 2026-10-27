"""$ per 1k queries, p50/p95 latency, index size. Components report spend to LEDGER."""
from __future__ import annotations

import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


class CostLedger:
    def __init__(self):
        self._lock = threading.Lock()
        self.usd, self.tokens = defaultdict(float), defaultdict(int)

    def add(self, category: str, usd: float = 0.0, tokens: int = 0):
        with self._lock:
            self.usd[category] += usd
            self.tokens[category] += tokens

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self.usd)

    def delta(self, before: dict) -> dict:
        with self._lock:
            return {k: v - before.get(k, 0.0) for k, v in self.usd.items() if v - before.get(k, 0.0) > 0}


LEDGER = CostLedger()


def summarize_latencies(ms) -> dict:
    a = np.asarray(list(ms), dtype=float)
    if a.size == 0:
        return {"p50": float("nan"), "p95": float("nan"), "mean": float("nan")}
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean())}


def usd_per_1k_queries(total_usd: float, n_queries: int) -> float:
    return 1000.0 * total_usd / max(n_queries, 1)


def dir_size_bytes(path) -> int:
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0