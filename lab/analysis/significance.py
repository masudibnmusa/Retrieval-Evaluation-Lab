"""Bootstrap CIs, paired permutation tests, multiple-comparison correction."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Returns (mean, lo, hi) of the percentile bootstrap CI of the mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    step = max(1, 2_000_000 // x.size)
    for i in range(0, n_boot, step):
        j = min(step, n_boot - i)
        means[i:i + j] = x[rng.integers(0, x.size, (j, x.size))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))


def paired_bootstrap_diff(a, b, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """CI for mean(a - b) over the same questions."""
    return bootstrap_ci(np.asarray(a, float) - np.asarray(b, float), n_boot, alpha, seed)


def paired_permutation_test(a, b, n_perm: int = 10000, seed: int = 0) -> float:
    """Two-sided sign-flip permutation test on per-question differences."""
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[~np.isnan(d)]
    if d.size == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    obs, count, done = abs(d.mean()), 0, 0
    step = max(1, 2_000_000 // d.size)
    while done < n_perm:
        j = min(step, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(j, d.size))
        count += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
        done += j
    return (count + 1) / (n_perm + 1)


def holm_correction(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    p = np.asarray(pvals, float)
    n, order = len(p), np.argsort(p)
    adj, prev = np.empty_like(p), 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, p[i] * n / (rank + 1))
        adj[i] = prev
    return adj