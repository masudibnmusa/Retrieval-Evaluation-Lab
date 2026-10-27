"""Quality vs. cost vs. latency frontier, and a recommendation rule."""
from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(df: pd.DataFrame, maximize: list[str], minimize: list[str]) -> np.ndarray:
    cols = [df[c].fillna(0).values.astype(float) for c in maximize] + \
           [-df[c].fillna(0).values.astype(float) for c in minimize]
    X = np.column_stack(cols)
    keep = np.ones(len(X), dtype=bool)
    for i in range(len(X)):
        if np.any(np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)):
            keep[i] = False
    return keep


def recommend(lb: pd.DataFrame, primary: str, cost_col: str = "query_usd_per_1k",
              latency_col: str = "latency_p50_ms"):
    """Best config by the primary metric; among configs whose score is not below the best's CI lower
    bound (statistically hard to distinguish), prefer the cheapest, then fastest."""
    best = lb.loc[lb[primary].idxmax()]
    cand = lb[lb[primary] >= best["ci_lo"]].sort_values([cost_col, latency_col, "index_usd"])
    return cand.iloc[0], best