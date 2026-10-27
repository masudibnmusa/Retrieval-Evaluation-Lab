"""recall@k, precision@k, MRR, nDCG@k, hit rate (+ full_recall for multi-hop) at fixed k
AND at fixed token budgets (fair across chunk sizes)."""
from __future__ import annotations

import math

from lab.metrics.span_matcher import MatchConfig, chunk_gain, span_coverage


def truncate_to_budget(chunks, budget: int):
    out, used = [], 0
    for c in chunks:
        if used + c.n_tokens > budget:
            break
        out.append(c)
        used += c.n_tokens
    return out


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def _score(retrieved, spans, ideal_gains, cfg, cutoff, denom):
    gains = [chunk_gain(c, spans, cfg) for c in retrieved]
    rel = [g > 0 for g in gains]
    n_rec = sum(span_coverage(retrieved, s) >= cfg.recall_threshold for s in spans)
    first = next((i for i, r in enumerate(rel) if r), None)
    idcg = _dcg(sorted(ideal_gains, reverse=True)[:cutoff])
    return {
        "recall": n_rec / len(spans),                       # fraction of gold spans recalled
        "full_recall": float(n_rec == len(spans)),          # all spans recalled (multi-hop success)
        "hit": float(any(rel)),
        "precision": sum(rel) / denom if denom else 0.0,
        "mrr": 1.0 / (first + 1) if first is not None else 0.0,
        "ndcg": _dcg(gains) / idcg if idcg > 0 else 0.0,
    }


def question_metrics(retrieved, spans, relevant: dict, ks, budgets=(), cfg: MatchConfig | None = None) -> dict:
    """Returns {} for unanswerable questions (no spans): they are excluded from retrieval metrics."""
    if not spans:
        return {}
    cfg, ideal, out = cfg or MatchConfig(), list(relevant.values()), {}
    for k in ks:
        for name, v in _score(retrieved[:k], spans, ideal, cfg, k, k).items():
            out[f"{name}@{k}"] = v
    for b in budgets:
        r = truncate_to_budget(retrieved, b)
        for name, v in _score(r, spans, ideal, cfg, len(r), len(r)).items():
            out[f"{name}@{b}tok"] = v
    return out