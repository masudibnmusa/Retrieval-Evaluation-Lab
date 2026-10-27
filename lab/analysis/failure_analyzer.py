"""Categorize misses and surface the worst questions."""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from lab.metrics.span_matcher import MatchConfig, overlap, relevant_chunks, span_coverage

_ID_LIKE = re.compile(r"\b(?:[A-Za-z]*\d[\w\-.]*|[A-Z][A-Z0-9_\-]{2,}|\w+_\w+)\b")


def classify_failure(row, q, index, corpus, k: int, has_reranker: bool, cfg: MatchConfig) -> str:
    retrieved = [r["chunk_id"] for r in row["retrieved"][:k]]
    docs = {s.doc_id for s in q.spans}
    doc_chunks = [c for d in docs for c in index.chunks_by_doc.get(d, [])]
    rel = relevant_chunks(doc_chunks, q.spans, cfg)
    if not rel:
        return "no_relevant_chunk"      # chunker produced nothing overlapping the gold span, or a labeling issue
    best_single = max((overlap(c.start, c.end, s.start, s.end) / (s.end - s.start)
                       for c in doc_chunks for s in q.spans if s.doc_id == c.doc_id), default=0.0)
    if best_single < cfg.recall_threshold:
        return "chunk_split"            # no single chunk holds enough of the answer
    if any(cid in rel and cid not in retrieved for cid in row["candidate_ids"]):
        return "reranker_demoted" if has_reranker else "ranked_below_k"
    gold = " ".join(corpus.span_text(s.doc_id, s.start, s.end) for s in q.spans).lower()
    if any(t.lower() in gold for t in _ID_LIKE.findall(q.question)):
        return "keyword_miss"           # exact term is in the answer span but retrieval missed it
    return "wrong_match"                # semantically plausible but wrong passages won


def analyze_failures(run, builder, corpus, config_id: str, k: int = 5, cfg: MatchConfig | None = None):
    cfg = cfg or MatchConfig()
    c = run.configs[config_id]
    spec = c["spec"]
    index = builder.get(spec["chunker"], spec["embedder"])
    has_rr = spec["reranker"].get("type", "none") != "none"
    col = f"full_recall@{k}"
    df = run.df[(run.df.config_id == config_id) & (run.df[col] == 0)]
    out = []
    for _, row in df.iterrows():
        q = run.questions[row["question_id"]]
        out.append({"question_id": q.id, "type": q.type, "question": q.question,
                    "cause": classify_failure(row, q, index, corpus, k, has_rr, cfg)})
    res = pd.DataFrame(out)
    counts = Counter(res["cause"]) if len(res) else Counter()
    return res, counts


def worst_questions(run, metric: str, n: int = 10) -> pd.DataFrame:
    g = run.df.groupby("question_id")[metric].mean().dropna().sort_values().head(n)
    return pd.DataFrame({"question_id": g.index, f"mean_{metric}": g.values,
                         "question": [run.questions[q].question for q in g.index],
                         "type": [run.questions[q].type for q in g.index]})