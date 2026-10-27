"""Context precision/recall: how much retrieved text is useful, and how much needed info is there."""
from lab.metrics.faithfulness import extract_claims, support_rate
from lab.metrics.span_matcher import MatchConfig, chunk_gain, overlap


def context_precision(chunks, spans, cfg: MatchConfig | None = None) -> dict:
    cfg = cfg or MatchConfig()
    if not chunks:
        return {"chunk_precision": 0.0, "useful_char_ratio": 0.0}
    rel = sum(chunk_gain(c, spans, cfg) > 0 for c in chunks)
    useful = sum(overlap(c.start, c.end, s.start, s.end) for c in chunks for s in spans if s.doc_id == c.doc_id)
    total = sum(c.end - c.start for c in chunks)
    return {"chunk_precision": rel / len(chunks), "useful_char_ratio": min(1.0, useful / total) if total else 0.0}


def context_recall(reference_answer: str, context: str, judge) -> float | None:
    """Fraction of the reference answer's claims that the context supports (LLM judge)."""
    return support_rate(extract_claims(reference_answer, judge.model), context, judge)