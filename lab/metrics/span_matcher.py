"""Map retrieved chunks to gold answer spans by character overlap.

Rules (all configurable):
  * A chunk is *relevant* if it overlaps a gold span by >= min(min_overlap_chars, span length).
    Its graded gain (for nDCG) is the fraction of that span it covers.
  * A gold span is *recalled* if the UNION of retrieved chunks covers >= recall_threshold of it.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchConfig:
    min_overlap_chars: int = 20
    recall_threshold: float = 0.5


def overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def chunk_gain(chunk, spans, cfg: MatchConfig) -> float:
    best = 0.0
    for s in spans:
        if s.doc_id != chunk.doc_id:
            continue
        ov = overlap(chunk.start, chunk.end, s.start, s.end)
        length = s.end - s.start
        if ov == 0 or ov < min(cfg.min_overlap_chars, length):
            continue
        best = max(best, ov / length)
    return best


def span_coverage(chunks, span) -> float:
    ivs = sorted((max(c.start, span.start), min(c.end, span.end))
                 for c in chunks if c.doc_id == span.doc_id and overlap(c.start, c.end, span.start, span.end) > 0)
    covered, cur_s, cur_e = 0, None, None
    for s, e in ivs:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        covered += cur_e - cur_s
    return covered / (span.end - span.start)


def relevant_chunks(chunks, spans, cfg: MatchConfig) -> dict:
    """chunk_id -> gain for every relevant chunk among `chunks` (pass all chunks of the gold docs)."""
    out = {}
    for c in chunks:
        g = chunk_gain(c, spans, cfg)
        if g > 0:
            out[c.chunk_id] = g
    return out