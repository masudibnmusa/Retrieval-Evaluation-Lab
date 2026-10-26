from types import SimpleNamespace as NS

from lab.metrics.span_matcher import MatchConfig, chunk_gain, overlap, relevant_chunks, span_coverage
from lab.testset.relevance_labels import Span

CFG = MatchConfig(min_overlap_chars=5, recall_threshold=0.5)
GOLD = Span("d", 100, 200)


def ch(cid, s, e, doc="d"):
    return NS(chunk_id=cid, doc_id=doc, start=s, end=e)


def test_overlap():
    assert overlap(0, 10, 5, 20) == 5
    assert overlap(0, 10, 10, 20) == 0


def test_gain_full_partial_none():
    assert chunk_gain(ch("a", 50, 250), [GOLD], CFG) == 1.0
    assert chunk_gain(ch("b", 150, 300), [GOLD], CFG) == 0.5
    assert chunk_gain(ch("c", 0, 50), [GOLD], CFG) == 0.0
    assert chunk_gain(ch("d", 198, 300), [GOLD], CFG) == 0.0   # 2 chars < min_overlap
    assert chunk_gain(ch("e", 100, 200, doc="other"), [GOLD], CFG) == 0.0


def test_union_coverage_of_split_answer():
    a, b = ch("a", 80, 140), ch("b", 140, 220)
    assert span_coverage([a], GOLD) == 0.4
    assert span_coverage([a, b], GOLD) == 1.0
    assert span_coverage([a, ch("dup", 90, 130)], GOLD) == 0.4   # overlapping chunks aren't double counted


def test_relevant_chunks_only_positive():
    rel = relevant_chunks([ch("a", 0, 50), ch("b", 90, 210)], [GOLD], CFG)
    assert list(rel) == ["b"]