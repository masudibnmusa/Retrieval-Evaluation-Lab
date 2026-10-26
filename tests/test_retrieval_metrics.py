import math
from types import SimpleNamespace as NS

from lab.metrics.retrieval_metrics import question_metrics, truncate_to_budget
from lab.metrics.span_matcher import MatchConfig, relevant_chunks
from lab.testset.relevance_labels import Span

CFG = MatchConfig(min_overlap_chars=5, recall_threshold=0.5)
SPANS = [Span("d", 100, 200)]


def ch(cid, s, e, n=100):
    return NS(chunk_id=cid, doc_id="d", start=s, end=e, n_tokens=n)


def run(retrieved, ks=(1, 3), budgets=()):
    allc = [ch("gold", 90, 210), ch("x", 0, 50), ch("y", 300, 400), ch("z", 500, 600)]
    return question_metrics(retrieved, SPANS, relevant_chunks(allc, SPANS, CFG), ks, budgets, CFG)


def test_oracle_gets_perfect_scores():
    m = run([ch("gold", 90, 210)])
    assert m["recall@1"] == m["mrr@1"] == m["ndcg@1"] == m["hit@1"] == m["full_recall@1"] == 1.0


def test_miss_scores_zero():
    m = run([ch("x", 0, 50), ch("y", 300, 400)])
    assert m["recall@3"] == m["mrr@3"] == m["ndcg@3"] == m["hit@3"] == 0.0


def test_rank_affects_mrr_and_ndcg():
    m = run([ch("x", 0, 50), ch("y", 300, 400), ch("gold", 90, 210)])
    assert m["mrr@3"] == 1 / 3
    assert math.isclose(m["ndcg@3"], 1 / math.log2(4))
    assert m["precision@3"] == 1 / 3
    assert m["recall@1"] == 0.0 and m["recall@3"] == 1.0


def test_unanswerable_has_no_metrics():
    assert question_metrics([ch("x", 0, 50)], [], {}, (1,)) == {}


def test_token_budget_truncation():
    chunks = [ch("a", 0, 1, n=400), ch("b", 0, 1, n=400), ch("c", 0, 1, n=400)]
    assert [c.chunk_id for c in truncate_to_budget(chunks, 1000)] == ["a", "b"]
    m = run([ch("x", 0, 50, 600), ch("gold", 90, 210, 600)], ks=(2,), budgets=(1000,))
    assert m["recall@2"] == 1.0 and m["recall@1000tok"] == 0.0   # gold doesn't fit in the budget