"""Expand configs, reuse shared indexes, retrieve, score, (optionally) generate, persist."""
from __future__ import annotations

import time

from lab import llm
from lab.components.registry import build, canon
from lab.corpus.registry import CorpusRegistry
from lab.experiments.experiment_spec import load_spec
from lab.experiments.parallel_executor import run_parallel
from lab.experiments.run_store import RunStore
from lab.indexing.index_builder import IndexBuilder
from lab.metrics.cost_latency import LEDGER
from lab.metrics.retrieval_metrics import question_metrics
from lab.metrics.span_matcher import MatchConfig, relevant_chunks
from lab.testset.relevance_labels import load_questions, validate_questions


def make_retriever(spec, index, embedder):
    spec = dict(canon(spec))
    transform = spec.pop("transform", None)
    r = build("retriever", spec, index=index, embedder=embedder)
    if transform:
        from lab.components.retrievers.query_transform import TransformedRetriever
        t = transform if isinstance(transform, dict) else {"mode": transform}
        r = TransformedRetriever(r, **t)
    return r


def run_config(cfg, spec, questions, builder, store, judge=None):
    index = builder.get(cfg.chunker, cfg.embedder)
    embedder = builder.embedder_for(cfg.embedder)
    retriever = make_retriever(cfg.retriever, index, embedder)
    reranker = build("reranker", cfg.reranker)
    mcfg = MatchConfig(**spec.match)
    max_k = max(spec.eval_k)
    depth = max(max_k, spec.candidates) if cfg.reranker.get("type", "none") != "none" else max_k
    hits_by_q = {}

    def work(q):
        llm.pop_cached_latency()
        t0 = time.perf_counter()
        cands = retriever.retrieve(q.question, depth, q.meta.get("doc_filter"))
        final = reranker.rerank(q.question, cands, max_k)
        ms = (time.perf_counter() - t0) * 1000 + llm.pop_cached_latency() * 1000
        hits_by_q[q.id] = final
        return {"question_id": q.id, "type": q.type, "answerable": q.answerable, "latency_ms": ms,
                "candidate_ids": [h.chunk.chunk_id for h in cands],
                "retrieved": [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "start": h.chunk.start,
                               "end": h.chunk.end, "score": float(h.score)} for h in final],
                "metrics": {}}

    before = LEDGER.snapshot()
    rows = run_parallel(work, questions, spec.workers, spec.rate_limit, desc="retrieve")
    retrieval_cost = LEDGER.delta(before)

    for q, row in zip(questions, rows):
        docs = {s.doc_id for s in q.spans}
        rel = relevant_chunks([c for d in docs for c in index.chunks_by_doc.get(d, [])], q.spans, mcfg)
        row["metrics"] = question_metrics([h.chunk for h in hits_by_q[q.id]], q.spans, rel,
                                          spec.eval_k, spec.token_budgets, mcfg)

    gen_cost = {}
    gen = spec.generation
    if gen.get("enabled") and judge is not None:
        from lab.metrics.generation_metrics import evaluate_generation, generate_answer
        ctx_k = int(gen.get("context_k", 5))
        before = LEDGER.snapshot()

        def gen_work(pair):
            q, row = pair
            chunks = [h.chunk for h in hits_by_q[q.id][:ctx_k]]
            ans = generate_answer(q.question, chunks, model=gen.get("model"))
            row["answer"] = ans
            row["metrics"].update(evaluate_generation(q, ans, chunks, judge, mcfg))
            return row

        run_parallel(gen_work, list(zip(questions, rows)), spec.workers, spec.rate_limit, desc="generate")
        gen_cost = LEDGER.delta(before)

    store.write_config(cfg, rows, index_stats=index.stats, retrieval_cost=retrieval_cost,
                       generation_cost=gen_cost)


def run_experiment(spec_path, run_id: str | None = None) -> str:
    spec = load_spec(spec_path)
    corpus = CorpusRegistry.from_path(spec.corpus)
    questions = load_questions(spec.testset)
    errs = validate_questions(questions, corpus)
    if errs:
        raise ValueError("Invalid test set:\n" + "\n".join(errs[:20]))
    run_id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{spec.name}"
    store = RunStore.create(run_id, spec, corpus, questions, spec_path)
    builder = IndexBuilder(corpus, backend=spec.backend)
    judge = None
    if spec.generation.get("enabled"):
        from lab.metrics.judge import Judge
        judge = Judge()
    for n, cfg in enumerate(spec.configs, 1):
        print(f"[{n}/{len(spec.configs)}] {cfg.label}")
        run_config(cfg, spec, questions, builder, store, judge)
    store.finalize()
    return run_id