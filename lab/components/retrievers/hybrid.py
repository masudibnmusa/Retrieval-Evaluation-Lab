from lab.components import Hit, Retriever
from lab.components.registry import register
from lab.components.retrievers.bm25 import BM25Retriever
from lab.components.retrievers.dense import DenseRetriever


def rrf_fuse(hit_lists: list[list[Hit]], k: int = 60) -> list[Hit]:
    scores, chunks = {}, {}
    for hits in hit_lists:
        for rank, h in enumerate(hits):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks[cid] = h.chunk
    return [Hit(chunks[c], s) for c, s in sorted(scores.items(), key=lambda x: -x[1])]


def _minmax(hits):
    if not hits:
        return {}
    vals = [h.score for h in hits]
    lo, hi = min(vals), max(vals)
    return {h.chunk.chunk_id: (h.score - lo) / (hi - lo) if hi > lo else 1.0 for h in hits}


@register("retriever", "hybrid")
class HybridRetriever(Retriever):
    def __init__(self, index, embedder, fusion: str = "rrf", rrf_k: int = 60, alpha: float = 0.5,
                 depth: int = 50, doc_filter: str | None = None, **_):
        assert fusion in ("rrf", "weighted")
        self.dense = DenseRetriever(index, embedder, doc_filter)
        self.sparse = BM25Retriever(index, embedder, doc_filter)
        self.fusion, self.rrf_k, self.alpha, self.depth = fusion, rrf_k, alpha, depth

    def retrieve(self, query, k, doc_filter=None):
        d = self.dense.retrieve(query, max(self.depth, k), doc_filter)
        s = self.sparse.retrieve(query, max(self.depth, k), doc_filter)
        if self.fusion == "rrf":
            return rrf_fuse([d, s], self.rrf_k)[:k]
        nd, ns = _minmax(d), _minmax(s)
        chunks = {h.chunk.chunk_id: h.chunk for h in d + s}
        fused = {c: self.alpha * nd.get(c, 0.0) + (1 - self.alpha) * ns.get(c, 0.0) for c in chunks}
        return [Hit(chunks[c], v) for c, v in sorted(fused.items(), key=lambda x: -x[1])][:k]