import numpy as np

from lab.components import Hit, Retriever
from lab.components.registry import register
from lab.indexing.index_builder import bm25_tokenize


@register("retriever", "bm25")
class BM25Retriever(Retriever):
    def __init__(self, index, embedder=None, doc_filter: str | None = None, **_):
        self.index, self.doc_filter = index, doc_filter

    def retrieve(self, query, k, doc_filter=None):
        scores = np.asarray(self.index.bm25().get_scores(bm25_tokenize(query)), dtype=float)
        pattern = doc_filter or self.doc_filter
        if pattern:
            scores = np.where(self.index.doc_mask(pattern), scores, -np.inf)
        k = min(k, len(scores))
        if k == 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self.index.chunks[i], float(scores[i])) for i in top if np.isfinite(scores[i]) and scores[i] > 0]