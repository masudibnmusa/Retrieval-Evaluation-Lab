from lab.components import Hit, Retriever
from lab.components.registry import register


@register("retriever", "dense")
class DenseRetriever(Retriever):
    def __init__(self, index, embedder, doc_filter: str | None = None, **_):
        self.index, self.embedder, self.doc_filter = index, embedder, doc_filter

    def retrieve(self, query, k, doc_filter=None):
        pattern = doc_filter or self.doc_filter
        mask = self.index.doc_mask(pattern) if pattern else None
        q = self.embedder.embed_queries([query])[0]
        return [Hit(self.index.chunks[i], s) for i, s in self.index.vindex().search(q, k, mask)]