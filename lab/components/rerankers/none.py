from lab.components import Reranker
from lab.components.registry import register


@register("reranker", "none")
class NoReranker(Reranker):
    def __init__(self, **_): ...

    def rerank(self, query, hits, top_n):
        return hits[:top_n]