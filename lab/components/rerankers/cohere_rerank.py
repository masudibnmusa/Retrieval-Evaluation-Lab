from lab import config
from lab.components import Hit, Reranker
from lab.components.registry import register
from lab.experiments.parallel_executor import retry
from lab.metrics.cost_latency import LEDGER


@register("reranker", "cohere")
class CohereReranker(Reranker):
    def __init__(self, model: str = "rerank-v3.5"):
        import cohere
        config.require_key("COHERE_API_KEY")
        self.client, self.model = cohere.ClientV2(), model

    @retry()
    def _call(self, query, docs, top_n):
        return self.client.rerank(model=self.model, query=query, documents=docs, top_n=top_n)

    def rerank(self, query, hits, top_n):
        if not hits:
            return []
        r = self._call(query, [h.chunk.embed_text for h in hits], min(top_n, len(hits)))
        LEDGER.add("rerank", config.RERANK_PRICE_PER_SEARCH.get(self.model, 0.002))
        return [Hit(hits[x.index].chunk, x.relevance_score) for x in r.results]