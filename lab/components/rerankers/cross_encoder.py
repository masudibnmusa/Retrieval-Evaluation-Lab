from lab.components import Hit, Reranker
from lab.components.registry import register


@register("reranker", "cross_encoder")
class CrossEncoderReranker(Reranker):
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", device: str | None = None):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model, device=device)

    def rerank(self, query, hits, top_n):
        if not hits:
            return []
        scores = self.model.predict([(query, h.chunk.embed_text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: -scores[i])[:top_n]
        return [Hit(hits[i].chunk, float(scores[i])) for i in order]