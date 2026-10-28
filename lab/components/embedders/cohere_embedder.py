import numpy as np

from lab import config
from lab.components.embedders.base import Embedder
from lab.components.registry import register
from lab.experiments.parallel_executor import retry


@register("embedder", "cohere")
class CohereEmbedder(Embedder):
    batch_size = 96

    def __init__(self, model: str = "embed-v4.0"):
        import cohere
        config.require_key("COHERE_API_KEY")
        self.client, self.model = cohere.ClientV2(), model
        self.name = f"cohere:{model}"
        self.price_per_m = config.EMBEDDING_PRICE_PER_M.get(model, 0.0)

    @retry()
    def _embed(self, texts, kind):
        r = self.client.embed(texts=[t or " " for t in texts], model=self.model,
                              input_type="search_document" if kind == "document" else "search_query",
                              embedding_types=["float"])
        vecs = getattr(r.embeddings, "float_", None) or r.embeddings.float
        return np.array(vecs)