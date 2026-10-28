import numpy as np

from lab import config
from lab.components.embedders.base import Embedder
from lab.components.registry import register
from lab.experiments.parallel_executor import retry


@register("embedder", "voyage")
class VoyageEmbedder(Embedder):
    batch_size = 64

    def __init__(self, model: str = "voyage-3.5"):
        import voyageai
        config.require_key("VOYAGE_API_KEY")
        self.client, self.model = voyageai.Client(), model
        self.name = f"voyage:{model}"
        self.price_per_m = config.EMBEDDING_PRICE_PER_M.get(model, 0.0)

    @retry()
    def _embed(self, texts, kind):
        r = self.client.embed([t or " " for t in texts], model=self.model, input_type=kind)
        return np.array(r.embeddings)