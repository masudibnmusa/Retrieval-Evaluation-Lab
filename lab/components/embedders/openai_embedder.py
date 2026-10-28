import numpy as np

from lab import config
from lab.components.embedders.base import Embedder
from lab.components.registry import register
from lab.experiments.parallel_executor import retry


@register("embedder", "openai")
class OpenAIEmbedder(Embedder):
    batch_size = 128

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int | None = None):
        from openai import OpenAI
        config.require_key("OPENAI_API_KEY")
        self.client, self.model, self.dimensions = OpenAI(), model, dimensions
        self.name = f"openai:{model}" + (f":{dimensions}" if dimensions else "")
        self.price_per_m = config.EMBEDDING_PRICE_PER_M.get(model, 0.0)

    @retry()
    def _embed(self, texts, kind):
        kw = {"model": self.model, "input": [t or " " for t in texts]}
        if self.dimensions:
            kw["dimensions"] = self.dimensions
        return np.array([d.embedding for d in self.client.embeddings.create(**kw).data])