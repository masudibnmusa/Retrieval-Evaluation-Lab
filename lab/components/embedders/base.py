import numpy as np
from abc import ABC, abstractmethod

from lab.components.registry import register


class Embedder(ABC):
    name = "embedder"
    price_per_m = 0.0   # $ per 1M tokens
    batch_size = 64

    @abstractmethod
    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        """kind: 'document' or 'query'."""

    def _run(self, texts, kind):
        outs = [np.asarray(self._embed(texts[i:i + self.batch_size], kind), dtype=np.float32)
                for i in range(0, len(texts), self.batch_size)]
        if not outs:
            return np.zeros((0, 1), np.float32)
        m = np.vstack(outs)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)

    def embed_documents(self, texts):
        return self._run(list(texts), "document")

    def embed_queries(self, texts):
        return self._run(list(texts), "query")


@register("embedder", "none")
class NoneEmbedder(Embedder):
    """Placeholder for BM25-only configs (no vectors needed)."""
    name = "none"

    def _embed(self, texts, kind):
        return np.ones((len(texts), 1), dtype=np.float32)