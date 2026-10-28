"""Local sentence-transformers models (BGE / E5 / MiniLM ...)."""
from lab.components.embedders.base import Embedder
from lab.components.registry import register

_BGE_QUERY = "Represent this sentence for searching relevant passages: "


def _prefixes(model: str):
    m = model.lower()
    if "e5" in m:
        return "passage: ", "query: "
    if "bge" in m and "m3" not in m:
        return "", _BGE_QUERY
    return "", ""


@register("embedder", "hf")
class HFEmbedder(Embedder):
    batch_size = 32

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 use_prefixes: bool = True):
        from sentence_transformers import SentenceTransformer
        self.model_name = model
        self.model = SentenceTransformer(model, device=device)
        self.name = f"hf:{model}"
        self.doc_prefix, self.query_prefix = _prefixes(model) if use_prefixes else ("", "")

    def _embed(self, texts, kind):
        pre = self.query_prefix if kind == "query" else self.doc_prefix
        return self.model.encode([pre + t for t in texts], batch_size=self.batch_size,
                                 normalize_embeddings=True, show_progress_bar=False)