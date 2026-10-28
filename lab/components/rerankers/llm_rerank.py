from lab.components import Hit, Reranker
from lab.components.registry import register
from lab.llm import complete_json

PROMPT = """Rate how useful each passage is for answering the question, from 0 (irrelevant) to 10 (directly answers it).

Question: {q}

{passages}

Return JSON: {{"scores": [<one number per passage, in order>]}}"""


@register("reranker", "llm")
class LLMReranker(Reranker):
    def __init__(self, model: str | None = None, max_chars: int = 1200):
        self.model, self.max_chars = model, max_chars

    def rerank(self, query, hits, top_n):
        if not hits:
            return []
        body = "\n\n".join(f"[{i}] {h.chunk.embed_text[:self.max_chars]}" for i, h in enumerate(hits))
        data = complete_json(PROMPT.format(q=query, passages=body), model=self.model,
                             max_tokens=400, category="rerank_llm")
        scores = data.get("scores") if isinstance(data, dict) else None
        if not isinstance(scores, list) or len(scores) != len(hits):
            return hits[:top_n]  # fall back to the incoming order
        order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))[:top_n]
        return [Hit(hits[i].chunk, float(scores[i])) for i in order]