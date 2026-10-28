"""Wrap any retriever with query rewriting, HyDE, or multi-query.
YAML: retriever: {type: dense, transform: hyde}   or   transform: {mode: multi_query, n: 3}"""
from lab.components import Hit, Retriever
from lab.components.registry import register
from lab.components.retrievers.hybrid import rrf_fuse
from lab.llm import complete

REWRITE = "Rewrite this question as a concise, self-contained search query. Output only the query.\n\nQuestion: {q}"
HYDE = ("Write a short passage (3-5 sentences) that would plausibly answer this question, as it might appear "
        "in documentation. Output only the passage.\n\nQuestion: {q}")
MULTI = "Write {n} different search queries for this question, one per line, no numbering.\n\nQuestion: {q}"


@register("retriever", "query_transform")
class TransformedRetriever(Retriever):
    def __init__(self, base: Retriever, mode: str = "rewrite", model: str | None = None, n: int = 3, **_):
        assert mode in ("rewrite", "hyde", "multi_query")
        self.base, self.mode, self.model, self.n = base, mode, model, n

    def queries(self, q: str) -> list[str]:
        if self.mode == "rewrite":
            return [complete(REWRITE.format(q=q), model=self.model, max_tokens=100).strip() or q]
        if self.mode == "hyde":  # keep the original query too, so exact terms still match for BM25
            return [q + "\n" + complete(HYDE.format(q=q), model=self.model, max_tokens=250).strip()]
        lines = complete(MULTI.format(q=q, n=self.n), model=self.model, max_tokens=200).splitlines()
        return [q] + [l.strip("-* \t") for l in lines if l.strip()][: self.n]

    def retrieve(self, query, k, doc_filter=None) -> list[Hit]:
        lists = [self.base.retrieve(q, k, doc_filter) for q in self.queries(query)]
        return lists[0] if len(lists) == 1 else rrf_fuse(lists)[:k]