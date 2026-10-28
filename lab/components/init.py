"""Shared types and base classes for pipeline components."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

_TOKEN = re.compile(r"\S+")


def token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _TOKEN.finditer(text)]


def approx_tokens(text: str) -> int:
    """Whitespace tokens. Not BPE tokens, but consistent across every config."""
    return sum(1 for _ in _TOKEN.finditer(text))


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    start: int            # offsets into the ORIGINAL (normalized) document text
    end: int
    text: str             # == doc_text[start:end]
    embed_text: str = ""  # what gets embedded / BM25-indexed (may include LLM context)
    n_tokens: int = 0

    def __post_init__(self):
        if not self.embed_text:
            self.embed_text = self.text
        if not self.n_tokens:
            self.n_tokens = approx_tokens(self.text)


@dataclass
class Hit:
    chunk: Chunk
    score: float


def make_chunks(doc_id: str, text: str, spans) -> list[Chunk]:
    chunks = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            chunks.append(Chunk(f"{doc_id}#{len(chunks)}", doc_id, s, e, text[s:e]))
    return chunks


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc_id: str, text: str) -> list[Chunk]: ...


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int, doc_filter: str | None = None) -> list[Hit]: ...


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, hits: list[Hit], top_n: int) -> list[Hit]: ...