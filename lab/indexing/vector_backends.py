"""In-memory (NumPy) / FAISS / Chroma backends behind one interface."""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod

import numpy as np


class VectorIndex(ABC):
    def __init__(self, vectors: np.ndarray):
        self.n = len(vectors)

    @abstractmethod
    def _search(self, q: np.ndarray, k: int) -> list[tuple[int, float]]: ...

    def search(self, q, k, mask=None):
        if self.n == 0:
            return []
        if mask is None:
            return self._search(q, min(k, self.n))
        n = k
        while True:  # over-fetch then filter (exact for NumpyIndex, approximate otherwise)
            res = [(i, s) for i, s in self._search(q, min(n, self.n)) if mask[i]]
            if len(res) >= k or n >= self.n:
                return res[:k]
            n *= 4


class NumpyIndex(VectorIndex):
    def __init__(self, vectors):
        super().__init__(vectors)
        self.m = vectors

    def _search(self, q, k):
        return self.search(q, k)

    def search(self, q, k, mask=None):
        if self.n == 0:
            return []
        s = self.m @ q
        if mask is not None:
            s = np.where(mask, s, -np.inf)
        k = min(k, self.n)
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if np.isfinite(s[i])]


class FaissIndex(VectorIndex):
    def __init__(self, vectors):
        import faiss
        super().__init__(vectors)
        self.ix = faiss.IndexFlatIP(vectors.shape[1])
        self.ix.add(np.ascontiguousarray(vectors, dtype=np.float32))

    def _search(self, q, k):
        s, i = self.ix.search(q[None, :].astype(np.float32), k)
        return [(int(a), float(b)) for a, b in zip(i[0], s[0]) if a >= 0]


class ChromaIndex(VectorIndex):
    def __init__(self, vectors):
        import chromadb
        super().__init__(vectors)
        self.col = chromadb.EphemeralClient().create_collection(
            f"c{uuid.uuid4().hex[:12]}", metadata={"hnsw:space": "cosine"})
        for i in range(0, self.n, 5000):
            part = vectors[i:i + 5000]
            self.col.add(ids=[str(j) for j in range(i, i + len(part))], embeddings=part.tolist())

    def _search(self, q, k):
        r = self.col.query(query_embeddings=[q.tolist()], n_results=k)
        return [(int(i), 1.0 - float(d)) for i, d in zip(r["ids"][0], r["distances"][0])]


def make_backend(name: str, vectors: np.ndarray) -> VectorIndex:
    return {"numpy": NumpyIndex, "faiss": FaissIndex, "chroma": ChromaIndex}[name](vectors)