"""Build one index per (chunker, embedder) pair; shared across configs and persisted on disk."""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
from collections import defaultdict

import numpy as np

from lab import config
from lab.components.registry import build, spec_key
from lab.indexing.embedding_cache import CachedEmbedder
from lab.indexing.vector_backends import make_backend
from lab.metrics.cost_latency import LEDGER, dir_size_bytes


def bm25_tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Index:
    def __init__(self, key, chunks, vectors, stats, backend="numpy"):
        self.key, self.chunks, self.vectors, self.stats, self.backend = key, chunks, vectors, stats, backend
        self.chunks_by_doc = defaultdict(list)
        for c in chunks:
            self.chunks_by_doc[c.doc_id].append(c)
        self._vindex = self._bm25 = None
        self._masks = {}

    def vindex(self):
        if self._vindex is None:
            self._vindex = make_backend(self.backend, self.vectors)
        return self._vindex

    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([bm25_tokenize(c.embed_text) for c in self.chunks])
        return self._bm25

    def doc_mask(self, pattern: str) -> np.ndarray:
        if pattern not in self._masks:
            rx = re.compile(pattern)
            self._masks[pattern] = np.array([bool(rx.search(c.doc_id)) for c in self.chunks])
        return self._masks[pattern]


class IndexBuilder:
    def __init__(self, corpus, backend: str = "numpy", use_disk: bool = True):
        self.corpus, self.backend, self.use_disk = corpus, backend, use_disk
        self._mem, self._emb = {}, {}

    def key(self, chunker, embedder) -> str:
        payload = json.dumps([self.corpus.corpus_hash(), spec_key(chunker), spec_key(embedder)])
        return hashlib.sha1(payload.encode()).hexdigest()[:16]

    def embedder_for(self, spec) -> CachedEmbedder:
        k = spec_key(spec)
        if k not in self._emb:
            self._emb[k] = CachedEmbedder(build("embedder", spec))
        return self._emb[k]

    def get(self, chunker, embedder) -> Index:
        key = self.key(chunker, embedder)
        if key in self._mem:
            return self._mem[key]
        path = config.INDEX_DIR / key
        if self.use_disk and (path / "meta.json").exists():
            with open(path / "chunks.pkl", "rb") as f:
                chunks = pickle.load(f)
            idx = Index(key, chunks, np.load(path / "vectors.npy"),
                        json.loads((path / "meta.json").read_text()), self.backend)
        else:
            idx = self._build(key, chunker, embedder)
            if self.use_disk:
                path.mkdir(parents=True, exist_ok=True)
                with open(path / "chunks.pkl", "wb") as f:
                    pickle.dump(idx.chunks, f)
                np.save(path / "vectors.npy", idx.vectors)
                (path / "meta.json").write_text(json.dumps(idx.stats, indent=2))
        self._mem[key] = idx
        return idx

    def _build(self, key, chunker_spec, embedder_spec) -> Index:
        before, t0 = LEDGER.snapshot(), time.perf_counter()
        chunker = build("chunker", chunker_spec)
        chunks = [c for d in self.corpus for c in chunker.chunk(d.doc_id, d.text)]
        emb = self.embedder_for(embedder_spec)
        vectors = emb.embed_documents([c.embed_text for c in chunks]) if chunks else np.zeros((0, 1), np.float32)
        secs = time.perf_counter() - t0   # clear data/cache for clean timing; cached embeddings are fast
        cost = LEDGER.delta(before)
        stats = {
            "n_chunks": len(chunks),
            "avg_chunk_tokens": float(np.mean([c.n_tokens for c in chunks])) if chunks else 0.0,
            "build_seconds": secs,
            "storage_bytes": int(vectors.nbytes + sum(len(c.embed_text.encode()) for c in chunks)),
            "index_usd": float(sum(cost.values())),
            "index_cost_breakdown": cost,
            "embed_cache_hits": emb.hits, "embed_cache_misses": emb.misses,
        }
        return Index(key, chunks, vectors, stats, self.backend)