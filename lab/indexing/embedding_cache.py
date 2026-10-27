"""Hash-keyed SQLite cache: never embed the same text twice. Costs are still charged to the
ledger as if uncached, so every config reports its true cost."""
from __future__ import annotations

import hashlib
import sqlite3

import numpy as np

from lab import config
from lab.components import approx_tokens
from lab.metrics.cost_latency import LEDGER


class EmbeddingCache:
    def __init__(self):
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = config.CACHE_DIR / "embeddings.sqlite"
        with self._conn() as c:
            c.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")

    def _conn(self):
        return sqlite3.connect(self.path, timeout=60)

    def get_many(self, keys):
        out = {}
        with self._conn() as c:
            for i in range(0, len(keys), 500):
                part = keys[i:i + 500]
                q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})"
                for k, v in c.execute(q, part):
                    out[k] = np.frombuffer(v, dtype=np.float32)
        return out

    def put_many(self, items: dict):
        with self._conn() as c:
            c.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                          [(k, np.asarray(v, np.float32).tobytes()) for k, v in items.items()])


class CachedEmbedder:
    """Wraps an Embedder. Documents are cached; queries are NOT cached by default so
    query latency measurements stay honest."""

    def __init__(self, inner, cache_queries: bool = False):
        self.inner, self.cache_queries = inner, cache_queries
        self.cache = EmbeddingCache()
        self.hits = self.misses = 0

    @property
    def name(self):
        return self.inner.name

    def embed_documents(self, texts):
        return self._embed(texts, "document", True)

    def embed_queries(self, texts):
        return self._embed(texts, "query", self.cache_queries)

    def _embed(self, texts, kind, use_cache):
        texts = list(texts)
        tokens = int(sum(approx_tokens(t) for t in texts) * 1.3)
        LEDGER.add("embedding", tokens * self.inner.price_per_m / 1e6, tokens)
        fn = getattr(self.inner, f"embed_{kind}s")
        if not use_cache or not texts:
            return fn(texts)
        keys = [hashlib.sha256(f"{self.inner.name}|{kind}|{t}".encode()).hexdigest() for t in texts]
        found = self.cache.get_many(keys)
        todo = [i for i, k in enumerate(keys) if k not in found]
        if todo:
            vecs = fn([texts[i] for i in todo])
            self.cache.put_many({keys[i]: v for i, v in zip(todo, vecs)})
            for i, v in zip(todo, vecs):
                found[keys[i]] = v
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        return np.stack([found[k] for k in keys])