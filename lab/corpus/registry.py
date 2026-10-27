"""Stable doc IDs, versions, hashes. All span offsets refer to Document.text (normalized)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from lab.corpus.loader import load_documents
from lab.corpus.normalizer import normalize


@dataclass
class Document:
    doc_id: str
    text: str
    sha256: str
    source: str
    offsets: list[int] = field(default_factory=list, repr=False)

    @property
    def version(self) -> str:
        return self.sha256[:12]


class CorpusRegistry:
    def __init__(self, docs: list[Document]):
        self.docs = {d.doc_id: d for d in docs}

    @classmethod
    def from_path(cls, path) -> "CorpusRegistry":
        docs = []
        for raw in load_documents(path):
            n = normalize(raw.text)
            docs.append(Document(raw.doc_id, n.text, hashlib.sha256(n.text.encode()).hexdigest(),
                                 raw.source, n.offsets))
        if not docs:
            raise ValueError(f"No documents found under {path}")
        return cls(docs)

    def __getitem__(self, doc_id): return self.docs[doc_id]
    def __iter__(self): return iter(self.docs.values())
    def __len__(self): return len(self.docs)
    def __contains__(self, doc_id): return doc_id in self.docs

    def corpus_hash(self) -> str:
        payload = json.dumps(sorted((d.doc_id, d.sha256) for d in self))
        return hashlib.sha256(payload.encode()).hexdigest()

    def span_text(self, doc_id: str, start: int, end: int) -> str:
        return self.docs[doc_id].text[start:end]

    def manifest(self) -> dict:
        return {d.doc_id: {"sha256": d.sha256, "chars": len(d.text), "source": d.source} for d in self}

    def save_manifest(self, path) -> None:
        Path(path).write_text(json.dumps(self.manifest(), indent=2))