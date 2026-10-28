"""Register components by name so YAML configs can reference them."""
from __future__ import annotations

import importlib
import json

_REG: dict[str, dict[str, type]] = {"chunker": {}, "embedder": {}, "retriever": {}, "reranker": {}}
_loaded = False

_BUILTIN = [
    "lab.components.chunkers.fixed", "lab.components.chunkers.recursive",
    "lab.components.chunkers.sentence", "lab.components.chunkers.semantic",
    "lab.components.chunkers.structure_aware", "lab.components.chunkers.contextual",
    "lab.components.embedders.base", "lab.components.embedders.openai_embedder",
    "lab.components.embedders.voyage_embedder", "lab.components.embedders.cohere_embedder",
    "lab.components.embedders.hf_embedder",
    "lab.components.retrievers.dense", "lab.components.retrievers.bm25",
    "lab.components.retrievers.hybrid", "lab.components.retrievers.query_transform",
    "lab.components.rerankers.none", "lab.components.rerankers.cross_encoder",
    "lab.components.rerankers.cohere_rerank", "lab.components.rerankers.llm_rerank",
]


def register(kind: str, name: str):
    def deco(cls):
        _REG[kind][name] = cls
        return cls
    return deco


def load_builtin() -> None:
    global _loaded
    if not _loaded:
        _loaded = True
        for m in _BUILTIN:
            importlib.import_module(m)


def canon(spec) -> dict:
    if spec is None:
        return {"type": "none"}
    if isinstance(spec, str):
        return {"type": spec}
    return dict(spec)


def spec_key(spec) -> str:
    return json.dumps(canon(spec), sort_keys=True)


def build(kind: str, spec, **ctx):
    load_builtin()
    spec = canon(spec)
    params = {k: v for k, v in spec.items() if k != "type"}
    try:
        cls = _REG[kind][spec["type"]]
    except KeyError:
        raise ValueError(f"Unknown {kind} '{spec.get('type')}'. Available: {sorted(_REG[kind])}")
    return cls(**params, **ctx)