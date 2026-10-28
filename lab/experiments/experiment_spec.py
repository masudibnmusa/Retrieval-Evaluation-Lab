"""Parse YAML: a `grid:` (cartesian product) or an explicit `configs:` list."""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from lab.components.registry import canon


def short_label(spec) -> str:
    s = canon(spec)
    extras = []
    for k, v in s.items():
        if k == "type":
            continue
        if isinstance(v, dict):
            v = short_label(v)
        elif isinstance(v, str) and "/" in v:
            v = v.split("/")[-1]
        extras.append(str(v))
    return f"{s['type']}({','.join(extras)})" if extras else s["type"]


@dataclass
class PipelineConfig:
    chunker: dict
    embedder: dict
    retriever: dict
    reranker: dict

    @property
    def id(self) -> str:
        return hashlib.sha1(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:10]

    @property
    def label(self) -> str:
        return " | ".join(short_label(x) for x in (self.chunker, self.embedder, self.retriever, self.reranker))

    def parts(self) -> dict:
        return {k: short_label(v) for k, v in asdict(self).items()}


@dataclass
class ExperimentSpec:
    name: str
    corpus: str
    testset: str
    configs: list[PipelineConfig]
    eval_k: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
    token_budgets: list[int] = field(default_factory=list)
    candidates: int = 50          # retrieval depth fed to the reranker
    workers: int = 1
    rate_limit: float | None = None
    backend: str = "numpy"
    primary_metric: str = ""
    baseline: int = 0             # index into configs for paired comparisons
    generation: dict = field(default_factory=dict)
    match: dict = field(default_factory=dict)


def _aslist(x, default):
    if x is None:
        return [default]
    return x if isinstance(x, list) else [x]


def _make(c, e, r, rr) -> PipelineConfig:
    c, e, r, rr = canon(c), canon(e), canon(r), canon(rr)
    if r["type"] == "bm25":
        e = {"type": "none"}   # BM25 needs no vectors; avoids duplicate runs per embedder
    return PipelineConfig(c, e, r, rr)


def expand_grid(grid: dict) -> list[PipelineConfig]:
    combos = itertools.product(
        _aslist(grid.get("chunker"), {"type": "recursive"}),
        _aslist(grid.get("embedder"), {"type": "none"}),
        _aslist(grid.get("retriever"), {"type": "dense"}),
        _aslist(grid.get("reranker"), {"type": "none"}))
    out, seen = [], set()
    for c, e, r, rr in combos:
        cfg = _make(c, e, r, rr)
        if cfg.id not in seen:
            seen.add(cfg.id)
            out.append(cfg)
    return out


def load_spec(path) -> ExperimentSpec:
    raw = yaml.safe_load(Path(path).read_text())
    if "configs" in raw:
        configs = [_make(c.get("chunker"), c.get("embedder"), c.get("retriever"), c.get("reranker"))
                   for c in raw["configs"]]
    else:
        configs = expand_grid(raw["grid"])
    ks = raw.get("eval_k") or raw.get("top_k") or [1, 3, 5, 10]
    ks = sorted({int(k) for k in (ks if isinstance(ks, list) else [ks])})
    return ExperimentSpec(
        name=raw.get("name", Path(path).stem), corpus=raw.get("corpus", "data/corpus"),
        testset=raw["testset"], configs=configs, eval_k=ks,
        token_budgets=[int(b) for b in raw.get("token_budgets", [])],
        candidates=int(raw.get("candidates", 50)), workers=int(raw.get("workers", 1)),
        rate_limit=raw.get("rate_limit"), backend=raw.get("backend", "numpy"),
        primary_metric=raw.get("primary_metric", f"ndcg@{max(ks)}"),
        baseline=int(raw.get("baseline", 0)),
        generation=raw.get("generation", {}) or {}, match=raw.get("match", {}) or {})