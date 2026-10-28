"""Persist per-question results for every config, plus a reproducibility manifest."""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path

import pandas as pd

from lab import config
from lab.testset.relevance_labels import load_questions


def _versions() -> dict:
    out = {"python": platform.python_version()}
    for p in ("numpy", "pandas", "rank-bm25", "sentence-transformers", "anthropic",
              "openai", "cohere", "voyageai", "faiss-cpu", "chromadb"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    return out


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RunStore:
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.dir = config.RUNS_DIR / run_id
        self.configs: list[dict] = []
        self.manifest: dict = {}

    @classmethod
    def create(cls, run_id, spec, corpus, questions, spec_path) -> "RunStore":
        s = cls(run_id)
        (s.dir / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy(spec_path, s.dir / "experiment.yaml")
        shutil.copy(spec.testset, s.dir / "testset.jsonl")
        corpus.save_manifest(s.dir / "corpus_manifest.json")
        s.manifest = {
            "run_id": run_id, "name": spec.name, "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": config.SEED, "llm_model": config.LLM_MODEL, "judge_model": config.JUDGE_MODEL,
            "corpus_hash": corpus.corpus_hash(), "n_docs": len(corpus),
            "testset_sha256": _sha(spec.testset), "n_questions": len(questions),
            "primary_metric": spec.primary_metric, "eval_k": spec.eval_k,
            "token_budgets": spec.token_budgets, "baseline": spec.baseline,
            "generation": spec.generation, "versions": _versions(),
        }
        s._write_manifest()
        return s

    def _write_manifest(self):
        (self.dir / "manifest.json").write_text(json.dumps(self.manifest, indent=2))

    def write_config(self, cfg, rows: list[dict], **extra):
        with open(self.dir / "results" / f"{cfg.id}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.configs.append({"id": cfg.id, "label": cfg.label, "spec": asdict(cfg),
                             "parts": cfg.parts(), "n_queries": len(rows), **extra})
        (self.dir / "configs.json").write_text(json.dumps(self.configs, indent=2))

    def finalize(self):
        self.manifest["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_manifest()


@dataclass
class RunData:
    run_id: str
    dir: Path
    manifest: dict
    configs: dict
    df: pd.DataFrame
    questions: dict


def load_run(run_id: str) -> RunData:
    d = config.RUNS_DIR / run_id
    if not d.exists():
        raise FileNotFoundError(f"No such run: {d}")
    manifest = json.loads((d / "manifest.json").read_text())
    configs = {c["id"]: c for c in json.loads((d / "configs.json").read_text())}
    rows = []
    for cid in configs:
        with open(d / "results" / f"{cid}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                flat = {"config_id": cid, "question_id": r["question_id"], "type": r["type"],
                        "answerable": r["answerable"], "latency_ms": r["latency_ms"],
                        "retrieved": r["retrieved"], "candidate_ids": r["candidate_ids"],
                        "answer": r.get("answer")}
                flat.update(r.get("metrics", {}))
                rows.append(flat)
    questions = {q.id: q for q in load_questions(d / "testset.jsonl")}
    return RunData(run_id, d, manifest, configs, pd.DataFrame(rows), questions)