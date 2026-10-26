import re
import zlib

import numpy as np
import pytest
import yaml

from lab import config
from lab.components.embedders.base import Embedder
from lab.components.registry import register
from lab.corpus.registry import CorpusRegistry
from lab.experiments.experiment_spec import expand_grid
from lab.experiments.grid_runner import run_experiment
from lab.experiments.run_store import load_run
from lab.testset.relevance_labels import Question, Span, save_questions

FACTS = {
    "alpha.md": "The zephyr protocol retries failed uploads exactly seven times before giving up. "
                "Unrelated filler text about gardens and tulips grows here for padding purposes.",
    "beta.md": "Quasar billing invoices are generated on the third business day of every month. "
               "More filler about mountains and rivers appears here to pad the document out.",
    "gamma.md": "The nimbus cache evicts entries using a least frequently used policy with decay. "
                "Even more filler about oceans and deserts is written here for length only.",
}


@register("embedder", "bow")
class BowEmbedder(Embedder):
    name = "bow"

    def _embed(self, texts, kind):
        out = np.zeros((len(texts), 64), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                out[i, zlib.crc32(w.encode()) % 64] += 1
        return out


def test_expand_grid_dedupes_bm25_across_embedders():
    cfgs = expand_grid({"chunker": [{"type": "fixed", "size": 50}],
                        "embedder": [{"type": "bow"}, {"type": "hf", "model": "x"}],
                        "retriever": [{"type": "bm25"}, {"type": "dense"}]})
    assert len(cfgs) == 3   # 1 bm25 + 2 dense


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    for name in ("CACHE_DIR", "INDEX_DIR", "RUNS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    for fn, text in FACTS.items():
        (corpus_dir / fn).write_text(text)
    corpus = CorpusRegistry.from_path(corpus_dir)
    qs = []
    for i, (fn, key, question) in enumerate([
            ("alpha.md", "retries failed uploads exactly seven times", "How many times does zephyr retry uploads?"),
            ("beta.md", "invoices are generated on the third business day", "When are quasar invoices generated?"),
            ("gamma.md", "least frequently used policy", "Which eviction policy does nimbus use?")]):
        t = corpus[fn].text
        s = t.index(key)
        qs.append(Question(f"q{i}", question, "factoid", key, [Span(fn, s, s + len(key))]))
    qs.append(Question("q_un", "What is the CEO's favourite colour?", "unanswerable", "", [], False))
    save_questions(qs, tmp_path / "dev.jsonl")
    spec = {"name": "t", "corpus": str(corpus_dir), "testset": str(tmp_path / "dev.jsonl"),
            "eval_k": [1, 3], "token_budgets": [40],
            "grid": {"chunker": [{"type": "fixed", "size": 12, "overlap": 2}],
                     "embedder": [{"type": "bow"}],
                     "retriever": [{"type": "bm25"}, {"type": "dense"}, {"type": "hybrid", "fusion": "rrf"}]}}
    (tmp_path / "exp.yaml").write_text(yaml.safe_dump(spec))
    return tmp_path / "exp.yaml"


def test_end_to_end_run(workspace):
    run_id = run_experiment(workspace, run_id="testrun")
    run = load_run(run_id)
    assert len(run.configs) == 3
    assert set(run.df.question_id) == {"q0", "q1", "q2", "q_un"}
    bm25_id = next(c for c, v in run.configs.items() if v["spec"]["retriever"]["type"] == "bm25")
    sub = run.df[(run.df.config_id == bm25_id) & run.df.answerable]
    assert sub["recall@3"].mean() == 1.0                       # distinctive keywords -> BM25 finds them
    assert run.df[run.df.question_id == "q_un"]["recall@3"].isna().all()   # unanswerable excluded
    assert (run.dir / "manifest.json").exists() and run.manifest["corpus_hash"]