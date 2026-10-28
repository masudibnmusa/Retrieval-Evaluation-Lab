"""Global settings, cache paths, API keys. Other modules read `config.X` at call time
so tests can monkeypatch paths."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(os.getenv("LAB_ROOT", ".")).resolve()
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
TESTSET_DIR = DATA_DIR / "testsets"
CACHE_DIR = DATA_DIR / "cache"
INDEX_DIR = DATA_DIR / "indexes"
RUNS_DIR = DATA_DIR / "runs"

SEED = int(os.getenv("LAB_SEED", "42"))
LLM_MODEL = os.getenv("LAB_LLM_MODEL", "claude-sonnet-5-5")
JUDGE_MODEL = os.getenv("LAB_JUDGE_MODEL", LLM_MODEL)

# $ per 1M input/output tokens for LLM_MODEL (placeholders: set to your real prices)
LLM_PRICE_PER_M = (float(os.getenv("LAB_LLM_PRICE_IN", "3.0")),
                   float(os.getenv("LAB_LLM_PRICE_OUT", "15.0")))

# $ per 1M tokens (placeholders: verify against current provider pricing; local models are free)
EMBEDDING_PRICE_PER_M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "voyage-3.5": 0.06,
    "voyage-3-large": 0.18,
    "embed-v4.0": 0.12,
}
RERANK_PRICE_PER_SEARCH = {"rerank-v3.5": 0.002}


def require_key(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing {name}. Add it to .env (see .env.example).")
    return val


def ensure_dirs() -> None:
    for d in (CORPUS_DIR, TESTSET_DIR, CACHE_DIR, INDEX_DIR, RUNS_DIR):
        d.mkdir(parents=True, exist_ok=True)