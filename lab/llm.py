"""Thin Anthropic wrapper: disk cache, cost ledger, JSON helper, cached-latency tracking."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time

from lab import config
from lab.experiments.parallel_executor import retry
from lab.metrics.cost_latency import LEDGER

_client = None
_tl = threading.local()


def _anthropic():
    global _client
    if _client is None:
        import anthropic
        config.require_key("ANTHROPIC_API_KEY")
        _client = anthropic.Anthropic()
    return _client


def pop_cached_latency() -> float:
    """Seconds of original latency 'replayed' from cache hits on this thread."""
    v = getattr(_tl, "extra", 0.0)
    _tl.extra = 0.0
    return v


@retry(attempts=5)
def _call(model, system, prompt, max_tokens, temperature):
    kw = dict(model=model, max_tokens=max_tokens,
              messages=[{"role": "user", "content": prompt}])
    if system:
        kw["system"] = system
    if temperature is not None:
        kw["temperature"] = temperature
    resp = _anthropic().messages.create(**kw)
    text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
    return text, resp.usage.input_tokens, resp.usage.output_tokens


def complete(prompt: str, *, system: str | None = None, model: str | None = None,
             max_tokens: int = 1024, temperature: float | None = 0.0,
             cache: bool = True, category: str = "llm") -> str:
    model = model or config.LLM_MODEL
    key = hashlib.sha256(json.dumps([model, system, prompt, max_tokens, temperature]).encode()).hexdigest()
    d = config.CACHE_DIR / "llm"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{key}.json"
    if cache and path.exists():
        rec = json.loads(path.read_text())
    else:
        t0 = time.perf_counter()
        text, tin, tout = _call(model, system, prompt, max_tokens, temperature)
        rec = {"text": text, "tin": tin, "tout": tout, "seconds": time.perf_counter() - t0, "model": model}
        if cache:
            path.write_text(json.dumps(rec))
    # Always charge (notional) cost and replay latency so reruns report honest numbers
    usd = (rec["tin"] * config.LLM_PRICE_PER_M[0] + rec["tout"] * config.LLM_PRICE_PER_M[1]) / 1e6
    LEDGER.add(category, usd, rec["tin"] + rec["tout"])
    _tl.extra = getattr(_tl, "extra", 0.0) + (rec["seconds"] if cache and path.exists() and rec.get("_hit", True) else 0.0)
    return rec["text"]


def complete_json(prompt: str, **kw):
    text = complete(prompt, **kw)
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None