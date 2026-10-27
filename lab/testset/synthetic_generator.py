"""LLM-generated questions from sampled passages. The evidence quote is located in the source
text to give a tight answer span (falls back to the whole passage)."""
from __future__ import annotations

import random
import re

from lab import config
from lab.llm import complete_json
from lab.testset.question_types import DEFAULT_MIX, SYSTEM, build_prompt
from lab.testset.quality_filter import check_question
from lab.testset.relevance_labels import Question, Span, find_quote


def _windows(text: str, target: int = 1200):
    paras, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        paras.append((pos, m.start()))
        pos = m.end()
    paras.append((pos, len(text)))
    paras = [p for p in paras if p[1] - p[0] >= 40]
    wins, cur = [], None
    for s, e in paras:
        if cur is None:
            cur = [s, e]
        elif e - cur[0] <= target:
            cur[1] = e
        else:
            wins.append(tuple(cur))
            cur = [s, e]
    if cur:
        wins.append(tuple(cur))
    return [(s, min(e, s + 2 * target)) for s, e in wins if e - s >= 200]


class PassageSampler:
    def __init__(self, corpus, rng: random.Random, target_chars: int = 1200):
        self.rng = rng
        w = {d.doc_id: _windows(d.text, target_chars) for d in corpus}
        self.windows = {k: v for k, v in w.items() if v}
        if not self.windows:
            raise ValueError("Corpus has no passages long enough to sample from.")

    def _doc(self, min_windows=1):
        ids = [k for k, v in self.windows.items() if len(v) >= min_windows]
        return self.rng.choices(ids, weights=[len(self.windows[i]) for i in ids])[0] if ids else None

    def one(self):
        d = self._doc()
        s, e = self.rng.choice(self.windows[d])
        return (d, s, e)

    def two(self):
        d = self._doc(min_windows=2)
        if d:
            (s1, e1), (s2, e2) = self.rng.sample(self.windows[d], 2)
            return [(d, s1, e1), (d, s2, e2)]
        return [self.one(), self.one()]


def _to_question(qtype, data, passages, texts, model) -> Question:
    spans = []
    if qtype != "unanswerable":
        ev = data.get("evidence") or []
        for i, (doc_id, ps, pe) in enumerate(passages):
            hit = find_quote(texts[i], ev[i]) if i < len(ev) and isinstance(ev[i], str) else None
            spans.append(Span(doc_id, ps + hit[0], ps + hit[1]) if hit else Span(doc_id, ps, pe))
    return Question(id="", question=data["question"].strip(), type=qtype,
                    answer=(data.get("answer") or "").strip(), spans=spans,
                    answerable=qtype != "unanswerable", source="synthetic",
                    meta={"passages": texts, "gen_model": model or config.LLM_MODEL})


def generate_questions(corpus, n: int, type_mix: dict | None = None, seed: int | None = None,
                       model: str | None = None, apply_filter: bool = True,
                       max_attempts_factor: int = 4) -> list[Question]:
    rng = random.Random(config.SEED if seed is None else seed)
    sampler = PassageSampler(corpus, rng)
    mix = type_mix or DEFAULT_MIX
    types, weights = list(mix), list(mix.values())
    out, seen, attempts = [], set(), 0
    while len(out) < n and attempts < n * max_attempts_factor:
        attempts += 1
        qtype = rng.choices(types, weights)[0]
        passages = sampler.two() if qtype == "multi_hop" else [sampler.one()]
        texts = [corpus.span_text(*p) for p in passages]
        data = complete_json(build_prompt(qtype, texts), system=SYSTEM, model=model,
                             temperature=0.3, max_tokens=600, category="testgen")
        if not isinstance(data, dict) or not isinstance(data.get("question"), str):
            continue
        q = _to_question(qtype, data, passages, texts, model)
        key = q.question.lower()
        if key in seen or (apply_filter and check_question(q)):
            continue
        seen.add(key)
        out.append(q)
    for i, q in enumerate(out):
        q.id = f"syn_{i:04d}"
    return out