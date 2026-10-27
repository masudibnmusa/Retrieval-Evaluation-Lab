"""Generate an answer from retrieved context, then score it."""
from __future__ import annotations

import re

from lab.llm import complete
from lab.metrics.context_metrics import context_precision, context_recall
from lab.metrics.faithfulness import faithfulness
from lab.metrics.span_matcher import MatchConfig

GEN_SYSTEM = ("Answer the question using ONLY the provided context. Be concise. "
              "If the context does not contain the answer, reply exactly: I don't know.")
_ABSTAIN = re.compile(r"\bi (do not|don't|dont) know\b|not (enough|sufficient) information|"
                      r"cannot be determined|does not (say|mention|contain)", re.I)


def generate_answer(question: str, chunks, model: str | None = None) -> str:
    ctx = "\n\n---\n\n".join(c.text for c in chunks)
    return complete(f"Context:\n{ctx}\n\nQuestion: {question}", system=GEN_SYSTEM, model=model,
                    max_tokens=500, category="generation").strip()


def is_abstention(answer: str) -> bool:
    return bool(_ABSTAIN.search(answer))


def evaluate_generation(q, answer: str, chunks, judge, cfg: MatchConfig | None = None) -> dict:
    abstained = is_abstention(answer)
    if not q.answerable:
        return {"gen_abstained_correctly": float(abstained)}
    ctx = "\n\n---\n\n".join(c.text for c in chunks)
    out = {"gen_answered": float(not abstained)}
    corr = judge.score("correctness", question=q.question, reference=q.answer, answer=answer)
    rel = judge.score("relevance", question=q.question, answer=answer)
    if corr is not None:
        out["gen_correctness"] = corr
    if rel is not None:
        out["gen_relevance"] = rel
    if not abstained:
        f = faithfulness(answer, ctx, judge)
        if f is not None:
            out["gen_faithfulness"] = f
    cr = context_recall(q.answer, ctx, judge) if q.answer else None
    if cr is not None:
        out["gen_context_recall"] = cr
    cp = context_precision(chunks, q.spans, cfg)
    out["gen_context_precision"] = cp["chunk_precision"]
    out["gen_useful_char_ratio"] = cp["useful_char_ratio"]
    return out