"""Drop ambiguous or leaky questions."""
from __future__ import annotations

import re

from lab.testset.relevance_labels import Question

_STOP = set("the a an of to in on for and or is are was were be been what which who whom how why when where "
            "does do did can could should would with from by as at that this these those it its into about "
            "than then there their they them you your i we our not".split())
_MAX_OVERLAP = {"paraphrase": 0.5, "exact_term": 0.95}  # default 0.8
_CONTEXT_DEP = re.compile(r"^(this|that|these|those|it|they|the (above|following|passage|text|document))\b", re.I)
_REFERS_SRC = re.compile(r"\b(according to the (passage|text|document|article)|in the (passage|text|above))\b", re.I)


def _content_tokens(s: str) -> set[str]:
    return {w for w in re.findall(r"\w+", s.lower()) if w not in _STOP and len(w) > 2}


def content_overlap(question: str, passage: str) -> float:
    q = _content_tokens(question)
    return len(q & _content_tokens(passage)) / len(q) if q else 1.0


def has_verbatim_ngram(question: str, passage: str, n: int = 6) -> bool:
    q = re.findall(r"\w+", question.lower())
    p = " ".join(re.findall(r"\w+", passage.lower()))
    return any(" ".join(q[i:i + n]) in p for i in range(len(q) - n + 1))


def check_question(q: Question) -> str | None:
    """Return a drop reason, or None if the question is fine."""
    words = q.question.split()
    if len(words) < 3 or len(words) > 60:
        return "length"
    if q.answerable and not q.spans:
        return "no_span"
    if _CONTEXT_DEP.match(q.question.strip()):
        return "context_dependent"
    if _REFERS_SRC.search(q.question):
        return "refers_to_source"
    passage = " ".join(q.meta.get("passages", []))
    if passage:
        if has_verbatim_ngram(q.question, passage):
            return "copies_source"
        if q.type != "unanswerable" and content_overlap(q.question, passage) > _MAX_OVERLAP.get(q.type, 0.8):
            return "too_similar_to_source"
    return None


def filter_questions(questions: list[Question]):
    kept, dropped = [], []
    for q in questions:
        r = check_question(q)
        (dropped.append((q, r)) if r else kept.append(q))
    return kept, dropped