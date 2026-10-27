"""Questions with answer spans (doc_id, start, end)."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Span:
    doc_id: str
    start: int
    end: int

    def to_dict(self):
        return asdict(self)


@dataclass
class Question:
    id: str
    question: str
    type: str = "factoid"
    answer: str = ""
    spans: list[Span] = field(default_factory=list)
    answerable: bool = True
    source: str = "manual"
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {"id": self.id, "question": self.question, "type": self.type, "answer": self.answer,
             "relevant_spans": [s.to_dict() for s in self.spans],
             "answerable": self.answerable, "source": self.source}
        if self.meta:
            d["meta"] = self.meta
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        spans = [Span(**s) for s in d.get("relevant_spans", [])]
        return cls(id=d["id"], question=d["question"], type=d.get("type", "factoid"),
                   answer=d.get("answer", ""), spans=spans,
                   answerable=d.get("answerable", bool(spans)),
                   source=d.get("source", "manual"), meta=d.get("meta", {}))


def load_questions(path) -> list[Question]:
    with open(path, encoding="utf-8") as f:
        return [Question.from_dict(json.loads(line)) for line in f if line.strip()]


def save_questions(questions: list[Question], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")


def find_quote(text: str, quote: str, start: int = 0):
    """Whitespace-insensitive search; returns (start, end) or None."""
    words = quote.split()
    if not words:
        return None
    m = re.compile(r"\s+".join(re.escape(w) for w in words)).search(text, start)
    return (m.start(), m.end()) if m else None


def validate_questions(questions: list[Question], corpus) -> list[str]:
    errs, seen = [], set()
    for q in questions:
        if q.id in seen:
            errs.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.answerable and not q.spans:
            errs.append(f"{q.id}: answerable but has no spans")
        for s in q.spans:
            if s.doc_id not in corpus:
                errs.append(f"{q.id}: unknown doc_id {s.doc_id}")
            elif not (0 <= s.start < s.end <= len(corpus[s.doc_id].text)):
                errs.append(f"{q.id}: span out of range {s}")
    return errs