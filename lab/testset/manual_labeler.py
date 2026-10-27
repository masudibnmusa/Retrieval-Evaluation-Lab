"""CSV flow for hand-labeling. Fill `evidence` with a verbatim snippet of the answer; offsets
are located for you. Use ' || ' to separate multiple evidence snippets (multi-hop)."""
from __future__ import annotations

import csv
from pathlib import Path

from lab.testset.relevance_labels import Question, Span, find_quote

COLUMNS = ["id", "question", "type", "answer", "evidence", "doc_id", "answerable"]
SEP = "||"


def write_template(path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerow(["m_001", "What does error E1042 mean?", "exact_term",
                    "An expired authentication token.", "E1042 indicates an expired authentication token.",
                    "docs/errors.md", "true"])
        w.writerow(["m_002", "Does the product support SSO via Kerberos?", "unanswerable", "", "", "", "false"])


def _locate(corpus, snippet: str, doc_id: str | None):
    docs = [corpus[doc_id]] if doc_id and doc_id in corpus else list(corpus)
    for d in docs:
        hit = find_quote(d.text, snippet)
        if hit:
            return Span(d.doc_id, hit[0], hit[1])
    return None


def import_labels(csv_path, corpus):
    questions, errors = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            qid = (row.get("id") or f"m_{i:03d}").strip()
            answerable = (row.get("answerable") or "true").strip().lower() not in ("false", "0", "no")
            snippets = [s.strip() for s in (row.get("evidence") or "").split(SEP) if s.strip()]
            doc_ids = [s.strip() for s in (row.get("doc_id") or "").split(SEP) if s.strip()]
            spans = []
            for j, snip in enumerate(snippets):
                did = doc_ids[j] if j < len(doc_ids) else (doc_ids[0] if len(doc_ids) == 1 else None)
                sp = _locate(corpus, snip, did)
                if sp:
                    spans.append(sp)
                else:
                    errors.append(f"row {i} ({qid}): evidence not found: {snip[:60]!r}")
            if answerable and not spans:
                errors.append(f"row {i} ({qid}): answerable but no span located; skipped")
                continue
            questions.append(Question(id=qid, question=row["question"].strip(),
                                      type=(row.get("type") or "factoid").strip(),
                                      answer=(row.get("answer") or "").strip(),
                                      spans=spans if answerable else [], answerable=answerable,
                                      source="manual"))
    return questions, errors