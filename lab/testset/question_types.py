"""Question types and generation prompts."""
from __future__ import annotations

from enum import Enum


class QuestionType(str, Enum):
    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    EXACT_TERM = "exact_term"
    PARAPHRASE = "paraphrase"
    UNANSWERABLE = "unanswerable"


DEFAULT_MIX = {"factoid": 0.35, "paraphrase": 0.20, "exact_term": 0.20,
               "multi_hop": 0.15, "unanswerable": 0.10}

SYSTEM = ("You write evaluation questions for a retrieval system. "
          "Respond with a single JSON object and nothing else.")

GUIDANCE = {
    "factoid": "Write one specific question whose answer is stated explicitly in the passage. "
               "It must make sense to someone who has not seen the passage.",
    "paraphrase": "Write one question answerable from the passage, phrased the way a confused user "
                  "would ask it. Deliberately avoid reusing distinctive words or phrases from the "
                  "passage; use synonyms and a different sentence structure.",
    "exact_term": "Write one question that hinges on an exact identifier appearing verbatim in the "
                  "passage (error code, function name, setting key, product name, ID) and include "
                  "that identifier in the question.",
    "multi_hop": "Write one question that can only be answered by combining information from BOTH "
                 "passages. Neither passage alone may be sufficient.",
    "unanswerable": "Write one plausible question on the same topic that the passage does NOT answer. "
                    "It should sound natural but require information absent from the passage.",
}

_OUT = ('Return JSON: {"question": str, "answer": str, "evidence": [str, ...]} where evidence has one '
        'short verbatim quote (copied exactly, under 40 words) per passage that supports the answer.')
_OUT_UNANS = 'Return JSON: {"question": str, "answer": "", "evidence": []}'


def build_prompt(qtype: str, passages: list[str]) -> str:
    body = "\n\n".join(f"[Passage {i + 1}]\n{p}" for i, p in enumerate(passages))
    out = _OUT_UNANS if qtype == "unanswerable" else _OUT
    return f"{GUIDANCE[qtype]}\n\n{body}\n\n{out}"