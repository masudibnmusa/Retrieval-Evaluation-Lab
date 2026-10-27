"""LLM-as-judge with rubrics, plus calibration against human labels. Pin LAB_JUDGE_MODEL."""
from __future__ import annotations

import csv

from lab import config
from lab.llm import complete_json

SYSTEM = "You are a strict, consistent evaluator. Respond with a single JSON object and nothing else."

RUBRICS = {
    "correctness": ("Compare the ANSWER to the REFERENCE for the QUESTION.\n"
                    "5 = fully correct and complete; 4 = correct, minor omission; 3 = partially correct; "
                    "2 = mostly wrong; 1 = wrong or contradicts the reference."),
    "relevance": ("Does the ANSWER address the QUESTION that was asked?\n"
                  "5 = directly and fully; 3 = partially or with much irrelevant content; 1 = off-topic."),
}


class Judge:
    def __init__(self, model: str | None = None):
        self.model = model or config.JUDGE_MODEL

    def score(self, rubric: str, **fields) -> float | None:
        """Normalized to 0..1, or None if the judge output was unparseable."""
        body = "\n\n".join(f"{k.upper()}:\n{v}" for k, v in fields.items())
        prompt = (f"{RUBRICS[rubric]}\n\n{body}\n\n"
                  'Return JSON: {"rationale": "<one sentence>", "score": <integer 1-5>}')
        data = complete_json(prompt, system=SYSTEM, model=self.model, max_tokens=300, category="judge")
        try:
            return (int(data["score"]) - 1) / 4
        except (TypeError, KeyError, ValueError):
            return None

    def yes_no_list(self, prompt: str, n: int) -> list[bool]:
        data = complete_json(prompt, system=SYSTEM, model=self.model, max_tokens=600, category="judge")
        v = data.get("verdicts") if isinstance(data, dict) else None
        v = v if isinstance(v, list) else []
        return [bool(x) for x in v[:n]] + [False] * max(0, n - len(v))


def calibrate(judge: Judge, csv_path, rubric: str = "correctness") -> dict:
    """CSV columns: question, reference, answer, human_score (1-5)."""
    import numpy as np
    from scipy.stats import spearmanr
    human, model = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = judge.score(rubric, question=r["question"], reference=r["reference"], answer=r["answer"])
            if s is not None:
                human.append((float(r["human_score"]) - 1) / 4)
                model.append(s)
    h, m = np.array(human), np.array(model)
    return {"n": len(h), "spearman": float(spearmanr(h, m).correlation),
            "mean_abs_error": float(np.abs(h - m).mean()), "exact_agreement": float((h == m).mean())}