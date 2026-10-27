"""Claim extraction + support check against context."""
from lab.llm import complete_json
from lab.metrics.judge import SYSTEM, Judge

EXTRACT = ('Break the text into atomic, self-contained factual claims.\n\nTEXT:\n{text}\n\n'
           'Return JSON: {{"claims": ["...", "..."]}}')
CHECK = ("For each numbered CLAIM, decide whether it is fully supported by the CONTEXT alone "
         "(true/false).\n\nCONTEXT:\n{context}\n\nCLAIMS:\n{claims}\n\n"
         'Return JSON: {{"verdicts": [true, false, ...]}} with one entry per claim, in order.')


def extract_claims(text: str, model=None) -> list[str]:
    data = complete_json(EXTRACT.format(text=text), system=SYSTEM, model=model, max_tokens=600, category="judge")
    claims = data.get("claims") if isinstance(data, dict) else []
    return [c for c in claims if isinstance(c, str) and c.strip()]


def support_rate(claims: list[str], context: str, judge: Judge) -> float | None:
    if not claims:
        return None
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    verdicts = judge.yes_no_list(CHECK.format(context=context, claims=numbered), len(claims))
    return sum(verdicts) / len(claims)


def faithfulness(answer: str, context: str, judge: Judge) -> float | None:
    """Fraction of the answer's claims supported by the retrieved context."""
    return support_rate(extract_claims(answer, judge.model), context, judge)