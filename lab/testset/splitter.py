"""Dev set for tuning, held-out set for final numbers (stratified by source+type)."""
from __future__ import annotations

import random
from collections import defaultdict

from lab.testset.relevance_labels import Question


def split_questions(questions: list[Question], heldout_frac: float = 0.3, seed: int = 42):
    groups = defaultdict(list)
    for q in questions:
        groups[(q.source, q.type)].append(q)
    rng = random.Random(seed)
    dev, held = [], []
    for key in sorted(groups):
        g = sorted(groups[key], key=lambda q: q.id)
        rng.shuffle(g)
        n_held = round(len(g) * heldout_frac)
        if len(g) >= 2:
            n_held = max(1, n_held)
        held += g[:n_held]
        dev += g[n_held:]
    return dev, held