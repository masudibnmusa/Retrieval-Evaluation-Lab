"""Leaderboard and breakdowns by question type, doc type, query length."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from lab.analysis.significance import bootstrap_ci

_METRIC = re.compile(r"^((full_)?recall|precision|mrr|ndcg|hit)@\d+(tok)?$|^gen_")


def metric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if _METRIC.match(c)]


def add_question_attrs(df: pd.DataFrame, questions: dict) -> pd.DataFrame:
    df = df.copy()

    def doc_type(qid):
        q = questions.get(qid)
        if not q or not q.spans:
            return "n/a"
        return q.spans[0].doc_id.rsplit(".", 1)[-1].lower() if "." in q.spans[0].doc_id else "none"

    def qlen(qid):
        n = len(questions[qid].question.split()) if qid in questions else 0
        return "<=6 words" if n <= 6 else ("7-12 words" if n <= 12 else ">12 words")

    df["doc_type"] = df["question_id"].map(doc_type)
    df["query_length"] = df["question_id"].map(qlen)
    return df


def leaderboard(run, primary: str) -> pd.DataFrame:
    df, rows = run.df, []
    mcols = metric_columns(df)
    for cid, g in df.groupby("config_id"):
        mean, lo, hi = bootstrap_ci(g[primary].dropna().values) if primary in g else (np.nan,) * 3
        c = run.configs[cid]
        ret_usd = sum(c.get("retrieval_cost", {}).values())
        gen_usd = c.get("generation_cost", {}).get("generation", 0.0)
        st = c.get("index_stats", {})
        row = {"config_id": cid, "label": c["label"], **c["parts"], primary: mean, "ci_lo": lo, "ci_hi": hi,
               "n": int(g[primary].notna().sum()) if primary in g else 0,
               "latency_p50_ms": float(np.percentile(g["latency_ms"], 50)),
               "latency_p95_ms": float(np.percentile(g["latency_ms"], 95)),
               "query_usd_per_1k": 1000 * (ret_usd + gen_usd) / max(c["n_queries"], 1),
               "index_usd": st.get("index_usd", 0.0), "index_seconds": st.get("build_seconds", 0.0),
               "index_mb": st.get("storage_bytes", 0) / 1e6, "n_chunks": st.get("n_chunks", 0)}
        for m in mcols:
            if m != primary:
                row[m] = g[m].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values(primary, ascending=False).reset_index(drop=True)


def by_group(run, primary: str, group_col: str) -> pd.DataFrame:
    df = add_question_attrs(run.df, run.questions)
    df = df[df[primary].notna()]
    piv = df.pivot_table(index="config_id", columns=group_col, values=primary, aggfunc="mean")
    piv.insert(0, "label", [run.configs[c]["label"] for c in piv.index])
    return piv


def by_question_type(run, primary):
    return by_group(run, primary, "type")