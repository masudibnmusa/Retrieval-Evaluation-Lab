"""Markdown report: leaderboard, paired comparisons, breakdowns, Pareto, failures, recommendation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from lab.analysis.breakdowns import by_question_type, leaderboard, metric_columns
from lab.analysis.pareto import pareto_front, recommend
from lab.analysis.significance import holm_correction, paired_bootstrap_diff, paired_permutation_test
from lab.experiments.run_store import load_run
from lab.reporting import plots


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    fmt = lambda v: floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def paired_table(run, lb, primary, baseline_id, top_n=8) -> pd.DataFrame:
    piv = run.df.pivot(index="question_id", columns="config_id", values=primary)
    rows = []
    for cid in lb["config_id"].head(top_n):
        if cid == baseline_id:
            continue
        pair = piv[[cid, baseline_id]].dropna()
        mean, lo, hi = paired_bootstrap_diff(pair[cid], pair[baseline_id])
        rows.append({"config": run.configs[cid]["label"], "diff_vs_baseline": mean, "ci_lo": lo, "ci_hi": hi,
                     "p": paired_permutation_test(pair[cid], pair[baseline_id])})
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = holm_correction(t["p"].values)
    return t


def build_report(run_id: str, primary: str | None = None, top_n: int = 10) -> str:
    run = load_run(run_id)
    m = run.manifest
    primary = primary or m["primary_metric"]
    lb = leaderboard(run, primary)
    lb["pareto"] = pareto_front(lb, [primary], ["query_usd_per_1k", "latency_p50_ms"])
    rec, best = recommend(lb, primary)
    ids = [c["id"] for c in run.configs.values()]
    base_id = ids[min(m.get("baseline", 0), len(ids) - 1)]

    pdir = run.dir / "plots"
    pdir.mkdir(exist_ok=True)
    plots.bar_with_ci(lb, primary, pdir / "leaderboard.png")
    plots.pareto_plot(lb, primary, pdir / "pareto.png", mask=lb["pareto"].values)
    if lb["chunker"].nunique() > 1 and lb["embedder"].nunique() > 1:
        plots.heatmap(lb, primary, pdir / "heatmap.png")

    show = ["label", primary, "ci_lo", "ci_hi", "latency_p50_ms", "latency_p95_ms",
            "query_usd_per_1k", "index_usd", "index_mb"]
    other = [c for c in metric_columns(lb) if c != primary][:4]
    out = [f"# Retrieval Evaluation Report: `{run_id}`", "",
           f"- Corpus: {m['n_docs']} docs (hash `{m['corpus_hash'][:12]}`)",
           f"- Test set: {m['n_questions']} questions (sha `{m['testset_sha256'][:12]}`)",
           f"- Configurations: {len(lb)} | Primary metric: **{primary}**", "",
           "## Recommendation", "",
           f"Best by {primary}: **{best['label']}** ({best[primary]:.3f}, CI {best['ci_lo']:.3f}-{best['ci_hi']:.3f}).",
           f"Cheapest config statistically hard to distinguish from the best: **{rec['label']}** "
           f"({rec[primary]:.3f}, ${rec['query_usd_per_1k']:.4f}/1k queries, p50 {rec['latency_p50_ms']:.0f} ms).", "",
           "## Leaderboard", "", md_table(lb.head(top_n)[show + other + ["pareto"]]), "",
           "![leaderboard](plots/leaderboard.png)", "",
           f"## Paired comparison vs baseline (`{run.configs[base_id]['label']}`)", "",
           "Per-question paired bootstrap CI and sign-flip permutation test; Holm-corrected across the "
           "rows below. With many configurations, some gaps appear 'significant' by chance, so treat the "
           "top few against the baseline as candidates and confirm on the held-out set.", "",
           md_table(paired_table(run, lb, primary, base_id, top_n)), "",
           "## By question type", "", md_table(by_question_type(run, primary).reset_index(drop=True)
                                               .assign(label=lambda d: d["label"]).head(top_n)), "",
           "## Cost / latency tradeoff", "", "![pareto](plots/pareto.png)", ""]
    if (pdir / "heatmap.png").exists():
        out += ["## Chunker x embedder", "", "![heatmap](plots/heatmap.png)", ""]
    path = run.dir / "report.md"
    path.write_text("\n".join(out), encoding="utf-8")
    try:  # optional HTML
        import markdown
        (run.dir / "report.html").write_text(markdown.markdown("\n".join(out), extensions=["tables"]), encoding="utf-8")
    except ImportError:
        pass
    return str(path)