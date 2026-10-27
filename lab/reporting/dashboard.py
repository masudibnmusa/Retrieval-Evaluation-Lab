import streamlit as st

from lab import config
from lab.analysis.breakdowns import by_group, leaderboard, metric_columns
from lab.analysis.pareto import pareto_front
from lab.experiments.run_store import load_run

st.set_page_config(page_title="Retrieval Eval Lab", layout="wide")
runs = sorted((p.name for p in config.RUNS_DIR.iterdir() if (p / "manifest.json").exists()), reverse=True) \
    if config.RUNS_DIR.exists() else []
if not runs:
    st.warning("No runs found in data/runs/")
    st.stop()

run = load_run(st.sidebar.selectbox("Run", runs))
metrics = metric_columns(run.df)
primary = st.sidebar.selectbox("Metric", metrics, index=metrics.index(run.manifest["primary_metric"])
                               if run.manifest["primary_metric"] in metrics else 0)
lb = leaderboard(run, primary)
lb["pareto"] = pareto_front(lb, [primary], ["query_usd_per_1k", "latency_p50_ms"])

t1, t2, t3, t4 = st.tabs(["Leaderboard", "Breakdowns", "Tradeoffs", "Questions"])
with t1:
    st.dataframe(lb, use_container_width=True)
with t2:
    group = st.selectbox("Group by", ["type", "doc_type", "query_length"])
    st.dataframe(by_group(run, primary, group), use_container_width=True)
with t3:
    st.scatter_chart(lb, x="query_usd_per_1k", y=primary, color="pareto")
    st.scatter_chart(lb, x="latency_p50_ms", y=primary, color="pareto")
with t4:
    cid = st.selectbox("Config", list(run.configs), format_func=lambda c: run.configs[c]["label"])
    sub = run.df[run.df.config_id == cid]
    qid = st.selectbox("Question", sub.question_id.tolist())
    q, row = run.questions[qid], sub[sub.question_id == qid].iloc[0]
    st.markdown(f"**{q.question}**  \n_type: {q.type}_  \nreference: {q.answer}")
    st.write({m: row[m] for m in metrics if m in row and row[m] == row[m]})
    st.caption("Gold spans: " + ", ".join(f"{s.doc_id}[{s.start}:{s.end}]" for s in q.spans))
    st.dataframe(row["retrieved"], use_container_width=True)
    if row.get("answer"):
        st.markdown(f"**Generated answer:** {row['answer']}")