# Retrieval Evaluation Lab

**Find out which chunking strategy, embedding model, search method, and reranker actually work best on *your* documents.**

Most RAG pipelines are built by copying a tutorial's defaults. This project replaces guesswork with measurement: generate a test set from your own corpus, sweep many pipeline configurations over it, and compare them using proper retrieval and generation metrics, alongside cost and latency.

---

## Core Idea

A RAG pipeline is just a **configuration**:

```
chunker + embedder + retriever + reranker + generator
```

The lab lets you:

1. Build a labeled question set where the answer locations are known
2. Sweep configurations over it
3. Score each on **retrieval quality** (recall@k, MRR, nDCG) and **answer quality** (faithfulness, correctness)
4. Report results with **cost and latency** so tradeoffs are visible
5. Tell whether a 2-point gain is real or noise

---

## Features

- **Fair comparison across chunkers** via span-level labels (character offsets) mapped to chunks by overlap
- **Flexible test sets**: synthetic (LLM-generated) and manual (hand-labeled gold standard)
- **Hard question types**: multi-hop, exact-term lookups (IDs, error codes), paraphrased, and unanswerable
- **Configurable sweeps** defined in YAML (grid or explicit list)
- **Cached embeddings and shared indexes** so large sweeps stay cheap
- **Retrieval metrics**: recall@k, precision@k, MRR, nDCG@k, hit rate
- **Generation metrics** (optional): faithfulness, answer correctness, answer relevance, context precision/recall via LLM judge
- **Cost and latency tracking**: index build time, storage, embedding cost, p50/p95 query latency, reranker cost
- **Statistics**: bootstrap confidence intervals and paired permutation tests
- **Error analysis**: bucket failures by cause and surface the worst questions
- **Reports**: leaderboard, per-question-type breakdown, Pareto plots, and a recommended configuration

---

## Configuration Space

| Component | Options |
|---|---|
| **Chunking** | Fixed-size (256/512/1024 tokens), recursive, sentence-based, semantic, structure-aware (by heading), configurable overlap, contextual augmentation |
| **Embeddings** | OpenAI, Cohere, Voyage, BGE, E5, sentence-transformers |
| **Retrieval** | Dense, BM25, hybrid (RRF or weighted), metadata filtering, query rewriting / HyDE / multi-query |
| **Reranking** | None, cross-encoder, Cohere Rerank, LLM-based |
| **Top-k / context size** | Number of chunks (or token budget) passed to the generator |

---

## How It Works

```
Corpus → loader → normalizer
              ↓
   synthetic_generator + manual_labeler → test set with answer spans
              ↓
   experiment.yaml → grid_runner (expand configurations)
              ↓
   chunker → embedder → index_builder (cached, shared across configs)
              ↓
   retriever (dense / BM25 / hybrid) → reranker → top-k chunks
              ↓
   span_matcher → retrieval_metrics (recall@k, MRR, nDCG)
              ↓
   [optional] generator → faithfulness / generation_metrics
              ↓
   run_store → significance → failure_analyzer → report_builder
```

1. **Corpus preparation**: load documents and assign stable document IDs
2. **Test set creation**: synthetic questions (the source passage is the label) plus a small hand-written set from real user questions
3. **Label alignment**: relevance is labeled as `(doc_id, start, end)` character spans and mapped to chunks by overlap, so different chunkers are scored on equal footing
4. **Indexing**: one index per `(chunker, embedder)` pair, with a hash-keyed embedding cache
5. **Evaluation**: retrieval metrics, then optional generation metrics
6. **Analysis and reporting**: significance tests, breakdowns, failure analysis, and a recommended configuration

---

## Project Structure

```
retrieval-eval-lab/
├── lab/
│   ├── cli.py                  # `lab run experiment.yaml`, `lab report <run-id>`
│   ├── config.py               # Global settings, cache paths, API keys
│   ├── corpus/                 # Loading, normalization (offset-preserving), doc registry
│   ├── testset/                # Synthetic + manual questions, quality filter, dev/held-out split
│   ├── components/
│   │   ├── chunkers/           # fixed, recursive, sentence, semantic, structure_aware, contextual
│   │   ├── embedders/          # openai, voyage, cohere, hf (BGE / E5 / MiniLM)
│   │   ├── retrievers/         # dense, bm25, hybrid, query_transform
│   │   └── rerankers/          # none, cross_encoder, cohere_rerank, llm_rerank
│   ├── indexing/               # Index builder, embedding cache, vector backends
│   ├── experiments/            # Spec parser, grid runner, parallel executor, run store
│   ├── metrics/                # Span matcher, retrieval/generation metrics, judge, cost & latency
│   ├── analysis/               # Significance, breakdowns, failure analysis, Pareto
│   └── reporting/              # Report builder, Streamlit dashboard, plots
├── experiments/                # Example YAML specs
├── data/                       # corpus, testsets, cache, indexes, runs
├── tests/
├── .env.example
├── pyproject.toml
├── README.md
└── run.sh
```

---

## Installation

```bash
git clone https://github.com/masudibnmusa/Retrieval-Evaluation-Lab.git
cd retrieval-eval-lab

python -m venv .venv
source .venv/bin/activate

pip install -e .
cp .env.example .env
```

Add the API keys you need to `.env`:

```env
OPENAI_API_KEY=
COHERE_API_KEY=
VOYAGE_API_KEY=
ANTHROPIC_API_KEY=
```

Local embedders (BGE, E5, sentence-transformers) and BM25 need no API keys.

---

## Quickstart

### 1. Add your documents

Place files in `data/corpus/`. Each document is assigned a stable ID and content hash.

### 2. Build a test set

```bash
# Hand-label real questions (recommended gold standard, even ~50 is valuable)
lab testset label --corpus data/corpus --out data/testsets/manual.jsonl

# Generate synthetic questions from sampled passages
lab testset generate --corpus data/corpus --n 300 --out data/testsets/synthetic.jsonl

# Split into dev (for tuning) and held-out (for final numbers)
lab testset split --in data/testsets/ --dev data/testsets/dev.jsonl --heldout data/testsets/heldout.jsonl
```

### 3. Define an experiment

`experiments/chunking_sweep.yaml`:

```yaml
name: chunking_sweep
testset: data/testsets/dev.jsonl

grid:
  chunker:
    - {type: fixed, size: 256, overlap: 32}
    - {type: fixed, size: 512, overlap: 64}
    - {type: fixed, size: 1024, overlap: 128}
    - {type: recursive, size: 512, overlap: 64}
    - {type: structure_aware}
  embedder:
    - {type: hf, model: BAAI/bge-small-en-v1.5}
  retriever:
    - {type: dense}
    - {type: bm25}
    - {type: hybrid, fusion: rrf}
  reranker:
    - {type: none}
  top_k: [5, 10]

evaluation:
  retrieval: [recall@k, mrr, ndcg@k, hit_rate]
  generation: false
```

### 4. Run and report

```bash
lab run experiments/chunking_sweep.yaml
lab report <run-id>

# Optional interactive explorer
streamlit run lab/reporting/dashboard.py
```

---

## Test Set Format

Each line in a `.jsonl` test set:

```json
{
  "id": "q_0042",
  "question": "What does error E1042 mean?",
  "type": "exact_term",
  "answer": "E1042 indicates an expired authentication token.",
  "relevant_spans": [
    {"doc_id": "docs/errors.md", "start": 10432, "end": 10611}
  ],
  "answerable": true,
  "source": "manual"
}
```

- `type`: `factoid`, `multi_hop`, `exact_term`, `paraphrase`, or `unanswerable`
- `relevant_spans`: character offsets into the **original** document (never the normalized or augmented text)
- Multi-hop questions list multiple spans
- Unanswerable questions have `answerable: false` and no spans

---

## Metrics

### Retrieval
| Metric | What it tells you |
|---|---|
| **Recall@k** | Whether the relevant spans appear in the top k chunks |
| **Precision@k** | How much of what you retrieved is relevant |
| **MRR** | How high the first relevant chunk is ranked |
| **nDCG@k** | Ranking quality with graded relevance |
| **Hit rate** | Fraction of questions with at least one relevant chunk retrieved |

> **Note:** Recall at a fixed *k* favors larger chunks. The lab also supports evaluation at a fixed **token budget** so chunk sizes can be compared fairly.

### Generation (optional)
- **Faithfulness**: are the answer's claims supported by the retrieved context?
- **Answer correctness**: agreement with the reference answer
- **Answer relevance**: does the answer address the question?
- **Context precision / recall**: how much retrieved text is useful, and how much needed information was retrieved

### Cost and latency
Index build time, storage size, embedding cost, p50/p95 query latency, reranker cost, and estimated $ per 1k queries.

---

## Statistical Comparison

- **Bootstrap confidence intervals** on every metric
- **Paired permutation tests** between configurations on the same questions
- Report the top few configurations against a baseline rather than declaring a winner from the full grid, since many comparisons produce false positives by chance

---

## Error Analysis

`failure_analyzer.py` buckets missed questions by likely cause:

- Chunk split the answer across a boundary
- Embedding retrieved a semantically similar but wrong passage
- Keyword miss (exact term not matched)
- Reranker demoted the correct chunk
- Label or test set issue

Findings feed directly into the next experiment.

---

## Recommended Practices

- **Start small.** Begin with 2 chunkers, 1 local embedder, dense + BM25, and ~50 manual questions before scaling up.
- **Use staged sweeps** (chunker, then embedder, then retriever, then reranker) to avoid grid explosion, and check key interactions afterward.
- **Tune on dev, report on held-out.** Never pick a configuration using the held-out set.
- **Compare synthetic vs. manual scores.** A large gap suggests question leakage.
- **Calibrate the LLM judge** against a small set of human labels, pin the judge model version, and cache its outputs.
- **Include sanity baselines**: BM25 alone, and an oracle that retrieves the gold span directly.
- **Record everything**: model versions, seeds, corpus hashes, and the full config are stored with each run for reproducibility.

---

## Testing

```bash
pytest tests/
```

Core tests cover the span matcher, retrieval metrics, chunkers, significance functions, and the grid runner.

---

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes, add tests for new components, and register new chunkers, embedders, retrievers, or rerankers through `components/registry.py`.

---

## License

MIT 