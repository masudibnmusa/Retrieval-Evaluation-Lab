"""CLI: `lab run experiment.yaml`, `lab report <run-id>`, plus testset/corpus helpers."""
from __future__ import annotations

import argparse
from pathlib import Path


def _corpus(path):
    from lab.corpus.registry import CorpusRegistry
    return CorpusRegistry.from_path(path)


def cmd_run(a):
    from lab.experiments.grid_runner import run_experiment
    run_id = run_experiment(a.experiment, run_id=a.run_id)
    print(f"\nrun id: {run_id}\nnext:   lab report {run_id}")


def cmd_report(a):
    from lab.reporting.report_builder import build_report
    print("report written to", build_report(a.run_id, primary=a.primary))


def cmd_corpus_info(a):
    c = _corpus(a.corpus)
    print(f"{len(c)} documents, hash {c.corpus_hash()[:12]}")
    for d in c:
        print(f"  {d.doc_id}  ({len(d.text)} chars)")


def cmd_testset(a):
    from lab.testset import manual_labeler, relevance_labels, splitter
    if a.action == "template":
        manual_labeler.write_template(a.out)
        print("wrote template", a.out)
    elif a.action == "import":
        qs, errs = manual_labeler.import_labels(a.csv, _corpus(a.corpus))
        relevance_labels.save_questions(qs, a.out)
        print(f"imported {len(qs)} questions -> {a.out}")
        for e in errs:
            print("  WARN:", e)
    elif a.action == "generate":
        from lab.testset.synthetic_generator import generate_questions
        qs = generate_questions(_corpus(a.corpus), a.n, seed=a.seed)
        relevance_labels.save_questions(qs, a.out)
        print(f"generated {len(qs)} questions -> {a.out}")
    elif a.action == "split":
        qs = [q for p in a.inputs for q in relevance_labels.load_questions(p)]
        dev, held = splitter.split_questions(qs, heldout_frac=a.heldout_frac, seed=a.seed)
        relevance_labels.save_questions(dev, a.dev)
        relevance_labels.save_questions(held, a.heldout)
        print(f"dev={len(dev)} heldout={len(held)}")
    elif a.action == "validate":
        errs = relevance_labels.validate_questions(relevance_labels.load_questions(a.testset), _corpus(a.corpus))
        print("OK" if not errs else "\n".join(errs))


def cmd_calibrate(a):
    from lab.metrics.judge import Judge, calibrate
    print(calibrate(Judge(), a.csv, a.rubric))


def main(argv=None):
    p = argparse.ArgumentParser(prog="lab")
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("run"); r.add_argument("experiment"); r.add_argument("--run-id")
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report"); rp.add_argument("run_id"); rp.add_argument("--primary")
    rp.set_defaults(func=cmd_report)

    c = sub.add_parser("corpus"); cs = c.add_subparsers(required=True)
    ci = cs.add_parser("info"); ci.add_argument("--corpus", default="data/corpus")
    ci.set_defaults(func=cmd_corpus_info)

    t = sub.add_parser("testset"); ts = t.add_subparsers(dest="action", required=True)
    x = ts.add_parser("template"); x.add_argument("--out", default="data/testsets/manual_template.csv")
    x = ts.add_parser("import"); x.add_argument("--csv", required=True)
    x.add_argument("--corpus", default="data/corpus"); x.add_argument("--out", default="data/testsets/manual.jsonl")
    x = ts.add_parser("generate"); x.add_argument("--corpus", default="data/corpus")
    x.add_argument("--n", type=int, default=200); x.add_argument("--seed", type=int, default=None)
    x.add_argument("--out", default="data/testsets/synthetic.jsonl")
    x = ts.add_parser("split"); x.add_argument("--inputs", nargs="+", required=True)
    x.add_argument("--dev", default="data/testsets/dev.jsonl")
    x.add_argument("--heldout", default="data/testsets/heldout.jsonl")
    x.add_argument("--heldout-frac", type=float, default=0.3); x.add_argument("--seed", type=int, default=42)
    x = ts.add_parser("validate"); x.add_argument("--testset", required=True); x.add_argument("--corpus", default="data/corpus")
    t.set_defaults(func=cmd_testset)

    k = sub.add_parser("calibrate"); k.add_argument("--csv", required=True)
    k.add_argument("--rubric", default="correctness"); k.set_defaults(func=cmd_calibrate)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()