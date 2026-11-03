"""
Run your retrieval + generation pipeline over rag-mini-wikipedia questions and score it.

Run from the app/ folder, after `python -m evaluation.ingest_corpus`:
    python -m evaluation.run_eval --n 50 --run-name baseline
    python -m evaluation.run_eval --n 50 --run-name no_expand --no-expand
    python -m evaluation.run_eval --n 50 --baseline          # also ask the LLM with NO retrieval

NOTE: expand_query(model, ...) and final_response(model, ...) must be called the same way main.py calls them.
run_pipeline() below mirrors retrieval_pipeline/response_pipeline in main.py.
If you change one, change the other (or move that logic into a shared module).
"""
import argparse
import csv
import json
import random
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from app.configs.config import settings
from ingestion.storage import Storage
from ingestion.embed import Embed
from retrieval import hybrid_search
from retrieval.hybrid_search import build_hybrid_retriever, drop_covered_nodes
from retrieval.expand import expand_query
from retrieval.reranker import rerank_documents
from app.generation.models import model
from generation.response import final_response
from evaluation import eval_prompts as P

RESULTS_DIR = Path(__file__).parent / "results"
RERANK_MODEL = getattr(settings, "rerank_model", "rerank-v4.0-pro")
DEFAULT_TOP_N = getattr(settings, "rerank_top_n", 3)

FIELDS = [
    "idx", "question", "gold", "expanded_query", "answer", "passage_ids", "contexts",
    "in_candidates", "in_context", "correct", "faithful",
    "baseline_answer", "baseline_correct", "seconds", "error",
]


# ---------- helpers ----------
def normalize(s):
    s = re.sub(r"[^\w\s]", " ", str(s).lower())
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(gold, text):
    g, t = normalize(gold), normalize(text)
    return bool(g) and f" {g} " in f" {t} "


def is_yes_no(gold):
    return normalize(gold) in {"yes", "no"}


def as_text(x):
    return str(getattr(x, "content", x) or "").strip()


def ask_llm(user_text, system_prompt):
    """One call to the shared chat model (same `model` that main.py uses)."""
    reply = model.invoke([("system", system_prompt), ("human", user_text)])
    return as_text(reply)


def with_retry(fn, *args, tries=3, **kwargs):
    for attempt in range(tries):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** (attempt + 1))   # simple backoff for rate limits


def judge(system_prompt, message, positive, negative):
    """Ask the LLM for a one-word verdict on the first line. Returns True/False/None."""
    try:
        reply = with_retry(ask_llm, message, system_prompt)
    except Exception:
        return None
    first = reply.strip().splitlines()[0].upper() if reply.strip() else ""
    if negative in first:          # check the negative first: "INCORRECT" contains "CORRECT"
        return False
    if positive in first:
        return True
    return None


# ---------- the pipeline under test (mirrors main.py) ----------
def run_pipeline(question, retriever, docstore, use_expand, top_n):
    search_query = expand_query(model, question) if use_expand else question

    merged = retriever.retrieve(search_query)
    candidates = drop_covered_nodes(merged, docstore)
    texts = [c.node.get_content() for c in candidates]
    if not texts:
        return {"expanded": search_query, "candidate_texts": [], "contexts": [], "ids": [], "answer": ""}

    reranked = rerank_documents(question, texts)
    picked = [candidates[r.index] for r in reranked.results]
    contexts = [c.node.get_content() for c in picked]
    ids = sorted({c.node.ref_doc_id for c in picked if c.node.ref_doc_id})

    context = "\n\n---\n\n".join(contexts)
    answer = as_text(final_response(model, f"Query: {question}\nSource Documents:\n{context}"))
    return {"expanded": search_query, "candidate_texts": texts, "contexts": contexts, "ids": ids, "answer": answer}


# ---------- reporting ----------
def rate(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None and r.get(key) != ""]
    return (sum(1 for v in vals if v) / len(vals), len(vals)) if vals else (None, 0)


def fmt(rt):
    r, n = rt
    return "n/a" if r is None else f"{r:.2f} (n={n})"


def summarize(rows, args, run_name):
    ok = [r for r in rows if not r.get("error")]
    avg_s = sum(r["seconds"] for r in ok) / len(ok) if ok else 0

    print("\n" + "=" * 62)
    print(f"Run: {run_name} | questions: {len(rows)} | errors: {len(rows) - len(ok)}")
    print("=" * 62)
    print(f"answer in candidates (before rerank): {fmt(rate(ok, 'in_candidates'))}")
    print(f"answer in final context:              {fmt(rate(ok, 'in_context'))}")
    print(f"answer correct (LLM judge):           {fmt(rate(ok, 'correct'))}")
    print(f"answer faithful to context (judge):   {fmt(rate(ok, 'faithful'))}")
    if args.baseline:
        print(f"baseline correct (NO retrieval):      {fmt(rate(ok, 'baseline_correct'))}")
    print(f"avg seconds per question:             {avg_s:.1f}")

    # Where do the failures come from?
    graded = [r for r in ok if r.get("in_candidates") is not None]
    retrieval_miss = [r for r in graded if not r["in_candidates"]]
    rerank_drop = [r for r in graded if r["in_candidates"] and not r["in_context"]]
    gen_miss = [r for r in graded if r["in_context"] and r.get("correct") is False]
    print("\nWhere questions went wrong (string-matchable answers only):")
    print(f"  retrieval missed it entirely:       {len(retrieval_miss)}")
    print(f"  found, but reranker/merge dropped:  {len(rerank_drop)}")
    print(f"  in context, answer still wrong:     {len(gen_miss)}")

    wrong = [r for r in ok if r.get("correct") is False][:8]
    if wrong:
        print("\nFirst wrong answers to read by hand:")
        for r in wrong:
            print(f"  Q: {r['question']}\n     gold: {r['gold']}\n     got:  {r['answer'][:160]}\n"
                  f"     in_context={r['in_context']}")

    # One summary row per run, with the settings that produced it.
    corpus_meta = {}
    meta_path = RESULTS_DIR / "corpus_meta.json"
    if meta_path.exists():
        corpus_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "questions": len(rows),
        "errors": len(rows) - len(ok),
        "expand": not args.no_expand,
        "top_k": hybrid_search.TOP_K,
        "merge_ratio": hybrid_search.MERGE_RATIO,
        "rerank_top_n": args.top_n,
        "chunk_sizes": corpus_meta.get("chunk_sizes"),
        "corpus_group": corpus_meta.get("group"),
        "in_candidates": rate(ok, "in_candidates")[0],
        "in_context": rate(ok, "in_context")[0],
        "correct": rate(ok, "correct")[0],
        "faithful": rate(ok, "faithful")[0],
        "baseline_correct": rate(ok, "baseline_correct")[0] if args.baseline else None,
        "avg_seconds": round(avg_s, 2),
    }
    path = RESULTS_DIR / "summary.csv"
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nAppended to {path}")


# ---------- main ----------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50, help="How many questions to sample (0 = all).")
    ap.add_argument("--seed", type=int, default=0, help="Same seed = same questions, so runs are comparable.")
    ap.add_argument("--run-name", default="run")
    ap.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Blocks sent to the LLM after reranking.")
    ap.add_argument("--no-expand", action="store_true", help="Skip query expansion.")
    ap.add_argument("--no-faithfulness", action="store_true", help="Skip the faithfulness judge (saves LLM calls).")
    ap.add_argument("--baseline", action="store_true", help="Also answer with NO retrieval, to see what retrieval adds.")
    ap.add_argument("--sleep", type=float, default=0.0, help="Seconds to wait between questions (API rate limits).")
    return ap.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    qa_all = load_dataset("rag-datasets/rag-mini-wikipedia", "question-answer")
    qa = qa_all["test"] if "test" in qa_all else qa_all[next(iter(qa_all))]
    idxs = list(range(len(qa)))
    if args.n and args.n < len(idxs):
        idxs = sorted(random.Random(args.seed).sample(idxs, args.n))
    print(f"Evaluating {len(idxs)} of {len(qa)} questions (seed={args.seed})")

    storage = Storage()
    embed_model = Embed().get_embedding()
    retriever = build_hybrid_retriever(storage, embed_model)
    docstore = storage.redis()

    run_name = args.run_name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"{run_name}_{stamp}.csv"
    rows = []

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        for k, i in enumerate(idxs, 1):
            row = qa[i]
            question, gold = str(row["question"]).strip(), str(row["answer"]).strip()
            rec = {f: None for f in FIELDS}
            rec.update(idx=i, question=question, gold=gold)
            t0 = time.time()
            try:
                out = with_retry(run_pipeline, question, retriever, docstore, not args.no_expand, args.top_n)
                rec.update(
                    expanded_query=out["expanded"],
                    answer=out["answer"],
                    passage_ids=",".join(out["ids"]),
                    contexts=" ||| ".join(out["contexts"]),
                )
                if not is_yes_no(gold):   # string matching is meaningless for yes/no answers
                    rec["in_candidates"] = any(contains(gold, t) for t in out["candidate_texts"])
                    rec["in_context"] = any(contains(gold, t) for t in out["contexts"])

                rec["correct"] = judge(P.CORRECTNESS_JUDGE_PROMPT,
                                       P.correctness_message(question, gold, out["answer"]),
                                       "CORRECT", "INCORRECT")
                if not args.no_faithfulness and out["answer"] and out["contexts"]:
                    rec["faithful"] = judge(P.FAITHFULNESS_JUDGE_PROMPT,
                                            P.faithfulness_message("\n\n---\n\n".join(out["contexts"]), out["answer"]),
                                            "SUPPORTED", "UNSUPPORTED")
                if args.baseline:
                    base = with_retry(ask_llm, question, P.BASELINE_PROMPT)
                    rec["baseline_answer"] = base
                    rec["baseline_correct"] = judge(P.CORRECTNESS_JUDGE_PROMPT,
                                                    P.correctness_message(question, gold, base),
                                                    "CORRECT", "INCORRECT")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            rec["seconds"] = round(time.time() - t0, 2)

            writer.writerow(rec)
            fh.flush()                      # a crash never loses finished rows
            rows.append(rec)
            status = rec["error"] or f"in_ctx={rec['in_context']} correct={rec['correct']}"
            print(f"[{k}/{len(idxs)}] {status} | {question[:60]}")
            if args.sleep:
                time.sleep(args.sleep)

    print(f"\nPer-question results: {out_path}")
    summarize(rows, args, run_name)


if __name__ == "__main__":
    main()