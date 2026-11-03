"""
Load the rag-mini-wikipedia passages into the SAME stores the app uses
(Redis docstore + Chroma + BM25), using your own chunking code.

Run from the app/ folder, with the FastAPI server stopped:
    python -m evaluation.ingest_corpus
    python -m evaluation.ingest_corpus --group 10     # 10 passages per Document
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from llama_index.core import Document

from app.configs.config import settings
from ingestion.storage import Storage
from ingestion.embed import Embed
from ingestion.reset_store import reset_all
from ingestion.chunk import chunk_documents

RESULTS_DIR = Path(__file__).parent / "results"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", type=int, default=1,
                    help="Passages per Document. 1 = one Document per passage (default). "
                         "Use e.g. 10 if passages are shorter than a leaf chunk, so the "
                         "2048/512/128 hierarchy actually gets exercised.")
    ap.add_argument("--limit", type=int, default=None, help="Only ingest the first N passages (quick tests).")
    ap.add_argument("--yes", action="store_true", help="Skip the confirmation prompt.")
    return ap.parse_args()


def main():
    args = parse_args()

    ds = load_dataset("rag-datasets/rag-mini-wikipedia", "text-corpus")
    split = ds["passages"] if "passages" in ds else ds[next(iter(ds))]
    cols = split.column_names
    text_col = "passage" if "passage" in cols else next(c for c in cols if c != "id")
    print(f"Corpus columns: {cols} | using '{text_col}' as text | {len(split)} passages")

    passages = []
    for i, row in enumerate(split):
        text = str(row[text_col]).strip()
        if text:
            passages.append((row.get("id", i), text))
    if args.limit:
        passages = passages[: args.limit]

    docs = []
    for start in range(0, len(passages), args.group):
        chunk = passages[start : start + args.group]
        if args.group == 1:
            pid, text = chunk[0]
            doc_id = f"passage-{pid}"
        else:
            text = "\n\n".join(t for _, t in chunk)
            doc_id = f"group-{chunk[0][0]}-{chunk[-1][0]}"
        # No metadata on purpose: metadata counts against the chunk size.
        docs.append(Document(text=text, id_=doc_id))

    words = sum(len(t.split()) for _, t in passages)
    print(f"{len(docs)} Documents, ~{words // max(1, len(passages))} words per passage on average")

    if not args.yes:
        ans = input("This WIPES the current Redis/Chroma/BM25 stores (including any document loaded "
                    "in the app). Continue? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            return

    storage = Storage()
    embed_model = Embed().get_embedding()

    t0 = time.time()
    reset_all(storage)
    stats = chunk_documents(storage, embed_model, docs)
    print(f"Ingested in {time.time() - t0:.0f}s: {stats}")

    # Remember how the corpus was built, so run_eval can record it next to the scores.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "ingested_at": datetime.now().isoformat(timespec="seconds"),
        "passages": len(passages),
        "group": args.group,
        "documents": len(docs),
        "chunk_sizes": getattr(settings, "chunk_sizes", None),
        **{k: v for k, v in (stats or {}).items()},
    }
    (RESULTS_DIR / "corpus_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("Saved", RESULTS_DIR / "corpus_meta.json")


if __name__ == "__main__":
    main()
