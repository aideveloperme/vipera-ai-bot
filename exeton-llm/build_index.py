"""
Step 2 — Build the retrieval index (the bot's "memory" of exeton.com).

Embeds every product card / page chunk with a local embedding model on the
DGX Spark GPU and saves:
  data/index/docs.jsonl        the text chunks
  data/index/embeddings.npy    L2-normalised vectors (float32)

Re-run this after every scrape (e.g. nightly cron) — no LLM retraining needed.
"""

import argparse
import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from knowledge import DATA_DIR, build_documents, load_jsonl

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-model", default=EMBED_MODEL)
    args = ap.parse_args()

    products = load_jsonl(DATA_DIR / "products.jsonl")
    pages = load_jsonl(DATA_DIR / "pages.jsonl")
    docs = build_documents(products, pages)
    if not docs:
        raise SystemExit("No data found — run scrape_exeton.py first.")
    print(f"{len(products)} products + {len(pages)} pages → {len(docs)} chunks")

    model = SentenceTransformer(args.embed_model)
    vectors = model.encode([d["text"] for d in docs], batch_size=64,
                           normalize_embeddings=True, show_progress_bar=True)

    out = DATA_DIR / "index"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", vectors.astype(np.float32))
    with (out / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    (out / "meta.json").write_text(json.dumps({"embed_model": args.embed_model}))
    print(f"Index saved → {out}")


if __name__ == "__main__":
    main()
