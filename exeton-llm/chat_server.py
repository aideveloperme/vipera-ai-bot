"""
Step 3 — Exeton Sales-Engineer chat API (RAG).

  question ─► retrieve top product/page chunks ─► (optional) live price/stock
           ─► local LLM on DGX Spark (vLLM / Ollama / NIM, OpenAI-compatible)
           ─► grounded answer + source links

Run:
  uvicorn chat_server:app --host 0.0.0.0 --port 8000

Env:
  LLM_BASE_URL   OpenAI-compatible endpoint   (default http://localhost:8001/v1 → vLLM)
  LLM_MODEL      model name served there      (default: your fine-tuned model or base)
  LLM_API_KEY    if the endpoint needs one
  LIVE_REFRESH   1 = re-check price/stock on exeton.com for the retrieved products
  TOP_K          chunks to retrieve (default 6)
"""

import json
import os
import re
import time

import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from knowledge import DATA_DIR, SYSTEM_PROMPT, load_jsonl, price_text

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8001/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "exeton-sales")
LLM_API_KEY = os.getenv("LLM_API_KEY", "not-needed")
LIVE_REFRESH = os.getenv("LIVE_REFRESH", "1") == "1"
TOP_K = int(os.getenv("TOP_K", "6"))
BASE_URL = os.getenv("EXETON_BASE_URL", "https://exeton.com")

INDEX_DIR = DATA_DIR / "index"
docs = load_jsonl(INDEX_DIR / "docs.jsonl")
embeddings = np.load(INDEX_DIR / "embeddings.npy")
meta = json.loads((INDEX_DIR / "meta.json").read_text())
embedder = SentenceTransformer(meta["embed_model"])

app = FastAPI(title="Exeton Sales Engineer")


# ── Retrieval ────────────────────────────────────────────────
MODEL_TOKEN = re.compile(r"[A-Za-z]*\d+[A-Za-z0-9\-]*")  # H200, RTX5090, B200, 8x, SYS-821GE


def retrieve(query, k=TOP_K):
    q = embedder.encode([query], normalize_embeddings=True)[0]
    scores = embeddings @ q
    # Keyword boost: exact model numbers (e.g. "H200") matter a lot in hardware sales.
    tokens = {t.lower() for t in MODEL_TOKEN.findall(query) if len(t) >= 3}
    if tokens:
        for i, d in enumerate(docs):
            hay = f"{d.get('name', '')} {d.get('sku', '')}".lower()
            if any(t in hay for t in tokens):
                scores[i] += 0.15
    top = np.argsort(-scores)[:k]
    return [docs[i] | {"score": float(scores[i])} for i in top]


# ── Live price/stock refresh (WooCommerce Store API) ─────────
_live_cache = {}


def live_product(product_id, ttl=300):
    if not product_id:
        return None
    hit = _live_cache.get(product_id)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    try:
        r = requests.get(f"{BASE_URL}/wp-json/wc/store/v1/products/{product_id}", timeout=5)
        r.raise_for_status()
        p = r.json()
        minor = int(p["prices"].get("currency_minor_unit", 2))
        data = {
            "price": {"amount": int(p["prices"]["price"]) / 10 ** minor,
                      "currency": p["prices"].get("currency_code", "USD")}
            if p["prices"].get("price") else None,
            "availability": (p.get("stock_availability") or {}).get("text")
            or ("In stock" if p.get("is_in_stock") else "Out of stock"),
        }
    except (requests.RequestException, KeyError, ValueError):
        data = None
    _live_cache[product_id] = (time.time(), data)
    return data


def context_block(hits):
    parts = []
    for h in hits:
        text = h["text"]
        if LIVE_REFRESH and h["type"] == "product":
            live = live_product(h.get("product_id"))
            if live:
                text += (f"\nLIVE PRICE (just checked): {price_text(live)}"
                         f"\nLIVE AVAILABILITY (just checked): {live['availability']}")
        parts.append(text)
    return "\n\n---\n\n".join(parts)


# ── LLM call ─────────────────────────────────────────────────
def ask_llm(messages):
    r = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {LLM_API_KEY}"},
        json={"model": LLM_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 700},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


@app.post("/chat")
def chat(req: ChatRequest):
    # Include the previous user turn so follow-ups ("what about the price?") still retrieve.
    last_user = next((m["content"] for m in reversed(req.history) if m["role"] == "user"), "")
    hits = retrieve(f"{last_user} {req.message}".strip())
    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context_block(hits)}"},
        *req.history[-6:],
        {"role": "user", "content": req.message},
    ]
    return {
        "reply": ask_llm(messages),
        "sources": [{"name": h["name"], "url": h["url"], "score": round(h["score"], 3)}
                    for h in hits],
    }


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(docs), "llm": LLM_MODEL}
