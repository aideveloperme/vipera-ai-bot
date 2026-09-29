"""
Step 4 (optional) — Build a fine-tuning dataset from the scraped catalog.

The goal of fine-tuning is to teach the model the Exeton *sales-engineer
behaviour* — reading the CONTEXT, quoting price/stock/URL exactly, refusing to
invent facts, asking qualifying questions — NOT to memorise prices (they change;
RAG supplies them at answer time). So every example contains a CONTEXT block,
exactly like chat_server.py builds at inference ("retrieval-augmented fine-tuning").

Answers are generated either by:
  • templates (default, no extra model needed), or
  • a larger "teacher" LLM served on the Spark (--teacher-url), which gives more
    natural answers. Review a sample by hand before training either way.

Output: data/sft.jsonl  — {"messages": [system, user, assistant]} per line
"""

import argparse
import json
import random

import requests

from knowledge import DATA_DIR, SYSTEM_PROMPT, load_jsonl, price_text, product_card

QUESTION_TEMPLATES = [
    "What is the price of the {name}?",
    "Is the {name} in stock?",
    "Can you give me the specs of {name}?",
    "I'm interested in {name}. Price and availability?",
    "Do you sell {short}? How much does it cost?",
    "Tell me about the {name}.",
    "Can I get a quote for {short}?",
]

VAGUE_QUESTIONS = [
    "I need a server for AI.",
    "What GPU should I buy?",
    "Recommend something for training LLMs.",
    "What do you have for inference?",
]

UNKNOWN_QUESTIONS = [
    "Do you sell {fake}?",
    "What's the price of the {fake}?",
]
FAKE_PRODUCTS = ["Quantum X9000 GPU", "NVIDIA H900 SXM9", "Supermicro SYS-999ZZ", "RTX 9090 Ti"]


def short_name(name):
    return " ".join(name.split()[:4])


def template_answer(p):
    specs = "; ".join(f"{k}: {v}" for k, v in list((p.get("attributes") or {}).items())[:6])
    summary = p.get("short_description") or (p.get("description") or "")[:300]
    lines = [f"**{p['name']}**" + (f" (SKU {p['sku']})" if p.get("sku") else "")]
    if summary:
        lines.append(summary)
    if specs:
        lines.append(f"Key specs: {specs}")
    lines.append(f"Price: {price_text(p)}")
    lines.append(f"Availability: {p.get('availability') or 'please confirm with sales'}")
    lines.append(f"Product page: {p.get('url')}")
    lines.append("Prices and availability can change — I'm happy to prepare a formal quote "
                 "or connect you with the Exeton sales team.")
    return "\n".join(lines)


def teacher_answer(url, model, system, question):
    r = requests.post(f"{url}/chat/completions", json={
        "model": model, "temperature": 0.3, "max_tokens": 600,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": question}],
    }, timeout=300)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-product", type=int, default=3)
    ap.add_argument("--distractors", type=int, default=3)
    ap.add_argument("--teacher-url", help="OpenAI-compatible URL of a larger LLM, e.g. http://localhost:8001/v1")
    ap.add_argument("--teacher-model", default="teacher")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    products = [p for p in load_jsonl(DATA_DIR / "products.jsonl") if p.get("name")]
    if len(products) < 2:
        raise SystemExit("Need scraped products — run scrape_exeton.py first.")
    rows = []

    def make(context_products, question, answer):
        ctx = "\n\n---\n\n".join(product_card(p) for p in context_products)
        system = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{ctx}"
        if args.teacher_url:
            answer = teacher_answer(args.teacher_url, args.teacher_model, system, question)
        rows.append({"messages": [{"role": "system", "content": system},
                                  {"role": "user", "content": question},
                                  {"role": "assistant", "content": answer}]})

    # 1. Grounded product questions (target product hidden among distractors)
    for p in products:
        for q in random.sample(QUESTION_TEMPLATES, min(args.per_product, len(QUESTION_TEMPLATES))):
            others = random.sample([o for o in products if o is not p],
                                   min(args.distractors, len(products) - 1))
            ctx = others + [p]
            random.shuffle(ctx)
            make(ctx, q.format(name=p["name"], short=short_name(p["name"])), template_answer(p))

    # 2. Vague requests → qualify the customer, then suggest options
    for q in VAGUE_QUESTIONS * 5:
        ctx = random.sample(products, min(4, len(products)))
        options = "\n".join(f"- {p['name']} — {price_text(p)} — {p.get('availability')} — {p.get('url')}"
                            for p in ctx)
        make(ctx, q,
             "Happy to help you pick the right system. A few quick questions:\n"
             "1. Is this for training, fine-tuning or inference?\n"
             "2. Which model sizes (e.g. 8B, 70B, 400B+) and how many users?\n"
             "3. Budget range and deployment (rack, power, cooling limits)?\n\n"
             f"Meanwhile, here are some options from our catalog:\n{options}")

    # 3. Not in context → say so, don't hallucinate
    for fake in FAKE_PRODUCTS:
        for q in UNKNOWN_QUESTIONS:
            ctx = random.sample(products, min(4, len(products)))
            make(ctx, q.format(fake=fake),
                 f"I couldn't find \"{fake}\" in the Exeton catalog I have access to. "
                 "Our sales team can check whether it can be sourced — would you like me to "
                 "connect you, or suggest a similar product that is available?")

    random.shuffle(rows)
    out = DATA_DIR / "sft.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} training examples → {out}")


if __name__ == "__main__":
    main()
