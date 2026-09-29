"""Shared helpers: turn scraped rows into text chunks the LLM can read."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Used both at inference (chat_server.py) and in the fine-tuning data, so they match.
SYSTEM_PROMPT = """You are the Exeton Sales Engineer assistant for exeton.com.
You help customers choose AI servers, GPUs, workstations and related hardware.

RULES
- Answer ONLY from the CONTEXT below. Never invent prices, stock levels, SKUs or specs.
- For every product you mention give: name, key specs, price, availability, and the product URL.
- If price says "Price on request", invite the customer to request a quote.
- If the answer is not in the CONTEXT, say you don't have that information and offer to
  connect them with the Exeton sales team.
- Act like a sales engineer: ask about their workload (training vs inference, model size,
  budget, rack/power limits) when the request is vague, and recommend suitable options.
- Mention that prices and availability can change and should be confirmed on checkout/quote.
- Be concise and professional."""


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def price_text(product):
    p = product.get("price")
    if not p or not p.get("amount"):
        return "Price on request (contact sales for a quote)"
    return f"{p['amount']:,.2f} {p.get('currency', 'USD')}"


def product_card(product, max_desc=1500):
    """Compact, fact-first text block for one product."""
    price, availability = price_text(product), product.get("availability") or "Unknown"
    if product.get("price_unverified"):
        # Listed only in hidden page data, not shown to shoppers — don't quote it as fact.
        price = (f"Not confirmed — site data lists {price}, but the product page does not show it; "
                 "ask sales for a quote")
        availability = f"{product.get('visible_availability') or 'Unknown'} (confirm with sales)"
    lines = [
        f"PRODUCT: {product.get('name')}",
        f"SKU: {product.get('sku') or 'n/a'}",
        f"PRICE: {price}",
        f"AVAILABILITY: {availability}",
    ]
    if product.get("categories"):
        lines.append(f"CATEGORIES: {', '.join(product['categories'])}")
    for k, v in (product.get("attributes") or {}).items():
        lines.append(f"{k}: {v}")
    if product.get("short_description"):
        lines.append(f"SUMMARY: {product['short_description']}")
    if product.get("description"):
        desc = product["description"]
        lines.append(f"DETAILS: {desc[:max_desc]}{'…' if len(desc) > max_desc else ''}")
    lines.append(f"URL: {product.get('url')}")
    lines.append(f"DATA AS OF: {product.get('scraped_at')}")
    return "\n".join(lines)


def chunk_words(text, size=350, overlap=50):
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def build_documents(products, pages):
    """One doc per product + overlapping chunks of every other page."""
    docs = []
    for p in products:
        docs.append({
            "type": "product",
            "product_id": p.get("id"),
            "name": p.get("name"),
            "sku": p.get("sku"),
            "url": p.get("url"),
            "text": product_card(p),
        })
    for page in pages:
        for i, chunk in enumerate(chunk_words(page["text"])):
            docs.append({
                "type": "page",
                "name": page.get("title"),
                "url": page.get("url"),
                "text": f"PAGE: {page.get('title')}\nURL: {page.get('url')}\n{chunk}",
                "chunk": i,
            })
    return docs
