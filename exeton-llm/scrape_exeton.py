"""
Step 1 — Scrape exeton.com into clean JSONL files.

Strategy (tried in order):
  1. WooCommerce Store API (/wp-json/wc/store/v1/products) — structured
     name, SKU, price, stock status, categories, attributes. Fastest + most accurate.
  2. Sitemap crawl — every product page is parsed for schema.org JSON-LD
     (Product / Offer), which gives price + availability even on non-Woo sites.
  3. All other sitemap pages (about, warranty, shipping, contact, blog...) are
     saved as plain text so the bot can answer company/policy questions too.

Outputs:
  data/products.jsonl   one product per line
  data/pages.jsonl      one non-product page per line

Usage:
  python scrape_exeton.py                 # full scrape
  python scrape_exeton.py --max-pages 50  # quick test
"""

import argparse
import json
import re
import time
import urllib.robotparser
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://exeton.com"
USER_AGENT = "ExetonSalesBot/1.0 (+internal knowledge-base crawler)"
DATA_DIR = Path(__file__).parent / "data"

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean_html(html):
    if not html:
        return ""
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def get(url, delay, **kwargs):
    time.sleep(delay)
    resp = session.get(url, timeout=30, **kwargs)
    resp.raise_for_status()
    return resp


# ── 1. WooCommerce Store API ─────────────────────────────────
def format_price(prices):
    if not prices or not prices.get("price"):
        return None
    minor = int(prices.get("currency_minor_unit", 2))
    value = int(prices["price"]) / (10 ** minor)
    return {"amount": value, "currency": prices.get("currency_code", "USD")}


def scrape_store_api(delay):
    products, page = [], 1
    while True:
        url = f"{BASE_URL}/wp-json/wc/store/v1/products"
        try:
            resp = get(url, delay, params={"per_page": 100, "page": page})
        except requests.RequestException as e:
            if page == 1:
                print(f"  Store API not available ({e}); falling back to sitemap crawl")
                return None
            break
        batch = resp.json()
        if not batch:
            break
        for p in batch:
            availability = (p.get("stock_availability") or {}).get("text") or (
                "In stock" if p.get("is_in_stock") else "Out of stock"
            )
            products.append({
                "id": p.get("id"),
                "source": "store_api",
                "name": clean_html(p.get("name")),
                "sku": p.get("sku") or None,
                "url": p.get("permalink"),
                "price": format_price(p.get("prices")),
                "on_backorder": p.get("is_on_backorder", False),
                "in_stock": p.get("is_in_stock"),
                "availability": availability,
                "categories": [c.get("name") for c in p.get("categories", [])],
                "attributes": {
                    a.get("name"): ", ".join(t.get("name") for t in a.get("terms", []))
                    for a in p.get("attributes", [])
                },
                "short_description": clean_html(p.get("short_description")),
                "description": clean_html(p.get("description")),
                "scraped_at": now_iso(),
            })
        print(f"  Store API page {page}: {len(batch)} products")
        total_pages = int(resp.headers.get("X-WP-TotalPages", page))
        if page >= total_pages:
            break
        page += 1
    return products


# ── 2. Sitemap crawl ─────────────────────────────────────────
def sitemap_urls(delay):
    """Return every <loc> reachable from the site's sitemap(s)."""
    candidates = [f"{BASE_URL}/sitemap_index.xml", f"{BASE_URL}/sitemap.xml",
                  f"{BASE_URL}/wp-sitemap.xml"]
    try:
        robots = get(f"{BASE_URL}/robots.txt", delay).text
        candidates = re.findall(r"(?im)^sitemap:\s*(\S+)", robots) + candidates
    except requests.RequestException:
        pass

    seen, urls, queue = set(), [], list(dict.fromkeys(candidates))
    while queue:
        sm = queue.pop(0)
        if sm in seen:
            continue
        seen.add(sm)
        try:
            root = ElementTree.fromstring(get(sm, delay).content)
        except (requests.RequestException, ElementTree.ParseError):
            continue
        ns = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}
        for loc in root.findall(".//s:sitemap/s:loc", ns):
            queue.append(loc.text.strip())
        for loc in root.findall(".//s:url/s:loc", ns):
            urls.append(loc.text.strip())
    return list(dict.fromkeys(urls))


def find_jsonld_product(soup):
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        nodes = data if isinstance(data, list) else data.get("@graph", [data])
        for node in nodes:
            types = node.get("@type")
            types = types if isinstance(types, list) else [types]
            if "Product" in types:
                return node
    return None


def parse_product_page(url, soup, ld):
    offers = ld.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}
    price = offers.get("price") or offers.get("lowPrice")
    availability = (offers.get("availability") or "").rsplit("/", 1)[-1]
    availability = re.sub(r"(?<!^)(?=[A-Z])", " ", availability) or "Unknown"
    return {
        "id": None,
        "source": "jsonld",
        "name": clean_html(ld.get("name")),
        "sku": ld.get("sku") or ld.get("mpn"),
        "url": url,
        "price": {"amount": float(price), "currency": offers.get("priceCurrency", "USD")}
        if price not in (None, "") else None,
        "on_backorder": "BackOrder" in availability.replace(" ", ""),
        "in_stock": availability.replace(" ", "") in ("InStock", "LimitedAvailability"),
        "availability": availability,
        "categories": [c.get_text(strip=True) for c in soup.select(".posted_in a")],
        "attributes": extract_spec_table(soup),
        "short_description": clean_html(str(soup.select_one(
            ".woocommerce-product-details__short-description") or "")),
        "description": clean_html(ld.get("description")) or main_text(soup)[:4000],
        "scraped_at": now_iso(),
    }


def extract_spec_table(soup):
    specs = {}
    for row in soup.select("table tr"):
        cells = row.find_all(["th", "td"])
        if len(cells) == 2:
            k, v = cells[0].get_text(" ", strip=True), cells[1].get_text(" ", strip=True)
            if k and v and len(k) < 80:
                specs[k] = v
    return specs


def main_text(soup):
    for sel in ["script", "style", "nav", "header", "footer", "noscript", "form"]:
        for t in soup.select(sel):
            t.decompose()
    main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    return re.sub(r"\s+", " ", main.get_text(" ", strip=True))


def crawl_sitemap(delay, max_pages, skip_products):
    robots = urllib.robotparser.RobotFileParser(f"{BASE_URL}/robots.txt")
    try:
        robots.read()
    except Exception:
        robots = None

    site = urlparse(BASE_URL).netloc.removeprefix("www.")
    urls = sitemap_urls(delay)
    print(f"  Sitemap: {len(urls)} URLs")
    products, pages = [], []
    for i, url in enumerate(urls[:max_pages] if max_pages else urls):
        if urlparse(url).netloc.removeprefix("www.") != site:
            continue
        if robots and not robots.can_fetch(USER_AGENT, url):
            continue
        try:
            soup = BeautifulSoup(get(url, delay).text, "html.parser")
        except requests.RequestException as e:
            print(f"  skip {url}: {e}")
            continue
        ld = find_jsonld_product(soup)
        if ld:
            if not skip_products:
                products.append(parse_product_page(url, soup, ld))
        else:
            title = soup.title.get_text(strip=True) if soup.title else url
            text = main_text(soup)
            if len(text) > 200:
                pages.append({"url": url, "title": title, "text": text, "scraped_at": now_iso()})
        if (i + 1) % 25 == 0:
            print(f"  crawled {i + 1} URLs ({len(products)} products, {len(pages)} pages)")
    return products, pages


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows → {path}")


def main():
    global BASE_URL
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE_URL)
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between requests")
    ap.add_argument("--max-pages", type=int, default=0, help="limit sitemap crawl (0 = all)")
    args = ap.parse_args()
    BASE_URL = args.base_url.rstrip("/")

    print("[1/2] WooCommerce Store API")
    products = scrape_store_api(args.delay)

    print("[2/2] Sitemap crawl")
    crawled_products, pages = crawl_sitemap(args.delay, args.max_pages,
                                            skip_products=products is not None)
    if products is None:
        products = crawled_products

    write_jsonl(DATA_DIR / "products.jsonl", products)
    write_jsonl(DATA_DIR / "pages.jsonl", pages)


if __name__ == "__main__":
    main()
