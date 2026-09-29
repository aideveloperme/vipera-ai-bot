"""
Step 1 — Scrape exeton.com into clean JSONL files.

Strategy (tried in order):
  1. WooCommerce Store API (/wp-json/wc/store/v1/products) — structured
     name, SKU, price, stock status, categories, attributes. Fastest + most accurate.
  2. Site crawl — URLs from the sitemap + links found on /all-products
     (exeton.com product pages live at /product-details/<slug>). Each product
     page is parsed from schema.org JSON-LD when present, otherwise from the
     visible page (h1, price text, stock wording, spec tables).
     Use --render if the site builds pages with JavaScript (needs Playwright).
  3. All other pages (about, warranty, shipping, contact, blog...) are
     saved as plain text so the bot can answer company/policy questions too.

Outputs:
  data/products.jsonl   one product per line
  data/pages.jsonl      one non-product page per line

Usage:
  python scrape_exeton.py                 # full scrape
  python scrape_exeton.py --max-pages 50  # quick test
  python scrape_exeton.py --render        # JS-rendered site (pip install playwright)
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
LISTING_PATHS = ["/all-products"]                        # pages that link to every product
PRODUCT_URL_RE = re.compile(r"/(product-details|product)/[^/?#]+/?$")
PRICE_RE = re.compile(r"(?:US\$|\$|USD\s?|AED\s?|€|£)\s?(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d{2}))?")
STOCK_PHRASES = [  # checked in order; first match wins
    "out of stock", "sold out", "pre-order", "preorder", "backorder", "limited stock",
    "limited availability", "in stock", "available now", "ships in", "lead time",
    "contact for pricing", "call for price", "request a quote", "get a quote", "contact sales",
    "inquire",
]
IN_STOCK = ("in stock", "available now", "limited stock", "limited availability")
QUOTE_ONLY = ("contact for pricing", "call for price", "request a quote", "get a quote",
              "contact sales", "inquire")
MPN_RE = re.compile(r"\b(?:MPN|SKU|Part\s*(?:No\.?|Number))\s*[:#]\s*([A-Z0-9][\w\-./+]{2,})", re.I)

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


_browser = None


def get_html(url, delay, render=False):
    """Page HTML — raw, or after JavaScript has run (Playwright) when render=True."""
    if not render:
        return get(url, delay).text
    global _browser
    if _browser is None:
        from playwright.sync_api import sync_playwright
        _browser = sync_playwright().start().chromium.launch()
    time.sleep(delay)
    page = _browser.new_page(user_agent=USER_AGENT)
    try:
        page.goto(url, wait_until="networkidle", timeout=60000)
        return page.content()
    finally:
        page.close()


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
        "categories": [c.get_text(strip=True) for c in soup.select(".posted_in a")] or breadcrumbs(soup),
        "attributes": extract_spec_table(soup),
        "short_description": clean_html(str(soup.select_one(
            ".woocommerce-product-details__short-description") or "")),
        "description": clean_html(ld.get("description")) or main_text(soup)[:4000],
        "scraped_at": now_iso(),
    }


def parse_visible_product(url, soup):
    """Fallback when a product page has no JSON-LD: read what a customer sees."""
    h1 = soup.find("h1")
    h1 = h1.get_text(" ", strip=True) if h1 else ""
    og = soup.find("meta", property="og:title")
    title = (og.get("content") if og else None) or (soup.title.get_text(strip=True) if soup.title else "")
    title = re.sub(r"^(buy|shop)\s+|\s+(online|for sale)$", "", title.split("|")[0].strip(), flags=re.I)
    # exeton.com often uses the bare model code as <h1> ("B343-C40"); prefer a longer page title.
    name = h1 or title or url
    if title and len(title) > len(h1) and h1.lower() in title.lower():
        name = title
    specs = extract_spec_table(soup)
    text = main_text(soup)

    # Look for the price after the product title, so related-product prices lower down lose.
    start = max(text.find(name[:40]), 0) if name else 0
    price = None
    m = PRICE_RE.search(text, start) or PRICE_RE.search(text)
    if m:
        amount = float(m.group(1).replace(",", "") + "." + (m.group(2) or "00"))
        cur = m.group(0).strip()
        currency = "AED" if cur.startswith("AED") else "EUR" if cur.startswith("€") \
            else "GBP" if cur.startswith("£") else "USD"
        if amount > 0:
            price = {"amount": amount, "currency": currency}

    lowered = text.lower()
    availability = next((p for p in STOCK_PHRASES if p in lowered), None)
    if availability in QUOTE_ONLY:
        price = None  # a "$" figure on a quote-only page is not this product's price
    mpn = MPN_RE.search(text)
    sku = (mpn and mpn.group(1)) or next(
        (v for k, v in specs.items() if k.strip().lower() in ("sku", "part number", "model", "mpn")), None)
    return {
        "id": None,
        "source": "html",
        "name": name,
        "sku": sku,
        "url": url,
        "price": price,
        "on_backorder": availability in ("backorder", "pre-order", "preorder"),
        "in_stock": None if availability in QUOTE_ONLY or not availability
        else availability in IN_STOCK,
        "availability": "Contact sales for pricing and availability (request a quote)"
        if availability in QUOTE_ONLY
        else availability.capitalize() if availability else "Unknown — confirm with sales",
        "categories": breadcrumbs(soup),
        "attributes": specs,
        "short_description": "",
        "description": text[:6000],
        "scraped_at": now_iso(),
    }


def check_jsonld_price(product, soup, visible):
    """Hidden JSON-LD prices can be template placeholders. Trust one only if the same
    amount is actually shown on the page; otherwise mark it unverified."""
    price = product.get("price")
    if not price:
        return product
    amount = price["amount"]
    shown = {float(m.group(1).replace(",", "") + "." + (m.group(2) or "00"))
             for m in PRICE_RE.finditer(main_text(soup))}
    if amount not in shown:
        product["price_unverified"] = True
        product["visible_price"] = visible.get("price")
        product["visible_availability"] = visible.get("availability")
    return product


def breadcrumbs(soup):
    """Category path from BreadcrumbList JSON-LD or a breadcrumb nav (minus Home and the product)."""
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except json.JSONDecodeError:
            continue
        for node in (data if isinstance(data, list) else data.get("@graph", [data])):
            if node.get("@type") == "BreadcrumbList":
                names = [(i.get("name") or (i.get("item") or {}).get("name") or "")
                         for i in node.get("itemListElement", [])]
                return [n for n in names[1:-1] if n]
    links = [a.get_text(strip=True) for a in soup.select(
        "nav[aria-label*=readcrumb] a, .breadcrumb a, [class*=breadcrumb] a")]
    return [n for n in links[1:] if n]


def extract_spec_table(soup):
    specs = {}
    for row in soup.select("table tr"):
        cells = row.find_all(["th", "td"])
        if all(c.name == "th" for c in cells):  # header row ("Specification | Value")
            continue
        if len(cells) == 2:
            k, v = cells[0].get_text(" ", strip=True), cells[1].get_text(" ", strip=True)
            if k and v and len(k) < 80:
                specs[k] = v
    return specs


def main_text(soup):
    soup = BeautifulSoup(str(soup), "html.parser")  # don't mutate the caller's tree
    for sel in ["script", "style", "nav", "header", "footer", "noscript", "form"]:
        for t in soup.select(sel):
            t.decompose()
    main = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    text = re.sub(r"[↑↓←→‹›]", " ", main.get_text(" ", strip=True))  # UI arrows
    return re.sub(r"\s+", " ", text).strip()


def listing_product_urls(delay, render, max_listing_pages=50):
    """Collect product links from listing pages (follows ?page=N style pagination).

    Also returns the site's other internal links (menu/footer: about, contact,
    warranty...) so company pages are covered even without a sitemap."""
    site = urlparse(BASE_URL).netloc.removeprefix("www.")
    found, other, seen, queue = [], [], set(), [BASE_URL + p for p in LISTING_PATHS]
    while queue and len(seen) < max_listing_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            soup = BeautifulSoup(get_html(url, delay, render), "html.parser")
        except Exception as e:  # noqa: BLE001 — a missing listing page is not fatal
            print(f"  listing {url}: {e}")
            continue
        for a in soup.find_all("a", href=True):
            link = urljoin(url, a["href"]).split("#")[0]
            if PRODUCT_URL_RE.search(urlparse(link).path):
                found.append(link)
            elif re.search(r"[?&]page=\d+|/page/\d+", link) and \
                    urlparse(link).path == urlparse(url).path:
                queue.append(link)
            elif urlparse(link).netloc.removeprefix("www.") == site and "?" not in link:
                other.append(link.rstrip("/") or BASE_URL)
    return list(dict.fromkeys(found)), list(dict.fromkeys(other))


def crawl_site(delay, max_pages, skip_products, render=False):
    robots = urllib.robotparser.RobotFileParser(f"{BASE_URL}/robots.txt")
    try:
        robots.read()
    except Exception:
        robots = None

    site = urlparse(BASE_URL).netloc.removeprefix("www.")
    urls = sitemap_urls(delay)
    print(f"  Sitemap: {len(urls)} URLs")
    listed, other = listing_product_urls(delay, render)
    print(f"  Listing pages: {len(listed)} product URLs, {len(other)} other site links")
    urls = list(dict.fromkeys(listed + urls + other))
    products, pages = [], []
    for i, url in enumerate(urls[:max_pages] if max_pages else urls):
        if urlparse(url).netloc.removeprefix("www.") != site:
            continue
        if robots and not robots.can_fetch(USER_AGENT, url):
            continue
        try:
            soup = BeautifulSoup(get_html(url, delay, render), "html.parser")
        except Exception as e:  # noqa: BLE001 — skip any page that fails to load
            print(f"  skip {url}: {e}")
            continue
        ld = find_jsonld_product(soup)
        if ld or PRODUCT_URL_RE.search(urlparse(url).path):
            if not skip_products:
                visible = parse_visible_product(url, soup)
                products.append(check_jsonld_price(parse_product_page(url, soup, ld), soup, visible)
                                if ld else visible)
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
    ap.add_argument("--force", action="store_true",
                    help="save results even if far fewer products were found than last time")
    ap.add_argument("--render", action="store_true",
                    help="render pages with a headless browser (for JavaScript-built sites)")
    args = ap.parse_args()
    BASE_URL = args.base_url.rstrip("/")

    print("[1/2] WooCommerce Store API")
    products = scrape_store_api(args.delay)

    print("[2/2] Site crawl (sitemap + listing pages)")
    crawled_products, pages = crawl_site(args.delay, args.max_pages,
                                         skip_products=products is not None, render=args.render)
    if products is None:
        products = crawled_products

    # Safety net for the nightly cron: if the site was down or changed, don't wipe good data.
    previous = sum(1 for _ in (DATA_DIR / "products.jsonl").open()) \
        if (DATA_DIR / "products.jsonl").exists() else 0
    if previous and len(products) < previous * 0.5 and not args.force:
        raise SystemExit(f"Only {len(products)} products found (last run: {previous}). Keeping the old "
                         "data. Check the site, then re-run with --force to accept the new result.")

    unverified = [p for p in products if p.get("price_unverified")]
    if unverified:
        print(f"\n⚠ {len(unverified)} products have a price in the site's hidden data (JSON-LD) that "
              "is NOT shown on the page — the bot will tell customers to confirm with sales:")
        for p in unverified[:15]:
            print(f"   {p['price']['amount']:>12,.2f}  {p['name'][:60]}  {p['url']}")
    write_jsonl(DATA_DIR / "products.jsonl", products)
    write_jsonl(DATA_DIR / "pages.jsonl", pages)


if __name__ == "__main__":
    main()
