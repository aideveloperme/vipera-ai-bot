"""
Find out how exeton.com is built and where its product data comes from.

  python probe_stack.py https://exeton.com/product-details/h274-a81
  python probe_stack.py <url> --browser    # also list the JSON/API calls the page makes
                                           # (pip install playwright && playwright install chromium)

Prints: server/framework clues, whether price/stock are in the raw HTML
(server-rendered) or only appear after JavaScript runs, structured data found,
and — with --browser — the API endpoints that return product JSON.
"""

import argparse
import json
import re

import requests
from bs4 import BeautifulSoup

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

MARKERS = {
    "Next.js (React)": [r"/_next/static", r"__NEXT_DATA__", r"next-route-announcer"],
    "Nuxt (Vue)": [r"/_nuxt/", r"__NUXT__"],
    "Angular": [r"ng-version=", r"ng-app"],
    "React (generic)": [r"data-reactroot", r"react-dom"],
    "Vue (generic)": [r"data-v-[0-9a-f]{6,}", r"vue(\.runtime)?(\.min)?\.js"],
    "Gatsby": [r"___gatsby"],
    "Laravel (PHP)": [r"laravel_session", r"XSRF-TOKEN", r'name="csrf-token"'],
    "Livewire (Laravel)": [r"wire:id", r"livewire"],
    "WordPress": [r"wp-content", r"wp-includes", r"/wp-json/"],
    "WooCommerce": [r"woocommerce", r"wc-blocks"],
    "Shopify": [r"cdn\.shopify\.com", r"_shopify_y", r"Shopify\.theme"],
    "Magento": [r"Magento_", r"mage/", r"X-Magento"],
    "BigCommerce": [r"bigcommerce", r"stencil"],
    "Wix": [r"wix\.com", r"_wixCIDX"],
    "Webflow": [r"webflow", r"data-wf-page"],
    "Squarespace": [r"squarespace"],
    "ASP.NET": [r"__VIEWSTATE", r"ASP\.NET", r"\.aspx"],
    "Django": [r"csrfmiddlewaretoken", r"django"],
    "Ruby on Rails": [r"csrf-param", r"authenticity_token"],
    "Cloudflare (CDN)": [r"cloudflare", r"cf-ray", r"__cf_bm"],
    "Vercel (hosting)": [r"x-vercel", r"vercel"],
    "Netlify (hosting)": [r"x-nf-request-id", r"netlify"],
    "Google Tag Manager": [r"googletagmanager\.com"],
}
PRICE_RE = re.compile(r"(?:US\$|\$|USD\s?|AED\s?|€|£)\s?\d[\d,]*(?:\.\d{2})?")
STOCK_RE = re.compile(r"in stock|out of stock|pre-?order|lead time|backorder|request a quote|call for price",
                      re.I)


def section(title):
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--browser", action="store_true", help="render with Playwright and log API calls")
    args = ap.parse_args()

    r = requests.get(args.url, headers={"User-Agent": UA}, timeout=30)
    html = r.text
    headers = "\n".join(f"{k}: {v}" for k, v in r.headers.items())
    cookies = "; ".join(r.cookies.keys())
    haystack = f"{headers}\n{cookies}\n{html}"

    section("HTTP")
    print(f"status {r.status_code}, {len(html):,} bytes of HTML")
    for h in ["server", "x-powered-by", "via", "x-vercel-id", "cf-ray", "x-cache", "set-cookie"]:
        if h in r.headers:
            print(f"{h}: {r.headers[h][:150]}")
    print(f"cookies: {cookies or '(none)'}")

    section("Technology clues")
    hits = [name for name, pats in MARKERS.items() if any(re.search(p, haystack, re.I) for p in pats)]
    print("\n".join(f"  ✔ {h}" for h in hits) or "  (no known markers)")

    soup = BeautifulSoup(html, "html.parser")
    scripts = [s["src"] for s in soup.find_all("script", src=True)]
    print(f"\nscript files ({len(scripts)}):")
    for s in scripts[:15]:
        print(f"  {s}")

    section("Is the product data in the raw HTML?")
    visible = soup.get_text(" ", strip=True)
    h1 = soup.find("h1")
    print(f"<h1>: {h1.get_text(' ', strip=True) if h1 else '(none — likely rendered by JavaScript)'}")
    print(f"prices in HTML: {sorted(set(PRICE_RE.findall(visible)))[:8] or 'none'}")
    print(f"stock wording : {sorted({m.lower() for m in STOCK_RE.findall(visible)}) or 'none'}")
    print(f"visible text  : {len(visible):,} chars")

    section("Structured data")
    for tag in soup.find_all("script", type="application/ld+json"):
        print("JSON-LD:", (tag.string or "")[:400].replace("\n", " "))
    nd = soup.find("script", id="__NEXT_DATA__")
    if nd:
        print("__NEXT_DATA__ keys:", list(json.loads(nd.string).get("props", {}).get("pageProps", {}))[:20])
    for m in re.findall(r"https?://[^\s\"'<>]*(?:/api/|graphql)[^\s\"'<>]*", html)[:10]:
        print("API URL in HTML:", m)
    for m in sorted(set(re.findall(r"[\"'](/api/[^\"'\s]+)", html)))[:10]:
        print("API path in HTML:", m)

    if args.browser:
        section("API calls made by the page (Playwright)")
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(user_agent=UA)
            calls = []

            def on_response(resp):
                ctype = resp.headers.get("content-type", "")
                if resp.request.resource_type in ("xhr", "fetch") or "json" in ctype:
                    try:
                        body = resp.text()[:300].replace("\n", " ")
                    except Exception:  # noqa: BLE001 — some bodies are unreadable (redirects)
                        body = ""
                    calls.append((resp.request.method, resp.status, resp.url, ctype, body))

            page.on("response", on_response)
            page.goto(args.url, wait_until="networkidle", timeout=60000)
            rendered = page.inner_text("body")
            browser.close()
        for method, status, url, ctype, body in calls:
            print(f"\n{method} {status} {url}\n  {ctype}\n  {body}")
        print(f"\nrendered text: {len(rendered):,} chars "
              f"(raw HTML had {len(visible):,}) — prices after JS: "
              f"{sorted(set(PRICE_RE.findall(rendered)))[:8] or 'none'}")

    section("What this means")
    if PRICE_RE.search(visible) or (h1 and len(visible) > 1500):
        print("Product content is server-rendered → scrape_exeton.py works without --render.")
    else:
        print("Little product content in raw HTML → run scrape_exeton.py --render, or better, "
              "use the product JSON API listed above (run this probe with --browser).")


if __name__ == "__main__":
    main()
