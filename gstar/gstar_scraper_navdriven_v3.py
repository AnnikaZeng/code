#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nav-driven product scraper (CSV -> products), with **--only-markets** to pick one or more countries
without splitting CSV files.

Input CSV schema (from nav dumper):
    market,section,branch,category,url
- `market` may be empty; we'll infer from url like https://www.g-star.com/en_gb/...

Examples
--------
# Only crawl UK from a merged all_nav_filtered.csv
python gstar_scraper_navdriven_v3.py \
  --nav-csv all_nav_filtered.csv --only-markets en_gb \
  --sections men,women --out uk_products.csv --xlsx uk_products.xlsx --headful

# Crawl FR + NL only
python gstar_scraper_navdriven_v3.py --nav-csv all_nav_filtered.csv \
  --only-markets en_fr,en_nl --out fr_nl.csv

# Crawl everything
python gstar_scraper_navdriven_v3.py --nav-csv all_nav_filtered.csv --sections men,women --out all.csv
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from playwright.sync_api import (
    Browser,
    Error,
    Page,
    Playwright,
    TimeoutError,
    sync_playwright,
)

BASE = "https://www.g-star.com"
LOCALE_RGX = re.compile(r"https?://www\.g-star\.com/([^/]+)/")

SEL: Dict[str, str] = {
    "plp_grid": 'section[data-testid="plp-grid"]',
    "product_tile": 'div[data-testid="product-tile"]',
    "tile_link": 'a[data-testid="product-tile-link"], a[href*="/product/"]',
    "tile_name": '[data-testid="product-title-title"], [data-testid="product-title"]',
    "tile_price": '[data-testid*="price" i], [class*="price" i]',
    "tile_imgs": "picture img, img",
    "show_next_btn": '[data-testid="productList-showNext"]',
    "total_products": '[data-testid="total-number-products"]',
}

CURRENCY_RGX = re.compile(
    r"([€£$])\s?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)"
)
COLORS_RGX = re.compile(r"(\d+)\s+colors?\s+available", re.IGNORECASE)


def _clean(s: str) -> str:
    s = re.sub(r"[\u00A0\s]+", " ", s or "")
    s = re.sub(r"\s*&\s*", " & ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Product:
    market: str
    section: str
    branch: str
    category: str
    title: str
    price_current: Optional[str]
    price_original: Optional[str]
    product_url: str
    image_url: Optional[str]
    colors_available: Optional[int]
    colors_text: Optional[str]
    order: int = 0
    expected_total: Optional[int] = None

    def key(self) -> str:
        return self.product_url.split("?")[0].split("#")[0]


class Scraper:
    def __init__(
        self,
        pw: Playwright,
        headless: bool,
        timeout_ms: int,
        max_per_cat: Optional[int],
        verbose: bool,
    ) -> None:
        self.browser: Browser = pw.chromium.launch(headless=headless)
        self.timeout = timeout_ms
        self.max_per_cat = max_per_cat
        self.verbose = verbose
        self.throttle = (0.5, 1.4)

    def run(
        self,
        tasks: List[Dict[str, str]],
        sections: Iterable[str],
        only_markets: Optional[Set[str]],
    ) -> List[Product]:
        page = self.browser.new_page()
        out: List[Product] = []
        dedupe: Set[str] = set()
        secset = {s.strip().lower() for s in sections}

        for r in tasks:
            sec = (r.get("section") or "").strip().lower()
            if sec and sec not in secset:
                continue
            url = r.get("url") or ""
            if not url:
                continue
            market = (r.get("market") or "").strip().lower() or self._market_from_url(
                url
            )
            if only_markets and market not in only_markets:
                continue
            branch = _clean(r.get("branch") or "")
            cat = _clean(r.get("category") or "")

            self._log(f"↳ {market} · {sec} · {branch} · {cat} -> {url}")
            arr = self._collect_products(page, market, sec, branch, cat, url)
            added = 0
            for p in arr:
                k = p.key() + f"|{market}"
                if k in dedupe:
                    continue
                dedupe.add(k)
                out.append(p)
                added += 1
            self._log(
                f"  [done] {cat}: {added} products (expected={arr[0].expected_total if arr else 'n/a'})"
            )

        page.close()
        self.browser.close()
        return out

    # ---- per PLP ----
    def _collect_products(
        self,
        page: Page,
        market: str,
        section: str,
        branch: str,
        category: str,
        url: str,
    ) -> List[Product]:
        self._goto(page, url)
        self._wait(page, SEL["plp_grid"], self.timeout)
        expected_total = None
        try:
            page.wait_for_selector(SEL["total_products"], timeout=3000)
            txt = page.locator(SEL["total_products"]).first.inner_text()
            m = re.search(r"(\d+)", txt or "")
            expected_total = int(m.group(1)) if m else None
        except Exception:
            expected_total = None
        self._load_all(page)
        items: List[Product] = []
        try:
            tiles = page.query_selector_all(SEL["product_tile"]) or []
        except Error:
            tiles = []
        for idx, t in enumerate(tiles, start=1):
            p = self._parse_tile(t, market, section, branch, category)
            if not p.product_url:
                continue
            p.order = idx
            p.expected_total = expected_total
            items.append(p)
            if self.max_per_cat and len(items) >= self.max_per_cat:
                break
        return items

    # ---- infinite load ----
    def _load_all(self, page: Page, max_clicks: int = 400) -> None:
        clicks = 0
        while clicks < max_clicks:
            before = len(page.query_selector_all(SEL["product_tile"]) or [])
            if self._try_more(page, before):
                clicks += 1
                continue
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            except Error:
                break
            time.sleep(random.uniform(*self.throttle))
            after = len(page.query_selector_all(SEL["product_tile"]) or [])
            if after <= before:
                break

    def _try_more(self, page: Page, before_count: int) -> bool:
        try:
            btn = page.locator(SEL["show_next_btn"]).first
            if btn and btn.is_visible():
                btn.click(timeout=1800)
                try:
                    page.wait_for_function(
                        "(s,n)=>document.querySelectorAll(s).length>n",
                        (SEL["product_tile"], before_count),
                        timeout=12000,
                    )
                except TimeoutError:
                    try:
                        page.wait_for_load_state("networkidle", timeout=3000)
                    except TimeoutError:
                        pass
                time.sleep(random.uniform(*self.throttle))
                return True
        except Exception:
            pass
        for text in (
            "Show the next",
            "Load more",
            "Show more",
            "Load More",
            "Show More",
        ):
            try:
                b = page.locator(f"button:has-text('{text}')").first
                if b and b.is_visible():
                    b.click(timeout=1200)
                    try:
                        page.wait_for_function(
                            "(s,n)=>document.querySelectorAll(s).length>n",
                            (SEL["product_tile"], before_count),
                            timeout=12000,
                        )
                    except TimeoutError:
                        try:
                            page.wait_for_load_state("networkidle", timeout=3000)
                        except TimeoutError:
                            pass
                    time.sleep(random.uniform(*self.throttle))
                    return True
            except Exception:
                continue
        return False

    # ---- tile parsing ----
    def _parse_tile(
        self, tile, market: str, section: str, branch: str, category: str
    ) -> Product:
        a = tile.query_selector(SEL["tile_link"]) or tile.query_selector("a")
        url_abs = self._abs(a.get_attribute("href") if a else None) or ""
        name_el = tile.query_selector(SEL["tile_name"]) or a
        name = (
            (name_el.inner_text().strip() if name_el else "").replace("\n", " ").strip()
        )
        price_texts = [
            (n.inner_text() or "").strip()
            for n in tile.query_selector_all(SEL["tile_price"]) or []
            if (n.inner_text() or "").strip()
        ]
        joined = " ".join(price_texts)
        prices = list(CURRENCY_RGX.finditer(joined))
        price_current = prices[0].group(0) if prices else None
        price_original = prices[-1].group(0) if len(prices) >= 2 else None
        colors_text, colors_available = None, None
        for p in tile.query_selector_all("p, span, div")[:8] or []:
            t = (p.inner_text() or "").strip()
            if "color" in t.lower():
                colors_text = t
                m = COLORS_RGX.search(t)
                if m:
                    try:
                        colors_available = int(m.group(1))
                    except ValueError:
                        pass
                break
        image_url = None
        for img in tile.query_selector_all(SEL["tile_imgs"]) or []:
            try:
                u = img.evaluate("el => el.currentSrc || el.src || ''")
            except Exception:
                u = img.get_attribute("src")
            if u:
                image_url = self._abs(u)
                break
        if not image_url:
            for img in tile.query_selector_all("img") or []:
                ss = img.get_attribute("srcset") or ""
                for part in ss.split(",") if ss else []:
                    u, _, _ = part.strip().partition(" ")
                    image_url = self._abs(u) or image_url
        return Product(
            market=market,
            section=section,
            branch=branch,
            category=category,
            title=name,
            price_current=price_current,
            price_original=price_original,
            product_url=url_abs,
            image_url=image_url,
            colors_available=colors_available,
            colors_text=colors_text,
        )

    # ---- helpers ----
    def _abs(self, href: Optional[str]) -> Optional[str]:
        if not href:
            return None
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"{BASE}{href}"
        return None

    def _market_from_url(self, url: str) -> str:
        m = LOCALE_RGX.search(url)
        return m.group(1) if m else ""

    def _goto(self, page: Page, url: str) -> None:
        try:
            page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
        except TimeoutError:
            self._log(f"[timeout] goto: {url}")
        except Error as e:
            self._log(f"[nav-error] {e}")
        time.sleep(random.uniform(*self.throttle))
        self._dismiss_cookie(page)

    def _wait(self, page: Page, selector: str, timeout_ms: int) -> None:
        try:
            page.wait_for_selector(selector, timeout=timeout_ms, state="visible")
        except TimeoutError:
            self._log(f"[timeout] wait: {selector}")

    def _dismiss_cookie(self, page: Page) -> None:
        for sel in (
            "button:has-text('Reject')",
            "button:has-text('Reject all')",
            "button:has-text('Decline')",
            "button:has-text('Only necessary')",
            "button:has-text('Tout refuser')",
            "button:has-text('Refuser')",
            "button:has-text('Weigeren')",
        ):
            try:
                btn = page.locator(sel).first
                if btn and btn.is_visible():
                    btn.click(timeout=1200)
                    time.sleep(0.2)
                    return
            except Exception:
                continue

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


# ---- CLI ----


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="G-STAR nav-driven product scraper (CSV, with --only-markets)"
    )
    p.add_argument(
        "--nav-csv",
        required=True,
        help="CSV columns: market,section,branch,category,url (market optional)",
    )
    p.add_argument(
        "--sections", default="men,women", help="men,women or 'all' (kids later)"
    )
    p.add_argument(
        "--only-markets",
        default=None,
        help="Comma-separated locales to include, e.g. en_gb,en_fr",
    )
    p.add_argument("--out", default="products.csv", help="CSV output path")
    p.add_argument("--jsonl", default=None, help="Optional JSONL path")
    p.add_argument("--xlsx", default=None, help="Optional Excel path (per leaf)")
    p.add_argument(
        "--headful", action="store_true", help="Run browser non-headless for debugging"
    )
    p.add_argument(
        "--max-per-cat", type=int, default=None, help="Limit products per leaf category"
    )
    p.add_argument(
        "--timeout", type=int, default=30000, help="Navigation timeout in ms"
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sections = (
        ["men", "women", "kids"]
        if args.sections.lower() == "all"
        else [s.strip().lower() for s in args.sections.split(",")]
    )

    df = pd.read_csv(args.nav_csv)
    # normalize columns
    cols = {c.lower(): c for c in df.columns}
    must = ["section", "category", "url"]
    if not set(must).issubset(set(cols)):
        raise SystemExit(
            "nav-csv must contain columns: market(optional), section, branch, category, url"
        )
    df = df.rename(
        columns={
            cols.get("market", "market"): "market",
            cols.get("section", "section"): "section",
            cols.get("branch", "branch"): "branch",
            cols.get("category", "category"): "category",
            cols.get("url", "url"): "url",
        }
    )

    only_markets: Optional[Set[str]] = None
    if args.only_markets:
        only_markets = {
            s.strip().lower() for s in args.only_markets.split(",") if s.strip()
        }

    rows = df.to_dict("records")

    with sync_playwright() as pw:
        scraper = Scraper(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            max_per_cat=args.max_per_cat,
            verbose=True,
        )
        products = scraper.run(rows, sections, only_markets)

    data = [asdict(p) for p in products]
    if not data:
        print("No products scraped.")
        return 1

    out_df = pd.DataFrame(data)
    out_df["title"] = (
        out_df["title"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    out_df.to_csv(args.out, index=False)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.xlsx:
        with pd.ExcelWriter(args.xlsx, engine="xlsxwriter") as writer:
            used: Set[str] = set()
            max_rows = 1_048_000

            def sheet(s: str) -> str:
                s = re.sub(r"[:\\/?*\[\]]", " ", s)
                s = re.sub(r"\s+", " ", s).strip()
                return s[:31]

            for (mkt, sec, br, cat), grp in out_df.groupby(
                ["market", "section", "branch", "category"], sort=False
            ):
                base = sheet(f"{mkt}-{sec}-{br}-{cat}")
                nm = base
                i = 2
                while nm in used:
                    suf = f"_{i}"
                    nm = sheet(base[: (31 - len(suf))] + suf)
                    i += 1
                used.add(nm)
                grp.to_excel(writer, sheet_name=nm, index=False)

    print(
        f"Saved {len(out_df)} products -> {args.out}"
        + (f" & {args.xlsx}" if args.xlsx else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
