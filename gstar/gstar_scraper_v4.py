#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-STAR RAW scraper (Playwright)
- Crawls Shop Men/Women/Kids
- Only required top-level branches (Men: Jeans & Bottoms, Tops & Hoodies, Jackets & Coats, Shoes & Accessories;
  Women: + Dresses & Jumpsuits; Kids: Boys, Girls), expands to **leaf** categories
- Product extraction per tile: title (name only), price_current/original, PDP link, image_url,
  colors_available/colors_text
- Infinite list handled via clicking "Show the next 36 results" until exhausted
- **Order preserved**: output sorted by (section, category, order) where `order` is the on-page index
- CSV default; optional Excel with one sheet per leaf category

Usage
-----
    pip install playwright pandas xlsxwriter
    playwright install

    # quick sample
    python gstar_scraper_v4.py --sections men --max-per-cat 50 --headful --out men.csv

    # full with Excel (one sheet per leaf)
    python gstar_scraper_v4.py --sections all --out products.csv --xlsx products.xlsx
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
LOCALE = "/en_us"
HOME = f"{BASE}{LOCALE}"

SELECTORS: Dict[str, str] = {
    "sidebar": '#sideNav, nav[aria-label="Sidebar"], aside[aria-label="Sidebar"]',
    "plp_grid": 'section[data-testid="plp-grid"]',
    "product_tile": 'div[data-testid="product-tile"]',
    "tile_link": 'a[data-testid="product-tile-link"], a[href*="/en_us/product/"]',
    "tile_name": '[data-testid="product-title-title"]',
    "tile_price": '[data-testid="product-tile-price"], [data-testid*="price" i], [class*="price" i]',
    "tile_imgs": "picture img, img",
    "show_next_btn": '[data-testid="productList-showNext"]',
}

ALLOWED_BRANCHES: Dict[str, List[str]] = {
    "men": [
        "Jeans & Bottoms",
        "Tops & Hoodies",
        "Jackets & Coats",
        "Shoes & Accessories",
    ],
    "women": [
        "Jeans & Bottoms",
        "Tops & Hoodies",
        "Jackets & Coats",
        "Dresses & Jumpsuits",
        "Shoes & Accessories",
    ],
    "kids": [
        "Boys",
        "Girls",
    ],
}

CURRENCY_RGX = re.compile(
    r"([€£$])\s?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)"
)
COLORS_RGX = re.compile(r"(\d+)\s+colors?\s+available", re.IGNORECASE)


@dataclass
class Product:
    section: str
    category: str
    title: str
    price_current: Optional[str]
    price_original: Optional[str]
    product_url: str
    image_url: Optional[str]
    colors_available: Optional[int]
    colors_text: Optional[str]
    order: int = 0  # on-page index, preserves visual order

    def key(self) -> str:
        return self.product_url.split("?")[0].split("#")[0]


class GStarScraper:
    def __init__(
        self,
        pw: Playwright,
        headless: bool = True,
        throttle: Tuple[float, float] = (0.5, 1.5),
        timeout_ms: int = 30000,
        max_per_cat: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        self.pw = pw
        self.browser: Browser = pw.chromium.launch(headless=headless)
        self.throttle = throttle
        self.timeout = timeout_ms
        self.max_per_cat = max_per_cat
        self.verbose = verbose

    # ---------------- Public API ---------------- #
    def run(self, sections: Iterable[str]) -> List[Product]:
        page = self.browser.new_page()
        out: List[Product] = []
        dedupe: Set[str] = set()

        for sec in sections:
            sec = sec.lower().strip()
            if sec not in {"men", "women", "kids"}:
                self._log(f"Skip unknown section: {sec}")
                continue

            self._safe_goto(page, f"{HOME}/shop/{sec}")
            links = self._get_leaf_links(page, sec)
            if not links:
                self._log(f"No subcategories found for {sec}.")
                continue

            for cat_name, cat_url in links:
                self._log(f"  ↳ {sec} · {cat_name} -> {cat_url}")
                for p in self._collect_products(page, sec, cat_name, cat_url):
                    if p.key() not in dedupe:
                        out.append(p)
                        dedupe.add(p.key())
                if self.max_per_cat:
                    self._log(f"    [limit] max-per-cat={self.max_per_cat} applied")

        page.close()
        self.browser.close()
        return out

    # ---------------- Sidebar navigation ---------------- #
    def _get_leaf_links(self, page: Page, section: str) -> List[Tuple[str, str]]:
        self._wait_visible(page, SELECTORS["sidebar"], self.timeout)
        sidebar = page.locator(SELECTORS["sidebar"]).first
        if not sidebar or not sidebar.is_visible():
            return []

        results: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        wanted = ALLOWED_BRANCHES.get(section, [])

        for label in wanted:
            # Find branch header (button or anchor) by exact text
            hdr = sidebar.locator(
                f"xpath=//*[self::button or self::a][normalize-space()='{label}']"
            ).first
            if not hdr or not hdr.count():
                self._log(f"[warn] branch not found: {label}")
                continue

            # Expand if collapsible
            try:
                expanded = hdr.get_attribute("aria-expanded")
                if expanded == "false" or expanded is None:
                    hdr.click(timeout=2000)
                    time.sleep(0.3)
            except Exception:
                pass

            # The branch container is the immediate next sibling
            container = sidebar.locator(
                f"xpath=(//*[self::button or self::a][normalize-space()='{label}']/following-sibling::*[1])"
            ).first
            if container and container.count():
                # Expand inner toggles, then collect anchors within this container only
                self._expand_all_toggles(container)
                anchors = container.locator(
                    f"xpath=.//a[contains(@href, '/shop/{section}/')]"
                ).all()
            else:
                anchors = []  # if no container, skip to be safe

            for a in anchors:
                try:
                    text = (a.inner_text() or "").strip()
                    href = a.get_attribute("href") or ""
                except Exception:
                    continue
                if not href:
                    continue
                if "shop all" in text.lower():
                    continue
                if f"/shop/{section}" not in href:
                    continue
                absu = self._abs_url(href)
                key = absu.split("?")[0]
                if key in seen:
                    continue
                results.append((text, absu))
                seen.add(key)

        return results

    def _expand_all_toggles(self, container) -> None:
        rounds = 0
        while rounds < 10:
            toggles = container.locator(
                "button[aria-expanded='false'], [role='button'][aria-expanded='false']"
            ).all()
            if not toggles:
                break
            for t in toggles:
                try:
                    if t.is_visible():
                        t.click(timeout=1000)
                        time.sleep(0.2)
                except Exception:
                    continue
            rounds += 1

    # ---------------- Collection ---------------- #
    def _collect_products(
        self, page: Page, section: str, category: str, url: str
    ) -> List[Product]:
        self._safe_goto(page, url)
        self._wait_visible(page, SELECTORS["plp_grid"], self.timeout)
        self._load_all_results(page)

        products: List[Product] = []
        tiles = page.query_selector_all(SELECTORS["product_tile"]) or []
        for idx, t in enumerate(tiles, start=1):
            p = self._parse_tile(t, section, category)
            if not p.product_url:
                continue
            p.order = idx
            products.append(p)
            if self.max_per_cat and len(products) >= self.max_per_cat:
                break
        return products

    def _load_all_results(self, page: Page, max_clicks: int = 400) -> None:
        clicks = 0
        while clicks < max_clicks:
            before = len(page.query_selector_all(SELECTORS["product_tile"]) or [])
            clicked = self._try_click_show_next(page, before)
            if not clicked:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(random.uniform(*self.throttle))
            after = len(page.query_selector_all(SELECTORS["product_tile"]) or [])
            if not clicked and after <= before:
                break
            clicks += 1

    def _try_click_show_next(self, page: Page, before_count: int = 0) -> bool:
        try:
            btn = page.locator(SELECTORS["show_next_btn"]).first
            if btn and btn.is_visible():
                btn.click(timeout=2000)
                # wait for either more tiles or network idle (handles slow network)
                try:
                    page.wait_for_function(
                        "(sel, n) => document.querySelectorAll(sel).length > n",
                        (SELECTORS["product_tile"], before_count),
                        timeout=10000,
                    )
                except TimeoutError:
                    try:
                        page.wait_for_load_state("networkidle", timeout=2500)
                    except TimeoutError:
                        pass
                time.sleep(random.uniform(*self.throttle))
                return True
        except Exception:
            pass
        for label in [
            "Show the next",
            "Load more",
            "Show more",
            "Load More",
            "Show More",
        ]:
            try:
                b = page.locator(f"button:has-text('{label}')").first
                if b and b.is_visible():
                    b.click(timeout=1500)
                    try:
                        page.wait_for_function(
                            "(sel, n) => document.querySelectorAll(sel).length > n",
                            (SELECTORS["product_tile"], before_count),
                            timeout=10000,
                        )
                    except TimeoutError:
                        try:
                            page.wait_for_load_state("networkidle", timeout=2500)
                        except TimeoutError:
                            pass
                    time.sleep(random.uniform(*self.throttle))
                    return True
            except Exception:
                continue
        return False

    # ---------------- Tile parsing ---------------- #
    def _parse_tile(self, tile, section: str, category: str) -> Product:
        a = tile.query_selector(SELECTORS["tile_link"]) or tile.query_selector("a")
        url_abs = self._abs_url(a.get_attribute("href") if a else None) or ""

        name_el = tile.query_selector(SELECTORS["tile_name"]) or a
        name = name_el.inner_text().strip() if name_el else ""

        price_texts = []
        for n in tile.query_selector_all(SELECTORS["tile_price"]) or []:
            txt = (n.inner_text() or "").strip()
            if txt:
                price_texts.append(txt)
        joined = " ".join(price_texts)
        prices = list(CURRENCY_RGX.finditer(joined))
        price_current = prices[0].group(0) if prices else None
        price_original = prices[-1].group(0) if len(prices) >= 2 else None

        colors_text = None
        for p in tile.query_selector_all("p, span, div")[:8] or []:
            t = (p.inner_text() or "").strip()
            if "color" in t.lower():
                colors_text = t
                break
        colors_available = None
        if colors_text:
            m = COLORS_RGX.search(colors_text)
            if m:
                try:
                    colors_available = int(m.group(1))
                except ValueError:
                    colors_available = None

        image_url = None
        for img in tile.query_selector_all(SELECTORS["tile_imgs"]) or []:
            try:
                url = img.evaluate("el => el.currentSrc || el.src || ''")
            except Exception:
                url = img.get_attribute("src")
            if url:
                image_url = self._abs_url(url)
                break
        if not image_url:
            for img in tile.query_selector_all("img") or []:
                ss = img.get_attribute("srcset") or ""
                cand = self._best_from_srcset(ss)
                if cand:
                    image_url = self._abs_url(cand)
                    break

        return Product(
            section=section,
            category=category,
            title=name,
            price_current=price_current,
            price_original=price_original,
            product_url=url_abs,
            image_url=image_url,
            colors_available=colors_available,
            colors_text=colors_text,
        )

    # ---------------- Utilities ---------------- #
    def _best_from_srcset(self, srcset: str) -> Optional[str]:
        best_w, best_url = 0, None
        for part in (srcset or "").split(","):
            part = part.strip()
            if not part:
                continue
            url, _, w = part.partition(" ")
            try:
                width = int((w or "0").replace("w", "").strip())
            except ValueError:
                width = 0
            if width >= best_w:
                best_w, best_url = width, url
        return best_url

    def _abs_url(self, href: Optional[str]) -> Optional[str]:
        if not href:
            return None
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"{BASE}{href}"
        return None

    def _safe_goto(self, page: Page, url: str) -> None:
        try:
            page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
        except TimeoutError:
            self._log(f"[timeout] goto: {url}")
        except Error as e:
            self._log(f"[nav-error] {e}")
        time.sleep(random.uniform(*self.throttle))
        self._dismiss_cookie_banner(page)

    def _wait_visible(self, page: Page, selector: str, timeout_ms: int) -> None:
        try:
            page.wait_for_selector(selector, timeout=timeout_ms, state="visible")
        except TimeoutError:
            self._log(f"[timeout] wait: {selector}")

    def _dismiss_cookie_banner(self, page: Page) -> None:
        for sel in [
            "button:has-text('Reject')",
            "button:has-text('Reject all')",
            "button:has-text('Decline')",
            "button:has-text('Only necessary')",
        ]:
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


# ---------------- CLI ---------------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G-STAR RAW category scraper (Playwright)")
    p.add_argument("--sections", default="all", help="men,women,kids or 'all'")
    p.add_argument("--out", default="products.csv", help="CSV output path")
    p.add_argument("--jsonl", default=None, help="Optional JSONL output path")
    p.add_argument(
        "--xlsx",
        default=None,
        help="Optional Excel .xlsx output path (per-leaf sheets)",
    )
    p.add_argument(
        "--headful", action="store_true", help="Run browser non-headless for debugging"
    )
    p.add_argument(
        "--max-per-cat", type=int, default=None, help="Limit products per leaf category"
    )
    p.add_argument(
        "--timeout", type=int, default=30000, help="Navigation timeout in ms"
    )
    p.add_argument("--quiet", action="store_true", help="Reduce console logs")
    p.add_argument(
        "--diagnose-url",
        default=None,
        help="PLP URL to diagnose selectors (prints samples and exits)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    with sync_playwright() as pw:
        scraper = GStarScraper(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            max_per_cat=args.max_per_cat,
            verbose=not args.quiet,
        )

        if args.diagnose_url:
            page = scraper.browser.new_page()
            scraper._safe_goto(page, args.diagnose_url)
            scraper._wait_visible(page, SELECTORS["plp_grid"], scraper.timeout)
            tiles = page.query_selector_all(SELECTORS["product_tile"]) or []
            print(f"Tiles matched: {len(tiles)}")
            if tiles:
                p = scraper._parse_tile(tiles[0], "diagnose", "diagnose")
                print(json.dumps(asdict(p), ensure_ascii=False, indent=2))
            page.close()
            scraper.browser.close()
            return 0

        sections = (
            ["men", "women", "kids"]
            if args.sections.lower() == "all"
            else [s.strip() for s in args.sections.split(",")]
        )
        products = scraper.run(sections)

    rows = [asdict(p) for p in products]
    if not rows:
        print("No products scraped.")
        return 1

    df = pd.DataFrame(rows)

    # Preserve order-of-appearance across (section, category)
    if "order" in df.columns:
        df.sort_values(
            ["section", "category", "order"], inplace=True, ignore_index=True
        )

    # Title to single line
    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df.to_csv(args.out, index=False)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if args.xlsx:
        with pd.ExcelWriter(args.xlsx, engine="xlsxwriter") as writer:
            max_rows = 1_048_000
            used: Set[str] = set()

            def sanitize(name: str) -> str:
                name = re.sub(r"[:\\/?*\[\]]", " ", name)
                name = re.sub(r"\s+", " ", name).strip()
                return name[:31]

            for (sec, cat), grp in df.groupby(["section", "category"], sort=True):
                base = sanitize(f"{sec}-{cat}") or "sheet"
                sheet = base
                i = 2
                while sheet in used:
                    suf = f"_{i}"
                    sheet = sanitize(base[: (31 - len(suf))] + suf)
                    i += 1
                used.add(sheet)

                if len(grp) <= max_rows:
                    grp.to_excel(writer, sheet_name=sheet, index=False)
                else:
                    parts = (len(grp) // max_rows) + 1
                    for p in range(parts):
                        chunk = grp.iloc[p * max_rows : (p + 1) * max_rows]
                        sfx = f"_part{p+1}"
                        sheet_p = sanitize(sheet[: (31 - len(sfx))] + sfx)
                        chunk.to_excel(writer, sheet_name=sheet_p, index=False)

    print(
        f"Saved {len(df)} products -> {args.out}"
        + (f" & {args.jsonl}" if args.jsonl else "")
        + (f" & {args.xlsx}" if args.xlsx else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
