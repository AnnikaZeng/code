#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: scrapers/gstar_scraper.py  (v9)
Purpose: Crawl G‑STAR RAW (https://www.g-star.com/en_us) PLPs for Men/Women/Kids
         and extract product tiles (title, prices, PDP link, image, colors),
         while also **recording per-leaf page totals & failures** for completeness checks.

This version (v9) focuses on fixing the issue where the script appears to
"keep browsing categories without actually scraping" by:
  1) Making PLP readiness detection resilient to selector changes.
  2) Aggressively dismissing cookie/offer overlays (including *Accept all*).
  3) Falling back to anchors when tile containers aren't found.
  4) Waiting for `networkidle` and first PDP link instead of a brittle grid testid.
  5) Adding robust logging & per-leaf new tab isolation (optional) to avoid state carryover.

Usage
-----
    pip install playwright pandas xlsxwriter openpyxl
    playwright install

    python gstar_scraper_v9.py --sections men --max-per-cat 50 --headful --out men.csv --xlsx men.xlsx --fails men_fails.csv

    # 限时（每叶 120s），并记录是否触发 deadline
    python gstar_scraper_v9.py --sections men --time-budget 120 --out men.csv --fails men_fails.csv

Diagnostics
    python scrapers/gstar_scraper.py --diagnose-url https://www.g-star.com/en_us/shop/men/jeans --headful
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
    BrowserContext,
    Error,
    Page,
    Playwright,
    TimeoutError,
    sync_playwright,
)

BASE = "https://www.g-star.com"
LOCALE = "/en_us"
HOME = f"{BASE}{LOCALE}"

# --- Selectors: broadened for resilience ---
SELECTORS: Dict[str, str] = {
    # Sidebar container on /shop/{section}
    "sidebar": '#sideNav, nav[aria-label="Sidebar"], aside[aria-label="Sidebar"], nav[aria-label*="Categories" i]',
    # Any container hinting PLP readiness
    "plp_ready": (
        'section[data-testid="plp-grid"], '
        '[data-testid*="plp" i], '
        '[data-testid*="productlist" i], '
        "main [data-grid], "
        'ul[role="list"]:has(a[href*="/product/"])'
    ),
    # Tiles and bits
    "product_tile": (
        'div[data-testid*="product-tile" i], '
        'li[data-testid*="product" i], '
        'article:has(a[href*="/product/"])'
    ),
    "tile_link": 'a[data-testid="product-tile-link"], a[href*="/en_us/product/"], a[href*="/product/"]',
    "tile_name": '[data-testid="product-title-title"], [data-testid="product-title"], h3, .product-title',
    "tile_price": '[data-testid*="price" i], [class*="price" i] , [data-test*="price" i]',
    "tile_imgs": "picture img, img",
    # Load-more variants
    "show_next_btn": '[data-testid="productList-showNext"]',
    # Total products text
    "total_products": '[data-testid="total-number-products"], [data-testid*="total" i]',
}

# Fallback URL fragments to locate links if container parsing fails
FALLBACK_HREFS: Dict[str, Dict[str, List[str]]] = {
    "men": {
        "Jeans & Bottoms": ["/shop/men/jeans", "/shop/men/pants", "/shop/men/shorts"],
        "Tops & Hoodies": [
            "/shop/men/t-shirts",
            "/shop/men/shirts",
            "/shop/men/sweatshirts-hoodies",
            "/shop/men/knitwear",
        ],
        "Jackets & Coats": [
            "/shop/men/jackets_and_blazers",
            "/shop/men/lightweight_jackets",
            "/shop/men/overshirts",
            "/shop/men/denim_jackets",
        ],
        "Shoes & Accessories": ["/shop/men/shoes", "/shop/men/accessories"],
    },
    "women": {
        "Jeans & Bottoms": [
            "/shop/women/jeans",
            "/shop/women/pants",
            "/shop/women/shorts",
        ],
        "Tops & Hoodies": [
            "/shop/women/t-shirts",
            "/shop/women/shirts",
            "/shop/women/sweatshirts-hoodies",
            "/shop/women/knitwear",
        ],
        "Jackets & Coats": [
            "/shop/women/jackets_and_blazers",
            "/shop/women/lightweight_jackets",
            "/shop/women/overshirts",
            "/shop/women/denim_jackets",
        ],
        "Dresses & Jumpsuits": ["/shop/women/dresses", "/shop/women/jumpsuits"],
        "Shoes & Accessories": ["/shop/women/shoes", "/shop/women/accessories"],
    },
    # kids 页面中 Boys/Girls 链接是 /en_us/shop/boys, /en_us/shop/girls
    "kids": {
        "Boys": ["/shop/boys"],
        "Girls": ["/shop/girls"],
    },
}

CURRENCY_RGX = re.compile(
    r"([€£$])\s?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)"
)
COLORS_RGX = re.compile(r"(\d+)\s+colors?\s+available", re.IGNORECASE)


@dataclass
class Product:
    section: str
    section_index: int
    category: str
    category_index: int
    order: int
    title: str
    price_current: Optional[str]
    price_original: Optional[str]
    product_url: str
    image_url: Optional[str]
    colors_available: Optional[int]
    colors_text: Optional[str]
    on_leaf_url: Optional[str] = None

    def key(self) -> str:
        return self.product_url.split("?")[0].split("#")[0]


@dataclass
class LeafFail:
    section: str
    category: str
    leaf_url: str
    expected_total: Optional[int]
    scraped: int
    max_per_cat: Optional[int]
    time_budget: Optional[int]
    deadline_hit: bool
    reason: str


class GStarScraper:
    def __init__(
        self,
        pw: Playwright,
        headless: bool = True,
        throttle: Tuple[float, float] = (0.5, 1.5),
        timeout_ms: int = 45000,
        max_per_cat: Optional[int] = None,
        time_budget_sec: Optional[int] = None,
        verbose: bool = True,
        isolate_leaf_tabs: bool = False,  # 打开每个叶子在独立 tab 里（更稳定）
    ) -> None:
        self.pw = pw
        self.browser: Browser = pw.chromium.launch(headless=headless)
        # Use a context with locale & UA to reduce geo/cookie variations
        self.context: BrowserContext = self.browser.new_context(
            locale="en-US",
            timezone_id="America/Los_Angeles",
            viewport={"width": 1440, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        self.throttle = throttle
        self.timeout = timeout_ms
        self.max_per_cat = max_per_cat
        self.time_budget_sec = time_budget_sec
        self.verbose = verbose
        self.isolate_leaf_tabs = isolate_leaf_tabs

    # ---------------- Public API ---------------- #
    def run(self, sections: Iterable[str]) -> Tuple[List[Product], List[LeafFail]]:
        page = self.context.new_page()
        out: List[Product] = []
        fails: List[LeafFail] = []
        dedupe: Set[str] = set()
        section_order = {sec: i for i, sec in enumerate(sections)}

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

            for cat_idx, (cat_name, cat_url) in enumerate(links):
                self._log(f"  ↳ {sec} · {cat_name or '[no-text]'} -> {cat_url}")
                start = time.time()

                # Optional isolation: open each leaf in a fresh tab to avoid overlays/state
                if self.isolate_leaf_tabs:
                    leaf_page = self.context.new_page()
                else:
                    leaf_page = page

                products, fail = self._collect_products(
                    leaf_page,
                    sec,
                    section_order[sec],
                    cat_name,
                    cat_idx,
                    cat_url,
                    start,
                )

                for p in products:
                    if p.key() in dedupe:
                        continue
                    out.append(p)
                    dedupe.add(p.key())

                if fail:
                    fails.append(fail)

                if self.isolate_leaf_tabs and leaf_page is not page:
                    try:
                        leaf_page.close()
                    except Exception:
                        pass

                if self.max_per_cat:
                    self._log(f"    [limit] max-per-cat={self.max_per_cat} applied")

        page.close()
        self.browser.close()
        return out, fails

    # ---------------- Sidebar ---------------- #
    def _get_leaf_links(self, page: Page, section: str) -> List[Tuple[str, str]]:
        self._wait_visible(page, SELECTORS["sidebar"], self.timeout)
        sidebar = page.locator(SELECTORS["sidebar"]).first
        if not sidebar or (hasattr(sidebar, "is_visible") and not sidebar.is_visible()):
            return []

        self._expand_all_toggles(sidebar)

        results: List[Tuple[str, str]] = []
        seen: Set[str] = set()
        for label, patterns in FALLBACK_HREFS[section].items():
            # 1) header → following container
            hdr = sidebar.locator(
                "xpath=(.//*[self::button or self::a or @role='button' or self::div][normalize-space()='%s'])[1]"
                % label
            )
            anchors: List = []
            try:
                if hdr and hdr.count():
                    container = hdr.locator("xpath=following-sibling::*[1]")
                    if container and container.count():
                        self._expand_all_toggles(container)
                        anchors = container.locator("xpath=.//a").all()
            except Exception:
                anchors = []

            # 2) fallback by href fragments
            if not anchors:
                xp_or = " or ".join([f"contains(@href, '{p}')" for p in patterns])
                # kids 不限制 /shop/{section}/
                if section == "kids":
                    anchors = sidebar.locator(
                        f"xpath=.//a[starts-with(@href,'{LOCALE}/shop/') and ({xp_or})]"
                    ).all()
                else:
                    anchors = sidebar.locator(
                        f"xpath=.//a[contains(@href,'/shop/{section}') and ({xp_or})]"
                    ).all()

            for a in anchors:
                try:
                    text = (a.inner_text() or "").strip()
                    href = a.get_attribute("href") or ""
                except Exception:
                    continue
                if not href or href.startswith("javascript") or href == "#":
                    continue
                if "shop all" in (text or "").lower():
                    continue
                # Boys/Girls 在 kids 下允许 /shop/boys|/shop/girls
                if section != "kids" and "/shop/" not in href:
                    continue
                url = self._abs_url(href)
                if not url:
                    continue
                key = url.split("?")[0]
                if key in seen:
                    continue
                results.append((text, url))
                seen.add(key)

        return results

    def _expand_all_toggles(self, container) -> None:
        for _ in range(10):
            try:
                loc = container.locator(
                    "button[aria-expanded='false'], [role='button'][aria-expanded='false']"
                )
                n = loc.count()
            except Exception:
                n = 0
            if not n:
                break
            for i in range(n):
                try:
                    el = loc.nth(i)
                    if el.is_visible():
                        el.click(timeout=1000)
                        time.sleep(0.2)
                except Exception:
                    continue

    # ---------------- PLP collection ---------------- #
    def _collect_products(
        self,
        page: Page,
        section: str,
        section_index: int,
        category: str,
        category_index: int,
        url: str,
        start_time: float,
    ) -> Tuple[List[Product], Optional[LeafFail]]:
        self._safe_goto(page, url)
        self._wait_plp_ready(page)
        expected_total = self._read_total(page)
        load_info = self._load_all_results(page, start_time)

        out: List[Product] = []
        tiles = self._find_tiles(page)
        for idx, t in enumerate(tiles, start=1):
            parsed = self._parse_tile(t)
            if not parsed.product_url:
                continue
            out.append(
                Product(
                    section=section,
                    section_index=section_index,
                    category=category,
                    category_index=category_index,
                    order=idx,
                    title=parsed.title,
                    price_current=parsed.price_current,
                    price_original=parsed.price_original,
                    product_url=parsed.product_url,
                    image_url=parsed.image_url,
                    colors_available=parsed.colors_available,
                    colors_text=parsed.colors_text,
                    on_leaf_url=url,
                )
            )
            if self.max_per_cat and len(out) >= self.max_per_cat:
                break

        self._log(f"    [scraped] {len(out)} products from {url}")

        fail = None
        scraped = len(out)
        reason = ""
        if scraped == 0:
            reason = "empty-or-blocked"
        elif expected_total is not None:
            cap = self.max_per_cat or expected_total
            if scraped < min(expected_total, cap):
                reason = "partial"
        if reason or load_info.get("deadline", False):
            fail = LeafFail(
                section=section,
                category=category,
                leaf_url=url,
                expected_total=expected_total,
                scraped=scraped,
                max_per_cat=self.max_per_cat,
                time_budget=self.time_budget_sec,
                deadline_hit=bool(load_info.get("deadline", False)),
                reason=reason
                or ("deadline" if load_info.get("deadline", False) else "ok"),
            )
        return out, fail

    def _wait_plp_ready(self, page: Page) -> None:
        """Robust PLP readiness: cookie dialog, networkidle, and first PDP link."""
        # Sometimes the cookie dialog blocks visibility; click it before waiting.
        self._dismiss_overlays(page)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=self.timeout)
        except TimeoutError:
            self._log("[timeout] domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=self.timeout)
        except TimeoutError:
            self._log("[timeout] networkidle")
        # Either the grid or at least one PDP link
        try:
            page.wait_for_selector(
                f"{SELECTORS['plp_ready']}, {SELECTORS['tile_link']}",
                state="attached",
                timeout=self.timeout,
            )
        except TimeoutError:
            self._log("[timeout] plp-ready selectors")
        self._dismiss_overlays(page)

    def _find_tiles(self, page: Page):
        tiles = page.query_selector_all(SELECTORS["product_tile"]) or []
        if tiles:
            return tiles
        # Fallback: unique PDP anchors as tiles
        anchors = page.query_selector_all(SELECTORS["tile_link"]) or []
        uniq: List = []
        seen: Set[str] = set()
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
            except Exception:
                continue
            if "/product/" not in href:
                continue
            key = href.split("?")[0]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(a)
        return uniq

    def _load_all_results(
        self, page: Page, start_time: float, max_clicks: int = 500
    ) -> Dict[str, object]:
        clicks = 0
        deadline = (
            start_time + float(self.time_budget_sec) if self.time_budget_sec else None
        )
        hit_deadline = False
        while clicks < max_clicks:
            if deadline and time.time() >= deadline:
                hit_deadline = True
                self._log("    [time-budget] stop loading more")
                break
            before = len(self._find_tiles(page))
            clicked = self._try_click_show_next(page, before)
            if not clicked:
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                except Exception:
                    pass
                time.sleep(random.uniform(*self.throttle))
            after = len(self._find_tiles(page))
            if not clicked and after <= before:
                break
            clicks += 1
        return {"clicks": clicks, "deadline": hit_deadline}

    def _try_click_show_next(self, page: Page, before_count: int) -> bool:
        def wait_growth():
            try:
                page.wait_for_function(
                    "(s,n)=>document.querySelectorAll(s).length>n",
                    (SELECTORS["tile_link"], before_count),
                    timeout=12000,
                )
            except TimeoutError:
                try:
                    page.wait_for_load_state("networkidle", timeout=3000)
                except TimeoutError:
                    pass
            time.sleep(random.uniform(*self.throttle))

        # Primary testid
        try:
            btn = page.locator(SELECTORS["show_next_btn"]).first
            if btn and btn.is_visible():
                btn.scroll_into_view_if_needed(timeout=1500)
                btn.click(timeout=2500)
                wait_growth()
                return True
        except Exception:
            pass
        # Label variants
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
                    b.scroll_into_view_if_needed(timeout=1500)
                    b.click(timeout=2500)
                    wait_growth()
                    return True
            except Exception:
                continue
        return False

    # ---------------- Tile parsing ---------------- #
    @dataclass
    class _ParsedTile:
        title: str
        price_current: Optional[str]
        price_original: Optional[str]
        product_url: str
        image_url: Optional[str]
        colors_available: Optional[int]
        colors_text: Optional[str]

    def _parse_tile(self, tile) -> "GStarScraper._ParsedTile":
        # Accept both a container or the anchor itself
        a = None
        try:
            tag = tile.evaluate("el => el.tagName")
            if isinstance(tag, str) and tag.upper() == "A":
                a = tile
            else:
                a = tile.query_selector(SELECTORS["tile_link"]) or tile.query_selector(
                    "a"
                )
        except Exception:
            a = tile.query_selector(SELECTORS["tile_link"]) or tile.query_selector("a")

        url_abs = self._abs_url(a.get_attribute("href") if a else None) or ""

        name_el = None
        try:
            name_el = tile.query_selector(SELECTORS["tile_name"]) or a
        except Exception:
            name_el = a
        title = (
            (name_el.inner_text().strip() if name_el else "").replace("\n", " ").strip()
        )

        price_texts = []
        for n in tile.query_selector_all(SELECTORS["tile_price"]) or []:
            try:
                txt = (n.inner_text() or "").strip()
            except Exception:
                txt = ""
            if txt:
                price_texts.append(txt)
        joined = " ".join(price_texts)
        prices = list(CURRENCY_RGX.finditer(joined))
        price_current = prices[0].group(0) if prices else None
        price_original = prices[-1].group(0) if len(prices) >= 2 else None

        colors_text = None
        try:
            ct = tile.locator(
                "xpath=.//p[contains(translate(normalize-space(.),'COLORS','colors'),'colors available')]"
            ).first
            if ct and ct.count():
                colors_text = (ct.inner_text() or "").strip()
        except Exception:
            pass
        if not colors_text:
            for p in tile.query_selector_all("p, span, div")[:8] or []:
                try:
                    t = (p.inner_text() or "").strip()
                except Exception:
                    t = ""
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
                u = img.evaluate("el=>el.currentSrc||el.src||''")
            except Exception:
                u = img.get_attribute("src")
            if u:
                image_url = self._abs_url(u)
                break
        if not image_url:
            for img in tile.query_selector_all("img") or []:
                ss = img.get_attribute("srcset") or ""
                cand = self._best_from_srcset(ss)
                if cand:
                    image_url = self._abs_url(cand)
                    break

        return GStarScraper._ParsedTile(
            title=title,
            price_current=price_current,
            price_original=price_original,
            product_url=url_abs,
            image_url=image_url,
            colors_available=colors_available,
            colors_text=colors_text,
        )

    # ---------------- Utils ---------------- #
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
        # Give the runtime a breath & clear overlays
        time.sleep(random.uniform(*self.throttle))
        self._dismiss_overlays(page)

    def _wait_visible(self, page: Page, selector: str, timeout_ms: int) -> None:
        try:
            page.wait_for_selector(selector, timeout=timeout_ms, state="visible")
        except TimeoutError:
            self._log(f"[timeout] wait: {selector}")

    def _dismiss_overlays(self, page: Page) -> None:
        # Broadened: include OneTrust and generic accept/agree variants
        selectors = [
            # OneTrust cookie
            "#onetrust-accept-btn-handler",
            "#onetrust-reject-all-handler",
            # Generic cookie buttons
            "button:has-text('Accept all')",
            "button:has-text('Accept All')",
            "button:has-text('Allow all')",
            "button:has-text('Accept')",
            "button:has-text('Agree')",
            "button:has-text('I agree')",
            "button:has-text('Got it')",
            "button:has-text('Reject')",
            "button:has-text('Reject all')",
            "button:has-text('Decline')",
            "button:has-text('Only necessary')",
            # promo/newsletter dialogs
            "[role='dialog'] button[aria-label*='close' i]",
            "[role='dialog'] button:has-text('No thanks')",
            "[role='dialog'] button:has-text('Not now')",
            "button:has-text('No thanks')",
            "button:has-text('Not now')",
            "button.close, .modal [data-testid*='close' i]",
        ]
        for sel in selectors:
            try:
                el = page.locator(sel).first
                if el and el.is_visible():
                    el.click(timeout=1200)
                    time.sleep(0.2)
            except Exception:
                continue

    def _read_total(self, page: Page) -> Optional[int]:
        try:
            el = page.locator(SELECTORS["total_products"]).first
            if el and el.count():
                txt = (el.inner_text() or "").strip()
                m = re.search(r"\d+", txt)
                if m:
                    return int(m.group(0))
        except Exception:
            pass
        return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


# ---------------- CLI ---------------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G-STAR RAW category scraper (Playwright)")
    p.add_argument("--sections", default="all", help="men,women,kids or 'all'")
    p.add_argument("--out", default="products.csv", help="CSV output path")
    p.add_argument(
        "--xlsx",
        default=None,
        help="Optional Excel .xlsx output path (per-leaf sheets)",
    )
    p.add_argument(
        "--fails",
        default=None,
        help="Optional CSV to save per-leaf failure/partial records",
    )
    p.add_argument(
        "--headful", action="store_true", help="Run browser non-headless for debugging"
    )
    p.add_argument(
        "--max-per-cat", type=int, default=None, help="Limit products per leaf category"
    )
    p.add_argument(
        "--time-budget",
        type=int,
        default=None,
        help="Seconds budget per leaf (stop loading when reached)",
    )
    p.add_argument(
        "--timeout", type=int, default=45000, help="Navigation timeout in ms"
    )
    p.add_argument("--quiet", action="store_true", help="Reduce console logs")
    p.add_argument(
        "--diagnose-url",
        default=None,
        help="PLP URL to diagnose selectors (prints first tile)",
    )
    p.add_argument(
        "--isolate-leaf-tabs",
        action="store_true",
        help="Open each leaf in a fresh tab to avoid overlay/state carryover",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    with sync_playwright() as pw:
        sections = (
            ["men", "women", "kids"]
            if args.sections.lower() == "all"
            else [s.strip() for s in args.sections.split(",")]
        )
        scraper = GStarScraper(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            max_per_cat=args.max_per_cat,
            time_budget_sec=args.time_budget,
            verbose=not args.quiet,
            isolate_leaf_tabs=args.isolate_leaf_tabs,
        )

        if args.diagnose_url:
            page = scraper.context.new_page()
            scraper._safe_goto(page, args.diagnose_url)
            scraper._wait_plp_ready(page)
            tiles = scraper._find_tiles(page)
            print(f"Tiles matched: {len(tiles)}")
            if tiles:
                sample = scraper._parse_tile(tiles[0])
                print(json.dumps(asdict(sample), ensure_ascii=False, indent=2))
            page.close()
            scraper.browser.close()
            return 0

        products, fails = scraper.run(sections)

    # ---------- Output ---------- #
    rows = [asdict(p) for p in products]
    if not rows:
        print("No products scraped.")
        return 1

    df = pd.DataFrame(rows)
    df.sort_values(
        ["section_index", "category_index", "order"], inplace=True, ignore_index=True
    )
    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df.to_csv(args.out, index=False)

    # optional fails
    if args.fails:
        df_f = pd.DataFrame([asdict(f) for f in fails])
        df_f.to_csv(args.fails, index=False)
        print(f"Saved fails log -> {args.fails}")

    # optional Excel
    if args.xlsx:
        engine = None
        try:
            import xlsxwriter  # noqa: F401

            engine = "xlsxwriter"
        except Exception:
            try:
                import openpyxl  # noqa: F401

                engine = "openpyxl"
            except Exception:
                engine = None
        if engine is None:
            print("[warn] neither xlsxwriter nor openpyxl installed; skip XLSX output")
        else:
            with pd.ExcelWriter(args.xlsx, engine=engine) as writer:
                max_rows = 1_048_000
                used: Set[str] = set()

                def sanitize(name: str) -> str:
                    name = re.sub(r"[:\\/?*\[\]]", " ", name)
                    name = re.sub(r"\s+", " ", name).strip()
                    return name[:31]

                for (sec, cat), grp in df.groupby(["section", "category"], sort=False):
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
        + (f" & {args.xlsx}" if args.xlsx else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
