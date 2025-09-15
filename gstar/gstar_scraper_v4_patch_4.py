#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-STAR RAW scraper (Playwright) – v4 patch (4th)
- Keep crawl order
- Exact sidebar mapping: branch=顶级分组, category=叶子锚点（与网站一致）
- Kids: /boys /girls
- Fix missing leaves under WOMEN (Skirts/Sweatpants/Basic T‑Shirts…) & add Underwear branch
- Print per-leaf product count
- Robust load-more

Examples
--------
python gstar_scraper_v4_patch_4.py --sections women --out women.csv
python gstar_scraper_v4_patch_4.py --sections men --only-branches "Shoes & Accessories,Underwear" --headful --out men_sa.csv
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
    "tile_name": '[data-testid="product-title-title"], [data-testid="product-title"]',
    "tile_price": '[data-testid="product-tile-price"], [data-testid*="price" i], [class*="price" i]',
    "tile_imgs": "picture img, img",
    "show_next_btn": '[data-testid="productList-showNext"]',
}

# Add "Underwear" as a top-level branch (site occasionally shows it separately)
ALLOWED_BRANCHES: Dict[str, List[str]] = {
    "men": [
        "Jeans & Bottoms",
        "Tops & Hoodies",
        "Jackets & Coats",
        "Shoes & Accessories",
        "Underwear",
    ],
    "women": [
        "Jeans & Bottoms",
        "Tops & Hoodies",
        "Jackets & Coats",
        "Dresses & Jumpsuits",
        "Shoes & Accessories",
        "Underwear",  # present in some locales; silently skipped if not found
    ],
    "kids": ["Boys", "Girls"],
}

# URL fragment fallbacks
FALLBACK_HREFS: Dict[str, Dict[str, List[str]]] = {
    "men": {
        "Jeans & Bottoms": [
            f"{LOCALE}/shop/men/jeans",
            f"{LOCALE}/shop/men/pants",
            f"{LOCALE}/shop/men/shorts",
        ],
        "Tops & Hoodies": [
            f"{LOCALE}/shop/men/t-shirts",
            f"{LOCALE}/shop/men/shirts",
            f"{LOCALE}/shop/men/sweatshirts-hoodies",
            f"{LOCALE}/shop/men/knitwear",
        ],
        "Jackets & Coats": [
            f"{LOCALE}/shop/men/jackets_and_blazers",
            f"{LOCALE}/shop/men/lightweight_jackets",
            f"{LOCALE}/shop/men/overshirts",
            f"{LOCALE}/shop/men/denim_jackets",
        ],
        "Shoes & Accessories": [
            f"{LOCALE}/shop/men/shoes",
            f"{LOCALE}/shop/men/accessories",
        ],
        "Underwear": [f"{LOCALE}/shop/men/underwear"],
    },
    "women": {
        "Jeans & Bottoms": [
            f"{LOCALE}/shop/women/jeans",
            f"{LOCALE}/shop/women/pants",
            f"{LOCALE}/shop/women/shorts",
            f"{LOCALE}/shop/women/skirts",
        ],
        "Tops & Hoodies": [
            f"{LOCALE}/shop/women/t-shirts",
            f"{LOCALE}/shop/women/shirts",
            f"{LOCALE}/shop/women/sweatshirts-hoodies",
            f"{LOCALE}/shop/women/knitwear",
        ],
        "Jackets & Coats": [
            f"{LOCALE}/shop/women/jackets_and_blazers",
            f"{LOCALE}/shop/women/lightweight_jackets",
            f"{LOCALE}/shop/women/overshirts",
            f"{LOCALE}/shop/women/denim_jackets",
        ],
        "Dresses & Jumpsuits": [
            f"{LOCALE}/shop/women/dresses",
            f"{LOCALE}/shop/women/jumpsuits-overalls",
        ],
        "Shoes & Accessories": [
            f"{LOCALE}/shop/women/shoes",
            f"{LOCALE}/shop/women/accessories",
        ],
        "Underwear": [f"{LOCALE}/shop/women/underwear"],
    },
    "kids": {"Boys": [f"{LOCALE}/shop/boys"], "Girls": [f"{LOCALE}/shop/girls"]},
}

CURRENCY_RGX = re.compile(
    r"([€£$])\s?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)"
)
COLORS_RGX = re.compile(r"(\d+)\s+colors?\s+available", re.IGNORECASE)


def _clean_label(s: str) -> str:
    s = re.sub(r"[\u00A0\s]+", " ", s or "")
    s = re.sub(r"\s*&\s*", " & ", s)
    return re.sub(r"\s+", " ", s).strip()


@dataclass
class Product:
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
        only_branches: Optional[Set[str]] = None,
    ) -> None:
        self.pw = pw
        self.browser: Browser = pw.chromium.launch(headless=headless)
        self.throttle = throttle
        self.timeout = timeout_ms
        self.max_per_cat = max_per_cat
        self.verbose = verbose
        self.only_branches = (
            {b.strip().lower() for b in only_branches} if only_branches else None
        )

    def run(self, sections: Iterable[str]) -> List[Product]:
        page = self.browser.new_page()
        out: List[Product] = []
        seen: Set[str] = set()
        for sec in sections:
            sec = sec.lower().strip()
            if sec not in {"men", "women", "kids"}:
                continue
            self._safe_goto(page, f"{HOME}/shop/{sec}")
            links = self._get_leaf_links(page, sec)  # (branch, leaf, url)
            if not links:
                self._log(f"No subcategories found for {sec}.")
                continue
            for branch, leaf, url in links:
                self._log(f"  ↳ {sec} · {leaf} -> {url}")
                arr = self._collect_products(page, sec, branch, leaf, url)
                added = 0
                for p in arr:
                    k = p.key()
                    if k not in seen:
                        out.append(p)
                        seen.add(k)
                        added += 1
                self._log(f"    [done] {leaf}: {added} products")
                if self.max_per_cat:
                    self._log(f"    [limit] max-per-cat={self.max_per_cat} applied")
        page.close()
        self.browser.close()
        return out

    def _get_leaf_links(self, page: Page, section: str) -> List[Tuple[str, str, str]]:
        self._wait_visible(page, SELECTORS["sidebar"], self.timeout)
        sidebar = page.locator(SELECTORS["sidebar"]).first
        if not sidebar or not sidebar.is_visible():
            return []
        out: List[Tuple[str, str, str]] = []
        dedupe: Set[str] = set()
        wanted = ALLOWED_BRANCHES.get(section, [])
        if self.only_branches is not None:
            wanted = [w for w in wanted if w.lower() in self.only_branches]
            if not wanted:
                return []
        for branch in wanted:
            hdr = sidebar.locator(
                f"xpath=(.//*[self::button or self::a or @role='button' or self::div][normalize-space()='{branch}'])[1]"
            )
            anchors = []
            if hdr and hdr.count():
                container = hdr.locator("xpath=following-sibling::*[1]")
                if container and container.count():
                    self._expand_all_toggles(container)
                    if section == "kids":
                        anchors = container.locator(
                            f"xpath=.//a[contains(@href,'{LOCALE}/shop/boys') or contains(@href,'{LOCALE}/shop/girls')]"
                        ).all()
                    else:
                        anchors = container.locator(
                            f"xpath=.//a[contains(@href,'/shop/{section}/')]"
                        ).all()
                if not anchors:
                    anc = hdr.locator(
                        "xpath=ancestor::*[contains(@class,'sideNav__branch')][1]"
                    )
                    if anc and anc.count():
                        self._expand_all_toggles(anc)
                        if section == "kids":
                            anchors = anc.locator(
                                f"xpath=.//a[contains(@href,'{LOCALE}/shop/boys') or contains(@href,'{LOCALE}/shop/girls')]"
                            ).all()
                        else:
                            anchors = anc.locator(
                                f"xpath=.//a[contains(@href,'/shop/{section}/')]"
                            ).all()
            if not anchors:
                pats = FALLBACK_HREFS.get(section, {}).get(branch, [])
                if pats:
                    xp = " or ".join([f"contains(@href,'{p}')" for p in pats])
                    anchors = sidebar.locator(f"xpath=.//a[{xp}]").all()
            for a in anchors:
                try:
                    txt = (a.inner_text() or a.text_content() or "").strip()
                    href = a.get_attribute("href") or ""
                except Exception:
                    continue
                if not href:
                    continue
                leaf = _clean_label(txt)
                if "shop all" in leaf.lower():
                    continue
                if section != "kids" and f"/shop/{section}" not in href:
                    continue
                absu = self._abs_url(href)
                key = (absu.split("?")[0], leaf)
                if key in dedupe:
                    continue
                dedupe.add(key)
                out.append((_clean_label(branch), leaf, absu))
        return out

    def _expand_all_toggles(self, root) -> None:
        rounds = 0
        while rounds < 10:
            toggles = root.locator(
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

    def _collect_products(
        self, page: Page, section: str, branch: str, category: str, url: str
    ) -> List[Product]:
        self._safe_goto(page, url)
        self._wait_visible(page, SELECTORS["plp_grid"], self.timeout)
        self._load_all_results(page)
        items: List[Product] = []
        try:
            tiles = page.query_selector_all(SELECTORS["product_tile"]) or []
        except Error:
            try:
                page.wait_for_load_state("domcontentloaded", timeout=3000)
                self._wait_visible(page, SELECTORS["plp_grid"], 5000)
                tiles = page.query_selector_all(SELECTORS["product_tile"]) or []
            except Exception:
                tiles = []
        for idx, t in enumerate(tiles, start=1):
            p = self._parse_tile(t, section, branch, category)
            if not p.product_url:
                continue
            p.order = idx
            items.append(p)
            if self.max_per_cat and len(items) >= self.max_per_cat:
                break
        return items

    def _load_all_results(self, page: Page, max_clicks: int = 400) -> None:
        clicks = 0
        while clicks < max_clicks:
            try:
                before = len(page.query_selector_all(SELECTORS["product_tile"]) or [])
            except Error:
                try:
                    page.wait_for_load_state("domcontentloaded", timeout=3000)
                    self._wait_visible(page, SELECTORS["plp_grid"], 5000)
                    before = len(
                        page.query_selector_all(SELECTORS["product_tile"]) or []
                    )
                except Exception:
                    break
            clicked = self._try_click_show_next(page, before)
            if not clicked:
                try:
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                except Error:
                    break
                time.sleep(random.uniform(*self.throttle))
            try:
                after = len(page.query_selector_all(SELECTORS["product_tile"]) or [])
            except Error:
                break
            if not clicked and after <= before:
                break
            clicks += 1

    def _try_click_show_next(self, page: Page, before_count: int) -> bool:
        try:
            btn = page.locator(SELECTORS["show_next_btn"]).first
            if btn and btn.is_visible():
                btn.click(timeout=2000)
                try:
                    page.wait_for_function(
                        "(s,n)=>document.querySelectorAll(s).length>n",
                        (SELECTORS["product_tile"], before_count),
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
        for label in (
            "Show the next",
            "Load more",
            "Show more",
            "Load More",
            "Show More",
        ):
            try:
                b = page.locator(f"button:has-text('{label}')").first
                if b and b.is_visible():
                    b.click(timeout=1500)
                    try:
                        page.wait_for_function(
                            "(s,n)=>document.querySelectorAll(s).length>n",
                            (SELECTORS["product_tile"], before_count),
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

    def _parse_tile(self, tile, section: str, branch: str, category: str) -> Product:
        a = tile.query_selector(SELECTORS["tile_link"]) or tile.query_selector("a")
        url_abs = self._abs_url(a.get_attribute("href") if a else None) or ""
        name_el = tile.query_selector(SELECTORS["tile_name"]) or a
        name = (
            (name_el.inner_text().strip() if name_el else "").replace("\n", " ").strip()
        )
        price_texts = [
            (n.inner_text() or "").strip()
            for n in tile.query_selector_all(SELECTORS["tile_price"]) or []
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
                for part in ss.split(",") if ss else []:
                    part = part.strip()
                    u, _, w = part.partition(" ")
                    image_url = self._abs_url(u) or image_url
        return Product(
            section=section,
            branch=_clean_label(branch),
            category=_clean_label(category),
            title=name,
            price_current=price_current,
            price_original=price_original,
            product_url=url_abs,
            image_url=image_url,
            colors_available=colors_available,
            colors_text=colors_text,
        )

    # utils
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
        for sel in (
            "button:has-text('Reject')",
            "button:has-text('Reject all')",
            "button:has-text('Decline')",
            "button:has-text('Only necessary')",
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
        print(msg, flush=True)


# CLI


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
    p.add_argument(
        "--only-branches",
        default=None,
        help="Comma-separated top-level branches (e.g. 'Jeans & Bottoms,Shoes & Accessories,Underwear')",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sections = (
        ["men", "women", "kids"]
        if args.sections.lower() == "all"
        else [s.strip() for s in args.sections.split(",")]
    )
    only: Optional[Set[str]] = None
    if args.only_branches:
        only = {s.strip().lower() for s in args.only_branches.split(",") if s.strip()}
    with sync_playwright() as pw:
        scraper = GStarScraper(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            max_per_cat=args.max_per_cat,
            only_branches=only,
        )
        products = scraper.run(sections)
    rows = [asdict(p) for p in products]
    if not rows:
        print("No products scraped.")
        return 1
    df = pd.DataFrame(rows)
    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    df.to_csv(args.out, index=False)
    if args.xlsx:
        with pd.ExcelWriter(args.xlsx, engine="xlsxwriter") as writer:
            used: Set[str] = set()
            max_rows = 1_048_000

            def name_ok(s: str) -> str:
                s = re.sub(r"[:\\/?*\[\]]", " ", s)
                s = re.sub(r"\s+", " ", s).strip()
                return s[:31]

            for (sec, br, cat), grp in df.groupby(
                ["section", "branch", "category"], sort=False
            ):
                base = name_ok(f"{sec}-{br}-{cat}")
                nm = base
                i = 2
                while nm in used:
                    suf = f"_{i}"
                    nm = name_ok(base[: (31 - len(suf))] + suf)
                    i += 1
                used.add(nm)
                if len(grp) <= max_rows:
                    grp.to_excel(writer, sheet_name=nm, index=False)
                else:
                    parts = (len(grp) // max_rows) + 1
                    for p in range(parts):
                        chunk = grp.iloc[p * max_rows : (p + 1) * max_rows]
                        nm2 = name_ok(f"{nm}_part{p+1}")
                        chunk.to_excel(writer, sheet_name=nm2, index=False)
    print(
        f"Saved {len(df)} products -> {args.out}"
        + (f" & {args.xlsx}" if args.xlsx else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
