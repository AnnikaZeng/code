#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-STAR RAW scraper (Playwright) – v4 patch (3rd)
Focus:
1) `category` 与站点左侧一致：
   - 新增 `branch` 列（顶级分组：Jeans & Bottoms / Tops & Hoodies / ...）。
   - `category` 保留最细分叶子名称（与左侧锚点一致）。
   - 统一空格（含 `&` 两侧、\u00a0）。
2) 每抓完一个叶子品类，打印抓取产品数量。
3) 保留 v4 的稳定性与 Kids/Boys/Girls、Load-more 等修复；支持 `--only-branches` 仅跑指定分组调试。

用法：
    pip install playwright pandas xlsxwriter
    playwright install

    # 仅跑 MEN 下 Jeans & Bottoms 与 Shoes & Accessories 两个分组
    python gstar_scraper_v4_patch_3.py --sections men --only-branches "Jeans & Bottoms,Shoes & Accessories" --max-per-cat 50 --headful --out men.csv

    python gstar_scraper_v4_patch_3.py --sections men --only-branches "Shoes & Accessories"  --headful --out men_branches.csv
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
    },
    "women": {
        "Jeans & Bottoms": [
            f"{LOCALE}/shop/women/jeans",
            f"{LOCALE}/shop/women/pants",
            f"{LOCALE}/shop/women/shorts",
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
    },
    "kids": {
        "Boys": [f"{LOCALE}/shop/boys"],
        "Girls": [f"{LOCALE}/shop/girls"],
    },
}

CURRENCY_RGX = re.compile(
    r"([€£$])\s?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)"
)
COLORS_RGX = re.compile(r"(\d+)\s+colors?\s+available", re.IGNORECASE)


def _clean_label(s: str) -> str:
    # why: 统一 & 两侧空格、去除不可见空格、压缩空白
    s = re.sub(r"[\u00A0\s]+", " ", s or "")
    s = re.sub(r"\s*&\s*", " & ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


@dataclass
class Product:
    section: str
    branch: str  # 顶级分组（左侧大类）
    category: str  # 叶子品类（左侧锚点）
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

    # -------- Public API -------- #
    def run(self, sections: Iterable[str]) -> List[Product]:
        page = self.browser.new_page()
        out: List[Product] = []
        dedupe: Set[str] = set()

        for sec in sections:
            sec = sec.lower().strip()
            if sec not in {"men", "women", "kids"}:  # 防呆
                continue
            self._safe_goto(page, f"{HOME}/shop/{sec}")
            links = self._get_leaf_links(page, sec)  # [(branch, leaf, url)]
            if not links:
                self._log(f"No subcategories found for {sec}.")
                continue

            for branch, leaf, url in links:
                self._log(f"  ↳ {sec} · {leaf} -> {url}")
                cat_products = self._collect_products(page, sec, branch, leaf, url)
                # 去重 & 追加
                added = 0
                for p in cat_products:
                    if p.key() not in dedupe:
                        out.append(p)
                        dedupe.add(p.key())
                        added += 1
                self._log(f"    [done] {leaf}: {added} products")
                if self.max_per_cat:
                    self._log(f"    [limit] max-per-cat={self.max_per_cat} applied")

        page.close()
        self.browser.close()
        return out

    # -------- Sidebar -------- #
    def _get_leaf_links(self, page: Page, section: str) -> List[Tuple[str, str, str]]:
        self._wait_visible(page, SELECTORS["sidebar"], self.timeout)
        sidebar = page.locator(SELECTORS["sidebar"]).first
        if not sidebar or not sidebar.is_visible():
            return []

        out: List[Tuple[str, str, str]] = []
        seen: Set[str] = set()
        wanted = ALLOWED_BRANCHES.get(section, [])
        if self.only_branches is not None:
            wanted = [w for w in wanted if w.lower() in self.only_branches]
            if not wanted:
                return []

        for branch in wanted:
            # header 可能是 a/button/div(role) 或纯 div
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
                            f"xpath=.//a[contains(@href, '/shop/{section}/')]"
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
                                f"xpath=.//a[contains(@href, '/shop/{section}/')]"
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
                key = absu.split("?")[0]
                if key in seen:
                    continue
                out.append((_clean_label(branch), leaf, absu))
                seen.add(key)

        return out

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

    # -------- Collect -------- #
    def _collect_products(
        self, page: Page, section: str, branch: str, category: str, url: str
    ) -> List[Product]:
        self._safe_goto(page, url)
        self._wait_visible(page, SELECTORS["plp_grid"], self.timeout)
        self._load_all_results(page)

        products: List[Product] = []
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
            products.append(p)
            if self.max_per_cat and len(products) >= self.max_per_cat:
                break
        return products

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

    def _try_click_show_next(self, page: Page, before_count: int = 0) -> bool:
        try:
            btn = page.locator(SELECTORS["show_next_btn"]).first
            if btn and btn.is_visible():
                btn.click(timeout=2000)
                try:
                    page.wait_for_function(
                        "(sel, n) => document.querySelectorAll(sel).length > n",
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
                            "(sel, n) => document.querySelectorAll(sel).length > n",
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

    # -------- Tile -------- #
    def _parse_tile(self, tile, section: str, branch: str, category: str) -> Product:
        a = tile.query_selector(SELECTORS["tile_link"]) or tile.query_selector("a")
        url_abs = self._abs_url(a.get_attribute("href") if a else None) or ""

        name_el = tile.query_selector(SELECTORS["tile_name"]) or a
        name = (
            (name_el.inner_text().strip() if name_el else "").replace("\n", " ").strip()
        )

        # 价格
        price_texts = []
        for n in tile.query_selector_all(SELECTORS["tile_price"]) or []:
            txt = (n.inner_text() or "").strip()
            if txt:
                price_texts.append(txt)
        joined = " ".join(price_texts)
        prices = list(CURRENCY_RGX.finditer(joined))
        price_current = prices[0].group(0) if prices else None
        price_original = prices[-1].group(0) if len(prices) >= 2 else None

        # 颜色
        colors_text = None
        for p in tile.query_selector_all("p, span, div")[:8] or []:
            t = (p.inner_text() or "").strip()
            if "color" in t.lower():
                colors_text = t
                break
        colors_available = None
        m = re.search(r"(\d+)\s+colors?\b", colors_text or "", flags=re.I)
        if m:
            try:
                colors_available = int(m.group(1))
            except ValueError:
                colors_available = None

        # 图片
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
                if ss:
                    # 取分辨率最大的
                    best_w, best_url = 0, None
                    for part in ss.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        u, _, w = part.partition(" ")
                        try:
                            wv = int((w or "0").replace("w", "").strip())
                        except ValueError:
                            wv = 0
                        if wv >= best_w:
                            best_w, best_url = wv, u
                    if best_url:
                        image_url = self._abs_url(best_url)
                        break

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

    # -------- Utils -------- #
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


# -------- CLI -------- #


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
        help="Comma-separated top-level branches to include (e.g. 'Jeans & Bottoms,Shoes & Accessories')",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    sections = (
        ["men", "women", "kids"]
        if args.sections.lower() == "all"
        else [s.strip() for s in args.sections.split(",")]
    )
    only_branches: Optional[Set[str]] = None
    if args.only_branches:
        only_branches = {
            s.strip().lower() for s in args.only_branches.split(",") if s.strip()
        }

    with sync_playwright() as pw:
        scraper = GStarScraper(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            max_per_cat=args.max_per_cat,
            only_branches=only_branches,
        )
        products = scraper.run(sections)

    rows = [asdict(p) for p in products]
    if not rows:
        print("No products scraped.")
        return 1

    df = pd.DataFrame(rows)

    # 标题单行
    df["title"] = (
        df["title"]
        .astype(str)
        .str.replace(r"[\r\n]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # 不改变抓取顺序（不排序）
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

            for (sec, br, cat), grp in df.groupby(
                ["section", "branch", "category"], sort=False
            ):
                base = sanitize(f"{sec}-{br}-{cat}") or "sheet"
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
