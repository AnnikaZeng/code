#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G-STAR RAW scraper (Playwright) – v4 patch (5th)

What’s new (per your requests):
1) **导航结构导出**：新增 `--dump-nav` 模式，先把左侧导航的 *branch*（顶级分组）与 *category*（叶子锚点）完整抓出并导出。
   - 不依赖白名单；直接以页面真实 DOM 为准（含 *Underwear* 若出现在 *Shoes & Accessories* 下）。
   - `--nav-out nav.csv` / `--nav-json nav.json` 可保存；控制台亦打印层级。
2) 保留既有商品抓取能力（与之前 v4 patch 逻辑一致），未改动输出顺序策略与 load-more 稳定性。

Examples
--------
# 仅导出 WOMEN 导航结构
python gstar_scraper_v4_patch_5.py --sections women --dump-nav --nav-out women_nav.csv

# 导出导航后再做商品抓取（常规）
python gstar_scraper_v4_patch_5.py --sections women --out women.csv
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

CURRENCY_RGX = re.compile(
    r"([€£$])\s?([0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]{2})?|[0-9]+)"
)
COLORS_RGX = re.compile(r"(\d+)\s+colors?\s+available", re.IGNORECASE)


def _clean_label(s: str) -> str:
    s = re.sub(r"[\u00A0\s]+", " ", s or "")  # nbsp/空白合并
    s = re.sub(r"\s*&\s*", " & ", s)  # & 两侧空格规范
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
    ) -> None:
        self.pw = pw
        self.browser: Browser = pw.chromium.launch(headless=headless)
        self.throttle = throttle
        self.timeout = timeout_ms
        self.max_per_cat = max_per_cat
        self.verbose = verbose

    # ---------- NAV DUMP ---------- #
    def dump_navigation(self, sections: Iterable[str]) -> List[Dict[str, str]]:
        page = self.browser.new_page()
        rows: List[Dict[str, str]] = []
        for sec in sections:
            sec = sec.strip().lower()
            if sec not in {"men", "women", "kids"}:
                continue
            self._safe_goto(page, f"{HOME}/shop/{sec}")
            self._wait_visible(page, SELECTORS["sidebar"], self.timeout)
            side = page.locator(SELECTORS["sidebar"]).first
            if not side or not side.is_visible():
                continue
            self._expand_all_toggles(side)

            # 采集所有 sidebar 链接（仅 /shop/{section}/…；Kids 允许 /shop/boys|/shop/girls）
            if sec == "kids":
                a_nodes = side.locator(
                    f"a[href*='{LOCALE}/shop/boys'], a[href*='{LOCALE}/shop/girls']"
                ).all()
            else:
                a_nodes = side.locator(f"a[href*='/shop/{sec}/']").all()

            seen: Set[Tuple[str, str, str]] = set()
            for a in a_nodes:
                try:
                    href = a.get_attribute("href") or ""
                    txt = (a.inner_text() or a.text_content() or "").strip()
                except Exception:
                    continue
                if not href or not txt:
                    continue
                if "shop all" in txt.lower():
                    continue
                leaf = _clean_label(txt)
                absu = self._abs_url(href)

                # 找到最近 sideNav__branch 祖先并提取其 header 文本
                branch = ""
                try:
                    anc = a.locator(
                        "xpath=ancestor::*[contains(@class,'sideNav__branch')][1]"
                    )
                    if anc and anc.count():
                        # header 通常是有 aria-expanded 的 button / role=button
                        hdr = anc.locator(
                            "xpath=.//*[(@aria-expanded or @role='button') and (self::button or self::a or self::div)][normalize-space()][1]"
                        ).first
                        if not hdr or not hdr.count():
                            hdr = anc.locator(
                                "xpath=.//*[self::button or self::a or @role='button' or self::div][normalize-space()][1]"
                            ).first
                        if hdr and hdr.count():
                            branch = _clean_label(
                                hdr.inner_text() or hdr.text_content() or ""
                            )
                except Exception:
                    branch = ""
                key = (sec, branch, leaf)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "section": sec,
                        "branch": branch,
                        "category": leaf,
                        "url": absu,
                    }
                )

            # 控制台分组打印，便于你对照：
            self._log(f"\n== {sec.upper()} NAV ==")
            groups: Dict[str, List[str]] = {}
            for r in rows:
                if r["section"] != sec:
                    continue
                groups.setdefault(r["branch"] or "(unknown)", []).append(r["category"])
            for b, cats in groups.items():
                self._log(f"  {b}:")
                for c in cats:
                    self._log(f"    - {c}")

        page.close()
        self.browser.close()
        return rows

    # ---------- PRODUCT SCRAPE ---------- #
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
            for branch, leaf, url in links:
                self._log(f"  ↳ {sec} · {leaf} -> {url}")
                for p in self._collect_products(page, sec, branch, leaf, url):
                    if p.key() not in dedupe:
                        out.append(p)
                        dedupe.add(p.key())
                if self.max_per_cat:
                    self._log(f"    [limit] max-per-cat={self.max_per_cat} applied")
        page.close()
        self.browser.close()
        return out

    # ---------- Sidebar helpers (for scrape) ---------- #
    def _get_leaf_links(self, page: Page, section: str) -> List[Tuple[str, str, str]]:
        self._wait_visible(page, SELECTORS["sidebar"], self.timeout)
        sidebar = page.locator(SELECTORS["sidebar"]).first
        if not sidebar or not sidebar.is_visible():
            return []
        self._expand_all_toggles(sidebar)
        if section == "kids":
            anchors = sidebar.locator(
                f"a[href*='{LOCALE}/shop/boys'], a[href*='{LOCALE}/shop/girls']"
            ).all()
        else:
            anchors = sidebar.locator(f"a[href*='/shop/{section}/']").all()
        out: List[Tuple[str, str, str]] = []
        seen: Set[Tuple[str, str]] = set()
        for a in anchors:
            try:
                href = a.get_attribute("href") or ""
                txt = (a.inner_text() or a.text_content() or "").strip()
            except Exception:
                continue
            if not href or not txt:
                continue
            if "shop all" in txt.lower():
                continue
            leaf = _clean_label(txt)
            absu = self._abs_url(href)
            # branch
            branch = ""
            try:
                anc = a.locator(
                    "xpath=ancestor::*[contains(@class,'sideNav__branch')][1]"
                )
                if anc and anc.count():
                    hdr = anc.locator(
                        "xpath=.//*[(@aria-expanded or @role='button') and (self::button or self::a or self::div)][normalize-space()][1]"
                    ).first
                    if not hdr or not hdr.count():
                        hdr = anc.locator(
                            "xpath=.//*[self::button or self::a or @role='button' or self::div][normalize-space()][1]"
                        ).first
                    if hdr and hdr.count():
                        branch = _clean_label(
                            hdr.inner_text() or hdr.text_content() or ""
                        )
            except Exception:
                pass
            key = (branch, absu.split("?")[0])
            if key in seen:
                continue
            seen.add(key)
            out.append((branch, leaf, absu))
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

    # ---------- Collection ---------- #
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
        self._log(f"    [done] {category}: {len(items)} products")
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
        price_texts = []
        for n in tile.query_selector_all(SELECTORS["tile_price"]) or []:
            txt = (n.inner_text() or "").strip()
            if txt:
                price_texts.append(txt)
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
                    u, _, _w = part.partition(" ")
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

    # ---------- Utils ---------- #
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
        if self.verbose:
            print(msg, flush=True)


# ---------- CLI ---------- #


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="G-STAR RAW scraper (Playwright)")
    p.add_argument("--sections", default="all", help="men,women,kids or 'all'")
    p.add_argument(
        "--dump-nav",
        action="store_true",
        help="Dump sidebar branches+categories only and exit",
    )
    p.add_argument(
        "--nav-out", default=None, help="Optional CSV path for --dump-nav output"
    )
    p.add_argument(
        "--nav-json", default=None, help="Optional JSON path for --dump-nav output"
    )
    p.add_argument(
        "--out", default="products.csv", help="CSV output path (product mode)"
    )
    p.add_argument(
        "--jsonl", default=None, help="Optional JSONL output path (product mode)"
    )
    p.add_argument("--xlsx", default=None, help="Optional Excel path (product mode)")
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
        else [s.strip() for s in args.sections.split(",")]
    )

    with sync_playwright() as pw:
        scraper = GStarScraper(pw, headless=not args.headful, timeout_ms=args.timeout)

        if args.dump_nav:
            nav_rows = scraper.dump_navigation(sections)
            if not nav_rows:
                print("No navigation found.")
                return 1
            df = pd.DataFrame(nav_rows)
            if args.nav_out:
                df.to_csv(args.nav_out, index=False)
            if args.nav_json:
                with open(args.nav_json, "w", encoding="utf-8") as f:
                    json.dump(nav_rows, f, ensure_ascii=False, indent=2)
            print(
                f"Dumped {len(df)} nav rows"
                + (f" -> {args.nav_out}" if args.nav_out else "")
            )
            return 0

        # product scraping path
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

    # 保留抓取顺序（不排序）
    df.to_csv(args.out, index=False)

    if args.jsonl:
        with open(args.jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

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
