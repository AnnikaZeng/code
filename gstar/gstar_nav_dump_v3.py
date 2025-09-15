#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依次爬取不同的国家导航栏
G-STAR RAW – Sidebar Navigation Dumper (v3)

Adds --markets support and outputs a CSV that matches your template:
    market,section,branch,category,url

Design
- Iterate markets (locales) and sections.
- For each /{market}/shop/{section} page, expand left sidebar groups softly.
- Branch container: [class*="sideNav__branch"].
- Leaf anchors:     [class*="sideNav__leaf"] a[href].
- Branch text: the first button/[role=button]/a with non-empty text inside branch.
- Filter out "Shop All" items only; keep the rest for manual screening.

Usage
-----
# Dump UK + NL + FR(=en_fr) + JP to one CSV
python gstar_nav_dump_v3.py \
  --markets en_gb,en_nl,en_fr,en_jp \
  --sections men,women \
  --out all_nav.csv --headful

# Dump single market to separate file (example: NL)
python gstar_nav_dump_v3.py --markets en_nl --sections men,women --out nl_nav.csv

Notes
- Kids later; you can pass --sections kids when needed.
- This script only collects navigation; use gstar_scraper_navdriven_v2.py to crawl products from this CSV.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from playwright.sync_api import (
    Browser,
    Locator,
    Page,
    Playwright,
    TimeoutError,
    sync_playwright,
)

BASE = "https://www.g-star.com"

SEL: Dict[str, str] = {
    "sidebar": 'aside[aria-label="Sidebar"], nav[aria-label="Sidebar"], #sideNav',
    "branch": '[class*="sideNav__branch"]',
    "leaf_a": '[class*="sideNav__leaf"] a[href]',
}


def clean(s: str) -> str:
    s = re.sub(r"[\u00A0\s]+", " ", s or "")
    s = re.sub(r"\s*&\s*", " & ", s)
    return re.sub(r"\s+", " ", s).strip()


class NavDump:
    def __init__(
        self, pw: Playwright, headless: bool, timeout_ms: int, verbose: bool
    ) -> None:
        self.browser: Browser = pw.chromium.launch(headless=headless)
        self.timeout = timeout_ms
        self.verbose = verbose

    def dump(
        self, markets: Iterable[str], sections: Iterable[str]
    ) -> List[Dict[str, str]]:
        page = self.browser.new_page()
        rows: List[Dict[str, str]] = []

        for market in markets:
            market = market.strip().lower()
            for sec in sections:
                sec = sec.strip().lower()
                if sec not in {"men", "women", "kids"}:
                    self._log(f"[skip] unknown section: {sec}")
                    continue

                url = f"{BASE}/{market}/shop/{sec}"
                self._goto(page, url)
                sidebar = self._wait_sidebar(page)
                if not sidebar:
                    self._log("[warn] sidebar not visible")
                    continue
                self._soft_expand(sidebar)

                branches = sidebar.locator(SEL["branch"]).all() or []
                self._log(f"[{market}/{sec}] branches: {len(branches)}")

                for b in branches:
                    hdr = b.locator(
                        "xpath=.//*[self::button or @role='button' or self::a][normalize-space()][1]"
                    ).first
                    try:
                        branch_name = (
                            clean(hdr.inner_text() or hdr.text_content() or "")
                            if hdr and hdr.count()
                            else ""
                        )
                    except Exception:
                        branch_name = ""

                    anchors = b.locator(SEL["leaf_a"]).all()
                    for a in anchors:
                        try:
                            txt = clean(
                                (a.inner_text() or a.text_content() or "").strip()
                            )
                            href = a.get_attribute("href") or ""
                        except Exception:
                            continue
                        if not txt or not href:
                            continue
                        if "shop all" in txt.lower():
                            continue
                        # Only keep leaves under this section's subtree
                        sec_prefix_ok = (f"/shop/{sec}/" in href) or (
                            sec == "kids"
                            and ("/shop/boys" in href or "/shop/girls" in href)
                        )
                        if not sec_prefix_ok:
                            continue
                        absu = self._abs(href)
                        rows.append(
                            {
                                "market": market,
                                "section": sec,
                                "branch": branch_name,
                                "category": txt,
                                "url": absu,
                            }
                        )

                # Pretty print tree for quick visual check
                self._log(f"\n== {market.upper()}/{sec.upper()} NAV ==")
                grouped: Dict[str, List[str]] = {}
                for r in rows:
                    if r["market"] == market and r["section"] == sec:
                        grouped.setdefault(r["branch"] or "(unknown)", []).append(
                            r["category"]
                        )
                for br, cats in grouped.items():
                    self._log(f"  {br}:")
                    for c in cats:
                        self._log(f"    - {c}")

        page.close()
        self.browser.close()
        return rows

    # ---- helpers ----
    def _goto(self, page: Page, url: str) -> None:
        try:
            page.goto(url, timeout=self.timeout, wait_until="domcontentloaded")
        except TimeoutError:
            self._log(f"[timeout] goto: {url}")
        time.sleep(0.3)
        self._dismiss_cookie(page)

    def _wait_sidebar(self, page: Page):
        try:
            page.wait_for_selector(
                SEL["sidebar"], timeout=self.timeout, state="visible"
            )
            return page.locator(SEL["sidebar"]).first
        except TimeoutError:
            return None

    def _soft_expand(self, sidebar: Locator) -> None:
        for _ in range(6):
            toggles = sidebar.locator(
                "button[aria-expanded='false'], [role='button'][aria-expanded='false']"
            ).all()
            if not toggles:
                break
            progressed = False
            for t in toggles[:30]:
                try:
                    if t.is_visible():
                        t.click(timeout=250)
                        time.sleep(0.08)
                        progressed = True
                except Exception:
                    continue
            if not progressed:
                break

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
                    btn.click(timeout=800)
                    time.sleep(0.1)
                    return
            except Exception:
                continue

    def _abs(self, href: Optional[str]) -> Optional[str]:
        if not href:
            return None
        if href.startswith("http"):
            return href
        if href.startswith("/"):
            return f"{BASE}{href}"
        return None

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


# ---- CLI ----


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump G-STAR sidebar navigation (v3)")
    p.add_argument(
        "--markets",
        required=True,
        help="Comma-separated locales, e.g. en_gb,en_nl,en_fr,en_jp",
    )
    p.add_argument(
        "--sections", default="men,women", help="Comma-separated: men,women[,kids]"
    )
    p.add_argument(
        "--out",
        default="nav.csv",
        help="CSV output path (market,section,branch,category,url)",
    )
    p.add_argument("--json", default=None, help="Optional JSON output path")
    p.add_argument(
        "--headful", action="store_true", help="Run non-headless browser for debugging"
    )
    p.add_argument("--timeout", type=int, default=30000, help="Page timeout ms")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    markets = [s.strip() for s in args.markets.split(",") if s.strip()]
    sections = [s.strip() for s in args.sections.split(",") if s.strip()]

    with sync_playwright() as pw:
        s = NavDump(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            verbose=not args.quiet,
        )
        rows = s.dump(markets, sections)

    if not rows:
        print("No nav rows.")
        return 1

    df = pd.DataFrame(
        rows, columns=["market", "section", "branch", "category", "url"]
    )  # ensure column order
    df.to_csv(args.out, index=False)

    if args.json:
        import json

        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Dumped {len(df)} nav rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
