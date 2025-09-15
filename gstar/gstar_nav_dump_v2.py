#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
爬出的导航很全，但要筛选想要的
G-STAR RAW – Sidebar Navigation Dumper (v2, DOM-accurate)

Reads the left sidebar on /shop/{men|women|kids} and outputs rows:
    section, branch, category, url

DOM contracts (from your screenshot):
- Branch containers have a class that contains "sideNav__branch".
- Leaf items are under elements whose class contains "sideNav__leaf" and hold an <a>.
- The branch header text is inside a <button> (or role=button) within that branch container.

Why this v2:
- Avoids over-clicking (soft-limit attempts) to prevent hangs.
- Traverses true DOM structure; no whitelist; kids supports /shop/boys & /shop/girls.

Usage
-----
    pip install playwright pandas
    playwright install

    # WOMEN (headful for debugging)
    python gstar_nav_dump_v2.py --sections women --headful --out women_nav.csv

    # MEN + WOMEN
    python gstar_nav_dump_v2.py --sections men,women --out nav.csv --json nav.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
from playwright.sync_api import Browser, Page, Playwright, TimeoutError, sync_playwright

BASE = "https://www.g-star.com"
LOCALE = "/en_us"
HOME = f"{BASE}{LOCALE}"

SEL: Dict[str, str] = {
    "sidebar": '#sideNav, nav[aria-label="Sidebar"], aside[aria-label="Sidebar"]',
    "branch": '[class*="sideNav__branch"]',  # branch container
    "leaf_a": '[class*="sideNav__leaf"] a[href]',
}


def clean(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"[\u00A0\s]+", " ", s)
    s = re.sub(r"\s*&\s*", " & ", s)
    return re.sub(r"\s+", " ", s).strip()


class NavDump:
    def __init__(
        self, pw: Playwright, headless: bool, timeout_ms: int, verbose: bool
    ) -> None:
        self.browser: Browser = pw.chromium.launch(headless=headless)
        self.timeout = timeout_ms
        self.verbose = verbose

    def dump(self, sections: Iterable[str]) -> List[Dict[str, str]]:
        page = self.browser.new_page()
        rows: List[Dict[str, str]] = []

        for sec in sections:
            sec = sec.strip().lower()
            if sec not in {"men", "women", "kids"}:
                self._log(f"[skip] unknown section: {sec}")
                continue

            url = f"{HOME}/shop/{sec}"
            self._goto(page, url)
            sidebar = self._wait_sidebar(page)
            if not sidebar:
                self._log("[warn] sidebar not visible")
                continue

            # Make a best-effort to expand collapsed groups (limited attempts)
            self._soft_expand(sidebar)

            branches = sidebar.locator(SEL["branch"]).all() or []
            self._log(f"[{sec}] branches found: {len(branches)}")

            seen: Set[Tuple[str, str]] = set()
            for b in branches:
                # branch header text (first clickable heading with text)
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

                # all leaves under this branch
                anchors = b.locator(SEL["leaf_a"]).all()
                for a in anchors:
                    try:
                        txt = clean((a.inner_text() or a.text_content() or "").strip())
                        href = a.get_attribute("href") or ""
                    except Exception:
                        continue
                    if not txt or not href:
                        continue
                    if "shop all" in txt.lower():
                        continue
                    # section filter: kids may link to /boys or /girls
                    if sec == "kids":
                        if not ("/shop/boys" in href or "/shop/girls" in href):
                            continue
                    else:
                        if f"/shop/{sec}/" not in href:
                            continue
                    absu = self._abs(href)
                    key = (branch_name, absu.split("?")[0])
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(
                        {
                            "section": sec,
                            "branch": branch_name,
                            "category": txt,
                            "url": absu,
                        }
                    )

            # Pretty print tree for this section
            self._log(f"\n== {sec.upper()} NAV ==")
            groups: Dict[str, List[str]] = {}
            for r in rows:
                if r["section"] != sec:
                    continue
                groups.setdefault(r["branch"] or "(unknown)", []).append(r["category"])
            for br, cats in groups.items():
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
        time.sleep(0.25)
        self._dismiss_cookie(page)

    def _wait_sidebar(self, page: Page):
        try:
            page.wait_for_selector(
                SEL["sidebar"], timeout=self.timeout, state="visible"
            )
            return page.locator(SEL["sidebar"]).first
        except TimeoutError:
            return None

    def _soft_expand(self, sidebar) -> None:
        # few rounds; short timeouts
        for _ in range(5):
            toggles = sidebar.locator(
                "button[aria-expanded='false'], [role='button'][aria-expanded='false']"
            ).all()
            if not toggles:
                break
            progressed = False
            for t in toggles[:30]:  # cap to avoid long loops
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
        ):
            try:
                btn = page.locator(sel).first
                if btn and btn.is_visible():
                    btn.click(timeout=600)
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
    p = argparse.ArgumentParser(description="Dump G-STAR sidebar navigation (v2)")
    p.add_argument("--sections", default="women", help="men,women,kids or 'all'")
    p.add_argument("--out", default=None, help="CSV output for nav dump")
    p.add_argument("--json", default=None, help="JSON output for nav dump")
    p.add_argument(
        "--headful", action="store_true", help="Run non-headless browser for debugging"
    )
    p.add_argument("--timeout", type=int, default=30000, help="page timeout ms")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    sections = (
        ["men", "women", "kids"]
        if args.sections.lower() == "all"
        else [s.strip() for s in args.sections.split(",")]
    )

    with sync_playwright() as pw:
        s = NavDump(
            pw,
            headless=not args.headful,
            timeout_ms=args.timeout,
            verbose=not args.quiet,
        )
        rows = s.dump(sections)

    if not rows:
        print("No nav rows.")
        return 1

    if args.out:
        pd.DataFrame(rows).to_csv(args.out, index=False)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)

    print(f"Dumped {len(rows)} nav rows" + (f" -> {args.out}" if args.out else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
