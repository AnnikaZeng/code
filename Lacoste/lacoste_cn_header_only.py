# -*- coding: utf-8 -*-
"""
File: scripts/lacoste_cn_header_v2.py
Goal: 点击城市后跳去别的国家/门店的问题，改为“提取城市 href → 直接 goto 该 URL”，
     不再触发复杂的前端 click 逻辑，避免 SPA 重排/虚拟列表/跨域跳转。

主要变化：
1) 不再对 <a> 触发 click；改为提前抓取每个城市的 href（形如 /us/stores?country=china&city=xxx）。
2) 对每个城市直接 page.goto(完整URL)，读取左列 header（.st-header），再返回城市目录页。
3) 保证 URL 始终带 country=china；若不带或被跳走，强制回到带参的 URL。
4) 解决 TargetClosedError：若 page 被关/被顶替，自动新开页并恢复。
5) 更稳的等待：页面 ready、header 可见与 header 文本变化。
"""
from __future__ import annotations
import re
import time
import random
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse

import pandas as pd
from playwright.sync_api import sync_playwright, Page, Locator, BrowserContext

START_URL = "https://www.lacoste.com/us/stores/china"
COUNTS_CSV = "lacoste_cn_city_counts.csv"
COUNTS_XLSX = "lacoste_cn_city_counts.xlsx"

# ====== 选择器（左列） ======
CITY_PANEL_SELECTOR = (
    ".js-storelocator.st-main .st-main-content.st-grid "
    ".st-grid-col2 #st-frame-container .st-frame-inner "
    "#page-search > ul.st-list.st-list-spacing"
)
CITY_LINKS_SELECTOR = "li.st-list-item a, li.st-list-item > a, a"

STORE_BLOCK_SELECTOR = (
    ".js-storelocator.st-main .st-main-content.st-grid "
    ".st-grid-col2 .st-frame .st-frame-inner .st-frame-item"
)
STORE_COUNT_SELECTOR = f"{STORE_BLOCK_SELECTOR} .st-header"

# ====== Tuning ======
SLOW_MIN_MS, SLOW_MAX_MS = 200, 500
WAIT_VISIBLE_TIMEOUT = 30000
HEADER_CHANGE_TIMEOUT_MS = 18000

USE_PERSISTENT_CONTEXT = True
USER_DATA_DIR = "pw-us"
PROXY = None


def wait_random(lo=SLOW_MIN_MS, hi=SLOW_MAX_MS) -> None:
    time.sleep(random.uniform(lo / 1000.0, hi / 1000.0))


def log(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class CityCountRow:
    city: str
    href: str
    store_count: int | None
    header_text: str | None


# ---------- utils ----------


def accept_cookies(page: Page) -> None:
    for kw in ["Accept", "Agree", "Consent", "同意", "接受"]:
        try:
            btn = page.get_by_role("button", name=re.compile(kw, re.I))
            if btn.count():
                btn.first.click(timeout=1500)
                wait_random(120, 240)
                return
        except Exception:
            pass
    try:
        page.locator(
            "button:has-text('cookie'), button:has-text('Cookie')"
        ).first.click(timeout=1200)
    except Exception:
        pass


def disable_geolocation(page: Page) -> None:
    page.add_init_script(
        """
Object.defineProperty(navigator,'geolocation',{value:{
  getCurrentPosition:(s,e)=>e&&e({code:1,message:'denied'}),
  watchPosition:(s,e)=>{if(e)e({code:1,message:'denied'});return 0;}
}});
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
"""
    )


def get_city_panel(scope) -> Locator:
    panel = scope.locator(CITY_PANEL_SELECTOR)
    if panel.count():
        return panel
    for sel in [
        ".js-storelocator.st-main .st-main-content.st-grid .st-grid-col2 .st-frame-inner .st-frame-item",
        ".js-storelocator.st-main .st-main-content.st-grid .st-grid-col2 .st-frame-inner",
        ".js-storelocator.st-main .st-main-content.st-grid .st-grid-col2",
    ]:
        p = scope.locator(sel)
        if p.count():
            return p
    return panel


def get_city_links(scope) -> Locator:
    return get_city_panel(scope).locator(CITY_LINKS_SELECTOR)


def ensure_on_china(page: Page) -> None:
    # 强制停留在 /us/stores?country=china（官方目录页会重定向到这个查询页）
    if "country=china" not in page.url.lower():
        target = "https://www.lacoste.com/us/stores?country=china"
        try:
            page.goto(target, wait_until="domcontentloaded")
        except Exception:
            pass
    wait_random(120, 240)


def page_alive(context: BrowserContext, page: Page) -> Page:
    if not page or page.is_closed():
        p = context.new_page()
        disable_geolocation(p)
        return p
    return page


def wait_for_store_header(page: Page) -> bool:
    try:
        page.locator(STORE_BLOCK_SELECTOR).wait_for(
            state="attached", timeout=WAIT_VISIBLE_TIMEOUT
        )
    except Exception:
        return False
    hdr = page.locator(STORE_COUNT_SELECTOR)
    try:
        if hdr.count():
            hdr.first.wait_for(state="visible", timeout=WAIT_VISIBLE_TIMEOUT)
            return True
    except Exception:
        return False
    return False


def get_store_header_text(page: Page) -> str | None:
    try:
        hdr = page.locator(STORE_COUNT_SELECTOR).first
        if hdr.count():
            return hdr.inner_text(timeout=1000).strip()
    except Exception:
        return None
    return None


def wait_for_store_header_change(
    page: Page, prev_header_text: str, timeout_ms: int = HEADER_CHANGE_TIMEOUT_MS
) -> str | None:
    deadline = time.monotonic() + timeout_ms / 1000.0
    last_txt = None
    while time.monotonic() < deadline:
        txt = get_store_header_text(page)
        if txt and txt != prev_header_text and txt.upper() not in ("CHINA",):
            return txt
        last_txt = txt or last_txt
        time.sleep(0.25)
    return last_txt if last_txt and last_txt != prev_header_text else None


def parse_header_count_text(txt: str) -> int | None:
    if not txt:
        return None
    m = re.search(r"(\d+)\s+STORE", txt, re.I)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    if txt.strip().upper().startswith("LACOSTE"):
        return 1
    return None


def looks_like_city(text: str) -> bool:
    if not text:
        return False
    if len(text) > 40:
        return False
    if re.search(r"\d{2,}", text):
        return False
    if re.search(
        r"(Street|Road|Rd\.|Avenue|District|Province|市|区|县|大道|路|街)", text, re.I
    ):
        if text.endswith("市") and len(text) <= 6:
            return True
        return False
    return True


def canonicalize_city_href(href: str) -> str:
    """保证 city 链接带 country=china，返回绝对 URL。"""
    base = "https://www.lacoste.com"
    if not href:
        return urljoin(base, "/us/stores?country=china")
    absu = urljoin(base, href)
    u = urlparse(absu)
    qs = parse_qs(u.query)
    if "country" not in {k.lower() for k in qs}:
        qs["country"] = ["china"]
    new_q = urlencode({k: v[0] if isinstance(v, list) else v for k, v in qs.items()})
    return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))


def collect_cities_and_hrefs(page: Page) -> List[Tuple[str, str]]:
    # 让 DOM 稳定
    page.wait_for_load_state("domcontentloaded")
    accept_cookies(page)
    ensure_on_china(page)

    # 精确到: #page-search > ul.st-list.st-list-spacing > li.st-list-item > a
    panel = get_city_panel(page)
    items = panel.locator("li.st-list-item")
    total = items.count()

    cities: List[Tuple[str, str]] = []
    for i in range(total):
        li = items.nth(i)
        try:
            txt = (li.inner_text(timeout=800) or "").strip()
        except Exception:
            txt = ""
        if not looks_like_city(txt):
            continue
        href = None
        try:
            a = li.locator("a").first
            if a.count():
                href = a.get_attribute("href")
        except Exception:
            href = None
        if not href:
            continue
        cities.append((re.sub(r"\s+", " ", txt), canonicalize_city_href(href)))
    return cities


def safe_goto(page: Page, url: str) -> None:
    try:
        page.goto(url, wait_until="domcontentloaded")
    except Exception:
        # 二次尝试：短等待 + reload
        wait_random(150, 260)
        try:
            page.goto(url, wait_until="domcontentloaded")
        except Exception:
            pass


def main() -> None:
    out_csv = Path(COUNTS_CSV)
    out_xlsx = Path(COUNTS_XLSX)

    with sync_playwright() as p:
        launch_kwargs: Dict[str, Any] = {
            "channel": "chrome",
            "headless": False,
            "args": [
                "--start-maximized",
                "--disable-blink-features=AutomationControlled",
                "--no-first-run",
                "--no-default-browser-check",
            ],
        }
        if PROXY:
            launch_kwargs["proxy"] = PROXY

        if USE_PERSISTENT_CONTEXT:
            context = p.chromium.launch_persistent_context(
                user_data_dir=USER_DATA_DIR, slow_mo=100, **launch_kwargs
            )
        else:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context()
        context.set_default_timeout(15000)

        page = context.new_page()
        disable_geolocation(page)
        safe_goto(page, START_URL)
        accept_cookies(page)
        ensure_on_china(page)

        # 先把“城市 → href”映射抓好
        try:
            get_city_panel(page).wait_for(
                state="attached", timeout=WAIT_VISIBLE_TIMEOUT
            )
        except Exception:
            log("未找到左侧城市列表面板。")
            return

        pairs = collect_cities_and_hrefs(page)
        if not pairs:
            log("未收集到城市链接，可能仍需手动选择 China/同意 Cookie。")
            return

        log(f"检测到 {len(pairs)} 个城市入口（按 href 导航）。开始抓取 header…")
        results: List[CityCountRow] = []

        for city_name, href in pairs:
            page = page_alive(context, page)
            full_url = canonicalize_city_href(href)

            # 进入城市页（带 country=china）
            safe_goto(page, full_url)
            prev_header_text = get_store_header_text(page) or ""

            # 等待 header 出现/变化
            new_header_text = wait_for_store_header_change(page, prev_header_text)
            if not new_header_text:
                ok_hdr = wait_for_store_header(page)
                if not ok_hdr:
                    log(f"[跳过] {city_name}：未检测到 .st-header")
                    # 回到目录页
                    safe_goto(page, START_URL)
                    ensure_on_china(page)
                    continue
                new_header_text = get_store_header_text(page) or ""

            # 若被重定向到其它国家，强制回到 china
            if "country=china" not in page.url.lower():
                # 强制回跳并再读一次 header（避免读到美国门店标题如 CITADEL OUTLETS）
                safe_goto(page, full_url)
                new_header_text = get_store_header_text(page) or new_header_text

            count = parse_header_count_text(new_header_text)
            log(
                f"{city_name}: {new_header_text} -> {count if count is not None else 'N/A'}"
            )
            results.append(
                CityCountRow(
                    city=city_name,
                    href=full_url,
                    store_count=count,
                    header_text=new_header_text,
                )
            )

            # 回到目录页继续
            safe_goto(page, START_URL)
            ensure_on_china(page)
            wait_random()

        if not results:
            log("未收集到任何城市 header。")
            return

        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(out_csv, index=False, encoding="utf-8-sig")
        try:
            df.to_excel(out_xlsx, index=False)
        except Exception as e:
            log(f"保存 Excel 失败：{e}")

        log(f"完成：{len(df)} 个城市 →")
        log(f"- CSV:  {out_csv.resolve()}")
        log(f"- XLSX: {out_xlsx.resolve()}")


if __name__ == "__main__":
    main()
