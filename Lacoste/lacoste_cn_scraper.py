# -*- coding: utf-8 -*-
"""
Lacoste China store city header scraper (FAST & safe)
- Target page: https://www.lacoste.com/us/stores/china
- Strategy: click each city on the LEFT list, read the LEFT column header (.st-header)
  to get store count / title, DO NOT parse .st-list (avoids hard navigations).
- Output: lacoste_cn_city_counts.csv / .xlsx

Notes:
- Uses persistent context to keep .com/us. If the site still geolocates you oddly,
  run once manually in that profile to choose "United States / English".
"""

from __future__ import annotations
import re, time, random, json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
from playwright.sync_api import sync_playwright, Page, Locator

START_URL = "https://www.lacoste.com/us/stores/china"
COUNTS_CSV = "lacoste_cn_city_counts.csv"
COUNTS_XLSX = "lacoste_cn_city_counts.xlsx"

# ====== LEFT column selectors (based on your DOM) ======
CITY_PANEL_SELECTOR = (
    ".js-storelocator.st-main .st-main-content.st-grid "
    ".st-grid-col2 .st-frame-inner .st-frame-item .st-list"
)
CITY_LINKS_SELECTOR = "li a, li button, li > a, a"

# After clicking a city, LEFT column becomes a "store block":
STORE_BLOCK_SELECTOR = (
    ".js-storelocator.st-main .st-main-content.st-grid "
    ".st-grid-col2 .st-frame .st-frame-inner .st-frame-item"
)
STORE_COUNT_SELECTOR = f"{STORE_BLOCK_SELECTOR} .st-header"  # header text (e.g., "8 STORES" or store title)

# ====== Tuning ======
SLOW_MIN_MS, SLOW_MAX_MS = 250, 650
CLICK_TIMEOUT = 4000
WAIT_VISIBLE_TIMEOUT = 50000
USE_PERSISTENT_CONTEXT = True
USER_DATA_DIR = "pw-us"  # change to "pw-us-2" if you want a fresh profile
PROXY = None  # e.g. {"server": "http://127.0.0.1:7890"}


def wait_random(lo=SLOW_MIN_MS, hi=SLOW_MAX_MS):
    time.sleep(random.uniform(lo / 1000.0, hi / 1000.0))


def log(msg: str):
    print(msg, flush=True)


@dataclass
class CityCountRow:
    city: str
    store_count: int | None
    header_text: str | None


def accept_cookies(page: Page):
    for kw in ["Accept", "Agree", "Consent", "同意", "接受"]:
        try:
            btn = page.get_by_role("button", name=re.compile(kw, re.I))
            if btn.count():
                btn.first.click(timeout=1500)
                wait_random(150, 300)
                return
        except:
            pass
    try:
        page.locator(
            "button:has-text('cookie'), button:has-text('Cookie')"
        ).first.click(timeout=1200)
    except:
        pass


def disable_geolocation(page: Page):
    page.add_init_script(
        """
Object.defineProperty(navigator,'geolocation',{value:{
  getCurrentPosition:(s,e)=>e&&e({code:1,message:'denied'}),
  watchPosition:(s,e)=>{if(e)e({code:1,message:'denied'});return 0;}
}});
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
"""
    )


def prevent_hard_nav_in_left_panel(page: Page):
    # Stop <a> default navigation inside left panel; keep SPA handlers only.
    selector_js = json.dumps(CITY_PANEL_SELECTOR)
    script = (
        """
(function(){
  const PANEL_SELECTOR = %s;
  document.addEventListener('click', function(e){
    const a = e.target && e.target.closest && e.target.closest('a');
    if(!a) return;
    const panel = document.querySelector(PANEL_SELECTOR);
    if(panel && panel.contains(a)){
      const href = a.getAttribute('href') || '';
      if (/^https?:/i.test(href) || href.includes('Lacoste_')){
        e.preventDefault(); e.stopPropagation();
      }
    }
  }, true);
})();
"""
        % selector_js
    )
    page.add_init_script(script)


def get_city_panel(scope) -> Locator:
    panel = scope.locator(CITY_PANEL_SELECTOR)
    if panel.count():
        return panel
    # broader fallbacks if needed
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


def ensure_on_china(page: Page):
    for label in ["China", "中国"]:
        try:
            page.get_by_text(label, exact=True).first.click(timeout=1200)
            wait_random(150, 300)
            return
        except:
            pass


def wait_for_store_header(page: Page) -> bool:
    try:
        page.locator(STORE_BLOCK_SELECTOR).wait_for(
            state="attached", timeout=WAIT_VISIBLE_TIMEOUT
        )
    except:
        return False
    hdr = page.locator(STORE_COUNT_SELECTOR)
    try:
        if hdr.count():
            hdr.first.wait_for(state="visible", timeout=WAIT_VISIBLE_TIMEOUT)
            return True
    except:
        return False
    return False


def get_store_count_text(page: Page) -> str | None:
    try:
        hdr = page.locator(STORE_COUNT_SELECTOR).first
        if hdr.count():
            return hdr.inner_text(timeout=800).strip()
    except:
        return None
    return None


def parse_header_count_text(txt: str) -> int | None:
    if not txt:
        return None
    m = re.search(r"(\d+)\s+STORE", txt, re.I)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    # If header is a single store title like "LACOSTE BEIJING ...", treat as 1
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


def go_to_city_directory(page: Page):
    try:
        page.goto(START_URL, wait_until="domcontentloaded")
    except:
        try:
            page.reload(wait_until="domcontentloaded")
        except:
            pass
    accept_cookies(page)
    ensure_on_china(page)
    wait_random(150, 300)


def click_city_by_index(page: Page, index: int) -> str | None:
    panel = get_city_panel(page)
    lis = panel.locator("li")
    if index >= lis.count():
        return None
    li = lis.nth(index)
    try:
        li.scroll_into_view_if_needed(timeout=2500)
    except:
        pass
    try:
        city_text = li.inner_text().strip()
    except:
        city_text = ""

    # Neutralize the first <a> to avoid hard nav, then dispatch click
    try:
        a = li.locator("a").first
        if a.count():
            a.evaluate(
                "el => { if (el) { el.setAttribute('href','#'); el.removeAttribute('target'); } }"
            )
            a.dispatch_event("click")
        else:
            li.dispatch_event("click")
        return city_text
    except:
        try:
            li.click(timeout=CLICK_TIMEOUT, force=True)
            return city_text
        except:
            return None


def main():
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
                user_data_dir=USER_DATA_DIR, slow_mo=120, **launch_kwargs
            )
        else:
            browser = p.chromium.launch(**launch_kwargs)
            context = browser.new_context()
        context.set_default_timeout(15000)
        page = context.new_page()

        prevent_hard_nav_in_left_panel(page)
        disable_geolocation(page)
        go_to_city_directory(page)

        links = get_city_links(page)
        try:
            get_city_panel(page).wait_for(
                state="attached", timeout=WAIT_VISIBLE_TIMEOUT
            )
        except:
            log("未找到左侧城市列表面板。")
            return

        total_entries = links.count()
        if total_entries == 0:
            log("左侧列表为空，可能仍需手动选择 China/同意 Cookie。")
            return

        # Build candidate (index, city_name)
        candidates: List[tuple[int, str]] = []
        for i in range(total_entries):
            try:
                t = links.nth(i).inner_text().strip()
            except:
                t = ""
            if looks_like_city(t):
                candidates.append((i, re.sub(r"\s+", " ", t)))

        log(f"检测到 {len(candidates)} 个城市入口。开始仅抓取 header（门店数量/标题）…")
        results: List[CityCountRow] = []

        for idx, city_name in candidates:
            go_to_city_directory(page)  # reset to city list
            links = get_city_links(page)
            if idx >= links.count():
                log(f"[跳过] 索引 {idx} 超出范围。")
                continue

            log(f"点击城市：{city_name} (index={idx})")
            clicked = click_city_by_index(page, idx)
            if not clicked:
                log(f"  - 跳过：无法点击 {city_name}")
                continue

            ok_hdr = wait_for_store_header(page)
            if not ok_hdr:
                log("  - 未检测到 .st-header，跳过")
                continue

            header_text = get_store_count_text(page) or ""
            count = parse_header_count_text(header_text)
            log(
                f"  - 门店数量提示: {header_text} -> 计数: {count if count is not None else 'N/A'}"
            )
            results.append(
                CityCountRow(city=city_name, store_count=count, header_text=header_text)
            )
            wait_random()

        if not results:
            log("未收集到任何城市 header。请把左列 DOM 截图发我以便再调。")
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
