"""
G-Star scraper scaffold

目标：抓取 https://www.g-star.com/en_us 下 Shop Men / Shop Women / Shop Kids
各品类（左侧 Category 中的子类，排除 "Shop All …"）的商品信息：图片、价格、链接。

说明：
- 站点较为动态，建议使用 Playwright 控制浏览器，从首页悬停顶部导航，采集各性别的分类入口，
  然后进入分类页读取商品卡片信息，并处理滚动/加载更多。
- 本脚本提供健壮的“候选选择器列表 + 回退策略”，你需要先用浏览器 DevTools 验证并在 TODO 处
  微调选择器，以确保准确抓取。

使用前准备：
- pip install playwright && playwright install chromium
- 也可改用 Chrome channel，如需登录或保留会话，可启用持久化上下文 USER_DATA_DIR。
"""

from __future__ import annotations

import re
import time
import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Dict, Any, Iterable, Tuple
from urllib.parse import urljoin

from playwright.sync_api import sync_playwright, Page, Locator, BrowserContext


START_URL = "https://www.g-star.com/en_us"
OUT_CSV = "gstar_products.csv"

# Playwright 启动参数
USE_PERSISTENT_CONTEXT = True
USER_DATA_DIR = "pw-us"  # 可复用浏览器会话（cookie 等）
HEADLESS = False
SLOW_MO_MS = 80
DEFAULT_TIMEOUT = 20000


def log(msg: str) -> None:
    print(msg, flush=True)


def wait(ms: int) -> None:
    time.sleep(ms / 1000)


def accept_cookies(page: Page) -> None:
    # 常见 cookie 文案尝试；若站点不同可在此扩展
    for kw in ["Accept", "Agree", "Consent", "I agree", "Allow all", "同意", "接受"]:
        try:
            btn = page.get_by_role("button", name=re.compile(kw, re.I))
            if btn.count():
                btn.first.click(timeout=1500)
                wait(200)
                return
        except Exception:
            pass
    # 常见选择器兜底
    for sel in [
        "button#onetrust-accept-btn-handler",
        "#onetrust-accept-btn-handler",
        "button:has-text('cookie'), button:has-text('Cookie')",
    ]:
        try:
            page.locator(sel).first.click(timeout=1200)
            wait(200)
            return
        except Exception:
            continue


# ========== 选择器策略（需在 DevTools 中确认/微调） ==========

# 顶部导航三大入口（通过可访问名称定位，避免类名易变）
TOP_TABS = {
    # 适配顶部导航或侧边栏中以 SHOP 开头的文案
    "Men": re.compile(r"^(shop\s+)?men$", re.I),
    "Women": re.compile(r"^(shop\s+)?women$", re.I),
    "Kids": re.compile(r"^(shop\s+)?kids$", re.I),
}

# Mega Menu 容器候选（不同站点/AB 方案类名可能变动）
MEGA_MENU_CANDIDATES = [
    # 顶部主导航
    "nav[aria-label*='Main'], nav[role='navigation']",
    "header",
    # 侧边栏（见截图中的 #sideNav / sideNav__branch）
    "#sideNav, nav#sideNav, [aria-label='Sidebar']",
]

# Mega Menu 中的类别链接候选选择器（需要过滤掉 "Shop All …"）
CATEGORY_LINK_CANDIDATES = [
    # 侧边栏分类链接优先
    "#sideNav a[href*='/en_us/shop/']",
    ".sideNav__branch a[href*='/en_us/']",
    # 兜底：站内链接
    "a[href*='/en_us/']:not([aria-hidden='true'])",
    "a:not([aria-hidden='true'])",
]

# 列表页商品网格/卡片候选
PRODUCT_GRID_CANDIDATES = [
    # 截图显示 data-testid=plp-grid
    "[data-testid='plp-grid']",
    "section[aria-label*='Products']",
    "div[class*='product-list'], ul[class*='product']",
    "main",
]

PRODUCT_CARD_CANDIDATES = [
    # 截图显示 data-testid=product-tile
    "[data-testid='product-tile']",
    "[data-qa='product-tile']",
    "li[class*='product'], article[class*='product'], div[class*='product']",
]

# 卡片内元素候选（名称/链接/价格/图片）
TITLE_CANDIDATES = [
    "[data-testid='product-title']",
    # 一些页面仅在链接的 aria-label 中提供标题
    "[data-testid='product-tile-link'][aria-label]",
    "h3, h2, .product-title, .product__title",
]

LINK_CANDIDATES = [
    # 截图显示 data-testid=product-tile-link
    "[data-testid='product-tile-link']",
    "a[href*='/en_us/']",
    "a[href]",
]

PRICE_CANDIDATES = [
    # 截图显示 data-testid=product-tile-price
    "[data-testid='product-tile-price']",
    "[data-testid='product-price']",
    "span[class*='price'], div[class*='price'], p[class*='price']",
]

IMAGE_CANDIDATES = [
    "img",
    "source",
]

# 过滤：排除 "Shop All …" 的正则
SHOP_ALL_RE = re.compile(r"^\s*shop\s+all\b", re.I)


@dataclass
class Product:
    gender: str
    category: str
    title: Optional[str]
    price_current: Optional[str]
    price_original: Optional[str]
    currency: Optional[str]
    product_url: Optional[str]
    image_url: Optional[str]


def find_first_existing(scope: Locator, selectors: Iterable[str]) -> Optional[Locator]:
    for sel in selectors:
        loc = scope.locator(sel)
        try:
            if loc.count() > 0:
                return loc
        except Exception:
            continue
    return None


def safe_text(loc: Optional[Locator]) -> Optional[str]:
    if not loc:
        return None
    try:
        txt = loc.inner_text(timeout=800)
        return (txt or "").strip()
    except Exception:
        return None


def safe_attr(loc: Optional[Locator], name: str) -> Optional[str]:
    if not loc:
        return None
    try:
        v = loc.get_attribute(name)
        return v.strip() if v else None
    except Exception:
        return None


def resolve_image_url(card: Locator) -> Optional[str]:
    # 优先 <img src/srcset>，再尝试 <source srcset>
    try:
        img = card.locator("img").first
        if img.count():
            for key in ("src", "data-src", "data-srcset", "srcset"):
                v = safe_attr(img, key)
                if v:
                    # 若是 srcset，取第一个 URL
                    if "," in v or " " in v:
                        return v.split(",")[0].strip().split(" ")[0].strip()
                    return v
    except Exception:
        pass
    try:
        src = card.locator("source").first
        if src.count():
            v = safe_attr(src, "srcset") or safe_attr(src, "data-srcset")
            if v:
                return v.split(",")[0].strip().split(" ")[0].strip()
    except Exception:
        pass
    return None


def get_currency_from_text(price_text: Optional[str]) -> Optional[str]:
    if not price_text:
        return None
    # 简单从符号推断；如需精确可在页面上读取 meta/定位 currency
    for sym, code in (("$", "USD"), ("€", "EUR"), ("£", "GBP")):
        if sym in price_text:
            return code
    return None


def collect_categories_from_mega_menu(page: Page, gender: str) -> List[Tuple[str, str]]:
    """从顶部导航悬停展开的 mega menu 收集分类名与链接。
    返回 (category_name, href)；过滤掉 "Shop All …" 项。
    注意：具体 DOM 需在 DevTools 中确认，如 mega menu 在 hover 才显示，需保持悬停状态。
    """
    # 优先尝试从侧边栏采集（若已渲染，会更稳定）
    side_pairs = collect_categories_from_side_nav(page, gender)
    if side_pairs:
        return side_pairs

    # 1) 悬停在对应性别 tab
    tab = page.get_by_role("link", name=TOP_TABS.get(gender, re.compile(gender, re.I)))
    if not tab.count():
        tab = page.get_by_role(
            "button", name=TOP_TABS.get(gender, re.compile(gender, re.I))
        )
    tab.first.hover()
    wait(300)

    # 2) 在多个候选容器里搜索链接
    found: List[Tuple[str, str]] = []
    container = None
    for root_sel in MEGA_MENU_CANDIDATES:
        c = page.locator(root_sel)
        try:
            if c.count():
                container = c
                # 直接从容器下收集所有 a，再由文本/路径筛选
                anchors = c.locator(", ".join(CATEGORY_LINK_CANDIDATES))
                total = anchors.count()
                for i in range(total):
                    a = anchors.nth(i)
                    name = safe_text(a) or ""
                    href = safe_attr(a, "href") or ""
                    if not href or not name:
                        continue
                    # 仅保留当前 gender 相关的链接
                    if not re.search(rf"/(men|women|kids)/", href, re.I):
                        continue
                    if not re.search(rf"/{gender.lower()}\b", href, re.I):
                        continue
                    # 过滤 Shop All …
                    if SHOP_ALL_RE.search(name):
                        continue
                    # 文案中若包含“Category”栏目标题等，可能不是最终分类，按需要过滤
                    if re.search(r"category|discover|new\s*in", name, re.I):
                        # 视站点实际决定是否跳过
                        pass
                    found.append((re.sub(r"\s+", " ", name), href))
        except Exception:
            continue

    # 去重
    uniq = {}
    for name, href in found:
        uniq[href] = name
    pairs = [(v, k) for k, v in uniq.items()]
    log(
        f"[{gender}] 发现候选分类：{len(pairs)} 条（含可能的非最终项，需 DevTools 复核）"
    )
    return pairs


def collect_categories_from_side_nav(page: Page, gender: str) -> List[Tuple[str, str]]:
    """从左侧 #sideNav 侧边栏抓取分类，匹配 /en_us/shop/{gender}/...，排除 Shop All。"""
    gender_lower = gender.lower()
    nav = page.locator("#sideNav, nav#sideNav, [aria-label='Sidebar']").first
    if not nav.count():
        return []

    # 展开 “Shop {Gender}” 分支按钮
    try:
        btn = nav.get_by_role(
            "button", name=re.compile(rf"shop\s+{gender_lower}", re.I)
        )
        if btn.count():
            expanded = (
                btn.first.get_attribute("aria-expanded") or ""
            ).lower() == "true"
            if not expanded:
                btn.first.click()
                wait(200)
    except Exception:
        pass
    # 就近取得该性别分支容器并尝试展开子叶
    branch_btn = None
    try:
        btn2 = nav.get_by_role("button", name=re.compile(rf"shop\s+{gender_lower}", re.I))
        if btn2.count():
            branch_btn = btn2.first
    except Exception:
        branch_btn = None
    branch = nav
    if branch_btn:
        try:
            b = branch_btn.locator("xpath=ancestor::*[contains(@class,'sideNav__branch')][1]")
            if b.count():
                branch = b
        except Exception:
            pass
    try:
        for _ in range(6):
            collapsed = branch.locator(".sideNav__leaf button[aria-expanded='false']")
            if not collapsed.count():
                break
            collapsed.first.click()
            wait(150)
    except Exception:
        pass

    # 放宽 href 匹配：支持 /en_us/shop/kids_boys 等
    anchors = branch.locator(f"a[href*='/en_us/shop/{gender_lower}']")
    if not anchors.count():
        anchors = branch.locator(".sideNav__leaf a[href*='/en_us/']")
    found: List[Tuple[str, str]] = []
    try:
        total = anchors.count()
    except Exception:
        total = 0
    for i in range(total):
        a = anchors.nth(i)
        href = safe_attr(a, "href") or ""
        if not href:
            continue
        # 仅保留当前 gender 的路径
        if not re.search(rf"/en_us/shop/{gender_lower}\\b", href, re.I):
            continue
        # 排除 /shop/{gender} 根路径及 Shop All
        if re.search(rf"/en_us/shop/{gender_lower}/?$", href, re.I):
            continue
        name = safe_attr(a, "data-title") or safe_text(a) or ""
        if not name or SHOP_ALL_RE.search(name):
            continue
        found.append((re.sub(r"\s+", " ", name), href))

    uniq = {}
    for name, href in found:
        uniq[href] = name
    pairs = [(v, k) for k, v in uniq.items()]
    if pairs:
        log(f"[{gender}] 从侧边栏获取分类：{len(pairs)} 条")
    return pairs


def locate_product_cards(page: Page) -> Tuple[Optional[Locator], Optional[Locator]]:
    grid = find_first_existing(page, PRODUCT_GRID_CANDIDATES)
    if not grid:
        grid = page
    cards = find_first_existing(grid, PRODUCT_CARD_CANDIDATES) or grid.locator("a")
    return grid, cards


def scroll_to_load_all(page: Page, max_loops: int = 20) -> None:
    prev_height = 0
    for i in range(max_loops):
        try:
            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        except Exception:
            break
        wait(600)
        try:
            cur_height = page.evaluate("document.body.scrollHeight")
        except Exception:
            break
        if cur_height == prev_height:
            # 尝试点击“加载更多”按钮
            clicked = False
            for kw in ["Load more", "Show more", "More", "加载更多"]:
                try:
                    btn = page.get_by_role("button", name=re.compile(kw, re.I))
                    if btn.count():
                        btn.first.click()
                        clicked = True
                        wait(800)
                        break
                except Exception:
                    pass
            if not clicked:
                break
        prev_height = cur_height


def extract_products_from_category(
    page: Page, gender: str, category_name: str
) -> List[Product]:
    accept_cookies(page)
    page.wait_for_load_state("domcontentloaded")
    scroll_to_load_all(page, max_loops=18)

    grid, cards = locate_product_cards(page)
    results: List[Product] = []
    total = 0
    try:
        total = cards.count()
    except Exception:
        total = 0

    for i in range(total):
        c = cards.nth(i)

        # 链接与标题
        a = find_first_existing(c, LINK_CANDIDATES) or c
        href = safe_attr(a, "href")
        if href and href.startswith("/"):
            href = urljoin(START_URL, href)
        title = safe_text(find_first_existing(c, TITLE_CANDIDATES))
        if not title:
            title = safe_attr(a, "aria-label")

        # 价格：处理促销（现价/原价）
        price_block = find_first_existing(c, PRICE_CANDIDATES) or c
        price_text = safe_text(price_block)
        price_cur = None
        price_org = None
        if price_text:
            # 常见格式：$99.00 $129.00 或 $99.00
            money = re.findall(r"[\$€£]\s?\d+[\d,]*(?:\.\d+)?", price_text)
            if money:
                price_cur = money[0]
                if len(money) > 1:
                    price_org = money[1]

        img_url = resolve_image_url(c)
        currency = get_currency_from_text(price_cur or price_org)

        # 过滤不完整项（必要时）
        if not href:
            continue

        results.append(
            Product(
                gender=gender,
                category=category_name,
                title=title,
                price_current=price_cur,
                price_original=price_org,
                currency=currency,
                product_url=href,
                image_url=img_url,
            )
        )

    log(f"[{gender} / {category_name}] 提取商品：{len(results)} 条")
    return results


def write_csv(rows: List[Product], outfile: str) -> None:
    if not rows:
        log("无数据可写入 CSV")
        return
    path = Path(outfile)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "gender",
                "category",
                "title",
                "price_current",
                "price_original",
                "currency",
                "product_url",
                "image_url",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    log(f"CSV 已保存：{path.resolve()}")


def ensure_context(p) -> BrowserContext:
    launch_kwargs: Dict[str, Any] = {
        "headless": HEADLESS,
        "slow_mo": SLOW_MO_MS,
        "channel": "chrome",  # 如无 Chrome，可去掉此行使用内置 Chromium
        "args": ["--start-maximized", "--disable-blink-features=AutomationControlled"],
    }
    if USE_PERSISTENT_CONTEXT:
        ctx = p.chromium.launch_persistent_context(
            user_data_dir=USER_DATA_DIR, **launch_kwargs
        )
    else:
        browser = p.chromium.launch(**launch_kwargs)
        ctx = browser.new_context()
    ctx.set_default_timeout(DEFAULT_TIMEOUT)
    return ctx


def main() -> None:
    all_rows: List[Product] = []
    with sync_playwright() as p:
        context = ensure_context(p)
        page = context.new_page()
        page.goto(START_URL, wait_until="domcontentloaded")
        accept_cookies(page)

        # 先在 DevTools 打开首页，悬停 Men/Women/Kids，确认 mega menu 中“类别链接”的位置。
        # 验证后可将 collect_categories_from_mega_menu 的过滤条件微调，确保捕获正确分类，
        # 并排除 "Shop All …"。
        genders = ["Men", "Women", "Kids"]
        for gender in genders:
            # 收集候选分类
            pairs = collect_categories_from_mega_menu(page, gender)
            if not pairs:
                log(
                    f"[{gender}] 未在 mega menu 捕获到分类链接，请在 DevTools 确认 DOM 并微调选择器。"
                )
                continue

            # 去重并按需裁剪（可在此进一步按 URL 结构仅保留 /c/{gender}/... 或 /shop/{gender}/...）
            # TODO：如站点实际分类落在某固定路径前缀，请在此正则过滤。

            # 依次进入分类页抓取商品
            for cat_name, href in pairs:
                try:
                    page.goto(href, wait_until="domcontentloaded")
                except Exception:
                    # 部分链接可能是促销或活动页，失败则跳过
                    continue
                accept_cookies(page)
                rows = extract_products_from_category(page, gender, cat_name)
                all_rows.extend(rows)

        write_csv(all_rows, OUT_CSV)


if __name__ == "__main__":
    main()
