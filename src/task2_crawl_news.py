"""
Task 2 — Crawl bài viết/thông báo.

Hướng dẫn:
    1. Điền tối thiểu 5 URL công khai vào ARTICLE_URLS.
    2. Crawl từng URL bằng Crawl4AI.
    3. Lưu mỗi bài thành một JSON trong data/landing/news/.
    4. Giữ đủ url, title, date_crawled và content_markdown.

Cài browser trước khi chạy:
    python -m playwright install chromium
    
-> Dùng Firecrawl or bất cứ công cụ nào bạn quen    
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import requests
from bs4 import BeautifulSoup


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

ARTICLE_URLS = [
    "https://xaydungchinhsach.chinhphu.vn/cam-nang-huong-dan-chung-ve-ho-kinh-doanh-119250812110959256.htm",
    "https://xaydungchinhsach.chinhphu.vn/quy-dinh-ho-so-trinh-tu-thu-tuc-dang-ky-thanh-lap-ho-kinh-doanh-119250702172916754.htm",
    "https://xaydungchinhsach.chinhphu.vn/luu-y-ve-dang-ky-dia-diem-kinh-doanh-va-dang-ky-su-dung-hoa-don-dien-tu-cua-ho-kinh-doanh-119251008173256643.htm",
    "https://xaydungchinhsach.chinhphu.vn/giai-dap-ve-chuyen-doi-ho-kinh-doanh-len-doanh-nghiep-119250813094416017.htm",
    "https://baochinhphu.vn/thuc-hien-nghi-dinh-70-2025-nd-cp-huong-toi-minh-bach-hoa-so-hoa-quan-ly-ho-kinh-doanh-102250529211550033.htm",
]


class _MarkdownExtractor(HTMLParser):
    """Chuyển phần HTML article đã chọn sang Markdown tối giản, dễ tái lập."""

    block_tags = {"p", "div", "section", "article", "br", "blockquote"}
    heading_tags = {"h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.heading_level: int | None = None
        self.in_li = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "img" and attributes.get("alt"):
            self.parts.append(f"\n\n![{attributes['alt']}]({attributes.get('src') or attributes.get('data-original') or ''})\n\n")
            return
        if tag in self.heading_tags:
            self.heading_level = int(tag[1])
        elif tag == "li":
            self.parts.append("\n- ")
            self.in_li = True
        elif tag in self.block_tags:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.heading_tags and self.heading_level:
            self.parts.append("\n\n")
            self.heading_level = None
        elif tag == "li":
            self.parts.append("\n")
            self.in_li = False

    def handle_data(self, data: str) -> None:
        text = " ".join(unescape(data).split())
        if text:
            self.parts.append(("#" * self.heading_level + " " if self.heading_level else "") + text)

    def markdown(self) -> str:
        text = "".join(self.parts)
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


async def crawl_article(url: str) -> dict:
    # requests chạy ổn định với các trang thông tin công khai này và tránh phụ
    # thuộc vào browser. Chạy trong thread để không chặn event loop.
    def fetch() -> str:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (educational RAG lab; contact: local)"},
            timeout=45,
        )
        response.raise_for_status()
        return response.text

    html = await asyncio.to_thread(fetch)
    title_match = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)', html, re.I)
    if not title_match:
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.I | re.S)
    title = unescape(title_match.group(1)).strip() if title_match else "Bài viết về hộ kinh doanh"

    # Các trang Chinhphu dùng article-body hoặc detail-content. Nếu giao diện
    # đổi, fallback về thẻ article rồi toàn trang để không tạo JSON rỗng.
    soup = BeautifulSoup(html, "html.parser")
    container = soup.select_one(".detail-content, .article-body, .detail__content, article")
    fragment = str(container) if container else html
    fragment = re.sub(r"<(script|style|nav|footer)[^>]*>.*?</\1>", "", fragment, flags=re.I | re.S)
    parser = _MarkdownExtractor()
    parser.feed(fragment)
    content = parser.markdown()
    if len(content) < 200:
        raise ValueError("Không trích xuất được đủ nội dung bài viết")
    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content,
    }


async def crawl_all() -> None:
    """Crawl và lưu từng bài thành một file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for index, url in enumerate(ARTICLE_URLS, 1):
        try:
            article = await crawl_article(url)
            output = DATA_DIR / f"article_{index:02d}.json"
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output}")
        except Exception as error:
            print(f"Failed: {url} — {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
