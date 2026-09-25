"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Hướng dẫn:
    1. Dùng MarkItDown để convert PDF/DOCX.
    2. Đọc JSON và giữ metadata ở đầu file Markdown.
    3. Giữ cấu trúc thư mục legal/ và news/.
    4. Không tạo file rỗng hoặc file trùng khi chạy lại.

Cài đặt:
    Dependency MarkItDown đã được khai báo trong pyproject.toml.
    
-> Hoặc dùng công cụ nào bạn quen khác Markitdown
"""

from pathlib import Path

import json

from markitdown import MarkItDown
import requests
from bs4 import BeautifulSoup


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Ba PDF chính thức đều là bản ký số dạng ảnh. MarkItDown không lấy được text
# layer từ chúng, nên dùng bản toàn văn trên cùng hệ thống Chinhphu.vn làm
# fallback có kiểm soát. File đầu vào vẫn chính là PDF được lưu ở landing.
LEGAL_TEXT_FALLBACKS = {
    "nghi-dinh-168-2025-dang-ky-doanh-nghiep.pdf": "https://xaydungchinhsach.chinhphu.vn/quy-dinh-ho-so-trinh-tu-thu-tuc-dang-ky-thanh-lap-ho-kinh-doanh-119250702172916754.htm",
    "thong-tu-94-2025-thue-ho-kinh-doanh.pdf": "https://vanban.chinhphu.vn/?classid=1&docid=215644&orggroupid=4&pageid=27160",
    "nghi-quyet-198-2025-phat-trien-kinh-te-tu-nhan.pdf": "https://xaydungchinhsach.chinhphu.vn/nghi-quyet-198-2025-qh15-ve-mot-so-co-che-chinh-sach-dac-biet-phat-trien-kinh-te-tu-nhan-119250517191622422.htm",
}


def _extract_page_text(url: str) -> str:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (educational RAG lab)"}, timeout=45)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    container = soup.select_one(".detail-content, .article-body, .detail__content, article")
    text = (container or soup).get_text("\n", strip=True)
    return "\n\n".join(line for line in text.splitlines() if line.strip())


def convert_legal_docs() -> None:
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    converter = MarkItDown()
    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        text = converter.convert(str(path)).text_content.strip()
        if not text:
            fallback_url = LEGAL_TEXT_FALLBACKS.get(path.name)
            if not fallback_url:
                raise ValueError(f"Không thể chuyển đổi hoặc tài liệu rỗng: {path}")
            text = _extract_page_text(fallback_url)
        if len(text) < 200:
            raise ValueError(f"Nội dung chuẩn hóa quá ngắn: {path}")
        header = (
            f"# {path.stem.replace('-', ' ').title()}\n\n"
            f"**Source file:** {path.name}\n\n"
            f"**Official text:** {LEGAL_TEXT_FALLBACKS.get(path.name, 'N/A')}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(header + text + "\n", encoding="utf-8")


def convert_news_articles() -> None:
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    required = {"url", "title", "date_crawled", "content_markdown"}
    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        missing = required - data.keys()
        if missing or not all(str(data[key]).strip() for key in required):
            raise ValueError(f"JSON thiếu metadata/nội dung: {path} ({sorted(missing)})")
        header = (
            f"# {data['title']}\n\n**Source:** {data['url']}\n\n"
            f"**Crawled:** {data['date_crawled']}\n\n---\n\n"
        )
        (output_dir / f"{path.stem}.md").write_text(
            header + data["content_markdown"].strip() + "\n", encoding="utf-8"
        )


def convert_all() -> None:
    """Convert toàn bộ dữ liệu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
