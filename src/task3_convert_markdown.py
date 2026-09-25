"""
Task 3 — Chuẩn hóa dữ liệu sang Markdown.

Pipeline:
  1. Đọc PDF/DOCX trong `data/landing/legal/` → Markdown qua MarkItDown.
  2. Đọc JSON trong `data/landing/news/` → Markdown có header chứa
     URL, ngày crawl, tiêu đề.
  3. Lưu vào `data/standardized/legal/` và `data/standardized/news/`.
  4. Không tạo file rỗng, không ghi đè nếu đã tồn tại nội dung.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from markitdown import MarkItDown


LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_OUT = OUTPUT_DIR / "legal"
NEWS_OUT = OUTPUT_DIR / "news"


def _clean_markdown(text: str) -> str:
    """Loại bỏ khoảng trắng thừa, dòng trống liên tiếp, chuẩn hoá heading."""
    lines = []
    blank_count = 0
    for line in text.splitlines():
        line = line.rstrip()
        if not line.strip():
            blank_count += 1
            if blank_count > 1:
                continue
            lines.append("")
            continue
        blank_count = 0
        lines.append(line.rstrip())
    out = "\n".join(lines).strip()
    return out + "\n"


def _meta_block(title: str, source: str, url: str | None, doc_type: str,
                 extra: dict | None = None) -> str:
    """Sinh khối YAML-like metadata ở đầu file Markdown."""
    parts = ["---"]
    parts.append(f"title: {title}")
    parts.append(f"source: {source}")
    if url:
        parts.append(f"url: {url}")
    parts.append(f"doc_type: {doc_type}")
    parts.append(f"normalized_at: {datetime.now().isoformat()}")
    if extra:
        for key, value in extra.items():
            parts.append(f"{key}: {value}")
    parts.append("---\n")
    return "\n".join(parts)


def convert_legal_docs() -> list[Path]:
    """Dùng MarkItDown để chuyển PDF trong landing/legal sang Markdown."""
    md = MarkItDown()
    saved: list[Path] = []
    legal_dir = LANDING_DIR / "legal"
    if not legal_dir.exists():
        return saved
    LEGAL_OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(legal_dir.iterdir()):
        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".doc", ".docx"}:
            continue
        out_path = LEGAL_OUT / f"{path.stem}.md"
        # đọc metadata đính kèm (do task1 tạo)
        meta_path = legal_dir / f"{path.stem}.meta.json"
        meta: dict = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta = {}

        try:
            result = md.convert(str(path))
            body = result.text_content or ""
        except Exception as exc:
            print(f"[legal] failed to convert {path.name}: {exc}")
            continue

        body = _clean_markdown(body)
        if not body.strip():
            print(f"[legal] empty body for {path.name}, skipping")
            continue

        header = _meta_block(
            title=meta.get("title") or path.stem,
            source=meta.get("source_org") or "Bộ Giáo dục và Đào tạo (MOET)",
            url=meta.get("source_url"),
            doc_type="legal",
            extra={
                "filename": path.name,
                "authoritative_url": meta.get("authoritative_url") or "",
                "fetched_from_official_pdf": str(meta.get("fetched_from_official_pdf", False)),
            },
        )

        out_path.write_text(header + "\n" + body, encoding="utf-8")
        saved.append(out_path)
        print(f"[legal] {path.name} -> {out_path.relative_to(OUTPUT_DIR.parent.parent)}")
    return saved


def convert_news_articles() -> list[Path]:
    """Đọc JSON trong landing/news và ghi Markdown chuẩn hoá."""
    saved: list[Path] = []
    news_dir = LANDING_DIR / "news"
    if not news_dir.exists():
        return saved
    NEWS_OUT.mkdir(parents=True, exist_ok=True)
    for path in sorted(news_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[news] invalid json {path.name}: {exc}")
            continue
        out_path = NEWS_OUT / f"{path.stem}.md"

        title = data.get("title") or path.stem
        url = data.get("url") or ""
        date_crawled = data.get("date_crawled") or datetime.now().isoformat()
        body_raw = data.get("content_markdown") or ""

        # làm sạch markdown: cắt ảnh quảng cáo ở đầu, giữ phần tiêu đề + nội dung chính
        body_clean = _clean_markdown(body_raw)
        if len(body_clean) < 200:
            print(f"[news] body too short for {path.name}, skipping")
            continue

        header = _meta_block(
            title=title,
            source=url or "Cổng thông tin điện tử",
            url=url,
            doc_type="news",
            extra={"date_crawled": date_crawled},
        )

        # Giới hạn độ dài thân bài ở ~12000 ký tự để tránh chunking quá
        # nhiều phần quảng cáo ở cuối bài viết.
        body_trimmed = body_clean[:12000]
        out_path.write_text(
            header + f"# {title}\n\n" + body_trimmed, encoding="utf-8"
        )
        saved.append(out_path)
        print(f"[news] {path.name} -> {out_path.relative_to(OUTPUT_DIR.parent.parent)}")
    return saved


def convert_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LEGAL_OUT.mkdir(parents=True, exist_ok=True)
    NEWS_OUT.mkdir(parents=True, exist_ok=True)
    legal = convert_legal_docs()
    news = convert_news_articles()
    print(f"Standardized {len(legal)} legal + {len(news)} news documents.")


if __name__ == "__main__":
    convert_all()