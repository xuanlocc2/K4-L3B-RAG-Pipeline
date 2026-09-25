"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Hướng dẫn:
    1. Chọn chủ đề của nhóm.
    2. Tìm tối thiểu 3 tài liệu PDF/DOCX từ nguồn công khai.
    3. Lưu file gốc vào data/landing/legal/.
    4. Đặt tên không dấu và thể hiện đúng nội dung.

Ví dụ tài liệu: học phí, học bổng, ký túc xá, quy trình đăng ký.
Nếu website chặn crawler, hãy chọn nguồn công khai khác; không vượt WAF.
"""

from pathlib import Path

import requests


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

# Nguồn chính thức của Cổng Thông tin điện tử Chính phủ.  Các văn bản được
# chọn bao phủ đăng ký hộ kinh doanh, biểu mẫu/thuế và chính sách hỗ trợ khu
# vực kinh tế tư nhân có áp dụng cho hộ kinh doanh.
DOCUMENT_SOURCES = {
    "nghi-dinh-168-2025-dang-ky-doanh-nghiep.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/7/168nd.signed.pdf"
    ),
    "thong-tu-94-2025-thue-ho-kinh-doanh.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/10/94-btc.signed.pdf"
    ),
    "nghi-quyet-198-2025-phat-trien-kinh-te-tu-nhan.pdf": (
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/5/198_nq.pdf"
    ),
}


def setup_directory() -> None:
    """Tạo thư mục lưu tài liệu gốc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def download_documents() -> None:
    """Tải ít nhất 3 PDF/DOCX từ nguồn công khai."""
    setup_directory()
    headers = {"User-Agent": "Mozilla/5.0 (educational RAG lab; contact: local)"}
    for filename, url in DOCUMENT_SOURCES.items():
        output = DATA_DIR / filename
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        if not response.content.startswith(b"%PDF"):
            raise ValueError(f"Nguồn không trả về PDF hợp lệ: {url}")
        output.write_bytes(response.content)
        print(f"Saved: {output} ({len(response.content):,} bytes)")


if __name__ == "__main__":
    setup_directory()
    download_documents()
