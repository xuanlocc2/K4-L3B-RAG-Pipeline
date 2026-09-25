"""Build / re-build ChromaDB index + corpus cache từ Markdown chuẩn hoá.

Idempotent: chạy nhiều lần không tạo duplicate chunks. In báo cáo indexing
thật từ `verify_index()` để xác nhận pipeline nối đúng.

Usage:
    python scripts/build_index.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.task4_chunking_indexing import (
    print_index_report,
    run_pipeline,
    verify_index,
)


def main() -> None:
    print("Building RAG index from Markdown corpus...")
    summary = run_pipeline()
    print(f"\nBuild summary: {summary}\n")

    print("Post-build verification...")
    verify_summary = verify_index()
    print_index_report(verify_summary)


if __name__ == "__main__":
    main()
