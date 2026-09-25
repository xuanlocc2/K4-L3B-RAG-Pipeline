"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

Triển khai thực tế:
    - PageIndex là dịch vụ ngoài; nếu thiếu key hoặc lỗi mạng, hàm
      `pageindex_search` trả về danh sách rỗng. Pipeline ở Task 9 sẽ
      xử lý an toàn, không bao giờ để UI crash.
    - Khi key được cấp, module sẽ thử upload các file PDF trong
      `data/landing/legal` và cache doc_id theo filename.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DOC_ID_CACHE = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"


_client: Any | None = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not PAGEINDEX_API_KEY:
        return None
    try:
        from pageindex import PageIndex

        _client = PageIndex(api_key=PAGEINDEX_API_KEY)
        return _client
    except Exception:
        return None


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _get_client()
    if client is None:
        # Không có key hoặc SDK không khả dụng — lưu cache rỗng để cache tồn tại.
        DOC_ID_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DOC_ID_CACHE.write_text(json.dumps({"doc_ids": {}}, indent=2), encoding="utf-8")
        return

    cache: dict[str, Any] = {"doc_ids": {}}
    if DOC_ID_CACHE.exists():
        try:
            cache = json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
        except Exception:
            cache = {"doc_ids": {}}

    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        if path.name in cache["doc_ids"]:
            continue
        try:
            result = client.upload(path=str(path))
            doc_id = None
            if isinstance(result, dict):
                doc_id = result.get("id") or result.get("doc_id")
            else:
                doc_id = getattr(result, "id", None) or getattr(result, "doc_id", None)
            if doc_id:
                cache["doc_ids"][path.name] = doc_id
        except Exception:
            # không push lỗi ra ngoài — fallback an toàn
            continue

    DOC_ID_CACHE.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _load_cache() -> dict[str, Any]:
    if not DOC_ID_CACHE.exists():
        return {"doc_ids": {}}
    try:
        return json.loads(DOC_ID_CACHE.read_text(encoding="utf-8"))
    except Exception:
        return {"doc_ids": {}}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult. Trả [] khi thiếu key/SDK lỗi."""
    client = _get_client()
    cache = _load_cache()
    if client is None or not cache["doc_ids"]:
        return []
    try:
        results: list[dict] = []
        for doc_id in list(cache["doc_ids"].values())[:top_k * 2]:
            response = client.search(doc_id=doc_id, query=query, top_k=top_k)
            nodes = response.get("nodes", []) if isinstance(response, dict) else []
            for rank, node in enumerate(nodes, 1):
                content = node.get("content") or node.get("text") or ""
                if not content:
                    continue
                results.append(
                    {
                        "id": f"{doc_id}::{node.get('node_id') or rank}",
                        "content": content,
                        "score": float(node.get("score") or 1.0 / rank),
                        "metadata": {
                            "source": f"pageindex:{doc_id}",
                            "title": node.get("title") or f"PageIndex {doc_id}",
                            "doc_type": "legal",
                            "url": None,
                            "chunk_index": rank - 1,
                        },
                        "retrieval_method": "pageindex",
                    }
                )
                if len(results) >= top_k:
                    break
            if len(results) >= top_k:
                break
        return results
    except Exception:
        return []


if __name__ == "__main__":
    upload_documents()