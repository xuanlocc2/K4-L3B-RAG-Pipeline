"""
Task 8 — PageIndex vectorless fallback.

Hướng dẫn:
    1. Đọc PAGEINDEX_API_KEY từ .env.
    2. Upload tài liệu ở định dạng PageIndex hỗ trợ.
    3. Cache document IDs để không upload lại.
    4. Parse kết quả thành SearchResult có method pageindex.

PageIndex là dịch vụ ngoài: cần timeout và xử lý lỗi để pipeline không crash.
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
CACHE_PATH = Path(__file__).parent.parent / "pageindex_doc_ids.json"
REQUEST_TIMEOUT_SECONDS = 30
PROCESSING_TIMEOUT_SECONDS = 90


def _client():
    if not PAGEINDEX_API_KEY:
        return None
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def _load_cache() -> dict:
    if not CACHE_PATH.exists():
        return {}
    try:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _call_with_timeout(function, *args):
    """Bound an SDK call because the current SDK exposes no timeout argument."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(function, *args)
    try:
        return future.result(timeout=REQUEST_TIMEOUT_SECONDS)
    except TimeoutError as error:
        future.cancel()
        raise TimeoutError("PageIndex request timed out") from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def upload_documents() -> None:
    """Upload tài liệu và lưu document IDs để tái sử dụng."""
    client = _client()
    if client is None:
        print("PAGEINDEX_API_KEY is not configured; skipping PageIndex upload.")
        return

    cache = _load_cache()
    for path in sorted(LEGAL_DIR.glob("*.pdf")):
        fingerprint = f"{path.stat().st_size}:{path.stat().st_mtime_ns}"
        cached = cache.get(path.name, {})
        if cached.get("fingerprint") == fingerprint and cached.get("doc_id"):
            continue
        response = _call_with_timeout(client.submit_document, str(path))
        doc_id = response.get("doc_id")
        if not isinstance(doc_id, str) or not doc_id:
            raise RuntimeError(f"PageIndex returned no doc_id for {path.name}")
        cache[path.name] = {"doc_id": doc_id, "fingerprint": fingerprint}
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Uploaded {path.name}: {doc_id}")


def _extract_nodes(payload: object) -> list[dict]:
    """Tolerate documented and beta response shapes without fabricating text."""
    if not isinstance(payload, dict):
        return []
    candidates = payload.get("nodes") or payload.get("results") or payload.get("retrieved_nodes")
    if candidates is None and isinstance(payload.get("result"), dict):
        candidates = payload["result"].get("nodes") or payload["result"].get("results")
    return [node for node in candidates if isinstance(node, dict)] if isinstance(candidates, list) else []


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Trả về pageindex SearchResult."""
    if top_k <= 0 or not query.strip():
        return []
    client = _client()
    if client is None:
        return []

    cache = _load_cache()
    if not cache:
        upload_documents()
        cache = _load_cache()
    results: list[dict] = []
    deadline = time.monotonic() + PROCESSING_TIMEOUT_SECONDS
    for filename, entry in cache.items():
        doc_id = entry.get("doc_id")
        if not isinstance(doc_id, str):
            continue
        # A document may still be OCR/tree-processing right after upload.
        while time.monotonic() < deadline and not _call_with_timeout(client.is_retrieval_ready, doc_id):
            time.sleep(2)
        if time.monotonic() >= deadline:
            continue
        submitted = _call_with_timeout(client.submit_query, doc_id, query)
        retrieval_id = submitted.get("retrieval_id")
        if not retrieval_id:
            continue
        payload = _call_with_timeout(client.get_retrieval, retrieval_id)
        for position, node in enumerate(_extract_nodes(payload), start=1):
            content = node.get("content") or node.get("text") or node.get("markdown")
            if not isinstance(content, str) or not content.strip():
                continue
            node_id = str(node.get("id") or node.get("node_id") or position)
            results.append({
                "id": f"pageindex:{doc_id}:{node_id}",
                "content": content.strip(),
                "score": float(node.get("score", 1.0 / position)),
                "metadata": {
                    "source": filename,
                    "title": Path(filename).stem.replace("-", " "),
                    "doc_type": "legal",
                    "url": None,
                    "chunk_index": position - 1,
                },
                "retrieval_method": "pageindex",
            })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    upload_documents()
