"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
       (Có thể thay bằng Weighted RRF thông qua ``retrieval_mode='weighted_rrf'``).
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Threshold SCORE_THRESHOLD được hiệu chỉnh trên tập validation gồm câu
in-domain và out-of-domain (xem `reports/threshold_calibration.md`).

Các chế độ (Bonus 1 + Bonus 2):
    ``baseline``          : dense + BM25 + RRF chuẩn (mặc định).
    ``dense_only``        : chỉ dense (dùng để benchmark).
    ``bm25_only``         : chỉ BM25 (dùng để benchmark).
    ``weighted_rrf``      : Bonus 1 - RRF có trọng số (dense_weight, bm25_weight).
    ``weighted_rrf_dedup``: Bonus 1 + Bonus 2 - Weighted RRF + document dedup.
    ``rrf_dedup``         : RRF chuẩn + document dedup (chỉ Bonus 2).
Các chế độ dedup sẽ lấy top-k lớn hơn từ fusion rồi dedup xuống top_k để
không mất chunk do chính sách lọc quá sớm.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import (
    deduplicate_by_document,
    rerank_rrf,
    rerank_weighted_rrf,
)
from .task8_pageindex_vectorless import pageindex_search


load_dotenv()

# Threshold được hiệu chỉnh trên tập validation (xem reports/threshold_calibration.md).
# Calibration 2026-09-25: in-domain min = 0.565, OOD max = 0.494.
# Chọn 0.50 để phân tách sạch giữa hai nhóm và cho phép fallback OOD kích hoạt.
# Có thể override qua env SCORE_THRESHOLD.
DEFAULT_SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.50"))
DEFAULT_TOP_K = 5

# Bonus 1: trọng số mặc định cho Weighted RRF.
# Lấy từ env để dễ cấu hình mà không cần sửa code.
# Grid search trên golden_dataset (xem scripts/weighted_rrf_grid.py) cho
# thấy dense_weight >= bm25_weight ổn định hơn; mặc định 1.5/1.0 được
# chọn để đề cao dense một chút mà vẫn dùng BM25 làm fallback cho truy vấn
# lexical-heavy.
DEFAULT_DENSE_WEIGHT = float(os.getenv("DENSE_WEIGHT", "1.5"))
DEFAULT_BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "1.0"))

# Có thể phá vỡ thế cân bằng theo retrieval_method khi dedup.
DEFAULT_DEDUP_PREFER = ["hybrid_weighted", "hybrid", "dense", "bm25", "pageindex"]

VALID_RETRIEVAL_MODES = {
    "baseline",
    "dense_only",
    "bm25_only",
    "weighted_rrf",
    "weighted_rrf_dedup",
    "rrf_dedup",
}


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    use_reranking: bool = True,
    *,
    retrieval_mode: str = "baseline",
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    dedup_top_k_factor: int = 2,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult.

    ``retrieval_mode`` chọn chiến lược fusion / dedup (xem docstring module).
    Chế độ mặc định ``baseline`` hành xử y hệt phiên bản gốc (RRF chuẩn,
    không dedup) để giữ tương thích với tests + evaluation cũ.
    """
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError(
            f"retrieval_mode phải thuộc {sorted(VALID_RETRIEVAL_MODES)}, got {retrieval_mode!r}"
        )
    if not query.strip():
        return []

    # Các chế độ đặc biệt: dedup cần nhiều candidate hơn top_k để còn
    # chunk sau khi gộp theo document.
    fetch_k = top_k * max(1, dedup_top_k_factor)
    dense = semantic_search(query, top_k=fetch_k)
    sparse = lexical_search(query, top_k=fetch_k)

    if retrieval_mode == "dense_only":
        hybrid = dense[:top_k]
    elif retrieval_mode == "bm25_only":
        hybrid = sparse[:top_k]
    elif retrieval_mode == "weighted_rrf":
        hybrid = rerank_weighted_rrf(
            [dense, sparse],
            top_k=top_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
    elif retrieval_mode == "weighted_rrf_dedup":
        fused = rerank_weighted_rrf(
            [dense, sparse],
            top_k=fetch_k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )
        hybrid = deduplicate_by_document(
            fused, top_k=top_k, prefer_method=DEFAULT_DEDUP_PREFER
        )
    elif retrieval_mode == "rrf_dedup":
        fused = rerank_rrf([dense, sparse], top_k=fetch_k)
        hybrid = deduplicate_by_document(
            fused, top_k=top_k, prefer_method=DEFAULT_DEDUP_PREFER
        )
    elif retrieval_mode == "baseline":
        if use_reranking:
            hybrid = rerank_rrf([dense, sparse], top_k=top_k)
        else:
            hybrid = dense[:top_k]
    else:  # pragma: no cover - đã validate ở đầu hàm
        hybrid = []

    # Fallback dùng dense score gốc (chuẩn hoá cosine similarity) - giữ
    # contract: so sánh threshold với cosine score gốc, không phải RRF score.
    best_dense_score = dense[0]["score"] if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            # không bao giờ để provider lỗi làm vỡ pipeline
            pass
    return hybrid


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    for result in retrieve("điều kiện xét tuyển đại học", top_k=3):
        print(result["id"], round(result["score"], 3), result["retrieval_method"])
