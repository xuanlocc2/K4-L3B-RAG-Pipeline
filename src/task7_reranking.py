"""
Task 7 — Reciprocal Rank Fusion.

RRF gộp nhiều bảng xếp hạng mà không cộng trực tiếp cosine score với BM25
score. Công thức: RRF(d) = sum(1 / (k + rank)), rank bắt đầu từ 1.

Lưu ý: RRF score chỉ phản ánh thứ hạng, không dùng để quyết định fallback.

<<<<<<< Updated upstream
-> Dùng Jina hoặc self host hoặc bất cứ công cụ nào bạn quen
"""

=======
Lưu ý quan trọng:
    RRF score chỉ phản ánh thứ hạng, KHÔNG dùng để quyết định fallback
    (Task 9 dùng cosine score gốc của dense result).
    RRF chỉ được áp dụng MỘT LẦN ở giai đoạn fusion cuối cùng của pipeline.

Score semantics:
    Mỗi kết quả hybrid giữ:
        - ``score``       : điểm RRF dùng để ranking (KHÔNG phải confidence).
        - ``rrf_score``   : giá trị RRF gốc (giống ``score`` sau fusion).
        - ``dense_score`` : cosine similarity gốc nếu chunk có trong dense list.
        - ``bm25_score``  : BM25 score gốc nếu chunk có trong BM25 list.
    Các trường optional giúp evidence_quality.py tín hiểu từng signal riêng
    biệt thay vì chỉ dựa vào RRF.

Weighted RRF (Bonus 1):
    score(d) = sum_i w_i * (1 / (k + rank_i))
    với mỗi hệ trọng số w_i được config qua ``weights``:
        - dense_weight  (mặc định 1.0)
        - bm25_weight   (mặc định 1.0)
    Hai bảng xếp hạng trong ``ranked_lists`` được ánh xạ theo thứ tự:
    vị trí 0 -> dense_weight, vị trí 1 -> bm25_weight.
    Nếu số bảng xếp hạng nhiều hơn, các vị trí còn lại giữ trọng số 1.0.
"""

from __future__ import annotations

from typing import Any


def _carry_original_score(item: dict, attr: str) -> float | None:
    """Đọc optional score (``dense_score``/``bm25_score``) nếu có."""
    value = item.get(attr)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _fuse(
    ranked_lists: list[list[dict]],
    top_k: int,
    k: int,
    weights: list[float] | None,
) -> list[dict]:
    """Shared fusion core cho RRF / Weighted RRF.

    Trả về list result với ``score = rrf_score`` và các original scores
    (``dense_score``/``bm25_score``) được gắn kèm nếu chunk có mặt trong
    list tương ứng. Chỉ số nào vắng mặt trong dense/bm25 sẽ không có
    original score cho signal đó.
    """
    if not ranked_lists:
        return []
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists "
            f"({len(ranked_lists)})"
        )
    for w in weights:
        if w < 0:
            raise ValueError("weights must be non-negative")

    # Track original scores per chunk để fusion giữ được signal gốc.
    dense_per_id: dict[str, float] = {}
    bm25_per_id: dict[str, float] = {}
    rrf_scores: dict[str, float] = {}
    items: dict[str, dict] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            rrf_scores[item_id] = rrf_scores.get(item_id, 0.0) + (
                weight * 1.0 / (k + rank)
            )
            items[item_id] = item
            dense_score = _carry_original_score(item, "dense_score")
            if dense_score is not None:
                dense_per_id[item_id] = max(
                    dense_per_id.get(item_id, dense_score), dense_score
                )
            bm25_score = _carry_original_score(item, "bm25_score")
            if bm25_score is not None:
                bm25_per_id[item_id] = max(
                    bm25_per_id.get(item_id, bm25_score), bm25_score
                )

    ranked_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        base: dict[str, Any] = dict(items[item_id])
        rrf_value = float(rrf_scores[item_id])
        base["score"] = rrf_value
        base["rrf_score"] = rrf_value
        if item_id in dense_per_id:
            base["dense_score"] = dense_per_id[item_id]
        if item_id in bm25_per_id:
            base["bm25_score"] = bm25_per_id[item_id]
        results.append(base)
    return results

>>>>>>> Stashed changes

def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
<<<<<<< Updated upstream
    """Fuse nhiều ranked lists và trả hybrid SearchResult."""
    # TODO: Implement RRF.
    #
    # scores = {}
    # items = {}
    # for ranked_list in ranked_lists:
    #     for rank, item in enumerate(ranked_list, 1):
    #         item_id = item["id"]
    #         scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
    #         items[item_id] = item
    #
    # ranked_ids = sorted(scores, key=scores.get, reverse=True)
    # results = []
    # for item_id in ranked_ids[:top_k]:
    #     result = items[item_id].copy()
    #     result["score"] = scores[item_id]
    #     result["retrieval_method"] = "hybrid"
    #     results.append(result)
    # return results
    raise NotImplementedError("Implement rerank_rrf")
=======
    """Fuse nhiều ranked lists bằng RRF chuẩn.

    ``score`` (và ``rrf_score``) trên kết quả là RRF gốc — KHÔNG dùng làm
    confidence. ``dense_score``/``bm25_score`` được preserve từ input list
    tương ứng.
    """
    results = _fuse(ranked_lists, top_k=top_k, k=k, weights=None)
    for item in results:
        item["retrieval_method"] = "hybrid"
    return results


def rerank_weighted_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
    *,
    weights: list[float] | None = None,
    dense_weight: float = 1.0,
    bm25_weight: float = 1.0,
) -> list[dict]:
    """Fuse nhiều ranked lists bằng Weighted RRF.

    Hỗ trợ hai cách truyền trọng số:
      * ``weights``: danh sách trọng số theo từng ranked_list.
      * ``dense_weight`` / ``bm25_weight``: shorthand cho 2 danh sách phổ biến
        (vị trí 0 -> dense, vị trí 1 -> bm25). Các vị trí còn lại mặc định 1.0.

    Nếu weights là ``None`` thì dùng ``[dense_weight, bm25_weight]`` cho 2
    danh sách đầu, còn lại 1.0.
    """
    if weights is None:
        weights = []
        for idx in range(len(ranked_lists)):
            if idx == 0:
                weights.append(dense_weight)
            elif idx == 1:
                weights.append(bm25_weight)
            else:
                weights.append(1.0)
    results = _fuse(ranked_lists, top_k=top_k, k=k, weights=weights)
    for item in results:
        item["retrieval_method"] = "hybrid_weighted"
    return results


def deduplicate_by_document(
    results: list[dict],
    *,
    top_k: int | None = None,
    prefer_method: list[str] | None = None,
) -> list[dict]:
    """Giảm thiểu nhiều chunk từ cùng một document trong top-K.

    Tham số:
        results: danh sách SearchResult đã được sort theo score giảm dần.
        top_k: nếu None thì giữ nguyên số lượng input; ngược lại cắt về top_k.
        prefer_method: thứ tự ưu tiên khi có nhiều chunk cùng document,
            ví dụ ``["hybrid_weighted", "hybrid", "dense", "bm25"]``.
            Chunk có retrieval_method đứng trước được ưu tiên giữ lại.

    Hành vi:
        Với mỗi document (xác định qua metadata.source hoặc id prefix trước
        '::chunk-'), giữ lại chunk có score cao nhất. Nếu prefer_method
        được cấp, dùng nó để phá vỡ thế cân bằng khi score rất sát nhau.
        KHÔNG thay đổi thứ tự tương đối của các document đã chọn.
    """
    if not results:
        return []
    if prefer_method is None:
        prefer_method = ["hybrid_weighted", "hybrid", "dense", "bm25", "pageindex"]

    method_priority = {m: idx for idx, m in enumerate(prefer_method)}

    def _doc_key(item: dict) -> str:
        meta = item.get("metadata") or {}
        if meta.get("source"):
            return str(meta["source"])
        # fallback: chunk-id thường có dạng <doc>::chunk-N
        return str(item["id"]).split("::chunk-")[0]

    selected: list[dict] = []
    seen_docs: set[str] = set()
    for item in results:
        key = _doc_key(item)
        if key in seen_docs:
            continue
        seen_docs.add(key)
        selected.append(item)

    # stable sort theo (-score, method_priority) để giữ đúng thứ tự gốc khi
    # score bằng nhau.
    def _sort_key(item: dict) -> tuple:
        neg_score = -float(item.get("score", 0.0))
        method = item.get("retrieval_method", "")
        return (neg_score, method_priority.get(method, len(method_priority)))

    # selection đã theo thứ tự score của input; chỉ sort lại khi user yêu cầu
    # ưu tiên method.
    if any(method_priority.get(it.get("retrieval_method", ""), -1) !=
           method_priority.get(results[0].get("retrieval_method", ""), -1)
           for it in selected[: min(len(selected), 5)]):
        selected.sort(key=_sort_key)
    if top_k is not None:
        selected = selected[:top_k]
    return selected
>>>>>>> Stashed changes


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
