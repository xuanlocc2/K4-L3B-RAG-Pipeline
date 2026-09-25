"""
Task 7 — Reciprocal Rank Fusion (RRF) và Weighted RRF.

Công thức chuẩn (RRF):
    score(d) = sum_i (1 / (k + rank_i))

Trong đó rank_i là thứ hạng của document ``d`` trong bảng xếp hạng thứ ``i``,
bắt đầu từ 1. RRf gộp nhiều bảng xếp hạng mà không cộng trực tiếp
cosine similarity với BM25 score (vốn là hai thang đo khác nhau).

Lưu ý quan trọng:
    RRF score chỉ phản ánh thứ hạng, KHÔNG dùng để quyết định fallback
    (Task 9 dùng cosine score gốc của dense result).
    RRF chỉ được áp dụng MỘT LẦN ở giai đoạn fusion cuối cùng của pipeline.

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


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhiều ranked lists bằng RRF chuẩn.

    Đây là implementation gốc, được giữ nguyên để tương thích ngược với
    test_contracts.py và pipeline mặc định. Weighted RRF được tách sang
    hàm ``rerank_weighted_rrf`` và chỉ được gọi khi người dùng chọn.
    """
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
            items[item_id] = item

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        base = dict(items[item_id])
        base["score"] = scores[item_id]
        base["retrieval_method"] = "hybrid"
        results.append(base)
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
    if not ranked_lists:
        return []
    if weights is None:
        weights = []
        for idx in range(len(ranked_lists)):
            if idx == 0:
                weights.append(dense_weight)
            elif idx == 1:
                weights.append(bm25_weight)
            else:
                weights.append(1.0)
    if len(weights) != len(ranked_lists):
        raise ValueError(
            f"weights length ({len(weights)}) must match ranked_lists ({len(ranked_lists)})"
        )

    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked_list, weight in zip(ranked_lists, weights):
        if weight < 0:
            raise ValueError("weights must be non-negative")
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + weight * 1.0 / (k + rank)
            items[item_id] = item

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    results: list[dict] = []
    for item_id in ranked_ids[:top_k]:
        base = dict(items[item_id])
        base["score"] = scores[item_id]
        base["retrieval_method"] = "hybrid_weighted"
        results.append(base)
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


if __name__ == "__main__":
    print("RRF ready. Use rerank_rrf([dense, bm25], top_k=5) or "
          "rerank_weighted_rrf(..., dense_weight=..., bm25_weight=...).")
