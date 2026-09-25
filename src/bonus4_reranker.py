"""
Bonus 4 — Reranker (lightweight, optional).

Chọn cách tiếp cận lightweight, reproducible, không cần download thêm model:

    ``rerank_cosine(query, candidates, top_k=None)``:
        Tính lại cosine similarity giữa query embedding và từng candidate
        embedding (nếu có sẵn), sort lại và trả top-k. Đây là "second-pass"
        reranker cực nhẹ: chỉ làm phẳng score scale giữa dense và BM25.

    ``rerank_keyword_overlap(query, candidates, top_k=None)``:
        Tính BM25-like overlap giữa query tokens và candidate content. Nhẹ,
        deterministic, không cần model.

Cả hai được thiết kế để chạy SAU fusion RRF và trên top-N candidates
(N >> top_k), KHÔNG thay thế dense hay BM25.

Nếu người dùng muốn dùng cross-encoder thật, họ có thể tự inject qua
``rerank_fn``. Đây là design choice để giữ pipeline reproducible trong
môi trường không có mạng / hạn chế model.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from .task4_chunking_indexing import embed_texts

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def rerank_cosine(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank bằng cosine similarity giữa query và content (đã embed sẵn)."""
    if not candidates:
        return []
    query_vec = embed_texts([query])[0] if query.strip() else None
    scored: list[dict[str, Any]] = []
    for cand in candidates:
        text = cand.get("content", "")
        # Ưu tiên dùng embedding sẵn trong candidate nếu có.
        emb = cand.get("embedding")
        if emb and query_vec:
            score = _cosine(query_vec, emb)
        else:
            cand_vec = embed_texts([text])[0] if text.strip() else None
            score = _cosine(query_vec, cand_vec) if cand_vec and query_vec else 0.0
        out = dict(cand)
        out["score"] = float(score)
        out["retrieval_method"] = "reranked_cosine"
        scored.append(out)
    scored.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def rerank_keyword_overlap(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rerank bằng Jaccard keyword overlap giữa query và content.

    Đơn giản, deterministic, không cần model. Phù hợp corpus nhỏ.
    """
    if not candidates:
        return []
    q_tokens = set(_tokenize(query))
    scored: list[dict[str, Any]] = []
    for cand in candidates:
        c_tokens = set(_tokenize(cand.get("content", "")))
        if not q_tokens or not c_tokens:
            jaccard = 0.0
        else:
            jaccard = len(q_tokens & c_tokens) / len(q_tokens | c_tokens)
        out = dict(cand)
        out["score"] = float(jaccard)
        out["retrieval_method"] = "reranked_keyword_overlap"
        scored.append(out)
    scored.sort(key=lambda x: x["score"], reverse=True)
    if top_k is not None:
        scored = scored[:top_k]
    return scored


def rerank_with_function(
    candidates: list[dict[str, Any]],
    rerank_fn: Callable[[list[dict[str, Any]]], list[dict[str, Any]]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Cho phép user inject rerank_fn tùy ý (cross-encoder, LLM judge, ...).

    Hàm này đảm bảo pipeline luôn có dạng:
        Dense → BM25 → RRF → top-N → rerank → top-K
    """
    if not candidates:
        return []
    out = rerank_fn(candidates)
    if top_k is not None:
        out = out[:top_k]
    return out


def rerank_dense_then_bm25_then_rrf_then_cosine(
    query: str,
    fused: list[dict[str, Any]],
    top_k: int = 5,
    candidate_pool: int = 10,
) -> list[dict[str, Any]]:
    """Convenience: rerank trên top ``candidate_pool`` rồi cắt top_k.

    Vì pipeline đã trả top_k, ta cần rerank trên chính các candidate đó.
    Hàm này KHÔNG truy xuất thêm; nó chỉ áp dụng reranker lên input list.
    """
    if not fused:
        return []
    pool = fused[:candidate_pool]
    reranked = rerank_cosine(query, pool)
    return reranked[:top_k]


if __name__ == "__main__":
    sample = [
        {"id": "a", "content": "IELTS xét tuyển đại học", "score": 0.9},
        {"id": "b", "content": "Tin tức thời tiết", "score": 0.7},
    ]
    for r in rerank_keyword_overlap("IELTS xét tuyển", sample):
        print(r["id"], round(r["score"], 3))
