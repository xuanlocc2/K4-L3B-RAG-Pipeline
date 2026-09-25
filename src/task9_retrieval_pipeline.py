"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

<<<<<<< Updated upstream
=======
from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

>>>>>>> Stashed changes
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search
from .evidence_quality import (
    assess_evidence,
    build_normalized_query,
    classify_domain,
    score_chunks,
    understand_query,
)


<<<<<<< Updated upstream
SCORE_THRESHOLD = 0.3
=======
load_dotenv()

# Threshold được hiệu chỉnh trên tập validation (xem reports/threshold_calibration.md).
# Calibration 2026-09-25: in-domain min = 0.565, OOD max = 0.494.
# Chọn 0.50 để phân tách sạch giữa hai nhóm và cho phép fallback OOD kích hoạt.
# Có thể override qua env SCORE_THRESHOLD. Giá trị rỗng hoặc không phải số sẽ
# fallback về mặc định 0.50 (không gây crash).
_env_threshold = os.getenv("SCORE_THRESHOLD", "0.50")
try:
    DEFAULT_SCORE_THRESHOLD = float(_env_threshold) if _env_threshold.strip() else 0.50
except (TypeError, ValueError):
    DEFAULT_SCORE_THRESHOLD = 0.50
>>>>>>> Stashed changes
DEFAULT_TOP_K = 5
DEFAULT_MAX_CITATIONS = 3  # 1–3 final supporting sources per audit §8.

<<<<<<< Updated upstream
=======
# Bonus 1: trọng số mặc định cho Weighted RRF.
# Lấy từ env để dễ cấu hình mà không cần sửa code.
# Grid search trên golden_dataset (xem scripts/weighted_rrf_grid.py) cho
# thấy dense_weight >= bm25_weight ổn định hơn; mặc định 1.5/1.0 được
# chọn để đề cao dense một chút mà vẫn dùng BM25 làm fallback cho truy vấn
# lexical-heavy. Giá trị rỗng / không phải số sẽ fallback về mặc định.
def _safe_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


DEFAULT_DENSE_WEIGHT = _safe_float("DENSE_WEIGHT", 1.5)
DEFAULT_BM25_WEIGHT = _safe_float("BM25_WEIGHT", 1.0)

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

>>>>>>> Stashed changes

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Trả về hybrid hoặc pageindex SearchResult."""
    # TODO: Implement full retrieval pipeline.
    #
    # dense = semantic_search(query, top_k=top_k * 2)
    # sparse = lexical_search(query, top_k=top_k * 2)
    # hybrid = (
    #     rerank_rrf([dense, sparse], top_k=top_k)
    #     if use_reranking else dense[:top_k]
    # )
    #
    # best_dense_score = dense[0]["score"] if dense else 0.0
    # if best_dense_score < score_threshold:
    #     try:
    #         fallback = pageindex_search(query, top_k=top_k)
    #         if fallback:
    #             return fallback
    #     except Exception:
    #         pass
    # return hybrid[:top_k]
    raise NotImplementedError("Implement retrieve")


def retrieve_with_evidence(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    *,
    retrieval_mode: str = "baseline",
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
    dedup_top_k_factor: int = 2,
    max_citations: int = DEFAULT_MAX_CITATIONS,
) -> dict:
    """Retrieve + assess evidence-quality gate + return diagnostics.

    Architecture (audit §3 / §5 / §8):

        query
          ↓
        Domain gate  ← classify_domain() — light, deterministic, runs FIRST.
          ↓ is_in_domain
        Candidate Retrieval (Dense + BM25, with PageIndex fallback)
          ↓
        Candidate Ranking (RRF / Weighted RRF / RRF+Dedup / ...)
          ↓
        Evidence Quality Gate  ← per-chunk filter (score_chunks):
                                    each chunk independently passes/fails
                                    on dense_score AND keyword_overlap.
          ↓ passed chunks
        Document Dedup  ← one entry per document.
          ↓
        Final Citations  ← capped to ``max_citations`` (default 3).

    If the domain gate fires OUT, retrieval does NOT run at all — we skip
    candidate retrieval, ranking, and per-chunk gating entirely. This is
    how "kết hôn" is refused without ever touching the corpus.

    Returns a dict with:
        chunks: list[SearchResult]    # candidates after fusion (pre-final)
        intent: dict                  # understand_query output
        domain_classification: dict   # classify_domain output
        normalized_query: str
        evidence: dict                # assess_evidence output
        final_evidence: list[dict]   # deduped, gated, capped — what citations render
        rejected_candidates: list[dict]  # each {"chunk": ..., "reasons": [str]}
        diagnostics: dict
        retrieval_method_used: str
    """
    empty_intent = {
        "education_level": "unspecified",
        "domain": "unknown",
        "intent": "other",
        "entities": [],
        "keywords": [],
        "domain_classification": classify_domain(query),
        "is_out_of_domain": True,
        "conflict": ["empty query"],
    }

    if not query or not query.strip():
        return {
            "chunks": [],
            "intent": empty_intent,
            "domain_classification": empty_intent["domain_classification"],
            "normalized_query": query or "",
            "evidence": {
                "status": "out_of_domain",
                "relevance_label": "Low",
                "suggested_action": "refuse",
                "signals": {},
                "per_chunk": [],
                "passed_chunk_ids": [],
                "rejected_chunk_ids": [],
                "top_chunk_id": None,
                "rationale": "empty query",
            },
            "final_evidence": [],
            "rejected_candidates": [],
            "diagnostics": {
                "dense_topk": [], "bm25_topk": [], "rrf_ranking": [],
            },
            "retrieval_method_used": retrieval_mode,
        }

    # Step 1: domain gate — runs BEFORE any retrieval so OOD queries
    # never touch the corpus (audit §1).
    intent = understand_query(query)
    classification = intent.get("domain_classification") or {}
    if classification.get("is_in_domain") is False:
        normalized_query = build_normalized_query(query, intent)
        evidence = assess_evidence(intent, [], score_threshold=score_threshold)
        return {
            "chunks": [],
            "intent": intent,
            "domain_classification": classification,
            "normalized_query": normalized_query,
            "evidence": evidence,
            "final_evidence": [],
            "rejected_candidates": [],
            "diagnostics": {
                "dense_topk": [], "bm25_topk": [], "rrf_ranking": [],
                "domain_gate": classification,
            },
            "retrieval_method_used": retrieval_mode,
        }

    # Step 2: candidate retrieval (in-domain only at this point).
    normalized_query = build_normalized_query(query, intent)
    fetch_k = top_k * max(1, dedup_top_k_factor)
    dense = semantic_search(normalized_query, top_k=fetch_k)
    sparse = lexical_search(normalized_query, top_k=fetch_k)

    d_w = dense_weight if dense_weight is not None else DEFAULT_DENSE_WEIGHT
    b_w = bm25_weight if bm25_weight is not None else DEFAULT_BM25_WEIGHT

    if retrieval_mode == "dense_only":
        chunks = dense[:top_k]
    elif retrieval_mode == "bm25_only":
        chunks = sparse[:top_k]
    elif retrieval_mode == "weighted_rrf":
        chunks = rerank_weighted_rrf(
            [dense, sparse], top_k=top_k,
            dense_weight=d_w, bm25_weight=b_w,
        )
    elif retrieval_mode == "weighted_rrf_dedup":
        fused = rerank_weighted_rrf(
            [dense, sparse], top_k=fetch_k,
            dense_weight=d_w, bm25_weight=b_w,
        )
        chunks = deduplicate_by_document(
            fused, top_k=top_k, prefer_method=DEFAULT_DEDUP_PREFER,
        )
    elif retrieval_mode == "rrf_dedup":
        fused = rerank_rrf([dense, sparse], top_k=fetch_k)
        chunks = deduplicate_by_document(
            fused, top_k=top_k, prefer_method=DEFAULT_DEDUP_PREFER,
        )
    else:  # "baseline"
        chunks = rerank_rrf([dense, sparse], top_k=top_k)

    # Step 3: evidence-quality gate (per-chunk). This is independent of
    # retrieval mode — even candidate-mode chunks get re-evaluated.
    evidence = assess_evidence(intent, chunks, score_threshold=score_threshold)
    passed_ids = set(evidence.get("passed_chunk_ids") or [])
    rejected_ids = set(evidence.get("rejected_chunk_ids") or [])

    # Step 4: build final_evidence from passed chunks, deduped by document,
    # capped to max_citations (audit §8 / §9). CRITICAL: only the "sufficient"
    # state produces citations — weak and insufficient must yield an empty
    # final_evidence list so the UI never shows them as supporting sources.
    if evidence.get("status") == "sufficient":
        passed = [c for c in chunks if c.get("id") in passed_ids]
        if passed:
            deduped = deduplicate_by_document(
                passed, top_k=max_citations,
                prefer_method=DEFAULT_DEDUP_PREFER,
            )
        else:
            deduped = []
        final_evidence = deduped
    else:
        final_evidence = []

    # Step 5: rejected candidates are surfaced for debug, NOT for citations.
    # Each entry is shaped ``{"chunk": raw_chunk, "reasons": [str, ...]}``
    # so the UI can show exactly why the evidence gate rejected it. Reasons
    # are sourced from ``evidence.per_chunk`` (per-chunk gate verdicts).
    per_chunk_by_id = {
        pc.get("id"): pc
        for pc in (evidence.get("per_chunk") or [])
    }
    rejected: list[dict[str, Any]] = []
    for c in chunks:
        if c.get("id") in rejected_ids:
            pc = per_chunk_by_id.get(c.get("id")) or {}
            rejected.append({
                "chunk": c,
                "reasons": pc.get("reasons") or ["failed evidence gate"],
            })

    # Build a minimal rrf_ranking diagnostic.
    rrf_ranking = [
        {
            "rank": idx + 1,
            "id": ch.get("id"),
            "rrf_score": ch.get("rrf_score"),
            "dense_score": ch.get("dense_score"),
            "bm25_score": ch.get("bm25_score"),
            "gate_status": (
                "passed" if ch.get("id") in passed_ids else "rejected"
            ),
        }
        for idx, ch in enumerate(chunks)
    ]
    diagnostics = {
        "dense_topk": [
            {
                "id": ch.get("id"),
                "score": ch.get("dense_score"),
                "title": (ch.get("metadata") or {}).get("title", ""),
            }
            for ch in dense[:top_k]
        ],
        "bm25_topk": [
            {
                "id": ch.get("id"),
                "score": ch.get("bm25_score"),
                "title": (ch.get("metadata") or {}).get("title", ""),
            }
            for ch in sparse[:top_k]
        ],
        "rrf_ranking": rrf_ranking,
        "domain_gate": classification,
    }

    return {
        "chunks": chunks,
        "intent": intent,
        "domain_classification": classification,
        "normalized_query": normalized_query,
        "evidence": evidence,
        "final_evidence": final_evidence,
        "rejected_candidates": rejected,
        "diagnostics": diagnostics,
        "retrieval_method_used": retrieval_mode,
    }


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
