"""Tests cho Bonus 4: Reranker (lightweight)."""

from __future__ import annotations

from src.bonus4_reranker import (
    rerank_cosine,
    rerank_dense_then_bm25_then_rrf_then_cosine,
    rerank_keyword_overlap,
    rerank_with_function,
)


def _cand(i: int, content: str) -> dict:
    return {
        "id": f"chunk-{i}",
        "content": content,
        "score": 1.0 - i / 10.0,
        "metadata": {"source": "doc.md", "title": "T", "doc_type": "legal",
                     "url": None, "chunk_index": i},
        "retrieval_method": "hybrid",
    }


def test_keyword_overlap_promotes_matching_chunk():
    candidates = [
        _cand(0, "IELTS xét tuyển đại học"),
        _cand(1, "Tin tức thời tiết hôm nay"),
        _cand(2, "IELTS và TOEIC đều là chứng chỉ quốc tế"),
    ]
    out = rerank_keyword_overlap("IELTS xét tuyển", candidates)
    # chunk-0 và chunk-2 có "IELTS"; chunk-1 không có.
    assert out[0]["id"] in ("chunk-0", "chunk-2")
    assert out[-1]["id"] == "chunk-1"
    assert all(r["retrieval_method"] == "reranked_keyword_overlap" for r in out)


def test_keyword_overlap_empty_input():
    assert rerank_keyword_overlap("q", []) == []


def test_keyword_overlap_respects_top_k():
    candidates = [_cand(i, f"content {i}") for i in range(5)]
    out = rerank_keyword_overlap("content", candidates, top_k=2)
    assert len(out) == 2


def test_rerank_with_custom_function():
    candidates = [_cand(i, f"x {i}") for i in range(3)]

    def my_reranker(items):
        # Đảo ngược thứ tự.
        return list(reversed(items))

    out = rerank_with_function(candidates, my_reranker)
    assert [r["id"] for r in out] == ["chunk-2", "chunk-1", "chunk-0"]


def test_rerank_with_custom_function_respects_top_k():
    candidates = [_cand(i, f"x {i}") for i in range(5)]

    def my_reranker(items):
        return list(reversed(items))

    out = rerank_with_function(candidates, my_reranker, top_k=2)
    assert len(out) == 2


def test_cosine_reranker_changes_retrieval_method():
    candidates = [_cand(i, "IELTS xét tuyển") for i in range(2)]
    out = rerank_cosine("IELTS xét tuyển", candidates)
    assert all(r["retrieval_method"] == "reranked_cosine" for r in out)


def test_convenience_pipeline_shape():
    fused = [_cand(i, "IELTS xét tuyển") for i in range(3)]
    out = rerank_dense_then_bm25_then_rrf_then_cosine("IELTS", fused, top_k=2)
    assert len(out) == 2


def test_convenience_empty_input():
    assert rerank_dense_then_bm25_then_rrf_then_cosine("q", [], top_k=3) == []
