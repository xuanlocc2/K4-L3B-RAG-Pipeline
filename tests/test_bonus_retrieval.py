"""
Tests cho các Bonus features:
- Bonus 1: Weighted RRF
- Bonus 2: Document-level deduplication

Các test này dùng monkeypatch giống test_contracts.py để không cần ChromaDB
hay BM25 thật.
"""

from __future__ import annotations

import pytest

from src.contracts import validate_search_results
from src.task7_reranking import (
    deduplicate_by_document,
    rerank_rrf,
    rerank_weighted_rrf,
)


def _result(item_id: str, score: float, method: str, *, source: str = "doc.md") -> dict:
    return {
        "id": item_id,
        "content": f"content of {item_id}",
        "score": score,
        "metadata": {
            "source": source,
            "title": f"Title of {item_id}",
            "doc_type": "legal",
            "url": None,
            "chunk_index": int(item_id.rsplit("-", 1)[-1]),
        },
        "retrieval_method": method,
    }


# --- Bonus 1: Weighted RRF ---------------------------------------------------


def test_weighted_rrf_default_weights_match_standard_rrf():
    dense = [
        _result("chunk-0", 0.9, "dense", source="doc-a.md"),
        _result("chunk-1", 0.8, "dense", source="doc-b.md"),
    ]
    bm25 = [
        _result("chunk-1", 7.0, "bm25", source="doc-b.md"),
        _result("chunk-2", 5.0, "bm25", source="doc-c.md"),
    ]
    standard = rerank_rrf([dense, bm25], top_k=3, k=60)
    weighted_default = rerank_weighted_rrf(
        [dense, bm25], top_k=3, k=60,
        dense_weight=1.0, bm25_weight=1.0,
    )
    assert [r["id"] for r in weighted_default] == [r["id"] for r in standard]
    assert all(r["retrieval_method"] == "hybrid_weighted" for r in weighted_default)


def test_weighted_rrf_weight_changes_ranking():
    dense = [
        _result("chunk-0", 0.9, "dense", source="doc-a.md"),
        _result("chunk-1", 0.8, "dense", source="doc-b.md"),
    ]
    bm25 = [
        _result("chunk-2", 9.0, "bm25", source="doc-c.md"),
        _result("chunk-0", 5.0, "bm25", source="doc-a.md"),
    ]
    # Khi đề cao dense, dense rank 1 của chunk-0 vẫn thắng vì BM25 chỉ cho
    # chunk-0 xếp hạng 2 (RRF weight còn lại).
    dense_focus = rerank_weighted_rrf(
        [dense, bm25], top_k=3, k=60,
        dense_weight=3.0, bm25_weight=1.0,
    )
    assert dense_focus[0]["id"] == "chunk-0"
    # Khi đề cao BM25, điểm BM25 của chunk-0 vẫn lớn nhất vì nó là chunk
    # DUY NHẤT có mặt ở cả 2 danh sách (RRF được cộng dồn từ cả hai).
    # Để BM25 có lợi thế thật sự, dùng dense không chứa chunk-0.
    bm25_focus = rerank_weighted_rrf(
        [dense, bm25], top_k=3, k=60,
        dense_weight=1.0, bm25_weight=3.0,
    )
    # chunk-0: 1/(60+1) + 3/(60+2) ≈ 0.0648
    # chunk-2: 3/(60+1) ≈ 0.0492
    # chunk-1: 1/(60+2) ≈ 0.0161
    assert bm25_focus[0]["id"] == "chunk-0"  # xuất hiện ở cả 2 bảng
    # Vẫn phải có chunk-2 ở top 3 (nó chỉ có BM25, điểm cao)
    assert "chunk-2" in [r["id"] for r in bm25_focus]


def test_weighted_rrf_rejects_invalid_weights():
    dense = [_result("chunk-0", 0.9, "dense")]
    bm25 = [_result("chunk-1", 5.0, "bm25")]
    with pytest.raises(ValueError):
        rerank_weighted_rrf(
            [dense, bm25], top_k=2,
            dense_weight=-0.1, bm25_weight=1.0,
        )
    with pytest.raises(ValueError):
        rerank_weighted_rrf(
            [dense, bm25], top_k=2,
            weights=[1.0],  # len mismatch
        )


def test_weighted_rrf_explicit_weights_override_shorthand():
    dense = [_result("chunk-0", 0.9, "dense")]
    bm25 = [_result("chunk-1", 5.0, "bm25")]
    explicit = rerank_weighted_rrf(
        [dense, bm25], top_k=2, k=60,
        weights=[2.0, 0.5],
        dense_weight=99.0,  # phải bị bỏ qua vì có weights
        bm25_weight=99.0,
    )
    # chunk-0: 2.0 / 61 = 0.0328; chunk-1: 0.5 / 61 = 0.0082
    assert explicit[0]["id"] == "chunk-0"
    assert explicit[0]["score"] == pytest.approx(2.0 / 61.0)


# --- Bonus 2: Document-level dedup ------------------------------------------


def test_dedup_keeps_top_chunk_per_document():
    results = [
        _result("chunk-0", 0.9, "dense", source="doc-a.md"),
        _result("chunk-1", 0.8, "dense", source="doc-a.md"),  # dup doc-a
        _result("chunk-2", 0.7, "dense", source="doc-b.md"),
        _result("chunk-3", 0.6, "dense", source="doc-c.md"),
    ]
    deduped = deduplicate_by_document(results, top_k=5)
    sources = [r["metadata"]["source"] for r in deduped]
    assert sources == ["doc-a.md", "doc-b.md", "doc-c.md"]
    assert deduped[0]["id"] == "chunk-0"  # score cao nhất của doc-a


def test_dedup_preserves_source_metadata():
    results = [
        _result("chunk-0", 0.9, "dense", source="https://example.com/a.htm"),
        _result("chunk-1", 0.7, "dense", source="https://example.com/a.htm"),
    ]
    deduped = deduplicate_by_document(results, top_k=5)
    assert len(deduped) == 1
    assert deduped[0]["metadata"]["source"] == "https://example.com/a.htm"
    assert deduped[0]["metadata"]["title"] == "Title of chunk-0"


def test_dedup_top_k_truncates_after_dedup():
    results = [
        _result("chunk-0", 0.9, "dense", source="doc-a.md"),
        _result("chunk-1", 0.8, "dense", source="doc-b.md"),
        _result("chunk-2", 0.7, "dense", source="doc-c.md"),
        _result("chunk-3", 0.6, "dense", source="doc-d.md"),
    ]
    deduped = deduplicate_by_document(results, top_k=2)
    assert len(deduped) == 2
    assert deduped[0]["id"] == "chunk-0"
    assert deduped[1]["id"] == "chunk-1"


def test_dedup_empty_input_returns_empty():
    assert deduplicate_by_document([]) == []


def test_dedup_via_retrieve_weighted_rrf_dedup(monkeypatch):
    """End-to-end: retrieval_mode='weighted_rrf_dedup' phải dedup theo doc."""
    import src.task9_retrieval_pipeline as pipeline

    # 5 dense candidates, 3 cùng doc-a (BM25 đẩy article phổ biến).
    dense = [
        _result("chunk-0", 0.9, "dense", source="doc-a.md"),
        _result("chunk-1", 0.8, "dense", source="doc-a.md"),
        _result("chunk-2", 0.7, "dense", source="doc-a.md"),
        _result("chunk-3", 0.6, "dense", source="doc-b.md"),
        _result("chunk-4", 0.5, "dense", source="doc-c.md"),
    ]
    bm25 = [
        _result("chunk-0", 9.0, "bm25", source="doc-a.md"),
        _result("chunk-1", 8.0, "bm25", source="doc-a.md"),
        _result("chunk-2", 7.0, "bm25", source="doc-a.md"),
        _result("chunk-3", 6.0, "bm25", source="doc-b.md"),
    ]
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: bm25)
    monkeypatch.setattr(pipeline, "pageindex_search", lambda q, top_k: [])

    # baseline (RRF chuẩn) sẽ trả về top-2 toàn doc-a vì BM25 đẩy mạnh.
    baseline = pipeline.retrieve("tuition", top_k=2, retrieval_mode="baseline")
    assert all(r["metadata"]["source"] == "doc-a.md" for r in baseline)

    # weighted_rrf_dedup phải đa dạng hoá theo document.
    diversified = pipeline.retrieve(
        "tuition", top_k=2, retrieval_mode="weighted_rrf_dedup",
    )
    sources = [r["metadata"]["source"] for r in diversified]
    assert len(set(sources)) == 2  # 2 document khác nhau
    validate_search_results(diversified, top_k=2)


def test_retrieve_rejects_invalid_mode(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: [])
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [])
    with pytest.raises(ValueError):
        pipeline.retrieve("tuition", top_k=3, retrieval_mode="bogus-mode")


def test_retrieve_dense_only_mode_skips_rrf(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [_result("chunk-0", 0.9, "dense")]
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [])
    monkeypatch.setattr(pipeline, "pageindex_search", lambda q, top_k: [])
    output = pipeline.retrieve("tuition", top_k=1, retrieval_mode="dense_only")
    assert output == dense
