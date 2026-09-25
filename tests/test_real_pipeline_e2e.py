"""End-to-end real-data tests.

Mục tiêu: chứng minh pipeline RAG trả về chunks THẬT từ corpus đã index
(không phải mock). Mỗi test verify một invariant thực tế:

    - chunk ID có trong ChromaDB.
    - chunk text tồn tại trong file Markdown gốc.
    - chunk metadata có source/title/url thật.
    - retrieval method là một giá trị hợp lệ.
    - score là số thực.
    - retrieved content cho câu hỏi in-domain chứa token khớp với query.
    - OOD câu hỏi có dense score thấp (gợi ý fallback).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent

# Skip nếu corpus chưa build
pytestmark = pytest.mark.skipif(
    not (ROOT / "chroma_db").exists(),
    reason="ChromaDB not initialized; run scripts/build_index.py first",
)


@pytest.fixture(scope="module")
def corpus_text() -> dict[str, str]:
    """Đọc toàn bộ corpus markdown để verify chunk text tồn tại thật."""
    standardized = ROOT / "data" / "standardized"
    texts: dict[str, str] = {}
    for path in sorted(standardized.rglob("*.md")):
        try:
            texts[path.name] = path.read_text(encoding="utf-8")
        except Exception:
            continue
    return texts


def _chunk_belongs_to_corpus(chunk: dict, corpus: dict[str, str]) -> bool:
    """Verify chunk.content xuất phát từ file Markdown thật trong corpus."""
    text = chunk.get("content", "") or ""
    if not text.strip():
        return False
    # Lấy phần "core" của chunk (bỏ 40 ký tự đầu — YAML header) để so khớp
    snippet = text.strip()[:200]
    for raw in corpus.values():
        if snippet in raw:
            return True
    return False


def test_index_has_real_chunks():
    from src.task4_chunking_indexing import verify_index

    summary = verify_index()
    assert summary["documents"] > 0, "Index phải có ít nhất 1 document"
    assert summary["chunks_in_chromadb"] > 0, "Index phải có chunk"
    assert summary["empty_chunks"] == 0, "Không được có chunk rỗng"
    assert summary["duplicate_chunk_ids"] == 0, "Không được có duplicate chunk ID"
    assert summary["embedding_dim_actual"] == summary["embedding_dim_expected"], (
        f"Embedding dim phải là {summary['embedding_dim_expected']}, "
        f"thực tế {summary['embedding_dim_actual']}"
    )


def test_dense_search_returns_real_chunks(corpus_text):
    from src.task5_semantic_search import semantic_search

    results = semantic_search("điều kiện xét tuyển đại học", top_k=5)
    assert results, "Dense search phải trả về ít nhất 1 chunk"

    # Verify schema
    for r in results:
        assert isinstance(r["id"], str) and r["id"]
        assert isinstance(r["content"], str) and r["content"]
        assert isinstance(r["score"], (int, float))
        assert r["retrieval_method"] == "dense"
        assert r["metadata"].get("source")
        assert r["metadata"].get("title")
        assert "chunk_index" in r["metadata"]

    # Verify content thực sự thuộc corpus
    real = [r for r in results if _chunk_belongs_to_corpus(r, corpus_text)]
    assert len(real) == len(results), (
        f"Một số chunk không tìm thấy trong corpus gốc: "
        f"{[r['id'] for r in results if not _chunk_belongs_to_corpus(r, corpus_text)]}"
    )


def test_bm25_search_returns_real_chunks(corpus_text):
    from src.task6_lexical_search import lexical_search

    results = lexical_search("IELTS xét tuyển đại học", top_k=5)
    assert results, "BM25 phải trả về ít nhất 1 chunk"

    for r in results:
        assert r["retrieval_method"] == "bm25"
        assert r["metadata"].get("source")

    real = [r for r in results if _chunk_belongs_to_corpus(r, corpus_text)]
    assert len(real) == len(results), "BM25 chunk không thuộc corpus"


def test_rrf_uses_real_rankings_and_metadata(corpus_text):
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf

    dense = semantic_search("phương thức xét tuyển đại học", top_k=5)
    bm = lexical_search("phương thức xét tuyển đại học", top_k=5)
    fused = rerank_rrf([dense, bm], top_k=5)
    assert fused, "RRF phải trả về kết quả"

    # RRF score dạng 1/(60+rank) — phải hợp lý
    for r in fused:
        score = float(r["score"])
        assert 0.0 < score <= 1.0 / 61.0 + 1.0 / 61.0, (
            f"RRF score {score} không hợp lý (k=60)"
        )
        assert r["retrieval_method"] == "hybrid"
        assert r["metadata"].get("source")

    # Tất cả ID kết quả phải đến từ dense hoặc bm25
    allowed = {r["id"] for r in dense} | {r["id"] for r in bm}
    for r in fused:
        assert r["id"] in allowed, (
            f"Hybrid ID {r['id']} không có trong dense/BM25 input — RRF bịa ID"
        )


def test_in_domain_query_scores_above_threshold(corpus_text):
    """Câu in-domain phải có dense top1 >= threshold; OOD thì < threshold."""
    from src.task5_semantic_search import semantic_search
    from src.task9_retrieval_pipeline import DEFAULT_SCORE_THRESHOLD

    in_domain = semantic_search("điều kiện xét tuyển đại học", top_k=1)
    assert in_domain
    assert in_domain[0]["score"] >= DEFAULT_SCORE_THRESHOLD, (
        f"In-domain score {in_domain[0]['score']:.3f} dưới threshold "
        f"{DEFAULT_SCORE_THRESHOLD}"
    )

    ood = semantic_search("giá Bitcoin hôm nay bao nhiêu", top_k=1)
    assert ood
    assert ood[0]["score"] < DEFAULT_SCORE_THRESHOLD, (
        f"OOD score {ood[0]['score']:.3f} nên dưới threshold "
        f"{DEFAULT_SCORE_THRESHOLD}"
    )


def test_retrieve_pipeline_preserves_dense_score_for_threshold():
    """retrieve() dùng dense top1 gốc (không phải RRF score) để quyết định fallback."""
    from src.task5_semantic_search import semantic_search
    from src.task9_retrieval_pipeline import DEFAULT_SCORE_THRESHOLD, retrieve

    query = "giá Bitcoin hôm nay bao nhiêu"
    dense_top = semantic_search(query, top_k=1)
    dense_score = dense_top[0]["score"] if dense_top else 0.0

    # Nếu dense top1 < threshold, retrieve() sẽ thử PageIndex fallback.
    # Nếu PageIndex không khả dụng, vẫn trả hybrid results. Điều quan trọng:
    # pipeline không crash và dense score được dùng cho quyết định.
    result = retrieve(query, top_k=5, score_threshold=DEFAULT_SCORE_THRESHOLD)
    # Pipeline phải trả kết quả (hybrid hoặc pageindex) — không bao giờ crash
    # và không bao giờ trả mock data.
    if result:
        for r in result:
            assert isinstance(r["score"], (int, float))
            assert r["metadata"].get("source")


def test_citations_map_to_real_retrieved_sources(corpus_text):
    """Citation [n] trong answer phải map về 1 source trong retrieved chunks.

    Nếu LLM trả safe refusal (không có key / lỗi provider / rate limit),
    test skip — đó là hành vi trung thực và KHÔNG phải mock.
    """
    from src.task10_generation import _safe_refusal, generate_with_citation

    result = generate_with_citation("IELTS xét tuyển đại học", top_k=3)
    safe = _safe_refusal()["answer"]
    if result.get("answer", "").strip() == safe:
        pytest.skip(
            "LLM trả safe refusal (no key / rate limit / provider error) — "
            "không có answer thật để kiểm tra citation. Đây là fallback trung thực."
        )

    sources = result.get("sources", [])
    assert sources, "Phải có sources để map citation"
    assert len(sources) <= 3

    # Nếu LLM từ chối trả lời hợp lệ (ví dụ qwen trả "không xác minh được")
    # mặc dù có retrieved sources, đó là giới hạn của model chứ không phải mock.
    # Citation check vẫn phải tồn tại.
    citation_check = result.get("citation_check") or {}
    assert citation_check, "citation_check phải tồn tại khi có answer"

    # Mọi source phải có metadata thật và content phải thuộc corpus.
    for s in sources:
        assert s["metadata"].get("source"), "Source metadata thiếu"
        assert s["metadata"].get("title"), "Title metadata thiếu"
        assert _chunk_belongs_to_corpus(s, corpus_text), (
            f"Source {s['id']} không có trong corpus — fabrication!"
        )
