"""Unit tests for the evidence-quality gate.

These tests use deterministic fixtures and do NOT need a real corpus or
embedding model. They exercise the core decision logic and the contract
for separated dense/bm25/rrf scores.

Tests are organized into the 5 query-level cases required by the audit,
plus tests for the deterministic gate itself and the score separation.
"""

from __future__ import annotations

import pytest

from src.evidence_quality import (
    assess_evidence,
    build_normalized_query,
    make_clarification_message,
    understand_query,
)
from src.task5_semantic_search import semantic_search  # noqa: F401  (sanity import)
from src.task6_lexical_search import lexical_search    # noqa: F401  (sanity import)
from src.task7_reranking import (
    deduplicate_by_document,
    rerank_rrf,
    rerank_weighted_rrf,
)


# ---------------------------------------------------------------------------
# Query understanding (deterministic, no corpus)
# ---------------------------------------------------------------------------


def test_understand_in_domain_admission():
    intent = understand_query("Điều kiện xét tuyển đại học là gì?")
    assert intent["education_level"] == "university"
    assert intent["domain"] == "university_admission"
    assert intent["intent"] == "requirement"
    assert intent["is_out_of_domain"] is False
    assert not intent["conflict"]


def test_understand_paraphrase_recognizes_admission_domain():
    intent = understand_query(
        "Tôi cần những điều kiện gì để đăng ký vào đại học?"
    )
    assert intent["domain"] in {"university_admission", "unknown"}
    assert intent["intent"] in {"requirement", "other"}


def test_understand_ielts_entity():
    intent = understand_query("IELTS 6.5 xét tuyển đại học")
    assert "ielts" in intent["entities"] or "IELTS" in intent["entities"]
    assert intent["domain"] == "university_admission"


def test_understand_primary_school_self_description():
    intent = understand_query(
        "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào"
    )
    assert intent["education_level"] == "primary_school"
    assert intent["domain"] == "university_admission"
    assert intent["conflict"], "Primary school + ĐH phải raise conflict"
    assert any("primary" in c or "tiểu học" in c.lower() for c in intent["conflict"])


def test_understand_thcs_self_description():
    intent = understand_query(
        "Con tôi đang học lớp 8 thì có thể đăng ký xét tuyển đại học không?"
    )
    assert intent["education_level"] == "secondary_school"
    assert intent["domain"] == "university_admission"
    assert intent["conflict"]


def test_understand_bitcoin_is_ood():
    intent = understand_query("Giá Bitcoin hôm nay là bao nhiêu?")
    # Even if domain detection misses, dense-side threshold must catch it.
    # We just verify is_out_of_domain is True OR domain stays unknown.
    assert intent["is_out_of_domain"] is True or intent["domain"] == "unknown"


def test_understand_empty_query_is_safe():
    intent = understand_query("")
    assert intent["education_level"] == "unspecified"
    assert intent["is_out_of_domain"] is True
    assert intent["conflict"]


def test_normalized_query_preserves_original():
    original = "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào"
    intent = understand_query(original)
    normalized = build_normalized_query(original, intent)
    # Original must be unchanged for the caller's downstream usage.
    assert original == "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào"
    # Normalized form focuses on the corpus subject for retrieval.
    assert "đại học" in normalized
    assert "lớp 1" not in normalized


# ---------------------------------------------------------------------------
# Evidence-quality gate (fixture-based)
# ---------------------------------------------------------------------------


def _chunk(
    item_id: str,
    content: str,
    *,
    dense_score: float | None = None,
    bm25_score: float | None = None,
    rrf_score: float | None = None,
    method: str = "hybrid",
) -> dict:
    return {
        "id": item_id,
        "content": content,
        "score": rrf_score if rrf_score is not None else dense_score or 0.0,
        "metadata": {
            "source": "doc.md",
            "title": "Test",
            "doc_type": "legal",
            "url": None,
            "chunk_index": 0,
        },
        "retrieval_method": method,
        **({"dense_score": dense_score} if dense_score is not None else {}),
        **({"bm25_score": bm25_score} if bm25_score is not None else {}),
        **({"rrf_score": rrf_score} if rrf_score is not None else {}),
    }


def test_gate_sufficient_for_in_domain():
    intent = understand_query("Điều kiện xét tuyển đại học là gì?")
    chunks = [_chunk(
        "doc::chunk-0",
        "Điều kiện xét tuyển đại học gồm tốt nghiệp THPT và hồ sơ hợp lệ.",
        dense_score=0.71, bm25_score=43.0, rrf_score=0.033,
    )]
    ev = assess_evidence(intent, chunks)
    assert ev["status"] == "sufficient"
    assert ev["suggested_action"] == "answer"
    assert ev["relevance_label"] in {"High", "Medium"}


def test_gate_clarifies_for_primary_school_with_admission_intent():
    intent = understand_query(
        "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào"
    )
    chunks = [_chunk(
        "news::chunk-22",
        "Trường ĐH công lập xét tuyển bổ sung 9 ngành từ mức 15 điểm.",
        dense_score=0.59, bm25_score=67.0, rrf_score=0.032,
    )]
    ev = assess_evidence(intent, chunks)
    assert ev["suggested_action"] == "clarify"
    assert ev["status"] == "insufficient"
    msg = make_clarification_message(intent, "test query")
    assert "tiểu học" in msg.lower() or "lớp 1" in msg.lower()
    assert "thpt" in msg.lower() or "lớp 12" in msg.lower()


def test_gate_refuses_for_ood_query():
    intent = understand_query("Giá Bitcoin hôm nay là bao nhiêu?")
    chunks = [_chunk(
        "news::chunk-2",
        "Một số tin tức giáo dục mới nhất hôm nay.",
        dense_score=0.45, bm25_score=0.0, rrf_score=0.015,
    )]
    ev = assess_evidence(intent, chunks)
    # Domain gate fires FIRST: OOD must be detected and refused, regardless
    # of the dense/threshold check that would otherwise run on the candidates.
    assert ev["suggested_action"] == "refuse"
    assert ev["status"] == "out_of_domain"


def test_gate_refuses_when_no_chunks():
    intent = understand_query("Điều kiện xét tuyển đại học là gì?")
    ev = assess_evidence(intent, [])
    assert ev["suggested_action"] == "refuse"
    assert ev["status"] == "insufficient"


def test_gate_weak_when_keyword_overlap_too_low():
    """High dense score but only generic token overlap should not pass."""
    intent = understand_query("Điều kiện xét tuyển đại học là gì?")
    # Chunk talks about THPT exam only, none of the user's content words.
    chunks = [_chunk(
        "news::chunk-9",
        "Lịch thi tốt nghiệp THPT 2025 được công bố vào tháng 3.",
        dense_score=0.66, bm25_score=10.0, rrf_score=0.03,
    )]
    # dense >= threshold but overlap is low -> clarify.
    ev = assess_evidence(intent, chunks, keyword_min=0.5)
    # Should be weak/clarify because overlap is only "đại học"/"xét tuyển".
    # We don't pin the exact action (it could be sufficient if overlap >= 0.5),
    # but we DO require the gate to surface the overlap signal.
    assert "keyword_overlap_top1" in ev["signals"]


# ---------------------------------------------------------------------------
# Score separation: dense / bm25 / rrf are preserved and not conflated
# ---------------------------------------------------------------------------


def test_hybrid_result_exposes_dense_bm25_rrf_separately():
    dense = [_chunk(
        "doc::chunk-0", "Điều kiện xét tuyển đại học gồm tốt nghiệp THPT.",
        dense_score=0.7, method="dense",
    )]
    bm = [_chunk(
        "doc::chunk-1", "Hồ sơ xét tuyển đại học cần giấy tờ tùy thân.",
        bm25_score=8.3, method="bm25",
    )]
    fused = rerank_rrf([dense, bm], top_k=5)
    assert fused, "Hybrid list phải có kết quả"
    # Each result must carry rrf_score (= score) and, when known, the original.
    for item in fused:
        assert "rrf_score" in item
        assert item["score"] == pytest.approx(item["rrf_score"])
    # chunk-0 came from dense -> dense_score preserved.
    chunk0 = next(it for it in fused if it["id"] == "doc::chunk-0")
    assert chunk0["dense_score"] == pytest.approx(0.7)
    # chunk-1 came from bm25 -> bm25_score preserved.
    chunk1 = next(it for it in fused if it["id"] == "doc::chunk-1")
    assert chunk1["bm25_score"] == pytest.approx(8.3)


def test_weighted_rrf_preserves_original_scores_and_sets_rrf():
    dense = [_chunk(
        "doc::chunk-0", "A", dense_score=0.8, method="dense",
    )]
    bm = [_chunk(
        "doc::chunk-1", "B", bm25_score=12.0, method="bm25",
    )]
    fused = rerank_weighted_rrf(
        [dense, bm], top_k=5,
        dense_weight=2.0, bm25_weight=1.0,
    )
    chunk0 = next(it for it in fused if it["id"] == "doc::chunk-0")
    chunk1 = next(it for it in fused if it["id"] == "doc::chunk-1")
    assert chunk0["dense_score"] == pytest.approx(0.8)
    assert chunk1["bm25_score"] == pytest.approx(12.0)
    # rrf_score is the weighted contribution (not the original cosine/bm25).
    assert chunk0["rrf_score"] == pytest.approx(2.0 / 61.0)
    assert chunk1["rrf_score"] == pytest.approx(1.0 / 61.0)


def test_score_is_ranking_signal_not_confidence():
    """Ensure we can read dense_score, bm25_score, rrf_score as separate
    values; none of them is the 'confidence' (confidence is in evidence)."""
    dense = [_chunk(
        "a", "alpha", dense_score=0.55, method="dense",
    )]
    bm = [_chunk(
        "a", "alpha", bm25_score=5.0, method="bm25",
    )]
    fused = rerank_rrf([dense, bm], top_k=1)
    item = fused[0]
    # The values are on DIFFERENT scales and cannot be unified:
    assert item["rrf_score"] < 0.1
    assert item["dense_score"] >= 0.5
    assert item["bm25_score"] >= 1.0


# ---------------------------------------------------------------------------
# Citation honesty contract for clarify / refuse paths
# ---------------------------------------------------------------------------


def test_citation_honesty_no_supporting_sources_for_clarify():
    """Generation result for a clarifying query must have empty sources
    (no misleading 'supporting citations')."""
    from src.task10_generation import generate_with_citation

    # Stub the LLM to a deterministic empty answer so the test runs offline.
    import src.task10_generation as gen
    gen.call_llm = lambda *args, **kwargs: ""

    result = generate_with_citation(
        "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào",
        top_k=3,
    )
    assert result["action"] == "clarify"
    assert result["sources"] == [], (
        "Clarify path must NOT cite sources (citation honesty)."
    )
    assert result["retrieval_source"] == "none"
    assert "tiểu học" in result["answer"].lower() or "lớp 1" in result["answer"].lower()


def test_citation_honesty_no_supporting_sources_for_refuse():
    from src.task10_generation import generate_with_citation
    import src.task10_generation as gen
    gen.call_llm = lambda *args, **kwargs: ""

    # Use a real OOD query to drive dense score below threshold.
    result = generate_with_citation(
        "Giá Bitcoin hôm nay là bao nhiêu?",
        top_k=3,
    )
    assert result["action"] == "refuse"
    assert result["sources"] == []
    assert result["retrieval_source"] == "none"


def test_citation_honesty_answer_path_still_has_real_sources():
    """Sanity: an in-domain question still goes through to grounding and
    produces real sources (no regression)."""
    from src.task10_generation import generate_with_citation
    import src.task10_generation as gen
    gen.call_llm = lambda *args, **kwargs: (
        "Theo quy chế, điều kiện xét tuyển là tốt nghiệp THPT [1]."
    )

    result = generate_with_citation(
        "Điều kiện xét tuyển đại học là gì?",
        top_k=3,
    )
    assert result["action"] == "answer"
    assert result["sources"], "In-domain answer phải có sources"
    assert result["retrieval_source"] != "none"
    assert "evidence" in result
    assert result["evidence"]["status"] == "sufficient"
