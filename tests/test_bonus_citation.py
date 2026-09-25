"""Tests cho Bonus 7: Citation grounding validation."""

from __future__ import annotations

from src.bonus7_citation_grounding import (
    extract_citation_indices,
    rewrite_with_safe_citations,
    validate_citations,
)


def _src(i: int) -> dict:
    return {"id": f"src-{i}", "content": f"content {i}", "metadata": {}}


def test_extract_citation_indices_basic():
    assert extract_citation_indices("Theo [1] và [2].") == [1, 2]
    assert extract_citation_indices("No citations here.") == []
    assert extract_citation_indices("Bad [abc] and [-1].") == []
    # [0] được trích ra nhưng sẽ bị validate_citations đánh dấu invalid.
    assert extract_citation_indices("[0]") == [0]


def test_validate_all_valid_citations():
    sources = [_src(1), _src(2), _src(3)]
    result = validate_citations("Theo [1] và [3].", sources)
    assert result["valid"] is True
    assert result["invalid_refs"] == []
    assert result["grounded_ratio"] == 1.0


def test_validate_invalid_refs_detected():
    sources = [_src(1), _src(2)]
    result = validate_citations("Bad refs [3] and [99].", sources)
    assert result["valid"] is False
    assert result["invalid_refs"] == [3, 99]
    assert result["grounded_ratio"] == 0.0


def test_validate_mixed_valid_invalid():
    sources = [_src(1), _src(2)]
    result = validate_citations("[1] ok [5] bad.", sources)
    assert result["invalid_refs"] == [5]
    assert result["grounded_ratio"] == 0.5


def test_validate_detects_missing_citation_when_claim_present():
    sources = [_src(1)]
    long_claim = (
        "Theo quy chế thì học sinh phải tốt nghiệp THPT và nộp hồ sơ đầy đủ."
    )
    result = validate_citations(long_claim, sources)
    assert result["missing_citation"] is True
    assert result["valid"] is False


def test_validate_short_text_without_citation_ok():
    sources = [_src(1)]
    # Câu quá ngắn / không có claim cue -> không flag.
    result = validate_citations("OK.", sources)
    assert result["missing_citation"] is False


def test_rewrite_replaces_invalid_refs():
    sources = [_src(1), _src(2)]
    out = rewrite_with_safe_citations("Theo [1] và [5] và [5].", sources)
    assert "[unverified]" in out["answer"]
    assert "[1]" in out["answer"]
    assert out["invalid_refs"] == [5]
    assert out["modified"] is True


def test_rewrite_passes_through_valid():
    sources = [_src(1), _src(2)]
    out = rewrite_with_safe_citations("Theo [1] và [2].", sources)
    assert out["modified"] is False
    assert "[1]" in out["answer"]


def test_rewrite_appends_warning_when_no_citation_at_all():
    sources = [_src(1)]
    out = rewrite_with_safe_citations(
        "Theo quy chế thì điều kiện rất rõ ràng và đầy đủ.", sources
    )
    assert "không chứa citation hợp lệ" in out["answer"]
    assert out["modified"] is True
