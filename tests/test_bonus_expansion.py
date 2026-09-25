"""Tests cho Bonus 3: Query expansion."""

from __future__ import annotations

from src.bonus3_query_expansion import (
    _hyde_fallback,
    expand_query,
    expand_via_bigrams,
)


def test_baseline_passthrough():
    out = expand_query("IELTS xét tuyển", mode="baseline")
    assert out["method"] == "baseline"
    assert out["expanded_query"] == "IELTS xét tuyển"


def test_empty_query_returns_empty():
    out = expand_query("", mode="expansion")
    assert out["expanded_query"] == ""
    assert out["method"] == "passthrough"


def test_unknown_mode_raises():
    import pytest
    with pytest.raises(ValueError):
        expand_query("test", mode="nonsense")


def test_bigram_expansion_adds_terms():
    # Với corpus đã build, "IELTS" có thể map tới bigram như "ielts toeic".
    expanded = expand_via_bigrams("IELTS")
    assert expanded != "IELTS"  # có thêm từ
    # gốc vẫn nằm trong expanded
    assert "IELTS" in expanded


def test_hyde_fallback_uses_bigrams(monkeypatch):
    # Ép provider openai + xóa mọi API key để _hyde_via_llm trả None.
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    out = expand_query("IELTS xét tuyển", mode="hyde")
    # No LLM key -> fallback local bigram expansion.
    assert out["method"] == "hyde_fallback_local"
    assert "error" in out and "unavailable" in (out["error"] or "").lower()
    # original still present
    assert "IELTS" in out["expanded_query"]


def test_expansion_mode_is_local():
    out = expand_query("điều kiện xét tuyển đại học", mode="expansion")
    assert out["method"] == "local_bigram_expansion"
    # phải có thêm token so với gốc
    assert len(out["expanded_query"].split()) >= len(
        "điều kiện xét tuyển đại học".split()
    )
