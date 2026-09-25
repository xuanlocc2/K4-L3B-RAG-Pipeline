"""Tests cho UI components (Bonus 6, 9, 10 helpers)."""

from __future__ import annotations

from src.ui_components import (
    render_analytics_table,
    render_citation_panel,
    render_debug_panel,
)


def _src(i: int, content: str = "Sample passage content", score: float = 0.8) -> dict:
    return {
        "id": f"chunk-{i}",
        "content": content,
        "score": score,
        "metadata": {
            "source": f"doc-{i}.md",
            "title": f"Title {i}",
            "doc_type": "legal",
            "url": f"https://example.com/{i}",
            "chunk_index": i,
        },
        "retrieval_method": "hybrid_weighted",
    }


def test_render_citation_panel_empty():
    assert render_citation_panel([]) == ""


def test_render_citation_panel_basic():
    html = render_citation_panel([_src(1), _src(2)])
    assert "Citation [1]" in html
    assert "Citation [2]" in html
    assert "doc-1.md" in html
    assert "Title 1" in html
    assert "https://example.com/1" in html
    # Passage được highlight.
    assert "Relevant passage" in html
    assert "Sample passage content" in html


def test_render_citation_panel_truncates_long_passage():
    long_content = "Xin chào. " * 100
    html = render_citation_panel([_src(1, content=long_content)], passage_max_chars=200)
    # phải có dấu "..." nếu cắt
    assert "..." in html
    # Không vượt quá passage_max_chars + một số overhead cho HTML.
    import re as _re
    block_match = _re.search(r"<blockquote[^>]*>(.*?)</blockquote>", html, _re.S)
    assert block_match
    inner = _re.sub(r"<[^>]+>", "", block_match.group(1)).strip()
    assert len(inner) <= 230


def test_render_citation_panel_escapes_html():
    html = render_citation_panel([_src(1, content='<script>alert("xss")</script>')])
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_render_debug_panel_basic():
    html = render_debug_panel(
        query="Q",
        original_query=None,
        rewrite_info=None,
        retrieval_mode="weighted_rrf",
        score_threshold=0.5,
        citation_check={
            "valid": True, "missing_citation": False,
            "invalid_refs": [], "grounded_ratio": 1.0, "total_refs": 2,
        },
        final_results=[_src(1), _src(2)],
    )
    assert "Debug" in html
    assert "weighted_rrf" in html
    assert "Citation grounding" in html
    assert "Final context" in html


def test_render_debug_panel_shows_rewrite():
    html = render_debug_panel(
        query="IELTS Còn vậy thì sao?",
        original_query="Còn vậy thì sao?",
        rewrite_info={"method": "local_heuristic", "last_topic": "IELTS"},
        retrieval_mode="baseline",
        score_threshold=0.5,
        citation_check=None,
    )
    assert "Original query" in html
    assert "local_heuristic" in html
    assert "IELTS" in html


def test_render_analytics_table():
    rows = [
        {"strategy": "A", "context_precision": 0.5, "context_recall": 0.84,
         "source_hit_rate": 0.9, "doc_diversity_avg": 2.2, "avg_latency_ms": 100.0},
        {"strategy": "B", "context_precision": 0.49, "context_recall": 0.79,
         "source_hit_rate": 0.86, "doc_diversity_avg": 2.3, "avg_latency_ms": 95.0},
    ]
    html = render_analytics_table(rows)
    assert "<table" in html
    assert ">A<" in html
    assert ">B<" in html
    assert "0.500" in html
    assert "100.0" in html


def test_render_analytics_table_empty():
    assert render_analytics_table([]) == ""
