"""UI helpers tests — render_rejected_candidates_panel + import surface."""

from __future__ import annotations

from src.ui_components import render_rejected_candidates_panel


def _raw_chunk(cid, content, *, source="https://example.com/article-1",
               method="hybrid", dense=0.5, bm25=10.0, rrf=0.02):
    return {
        "id": cid,
        "content": content,
        "score": rrf,
        "dense_score": dense,
        "bm25_score": bm25,
        "rrf_score": rrf,
        "retrieval_method": method,
        "metadata": {"title": f"Title for {cid}", "source": source, "url": source},
    }


def test_render_rejected_returns_empty_when_no_rejected():
    html_out = render_rejected_candidates_panel([])
    assert html_out == ""


def test_render_rejected_accepts_pipeline_shape():
    """The pipeline produces ``{"chunk": ..., "reasons": [...]}``."""
    rejected = [
        {
            "chunk": _raw_chunk(
                "doc::c1", "Nội dung chunk A bị loại vì không liên quan.",
                source="https://dantri.com.vn/x", method="hybrid",
                dense=0.42,
            ),
            "reasons": ["dense_score=0.420 dưới ngưỡng 0.500"],
        },
        {
            "chunk": _raw_chunk(
                "doc::c2", "Nội dung chunk B không có overlap từ khoá.",
                source="https://thanhnien.vn/y", method="dense",
                dense=0.66,
            ),
            "reasons": ["keyword_overlap=0.10 dưới ngưỡng 0.20"],
        },
    ]
    out = render_rejected_candidates_panel(rejected)
    assert out, "panel must render non-empty"
    # Must NOT label rejected items as "Citation".
    assert "Citation" not in out
    # Must include the explicit "rejected" / "KHÔNG" disclaimers.
    assert "Rejected candidates" in out
    # The Vietnamese "KHÔNG dùng" disclaimer is mandatory (audit §14).
    assert "dùng" in out
    # Must surface the rejection reasons.
    assert "dense_score=0.420" in out
    assert "keyword_overlap=0.10" in out
    # Must show method, dense, bm25, rrf as separate fields (not "confidence").
    assert "method=" in out
    assert "dense=" in out
    assert "bm25=" in out
    assert "rrf=" in out
    assert "confidence" not in out.lower()
    # Both titles must appear.
    assert "Title for doc::c1" in out
    assert "Title for doc::c2" in out
    # Sources must appear.
    assert "dantri.com.vn" in out
    assert "thanhnien.vn" in out


def test_render_rejected_accepts_raw_chunk_fallback():
    """Defensive: raw SearchResult (no reasons wrapper) still renders."""
    rejected = [_raw_chunk("doc::c3", "Chunk C")]
    out = render_rejected_candidates_panel(rejected)
    assert out
    assert "Title for doc::c3" in out
    assert "failed evidence gate" in out or "rejected" in out.lower()
    assert "Citation" not in out
