"""UI helpers cho citation highlighting (Bonus 6) và debug mode (Bonus 9).

Tách logic render khỏi app.py để dễ test và tái sử dụng nếu sau này muốn
đổi UI framework (Gradio, FastAPI + HTML, ...).
"""

from __future__ import annotations

import html
import re
from typing import Any


def _extract_relevant_passage(content: str, max_chars: int = 320) -> str:
    """Trích đoạn đầu tiên có ích từ chunk content làm passage nổi bật.

    Bỏ qua khoảng trắng thừa ở đầu; cắt ở ranh giới câu gần nhất.
    """
    if not content:
        return ""
    text = re.sub(r"\s+", " ", content).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = max(truncated.rfind(". "), truncated.rfind("; "))
    if last_period > max_chars // 2:
        truncated = truncated[: last_period + 1]
    return truncated.rstrip() + "..."


def _safe_num(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if f != f or f in (float("inf"), float("-inf")):
            return None
        return f
    return None


def _relevance_label_from_scores(chunk: dict[str, Any]) -> str:
    """Map chunk scores to a coarse relevance label for the user.

    Uses only dense_score (cosine similarity), which is the only score that
    has a stable [0, 1] interpretation. RRF and BM25 are shown separately
    in the developer/debug block.
    """
    dense = _safe_num(chunk.get("dense_score"))
    if dense is None:
        # dense_score missing (e.g. pageindex-only result): fall back to RRF
        # magnitude, which is not comparable across corpora but gives a
        # coarse signal.
        rrf = _safe_num(chunk.get("rrf_score"))
        if rrf is None:
            return "Low"
        return "Medium" if rrf >= 0.02 else "Low"
    if dense >= 0.65:
        return "High"
    if dense >= 0.5:
        return "Medium"
    return "Low"


def render_state_banner(
    action: str,
    answerability_state: str | None = None,
    evidence_status: str | None = None,
) -> str:
    """Small caption that surfaces the answerability state of the last turn.

    Audit §11/§13: the user should see which branch of the state machine
    produced the answer. This avoids "I cannot verify..." + a citation panel
    that look contradictory.

    Returns a short HTML string. Empty string for the trivial "answer" case.
    """
    label_map = {
        "out_of_domain": "🚫 Câu hỏi nằm ngoài phạm vi dữ liệu tuyển sinh",
        "evidence_insufficient": (
            "⚠️ Bằng chứng trong corpus chưa đủ để trả lời"
        ),
        "evidence_weak": (
            "❓ Bằng chứng quá mỏng — cần thêm ngữ cảnh"
        ),
        "evidence_sufficient": "✅ Có bằng chứng hỗ trợ trong corpus",
        "answer": "",
        "clarify": "❓ Cần làm rõ thêm",
        "refuse": "⚠️ Đã từ chối an toàn",
    }
    state = answerability_state or action
    if state not in label_map or not label_map[state]:
        return ""
    return (
        "<div style='font-size:0.8em; color:#888; margin-top:4px;'>"
        + html.escape(label_map[state])
        + "</div>"
    )


def render_rejected_candidates_panel(
    rejected: list[Any],
    *,
    passage_max_chars: int = 160,
) -> str:
    """Debug-only panel listing candidates that retrieval returned but the
    evidence gate rejected.

    Audit §14: rejected candidates must NEVER appear in the normal
    user-facing citation section, but they may be surfaced in the debug
    view with the rejection reason.

    Input contract — each item may be either::

        # preferred shape (what the pipeline produces):
        {
            "chunk":   dict          # raw SearchResult
            "reasons": list[str]     # per-chunk rejection reasons
        }

        # or a raw SearchResult dict (defensive fallback):
        {"id": "...", "content": "...", "metadata": {...}, ...}

    Each row shows:
        * document title
        * source URL / source label
        * retrieval method (dense / bm25 / hybrid / ...)
        * dense_score, bm25_score, rrf_score (separated, NOT labelled
          "confidence")
        * rejection reason(s)
        * short relevant passage

    Returns an empty string for empty input so the caller can safely
    concatenate.
    """
    if not rejected:
        return ""
    rows: list[str] = []
    for idx, item in enumerate(rejected, 1):
        # Normalize item → (chunk, reasons).
        if isinstance(item, dict) and "chunk" in item and isinstance(
            item["chunk"], dict
        ):
            chunk = item["chunk"]
            reasons = item.get("reasons") or []
        elif isinstance(item, dict):
            chunk = item
            reasons = []
        else:
            continue
        meta = chunk.get("metadata") or {}
        title = meta.get("title") or "(không tiêu đề)"
        source = meta.get("source") or meta.get("url") or "(không rõ nguồn)"
        method = chunk.get("retrieval_method", "")
        dense_v = _safe_num(chunk.get("dense_score"))
        bm25_v = _safe_num(chunk.get("bm25_score"))
        rrf_v = _safe_num(chunk.get("rrf_score"))

        reason_text = (
            "; ".join(reasons) if reasons else "failed evidence gate"
        )
        passage = _extract_relevant_passage(
            chunk.get("content", ""), passage_max_chars
        )

        score_parts: list[str] = []
        if method:
            score_parts.append(f"method=<code>{html.escape(str(method))}</code>")
        if dense_v is not None:
            score_parts.append(f"dense=<code>{dense_v:.3f}</code>")
        if bm25_v is not None:
            score_parts.append(f"bm25=<code>{bm25_v:.2f}</code>")
        if rrf_v is not None:
            score_parts.append(f"rrf=<code>{rrf_v:.3f}</code>")
        score_html = (
            " · ".join(score_parts) if score_parts else "scores=?"
        )

        rows.append(
            "<div style='border:1px dashed #c66; padding:6px 8px; "
            "margin:4px 0; background:#fdf5f5;'>"
            f"<b>[{idx}] {html.escape(title)}</b> "
            f"<span style='color:#888;'>— {html.escape(str(source))}</span><br>"
            f"<span style='font-size:0.85em; color:#444;'>"
            f"{score_html}</span><br>"
            f"<span style='color:#a44; font-size:0.85em;'>"
            f"reason: {html.escape(reason_text)}</span><br>"
            f"<span style='font-size:0.85em; color:#444;'>"
            f"{html.escape(passage)}</span>"
            "</div>"
        )
    return (
        "<div style='border:1px dashed #c66; padding:8px 10px; "
        "margin-top:8px;'>"
        "<div style='font-weight:600; color:#a44;'>"
        "❌ Rejected candidates (retrieved nhưng KHÔNG vượt được evidence gate) "
        "— KHÔNG dùng làm nguồn trích dẫn</div>"
        + "".join(rows)
        + "</div>"
    )


def render_citation_panel(
    sources: list[dict[str, Any]],
    *,
    highlight_passage: bool = True,
    passage_max_chars: int = 320,
    show_debug_scores: bool = False,
) -> str:
    """Trả về HTML cho citation panel.

    User-facing block (always shown):
        Citation [n]
        Title
        Source: ...
        Relevance: High / Medium / Low       ← new, derived from dense score
        Relevant passage (quote)

    Developer/debug block (shown only when ``show_debug_scores=True``):
        Retrieval: hybrid_weighted
        Dense similarity: 0.42
        BM25 score:       8.31
        RRF score:        0.031
    """
    if not sources:
        return ""
    blocks: list[str] = []
    for idx, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        title = meta.get("title") or "(không tiêu đề)"
        source = meta.get("source") or "(không rõ nguồn)"
        url = meta.get("url") or ""
        method = src.get("retrieval_method", "")
        relevance = _relevance_label_from_scores(src)

        passage_html = ""
        if highlight_passage:
            passage = _extract_relevant_passage(src.get("content", ""), passage_max_chars)
            if passage:
                passage_html = (
                    "<div style='margin-top:6px;'>"
                    "<div style='font-size:0.85em; color:#555;'>Relevant passage</div>"
                    "<blockquote style='margin:4px 0; padding:6px 10px; "
                    "border-left:3px solid #888; background:#f7f7f7; "
                    "color:#222;'>"
                    f"{html.escape(passage)}"
                    "</blockquote></div>"
                )

        url_html = (
            f"<div style='margin-top:4px;'><a href='{html.escape(url)}' "
            f"target='_blank' rel='noopener'>{html.escape(url)}</a></div>"
            if url else ""
        )

        debug_html = ""
        if show_debug_scores:
            dense_v = _safe_num(src.get("dense_score"))
            bm25_v = _safe_num(src.get("bm25_score"))
            rrf_v = _safe_num(src.get("rrf_score"))
            score_v = _safe_num(src.get("score"))
            pieces = []
            if method:
                pieces.append(f"method={html.escape(str(method))}")
            if score_v is not None:
                pieces.append(f"score={score_v:.3f}")
            if dense_v is not None:
                pieces.append(f"dense={dense_v:.3f}")
            if bm25_v is not None:
                pieces.append(f"bm25={bm25_v:.2f}")
            if rrf_v is not None:
                pieces.append(f"rrf={rrf_v:.3f}")
            debug_html = (
                "<div style='font-size:0.78em; color:#777; margin-top:2px;'>"
                + " · ".join(pieces)
                + "</div>"
            )

        blocks.append(
            "<div style='border:1px solid #ddd; border-radius:6px; "
            "padding:10px 12px; margin-bottom:8px; background:#fff;'>"
            f"<div style='font-weight:600;'>Citation [{idx}]</div>"
            f"<div style='font-weight:500; margin-top:2px;'>{html.escape(title)}</div>"
            f"<div style='font-size:0.85em; color:#444;'>Source: "
            f"{html.escape(str(source))}</div>"
            f"<div style='font-size:0.85em; color:#444;'>Relevance: "
            f"<b>{html.escape(relevance)}</b></div>"
            f"{debug_html}"
            f"{passage_html}"
            f"{url_html}"
            "</div>"
        )
    return "<div>" + "".join(blocks) + "</div>"


def render_insufficient_evidence_panel(
    sources: list[dict[str, Any]],
    *,
    passage_max_chars: int = 200,
) -> str:
    """HTML cho panel 'Retrieved nhưng KHÔNG đủ bằng chứng'.

    Được hiển thị khi evidence gate quyết định clarify / refuse, thay vì
    hiển thị nguồn như 'trích dẫn hỗ trợ'. Mục đích: tránh ấn tượng sai
    rằng các đoạn này trả lời câu hỏi.
    """
    if not sources:
        return ""
    rows: list[str] = []
    for idx, src in enumerate(sources, 1):
        meta = src.get("metadata") or {}
        title = meta.get("title") or "(không tiêu đề)"
        relevance = _relevance_label_from_scores(src)
        passage = _extract_relevant_passage(
            src.get("content", ""), passage_max_chars,
        )
        rows.append(
            "<div style='border:1px dashed #ccc; padding:6px 8px; "
            "margin:4px 0; background:#fafafa;'>"
            f"<b>[{idx}]</b> {html.escape(title)} "
            f"<span style='color:#888;'>(relevance: {html.escape(relevance)})</span><br>"
            f"<span style='font-size:0.85em; color:#444;'>{html.escape(passage)}</span>"
            "</div>"
        )
    return (
        "<div style='border:1px dashed #aaa; padding:6px 8px; "
        "margin-top:6px;'>"
        "<div style='font-weight:600;'>⚠ Retrieved nhưng chưa đủ bằng chứng "
        "(chỉ để debug — KHÔNG dùng làm nguồn trích dẫn)</div>"
        + "".join(rows)
        + "</div>"
    )


def render_debug_panel(
    *,
    query: str,
    original_query: str | None,
    rewrite_info: dict | None,
    retrieval_mode: str,
    score_threshold: float,
    citation_check: dict | None,
    dense_results: list[dict[str, Any]] | None = None,
    bm25_results: list[dict[str, Any]] | None = None,
    final_results: list[dict[str, Any]] | None = None,
    retrieval_source: str = "hybrid",
    evidence: dict | None = None,
    intent: dict | None = None,
    normalized_query: str | None = None,
    diagnostics: dict | None = None,
) -> str:
    """Trả về HTML cho debug / observability panel (Bonus 9).

    Hiển thị:
        - Query (effective vs original vs normalized)
        - Query understanding (education_level, domain, intent, ...)
        - Retrieval mode + threshold
        - Optional: top dense, top BM25, final
        - Evidence-quality gate decision
        - Citation grounding report
    """
    rows: list[str] = []

    def _row(label: str, value: Any) -> None:
        if value is None or value == "":
            return
        rows.append(
            "<div style='margin-bottom:4px;'>"
            f"<span style='color:#888;'>{html.escape(label)}:</span> "
            f"<code>{html.escape(str(value))}</code></div>"
        )

    _row("Original query", original_query or query)
    _row("Normalized query (retrieval-side)", normalized_query)
    if query and query != (original_query or query) and query != (normalized_query or ""):
        _row("Effective query", query)
    if rewrite_info:
        _row("Rewrite method", rewrite_info.get("method"))
        _row("Last topic", rewrite_info.get("last_topic"))
    _row("Retrieval mode", retrieval_mode)
    _row("Score threshold", score_threshold)
    _row("Retrieval source", retrieval_source)

    if intent:
        rows.append(
            "<div style='margin-top:8px;'><b>Query understanding</b></div>"
            "<div style='font-size:0.85em;'>"
            f"education_level=<code>{html.escape(str(intent.get('education_level')))}</code>, "
            f"domain=<code>{html.escape(str(intent.get('domain')))}</code>, "
            f"intent=<code>{html.escape(str(intent.get('intent')))}</code>, "
            f"entities=<code>{html.escape(str(intent.get('entities', [])))}</code>, "
            f"is_out_of_domain=<code>{html.escape(str(intent.get('is_out_of_domain')))}</code>"
            "</div>"
        )
        conflict = intent.get("conflict") or []
        if conflict:
            rows.append(
                "<div style='font-size:0.85em; color:#a55;'>"
                "<b>Conflict:</b> " + html.escape("; ".join(conflict))
                + "</div>"
            )

    if evidence:
        rows.append("<div style='margin-top:8px;'><b>Evidence quality</b></div>")
        sig = evidence.get("signals") or {}
        rows.append(
            "<div style='font-size:0.85em;'>"
            f"status=<code>{html.escape(str(evidence.get('status')))}</code>, "
            f"relevance_label=<code>{html.escape(str(evidence.get('relevance_label')))}</code>, "
            f"suggested_action=<code>{html.escape(str(evidence.get('suggested_action')))}</code>, "
            f"dense_top1=<code>{sig.get('dense_top1', 0):.3f}</code>, "
            f"bm25_top1=<code>{sig.get('bm25_top1', 0):.2f}</code>, "
            f"rrf_top1=<code>{sig.get('rrf_top1', 0):.4f}</code>, "
            f"keyword_overlap_top1=<code>{sig.get('keyword_overlap_top1', 0):.2f}</code>"
            "</div>"
        )
        rationale = evidence.get("rationale")
        if rationale:
            rows.append(
                "<div style='font-size:0.85em; color:#555;'>"
                "Rationale: " + html.escape(str(rationale)) + "</div>"
            )

    def _block(title: str, items: list[dict[str, Any]] | None) -> None:
        if not items:
            return
        rows.append(f"<div style='margin-top:8px;'><b>{html.escape(title)}</b></div>")
        for idx, item in enumerate(items, 1):
            meta = item.get("metadata") or {}
            score_v = _safe_num(item.get("score"))
            score_str = f"score={score_v:.3f}" if score_v is not None else "score=?"
            rows.append(
                "<div style='border-left:3px solid #aaa; padding:4px 8px; "
                f"margin:4px 0; font-size:0.85em;'>"
                f"<b>[{idx}]</b> {html.escape(str(item.get('id', '')))} "
                f"<span style='color:#666;'>({item.get('retrieval_method', '')} · "
                f"{score_str})</span><br>"
                f"<span style='color:#444;'>{html.escape(str(meta.get('title', '')))}</span>"
                "</div>"
            )

    _block("Dense top-K", dense_results)
    _block("BM25 top-K", bm25_results)
    _block("Final context", final_results)

    if citation_check:
        rows.append("<div style='margin-top:8px;'><b>Citation grounding</b></div>")
        rows.append(
            "<div style='font-size:0.85em;'>"
            f"valid=<code>{citation_check.get('valid')}</code>, "
            f"missing=<code>{citation_check.get('missing_citation')}</code>, "
            f"invalid_refs=<code>{citation_check.get('invalid_refs')}</code>, "
            f"grounded_ratio=<code>{citation_check.get('grounded_ratio')}</code>, "
            f"total_refs=<code>{citation_check.get('total_refs')}</code>"
            "</div>"
        )

    if diagnostics and isinstance(diagnostics, dict):
        rows.append(
            "<div style='margin-top:8px;'><b>Retrieval diagnostics</b></div>"
        )
        rows.append(
            "<div style='font-size:0.85em;'>"
            f"dense_topk_count=<code>{len(diagnostics.get('dense_topk', []))}</code>, "
            f"bm25_topk_count=<code>{len(diagnostics.get('bm25_topk', []))}</code>, "
            f"rrf_ranking_count=<code>{len(diagnostics.get('rrf_ranking', []))}</code>"
            "</div>"
        )

    if not rows:
        return ""
    return (
        "<div style='border:1px dashed #999; background:#fafafa; padding:8px 10px; "
        "border-radius:6px; margin-top:6px;'>"
        "<div style='font-weight:600;'>🔍 Debug / Observability</div>"
        + "".join(rows)
        + "</div>"
    )


def render_analytics_table(rows: list[dict[str, Any]]) -> str:
    """Bảng HTML nhỏ cho retrieval analytics (Bonus 10)."""
    if not rows:
        return ""
    head = (
        "<tr><th>Strategy</th><th>CP</th><th>CR</th><th>Hit</th><th>Div</th>"
        "<th>Latency(ms)</th></tr>"
    )
    body = ""
    for row in rows:
        body += (
            "<tr>"
            f"<td>{html.escape(str(row.get('strategy', '')))}</td>"
            f"<td>{row.get('context_precision', 0):.3f}</td>"
            f"<td>{row.get('context_recall', 0):.3f}</td>"
            f"<td>{row.get('source_hit_rate', 0):.3f}</td>"
            f"<td>{row.get('doc_diversity_avg', 0):.2f}</td>"
            f"<td>{row.get('avg_latency_ms', 0):.1f}</td>"
            "</tr>"
        )
    return (
        "<table style='border-collapse:collapse; font-size:0.9em;'>"
        f"<thead>{head}</thead><tbody>{body}</tbody></table>"
    )
