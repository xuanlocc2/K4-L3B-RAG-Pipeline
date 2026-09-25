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


def render_citation_panel(
    sources: list[dict[str, Any]],
    *,
    highlight_passage: bool = True,
    passage_max_chars: int = 320,
) -> str:
    """Trả về HTML cho citation panel.

    Format (Bonus 6):
        Citation [1]
        ────────────
        Document title
        Source: ...

        Relevant passage
        > "...exact supporting passage..."

        Source URL
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
        score = float(src.get("score", 0.0))

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
        blocks.append(
            "<div style='border:1px solid #ddd; border-radius:6px; "
            "padding:10px 12px; margin-bottom:8px; background:#fff;'>"
            f"<div style='font-weight:600;'>Citation [{idx}]</div>"
            f"<div style='font-weight:500; margin-top:2px;'>{html.escape(title)}</div>"
            f"<div style='font-size:0.85em; color:#444;'>Source: "
            f"{html.escape(str(source))}</div>"
            f"<div style='font-size:0.8em; color:#666;'>"
            f"method={html.escape(str(method))} · score={score:.3f}</div>"
            f"{passage_html}"
            f"{url_html}"
            "</div>"
        )
    return "<div>" + "".join(blocks) + "</div>"


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
) -> str:
    """Trả về HTML cho debug / observability panel (Bonus 9).

    Hiển thị:
        - Query (effective vs original)
        - Retrieval mode + threshold
        - Optional: top dense, top BM25, final
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

    _row("Effective query", query)
    if original_query and original_query != query:
        _row("Original query", original_query)
    if rewrite_info:
        _row("Rewrite method", rewrite_info.get("method"))
        _row("Last topic", rewrite_info.get("last_topic"))
    _row("Retrieval mode", retrieval_mode)
    _row("Score threshold", score_threshold)
    _row("Retrieval source", retrieval_source)

    def _block(title: str, items: list[dict[str, Any]] | None) -> None:
        if not items:
            return
        rows.append(f"<div style='margin-top:8px;'><b>{html.escape(title)}</b></div>")
        for idx, item in enumerate(items, 1):
            meta = item.get("metadata") or {}
            rows.append(
                "<div style='border-left:3px solid #aaa; padding:4px 8px; "
                f"margin:4px 0; font-size:0.85em;'>"
                f"<b>[{idx}]</b> {html.escape(str(item.get('id', '')))} "
                f"<span style='color:#666;'>({item.get('retrieval_method', '')} · "
                f"score={float(item.get('score', 0.0)):.3f})</span><br>"
                f"<span style='color:#444;'>{html.escape(str(meta.get('title', '')))}</span>"
                "</div>"
            )

    _block("Dense results", dense_results)
    _block("BM25 results", bm25_results)
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
