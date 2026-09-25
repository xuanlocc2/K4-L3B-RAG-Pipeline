"""Multi-query smoke test: 5 loại câu hỏi khác nhau để chứng minh retrieval
thật phản hồi theo nội dung, không phải mock.

Các loại câu hỏi:
    1. Factual:        "Điều kiện xét tuyển đại học là gì?"
    2. Paraphrase:     "Tôi có thể dùng chứng chỉ IELTS để vào đại học không?"
    3. Keyword-heavy:  "IELTS 6.5 xét tuyển 2025"
    4. Multi-document: "Những phương thức xét tuyển đại học hiện nay là gì?"
    5. OOD:            "Giá Bitcoin hôm nay là bao nhiêu?" — phải fallback safe.

Usage:
    python scripts/multi_query_test.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf
from src.task9_retrieval_pipeline import DEFAULT_SCORE_THRESHOLD, retrieve


QUERIES = [
    ("Factual",        "Điều kiện xét tuyển đại học là gì?"),
    ("Paraphrase",     "Tôi có thể dùng chứng chỉ IELTS để vào đại học không?"),
    ("Keyword-heavy",  "IELTS 6.5 xét tuyển 2025"),
    ("Multi-document", "Những phương thức xét tuyển đại học hiện nay là gì?"),
    ("OOD",            "Giá Bitcoin hôm nay là bao nhiêu?"),
]


def _show(label: str, query: str, results: list[dict]) -> dict:
    print(f"\n--- {label} ---")
    print(f"Q: {query}")
    print(f"top-{len(results)} chunks:")
    top1_score = float(results[0]["score"]) if results else 0.0
    for idx, r in enumerate(results[:3], 1):
        meta = r.get("metadata") or {}
        print(
            f"  [{idx}] score={float(r.get('score', 0.0)):.4f} "
            f"method={r.get('retrieval_method', '')} "
            f"id={r.get('id', '')}"
        )
        print(f"       title : {meta.get('title', '')[:80]}")
    return {
        "label": label,
        "query": query,
        "top1_score": top1_score,
        "top1_id": results[0]["id"] if results else None,
        "results": [
            {"id": r["id"], "score": float(r["score"]),
             "method": r["retrieval_method"], "title": (r.get("metadata") or {}).get("title", "")}
            for r in results
        ],
    }


def main() -> None:
    print("=" * 70)
    print(f"Multi-query retrieval test (threshold={DEFAULT_SCORE_THRESHOLD})")
    print("=" * 70)

    report = []
    for label, query in QUERIES:
        # DENSE
        dense = semantic_search(query, top_k=5)
        # BM25
        bm = lexical_search(query, top_k=5)
        # HYBRID qua pipeline Task 9 (xử lý threshold + fallback)
        fused_pipeline = retrieve(query, top_k=5, score_threshold=DEFAULT_SCORE_THRESHOLD)

        print(f"\n## {label}")
        dense_summary = _show("DENSE", query, dense)
        bm_summary = _show("BM25", query, bm)
        pipe_summary = _show(f"HYBRID (Task9 pipeline, threshold={DEFAULT_SCORE_THRESHOLD})",
                             query, fused_pipeline)

        report.append({
            "label": label,
            "query": query,
            "dense": dense_summary,
            "bm25": bm_summary,
            "pipeline": pipe_summary,
        })

    # Summary
    print("\n" + "=" * 70)
    print("Top-1 source per query")
    print("=" * 70)
    for entry in report:
        d = entry["dense"]
        print(
            f"  {entry['label']:15s} | dense_score={d['top1_score']:.4f} | "
            f"id={d['top1_id']}"
        )

    # Lưu kết quả JSON để audit
    out_path = ROOT / "reports" / "multi_query_test.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved report to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
