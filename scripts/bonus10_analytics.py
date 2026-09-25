"""Bonus 10 — Retrieval analytics utility.

Tính và in báo cáo analytics trên golden_dataset:
    * Dense-only hit rate
    * BM25-only hit rate
    * Hybrid (RRF) hit rate
    * Weighted RRF hit rate
    * Document-level dedup hit rate
    * RRF contribution (chunk xuất hiện ở cả 2 bảng vs chỉ 1)
    * Fallback frequency (best dense score < threshold)
    * Average retrieved documents
    * Document diversity (unique sources / query)
    * Average retrieval latency

Output:
    group_project/evaluation/bonus10_analytics.json
    group_project/evaluation/bonus10_analytics.md
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import (  # noqa: E402
    deduplicate_by_document,
    rerank_rrf,
    rerank_weighted_rrf,
)

GOLDEN = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUT_JSON = ROOT / "group_project" / "evaluation" / "bonus10_analytics.json"
OUT_MD = ROOT / "group_project" / "evaluation" / "bonus10_analytics.md"
TOP_K = 5
SCORE_THRESHOLD = 0.5


def _expected(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _hit(ids, expected):
    return any(e in rid for rid in ids for e in expected if e)


def _safe_metrics(retrieved: list[dict], expected: list[str]) -> dict:
    ids = [r["id"] for r in retrieved]
    diversity = len({r["metadata"].get("source", r["id"]) for r in retrieved})
    if not expected:
        return {"hit": False, "cp": 0.0, "cr": 0.0, "n": len(ids), "diversity": diversity}
    cp = sum(1 for rid in ids if any(e in rid for e in expected if e)) / max(len(ids), 1)
    s = set(ids)
    cr = sum(1 for e in expected if any(e in r for r in s if e)) / len(expected)
    return {
        "hit": _hit(ids, expected),
        "cp": cp,
        "cr": cr,
        "n": len(ids),
        "diversity": diversity,
    }


def main() -> None:
    if not GOLDEN.exists():
        print(f"Missing {GOLDEN}")
        sys.exit(1)
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    print(f"Computing retrieval analytics over {len(dataset)} cases...\n")

    stats: dict[str, list[dict]] = {
        "dense_only": [],
        "bm25_only": [],
        "hybrid_rrf": [],
        "weighted_rrf": [],
        "weighted_rrf_dedup": [],
    }
    fallback_count = 0
    rrf_contrib: dict[str, int] = {"both": 0, "dense_only": 0, "bm25_only": 0}

    for case in dataset:
        q = case["question"]
        exp = _expected(case)

        t0 = time.perf_counter()
        dense = semantic_search(q, top_k=TOP_K * 2)
        bm = lexical_search(q, top_k=TOP_K * 2)
        retrieval_latency_ms = (time.perf_counter() - t0) * 1000.0

        # RRF contribution: đếm id xuất hiện ở cả hai, chỉ dense, chỉ bm25.
        dense_ids = {r["id"] for r in dense}
        bm_ids = {r["id"] for r in bm}
        for rid in (dense_ids & bm_ids):
            pass  # counted below
        for rid in (dense_ids | bm_ids):
            if rid in dense_ids and rid in bm_ids:
                rrf_contrib["both"] += 1
            elif rid in dense_ids:
                rrf_contrib["dense_only"] += 1
            else:
                rrf_contrib["bm25_only"] += 1

        # Fallback frequency: dense top-1 < threshold
        best_dense = dense[0]["score"] if dense else 0.0
        if best_dense < SCORE_THRESHOLD:
            fallback_count += 1

        stats["dense_only"].append(_safe_metrics(dense[:TOP_K], exp))
        stats["bm25_only"].append(_safe_metrics(bm[:TOP_K], exp))
        stats["hybrid_rrf"].append(
            _safe_metrics(rerank_rrf([dense, bm], top_k=TOP_K), exp)
        )
        stats["weighted_rrf"].append(
            _safe_metrics(
                rerank_weighted_rrf(
                    [dense, bm], top_k=TOP_K,
                    dense_weight=1.5, bm25_weight=1.0,
                ),
                exp,
            )
        )
        stats["weighted_rrf_dedup"].append(
            _safe_metrics(
                deduplicate_by_document(
                    rerank_weighted_rrf(
                        [dense, bm], top_k=TOP_K * 2,
                        dense_weight=1.5, bm25_weight=1.0,
                    ),
                    top_k=TOP_K,
                ),
                exp,
            )
        )

    def _agg(rows: list[dict]) -> dict:
        n = len(rows) or 1
        return {
            "hit_rate": sum(r["hit"] for r in rows) / n,
            "context_precision": sum(r["cp"] for r in rows) / n,
            "context_recall": sum(r["cr"] for r in rows) / n,
            "avg_docs": sum(r["n"] for r in rows) / n,
            "doc_diversity": sum(r["diversity"] for r in rows) / n,
        }

    summary = {key: _agg(rows) for key, rows in stats.items()}
    summary["fallback_frequency"] = fallback_count / max(len(dataset), 1)
    summary["rrf_contribution"] = {
        k: v / max(len(dataset), 1) for k, v in rrf_contrib.items()
    }
    summary["config"] = {
        "top_k": TOP_K,
        "score_threshold": SCORE_THRESHOLD,
        "n_cases": len(dataset),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Markdown summary
    lines = [
        "# Retrieval Analytics (Bonus 10)",
        "",
        f"- Số case: **{summary['config']['n_cases']}**",
        f"- top_k: **{TOP_K}**, score_threshold: **{SCORE_THRESHOLD}**",
        "",
        "## Per-strategy metrics",
        "",
        "| Strategy | Hit | CP | CR | Avg docs | Doc diversity |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in ("dense_only", "bm25_only", "hybrid_rrf", "weighted_rrf", "weighted_rrf_dedup"):
        m = summary[key]
        lines.append(
            f"| `{key}` | {m['hit_rate']:.3f} | {m['context_precision']:.3f} "
            f"| {m['context_recall']:.3f} | {m['avg_docs']:.2f} "
            f"| {m['doc_diversity']:.2f} |"
        )
    lines += [
        "",
        "## Hybrid contribution",
        "",
        "Trung bình mỗi case có bao nhiêu chunk xuất hiện ở cả 2 bảng xếp hạng.",
        "",
        f"- Cả dense + BM25: **{summary['rrf_contribution']['both']:.2f}** chunk/case",
        f"- Chỉ dense: **{summary['rrf_contribution']['dense_only']:.2f}** chunk/case",
        f"- Chỉ BM25: **{summary['rrf_contribution']['bm25_only']:.2f}** chunk/case",
        "",
        "## Fallback frequency",
        "",
        f"Tỉ lệ câu có best_dense_score < {SCORE_THRESHOLD}: "
        f"**{summary['fallback_frequency']:.2%}**",
        "",
        f"Dữ liệu JSON đầy đủ: `group_project/evaluation/bonus10_analytics.json`.",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved {OUT_JSON.relative_to(ROOT)} and {OUT_MD.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
