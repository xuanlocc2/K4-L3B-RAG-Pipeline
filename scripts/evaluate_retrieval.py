"""Retrieval-only evaluation: chạy dense vs hybrid trên golden dataset,
TÍNH METRICS TỪ KẾT QUẢ THẬT. Không gọi LLM để chạy nhanh.

Output: group_project/evaluation/retrieval_evaluation.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf

GOLDEN_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
RESULTS_PATH = ROOT / "group_project" / "evaluation" / "retrieval_evaluation.json"
TOP_K = 5


def _expected_sources_from_case(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _source_hit(retrieved: list[dict], expected: list[str]) -> bool:
    if not expected:
        return False
    for r in retrieved:
        for exp in expected:
            if exp and exp in r["id"]:
                return True
    return False


def _context_precision(retrieved: list[dict], expected: list[str]) -> float:
    if not retrieved or not expected:
        return 0.0
    hits = sum(
        1 for r in retrieved
        if any(exp in r["id"] for exp in expected)
    )
    return hits / len(retrieved)


def _context_recall(retrieved: list[dict], expected: list[str]) -> float:
    if not expected:
        return 0.0
    retrieved_ids = {r["id"] for r in retrieved}
    hits = sum(
        1 for exp in expected
        if any(exp in rid for rid in retrieved_ids)
    )
    return hits / len(expected)


def _evaluate(retrieval_fn, dataset: list[dict]) -> dict:
    rows = []
    for case in dataset:
        expected = _expected_sources_from_case(case)
        start = time.perf_counter()
        try:
            results = retrieval_fn(case["question"], TOP_K)
        except Exception as exc:
            print(f"  ERR {case.get('id')}: {exc}")
            results = []
        latency = (time.perf_counter() - start) * 1000.0
        rows.append({
            "id": case["id"],
            "category": case.get("category", ""),
            "difficulty": case.get("difficulty", ""),
            "expected_sources": expected,
            "retrieved_ids": [r["id"] for r in results],
            "top1_id": results[0]["id"] if results else None,
            "top1_score": float(results[0]["score"]) if results else 0.0,
            "context_precision": _context_precision(results, expected),
            "context_recall": _context_recall(results, expected),
            "source_hit": _source_hit(results, expected),
            "latency_ms": latency,
        })

    in_domain = [r for r in rows if r["expected_sources"]]
    ood = [r for r in rows if not r["expected_sources"]]
    total = len(rows)

    def _mean(items, key):
        return sum(r[key] for r in items) / max(len(items), 1)

    return {
        "aggregate": {
            "total_cases": total,
            "in_domain_cases": len(in_domain),
            "ood_cases": len(ood),
            "context_precision_overall": _mean(rows, "context_precision"),
            "context_recall_overall": _mean(rows, "context_recall"),
            "context_precision_in_domain": _mean(in_domain, "context_precision"),
            "context_recall_in_domain": _mean(in_domain, "context_recall"),
            "source_hit_rate_in_domain": _mean(in_domain, "source_hit"),
            "avg_latency_ms": _mean(rows, "latency_ms"),
            "avg_top1_score_in_domain": _mean(in_domain, "top1_score"),
            "avg_top1_score_ood": _mean(ood, "top1_score"),
        },
        "per_case": rows,
    }


def main() -> None:
    if not GOLDEN_PATH.exists():
        print(f"Missing golden dataset: {GOLDEN_PATH}")
        sys.exit(1)

    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    print(f"Evaluating {len(dataset)} cases (top_k={TOP_K})...")

    # A: dense-only
    print("\n--- Strategy A: dense-only ---")
    dense_only = _evaluate(
        lambda q, k: semantic_search(q, top_k=k), dataset,
    )
    agg_a = dense_only["aggregate"]
    print(json.dumps(agg_a, indent=2, ensure_ascii=False))

    # B: hybrid (dense + BM25 + RRF)
    print("\n--- Strategy B: hybrid (dense + BM25 + RRF) ---")
    def _hybrid(q, k):
        dense = semantic_search(q, top_k=k)
        bm = lexical_search(q, top_k=k)
        return rerank_rrf([dense, bm], top_k=k)
    hybrid = _evaluate(_hybrid, dataset)
    agg_b = hybrid["aggregate"]
    print(json.dumps(agg_b, indent=2, ensure_ascii=False))

    # Winner
    print("\n--- Verdict ---")
    if agg_b["context_precision_in_domain"] >= agg_a["context_precision_in_domain"]:
        print(
            f"HYBRID wins: precision {agg_b['context_precision_in_domain']:.3f} "
            f">= DENSE {agg_a['context_precision_in_domain']:.3f}"
        )
    else:
        print(
            f"DENSE wins: precision {agg_a['context_precision_in_domain']:.3f} "
            f"> HYBRID {agg_b['context_precision_in_domain']:.3f}. "
            "Honest failure analysis: hybrid adds BM25 noise on this corpus."
        )

    out = {
        "config": {
            "top_k": TOP_K,
            "embedding_model": "BAAI/bge-m3",
        },
        "strategy_a_dense_only": dense_only,
        "strategy_b_hybrid_rrf": hybrid,
    }
    RESULTS_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
