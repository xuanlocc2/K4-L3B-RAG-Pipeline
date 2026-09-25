"""Benchmark Bonus 1 + Bonus 2: A/B/C/D comparison on the golden dataset.

Configurations:
    A = dense-only
    B = dense + BM25 + standard RRF (baseline hybrid)
    C = dense + BM25 + weighted RRF (Bonus 1)
    D = dense + BM25 + weighted RRF + document-level dedup (Bonus 1 + 2)

For each strategy we measure:
    - Context Precision
    - Context Recall
    - Source Hit Rate
    - Average retrieved docs (document diversity)
    - Average retrieval latency (ms)
    - Answered rate (always 0 here vì no LLM key, nhưng vẫn log)

Output:
    group_project/evaluation/bonus_benchmark.json
    reports/BONUS_BENCHMARK.md (table summary)
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
OUT_JSON = ROOT / "group_project" / "evaluation" / "bonus_benchmark.json"
TOP_K = 5


def _expected(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _hit(rec_ids: list[str], expected: list[str]) -> bool:
    if not expected:
        return False
    return any(exp in rid for rid in rec_ids for exp in expected if exp)


def _precision(rec_ids: list[str], expected: list[str]) -> float:
    if not rec_ids or not expected:
        return 0.0
    return sum(
        1 for rid in rec_ids
        if any(exp in rid for exp in expected if exp)
    ) / len(rec_ids)


def _recall(rec_ids: list[str], expected: list[str]) -> float:
    if not expected:
        return 0.0
    rec_set = set(rec_ids)
    return sum(
        1 for exp in expected
        if any(exp in rid for rid in rec_set if exp)
    ) / len(expected)


def _doc_diversity(rec: list[dict]) -> float:
    """Trung bình số document unique trong mỗi kết quả trả về (per-query)."""
    sources = {r["metadata"].get("source", r["id"]) for r in rec}
    return len(sources)


def _strategy(name: str, fn):
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    per_case = []
    for case in dataset:
        q = case["question"]
        exp = _expected(case)
        t0 = time.perf_counter()
        try:
            rec = fn(q)
        except Exception as exc:
            print(f"[{name}] {case['id']} error: {exc}")
            rec = []
        latency_ms = (time.perf_counter() - t0) * 1000.0
        rec_ids = [r["id"] for r in rec]
        per_case.append({
            "id": case["id"],
            "retrieved_ids": rec_ids,
            "source_hit": _hit(rec_ids, exp),
            "context_precision": _precision(rec_ids, exp),
            "context_recall": _recall(rec_ids, exp),
            "doc_diversity": _doc_diversity(rec),
            "latency_ms": latency_ms,
        })

    n = len(per_case) or 1
    return {
        "strategy": name,
        "n_cases": len(per_case),
        "context_precision": sum(c["context_precision"] for c in per_case) / n,
        "context_recall": sum(c["context_recall"] for c in per_case) / n,
        "source_hit_rate": sum(c["source_hit"] for c in per_case) / n,
        "doc_diversity_avg": sum(c["doc_diversity"] for c in per_case) / n,
        "avg_latency_ms": sum(c["latency_ms"] for c in per_case) / n,
        "per_case": per_case,
    }


def _dense_only(q: str) -> list[dict]:
    return semantic_search(q, top_k=TOP_K)


def _baseline_rrf(q: str) -> list[dict]:
    dense = semantic_search(q, top_k=TOP_K * 2)
    bm = lexical_search(q, top_k=TOP_K * 2)
    return rerank_rrf([dense, bm], top_k=TOP_K)


def _weighted_rrf(q: str) -> list[dict]:
    dense = semantic_search(q, top_k=TOP_K * 2)
    bm = lexical_search(q, top_k=TOP_K * 2)
    return rerank_weighted_rrf(
        [dense, bm], top_k=TOP_K,
        dense_weight=1.0, bm25_weight=0.7,
    )


def _weighted_rrf_dedup(q: str) -> list[dict]:
    fetch = TOP_K * 2
    dense = semantic_search(q, top_k=fetch)
    bm = lexical_search(q, top_k=fetch)
    fused = rerank_weighted_rrf(
        [dense, bm], top_k=fetch,
        dense_weight=1.0, bm25_weight=0.7,
    )
    return deduplicate_by_document(fused, top_k=TOP_K)


def main() -> None:
    if not GOLDEN.exists():
        print(f"Missing {GOLDEN}")
        sys.exit(1)
    print(f"Benchmarking Bonus 1 + 2 on {GOLDEN.name} (top_k={TOP_K})...\n")

    results = {
        "A_dense_only": _strategy("A_dense_only", _dense_only),
        "B_baseline_rrf": _strategy("B_baseline_rrf", _baseline_rrf),
        "C_weighted_rrf": _strategy("C_weighted_rrf", _weighted_rrf),
        "D_weighted_rrf_dedup": _strategy("D_weighted_rrf_dedup", _weighted_rrf_dedup),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {OUT_JSON.relative_to(ROOT)}")

    # in bảng tóm tắt
    print("\n| Strategy                | CP    | CR    | Hit   | Div | Latency(ms) |")
    print("| ----------------------- | ----- | ----- | ----- | --- | ----------- |")
    for key in ("A_dense_only", "B_baseline_rrf", "C_weighted_rrf", "D_weighted_rrf_dedup"):
        r = results[key]
        print(
            f"| {key:<23} | {r['context_precision']:.3f} | {r['context_recall']:.3f} "
            f"| {r['source_hit_rate']:.3f} | {r['doc_diversity_avg']:.2f} "
            f"| {r['avg_latency_ms']:>11.1f} |"
        )


if __name__ == "__main__":
    main()
