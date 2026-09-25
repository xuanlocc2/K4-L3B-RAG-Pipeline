"""Benchmark Bonus 4: Reranker (lightweight).

So sánh:
    A = Hybrid + RRF (baseline)
    B = Hybrid + RRF + Cosine reranker (top-10 -> top-5)
    C = Hybrid + RRF + Keyword overlap reranker

Pipeline shape (Bonus 4 rule):
    Dense -> BM25 -> RRF -> top-N candidates -> reranker -> top-K final
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.bonus4_reranker import (  # noqa: E402
    rerank_cosine,
    rerank_keyword_overlap,
)
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import rerank_rrf  # noqa: E402

GOLDEN = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUT = ROOT / "group_project" / "evaluation" / "bonus4_benchmark.json"
TOP_K = 5
CANDIDATE_POOL = 10


def _expected(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _hit(ids, exp):
    return any(e in rid for rid in ids for e in exp if e)


def _precision(ids, exp):
    if not ids or not exp:
        return 0.0
    return sum(1 for r in ids if any(e in r for e in exp if e)) / len(ids)


def _recall(ids, exp):
    if not exp:
        return 0.0
    s = set(ids)
    return sum(1 for e in exp if any(e in r for r in s if e)) / len(exp)


def _aggregate(rows):
    n = len(rows) or 1
    return {
        "n": len(rows),
        "context_precision": sum(r["cp"] for r in rows) / n,
        "context_recall": sum(r["cr"] for r in rows) / n,
        "source_hit_rate": sum(r["hit"] for r in rows) / n,
        "avg_latency_ms": sum(r["latency_ms"] for r in rows) / n,
    }


def main():
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    a, b, c = [], [], []

    for case in dataset:
        q = case["question"]
        exp = _expected(case)

        # A: hybrid RRF no rerank
        t0 = time.perf_counter()
        d = semantic_search(q, top_k=TOP_K)
        b_ = lexical_search(q, top_k=TOP_K)
        a_rec = rerank_rrf([d, b_], top_k=TOP_K)
        a_ms = (time.perf_counter() - t0) * 1000.0
        a_ids = [r["id"] for r in a_rec]
        a.append({"cp": _precision(a_ids, exp), "cr": _recall(a_ids, exp),
                  "hit": _hit(a_ids, exp), "latency_ms": a_ms})

        # B: hybrid RRF + cosine rerank trên top-10
        t0 = time.perf_counter()
        d2 = semantic_search(q, top_k=CANDIDATE_POOL)
        b2 = lexical_search(q, top_k=CANDIDATE_POOL)
        fused = rerank_rrf([d2, b2], top_k=CANDIDATE_POOL)
        b_rec = rerank_cosine(q, fused, top_k=TOP_K)
        b_ms = (time.perf_counter() - t0) * 1000.0
        b_ids = [r["id"] for r in b_rec]
        b.append({"cp": _precision(b_ids, exp), "cr": _recall(b_ids, exp),
                  "hit": _hit(b_ids, exp), "latency_ms": b_ms})

        # C: hybrid RRF + keyword overlap rerank
        t0 = time.perf_counter()
        d3 = semantic_search(q, top_k=CANDIDATE_POOL)
        b3 = lexical_search(q, top_k=CANDIDATE_POOL)
        fused3 = rerank_rrf([d3, b3], top_k=CANDIDATE_POOL)
        c_rec = rerank_keyword_overlap(q, fused3, top_k=TOP_K)
        c_ms = (time.perf_counter() - t0) * 1000.0
        c_ids = [r["id"] for r in c_rec]
        c.append({"cp": _precision(c_ids, exp), "cr": _recall(c_ids, exp),
                  "hit": _hit(c_ids, exp), "latency_ms": c_ms})

    summary = {
        "A_hybrid_rrf_no_rerank": _aggregate(a),
        "B_hybrid_rrf_cosine_rerank": _aggregate(b),
        "C_hybrid_rrf_keyword_rerank": _aggregate(c),
        "config": {"top_k": TOP_K, "candidate_pool": CANDIDATE_POOL,
                   "n_cases": len(dataset)},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
