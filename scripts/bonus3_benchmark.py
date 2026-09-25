"""Benchmark Bonus 3: Query expansion.

So sánh 3 chế độ trên golden_dataset:
    A = baseline (không expansion)
    B = bigram expansion (local, không LLM)
    C = hyde (LLM nếu có key, fallback local nếu không)

Vì không có LLM key trong môi trường này, C sẽ chạy local fallback.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.bonus3_query_expansion import expand_query  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import rerank_weighted_rrf  # noqa: E402

GOLDEN = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUT = ROOT / "group_project" / "evaluation" / "bonus3_benchmark.json"
TOP_K = 5


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


def _retrieve(q: str) -> list[dict]:
    d = semantic_search(q, top_k=TOP_K * 2)
    b = lexical_search(q, top_k=TOP_K * 2)
    return rerank_weighted_rrf([d, b], top_k=TOP_K, dense_weight=1.5, bm25_weight=1.0)


def _aggregate(rows):
    n = len(rows) or 1
    return {
        "n": len(rows),
        "context_precision": sum(r["cp"] for r in rows) / n,
        "context_recall": sum(r["cr"] for r in rows) / n,
        "source_hit_rate": sum(r["hit"] for r in rows) / n,
    }


def main():
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))
    a, b, c = [], [], []
    examples = []
    for case in dataset:
        q = case["question"]
        exp = _expected(case)
        # A: baseline
        rec_a = _retrieve(q)
        ids_a = [r["id"] for r in rec_a]
        a.append({"id": case["id"], "cp": _precision(ids_a, exp),
                  "cr": _recall(ids_a, exp), "hit": _hit(ids_a, exp)})
        # B: expansion
        out_b = expand_query(q, mode="expansion")
        rec_b = _retrieve(out_b["expanded_query"])
        ids_b = [r["id"] for r in rec_b]
        b.append({"id": case["id"], "cp": _precision(ids_b, exp),
                  "cr": _recall(ids_b, exp), "hit": _hit(ids_b, exp)})
        # C: hyde
        out_c = expand_query(q, mode="hyde")
        rec_c = _retrieve(out_c["expanded_query"])
        ids_c = [r["id"] for r in rec_c]
        c.append({"id": case["id"], "cp": _precision(ids_c, exp),
                  "cr": _recall(ids_c, exp), "hit": _hit(ids_c, exp)})
        if case["id"] in ("Q01", "Q08", "Q13", "Q19"):
            examples.append({
                "id": case["id"],
                "original": q,
                "expanded_b": out_b["expanded_query"],
                "expanded_c": out_c["expanded_query"],
                "expanded_c_method": out_c["method"],
            })

    summary = {
        "A_baseline": _aggregate(a),
        "B_bigram_expansion": _aggregate(b),
        "C_hyde": _aggregate(c),
        "expansion_examples": examples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "expansion_examples"},
                     ensure_ascii=False, indent=2))
    print("\nSaved", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
