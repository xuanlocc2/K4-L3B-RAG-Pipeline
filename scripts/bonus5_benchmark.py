"""Benchmark Bonus 5: Conversation memory.

Đo:
    A = standalone query, không rewrite.
    B = standalone query CÓ conversation memory bật (passthrough vì
        đã là câu độc lập, kỳ vọng metric giữ nguyên).
    C = follow-up query ngắn (giả lập) với conversation memory.

Mục tiêu: Bonus 5 KHÔNG được làm giảm chất lượng standalone queries
(Requirement: "memory feature must not degrade standalone query performance")
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.bonus5_conversation_memory import rewrite_followup  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import rerank_weighted_rrf  # noqa: E402

GOLDEN = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUT = ROOT / "group_project" / "evaluation" / "bonus5_benchmark.json"
TOP_K = 5


def _expected(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _hit(ids, expected):
    return any(e in rid for rid in ids for e in expected if e)


def _precision(ids, expected):
    if not ids or not expected:
        return 0.0
    return sum(1 for rid in ids if any(e in rid for e in expected if e)) / len(ids)


def _recall(ids, expected):
    if not expected:
        return 0.0
    s = set(ids)
    return sum(1 for e in expected if any(e in r for r in s if e)) / len(expected)


def _retrieve(q: str) -> list[dict]:
    d = semantic_search(q, top_k=TOP_K * 2)
    b = lexical_search(q, top_k=TOP_K * 2)
    return rerank_weighted_rrf([d, b], top_k=TOP_K, dense_weight=1.5, bm25_weight=1.0)


def _aggregate(per_case):
    n = len(per_case) or 1
    return {
        "n": len(per_case),
        "context_precision": sum(c["cp"] for c in per_case) / n,
        "context_recall": sum(c["cr"] for c in per_case) / n,
        "source_hit_rate": sum(c["hit"] for c in per_case) / n,
        "rewrites": sum(1 for c in per_case if c["rewrote"]),
        "passthrough": sum(1 for c in per_case if not c["rewrote"]),
    }


def main():
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))

    standalone = []
    standalone_with_memory = []
    followup_with_memory = []
    # giả lập history có 1 turn trước cùng chủ đề "điều kiện xét tuyển"
    fake_history = [
        {"role": "user", "content": "Điều kiện xét tuyển đại học gồm những yêu cầu nào?"},
    ]

    for case in dataset:
        q = case["question"]
        exp = _expected(case)

        # A: standalone, no memory
        t0 = time.perf_counter()
        rec = _retrieve(q)
        ids = [r["id"] for r in rec]
        standalone.append({
            "id": case["id"], "cp": _precision(ids, exp),
            "cr": _recall(ids, exp), "hit": _hit(ids, exp),
            "rewrote": False,
        })

        # B: standalone + memory (passthrough)
        rewrite_info = rewrite_followup(fake_history, q, use_llm=False)
        effective_q = rewrite_info["standalone_query"]
        rec_b = _retrieve(effective_q)
        ids_b = [r["id"] for r in rec_b]
        standalone_with_memory.append({
            "id": case["id"], "cp": _precision(ids_b, exp),
            "cr": _recall(ids_b, exp), "hit": _hit(ids_b, exp),
            "rewrote": rewrite_info["rewrote"],
        })

        # C: biến câu thành follow-up ngắn để test rewrite.
        followup_q = "Còn vậy thì sao?"
        # tạo history riêng cho từng case từ topic đã biết
        per_case_history = [
            {"role": "user", "content": q},  # conversation trước đó về chính case này
        ]
        rewrite_info_c = rewrite_followup(per_case_history, followup_q, use_llm=False)
        rec_c = _retrieve(rewrite_info_c["standalone_query"])
        ids_c = [r["id"] for r in rec_c]
        followup_with_memory.append({
            "id": case["id"], "cp": _precision(ids_c, exp),
            "cr": _recall(ids_c, exp), "hit": _hit(ids_c, exp),
            "rewrote": rewrite_info_c["rewrote"],
            "rewritten_query": rewrite_info_c["standalone_query"],
        })

    out = {
        "A_standalone_no_memory": _aggregate(standalone),
        "B_standalone_with_memory_passthrough": _aggregate(standalone_with_memory),
        "C_followup_with_memory": _aggregate(followup_with_memory),
        "per_case_followup": followup_with_memory,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {k: _aggregate(v) if isinstance(v, list) else {kk: vv for kk, vv in v.items() if kk != "per_case_followup"}
               for k, v in out.items() if k != "per_case_followup"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved {OUT.relative_to(ROOT)}")
    # in 3 case followup đầu để demo
    print("\nMột số ví dụ followup rewrite:")
    for c in followup_with_memory[:5]:
        print(f"  [{c['id']}] hit={c['hit']} cp={c['cp']:.2f} -> '{c['rewritten_query']}'")


if __name__ == "__main__":
    main()
