"""Quick grid search over dense_weight / bm25_weight for Weighted RRF.

Tìm cặp (dw, bw) tốt nhất trên tập golden_dataset. Kết quả dùng để chọn
default cho retrieve() nếu tốt hơn baseline.
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
from src.task7_reranking import rerank_weighted_rrf  # noqa: E402

GOLDEN = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
TOP_K = 5


def _expected(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _metrics(dataset, fn):
    cps, crs, hits = [], [], []
    for case in dataset:
        expected = _expected(case)
        rec = fn(case["question"])
        ids = [r["id"] for r in rec]
        if not expected or not ids:
            cps.append(0.0); crs.append(0.0); hits.append(False); continue
        cp = sum(1 for rid in ids if any(e in rid for e in expected if e)) / len(ids)
        rec_set = set(ids)
        cr = sum(1 for e in expected if any(e in rid for rid in rec_set if e)) / len(expected)
        hit = any(e in rid for rid in ids for e in expected if e)
        cps.append(cp); crs.append(cr); hits.append(hit)
    n = len(cps) or 1
    return sum(cps)/n, sum(crs)/n, sum(hits)/n


def main() -> None:
    dataset = json.loads(GOLDEN.read_text(encoding="utf-8"))

    def make_fn(dw: float, bw: float):
        def _fn(q):
            d = semantic_search(q, top_k=TOP_K * 2)
            s = lexical_search(q, top_k=TOP_K * 2)
            return rerank_weighted_rrf(
                [d, s], top_k=TOP_K,
                dense_weight=dw, bm25_weight=bw,
            )
        return _fn

    grid = [
        (1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0),
        (1.0, 0.5), (1.0, 0.7), (1.5, 0.5), (2.0, 0.7),
        (1.0, 0.3), (0.7, 1.0), (0.5, 1.0),
    ]
    print(f"| dense_weight | bm25_weight | CP    | CR    | Hit   |")
    print(f"| ------------ | ----------- | ----- | ----- | ----- |")
    best = None
    for dw, bw in grid:
        cp, cr, hit = _metrics(dataset, make_fn(dw, bw))
        print(f"| {dw:<12} | {bw:<11} | {cp:.3f} | {cr:.3f} | {hit:.3f} |")
        score = cp + cr + hit  # tổng đơn giản
        if best is None or score > best["score"]:
            best = {"dw": dw, "bw": bw, "cp": cp, "cr": cr, "hit": hit, "score": score}
    print(f"\nBest: dense_weight={best['dw']}, bm25_weight={best['bw']} "
          f"(CP={best['cp']:.3f}, CR={best['cr']:.3f}, Hit={best['hit']:.3f})")
    # Lưu default recommendation
    out = {"recommendation": best, "grid": [{"dw": d, "bw": b} for d, b in grid]}
    (ROOT / "group_project" / "evaluation" / "weighted_rrf_grid.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
