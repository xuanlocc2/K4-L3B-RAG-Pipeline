"""Calibrate the dense-score threshold using a validation set.

We probe both in-domain queries (must succeed) and out-of-domain queries
(must trigger fallback / safe refusal). The script records the best dense
cosine score for each query and recommends a threshold that balances
successful in-domain retrieval with effective fallback for OOD queries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.task5_semantic_search import semantic_search

IN_DOMAIN_QUERIES = [
    "điều kiện xét tuyển đại học",
    "các phương thức xét tuyển",
    "tổ hợp môn xét tuyển",
    "điểm ưu tiên khu vực",
    "các bài thi tốt nghiệp THPT",
    "chứng chỉ IELTS xét tuyển",
    "thông tư 08/2022",
    "hồ sơ dự tuyển đại học",
    "tuyển sinh bổ sung 2026",
    "đại học Mỹ cho sinh viên Việt",
]

OUT_OF_DOMAIN_QUERIES = [
    "công thức nấu phở bò",
    "giá vàng hôm nay",
    "thủ tục đăng ký kết hôn",
    "lịch thi đấu World Cup",
    "cách trồng cây cảnh",
]


def best_score(query: str) -> float:
    results = semantic_search(query, top_k=5)
    if not results:
        return 0.0
    return float(results[0]["score"])


def main() -> None:
    in_domain_scores = [(q, best_score(q)) for q in IN_DOMAIN_QUERIES]
    ood_scores = [(q, best_score(q)) for q in OUT_OF_DOMAIN_QUERIES]

    print("In-domain best dense scores:")
    for q, s in in_domain_scores:
        print(f"  {s:.3f}  {q}")
    print("\nOut-of-domain best dense scores:")
    for q, s in ood_scores:
        print(f"  {s:.3f}  {q}")

    in_min = min(s for _, s in in_domain_scores)
    ood_max = max(s for _, s in ood_scores)
    in_mean = sum(s for _, s in in_domain_scores) / len(in_domain_scores)
    ood_mean = sum(s for _, s in ood_scores) / len(ood_scores)

    # chọn threshold giữa in_min và ood_max, ưu tiên lệch về phía OOD
    # để fallback có cơ hội kích hoạt
    candidates = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    print("\nThreshold candidates:")
    summary = []
    for t in candidates:
        in_below = sum(1 for _, s in in_domain_scores if s < t)
        ood_above = sum(1 for _, s in ood_scores if s >= t)
        summary.append(
            {
                "threshold": t,
                "in_domain_below": in_below,
                "ood_at_or_above": ood_above,
            }
        )
        print(f"  t={t:.2f}  in-domain misses={in_below}  OOD false_keep={ood_above}")

    chosen = max(
        candidates,
        key=lambda t: (
            sum(1 for _, s in in_domain_scores if s >= t),  # maximize in-domain kept
            -sum(1 for _, s in ood_scores if s >= t),  # minimize OOD kept
        ),
    )
    print(f"\nIn-domain min={in_min:.3f}  OOD max={ood_max:.3f}")
    print(f"In-domain mean={in_mean:.3f}  OOD mean={ood_mean:.3f}")
    print(f"Recommended threshold: {chosen:.2f}")

    out = {
        "in_domain_scores": in_domain_scores,
        "ood_scores": ood_scores,
        "candidate_summary": summary,
        "recommended_threshold": chosen,
        "in_domain_min": in_min,
        "ood_max": ood_max,
        "in_domain_mean": in_mean,
        "ood_mean": ood_mean,
    }
    out_path = ROOT / "reports" / "threshold_calibration.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved calibration to {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()