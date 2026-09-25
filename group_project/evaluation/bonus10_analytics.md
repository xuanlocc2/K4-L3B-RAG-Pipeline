# Retrieval Analytics (Bonus 10)

- Số case: **22**
- top_k: **5**, score_threshold: **0.5**

## Per-strategy metrics

| Strategy | Hit | CP | CR | Avg docs | Doc diversity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense_only` | 0.909 | 0.509 | 0.841 | 5.00 | 2.23 |
| `bm25_only` | 0.818 | 0.427 | 0.727 | 5.00 | 2.64 |
| `hybrid_rrf` | 0.864 | 0.491 | 0.795 | 5.00 | 2.32 |
| `weighted_rrf` | 0.909 | 0.500 | 0.841 | 5.00 | 2.27 |
| `weighted_rrf_dedup` | 0.864 | 0.430 | 0.795 | 3.00 | 3.00 |

## Hybrid contribution

Trung bình mỗi case có bao nhiêu chunk xuất hiện ở cả 2 bảng xếp hạng.

- Cả dense + BM25: **4.45** chunk/case
- Chỉ dense: **5.55** chunk/case
- Chỉ BM25: **5.55** chunk/case

## Fallback frequency

Tỉ lệ câu có best_dense_score < 0.5: **9.09%**

Dữ liệu JSON đầy đủ: `group_project/evaluation/bonus10_analytics.json`.