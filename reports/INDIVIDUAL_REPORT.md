# Individual contribution report — Hung

---

## Thông tin

- Họ và tên: Lê Đức Hùng
- Mã học viên: 2A202602849
- Nhóm: LaoGaKho
- Repository/branch: hung

---

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Semantic search (dense retrieval) | Implement `semantic_search()` — embed query, query ChromaDB, cosine distance → similarity | `src/task5_semantic_search.py` | Done |
| Lexical search (BM25) | Implement `lexical_search()` — BM25Plus trên corpus chunks, trả về scored results | `src/task6_lexical_search.py` | Done |
| RRF reranking | Implement `rerank_rrf()` — Reciprocal Rank Fusion với k=60; `rerank_weighted_rrf()` với dense_weight=1.5, bm25_weight=1.0 | `src/task7_reranking.py` | Done |
| Document-level dedup | Implement `deduplicate_by_document()` — giữ chunk có score cao nhất per source, cap top-k | `src/task7_reranking.py` | Done |
| Lightweight reranker | Implement `rerank_cosine()` — cosine rerank trên top-k chunks (Bonus 4) | `src/bonus4_reranker.py` | Done |
| Retrieval pipeline integration | Tích hợp dense + BM25 + RRF vào `task9_retrieval_pipeline.py`, wired evidence gate và dedup | `src/task9_retrieval_pipeline.py` | Done |
| Dense-only benchmark | So sánh dense-only vs hybrid RRF trên 22 golden cases — dense thắng trên corpus này | `scripts/evaluate_retrieval.py`, `group_project/evaluation/retrieval_evaluation.json` | Done |
| Weighted RRF grid search | Tìm dense_weight/bm25_weight tối ưu; kết quả: dense_weight=1.5 khớp dense-only trên CP/CR/Hit | `scripts/weighted_rrf_grid.py`, `group_project/evaluation/weighted_rrf_grid.json` | Done |
| Bonus 4 benchmark | Đo reranker trên 22 case: hit_rate=0.909 vs baseline 0.909 (không cải thiện trên corpus nhỏ) | `scripts/bonus4_benchmark.py`, `group_project/evaluation/bonus4_benchmark.json` | Done |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Chọn `k=60` cho RRF formula `sum(1/(k+rank))`.
   **Lý do/evidence:** k=60 là giá trị phổ biến nhất trong literature (Baeza-Yates et al.); giá trị lớn giảm impact của rank difference, phù hợp khi dense và BM25 có correlation thấp. Benchmark trên 22 case xác nhận k=60 hoạt động ổn định.
   **Trade-off:** k lớn → RRF gần như uniform averaging; k nhỏ (k=1) quá nhạy với rank noise. Grid search trên k={20,40,60,80} có thể cải thiện nhưng chưa làm.

2. **Quyết định:** Dùng `score_threshold=0.50` thay vì default 0.30 cho dense gate.
   **Lý do/evidence:** Calibration trên 22 case (xem `reports/threshold_calibration.md`): 0.50 giữ nguyên in-domain recall đồng thời tăng OOD recall từ 0.368→0.494. Threshold thấp hơn làm tăng false positive — trả lời sai cho câu ngoài domain.
   **Trade-off:** Threshold cao hơn (0.60) sẽ bỏ sót thêm case borderline nhưng an toàn hơn. Threshold 0.50 là sweet spot.

---

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:** 22 golden cases chạy qua `scripts/evaluate_retrieval.py`; demo queries "IELTS có được dùng để xét tuyển đại học không?" và "Điều kiện xét tuyển đại học gồm những yêu cầu nào?"
- **Kết quả trước/sau nếu có:**
  - Dense-only: CP=0.509, CR=0.841, SourceHit=0.909
  - Hybrid RRF: CP=0.491, CR=0.795, SourceHit=0.864
  - Weighted RRF (dense_w=1.5): CP=0.500, CR=0.841, SourceHit=0.909 (khớp dense-only)
- **Lỗi đã phát hiện và cách xử lý:** BM25 ưu tiên chunks từ article_01/article_02 (popular news) — làm RRF đẩy rare-but-correct chunks (article_08) ra khỏi top-5. Đã ghi nhận trong `group_project/evaluation/RESULT.md` là known limitation; weighted RRF giảm được vấn đề này.

---

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** BM25 fallback khi dense score < 0.50 chưa thực sự trigger vì `pageindex_search` luôn trả `[]` (không có key). Retrieval pipeline chưa có đường retrieval thứ ba thực sự.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Thêm PageIndex key và đo fallback trên 2 câu OOD để xác nhận recall cải thiện. Sau đó thử `k=20` cho RRF để tăng contribution của rank difference.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 25/9/2026
- Tên thành viên: Lê Đức Hùng
