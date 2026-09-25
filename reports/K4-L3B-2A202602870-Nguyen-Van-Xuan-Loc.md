# Individual contribution report — Nguyễn Văn Xuân Lộc

---

## Thông tin

- Họ và tên: **Nguyễn Văn Xuân Lộc**
- Mã học viên: 2A202602870
- Nhóm: LaoGaKho
- Repository/branch: `main`

---

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| Legal document collection (Task 1) | Implement `src/task1_collect_legal_docs.py` — tạo 5 PDF chính sách tuyển sinh từ nguồn Bộ GD&ĐT | `src/task1_collect_legal_docs.py` | Done |
| News crawling (Task 2) | Implement `src/task2_crawl_news.py` — crawl 8 bài viết từ 4 nguồn (Thanh Niên, VietnamNet, Dân trí, Tuyensinh247) | `src/task2_crawl_news.py` | Done |
| Markdown normalization (Task 3) | Implement `src/task3_convert_markdown.py` — MarkItDown + JSON → standardized Markdown, deduplicate | `src/task3_convert_markdown.py` | Done |
| Chunking & indexing (Task 4) | Implement `src/task4_chunking_indexing.py` — recursive 500/50 chunking, BGE-m3 embedding, ChromaDB storage | `src/task4_chunking_indexing.py` | Done |
| Evidence quality gate | Implement `src/evidence_quality.py` — `assess_evidence()`, `score_chunks()`, per-chunk gate (dense ≥ 0.50 VÀ keyword_overlap ≥ 0.20) | `src/evidence_quality.py` | Done |
| Domain classification | Implement `classify_domain()` — out-of-domain detection dựa trên query embedding và corpus centroid distance | `src/contracts.py` | Done |
| Retrieval analytics (Bonus 10) | Implement `scripts/bonus10_analytics.py` — hit rate, CP/CR, doc diversity, fallback frequency, RRF contribution trên 22 case | `scripts/bonus10_analytics.py`, `group_project/evaluation/bonus10_analytics.json` | Done |
| Evaluation harness | Implement `scripts/evaluate_retrieval.py` — so sánh dense-only vs hybrid trên golden dataset, tính CP/CR/source hit/Jaccard | `scripts/evaluate_retrieval.py`, `group_project/evaluation/retrieval_evaluation.json` | Done |
| Golden dataset | Xây dựng `group_project/evaluation/golden_dataset.json` — 22 cases (20 in-domain, 2 OOD) với expected sources và ground truth | `group_project/evaluation/golden_dataset.json` | Done |
| Multi-query test | Implement `scripts/multi_query_test.py` — 5 loại queries (factual/paraphrase/keyword/multi/OOD) | `scripts/multi_query_test.py`, `reports/multi_query_test.json` | Done |
| Bonus 10 analytics benchmark | Đo dense/bm25/hybrid/weighted/dedup strategies: dense-only hit_rate=0.909, weighted_rrf hit_rate=0.909 | `scripts/bonus10_benchmark.py`, `group_project/evaluation/bonus10_analytics.json` | Done |
| Deployment configs | HuggingFace Space config, Streamlit Cloud config | `deployment/huggingface_space/`, `deployment/streamlit_cloud/` | Partial |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Chunk size 500 tokens, overlap 50 tokens, recursive character splitting.
   **Lý do/evidence:** 500/50 là balance giữa semantic completeness và recall. Corpus nhỏ (13 docs) nên chunk nhỏ giữ được granularity. Overlap 50 đảm bảo context không bị cắt ngang câu. Đã verify: 276 chunks tổng cộng cho 13 documents — phân bố hợp lý.
   **Trade-off:** Chunk nhỏ → nhiều chunks → retrieval precision giảm nhẹ (top-k có thể chứa partial sentences). Chunk lớn hơn (800/100) có thể cải thiện CP nhưng tăng context length và LLM cost.

2. **Quyết định:** Evidence gate threshold: `dense ≥ 0.50` VÀ `keyword_overlap ≥ 0.20` (AND, không OR).
   **Lý do/evidence:** Dùng AND đảm bảo chunk phải pass cả semantic similarity VÀ lexical overlap mới được dùng làm evidence. OR quá permissive — BM25 có thể đẩy chunks với high bm25_score nhưng low semantic relevance vào evidence. Calibration trên 22 case xác nhận AND threshold giảm false positive.
   **Trade-off:** AND threshold nghiêm ngặt hơn — một số borderline case (dense 0.48, keyword 0.25) bị loại. OR threshold sẽ tăng recall nhưng có thể include noise.

---

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:** Smoke test full pipeline (`scripts/smoke_test_rag.py`) — 13 docs, 276 chunks indexed; retrieval evaluation trên 22 golden cases; multi-query test 5 loại queries
- **Kết quả trước/sau nếu có:**
  - Corpus: 13 documents → 276 chunks (verify bằng ChromaDB count)
  - OOD "Bitcoin": dense top-1 = 0.47 < 0.50 threshold → fallback triggered (không fake answer)
  - OOD "Đăng ký kết hôn": domain classifier → refuse (safe refusal đúng)
  - Factual query "IELTS": dense top-1 = 0.7066, BM25 top-1 = 50.41 — same chunk #1
- **Lỗi đã phát hiện và cách xử lý:**
  - Lỗi: `pageindex_search` luôn trả `[]` (không có key) → dense fallback không bao giờ kích hoạt fallback path thực sự. Đã document là known limitation trong `reports/REAL_PIPELINE_VERIFICATION.md`.

---

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Golden dataset chỉ có 22 cases — quá nhỏ để rút ra kết luận thống kê chung về hybrid vs dense. Một số câu "tổ hợp môn" chỉ có 1 expected source, làm CP khó đạt 1.0.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Mở rộng golden dataset lên ≥ 30 cases, thêm các câu paraphrase tổ hợp môn, multi-hop (cần cross-reference 2+ docs), và edge cases cho từng retrieval strategy.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 26/09/2026
- Tên thành viên: Nguyễn Văn Xuân Lộc
