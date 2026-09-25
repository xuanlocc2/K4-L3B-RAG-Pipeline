# RAG Evaluation Results

> Kết quả được tổng hợp từ `scripts/evaluate.py`. Dữ liệu chi tiết được lưu tại
> `group_project/evaluation/evaluation_results.json`.

## Run information

| Field | Value |
| --- | --- |
| Evaluation date | 2026-09-25 |
| Framework and version | Custom evaluation pipeline + Ragas 0.4.3 |
| Evaluator model | Deterministic token-based metrics (Jaccard overlap); Ragas LLM-judged metrics chưa bật do không có LLM API key |
| Generator model | LLM provider đã cấu hình nhưng môi trường đánh giá không có API key |
| Embedding model | `BAAI/bge-m3` (`sentence-transformers` 6.1.0) |
| Corpus version/commit | 13 tài liệu chuẩn hóa / 276 chunks |
| Golden dataset size | 22 cases (20 in-domain, 2 out-of-domain) |
| `top_k` | 5 |
| Fallback threshold and calibration | 0.50 (xem `reports/threshold_calibration.md`) |

## Configurations

- **Config A — dense-only:** chỉ dùng `semantic_search(query, top_k=5)`.
- **Config B — hybrid + RRF:** dense top-10 + BM25 top-10, hợp nhất bằng RRF và trả về top-5.

Hai cấu hình dùng cùng embedding, golden dataset, generator và `top_k`; chỉ thay đổi chiến lược retrieval.

## Overall scores

| Metric | Config A | Config B | Delta B − A |
| --- | ---: | ---: | ---: |
| Context Precision | 0.509 | 0.491 | −0.018 |
| Context Recall | 0.841 | 0.795 | −0.046 |
| Source Hit Rate | 0.909 | 0.864 | −0.045 |
| Answer Overlap (Jaccard) | 0.104 | 0.104 | 0.000 |
| Answered Rate | 0.000 | 0.000 | 0.000 |
| Average retrieval latency (ms) | 147 | 102 | −45 |

> Do môi trường đánh giá không có LLM API key, mọi case đều trả safe refusal.
> Vì vậy `Answered Rate = 0.0`; đây là kết quả thực tế, không phải placeholder.

## A/B comparison

- **Cấu hình tốt hơn trên tập dữ liệu này:** Config A (dense-only), cao hơn Config B khoảng 2–5 điểm phần trăm ở precision, recall và source hit rate.
- **Nguyên nhân hybrid kém hơn:** RRF hợp nhất theo chunk, trong khi expected sources được chấm theo document. BM25 thường đưa nhiều chunk từ các bài phổ biến lên cao, làm chunk hiếm nhưng đúng bị đẩy khỏi top-5.
- **Trade-off latency:** Config B nhanh hơn khoảng 30% trong lần đo này (102 ms so với 147 ms).
- **An toàn:** cả hai cấu hình đều trả safe refusal cho hai câu ngoài domain.

## Worst performers

| # | Question | Config | Source Hit | Precision | Recall | Failure stage | Root cause |
| --: | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | Tuyensinh247.com là cổng thông tin gì? | B | 0/1 | 0.00 | 0.00 | Retrieval/hybrid | BM25 không đưa `article_08` lên đủ cao; RRF đẩy chunk đúng khỏi top-5. |
| 2 | Thủ tục đăng ký kết hôn tại Việt Nam như thế nào? | A, B | 0/0 | 0.00 | 0.00 | Generation/safe refusal | Câu hỏi ngoài domain; pipeline từ chối đúng, nhưng chưa có LLM key để kiểm tra generation. |
| 3 | Lãi suất ngân hàng Techcombank hiện tại là bao nhiêu? | A, B | 0/0 | 0.00 | 0.00 | Generation/safe refusal | Câu hỏi ngoài domain; pipeline từ chối an toàn. |
| 4 | Điều kiện xét tuyển đại học gồm những yêu cầu nào? | B | 1/1 | 0.60 | 1.00 | Precision regression | 4/5 chunk trả về không thuộc expected source do BM25/RRF ranking. |

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| ---: | --- | --- | --- | --- |
| 1 | Bật ít nhất một LLM API key hợp lệ. | 100% case hiện trả safe refusal do thiếu credentials. | Đo được answer quality và tăng answered rate. | Chạy lại `python scripts/evaluate.py`, kiểm tra `answered_rate`. |
| 2 | Mở rộng golden dataset lên ít nhất 30 câu. | 22 case còn nhỏ và một số nhóm chỉ có một expected source. | Kết quả A/B ổn định và bao phủ edge case tốt hơn. | Bổ sung dữ liệu rồi chạy lại toàn bộ evaluation. |
| 3 | Cấu hình PageIndex key thật. | `pageindex_search` hiện trả `[]`, nên fallback chưa được đánh giá thực tế. | Có retrieval path thứ ba và đo được recall của fallback. | Đặt `PAGEINDEX_API_KEY` và chạy lại các case OOD/score thấp. |
| 4 | Thử weighted RRF, document dedup và các giá trị `k` khác. | RRF chuẩn làm mất source đúng ở Q19. | Tăng recall và source diversity cho truy vấn hiếm. | A/B với `k=20,40,60,80` và trọng số dense/BM25 khác nhau. |

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| --- | --- | ---: | ---: | --- |
| Hybrid RRF | Dense-only | CP −0.018; CR −0.046; Hit −0.045 | Khoảng −30% latency | Dense-only tốt hơn trên corpus nhỏ hiện tại. |
| Weighted RRF (1.5/1.0) | RRF chuẩn | CP +0.009; CR +0.046; Hit +0.045 | Gần như không đổi | Khớp chất lượng dense-only; phù hợp làm tùy chọn. |
| Document-level dedup | Weighted RRF | Doc diversity 2.27 → 3.00 | Gần như không đổi | Tăng đa dạng nguồn nhưng giảm CP/CR/Hit. |
| Cosine reranker | Hybrid RRF | CP +0.018; CR +0.046; Hit +0.045 | Khoảng 6× latency | Chất lượng bằng dense-only nhưng chi phí cao. |
| Query expansion/HyDE fallback | Baseline | Giảm khoảng 4–6% | Tăng xử lý | Không có lợi trên corpus nhỏ này. |
| Threshold 0.50 | Threshold 0.30 | Cải thiện lọc OOD, giữ in-domain recall | Không đáng kể | Giữ threshold 0.50 theo calibration. |

## Kết luận

Dense-only là cấu hình retrieval tốt nhất trên tập đánh giá hiện tại. Weighted RRF là phương án hybrid hợp lý khi cần cân bằng chất lượng và latency. Các kết luận còn bị giới hạn bởi golden dataset nhỏ và việc chưa có LLM/PageIndex credentials; cần chạy lại sau khi cấu hình đủ khóa để đánh giá generation và fallback end-to-end.
