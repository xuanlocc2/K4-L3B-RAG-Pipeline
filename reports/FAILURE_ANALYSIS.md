# Failure Analysis

> Phân tích thất bại thực tế từ `scripts/evaluate.py` (A/B comparison).

## 1. Phương pháp

Chạy A/B comparison với 22 golden cases (20 in-domain, 2 out-of-domain) trên
hai chiến lược:

* **A — dense-only**: `semantic_search(query, top_k=5)`.
* **B — hybrid + RRF**: `rerank_rrf([semantic, bm25], top_k=5)`.

Cùng `embed_texts`, cùng generator, cùng `top_k`.

## 2. Trường hợp thất bại

### Case 1 — Q19 "Tuyensinh247.com là cổng thông tin gì?"

| Field | Dense-only (A) | Hybrid (B) |
| --- | --- | --- |
| Source Hit | 1 (article_08.md ở rank 5/5) | 0 (article_08 bị đẩy ra ngoài top-5) |
| Context Precision | 0.20 | 0.00 |
| Context Recall | 1.00 | 0.00 |
| Top retrieved | `article_01::chunk-10, article_02::chunk-10, article_01::chunk-9, article_02::chunk-9, article_08::chunk-1` | `article_01::chunk-10, article_02::chunk-10, article_01::chunk-9, article_02::chunk-9, article_01::chunk-12` |

**Root cause**: BM25 trả về nhiều chunk từ `article_01` và `article_02` vì
chúng chứa nhiều từ khoá phổ biến ("đại học", "Mỹ", "Việt", ...). RRF ưu
tiên các chunk xuất hiện ở cả dense và BM25 (cộng dồn điểm), nên
`article_08` chỉ xuất hiện trong dense — không có điểm cộng — và bị đẩy ra
ngoài top-5.

**Fix đề xuất**:
1. Dùng **weighted RRF** (trọng số dense cao hơn BM25).
2. Hoặc tăng `top_k` lên 8–10 khi gọi BM25/dense trước khi RRF.
3. Hoặc deduplicate theo `metadata.source` (document) thay vì theo `chunk_id`.

**Kết quả sau khi điều chỉnh**: chưa đo lại trong bản này; cần thực nghiệm
lần sau khi bật LLM API key để đo đồng thời Answer Overlap.

### Case 2 — Q20, Q21 (out-of-domain)

Câu hỏi "Thủ tục đăng ký kết hôn" và "Lãi suất ngân hàng" — cả hai đều
nằm ngoài domain tuyển sinh.

* Dense trả về các chunk không liên quan với score thấp (0.46–0.49).
* `best_dense_score < 0.50` → fallback PageIndex được kích hoạt.
* PageIndex không có key → trả `[]`.
* Pipeline trả hybrid (các chunk noise) thay vì crash.
* Generator không có key → safe refusal được in ra.

**Root cause**: thiếu LLM key → user chỉ thấy safe refusal dù retrieval
có trả về gì đi nữa. Khi có key, answer sẽ phụ thuộc vào các chunk noise
(because the model will try to answer from irrelevant context).

**Fix đề xuất**:
1. Khi `best_dense_score < threshold` VÀ tất cả retrieved đều có score < 0.5,
   pipeline nên trả về safe refusal trực tiếp, thay vì chuyển sang LLM với
   context noise.
2. Thêm một `min_answer_score` sau retrieval: nếu max(score) < `min_answer_score`
   thì trả về safe refusal.

### Case 3 — Q01 "Điều kiện xét tuyển đại học gồm những yêu cầu nào?"

Dense-only trả về 5 chunk, trong đó 4/5 thuộc về `thong_tu_dieu_kien_xet_tuyen.md`
và 1/5 thuộc về `quy_che_tuyen_sinh_dh_2025.md`. Expected source là cả hai
file, nên Source Hit = 1, CP = 0.80, CR = 1.00.

Hybrid trả về 5 chunk, nhưng RRF rerank khiến 1 chunk từ expected source
bị đẩy ra. CP giảm còn 0.60.

**Root cause**: tương tự Case 1 — RRF rerank ưu tiên chunk xuất hiện ở cả
hai bảng xếp hạng.

### Case 4 — Q10 "Chính sách cho học sinh giỏi quốc gia"

Câu này cần ghép thông tin từ `quy_che_tuyen_sinh_dh_2025.md` (policy)
và `article_06.md` (news). Cả hai đều trả về đúng.

**Lesson learned**: dense-only vẫn trả về đủ multi-document retrieval nhờ
BGE-m3 nắm được ngữ nghĩa. Hybrid không cải thiện thêm.

## 3. Tổng kết

| Loại thất bại | Nguyên nhân | Mức ảnh hưởng |
| --- | --- | --- |
| Hybrid đẩy mất chunk đúng | BM25 rerank quá mạnh trên common terms | 1/22 case (Q19) |
| Generator safe refusal (100% câu) | Thiếu LLM API key | Tất cả câu |
| Context Precision thấp trên multi-source | Chunk đúng không nằm trong top-5 | 4/20 in-domain case |
| Out-of-domain vẫn trigger retrieval | Threshold chỉ chặn retrieval rất thấp | 2/2 OOD case |
| PageIndex fallback chưa hoạt động | Không có key | Tất cả case có score < threshold |

## 4. Hành động khắc phục đã thực hiện

1. ✅ Chọn `BM25Plus` thay cho `BM25Okapi` để tránh IDF=0 trên corpus nhỏ.
2. ✅ Hiệu chỉnh `SCORE_THRESHOLD = 0.50` dựa trên calibration.
3. ✅ `safe_refusal` đảm bảo UI không crash khi provider lỗi.
4. ✅ PageIndex fallback `try/except` để không bao giờ làm vỡ pipeline.

## 5. Hành động khắc phục còn lại

1. Đặt `OPENAI_API_KEY` hoặc `GEMINI_API_KEY` thật để generator chạy.
2. Đặt `PAGEINDEX_API_KEY` để fallback thực sự hoạt động trên câu OOD.
3. Thử weighted-RRF hoặc deduplicate-by-document.
4. Mở rộng golden dataset lên ≥ 30 case để có dữ liệu A/B đáng tin hơn.