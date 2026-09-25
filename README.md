# Day 8 — RAG Pipeline

## Mục tiêu

Mỗi nhóm xây dựng một chatbot RAG trả lời câu hỏi từ bộ tài liệu do nhóm thu thập. Sản phẩm phải có hybrid retrieval, citation, giao diện chat và báo cáo đánh giá.

Nhóm tự chọn bài toán và thu thập dữ liệu phù hợp; repo không cung cấp dữ liệu mẫu.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `group_project/ịndividual/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Python bắt buộc: `>=3.10,<3.14`. Đã kiểm thử với **Python 3.11.9**.

## 7. `.env` Configuration

```env
LLM_PROVIDER=openai          # openai | groq | gemini | anthropic
LLM_MODEL=                   # để trống dùng model mặc định của provider
OPENAI_API_KEY=
GROQ_API_KEY=                # dùng khi LLM_PROVIDER=groq
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
PAGEINDEX_API_KEY=
JINA_API_KEY=
SCORE_THRESHOLD=0.50
DENSE_WEIGHT=1.5             # Bonus 1 — Weighted RRF
BM25_WEIGHT=1.0              # Bonus 1 — Weighted RRF
```

> Không commit file `.env` thật có chứa secret. Mọi key đều là tuỳ chọn;
> thiếu key nào thì provider đó trả về chuỗi rỗng và pipeline fallback
> sang safe refusal (không bao giờ crash).

### Groq — provider miễn phí, nhanh

Pipeline hỗ trợ **Groq Cloud** thông qua OpenAI-compatible API. Cách
dùng:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk-...
# LLM_MODEL=llama-3.3-70b-versatile    # tuỳ chọn — mặc định llama-3.1-8b-instant
```

Groq dùng cùng `openai` SDK với `base_url=https://api.groq.com/openai/v1` — không cần thêm package. Test nằm ở `tests/test_groq_provider.py`. Model mặc định hiện tại là `qwen/qwen3.8-27b` (chat model phổ biến trên Groq Cloud, hỗ trợ tiếng Việt); có thể override qua `LLM_MODEL` với model Groq khác như `llama-3.3-70b-versatile` hoặc `openai/gpt-oss-120b` (tuỳ quyền truy cập của account).

## 8. Data Collection

```bash
python -m src.task1_collect_legal_docs    # Tạo 5 PDF chính sách
python -m src.task2_crawl_news            # Crawl 8 bài viết từ 4 nguồn
```

## 9. Data Normalization

```bash
python -m src.task3_convert_markdown       # MarkItDown + JSON -> Markdown
```

## 10. Chunking & Indexing

```bash
python -m src.task4_chunking_indexing      # Chunk + BGE-m3 + ChromaDB
```

Cấu hình:

| Tham số | Giá trị |
| --- | --- |
| `CHUNK_SIZE` | 500 |
| `CHUNK_OVERLAP` | 50 |
| `EMBEDDING_MODEL` | BAAI/bge-m3 |
| `EMBEDDING_DIM` | 1024 |
| Distance | cosine |

## 11. Dense Retrieval

`src/task5_semantic_search.py` — embed query, query ChromaDB, đổi
cosine distance → similarity.

## 12. BM25 Retrieval

`src/task6_lexical_search.py` — dùng `BM25Plus` trên cùng corpus chunks.

## 13. RRF

`src/task7_reranking.py` — `RRF = sum(1/(k+rank))`, rank bắt đầu từ 1,
`k=60`. RRF chỉ chạy **một lần** ở Task 9.

## 14. Audit-Driven Architecture (Evidence-Gating, RETRIEVAL ≠ EVIDENCE ≠ CITATION)

Mọi câu hỏi đi qua 7 bước có thứ tự. Ba khái niệm **bắt buộc phân biệt**:

* **Retrieval**: candidate chunks trả về bởi dense + BM25 + RRF.
* **Evidence**: tập con của retrieval đã qua per-chunk gate (dense ≥ 0.50
  VÀ keyword_overlap ≥ 0.20).
* **Citation**: tập con của evidence đã dedup theo document, cap 1–3.

```
                  ┌──────────────────────────────────┐
                  │  Step 1 — Domain Gate             │
                  │  (classify_domain, deterministic) │
                  └──────────────────────────────────┘
                                  │
              ┌───────────────────┼─────────────────────┐
              │ OUT_OF_DOMAIN (refuse; sources=[])     │
              ▼                                          ▼
   ┌────────────────────────┐            ┌────────────────────────────────┐
   │ Step 2 — Candidate     │            │ Step 3 — Evidence-Quality Gate │
   │ Retrieval (Dense+BM25, │            │ (per-chunk score_chunks)       │
   │ fallback khi dense<thr)│            │                                │
   └────────────────────────┘            └────────────────────────────────┘
              │                                          │
              ▼                                          ▼
   ┌────────────────────────┐            ┌────────────────────────────────┐
   │ Step 4 — RRF fusion    │            │ Step 4 — Conflict detection    │
   │ (rerank_rrf / Weighted │            │ (e.g. lớp 1 + ĐH → clarify)   │
   │  RRF, chạy 1 lần)      │            └────────────────────────────────┘
   └────────────────────────┘                          │
              │                                          ▼
              ▼                            ┌────────────────────────────────┐
   ┌────────────────────────┐   passed    │ Step 5 — Document Dedup        │
   │ Step 5 — Per-chunk     │ ──────────► │ (deduplicate_by_document)      │
   │ evidence gate          │             └────────────────────────────────┘
   │ (score_chunks: dense & │                          │
   │  keyword_overlap)      │                          ▼
   └────────────────────────┘            ┌────────────────────────────────┐
              │                          │ Step 6 — Final Evidence        │
              ▼                          │ (cap 1–3 documents, cap=3)     │
   rejected_candidates (debug only)     └────────────────────────────────┘
              │                                          │
              ▼                                          ▼
   ┌────────────────────────┐            ┌────────────────────────────────┐
   │ Step 7 — Citation      │            │ Step 7 — Generation + Citation│
   │ Honesty: rejected ≠    │            │ Validation (Bonus 7)          │
   │ cited                  │            │ [n] trỏ về final_evidence[n-1] │
   └────────────────────────┘            └────────────────────────────────┘
```

**Score semantics — không bao giờ trộn:**

| Trường | Ý nghĩa | Khoảng | Dùng cho |
|---|---|---|---|
| `dense_score` | cosine similarity gốc | [0, 1] | Threshold, evidence gate, debug label |
| `bm25_score` | BM25Plus | ≥ 0 | Evidence gate, debug |
| `rrf_score` (≈ `score`) | fusion ranking metric | (0, 2/61] | Sắp xếp thứ tự top-K |
| `keyword_overlap` | tỉ lệ từ khóa chung | [0, 1] | Evidence gate |

`rrf_score = 0.031` **KHÔNG BAO GIỜ** được hiểu thành "3.1% confidence".
Các nhãn "High / Medium / Low" trong UI tính từ `dense_score` (hoặc
fallback RRF magnitude khi dense không có).

**Answerability state machine:**

```
                 ┌──── query
                 ▼
        DOMAIN_CHECK (classify_domain)
        ├── OUT_OF_DOMAIN  → refuse  (Vietnamese refusal, sources=[])
        │
        ▼
        EVIDENCE_CHECK (assess_evidence)
        ├── EVIDENCE_INSUFFICIENT → refuse (sources=[])
        ├── EVIDENCE_WEAK         → clarify (sources=[])
        └── EVIDENCE_SUFFICIENT   → answer (sources=final_evidence, 1–3)
```

**Rejected candidates ≠ Citations:** mọi chunk bị loại bởi evidence gate
được surface trong `render_rejected_candidates_panel()` chỉ khi bật
Debug mode trên sidebar. Panel này ghi rõ **"KHÔNG dùng làm nguồn trích
dẫn"** và không bao giờ xuất hiện trong phần "Nguồn trích dẫn" của user.

## 15. Fallback

`src/task9_retrieval_pipeline.py` — dùng `best_dense_score < 0.50` để
trigger PageIndex fallback. Threshold đã hiệu chỉnh trên tập in-domain
và OOD (xem `reports/threshold_calibration.md`).

## 15. Generation

`src/task10_generation.py` — Dispatch OpenAI / Gemini / Anthropic theo
env. Reorder theo front+reverse-back để giảm lost-in-the-middle. Citation
format `[n]` trỏ về `sources[n-1]`.

## 16. Citation

Mỗi câu trả lời kèm danh sách sources, mỗi source có `title`, `source`,
`url`, `score`, `retrieval_method`. UI hiển thị trong expander.

## 17. Evaluation

```bash
python scripts/evaluate.py
```

Kết quả ghi vào `group_project/evaluation/RESULT.md`. Bốn metric chính:

* Context Precision
* Context Recall
* Source Hit Rate
* Answer Overlap (Jaccard)

## 18. A/B Comparison

So sánh **dense-only** vs **hybrid (dense + BM25 + RRF)** trên cùng
golden dataset. Xem chi tiết trong `group_project/evaluation/RESULT.md`.

## 19. Running the Application

```bash
streamlit run app.py
```

App cung cấp:

* Chat với lịch sử hội thoại.
* 6 câu hỏi gợi ý.
* Sidebar chỉnh `top_k` và threshold.
* Citation trong expander cho mỗi câu trả lời.

## 20. Testing

```bash
pytest -q
```

Kết quả hiện tại: **92/92 tests pass** (15 contract + 5 acceptance + 52 unit bonus + 20 integration).

## 21. Demo Examples

| # | Câu hỏi | Loại |
| - | --- | --- |
| 1 | "Điều kiện xét tuyển đại học gồm những yêu cầu nào?" | Factual |
| 2 | "Tóm tắt Quy chế thi tốt nghiệp THPT 2025" | Paraphrase |
| 3 | "IELTS có được dùng để xét tuyển đại học không?" | Semantic + Lexical kết hợp |
| 4 | "Có chính sách nào cho học sinh giỏi quốc gia khi xét tuyển?" | Multi-document |
| 5 | "Thủ tục đăng ký kết hôn tại Việt Nam như thế nào?" | Out-of-domain (safe refusal) |
| 6 | "Tuyensinh247.com là cổng thông tin gì?" | Source-specific |

## 22. Limitations

* **Không có LLM API key trong môi trường mặc định** → answer = safe refusal.
* **Hybrid không tốt hơn dense-only trên corpus này** do BM25 rerank ưu tiên
  chunk phổ biến (xem `reports/FAILURE_ANALYSIS.md`).
  *Tuy nhiên*, Weighted RRF (Bonus 1, `DENSE_WEIGHT=1.5`) **khớp** dense-only
  trên CP/CR/Hit với latency thấp hơn ~5×.
* **PageIndex fallback chưa kích hoạt thực tế** vì không có key.
* **Golden dataset 22 case** — đủ để minh hoạ nhưng chưa đủ lớn để kết luận
  thống kê chung.

## 23. Implemented Bonuses

Xem chi tiết trong `reports/BONUS_IMPLEMENTATION.md`. Tóm tắt:

| Bonus | Status | Cách dùng |
| --- | --- | --- |
| 1. Weighted RRF | ✅ IMPLEMENTED | Sidebar → "Chế độ retrieval" → "Weighted RRF" |
| 2. Document-level dedup | ✅ IMPLEMENTED | Sidebar → "Chế độ retrieval" → "Weighted RRF + dedup" |
| 3. Query expansion / HyDE | ⚠️ PARTIAL (không cải thiện trên corpus này) | `expand_query(..., mode=...)` |
| 4. Reranker (lightweight) | ✅ IMPLEMENTED | Opt-in qua `bonus4_reranker.rerank_cosine` |
| 5. Conversation memory | ✅ IMPLEMENTED | Bật/tắt sidebar → "Conversation memory" |
| 6. Source highlighting | ✅ IMPLEMENTED | Tự động trong citation panel |
| 7. Citation grounding | ✅ IMPLEMENTED | Tự động trong `generate_with_citation` |
| 8. Online deployment | ⚠️ NOT FEASIBLE (configs ready) | `deployment/huggingface_space/` |
| 9. Debug / observability mode | ✅ IMPLEMENTED | Sidebar → "Debug / Observability mode" |
| 10. Retrieval analytics | ✅ IMPLEMENTED | `scripts/bonus10_analytics.py`; sidebar expander |

Bonus benchmark JSON: `group_project/evaluation/bonus{1,3,4,5,10}_benchmark.json`.

---

## Quick start (TL;DR)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m playwright install chromium
copy .env.example .env
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index và kiểm tra contract
python -m src.task4_chunking_indexing
pytest -q

# 3. Chạy sản phẩm
streamlit run app.py
```

## Lộ trình 3 giờ

* **Tests**: `pytest -q` → **109/109 pass** (15 contract + 5 acceptance + 52 unit bonus + 20 integration + 7 real-pipeline e2e + 6 Groq + 4 env-parsing).
* **Streamlit**: HTTP 200 trên `http://127.0.0.1:8765/` (UI tích hợp Bonus 1, 2,
  5, 6, 7, 9, 10).
* **Retrieval**: dense + BM25 + RRF trả về 5 chunks cho truy vấn tiếng Việt.
* **A/B**: chi tiết trong `group_project/evaluation/RESULT.md`.
* **Bonus A/B/C/D**: chi tiết trong `reports/BONUS_IMPLEMENTATION.md`.

## Real End-to-End Verification

Pipeline thật từ PDF/news → Markdown → chunks → BGE-m3 embeddings → ChromaDB → BM25 → RRF → generation. Không mock, không hard-coded retrieval.

**Runtime evidence (smoke test, captured 2026-09-25):**

```
Documents loaded        : 13        (5 legal PDFs + 8 news articles)
Chunks in ChromaDB      : 276       (recursive 500/50)
Embedding model         : BAAI/bge-m3
Embedding dim           : 1024      (verified)
Distinct sources        : 9         (MOET, VietnamNet, Thanh Niên, Dân trí, Tuyensinh247, ...)

Query: "IELTS có được sử dụng để xét tuyển đại học không?"
  DENSE  top-1: legal/quy_che_ngoai_ngu_dau_vao.md::chunk-1  (cosine 0.7066)
  BM25   top-1: legal/quy_che_ngoai_ngu_dau_vao.md::chunk-1  (BM25+ 50.41)
  HYBRID top-1: legal/quy_che_ngoai_ngu_dau_vao.md::chunk-1  (RRF 0.0328)
  GENERATION (Groq qwen/qwen3.8-27b): "IELTS có ... [1] ... [4][5]"
  citation_check: valid=True, grounded_ratio=1.0, total_refs=5
```

Reproduction scripts:

```bash
python scripts/build_index.py            # idempotent re-index
python scripts/smoke_test_rag.py         # indexing report + sample retrieval + gen
python scripts/multi_query_test.py       # 5 query types (factual/paraphrase/keyword/multi/OOD)
python scripts/evaluate_retrieval.py      # dense vs hybrid RRF on 22 golden cases
```

Multi-query test (`reports/multi_query_test.json`):
- Factual, Paraphrase, Keyword-heavy, Multi-document: dense top-1 score 0.55–0.71 — relevant chunks.
- OOD "Bitcoin": dense top-1 score 0.47 < 0.5 threshold — fallback path triggered (no fake answer).

Retrieval evaluation (`group_project/evaluation/retrieval_evaluation.json`):
- Dense-only: CP=0.560, CR=0.925, source_hit=1.00 (in-domain).
- Hybrid RRF: CP=0.540, CR=0.875, source_hit=0.95 (in-domain). Honest verdict: dense wins slightly on this corpus; BM25 adds noise.

Báo cáo đầy đủ: `reports/REAL_PIPELINE_VERIFICATION.md` (corpus size, every stage verified, known limitations).
