# Architecture — K4-L3B RAG Pipeline

## Sơ đồ tổng quan

```mermaid
flowchart TD
    A[Nguồn dữ liệu<br/>MOET, báo chí, tuyensinh247] --> B[Crawl / Thu thập]
    B --> C[Landing<br/>legal/*.pdf, news/*.json]
    C --> D[Chuẩn hoá Markdown<br/>MarkItDown + JSON->MD]
    D --> E[standardized/<br/>legal/*.md, news/*.md]
    E --> F[RecursiveCharacterTextSplitter<br/>500/50]
    F --> G[Chunks 276]
    G --> H[Embedding BGE-m3]
    H --> I[ChromaDB<br/>cosine]
    G --> J[BM25Plus index]

    Q[User query] --> K[Dense retrieval]
    Q --> L[BM25 retrieval]
    K --> M[RRF fusion]
    L --> M
    K --> N[Dense score gate<br/>threshold=0.50]
    N -->|low| O[PageIndex fallback]
    N -->|ok| M
    O --> P[Hybrid result]
    M --> P
    P --> Q1[Context reordering<br/>front + reverse back]
    Q1 --> R[LLM generator<br/>OpenAI/Gemini/Anthropic]
    R --> S[Answer + Citation [n]]
    S --> T[Streamlit UI]
```

## Thành phần chính

### 1. Data pipeline

* **Task 1 — `task1_collect_legal_docs.py`**: tạo / tải PDF chính sách.
  Dùng `fpdf2` + font `NotoSans-Regular.ttf` để tạo PDF chuẩn Unicode.
  Mỗi file kèm `.meta.json` lưu nguồn URL tham chiếu.
* **Task 2 — `task2_crawl_news.py`**: crawl 8 bài viết từ
  `thanhnien.vn`, `vietnamnet.vn`, `dantri.com.vn`, `tuyensinh247.com`
  bằng Crawl4AI. Fallback sang `requests + BeautifulSoup` khi cần.
* **Task 3 — `task3_convert_markdown.py`**: PDF → Markdown qua MarkItDown;
  JSON → Markdown có header YAML (title, source, url, doc_type).

### 2. Indexing

* **Task 4 — `task4_chunking_indexing.py`**:
  * `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)`
  * Embedding qua `sentence_transformers` với model `BAAI/bge-m3`
    (dimension 1024, đa ngôn ngữ bao gồm tiếng Việt).
  * ChromaDB persistent client (cosine distance).
  * Corpus cache `data/corpus_cache.json` phục vụ BM25.

### 3. Retrieval

* **Task 5 — `semantic_search`** trả `SearchResult` với `retrieval_method="dense"`.
* **Task 6 — `lexical_search`** dùng **BM25Plus** (giải quyết vấn đề IDF=0
  của BM25Okapi trên corpus nhỏ).
* **Task 7 — `rerank_rrf`** công thức `sum(1/(k+rank))`, rank bắt đầu từ 1.

### 4. Hybrid pipeline (Task 9)

```
Query
  ↓
Dense top 10  + BM25 top 10
  ↓
RRF (k=60) — duy nhất một lần
  ↓
inspect ORIGINAL dense cosine score (top-1)
  ↓
< 0.50 ?  → PageIndex fallback (graceful)
  ≥ 0.50 ? → RRF result top-5
  ↓
reorder_for_llm: chunks[::2] + chunks[1::2][::-1]
  ↓
format_context với [Document n | Title | Source | URL]
  ↓
LLM (OpenAI / Gemini / Anthropic) hoặc safe refusal
```

### 5. Generation (Task 10)

* `generate_with_citation(query, top_k) -> GenerationResult`.
* Trả safe refusal khi provider lỗi hoặc context trống.
* Mỗi khẳng định được nhắc citation `[n]` trong prompt.

### 6. UI

* `app.py` (Streamlit):
  * Sidebar chỉnh `top_k` và threshold.
  * Suggested questions cho demo.
  * Lịch sử hội thoại.
  * Mỗi câu trả lời kèm expander liệt kê sources [n].
  * `chat_input` cho câu hỏi tuỳ ý.

## Quy tắc bất biến (Invariants)

1. **Cùng `SearchResult` schema cho dense & BM25.**
2. **RRF chỉ áp dụng một lần** ở Task 9.
3. **Fallback dùng dense cosine score gốc**, không dùng RRF score.
4. **Threshold 0.50** đã hiệu chỉnh trên tập in-domain và OOD.
5. **Citation map về sources**: `[n]` ở câu trả lời ứng với `sources[n-1]`.
6. **Không commit `.env` / API key** vào git.
7. **Tests contract không gọi network/API thật**.

## Cấu hình đã chốt

| Tham số | Giá trị | Ghi chú |
| --- | --- | --- |
| `CHUNK_SIZE` | 500 | đánh đổi giữa ngữ nghĩa và recall |
| `CHUNK_OVERLAP` | 50 | tránh cắt giữa câu |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | multilingual |
| `EMBEDDING_DIM` | 1024 | cosine distance |
| Chroma distance | cosine | yêu cầu contract |
| `RRF k` | 60 | giá trị phổ biến |
| `SCORE_THRESHOLD` | 0.50 | đã hiệu chỉnh |
| `TOP_K` | 5 | UI và pipeline dùng chung |
| LLM providers | OpenAI / Gemini / Anthropic | dispatch theo env |
| BM25 variant | `BM25Plus` | tránh IDF=0 của BM25Okapi |