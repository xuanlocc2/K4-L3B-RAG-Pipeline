# Environment Setup — K4-L3B RAG Pipeline

Tài liệu này mô tả cách dựng môi trường để chạy dự án RAG trên máy mới.

## 1. Yêu cầu hệ thống

| Yêu cầu               | Khuyến nghị                                 |
| ---------------------- | ------------------------------------------ |
| OS                     | Windows 10/11, macOS, Linux                 |
| Python                 | **3.10 ≤ Python < 3.14** (đã test với 3.11) |
| RAM                    | ≥ 8 GB (16 GB khuyến nghị cho BGE-m3)      |
| Ổ đĩa                  | ≥ 5 GB trống (model + corpus + cache)      |
| Kết nối Internet      | Cần cho lần đầu tải model và crawl bài viết |

## 2. Tạo môi trường ảo

### Windows (PowerShell)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Nếu PowerShell chặn script activation:

```powershell
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Phiên bản Python trong dự án

`pyproject.toml` đặt `requires-python = ">=3.10,<3.14"`. Phiên bản đã được
kiểm thử trong repo này là **Python 3.11.9**. Không nên dùng Python ≥ 3.14
vì một số thư viện (chromadb, sentence-transformers) chưa hỗ trợ wheel.

## 3. Nâng cấp build tooling

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Cài đặt dependencies

```bash
python -m pip install -e ".[dev]"
```

Lệnh này dựng project ở chế độ editable (`src/`) và cài đặt mọi phụ thuộc
đã khai báo trong `pyproject.toml`. Danh sách bao gồm:

* `chromadb`, `sentence-transformers`, `rank-bm25`, `markitdown`,
  `crawl4ai`, `pageindex`, `ragas`, `streamlit`
* `openai`, `anthropic`, `google-genai`
* `langchain`, `langchain-openai`, `langchain-community`, `langchain-text-splitters`
* `fpdf2` (dùng để tạo PDF khi cần)
* `pytest` (dev)

## 5. Cài đặt Playwright (browser)

Cần cho `crawl4ai` khi crawl trang HTML động:

```bash
python -m playwright install chromium
```

Kiểm tra Chromium đã cài:

```bash
python -c "from playwright.sync_api import sync_playwright; p = sync_playwright().start(); b = p.chromium.launch(); b.close(); p.stop(); print('chromium OK')"
```

## 6. Font tiếng Việt cho PDF

Một số tài liệu PDF được tạo bằng `fpdf2` cần font Unicode. Repo đã đặt sẵn
font `assets/NotoSans-Regular.ttf`. Không cần cài thêm.

## 7. Biến môi trường

Copy file mẫu và điền key thật khi cần:

```bash
cp .env.example .env       # macOS/Linux
copy .env.example .env     # Windows
```

Mặc định các giá trị:

```env
LLM_PROVIDER=openai
LLM_MODEL=
OPENAI_API_KEY=
GEMINI_API_KEY=
ANTHROPIC_API_KEY=
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3
PAGEINDEX_API_KEY=
JINA_API_KEY=
SCORE_THRESHOLD=0.50
```

* Nếu không có key cho LLM, generator trả về **safe refusal**. Pipeline vẫn
  hoạt động và trả về retrieval kèm citation.
* `PAGEINDEX_API_KEY` là tùy chọn; nếu thiếu, `pageindex_search` trả về `[]`
  và pipeline fallback sang hybrid thay vì crash.

## 8. Smoke test

Sau khi cài đặt, chạy các lệnh sau để xác nhận:

```bash
python -c "import chromadb; print('chromadb OK')"
python -c "import streamlit; print('streamlit OK')"
python -c "import sentence_transformers; print('sentence-transformers OK')"
python -c "import rank_bm25; print('rank-bm25 OK')"
python -c "from langchain_text_splitters import RecursiveCharacterTextSplitter; print('langchain OK')"
```

## 9. Chạy pipeline

```bash
# 1. Thu thập dữ liệu
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index
python -m src.task4_chunking_indexing

# 3. Smoke test retrieval
python -m src.task9_retrieval_pipeline

# 4. Chạy app
streamlit run app.py
```

## 10. Chạy tests

```bash
pytest -q
```

## 11. Lỗi thường gặp

| Lỗi | Nguyên nhân | Cách xử lý |
| --- | --- | --- |
| `UnicodeEncodeError` khi `print()` chuỗi tiếng Việt | Terminal mặc định cp1252 | `python -c "import sys; sys.stdout.reconfigure(encoding='utf-8')"` hoặc dùng `PYTHONIOENCODING=utf-8`. |
| `chromadb sqlite3` không tương thích | Python ≥ 3.14 | Dùng Python 3.11/3.12. |
| `crawl4ai` thiếu browser | Chưa `playwright install` | Chạy lại bước 5. |
| OpenAI trả "Missing credentials" | Không có key | Đặt `OPENAI_API_KEY` trong `.env`. |
| Embedding model không tải được | Mạng chặn HuggingFace Hub | Đặt `HF_TOKEN` hoặc dùng cache local. |
| `OSError: [Errno 28] No space left` | Thiếu dung lượng | Dọn `.venv`, `chroma_db`, `data/_tmp_pdf`. |

## 12. Verified working commands (đã kiểm thử trong dự án)

```bash
# Windows PowerShell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
copy .env.example .env
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown
python -m src.task4_chunking_indexing
pytest -q
streamlit run app.py
```

Kết quả thực tế (xem `docs/PROJECT_AUDIT.md`):

* 5 PDF chính sách + 8 bài viết tin tức crawl từ nguồn thật.
* 13 tài liệu được chuẩn hoá, 276 chunk embedding bằng `BAAI/bge-m3`.
* 22 golden questions, A/B giữa dense-only và hybrid + RRF.
* Tất cả 20 tests pass.