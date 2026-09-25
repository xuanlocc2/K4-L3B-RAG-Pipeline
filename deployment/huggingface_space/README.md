# Hugging Face Space — RAG Tuyển sinh ĐH Việt Nam

Space này deploy `app.py` lên Hugging Face Spaces (Streamlit runtime).

## Cấu trúc thư mục

```
.
├── app.py                     # entry point (đã được HF Space tự nhận)
├── requirements.txt           # dependency list (copy từ pyproject.toml)
├── packages.txt               # system packages (nếu cần; trống cho app này)
├── README.md                  # Space metadata + mô tả (file này)
├── src/                       # source modules
├── data/corpus_cache.json     # cache corpus (commit vào repo)
├── chroma_db/                 # ChromaDB persistent (commit vào repo)
└── scripts/                   # benchmark + evaluation scripts
```

## Cách deploy lên HF Space

1. Tạo Space mới tại https://huggingface.co/new-space, chọn SDK = **Streamlit**.
2. Clone Space repo về máy.
3. Copy toàn bộ nội dung repo này vào Space repo (trừ `.venv`, `__pycache__`, `.env` thật).
4. Trong Space → **Settings → Variables and secrets**, thêm các biến môi trường
   cần thiết (xem bảng dưới).
5. Push lên Space. Space tự động build và chạy `app.py`.

## Biến môi trường cần thiết

| Tên | Bắt buộc? | Mô tả |
| --- | --- | --- |
| `LLM_PROVIDER` | không | `openai` \| `groq` \| `gemini` \| `anthropic`. Mặc định `openai`. |
| `LLM_MODEL` | không | Tên model cụ thể (ví dụ `gpt-4o-mini`, `llama-3.1-8b-instant`). |
| `OPENAI_API_KEY` | nếu provider=openai | Secret. |
| `GROQ_API_KEY` | nếu provider=groq | Secret. Groq dùng OpenAI-compatible API, base URL `https://api.groq.com/openai/v1`. |
| `GEMINI_API_KEY` | nếu provider=gemini | Secret. |
| `ANTHROPIC_API_KEY` | nếu provider=anthropic | Secret. |
| `EMBEDDING_PROVIDER` | không | Mặc định `sentence_transformers`. |
| `EMBEDDING_MODEL` | không | Mặc định `BAAI/bge-m3` (~2.3GB). |
| `SCORE_THRESHOLD` | không | Mặc định `0.50`. |
| `PAGEINDEX_API_KEY` | không | Để bật fallback vectorless. |
| `DENSE_WEIGHT` | không | Mặc định `1.5` (Bonus 1). |
| `BM25_WEIGHT` | không | Mặc định `1.0` (Bonus 1). |

## Lưu ý quan trọng

* **`BAAI/bge-m3` rất nặng (~2.3GB)** — lần đầu khởi động Space sẽ tải về.
  Nếu muốn Space khởi động nhanh, hãy đổi sang model nhỏ hơn (ví dụ
  `intfloat/multilingual-e5-small`) bằng cách đặt `EMBEDDING_MODEL` ở Space
  settings.
* `chroma_db/` và `data/corpus_cache.json` cần được commit vào repo
  (hoặc build lại qua `python -m src.task4_chunking_indexing` trong Space).
* Không commit file `.env` thật. Sử dụng **Secrets** trong Space settings.

## Trạng thái deployment

> **Không thể tự động verify deployment** vì môi trường này không có HF
> credentials. Cấu hình trên đây là **deployment-ready**: khi copy sang
> Space repo, app sẽ chạy sau khi đặt secrets.

## Verification thủ công

```bash
# Local smoke test (sau khi pip install):
streamlit run app.py --server.port 7860 --server.address 0.0.0.0
# Mở http://localhost:7860 và thử câu hỏi trong sidebar.
```
