# Streamlit Community Cloud deployment

## Cách deploy

1. Push repo lên GitHub (không commit `.env` thật).
2. Vào https://share.streamlit.io/, connect GitHub repo.
3. Chọn branch + main file = `app.py`.
4. Trong **Advanced settings → Secrets**, paste nội dung `.env` thật
   (chỉ các key cần thiết).

## Secrets mẫu

```toml
LLM_PROVIDER = "openai"
OPENAI_API_KEY = "sk-..."
SCORE_THRESHOLD = "0.50"
```

## Lưu ý

- BAAI/bge-m3 (~2.3GB) sẽ tải về lần đầu. Lần sau nhanh hơn vì cache.
- ChromaDB persistent dir `chroma_db/` cần được commit hoặc build lại.
- Streamlit Cloud có giới hạn resource; nếu gặp OOM, hãy đổi sang
  `EMBEDDING_MODEL = "intfloat/multilingual-e5-small"`.
