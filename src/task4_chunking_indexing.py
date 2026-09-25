"""
Task 4 — Chunking, embedding và indexing.

Pipeline:
    1. Đọc toàn bộ Markdown đã chuẩn hoá trong `data/standardized/`.
    2. Chia văn bản bằng `RecursiveCharacterTextSplitter` với cấu hình
       CHUNK_SIZE=500 và CHUNK_OVERLAP=50.
    3. Embed chunks bằng `BAAI/bge-m3` qua sentence-transformers.
       Hỗ trợ override OPENAI qua env `EMBEDDING_PROVIDER=openai`.
    4. Upsert vào ChromaDB (persistent, cosine distance) và đồng thời
       cache corpus chunks để phục vụ BM25 và RRF.

Đặc tả API:
    load_documents() -> list[Document]
    chunk_documents(documents) -> list[Document]   (Chunk = Document + chunk_index)
    embed_texts(texts) -> list[list[float]]
    embed_chunks(chunks) -> list[EmbeddedChunk]
    get_collection() -> chroma Collection
    index_to_vectorstore(chunks) -> None

ID chunks là đường dẫn tương đối của file Markdown + '::chunk-N'.
Việc này đảm bảo ổn định khi chạy lại pipeline.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
CORPUS_CACHE = Path(__file__).parent.parent / "data" / "corpus_cache.json"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


# --- Load & chunk -----------------------------------------------------------


def load_documents() -> list[dict]:
    """Đọc mọi Markdown và tạo Document theo contract."""
    documents: list[dict] = []
    if not STANDARDIZED_DIR.exists():
        return documents
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        rel = path.relative_to(STANDARDIZED_DIR).as_posix()

        doc_type = "legal" if "legal" in rel.lower() else "news"
        # cố gắng đọc metadata từ header YAML
        meta = _parse_yaml_header(text) or {}
        title = meta.get("title") or path.stem
        source = meta.get("source") or path.name
        url = meta.get("url")

        documents.append(
            {
                "id": rel,
                "content": text,
                "metadata": {
                    "source": source,
                    "title": title,
                    "doc_type": doc_type,
                    "url": url,
                },
            }
        )
    return documents


def _parse_yaml_header(text: str) -> dict | None:
    """Trích các trường metadata ở khối --- ở đầu file Markdown."""
    if not text.startswith("---"):
        return None
    lines = text.splitlines()
    if len(lines) < 3 or lines[1:].count("---") == 0:
        return None
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        return None
    header_lines = lines[1:end]
    out: dict[str, str] = {}
    for line in header_lines:
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip().strip('"')
    return out


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for document in documents:
        text = document["content"]
        for index, segment in enumerate(splitter.split_text(text)):
            segment = segment.strip()
            if not segment:
                continue
            chunks.append(
                {
                    "id": f"{document['id']}::chunk-{index}",
                    "content": segment,
                    "metadata": {**document["metadata"], "chunk_index": index},
                }
            )
    return chunks


# --- Embedding --------------------------------------------------------------


_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed danh sách văn bản theo provider đã khai báo trong .env."""
    if not texts:
        return []
    if EMBEDDING_PROVIDER == "openai":
        try:
            from openai import OpenAI

            client = OpenAI()
            response = client.embeddings.create(
                model=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large"),
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception:
            # fallback về local model khi không có key OpenAI
            model = _get_model()
            return [list(map(float, vec)) for vec in model.encode(texts, normalize_embeddings=True).tolist()]

    # default: sentence_transformers
    model = _get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return [list(map(float, vec)) for vec in vectors.tolist()]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vector vào từng chunk."""
    if not chunks:
        return chunks
    texts = [chunk["content"] for chunk in chunks]
    vectors = embed_texts(texts)
    for chunk, vector in zip(chunks, vectors):
        chunk["embedding"] = vector
    return chunks


# --- ChromaDB ----------------------------------------------------------------


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB và cache corpus phục vụ BM25."""
    if not chunks:
        return
    collection = get_collection()

    # xác định id đã tồn tại để bỏ qua upsert lặp
    existing_ids = set()
    try:
        existing_ids.update(collection.get(include=[])["ids"])
    except Exception:
        pass

    new_chunks: list[dict] = []
    for chunk in chunks:
        if chunk["id"] in existing_ids:
            continue
        new_chunks.append(chunk)

    if new_chunks:
        collection.upsert(
            ids=[c["id"] for c in new_chunks],
            documents=[c["content"] for c in new_chunks],
            embeddings=[c["embedding"] for c in new_chunks],
            metadatas=[c["metadata"] for c in new_chunks],
        )

    # Lưu corpus cache cho BM25
    cache = {
        "chunks": [
            {
                "id": c["id"],
                "content": c["content"],
                "metadata": c["metadata"],
            }
            for c in chunks
        ]
    }
    CORPUS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CORPUS_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Indexed {len(new_chunks)} new chunks ({len(chunks) - len(new_chunks)} existing). Total in collection: {collection.count()}.")


# --- Pipeline driver --------------------------------------------------------


def run_pipeline() -> dict[str, Any]:
    """Chạy load → chunk → embed → index; trả về summary."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded = embed_chunks(chunks)
    index_to_vectorstore(embedded)

    # tính content hash cho audit
    text_dump = json.dumps([c["content"] for c in chunks], ensure_ascii=False).encode()
    summary = {
        "documents": len(documents),
        "chunks": len(chunks),
        "collection": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "corpus_hash": hashlib.md5(text_dump).hexdigest(),
    }
    print(f"Summary: {summary}")
    return summary


if __name__ == "__main__":
    run_pipeline()