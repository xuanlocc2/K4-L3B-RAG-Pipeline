"""
Task 4 — Chunking, embedding và indexing.

Hướng dẫn:
    1. Đọc toàn bộ Markdown trong data/standardized/.
    2. Chia văn bản bằng strategy đã chọn.
    3. Embed chunks bằng một provider duy nhất.
    4. Upsert vào ChromaDB với cosine distance.

Mỗi document/chunk phải theo docs/MODULE_CONTRACTS.md. ID cần ổn định để
chạy lại pipeline không tạo dữ liệu trùng. Task 5 phải dùng chung embed_texts().
"""

import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


def _metadata_from_markdown(path: Path, content: str, doc_type: str) -> dict:
    """Extract the explicit source metadata written by Task 3's Markdown."""
    title_match = re.search(r"^#\s+(.+?)\s*$", content, flags=re.MULTILINE)
    source_match = re.search(r"^\*\*Source(?: file)?:\*\*\s*(.+?)\s*$", content, flags=re.MULTILINE)
    url_match = re.search(r"^\*\*(?:Source|Official text):\*\*\s*(https?://\S+)", content, flags=re.MULTILINE)
    return {
        "source": source_match.group(1).strip() if source_match else path.name,
        "title": title_match.group(1).strip() if title_match else path.stem.replace("-", " "),
        "doc_type": doc_type,
        "url": url_match.group(1).strip() if url_match else None,
    }


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load one local model per process; both corpus and query use this object."""
    from sentence_transformers import SentenceTransformer

    load_dotenv()
    provider = os.getenv("EMBEDDING_PROVIDER", "sentence_transformers").strip()
    if provider != "sentence_transformers":
        raise ValueError("This lab is configured for local sentence_transformers embeddings")
    model_name = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL).strip() or EMBEDDING_MODEL
    if model_name != EMBEDDING_MODEL:
        raise ValueError(f"Expected local embedding model {EMBEDDING_MODEL}, got {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed text with normalized BGE-M3 vectors for cosine similarity."""
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("embed_texts only accepts non-empty strings")
    vectors = _get_embedding_model().encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()
    if any(len(vector) != EMBEDDING_DIM for vector in vectors):
        raise ValueError(f"Unexpected embedding dimension; expected {EMBEDDING_DIM}")
    return vectors


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative = path.relative_to(STANDARDIZED_DIR)
        doc_type = "legal" if relative.parts[0] == "legal" else "news"
        documents.append({
            "id": relative.as_posix(),
            "content": content,
            "metadata": _metadata_from_markdown(path, content, doc_type),
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[dict] = []
    for document in documents:
        for index, text in enumerate(splitter.split_text(document["content"])):
            text = text.strip()
            if text:
                chunks.append({
                    "id": f"{document['id']}::chunk-{index}",
                    "content": text,
                    "metadata": {**document["metadata"], "chunk_index": index},
                })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    return [{**chunk, "embedding": vector} for chunk, vector in zip(chunks, vectors)]


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    if not chunks:
        return
    collection = get_collection()
    # Chroma metadata cannot contain None. It is restored to None by Task 5
    # when absent, preserving the public contract for legal source files.
    metadatas = [{key: value for key, value in chunk["metadata"].items() if value is not None} for chunk in chunks]
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=metadatas,
    )


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
