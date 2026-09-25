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

from pathlib import Path


STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"

# Giải thích lựa chọn tham số trong báo cáo nhóm.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

COLLECTION_NAME = "rag_documents"


<<<<<<< Updated upstream
=======
# --- Load & chunk -----------------------------------------------------------


def load_documents() -> list[dict]:
    """Đọc mọi Markdown và tạo Document theo contract.

    Raise nếu:
        - thư mục STANDARDIZED_DIR không tồn tại.
        - không tìm thấy file .md nào (corpus rỗng).
        - một file có nội dung rỗng / không đọc được.
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"STANDARDIZED_DIR không tồn tại: {STANDARDIZED_DIR}. "
            "Chạy Task 3 (convert markdown) trước."
        )
    documents: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            raise RuntimeError(f"Không đọc được file Markdown {path}: {exc}") from exc
        if not text.strip():
            raise RuntimeError(
                f"File Markdown rỗng: {path}. Bỏ qua sẽ gây mất dữ liệu."
            )
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
    if not documents:
        raise RuntimeError(
            f"Không tìm thấy file .md nào trong {STANDARDIZED_DIR}. "
            "Corpus rỗng — không thể build index."
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


>>>>>>> Stashed changes
def embed_texts(texts: list[str]) -> list[list[float]]:
    # TODO: Dispatch theo EMBEDDING_PROVIDER trong .env.
    #
    # Provider local gợi ý:
    # from sentence_transformers import SentenceTransformer
    # model = SentenceTransformer(EMBEDDING_MODEL)
    # return model.encode(texts).tolist()
    raise NotImplementedError("Implement embed_texts")


def get_collection():
    """Mở Chroma collection dùng cosine distance."""
    # TODO: Tạo hoặc mở persistent collection.
    #
    # import chromadb
    # CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    # return client.get_or_create_collection(
    #     name=COLLECTION_NAME,
    #     metadata={"hnsw:space": "cosine"},
    # )
    raise NotImplementedError("Implement get_collection")


def load_documents() -> list[dict]:
    """Đọc Markdown và trả về danh sách Document."""
    # TODO: Đọc mọi .md và tạo Document theo contract.
    #
    # documents = []
    # for path in STANDARDIZED_DIR.rglob("*.md"):
    #     doc_type = "legal" if "legal" in path.parts else "news"
    #     documents.append({
    #         "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
    #         "content": path.read_text(encoding="utf-8"),
    #         "metadata": {
    #             "source": path.name,
    #             "title": path.stem,
    #             "doc_type": doc_type,
    #             "url": None,
    #         },
    #     })
    # return documents
    raise NotImplementedError("Implement load_documents")


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thành chunks có id và chunk_index."""
    # TODO: Chunk bằng RecursiveCharacterTextSplitter.
    #
    # from langchain_text_splitters import RecursiveCharacterTextSplitter
    # splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=CHUNK_SIZE,
    #     chunk_overlap=CHUNK_OVERLAP,
    #     separators=["\n\n", "\n", ". ", " ", ""],
    # )
    # chunks = []
    # for document in documents:
    #     for index, text in enumerate(splitter.split_text(document["content"])):
    #         chunks.append({
    #             "id": f"{document['id']}::chunk-{index}",
    #             "content": text,
    #             "metadata": {**document["metadata"], "chunk_index": index},
    #         })
    # return chunks
    raise NotImplementedError("Implement chunk_documents")


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Thêm embedding vào từng chunk."""
    # TODO: Embed theo batch và giữ nguyên các field của chunk.
    #
    # vectors = embed_texts([chunk["content"] for chunk in chunks])
    # for chunk, vector in zip(chunks, vectors):
    #     chunk["embedding"] = vector
    # return chunks
    raise NotImplementedError("Implement embed_chunks")


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vào ChromaDB."""
    # TODO: Upsert ids, documents, embeddings và metadatas.
    #
    # collection = get_collection()
    # collection.upsert(
    #     ids=[chunk["id"] for chunk in chunks],
    #     documents=[chunk["content"] for chunk in chunks],
    #     embeddings=[chunk["embedding"] for chunk in chunks],
    #     metadatas=[chunk["metadata"] for chunk in chunks],
    # )
    raise NotImplementedError("Implement index_to_vectorstore")


def run_pipeline() -> None:
    """Chạy load, chunk, embed và index."""
    documents = load_documents()
    chunks = chunk_documents(documents)
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


def verify_index() -> dict[str, Any]:
    """Đọc ChromaDB + corpus cache thật để in báo cáo indexing (dùng cho debug).

    Trả về dict chứa các số liệu runtime; raise nếu index trống hoặc thiếu.
    Hàm này KHÔNG chạy lại indexing — chỉ đọc trạng thái hiện tại.
    """
    documents = load_documents()
    chunks = chunk_documents(documents)
    collection = get_collection()
    db_count = collection.count()
    if db_count == 0:
        raise RuntimeError(
            "ChromaDB collection rỗng. Chạy `python -m src.task4_chunking_indexing` "
            "hoặc `python scripts/build_index.py` để build lại index."
        )

    lengths = [len(c["content"]) for c in chunks]
    avg_len = sum(lengths) / max(len(lengths), 1)
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    empty = sum(1 for l in lengths if l == 0)
    dupes = len(chunks) - len({c["id"] for c in chunks})
    sources = sorted({c["metadata"].get("source", "?") for c in chunks})

    # Lấy 1 embedding thật từ DB để xác nhận dim. peek() mặc định đã trả
    # embeddings (numpy array). Một số version Chroma không nhận `include`.
    db_dim = 0
    try:
        peek = collection.peek(limit=1)
        emb_raw = peek.get("embeddings") if isinstance(peek, dict) else None
        if emb_raw is not None and len(emb_raw) > 0:
            db_dim = len(emb_raw[0])
    except Exception:
        db_dim = 0

    summary = {
        "documents": len(documents),
        "chunks_in_chromadb": db_count,
        "chunks_in_memory": len(chunks),
        "avg_chunk_length_chars": round(avg_len, 1),
        "min_chunk_length_chars": min_len,
        "max_chunk_length_chars": max_len,
        "empty_chunks": empty,
        "duplicate_chunk_ids": dupes,
        "distinct_sources": len(sources),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim_expected": EMBEDDING_DIM,
        "embedding_dim_actual": db_dim,
        "collection": COLLECTION_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunking_method": CHUNKING_METHOD,
    }
    return summary


def print_index_report(summary: dict[str, Any] | None = None) -> dict[str, Any]:
    """In báo cáo indexing ra stdout. Trả về summary để caller dùng tiếp."""
    summary = summary or verify_index()
    print("=== Indexing report ===")
    print(f"Documents loaded        : {summary['documents']}")
    print(f"Chunks in ChromaDB      : {summary['chunks_in_chromadb']}")
    print(f"Chunks (in-memory)      : {summary['chunks_in_memory']}")
    print(f"Chunking                : {summary['chunking_method']} "
          f"size={summary['chunk_size']} overlap={summary['chunk_overlap']}")
    print(f"Avg chunk length (chars): {summary['avg_chunk_length_chars']}")
    print(f"Min/max chunk length    : {summary['min_chunk_length_chars']}/"
          f"{summary['max_chunk_length_chars']}")
    print(f"Empty chunks            : {summary['empty_chunks']}")
    print(f"Duplicate chunk IDs     : {summary['duplicate_chunk_ids']}")
    print(f"Distinct sources        : {summary['distinct_sources']}")
    print(f"Embedding model         : {summary['embedding_model']}")
    print(f"Embedding dim (expected): {summary['embedding_dim_expected']}")
    print(f"Embedding dim (actual)  : {summary['embedding_dim_actual']}")
    print(f"Chroma collection       : {summary['collection']}")
    if summary["embedding_dim_actual"] not in (0, summary["embedding_dim_expected"]):
        print(
            "  WARNING: embedding dim không khớp với mong đợi; "
            "xem lại model đã load."
        )
    return summary


if __name__ == "__main__":
    run_pipeline()
