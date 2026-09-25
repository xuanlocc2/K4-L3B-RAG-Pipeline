"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []


def build_bm25_index(corpus: list[dict]):
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    from rank_bm25 import BM25Okapi

    return BM25Okapi([_tokenize(item["content"]) for item in corpus])


def _tokenize(text: str) -> list[str]:
    """Unicode-friendly tokenizer retaining legal document numbers such as 168/2025."""
    import re

    return re.findall(r"[\wÀ-ỹ]+(?:[/-][\wÀ-ỹ]+)*", text.lower(), flags=re.UNICODE)


def _get_corpus() -> list[dict]:
    if CORPUS:
        return CORPUS
    from .task4_chunking_indexing import chunk_documents, load_documents

    return chunk_documents(load_documents())


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []
    corpus = _get_corpus()
    if not corpus:
        return []
    scores = build_bm25_index(corpus).get_scores(_tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda item: (-float(item[1]), corpus[item[0]]["id"]))
    results: list[dict] = []
    query_tokens = set(_tokenize(query))
    for index, score in ranked:
        if not (query_tokens & set(_tokenize(corpus[index]["content"]))) or len(results) >= top_k:
            continue
        item = corpus[index]
        results.append({
            "id": item["id"],
            "content": item["content"],
            "score": float(score),
            "metadata": item["metadata"],
            "retrieval_method": "bm25",
        })
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
