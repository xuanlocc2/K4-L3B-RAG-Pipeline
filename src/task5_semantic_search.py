"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if top_k <= 0 or not query.strip():
        return []
    collection = get_collection()
    count = collection.count() if hasattr(collection, "count") else top_k
    if count == 0:
        return []
    response = collection.query(
        query_embeddings=[embed_texts([query])[0]],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )
    results: list[dict] = []
    seen: set[str] = set()
    for item_id, content, metadata, distance in zip(
        response["ids"][0], response["documents"][0], response["metadatas"][0], response["distances"][0]
    ):
        if item_id in seen or content is None:
            continue
        seen.add(item_id)
        results.append({
            "id": item_id,
            "content": content,
            "score": max(0.0, 1.0 - float(distance)),
            "metadata": {**metadata, "url": metadata.get("url")},
            "retrieval_method": "dense",
        })
    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
