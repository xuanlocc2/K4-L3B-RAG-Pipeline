"""
Task 5 — Semantic search.

Embed query bằng chính hàm của Task 4, query ChromaDB và đổi cosine distance
thành similarity. Output phải theo SearchResult, sort giảm dần và không quá top_k.
"""

from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về dense SearchResult theo score giảm dần."""
    if not query.strip():
        return []
    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results: list[dict] = []
    ids = response.get("ids", [[]])[0]
    documents = response.get("documents", [[]])[0]
    metadatas = response.get("metadatas", [[]])[0]
    distances = response.get("distances", [[]])[0]
    for item_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
        sim = max(0.0, 1.0 - float(distance))
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": sim,
                "dense_score": sim,
                "metadata": dict(metadata),
                "retrieval_method": "dense",
            }
        )
    return results


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
