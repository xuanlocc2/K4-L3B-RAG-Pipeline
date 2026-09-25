import inspect

import pytest

from src.contracts import (
    assert_extended_score_fields,
    validate_document,
    validate_generation_result,
    validate_search_results,
)


def metadata(source: str = "tuition.md", chunk_index: int = 0) -> dict:
    return {
        "source": source,
        "title": "Tuition policy",
        "doc_type": "legal",
        "url": "https://university.example/tuition",
        "chunk_index": chunk_index,
    }


def result(
    item_id: str,
    score: float,
    method: str = "dense",
    content: str = "Tuition is paid per semester.",
) -> dict:
    return {
        "id": item_id,
        "content": content,
        "score": score,
        "metadata": metadata(chunk_index=int(item_id.rsplit("-", 1)[-1])),
        "retrieval_method": method,
    }


def test_public_function_signatures_are_stable():
    from src.task4_chunking_indexing import chunk_documents, load_documents
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search
    from src.task9_retrieval_pipeline import retrieve
    from src.task10_generation import generate_with_citation

    assert list(inspect.signature(load_documents).parameters) == []
    assert list(inspect.signature(chunk_documents).parameters) == ["documents"]
    assert list(inspect.signature(semantic_search).parameters) == ["query", "top_k"]
    assert list(inspect.signature(lexical_search).parameters) == ["query", "top_k"]
    assert list(inspect.signature(rerank_rrf).parameters) == ["ranked_lists", "top_k", "k"]
    assert list(inspect.signature(pageindex_search).parameters) == ["query", "top_k"]
    assert list(inspect.signature(retrieve).parameters) == [
        "query", "top_k", "score_threshold", "use_reranking"
    ]
    assert list(inspect.signature(generate_with_citation).parameters) == ["query", "top_k"]


def test_document_validator_accepts_contract():
    validate_document(
        {
            "id": "tuition",
            "content": "Tuition policy content",
            "metadata": {key: value for key, value in metadata().items() if key != "chunk_index"},
        }
    )


@pytest.mark.parametrize("missing", ["id", "content", "metadata"])
def test_document_validator_rejects_missing_fields(missing):
    document = {
        "id": "tuition",
        "content": "Tuition policy content",
        "metadata": {key: value for key, value in metadata().items() if key != "chunk_index"},
    }
    document.pop(missing)
    with pytest.raises(ValueError):
        validate_document(document)


def test_search_result_validator_checks_order_method_and_uniqueness():
    valid = [result("chunk-0", 0.9), result("chunk-1", 0.7)]
    validate_search_results(valid, top_k=2, expected_method="dense")

    with pytest.raises(ValueError, match="sorted"):
        validate_search_results(list(reversed(valid)))
    with pytest.raises(ValueError, match="unique"):
        validate_search_results([valid[0], valid[0]])
    with pytest.raises(ValueError, match="expected"):
        validate_search_results(valid, expected_method="bm25")


def test_chunk_documents_preserves_identity_and_metadata():
    from src.task4_chunking_indexing import CHUNK_SIZE, chunk_documents

    document = {
        "id": "tuition",
        "content": "Tuition policy. " * 100,
        "metadata": {
            "source": "tuition.md",
            "title": "Tuition policy",
            "doc_type": "legal",
            "url": None,
        },
    }
    chunks = chunk_documents([document])

    assert chunks
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    for index, chunk in enumerate(chunks):
        validate_document(chunk, require_chunk=True)
        assert chunk["metadata"]["source"] == "tuition.md"
        assert chunk["metadata"]["chunk_index"] == index
        assert len(chunk["content"]) <= int(CHUNK_SIZE * 1.1)


def test_semantic_search_uses_shared_embedding_and_contract(monkeypatch):
    import src.task5_semantic_search as semantic

    class FakeCollection:
        def query(self, **kwargs):
            assert kwargs["query_embeddings"] == [[0.1, 0.2]]
            return {
                "ids": [["chunk-0", "chunk-1"]],
                "documents": [["Relevant tuition text", "Less relevant text"]],
                "metadatas": [[metadata(chunk_index=0), metadata(chunk_index=1)]],
                "distances": [[0.1, 0.4]],
            }

    monkeypatch.setattr(semantic, "embed_texts", lambda texts: [[0.1, 0.2]])
    monkeypatch.setattr(semantic, "get_collection", lambda: FakeCollection())
    output = semantic.semantic_search("tuition", top_k=2)
    validate_search_results(output, top_k=2, expected_method="dense")


def test_lexical_search_returns_bm25_contract(monkeypatch):
    import src.task6_lexical_search as lexical

    corpus = [
        {
            "id": "chunk-0",
            "content": "tuition fee payment policy",
            "metadata": metadata(chunk_index=0),
        },
        {
            "id": "chunk-1",
            "content": "library opening hours",
            "metadata": metadata(source="library.md", chunk_index=1),
        },
    ]
    # Use direct monkeypatching on both CORPUS and _BM25 so the cached
    # BM25 index is invalidated. If a previous test loaded the real
    # Vietnamese corpus, _BM25 stays bound to it and direct patching of
    # CORPUS alone is not enough to force re-indexing.
    monkeypatch.setattr(lexical, "CORPUS", corpus)
    monkeypatch.setattr(lexical, "_BM25", None)
    output = lexical.lexical_search("tuition fee", top_k=2)
    validate_search_results(output, top_k=2, expected_method="bm25")
    assert output[0]["id"] == "chunk-0"


def test_rrf_uses_rank_deduplicates_and_marks_hybrid():
    from src.task7_reranking import rerank_rrf

    dense = [result("chunk-0", 0.9), result("chunk-1", 0.8)]
    bm25 = [
        result("chunk-1", 7.0, "bm25"),
        result("chunk-2", 5.0, "bm25"),
    ]
    fused = rerank_rrf([dense, bm25], top_k=3, k=60)

    validate_search_results(fused, top_k=3, expected_method="hybrid")
    assert [item["id"] for item in fused][0] == "chunk-1"
    expected = 1 / 62 + 1 / 61
    assert fused[0]["score"] == pytest.approx(expected)


def test_reorder_is_non_mutating_and_context_contains_source():
    from src.task10_generation import format_context, reorder_for_llm

    chunks = [result(f"chunk-{index}", 1 - index / 10, "hybrid") for index in range(5)]
    original_ids = [item["id"] for item in chunks]
    reordered = reorder_for_llm(chunks)

    assert [item["id"] for item in chunks] == original_ids
    assert sorted(item["id"] for item in reordered) == sorted(original_ids)
    assert reordered[0]["id"] == "chunk-0"
    context = format_context(reordered)
    assert "tuition.md" in context
    assert "Tuition policy" in context


def test_retrieve_uses_dense_score_for_fallback(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [result("chunk-0", 0.2, "dense")]
    sparse = [result("chunk-1", 4.0, "bm25")]
    fallback = [result("chunk-2", 1.0, "pageindex")]

    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: sparse)
    monkeypatch.setattr(pipeline, "rerank_rrf", lambda lists, top_k: [])
    monkeypatch.setattr(pipeline, "pageindex_search", lambda query, top_k: fallback)

    output = pipeline.retrieve("tuition", top_k=2, score_threshold=0.5)
    assert output == fallback
    validate_search_results(output, top_k=2, expected_method="pageindex")


def test_retrieve_fuses_once_when_dense_is_confident(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [result("chunk-0", 0.9, "dense")]
    sparse = [result("chunk-1", 4.0, "bm25")]
    fused = [result("chunk-0", 0.03, "hybrid")]
    calls = {"rrf": 0, "fallback": 0}

    def fake_rrf(lists, top_k):
        calls["rrf"] += 1
        assert lists == [dense, sparse]
        return fused

    def fake_fallback(query, top_k):
        calls["fallback"] += 1
        return []

    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: sparse)
    monkeypatch.setattr(pipeline, "rerank_rrf", fake_rrf)
    monkeypatch.setattr(pipeline, "pageindex_search", fake_fallback)

    output = pipeline.retrieve("tuition", top_k=2, score_threshold=0.5)
    assert output == fused
    assert calls == {"rrf": 1, "fallback": 0}


def test_retrieve_survives_fallback_provider_error(monkeypatch):
    import src.task9_retrieval_pipeline as pipeline

    dense = [result("chunk-0", 0.2, "dense")]
    hybrid = [result("chunk-0", 0.02, "hybrid")]
    monkeypatch.setattr(pipeline, "semantic_search", lambda query, top_k: dense)
    monkeypatch.setattr(pipeline, "lexical_search", lambda query, top_k: [])
    monkeypatch.setattr(pipeline, "rerank_rrf", lambda lists, top_k: hybrid)

    def unavailable(query, top_k):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(pipeline, "pageindex_search", unavailable)
    output = pipeline.retrieve("tuition", top_k=2, score_threshold=0.5)
    assert output == hybrid


def test_generation_result_validator_accepts_safe_refusal():
    validate_generation_result(
        {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }
    )


# ---------------------------------------------------------------------------
# Extension: optional dense/bm25/rrf_score fields (evidence-quality work)
# ---------------------------------------------------------------------------


def test_assert_extended_score_fields_accepts_absent_fields():
    """Optional fields may be absent — backward-compatible."""
    items = [
        {
            "id": "chunk-0",
            "content": "x",
            "score": 0.5,
            "metadata": metadata(chunk_index=0),
            "retrieval_method": "dense",
        },
    ]
    # Must not raise when optional fields are absent.
    assert_extended_score_fields(items)


def test_assert_extended_score_fields_accepts_present_finite_numbers():
    items = [
        {
            "id": "chunk-0",
            "content": "x",
            "score": 0.5,
            "metadata": metadata(chunk_index=0),
            "retrieval_method": "hybrid",
            "dense_score": 0.71,
            "bm25_score": 12.3,
            "rrf_score": 0.032,
        },
    ]
    assert_extended_score_fields(items)


def test_assert_extended_score_fields_rejects_non_numeric_value():
    items = [
        {
            "id": "chunk-0",
            "content": "x",
            "score": 0.5,
            "metadata": metadata(chunk_index=0),
            "retrieval_method": "hybrid",
            "dense_score": "0.71",  # wrong type
        },
    ]
    with pytest.raises(ValueError, match="dense_score"):
        assert_extended_score_fields(items)


def test_assert_extended_score_fields_rejects_nan():
    items = [
        {
            "id": "chunk-0",
            "content": "x",
            "score": 0.5,
            "metadata": metadata(chunk_index=0),
            "retrieval_method": "hybrid",
            "rrf_score": float("nan"),
        },
    ]
    with pytest.raises(ValueError, match="rrf_score"):
        assert_extended_score_fields(items)


def test_assert_extended_score_fields_accepts_none():
    items = [
        {
            "id": "chunk-0",
            "content": "x",
            "score": 0.5,
            "metadata": metadata(chunk_index=0),
            "retrieval_method": "hybrid",
            "dense_score": None,
            "bm25_score": None,
            "rrf_score": None,
        },
    ]
    assert_extended_score_fields(items)


def test_generation_result_validator_accepts_evidence_payload():
    """GenerationResult may carry optional evidence/intent/diagnostics."""
    validate_generation_result(
        {
            "answer": "Theo [1], tốt nghiệp THPT.",
            "sources": [result("chunk-0", 0.7)],
            "retrieval_source": "hybrid",
            "evidence": {
                "status": "sufficient",
                "relevance_label": "High",
                "suggested_action": "answer",
            },
            "intent": {
                "education_level": "high_school",
                "domain": "university_admission",
                "intent": "requirement",
            },
        }
    )
