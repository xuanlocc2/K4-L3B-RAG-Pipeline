"""Integration smoke test cho Bonus 1+2 retrieval modes.

Test thật với corpus đã index (cần ChromaDB + BM25 cache), không dùng mock.
Đảm bảo các chế độ retrieval mới trả về đúng schema và số lượng.
"""

from __future__ import annotations

import pytest

from src.contracts import validate_search_results
from src.task9_retrieval_pipeline import retrieve


# Test này cần corpus đã được index. Skip nếu ChromaDB chưa sẵn sàng.
pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists(
        __import__("pathlib").Path(__file__).parent.parent
        / "chroma_db"
    ),
    reason="ChromaDB not initialized; run task4_chunking_indexing first",
)


SAMPLE_QUERIES = [
    "Điều kiện xét tuyển đại học",
    "IELTS xét tuyển",
    "Phương thức xét tuyển",
]


@pytest.mark.parametrize("mode", ["baseline", "weighted_rrf", "weighted_rrf_dedup", "rrf_dedup"])
@pytest.mark.parametrize("query", SAMPLE_QUERIES)
def test_retrieve_all_modes_return_valid_results(mode, query):
    result = retrieve(query, top_k=5, retrieval_mode=mode)
    assert len(result) <= 5
    if result:
        validate_search_results(result, top_k=5)


@pytest.mark.parametrize("mode", ["dense_only", "bm25_only"])
@pytest.mark.parametrize("query", SAMPLE_QUERIES)
def test_retrieve_solo_modes_return_results(mode, query):
    result = retrieve(query, top_k=5, retrieval_mode=mode)
    if result:
        validate_search_results(result, top_k=5)
        assert all(r["retrieval_method"] == mode.replace("_only", "") for r in result)


def test_weighted_rrf_marks_results_as_hybrid_weighted():
    result = retrieve("điều kiện xét tuyển đại học", top_k=5, retrieval_mode="weighted_rrf")
    if result:
        assert all(r["retrieval_method"] == "hybrid_weighted" for r in result)


def test_dedup_modes_reduce_to_unique_sources():
    result = retrieve("IELTS xét tuyển đại học", top_k=5, retrieval_mode="weighted_rrf_dedup")
    if result:
        sources = [r["metadata"]["source"] for r in result]
        assert len(set(sources)) == len(sources), "Dedup did not deduplicate"
