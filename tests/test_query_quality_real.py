"""Real-corpus tests for the new evidence-quality + score-separation
behavior. Skipped automatically when ChromaDB / BM25 cache is not
present (so this file works in any CI environment).

These tests are the integration counterpart to ``test_evidence_quality.py``:
they exercise the actual BGE-m3 embeddings + BM25 + RRF against the real
indexed corpus, and they verify the 5 audit-required query categories.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


pytestmark = pytest.mark.skipif(
    not (ROOT / "chroma_db").exists(),
    reason="ChromaDB not initialized; run scripts/build_index.py first",
)


@pytest.fixture(scope="module")
def retrieval():
    """Lazy import + warm up the pipeline once."""
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank_rrf
    from src.task9_retrieval_pipeline import retrieve_with_evidence
    from src.evidence_quality import (
        assess_evidence,
        build_normalized_query,
        understand_query,
    )
    return {
        "semantic_search": semantic_search,
        "lexical_search": lexical_search,
        "rerank_rrf": rerank_rrf,
        "retrieve_with_evidence": retrieve_with_evidence,
        "assess_evidence": assess_evidence,
        "build_normalized_query": build_normalized_query,
        "understand_query": understand_query,
    }


def _retrieve(retrieval, query: str, top_k: int = 5):
    return retrieval["retrieve_with_evidence"](query, top_k=top_k)


def _source_titles(chunks) -> list[str]:
    return [(c.get("metadata") or {}).get("title", "") for c in chunks]


def _doc_id(chunk) -> str:
    return str(chunk.get("id", "")).split("::chunk-")[0]


# ---------------------------------------------------------------------------
# Audit-required tests
# ---------------------------------------------------------------------------


def test_real_in_domain_admission_retrieves_relevant_docs(retrieval):
    """T1: in-domain admissions query → relevant documents."""
    res = _retrieve(retrieval, "Điều kiện xét tuyển đại học là gì?", top_k=5)
    assert res["evidence"]["status"] == "sufficient"
    assert res["evidence"]["suggested_action"] == "answer"
    docs = {_doc_id(c) for c in res["chunks"]}
    assert any("dieu_kien" in d or "tuyen_sinh" in d for d in docs), (
        f"In-domain query phải hit tuyển sinh docs, got {docs}"
    )


def test_real_paraphrase_finds_same_doc_category(retrieval):
    """T2: paraphrase retrieves same document category as T1."""
    a = _retrieve(retrieval, "Điều kiện xét tuyển đại học là gì?", top_k=3)
    b = _retrieve(
        retrieval,
        "Tôi cần những điều kiện gì để đăng ký vào đại học?",
        top_k=3,
    )
    docs_a = {_doc_id(c) for c in a["chunks"]}
    docs_b = {_doc_id(c) for c in b["chunks"]}
    assert docs_a & docs_b, (
        f"Paraphrase phải overlap ít nhất 1 doc với original; "
        f"got a={docs_a} b={docs_b}"
    )


def test_real_ielts_query_retrieves_ielts_doc(retrieval):
    """T3: keyword-heavy query → IELTS / foreign-language doc."""
    res = _retrieve(retrieval, "IELTS 6.5 xét tuyển đại học", top_k=5)
    assert res["evidence"]["status"] == "sufficient"
    assert res["evidence"]["suggested_action"] == "answer"
    docs = {_doc_id(c) for c in res["chunks"]}
    titles = " | ".join(_source_titles(res["chunks"]))
    assert "ielts" in titles.lower() or any("ngoai_ngu" in d for d in docs), (
        f"IELTS query phải hit IELTS/ngoại ngữ doc; titles={titles}"
    )


def test_real_lop1_query_is_clarified_no_supporting_sources(retrieval):
    """T4: primary-school + university-admission → clarify (no sources)."""
    from src.task10_generation import generate_with_citation
    import src.task10_generation as gen
    gen.call_llm = lambda *a, **kw: ""

    res = generate_with_citation(
        "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào",
        top_k=3,
    )
    assert res["action"] == "clarify", (
        f"Primary-school + ĐH phải clarify; got action={res['action']}"
    )
    assert res["sources"] == [], (
        "Clarify path KHÔNG được trả sources như 'trích dẫn hỗ trợ'."
    )
    assert res["retrieval_source"] == "none"
    # The clarification message must ask the user to disambiguate.
    assert any(
        word in res["answer"].lower()
        for word in ("tiểu học", "lớp 1", "thpt", "lớp 12")
    )


def test_real_bitcoin_query_is_refused(retrieval):
    """T5: OOD query → safe refusal, no supporting sources."""
    from src.task10_generation import generate_with_citation
    import src.task10_generation as gen
    gen.call_llm = lambda *a, **kw: ""

    res = generate_with_citation(
        "Giá Bitcoin hôm nay là bao nhiêu?",
        top_k=3,
    )
    assert res["action"] == "refuse"
    assert res["sources"] == []
    assert res["retrieval_source"] == "none"
    # Domain-gate OOD message (audit §12):
    assert "nằm ngoài phạm vi" in res["answer"].lower()
    assert res["answerability_state"] == "out_of_domain"


# ---------------------------------------------------------------------------
# Score separation: real corpus
# ---------------------------------------------------------------------------


def test_real_hybrid_results_expose_dense_bm25_rrf(retrieval):
    """Real hybrid result must carry all three optional score fields when
    applicable (chunk present in dense OR bm25 OR both)."""
    res = _retrieve(retrieval, "IELTS 6.5 xét tuyển đại học", top_k=3)
    assert res["chunks"]
    at_least_one_with_dense = False
    at_least_one_with_bm25 = False
    at_least_one_with_rrf = False
    for c in res["chunks"]:
        if c.get("dense_score") is not None:
            at_least_one_with_dense = True
        if c.get("bm25_score") is not None:
            at_least_one_with_bm25 = True
        if c.get("rrf_score") is not None:
            at_least_one_with_rrf = True
    assert at_least_one_with_dense, "Ít nhất 1 chunk phải có dense_score"
    assert at_least_one_with_bm25, "Ít nhất 1 chunk phải có bm25_score"
    assert at_least_one_with_rrf, "Mọi hybrid chunk phải có rrf_score"


def test_real_rrf_score_is_not_a_confidence(retrieval):
    """RRF score must not be presented as a percentage."""
    res = _retrieve(retrieval, "Điều kiện xét tuyển đại học là gì?", top_k=3)
    for c in res["chunks"]:
        if "rrf_score" in c:
            rrf = float(c["rrf_score"])
            # RRF magnitudes are typically < 0.05; if anyone interpreted it
            # as "3% confidence" they'd be misleading users.
            assert rrf < 0.1, (
                f"RRF={rrf} lớn bất thường — UI không được hiểu là confidence."
            )


# ---------------------------------------------------------------------------
# Diagnostics payload: original/normalized queries + dense/bm25/rrf rankings
# ---------------------------------------------------------------------------


def test_real_diagnostics_contain_all_required_signals(retrieval):
    res = _retrieve(retrieval, "Điều kiện xét tuyển đại học là gì?", top_k=3)
    assert "diagnostics" in res
    diag = res["diagnostics"]
    assert "dense_topk" in diag
    assert "bm25_topk" in diag
    assert "rrf_ranking" in diag
    assert "normalized_query" in res
    assert "intent" in res
    assert "evidence" in res
    # The original query must be preserved somewhere reachable.
    assert res["normalized_query"] or res["intent"]


# ---------------------------------------------------------------------------
# Original query preservation (contract test)
# ---------------------------------------------------------------------------


def test_real_conversation_memory_preserves_original(retrieval):
    """When conversation-memory rewrites the query, the original is preserved
    on the generation result (not the retrieval-side normalized query)."""
    from src.task10_generation import generate_with_citation
    import src.task10_generation as gen
    gen.call_llm = lambda *a, **kw: ""

    history = [
        {"role": "user", "content": "IELTS có được dùng để xét tuyển đại học không?"},
        {"role": "assistant", "content": "..."},
    ]
    res = generate_with_citation(
        "Còn vậy thì sao?",
        top_k=3,
        history=history,
        use_conversation_memory=True,
    )
    assert res["original_query"] == "Còn vậy thì sao?"
    assert res.get("rewrite_info") is not None
    assert res["rewrite_info"]["rewrote"] is True
