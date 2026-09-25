"""Audit-driven answerability state machine tests.

These tests exercise the new end-to-end pipeline layering the audit
demands, using mocks so they run offline. They cover:

    1. Domain gate fires on OOD topics (no retrieval, no citations).
    2. Per-chunk evidence gate filters individually.
    3. Document-level deduplication of final evidence.
    4. RRF is not treated as confidence.
    5. Citation eligibility: only final evidence becomes citations.
    6. The 5 representative query classes (Factual / Paraphrase /
       Keyword-heavy / Ambiguous / OOD) all produce the right action.

No real retrievers are touched — the retrieval layer is monkey-patched.
"""

from __future__ import annotations

import pytest

from src.evidence_quality import (
    classify_domain,
    understand_query,
    score_chunks,
    assess_evidence,
)
from src.task7_reranking import deduplicate_by_document


# ---------------------------------------------------------------------------
# Mock builders (so the integration test does NOT load the embedding model).
# ---------------------------------------------------------------------------

def _chunk(cid: str, content: str, *, dense=0.6, bm25=5.0, rrf=0.03,
           doc=None, source: str | None = None, method: str = "hybrid"):
    """Construct a mock SearchResult.

    Note: ``deduplicate_by_document`` keys on ``metadata.source`` first,
    then the id prefix. To make dedup tests meaningful, we set
    ``metadata.source`` to a unique value per document by default.
    """
    if doc is None:
        doc = cid.split("::")[0] if "::" in cid else "doc-" + cid
    if source is None:
        source = doc  # one unique source per doc → dedup works
    return {
        "id": cid,
        "content": content,
        "score": rrf,
        "dense_score": dense,
        "bm25_score": bm25,
        "rrf_score": rrf,
        "retrieval_method": method,
        "metadata": {
            "title": f"doc {doc}",
            "source": source,
            "url": "",
            "doc_id": doc,
        },
    }


@pytest.fixture
def patched_retrieval(monkeypatch):
    """Patch the two retrievers and PageIndex fallback so integration tests
    can run offline. Returns a factory that lets each test set its own
    candidates."""
    from src import task9_retrieval_pipeline as pipe

    state = {"dense": [], "sparse": [], "pageindex": []}

    def _semantic(q, top_k=10, **_):
        return state["dense"][:top_k]

    def _lexical(q, top_k=10, **_):
        return state["sparse"][:top_k]

    def _pageindex(q, top_k=5, **_):
        return state["pageindex"][:top_k]

    monkeypatch.setattr(pipe, "semantic_search", _semantic)
    monkeypatch.setattr(pipe, "lexical_search", _lexical)
    monkeypatch.setattr(pipe, "pageindex_search", _pageindex)

    def _set(dense=None, sparse=None, pageindex=None):
        if dense is not None:
            state["dense"] = dense
        if sparse is not None:
            state["sparse"] = sparse
        if pageindex is not None:
            state["pageindex"] = pageindex

    return _set


# ---------------------------------------------------------------------------
# 1. Domain gate — runs BEFORE retrieval
# ---------------------------------------------------------------------------

class TestDomainGate:
    @pytest.mark.parametrize(
        "query",
        [
            "Thủ tục đăng ký kết hôn tại Việt Nam như thế nào?",  # marriage
            "Giá Bitcoin hôm nay?",  # crypto
            "Dự báo thời tiết Hà Nội tuần này",  # weather
            "Công thức nấu phở bò gia đình",  # cooking
            "Tôi bị viêm phổi, có nên dùng thuốc kháng sinh?",  # medical
            "Trận chung kết World Cup 2026 diễn ra khi nào?",  # sports
            "Đăng ký xe ô tô mới cần giấy tờ gì?",  # vehicle
            "Mua chung cư Hà Nội năm 2025 cần thủ tục gì?",  # real_estate
            "Thủ tục cấp hộ chiếu cho người lần đầu?",  # procedural_law
        ],
    )
    def test_ood_topics_are_detected(self, query):
        cls = classify_domain(query)
        assert cls["is_in_domain"] is False, (
            f"Expected OOD for: {query!r}, got {cls!r}"
        )
        assert cls["ood_topic"] is not None

    @pytest.mark.parametrize(
        "query",
        [
            "Điều kiện xét tuyển đại học là gì?",
            "Phương thức xét tuyển 2025 gồm những gì?",
            "Tôi có IELTS 6.5 thì có thể xét tuyển đại học không?",
            "Lịch thi tốt nghiệp THPT 2025",
            "Học phí đại học Bách Khoa Hà Nội 2025",
            "Điểm chuẩn ngành CNTT năm 2024?",
        ],
    )
    def test_in_domain_queries_pass_domain_gate(self, query):
        cls = classify_domain(query)
        assert cls["is_in_domain"] is True, (
            f"Expected in-domain for: {query!r}, got {cls!r}"
        )
        assert cls["in_domain_signal"] is not None


# ---------------------------------------------------------------------------
# 2. Per-chunk evidence gate — filters individually
# ---------------------------------------------------------------------------

class TestPerChunkGate:
    def test_chunks_that_do_not_meet_threshold_are_filtered(self):
        intent = understand_query("điều kiện xét tuyển đại học")
        chunks = [
            _chunk("a::1", "Điều kiện xét tuyển đại học 2025", dense=0.71),
            _chunk("b::1", "Trường X thông báo tuyển sinh bổ sung 2025", dense=0.49),
        ]
        scored = score_chunks(intent, chunks, score_threshold=0.50)
        assert scored["passed_count"] == 1
        assert scored["passed"][0]["id"] == "a::1"
        assert scored["rejected"][0]["chunk"]["id"] == "b::1"
        assert "dense_score" in scored["rejected"][0]["reasons"][0]

    def test_chunks_that_fail_keyword_overlap_are_filtered(self):
        intent = understand_query("điều kiện xét tuyển đại học")
        # High dense but no overlap with query content tokens.
        chunks = [
            _chunk("a::1", "Lịch thi tốt nghiệp THPT 2025", dense=0.66),
            _chunk("b::1", "Học phí ngành CNTT 2025", dense=0.66),
        ]
        scored = score_chunks(intent, chunks, score_threshold=0.50)
        # Both fail keyword_overlap.
        assert scored["passed_count"] == 0
        assert scored["rejected"][0]["chunk"]["id"] == "a::1"


# ---------------------------------------------------------------------------
# 3. Document-level deduplication
# ---------------------------------------------------------------------------

class TestDocumentDedup:
    def test_multiple_chunks_from_same_doc_collapse_to_one(self):
        a1 = _chunk("doc-A::chunk-1", "nội dung 1", doc="doc-A")
        a2 = _chunk("doc-A::chunk-2", "nội dung 2", doc="doc-A")
        b = _chunk("doc-B::chunk-1", "nội dung 3", doc="doc-B")
        deduped = deduplicate_by_document([a1, a2, b], top_k=5)
        ids = [c["id"] for c in deduped]
        # doc-A appears at most once.
        assert ids.count("doc-A::chunk-1") + ids.count("doc-A::chunk-2") == 1
        assert "doc-B::chunk-1" in ids

    def test_audit_final_evidence_caps_to_three(self):
        # 6 distinct docs → final must be at most 3.
        chunks = [
            _chunk(f"doc{i}::c1", "nội dung", doc=f"doc{i}", dense=0.7 - i * 0.05)
            for i in range(6)
        ]
        deduped = deduplicate_by_document(chunks, top_k=3)
        assert len(deduped) == 3


# ---------------------------------------------------------------------------
# 4. RRF is not treated as confidence
# ---------------------------------------------------------------------------

class TestRRFNotConfidence:
    def test_assess_evidence_does_not_use_rrf_as_threshold(self):
        # A fake chunk with high dense AND high keyword_overlap must pass,
        # even though RRF is small (RRF is a fusion metric, not evidence).
        intent = understand_query("điều kiện xét tuyển đại học")
        chunks = [_chunk(
            "doc::chunk-1",
            "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT",
            dense=0.71, bm25=10.0, rrf=0.008,
        )]
        ev = assess_evidence(intent, chunks)
        # Threshold metric that could be mis-confused with confidence:
        # we use dense_score (cosine) — NOT RRF — to decide.
        assert ev["status"] == "sufficient"
        assert ev["signals"]["rrf_top1"] < 0.05
        assert ev["signals"]["dense_top1"] >= 0.5

    def test_rrf_below_zero_threshold_does_not_change_pass_fail(self):
        # Same content, two different RRF magnitudes — both should yield
        # the same pass/fail decision because RRF is NOT the threshold.
        intent = understand_query("điều kiện xét tuyển đại học")
        a = _chunk(
            "doc::c1",
            "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT",
            dense=0.7, rrf=0.005,
        )
        b = _chunk(
            "doc::c1",
            "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT",
            dense=0.7, rrf=0.05,
        )
        ev_a = assess_evidence(intent, [a])
        ev_b = assess_evidence(intent, [b])
        # Single strong chunk -> sufficient (strong-single-exception).
        assert ev_a["status"] == ev_b["status"] == "sufficient"


# ---------------------------------------------------------------------------
# 5. End-to-end answerability state machine
# ---------------------------------------------------------------------------

class TestAnswerabilityStateMachine:
    def test_ood_query_returns_out_of_domain_state(self, patched_retrieval):
        # Even with non-empty retriever output, OOD must short-circuit.
        patched_retrieval(
            dense=[_chunk(
                "doc::c1",
                "Trường X tuyển sinh bổ sung năm 2025",
                dense=0.71,
            )],
            sparse=[],
        )
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence(
            "Thủ tục đăng ký kết hôn tại Việt Nam như thế nào?",
            top_k=5,
        )
        assert res["evidence"]["status"] == "out_of_domain"
        assert res["domain_classification"]["ood_topic"] == "marriage"
        assert res["final_evidence"] == []
        assert res["chunks"] == []  # retrieval never even ran

    def test_weak_evidence_falls_through_to_insufficient(self, patched_retrieval):
        # Only one weakly-passing chunk in the corpus.
        patched_retrieval(
            dense=[_chunk(
                "doc::c1",
                "Trường X tuyển sinh bổ sung 2025 với điểm chuẩn a00 cao.",
                dense=0.51,
            )],
            sparse=[],
        )
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence("tổng hợp IELTS 2025", top_k=5)
        # 1 weak-only pass → status=weak → suggested_action=clarify.
        ev = res["evidence"]
        assert ev["status"] in {"weak", "insufficient"}
        if ev["status"] == "weak":
            assert ev["suggested_action"] == "clarify"
        # final_evidence still empty because "weak" → 1-pass rule.
        assert res["final_evidence"] == []

    def test_strong_evidence_with_dedup(self, patched_retrieval):
        # Two distinct docs, each with 2 chunks; we want final = 2 docs (cap 3).
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        chunks = [
            _chunk("doc-A::c1", "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT.", doc="doc-A"),
            _chunk("doc-A::c2", "Phương thức xét tuyển đại học gồm thi THPT và học bạ.", doc="doc-A"),
            _chunk("doc-B::c1", "Tin tức tuyển sinh đại học 2025 mới nhất.", doc="doc-B"),
            _chunk("doc-B::c2", "Lịch thi tốt nghiệp THPT 2025.", doc="doc-B"),
        ]
        patched_retrieval(dense=chunks, sparse=chunks)
        res = retrieve_with_evidence(
            "điều kiện xét tuyển đại học", top_k=5,
        )
        ev = res["evidence"]
        # Multiple chunks pass the gate.
        assert ev["status"] == "sufficient"
        # final_evidence is deduped by document.
        doc_ids = {
            c["metadata"]["doc_id"] for c in res["final_evidence"]
        }
        assert len(doc_ids) == len(res["final_evidence"]), (
            "Each final-evidence entry must belong to a distinct document."
        )
        assert 1 <= len(res["final_evidence"]) <= 3, (
            f"final_evidence must be capped to 1–3, got {len(res['final_evidence'])}"
        )

    def test_one_pass_strong_chunk_is_sufficient(self, patched_retrieval):
        # A single perfect chunk IS sufficient (audit §8: 1–3 citations).
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        chunks = [
            _chunk(
                "doc-A::c1",
                "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT và "
                "hồ sơ hợp lệ, theo quy định của Bộ Giáo dục.",
                dense=0.78, rrf=0.05,
                doc="doc-A",
            )
        ]
        patched_retrieval(dense=chunks, sparse=chunks)
        res = retrieve_with_evidence(
            "điều kiện xét tuyển đại học", top_k=3,
        )
        assert res["evidence"]["status"] == "sufficient"
        assert len(res["final_evidence"]) == 1
        assert res["final_evidence"][0]["id"] == "doc-A::c1"


# ---------------------------------------------------------------------------
# 6. The 5 representative query classes
# ---------------------------------------------------------------------------

class TestFiveQueryClasses:
    """Each test class below corresponds to one of the 5 query classes
    required by the audit. The actions are checked via the in-domain
    ``classify_domain()`` + offline-retrieval mock."""

    @pytest.fixture
    def pipeline(self, monkeypatch):
        """Allow setting retrieval candidates per query."""
        from src import task9_retrieval_pipeline as pipe
        state = {"dense": [], "sparse": []}

        def _sem(q, top_k=10, **_):
            return state["dense"][:top_k]

        def _lex(q, top_k=10, **_):
            return state["sparse"][:top_k]

        monkeypatch.setattr(pipe, "semantic_search", _sem)
        monkeypatch.setattr(pipe, "lexical_search", _lex)
        monkeypatch.setattr(pipe, "pageindex_search", lambda q, top_k=5, **_: [])

        def _set(dense, sparse):
            state["dense"] = dense
            state["sparse"] = sparse

        return _set

    def test_class_1_factual_in_domain(self, pipeline):
        chunks = [
            _chunk("doc-A::c1",
                   "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT.",
                   dense=0.78, doc="doc-A"),
            _chunk("doc-B::c1",
                   "Phương thức xét tuyển đại học 2025 gồm thi và học bạ.",
                   dense=0.72, doc="doc-B"),
        ]
        pipeline(chunks, chunks)
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence("điều kiện xét tuyển đại học", top_k=5)
        assert res["evidence"]["status"] == "sufficient"
        assert 1 <= len(res["final_evidence"]) <= 3

    def test_class_2_paraphrase_in_domain(self, pipeline):
        chunks = [
            _chunk("doc-A::c1",
                   "Quy định về đầu vào đại học gồm có tốt nghiệp THPT.",
                   dense=0.75, doc="doc-A"),
            _chunk("doc-B::c1",
                   "Hồ sơ xét tuyển đại học cần giấy tờ tùy thân.",
                   dense=0.70, doc="doc-B"),
        ]
        pipeline(chunks, chunks)
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence(
            "để vào được đại học thì cần những gì", top_k=5,
        )
        assert res["evidence"]["status"] == "sufficient"
        assert res["final_evidence"]

    def test_class_3_keyword_heavy_in_domain(self, pipeline):
        chunks = [
            _chunk("doc-A::c1",
                   "IELTS 6.5 được chấp nhận làm chứng chỉ xét tuyển đại học.",
                   dense=0.74, doc="doc-A"),
            _chunk("doc-B::c1",
                   "Quy định xét tuyển bằng IELTS 2025 theo từng trường.",
                   dense=0.71, doc="doc-B"),
        ]
        pipeline(chunks, chunks)
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence(
            "IELTS TOEIC VSTEP SAT chứng chỉ xét tuyển đại học",
            top_k=5,
        )
        assert res["evidence"]["status"] == "sufficient"
        assert res["final_evidence"]

    def test_class_4_ambiguous_lop1_clarifies(self, monkeypatch):
        """Primary school + university_admission → conflict → clarify."""
        from src import task9_retrieval_pipeline as pipe
        state = {"dense": [], "sparse": []}
        monkeypatch.setattr(pipe, "semantic_search",
                            lambda q, top_k=10, **_: state["dense"][:top_k])
        monkeypatch.setattr(pipe, "lexical_search",
                            lambda q, top_k=10, **_: state["sparse"][:top_k])
        monkeypatch.setattr(pipe, "pageindex_search",
                            lambda q, top_k=5, **_: [])
        state["dense"] = [
            _chunk("doc::c1",
                   "Phương thức xét tuyển đại học 2025 gồm thi và học bạ.",
                   dense=0.75),
        ]
        state["sparse"] = state["dense"]
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence(
            "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào",
            top_k=5,
        )
        assert res["evidence"]["suggested_action"] == "clarify"
        assert res["final_evidence"] == []

    def test_class_5_out_of_domain_kethon(self, pipeline):
        pipeline(
            dense=[_chunk(
                "doc::c1", "Trường đại học X tuyển sinh bổ sung 2025.",
                dense=0.71)],
            sparse=[],
        )
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        res = retrieve_with_evidence(
            "Thủ tục đăng ký kết hôn tại Việt Nam như thế nào?",
            top_k=5,
        )
        assert res["evidence"]["status"] == "out_of_domain"
        assert res["evidence"]["suggested_action"] == "refuse"
        assert res["final_evidence"] == []
        assert res["chunks"] == []  # retrieval did NOT run


# ---------------------------------------------------------------------------
# 7. End-to-end generate_with_citation contract
# ---------------------------------------------------------------------------

class TestGenerationContract:
    """End-to-end through task10.generate_with_citation."""

    def _setup(self, monkeypatch, dense_chunks):
        from src import task9_retrieval_pipeline as pipe
        monkeypatch.setattr(pipe, "semantic_search",
                            lambda q, top_k=10, **_: dense_chunks[:top_k])
        monkeypatch.setattr(pipe, "lexical_search",
                            lambda q, top_k=10, **_: dense_chunks[:top_k])
        monkeypatch.setattr(pipe, "pageindex_search",
                            lambda q, top_k=5, **_: [])

    def test_ood_kethon_returns_no_sources_and_state_out_of_domain(self, monkeypatch):
        self._setup(monkeypatch, [
            _chunk("doc::c1", "Trường ĐH X tuyển sinh bổ sung.", dense=0.71)
        ])
        from src.task10_generation import generate_with_citation
        import src.task10_generation as gen
        gen.call_llm = lambda *a, **kw: ""  # no real LLM
        res = generate_with_citation(
            "Thủ tục đăng ký kết hôn tại Việt Nam như thế nào?", top_k=5,
        )
        assert res["answerability_state"] == "out_of_domain"
        assert res["sources"] == []
        assert res["action"] == "refuse"
        assert "nằm ngoài phạm vi" in res["answer"].lower()

    def test_weak_in_domain_returns_no_sources(self, monkeypatch):
        self._setup(monkeypatch, [
            _chunk("doc::c1",
                   "Tin tức giáo dục mới nhất hôm nay.", dense=0.51)
        ])
        from src.task10_generation import generate_with_citation
        import src.task10_generation as gen
        gen.call_llm = lambda *a, **kw: ""
        res = generate_with_citation("điều kiện xét tuyển đại học", top_k=5)
        # Either weak → clarify / refuse → no sources.
        assert res["sources"] == []

    def test_strong_in_domain_returns_grounded_citations(self, monkeypatch):
        self._setup(monkeypatch, [
            _chunk("doc-A::c1",
                   "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT.",
                   dense=0.78),
            _chunk("doc-B::c1",
                   "Phương thức xét tuyển đại học gồm thi và học bạ.",
                   dense=0.72),
        ])
        from src.task10_generation import generate_with_citation
        import src.task10_generation as gen
        gen.call_llm = (
            lambda *a, **kw: "Theo quy chế, điều kiện xét tuyển là tốt nghiệp THPT [1]."
        )
        res = generate_with_citation("điều kiện xét tuyển đại học", top_k=5)
        assert res["action"] == "answer"
        assert res["answerability_state"] == "evidence_sufficient"
        assert res["sources"], "In-domain answer must surface citations"
        # Citations are a subset of final_evidence.
        assert {s["id"] for s in res["sources"]}.issubset(
            {s["id"] for s in res["final_evidence"]}
        )
        # Cap enforced.
        assert 1 <= len(res["sources"]) <= 3
        # No duplicate docs in final_evidence.
        docs = {s["metadata"].get("doc_id") for s in res["final_evidence"]}
        assert len(docs) == len(res["final_evidence"])

    def test_duplicate_docs_collapse_in_citations(self, monkeypatch):
        self._setup(monkeypatch, [
            _chunk("doc-A::c1",
                   "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT.",
                   dense=0.78, doc="doc-A"),
            _chunk("doc-A::c2",
                   "Phương thức xét tuyển đại học là thi và học bạ.",
                   dense=0.72, doc="doc-A"),
            _chunk("doc-B::c1",
                   "Hồ sơ xét tuyển đại học theo quy định mới.",
                   dense=0.70, doc="doc-B"),
        ])
        from src.task10_generation import generate_with_citation
        import src.task10_generation as gen
        gen.call_llm = lambda *a, **kw: "Câu trả lời có trích dẫn [1]."
        res = generate_with_citation("điều kiện xét tuyển đại học", top_k=5)
        # A single document must not appear twice in citations.
        docs = [s["metadata"].get("doc_id") for s in res["final_evidence"]]
        assert len(set(docs)) == len(docs), (
            f"final_evidence must be deduped by document, got {docs}"
        )

    def test_rejected_candidates_shape_is_chunk_plus_reasons(self, monkeypatch):
        """The pipeline must emit ``{"chunk": ..., "reasons": [...]}``
        entries for rejected candidates so the UI can show the reason
        next to each rejected document."""
        # Mix of pass / fail so we have both passed and rejected candidates.
        chunks = [
            _chunk("doc-A::c1",
                   "Điều kiện xét tuyển đại học 2025 gồm tốt nghiệp THPT.",
                   dense=0.78, doc="doc-A"),
            _chunk("doc-B::c1",
                   "Lịch thi đấu World Cup 2026 khi nào diễn ra.",
                   dense=0.71, doc="doc-B"),
            _chunk("doc-C::c1",
                   "Tin tức bất động sản Hà Nội tuần qua.",
                   dense=0.69, doc="doc-C"),
        ]
        self._setup(monkeypatch, chunks)
        from src.task10_generation import generate_with_citation
        import src.task10_generation as gen
        gen.call_llm = lambda *a, **kw: "Câu trả lời [1]."
        res = generate_with_citation("điều kiện xét tuyển đại học", top_k=5)
        # Rejected candidates is a list of {"chunk", "reasons"} dicts.
        rc = res["rejected_candidates"]
        assert isinstance(rc, list)
        for entry in rc:
            assert isinstance(entry, dict)
            assert "chunk" in entry and "reasons" in entry
            assert isinstance(entry["chunk"], dict)
            assert isinstance(entry["reasons"], list)
            assert all(isinstance(r, str) for r in entry["reasons"])

        # The rejected-candidates UI helper must consume the same shape.
        from src.ui_components import render_rejected_candidates_panel
        out = render_rejected_candidates_panel(rc)
        assert out
        assert "Rejected candidates" in out
        assert "Citation" not in out
        assert "confidence" not in out.lower()

    def test_rejected_panel_renders_with_real_pipeline(self):
        """Real retrieval call (no monkeypatch): confirm the UI helper accepts
        whatever shape the live pipeline produces."""
        from src.task9_retrieval_pipeline import retrieve_with_evidence
        from src.ui_components import render_rejected_candidates_panel
        # Weak-ambiguous query: should produce some rejected candidates.
        res = retrieve_with_evidence(
            "tôi là học sinh lớp 1 thì nên xét tuyển phương thức đại học nào",
            top_k=5,
        )
        rc = res["rejected_candidates"]
        # May be empty if every retrieved chunk is at the conflict branch,
        # but if non-empty must render correctly.
        out = render_rejected_candidates_panel(rc)
        # Either empty input -> empty string, or non-empty -> non-empty HTML.
        assert isinstance(out, str)
        if rc:
            assert "Rejected candidates" in out
