"""Streamlit RAG Chatbot — Tuyển sinh Đại học Việt Nam.

Giao diện chat với:
    - Lịch sử hội thoại và hiển thị nguồn trích dẫn.
    - Câu hỏi gợi ý cho demo.
    - Điều chỉnh `top_k`, threshold, retrieval_mode (Bonus 1+2).
    - Citation panel với Relevant passage highlight (Bonus 6).
    - Debug / observability panel để minh hoạ hybrid retrieval (Bonus 9).
    - Conversation memory: bật/tắt rewrite follow-up (Bonus 5).
    - Trạng thái fallback PageIndex khi điểm dense thấp.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Đảm bảo import được các module src.* khi chạy `streamlit run app.py`
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from src.task10_generation import generate_with_citation  # noqa: E402
from src.task9_retrieval_pipeline import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    VALID_RETRIEVAL_MODES,
)
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import (  # noqa: E402
    deduplicate_by_document,
    rerank_rrf,
    rerank_weighted_rrf,
)
from src.ui_components import (  # noqa: E402
    render_analytics_table,
    render_citation_panel,
    render_debug_panel,
    render_insufficient_evidence_panel,
    render_rejected_candidates_panel,
    render_state_banner,
)


load_dotenv()

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.title("RAG Chatbot")
    st.caption("Thay mô tả theo đề tài của nhóm")
    top_k = st.slider("Số chunks", 3, 10, 5)

st.title("RAG Chatbot")
st.caption("Thay tiêu đề và hướng dẫn sử dụng")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            # Citation panel is ONLY shown for grounded answers (audit §13).
            # For refuse / clarify / out-of_domain there must be no citations
            # even if a previous bug stored non-empty sources.
            if message.get("action") == "answer" and message.get("sources"):
                with st.expander("📚 Nguồn trích dẫn", expanded=False):
                    st.markdown(
                        render_citation_panel(message["sources"]),
                        unsafe_allow_html=True,
                    )
                if message.get("debug_html"):
                    with st.expander("🔍 Debug", expanded=False):
                        st.markdown(message["debug_html"], unsafe_allow_html=True)
                st.caption(
                    f"Retrieval source: `{message.get('retrieval_source', '?')}`"
                )
            else:
                st.caption(
                    f"Action: `{message.get('action', '?')}`"
                    f" · State: `{message.get('answerability_state', '?')}`"
                )
                if message.get("debug_html"):
                    with st.expander("🔍 Debug", expanded=False):
                        st.markdown(message["debug_html"], unsafe_allow_html=True)

query = st.chat_input("Nhập câu hỏi...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            settings = st.session_state.settings
            history = st.session_state.messages[:-1]
            try:
                result = generate_with_citation(
                    user_input,
                    top_k=settings["top_k"],
                    history=history,
                    use_conversation_memory=settings["use_conversation_memory"],
                    retrieval_mode=settings["retrieval_mode"],
                )
                answer = result["answer"]
                sources = result["sources"]
                retrieval_source = result["retrieval_source"]
                citation_check = result.get("citation_check")
                original_query = result.get("original_query")
                rewrite_info = result.get("rewrite_info")
                evidence = result.get("evidence") or {}
                intent = result.get("intent") or {}
                normalized_query = result.get("normalized_query")
                diagnostics = result.get("diagnostics")
                action = result.get("action", "answer")
                answerability_state = result.get(
                    "answerability_state", action,
                )
                rejected_candidates = result.get("rejected_candidates") or []
                st.markdown(answer)

                # Visible answerability state (audit §13 / §11).
                banner_html = render_state_banner(
                    action=action,
                    answerability_state=answerability_state,
                    evidence_status=evidence.get("status"),
                )
                if banner_html:
                    st.markdown(banner_html, unsafe_allow_html=True)

                # Evidence-quality routing:
                #   - answer   : citation panel như bình thường
                #   - clarify  : KHÔNG hiện "Nguồn trích dẫn"; chỉ hiện
                #                "Retrieved nhưng chưa đủ bằng chứng" trong
                #                block debug.
                #   - refuse   : tương tự clarify.
                if action == "answer" and sources:
                    with st.expander("📚 Nguồn trích dẫn", expanded=True):
                        st.markdown(
                            render_citation_panel(
                                sources,
                                show_debug_scores=settings["debug_mode"],
                            ),
                            unsafe_allow_html=True,
                        )
                    debug_html = ""
                    if settings["debug_mode"]:
                        debug_html = render_debug_panel(
                            query=user_input,
                            original_query=original_query,
                            rewrite_info=rewrite_info,
                            retrieval_mode=settings["retrieval_mode"],
                            score_threshold=float(settings["score_threshold"]),
                            citation_check=citation_check,
                            dense_results=(
                                diagnostics or {}
                            ).get("dense_topk", []),
                            bm25_results=(
                                diagnostics or {}
                            ).get("bm25_topk", []),
                            final_results=sources,
                            retrieval_source=retrieval_source,
                            evidence=evidence,
                            intent=intent,
                            normalized_query=normalized_query,
                            diagnostics=diagnostics,
                        )
                        # Surface rejected candidates in debug only (audit §14).
                        debug_html += render_rejected_candidates_panel(
                            rejected_candidates
                        )
                        with st.expander("🔍 Debug / Observability", expanded=True):
                            st.markdown(debug_html, unsafe_allow_html=True)
                    st.caption(
                        f"Retrieval source: `{retrieval_source}` "
                        f"· Evidence: `{evidence.get('status', '?')}` "
                        f"· Relevance: `{evidence.get('relevance_label', '?')}`"
                    )
                else:
                    # clarify / refuse — KHÔNG hiển thị nguồn trích dẫn.
                    st.caption(
                        f"Retrieval source: `{retrieval_source}` "
                        f"· Action: `{action}` "
                        f"· Evidence: `{evidence.get('status', '?')}` "
                        f"· Relevance: `{evidence.get('relevance_label', '?')}`"
                    )
                    debug_html = ""
                    if settings["debug_mode"]:
                        # Debug block vẫn cho thấy retrieved chunks — nhưng
                        # ghi rõ "KHÔNG dùng làm nguồn trích dẫn".
                        insufficient_html = render_insufficient_evidence_panel(
                            diagnostics.get("rrf_ranking", []) if diagnostics else []
                        )
                        debug_html = render_debug_panel(
                            query=user_input,
                            original_query=original_query,
                            rewrite_info=rewrite_info,
                            retrieval_mode=settings["retrieval_mode"],
                            score_threshold=float(settings["score_threshold"]),
                            citation_check=citation_check,
                            dense_results=(
                                diagnostics or {}
                            ).get("dense_topk", []),
                            bm25_results=(
                                diagnostics or {}
                            ).get("bm25_topk", []),
                            final_results=sources,
                            retrieval_source=retrieval_source,
                            evidence=evidence,
                            intent=intent,
                            normalized_query=normalized_query,
                            diagnostics=diagnostics,
                        )
                        debug_html = debug_html + insufficient_html
                        with st.expander("🔍 Debug / Observability", expanded=True):
                            st.markdown(debug_html, unsafe_allow_html=True)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieval_source": retrieval_source,
                        "action": action,
                        "answerability_state": answerability_state,
                        "debug_html": debug_html,
                    }
                )
            except Exception as exc:
                st.error(f"Lỗi khi sinh câu trả lời: {exc}")
