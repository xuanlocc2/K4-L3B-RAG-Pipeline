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
)


load_dotenv()

st.set_page_config(
    page_title="RAG Tuyển sinh ĐH Việt Nam",
    page_icon="🎓",
    layout="wide",
)

PROJECT_TITLE = "RAG Tuyển sinh Đại học Việt Nam"
PROJECT_DESCRIPTION = (
    "Chatbot trả lời câu hỏi về quy chế tuyển sinh, điều kiện xét tuyển, "
    "phương thức xét tuyển, kỳ thi tốt nghiệp THPT và tin tức giáo dục Việt Nam. "
    "Dữ liệu được tổng hợp từ Bộ Giáo dục & Đào tạo (MOET) và các nguồn báo "
    "chí uy tín (Thanh Niên, VietnamNet, Dân trí, Tuyensinh247)."
)

SUGGESTED_QUESTIONS = [
    "Điều kiện xét tuyển đại học gồm những yêu cầu nào?",
    "Các phương thức xét tuyển phổ biến hiện nay là gì?",
    "Các bài thi tốt nghiệp THPT 2025 gồm những môn nào?",
    "IELTS có được dùng để xét tuyển đại học không?",
    "Tóm tắt Quy chế thi tốt nghiệp THPT 2025",
    "Thủ tục đăng ký kết hôn tại Việt Nam như thế nào? (Câu ngoài domain)",
]

RETRIEVAL_MODE_LABELS = {
    "baseline": "Hybrid + RRF (mặc định)",
    "dense_only": "Dense-only (benchmark)",
    "bm25_only": "BM25-only (benchmark)",
    "weighted_rrf": "Weighted RRF (Bonus 1)",
    "weighted_rrf_dedup": "Weighted RRF + dedup (Bonus 1+2)",
    "rrf_dedup": "RRF + dedup (Bonus 2)",
}

BENCHMARK_PATH = ROOT / "group_project" / "evaluation" / "bonus_benchmark.json"


# --- State ------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "settings" not in st.session_state:
    st.session_state.settings = {
        "top_k": DEFAULT_TOP_K,
        "score_threshold": DEFAULT_SCORE_THRESHOLD,
        "retrieval_mode": "baseline",
        "use_conversation_memory": True,
        "debug_mode": False,
    }


# --- Sidebar ----------------------------------------------------------------


def _render_sidebar() -> None:
    with st.sidebar:
        st.title(PROJECT_TITLE)
        st.caption("Lab Day 8 — Hybrid RAG Pipeline")

        st.subheader("Cấu hình truy vấn")
        top_k = st.slider(
            "Số chunk truy xuất (top_k)",
            min_value=1,
            max_value=10,
            value=st.session_state.settings["top_k"],
            step=1,
        )
        threshold = st.slider(
            "Ngưỡng fallback (cosine score)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.settings["score_threshold"]),
            step=0.05,
        )
        retrieval_mode = st.selectbox(
            "Chế độ retrieval (Bonus 1+2)",
            options=sorted(VALID_RETRIEVAL_MODES),
            format_func=lambda v: RETRIEVAL_MODE_LABELS.get(v, v),
            index=sorted(VALID_RETRIEVAL_MODES).index(
                st.session_state.settings["retrieval_mode"]
            ),
        )
        use_memory = st.checkbox(
            "Conversation memory (Bonus 5)",
            value=st.session_state.settings["use_conversation_memory"],
            help="Viết lại câu hỏi follow-up thành câu độc lập dựa trên lịch sử.",
        )
        debug_mode = st.checkbox(
            "Debug / Observability mode (Bonus 9)",
            value=st.session_state.settings["debug_mode"],
            help="Hiển thị query gốc/sau rewrite, dense/BM25 candidates, citation grounding.",
        )

        st.session_state.settings.update({
            "top_k": top_k,
            "score_threshold": threshold,
            "retrieval_mode": retrieval_mode,
            "use_conversation_memory": use_memory,
            "debug_mode": debug_mode,
        })

        st.subheader("Provider")
        st.write(f"LLM: `{os.getenv('LLM_PROVIDER', 'openai')}`")
        st.write(f"Model: `{os.getenv('LLM_MODEL', 'auto') or 'auto'}`")
        st.write(f"Embedding: `{os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')}`")

        st.subheader("Giới thiệu")
        st.markdown(
            "- Hybrid retrieval: dense (BGE-m3) + BM25 + RRF\n"
            "- Weighted RRF + document dedup (Bonus 1+2)\n"
            "- Citation grounding (Bonus 7)\n"
            "- Fallback khi dense score thấp\n"
            "- Safe refusal cho câu ngoài domain"
        )

        if st.button("🧹 Xóa hội thoại", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


_render_sidebar()


# --- Analytics (Bonus 10) ---------------------------------------------------


def _render_analytics() -> None:
    if not BENCHMARK_PATH.exists():
        return
    with st.expander("📊 Retrieval analytics (Bonus 10)", expanded=False):
        try:
            data = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
        except Exception:
            return
        rows = []
        for key in ("A_dense_only", "B_baseline_rrf", "C_weighted_rrf", "D_weighted_rrf_dedup"):
            item = data.get(key)
            if not item:
                continue
            rows.append({
                "strategy": RETRIEVAL_MODE_LABELS.get(
                    {"A_dense_only": "dense_only",
                     "B_baseline_rrf": "baseline",
                     "C_weighted_rrf": "weighted_rrf",
                     "D_weighted_rrf_dedup": "weighted_rrf_dedup"}[key],
                    key,
                ),
                "context_precision": item.get("context_precision", 0),
                "context_recall": item.get("context_recall", 0),
                "source_hit_rate": item.get("source_hit_rate", 0),
                "doc_diversity_avg": item.get("doc_diversity_avg", 0),
                "avg_latency_ms": item.get("avg_latency_ms", 0),
            })
        st.markdown(
            "Bảng dưới được sinh từ `scripts/bonus_benchmark.py` trên "
            "`group_project/evaluation/bonus_benchmark.json`. Số liệu thật, "
            "không tổng hợp tay."
        )
        st.markdown(render_analytics_table(rows), unsafe_allow_html=True)


# --- Main layout ------------------------------------------------------------

st.title(PROJECT_TITLE)
st.caption(PROJECT_DESCRIPTION)

st.markdown("### Câu hỏi gợi ý")
cols = st.columns(2)
for index, q in enumerate(SUGGESTED_QUESTIONS):
    col = cols[index % 2]
    with col:
        if st.button(q, use_container_width=True, key=f"suggest_{index}"):
            st.session_state.pending_question = q

_render_analytics()


# --- Render chat history ----------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
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


# --- Handle user input ------------------------------------------------------

pending = st.session_state.pop("pending_question", None)
typed = st.chat_input("Nhập câu hỏi về tuyển sinh đại học...")
user_input = pending or typed

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Đang truy xuất và sinh câu trả lời..."):
            settings = st.session_state.settings
            # Bonus 5: build history (trừ message hiện tại vừa push ở trên)
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
                st.markdown(answer)

                # Bonus 6: Citation panel với highlight
                if sources:
                    with st.expander("📚 Nguồn trích dẫn", expanded=True):
                        st.markdown(
                            render_citation_panel(sources),
                            unsafe_allow_html=True,
                        )
                    # Bonus 9: Debug panel
                    debug_html = ""
                    if settings["debug_mode"]:
                        # Trong debug mode, tái chạy retrieval để show dense + BM25.
                        dense = semantic_search(user_input, top_k=settings["top_k"] * 2)
                        bm = lexical_search(user_input, top_k=settings["top_k"] * 2)
                        debug_html = render_debug_panel(
                            query=user_input,
                            original_query=original_query if original_query != user_input else None,
                            rewrite_info=rewrite_info,
                            retrieval_mode=settings["retrieval_mode"],
                            score_threshold=float(settings["score_threshold"]),
                            citation_check=citation_check,
                            dense_results=dense[:5],
                            bm25_results=bm[:5],
                            final_results=sources,
                            retrieval_source=retrieval_source,
                        )
                        with st.expander("🔍 Debug / Observability", expanded=True):
                            st.markdown(debug_html, unsafe_allow_html=True)
                    st.caption(f"Retrieval source: `{retrieval_source}`")
                else:
                    st.caption("Không truy xuất được nguồn — safe refusal.")
                    debug_html = ""

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "retrieval_source": retrieval_source,
                        "debug_html": debug_html,
                    }
                )
            except Exception as exc:
                st.error(f"Lỗi khi sinh câu trả lời: {exc}")
