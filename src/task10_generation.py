"""
Task 10 — Generation có citation.

Hướng dẫn:
    1. Retrieve top-k chunks.
    2. Reorder để giảm lost-in-the-middle.
    3. Format context kèm title và source.
    4. Gọi provider được chọn trong .env.
    5. Trả answer, sources và retrieval_source.

Nếu context không đủ hoặc provider lỗi, trả safe refusal; không bịa thông tin.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from .bonus3_query_expansion import expand_query
from .bonus5_conversation_memory import rewrite_followup
from .bonus7_citation_grounding import (
    rewrite_with_safe_citations,
    validate_citations,
)
from .task9_retrieval_pipeline import retrieve


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Bạn là trợ lý RAG cho đề tài Tuyển sinh Đại học Việt Nam.
Chỉ trả lời dựa trên context được cung cấp.
Mỗi khẳng định phải có citation kiểu [n] trỏ về tài liệu trong context.
Nếu context không chứa thông tin để trả lời, hãy từ chối một cách an toàn
bằng câu: 'Tôi không thể xác minh thông tin này từ nguồn hiện có.'
Không bịa thêm tài liệu, không đưa ra khẳng định ngoài context.
"""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context (giảm lost-in-the-middle).

    Chiến lược: lấy các phần tử ở vị trí chẵn làm phần đầu, các phần tử ở vị trí
    lẻ đảo ngược để làm phần cuối. Phương pháp này được hỗ trợ bởi nghiên cứu
    về thứ tự long-context (Liu et al., 2023 — "Lost in the Middle").
    """
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label theo citation [n]."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        title = metadata.get("title") or "(không tiêu đề)"
        source = metadata.get("source") or "(không rõ nguồn)"
        url = metadata.get("url") or ""
        header = (
            f"[Document {index} | Title: {title} | Source: {source}"
            + (f" | URL: {url}" if url else "")
            + "]"
        )
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    provider = LLM_PROVIDER
    try:
        if provider == "openai":
            from openai import OpenAI

            client = OpenAI()
            model = LLM_MODEL or "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )
            return response.choices[0].message.content or ""
        if provider == "gemini":
            from google import genai

            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return ""
            client = genai.Client(api_key=api_key)
            model = LLM_MODEL or "gemini-2.0-flash"
            response = client.models.generate_content(
                model=model,
                contents=f"{system_prompt}\n\n{user_message}",
                config={"temperature": TEMPERATURE, "top_p": TOP_P},
            )
            return getattr(response, "text", "") or ""
        if provider == "anthropic":
            from anthropic import Anthropic

            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return ""
            client = Anthropic(api_key=api_key)
            model = LLM_MODEL or "claude-3-5-haiku-20241022"
            response = client.messages.create(
                model=model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                max_tokens=1024,
                temperature=TEMPERATURE,
            )
            # ghép các content block
            return "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
    except Exception as exc:
        print(f"[call_llm] provider={provider} error: {exc}")
    return ""


def _safe_refusal() -> dict:
    return {
        "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
        "sources": [],
        "retrieval_source": "none",
    }


def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    *,
    history: list[dict] | None = None,
    use_conversation_memory: bool = False,
    use_llm_rewrite: bool = False,
    retrieval_mode: str = "baseline",
    dense_weight: float | None = None,
    bm25_weight: float | None = None,
    query_expansion_mode: str = "baseline",
) -> dict:
    """Trả về GenerationResult.

    Sau khi LLM sinh answer, Bonus 7 sẽ kiểm tra các citation ``[n]`` có
    hợp lệ (trỏ về sources) không. Citation không hợp lệ được đánh dấu
    ``[unverified]`` để tránh fabricate nguồn.

    Bonus 5 (Conversation Memory): khi ``use_conversation_memory=True`` và
    ``history`` được cung cấp, câu follow-up sẽ được viết lại thành câu
    độc lập trước khi retrieval. Original query luôn được giữ lại để debug.

    Bonus 3 (Query Expansion): ``query_expansion_mode`` là một trong
    ``baseline`` | ``expansion`` | ``hyde``. ``hyde`` dùng LLM nếu có key,
    fallback về bigram local nếu không. Original query luôn được giữ lại.
    """
    if not query.strip():
        return _safe_refusal()

    # --- Bonus 5: conversation-aware query rewriting ------------------------
    original_query = query
    rewrite_info: dict | None = None
    if use_conversation_memory and history:
        rewrite_info = rewrite_followup(history, query, use_llm=use_llm_rewrite)
        if rewrite_info["rewrote"]:
            query = rewrite_info["standalone_query"]
    # -----------------------------------------------------------------------

    # --- Bonus 3: query expansion --------------------------------------------
    expansion_info = expand_query(query, mode=query_expansion_mode)
    effective_query = expansion_info["expanded_query"]
    # -----------------------------------------------------------------------

    kwargs: dict = {"top_k": top_k}
    if retrieval_mode != "baseline":
        kwargs["retrieval_mode"] = retrieval_mode
    if dense_weight is not None:
        kwargs["dense_weight"] = dense_weight
    if bm25_weight is not None:
        kwargs["bm25_weight"] = bm25_weight
    chunks = retrieve(effective_query, **kwargs)
    if not chunks:
        return _safe_refusal()

    # Bỏ qua các chunk có score quá thấp để tránh đưa thông tin nhiễu vào LLM.
    usable = [c for c in chunks if c.get("score", 0.0) > 0]
    if not usable:
        return _safe_refusal()

    reordered = reorder_for_llm(usable)
    context = format_context(reordered)
    user_message = (
        "Context (hãy trích dẫn [n] trỏ về Document n ở trên):\n"
        f"{context}\n\n"
        f"Câu hỏi: {effective_query}"
    )
    answer = call_llm(SYSTEM_PROMPT, user_message)
    if not answer.strip():
        # Provider lỗi hoặc thiếu key — trả về dạng evidence-only với safe refusal.
        return _safe_refusal()

    # --- Bonus 7: citation grounding -----------------------------------------
    citation_check = validate_citations(answer, chunks)
    grounding = rewrite_with_safe_citations(answer, chunks)
    if grounding["modified"]:
        answer = grounding["answer"]
    # -----------------------------------------------------------------------

    retrieval_source = "hybrid" if chunks[0]["retrieval_method"] in {"hybrid", "hybrid_weighted", "dense", "bm25"} else chunks[0]["retrieval_method"]
    return {
        "answer": answer.strip(),
        "sources": chunks,
        "retrieval_source": retrieval_source,
        "citation_check": citation_check,
        "original_query": original_query,
        "rewrite_info": rewrite_info,
        "expansion_info": expansion_info,
    }


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    result = generate_with_citation("Điều kiện xét tuyển đại học là gì?", top_k=3)
    print(result["answer"])