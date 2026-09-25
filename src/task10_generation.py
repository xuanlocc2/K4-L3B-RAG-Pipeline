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

import os

from dotenv import load_dotenv

<<<<<<< Updated upstream
from .task9_retrieval_pipeline import retrieve
=======
from .bonus3_query_expansion import expand_query
from .bonus5_conversation_memory import rewrite_followup
from .bonus7_citation_grounding import (
    rewrite_with_safe_citations,
    validate_citations,
)
from .evidence_quality import (
    make_clarification_message,
    make_out_of_domain_message,
    make_weak_evidence_message,
)
from .task9_retrieval_pipeline import retrieve_with_evidence
>>>>>>> Stashed changes


load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời chỉ từ context được cung cấp.
Mỗi khẳng định phải có citation. Nếu thiếu evidence, hãy từ chối xác minh."""


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Đưa chunks quan trọng về đầu và cuối context."""
    # TODO: Implement document reordering.
    #
    # if len(chunks) <= 2:
    #     return list(chunks)
    # front = chunks[::2]
    # back = chunks[1::2]
    # return front + back[::-1]
    raise NotImplementedError("Implement reorder_for_llm")


def format_context(chunks: list[dict]) -> str:
    """Tạo context có title và source label."""
    # TODO: Format chunks để LLM tạo citation kiểm chứng được.
    #
    # parts = []
    # for index, chunk in enumerate(chunks, 1):
    #     metadata = chunk["metadata"]
    #     parts.append(
    #         f"[Document {index} | Title: {metadata['title']} | "
    #         f"Source: {metadata['source']}]\n{chunk['content']}"
    #     )
    # return "\n\n---\n\n".join(parts)
    raise NotImplementedError("Implement format_context")


def call_llm(system_prompt: str, user_message: str) -> str:
<<<<<<< Updated upstream
    """Gọi OpenAI, Gemini hoặc Anthropic theo cấu hình."""
    # TODO: Dispatch theo LLM_PROVIDER.
    #
    # - openai    -> OPENAI_API_KEY
    # - gemini    -> GEMINI_API_KEY
    # - anthropic -> ANTHROPIC_API_KEY
    #
    # Dùng LLM_MODEL và trả về text thuần cho cả ba nhánh.
    raise NotImplementedError("Implement call_llm")


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Trả về GenerationResult."""
    # TODO: Implement end-to-end generation.
    #
    # chunks = retrieve(query, top_k=top_k)
    # if not chunks:
    #     return {
    #         "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
    #         "sources": [],
    #         "retrieval_source": "none",
    #     }
    # reordered = reorder_for_llm(chunks)
    # context = format_context(reordered)
    # user_message = f"Context:\n{context}\n\nQuestion: {query}"
    # answer = call_llm(SYSTEM_PROMPT, user_message)
    # return {
    #     "answer": answer,
    #     "sources": chunks,
    #     "retrieval_source": chunks[0]["retrieval_method"],
    # }
    raise NotImplementedError("Implement generate_with_citation")
=======
    """Gọi OpenAI, Groq, Gemini hoặc Anthropic theo cấu hình.

    Supported providers:
        - ``openai``   : OpenAI chính thức, dùng ``OPENAI_API_KEY``.
        - ``groq``     : Groq Cloud (OpenAI-compatible API),
          dùng ``GROQ_API_KEY`` và base URL ``https://api.groq.com/openai/v1``.
        - ``gemini``   : Google Gemini, dùng ``GEMINI_API_KEY``.
        - ``anthropic``: Anthropic Claude, dùng ``ANTHROPIC_API_KEY``.

    Provider nào thiếu key sẽ trả về chuỗi rỗng -> ``generate_with_citation``
    sẽ fallback sang safe refusal (không bao giờ làm UI crash).
    """
    provider = LLM_PROVIDER
    try:
        if provider in ("openai", "groq"):
            # Cả OpenAI và Groq đều dùng OpenAI SDK; Groq trỏ base_url khác.
            from openai import OpenAI

            api_key = (
                os.getenv("GROQ_API_KEY") if provider == "groq"
                else os.getenv("OPENAI_API_KEY")
            )
            if not api_key:
                return ""
            base_url = (
                "https://api.groq.com/openai/v1" if provider == "groq" else None
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
            # Default model Groq: qwen/qwen3-32b nếu có, fallback về
            # qwen/qwen3.8-27b (chat model phổ biến trên hầu hết account,
            # kể cả free tier; cho chất lượng tốt và hỗ trợ tiếng Việt).
            # Có thể override qua LLM_MODEL.
            default_model = (
                "qwen/qwen3.8-27b" if provider == "groq" else "gpt-4o-mini"
            )
            model = LLM_MODEL or default_model
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

    Flow:
        1. Optional conversation-memory rewrite (Bonus 5) — keeps original.
        2. Optional query expansion (Bonus 3) — keeps original.
        3. ``retrieve_with_evidence`` runs dense + BM25 + RRF, then
           evidence-quality gate decides:
             * answer -> grounded generation with separated scores
             * clarify -> clarification message, NO cited sources
             * refuse -> safe refusal, NO cited sources
        4. After LLM answer is produced, citation grounding (Bonus 7) still
           validates ``[n]`` markers against the actually-used sources.

    Citation honesty:
        If evidence.status is not "sufficient", we DO NOT show retrieved
        chunks as "supporting sources" for the answer. They are still
        available via the diagnostics block.
    """
    if not query or not query.strip():
        return _safe_refusal()

    original_query = query
    rewrite_info: dict | None = None
    if use_conversation_memory and history:
        rewrite_info = rewrite_followup(history, query, use_llm=use_llm_rewrite)
        if rewrite_info["rewrote"]:
            query = rewrite_info["standalone_query"]

    expansion_info = expand_query(query, mode=query_expansion_mode)
    effective_query = expansion_info["expanded_query"]

    kwargs: dict = {"top_k": top_k}
    if retrieval_mode != "baseline":
        kwargs["retrieval_mode"] = retrieval_mode
    if dense_weight is not None:
        kwargs["dense_weight"] = dense_weight
    if bm25_weight is not None:
        kwargs["bm25_weight"] = bm25_weight

    retrieved = retrieve_with_evidence(effective_query, **kwargs)
    chunks = retrieved["chunks"]
    intent = retrieved["intent"]
    evidence = retrieved["evidence"]
    diagnostics = retrieved["diagnostics"]
    normalized_query = retrieved["normalized_query"]
    classification = retrieved.get("domain_classification") or {}
    final_evidence = retrieved.get("final_evidence") or []
    rejected_candidates = retrieved.get("rejected_candidates") or []

    # Answerability state machine (audit §11).
    #   DOMAIN_CHECK
    #   ├── OUT_OF_DOMAIN  → refuse        (Vietnamese refusal, no citations)
    #   EVIDENCE_CHECK
    #   ├── EVIDENCE_SUFFICIENT → answer  (final_evidence has 1–3 support)
    #   ├── EVIDENCE_WEAK → clarify       (no citations)
    #   └── EVIDENCE_INSUFFICIENT → refuse (no citations)

    # State 1: OOD short-circuit.
    if classification.get("is_in_domain") is False:
        return {
            "answer": make_out_of_domain_message(classification),
            "sources": [],
            "retrieval_source": "none",
            "evidence": evidence,
            "intent": intent,
            "domain_classification": classification,
            "normalized_query": normalized_query,
            "diagnostics": diagnostics,
            "rejected_candidates": rejected_candidates,
            "final_evidence": [],
            "original_query": original_query,
            "rewrite_info": rewrite_info,
            "expansion_info": expansion_info,
            "action": "refuse",
            "answerability_state": "out_of_domain",
            "citation_check": {"valid": True, "grounded_ratio": 1.0,
                               "missing_citation": False,
                               "invalid_refs": [], "total_refs": 0},
        }

    # Evidence gate routing.
    action = evidence.get("suggested_action", "refuse")
    state = evidence.get("status", "insufficient")  # "weak"/"sufficient"/"insufficient"

    if action == "clarify":
        return {
            "answer": make_clarification_message(intent, original_query),
            "sources": [],
            "retrieval_source": "none",
            "evidence": evidence,
            "intent": intent,
            "domain_classification": classification,
            "normalized_query": normalized_query,
            "diagnostics": diagnostics,
            "rejected_candidates": rejected_candidates,
            "final_evidence": [],
            "original_query": original_query,
            "rewrite_info": rewrite_info,
            "expansion_info": expansion_info,
            "action": "clarify",
            "answerability_state": "evidence_weak",
            "citation_check": {"valid": True, "grounded_ratio": 1.0,
                               "missing_citation": False,
                               "invalid_refs": [], "total_refs": 0},
        }

    if action == "refuse" or not final_evidence:
        # NO citations — same honesty rule. Choose message by state:
        answer_text = (
            make_weak_evidence_message(intent)
            if state == "weak"
            else _safe_refusal()["answer"]
        )
        return {
            "answer": answer_text,
            "sources": [],
            "retrieval_source": "none",
            "evidence": evidence,
            "intent": intent,
            "domain_classification": classification,
            "normalized_query": normalized_query,
            "diagnostics": diagnostics,
            "rejected_candidates": rejected_candidates,
            "final_evidence": [],
            "original_query": original_query,
            "rewrite_info": rewrite_info,
            "expansion_info": expansion_info,
            "action": "refuse",
            "answerability_state": (
                "evidence_weak" if state == "weak"
                else "evidence_insufficient"
            ),
            "citation_check": {"valid": True, "grounded_ratio": 1.0,
                               "missing_citation": False,
                               "invalid_refs": [], "total_refs": 0},
        }

    # Action == "answer": proceed with grounded generation.
    # CRITICAL (audit §3 / §5): the LLM context is built from final_evidence,
    # NOT from the raw retrieval top-K. This keeps retrieval and citation
    # strictly separate.
    usable = [c for c in final_evidence if float(c.get("score", 0.0)) > 0]
    if not usable:
        return {
            "answer": _safe_refusal()["answer"],
            "sources": [],
            "retrieval_source": "none",
            "evidence": evidence,
            "intent": intent,
            "domain_classification": classification,
            "normalized_query": normalized_query,
            "diagnostics": diagnostics,
            "rejected_candidates": rejected_candidates,
            "final_evidence": [],
            "original_query": original_query,
            "rewrite_info": rewrite_info,
            "expansion_info": expansion_info,
            "action": "refuse",
            "answerability_state": "evidence_insufficient",
            "citation_check": {"valid": True, "grounded_ratio": 1.0,
                               "missing_citation": False,
                               "invalid_refs": [], "total_refs": 0},
        }

    reordered = reorder_for_llm(usable)
    context = format_context(reordered)
    user_message = (
        "Context (hãy trích dẫn [n] trỏ về Document n ở trên):\n"
        f"{context}\n\n"
        f"Câu hỏi: {effective_query}"
    )
    answer = call_llm(SYSTEM_PROMPT, user_message)
    if not answer.strip():
        # Provider lỗi hoặc thiếu key — fallback evidence-only.
        return {
            "answer": _safe_refusal()["answer"],
            "sources": [],
            "retrieval_source": "none",
            "evidence": evidence,
            "intent": intent,
            "domain_classification": classification,
            "normalized_query": normalized_query,
            "diagnostics": diagnostics,
            "rejected_candidates": rejected_candidates,
            "final_evidence": [],
            "original_query": original_query,
            "rewrite_info": rewrite_info,
            "expansion_info": expansion_info,
            "action": "refuse",
            "answerability_state": "evidence_insufficient",
            "citation_check": {"valid": True, "grounded_ratio": 1.0,
                               "missing_citation": False,
                               "invalid_refs": [], "total_refs": 0},
        }

    citation_check = validate_citations(answer, usable)
    grounding = rewrite_with_safe_citations(answer, usable)
    if grounding["modified"]:
        answer = grounding["answer"]

    retrieval_source = (
        usable[0]["retrieval_method"] if usable else "none"
    )
    return {
        "answer": answer.strip(),
        "sources": usable,           # = final_evidence (cap 1–3, deduped)
        "retrieval_source": retrieval_source,
        "citation_check": citation_check,
        "evidence": evidence,
        "intent": intent,
        "domain_classification": classification,
        "normalized_query": normalized_query,
        "diagnostics": diagnostics,
        "rejected_candidates": rejected_candidates,
        "final_evidence": usable,
        "original_query": original_query,
        "rewrite_info": rewrite_info,
        "expansion_info": expansion_info,
        "action": "answer",
        "answerability_state": "evidence_sufficient",
    }
>>>>>>> Stashed changes


if __name__ == "__main__":
    print(generate_with_citation("test query"))
