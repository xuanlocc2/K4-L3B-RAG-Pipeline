"""
Bonus 5 — Conversation memory (query rewriting).

Vấn đề cần giải:
    Chatbot hiện chỉ giữ lịch sử UI, không "nhớ" ngữ cảnh cho retrieval.
    Câu hỏi tiếp theo kiểu "Còn IELTS thì sao?" hoặc "What about IELTS?"
    cần được hiểu theo ngữ cảnh của câu trước đó.

Giải pháp:
    ``rewrite_followup(history, query)`` -> standalone query (string).
    - Nếu query đã đứng độc lập (có đủ thực thể), trả về nguyên văn.
    - Nếu query là follow-up ngắn / thiếu thực thể, dùng heuristic
      (keyword carryover + pronoun resolution) để bổ sung thực thể từ
      câu trước.
    - Nếu LLM provider khả dụng, có thể dùng LLM để rewrite chất lượng
      hơn; nhưng phải có fallback local để pipeline không phụ thuộc.

Tham số:
    history: list các tin nhắn [{"role": "user"|"assistant", "content": str}]
    query: câu hỏi hiện tại của user.

Trả về:
    dict:
        - standalone_query: str  (query đã được viết lại)
        - rewrote: bool           (True nếu query đã được sửa)
        - method: str             ("passthrough" | "local_heuristic" | "llm")
        - last_topic: str | None  (topic trích từ turn trước, nếu có)
"""

from __future__ import annotations

import os
import re
from typing import Any

# Pronoun / cue thường gặp ở câu follow-up tiếng Việt + Anh.
# Chỉ áp dụng khi query đủ NGẮN; tránh match "những" trong câu dài độc lập.
_FOLLOWUP_CUES = (
    "còn ", "còn?", "thì sao", "thế thì", "vậy ", "vậy?", "thì ",
    "what about", "how about", "nó ", "so với", "liệu ", "có phải",
    # các pronoun ngắn đứng đầu câu
    "còn ",
)

# Cue chỉ kích hoạt khi query ngắn (ít token). Tránh false positive.
_FOLLOWUP_CUES_SHORT_ONLY = ("những ", "ở trên", "trên", "họ ", "các ")

# Token tối thiểu để coi là standalone query.
_STANDALONE_MIN_TOKENS = 4

# Các topic "nổi tiếng" của domain này — dùng làm anchor khi heuristic tìm
# thực thể từ lịch sử. Lấy từ golden_dataset.
_KNOWN_TOPICS = [
    "IELTS", "thi tốt nghiệp THPT", "tốt nghiệp THPT", "phương thức xét tuyển",
    "xét tuyển đại học", "điều kiện xét tuyển", "nguyện vọng",
    "học sinh giỏi quốc gia", "điểm ưu tiên", "tổ hợp môn", "A00",
    "VSTEP", "Toán", "Ngữ văn", "Ngoại ngữ", "hồ sơ dự tuyển",
    "tuyển sinh bổ sung", "Tuyensinh247",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text or "", flags=re.UNICODE)


def _looks_like_followup(query: str) -> bool:
    """Câu ngắn / có cue follow-up / chỉ chứa pronoun + entity."""
    lowered = (query or "").lower().strip()
    if not lowered:
        return False
    if any(cue in lowered for cue in _FOLLOWUP_CUES):
        return True
    tokens = _tokenize(lowered)
    # Câu quá ngắn -> coi như follow-up.
    if len(tokens) < _STANDALONE_MIN_TOKENS:
        return True
    # Cue chỉ áp dụng khi query ngắn.
    if len(tokens) <= 8 and any(cue in lowered for cue in _FOLLOWUP_CUES_SHORT_ONLY):
        return True
    return False


def _topic_from_history(history: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    """Trích thực thể / topic từ 1-2 turn gần nhất.

    Trả về (entity, topic_long) hoặc (None, None) nếu không tìm thấy.
    Ưu tiên user turn ngay trước (câu hỏi trước), sau đó mới tới assistant.

    Chọn topic theo thứ tự xuất hiện trong text (salient topic), không
    theo độ dài.
    """
    candidates: list[str] = []
    # duyệt ngược tối đa 3 turn, lấy text của user và assistant.
    for message in reversed(history[-3:]):
        if message.get("role") in {"user", "assistant"}:
            content = (message.get("content") or "").strip()
            if content:
                candidates.append(content)
        if len(candidates) >= 2:
            break
    # entity: tìm topic đã biết xuất hiện SỚM NHẤT trong candidates (ưu tiên
    # topic xuất hiện đầu tiên, không phải topic dài nhất).
    entity = None
    earliest_pos = None
    for cand in candidates:
        lowered = cand.lower()
        for topic in _KNOWN_TOPICS:
            pos = lowered.find(topic.lower())
            if pos < 0:
                continue
            if earliest_pos is None or pos < earliest_pos:
                earliest_pos = pos
                entity = topic
                # không break — tiếp tục tìm nếu có topic xuất hiện sớm hơn
    # Nếu không có topic nào khớp, lấy 4-6 token quan trọng nhất của
    # user turn gần nhất (heuristic đơn giản: bỏ stopword ngắn).
    topic_long = None
    if not entity and candidates:
        tokens = [t for t in _tokenize(candidates[0]) if len(t) > 2]
        topic_long = " ".join(tokens[:5]) or None
    return entity, topic_long


def _try_llm_rewrite(history: list[dict[str, Any]], query: str) -> str | None:
    """Thử dùng LLM để rewrite. Trả None nếu không có key / lỗi.

    Hỗ trợ ``openai`` và ``groq`` (cùng dùng OpenAI SDK). Các provider
    khác hiện không được dùng cho rewrite.
    Chỉ dùng khi provider thật sự khả dụng — không được làm pipeline
    crash nếu thiếu key.
    """
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    try:
        if provider in ("openai", "groq"):
            from openai import OpenAI

            api_key = (
                os.getenv("GROQ_API_KEY") if provider == "groq"
                else os.getenv("OPENAI_API_KEY")
            )
            if not api_key:
                return None
            base_url = (
                "https://api.groq.com/openai/v1" if provider == "groq" else None
            )
            client = OpenAI(api_key=api_key, base_url=base_url)
            default_model = (
                "qwen/qwen3.8-27b" if provider == "groq" else "gpt-4o-mini"
            )
            msgs = [
                {"role": "system", "content": (
                    "Bạn là trợ lý viết lại câu hỏi. Nhiệm vụ: chuyển câu "
                    "follow-up (có thể thiếu thực thể) thành câu độc lập đầy "
                    "đủ nghĩa để truy xuất. Chỉ trả về câu đã viết lại, KHÔNG "
                    "giải thích."
                )},
            ]
            # chỉ gửi tối đa 2 turn trước + query để context nhỏ.
            for msg in history[-2:]:
                if msg.get("role") in {"user", "assistant"}:
                    msgs.append({
                        "role": msg["role"],
                        "content": (msg.get("content") or "")[:300],
                    })
            msgs.append({"role": "user", "content": query})
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "") or default_model,
                messages=msgs,
                temperature=0.0,
                max_tokens=120,
            )
            rewritten = (response.choices[0].message.content or "").strip()
            return rewritten or None
    except Exception:
        return None
    return None


def rewrite_followup(
    history: list[dict[str, Any]],
    query: str,
    *,
    use_llm: bool = False,
) -> dict[str, Any]:
    """Viết lại câu hỏi follow-up thành câu đứng độc lập.

    Quy tắc:
        1. Nếu query rỗng -> trả về query.
        2. Nếu query đã đứng độc lập -> passthrough.
        3. Nếu query là follow-up và có entity anchor trong history -> nối
           entity vào đầu/cuối câu (heuristic, không thay đổi nghĩa).
        4. Nếu ``use_llm=True`` và LLM khả dụng -> dùng LLM rewrite.
    """
    if not query or not query.strip():
        return {
            "standalone_query": query or "",
            "rewrote": False,
            "method": "passthrough",
            "last_topic": None,
        }
    if not _looks_like_followup(query):
        return {
            "standalone_query": query.strip(),
            "rewrote": False,
            "method": "passthrough",
            "last_topic": _topic_from_history(history)[0],
        }

    # Thử LLM nếu được yêu cầu.
    if use_llm:
        llm_out = _try_llm_rewrite(history, query)
        if llm_out:
            entity, _ = _topic_from_history(history)
            return {
                "standalone_query": llm_out,
                "rewrote": llm_out.strip().lower() != query.strip().lower(),
                "method": "llm",
                "last_topic": entity,
            }

    # Fallback: heuristic.
    entity, topic_long = _topic_from_history(history)
    if entity:
        # Nối entity vào đầu nếu query không chứa entity đó.
        if entity.lower() not in query.lower():
            rewritten = f"{entity} {query.rstrip('?').strip()}?"
        else:
            rewritten = query
        return {
            "standalone_query": rewritten.strip(),
            "rewrote": rewritten.strip() != query.strip(),
            "method": "local_heuristic",
            "last_topic": entity,
        }
    if topic_long:
        rewritten = f"{topic_long} {query.rstrip('?').strip()}?"
        return {
            "standalone_query": rewritten.strip(),
            "rewrote": True,
            "method": "local_heuristic",
            "last_topic": topic_long,
        }
    return {
        "standalone_query": query.strip(),
        "rewrote": False,
        "method": "passthrough",
        "last_topic": None,
    }


if __name__ == "__main__":
    history = [
        {"role": "user", "content": "Điều kiện xét tuyển đại học gồm những yêu cầu nào?"},
        {"role": "assistant", "content": "Học sinh cần tốt nghiệp THPT..."},
    ]
    for q in [
        "Còn IELTS thì sao?",
        "Các bài thi tốt nghiệp THPT 2025 gồm những môn nào?",
        "Thì sao?",
    ]:
        print(q, "->", rewrite_followup(history, q))
