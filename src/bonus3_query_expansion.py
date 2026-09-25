"""
Bonus 3 — Query expansion (optional, lightweight).

Hai chế độ:
    - ``expansion``  : mở rộng query bằng cách thêm bigram tiếng Việt hay đi
      cùng các từ trong query (dựa trên corpus). Không cần LLM.
    - ``hyde``       : sinh "hypothetical document" bằng LLM nếu có key,
      fallback về một mẫu deterministic nếu không.

Giao diện:
    ``expand_query(query, mode='expansion'|'hyde'|'baseline')``
        - ``baseline``  : trả về query nguyên văn.
        - ``expansion`` : thêm bigram lấy từ corpus (top-K=3).
        - ``hyde``      : dùng LLM nếu có key, nếu không thì dùng fallback.

Trả về dict:
    - expanded_query: str
    - method: str
    - error: str | None
"""

from __future__ import annotations

import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

CORPUS_PATH = Path(__file__).parent.parent / "data" / "corpus_cache.json"

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)


def _load_corpus() -> list[dict[str, Any]]:
    if not CORPUS_PATH.exists():
        return []
    import json
    try:
        payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
        # cache file có cấu trúc {"chunks": [...]}
        if isinstance(payload, dict):
            return list(payload.get("chunks", []))
        return list(payload)
    except Exception:
        return []


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _build_bigram_index(corpus: list[dict]) -> dict[str, Counter]:
    """Với mỗi token, đếm bigram (token, next_token) xuất hiện trong corpus."""
    index: dict[str, Counter] = {}
    for doc in corpus:
        tokens = _tokenize(doc.get("content", ""))
        for i in range(len(tokens) - 1):
            cur, nxt = tokens[i], tokens[i + 1]
            if len(cur) < 3 or len(nxt) < 2:
                continue
            index.setdefault(cur, Counter())[nxt] += 1
    return index


_BIGRAM_INDEX: dict[str, Counter] | None = None


def _get_index() -> dict[str, Counter]:
    global _BIGRAM_INDEX
    if _BIGRAM_INDEX is None:
        _BIGRAM_INDEX = _build_bigram_index(_load_corpus())
    return _BIGRAM_INDEX


def expand_via_bigrams(query: str, top_k: int = 3) -> str:
    """Thêm bigram phổ biến nhất theo từng token trong query.

    Ví dụ: "IELTS xét tuyển" -> có thể thêm "ielts toeic quốc tế" nếu corpus
    chứa cụm này nhiều.
    """
    tokens = _tokenize(query)
    if not tokens:
        return query
    index = _get_index()
    expansions: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if len(token) < 3:
            continue
        candidates = index.get(token)
        if not candidates:
            continue
        for nxt, _ in candidates.most_common(top_k):
            bigram = f"{token} {nxt}"
            if bigram not in seen and nxt not in tokens:
                expansions.append(nxt)
                seen.add(bigram)
    if not expansions:
        return query
    return f"{query} {' '.join(expansions)}".strip()


def _hyde_via_llm(query: str) -> str | None:
    """Sinh hypothetical document bằng LLM. Trả None nếu không có key.

    Hỗ trợ ``openai`` và ``groq`` (cùng dùng OpenAI SDK). Khi thiếu key
    hoặc lỗi sẽ trả về None để fallback dùng local bigram expansion.
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
            response = client.chat.completions.create(
                model=os.getenv("LLM_MODEL", "") or default_model,
                messages=[
                    {"role": "system", "content": (
                        "Bạn tạo một đoạn văn ngắn (3-4 câu) trả lời câu hỏi về "
                        "tuyển sinh đại học Việt Nam. Đoạn văn có thể chứa thông "
                        "tin chưa chính xác — mục đích chỉ là tạo query mở rộng "
                        "cho retrieval."
                    )},
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=200,
            )
            text = (response.choices[0].message.content or "").strip()
            return text or None
    except Exception:
        return None
    return None


def _hyde_fallback(query: str) -> str:
    """Fallback khi không có LLM key: tạo pseudo-HyDE bằng bigram expansion.

    Không claim là hypothetical document thật; chỉ mở rộng query để retrieval
    có thêm từ khoá liên quan.
    """
    return expand_via_bigrams(query, top_k=5)


def expand_query(query: str, *, mode: str = "baseline") -> dict[str, Any]:
    """Mở rộng query theo mode.

    Args:
        mode: ``baseline`` | ``expansion`` | ``hyde``
    """
    if not query or not query.strip():
        return {"expanded_query": query or "", "method": "passthrough", "error": None}
    mode = (mode or "baseline").lower()
    if mode == "baseline":
        return {"expanded_query": query.strip(), "method": "baseline", "error": None}
    if mode == "expansion":
        expanded = expand_via_bigrams(query)
        return {
            "expanded_query": expanded,
            "method": "local_bigram_expansion",
            "error": None,
        }
    if mode == "hyde":
        hyde = _hyde_via_llm(query)
        if hyde:
            return {"expanded_query": hyde, "method": "hyde_llm", "error": None}
        return {
            "expanded_query": _hyde_fallback(query),
            "method": "hyde_fallback_local",
            "error": "LLM key unavailable; using local bigram expansion.",
        }
    raise ValueError(f"Unknown expansion mode: {mode!r}")


if __name__ == "__main__":
    for q in ["IELTS xét tuyển", "điều kiện xét tuyển đại học"]:
        print(q)
        for mode in ("baseline", "expansion", "hyde"):
            print(" ", mode, "->", expand_query(q, mode=mode))
