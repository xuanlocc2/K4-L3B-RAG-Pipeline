"""
Bonus 7 — Citation grounding validation.

Kiểm tra câu trả lời có dùng citation hợp lệ hay không.

Quy tắc phát hiện:
    * Citation có dạng ``[n]`` với n là số nguyên dương.
    * Citation phải nằm trong khoảng ``[1, len(sources)]``.
    * Mỗi ``[n]`` trong answer phải map về đúng ``sources[n-1]``.

Output:
    ``validate_citations(answer, sources)`` trả về:
        - valid: bool
        - invalid_refs: list[int]  (các index citation không hợp lệ)
        - missing_citation: bool   (answer có khẳng định mà không cite)
        - grounded_ratio: float   (tỉ lệ [n] hợp lệ trên tổng số [n])

    ``rewrite_with_safe_citations(answer, sources)`` thay thế citation không
    hợp lệ bằng marker ``[unverified]`` và đính kèm cảnh báo, KHÔNG bịa nguồn.
"""

from __future__ import annotations

import re
from typing import Any

# Pattern khớp [n] với n là 1+ chữ số.
_CITATION_RE = re.compile(r"\[(\d+)\]")

# Các cụm từ thường đi kèm khẳng định có thật. Dùng heuristic nhẹ để phát
# hiện "missing citation" — không thay thế LLM judge, chỉ là safety net.
_CLAIM_CUES = (
    "là", "gồm", "theo", "được", "phải", "có thể", "không thể",
    "bao gồm", "áp dụng", "qui định", "quy định", "thuộc",
)


def extract_citation_indices(answer: str) -> list[int]:
    """Trích tất cả ``[n]`` trong answer, trả về list[int] các chỉ số."""
    if not answer:
        return []
    return [int(m.group(1)) for m in _CITATION_RE.finditer(answer)]


def validate_citations(answer: str, sources: list[Any]) -> dict[str, Any]:
    """Kiểm tra answer có citation hợp lệ hay không.

    Trả về dict với các key:
        - valid: bool
        - invalid_refs: list[int] (các index trỏ ra ngoài sources)
        - missing_citation: bool (answer có câu khẳng định nhưng không cite)
        - grounded_ratio: float (tỉ lệ citation hợp lệ / tổng citation)
        - total_refs: int
    """
    refs = extract_citation_indices(answer)
    n_sources = len(sources)

    invalid_refs = [n for n in refs if n < 1 or n > n_sources]
    valid_refs = [n for n in refs if 1 <= n <= n_sources]
    grounded_ratio = (len(valid_refs) / len(refs)) if refs else 1.0

    # Heuristic missing citation: nếu answer có dấu chấm câu và chứa claim
    # cue mà không cite gì cả thì coi là missing.
    missing = False
    if not refs:
        lowered = (answer or "").lower()
        if any(cue in lowered for cue in _CLAIM_CUES) and len(answer.strip()) > 20:
            missing = True

    return {
        "valid": not invalid_refs and not missing,
        "invalid_refs": invalid_refs,
        "missing_citation": missing,
        "grounded_ratio": grounded_ratio,
        "total_refs": len(refs),
    }


def rewrite_with_safe_citations(answer: str, sources: list[Any]) -> dict[str, Any]:
    """Thay thế citation không hợp lệ bằng marker ``[unverified]``.

    Không bịa nguồn. Nếu toàn bộ answer thiếu citation, đính kèm cảnh báo
    ở cuối.
    """
    if not answer:
        return {"answer": answer or "", "modified": False, "invalid_refs": []}

    n_sources = len(sources)
    invalid_refs: list[int] = []
    seen_invalid: set[int] = set()

    def _replace(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if 1 <= n <= n_sources:
            return match.group(0)
        invalid_refs.append(n)
        if n not in seen_invalid:
            seen_invalid.add(n)
        return "[unverified]"

    new_answer = _CITATION_RE.sub(_replace, answer)
    modified = new_answer != answer

    # Nếu thiếu citation hoàn toàn nhưng có claim, thêm cảnh báo.
    if not _CITATION_RE.search(new_answer) and any(
        cue in (answer or "").lower() for cue in _CLAIM_CUES
    ):
        new_answer = (
            new_answer.rstrip()
            + "\n\n_(Lưu ý: câu trả lời này không chứa citation hợp lệ; "
            "vui lòng kiểm tra nguồn trước khi sử dụng.)_"
        )
        modified = True

    return {
        "answer": new_answer,
        "modified": modified,
        "invalid_refs": sorted(set(invalid_refs)),
    }


if __name__ == "__main__":
    demo_sources = [{"id": "a"}, {"id": "b"}]
    print(validate_citations("Theo quy chế [1] thì [3] cũng đúng.", demo_sources))
    print(rewrite_with_safe_citations("Theo [1] và [5].", demo_sources))
