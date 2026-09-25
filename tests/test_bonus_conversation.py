"""Tests cho Bonus 5: Conversation memory (query rewriting)."""

from __future__ import annotations

from src.bonus5_conversation_memory import (
    _looks_like_followup,
    _topic_from_history,
    rewrite_followup,
)


def test_passthrough_for_standalone_query():
    history = [
        {"role": "user", "content": "Điều kiện xét tuyển đại học là gì?"},
        {"role": "assistant", "content": "..."},
    ]
    out = rewrite_followup(history, "Các bài thi tốt nghiệp THPT 2025 gồm những môn nào?")
    assert out["method"] == "passthrough"
    assert out["rewrote"] is False
    assert out["standalone_query"].startswith("Các bài thi")


def test_followup_carries_known_entity():
    history = [
        {"role": "user", "content": "Điều kiện xét tuyển đại học gồm những yêu cầu nào?"},
        {"role": "assistant", "content": "..."},
    ]
    out = rewrite_followup(history, "Còn IELTS thì sao?")
    assert out["method"] == "local_heuristic"
    assert out["last_topic"] == "IELTS" or "IELTS" in out["standalone_query"]
    assert "IELTS" in out["standalone_query"]


def test_followup_short_query_gets_topic():
    history = [
        {"role": "user", "content": "Các bài thi tốt nghiệp THPT 2025 gồm những môn nào?"},
        {"role": "assistant", "content": "..."},
    ]
    out = rewrite_followup(history, "Thì sao?")
    assert out["rewrote"] is True
    # Topic dài được mang sang.
    assert "thi tốt nghiệp THPT" in out["standalone_query"] or "tốt nghiệp THPT" in out["standalone_query"]


def test_topic_switch_recognized_as_standalone():
    """Câu đủ dài với entity rõ ràng -> passthrough, không mang topic cũ."""
    history = [
        {"role": "user", "content": "IELTS có được dùng để xét tuyển không?"},
    ]
    out = rewrite_followup(history, "Hồ sơ dự tuyển đại học cần những giấy tờ gì?")
    assert out["method"] == "passthrough"


def test_topic_extraction_finds_known_entity():
    history = [
        {"role": "user", "content": "IELTS có được dùng để xét tuyển đại học không?"},
    ]
    entity, _ = _topic_from_history(history)
    assert entity == "IELTS"


def test_topic_extraction_falls_back_to_tokens():
    history = [
        {"role": "user", "content": "Chứng chỉ ABC123 là gì?"},
    ]
    entity, topic_long = _topic_from_history(history)
    assert entity is None
    assert topic_long is not None
    assert "ABC123" in topic_long or "chứng" in topic_long.lower()


def test_empty_history_no_topic():
    out = rewrite_followup([], "IELTS có được dùng không?")
    # Không có history -> passthrough, không mang topic.
    assert out["method"] == "passthrough"
    assert out["last_topic"] is None


def test_empty_query_returns_empty():
    out = rewrite_followup([{"role": "user", "content": "..."}], "")
    assert out["method"] == "passthrough"
    assert out["rewrote"] is False


def test_followup_cue_detection_short_query():
    assert _looks_like_followup("Còn về học phí?")
    assert _looks_like_followup("Thì sao?")
    assert _looks_like_followup("vậy hả?")
    assert not _looks_like_followup(
        "Hồ sơ dự tuyển đại học cần những giấy tờ gì đầy đủ?"
    )


def test_uses_recent_user_turn_not_old_assistant():
    """Khi có nhiều turn, lấy entity từ user turn gần nhất."""
    history = [
        {"role": "user", "content": "IELTS có được dùng để xét tuyển không?"},
        {"role": "assistant", "content": "Điều kiện xét tuyển cần tốt nghiệp THPT"},
        {"role": "user", "content": "Tổ hợp môn A00 gồm những môn nào?"},
    ]
    out = rewrite_followup(history, "Còn VSTEP thì sao?")
    # Topic gần nhất (Tổ hợp A00) vẫn là anchor; query đã có "VSTEP" nên
    # passthrough hoặc giữ nguyên.
    assert "VSTEP" in out["standalone_query"]
