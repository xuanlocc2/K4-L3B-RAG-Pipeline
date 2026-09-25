# Individual contribution report — Bùi Hải Nam

---

## Thông tin

- Họ và tên: **Bùi Hải Nam**
- Mã học viên: 2A202602636
- Nhóm: LaoGaKho
- Repository/branch: `main`

---

## Phần việc đã thực hiện

| Module/deliverable | Việc tôi trực tiếp làm | File/commit/PR | Trạng thái |
|---|---|---|---|
| LLM generation dispatch | Implement `generate_with_citation()` — OpenAI/Groq/Gemini/Anthropic dispatch theo env, context reorder (front+reverse-back) để giảm lost-in-the-middle | `src/task10_generation.py` | Done |
| Citation grounding (Bonus 7) | Implement `check_citation_grounding()` — validate [n] format, đo grounded_ratio, detect missing/invalid refs | `src/bonus7_citation_grounding.py` | Done |
| Source highlighting (Bonus 6) | Implement `_highlight_citations()` trong generation — trả về câu trả lời với `[n]` được highlight | `src/task10_generation.py` | Done |
| Conversation memory (Bonus 5) | Implement `rewrite_query_with_history()` — gộp conversation history vào query expansion, multi-turn rewrite | `src/bonus5_conversation_memory.py` | Done |
| Streamlit UI | Xây dựng `app.py` — chat interface, suggested questions, session state, expander cho citations | `app.py` | Done |
| UI components | Implement `render_citation_panel()`, `render_insufficient_evidence_panel()`, `render_state_banner()`, `render_rejected_candidates_panel()`, `render_debug_panel()` — tách logic render khỏi app.py | `src/ui_components.py` | Done |
| Sidebar controls | Chế độ retrieval (simple/hybrid/weighted), top_k slider, score threshold, debug toggle, analytics toggle | `app.py` sidebar | Done |
| Safe refusal messages | Vietnamese refusal messages cho từng answerability state (out_of_domain, evidence_insufficient, evidence_weak, clarify, refuse) | `src/task10_generation.py` | Done |
| Bonus 5 benchmark | Đo conversation memory trên multi-turn queries: latent shift + topic narrowing — không cải thiện trên corpus này | `scripts/bonus5_benchmark.py`, `group_project/evaluation/bonus5_benchmark.json` | Done |

---

## Quyết định kỹ thuật quan trọng

1. **Quyết định:** Context reorder: đặt chunk đầu và cuối, đảo ngược middle chunk để giảm lost-in-the-middle.
   **Lý do/evidence:** Liu et al. (2023) "Lost in the Middle" chỉ ra LLM tập trung vào vị trí đầu/cuối. Đặt 2 chunk đầu + đảo ngược 3 chunk giữa → chunk 4,3,2 ở giữa nhưng LLM không bỏ qua.
   **Trade-off:** Reorder tăng overhead và có thể phá vỡ context flow. Đã test với 5 queries — không thấy regression.

2. **Quyết định:** Citation format `[n]` thay vì Markdown footnotes hoặc superscript.
   **Lý do/evidence:** `[n]` là format LLM tự nhiên generate khi được prompt đúng, ít hallucination hơn superscript. Streamlit markdown render `[1][2]` tốt. Grounding check dùng regex `\[(\d+)\]` đơn giản và chính xác.
   **Trade-off:** Người dùng có thể quen với superscript; UI có thể upgrade lên clickable footnote sau.

---

## Kiểm thử và kết quả

- **Test hoặc query tôi đã dùng:** 5 demo queries (factual, paraphrase, semantic+lexical, multi-document, OOD); 6 Groq test cases (`tests/test_groq_provider.py`); real-pipeline smoke test (`scripts/smoke_test_rag.py`)
- **Kết quả trước/sau nếu có:**
  - Groq provider: `qwen/qwen3.8-27b` trả về câu trả lời có citation, `citation_check: valid=True, grounded_ratio=1.0, total_refs=5` cho query "IELTS có được dùng để xét tuyển đại học không?"
  - Conversation memory: rewrite "vậy còn gì khác?" → "ngoài IELTS, còn phương thức nào khác để xét tuyển đại học?" — test pass trên 3 multi-turn pairs
- **Lỗi đã phát hiện và cách xử lý:**
  - Lỗi: `ImportError: cannot import name 'render_rejected_candidates_panel'` — function đã tồn tại trong `src/ui_components.py`, import path đúng. Xác nhận bằng `python -c "from src.ui_components import render_rejected_candidates_panel"`.
  - Lỗi: Streamlit widget-state "Error(s) Saving Not Allowed" — benign warning do widget ID không persistent. Không ảnh hưởng pipeline.

---

## Điều còn hạn chế

- **Một hạn chế cụ thể của phần tôi làm:** Citation grounding không kiểm tra semantic correctness của citation (chỉ kiểm tra format và source tồn tại). LLM có thể cite đúng source nhưng trích dẫn sai nội dung.
- **Nếu có thêm thời gian, thay đổi đầu tiên tôi sẽ thực hiện:** Thêm claim-level citation verification — dùng LLM để check mỗi câu có citation [n] thực sự supported bởi source nội dung, không chỉ format.

---

## Xác nhận đóng góp

Tôi xác nhận nội dung trên phản ánh đúng phần việc của mình và có thể giải thích hoặc chạy lại trong buổi demo.

- Ngày: 26/09/2026
- Tên thành viên: Bùi Hải Nam
