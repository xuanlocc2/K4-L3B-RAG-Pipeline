"""
Task 6 — Lexical search bằng BM25.

Dùng cùng corpus chunks với Task 5. BM25 phù hợp với từ khóa chính xác, mã tài
liệu và tên riêng. Output phải theo SearchResult và sort score giảm dần.
"""


CORPUS: list[dict] = []


def _tokenize(text: str) -> list[str]:
    # tách theo ký tự chữ/số để giữ nguyên tiếng Việt có dấu.
    return [token.lower() for token in _TOKEN_RE.findall(text or "")]


def _load_corpus_from_cache() -> list[dict]:
    if not _CORPUS_PATH.exists():
        return []
    try:
        payload = json.loads(_CORPUS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    return list(payload.get("chunks", []))


def set_corpus(corpus: list[dict]) -> None:
    """Cho phép test fixture gán corpus tuỳ ý."""
    global CORPUS, _BM25
    CORPUS = corpus
    _BM25 = None  # invalidate


def _ensure_corpus() -> None:
    global CORPUS, _BM25
    # Nếu CORPUS rỗng thì load từ cache.
    if not CORPUS:
        CORPUS = _load_corpus_from_cache()
        # Sau khi reload corpus, phải invalidate _BM25 để nó được build
        # lại với đúng corpus hiện tại. Tránh leak state từ test fixture.
        if CORPUS:
            _BM25 = None
    # Nếu _BM25 chưa được build (hoặc vừa bị invalidate), build lại.
    if _BM25 is None and CORPUS:
        tokenized = [_tokenize(item["content"]) for item in CORPUS]
        _BM25 = BM25Plus(tokenized)


def build_bm25_index(corpus: list[dict]) -> BM25Plus:
    """Tạo BM25 index từ cùng corpus chunks của Task 4."""
    # TODO: Tokenize và tạo BM25 index.
    #
    # from rank_bm25 import BM25Okapi
    # tokenized = [item["content"].lower().split() for item in corpus]
    # return BM25Okapi(tokenized)
    raise NotImplementedError("Implement build_bm25_index")


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Trả về BM25 SearchResult theo score giảm dần."""
    if not query.strip():
        return []
    _ensure_corpus()
    if _BM25 is None or not CORPUS:
        return []

    tokens = _tokenize(query)
    if not tokens:
        return []

    scores = _BM25.get_scores(tokens)
    order = np.argsort(scores)[::-1]

    results: list[dict] = []
    seen: set[str] = set()
    for index in order:
        score = float(scores[index])
        # BM25Plus có thể trả score rất nhỏ (gần 0) cho tài liệu không khớp.
        # Bỏ qua các tài liệu có score <= 0 để đảm bảo chất lượng.
        if score <= 0:
            continue
        item = CORPUS[index]
        if item["id"] in seen:
            continue
        seen.add(item["id"])
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": score,
                "bm25_score": score,
                "metadata": dict(item["metadata"]),
                "retrieval_method": "bm25",
            }
        )
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
