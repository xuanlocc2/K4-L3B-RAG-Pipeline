"""
Task 9 — Retrieval pipeline hoàn chỉnh.

Luồng xử lý:
    1. Chạy semantic_search và lexical_search.
    2. Fuse hai danh sách bằng RRF đúng một lần.
    3. Lấy best cosine score gốc từ dense results.
    4. Nếu score dưới threshold, thử PageIndex fallback.
    5. Nếu fallback lỗi, trả hybrid results thay vì crash.

Không so sánh threshold với RRF score vì hai thang đo khác nhau.
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5


def calibrate_threshold(
    in_domain_queries: list[str], out_of_domain_queries: list[str]
) -> dict:
    """Choose a cosine gate by maximizing balanced accuracy on labeled queries.

    In-domain queries should retrieve from this legal corpus; out-of-domain
    queries should not. The caller records the returned threshold and score
    distributions in the evaluation report rather than guessing a value.
    """
    if not in_domain_queries or not out_of_domain_queries:
        raise ValueError("Provide at least one in-domain and one out-of-domain query")

    def score(query: str) -> float:
        results = semantic_search(query, top_k=1)
        return float(results[0]["score"]) if results else 0.0

    in_scores = [score(query) for query in in_domain_queries]
    out_scores = [score(query) for query in out_of_domain_queries]
    candidates = sorted({0.0, 1.0, *in_scores, *out_scores, *[(a + b) / 2 for a in in_scores for b in out_scores]})
    evaluations = []
    for threshold in candidates:
        true_positive_rate = sum(value >= threshold for value in in_scores) / len(in_scores)
        true_negative_rate = sum(value < threshold for value in out_scores) / len(out_scores)
        evaluations.append((true_positive_rate + true_negative_rate) / 2)
    best_index = max(range(len(candidates)), key=lambda index: (evaluations[index], candidates[index]))
    return {
        "threshold": candidates[best_index],
        "balanced_accuracy": evaluations[best_index],
        "in_domain_scores": in_scores,
        "out_of_domain_scores": out_scores,
    }


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Retrieve with a demo-friendly A/B switch.

    ``use_reranking=False`` is the semantic-only baseline. ``True`` is the
    hybrid candidate (dense + BM25 + one RRF pass). Both use the same query,
    corpus, top_k and dense fallback gate, making their evaluation comparable.
    """
    if top_k <= 0 or not query.strip():
        return []

    candidate_k = top_k * 2
    dense = semantic_search(query, top_k=candidate_k)
    if use_reranking:
        sparse = lexical_search(query, top_k=candidate_k)
        result = rerank_rrf([dense, sparse], top_k=top_k)
    else:
        result = dense[:top_k]

    # Only raw dense cosine controls fallback; RRF scores have another scale.
    best_dense_score = float(dense[0]["score"]) if dense else 0.0
    if best_dense_score < score_threshold:
        try:
            fallback = pageindex_search(query, top_k=top_k)
            if fallback:
                return fallback
        except Exception:
            pass
    return result[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
