from typing import Literal, TypedDict


RetrievalMethod = Literal["dense", "bm25", "hybrid", "pageindex"]
RetrievalSource = Literal["hybrid", "pageindex", "none"]


class DocumentMetadata(TypedDict):
    source: str
    title: str
    doc_type: str
    url: str | None


class ChunkMetadata(DocumentMetadata):
    chunk_index: int


class Document(TypedDict):
    id: str
    content: str
    metadata: DocumentMetadata


class Chunk(TypedDict):
    id: str
    content: str
    metadata: ChunkMetadata


class EmbeddedChunk(Chunk):
    embedding: list[float]


class SearchResult(TypedDict, total=False):
    """Public search-result schema.

    Required (existing contract — backward-compatible):
        id, content, score, metadata, retrieval_method.

    Optional (added by evidence-quality work — populated when available):
        dense_score, bm25_score, rrf_score.

    The required ``score`` is whatever number was used for ranking this result
    in its final list — for a hybrid result this is the RRF score, for a
    dense-only result it is the cosine similarity, etc. It is a ranking signal,
    NOT a confidence percentage. Callers that need to talk about confidence
    must consult the evidence-quality gate (see ``src/evidence_quality.py``).
    """

    id: str
    content: str
    score: float
    metadata: ChunkMetadata
    retrieval_method: RetrievalMethod
    dense_score: float
    bm25_score: float
    rrf_score: float


class GenerationResult(TypedDict, total=False):
    """Public generation result schema.

    Required (existing contract):
        answer, sources, retrieval_source.

    Optional (added by evidence-quality work):
        evidence, intent, normalized_query, diagnostics, citation_check.
    """

    answer: str
    sources: list[SearchResult]
    retrieval_source: RetrievalSource
    evidence: dict
    intent: dict
    normalized_query: str | None
    diagnostics: dict
    citation_check: dict


def validate_document(item: object, *, require_chunk: bool = False) -> None:
    """Raise ``ValueError`` when an item violates the document contract."""
    if not isinstance(item, dict):
        raise ValueError("document must be a dict")
    if not isinstance(item.get("id"), str) or not item["id"].strip():
        raise ValueError("document.id must be a non-empty string")
    if not isinstance(item.get("content"), str) or not item["content"].strip():
        raise ValueError("document.content must be a non-empty string")

    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("document.metadata must be a dict")
    for key in ("source", "title", "doc_type"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise ValueError(f"metadata.{key} must be a non-empty string")
    if "url" not in metadata or not (
        metadata["url"] is None or isinstance(metadata["url"], str)
    ):
        raise ValueError("metadata.url must be a string or None")
    if require_chunk and (
        not isinstance(metadata.get("chunk_index"), int)
        or metadata["chunk_index"] < 0
    ):
        raise ValueError("metadata.chunk_index must be a non-negative integer")


def validate_search_results(
    results: object,
    *,
    top_k: int | None = None,
    expected_method: RetrievalMethod | None = None,
) -> None:
    """Validate schema, ordering, uniqueness and result count."""
    if not isinstance(results, list):
        raise ValueError("search results must be a list")
    if top_k is not None and len(results) > max(top_k, 0):
        raise ValueError("search results exceed top_k")

    ids: list[str] = []
    scores: list[float] = []
    valid_methods = {"dense", "bm25", "hybrid", "pageindex"}
    for item in results:
        validate_document(item, require_chunk=True)
        score = item.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("result.score must be numeric")
        method = item.get("retrieval_method")
        if method not in valid_methods:
            raise ValueError("result.retrieval_method is invalid")
        if expected_method is not None and method != expected_method:
            raise ValueError(f"expected retrieval_method={expected_method}")
        ids.append(item["id"])
        scores.append(float(score))

    if len(ids) != len(set(ids)):
        raise ValueError("search result IDs must be unique")
    if scores != sorted(scores, reverse=True):
        raise ValueError("search results must be sorted by score descending")


def _is_optional_score(value: object) -> bool:
    return value is None or (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float("-inf") < float(value) < float("inf")
    )


def assert_extended_score_fields(results: object) -> None:
    """Soft-check for the optional ``dense_score``/``bm25_score``/``rrf_score``.

    This is intentionally permissive: it only validates type when the field is
    present. It never raises when the field is absent (the legacy contract
    doesn't require it), so it is safe to call from any existing test path.
    """
    if not isinstance(results, list):
        return
    for item in results:
        if not isinstance(item, dict):
            continue
        for key in ("dense_score", "bm25_score", "rrf_score"):
            if key in item and not _is_optional_score(item[key]):
                raise ValueError(
                    f"result.{key} must be a finite number or None, "
                    f"got {item[key]!r}"
                )


def validate_generation_result(result: object) -> None:
    """Validate the public output of ``generate_with_citation``."""
    if not isinstance(result, dict):
        raise ValueError("generation result must be a dict")
    if not isinstance(result.get("answer"), str) or not result["answer"].strip():
        raise ValueError("generation answer must be a non-empty string")
    validate_search_results(result.get("sources"))
    if result.get("retrieval_source") not in {"hybrid", "pageindex", "none"}:
        raise ValueError("generation retrieval_source is invalid")
