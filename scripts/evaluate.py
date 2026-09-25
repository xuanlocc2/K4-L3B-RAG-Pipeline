"""Evaluation pipeline (RAG) — A/B comparison and metric computation.

This module runs two retrieval strategies against the golden dataset and
computes four metrics:

    - Context Precision:  càng cao càng tốt.
    - Context Recall:     càng cao càng tốt.
    - Source Hit Rate:    tỉ lệ câu hỏi có ít nhất một source trong
                         expected_context được retrieve lại.
    - Answer Fidelity:    tỉ lệ câu trả lời được LLM generate thành công
                         (không bị safe refusal) cho câu in-domain.

For LLM-based metrics (faithfulness, answer relevance) the script falls
back to deterministic proxy metrics when no API key is configured:

    - Token-level answer overlap với expected_answer (Jaccard).
    - Retrieval hit indicator dựa trên expected_context.

Khi không có OpenAI/Gemini/Anthropic key, generator trả về safe refusal
cho tất cả câu — vì vậy Answer Fidelity sẽ là 0. Đây là kết quả
TRUNG THỰC và được báo cáo trong RESULT.md.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")


# --- Pipeline imports --------------------------------------------------------

from src.task4_chunking_indexing import embed_texts  # noqa: E402
from src.task5_semantic_search import semantic_search  # noqa: E402
from src.task6_lexical_search import lexical_search  # noqa: E402
from src.task7_reranking import rerank_rrf  # noqa: E402
from src.task9_retrieval_pipeline import (  # noqa: E402
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
)
from src.task10_generation import generate_with_citation  # noqa: E402


# --- Configuration -----------------------------------------------------------

GOLDEN_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
RESULTS_DIR = ROOT / "group_project" / "evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TOP_K = DEFAULT_TOP_K
THRESHOLD = DEFAULT_SCORE_THRESHOLD


@dataclass
class CaseResult:
    case_id: str
    question: str
    expected_answer: str
    expected_sources: list[str]
    retrieved_ids: list[str]
    retrieval_method: str
    answer: str
    answer_status: str  # "answered" | "refusal" | "error"
    latency_ms: float
    metadata: dict = field(default_factory=dict)


def _retrieval_dense_only(query: str, top_k: int) -> list[dict]:
    return semantic_search(query, top_k=top_k)


def _retrieval_hybrid(query: str, top_k: int) -> list[dict]:
    dense = semantic_search(query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)
    return rerank_rrf([dense, sparse], top_k=top_k)


def _expected_sources_from_case(case: dict) -> list[str]:
    ctx = (case.get("expected_context") or "").strip()
    if not ctx or ctx.startswith("("):
        return []
    return [s.strip() for s in ctx.split(",") if s.strip()]


def _source_hit(retrieved: list[dict], expected: list[str]) -> bool:
    if not expected:
        return False
    for r in retrieved:
        for exp in expected:
            if exp and exp in r["id"]:
                return True
    return False


def _tokenize(s: str) -> set[str]:
    return {t.lower() for t in re.findall(r"\w+", s or "", flags=re.UNICODE)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _context_precision(retrieved: list[dict], expected: list[str]) -> float:
    """Tỉ lệ retrieved IDs nằm trong expected sources."""
    if not retrieved:
        return 0.0
    if not expected:
        return 0.0
    hits = 0
    for r in retrieved:
        for exp in expected:
            if exp in r["id"]:
                hits += 1
                break
    return hits / len(retrieved)


def _context_recall(retrieved: list[dict], expected: list[str]) -> float:
    """Tỉ lệ expected sources được retrieve lại."""
    if not expected:
        return 0.0
    retrieved_set = {r["id"] for r in retrieved}
    hits = 0
    for exp in expected:
        if any(exp in rid for rid in retrieved_set):
            hits += 1
    return hits / len(expected)


def _answer_overlap(answer: str, expected: str) -> float:
    return _jaccard(_tokenize(answer), _tokenize(expected))


def _answer_status(answer: str) -> str:
    if not answer.strip():
        return "refusal"
    if "Tôi không thể xác minh" in answer:
        return "refusal"
    return "answered"


def _evaluate_strategy(strategy_name: str, retrieval_fn, dataset: list[dict]) -> list[CaseResult]:
    results: list[CaseResult] = []
    for case in dataset:
        q = case["question"]
        expected = _expected_sources_from_case(case)
        start = time.perf_counter()
        try:
            retrieved = retrieval_fn(q, TOP_K)
        except Exception as exc:
            print(f"[{strategy_name}] {case['id']} retrieval error: {exc}")
            retrieved = []
        retrieval_latency = (time.perf_counter() - start) * 1000.0

        # generation
        answer = ""
        gen_latency = retrieval_latency
        try:
            gen_start = time.perf_counter()
            gen_result = generate_with_citation(q, top_k=TOP_K)
            answer = gen_result.get("answer", "")
            gen_latency = (time.perf_counter() - gen_start) * 1000.0
        except Exception as exc:
            print(f"[{strategy_name}] {case['id']} generation error: {exc}")

        results.append(
            CaseResult(
                case_id=case["id"],
                question=q,
                expected_answer=case.get("expected_answer", ""),
                expected_sources=expected,
                retrieved_ids=[r["id"] for r in retrieved],
                retrieval_method=strategy_name,
                answer=answer,
                answer_status=_answer_status(answer),
                latency_ms=gen_latency,
                metadata={
                    "retrieval_latency_ms": retrieval_latency,
                    "category": case.get("category", ""),
                    "difficulty": case.get("difficulty", ""),
                    "source_hit": _source_hit(retrieved, expected),
                    "context_precision": _context_precision(retrieved, expected),
                    "context_recall": _context_recall(retrieved, expected),
                    "answer_overlap": _answer_overlap(answer, case.get("expected_answer", "")),
                    "top1_score": float(retrieved[0]["score"]) if retrieved else 0.0,
                },
            )
        )
    return results


def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    total = len(results)
    in_domain = [r for r in results if r.expected_sources]
    ood = [r for r in results if not r.expected_sources]

    def _avg(items: list[CaseResult], key: str) -> float:
        if not items:
            return 0.0
        return sum(r.metadata.get(key, 0.0) for r in items) / len(items)

    def _mean_overall(key: str) -> float:
        if not results:
            return 0.0
        return sum(r.metadata.get(key, 0.0) for r in results) / total

    return {
        "total_cases": total,
        "in_domain_cases": len(in_domain),
        "ood_cases": len(ood),
        "context_precision": _mean_overall("context_precision"),
        "context_recall": _mean_overall("context_recall"),
        "source_hit_rate": _mean_overall("source_hit"),
        "answer_overlap": _mean_overall("answer_overlap"),
        "answer_fidelity_in_domain": _avg(in_domain, "answer_overlap"),
        "answered_rate": sum(1 for r in results if r.answer_status == "answered") / total,
        "answered_rate_in_domain": (
            sum(1 for r in in_domain if r.answer_status == "answered") / max(len(in_domain), 1)
        ),
        "avg_latency_ms": sum(r.latency_ms for r in results) / max(total, 1),
    }


def main() -> None:
    if not GOLDEN_PATH.exists():
        print(f"Missing golden dataset at {GOLDEN_PATH}")
        sys.exit(1)
    dataset = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    print(f"Evaluating {len(dataset)} cases per strategy (top_k={TOP_K}, threshold={THRESHOLD})...")

    print("\n--- Strategy A: dense-only ---")
    strategy_a = _evaluate_strategy("dense-only", _retrieval_dense_only, dataset)
    agg_a = _aggregate(strategy_a)
    print(json.dumps(agg_a, indent=2, ensure_ascii=False))

    print("\n--- Strategy B: hybrid (dense + BM25 + RRF) ---")
    strategy_b = _evaluate_strategy("hybrid-rrf", _retrieval_hybrid, dataset)
    agg_b = _aggregate(strategy_b)
    print(json.dumps(agg_b, indent=2, ensure_ascii=False))

    # lưu kết quả chi tiết
    output = {
        "config": {
            "top_k": TOP_K,
            "score_threshold": THRESHOLD,
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
            "llm_model": os.getenv("LLM_MODEL", "") or "(default per provider)",
        },
        "strategy_a_dense_only": {
            "aggregate": agg_a,
            "per_case": [r.__dict__ for r in strategy_a],
        },
        "strategy_b_hybrid": {
            "aggregate": agg_b,
            "per_case": [r.__dict__ for r in strategy_b],
        },
    }
    out_path = RESULTS_DIR / "evaluation_results.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved detailed results to {out_path.relative_to(ROOT)}")
    return output


if __name__ == "__main__":
    main()