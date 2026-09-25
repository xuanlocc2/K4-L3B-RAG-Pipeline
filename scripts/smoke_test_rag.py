"""Smoke test cho RAG pipeline thật.

Mục đích: chứng minh pipeline end-to-end hoạt động với indexed corpus
hiện tại, không phải mock. In ra:
    1. Indexing report thật (số chunks, dim, sources...).
    2. Top-K dense + BM25 cho một câu hỏi tiếng Việt.
    3. Top-K hybrid (RRF) kết hợp.
    4. Sample context (chunks thật + metadata).
    5. Generation nếu có API key, ngược lại báo "generation skipped".

Usage:
    python scripts/smoke_test_rag.py
    python scripts/smoke_test_rag.py --query "IELTS có dùng xét tuyển được không?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from src.task4_chunking_indexing import print_index_report, verify_index
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search
from src.task7_reranking import rerank_rrf


DEFAULT_QUERY = "IELTS có được sử dụng để xét tuyển đại học không?"


def _print_results(label: str, results: list[dict], limit: int = 5) -> None:
    print(f"\n--- {label} ({len(results)} results) ---")
    for idx, item in enumerate(results[:limit], 1):
        meta = item.get("metadata") or {}
        print(
            f"[{idx}] score={float(item.get('score', 0.0)):.4f} "
            f"method={item.get('retrieval_method', '')} "
            f"id={item.get('id', '')}"
        )
        title = meta.get("title") or "(no title)"
        source = meta.get("source") or "(no source)"
        print(f"     title : {title}")
        print(f"     source: {source}")
        if meta.get("url"):
            print(f"     url   : {meta['url']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test RAG pipeline.")
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Câu hỏi tiếng Việt để truy vấn.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 1: Indexing report (real values from ChromaDB)")
    print("=" * 60)
    print_index_report(verify_index())

    print("\n" + "=" * 60)
    f"STEP 2: Query='{args.query}' (top_k={args.top_k})"
    print("=" * 60)

    dense = semantic_search(args.query, top_k=args.top_k)
    bm25 = lexical_search(args.query, top_k=args.top_k)
    fused = rerank_rrf([dense, bm25], top_k=args.top_k)

    _print_results("DENSE (BGE-m3 cosine)", dense)
    _print_results("BM25 (lexical)", bm25)
    _print_results("HYBRID (RRF, k=60)", fused)

    print("\n" + "=" * 60)
    print("STEP 3: Sample chunks (first 2 of hybrid)")
    print("=" * 60)
    for idx, item in enumerate(fused[:2], 1):
        meta = item.get("metadata") or {}
        content = item.get("content", "")
        print(f"\n[{idx}] {item.get('id', '')} ({meta.get('title', '')})")
        snippet = content[:400].replace("\n", " ")
        if len(content) > 400:
            snippet += "..."
        print(f"     text: {snippet}")

    # Step 4: attempt generation
    print("\n" + "=" * 60)
    print("STEP 4: Generation attempt")
    print("=" * 60)
    provider = os.getenv("LLM_PROVIDER", "openai")
    key_env = {
        "openai": "OPENAI_API_KEY",
        "groq": "GROQ_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }.get(provider, "")
    has_key = bool(os.getenv(key_env, "")) if key_env else False

    if not has_key:
        print(
            f"SKIPPED: LLM_PROVIDER={provider} nhưng {key_env} rỗng — không gọi LLM. "
            "Đây là kết quả trung thực."
        )
    else:
        try:
            from src.task10_generation import generate_with_citation

            result = generate_with_citation(args.query, top_k=args.top_k)
            print(f"Answer (first 600 chars):\n{result['answer'][:600]}")
            print(f"\nretrieval_source: {result['retrieval_source']}")
            citation_check = result.get("citation_check") or {}
            print(
                f"citation_check: valid={citation_check.get('valid')} "
                f"grounded_ratio={citation_check.get('grounded_ratio')} "
                f"total_refs={citation_check.get('total_refs')}"
            )
        except Exception as exc:
            print(f"Generation error (trung thực, không giả lập): {exc}")


if __name__ == "__main__":
    main()
