# Project Audit — K4-L3B RAG Pipeline

> Audit snapshot taken before any modifications to the skeleton repository.

## 1. Repository State

The repository is a **day-8 lab skeleton** distributed to students. It declares the
RAG architecture in `README.md`, the contract schemas in `docs/MODULE_CONTRACTS.md`,
and a 10-task implementation plan in `docs/STEP_BY_STEP.md`. The grader's rubric
is in `docs/GRADING_RUBRIC.md`. However, every Task 1–10 module is a **stub**
that raises `NotImplementedError`. The data pipeline, indexes, evaluation dataset,
and Streamlit chat are all empty.

| Aspect               | Status                                             |
| -------------------- | -------------------------------------------------- |
| Working tree         | Clean (`git status`)                               |
| Branch               | `main`, up-to-date with `origin/main`              |
| Python interpreter   | 3.11.9 (satisfies `>=3.10,<3.14`)                  |
| Virtualenv           | Created (`.venv`)                                  |
| Dependencies         | Installed (`pip install -e ".[dev]"`)              |
| Playwright Chromium  | Installed and verified to launch                   |
| Tests                | 13 fail, 7 pass (mostly signature checks)          |

## 2. Module Audit (per `src/`)

| Module | File | State | Notes |
| ------ | ---- | ----- | ----- |
| Contracts | `src/contracts.py` | WORKING | TypedDict schema + validators used by tests |
| Task 1 — Legal docs | `src/task1_collect_legal_docs.py` | MISSING | `NotImplementedError`; `data/landing/legal/` is empty |
| Task 2 — News crawl | `src/task2_crawl_news.py` | MISSING | `NotImplementedError`; `data/landing/news/` is empty; URL list empty |
| Task 3 — Markdown conversion | `src/task3_convert_markdown.py` | MISSING | `NotImplementedError`; `data/standardized/` is empty |
| Task 4 — Chunk + embed + index | `src/task4_chunking_indexing.py` | MISSING | All functions raise `NotImplementedError` |
| Task 5 — Semantic search | `src/task5_semantic_search.py` | MISSING | `NotImplementedError` |
| Task 6 — BM25 search | `src/task6_lexical_search.py` | MISSING | `NotImplementedError` |
| Task 7 — RRF | `src/task7_reranking.py` | MISSING | `NotImplementedError` |
| Task 8 — PageIndex fallback | `src/task8_pageindex_vectorless.py` | MISSING | `NotImplementedError`; no key configured |
| Task 9 — Retrieval pipeline | `src/task9_retrieval_pipeline.py` | MISSING | `NotImplementedError` |
| Task 10 — Generation + citation | `src/task10_generation.py` | MISSING | `NotImplementedError` |
| Streamlit app | `app.py` | PARTIAL | Imports `dotenv`; chat input wires up but does not call the pipeline; only echoes `TODO` |
| Evaluation | `group_project/evaluation/golden_dataset.json` | MISSING | Empty file (0 bytes) |
| Evaluation report | `group_project/evaluation/RESULT.md` | NEEDS IMPROVEMENT | Template with `TODO` placeholders |

## 3. Baseline Test Run

```
$ pytest -q
13 failed, 7 passed
```

Failures are concentrated in `test_contracts.py` (8) and `test_acceptance.py` (5).
The 7 passing tests are mostly the contract validator unit tests and the public
signature stability checks.

## 4. Risk and Gap Analysis

1. **No data** — the project has no real documents, which blocks acceptance tests.
2. **No indexes** — ChromaDB collection has not been created.
3. **No retrieval** — semantic / BM25 / RRF are not implemented.
4. **No generation** — no LLM dispatcher, no context formatting, no citation mapping.
5. **No evaluation** — golden dataset is empty; `RESULT.md` is a TODO template.
6. **No UI** — `app.py` is a placeholder; it never invokes the pipeline.

## 5. Constraints Respected

* `src.contracts` is treated as the immutable schema source for all modules.
* `SearchResult` and `GenerationResult` schemas from `MODULE_CONTRACTS.md` must be honored.
* `metadata.url` may be `None` for legal PDFs but is required for news JSON.
* Tests under `tests/test_contracts.py` must pass without making any real
  network or external API call. This invariant is enforced by stubbing
  `embed_texts`, `get_collection`, and provider calls in the test code.

## 6. Architectural Decisions Locked In

* **Embedding model**: `BAAI/bge-m3` (declared in `task4_chunking_indexing.py`,
  supports multilingual including Vietnamese, dimension 1024).
* **Vector store**: ChromaDB persistent, cosine distance.
* **Chunk size**: 500 / overlap 50 (matches the placeholder, evaluated below).
* **Hybrid fusion**: RRF only, applied exactly once.
* **Fallback decision metric**: original dense cosine score (NOT RRF score).
* **Reordering**: even-indexed first + reversed odd-indexed (lost-in-the-middle mitigation).
* **PageIndex**: optional external fallback — implemented as a safe stub that
  returns `[]` when the API key is absent so the pipeline never crashes.

## 7. Required Improvements (to be executed)

1. Pick a topic (see `docs/TOPIC_SELECTION.md`).
2. Collect ≥ 3 policy documents and ≥ 5 public news JSONs.
3. Implement Tasks 1–10.
4. Calibrate the dense-score threshold.
5. Build the Streamlit chat UI.
6. Build a ≥ 15-question golden dataset grounded in real sources.
7. Run Ragas evaluation with 4 metrics and A/B comparison.
8. Document everything in `docs/` and update the README.