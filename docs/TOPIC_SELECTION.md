# Topic Selection — K4-L3B RAG Pipeline

## 1. Selection Process

We evaluated five candidate topics against the published selection criteria before
committing to a single domain. All candidates were checked for data availability,
domain authority, and reproducibility before scoring.

## 2. Candidates Considered

| # | Candidate | Rationale | Concerns |
| - | --------- | --------- | -------- |
| 1 | Vietnamese Higher Education Admissions | Multiple official sources (MOET, individual universities); high lexical diversity (codes, dates, thresholds); strong cross-document reasoning needs | Vietnamese text requires a multilingual embedding model |
| 2 | Vietnamese Tourism / Destinations | Many travel guides and news pages | Sources change quickly; weaker on policy/guideline documents |
| 3 | Personal Income Tax (Thuế TNCN) | Stable regulations from the General Department of Taxation | Very narrow lexical vocabulary, less interesting retrieval |
| 4 | IELTS Writing Band Descriptors | Stable, English source | Limited public news pages, mostly one-document topic |
| 5 | Vietnamese Public Healthcare Services | Many news pages | Highly variable policies across regions |

## 3. Scoring (out of 100)

| Criterion                                  | Weight | Higher Ed Admissions | Tourism | Tax | IELTS | Healthcare |
| ------------------------------------------ | -----: | -------------------: | ------: | --: | ----: | ---------: |
| Quality and authority of sources           |   25%  |                 22   |    16   | 20  |   19  |        15  |
| ≥ 3 policy/guideline documents             |   15%  |                 14   |    10   | 13  |   12  |        12  |
| ≥ 5 public pages/articles                  |   15%  |                 15   |    14   | 10  |    5  |        13  |
| Data stability                             |   15%  |                 12   |     9   | 15  |   15  |        10  |
| Retrieval / evaluation potential          |   10%  |                  9   |     6   |  7  |    6  |         8  |
| Demo value                                 |   10%  |                  9   |     8   |  7  |    5  |         7  |
| Bonus / extension potential                |   10%  |                  9   |     6   |  6  |    7  |         7  |
| **Total**                                  |        |              **90** |   **69** | 78 |   **69** |     **72** |

## 4. Selected Topic

**Vietnamese Higher Education Admissions (Xét tuyển Đại học / Cao đẳng Việt Nam)**

### Why this topic scored highest

1. **Authoritative sources** — MOET (`moet.gov.vn`) is the issuing body for the
   admissions regulations. Major universities (HUST, VNU-USSH, HCMUS) publish
   their own admission plans every year. The corpus mixes government policy
   documents with reporting from reputable outlets (`tuoitre.vn`, `thanhnien.vn`,
   `vietnamnet.vn`, `dantri.com.vn`, `tuyensinh247.com`).
2. **Required document counts are comfortably exceeded.** Plans exist to collect
   ≥ 5 policy PDFs / government announcements and ≥ 8 news articles.
3. **Cross-document reasoning**: a typical user question may require combining a
   MOET regulation with a university's specific implementation plan.
4. **Lexical vs semantic tension**: official codes (e.g. "Phương thức 5",
   "Điều kiện xét tuyển", "ĐXT") are perfect for BM25, while paraphrases
   ("em muốn hỏi điều kiện thi đại học như thế nào") need dense retrieval.
5. **Stable URLs** — MOET pages and university portals rarely disappear.
6. **Out-of-domain probe questions** are easy to construct (e.g. "Thủ tục đăng ký
   xe ô tô").

## 5. Expected Retrieval Challenges

* Vietnamese tokenisation; BGE-m3 handles this well.
* Long policy PDFs need sensible chunking to keep semantic coherence.
* Mixed language (some policy phrases are formally Vietnamese; news uses
  conversational Vietnamese).
* Date ambiguity (admission year vs publication year).

## 6. Expected Evaluation Strategy

* Four Ragas metrics: **Faithfulness**, **Answer Relevancy**, **Context Precision**,
  **Context Recall**.
* A/B comparison: dense-only vs hybrid (dense + BM25 + RRF + fallback).
* Categories of golden questions: factual, paraphrase, lexical, multi-document,
  comparison, date/number, source-specific, ambiguous, refusal.

## 7. Confirmed Real Sources

* MOET — `https://moet.gov.vn`
* HUST — `https://hust.edu.vn/tuyen-sinh`
* VNU-USSH — `https://ussh.vnu.edu.vn`
* HCMUS — `https://hcmus.edu.vn`
* Thanh Niên — `https://thanhnien.vn/giao-duc`
* VietnamNet — `https://vietnamnet.vn/giao-duc`
* Dân trí — `https://dantri.com.vn/giao-duc.htm`
* Tuyensinh247 — `https://tuyensinh247.com`

All sources verified with HTTP 200 and substantial page content during the
connectivity smoke test (see `docs/ENVIRONMENT_SETUP.md`).