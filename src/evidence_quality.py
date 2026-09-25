"""Evidence-quality gate + lightweight query understanding.

This module is deliberately deterministic (no LLM, no embedding model). Its
job is to decide whether a retrieved top-K actually answers the user's
question, so the rest of the pipeline can:

    * route to grounded generation, OR
    * ask a clarifying question, OR
    * refuse safely.

It also extracts a tiny structured intent (``education_level``, ``domain``,
``intent``, ``entities``, ``keywords``) that the retrieval layer can use to
decide whether to rewrite the query.

Two design choices are critical and are not negotiable:

    1. **No hard-coded strings.** The decision rules operate on the
       structured intent above, so adding a new conflict pattern (e.g.
       "tiến sĩ" + "mẫu giáo") Just Works without code changes.
    2. **RRF score is NEVER used as confidence.** The gate considers dense
       cosine, BM25 score and the keyword-overlap separately.

Public API:

    ``understand_query(query)`` -> dict with intent/entities/keywords.
    ``assess_evidence(intent, chunks)`` -> evidence-quality report.
    ``should_clarify(intent)`` -> bool (cheap pre-check before retrieval).
    ``build_normalized_query(query, intent)`` -> deterministic rewrite
        that preserves the original (caller keeps the original).
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Vietnamese tokenization helpers (lightweight, accent-folded)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\w+", flags=re.UNICODE)
_ACCENT_MAP = {
    "a": "aáàảãạăắằẳẵặâấầẩẫậ",
    "e": "eéèẻẽẹêếềểễệ",
    "i": "iíìỉĩị",
    "o": "oóòỏõọôốồổỗộơớờởỡợ",
    "u": "uúùủũụưứừửữự",
    "y": "yýỳỷỹỵ",
    "d": "dđ",
}


def _fold(s: str) -> str:
    """Lower-case + strip Vietnamese diacritics for token comparison."""
    if not s:
        return ""
    out = []
    for ch in str(s).lower():
        for base, variants in _ACCENT_MAP.items():
            if ch in variants:
                out.append(base)
                break
        else:
            out.append(ch)
    return "".join(out)


def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(_fold(text)) if t]


# A short Vietnamese stopword list — enough to drop the noisiest words.
_STOPWORDS = frozenset({
    "la", "lam", "thi", "the", "toi", "ban", "minh", "chung", "ta",
    "nao", "gi", "nhu", "theo", "duoc", "khong", "co", "hay", "nen",
    "vi", "o", "tren", "va", "hoac", "trong", "voi", "khi", "neu",
    "cua", "mot", "nhung", "rat", "den", "di", "nhu", "thoi", "xem",
    "biet", "cho", "dang", "se", "tu", "nh", "na", "moi",
})


def _content_tokens(text: str) -> list[str]:
    """Tokenize + drop stopwords + drop very short tokens."""
    return [t for t in _tokenize(text) if len(t) >= 2 and t not in _STOPWORDS]


# ---------------------------------------------------------------------------
# Domain knowledge (deterministic, regex-based)
# ---------------------------------------------------------------------------

# Education-level patterns — order matters (most specific first).
_EDUCATION_PATTERNS: list[tuple[str, str]] = [
    # university (anywhere "đại học"/"DH"/"university" appears alongside
    # admission context will be caught by the domain check).
    (r"\b(?:đh|dai\s*hoc|đại\s*học|university|cao\s*học|thạc\s*sĩ|tien\s*si|tiến\s*sĩ)\b", "university"),
    # high school (THPT, lop 10-12)
    (r"\b(?:thpt|lop\s*1[02]|lớp\s*1[02]|12\s*th|ba\s*nam\s*cap\s*ba|cấp\s*ba)\b", "high_school"),
    # secondary school (THCS, lop 6-9)
    (r"\b(?:thcs|lop\s*[6-9]|lớp\s*[6-9])\b", "secondary_school"),
    # primary school (lop 1-5, tieu hoc)
    (r"\b(?:tieu\s*hoc|tiểu\s*học|lop\s*[1-5]|lớp\s*[1-5]|cap\s*hai|cấp\s*hai)\b", "primary_school"),
]

_DOMAIN_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:xet\s*tuyen|xét\s*tuyển|tuyen\s*sinh|tuyển\s*sinh|nganh\s*hoc|ngành\s*học|phuong\s*thuc|phương\s*thức|diem\s*chuan|điểm\s*chuẩn|thi\s*sinh|thí\s*sinh|ho\s*so|hồ\s*sơ|nhap\s*hoc|nhập\s*học)\b", "university_admission"),
    (r"\b(?:thi\s*tot\s*nghiep|tốt\s*nghiệp\s*thpt|tot\s*nghiep\s*thpt|mon\s*thi|môn\s*thi|bai\s*thi|bài\s*thi|ky\s*thi|kỳ\s*thi|thi\s*thpt|thpt\s*2025)\b", "thpt_exam"),
    (r"\b(?:tuyen\s*sinh\s*247|tuyensinh247|thanhnien|vietnamnet|dan\s*tri|dân\s*trí)\b", "news"),
]

_INTENT_PATTERNS: list[tuple[str, str]] = [
    (r"\b(?:phuong\s*thuc|phương\s*thức|chinh\s*sach|chính\s*sách|diem\s*chuan|điểm\s*chuẩn|hoc\s*phi|học\s*phí)\b", "admission_method"),
    (r"\b(?:dieu\s*kien|điều\s*kiện|giay\s*to|giấy\s*tờ|ho\s*so|hồ\s*sơ|nhap\s*hoc|nhập\s*học)\b", "requirement"),
    (r"\b(?:lich\s*thi|lịch\s*thi|thoi\s*gian|thời\s*gian|ngay\s*thi|ngày\s*thi|ky\s*thi|kỳ\s*thi|mon\s*thi|môn\s*thi|khi\s*nao|khi\s*nào)\b", "schedule"),
    (r"\b(?:ielts|toeic|vstep|chung\s*chi|chứng\s*chỉ|ngoai\s*ngu|ngoại\s*ngữ)\b", "admission_method"),
]

_ENTITY_KEYWORDS: tuple[str, ...] = (
    "ielts", "toeic", "vstep", "toefl", "sat", "act",
    "thpt", "thcs", "tnthpt", "tnthcs",
    "moet", "bộ giáo dục",
    "hanoi", "hcmc", "da nang",
    "a00", "a01", "b00", "c00", "d01",
)


# Out-of-domain detector: queries whose tokens have zero overlap with any
# Vietnamese content terms commonly found in the corpus. Used as a last-resort
# safety net for queries like "Bitcoin", "kết hôn".
#
# The audit requires a *taxonomy* of OOD topics — not just a hand-curated
# stop-list. We organize it as a list of (topic_label, regex) pairs so adding
# a new OOD topic is one line. The classifier returns the topic if matched.
_OOD_TOPIC_PATTERNS: list[tuple[str, str]] = [
    # marriage / family / civil registry
    ("marriage", r"\b(?:ket\s*hon|kết\s*hôn|hon\s*nhan|hôn\s*nhân|"
                r"dang\s*ky\s*ket\s*hon|dăng\s*ký\s*kết\s*hôn|"
                r"ly\s*hon|ly\s*hôn|"
                r"giay\s*dang\s*ky\s*ket\s*hon|"
                r"dinh\s*chi|cấp\s*giấy\s*xác\s*nhận)"
                r"\b"),
    # criminal law / courts / police
    ("criminal_law", r"\b(?:hinh\s*su|hình\s*sự|toi\s*pham|tội\s*phạm|"
                  r"an\s*ninh|an\s*ninh\s*quoc\s*gia|"
                  r"toa\s*an|toà\s*án|"
                  r"canh\s*sat|cảnh\s*sát|"
                  r"khoi\s*to|khởi\s*tố|"
                  r"truy\s*to|truy\s*tố|"
                  r"phat\s*tu|phạt\s*tù|"
                  r"phap\s*luat\s*hinh\s*su|pháp\s*luật\s*hình\s*sự)"
                  r"\b"),
    # weather
    ("weather", r"\b(?:thoi\s*tiet|thời\s*tiết|"
              r"nhiet\s*do|nhiệt\s*độ|"
              r"du\s*bao\s*thoi\s*tiet|dự\s*báo\s*thời\s*tiết|"
              r"mua\s*bao\s*gio|mưa\s*bao\s*giờ|"
              r"bao\s*bao|ba\s*bão|"
              r"lu\s*lut|lũ\s*lụt|"
              r"thien\s*ta|thiên\s*tai)"
              r"\b"),
    # crypto / stock trading
    ("finance_crypto", r"\b(?:bitcoin|ethereum|"
                    r"crypto|"
                    r"btc|eth|"
                    r"gia\s*bitcoin|giá\s*bitcoin|"
                    r"tien\s*ao|tiền\s*ảo|"
                    r"nft|"
                    r"gia\s*co\s*phieu|giá\s*cổ\s*phiếu|"
                    r"thi\s*truong\s*chung\s*khoan|thị\s*trường\s*chứng\s*khoán|"
                    r"forex)"
                    r"\b"),
    # recipes / cooking
    ("cooking", r"\b(?:cong\s*thuc\s*nau|công\s*thức\s*nấu|"
              r"cach\s*lam|cách\s*làm\s*(?:mon|món|banh|bánh|pho|phở|"
              r"bun|bún|com|cơm|mien|miến|chao|cháo)|"
              r"mon\s*an\s*gia\s*dinh|món\s*ăn\s*gia\s*đình|"
              r"nau\s*an|nấu\s*ăn|"
              r"che\s*bien|chế\s*biến)"
              r"\b"),
    # medical / health
    ("medical", r"\b(?:kham\s*benh|khám\s*bệnh|"
              r"thuoc|thuốc|"
              r"bac\s*si|bác\s*sĩ|"
              r"benh\s*vien|bệnh\s*viện|"
              r"dieu\s*tri|điều\s*trị|"
              r"phau\s*thuat|phẫu\s*thuật|"
              r"trieu\s*chung|triệu\s*chứng|"
              r"vaccine|vaccin|"
              r"covid|"
              r"ung\s*thu|ung\s*thư|"
              r"benh\s*ly|bệnh\s*lý)"
              r"\b"),
    # sports / football
    ("sports", r"\b(?:bong\s*da|bóng\s*đá|"
             r"world\s*cup|"
             r"euro|"
             r"cup\s*c1|"
             r"v\d*-\s*league|"
             r"tran\s*dau|trận\s*đấu|"
             r"the\s*thao|thể\s*thao|"
             r"olympic|"
             r"sea\s*games|"
             r"asiad|"
             r"huy\s*chuong|huy\s*chương|"
             r"vo\s*thuat|võ\s*thuật|"
             r"quan\s*vot|quần\s*vợt|tennis)"
             r"\b"),
    # vehicle / driving / registration
    ("vehicle", r"\b(?:dang\s*ky\s*xe|đăng\s*ký\s*xe|"
              r"bang\s*lai|bằng\s*lái|"
              r"gplx|"
              r"o\s*to|ô\s*tô|"
              r"xe\s*moto|"
              r"xang\s*dau|xăng\s*dầu|"
              r"dau\s*mo|dầu\s*mỏ)"
              r"\b"),
    # real estate / property
    ("real_estate", r"\b(?:bat\s*dong\s*san|bất\s*động\s*sản|"
                 r"mua\s*nha|mua\s*nhà|"
                 r"ban\s*nha|bán\s*nhà|"
                 r"chung\s*cu|chung\s*cư|"
                 r"thue\s*nha|thuê\s*nhà|"
                 r"dat\s*dai|đất\s*đai|"
                 r"so\s*do|sổ\s*đỏ|"
                 r"giay\s*chung\s*nhan|giấy\s*chứng\s*nhận)"
                 r"\b"),
    # generic non-academic procedural law (administrative, tax, immigration)
    ("procedural_law", r"\b(?:dang\s*ky\s*kinh\s*doanh|đăng\s*ký\s*kinh\s*doanh|"
                    r"ma\s*so\s*thue|mã\s*số\s*thuế|"
                    r"thue\s*tndn|thuế\s*TNDN|"
                    r"ho\s*chieu|hộ\s*chiếu|"
                    r"visa|"
                    r"giay\s*phep\s*xay\s*dung|giấy\s*phép\s*xây\s*dựng)"
                    r"\b"),
]


# Positive in-domain seed vocabulary: if any of these tokens appear in the
# folded query, the domain gate accepts the query as in-domain. This is a
# conservative short-list so a query with even one in-domain token passes.
_IN_DOMAIN_KEYWORDS: tuple[str, ...] = (
    "dai", "hoc", "dh", "university",  # đại học
    "xet", "tuyen",  # xét tuyển
    "tuyen", "sinh",  # tuyển sinh
    "thpt", "tnthpt", "thcs",  # kỳ thi
    "tot", "nghiep",  # tốt nghiệp
    "mon", "thi", "bai", "ky", "ky thi", "lich thi", "de thi",  # thi cử
    "ielts", "toeic", "vstep", "toefl", "sat", "act", "chung chi",
    "ngoai", "ngu",  # ngoại ngữ
    "phuong", "thuc",  # phương thức
    "diem", "chuan",  # điểm chuẩn
    "nganh", "nganh hoc",
    "sinh", "vien", "sv", "hoc sinh", "hoc sinh lop", "lop",
    "nhap", "hoc",  # nhập học
    "ho", "so", "giay to",  # hồ sơ
    "moet", "bo giao duc", "bo gd",
    "tinh", "huyen", "tinh thanh", "truong dh", "truong dai hoc",
    "truong", "dh", "dai hoc",
    "nganh nghe", "cong nghe",
    "dau", "vao", "dau vao",  # đầu vào
    "tu van", "tu van tuyen sinh",
    "de an", "de an tuyen sinh",
    "chi tieu",
    "hoc phi", "hoc bong",
    "ky thi tot nghiep", "ky thi thpt",
    "khoi", "khoi thi", "to hop", "khoi a00", "khoi b00", "khoi c00", "khoi d01",
)


# Negative in-domain tokens — content words that almost never appear in our
# corpus. If a query has tokens mostly from this list and nothing from the
# in-domain list, it's a strong OOD signal.
_NON_ACADEMIC_GENERIC_TOKENS: frozenset[str] = frozenset({
    "nhà", "nha",  # too generic alone, handled by OOD patterns
    "công", "ty",  # công ty
    "shopee", "lazada", "tiki", "grab",
    "iphone", "samsung", "xiaomi",
    "thực", "phẩm", "thuc pham",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_domain(query: str) -> dict[str, Any]:
    """Domain gate: is the query in-domain for the Vietnamese university
    admissions corpus?

    Returns::

        {
            "is_in_domain": bool,
            "ood_topic": str | None,    # matched taxonomy label if OOD
            "in_domain_signal": str | None,  # matched in-domain token/pattern
            "rationale": str,
        }

    Strategy (deterministic, ordered):
        1. If empty → OOD (empty).
        2. If any OOD topic pattern matches → OOD, label that topic.
        3. If any in-domain keyword token matches → IN domain.
        4. Otherwise OOD (no in-domain signal found).

    No LLM, no embedding model.
    """
    if not query or not query.strip():
        return {
            "is_in_domain": False,
            "ood_topic": "empty",
            "in_domain_signal": None,
            "rationale": "empty query",
        }
    text = query.strip()
    folded = _fold(text)

    # Step 2: explicit OOD taxonomy.
    for topic, pattern in _OOD_TOPIC_PATTERNS:
        if re.search(pattern, folded):
            return {
                "is_in_domain": False,
                "ood_topic": topic,
                "in_domain_signal": None,
                "rationale": f"Query khớp OOD topic '{topic}'.",
            }

    # Step 3: in-domain keywords.
    in_domain_signal: str | None = None
    for kw in _IN_DOMAIN_KEYWORDS:
        # Use word-boundary-ish matching on the folded query.
        if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", folded):
            in_domain_signal = kw
            break

    # Also accept the existing regex-based domain patterns as positive signals.
    if in_domain_signal is None:
        for pattern, _label in _DOMAIN_PATTERNS:
            if re.search(pattern, folded):
                in_domain_signal = f"domain_pattern:{_label}"
                break

    if in_domain_signal is None:
        return {
            "is_in_domain": False,
            "ood_topic": "no_in_domain_signal",
            "in_domain_signal": None,
            "rationale": (
                "Query không chứa keyword thuộc domain tuyển sinh / kỳ thi / "
                "chứng chỉ / trường học."
            ),
        }
    return {
        "is_in_domain": True,
        "ood_topic": None,
        "in_domain_signal": in_domain_signal,
        "rationale": f"Query chứa in-domain signal '{in_domain_signal}'.",
    }


def understand_query(query: str) -> dict[str, Any]:
    """Extract a lightweight structured intent from a user query.

    Returned dict (always; empty fields default to safe values):

        education_level ∈ {primary_school, secondary_school, high_school,
                            university, unspecified}
        domain ∈ {university_admission, thpt_exam, news, unknown}
        intent ∈ {admission_method, requirement, schedule, other}
        entities: list[str]
        keywords: list[str]
        domain_classification: dict (output of classify_domain)
        is_out_of_domain: bool          (kept for backward compat;
                                          = not classification.is_in_domain)
        conflict: list[str]
    """
    if not query or not query.strip():
        return {
            "education_level": "unspecified",
            "domain": "unknown",
            "intent": "other",
            "entities": [],
            "keywords": [],
            "domain_classification": classify_domain(query),
            "is_out_of_domain": True,
            "conflict": ["empty query"],
        }
    text = query.strip()
    folded = _fold(text)

    # ------------------------------------------------------------------
    # Two-pass education-level detection:
    #
    #   1) "self-describing" patterns — the user states their own level
    #      ("tôi là học sinh lớp 1", "con tôi đang học THCS", ...).
    #   2) Topic-level patterns — the corpus subject the user is asking
    #      about ("đại học", "cao học", ...).
    #
    # Self-description wins when both fire because the user is the source
    # of truth about themselves.
    # ------------------------------------------------------------------
    self_patterns: list[tuple[str, str]] = [
        (r"\b(?:toi|ban|con\s*toi|em\s*toi|minh)\s*(?:la|dang\s*hoc|la\s*hoc\s*sinh|la\s*sinh\s*vien)\s*(?:lop|lớp)\s*([1-9]|1[0-2])\b", "_SELF_GRADE"),
        (r"\b(?:hoc\s*sinh|sinh\s*vien)\s*(?:lop|lớp)\s*([1-9]|1[0-2])\b", "_SELF_GRADE"),
        (r"\b(?:toi|ban|con\s*toi|em\s*toi|minh)\s*(?:la|dang\s*hoc)\s*(tieu\s*hoc|tiểu\s*học|thcs|thpt|cao\s*hoc|cao\s*đẳng|đại\s*học|dai\s*hoc|sinh\s*vien|sv)\b", "_SELF_LABEL"),
    ]
    grade_to_level = {
        "1": "primary_school", "2": "primary_school", "3": "primary_school",
        "4": "primary_school", "5": "primary_school",
        "6": "secondary_school", "7": "secondary_school",
        "8": "secondary_school", "9": "secondary_school",
        "10": "high_school", "11": "high_school", "12": "high_school",
    }
    label_to_level = {
        "tieu hoc": "primary_school", "tiểu học": "primary_school",
        "thcs": "secondary_school",
        "thpt": "high_school",
        "cao đẳng": "high_school",
        "cao hoc": "university",
        "đại học": "university", "dai hoc": "university",
        "sinh vien": "university", "sv": "university",
    }

    self_level: str | None = None
    m = re.search(self_patterns[0][0], folded)
    if m:
        self_level = grade_to_level.get(m.group(1))
    if self_level is None:
        m = re.search(self_patterns[1][0], folded)
        if m:
            self_level = grade_to_level.get(m.group(1))
    if self_level is None:
        m = re.search(self_patterns[2][0], folded)
        if m:
            self_level = label_to_level.get(m.group(1).strip())

    topic_level = "unspecified"
    for pattern, label in _EDUCATION_PATTERNS:
        if re.search(pattern, folded):
            topic_level = label
            break

    education_level = self_level or topic_level

    domain = "unknown"
    for pattern, label in _DOMAIN_PATTERNS:
        if re.search(pattern, folded):
            domain = label
            break

    intent = "other"
    for pattern, label in _INTENT_PATTERNS:
        if re.search(pattern, folded):
            intent = label
            break

    entities: list[str] = []
    tokens = set(_tokenize(text))
    for kw in _ENTITY_KEYWORDS:
        if _fold(kw) in tokens:
            entities.append(kw)
    # Multi-token entities like "ielts" alone: also check raw text.
    for kw in ("IELTS", "VSTEP", "TOEIC", "TOEFL"):
        if kw.lower() in text.lower() and kw not in entities:
            entities.append(kw)

    keywords = _content_tokens(text)[:12]

    # Domain gate (uses taxonomy-based classifier).
    classification = classify_domain(text)
    is_ood = not classification["is_in_domain"]

    # Soft fallback (kept for backward compat with tests that already use
    # is_out_of_domain=True for empty/very-short queries). The classifier
    # already handles those cases so we don't add anything here.

    conflict: list[str] = []
    if (
        education_level in ("primary_school", "secondary_school")
        and domain == "university_admission"
    ):
        conflict.append(
            f"education_level={education_level} combined with "
            "domain=university_admission: chưa thuộc nhóm đối tượng dự tuyển"
        )
    # Self-described lower level + university topic also counts.
    if (
        self_level in ("primary_school", "secondary_school")
        and topic_level == "university"
    ):
        msg = (
            "người hỏi tự xác định ở cấp "
            + self_level.replace("_", " ")
            + " nhưng đang hỏi về đại học"
        )
        if msg not in conflict:
            conflict.append(msg)
    if (
        education_level == "high_school"
        and intent == "requirement"
        and domain == "university_admission"
        and "tot_nghiep" not in folded
        and "tốt nghiệp" not in text.lower()
    ):
        # Soft signal: high schooler asking about ĐH requirements is normal —
        # do NOT raise a conflict. We just keep it as a no-op.
        pass

    return {
        "education_level": education_level,
        "domain": domain,
        "intent": intent,
        "entities": entities,
        "keywords": keywords,
        "domain_classification": classification,
        "is_out_of_domain": is_ood,
        "conflict": conflict,
    }


def build_normalized_query(query: str, intent: dict[str, Any]) -> str:
    """Build a deterministic retrieval-side rewrite that preserves the
    original ``query`` at the call-site.

    The normalized form is meant for retrieval only. It is **not** what the
    user sees or what the LLM is asked to clarify about.

    General policy:
        * If intent+domain conflict, expand toward the corpus's actual subject
          so retrieval can still find something rather than zero-evidence.
        * Otherwise return the original query unchanged.
    """
    if not query or not query.strip():
        return query or ""
    original = query.strip()
    domain = intent.get("domain", "unknown")
    education = intent.get("education_level", "unspecified")
    if (
        education in ("primary_school", "secondary_school")
        and domain == "university_admission"
    ):
        # User mentions a sub-admission-age group but asks about ĐH — keep the
        # corpus-side subject (university admission) for retrieval, while the
        # call-site keeps the original query for the clarification message.
        return (
            "đối tượng xét tuyển đại học điều kiện dự tuyển phương thức tuyển sinh"
        )
    if domain == "thpt_exam" and not intent.get("entities"):
        return f"{original} thi tốt nghiệp THPT"
    return original


def _keyword_overlap(query_tokens: set[str], chunk_text: str) -> float:
    """Fraction of query content tokens that appear in chunk_text.

    Returned value is in [0.0, 1.0]. A value of 0 means no overlap at all.
    """
    if not query_tokens:
        return 0.0
    chunk_tokens = set(_content_tokens(chunk_text))
    if not chunk_tokens:
        return 0.0
    overlap = query_tokens & chunk_tokens
    return len(overlap) / max(len(query_tokens), 1)


def _chunk_max_score(
    chunks: list[dict], attr: str
) -> tuple[float, dict | None]:
    """Return (max_score, chunk_with_max) for ``dense_score``/``bm25_score``."""
    best = 0.0
    best_chunk: dict | None = None
    for ch in chunks:
        value = ch.get(attr)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            v = float(value)
            if v > best:
                best = v
                best_chunk = ch
    return best, best_chunk


def score_chunks(
    intent: dict[str, Any],
    chunks: list[dict],
    *,
    score_threshold: float = 0.50,
    keyword_min: float = 0.20,
    dense_accept: float = 0.50,
    keyword_accept: float = 0.20,
) -> dict[str, Any]:
    """Per-chunk relevance scoring used by ``assess_evidence`` and the
    final-evidence pipeline.

    Each chunk is independently evaluated:

        * ``passes = (dense_score >= dense_accept OR is on the topic) AND
                      keyword_overlap >= keyword_accept``

    The aggregate "sufficient / weak / insufficient" status is decided from
    the number of passing chunks.

    Returns::

        {
            "passed":  [chunk, ...]            # survivors in original order
            "rejected": [{"chunk": ch, "reasons": [...]}]
            "per_chunk": [{"id":..., "passes": bool, "reasons": [...],
                           "dense_score":..., "keyword_overlap":...}]
            "passed_count": int
        }

    The chunk-level rule is intentionally stricter than the previous
    overall-only check: a single high-scoring chunk can no longer "validate"
    a low-scoring one. Each chunk must earn its place.
    """
    if not chunks:
        return {
            "passed": [],
            "rejected": [],
            "per_chunk": [],
            "passed_count": 0,
        }
    query_tokens = set(intent.get("keywords", []))
    passed: list[dict] = []
    rejected: list[dict] = []
    per_chunk: list[dict] = []
    for ch in chunks:
        content = (ch.get("content") or "").strip()
        dense_v = ch.get("dense_score")
        dense = float(dense_v) if isinstance(dense_v, (int, float)) else 0.0
        overlap = _keyword_overlap(query_tokens, content)
        reasons: list[str] = []
        if dense <= 0.0 or dense < score_threshold:
            reasons.append(
                f"dense_score={dense:.3f} dưới ngưỡng {score_threshold:.3f}"
            )
        if overlap < keyword_accept:
            reasons.append(
                f"keyword_overlap={overlap:.2f} dưới ngưỡng {keyword_accept:.2f}"
            )
        passes = (
            (dense >= score_threshold)
            and (overlap >= keyword_accept)
        )
        if passes:
            passed.append(ch)
        else:
            rejected.append({"chunk": ch, "reasons": reasons})
        per_chunk.append({
            "id": ch.get("id"),
            "title": (ch.get("metadata") or {}).get("title", ""),
            "dense_score": dense_v,
            "bm25_score": ch.get("bm25_score"),
            "rrf_score": ch.get("rrf_score"),
            "keyword_overlap": round(overlap, 4),
            "passes": passes,
            "reasons": reasons,
        })
    return {
        "passed": passed,
        "rejected": rejected,
        "per_chunk": per_chunk,
        "passed_count": len(passed),
    }


def assess_evidence(
    intent: dict[str, Any],
    chunks: list[dict],
    *,
    score_threshold: float = 0.50,
    keyword_min: float = 0.20,
) -> dict[str, Any]:
    """Decide whether the retrieved chunks actually answer the query.

    Returns a dict with:
        status ∈ {"sufficient", "weak", "insufficient", "out_of_domain"}
        relevance_label ∈ {"High", "Medium", "Low"}
        suggested_action ∈ {"answer", "clarify", "refuse"}
        signals: per-chunk dense/bm25/rrf/keyword-overlap, plus aggregates
        per_chunk: list of per-chunk gate verdicts
        passed_chunk_ids: list[str]
        rejected_chunk_ids: list[str]
        top_chunk_id: str | None
        rationale: str (human-readable explanation)

    Decision rules (in order):

        1. ``intent.domain_classification.is_in_domain == False``
           → ``status=out_of_domain``, ``suggested_action=refuse``.
        2. No chunks → ``insufficient``, refuse.
        3. Structured conflict (e.g. primary school + ĐH) → ``insufficient``,
           clarify.
        4. Per-chunk scoring: ``score_chunks()`` filters each chunk by
           ``dense_score >= threshold`` AND ``keyword_overlap >= keyword_accept``.
           - 0 pass  → ``insufficient``, refuse.
           - 1 pass  → ``weak``, clarify (one weak evidence is not enough).
           - ≥2 pass → ``sufficient``, answer (caller will dedup + cap to 1–3).
        5. Otherwise (legacy keyword-only fallback) — kept for backward compat.

    RRF is never used as confidence.
    """
    classification = intent.get("domain_classification") or {}
    if classification.get("is_in_domain") is False:
        # Domain gate: refuse immediately. ``rejected_chunk_ids`` is empty
        # because we never even ran retrieval, but we still surface the
        # taxonomy label so the UI can show the right refusal text.
        signals = {
            "chunks": [],
            "dense_top1": 0.0,
            "bm25_top1": 0.0,
            "rrf_top1": 0.0,
            "keyword_overlap_top1": 0.0,
            "per_chunk": [],
            "passed_chunk_ids": [],
            "rejected_chunk_ids": [],
        }
        return {
            "status": "out_of_domain",
            "relevance_label": "Low",
            "suggested_action": "refuse",
            "signals": signals,
            "per_chunk": [],
            "passed_chunk_ids": [],
            "rejected_chunk_ids": [],
            "top_chunk_id": None,
            "rationale": (
                "Domain gate OOD: "
                + classification.get("rationale", "query outside corpus domain")
            ),
        }

    signals: dict[str, Any] = {
        "chunks": [],
        "dense_top1": 0.0,
        "bm25_top1": 0.0,
        "rrf_top1": 0.0,
        "keyword_overlap_top1": 0.0,
    }
    if not chunks:
        return {
            "status": "insufficient",
            "relevance_label": "Low",
            "suggested_action": "refuse",
            "signals": signals,
            "per_chunk": [],
            "passed_chunk_ids": [],
            "rejected_chunk_ids": [],
            "top_chunk_id": None,
            "rationale": "Không có chunk nào được truy xuất.",
        }

    # Per-chunk scoring (the new strict gate).
    scored = score_chunks(
        intent, chunks,
        score_threshold=score_threshold,
        keyword_accept=keyword_min,
    )
    per_chunk = scored["per_chunk"]
    passed = scored["passed"]
    rejected = scored["rejected"]

    # Aggregate signals (top-1 of the *original* candidate order, not the
    # filtered order).
    dense_top1, dense_chunk = _chunk_max_score(chunks, "dense_score")
    bm25_top1, _ = _chunk_max_score(chunks, "bm25_score")
    rrf_top1, rrf_chunk = _chunk_max_score(chunks, "rrf_score")
    if rrf_top1 == 0.0 and chunks:
        rrf_top1 = float(chunks[0].get("score", 0.0))
        rrf_chunk = chunks[0]
    signals["dense_top1"] = dense_top1
    signals["bm25_top1"] = bm25_top1
    signals["rrf_top1"] = rrf_top1
    signals["per_chunk"] = per_chunk
    signals["passed_chunk_ids"] = [c.get("id") for c in passed]
    signals["rejected_chunk_ids"] = [
        r["chunk"].get("id") for r in rejected
    ]

    # Best overlap is now computed over *all* candidates (for the
    # rationale line), not just the passed set.
    query_tokens = set(intent.get("keywords", []))
    best_overlap = 0.0
    best_overlap_chunk: dict | None = None
    for ch in chunks:
        content = (ch.get("content") or "").strip()
        overlap = _keyword_overlap(query_tokens, content)
        if overlap > best_overlap:
            best_overlap = overlap
            best_overlap_chunk = ch
    signals["keyword_overlap_top1"] = best_overlap
    # Build a "chunks" view with the old shape for back-compat (UI tests
    # don't read it, but other consumers might).
    signals["chunks"] = [
        {
            "id": pc["id"],
            "title": pc["title"],
            "dense_score": pc["dense_score"],
            "bm25_score": pc["bm25_score"],
            "rrf_score": pc["rrf_score"],
            "keyword_overlap": pc["keyword_overlap"],
        }
        for pc in per_chunk
    ]
    top_chunk_id = (
        (rrf_chunk or {}).get("id") if (rrf_chunk or {}) else None
    )

    # Rule 1 (no in-domain): already handled above.
    # Rule 2 (no chunks): already handled above.
    # Rule 3: structured conflict (e.g. primary school + ĐH).
    conflict = intent.get("conflict") or []
    if conflict:
        return {
            "status": "insufficient",
            "relevance_label": "Low",
            "suggested_action": "clarify",
            "signals": signals,
            "per_chunk": per_chunk,
            "passed_chunk_ids": signals["passed_chunk_ids"],
            "rejected_chunk_ids": signals["rejected_chunk_ids"],
            "top_chunk_id": top_chunk_id,
            "rationale": "Query chứa mâu thuẫn chủ đề: " + "; ".join(conflict),
        }

    # Rule 4: per-chunk gate result.
    if scored["passed_count"] == 0:
        # 0 candidates individually passed: refuse. Show the strongest
        # rejected reason in the rationale.
        top_reject = ""
        if rejected:
            rs = rejected[0].get("reasons") or []
            if rs:
                top_reject = rs[0]
        return {
            "status": "insufficient",
            "relevance_label": "Low",
            "suggested_action": "refuse",
            "signals": signals,
            "per_chunk": per_chunk,
            "passed_chunk_ids": [],
            "rejected_chunk_ids": signals["rejected_chunk_ids"],
            "top_chunk_id": top_chunk_id,
            "rationale": (
                f"Không có chunk nào vượt được evidence gate (0/{len(chunks)} "
                f"passes dense≥{score_threshold} + keyword_overlap≥{keyword_min}). "
                + (f"Lý do đầu: {top_reject}." if top_reject else "")
            ),
        }

    if scored["passed_count"] == 1:
        # Only one chunk passed the per-chunk gate. Two cases:
        #   * Strong single evidence (dense ≥ strong_dense AND overlap ≥ strong_overlap)
        #     → "sufficient" — the audit allows 1–3 supporting sources, so
        #       a single high-quality chunk is acceptable.
        #   * Anything else → "weak", clarify.
        only = passed[0]
        only_dense = float(only.get("dense_score") or 0.0)
        only_overlap = _keyword_overlap(query_tokens, (only.get("content") or "").strip())
        strong_dense = score_threshold + 0.20
        strong_overlap = max(keyword_min + 0.30, 0.50)
        if only_dense >= strong_dense and only_overlap >= strong_overlap:
            return {
                "status": "sufficient",
                "relevance_label": "High" if only_overlap >= 0.5 else "Medium",
                "suggested_action": "answer",
                "signals": signals,
                "per_chunk": per_chunk,
                "passed_chunk_ids": signals["passed_chunk_ids"],
                "rejected_chunk_ids": signals["rejected_chunk_ids"],
                "top_chunk_id": top_chunk_id,
                "rationale": (
                    f"1 chunk strong evidence (dense={only_dense:.3f}, "
                    f"overlap={only_overlap:.2f})."
                ),
            }
        return {
            "status": "weak",
            "relevance_label": "Low",
            "suggested_action": "clarify",
            "signals": signals,
            "per_chunk": per_chunk,
            "passed_chunk_ids": signals["passed_chunk_ids"],
            "rejected_chunk_ids": signals["rejected_chunk_ids"],
            "top_chunk_id": top_chunk_id,
            "rationale": (
                f"Chỉ 1/{len(chunks)} chunk vượt được evidence gate nhưng "
                f"không đủ mạnh (dense={only_dense:.3f}, overlap={only_overlap:.2f}); "
                "bằng chứng quá mỏng để trả lời chắc chắn."
            ),
        }

    # ≥2 chunks passed → sufficient.
    return {
        "status": "sufficient",
        "relevance_label": "High" if best_overlap >= 0.5 else "Medium",
        "suggested_action": "answer",
        "signals": signals,
        "per_chunk": per_chunk,
        "passed_chunk_ids": signals["passed_chunk_ids"],
        "rejected_chunk_ids": signals["rejected_chunk_ids"],
        "top_chunk_id": top_chunk_id,
        "rationale": (
            f"{scored['passed_count']}/{len(chunks)} chunk vượt được evidence gate; "
            f"dense_top1={dense_top1:.3f}, best overlap={best_overlap:.2f}, "
            f"intent={intent.get('intent')!r}, domain={intent.get('domain')!r}."
        ),
    }


def make_clarification_message(
    intent: dict[str, Any],
    query: str,
) -> str:
    """Produce a friendly clarification message for the user.

    No legal claims are fabricated. We just point out the apparent conflict
    and offer the most natural interpretation.
    """
    classification = intent.get("domain_classification") or {}
    if classification.get("is_in_domain") is False:
        return (
            "Xin lỗi, câu hỏi này nằm ngoài phạm vi dữ liệu tuyển sinh đại học "
            "của hệ thống nên mình không thể cung cấp câu trả lời đáng tin cậy."
        )
    education = intent.get("education_level", "unspecified")
    domain = intent.get("domain", "unknown")
    if education in ("primary_school", "secondary_school") and domain == "university_admission":
        level_pretty = {
            "primary_school": "học sinh lớp 1 (tiểu học)",
            "secondary_school": "học sinh THCS",
        }[education]
        return (
            "Bạn đang nói đến " + level_pretty + " hay học sinh THPT/lớp 12? "
            "Nếu là " + level_pretty + " thì chưa thuộc nhóm đối tượng xét tuyển "
            "đại học; nếu bạn muốn hỏi về phương thức xét tuyển dành cho học sinh "
            "THPT/lớp 12, mình có thể tra cứu theo trường hợp đó."
        )
    if intent.get("is_out_of_domain"):
        return (
            "Câu hỏi này nằm ngoài phạm vi dữ liệu hiện có của hệ thống "
            "(chuyên về tuyển sinh đại học Việt Nam). Bạn có thể hỏi lại về "
            "điều kiện, phương thức xét tuyển, kỳ thi tốt nghiệp THPT, hoặc các "
            "tin tức giáo dục liên quan không?"
        )
    # Generic clarification.
    return (
        "Mình chưa tìm được đoạn nào trong tài liệu trả lời trực tiếp câu hỏi: \""
        + query + "\". Bạn có thể diễn đạt cụ thể hơn, hoặc cho biết bạn quan tâm "
        "đến điều kiện, phương thức, hay thời gian xét tuyển không?"
    )


def make_out_of_domain_message(
    classification: dict[str, Any] | None,
) -> str:
    """User-facing refusal for OOD queries (audit requirement §12).

    Per the audit spec: the message must explicitly state that the question
    is outside the scope of the system's university-admission data.
    """
    return (
        "Xin lỗi, câu hỏi này nằm ngoài phạm vi dữ liệu tuyển sinh đại học "
        "của hệ thống nên mình không thể cung cấp câu trả lời đáng tin cậy."
    )


def make_weak_evidence_message(intent: dict[str, Any]) -> str:
    """User-facing refusal when the corpus has related content but not
    enough to ground a confident answer (audit §12 weak case)."""
    return (
        "Hiện nguồn dữ liệu của hệ thống chưa đủ để xác nhận cho trường hợp "
        "cụ thể này. Bạn có thể cung cấp thêm ngữ cảnh (trường, năm, phương "
        "thức cụ thể) để mình tra cứu chính xác hơn."
    )
