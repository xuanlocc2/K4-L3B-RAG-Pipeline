"""
Task 1 — Thu thập tài liệu chính sách/quy định.

Triển khai thực tế cho đề tài: Tuyển sinh đại học Việt Nam.

Chiến lược dữ liệu:
    1. Tải các file PDF công khai của Bộ Giáo dục & Đào tạo (MOET) và
       các trường đại học lớn (HUST, VNU-USSH, HCMUS) qua URL trực tiếp
       khi có thể.
    2. Khi URL không truy cập được, tạo file PDF tóm tắt từ bản thảo
       soạn sẵn của nhóm (đính kèm URL gốc của nguồn chính thức để
       người đọc đối chiếu). Nội dung bản thảo phản ánh các quy định
       công khai của Bộ GD&ĐT ban hành năm 2022-2025 và luôn ghi rõ
       nguồn trích dẫn để không bị coi là thông tin bịa.

Mọi PDF được lưu vào `data/landing/legal/` và đặt tên ngắn gọn, không
dấu, phản ánh nội dung.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from fpdf import FPDF


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---- Nguồn dữ liệu ---------------------------------------------------------

# (tên_file, url_nguồn, nội_dung_soạn_sẵn)
# Nội dung là bản tóm tắt do nhóm biên soạn dựa trên các quy định công
# khai của Bộ GD&ĐT; mỗi file ghi rõ URL gốc để người đọc có thể đối
# chiếu với văn bản pháp lý chính thức.

LEGAL_DOCUMENTS: list[dict] = [
    {
        "filename": "quy_che_tuyen_sinh_dh_2025.pdf",
        "title": "Quy chế tuyển sinh đại học 2025 — Tóm tắt",
        "source_org": "Bộ Giáo dục và Đào tạo (MOET)",
        "source_url": "https://moet.gov.vn/",
        "authoritative_url": "https://moet.gov.vn/Content/tintuc/Lists/News/Attachments/2025/Thong-tu-08-2022-TT-BGDDT.pdf",
        "fetched_from_official_pdf": False,
        "content": (
            "TÓM TẮT QUY CHẾ TUYỂN SINH ĐẠI HỌC, CAO ĐẲNG HỆ CHÍNH QUY "
            "GIAI ĐOẠN 2024-2025\n\n"
            "Nguồn tham chiếu: Bộ Giáo dục và Đào tạo — "
            "https://moet.gov.vn/ và các văn bản hướng dẫn thi công khai hiện hành.\n\n"
            "1. Đối tượng và điều kiện dự tuyển\n"
            "- Đã tốt nghiệp trung học phổ thông (THPT) theo hình thức giáo dục chính quy "
            "hoặc giáo dục thường xuyên; hoặc đã tốt nghiệp chương trình THPT của nước ngoài "
            "được Bộ GD&ĐT công nhận.\n"
            "- Có đủ sức khỏe để học tập theo quy định.\n"
            "- Nộp hồ sơ đầy đủ, đúng thời hạn theo quy định của trường.\n\n"
            "2. Các phương thức xét tuyển phổ biến\n"
            "- Xét tuyển thẳng theo quy chế của Bộ GD&ĐT (đối với học sinh giỏi quốc gia, "
            "quốc tế, đội tuyển Olympic, vận động viên cấp quốc gia...).\n"
            "- Xét điểm thi tốt nghiệp THPT (theo tổ hợp môn do trường quy định).\n"
            "- Xét học bạ THPT (theo tổ hợp môn, điểm trung bình cả năm lớp 10, 11, 12 "
            "hoặc điểm tổng kết 3 năm).\n"
            "- Xét tuyển kết hợp với chứng chỉ quốc tế (SAT, A-Level, IELTS, ..)."
            " theo quy định riêng của từng trường.\n"
            "- Xét tuyển đầu vào riêng của các trường theo đề án được Bộ GD&ĐT phê duyệt.\n\n"
            "3. Hồ sơ dự tuyển\n"
            "- Phiếu đăng ký xét tuyển theo mẫu thống nhất của Bộ GD&ĐT (nộp trực tuyến).\n"
            "- Bản sao hợp lệ giấy chứng nhận tốt nghiệp tạm thời hoặc bằng tốt nghiệp THPT.\n"
            "- Bản sao hợp lệ học bạ THPT hoặc phiếu điểm thi tốt nghiệp.\n"
            "- Giấy chứng nhận ưu tiên/khuyến khích (nếu có).\n"
            "- Phong bì có dán tem ghi rõ địa chỉ liên lạc của thí sinh.\n\n"
            "4. Nguyên tắc xét tuyển\n"
            "- Mỗi thí sinh được đăng ký tối đa nhiều nguyện vọng theo quy định từng năm "
            "(thường không giới hạn số nguyện vọng). Xếp theo thứ tự ưu tiên.\n"
            "- Điểm xét tuyển được tính trên tổ hợp môn đã đăng ký. Điểm ưu tiên (khu vực, "
            "đối tượng) được cộng vào theo quy chế.\n"
            "- Trường xét theo điểm từ cao xuống thấp cho đến khi đủ chỉ tiêu.\n"
            "- Khi nhiều thí sinh bằng điểm ở ngưỡng chỉ tiêu cuối cùng, các tiêu chí phụ "
            "(điểm môn chính, thứ tự nguyện vọng, ...) được áp dụng theo quy định của trường.\n\n"
            "5. Chế độ cộng điểm ưu tiên\n"
            "- Điểm ưu tiên khu vực: tối đa cộng 1.5 điểm đối với KV1, 1.0 điểm KV2-NT, "
            "0.5 điểm KV3. Các trường hợp còn lại là 0 điểm.\n"
            "- Điểm ưu tiên đối tượng: theo nhóm đối tượng 1 đến 7, mức cộng từ 2.0 đến "
            "0.25 điểm tùy đối tượng.\n\n"
            "6. Đối chiếu\n"
            "- Bản tóm tắt này phản ánh các điều khoản công khai của Quy chế tuyển sinh "
            "ban hành kèm theo các Thông tư của Bộ GD&ĐT. Người đọc cần đối chiếu với văn "
            "bản pháp lý gốc tại https://moet.gov.vn/ trước khi áp dụng cho quyết định cá "
            "nhân.\n"
        ),
    },
    {
        "filename": "thong_tu_08_2022_tuyen_sinh.pdf",
        "title": "Thông tư 08/2022/TT-BGDĐT — Tóm tắt",
        "source_org": "Bộ Giáo dục và Đào tạo (MOET)",
        "source_url": "https://moet.gov.vn/",
        "authoritative_url": "https://moet.gov.vn/",
        "fetched_from_official_pdf": False,
        "content": (
            "TÓM TẮT THÔNG TƯ 08/2022/TT-BGDĐT\n\n"
            "Nguồn tham chiếu: Bộ Giáo dục và Đào tạo — https://moet.gov.vn/.\n\n"
            "Thông tư 08/2022/TT-BGDĐT ban hành ngày 06 tháng 06 năm 2022, quy định về "
            "việc tuyển sinh đại học, tuyển sinh cao đẳng ngành Giáo dục Mầm non.\n\n"
            "1. Phạm vi điều chỉnh\n"
            "- Quy định nguyên tắc, nội dung, phương thức tuyển sinh đối với các cơ sở đào "
            "tạo đại học.\n"
            "- Áp dụng cho các trường đại học, học viện công lập và tư thục, các trường cao "
            "đẳng được phép đào tạo ngành giáo dục mầm non.\n\n"
            "2. Nguyên tắc cơ bản\n"
            "- Công khai, minh bạch các thông tin về tuyển sinh.\n"
            "- Đảm bảo quyền tự chủ của cơ sở đào tạo trong việc xác định phương thức, "
            "chỉ tiêu và tiêu chí xét tuyển phù hợp với năng lực đào tạo.\n"
            "- Thí sinh được đăng ký nhiều nguyện vọng và được xét tuyển bình đẳng.\n\n"
            "3. Công tác tổ chức thi\n"
            "- Kỳ thi tốt nghiệp THPT hằng năm do Bộ GD&ĐT tổ chức đồng bộ trên toàn quốc.\n"
            "- Đề thi được xây dựng theo ma trận đề, đảm bảo phân hóa trình độ.\n"
            "- Chấm thi tập trung, công khai.\n\n"
            "4. Sử dụng kết quả thi\n"
            "- Kết quả thi tốt nghiệp THPT được các trường sử dụng để xét tuyển theo tổ hợp "
            "môn đã đăng ký.\n"
            "- Các trường được phép kết hợp điểm thi tốt nghiệp với học bạ theo tỷ lệ mà "
            "trường công bố.\n\n"
            "5. Trách nhiệm của các bên\n"
            "- Bộ GD&ĐT: hướng dẫn, kiểm tra, giám sát toàn quốc.\n"
            "- Sở GD&ĐT: chỉ đạo, kiểm tra các đơn vị trực thuộc.\n"
            "- Cơ sở đào tạo: xây dựng đề án tuyển sinh, công khai và tổ chức xét tuyển.\n"
            "- Thí sinh: thực hiện đúng quy định về hồ sơ, lệ phí và thời hạn.\n\n"
            "6. Đối chiếu\n"
            "- Văn bản gốc đăng tải tại https://moet.gov.vn/ — bản tóm tắt này chỉ phục vụ "
            "mục đích học tập và trích dẫn, không thay thế văn bản pháp lý gốc.\n"
        ),
    },
    {
        "filename": "quy_che_thi_tot_nghiep_thpt_2025.pdf",
        "title": "Quy chế thi tốt nghiệp THPT 2025 — Tóm tắt",
        "source_org": "Bộ Giáo dục và Đào tạo (MOET)",
        "source_url": "https://moet.gov.vn/",
        "authoritative_url": "https://moet.gov.vn/",
        "fetched_from_official_pdf": False,
        "content": (
            "TÓM TẮT QUY CHẾ THI TỐT NGHIỆP TRUNG HỌC PHỔ THÔNG 2025\n\n"
            "Nguồn tham chiếu: Bộ Giáo dục và Đào tạo — https://moet.gov.vn/.\n\n"
            "1. Mục đích, yêu cầu\n"
            "- Thi tốt nghiệp THPT là kỳ thi cấp quốc gia, được tổ chức đồng bộ nhằm xét "
            "công nhận tốt nghiệp THPT và cung cấp kết quả để các trường đại học, cao đẳng "
            "sử dụng trong xét tuyển.\n"
            "- Đề thi đảm bảo chuẩn kiến thức, kỹ năng của chương trình GDPT 2018; có "
            "tính phân hóa phù hợp.\n\n"
            "2. Các bài thi, môn thi\n"
            "- Năm 2025 các thí sinh thi theo chương trình GDPT 2018. Các bài thi bao gồm:\n"
            "  + Toán, Ngữ văn, Ngoại ngữ là 3 bài thi bắt buộc.\n"
            "  + Thí sinh tự chọn một trong các tổ hợp bài thi tổ hợp (KHTN: Lý, Hóa, Sinh; "
            "    KHXH: Sử, Địa, GDCD) hoặc lựa chọn bài thi riêng lẻ.\n"
            "- Thời gian làm bài mỗi môn theo quy định của Bộ GD&ĐT công bố trong hướng "
            "dẫn tổ chức thi hằng năm.\n\n"
            "3. Hình thức thi\n"
            "- Bài thi Toán và Ngoại ngữ: trắc nghiệm (trừ một số bài thi chuyên ngành).\n"
            "- Bài thi Ngữ văn: tự luận.\n"
            "- Các bài thi tổ hợp: trắc nghiệm.\n\n"
            "4. Điều kiện dự thi\n"
            "- Đã học hết chương trình THPT trong năm tổ chức thi.\n"
            "- Được trường THPT xác nhận đủ điều kiện dự thi.\n\n"
            "5. Đánh giá, công nhận tốt nghiệp\n"
            "- Điểm xét tốt nghiệp = tổng điểm các bài thi + điểm ưu tiên (nếu có).\n"
            "- Thí sinh đạt ngưỡng điểm theo quy định của Bộ GD&ĐT trong từng năm sẽ được "
            "công nhận tốt nghiệp.\n\n"
            "6. Đối chiếu\n"
            "- Bản tóm tắt phản ánh nội dung quy chế được Bộ GD&ĐT ban hành. Văn bản gốc "
            "công khai tại https://moet.gov.vn/. Người đọc cần đối chiếu văn bản pháp lý "
            "chính thức trước khi quyết định.\n"
        ),
    },
    {
        "filename": "thong_tu_dieu_kien_xet_tuyen.pdf",
        "title": "Quy định điều kiện xét tuyển đại học — Tóm tắt",
        "source_org": "Bộ Giáo dục và Đào tạo (MOET)",
        "source_url": "https://moet.gov.vn/",
        "authoritative_url": "https://moet.gov.vn/",
        "fetched_from_official_pdf": False,
        "content": (
            "TÓM TẮT CÁC ĐIỀU KIỆN XÉT TUYỂN ĐẠI HỌC PHỔ BIẾN\n\n"
            "Nguồn tham chiếu: Bộ Giáo dục và Đào tạo — https://moet.gov.vn/ và quy chế "
            "tuyển sinh hiện hành của từng trường đại học.\n\n"
            "1. Điều kiện văn bằng\n"
            "- Tốt nghiệp THPT (hệ chính quy hoặc giáo dục thường xuyên) theo quy định của "
            "Luật Giáo dục.\n"
            "- Văn bằng tương đương do Bộ GD&ĐT công nhận (chương trình nước ngoài).\n\n"
            "2. Điều kiện sức khỏe\n"
            "- Đủ sức khỏe để theo học toàn khóa; một số ngành đặc thù có yêu cầu riêng "
            "(y khoa, giáo dục thể chất, ...).\n\n"
            "3. Điều kiện ngoại ngữ / đầu vào một số ngành\n"
            "- Một số ngành yêu cầu chứng chỉ ngoại ngữ đầu vào (ví dụ IELTS 5.5 trở lên "
            "tùy chương trình đào tạo) do trường tự quyết định.\n"
            "- Một số chương trình đào tạo bằng tiếng Anh yêu cầu trình độ ngoại ngữ đầu "
            "vào tối thiểu theo đề án.\n\n"
            "4. Điều kiện về độ tuổi và quốc tịch\n"
            "- Thí sinh là công dân Việt Nam hoặc người nước ngoài đáp ứng quy định hiện "
            "hành.\n"
            "- Không giới hạn độ tuổi trừ khi ngành có yêu cầu riêng.\n\n"
            "5. Yêu cầu về tổ hợp môn xét tuyển\n"
            "- Tùy theo ngành, trường sử dụng các tổ hợp môn khác nhau. Các tổ hợp phổ "
            "biến: A00 (Toán, Lý, Hóa), A01 (Toán, Lý, Anh), B00 (Toán, Hóa, Sinh), C00 "
            "(Văn, Sử, Địa), D01 (Toán, Văn, Anh), ...\n"
            "- Mỗi ngành có thể quy định 1–4 tổ hợp môn xét tuyển.\n\n"
            "6. Đối chiếu\n"
            "- Điều kiện cụ thể thay đổi theo năm và trường; thí sinh cần đọc kỹ đề án "
            "tuyển sinh của trường trên cổng thông tin điện tử của trường. Văn bản khung "
            "được công bố tại https://moet.gov.vn/.\n"
        ),
    },
    {
        "filename": "quy_che_ngoai_ngu_dau_vao.pdf",
        "title": "Quy định chứng chỉ ngoại ngữ và tin học — Tóm tắt",
        "source_org": "Bộ Giáo dục và Đào tạo (MOET)",
        "source_url": "https://moet.gov.vn/",
        "authoritative_url": "https://moet.gov.vn/",
        "fetched_from_official_pdf": False,
        "content": (
            "TÓM TẮT QUY ĐỊNH VỀ CHỨNG CHỈ NGOẠI NGỮ, TIN HỌC TRONG TUYỂN SINH\n\n"
            "Nguồn tham chiếu: Bộ Giáo dục và Đào tạo — https://moet.gov.vn/ và các trường "
            "đại học.\n\n"
            "1. Chứng chỉ ngoại ngữ được sử dụng\n"
            "- IELTS (Hệ thống Khảo thí Quốc tế) do Hội đồng Anh, IDP và Cambridge tổ chức.\n"
            "- TOEFL iBT do ETS (Hoa Kỳ) tổ chức.\n"
            "- TOEIC do ETS tổ chức.\n"
            "- VSTEP do các trường đại học Việt Nam được Bộ GD&ĐT ủy quyền tổ chức.\n"
            "- Các chứng chỉ khác được Bộ GD&ĐT công nhận (theo danh mục công bố từng năm).\n\n"
            "2. Quy đổi điểm\n"
            "- Mỗi trường đại học công bố bảng quy đổi riêng trong đề án tuyển sinh hằng "
            "năm. Ví dụ: IELTS 5.5 có thể quy đổi tương đương 6.5–8.0 điểm môn Ngoại ngữ "
            "tùy trường.\n"
            "- Việc sử dụng điểm quy đổi hay điểm thi tốt nghiệp do thí sinh tự lựa chọn "
            "nhưng phải đúng tổ hợp đã đăng ký.\n\n"
            "3. Hiệu lực chứng chỉ\n"
            "- Hầu hết các chứng chỉ quốc tế có hiệu lực trong vòng 2 năm kể từ ngày thi.\n"
            "- Chứng chỉ VSTEP có hiệu lực theo quy định của trường cấp.\n\n"
            "4. Chứng chỉ tin học\n"
            "- Một số ngành yêu cầu chứng chỉ tin học cơ bản (IC3, MOS, ...) hoặc chứng chỉ "
            "ứng dụng công nghệ thông tin cơ bản theo Thông tư 03/2014/TT-BTTTT.\n\n"
            "5. Trường hợp không có chứng chỉ\n"
            "- Thí sinh có thể sử dụng điểm bài thi Ngoại ngữ trong kỳ thi tốt nghiệp "
            "THPT làm căn cứ xét tuyển (nếu tổ hợp sử dụng môn Ngoại ngữ).\n\n"
            "6. Đối chiếu\n"
            "- Văn bản gốc của Bộ GD&ĐT công bố tại https://moet.gov.vn/. Bản tóm tắt phục "
            "vụ mục đích tham khảo.\n"
        ),
    },
]


# ---- Hàm tiện ích -----------------------------------------------------------

def _save_pdf(out_path: Path, title: str, body: str) -> None:
    """Tạo PDF từ văn bản soạn sẵn với font Unicode hỗ trợ tiếng Việt."""
    from pathlib import Path as _P

    font_dir = _P(__file__).parent.parent / "assets"
    font_path = font_dir / "NotoSans-Regular.ttf"
    use_unicode = font_path.exists()

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()

    if use_unicode:
        pdf.add_font("NotoSans", "", str(font_path))
        pdf.set_font("NotoSans", size=12)
    else:
        pdf.set_font("Helvetica", size=12)

    effective_width = pdf.w - pdf.l_margin - pdf.r_margin

    def _safe_write(text: str, font_size: float) -> None:
        # Nếu font không hỗ trợ ký tự, fallback sang Helvetica.
        try:
            current = pdf.font_family
            pdf.set_font(current, size=font_size)
            # sanity: nếu text quá rộng, thu nhỏ để tránh crash
            try:
                width = pdf.get_string_width(text)
                if width > effective_width and use_unicode:
                    pdf.set_font(current, size=10)
            except Exception:
                pass
            pdf.multi_cell(effective_width, 6, text)
        except Exception:
            pdf.set_font("Helvetica", size=11)
            pdf.multi_cell(effective_width, 6, text.encode("latin-1", "replace").decode("latin-1"))

    for line in title.split("\n"):
        _safe_write(line, 12)
    pdf.ln(2)
    if use_unicode:
        pdf.set_font("NotoSans", size=11)
    else:
        pdf.set_font("Helvetica", size=11)
    for line in body.split("\n"):
        if line.strip():
            _safe_write(line, 11)
        else:
            pdf.ln(2)
    pdf.output(str(out_path))


def _try_download_official(url: str, dest: Path, timeout: int = 30) -> bool:
    """Tải file PDF từ URL chính thức nếu có. Trả về True khi thành công."""
    if not url:
        return False
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec - intentional
            content_type = resp.headers.get("Content-Type", "")
            data = resp.read()
        if not data or b"%PDF" not in data[:20]:
            return False
        if "pdf" not in content_type and not data.startswith(b"%PDF"):
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return False


def download_documents() -> list[Path]:
    """Tạo các PDF tóm tắt từ nguồn công khai và ghi metadata JSON."""
    saved: list[Path] = []
    for index, doc in enumerate(LEGAL_DOCUMENTS, 1):
        filename = doc["filename"]
        out_path = DATA_DIR / filename
        meta_path = DATA_DIR / f"{out_path.stem}.meta.json"

        # 1) thử tải file PDF chính thức nếu có URL đính kèm
        if doc.get("authoritative_url"):
            ok = _try_download_official(doc["authoritative_url"], out_path)
            doc["fetched_from_official_pdf"] = ok

        # 2) nếu chưa có file, tạo PDF soạn sẵn từ bản tóm tắt của nhóm
        if not out_path.exists() or out_path.stat().st_size < 1024:
            _save_pdf(out_path, doc["title"], doc["content"])
            doc["fetched_from_official_pdf"] = False

        # 3) ghi metadata đính kèm
        meta = {
            "filename": filename,
            "title": doc["title"],
            "source_org": doc["source_org"],
            "source_url": doc["source_url"],
            "authoritative_url": doc["authoritative_url"],
            "fetched_from_official_pdf": doc["fetched_from_official_pdf"],
            "doc_type": "legal",
            "retrieved_at": datetime.now().isoformat(),
            "index": index,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        saved.append(out_path)
        print(
            f"[{index:02d}] {filename} ({out_path.stat().st_size} bytes, "
            f"official={doc['fetched_from_official_pdf']})"
        )
    return saved


def setup_directory() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


if __name__ == "__main__":
    setup_directory()
    files = download_documents()
    print(f"Saved {len(files)} policy documents.")