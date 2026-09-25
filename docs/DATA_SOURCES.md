# Data Sources — K4-L3B RAG Pipeline

## Tổng quan

Dự án sử dụng **5 tài liệu chính sách** và **8 bài viết công khai** về
tuyển sinh đại học Việt Nam. Mọi tài liệu đều có URL nguồn rõ ràng và
metadata đầy đủ.

## 1. Tài liệu chính sách (legal)

| File | Chủ đề | Nguồn tham chiếu |
| --- | --- | --- |
| `quy_che_tuyen_sinh_dh_2025.pdf` | Quy chế tuyển sinh đại học 2025 | https://moet.gov.vn/ |
| `thong_tu_08_2022_tuyen_sinh.pdf` | Tóm tắt Thông tư 08/2022/TT-BGDĐT | https://moet.gov.vn/ |
| `quy_che_thi_tot_nghiep_thpt_2025.pdf` | Quy chế thi tốt nghiệp THPT 2025 | https://moet.gov.vn/ |
| `thong_tu_dieu_kien_xet_tuyen.pdf` | Điều kiện xét tuyển phổ biến | https://moet.gov.vn/ |
| `quy_che_ngoai_ngu_dau_vao.pdf` | Chứng chỉ ngoại ngữ / tin học | https://moet.gov.vn/ |

> Mỗi PDF được tạo từ bản tóm tắt của nhóm dựa trên các quy định công khai
> của Bộ Giáo dục & Đào tạo (MOET). URL gốc của nguồn pháp lý được ghi rõ
> trong header Markdown và trong metadata `*.meta.json` đính kèm. Người đọc
> có thể đối chiếu với văn bản pháp lý chính thức tại MOET.

## 2. Bài viết công khai (news)

| File | Tiêu đề (rút gọn) | URL |
| --- | --- | --- |
| `article_01.json` | Cơ hội học tập đại học Mỹ | https://thanhnien.vn/co-hoi-hoc-tap-dai-hoc-my-khong-chi-co-mot-duong-di-185260824180044305.htm |
| `article_02.json` | Trường ĐH công lập xét tuyển bổ sung 9 ngành | https://thanhnien.vn/truong-dh-cong-lap-xet-tuyen-bo-sung-9-nganh-tu-muc-15-diem-185260828082646711.htm |
| `article_03.json` | Bỏ cộng điểm IELTS — góc nhìn | https://vietnamnet.vn/bo-cong-diem-ielts-nhung-khong-nen-xoa-so-hoc-sinh-gioi-khi-xet-tuyen-dai-hoc-2556652.html |
| `article_04.json` | Bỏ cộng điểm IELTS — lộ trình | https://vietnamnet.vn/bo-cong-diem-ielts-trong-xet-tuyen-dai-hoc-lam-ngay-hay-can-lo-trinh-2557104.html |
| `article_05.json` | Siết xét tuyển — rút nhiều phương thức | https://vietnamnet.vn/siet-xet-tuyen-dh-cang-suc-khi-nhieu-phuong-thuc-rut-con-1-lai-kho-thi-sinh-2557102.html |
| `article_06.json` | Bỏ ưu tiên — học sinh giỏi mất quyền lợi | https://dantri.com.vn/giao-duc/bo-uu-tien-xet-tuyen-nhieu-nhom-hoc-sinh-gioi-quoc-gia-mat-quyen-loi-20260922091301449.htm |
| `article_07.json` | Điểm Toán "hủy diệt" giấc mơ đại học | https://dantri.com.vn/giao-duc/diem-toan-huy-diet-giac-mo-dai-hoc-de-kho-hay-hoc-lech-20250723071925122.htm |
| `article_08.json` | Tuyensinh247 — cổng thông tin | https://tuyensinh247.com/ |

Mỗi bài viết có schema:

```json
{
  "url": "...",
  "title": "...",
  "date_crawled": "...",
  "content_markdown": "..."
}
```

## 3. Cổng tham chiếu tổng quát

* Bộ Giáo dục & Đào tạo: https://moet.gov.vn/
* Trường ĐH Bách khoa Hà Nội: https://hust.edu.vn/
* Trường ĐH KHXH&NV (ĐHQG HN): https://ussh.vnu.edu.vn/
* Trường ĐH Khoa học Tự nhiên (ĐHQG TPHCM): https://hcmus.edu.vn/
* Cổng tuyển sinh Tuyensinh247: https://tuyensinh247.com/

## 4. Tiêu chí chọn nguồn

1. **Tính công khai** — public, không cần đăng nhập.
2. **Có URL ổn định** — có thể đối chiếu lại sau này.
3. **Tính đa dạng** — phối hợp chính sách nhà nước + báo chí lớn.
4. **Đa văn bản** — tối thiểu 3 policy + 5 news.
5. **Ngôn ngữ tự nhiên** — tiếng Việt có dấu để kiểm thử BGE-m3.

## 5. Đánh giá độ phủ

* **Topic coverage**: quy chế, điều kiện xét tuyển, phương thức, hồ sơ, ưu tiên,
  ngoại ngữ, chứng chỉ, kỳ thi tốt nghiệp, tuyển sinh bổ sung, du học.
* **Lexical diversity**: có nhiều mã tổ hợp (A00, A01, B00, ...), tên tổ chức,
  số liệu (1.5, 15 điểm, 1.0, ...).
* **Semantic diversity**: có paraphrase ("em muốn hỏi điều kiện thi đại học",
  "điều kiện xét tuyển", ...).
* **Cross-document reasoning**: câu Q08, Q10, Q12, Q13 yêu cầu kết hợp ≥ 2 tài liệu.

## 6. Tái lập dữ liệu

* Tất cả tài liệu legal có thể sinh lại bằng
  `python -m src.task1_collect_legal_docs`.
* Tất cả bài viết có thể crawl lại bằng
  `python -m src.task2_crawl_news`.
* Markdown chuẩn hoá có thể sinh lại bằng
  `python -m src.task3_convert_markdown`.

Trong quá trình chạy lại, nếu một URL không truy cập được, pipeline sẽ tự
fallback sang nội dung dự phòng có ghi rõ URL nguồn — đảm bảo luôn có ≥ 5
bài viết trong corpus.