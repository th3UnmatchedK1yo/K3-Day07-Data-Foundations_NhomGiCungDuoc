# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Nhóm Gì Cũng Được (`NhomGiCungDuoc`)
**Thành viên:** Võ Hà Minh Huy (2A202601373) _(điền thêm tên các thành viên còn lại)_
**Ngày:** 2026-08-03

> **Nộp 1 bản / nhóm.** Phần cá nhân (hướng tiếp cận, kết quả riêng, dự đoán…) mỗi thành viên nộp riêng trong `REPORT_CANHAN.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần nhóm: 40** = Lựa chọn tài liệu (10) + Thiết kế chiến lược (15) + Chất lượng truy xuất (10) + Thuyết trình (5).

---

## 1. Lựa chọn tài liệu (Document Set Quality) — Nhóm (10 điểm)

### Phạm vi bộ tài liệu (Scope)

**Chủ đề (cố định theo lớp K3):** Dịch vụ / quy định đại học (đăng ký môn, học phí, học bổng, thư viện, ký túc xá…).

**Phạm vi cụ thể nhóm tập trung:**

> Quy trình hành chính dành cho sinh viên UIT (ĐKHP, tạm dừng/bảo lưu, xét tốt nghiệp, phúc khảo, chuyển ngành/trường, biểu mẫu…).

### Danh sách tài liệu (Data Inventory)

| #   | Tên tài liệu                                                                     | Nguồn (Source URL)                                             | Ngày lấy / Phiên bản                  | Số ký tự                | Metadata đã gán                                                                                                                                  |
| --- | -------------------------------------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | Một số quy trình dành cho sinh viên (UIT)                                        | https://student.uit.edu.vn/mot-so-quy-trinh-danh-cho-sinh-vien | 2026-08-03 / crawl Markdown công khai | 17,560                  | `source_id`, `source_title`, `source_url`, `institution=UIT`, `audience=student`, `document_type=procedure`, `procedure_slug`, `procedure_title` |
| 2   | (các section quy trình sau khi tách) — ví dụ Đăng ký học phần                    | cùng nguồn #1                                                  | cùng #1                               | 3,520                   | như trên + `section_index`                                                                                                                       |
| 3   | Tạm dừng học tập / bảo lưu                                                       | cùng nguồn #1                                                  | cùng #1                               | 1,402                   | như trên                                                                                                                                         |
| 4   | Xét tốt nghiệp                                                                   | cùng nguồn #1                                                  | cùng #1                               | 1,867                   | như trên                                                                                                                                         |
| 5   | Các quy trình còn lại trong file (bảng điểm, chuyển ngành, phúc khảo, biểu mẫu…) | cùng nguồn #1                                                  | cùng #1                               | phần còn lại của 17,560 | như trên — tổng **20 procedure sections** (xem `data/uit/procedure_boundaries.json`)                                                             |

> Ghi chú: Nguồn chính là **một file Markdown công khai** chứa nhiều quy trình; nhóm **không ghi đè nội dung gốc**, chỉ normalize định dạng + tách section bằng anchor phrase + gắn metadata. File cấu hình ranh giới: `data/uit/procedure_boundaries.json`.

**Danh sách kiểm tra quản trị dữ liệu (Data governance checklist):**

- [x] Tập tài liệu (Corpus) chỉ chứa nguồn công khai/được phép dùng và không chứa dữ liệu cá nhân, thông tin đăng nhập hoặc tài liệu nội bộ.
- [x] Mỗi tài liệu/section có `source_url` và metadata truy vết; `document_version`/ngày hiệu lực **không bịa** vì nguồn không nêu rõ (đúng yêu cầu lab: không tự thêm `effective_date`/`academic_year` nếu tài liệu không nói).

### Cấu trúc Metadata (Metadata Schema)

| Trường metadata                       | Kiểu         | Ví dụ giá trị                    | Tại sao hữu ích cho truy xuất (retrieval)?            |
| ------------------------------------- | ------------ | -------------------------------- | ----------------------------------------------------- |
| `source_id`                           | string       | `uit_student_procedures`         | Định danh corpus để citation ổn định                  |
| `source_url`                          | string       | `https://student.uit.edu.vn/...` | Truy vết nguồn gốc câu trả lời                        |
| `audience`                            | string       | `student`                        | Lọc đúng đối tượng (yêu cầu K3)                       |
| `institution`                         | string       | `UIT`                            | Phân biệt nếu sau này gộp nhiều trường                |
| `document_type`                       | string       | `procedure`                      | Phân loại loại tài liệu                               |
| `procedure_slug`                      | string       | `dang_ky_hoc_phan`               | Metadata pre-filter theo quy trình khi confidence cao |
| `procedure_title`                     | string       | `Đăng ký học phần (ĐKHP)`        | Hiển thị + header embedding                           |
| `section_index` / `chunk_index`       | int          | `2`, `0`                         | Định vị vị trí trong tài liệu                         |
| `previous_chunk_id` / `next_chunk_id` | string\|null | `...::chunk_001`                 | Adjacent expansion giữ trọn bước/điều kiện            |
| `content_hash`                        | string       | `a1b2c3...`                      | Dedup chunk trùng nội dung                            |

---

## 2. Thiết kế chiến lược (Strategy Design) — Nhóm (15 điểm)

> Mỗi thành viên thử **một chiến lược khác nhau** trên cùng bộ tài liệu; nhóm tổng hợp và so sánh ở đây.

### Phân tích đường cơ sở (Baseline Analysis)

Chạy `ChunkingStrategyComparator().compare(text, chunk_size=500)` trên 3 section UIT (Python 3.11):

| Tài liệu                       | Chiến lược (Strategy)            | Số lượng Chunk | Độ dài trung bình     | Giữ được ngữ cảnh không?                                     |
| ------------------------------ | -------------------------------- | -------------- | --------------------- | ------------------------------------------------------------ |
| Đăng ký học phần (3,520 ký tự) | FixedSizeChunker (`fixed_size`)  | 8              | 483.8                 | Trung bình — dễ cắt giữa “Bước N” và nội dung                |
| Đăng ký học phần               | SentenceChunker (`by_sentences`) | 11             | 317.1                 | Tốt hơn ở ranh giới câu, nhưng vẫn trộn ý nếu nhiều câu ngắn |
| Đăng ký học phần               | RecursiveChunker (`recursive`)   | 8              | 438.2                 | Khá — ưu tiên `\n\n` nên gần đoạn văn hơn FixedSize          |
| Tạm dừng học tập (1,402 ký tự) | FixedSize / Sentence / Recursive | 4 / 3 / 4      | 388.0 / 464.0 / 349.0 | Sentence giữ đoạn điều kiện tốt hơn trên section ngắn        |
| Xét tốt nghiệp (1,867 ký tự)   | FixedSize / Sentence / Recursive | 5 / 5 / 5      | 413.4 / 371.0 / 371.8 | Ba chiến lược gần nhau về số chunk                           |

### Chiến lược của từng thành viên

> Mỗi thành viên điền một khối dưới đây (copy thêm nếu nhóm có nhiều hơn 3 người).

**Thành viên 1 — Võ Hà Minh Huy**

- **Loại chiến lược:** custom — **StructureAwareChunker** (theo heading/section/procedure) + hybrid retrieval
- **Mô tả & lý do chọn cho chủ đề này:** Tài liệu UIT gộp nhiều quy trình trong một Markdown; một số section mất tiêu đề và bắt đầu thẳng bằng “Bước 1”. Chunk toàn văn theo FixedSize dễ trộn hai quy trình và cắt mất điều kiện (“tối đa”, “dưới 30%”, “chậm nhất”). Vì vậy tách section bằng anchor phrase (`procedure_boundaries.json`), chunk theo đoạn văn/bước trong từng quy trình, gắn `procedure_slug`, rồi retrieve bằng dense + BM25 + RRF; chỉ metadata-filter khi confidence cao; mở rộng chunk liền kề cùng quy trình trước khi gọi DeepSeek.
- **Code snippet (nếu custom):**

```python
# src/structure_chunking.py — ý chính
class StructureAwareChunker:
    def chunk_sections(self, sections: list[dict]) -> list[Chunk]:
        # Mỗi procedure section chunk độc lập — không trộn 2 quy trình.
        # Ưu tiên ranh giới: paragraph (Bước/bullet) > câu > khoảng trắng.
        # Prepend header cho embedding; giữ raw_text riêng để citation sạch.
        ...
```

Pipeline đầy đủ: `src/uit_preprocessing.py` → `StructureAwareChunker` → `LocalRAGEmbedder` (`BAAI/bge-m3`) → `HybridRetriever` → `DeepSeekClient`. Chi tiết: `docs/ARCHITECTURE_FLOW.md`. Benchmark: `python bench.py --compare-strategies`.

**Thành viên 2 — [Tên thành viên 2]**

- **Loại chiến lược:** _(điền: FixedSize / Sentence / Recursive / custom)_
- **Mô tả & lý do chọn:** _(điền 2-3 câu)_
- **Code snippet (nếu custom):**

**Thành viên 3 — [Tên thành viên 3]**

- **Loại chiến lược:**
- **Mô tả & lý do chọn:**
- **Code snippet (nếu custom):**

### So Sánh Giữa Các Thành Viên

| Thành viên   | Chiến lược (Strategy)                             | Điểm truy xuất (/10)                                                        | Điểm mạnh                                                             | Điểm yếu                                                               |
| ------------ | ------------------------------------------------- | --------------------------------------------------------------------------- | --------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| Võ Hà Minh Huy (2A202601373) | StructureAware + hybrid (dense+BM25+RRF+metadata) | 10/10 (5/5 câu top-3 relevant + agent có citation đúng trên benchmark thật) | Đúng procedure, giữ điều kiện/giới hạn, refusal đúng câu ngoài corpus | Latency cao hơn baseline; phụ thuộc cấu hình anchor + model local nặng |
| [TV2]        |                                                   |                                                                             |                                                                       |                                                                        |
| [TV3]        |                                                   |                                                                             |                                                                       |                                                                        |

**So sánh nhanh baseline vs chiến lược của Thành viên 1 (cùng 5 câu nhóm):**

| Metric                    | baseline (FixedSize + MiniLM + dense-only) | high_accuracy (StructureAware + BGE-M3 + hybrid) |
| ------------------------- | ------------------------------------------ | ------------------------------------------------ |
| Hit@1                     | 20%                                        | **100%**                                         |
| Recall@5                  | 60%                                        | **100%**                                         |
| MRR@5                     | 0.367                                      | **1.000**                                        |
| Procedure accuracy        | 0%                                         | **100%**                                         |
| Keyword coverage (answer) | 48.3%                                      | **83.3%**                                        |
| Citation present          | 60%                                        | **100%**                                         |

Nguồn số liệu: `benchmark/results/latest.md` (đã chạy **retrieval + DeepSeek generation**, không chỉ retrieval).

**Chiến lược nào tốt nhất cho chủ đề này? Tại sao?**

> Trên corpus quy trình UIT này, high_accuracy (structure-aware + hybrid) rõ ràng tốt hơn baseline FixedSize dense-only: đúng procedure ở cả 5 câu, giữ được số liệu/điều kiện trong answer, và từ chối đúng câu hỏi ngoài tài liệu. Lý do chính là **không trộn quy trình** + **lexical match (BM25)** bù cho mã biểu mẫu/số liệu mà dense dễ bỏ sót + **metadata filter có ngưỡng** giúp thu hẹp ứng viên khi query đã đủ rõ.

---

## 3. Câu hỏi đánh giá & Chất lượng truy xuất (Retrieval Quality) — Nhóm (10 điểm)

### Câu hỏi đánh giá & Câu trả lời chuẩn (nhóm thống nhất)

> **Đúng 5 câu hỏi**, đa dạng, có thể kiểm chứng; **ít nhất 1 câu** cần lọc metadata mới trả lời tốt. Đây là bộ câu hỏi chung cho mọi thành viên chạy. (Lưu trong `benchmark/gold_queries.json`.)

| #   | Câu hỏi (Query)                                                                                              | Câu trả lời chuẩn (Gold Answer)                                                                                                       | Chunk nào chứa thông tin?                                                     |
| --- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| 1   | Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ? Khi nào được đăng ký 30 tín chỉ?           | Tối đa **24 tín chỉ**/học kỳ; được **30 tín chỉ** nếu **ĐTBC ≥ 8,0**.                                                                 | Section `dang_ky_hoc_phan` (gold keywords: 24 tín chỉ, ĐTBC, 8,0, 30 tín chỉ) |
| 2   | Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi những gì và có được đăng ký môn mới không?       | Chỉ xử lý **học lại / cải thiện / sửa đổi dưới 30%**; **không được phép ĐKHP mới**.                                                   | Section `dang_ky_hoc_phan` (đợt cứu xét)                                      |
| 3   | Sinh viên xin tạm dừng học tập vì lý do cá nhân phải đáp ứng điều kiện gì và được tạm dừng tối đa bao lâu?   | Đã **học ít nhất 1 học kỳ**, **không bị đình chỉ**; tạm dừng **01 đến tối đa 02 học kỳ chính liên tiếp**.                             | Section `tam_dung_hoc_tap`                                                    |
| 4   | Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào?                                  | Nộp **đơn xin nhập học lại** (**Mẫu 07** / **Mẫu 09**), **chậm nhất 1 tháng trước** học kỳ mới.                                       | Section `tam_dung_hoc_tap` (phần sau bảo lưu)                                 |
| 5   | Để đăng ký xét tốt nghiệp, sinh viên cần đóng khoản gì, chuẩn bị hồ sơ nào và hoàn thành những nghĩa vụ nào? | Đóng **lệ phí xét cấp bằng**; hồ sơ **bằng THPT, giấy khai sinh, chứng chỉ ngoại ngữ**; hoàn thành nghĩa vụ như **nợ sách thư viện**… | Section `xet_tot_nghiep`                                                      |

> Câu 1–5 đều hưởng lợi từ `metadata_filter` theo `procedure_slug` (audience=`student` trên toàn corpus). Diagnostic ngoài báo cáo: “Chi phí ký túc xá UIT…?” — không có trong corpus, hệ thống phải từ chối.

### Tổng hợp chất lượng truy xuất của nhóm

> Cách chấm (theo `docs/SCORING.md`): **2 điểm/câu** — top-3 chứa chunk liên quan + agent trả lời đúng (2), có liên quan nhưng thiếu/không ở top-1 (1), không có trong top-3 (0).

| #   | Câu hỏi                        | Chiến lược tốt nhất cho câu này | Có chunk liên quan trong top-3? | Ghi chú                                                     |
| --- | ------------------------------ | ------------------------------- | ------------------------------- | ----------------------------------------------------------- |
| 1   | Tín chỉ tối đa / 30 tín chỉ    | StructureAware + hybrid (TV1)   | Có (top-1)                      | Baseline dense-only hay lệch sang `cong_nhan_tin_chi`       |
| 2   | Đợt cứu xét ĐKHP               | StructureAware + hybrid (TV1)   | Có (top-1)                      | Baseline miss trong top-5 trên lần chạy thật                |
| 3   | Tạm dừng học tập               | StructureAware + hybrid (TV1)   | Có (top-1)                      | Baseline đôi khi cũng hit nhờ từ khóa “tạm dừng”            |
| 4   | Hết hạn bảo lưu / nhập học lại | StructureAware + hybrid (TV1)   | Có (top-1)                      | Baseline miss; metadata `tam_dung_hoc_tap` giúp rõ          |
| 5   | Xét tốt nghiệp                 | StructureAware + hybrid (TV1)   | Có (top-1)                      | Cả hai strategy có thể trả lời được phần hồ sơ nếu chunk đủ |

**Điểm Retrieval Quality (theo chiến lược TV1 trên 5 câu nhóm):** 5 × 2 = **10 / 10** (top-3 relevant + agent grounded có citation trên mọi câu). Thành viên khác điền cột chiến lược của mình khi đã chạy xong.

**Lọc bằng metadata có giúp ích không? Ở câu hỏi nào?**

> Có — trên high_accuracy, metadata filter được áp dụng cho **5/5** core queries và **không làm mất** chunk relevant nào (0/5 bị “lọc quá chặt”). Đặc biệt hữu ích ở câu 1–2 (`dang_ky_hoc_phan`) và câu 3–4 (`tam_dung_hoc_tap`) vì query có từ khóa rõ (ĐKHP, tín chỉ, tạm dừng, bảo lưu). Filter chỉ bật khi confidence ≥ ngưỡng (~0.6); câu mơ hồ/ngoài corpus (ký túc xá) không bị ép vào một procedure sai.

---

## 4. Thuyết trình (Demo) & Bài học nhóm — Nhóm (5 điểm)

**Những phân tích (insights) hay nhất nhóm sẽ trình bày:**

> 1. Cùng một Markdown UIT: FixedSize+dense-only chỉ đạt Recall@5=60%, trong khi structure-aware + hybrid đạt 100% trên 5 câu nhóm — chứng minh **chiến lược dữ liệu/chunking > chỉ đổi model**.
> 2. Metadata filter **giúp** khi confidence cao (5/5 câu vẫn giữ relevant) nhưng phải có ngưỡng; filter cứng trên query mơ hồ sẽ loại mất kết quả đúng.
> 3. Failure mode ngoài corpus: câu “chi phí ký túc xá” — retrieval score thấp + DeepSeek từ chối đúng, không bịa (diagnostic trong `benchmark/diagnostic_queries.json`).

**Bài học rút ra khi so sánh trong nhóm:**

> Cùng tài liệu và cùng 5 câu hỏi, khác chunking/retrieval dẫn tới khác biệt lớn ở Hit@1 và procedure accuracy. Dense mạnh về paraphrase nhưng yếu với mã biểu mẫu/số liệu; BM25 bù lại. Chunk cắt giữa “Bước N” và điều kiện làm agent trả lời thiếu dù “có vẻ retrieve được”.

**Nếu làm lại, nhóm sẽ thay đổi gì trong chiến lược dữ liệu (data strategy)?**

> Bổ sung thêm 4–9 tài liệu UIT/khoa khác (học phí, thư viện, ký túc xá) thành corpus 5–10 file đúng rubric; thống nhất `sources.csv` đầy đủ; thử thêm reranker local trên top-10; và mỗi thành viên còn lại nên chạy đúng một chiến lược built-in (Sentence/Recursive/FixedSize) trên cùng gold queries để bảng so sánh nhóm đầy đủ hơn.

---

## Tự Đánh Giá (Phần Nhóm)

| Tiêu chí                                 | Điểm tự đánh giá                                                                                |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Lựa chọn tài liệu (Document Set Quality) | 8 / 10 _(mạnh về metadata/UIT thật; chưa đủ 5–10 file độc lập — chủ yếu 1 nguồn nhiều section)_ |
| Thiết kế chiến lược (Strategy Design)    | 12 / 15 _(TV1 đầy đủ + baseline số liệu thật; TV2/TV3 còn placeholder)_                         |
| Chất lượng truy xuất (Retrieval Quality) | 10 / 10 _(theo chiến lược TV1 trên 5 câu nhóm)_                                                 |
| Thuyết trình (Demo)                      | 4 / 5 _(đã có insight + failure analysis; demo miệng còn tùy buổi thuyết trình)_                |
| **Tổng phần nhóm**                       | **34 / 40** _(ước lượng — cập nhật khi các thành viên còn lại điền xong)_                       |
