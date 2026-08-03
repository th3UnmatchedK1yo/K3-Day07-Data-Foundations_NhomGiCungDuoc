# Báo Cáo Nhóm — Lab 7: Embedding & Vector Store

**Nhóm:** Nhóm Gì Cũng Được (`NhomGiCungDuoc`)  
**Thành viên:**

1. Võ Hà Minh Huy — MSSV: `2A202601373`
2. Đỗ Duy Đông — MSSV: _(bổ sung trước khi nộp)_
3. Nguyễn Minh Thái — MSSV: _(bổ sung trước khi nộp)_

**Ngày hoàn thiện báo cáo:** 2026-08-03

> **Phạm vi báo cáo:** Báo cáo này tổng hợp kết quả từ ba báo cáo cá nhân của nhóm, sử dụng cùng corpus UIT và cùng năm câu hỏi đánh giá. Những số liệu không xuất hiện trong báo cáo cá nhân hoặc kết quả benchmark đã có sẽ không được tự suy đoán thêm.

**Tổng điểm phần nhóm: 40 điểm** gồm:

- Lựa chọn tài liệu: 10 điểm
- Thiết kế chiến lược: 15 điểm
- Chất lượng truy xuất: 10 điểm
- Thuyết trình và phân tích: 5 điểm

---

## Tóm tắt kết quả chính

Nhóm xây dựng hệ thống truy xuất thông tin trên tài liệu công khai của Trường Đại học Công nghệ Thông tin — ĐHQG TP.HCM, tập trung vào các quy trình hành chính dành cho sinh viên như đăng ký học phần, tạm dừng/bảo lưu, phúc khảo, chuyển ngành, xét tốt nghiệp và các biểu mẫu liên quan.

Ba thành viên cùng hoàn thiện phần core của Lab và đều đạt **42/42 test**. Nhóm triển khai hai nhóm chiến lược thực nghiệm:

1. **RecursiveChunker + local multilingual MiniLM + dense retrieval**, được Đỗ Duy Đông và Nguyễn Minh Thái chạy độc lập trên cùng cấu hình, tạo 42 chunks và đạt 5/5 câu có chunk liên quan trong top-3.
2. **StructureAwareChunker + BGE-M3 + hybrid retrieval**, do Võ Hà Minh Huy triển khai, kết hợp dense retrieval, BM25, Reciprocal Rank Fusion, metadata filter và adjacent expansion trước khi gọi DeepSeek. Chiến lược này đạt Hit@1 = 100%, Recall@5 = 100%, MRR@5 = 1.000 và citation presence = 100% trên năm câu hỏi chung.

Kết quả cho thấy việc giữ đúng ranh giới quy trình, kết hợp semantic search với lexical search và sử dụng metadata đúng lúc tạo khác biệt rõ rệt so với việc chỉ chia theo kích thước hoặc chỉ dùng dense retrieval.

---

# 1. Lựa chọn tài liệu — Document Set Quality (10 điểm)

## 1.1. Chủ đề và phạm vi

**Chủ đề theo yêu cầu lớp K3:** Dịch vụ và quy định đại học.

**Phạm vi nhóm lựa chọn:**

> Quy trình hành chính dành cho sinh viên UIT, gồm đăng ký học phần, chuyển ngành, chuyển trường, tạm dừng/bảo lưu, hoãn thi, thôi học, phúc khảo, miễn học phần ngoại ngữ, xét tốt nghiệp, học chương trình thứ hai, công nhận tín chỉ và hệ thống biểu mẫu.

Phạm vi này phù hợp với mục tiêu của Lab vì tài liệu chứa nhiều dạng thông tin cần truy xuất chính xác:

- Trình tự nhiều bước: `Bước 1`, `Bước 2`, `Bước 3`.
- Điều kiện bắt buộc: “phải”, “chỉ”, “không được”.
- Giới hạn định lượng: 24 tín chỉ, 30 tín chỉ, dưới 30%, 01–02 học kỳ.
- Mốc thời gian: chậm nhất 1 tháng trước học kỳ mới, trong vòng 2 ngày làm việc.
- Tên biểu mẫu và mã nghiệp vụ: Mẫu 07, Mẫu 09, ĐKHP, ĐTBC.

Đây là corpus phù hợp để quan sát rõ ảnh hưởng của chunking, metadata và retrieval strategy đến câu trả lời của RAG.

## 1.2. Nguồn dữ liệu

| Thuộc tính | Giá trị |
|---|---|
| Tên nguồn | Một số quy trình dành cho sinh viên |
| Đơn vị cung cấp | Trường Đại học Công nghệ Thông tin — ĐHQG TP.HCM |
| URL | `https://student.uit.edu.vn/mot-so-quy-trinh-danh-cho-sinh-vien` |
| Ngày thu thập | 2026-08-03 |
| Định dạng sau thu thập | Markdown UTF-8 |
| Kích thước được ghi nhận | Khoảng 17.560 ký tự |
| Loại dữ liệu | Quy trình, điều kiện, hướng dẫn, biểu mẫu |
| Quyền truy cập | Nguồn công khai |

Tài liệu được tải từ trang công khai và chuyển sang Markdown. Nhóm không thay đổi nội dung quy định; các thao tác xử lý chỉ gồm:

- Chuẩn hóa newline và khoảng trắng.
- Giữ nguyên số liệu, URL, điều kiện, biểu mẫu và thứ tự bước.
- Xác định ranh giới các quy trình.
- Gắn metadata phục vụ truy xuất và citation.

## 1.3. Đơn vị dữ liệu trong corpus

Nguồn gốc hiện tại là **một tài liệu công khai**, nhưng nội dung chứa khoảng **20 quy trình logic**. Sau preprocessing, mỗi quy trình được xem là một section độc lập để tránh trộn ngữ cảnh.

| Nhóm nội dung | Ví dụ section | Vai trò trong benchmark |
|---|---|---|
| Đăng ký học phần | `dang_ky_hoc_phan` | Câu 1 và câu 2 |
| Tạm dừng/bảo lưu | `tam_dung_hoc_tap` | Câu 3 và câu 4 |
| Xét tốt nghiệp | `xet_tot_nghiep` | Câu 5 |
| Chuyển ngành/trường | `chuyen_nganh`, `chuyen_truong` | Tạo nhiễu semantic gần chủ đề học vụ |
| Phúc khảo/hoãn thi | `phuc_khao`, `hoan_thi` | Kiểm tra khả năng phân biệt quy trình |
| Biểu mẫu | `bieu_mau` | Kiểm tra retrieval với mã biểu mẫu |
| Công nhận tín chỉ | `cong_nhan_tin_chi` | Có thể gây nhầm với câu hỏi về giới hạn tín chỉ |

### Lưu ý về rubric 5–10 tài liệu

Corpus hiện có **một nguồn gốc độc lập** và nhiều section bên trong. Điều này mạnh về tính nhất quán và truy vết, nhưng chưa tương đương hoàn toàn với yêu cầu 5–10 tài liệu độc lập. Nhóm ghi nhận đây là hạn chế dữ liệu và không trình bày các section như các nguồn độc lập khác nhau.

Nếu tiếp tục mở rộng, nhóm sẽ bổ sung các tài liệu UIT công khai riêng biệt về:

- Học phí.
- Học bổng.
- Chính sách thư viện.
- Ký túc xá.
- Kế hoạch đăng ký học phần theo học kỳ.

## 1.4. Metadata schema

| Trường | Kiểu | Ví dụ | Mục đích |
|---|---|---|---|
| `source_id` | string | `uit_student_procedures` | Định danh corpus và xây citation ổn định |
| `source_title` | string | `Một số quy trình dành cho sinh viên` | Hiển thị tên nguồn |
| `source_url` | string | URL UIT | Truy vết nguồn gốc |
| `institution` | string | `UIT` | Phân biệt khi mở rộng nhiều trường |
| `audience` | string | `student` | Lọc đúng đối tượng |
| `document_type` | string | `procedure` | Phân biệt quy trình với thông báo/FAQ |
| `procedure_slug` | string | `dang_ky_hoc_phan` | Pre-filter đúng quy trình |
| `procedure_title` | string | `Đăng ký học phần` | Header embedding và hiển thị |
| `section_index` | integer | `2` | Giữ thứ tự section |
| `chunk_index` | integer | `0` | Xác định vị trí chunk trong section |
| `previous_chunk_id` | string/null | `...::chunk_000` | Mở rộng chunk liền trước |
| `next_chunk_id` | string/null | `...::chunk_002` | Mở rộng chunk liền sau |
| `content_hash` | string | SHA/hash ổn định | Dedup và cache embedding |

## 1.5. Quản trị và chất lượng dữ liệu

- [x] Chỉ sử dụng nguồn công khai.
- [x] Không chứa tài khoản, mật khẩu hoặc dữ liệu cá nhân của sinh viên.
- [x] Giữ URL nguồn để truy vết.
- [x] Không tự thêm ngày hiệu lực hoặc năm học nếu nguồn không nêu.
- [x] Không ghi đè file raw trong bước preprocessing.
- [x] Giữ nguyên các từ điều kiện như “tối đa”, “tối thiểu”, “chỉ”, “không được”, “chậm nhất”.
- [x] Giữ nguyên mã biểu mẫu, số liệu và viết tắt nghiệp vụ.

---

# 2. Thiết kế chiến lược — Strategy Design (15 điểm)

## 2.1. Pipeline chung của Lab

```text
Tài liệu Markdown
    ↓
Chunking
    ↓
Embedding từng chunk
    ↓
EmbeddingStore / Vector Store
    ↓
Embedding câu hỏi
    ↓
Tìm các chunk gần nhất
    ↓
Ghép context
    ↓
Agent / DeepSeek sinh câu trả lời
```

Phần core được cả ba thành viên hoàn thiện gồm:

- `SentenceChunker`
- `RecursiveChunker`
- `compute_similarity`
- `ChunkingStrategyComparator`
- `EmbeddingStore`
- `search_with_filter`
- `delete_document`
- `KnowledgeBaseAgent`

Cả ba thành viên đều báo cáo **42/42 tests passed**.

## 2.2. Phân tích ba chunker cơ bản

Kết quả baseline đã được ghi nhận trên các section chính của tài liệu UIT:

| Section | FixedSizeChunker | SentenceChunker | RecursiveChunker | Nhận xét |
|---|---:|---:|---:|---|
| Đăng ký học phần — 3.520 ký tự | 8 chunks, TB 483,8 | 11 chunks, TB 317,1 | 8 chunks, TB 438,2 | FixedSize dễ cắt giữa bước; Recursive giữ đoạn tốt hơn |
| Tạm dừng học tập — 1.402 ký tự | 4 chunks | 3 chunks | 4 chunks | Sentence giữ nguyên cụm điều kiện tốt trên section ngắn |
| Xét tốt nghiệp — 1.867 ký tự | 5 chunks | 5 chunks | 5 chunks | Khác biệt chủ yếu nằm ở vị trí cắt, không chỉ số chunk |

### Nhận xét

- **FixedSizeChunker:** đơn giản, nhanh, dễ benchmark nhưng không hiểu cấu trúc.
- **SentenceChunker:** giữ ranh giới câu, nhưng một quy trình có thể gồm nhiều câu liên tiếp nên chunk vẫn có thể thiếu bước trước/sau.
- **RecursiveChunker:** ưu tiên đoạn văn và xuống dòng trước khi cắt nhỏ, phù hợp hơn với tài liệu hành chính.
- **StructureAwareChunker:** phù hợp nhất với tài liệu hiện tại vì sử dụng ranh giới procedure và bước nghiệp vụ.

## 2.3. Chiến lược của từng thành viên

### Thành viên 1 — Võ Hà Minh Huy

**Chiến lược:** Structure-aware chunking + hybrid retrieval + DeepSeek generation.

**Embedding:** `BAAI/bge-m3`.

**Thành phần:**

```text
UIT Markdown
→ Procedure detection
→ StructureAwareChunker
→ BGE-M3 dense embedding
→ Dense retrieval + BM25
→ Reciprocal Rank Fusion
→ Metadata filter có confidence
→ Adjacent expansion
→ Context
→ DeepSeek
→ Answer + citation
```

**Lý do lựa chọn:**

Tài liệu UIT là một file lớn chứa nhiều quy trình. Nếu chunk toàn văn theo số ký tự, một chunk có thể chứa phần cuối của quy trình trước và phần đầu của quy trình sau. Điều này đặc biệt nguy hiểm khi câu trả lời phụ thuộc vào các từ điều kiện như:

- “dưới 30%”
- “không được phép ĐKHP mới”
- “chậm nhất 1 tháng trước”
- “tối đa 24 tín chỉ”

Structure-aware chunking chia độc lập theo từng `procedure_slug`, sau đó mới chia trong section. Hybrid retrieval kết hợp:

- Dense retrieval để bắt paraphrase.
- BM25 để bắt số liệu, mã biểu mẫu và từ viết tắt.
- Metadata filter để thu hẹp đúng quy trình khi query đủ rõ.
- Adjacent expansion để lấy lại bước hoặc điều kiện ở chunk liền kề.

**Kết quả nổi bật:**

- 5/5 câu có chunk đúng ở top-1.
- Hit@1 = 100%.
- Recall@5 = 100%.
- MRR@5 = 1.000.
- Procedure accuracy = 100%.
- Citation presence = 100%.
- Diagnostic ngoài corpus được từ chối đúng.

### Thành viên 2 — Đỗ Duy Đông

**Chiến lược:** `RecursiveChunker` + local multilingual MiniLM + dense retrieval.

**Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

**Cấu hình được báo cáo:**

- Corpus: tài liệu quy trình sinh viên UIT.
- Số chunks: 42.
- Store: `EmbeddingStore`.
- Retrieval: semantic dense search theo cosine/dot product trên embedding đã chuẩn hóa.
- Agent: lấy top-k chunk, ghép context và truyền vào LLM.

**Lý do lựa chọn:**

RecursiveChunker ưu tiên tách theo `\n\n`, `\n`, dấu câu và khoảng trắng. So với FixedSize, chiến lược này giảm khả năng cắt giữa đoạn hoặc giữa các bước của quy trình mà vẫn giữ cách triển khai gần với core lab.

**Kết quả được báo cáo:**

- 5/5 câu có chunk liên quan trong top-3.
- Câu 1 và câu 4 có top-1 chỉ liên quan một phần, nhưng agent vẫn tổng hợp được câu trả lời đúng khi dùng các chunk retrieval khác.
- Điểm cosine top-1 nằm trong khoảng 0,6624–0,6843.
- Hệ thống trả lời đúng các giới hạn tín chỉ, điều kiện tạm dừng, thời hạn nhập học lại và hồ sơ tốt nghiệp.

### Thành viên 3 — Nguyễn Minh Thái

**Chiến lược:** `RecursiveChunker` + local multilingual MiniLM + dense retrieval.

**Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, vector 384 chiều.

**Cấu hình được báo cáo:**

- Corpus: `data/uit/quy-trinh-danh-cho-sinh-vien.md`.
- Số chunks: 42.
- Retrieval: dense search.
- Top-k được đưa vào `KnowledgeBaseAgent` để dựng context.

**Lý do lựa chọn:**

Chiến lược Recursive giữ cấu trúc đoạn tốt hơn cắt cứng, trong khi MiniLM multilingual phù hợp với tiếng Việt và không yêu cầu API key. Đây là cấu hình nhẹ, dễ tái lập trên máy cá nhân.

**Kết quả được báo cáo:**

- 5/5 câu có chunk liên quan trong top-3.
- Câu 1 và câu 4 có top-1 liên quan một phần.
- Các câu 2, 3 và 5 có câu trả lời đúng dù top-1 chưa phải lúc nào cũng chứa toàn bộ thông tin cần thiết.
- Kết quả tương đồng và retrieval khớp với báo cáo của Đỗ Duy Đông, cho thấy cấu hình có tính lặp lại khi chạy trên cùng dữ liệu.

## 2.4. Mức độ khác biệt giữa ba chiến lược

Hai thành viên Đỗ Duy Đông và Nguyễn Minh Thái sử dụng cùng một chiến lược `RecursiveChunker + MiniLM + dense retrieval`. Vì vậy, hai kết quả này nên được xem là **hai lần chạy tái lập độc lập**, không phải hai chiến lược hoàn toàn khác nhau.

Điều này có giá trị ở khía cạnh reproducibility, nhưng chưa đáp ứng tối đa mục tiêu “mỗi thành viên thử một strategy riêng”. Nếu còn thời gian trước khi nộp, nhóm nên cho một trong hai thành viên chạy thêm:

- `SentenceChunker + MiniLM + dense retrieval`, hoặc
- `FixedSizeChunker + MiniLM + dense retrieval`.

Nhóm không tự sửa báo cáo cá nhân để tạo ra sự khác biệt không có trong kết quả thực tế.

## 2.5. Bảng so sánh giữa các thành viên

| Thành viên | Chunking | Embedding | Retrieval | Kết quả top-3 | Điểm mạnh | Hạn chế |
|---|---|---|---|---|---|---|
| Võ Hà Minh Huy | StructureAware | BGE-M3 | Dense + BM25 + RRF + metadata + adjacent | 5/5; cả 5 ở top-1 | Đúng procedure, giữ số liệu và điều kiện, có citation | Kiến trúc phức tạp, model nặng hơn, latency cao hơn |
| Đỗ Duy Đông | Recursive | MiniLM multilingual | Dense-only | 5/5 | Nhẹ, dễ chạy, semantic retrieval tốt hơn mock | Top-1 ở câu 1 và 4 chỉ liên quan một phần |
| Nguyễn Minh Thái | Recursive | MiniLM multilingual | Dense-only | 5/5 | Cấu hình dễ tái lập, vector 384 chiều, không cần API | Trùng strategy với thành viên 2; top-1 chưa luôn tối ưu |

## 2.6. So sánh baseline và high-accuracy

| Metric | Baseline: FixedSize + MiniLM + dense-only | High-accuracy: StructureAware + BGE-M3 + hybrid |
|---|---:|---:|
| Hit@1 | 20% | **100%** |
| Recall@5 | 60% | **100%** |
| MRR@5 | 0,367 | **1,000** |
| Procedure accuracy | 0% | **100%** |
| Keyword coverage của answer | 48,3% | **83,3%** |
| Citation presence | 60% | **100%** |

### Lưu ý khi đọc score

Score `0,0328` trong chiến lược hybrid là **RRF score**, còn score khoảng `0,66–0,68` trong chiến lược Recursive là **cosine/dense similarity**. Hai loại score có thang đo khác nhau và **không được so trực tiếp theo giá trị tuyệt đối**.

Chỉ nên so sánh bằng các metric chung như:

- Hit@1.
- Recall@3/5.
- MRR.
- Procedure accuracy.
- Keyword coverage.
- Citation validity.

## 2.7. Kết luận về chiến lược

Trên corpus hiện tại, StructureAware + hybrid đạt kết quả tốt nhất vì:

1. Không trộn hai quy trình trong cùng chunk.
2. Metadata thu hẹp đúng section khi query rõ nghĩa.
3. BM25 bắt tốt các cụm như `24 tín chỉ`, `Mẫu 07`, `ĐKHP`, `ĐTBC`.
4. Dense retrieval bắt được câu hỏi diễn đạt khác với tài liệu.
5. RRF kết hợp hai danh sách kết quả mà không phụ thuộc vào cùng một thang score.
6. Adjacent expansion giữ điều kiện hoặc bước nằm ở chunk liền kề.
7. DeepSeek được cung cấp context đã chọn lọc và citation rõ ràng.

---

# 3. Bộ câu hỏi đánh giá và chất lượng truy xuất — Retrieval Quality (10 điểm)

## 3.1. Nguyên tắc xây benchmark

Nhóm thống nhất đúng năm câu hỏi dùng chung cho cả ba thành viên. Mỗi câu hỏi phải:

- Có câu trả lời kiểm chứng được trong corpus.
- Chứa điều kiện hoặc chi tiết dễ bị bỏ sót.
- Có gold keywords.
- Có `expected_procedure` để đánh giá metadata/procedure accuracy.
- Không phụ thuộc vào chunk ID cố định, vì mỗi chiến lược tạo chunk khác nhau.

## 3.2. Năm câu hỏi và gold answer

| # | Câu hỏi | Gold answer | Expected procedure | Gold keywords chính |
|---:|---|---|---|---|
| 1 | Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ? Khi nào được đăng ký 30 tín chỉ? | Tối đa 24 tín chỉ/học kỳ; được đăng ký 30 tín chỉ nếu ĐTBC ≥ 8,0 tại thời điểm đăng ký. | `dang_ky_hoc_phan` | `24 tín chỉ`, `ĐTBC`, `8,0`, `30 tín chỉ` |
| 2 | Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi những gì và có được đăng ký môn mới không? | Chỉ xem xét học lại, cải thiện, sửa đổi hoặc điều chỉnh nhỏ dưới 30% số môn đã đăng ký; không được đăng ký học phần mới. | `dang_ky_hoc_phan` | `học lại`, `cải thiện`, `dưới 30%`, `không được phép ĐKHP mới` |
| 3 | Sinh viên xin tạm dừng học tập vì lý do cá nhân phải đáp ứng điều kiện gì và được tạm dừng tối đa bao lâu? | Phải học ít nhất 1 học kỳ, không thuộc trường hợp bị đình chỉ; thời gian từ 01 đến tối đa 02 học kỳ chính liên tiếp. | `tam_dung_hoc_tap` | `học ít nhất 1 học kỳ`, `không bị đình chỉ`, `01`, `02 học kỳ` |
| 4 | Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào? | Làm đơn xin nhập học lại, dùng Mẫu 07 hoặc Mẫu 09 tùy trường hợp, nộp chậm nhất 1 tháng trước học kỳ mới. | `tam_dung_hoc_tap` | `đơn xin nhập học lại`, `Mẫu 07`, `Mẫu 09`, `chậm nhất 1 tháng trước` |
| 5 | Để đăng ký xét tốt nghiệp, sinh viên cần đóng khoản gì, chuẩn bị hồ sơ nào và hoàn thành những nghĩa vụ nào? | Đóng lệ phí xét cấp bằng; chuẩn bị bản scan bằng THPT, giấy khai sinh, chứng chỉ ngoại ngữ; hoàn thành nghĩa vụ như nợ sách thư viện và các khoản liên quan. | `xet_tot_nghiep` | `lệ phí`, `bằng THPT`, `giấy khai sinh`, `chứng chỉ ngoại ngữ`, `nợ sách thư viện` |

## 3.3. Kết quả chi tiết theo từng thành viên

### Câu 1 — Giới hạn tín chỉ

| Thành viên | Top-1 | Đánh giá |
|---|---|---|
| Võ Hà Minh Huy | `dang_ky_hoc_phan::chunk_001` | Đúng top-1; answer đủ 24 tín chỉ, ĐTBC ≥ 8,0 và 30 tín chỉ |
| Đỗ Duy Đông | Chunk chứa giới hạn 24 tín chỉ và điều kiện ĐTBC ≥ 8 | Liên quan một phần theo báo cáo, nhưng agent trả lời đủ |
| Nguyễn Minh Thái | Cùng nội dung và score 0,6843 | Liên quan một phần, answer đầy đủ |

**Phân tích:** Đây là câu dense retrieval dễ nhầm sang section `cong_nhan_tin_chi` vì cùng chứa từ “tín chỉ”. Metadata filter theo `dang_ky_hoc_phan` giúp loại nhiễu rõ rệt.

### Câu 2 — Đợt cứu xét ĐKHP

| Thành viên | Top-1 | Đánh giá |
|---|---|---|
| Võ Hà Minh Huy | Chunk ĐKHP chứa đợt cứu xét | Đúng top-1; giữ “dưới 30%” và “không được ĐKHP mới” |
| Đỗ Duy Đông | Chunk về chọn loại đăng ký/hủy đăng ký | Top-1 chưa phải đoạn tối ưu, nhưng top-k đủ để agent trả lời đúng |
| Nguyễn Minh Thái | Cùng kết quả, score 0,6624 | Dense retrieval bị kéo về đoạn có từ “đăng ký/hủy đăng ký” |

**Phân tích:** Câu này cho thấy semantic similarity không đồng nghĩa với relevance hoàn toàn. Các đoạn cùng nói về đăng ký/hủy môn có thể có score cao, nhưng chỉ section đợt cứu xét chứa điều kiện “dưới 30%” và cấm đăng ký mới.

### Câu 3 — Tạm dừng học tập

| Thành viên | Top-1 | Đánh giá |
|---|---|---|
| Võ Hà Minh Huy | `tam_dung_hoc_tap::chunk_000` | Đúng top-1, đủ điều kiện và thời hạn |
| Đỗ Duy Đông | Chunk chứa thời gian 01–02 học kỳ | Có liên quan; agent kết hợp thêm điều kiện học ít nhất 1 học kỳ |
| Nguyễn Minh Thái | Cùng kết quả, score 0,6839 | Có liên quan; câu trả lời đúng |

**Phân tích:** Từ khóa “tạm dừng học tập” khá đặc trưng nên cả dense và hybrid đều hoạt động tốt. Tuy nhiên, điều kiện và thời hạn có thể nằm ở hai đoạn gần nhau; adjacent expansion giúp giữ trọn ý.

### Câu 4 — Nhập học lại sau bảo lưu

| Thành viên | Top-1 | Đánh giá |
|---|---|---|
| Võ Hà Minh Huy | Chunk tạm dừng/bảo lưu có phần nhập học lại | Đúng top-1, đủ Mẫu 07/09 và hạn 1 tháng |
| Đỗ Duy Đông | Chunk về thời gian tạm dừng 01–02 học kỳ | Chỉ liên quan một phần; top-1 không chứa trực tiếp toàn bộ thủ tục nhập học lại |
| Nguyễn Minh Thái | Cùng kết quả, score 0,6781 | Liên quan một phần; agent vẫn trả lời đúng từ top-k context |

**Phân tích:** Đây là failure case tốt để minh họa. Query dùng từ “bảo lưu”, còn đoạn trả lời cụ thể nằm ở bước sau trong cùng quy trình. Chỉ dense top-1 có thể chọn đoạn nói về thời gian bảo lưu thay vì thủ tục nhập học lại. Metadata filter + adjacent expansion giải quyết tốt hơn.

### Câu 5 — Xét tốt nghiệp

| Thành viên | Top-1 | Đánh giá |
|---|---|---|
| Võ Hà Minh Huy | `xet_tot_nghiep::chunk_000` | Đúng top-1, answer có hồ sơ, lệ phí, nghĩa vụ và citation |
| Đỗ Duy Đông | Chunk về nhận bằng tốt nghiệp | Có liên quan nhưng top-1 nằm cuối quy trình; agent vẫn lấy đủ thông tin từ top-k |
| Nguyễn Minh Thái | Cùng kết quả, score 0,6719 | Dense top-1 chưa phải đoạn hồ sơ, nhưng answer đúng |

**Phân tích:** Một quy trình dài có nhiều đoạn cùng chung chủ đề “tốt nghiệp”. Nếu chỉ nhìn top-1, hệ thống có thể chọn bước nhận bằng thay vì bước đăng ký và chuẩn bị hồ sơ. Structure-aware chunking và reranking theo query giúp đưa đoạn phù hợp lên đầu.

## 3.4. Bảng tổng hợp kết quả

| Thành viên/strategy | Top-3 relevant | Top-1 hoàn toàn đúng | Top-1 liên quan một phần | Citation được báo cáo | Out-of-corpus refusal |
|---|---:|---:|---:|---:|---:|
| Võ Hà Minh Huy — StructureAware hybrid | 5/5 | 5/5 | 0/5 | 5/5 | Đúng |
| Đỗ Duy Đông — Recursive MiniLM dense | 5/5 | 3/5 | 2/5 | Không có metric citation riêng trong báo cáo | Chưa báo cáo |
| Nguyễn Minh Thái — Recursive MiniLM dense | 5/5 | 3/5 | 2/5 | Không có metric citation riêng trong báo cáo | Chưa báo cáo |

> “Top-1 hoàn toàn đúng” và “liên quan một phần” ở hai chiến lược Recursive được tổng hợp từ chính cột đánh giá trong báo cáo cá nhân, không phải metric tự động mới.

## 3.5. Metadata filter có giúp không?

Có. Tác động rõ nhất xuất hiện ở:

- **Câu 1–2:** lọc về `dang_ky_hoc_phan` để tránh `cong_nhan_tin_chi` hoặc các đoạn điều chỉnh đăng ký khác.
- **Câu 3–4:** lọc về `tam_dung_hoc_tap` để giữ cùng một quy trình bảo lưu/nhập học lại.
- **Câu 5:** lọc về `xet_tot_nghiep` để tránh các section biểu mẫu hoặc giấy giới thiệu.

Tuy nhiên, metadata filter không nên bật cứng với mọi query. Nếu query mơ hồ hoặc hỏi ngoài corpus, việc ép vào một procedure có thể làm retrieval trả về nội dung gần nhất nhưng sai. Vì vậy chiến lược high-accuracy chỉ filter khi confidence đủ cao.

## 3.6. Diagnostic query ngoài corpus

**Query:** “Chi phí ký túc xá UIT hiện tại là bao nhiêu?”

Corpus hiện tại không chứa thông tin này. Hành vi đúng là:

```text
Không tìm thấy thông tin này trong tài liệu UIT đã nạp.
```

Chiến lược high-accuracy đã báo cáo từ chối đúng, thay vì dùng kiến thức ngoài context hoặc bịa ra mức phí.

---

# 4. Phân tích lỗi và bài học nhóm

## 4.1. Failure case 1 — Mock embedding không phản ánh ngữ nghĩa

Đỗ Duy Đông thử `_mock_embed` trong phần similarity và ghi nhận nhiều cặp cùng nghĩa có score thấp hoặc cặp khác nghĩa có score không thấp như kỳ vọng. Nguyên nhân là mock embedding dựa trên chuỗi/hash để phục vụ unit test, không phải model ngôn ngữ.

**Kết luận:**

- Mock phù hợp để kiểm tra interface, sort, filter và delete.
- Mock không phù hợp để đánh giá retrieval tiếng Việt.
- Benchmark strategy phải dùng local multilingual embedding hoặc backend thật.

## 4.2. Failure case 2 — Similarity cao nhưng không cùng ý định

Nguyễn Minh Thái ghi nhận cặp:

```text
“Sinh viên đăng ký học phần”
“Sinh viên xin giấy giới thiệu”
```

có similarity 0,7147, dù hai câu thuộc hai thủ tục khác nhau. Cả hai cùng chứa ngữ cảnh chung “sinh viên thực hiện thủ tục”, khiến embedding đặt gần nhau.

**Kết luận:** Dense retrieval có thể bị ảnh hưởng bởi chủ đề chung. Metadata, lexical features và reranker giúp phân biệt intent cụ thể.

## 4.3. Failure case 3 — Thuật ngữ nghiệp vụ gần nhau nhưng dense score chưa cao

Võ Hà Minh Huy ghi nhận “tạm dừng học tập” và “bảo lưu kết quả học tập” có score khoảng 0,53, thấp hơn paraphrase trực tiếp.

**Kết luận:** Embedding hiểu paraphrase từ vựng tốt hơn quan hệ nghiệp vụ ngầm. Đây là lý do cần BM25, metadata mapping và cấu trúc procedure.

## 4.4. Failure case 4 — Top-1 đúng chủ đề nhưng sai phần của quy trình

Ở câu 4 và câu 5, chiến lược Recursive dense-only có thể lấy đúng quy trình nhưng chọn nhầm bước:

- Câu 4: lấy đoạn nói về thời gian tạm dừng thay vì nhập học lại.
- Câu 5: lấy đoạn nhận bằng thay vì đoạn đăng ký và chuẩn bị hồ sơ.

**Kết luận:** Relevance phải xét ở mức “đúng thông tin trả lời”, không chỉ “đúng chủ đề”.

## 4.5. Failure case 5 — Chunk cắt mất từ điều kiện

Nếu chunk chỉ chứa “được đăng ký 30 tín chỉ” nhưng mất cụm “ĐTBC ≥ 8,0”, câu trả lời trở nên sai. Tương tự, nếu mất “không được phép ĐKHP mới”, agent có thể đưa ra hướng dẫn nguy hiểm.

**Kết luận:** Với tài liệu quy định, từ điều kiện quan trọng hơn việc chunk có vẻ đọc tự nhiên. Cần bảo toàn:

- Điều kiện.
- Ngoại lệ.
- Giới hạn.
- Thời hạn.
- Phủ định.

## 4.6. Failure case 6 — So sánh score khác thang đo

RRF score và cosine similarity không cùng ý nghĩa. RRF thường tạo số nhỏ vì score dựa trên thứ hạng, không phải góc giữa vector.

**Kết luận:** Không đánh giá strategy bằng việc nhìn score nào “to hơn”. Phải dùng metric retrieval thống nhất.

---

# 5. Tính tái lập và kiểm thử

## 5.1. Kết quả test của ba thành viên

| Thành viên | Python trong báo cáo | Kết quả |
|---|---|---:|
| Võ Hà Minh Huy | Python 3.11.9 | 42/42 passed |
| Đỗ Duy Đông | Python 3.13.3 | 42/42 passed |
| Nguyễn Minh Thái | Python 3.13.5 | 42/42 passed |

Cả ba đều hoàn thành core implementation. Tuy nhiên, README của Lab quy định Python 3.11 là môi trường chuẩn. Do đó, trước khi nộp nhóm nên chạy lại toàn bộ test trong Python 3.11 để bảo đảm tính tương thích chính thức.

## 5.2. Lệnh kiểm tra core

```bash
python -m pytest tests -v
```

Kết quả mong đợi:

```text
42 passed
```

## 5.3. Lệnh benchmark

Benchmark retrieval không cần gọi DeepSeek:

```bash
python bench.py --compare-strategies --skip-generation
```

Benchmark đầy đủ có generation:

```bash
python bench.py --compare-strategies
```

Kết quả được lưu tại:

```text
benchmark/results/latest.json
benchmark/results/latest.md
```

## 5.4. Điều kiện để so sánh công bằng

Khi so chunking/retrieval strategy, nhóm cần giữ cố định:

- Cùng corpus.
- Cùng năm query.
- Cùng gold answer.
- Cùng embedding model nếu mục tiêu là đo riêng chunking/retrieval.
- Cùng top-k.
- Cùng cách xác định relevant chunk.

Khi so embedding model, phải giữ cố định chunking và retrieval strategy. Không nên đồng thời đổi chunking, embedding và retrieval rồi kết luận toàn bộ cải thiện đến từ embedding.

---

# 6. Phân công và đóng góp thể hiện trong báo cáo cá nhân

| Thành viên | Đóng góp được thể hiện trong báo cáo |
|---|---|
| Võ Hà Minh Huy | Core implementation; StructureAwareChunker; hybrid retrieval; metadata filter; benchmark baseline/high-accuracy; DeepSeek generation; citation và diagnostic refusal |
| Đỗ Duy Đông | Core implementation; RecursiveChunker; EmbeddingStore; MiniLM local retrieval; chạy 42 tests; chạy năm query nhóm và phân tích top-3 |
| Nguyễn Minh Thái | Core implementation; RecursiveChunker; MiniLM 384 chiều; kiểm tra similarity; chạy 42 tests; tái lập năm query và phân tích các trường hợp top-1 chưa tối ưu |

Bảng này chỉ tổng hợp những gì xuất hiện trong ba báo cáo cá nhân. Nhóm không gán thêm phần việc không có bằng chứng trong tài liệu.

---

# 7. Kịch bản demo và thuyết trình — 5 điểm

## 7.1. Mục tiêu demo

Trong thời gian ngắn, nhóm cần chứng minh ba ý:

1. Chunking khác nhau tạo kết quả retrieval khác nhau.
2. Dense retrieval có thể đúng chủ đề nhưng sai chi tiết.
3. Structure-aware + hybrid giúp đưa đúng chunk lên top-1 và giảm hallucination.

## 7.2. Kịch bản demo đề xuất

### Bước 1 — Giới thiệu corpus

- Mở tài liệu UIT.
- Chỉ ra nhiều quy trình nằm trong cùng một file.
- Chỉ ra các cụm dễ bị cắt: “dưới 30%”, “không được phép ĐKHP mới”, “chậm nhất 1 tháng trước”.

### Bước 2 — So chunking

- Hiển thị một chunk FixedSize bị cắt giữa bước.
- Hiển thị chunk Recursive giữ đoạn tốt hơn.
- Hiển thị StructureAware chunk không trộn hai procedure.

### Bước 3 — Chạy câu hỏi khó

Dùng câu 4:

```text
Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào?
```

So sánh:

- Recursive dense-only: top-1 có thể là đoạn thời gian tạm dừng.
- Structure-aware hybrid: top-1 là đoạn nhập học lại, Mẫu 07/09, hạn 1 tháng.

### Bước 4 — Demo câu ngoài corpus

```text
Chi phí ký túc xá UIT hiện tại là bao nhiêu?
```

Kết quả đúng phải là từ chối dựa trên thiếu context.

### Bước 5 — Chốt metric

- Baseline Hit@1: 20%.
- High-accuracy Hit@1: 100%.
- Baseline Recall@5: 60%.
- High-accuracy Recall@5: 100%.
- Citation presence tăng từ 60% lên 100%.

## 7.3. Ba insight chính để trình bày

1. **Chất lượng dữ liệu và chunking quyết định chất lượng context.** LLM tốt không thể sửa hoàn toàn context sai hoặc thiếu.
2. **Dense và BM25 bổ trợ nhau.** Dense hiểu paraphrase; BM25 giữ số liệu, mã biểu mẫu và từ khóa nghiệp vụ.
3. **Metadata filter chỉ có lợi khi dùng đúng lúc.** Filter theo confidence giúp giảm nhiễu, nhưng filter cứng có thể làm mất đáp án.

---

# 8. Hạn chế của bài làm

1. Corpus hiện chỉ có một nguồn độc lập, dù có nhiều section logic.
2. Hai thành viên dùng cùng chiến lược Recursive + MiniLM, nên độ đa dạng strategy chưa tối đa.
3. Đông và Thái chạy test trên Python 3.13 thay vì chuẩn Python 3.11 của Lab.
4. Hai báo cáo Recursive không cung cấp đầy đủ Hit@1, MRR, procedure accuracy và citation metrics tự động; chỉ có top-3 5/5 và đánh giá top-1 theo từng câu.
5. Chưa có benchmark riêng để tách ảnh hưởng của BGE-M3 khỏi ảnh hưởng của StructureAware + hybrid retrieval.
6. Tài liệu crawl có thể mất một số heading gốc, nên procedure detection phụ thuộc anchor mapping.
7. Chưa có kết quả reranker riêng để đo mức cải thiện và chi phí latency.

---

# 9. Hướng cải tiến

## 9.1. Trước khi nộp

- Bổ sung MSSV của Đỗ Duy Đông và Nguyễn Minh Thái.
- Chạy lại `42/42 tests` trên Python 3.11.
- Cho một thành viên chạy thêm SentenceChunker hoặc FixedSize để có ba strategy thực sự khác nhau.
- Kiểm tra `sources.csv` và URL nguồn.
- Bảo đảm `benchmark/results/latest.md` khớp với số liệu trong báo cáo.
- Kiểm tra Mermaid trong `docs/ARCHITECTURE_FLOW.md` render được.

## 9.2. Nếu có thêm thời gian

- Bổ sung 4–9 tài liệu UIT độc lập.
- Thêm tài liệu học phí, học bổng, thư viện và ký túc xá.
- Tạo 20–50 query để metric ổn định hơn.
- Thử reranker local trên top-10.
- Đo p95 latency và dung lượng index.
- So sánh BGE-M3 với MiniLM khi giữ nguyên chunking và hybrid retrieval.
- Thêm query khó có phủ định, ngoại lệ và nhiều điều kiện.

---

# 10. Kết luận

Nhóm đã hoàn thành phần core của Lab với 42/42 tests ở cả ba thành viên và xây dựng được một benchmark chung gồm năm câu hỏi có gold answer rõ ràng.

Kết quả thực nghiệm cho thấy `RecursiveChunker + MiniLM + dense retrieval` là baseline thực tế tốt hơn mock embedding và có thể đưa chunk liên quan vào top-3 cho cả năm câu. Tuy nhiên, top-1 vẫn có thể chỉ đúng một phần hoặc chọn sai bước trong cùng quy trình.

Chiến lược `StructureAwareChunker + BGE-M3 + dense + BM25 + RRF + metadata + adjacent expansion` đạt kết quả tốt nhất trên corpus hiện tại. Điểm cải thiện không chỉ đến từ embedding model, mà chủ yếu từ việc:

- Tách đúng ranh giới quy trình.
- Bảo toàn điều kiện và số liệu.
- Kết hợp semantic với lexical retrieval.
- Lọc metadata có ngưỡng.
- Mở rộng chunk liền kề.
- Bắt DeepSeek chỉ trả lời từ context và đưa citation.

Bài học quan trọng nhất của nhóm là:

> Trong hệ thống RAG, LLM chỉ là bước cuối. Nếu tài liệu bị chia sai hoặc retrieval lấy sai đoạn, câu trả lời vẫn có thể sai dù LLM mạnh. Chất lượng dữ liệu, chunking và retrieval strategy quyết định mức độ grounded của toàn hệ thống.

---

# Tự đánh giá phần nhóm

| Tiêu chí | Điểm tự đánh giá | Giải thích |
|---|---:|---|
| Lựa chọn tài liệu | 8/10 | Nguồn UIT thật, metadata tốt, nội dung giàu điều kiện; nhưng mới có một nguồn độc lập |
| Thiết kế chiến lược | 13/15 | Có baseline, Recursive và high-accuracy; hai thành viên vẫn trùng strategy |
| Chất lượng truy xuất | 10/10 | Strategy tốt nhất đạt 5/5 top-1, citation đầy đủ và refusal đúng |
| Thuyết trình và phân tích | 4/5 | Có metric, failure case và demo flow; điểm cuối còn phụ thuộc phần trình bày thực tế |
| **Tổng** | **35/40** | Mức tự đánh giá có nêu rõ hạn chế, không làm tròn thành tích |

---

# Checklist trước khi nộp

- [ ] Bổ sung MSSV Đỗ Duy Đông.
- [ ] Bổ sung MSSV Nguyễn Minh Thái.
- [ ] Kiểm tra tên file corpus đúng với repo.
- [ ] Kiểm tra URL trong `sources.csv`.
- [ ] Chạy `python -m pytest tests -v` bằng Python 3.11.
- [ ] Xác nhận kết quả là 42 passed.
- [ ] Chạy benchmark retrieval-only.
- [ ] Chạy benchmark có DeepSeek nếu API key hợp lệ.
- [ ] Đối chiếu `latest.md` với bảng metric trong báo cáo.
- [ ] Không commit API key, cache model hoặc index lớn.
- [ ] Kiểm tra REPORT_CANHAN của ba thành viên dùng đúng năm query chung.
- [ ] Commit `REPORT_NHOM.md` cuối cùng lên repo nhóm.
