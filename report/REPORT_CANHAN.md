# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Nguyễn Minh Thái
**Nhóm:** NhomGiCungDuoc
**Ngày:** 03/08/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

Độ tương tự cosine cao cho thấy hai vector embedding có hướng gần nhau, nghĩa là hai câu có nội dung hoặc ngữ nghĩa tương đối giống nhau. Giá trị càng gần 1 thì mức độ tương đồng về ngữ nghĩa càng cao.

**Ví dụ có độ tương tự CAO:**

* Câu A: `Sinh viên đăng ký học phần`
* Câu B: `Sinh viên thực hiện đăng ký môn học`
* Tại sao tương đồng: Hai câu đều nói về việc sinh viên đăng ký học phần/môn học và có cùng ngữ cảnh.

**Ví dụ có độ tương tự THẤP:**

* Câu A: `Đăng ký học phần`
* Câu B: `Thời tiết hôm nay rất đẹp`
* Tại sao khác: Hai câu thuộc hai chủ đề hoàn toàn khác nhau. Câu thứ nhất nói về hoạt động học tập, trong khi câu thứ hai nói về thời tiết.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**

Cosine similarity tập trung vào hướng của vector thay vì độ lớn của vector, phù hợp với việc so sánh ngữ nghĩa của text embeddings. Vì vậy, các câu có ý nghĩa gần nhau vẫn có thể được đánh giá là tương đồng ngay cả khi độ dài hoặc độ lớn vector khác nhau.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

**Trình bày phép tính:**

Bước dịch chuyển giữa hai chunk:

```text
step = chunk_size - overlap
     = 500 - 50
     = 450
```

Số chunk:

```text
chunks = ceil((10,000 - 500) / 450) + 1
       = ceil(9,500 / 450) + 1
       = 22 + 1
       = 23
```

**Đáp án: 23 chunks.**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**

Khi overlap tăng lên 100:

```text
step = 500 - 100 = 400

chunks = ceil((10,000 - 500) / 400) + 1
       = ceil(9,500 / 400) + 1
       = 24 + 1
       = 25
```

Số chunk tăng từ **23 lên 25**. Overlap lớn giúp giữ lại nhiều ngữ cảnh ở ranh giới giữa hai chunk, nhưng đồng thời làm tăng số lượng chunk và chi phí embedding cũng như retrieval.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk` — hướng tiếp cận:**

`SentenceChunker.chunk` sử dụng biểu thức chính quy để nhận diện các ranh giới câu dựa trên các dấu kết thúc câu như `.`, `!`, `?` và các dòng văn bản. Sau đó, các câu được gom lại thành từng chunk theo số lượng câu tối đa được cấu hình. Trường hợp văn bản rỗng hoặc không có đủ dấu câu cũng được xử lý để hàm vẫn trả về danh sách chunk hợp lệ.

**`RecursiveChunker.chunk / _split` — hướng tiếp cận:**

`RecursiveChunker` chia văn bản theo nhiều cấp độ separator, từ các ranh giới lớn như đoạn văn và xuống dòng đến khoảng trắng khi cần thiết. Hàm `_split` tiếp tục chia nhỏ đoạn văn nếu kích thước vẫn vượt quá `chunk_size`. Trường hợp cơ sở là khi đoạn văn đã nằm trong kích thước cho phép hoặc không còn separator phù hợp thì giữ lại phần văn bản hiện tại.

### Lớp EmbeddingStore

**`add_documents` + `search` — hướng tiếp cận:**

`add_documents` chuyển mỗi `Document` thành một record gồm `id`, `content`, `metadata` và embedding rồi lưu vào vector store. Khi tìm kiếm, hệ thống tạo embedding cho câu hỏi và tính độ tương đồng giữa query embedding với embedding của các chunk. Các kết quả được sắp xếp theo score giảm dần và trả về tối đa `top_k` kết quả.

Trong phần thực nghiệm, tôi sử dụng local embedding:

```text
Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Embedding dimension: 384
Provider: local
```

### `search_with_filter` + `delete_document` — hướng tiếp cận:

`search_with_filter` thực hiện lọc metadata trước khi tính similarity, giúp giới hạn phạm vi tìm kiếm vào những chunk phù hợp với điều kiện metadata. `delete_document` dựa trên `doc_id` để xác định và xóa toàn bộ các chunk thuộc cùng một tài liệu khỏi store.

Cách tiếp cận này giúp vector store không chỉ tìm kiếm theo ngữ nghĩa mà còn hỗ trợ quản lý và truy xuất dữ liệu dựa trên metadata.

### Tác tử KnowledgeBaseAgent

**`answer` — hướng tiếp cận:**

`KnowledgeBaseAgent.answer` trước tiên gửi câu hỏi vào `EmbeddingStore.search` để lấy các chunk liên quan nhất. Nội dung của các chunk được ghép thành `context` và đưa vào prompt cùng với câu hỏi trước khi gọi hàm LLM.

Pipeline được triển khai theo mô hình RAG:

```text
Question
   ↓
EmbeddingStore.search()
   ↓
Top-k relevant chunks
   ↓
Build context
   ↓
Inject context into prompt
   ↓
LLM
   ↓
Answer
```

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

Đã chạy:

```powershell
pytest tests/ -v
```

Kết quả:

```text
========================= test session starts ==========================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
collected 42 items

42 passed in 0.25s
```

**Số lượng bài test vượt qua (pass): 42 / 42**

**Tỷ lệ pass: 100%.**

Các nhóm chức năng chính đều vượt qua kiểm thử, bao gồm:

* Project structure và class interfaces.
* `FixedSizeChunker`.
* `SentenceChunker`.
* `RecursiveChunker`.
* `EmbeddingStore`.
* Similarity computation.
* `search_with_filter`.
* `delete_document`.
* `KnowledgeBaseAgent`.
* So sánh các chiến lược chunking.

Kết quả này cho thấy phần implementation trong `src` đáp ứng đầy đủ các yêu cầu của bộ test được cung cấp cho Lab 7.

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Tôi sử dụng local embedding model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

để đo cosine similarity giữa các cặp câu.

| Cặp | Câu A                      | Câu B                                  | Dự đoán | Điểm thực tế | Đúng? |
| --- | -------------------------- | -------------------------------------- | ------- | -----------: | ----- |
| 1   | Sinh viên đăng ký học phần | Sinh viên thực hiện đăng ký môn học    | Cao     |       0.9282 | Có    |
| 2   | Sinh viên đăng ký học phần | Sinh viên xin giấy giới thiệu          | Thấp    |       0.7147 | Không |
| 3   | Học phí học kỳ             | Mức học phí cần đóng trong học kỳ      | Cao     |       0.8632 | Có    |
| 4   | Đăng ký học phần           | Thời tiết hôm nay rất đẹp              | Thấp    |       0.0697 | Có    |
| 5   | Sinh viên cần làm thủ tục  | Sinh viên thực hiện thủ tục hành chính | Cao     |       0.7638 | Có    |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

Kết quả bất ngờ nhất là cặp “Sinh viên đăng ký học phần” và “Sinh viên xin giấy giới thiệu” đạt similarity **0.7147**, mặc dù hai câu không hoàn toàn cùng chủ đề. Điều này cho thấy embedding không chỉ dựa trên việc hai câu có cùng từ khóa mà còn biểu diễn ngữ nghĩa và ngữ cảnh chung, chẳng hạn cùng liên quan đến hoạt động của sinh viên. Vì vậy, similarity cao không phải lúc nào cũng có nghĩa hai câu hoàn toàn tương đương về nội dung.

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Tôi chạy 5 câu hỏi đánh giá trên mã nguồn cá nhân trong gói `src`.

Cấu hình thực nghiệm:

```text
Embedding provider: local
Embedding model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
Embedding dimension: 384
Chunking strategy: RecursiveChunker
Dataset: data/uit/quy-trinh-danh-cho-sinh-vien.md
Number of chunks: 42
```

| # | Câu hỏi (Query)                                                                                              | Top-1 Chunk truy xuất được (tóm tắt)                                                                                                                      | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt)                                                                                                                                                                                                                                                                                                           |
| - | ------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------: | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1 | Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ? Khi nào được đăng ký 30 tín chỉ?           | Chỉ được đăng ký tối đa 24 tín chỉ/học kỳ, bao gồm các học phần đăng ký học lại và học cải thiện. Sinh viên có ĐTBC ≥ 8.0 được đăng ký tối đa 30 tín chỉ. |     0.6843 | Một phần                       | Sinh viên thường được đăng ký tối đa 24 tín chỉ/học kỳ, bao gồm học lại và học cải thiện. Nếu ĐTBC từ 8.0 trở lên tại thời điểm đăng ký thì được đăng ký tối đa 30 tín chỉ.                                                                                                                                                               |
| 2 | Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi những gì và có được đăng ký môn mới không?       | Sinh viên chọn “Loại” (“Đăng ký” hoặc “Hủy đăng ký”) và “Mã lớp” tương ứng với loại đăng ký hoặc hủy đăng ký.                                             |     0.6624 | Có                             | Trong đợt cứu xét, sinh viên chỉ được sửa đổi/điều chỉnh nhỏ dưới 30% các môn đã đăng ký ở đợt 1 và đợt 2, chủ yếu cho học lại, cải thiện hoặc điều chỉnh. Sinh viên không được đăng ký học phần mới trong đợt này.                                                                                                                       |
| 3 | Sinh viên xin tạm dừng học tập vì lý do cá nhân phải đáp ứng điều kiện gì và được tạm dừng tối đa bao lâu?   | Thời gian tạm dừng học tập từ 01 đến tối đa 02 học kỳ chính liên tiếp, được tính vào thời gian tối đa hoàn thành khóa học.                                |     0.6839 | Có                             | Nếu xin tạm dừng vì lý do cá nhân/lý do khác, sinh viên phải học ít nhất 1 học kỳ tại Trường và không thuộc trường hợp bị đình chỉ học tập. Thời gian tạm dừng từ 01 đến tối đa 02 học kỳ chính liên tiếp.                                                                                                                                |
| 4 | Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào?                                  | Thời gian tạm dừng học tập từ 01 đến tối đa 02 học kỳ chính liên tiếp, được tính vào thời gian tối đa hoàn thành khóa học.                                |     0.6781 | Một phần                       | Sau khi hết thời hạn bảo lưu, sinh viên phải làm đơn xin nhập học lại và nộp tại P.ĐTĐH. Hạn chót là chậm nhất 1 tháng trước khi bắt đầu học kỳ mới.                                                                                                                                                                                      |
| 5 | Để đăng ký xét tốt nghiệp, sinh viên cần đóng khoản gì, chuẩn bị hồ sơ nào và hoàn thành những nghĩa vụ nào? | Sinh viên nhận bằng tốt nghiệp tại P.ĐTĐH/VPCCTĐB theo thông báo kế hoạch tổ chức lễ tốt nghiệp của Trường.                                               |     0.6719 | Có                             | Sinh viên cần đóng lệ phí xét cấp bằng tốt nghiệp, chuẩn bị hồ sơ gồm scan bằng tốt nghiệp THPT, scan giấy khai sinh và chứng chỉ ngoại ngữ dùng xét chuẩn đầu ra. Ngoài ra phải có chứng chỉ Giáo dục Quốc phòng & An ninh, hoàn thành các khoản nợ như nợ sách thư viện, nợ tiền giấy xác nhận và kiểm tra chính xác thông tin cá nhân. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** **5 / 5**

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**

Tôi nhận thấy chất lượng retrieval phụ thuộc nhiều vào cách chia chunk và lựa chọn embedding backend. Khi sử dụng local embedding đa ngữ `paraphrase-multilingual-MiniLM-L12-v2`, các kết quả có ý nghĩa ngữ nghĩa tốt hơn so với mock embedding. Tuy nhiên, vẫn có trường hợp top-1 chưa phải chunk phù hợp nhất, vì vậy cần xem xét cả top-3 kết quả và phân tích lỗi retrieval thay vì chỉ dựa vào kết quả top-1.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí                                        | Điểm tự đánh giá |
| ----------------------------------------------- | ---------------: |
| Khởi động (Warm-up)                             |        **5 / 5** |
| Hướng tiếp cận của tôi (My Approach)            |      **10 / 10** |
| Hoàn thiện code (Core Implementation — tests)   |      **30 / 30** |
| Dự đoán độ tương tự (Similarity Predictions)    |        **5 / 5** |
| Kết quả truy xuất của tôi (Competition Results) |      **10 / 10** |
| **Tổng phần cá nhân**                           |      **60 / 60** |

### Tổng kết cá nhân

Qua Lab 7, tôi hiểu rõ hơn quy trình xây dựng một hệ thống retrieval dựa trên embedding, từ bước chia nhỏ tài liệu, tạo embedding, lưu trữ vector, tìm kiếm theo similarity đến việc đưa các chunk liên quan vào context cho RAG. Tôi cũng nhận thấy việc lựa chọn chunking strategy và embedding model có ảnh hưởng trực tiếp đến chất lượng retrieval. Đặc biệt, kết quả thực nghiệm cho thấy local multilingual embedding phù hợp với dữ liệu tiếng Việt và có khả năng biểu diễn sự tương đồng về ngữ nghĩa tốt hơn mock embedding.
