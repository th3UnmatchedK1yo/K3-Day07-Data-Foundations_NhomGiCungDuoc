# Báo Cáo Cá Nhân - Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

> **Nộp 1 bản / sinh viên.** Phần nhóm nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) - Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**

Độ tương tự cosine cao nghĩa là hai vector embedding có hướng gần giống nhau trong không gian vector. Với văn bản, điều này thường cho thấy hai câu/đoạn đang nói về nội dung hoặc ý nghĩa tương tự nhau, dù có thể dùng từ ngữ khác nhau. Score càng cao thì mức độ tương đồng về ý nghĩa càng lớn.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên cần đăng ký học phần trước thời hạn của trường.
- Câu B: Người học phải hoàn tất đăng ký môn học đúng hạn.
- Tại sao tương đồng: Hai câu đều nói về việc sinh viên/người học phải đăng ký học phần/môn học trước hạn.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Thư viện cho phép sinh viên mượn sách trong 14 ngày.
- Câu B: Căn tin phục vụ bữa trưa từ 11 giờ đến 13 giờ.
- Tại sao khác: Hai câu nói về hai chủ đề khác nhau: một câu về dịch vụ thư viện, câu còn lại về căn tin.

**Tại sao độ tương tự cosine được ưu tiên hơn khoảng cách Euclid cho text embeddings?**

Cosine similarity tập trung vào hướng của vector, tức là tập trung vào ý nghĩa tương đối của văn bản hơn là độ lớn tuyệt đối của vector. Với text embeddings, hai câu có cùng ý nghĩa có thể có độ dài hoặc cường độ vector khác nhau, nên cosine thường phù hợp hơn Euclidean distance.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**

Công thức:

```text
số lượng chunk = ceil((độ_dài_tài_liệu - overlap) / (chunk_size - overlap))
```

Thay số:

```text
ceil((10000 - 50) / (500 - 50))
= ceil(9950 / 450)
= ceil(22.11)
= 23 chunks
```

**Đáp án:** 23 chunks.

**Nếu overlap tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**

Khi `overlap=100`:

```text
ceil((10000 - 100) / (500 - 100))
= ceil(9900 / 400)
= ceil(24.75)
= 25 chunks
```

Số chunk tăng từ 23 lên 25 vì bước nhảy giữa hai chunk nhỏ hơn. Tăng overlap giúp giữ lại ngữ cảnh ở ranh giới giữa các chunk, giảm khả năng một ý quan trọng bị cắt đôi giữa hai đoạn.

---

## 2. Hướng tiếp cận của tôi (My Approach) - Cá nhân (10 điểm)

Giải thích cách tiếp cận khi lập trình các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk` - hướng tiếp cận:**

Em dùng regex `(?<=[.!?])\s+` để tách văn bản tại khoảng trắng đứng sau các dấu kết thúc câu như `.`, `!`, `?`. Sau khi tách, Em loại bỏ khoảng trắng thừa và nhóm các câu lại theo `max_sentences_per_chunk`. Trường hợp văn bản rỗng thì trả về danh sách rỗng để tránh tạo chunk không có nội dung.

**`RecursiveChunker.chunk` / `_split` - hướng tiếp cận:**

Em ưu tiên dùng `RecursiveCharacterTextSplitter` của LangChain để chia văn bản theo danh sách separator `['\n\n', '\n', '. ', ' ', '']`. Thuật toán thử separator lớn trước để giữ cấu trúc đoạn/câu, nếu phần nào vẫn quá dài thì tiếp tục chia bằng separator nhỏ hơn. base case là khi đoạn hiện tại đã có độ dài nhỏ hơn hoặc bằng `chunk_size`.

### Lớp EmbeddingStore

**`add_documents` + `search` - hướng tiếp cận:**

Trong `add_documents`, mỗi `Document` được chuyển thành một record gồm `id`, `content`, `metadata`, `embedding` và thứ tự thêm vào store. Embedding được tạo bằng `embedding_fn`, mặc định là `_mock_embed` để test không cần API key. Trong `search`, tôi nhúng câu truy vấn rồi tính dot product giữa query embedding và embedding của từng document, sau đó sắp xếp giảm dần theo score và lấy `top_k` kết quả.

**`search_with_filter` + `delete_document` - hướng tiếp cận:**

Với `search_with_filter`, tôi lọc metadata trước rồi mới tính similarity để truy xuất trên tập ứng viên nhỏ và đúng ngữ cảnh hơn. Điều kiện lọc yêu cầu các cặp `key/value` trong `metadata_filter` phải khớp với metadata của record. Với `delete_document`, tôi xóa tất cả record có `id` bằng `doc_id` hoặc có `metadata['doc_id']` bằng `doc_id`, phù hợp với dữ liệu đã được chunk từ một tài liệu gốc.

### Tác tử KnowledgeBaseAgent

**`answer` - hướng tiếp cận:**

Trong `answer`, agent lấy `top_k` chunk liên quan nhất từ `EmbeddingStore.search`. Sau đó tôi tạo prompt gồm phần hướng dẫn, danh sách context chunk kèm score/source, câu hỏi của người dùng và nhãn `Answer:`. Cuối cùng prompt được truyền vào `llm_fn` để sinh câu trả lời dựa trên ngữ cảnh truy xuất được.

---

## 3. Hoàn thiện code (Core Implementation) - Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```text
=============================== test session starts ================================
platform win32 -- Python 3.13.3, pytest-9.1.1, pluggy-1.6.0 -- D:\VinAI\Day07-Lab\K3-Day07-Data-Foundations\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\VinAI\Day07-Lab\K3-Day07-Data-Foundations
plugins: anyio-4.14.2, langsmith-0.10.15
collected 42 items                                                                  

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED        [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

================================ 42 passed in 0.42s ================================
```

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) - Cá nhân (5 điểm)

Tôi dùng `_mock_embed` của repo để tạo vector cho từng câu, sau đó gọi `compute_similarity()` để tính cosine similarity. Vì `_mock_embed` là embedding giả lập dựa trên hash chuỗi, điểm số có tính ổn định khi chạy lại nhưng không phản ánh tốt ý nghĩa ngôn ngữ như embedding thật.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên cần đăng ký học phần đúng thời hạn. | Người học phải hoàn tất đăng ký môn học trước hạn. | cao | 0.1824 | Đúng một phần |
| 2 | Thư viện cho phép mượn sách trong 14 ngày. | Sinh viên có thể gia hạn tài liệu thư viện trực tuyến. | cao | 0.0175 | Sai |
| 3 | Học phí phải được thanh toán qua cổng thông tin sinh viên. | Căn tin phục vụ bữa trưa từ 11 giờ. | thấp | 0.1535 | Sai |
| 4 | Ký túc xá ưu tiên sinh viên năm nhất ở xa. | Chính sách học bổng xét theo kết quả học tập. | thấp | -0.2028 | Đúng |
| 5 | Python là một ngôn ngữ lập trình phổ biến. | Python is a popular programming language. | cao | 0.0579 | Sai |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**

Kết quả bất ngờ nhất là cặp 5, vì hai câu có cùng ý nghĩa nhưng khác ngôn ngữ lại chỉ đạt 0.0579 với `_mock_embed`. Điều này cho thấy mock embedding chỉ phù hợp để kiểm thử code chạy đúng, không phù hợp để đánh giá chất lượng ngữ nghĩa. Nếu dùng embedding thật như multilingual sentence-transformers, tôi kỳ vọng các cặp cùng ý nghĩa sẽ có điểm cao hơn rõ rệt.

---

## 5. Kết quả truy xuất của tôi (Competition Results) - Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

Tôi chạy với `EMBEDDING_PROVIDER=local`, dùng model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, dữ liệu `data/quy-trinh-danh-cho-sinh-vien.md` và `RecursiveChunker`. Store nạp được 42 chunk.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ? Khi nào được đăng ký 30 tín chỉ | Chỉ được đăng ký tối đa 24 tín chỉ/ học kỳ (bao gồm cả các học phần đăng ký học lại và học cải thiện). SV có ĐTBC ≥ 8. | 0.6843 | Một phần | Sinh viên thường được đăng ký tối đa 24 tín chỉ/học kỳ, bao gồm học lại và học cải thiện. Nếu ĐTBC từ 8.0 trở lên tại thời điểm đăng ký thì được đăng ký tối đa 30 tín chỉ. |
| 2 | Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi những gì và có được đăng ký môn mới không | Sinh viên chọn “Loại” (“Đăng ký” hoặc “Hủy đăng ký”), “Mã lớp” tương ứng với loại đăng ký hoặc hủy đăng ký (chú ... | 0.6624 | Có | Trong đợt cứu xét, sinh viên chỉ được sửa đổi/điều chỉnh nhỏ dưới 30% các môn đã đăng ký ở đợt 1 và đợt 2, chủ yếu cho học lại, cải thiện hoặc điều chỉnh. Sinh viên không được đăng ký học phần mới trong đợt này. |
| 3 | Sinh viên xin tạm dừng học tập vì lý do cá nhân phải đáp ứng điều kiện gì và được tạm dừng tối đa bao lâu | Thời gian tạm dừng học tập từ 01 đến tối đa 02 học kỳ chính liên tiếp, được tính vào thời gian tối đa hoàn thành khóa họ... | 0.6839 | Có | Nếu xin tạm dừng vì lý do cá nhân/lý do khác, sinh viên phải học ít nhất 1 học kỳ tại Trường và không thuộc trường hợp bị đình chỉ học tập. Thời gian tạm dừng từ 01 đến tối đa 02 học kỳ chính liên tiếp. |
| 4 | Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào | Thời gian tạm dừng học tập từ 01 đến tối đa 02 học kỳ chính liên tiếp, được tính vào thời gian tối đa hoàn thành khóa họ... | 0.6781 | Một phần | Sau khi hết thời hạn bảo lưu, sinh viên phải làm đơn xin nhập học lại và nộp tại P.ĐTĐH. Hạn chót là chậm nhất 1 tháng trước khi bắt đầu học kỳ mới. |
| 5 | Để đăng ký xét tốt nghiệp, sinh viên cần đóng khoản gì, chuẩn bị hồ sơ nào và hoàn thành những nghĩa vụ nào | Sinh viên nhận bằng tốt nghiệp tại P.ĐTĐH/VPCCTĐB theo thông báo kế hoạch tổ chức lễ tốt nghiệp của Trường. ... | 0.6719 | Có | Sinh viên cần đóng lệ phí xét cấp bằng tốt nghiệp, chuẩn bị hồ sơ gồm scan bằng tốt nghiệp THPT, scan giấy khai sinh và chứng chỉ ngoại ngữ dùng xét chuẩn đầu ra. Ngoài ra phải có chứng chỉ Giáo dục Quốc phòng & An ninh, hoàn thành các khoản nợ như nợ sách thư viện, nợ tiền giấy xác nhận và kiểm tra chính xác thông tin cá nhân. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> Tôi thấy chất lượng retrieval phụ thuộc rất nhiều vào cách chia chunk và embedding backend. Khi dùng local embedding đa ngữ, kết quả có ý nghĩa hơn mock embedding, nhưng vẫn có trường hợp top-1 chưa đúng nhất nên cần đọc cả top-3 và phân tích lỗi retrieval.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) |5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10/ 10 |
| Hoàn thiện code (Core Implementation - tests) | 30/ 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5/ 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10/ 10 |
| **Tổng phần cá nhân** | **60/ 60** |
