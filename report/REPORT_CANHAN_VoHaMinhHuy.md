# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Võ Hà Minh Huy (2A202601373)
**Nhóm:** Nhóm Gì Cũng Được (`NhomGiCungDuoc`)
**Ngày:** 2026-08-03

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> Hai vector embedding hướng gần giống nhau trong không gian vector (góc giữa chúng nhỏ), nên nội dung hai đoạn văn bản thường cùng chủ đề/cùng ý nghĩa dù câu chữ có thể khác nhau.

**Ví dụ có độ tương tự CAO:**
- Câu A: Sinh viên được đăng ký tối đa 24 tín chỉ mỗi học kỳ.
- Câu B: Mỗi học kỳ sinh viên đăng ký không quá 24 tín chỉ.
- Tại sao tương đồng: Cùng nói về giới hạn tín chỉ đăng ký trong một học kỳ; chỉ khác cách diễn đạt.

**Ví dụ có độ tương tự THẤP:**
- Câu A: Sinh viên được đăng ký tối đa 24 tín chỉ mỗi học kỳ.
- Câu B: Trường tổ chức xét tốt nghiệp mỗi năm 04 đợt.
- Tại sao khác: Một câu về ĐKHP/tín chỉ, một câu về lịch xét tốt nghiệp — chủ đề khác nhau nên vector lệch hướng.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> Cosine chỉ quan tâm hướng (góc) của vector, ít bị ảnh hưởng bởi độ lớn/độ dài văn bản; còn khoảng cách Euclid phụ thuộc cả độ lớn nên hai câu cùng nghĩa nhưng độ dài khác có thể bị xem là “xa” hơn mức cần thiết.

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:*  
> Công thức: `ceil((độ_dài - overlap) / (chunk_size - overlap))`  
> `= ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.111...) = 23`
> *Đáp án:* **23 chunks**

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> Số chunk tăng: `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25` (tăng từ 23 lên 25). Overlap lớn hơn giúp giữ ngữ cảnh ở ranh giới chunk (ví dụ điều kiện “tối đa 24 tín chỉ” không bị tách khỏi câu giải thích ngay sau), nên retrieval ít mất thông tin quan trọng hơn dù tốn thêm bộ nhớ/index.

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> Dùng regex `(?<=[.!?])\s+` để tách câu theo dấu kết thúc câu rồi nhóm tối đa `max_sentences_per_chunk` câu thành một chunk bằng `range(0, n, max)`. Edge case: chuỗi rỗng/`None`-like trả về `[]`; câu không có dấu vẫn được giữ nguyên sau khi `strip`; mỗi nhóm được `join` bằng khoảng trắng và `strip` lại trước khi trả về.

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> Thử lần lượt các separator `["\n\n", "\n", ". ", " ", ""]`. Base case: text ngắn hơn `chunk_size` thì trả về nguyên văn; nếu còn separator rỗng hoặc hết separator thì cắt cứng theo độ dài. Khi một phần vẫn quá dài sau khi tách bằng separator hiện tại thì đệ quy với danh sách separator còn lại; các phần vừa đủ được ghép lại cho đến khi vượt `chunk_size` rồi flush thành chunk mới.

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> Mỗi `Document` được embed bằng `embedding_fn` rồi lưu thành record `{id, content, metadata, embedding}` — ưu tiên ChromaDB nếu import được, không thì append vào list in-memory. `search` embed query rồi tính độ tương tự bằng dot product (embedding đã normalize nên tương đương cosine), sắp xếp giảm dần theo score và cắt `top_k`.

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> Filter **trước** rồi mới search: giữ các record có metadata khớp mọi cặp key/value trong `metadata_filter`, sau đó mới chạy similarity trên tập đã lọc. `delete_document` xóa mọi chunk thuộc `doc_id` bằng cách so `metadata['doc_id']` (fallback `record['id']` cho document không qua ingest); trên Chroma gọi `collection.delete(ids=...)`, trên in-memory rebuild list đã lọc.

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> Gọi `store.search(question, top_k)` lấy context, ghép các chunk thành danh sách gạch đầu dòng trong prompt dạng “Use the context below… If the context does not contain the answer, say so.” rồi truyền prompt đó vào `llm_fn`. Không tự sinh câu trả lời trong agent — chỉ retrieve + dựng prompt + ủy quyền LLM.

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::... PASSED (7 tests)
tests/test_solution.py::TestSentenceChunker::... PASSED (4 tests)
tests/test_solution.py::TestRecursiveChunker::... PASSED (4 tests)
tests/test_solution.py::TestEmbeddingStore::... PASSED (8 tests)
tests/test_solution.py::TestKnowledgeBaseAgent::... PASSED (2 tests)
tests/test_solution.py::TestComputeSimilarity::... PASSED (4 tests)
tests/test_solution.py::TestCompareChunkingStrategies::... PASSED (3 tests)
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::... PASSED (3 tests)
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::... PASSED (3 tests)

============================= 42 passed in 0.07s ==============================
```

Lệnh chạy: `python -m pytest tests -v` (Python 3.11).

**Số lượng bài test vượt qua (pass):** 42 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

Embedding dùng: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (local, không dùng mock). Score = `compute_similarity(embed(A), embed(B))`.

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Sinh viên được đăng ký tối đa 24 tín chỉ mỗi học kỳ. | Mỗi học kỳ sinh viên đăng ký không quá 24 tín chỉ. | cao | 0.8763 | Có |
| 2 | Sinh viên được đăng ký tối đa 24 tín chỉ mỗi học kỳ. | Trường tổ chức xét tốt nghiệp mỗi năm 04 đợt. | thấp | 0.3735 | Có |
| 3 | Phúc khảo điểm thi cuối kỳ. | Xin xem lại điểm bài thi cuối học kỳ. | cao | 0.8210 | Có |
| 4 | Tạm dừng học tập vì lý do cá nhân tối đa 02 học kỳ. | Bảo lưu kết quả học tập khi nghỉ học tạm thời. | cao | 0.5301 | Một phần (cao hơn cặp 2/5 nhưng thấp hơn kỳ vọng “cao rõ”) |
| 5 | Chi phí ký túc xá UIT hiện tại là bao nhiêu? | Sinh viên nộp lệ phí xét cấp bằng tốt nghiệp. | thấp | 0.3730 | Có |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> Cặp 4 bất ngờ nhất: về mặt nghiệp vụ UIT “tạm dừng học tập” và “bảo lưu” rất gần nhau, nhưng score chỉ ~0.53 — thấp hơn cặp paraphrase tường minh (cặp 1, 3). Embedding đa ngữ nắm được paraphrase gần từ vựng tốt, nhưng chưa chắc “hiểu” quan hệ thuật ngữ hành chính trong cùng một quy trình nếu câu chữ không overlap mạnh. Vì vậy chỉ dựa dense retrieval dễ miss; cần bổ sung BM25/metadata (đúng hướng chiến lược high_accuracy của nhóm).

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên chiến lược cá nhân của tôi: **StructureAwareChunker + hybrid retrieval (dense + BM25 + RRF + metadata filter + adjacent expansion) + DeepSeek generation**, corpus `data/uit/quy-trinh-danh-cho-sinh-vien.md`. Embedding: `BAAI/bge-m3`. Không dùng mock. Kết quả lấy từ lần chạy thật `python bench.py --compare-strategies` (có generation, không chỉ retrieval) — xem `benchmark/results/latest.md`.

| # | Câu hỏi (Query) | Top-1 Chunk truy xuất được (tóm tắt) | Điểm Score | Có liên quan không? (Relevant) | Câu trả lời của Agent (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ? Khi nào được đăng ký 30 tín chỉ? | `dang_ky_hoc_phan::chunk_001` — quy định lớp mở / ĐKHP / giới hạn tín chỉ | 0.0328 (RRF) | Có (top-1) | Tối đa **24 tín chỉ/học kỳ**; được **30 tín chỉ** nếu **ĐTBC ≥ 8,0**. Có citation. |
| 2 | Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi những gì và có được đăng ký môn mới không? | `dang_ky_hoc_phan::chunk_001` — đợt cứu xét / học lại / cải thiện | 0.0328 (RRF) | Có (top-1) | Chỉ xử lý học lại, cải thiện, sửa đổi dưới 30%; **không được ĐKHP mới**. Có citation. |
| 3 | Sinh viên xin tạm dừng học tập vì lý do cá nhân phải đáp ứng điều kiện gì và được tạm dừng tối đa bao lâu? | `tam_dung_hoc_tap::chunk_000` — điều kiện tạm dừng & thời hạn | 0.0328 (RRF) | Có (top-1) | Đã học ≥1 HK, không bị đình chỉ; tạm dừng **01–02 học kỳ chính liên tiếp**. Có citation. |
| 4 | Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào? | `tam_dung_hoc_tap::chunk_000` — nhập học lại sau bảo lưu | 0.0328 (RRF) | Có (top-1) | Nộp đơn nhập học lại (**Mẫu 07/09**), **chậm nhất 1 tháng trước** học kỳ mới. Có citation. |
| 5 | Để đăng ký xét tốt nghiệp, sinh viên cần đóng khoản gì, chuẩn bị hồ sơ nào và hoàn thành những nghĩa vụ nào? | `xet_tot_nghiep::chunk_000` — lệ phí, hồ sơ, nghĩa vụ | 0.0328 (RRF) | Có (top-1) | Đóng lệ phí xét bằng; hồ sơ THPT/khai sinh/chứng chỉ NN; hoàn thành nợ thư viện… Có citation. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Metric tổng hợp chiến lược của tôi (high_accuracy):** Hit@1 = 100%, Recall@3/5 = 100%, MRR@5 = 1.000, procedure accuracy = 100%, citation present = 100%, unsupported citations = 0. Diagnostic out-of-corpus (“Chi phí ký túc xá…”) từ chối đúng: *Không tìm thấy thông tin này trong tài liệu UIT đã nạp.*

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> So sánh với baseline FixedSize + dense-only trên cùng 5 câu (Hit@1 chỉ 20%, Recall@5 60%) cho thấy metadata filter + BM25/RRF quan trọng hơn việc chỉ “đổi embedder”. Chunk cắt giữa điều kiện và bước dễ làm agent trả lời thiếu; structure-aware + adjacent expansion giữ được các từ như “tối đa”, “chậm nhất”, “dưới 30%”.

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
