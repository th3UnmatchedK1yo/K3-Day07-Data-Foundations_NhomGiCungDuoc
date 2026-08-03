# UIT RAG Benchmark Report

- Generated: 2026-08-03T06:25:58.282560+00:00
- Python: 3.11.9
- Source document: `C:\Users\Legion\Documents\THUCHANH-AITHUCCHIEN-E403\K3-Day07-Data-Foundations_NhomGiCungDuoc\data\uit\quy-trinh-danh-cho-sinh-vien.md`
- LLM judge used (supplementary only): False

## Bảng so sánh Retrieval Metrics

| Metric | baseline | high_accuracy |
|---|---|---|
| Hit@1 | 20.0% | 100.0% |
| Recall@3 | 60.0% | 100.0% |
| Recall@5 | 60.0% | 100.0% |
| MRR@5 | 0.367 | 1.000 |
| Procedure accuracy | 0.0% | 100.0% |
| Mean retrieval latency | 21.3 ms | 131.8 ms |
| P95 retrieval latency | 22.7 ms | 253.0 ms |

## Bảng so sánh Answer Metrics (nếu có gọi DeepSeek)

| Metric | baseline | high_accuracy |
|---|---|---|
| Mean gold keyword coverage | 48.3% | 83.3% |
| Citation present rate | 60.0% | 100.0% |
| Total unsupported citations | 0 | 0 |
| Mean generation latency | 8969.7 ms | 6125.9 ms |
| Diagnostic refusal correct rate | 100.0% | 100.0% |

## Cấu hình mỗi strategy

### baseline
- Embedding model thực tế: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- DeepSeek model: `deepseek-v4-flash`
- Chunker: `baseline`
- BM25: False, Metadata filter: False
- Reranker yêu cầu: False, reranker thực sự hoạt động: False
- Adjacent expansion: False
- Số chunks trong index: 47 (build in 2124.3 ms)

### high_accuracy
- Embedding model thực tế: `BAAI/bge-m3`
- DeepSeek model: `deepseek-v4-flash`
- Chunker: `structure_aware`
- BM25: True, Metadata filter: True
- Reranker yêu cầu: False, reranker thực sự hoạt động: False
- Adjacent expansion: True
- Số chunks trong index: 23 (build in 22282.8 ms)

## Kết quả từng câu hỏi (5 core queries)

### Q1: Sinh viên được đăng ký tối đa bao nhiêu tín chỉ trong một học kỳ? Khi nào được đăng ký 30 tín chỉ?
- Expected procedure: `dang_ky_hoc_phan`
  - **baseline**: predicted_procedure=`None` (confidence=0.00, filter_applied=False); top-1 chunk=`uit_student_procedures::cong_nhan_tin_chi::baseline_000` (relevant=False)
    - answer: gold_keyword_coverage=75.0%, citations=1, unsupported_citations=0
  - **high_accuracy**: predicted_procedure=`dang_ky_hoc_phan` (confidence=0.75, filter_applied=True); top-1 chunk=`uit_student_procedures::dang_ky_hoc_phan::chunk_001` (relevant=True)
    - answer: gold_keyword_coverage=100.0%, citations=2, unsupported_citations=0

### Q2: Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi những gì và có được đăng ký môn mới không?
- Expected procedure: `dang_ky_hoc_phan`
  - **baseline**: predicted_procedure=`None` (confidence=0.00, filter_applied=False); top-1 chunk=`uit_student_procedures::chuyen_nganh::baseline_000` (relevant=False)
    - answer: gold_keyword_coverage=0.0%, citations=0, unsupported_citations=0
  - **high_accuracy**: predicted_procedure=`dang_ky_hoc_phan` (confidence=1.00, filter_applied=True); top-1 chunk=`uit_student_procedures::dang_ky_hoc_phan::chunk_001` (relevant=True)
    - answer: gold_keyword_coverage=75.0%, citations=2, unsupported_citations=0

### Q3: Sinh viên xin tạm dừng học tập vì lý do cá nhân phải đáp ứng điều kiện gì và được tạm dừng tối đa bao lâu?
- Expected procedure: `tam_dung_hoc_tap`
  - **baseline**: predicted_procedure=`None` (confidence=0.00, filter_applied=False); top-1 chunk=`uit_student_procedures::tam_dung_hoc_tap::baseline_000` (relevant=True)
    - answer: gold_keyword_coverage=66.7%, citations=3, unsupported_citations=0
  - **high_accuracy**: predicted_procedure=`tam_dung_hoc_tap` (confidence=0.75, filter_applied=True); top-1 chunk=`uit_student_procedures::tam_dung_hoc_tap::chunk_000` (relevant=True)
    - answer: gold_keyword_coverage=66.7%, citations=4, unsupported_citations=0

### Q4: Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn chót là khi nào?
- Expected procedure: `tam_dung_hoc_tap`
  - **baseline**: predicted_procedure=`None` (confidence=0.00, filter_applied=False); top-1 chunk=`uit_student_procedures::dieu_chinh_dang_ky_hoc_phan::baseline_001` (relevant=False)
    - answer: gold_keyword_coverage=0.0%, citations=0, unsupported_citations=0
  - **high_accuracy**: predicted_procedure=`tam_dung_hoc_tap` (confidence=0.75, filter_applied=True); top-1 chunk=`uit_student_procedures::tam_dung_hoc_tap::chunk_000` (relevant=True)
    - answer: gold_keyword_coverage=75.0%, citations=1, unsupported_citations=0

### Q5: Để đăng ký xét tốt nghiệp, sinh viên cần đóng khoản gì, chuẩn bị hồ sơ nào và hoàn thành những nghĩa vụ nào?
- Expected procedure: `xet_tot_nghiep`
  - **baseline**: predicted_procedure=`None` (confidence=0.00, filter_applied=False); top-1 chunk=`uit_student_procedures::xet_tot_nghiep::baseline_004` (relevant=False)
    - answer: gold_keyword_coverage=100.0%, citations=6, unsupported_citations=0
  - **high_accuracy**: predicted_procedure=`xet_tot_nghiep` (confidence=1.00, filter_applied=True); top-1 chunk=`uit_student_procedures::xet_tot_nghiep::chunk_000` (relevant=True)
    - answer: gold_keyword_coverage=100.0%, citations=6, unsupported_citations=0

## Diagnostic query (không tính vào 5 core queries)

- **baseline** — `Chi phí ký túc xá UIT hiện tại là bao nhiêu?`
  - predicted_procedure=`None` (confidence=0.00), top chunk scores=[0.38556741004934625, 0.37260229780154647, 0.31345404266554955, 0.30253849887316786, 0.29476108074653645]
  - refusal_correct=True
  - answer: Không tìm thấy thông tin này trong tài liệu UIT đã nạp.
- **high_accuracy** — `Chi phí ký túc xá UIT hiện tại là bao nhiêu?`
  - predicted_procedure=`None` (confidence=0.00), top chunk scores=[0.032266458495966696, 0.031544957774465976, 0.030776515151515152, 0.03076923076923077, 0.03036576949620428]
  - refusal_correct=True
  - answer: Không tìm thấy thông tin này trong tài liệu UIT đã nạp.

## Failure cases & nguyên nhân có thể

- **baseline / Q2** (`Trong đợt cứu xét đăng ký học phần, sinh viên được thay đổi ...`): không có chunk relevant trong top-5. Nguyên nhân có thể: chunk không bao phủ đủ gold keywords hoặc embedding không đủ mạnh để xếp hạng đúng chunk.
- **baseline / Q4** (`Sau khi hết thời hạn bảo lưu, sinh viên phải làm gì và hạn c...`): không có chunk relevant trong top-5. Nguyên nhân có thể: chunk không bao phủ đủ gold keywords hoặc embedding không đủ mạnh để xếp hạng đúng chunk.

## Metadata filter: giúp ích hay gây hại?

- Metadata filter được áp dụng cho 5/5 core queries trong strategy `high_accuracy`.
- Trong số đó: 5 câu vẫn có chunk relevant, 0 câu KHÔNG có chunk relevant.

## Kết luận

Trên bộ 5 core queries này, `high_accuracy` cho Recall@5=100.0% và MRR@5=1.000, cao hơn hoặc bằng `baseline` (Recall@5=60.0%, MRR@5=0.367). high_accuracy có xu hướng tốt hơn trên corpus này.
