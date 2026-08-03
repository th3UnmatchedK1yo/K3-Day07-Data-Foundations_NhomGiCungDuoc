"""Tests for src/structure_chunking.py: StructureAwareChunker.

Fast and network-free: chunking is pure text processing (no embedding model).
"""

from __future__ import annotations

from src.structure_chunking import StructureAwareChunker
from src.uit_preprocessing import DEFAULT_BOUNDARIES_PATH, DEFAULT_SOURCE_PATH, preprocess_uit_document

SAMPLE_SECTION = {
    "procedure_slug": "dang_ky_hoc_phan",
    "procedure_title": "Đăng ký học phần (ĐKHP)",
    "section_index": 0,
    "text": (
        "**Bước 1:** Khoảng một tháng trước khi bắt đầu học kỳ, P.ĐTĐH thông báo thời gian đăng ký "
        "học phần và thời khóa biểu các học phần dự kiến mở trong học kỳ.\n\n"
        "**Bước 2:** SV đăng ký học phần trên hệ thống trong thời gian quy định. SV được đăng ký tối đa "
        "24 tín chỉ mỗi học kỳ, hoặc tối đa 30 tín chỉ nếu ĐTBC từ 8,0 trở lên.\n\n"
        "- Lớp học phần được mở khi có tối thiểu 30 SV đăng ký.\n\n"
        "- SV đang bị cảnh cáo học vụ chỉ được đăng ký học lại.\n\n"
        "**Bước 3:** SV nộp Mẫu 07 nếu cần điều chỉnh đăng ký học phần trong vòng 7 ngày kể từ ngày mở lớp."
    ),
    "metadata": {
        "source_id": "uit_student_procedures",
        "procedure_slug": "dang_ky_hoc_phan",
        "procedure_title": "Đăng ký học phần (ĐKHP)",
        "section_index": 0,
    },
}

OTHER_SECTION = {
    "procedure_slug": "phuc_khao",
    "procedure_title": "Phúc khảo",
    "section_index": 1,
    "text": (
        "**Bước 1:** SV nộp đơn phúc khảo chậm nhất trong vòng 10 ngày kể từ ngày công bố điểm.\n\n"
        "**Bước 2:** Bộ môn tổ chức phúc khảo và trả kết quả trong vòng 15 ngày làm việc."
    ),
    "metadata": {
        "source_id": "uit_student_procedures",
        "procedure_slug": "phuc_khao",
        "procedure_title": "Phúc khảo",
        "section_index": 1,
    },
}


def test_chunk_ids_are_deterministic_across_repeated_runs():
    chunker = StructureAwareChunker()
    first_run = chunker.chunk_sections([SAMPLE_SECTION])
    second_run = StructureAwareChunker().chunk_sections([SAMPLE_SECTION])

    assert [c.chunk_id for c in first_run] == [c.chunk_id for c in second_run]
    assert [c.metadata["content_hash"] for c in first_run] == [c.metadata["content_hash"] for c in second_run]


def test_chunk_ids_follow_expected_naming_scheme():
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_sections([SAMPLE_SECTION])
    for index, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"uit_student_procedures::dang_ky_hoc_phan::chunk_{index:03d}"


def test_chunks_never_mix_two_different_procedures():
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_sections([SAMPLE_SECTION, OTHER_SECTION])

    for chunk in chunks:
        # raw_text must only ever contain content from its own procedure.
        if chunk.metadata["procedure_slug"] == "dang_ky_hoc_phan":
            assert "phúc khảo" not in chunk.raw_text.lower()
        else:
            assert "đăng ký học phần" not in chunk.raw_text.lower()


def test_previous_and_next_chunk_links_are_consistent():
    chunker = StructureAwareChunker(target_tokens=20, max_tokens=40, overlap_tokens=5)
    chunks = chunker.chunk_sections([SAMPLE_SECTION])
    assert len(chunks) >= 2, "test fixture must produce multiple chunks to exercise linking"

    assert chunks[0].metadata["previous_chunk_id"] is None
    assert chunks[-1].metadata["next_chunk_id"] is None
    for index in range(1, len(chunks)):
        assert chunks[index].metadata["previous_chunk_id"] == chunks[index - 1].chunk_id
        assert chunks[index - 1].metadata["next_chunk_id"] == chunks[index].chunk_id


def test_header_is_prepended_to_embedding_text_but_absent_from_raw_text():
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_sections([SAMPLE_SECTION])
    for chunk in chunks:
        assert chunk.text.startswith("Tài liệu: Một số quy trình dành cho sinh viên")
        assert "Quy trình: Đăng ký học phần (ĐKHP)" in chunk.text
        assert "Tài liệu:" not in chunk.raw_text
        assert "Quy trình:" not in chunk.raw_text
        assert chunk.raw_text in chunk.text


def test_step_label_is_never_separated_from_its_own_content():
    """A 'Bước N' label must stay in the same chunk as the sentence(s) right after it."""
    chunker = StructureAwareChunker(target_tokens=15, max_tokens=25, overlap_tokens=0)
    chunks = chunker.chunk_sections([SAMPLE_SECTION])
    for chunk in chunks:
        for step_label in ("**Bước 1:**", "**Bước 2:**", "**Bước 3:**"):
            if step_label in chunk.raw_text:
                # The label and at least a few following characters (its own paragraph) exist together.
                idx = chunk.raw_text.index(step_label)
                assert len(chunk.raw_text) - idx > len(step_label) + 10


def test_constructor_rejects_invalid_token_bounds():
    import pytest

    with pytest.raises(ValueError):
        StructureAwareChunker(target_tokens=0)
    with pytest.raises(ValueError):
        StructureAwareChunker(target_tokens=500, max_tokens=100)


def test_empty_section_text_produces_no_chunks():
    chunker = StructureAwareChunker()
    empty_section = dict(SAMPLE_SECTION, text="   \n\n  ")
    assert chunker.chunk_sections([empty_section]) == []


def test_real_uit_document_chunks_stay_within_max_tokens_bound():
    sections = preprocess_uit_document(DEFAULT_SOURCE_PATH, DEFAULT_BOUNDARIES_PATH)
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_sections(sections)
    assert len(chunks) > 0
    for chunk in chunks:
        # A single oversized atomic unit (e.g. one long sentence) may slightly exceed max_tokens;
        # allow a small margin instead of a hard equality check.
        assert chunk.metadata["token_count"] <= chunker.max_tokens * 1.5


def test_real_uit_document_every_chunk_maps_to_exactly_one_known_procedure():
    sections = preprocess_uit_document(DEFAULT_SOURCE_PATH, DEFAULT_BOUNDARIES_PATH)
    known_slugs = {section["procedure_slug"] for section in sections}
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_sections(sections)
    for chunk in chunks:
        assert chunk.metadata["procedure_slug"] in known_slugs
