"""Tests for src/uit_preprocessing.py: normalization + procedure detection.

Fast and network-free: only reads local text/JSON files.
"""

from __future__ import annotations

import pytest

from src.uit_preprocessing import (
    DEFAULT_BOUNDARIES_PATH,
    DEFAULT_SOURCE_PATH,
    ProcedureBoundaryError,
    detect_sections,
    load_procedure_boundaries,
    normalize_text,
    preprocess_uit_document,
    read_uit_markdown,
)

# The 19 procedures the task requires to be detectable in the UIT source document.
REQUIRED_PROCEDURE_SLUGS = {
    "bang_diem",
    "dang_ky_hoc_phan",
    "chuyen_nganh",
    "chuyen_truong",
    "bao_luu_trung_tuyen",
    "tam_dung_hoc_tap",
    "hoan_thi",
    "thoi_hoc",
    "gia_han_hoc_tap",
    "phuc_khao",
    "giay_gioi_thieu",
    "mien_ngoai_ngu",
    "xet_tot_nghiep",
    "dieu_chinh_dang_ky_hoc_phan",
    "hoc_chuong_trinh_thu_hai",
    "khieu_nai",
    "chuyen_he_dao_tao",
    "cong_nhan_tin_chi",
    "bieu_mau",
}


def test_normalize_text_unifies_newlines_and_unicode_spaces():
    raw = "A\r\nB\rC\u00a0D\ufeffE"
    normalized = normalize_text(raw)
    assert "\r" not in normalized
    assert "\u00a0" not in normalized
    assert "\ufeff" not in normalized
    # \r\n / \r -> \n ; non-breaking space -> regular space ; BOM/ZWNBSP -> removed entirely (not a space)
    assert "A\nB\nC DE" == normalized


def test_normalize_text_collapses_many_blank_lines_to_one():
    raw = "Para 1\n\n\n\n\nPara 2"
    normalized = normalize_text(raw)
    assert "\n\n\n" not in normalized
    assert normalized == "Para 1\n\nPara 2"


def test_normalize_text_unifies_bullet_glyphs_without_changing_wording():
    raw = "• Điều kiện A\n\u25e6 Điều kiện B"
    normalized = normalize_text(raw)
    assert normalized.splitlines()[0] == "- Điều kiện A"
    assert normalized.splitlines()[1] == "- Điều kiện B"


def test_normalize_text_preserves_numbers_urls_and_form_names():
    raw = (
        "SV đăng ký tại: https://student.uit.edu.vn/sinhvien/dangky-giaygioithieu\n"
        "Nộp Mẫu 07 trong vòng 30 ngày, ĐTBC tối thiểu 8,0."
    )
    normalized = normalize_text(raw)
    assert "https://student.uit.edu.vn/sinhvien/dangky-giaygioithieu" in normalized
    assert "Mẫu 07" in normalized
    assert "30 ngày" in normalized
    assert "8,0" in normalized


def test_normalize_text_empty_input_returns_empty_string():
    assert normalize_text("") == ""


def test_read_uit_markdown_does_not_modify_the_file_on_disk():
    before = DEFAULT_SOURCE_PATH.read_bytes()
    read_uit_markdown(DEFAULT_SOURCE_PATH)
    after = DEFAULT_SOURCE_PATH.read_bytes()
    assert before == after


def test_load_procedure_boundaries_rejects_empty_config(tmp_path):
    bad_config = tmp_path / "empty_boundaries.json"
    bad_config.write_text('{"procedures": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_procedure_boundaries(bad_config)


def test_load_procedure_boundaries_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_procedure_boundaries(tmp_path / "does_not_exist.json")


def test_detect_sections_raises_on_missing_anchor():
    normalized_text = normalize_text("Nội dung không liên quan gì đến quy trình nào cả.")
    boundaries = {
        "source_document": {"source_id": "uit_student_procedures"},
        "procedures": [
            {"slug": "khong_ton_tai", "title": "Quy trình không tồn tại", "anchor": "CHUỖI KHÔNG XUẤT HIỆN TRONG VĂN BẢN"}
        ],
    }
    with pytest.raises(ProcedureBoundaryError):
        detect_sections(normalized_text, boundaries)


def test_detect_sections_covers_all_required_procedure_slugs():
    sections = preprocess_uit_document(DEFAULT_SOURCE_PATH, DEFAULT_BOUNDARIES_PATH)
    found_slugs = {section["procedure_slug"] for section in sections}
    missing = REQUIRED_PROCEDURE_SLUGS - found_slugs
    assert not missing, f"Missing required procedure slugs: {missing}"


def test_detect_sections_are_sequential_and_non_overlapping():
    sections = preprocess_uit_document(DEFAULT_SOURCE_PATH, DEFAULT_BOUNDARIES_PATH)
    raw_text = normalize_text(read_uit_markdown(DEFAULT_SOURCE_PATH))
    offsets = [raw_text.find(section["text"][:40]) for section in sections]
    assert all(offset != -1 for offset in offsets)
    assert offsets == sorted(offsets), "Sections must appear in the same order as in the source document"


def test_each_section_has_required_shape_and_metadata():
    sections = preprocess_uit_document(DEFAULT_SOURCE_PATH, DEFAULT_BOUNDARIES_PATH)
    for index, section in enumerate(sections):
        assert section["section_index"] == index
        assert section["text"].strip(), "section text must not be empty"
        metadata = section["metadata"]
        for required_key in (
            "source_id",
            "source_title",
            "source_url",
            "institution",
            "audience",
            "document_type",
            "procedure_slug",
            "procedure_title",
            "section_index",
        ):
            assert required_key in metadata
        assert metadata["procedure_slug"] == section["procedure_slug"]
        # The task forbids inventing effective_date/academic_year when the source doesn't state them.
        assert "effective_date" not in metadata
        assert "academic_year" not in metadata


def test_sections_do_not_mix_two_different_procedures():
    """A section for procedure X must not contain the *anchor* of the next procedure Y."""
    boundaries = load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH)
    procedures = boundaries["procedures"]
    sections = preprocess_uit_document(DEFAULT_SOURCE_PATH, DEFAULT_BOUNDARIES_PATH)

    for section, next_procedure in zip(sections[:-1], procedures[1:]):
        assert next_procedure["anchor"] not in section["text"], (
            f"Section '{section['procedure_slug']}' leaked into the next procedure "
            f"'{next_procedure['slug']}'"
        )
