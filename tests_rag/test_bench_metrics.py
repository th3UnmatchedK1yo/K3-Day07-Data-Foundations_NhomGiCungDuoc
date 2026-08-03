"""Tests for the pure, network-free metric helpers in bench.py.

These are the building blocks of the retrieval/answer benchmark: relevance
judging, keyword coverage, citation extraction and the aggregate retrieval
metrics (Hit@1 / Recall@k / MRR@5 / procedure accuracy).
"""

from __future__ import annotations

import json

from bench import (
    DIAGNOSTIC_QUERIES_PATH,
    GOLD_QUERIES_PATH,
    compute_retrieval_metrics,
    extract_citations,
    is_relevant_chunk,
    keyword_coverage,
)


def _chunk(procedure_slug: str, raw_text: str) -> dict:
    return {"metadata": {"procedure_slug": procedure_slug}, "raw_text": raw_text}


def test_is_relevant_chunk_requires_correct_procedure_slug():
    chunk = _chunk("phuc_khao", "24 tín chỉ ĐTBC 8,0 30 tín chỉ")
    assert is_relevant_chunk(chunk, "dang_ky_hoc_phan", ["24 tín chỉ", "ĐTBC", "8,0", "30 tín chỉ"]) is False


def test_is_relevant_chunk_true_when_procedure_matches_and_keywords_covered():
    chunk = _chunk("dang_ky_hoc_phan", "SV được đăng ký tối đa 24 tín chỉ, hoặc 30 tín chỉ nếu ĐTBC đạt 8,0")
    assert is_relevant_chunk(chunk, "dang_ky_hoc_phan", ["24 tín chỉ", "ĐTBC", "8,0", "30 tín chỉ"]) is True


def test_is_relevant_chunk_false_when_keyword_coverage_below_threshold():
    chunk = _chunk("dang_ky_hoc_phan", "Chỉ có 24 tín chỉ ở đây, không có gì khác.")
    # only 1/4 gold keywords present -> below the 0.5 threshold
    assert is_relevant_chunk(chunk, "dang_ky_hoc_phan", ["24 tín chỉ", "ĐTBC", "8,0", "30 tín chỉ"]) is False


def test_is_relevant_chunk_true_when_no_gold_keywords_given():
    chunk = _chunk("dang_ky_hoc_phan", "bất kỳ nội dung gì")
    assert is_relevant_chunk(chunk, "dang_ky_hoc_phan", []) is True


def test_is_relevant_chunk_matching_is_case_and_unicode_insensitive():
    chunk = _chunk("dang_ky_hoc_phan", "sv được đăng ký TỐI ĐA 24 TÍN CHỈ")
    assert is_relevant_chunk(chunk, "dang_ky_hoc_phan", ["24 tín chỉ"]) is True


def test_keyword_coverage_full_and_partial():
    assert keyword_coverage("24 tín chỉ và ĐTBC 8,0", ["24 tín chỉ", "ĐTBC", "8,0"]) == 1.0
    assert keyword_coverage("24 tín chỉ", ["24 tín chỉ", "ĐTBC"]) == 0.5
    assert keyword_coverage("không liên quan", ["24 tín chỉ"]) == 0.0


def test_keyword_coverage_returns_one_when_no_gold_keywords():
    assert keyword_coverage("bất kỳ gì", []) == 1.0


def test_extract_citations_returns_procedure_and_chunk_id_tuples():
    text = "Ý 1. [uit_student_procedures/dang_ky_hoc_phan/uit_student_procedures::dang_ky_hoc_phan::chunk_001]"
    citations = extract_citations(text)
    assert citations == [("dang_ky_hoc_phan", "uit_student_procedures::dang_ky_hoc_phan::chunk_001")]


def test_extract_citations_returns_empty_for_refusal_text():
    assert extract_citations("Không tìm thấy thông tin này trong tài liệu UIT đã nạp.") == []


def test_compute_retrieval_metrics_hit_at_1_and_mrr():
    per_query = [
        {
            "relevance_flags": [True, False, False, False, False],
            "predicted_procedure": "dang_ky_hoc_phan",
            "expected_procedure": "dang_ky_hoc_phan",
            "retrieval_latency_ms": 10.0,
        },
        {
            "relevance_flags": [False, True, False, False, False],
            "predicted_procedure": None,
            "expected_procedure": "phuc_khao",
            "retrieval_latency_ms": 20.0,
        },
    ]
    metrics = compute_retrieval_metrics(per_query)
    assert metrics["hit_at_1"] == 0.5
    assert metrics["recall_at_3"] == 1.0
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_5"] == (1.0 + 0.5) / 2
    assert metrics["procedure_accuracy"] == 0.5
    assert metrics["mean_retrieval_latency_ms"] == 15.0


def test_compute_retrieval_metrics_zero_when_nothing_relevant():
    per_query = [
        {
            "relevance_flags": [False, False, False, False, False],
            "predicted_procedure": "x",
            "expected_procedure": "y",
            "retrieval_latency_ms": 5.0,
        }
    ]
    metrics = compute_retrieval_metrics(per_query)
    assert metrics["hit_at_1"] == 0.0
    assert metrics["recall_at_5"] == 0.0
    assert metrics["mrr_at_5"] == 0.0


# --------------------------------------------------------------------------
# Gold / diagnostic query files: shape required by the task specification.
# --------------------------------------------------------------------------


def test_gold_queries_file_has_exactly_five_core_queries():
    gold_queries = json.loads(GOLD_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    assert len(gold_queries) == 5
    for query in gold_queries:
        assert {"id", "question", "expected_procedure", "gold_keywords"} <= query.keys()


def test_diagnostic_queries_file_has_at_least_one_out_of_corpus_query():
    diagnostic_queries = json.loads(DIAGNOSTIC_QUERIES_PATH.read_text(encoding="utf-8"))["queries"]
    assert len(diagnostic_queries) >= 1
    assert any(query.get("in_corpus") is False for query in diagnostic_queries)
