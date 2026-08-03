"""Tests for src/retrieval.py: MetadataFilter, RRF, dedup, adjacent expansion.

Fast and network-free: no embedding model or DeepSeek calls. BM25Retriever
tests require ``rank-bm25`` (see requirements-rag.txt).
"""

from __future__ import annotations

from src.retrieval import (
    BM25Retriever,
    MetadataFilter,
    RetrievedChunk,
    deduplicate,
    expand_adjacent,
    reciprocal_rank_fusion,
)
from src.uit_preprocessing import DEFAULT_BOUNDARIES_PATH, load_procedure_boundaries


def _chunk(chunk_id: str, score: float, procedure_slug: str = "dang_ky_hoc_phan", **extra_metadata) -> RetrievedChunk:
    metadata = {"procedure_slug": procedure_slug, "chunk_id": chunk_id, **extra_metadata}
    return RetrievedChunk(chunk_id=chunk_id, content=f"content {chunk_id}", raw_text=f"raw {chunk_id}", metadata=metadata, score=score)


# --------------------------------------------------------------------------
# MetadataFilter
# --------------------------------------------------------------------------


def test_metadata_filter_high_confidence_for_unambiguous_keyword():
    metadata_filter = MetadataFilter(load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH))
    slug, confidence = metadata_filter.classify("Sinh viên đăng ký học phần ĐKHP thế nào?")
    assert slug == "dang_ky_hoc_phan"
    assert MetadataFilter.should_filter(confidence) is True


def test_metadata_filter_recognizes_phuc_khao():
    metadata_filter = MetadataFilter(load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH))
    slug, confidence = metadata_filter.classify("Thủ tục phúc khảo điểm thi như thế nào?")
    assert slug == "phuc_khao"
    assert confidence > 0


def test_metadata_filter_recognizes_xet_tot_nghiep():
    metadata_filter = MetadataFilter(load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH))
    slug, _confidence = metadata_filter.classify("Điều kiện xét tốt nghiệp là gì?")
    assert slug == "xet_tot_nghiep"


def test_metadata_filter_low_confidence_for_ambiguous_query_returns_none_or_low_confidence():
    metadata_filter = MetadataFilter(load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH))
    slug, confidence = metadata_filter.classify("Trường đại học UIT ở đâu?")
    assert slug is None
    assert confidence == 0.0
    assert MetadataFilter.should_filter(confidence) is False


def test_metadata_filter_should_filter_respects_custom_threshold():
    assert MetadataFilter.should_filter(0.5, threshold=0.6) is False
    assert MetadataFilter.should_filter(0.6, threshold=0.6) is True


# --------------------------------------------------------------------------
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def test_rrf_is_deterministic_across_repeated_calls():
    dense = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
    bm25 = [_chunk("b", 5.0), _chunk("a", 4.0), _chunk("d", 3.0)]

    first = reciprocal_rank_fusion([dense, bm25])
    second = reciprocal_rank_fusion([dense, bm25])

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.score for c in first] == [c.score for c in second]


def test_rrf_ranks_chunks_present_in_both_lists_higher():
    dense = [_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]
    bm25 = [_chunk("b", 5.0), _chunk("d", 3.0), _chunk("e", 1.0)]

    fused = reciprocal_rank_fusion([dense, bm25])
    fused_ids = [c.chunk_id for c in fused]
    # "b" is top-of-BM25 and 2nd-in-dense -> should outrank chunks present in only one list.
    assert fused_ids[0] == "b"


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    dense = [_chunk("a", 0.9)]
    assert [c.chunk_id for c in reciprocal_rank_fusion([dense, []])] == ["a"]


def test_rrf_tie_breaks_deterministically_by_chunk_id():
    # Two chunks with an identical single appearance at rank 1 in disjoint lists get an equal RRF score.
    list_one = [_chunk("z", 1.0)]
    list_two = [_chunk("a", 1.0)]
    fused = reciprocal_rank_fusion([list_one, list_two])
    assert fused[0].score == fused[1].score
    assert fused[0].chunk_id == "a"  # alphabetical tie-break


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------


def test_deduplicate_drops_repeated_content_hash_keeping_first():
    chunks = [
        _chunk("a", 0.9, content_hash="hash1"),
        _chunk("b", 0.8, content_hash="hash1"),  # duplicate content
        _chunk("c", 0.7, content_hash="hash2"),
    ]
    deduped = deduplicate(chunks)
    assert [c.chunk_id for c in deduped] == ["a", "c"]


def test_deduplicate_falls_back_to_chunk_id_when_no_content_hash():
    chunks = [_chunk("a", 0.9), _chunk("a", 0.8)]
    deduped = deduplicate(chunks)
    assert len(deduped) == 1


# --------------------------------------------------------------------------
# Adjacent expansion
# --------------------------------------------------------------------------


def test_expand_adjacent_pulls_in_same_procedure_neighbors():
    lookup = {
        "prev": {"chunk_id": "prev", "text": "prev text", "raw_text": "prev raw", "metadata": {"procedure_slug": "dang_ky_hoc_phan"}},
        "next": {"chunk_id": "next", "text": "next text", "raw_text": "next raw", "metadata": {"procedure_slug": "dang_ky_hoc_phan"}},
    }
    main_chunk = _chunk("main", 0.9, previous_chunk_id="prev", next_chunk_id="next")
    expanded = expand_adjacent([main_chunk], lookup)
    expanded_ids = {c.chunk_id for c in expanded}
    assert expanded_ids == {"main", "prev", "next"}


def test_expand_adjacent_never_pulls_a_different_procedure():
    lookup = {
        "other_proc": {
            "chunk_id": "other_proc",
            "text": "t",
            "raw_text": "t",
            "metadata": {"procedure_slug": "phuc_khao"},
        },
    }
    main_chunk = _chunk("main", 0.9, procedure_slug="dang_ky_hoc_phan", next_chunk_id="other_proc")
    expanded = expand_adjacent([main_chunk], lookup)
    assert [c.chunk_id for c in expanded] == ["main"]


def test_expand_adjacent_respects_token_budget():
    huge_text = "từ " * 5000  # way over any reasonable budget
    lookup = {
        "next": {"chunk_id": "next", "text": huge_text, "raw_text": huge_text, "metadata": {"procedure_slug": "dang_ky_hoc_phan"}},
    }
    main_chunk = _chunk("main", 0.9, next_chunk_id="next")
    expanded = expand_adjacent([main_chunk], lookup, max_total_tokens=10)
    assert [c.chunk_id for c in expanded] == ["main"]


# --------------------------------------------------------------------------
# BM25Retriever (basic Vietnamese handling)
# --------------------------------------------------------------------------


def _bm25_corpus() -> list[dict]:
    # BM25Okapi's idf formula can be exactly 0 when a term appears in exactly
    # half of a tiny corpus, so these tests use enough unrelated filler chunks
    # for the query terms to have a clearly-below-50% document frequency.
    return [
        {
            "chunk_id": "c1",
            "text": "Đăng ký học phần ĐKHP tín chỉ",
            "raw_text": "Đăng ký học phần ĐKHP tín chỉ",
            "metadata": {"procedure_slug": "dang_ky_hoc_phan"},
        },
        {
            "chunk_id": "c2",
            "text": "Phúc khảo điểm thi cuối kỳ",
            "raw_text": "Phúc khảo điểm thi cuối kỳ",
            "metadata": {"procedure_slug": "phuc_khao"},
        },
        {
            "chunk_id": "c3",
            "text": "Chuyển trường sang cơ sở đào tạo khác",
            "raw_text": "Chuyển trường sang cơ sở đào tạo khác",
            "metadata": {"procedure_slug": "chuyen_truong"},
        },
        {
            "chunk_id": "c4",
            "text": "Xét tốt nghiệp và cấp bằng cho sinh viên",
            "raw_text": "Xét tốt nghiệp và cấp bằng cho sinh viên",
            "metadata": {"procedure_slug": "xet_tot_nghiep"},
        },
    ]


def test_bm25_retriever_ranks_matching_chunk_first():
    retriever = BM25Retriever(_bm25_corpus())
    results = retriever.retrieve("đăng ký học phần ĐKHP")
    assert results
    assert results[0].chunk_id == "c1"


def test_bm25_retriever_preserves_form_codes_and_numbers():
    """BM25 tokenization must not strip digits or keywords like ĐKHP/ĐTBC."""
    chunks = _bm25_corpus() + [
        {"chunk_id": "c5", "text": "Nộp Mẫu 07 trong vòng 30 ngày", "raw_text": "Nộp Mẫu 07 trong vòng 30 ngày", "metadata": {}},
    ]
    retriever = BM25Retriever(chunks)
    results = retriever.retrieve("Mẫu 07")
    assert results and results[0].chunk_id == "c5"


def test_bm25_retriever_handles_empty_corpus():
    retriever = BM25Retriever([])
    assert retriever.retrieve("bất kỳ câu hỏi nào") == []
