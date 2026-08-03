"""Tests for src/rag_pipeline.py: UITRAGPipeline (build_index/retrieve/answer + CLI).

Fast and network-free:
  - Embeddings use a deterministic hash-based fake embedder (no sentence-transformers
    model is ever loaded), matching the lab's own MockEmbedder approach.
  - DeepSeek is always a fake/mocked client (no network call, no real API key).
"""

from __future__ import annotations

import hashlib
import math

import pytest

import src.rag_pipeline as rag_pipeline_module
from src.deepseek_client import REFUSAL_MESSAGE
from src.rag_pipeline import DEFAULT_INDEX_CACHE, UITRAGPipeline, normalize_query
from src.uit_preprocessing import DEFAULT_SOURCE_PATH


class FakeEmbedder:
    """Deterministic, network-free stand-in for LocalRAGEmbedder (mirrors src.embeddings.MockEmbedder)."""

    model_name = "fake-mock-embedder"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def __call__(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        seed = int(digest, 16)
        vector = []
        for _ in range(self.dim):
            seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
            vector.append((seed / 0xFFFFFFFF) * 2 - 1)
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class FakeDeepSeekClient:
    """Records the last prompt/context and returns a scripted answer -- never calls the network."""

    model = "fake-deepseek-model-for-tests"

    def __init__(self, scripted_answer: str) -> None:
        self._scripted_answer = scripted_answer
        self.last_question: str | None = None
        self.last_context: str | None = None

    def generate(self, question: str, context: str, temperature: float = 0.0) -> str:
        self.last_question = question
        self.last_context = context
        return self._scripted_answer


def _build_pipeline(chunker: str = "structure_aware") -> UITRAGPipeline:
    pipeline = UITRAGPipeline(embedder=FakeEmbedder(), chunker=chunker)
    pipeline.build_index(str(DEFAULT_SOURCE_PATH))
    return pipeline


# --------------------------------------------------------------------------
# normalize_query
# --------------------------------------------------------------------------


def test_normalize_query_collapses_whitespace_and_unicode_spaces():
    assert normalize_query("  Đăng   ký\u00a0học   phần?  ") == "Đăng ký học phần?"


# --------------------------------------------------------------------------
# build_index / retrieve (no DeepSeek call needed)
# --------------------------------------------------------------------------


def test_build_index_produces_chunks_with_procedure_metadata():
    pipeline = _build_pipeline()
    assert len(pipeline.chunks) > 0
    for chunk in pipeline.chunks:
        assert chunk["metadata"].get("procedure_slug")


def test_retrieve_before_build_index_raises_runtime_error():
    pipeline = UITRAGPipeline(embedder=FakeEmbedder())
    with pytest.raises(RuntimeError):
        pipeline.retrieve("bất kỳ câu hỏi nào")


def test_retrieve_returns_chunks_with_expected_shape():
    pipeline = _build_pipeline()
    results = pipeline.retrieve("Sinh viên đăng ký học phần như thế nào?", top_k=5)
    assert isinstance(results, list)
    assert results
    for chunk in results:
        assert {"chunk_id", "content", "raw_text", "metadata", "score", "source"} <= chunk.keys()


def test_retrieve_never_requires_a_deepseek_client():
    pipeline = UITRAGPipeline(embedder=FakeEmbedder())
    assert pipeline._deepseek_client is None
    pipeline.build_index(str(DEFAULT_SOURCE_PATH))
    pipeline.retrieve("Điều kiện phúc khảo là gì?")
    assert pipeline._deepseek_client is None  # retrieve() must never instantiate DeepSeekClient


def test_baseline_and_high_accuracy_chunkers_both_build_successfully():
    baseline = _build_pipeline(chunker="baseline")
    high_accuracy = _build_pipeline(chunker="structure_aware")
    assert len(baseline.chunks) > 0
    assert len(high_accuracy.chunks) > 0


def test_constructor_rejects_unknown_chunker_kind():
    with pytest.raises(ValueError):
        UITRAGPipeline(embedder=FakeEmbedder(), chunker="not-a-real-chunker")


# --------------------------------------------------------------------------
# Citation extraction
# --------------------------------------------------------------------------


def test_extract_citations_matches_the_documented_format():
    text = (
        "SV được đăng ký tối đa 24 tín chỉ. [uit_student_procedures/dang_ky_hoc_phan/uit_student_procedures::dang_ky_hoc_phan::chunk_000]\n"
        "Ngoại lệ: 30 tín chỉ nếu ĐTBC >= 8,0. [uit_student_procedures/dang_ky_hoc_phan/uit_student_procedures::dang_ky_hoc_phan::chunk_001]"
    )
    citations = UITRAGPipeline._extract_citations(text)
    assert len(citations) == 2
    assert citations[0].startswith("[uit_student_procedures/dang_ky_hoc_phan/")


def test_extract_citations_returns_empty_list_when_no_citation_present():
    assert UITRAGPipeline._extract_citations("Không tìm thấy thông tin này trong tài liệu UIT đã nạp.") == []


# --------------------------------------------------------------------------
# answer() with a fully mocked DeepSeek client (no network)
# --------------------------------------------------------------------------


def test_answer_includes_grounded_citation_from_a_retrieved_chunk():
    pipeline = _build_pipeline()
    top_chunk = pipeline.retrieve("Sinh viên đăng ký học phần như thế nào?", top_k=1)[0]
    scripted = (
        f"Câu trả lời mẫu. [uit_student_procedures/{top_chunk['metadata']['procedure_slug']}/{top_chunk['chunk_id']}]"
    )
    pipeline._deepseek_client = FakeDeepSeekClient(scripted)

    result = pipeline.answer("Sinh viên đăng ký học phần như thế nào?", top_k=5)

    assert result["answer"] == scripted
    assert result["citations"] == [
        f"[uit_student_procedures/{top_chunk['metadata']['procedure_slug']}/{top_chunk['chunk_id']}]"
    ]
    assert result["model"] == "fake-deepseek-model-for-tests"
    assert result["embedding_model"] == "fake-mock-embedder"
    assert set(result["latency_ms"].keys()) == {"retrieval", "generation", "total"}
    assert result["retrieved_chunks"]


def test_answer_never_calls_deepseek_with_an_empty_context_key_missing():
    pipeline = _build_pipeline()
    fake_client = FakeDeepSeekClient("Câu trả lời.")
    pipeline._deepseek_client = fake_client
    pipeline.answer("Điều kiện tạm dừng học tập?", top_k=3)
    assert fake_client.last_context  # context was built and passed through
    assert "Quy trình:" in fake_client.last_context


def test_diagnostic_query_triggers_refusal_message_when_deepseek_says_so():
    """The diagnostic query is out-of-corpus; DeepSeek (mocked here) must refuse, not hallucinate."""
    pipeline = _build_pipeline()
    pipeline._deepseek_client = FakeDeepSeekClient(REFUSAL_MESSAGE)

    result = pipeline.answer("Chi phí ký túc xá UIT hiện tại là bao nhiêu?", top_k=5)

    assert result["answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []  # a refusal must carry no fabricated citation


# --------------------------------------------------------------------------
# save_index / load_index round-trip (still no real embedding model loaded)
# --------------------------------------------------------------------------


def test_save_and_load_index_roundtrip_preserves_retrieval(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_pipeline_module, "LocalRAGEmbedder", lambda model_name=None: FakeEmbedder())

    pipeline = _build_pipeline()
    query = "Sinh viên đăng ký học phần như thế nào?"
    expected = pipeline.retrieve(query, top_k=3)

    index_path = tmp_path / "index.pkl"
    pipeline.save_index(index_path)

    reloaded = UITRAGPipeline.load_index(index_path)
    actual = reloaded.retrieve(query, top_k=3)

    assert [c["chunk_id"] for c in actual] == [c["chunk_id"] for c in expected]


def test_load_index_raises_file_not_found_for_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError):
        UITRAGPipeline.load_index(tmp_path / "missing_index.pkl")


def test_default_index_cache_path_is_gitignored_cache_dir():
    assert DEFAULT_INDEX_CACHE.parts[0] == ".rag_cache"
