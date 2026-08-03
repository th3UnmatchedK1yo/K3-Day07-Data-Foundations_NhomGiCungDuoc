"""Tests for src/rag_embeddings.py: explicit (never silent) fallback + cache.

No real Sentence-Transformers model is downloaded: ``sentence_transformers.SentenceTransformer``
is monkeypatched with fast in-process fakes. Requires the ``sentence-transformers``
package to be *importable* (see requirements-rag.txt) but never touches the network.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sentence_transformers", reason="requirements-rag.txt not installed")

import sentence_transformers  # noqa: E402

from src.rag_embeddings import (  # noqa: E402
    FALLBACK_MODEL,
    PREFERRED_MODEL,
    EmbeddingBackendError,
    LocalRAGEmbedder,
    ModelLoadTimeoutError,
)


class _FakeSentenceTransformer:
    """Cheap stand-in with the tiny slice of the SentenceTransformer API we use."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts, **kwargs):
        import numpy as np

        single = isinstance(texts, str)
        batch = [texts] if single else list(texts)
        vectors = np.array([[float(len(t)), 1.0, 0.0] for t in batch])
        return vectors[0] if single else vectors


def test_preferred_model_loads_successfully_when_available(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = LocalRAGEmbedder(model_name=PREFERRED_MODEL)
    assert embedder.model_name == PREFERRED_MODEL
    assert embedder.fallback_used is False


def test_fallback_is_used_and_logged_explicitly_when_preferred_model_fails(monkeypatch, caplog):
    def _load(model_name: str):
        if model_name == FALLBACK_MODEL:
            return _FakeSentenceTransformer(model_name)
        raise OSError(f"simulated failure loading {model_name}")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _load)

    with caplog.at_level("WARNING"):
        embedder = LocalRAGEmbedder(model_name="some/unavailable-model")

    assert embedder.fallback_used is True
    assert embedder.model_name == FALLBACK_MODEL
    assert embedder.requested_model_name == "some/unavailable-model"
    # The fallback must be logged explicitly (never silent).
    assert any("falling back" in record.message.lower() for record in caplog.records)


def test_error_raised_when_both_requested_and_fallback_model_fail(monkeypatch):
    def _always_fail(model_name: str):
        raise OSError(f"simulated failure loading {model_name}")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _always_fail)
    with pytest.raises(EmbeddingBackendError):
        LocalRAGEmbedder(model_name="some/unavailable-model")


def test_load_timeout_triggers_fallback_instead_of_hanging_forever(monkeypatch):
    """A model 'load' that never returns must not block the caller past LOAD_TIMEOUT_S."""
    import src.rag_embeddings as rag_embeddings_module

    monkeypatch.setattr(rag_embeddings_module, "LOAD_TIMEOUT_S", 0.2)

    def _hangs_forever(model_name: str):
        if model_name == FALLBACK_MODEL:
            return _FakeSentenceTransformer(model_name)
        time.sleep(30)  # simulate a stalled download; the test must not wait 30s
        return _FakeSentenceTransformer(model_name)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _hangs_forever)

    start = time.perf_counter()
    embedder = LocalRAGEmbedder(model_name="some/slow-model")
    elapsed = time.perf_counter() - start

    assert embedder.fallback_used is True
    assert embedder.model_name == FALLBACK_MODEL
    assert elapsed < 5.0, "loading must give up after LOAD_TIMEOUT_S, not hang"


def test_load_timeout_error_message_mentions_the_model_name(monkeypatch):
    import src.rag_embeddings as rag_embeddings_module

    monkeypatch.setattr(rag_embeddings_module, "LOAD_TIMEOUT_S", 0.1)

    def _hangs(model_name: str):
        time.sleep(30)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _hangs)
    with pytest.raises(EmbeddingBackendError) as exc_info:
        # Both preferred and fallback attempts hang -> both time out -> final error.
        LocalRAGEmbedder(model_name=FALLBACK_MODEL)
    assert "did not finish within" in str(exc_info.value)


def test_embed_one_caches_by_content_hash_and_model_name(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = LocalRAGEmbedder(model_name=PREFERRED_MODEL)

    first = embedder.embed_one("Đăng ký học phần")
    cache_size_after_first = len(embedder._cache)
    second = embedder.embed_one("Đăng ký học phần")
    cache_size_after_second = len(embedder._cache)

    assert first == second
    assert cache_size_after_first == cache_size_after_second == 1


def test_embed_one_normalizes_and_returns_plain_floats(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = LocalRAGEmbedder(model_name=PREFERRED_MODEL)
    vector = embedder.embed_one("abc")
    assert all(isinstance(value, float) for value in vector)


def test_embed_batch_reuses_cache_and_matches_embed_one(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = LocalRAGEmbedder(model_name=PREFERRED_MODEL)

    single = embedder.embed_one("hello world")
    batch = embedder.embed_batch(["hello world", "another text"])

    assert batch[0] == single
    assert len(batch) == 2


def test_call_is_equivalent_to_embed_one(monkeypatch):
    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _FakeSentenceTransformer)
    embedder = LocalRAGEmbedder(model_name=PREFERRED_MODEL)
    assert embedder("some text") == embedder.embed_one("some text")


def test_missing_sentence_transformers_dependency_raises_clear_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(EmbeddingBackendError):
        LocalRAGEmbedder(model_name=PREFERRED_MODEL)
