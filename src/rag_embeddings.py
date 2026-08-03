"""Configurable local embedding backend for the advanced UIT RAG pipeline.

Kept separate from ``src/embeddings.py`` (graded lab core, already complete)
so the lab's ``LocalEmbedder``/``OpenAIEmbedder``/``MockEmbedder`` classes
and their public signatures are never touched.

Configuration (environment variables):
    EMBEDDING_PROVIDER=local        Only "local" is supported here. DeepSeek
                                     is never used to produce embeddings.
    EMBEDDING_MODEL=BAAI/bge-m3     Preferred model for the high_accuracy
                                     benchmark strategy.

If the requested model cannot be loaded (missing dependency, out of memory,
no network access, ...), this module falls back to
``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`` and logs the
fallback explicitly -- it never fails silently. Documents and queries must
share one embedder instance (and therefore one model) so their vectors stay
comparable; callers are responsible for reusing the same instance.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from typing import Sequence

logger = logging.getLogger(__name__)

PREFERRED_MODEL = "BAAI/bge-m3"
FALLBACK_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL_ENV = "EMBEDDING_MODEL"
EMBEDDING_PROVIDER_ENV = "EMBEDDING_PROVIDER"
# How long to wait for a model download/load before treating it as "cannot be
# loaded" and falling back. Large models (e.g. BAAI/bge-m3, ~2.3GB) can hang
# indefinitely on a slow/blocked network instead of raising -- without this
# bound the documented fallback behaviour would never trigger in that case.
LOAD_TIMEOUT_S = float(os.getenv("EMBEDDING_LOAD_TIMEOUT_S", "120"))


class ModelLoadTimeoutError(RuntimeError):
    """Raised when loading a Sentence-Transformers model exceeds LOAD_TIMEOUT_S."""


class EmbeddingBackendError(RuntimeError):
    """Raised when neither the requested model nor the fallback model can be loaded."""


class LocalRAGEmbedder:
    """Sentence-Transformers embedder with explicit fallback and content-hash cache.

    Embeddings are L2-normalized so downstream similarity search can use a
    plain dot product as cosine similarity. Embeddings are cached by
    ``(model_name, sha256(text))`` so identical content is never re-embedded
    while the model stays the same.
    """

    def __init__(self, model_name: str | None = None) -> None:
        requested_model = model_name or os.getenv(EMBEDDING_MODEL_ENV, PREFERRED_MODEL)
        self._cache: dict[tuple[str, str], list[float]] = {}
        self.model, self.model_name, self.fallback_used = self._load_model(requested_model)
        self.requested_model_name = requested_model
        self._backend_name = self.model_name

        if self.fallback_used:
            logger.warning(
                "Requested embedding model '%s' could not be loaded; falling back to '%s'. "
                "(explicit fallback -- not silent)",
                requested_model,
                self.model_name,
            )
        else:
            logger.info("LocalRAGEmbedder loaded embedding model '%s'", self.model_name)

    @staticmethod
    def _load_sentence_transformer(model_name: str):
        """Load a SentenceTransformer, bounded by LOAD_TIMEOUT_S.

        A stalled download (e.g. a large model on a slow/blocked network) hangs
        rather than raising, so a plain try/except would never reach the
        fallback path. The load runs on a *daemon* thread: ``Thread.join(timeout=...)``
        lets us give up after LOAD_TIMEOUT_S without waiting for the load to
        finish, and being a daemon thread means it cannot block process exit
        even if the download never completes (unlike a ThreadPoolExecutor,
        whose context manager / atexit hook would join the worker and hang
        regardless of any timeout passed to ``future.result()``).
        """
        from sentence_transformers import SentenceTransformer

        outcome: dict[str, object] = {}

        def _worker() -> None:
            try:
                outcome["model"] = SentenceTransformer(model_name)
            except Exception as exc:  # noqa: BLE001 - re-raised on the caller's thread below
                outcome["error"] = exc

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()
        thread.join(timeout=LOAD_TIMEOUT_S)

        if thread.is_alive():
            raise ModelLoadTimeoutError(
                f"Loading '{model_name}' did not finish within {LOAD_TIMEOUT_S:.0f}s "
                "(likely a slow/blocked network for a large model download)"
            )
        if "error" in outcome:
            raise outcome["error"]  # type: ignore[misc]
        return outcome["model"]

    @classmethod
    def _load_model(cls, requested_model: str):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise EmbeddingBackendError(
                "sentence-transformers is not installed. Run: "
                "python -m pip install -r requirements-rag.txt"
            ) from exc

        try:
            return cls._load_sentence_transformer(requested_model), requested_model, False
        except Exception as exc:  # noqa: BLE001 - broad on purpose: any load failure triggers fallback
            if requested_model == FALLBACK_MODEL:
                raise EmbeddingBackendError(
                    f"Failed to load fallback embedding model '{FALLBACK_MODEL}': {exc}"
                ) from exc
            logger.warning(
                "Could not load embedding model '%s' (%s); trying fallback '%s'",
                requested_model,
                exc,
                FALLBACK_MODEL,
            )
            try:
                return cls._load_sentence_transformer(FALLBACK_MODEL), FALLBACK_MODEL, True
            except Exception as fallback_exc:
                raise EmbeddingBackendError(
                    f"Failed to load requested model '{requested_model}' ({exc}) and fallback "
                    f"model '{FALLBACK_MODEL}' ({fallback_exc})"
                ) from fallback_exc

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str) -> tuple[str, str]:
        return (self.model_name, self._content_hash(text))

    def embed_one(self, text: str) -> list[float]:
        key = self._cache_key(text)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        result = vector.tolist() if hasattr(vector, "tolist") else [float(v) for v in vector]
        self._cache[key] = result
        return result

    def embed_batch(self, texts: Sequence[str], batch_size: int = 32) -> list[list[float]]:
        """Batch-embed texts, reusing the content-hash cache for already-seen text."""
        results: list[list[float] | None] = [None] * len(texts)
        pending_indices: list[int] = []
        pending_texts: list[str] = []

        for index, text in enumerate(texts):
            cached = self._cache.get(self._cache_key(text))
            if cached is not None:
                results[index] = cached
            else:
                pending_indices.append(index)
                pending_texts.append(text)

        if pending_texts:
            vectors = self.model.encode(
                pending_texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for position, index in enumerate(pending_indices):
                vector = vectors[position]
                result = vector.tolist() if hasattr(vector, "tolist") else [float(v) for v in vector]
                self._cache[self._cache_key(texts[index])] = result
                results[index] = result

        return [result for result in results if result is not None]

    def __call__(self, text: str) -> list[float]:
        return self.embed_one(text)


__all__ = [
    "LocalRAGEmbedder",
    "EmbeddingBackendError",
    "ModelLoadTimeoutError",
    "PREFERRED_MODEL",
    "FALLBACK_MODEL",
    "EMBEDDING_MODEL_ENV",
    "EMBEDDING_PROVIDER_ENV",
    "LOAD_TIMEOUT_S",
]
