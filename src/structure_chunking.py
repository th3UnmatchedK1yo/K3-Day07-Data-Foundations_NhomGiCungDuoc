"""Structure-aware chunking for the UIT student-procedures document.

Unlike the generic lab chunkers (``FixedSizeChunker``, ``SentenceChunker``,
``RecursiveChunker`` in ``src/chunking.py``), :class:`StructureAwareChunker`
is designed for already procedure-scoped text produced by
``src.uit_preprocessing.preprocess_uit_document``. It never mixes content
from two different procedures into one chunk, and it tries hard not to
separate a numbered step ("Bước N") from the sentence(s) describing it, a
bullet condition from its own text, or a form name ("Mẫu 07") from the
clause that explains it.

Token counting
---------------
If the optional ``tiktoken`` package is installed, chunk sizes are measured
in real GPT-style tokens (``cl100k_base`` encoding). Otherwise a stable
estimator is used: ``tokens ≈ round(word_count * 1.5)``, since Vietnamese
syllables are usually split into 1-2 sub-word tokens by BPE tokenizers.
This estimator is deliberately simple and documented here so benchmark
numbers can be interpreted correctly when ``tiktoken`` is unavailable.
The tokenizer (if any) is lazy-loaded on first use, never at import time.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

DEFAULT_TARGET_TOKENS = 450
DEFAULT_MAX_TOKENS = 650
DEFAULT_OVERLAP_TOKENS = 80

DOCUMENT_LABEL = "Một số quy trình dành cho sinh viên"

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

_token_counter_cache: dict[str, Callable[[str], int]] = {}


def _word_count_estimator(text: str) -> int:
    """Stable fallback token estimator: ~1.5 tokens per whitespace-separated word."""
    words = text.split()
    return max(1, round(len(words) * 1.5))


def _get_token_counter() -> Callable[[str], int]:
    """Lazily resolve a token counter, preferring tiktoken if installed."""
    if "counter" in _token_counter_cache:
        return _token_counter_cache["counter"]

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        counter: Callable[[str], int] = lambda text: len(encoding.encode(text))
        logger.info("StructureAwareChunker: using tiktoken cl100k_base for token counting")
    except Exception:
        counter = _word_count_estimator
        logger.info(
            "StructureAwareChunker: tiktoken not available, using word-count estimator "
            "(tokens ~= words * 1.5)"
        )

    _token_counter_cache["counter"] = counter
    return counter


def _split_into_paragraphs(text: str) -> list[str]:
    """Split on blank lines. Each UIT step/bullet already sits in its own paragraph."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paragraphs


def _split_into_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text.strip()) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _split_by_whitespace(text: str, max_tokens: int, count_tokens: Callable[[str], int]) -> list[str]:
    """Last-resort split (no heading/paragraph/sentence boundary left to use).

    Greedily accumulates whitespace-separated words until ``max_tokens`` is
    reached. Always makes forward progress (unlike re-attempting sentence
    splitting on text with no sentence boundary), so it never recurses.
    """
    words = text.split()
    if not words:
        return []
    pieces: list[str] = []
    current_words: list[str] = []
    for word in words:
        current_words.append(word)
        if count_tokens(" ".join(current_words)) >= max_tokens:
            pieces.append(" ".join(current_words))
            current_words = []
    if current_words:
        pieces.append(" ".join(current_words))
    return pieces or [text]


@dataclass(frozen=True)
class Chunk:
    """A single retrieval-ready chunk with header-augmented text and clean raw_text."""

    chunk_id: str
    text: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "raw_text": self.raw_text,
            "metadata": dict(self.metadata),
        }


class StructureAwareChunker:
    """Procedure-scoped, boundary-aware chunker for the UIT source document.

    Boundary priority (highest first): heading/procedure boundary (handled
    upstream by ``uit_preprocessing``) > paragraph (a step or bullet in this
    source) > sentence > whitespace (last resort for a single oversized
    "sentence" with no punctuation).
    """

    def __init__(
        self,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
        document_label: str = DOCUMENT_LABEL,
    ) -> None:
        if target_tokens <= 0 or max_tokens <= 0:
            raise ValueError("target_tokens and max_tokens must be positive")
        if max_tokens < target_tokens:
            raise ValueError("max_tokens must be >= target_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = max(0, overlap_tokens)
        self.document_label = document_label

    def _count_tokens(self, text: str) -> int:
        return _get_token_counter()(text)

    def _pack_units(self, units: list[str]) -> list[list[str]]:
        """Greedily pack atomic units into chunks bounded by target/max tokens."""
        chunks: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._count_tokens(unit)

            if unit_tokens > self.max_tokens:
                if current:
                    chunks.append(current)
                    current, current_tokens = [], 0
                sentences = _split_into_sentences(unit)
                if len(sentences) <= 1:
                    # No internal sentence boundary (e.g. one long sentence with no
                    # ".!?"): splitting into "sentences" would just return the same
                    # unit again, which would recurse forever. Fall back to the
                    # whitespace level instead -- this always makes forward progress.
                    whitespace_pieces = _split_by_whitespace(unit, self.max_tokens, self._count_tokens)
                    chunks.extend([[piece] for piece in whitespace_pieces])
                else:
                    chunks.extend(self._pack_units(sentences))
                continue

            if current and current_tokens + unit_tokens > self.max_tokens:
                chunks.append(current)
                current, current_tokens = [], 0

            current.append(unit)
            current_tokens += unit_tokens

            if current_tokens >= self.target_tokens:
                chunks.append(current)
                current, current_tokens = [], 0

        if current:
            chunks.append(current)
        return chunks

    def _apply_overlap(self, chunk_units: list[list[str]]) -> list[list[str]]:
        """Prepend trailing units of the previous chunk (~overlap_tokens) to each chunk."""
        if self.overlap_tokens == 0 or len(chunk_units) <= 1:
            return chunk_units

        overlapped: list[list[str]] = [chunk_units[0]]
        for index in range(1, len(chunk_units)):
            previous_units = chunk_units[index - 1]
            overlap_units: list[str] = []
            accumulated = 0
            for unit in reversed(previous_units):
                unit_tokens = self._count_tokens(unit)
                if overlap_units and accumulated + unit_tokens > self.overlap_tokens:
                    break
                overlap_units.insert(0, unit)
                accumulated += unit_tokens
                if accumulated >= self.overlap_tokens:
                    break
            overlapped.append(overlap_units + chunk_units[index])
        return overlapped

    def _build_header(self, procedure_title: str) -> str:
        return f"Tài liệu: {self.document_label}\nQuy trình: {procedure_title}"

    def chunk_section(self, section: dict[str, Any]) -> list[Chunk]:
        """Chunk a single procedure section (as produced by uit_preprocessing)."""
        procedure_slug = section["procedure_slug"]
        procedure_title = section["procedure_title"]
        section_index = section["section_index"]
        base_metadata = dict(section.get("metadata", {}))
        source_id = base_metadata.get("source_id", "uit_student_procedures")

        paragraphs = _split_into_paragraphs(section["text"])
        if not paragraphs:
            return []

        packed = self._pack_units(paragraphs)
        packed = self._apply_overlap(packed)

        header = self._build_header(procedure_title)
        chunks: list[Chunk] = []
        chunk_ids = [f"{source_id}::{procedure_slug}::chunk_{i:03d}" for i in range(len(packed))]

        for chunk_index, units in enumerate(packed):
            raw_text = "\n\n".join(units)
            embedding_text = f"{header}\n\n{raw_text}"
            content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
            metadata = dict(base_metadata)
            metadata.update(
                {
                    "chunk_id": chunk_ids[chunk_index],
                    "source_id": source_id,
                    "procedure_slug": procedure_slug,
                    "procedure_title": procedure_title,
                    "section_index": section_index,
                    "chunk_index": chunk_index,
                    "previous_chunk_id": chunk_ids[chunk_index - 1] if chunk_index > 0 else None,
                    "next_chunk_id": chunk_ids[chunk_index + 1] if chunk_index + 1 < len(chunk_ids) else None,
                    "content_hash": content_hash,
                    "token_count": self._count_tokens(raw_text),
                }
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_ids[chunk_index],
                    text=embedding_text,
                    raw_text=raw_text,
                    metadata=metadata,
                )
            )
        return chunks

    def chunk_sections(self, sections: list[dict[str, Any]]) -> list[Chunk]:
        """Chunk every procedure section independently (never mixes procedures)."""
        all_chunks: list[Chunk] = []
        for section in sections:
            all_chunks.extend(self.chunk_section(section))
        logger.info(
            "StructureAwareChunker produced %d chunks from %d procedure sections",
            len(all_chunks),
            len(sections),
        )
        return all_chunks


__all__ = ["StructureAwareChunker", "Chunk", "DEFAULT_TARGET_TOKENS", "DEFAULT_MAX_TOKENS", "DEFAULT_OVERLAP_TOKENS"]
