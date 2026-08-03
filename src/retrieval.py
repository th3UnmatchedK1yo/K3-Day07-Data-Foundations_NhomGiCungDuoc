"""Hybrid retrieval for the UIT RAG pipeline.

Implements the query-side pipeline described in docs/ARCHITECTURE_FLOW.md:

    query -> normalize -> procedure classification (confidence-gated)
          -> dense retrieval + BM25 retrieval -> Reciprocal Rank Fusion
          -> optional reranker -> dedup -> adjacent chunk expansion

Every piece is usable independently (``DenseRetriever``, ``BM25Retriever``,
``MetadataFilter``, ``reciprocal_rank_fusion``, ``Reranker``,
``deduplicate``, ``expand_adjacent``) and is also wired together by
``HybridRetriever`` for convenience.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Optional

from .store import EmbeddingStore
from .uit_preprocessing import DEFAULT_BOUNDARIES_PATH, load_procedure_boundaries

logger = logging.getLogger(__name__)

DEFAULT_DENSE_TOP_K = 20
DEFAULT_BM25_TOP_K = 20
DEFAULT_RRF_K = 60
DEFAULT_METADATA_FILTER_CONFIDENCE = 0.6
DEFAULT_CONTEXT_TOKEN_BUDGET = 1600

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize_vi(text: str) -> list[str]:
    """Minimal Vietnamese-aware tokenizer: lowercase + NFC, keep digits/form codes."""
    normalized = unicodedata.normalize("NFC", text).lower()
    return _TOKEN_RE.findall(normalized)


def _approx_tokens(text: str) -> int:
    return max(1, round(len(text.split()) * 1.5))


@dataclass
class RetrievedChunk:
    """A single retrieval candidate flowing through the hybrid pipeline."""

    chunk_id: str
    content: str
    raw_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    source: str = "dense"

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "raw_text": self.raw_text,
            "metadata": dict(self.metadata),
            "score": self.score,
            "source": self.source,
        }


class DenseRetriever:
    """Thin wrapper around ``EmbeddingStore`` for dense (embedding) retrieval."""

    def __init__(self, store: EmbeddingStore) -> None:
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_DENSE_TOP_K,
        metadata_filter: Optional[dict[str, Any]] = None,
    ) -> list[RetrievedChunk]:
        if metadata_filter:
            raw_results = self._store.search_with_filter(query, top_k=top_k, metadata_filter=metadata_filter)
        else:
            raw_results = self._store.search(query, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=result["metadata"].get("chunk_id", result.get("id", "")),
                content=result["content"],
                raw_text=result["metadata"].get("raw_text", result["content"]),
                metadata=result["metadata"],
                score=float(result["score"]),
                source="dense",
            )
            for result in raw_results
        ]


class BM25Retriever:
    """BM25 lexical retrieval over normalized chunk text (basic Vietnamese handling)."""

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank_bm25 is required for BM25Retriever. Install it with: "
                "python -m pip install -r requirements-rag.txt"
            ) from exc

        self._chunks = chunks
        tokenized_corpus = [_tokenize_vi(chunk["raw_text"]) for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def retrieve(self, query: str, top_k: int = DEFAULT_BM25_TOP_K) -> list[RetrievedChunk]:
        if not self._chunks or self._bm25 is None:
            return []
        tokenized_query = _tokenize_vi(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], self._chunks[i]["chunk_id"]))
        results = []
        for index in ranked_indices[:top_k]:
            if scores[index] <= 0:
                continue
            chunk = self._chunks[index]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk["chunk_id"],
                    content=chunk["text"],
                    raw_text=chunk["raw_text"],
                    metadata=chunk["metadata"],
                    score=float(scores[index]),
                    source="bm25",
                )
            )
        return results


class MetadataFilter:
    """Infers a likely ``procedure_slug`` for a query from keyword mapping.

    The keyword -> procedure mapping is loaded from
    ``data/uit/procedure_boundaries.json`` (``query_keywords`` per
    procedure) instead of being hard-coded in Python.
    """

    def __init__(self, boundaries: Optional[dict[str, Any]] = None) -> None:
        boundaries = boundaries or load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH)
        self._keyword_map: list[tuple[str, list[str]]] = [
            (procedure["slug"], [kw.lower() for kw in procedure.get("query_keywords", [])])
            for procedure in boundaries["procedures"]
        ]

    def classify(self, query: str) -> tuple[Optional[str], float]:
        """Return ``(procedure_slug_or_None, confidence)`` with confidence in [0, 1]."""
        normalized_query = unicodedata.normalize("NFC", query).lower()
        hits_by_slug: dict[str, int] = {}
        for slug, keywords in self._keyword_map:
            hits = sum(1 for keyword in keywords if keyword in normalized_query)
            if hits:
                hits_by_slug[slug] = hits

        if not hits_by_slug:
            return None, 0.0

        best_slug = max(hits_by_slug, key=lambda slug: (hits_by_slug[slug], slug))
        best_hits = hits_by_slug[best_slug]
        total_hits = sum(hits_by_slug.values())

        dominance = best_hits / total_hits  # how much this procedure "owns" the matched keywords
        magnitude = min(1.0, best_hits / 2)  # more than 1 distinct keyword match increases confidence
        confidence = round(0.5 * dominance + 0.5 * magnitude, 4)
        return best_slug, confidence

    @staticmethod
    def should_filter(confidence: float, threshold: float = DEFAULT_METADATA_FILTER_CONFIDENCE) -> bool:
        """Ambiguous queries (low confidence) should NOT be hard-filtered -- it can drop correct hits."""
        return confidence >= threshold


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]], k: int = DEFAULT_RRF_K
) -> list[RetrievedChunk]:
    """Deterministic Reciprocal Rank Fusion across multiple ranked candidate lists."""
    scores: dict[str, float] = {}
    representative: dict[str, RetrievedChunk] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            representative.setdefault(chunk.chunk_id, chunk)

    fused = [
        RetrievedChunk(
            chunk_id=chunk_id,
            content=representative[chunk_id].content,
            raw_text=representative[chunk_id].raw_text,
            metadata=representative[chunk_id].metadata,
            score=score,
            source="fused",
        )
        for chunk_id, score in scores.items()
    ]
    # Deterministic ordering: score desc, chunk_id asc as a stable tie-break.
    fused.sort(key=lambda chunk: (-chunk.score, chunk.chunk_id))
    return fused


class RerankerUnavailableError(RuntimeError):
    """Raised when the optional cross-encoder reranker cannot be loaded."""


class Reranker:
    """Optional cross-encoder reranker (default: BAAI/bge-reranker-v2-m3).

    Never crashes the pipeline on its own: construction raises
    ``RerankerUnavailableError`` if the dependency/model is missing, and
    callers are expected to catch it and continue with the RRF ranking.
    """

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerUnavailableError(f"sentence-transformers CrossEncoder unavailable: {exc}") from exc
        try:
            self._model = CrossEncoder(model_name)
        except Exception as exc:  # noqa: BLE001 - any load failure means "unavailable"
            raise RerankerUnavailableError(f"Could not load reranker model '{model_name}': {exc}") from exc
        self.model_name = model_name

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int = 5) -> list[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, chunk.raw_text) for chunk in chunks]
        raw_scores = self._model.predict(pairs)
        scored = sorted(
            zip(chunks, raw_scores), key=lambda item: (-float(item[1]), item[0].chunk_id)
        )
        return [
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                raw_text=chunk.raw_text,
                metadata=chunk.metadata,
                score=float(score),
                source="reranked",
            )
            for chunk, score in scored[:top_k]
        ]


def deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks with a repeated content_hash, keeping the first (highest-ranked) copy."""
    seen: set[str] = set()
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        key = chunk.metadata.get("content_hash") or chunk.chunk_id
        if key in seen:
            continue
        seen.add(key)
        deduped.append(chunk)
    return deduped


def diversify(chunks: list[RetrievedChunk], top_k: int, score_epsilon: float = 0.02) -> list[RetrievedChunk]:
    """Among near-tied candidates, prefer results from a procedure/section not already picked."""
    if not chunks:
        return []
    result = [chunks[0]]
    seen_slugs = {chunks[0].metadata.get("procedure_slug")}
    remaining = list(chunks[1:])

    while remaining and len(result) < top_k:
        best_score = remaining[0].score
        close_band = [c for c in remaining if best_score - c.score <= score_epsilon]
        pick = next((c for c in close_band if c.metadata.get("procedure_slug") not in seen_slugs), close_band[0])
        result.append(pick)
        seen_slugs.add(pick.metadata.get("procedure_slug"))
        remaining.remove(pick)
    return result


def expand_adjacent(
    chunks: list[RetrievedChunk],
    chunk_lookup: dict[str, dict[str, Any]],
    max_total_tokens: int = DEFAULT_CONTEXT_TOKEN_BUDGET,
) -> list[RetrievedChunk]:
    """Pull in previous/next chunks of the same procedure while staying within a token budget."""
    expanded: list[RetrievedChunk] = []
    used_ids = {chunk.chunk_id for chunk in chunks}
    total_tokens = sum(_approx_tokens(chunk.raw_text) for chunk in chunks)

    for chunk in chunks:
        expanded.append(chunk)
        for neighbor_key in ("previous_chunk_id", "next_chunk_id"):
            neighbor_id = chunk.metadata.get(neighbor_key)
            if not neighbor_id or neighbor_id in used_ids:
                continue
            neighbor_record = chunk_lookup.get(neighbor_id)
            if not neighbor_record:
                continue
            if neighbor_record["metadata"].get("procedure_slug") != chunk.metadata.get("procedure_slug"):
                continue
            neighbor_tokens = _approx_tokens(neighbor_record["raw_text"])
            if total_tokens + neighbor_tokens > max_total_tokens:
                continue
            used_ids.add(neighbor_id)
            total_tokens += neighbor_tokens
            expanded.append(
                RetrievedChunk(
                    chunk_id=neighbor_id,
                    content=neighbor_record["text"],
                    raw_text=neighbor_record["raw_text"],
                    metadata=neighbor_record["metadata"],
                    score=chunk.score,
                    source="adjacent",
                )
            )
    return expanded


class HybridRetriever:
    """Wires dense + BM25 + RRF + optional metadata filter/reranker/adjacent expansion.

    Configuration flags map directly to the baseline vs. high_accuracy
    strategies compared by ``bench.py``.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        chunks: list[dict[str, Any]],
        use_bm25: bool = True,
        use_metadata_filter: bool = True,
        use_reranker: bool = False,
        use_adjacent_expansion: bool = True,
        enable_diversity: bool = True,
        dense_top_k: int = DEFAULT_DENSE_TOP_K,
        bm25_top_k: int = DEFAULT_BM25_TOP_K,
        metadata_filter_confidence: float = DEFAULT_METADATA_FILTER_CONFIDENCE,
        boundaries: Optional[dict[str, Any]] = None,
    ) -> None:
        self._chunk_lookup = {chunk["chunk_id"]: chunk for chunk in chunks}
        self._dense = DenseRetriever(store)
        self._bm25 = BM25Retriever(chunks) if use_bm25 else None
        self._metadata_filter = MetadataFilter(boundaries) if use_metadata_filter else None
        self._use_adjacent_expansion = use_adjacent_expansion
        self._enable_diversity = enable_diversity
        self.dense_top_k = dense_top_k
        self.bm25_top_k = bm25_top_k
        self.metadata_filter_confidence = metadata_filter_confidence

        self._reranker: Optional[Reranker] = None
        self.reranker_requested = use_reranker
        self.reranker_active = False
        if use_reranker:
            try:
                self._reranker = Reranker()
                self.reranker_active = True
            except RerankerUnavailableError as exc:
                logger.warning("Reranker requested but unavailable (%s); continuing without it.", exc)

    def retrieve(self, query: str, top_k: int = 5) -> dict[str, Any]:
        procedure_slug: Optional[str] = None
        confidence = 0.0
        metadata_filter: Optional[dict[str, Any]] = None

        if self._metadata_filter is not None:
            procedure_slug, confidence = self._metadata_filter.classify(query)
            if procedure_slug and self._metadata_filter.should_filter(confidence, self.metadata_filter_confidence):
                metadata_filter = {"procedure_slug": procedure_slug}

        dense_results = self._dense.retrieve(query, top_k=self.dense_top_k, metadata_filter=metadata_filter)

        bm25_results: list[RetrievedChunk] = []
        if self._bm25 is not None:
            bm25_results = self._bm25.retrieve(query, top_k=self.bm25_top_k)
            if metadata_filter:
                bm25_results = [c for c in bm25_results if c.metadata.get("procedure_slug") == procedure_slug]

        fused = reciprocal_rank_fusion([dense_results, bm25_results]) if bm25_results else dense_results
        fused = deduplicate(fused)

        if self.reranker_active and self._reranker is not None and fused:
            candidate_pool = fused[:10]
            primary = self._reranker.rerank(query, candidate_pool, top_k=top_k)
        elif self._enable_diversity:
            primary = diversify(fused, top_k)
        else:
            primary = fused[:top_k]

        final = expand_adjacent(primary, self._chunk_lookup) if self._use_adjacent_expansion else primary

        return {
            # `chunks`: primary ranking + adjacent expansion, used for context building / answers.
            "chunks": final,
            # `primary_chunks`: ranking BEFORE adjacent expansion, used for retrieval metrics
            # (Hit@1/Recall@k/MRR@5 should reflect ranking quality, not neighbor padding).
            "primary_chunks": primary,
            "procedure_slug": procedure_slug,
            "procedure_confidence": confidence,
            "metadata_filter_applied": metadata_filter is not None,
            "reranker_used": self.reranker_active and self._reranker is not None,
        }


__all__ = [
    "RetrievedChunk",
    "DenseRetriever",
    "BM25Retriever",
    "MetadataFilter",
    "reciprocal_rank_fusion",
    "Reranker",
    "RerankerUnavailableError",
    "deduplicate",
    "diversify",
    "expand_adjacent",
    "HybridRetriever",
    "DEFAULT_DENSE_TOP_K",
    "DEFAULT_BM25_TOP_K",
    "DEFAULT_RRF_K",
    "DEFAULT_METADATA_FILTER_CONFIDENCE",
    "DEFAULT_CONTEXT_TOKEN_BUDGET",
]
