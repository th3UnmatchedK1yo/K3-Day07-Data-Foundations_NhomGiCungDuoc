"""End-to-end UIT RAG pipeline: preprocessing -> chunking -> hybrid retrieval -> DeepSeek.

Public API (kept stable for benchmark/CLI use)::

    class UITRAGPipeline:
        def build_index(self, source_path: str) -> None: ...
        def retrieve(self, query: str, top_k: int = 5) -> list: ...
        def answer(self, query: str) -> dict: ...

Also exposes a small CLI (``python -m src.rag_pipeline {build,retrieve,ask}``)
that persists the built index to a local (gitignored) cache file so
``retrieve``/``ask`` can run as separate process invocations without
re-embedding the corpus, and without ever requiring a DeepSeek API key
except for ``ask``.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

from .chunking import FixedSizeChunker
from .deepseek_client import DeepSeekClient, DeepSeekConfigurationError
from .models import Document
from .rag_embeddings import FALLBACK_MODEL, LocalRAGEmbedder
from .retrieval import HybridRetriever, RetrievedChunk
from .store import EmbeddingStore
from .structure_chunking import StructureAwareChunker
from .uit_preprocessing import DEFAULT_BOUNDARIES_PATH, load_procedure_boundaries, preprocess_uit_document

logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_INDEX_CACHE = Path(".rag_cache") / "uit_index.pkl"
CITATION_RE = re.compile(r"\[uit_student_procedures/[^/\]]+/[^\]]+\]")

BASELINE_CHUNK_SIZE = 500
BASELINE_CHUNK_OVERLAP = 50


def normalize_query(query: str) -> str:
    """Normalize a user query (unicode/whitespace) the same way the source document is normalized."""
    text = unicodedata.normalize("NFC", query.strip())
    text = text.translate({0x00A0: " ", 0xFEFF: ""})
    return re.sub(r"\s+", " ", text)


def _baseline_chunk_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Baseline chunking: plain FixedSizeChunker per procedure section (no header, no overlap tuning).

    Chunking is still scoped per procedure section so every chunk carries an
    unambiguous ``procedure_slug`` for evaluation, matching the lab's
    original ``FixedSizeChunker`` behavior as closely as possible while
    keeping the baseline vs. high_accuracy comparison meaningful.
    """
    chunker = FixedSizeChunker(chunk_size=BASELINE_CHUNK_SIZE, overlap=BASELINE_CHUNK_OVERLAP)
    chunks: list[dict[str, Any]] = []
    for section in sections:
        pieces = chunker.chunk(section["text"])
        source_id = section["metadata"].get("source_id", "uit_student_procedures")
        chunk_ids = [f"{source_id}::{section['procedure_slug']}::baseline_{i:03d}" for i in range(len(pieces))]
        for index, piece in enumerate(pieces):
            metadata = dict(section["metadata"])
            metadata.update(
                {
                    "chunk_id": chunk_ids[index],
                    "chunk_index": index,
                    "previous_chunk_id": chunk_ids[index - 1] if index > 0 else None,
                    "next_chunk_id": chunk_ids[index + 1] if index + 1 < len(chunk_ids) else None,
                    "content_hash": hashlib.sha256(piece.encode("utf-8")).hexdigest()[:16],
                }
            )
            chunks.append({"chunk_id": chunk_ids[index], "text": piece, "raw_text": piece, "metadata": metadata})
    return chunks


class UITRAGPipeline:
    """High-accuracy-by-default RAG pipeline for the UIT student-procedures document."""

    def __init__(
        self,
        embedder: Optional[LocalRAGEmbedder] = None,
        deepseek_client: Optional[DeepSeekClient] = None,
        use_bm25: bool = True,
        use_metadata_filter: bool = True,
        use_reranker: bool = False,
        use_adjacent_expansion: bool = True,
        enable_diversity: bool = True,
        chunker: str = "structure_aware",
        collection_name: str = "uit_student_procedures",
    ) -> None:
        if chunker not in ("structure_aware", "baseline"):
            raise ValueError(f"Unknown chunker kind: {chunker!r}. Expected 'structure_aware' or 'baseline'.")

        self._embedder = embedder
        self._deepseek_client = deepseek_client
        self._use_bm25 = use_bm25
        self._use_metadata_filter = use_metadata_filter
        self._use_reranker = use_reranker
        self._use_adjacent_expansion = use_adjacent_expansion
        self._enable_diversity = enable_diversity
        self._chunker_kind = chunker
        self._collection_name = collection_name

        self._store: Optional[EmbeddingStore] = None
        self._chunks: list[dict[str, Any]] = []
        self._retriever: Optional[HybridRetriever] = None
        self._boundaries = load_procedure_boundaries(DEFAULT_BOUNDARIES_PATH)

    @classmethod
    def for_strategy(
        cls,
        strategy: str,
        embedding_model: Optional[str] = None,
        use_reranker: bool = False,
    ) -> "UITRAGPipeline":
        """Factory matching the benchmark's baseline / high_accuracy strategy presets."""
        if strategy == "baseline":
            embedder = LocalRAGEmbedder(model_name=embedding_model or FALLBACK_MODEL)
            return cls(
                embedder=embedder,
                use_bm25=False,
                use_metadata_filter=False,
                use_reranker=False,
                use_adjacent_expansion=False,
                enable_diversity=False,
                chunker="baseline",
            )
        if strategy == "high_accuracy":
            embedder = LocalRAGEmbedder(model_name=embedding_model) if embedding_model else LocalRAGEmbedder()
            return cls(
                embedder=embedder,
                use_bm25=True,
                use_metadata_filter=True,
                use_reranker=use_reranker,
                use_adjacent_expansion=True,
                enable_diversity=True,
                chunker="structure_aware",
            )
        raise ValueError(f"Unknown strategy: {strategy!r}. Expected 'baseline' or 'high_accuracy'.")

    @property
    def embedding_model(self) -> str:
        if self._embedder is None:
            return "not-loaded"
        return getattr(self._embedder, "model_name", self._embedder.__class__.__name__)

    @property
    def chunks(self) -> list[dict[str, Any]]:
        return self._chunks

    def build_index(self, source_path: str) -> None:
        """Ingest, chunk, embed and index the UIT source document. Never calls DeepSeek."""
        start = time.perf_counter()
        if self._embedder is None:
            self._embedder = LocalRAGEmbedder()

        sections = preprocess_uit_document(source_path, DEFAULT_BOUNDARIES_PATH)
        if self._chunker_kind == "baseline":
            self._chunks = _baseline_chunk_sections(sections)
        else:
            chunk_objects = StructureAwareChunker().chunk_sections(sections)
            self._chunks = [chunk.to_dict() for chunk in chunk_objects]

        self._store = EmbeddingStore(collection_name=self._collection_name, embedding_fn=self._embedder)
        documents = [
            Document(id=chunk["chunk_id"], content=chunk["text"], metadata=chunk["metadata"])
            for chunk in self._chunks
        ]
        self._store.add_documents(documents)

        self._retriever = HybridRetriever(
            store=self._store,
            chunks=self._chunks,
            use_bm25=self._use_bm25,
            use_metadata_filter=self._use_metadata_filter,
            use_reranker=self._use_reranker,
            use_adjacent_expansion=self._use_adjacent_expansion,
            enable_diversity=self._enable_diversity,
            boundaries=self._boundaries,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "UITRAGPipeline index built: %d sections -> %d chunks in %.1fms (embedding_model=%s, chunker=%s)",
            len(sections),
            len(self._chunks),
            elapsed_ms,
            self.embedding_model,
            self._chunker_kind,
        )

    def _ensure_index(self) -> None:
        if self._retriever is None or self._store is None:
            raise RuntimeError("Index not built yet. Call build_index(source_path) first.")

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        """Run retrieval only (no DeepSeek call). Returns a list of chunk dicts with scores."""
        self._ensure_index()
        result = self._retriever.retrieve(normalize_query(query), top_k=top_k)
        return [chunk.to_dict() for chunk in result["chunks"]]

    def retrieve_detailed(self, query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Like retrieve(), but also exposes the pre-adjacent-expansion ranking and
        procedure-classification info. Used by bench.py for retrieval metrics."""
        self._ensure_index()
        result = self._retriever.retrieve(normalize_query(query), top_k=top_k)
        return {
            "chunks": [chunk.to_dict() for chunk in result["chunks"]],
            "primary_chunks": [chunk.to_dict() for chunk in result["primary_chunks"]],
            "procedure_slug": result["procedure_slug"],
            "procedure_confidence": result["procedure_confidence"],
            "metadata_filter_applied": result["metadata_filter_applied"],
            "reranker_used": result["reranker_used"],
        }

    @staticmethod
    def _build_context(chunks: list[RetrievedChunk]) -> str:
        blocks = []
        for chunk in chunks:
            meta = chunk.metadata
            citation_tag = f"[uit_student_procedures/{meta.get('procedure_slug')}/{chunk.chunk_id}]"
            blocks.append(
                f"### Quy trình: {meta.get('procedure_title')} (citation: {citation_tag})\n{chunk.raw_text}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _extract_citations(answer_text: str) -> list[str]:
        return CITATION_RE.findall(answer_text)

    def answer(self, query: str, top_k: int = DEFAULT_TOP_K) -> dict[str, Any]:
        """Retrieve context, call DeepSeek, and return a grounded answer with citations."""
        self._ensure_index()
        normalized_query = normalize_query(query)

        retrieval_start = time.perf_counter()
        retrieval_result = self._retriever.retrieve(normalized_query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        chunks: list[RetrievedChunk] = retrieval_result["chunks"]

        if self._deepseek_client is None:
            self._deepseek_client = DeepSeekClient()

        context = self._build_context(chunks)
        generation_start = time.perf_counter()
        answer_text = self._deepseek_client.generate(normalized_query, context)
        generation_ms = (time.perf_counter() - generation_start) * 1000

        return {
            "answer": answer_text,
            "citations": self._extract_citations(answer_text),
            "retrieved_chunks": [chunk.to_dict() for chunk in chunks],
            "model": self._deepseek_client.model,
            "embedding_model": self.embedding_model,
            "procedure_slug": retrieval_result["procedure_slug"],
            "procedure_confidence": retrieval_result["procedure_confidence"],
            "metadata_filter_applied": retrieval_result["metadata_filter_applied"],
            "reranker_used": retrieval_result["reranker_used"],
            "latency_ms": {
                "retrieval": round(retrieval_ms, 2),
                "generation": round(generation_ms, 2),
                "total": round(retrieval_ms + generation_ms, 2),
            },
        }

    def save_index(self, path: str | Path) -> None:
        """Persist the built index (chunks + embeddings + config) to a local pickle file."""
        self._ensure_index()
        payload = {
            "chunks": self._chunks,
            "records": self._store._store,  # in-memory EmbeddingStore records
            "embedding_model": self.embedding_model,
            "chunker_kind": self._chunker_kind,
            "use_bm25": self._use_bm25,
            "use_metadata_filter": self._use_metadata_filter,
            "use_reranker": self._use_reranker,
            "use_adjacent_expansion": self._use_adjacent_expansion,
            "enable_diversity": self._enable_diversity,
            "collection_name": self._collection_name,
        }
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(payload, handle)

    @classmethod
    def load_index(cls, path: str | Path) -> "UITRAGPipeline":
        """Reload a previously saved index without recomputing embeddings."""
        index_path = Path(path)
        if not index_path.exists():
            raise FileNotFoundError(
                f"No index found at {index_path}. Run `python -m src.rag_pipeline build --source ...` first."
            )
        with index_path.open("rb") as handle:
            payload = pickle.load(handle)

        embedder = LocalRAGEmbedder(model_name=payload["embedding_model"])
        pipeline = cls(
            embedder=embedder,
            use_bm25=payload["use_bm25"],
            use_metadata_filter=payload["use_metadata_filter"],
            use_reranker=payload["use_reranker"],
            use_adjacent_expansion=payload["use_adjacent_expansion"],
            enable_diversity=payload.get("enable_diversity", True),
            chunker=payload["chunker_kind"],
            collection_name=payload["collection_name"],
        )
        pipeline._chunks = payload["chunks"]
        store = EmbeddingStore(collection_name=pipeline._collection_name, embedding_fn=embedder)
        store._store = payload["records"]
        pipeline._store = store
        pipeline._retriever = HybridRetriever(
            store=store,
            chunks=pipeline._chunks,
            use_bm25=pipeline._use_bm25,
            use_metadata_filter=pipeline._use_metadata_filter,
            use_reranker=pipeline._use_reranker,
            use_adjacent_expansion=pipeline._use_adjacent_expansion,
            enable_diversity=pipeline._enable_diversity,
            boundaries=pipeline._boundaries,
        )
        return pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.rag_pipeline",
        description="UIT RAG pipeline CLI (build index / retrieve / ask DeepSeek).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build and persist the retrieval index")
    build_parser.add_argument("--source", required=True, help="Path to the UIT markdown source file")
    build_parser.add_argument("--strategy", choices=["baseline", "high_accuracy"], default="high_accuracy")
    build_parser.add_argument("--embedding-model", default=None, help="Override the embedding model name")
    build_parser.add_argument("--reranker", action="store_true", help="Enable the optional local reranker")
    build_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_CACHE))

    retrieve_parser = subparsers.add_parser("retrieve", help="Run retrieval only (no DeepSeek call needed)")
    retrieve_parser.add_argument("--query", required=True)
    retrieve_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    retrieve_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_CACHE))

    ask_parser = subparsers.add_parser("ask", help="Run retrieval + DeepSeek generation (needs DEEPSEEK_API_KEY)")
    ask_parser.add_argument("--query", required=True)
    ask_parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    ask_parser.add_argument("--index-path", default=str(DEFAULT_INDEX_CACHE))

    return parser


def _ensure_utf8_console() -> None:
    """Best-effort: avoid UnicodeEncodeError when printing Vietnamese text on
    Windows consoles whose default codepage is not UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _load_dotenv_if_available() -> None:
    """Load ``.env`` (DEEPSEEK_API_KEY, EMBEDDING_MODEL, ...) if python-dotenv is
    installed, mirroring ``main.py``'s ``load_dotenv(override=False)`` behaviour.
    Never overrides variables already exported in the shell."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=False)


def main(argv: Optional[list[str]] = None) -> int:
    _ensure_utf8_console()
    _load_dotenv_if_available()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        if args.command == "build":
            pipeline = UITRAGPipeline.for_strategy(
                args.strategy, embedding_model=args.embedding_model, use_reranker=args.reranker
            )
            pipeline.build_index(args.source)
            pipeline.save_index(args.index_path)
            print(
                f"OK: index built with {len(pipeline.chunks)} chunks "
                f"(strategy={args.strategy}, embedding_model={pipeline.embedding_model})"
            )
            print(f"Saved index to: {args.index_path}")
            return 0

        if args.command == "retrieve":
            pipeline = UITRAGPipeline.load_index(args.index_path)
            results = pipeline.retrieve(args.query, top_k=args.top_k)
            if not results:
                print("No results.")
            for rank, chunk in enumerate(results, start=1):
                meta = chunk["metadata"]
                preview = chunk["raw_text"][:200].replace("\n", " ")
                print(f"{rank}. score={chunk['score']:.4f} chunk_id={chunk['chunk_id']} procedure={meta.get('procedure_slug')}")
                print(f"   {preview}")
            return 0

        if args.command == "ask":
            pipeline = UITRAGPipeline.load_index(args.index_path)
            result = pipeline.answer(args.query, top_k=args.top_k)
            print(result["answer"])
            print()
            print(f"Citations: {result['citations']}")
            print(
                f"Model: {result['model']} | Embedding: {result['embedding_model']} | "
                f"Latency(ms): {result['latency_ms']}"
            )
            return 0

    except (FileNotFoundError, DeepSeekConfigurationError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["UITRAGPipeline", "normalize_query", "DEFAULT_TOP_K", "DEFAULT_INDEX_CACHE"]
