from .agent import KnowledgeBaseAgent
from .chunking import (
    ChunkingStrategyComparator,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    compute_similarity,
)
from .embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    MockEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from .models import Document
from .store import EmbeddingStore

# Advanced UIT RAG pipeline (Phase 2+). These modules only import optional
# heavy dependencies (sentence-transformers, rank_bm25, openai) lazily
# inside functions/methods, never at import time, so importing `src` never
# requires `requirements-rag.txt` to be installed.
from .deepseek_client import DeepSeekClient
from .rag_embeddings import LocalRAGEmbedder
from .rag_pipeline import UITRAGPipeline
from .retrieval import BM25Retriever, DenseRetriever, HybridRetriever, MetadataFilter
from .structure_chunking import StructureAwareChunker
from .uit_preprocessing import preprocess_uit_document

__all__ = [
    "Document",
    "FixedSizeChunker",
    "SentenceChunker",
    "RecursiveChunker",
    "ChunkingStrategyComparator",
    "compute_similarity",
    "EmbeddingStore",
    "KnowledgeBaseAgent",
    "MockEmbedder",
    "LocalEmbedder",
    "OpenAIEmbedder",
    "_mock_embed",
    "LOCAL_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "EMBEDDING_PROVIDER_ENV",
    # Advanced UIT RAG pipeline
    "UITRAGPipeline",
    "LocalRAGEmbedder",
    "DeepSeekClient",
    "DenseRetriever",
    "BM25Retriever",
    "MetadataFilter",
    "HybridRetriever",
    "StructureAwareChunker",
    "preprocess_uit_document",
]
