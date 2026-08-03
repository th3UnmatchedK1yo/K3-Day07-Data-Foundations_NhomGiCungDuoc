from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # Dung in-memory client de lab/test khong tao file phu tren may nguoi hoc.
            client = chromadb.Client()
            self._collection = client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # Metadata duoc copy de tranh sua ngoai y muon tu object Document goc.
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": dict(doc.metadata or {}),
            "embedding": self._embedding_fn(doc.content),
            "index": self._next_index,
        }

    def _format_record(self, record: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": record["id"],
            "content": record["content"],
            "metadata": record["metadata"],
            "score": score,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        query_embedding = self._embedding_fn(query)
        scored = [
            self._format_record(record, _dot(query_embedding, record["embedding"]))
            for record in records
        ]
        # Sort theo score giam dan; index giu thu tu them vao khi score bang nhau.
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[: max(0, top_k)]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        records: list[dict[str, Any]] = []
        for doc in docs:
            record = self._make_record(doc)
            records.append(record)
            self._store.append(record)
            self._next_index += 1

        if self._use_chroma and self._collection is not None and records:
            try:
                self._collection.add(
                    ids=[record["id"] for record in records],
                    documents=[record["content"] for record in records],
                    metadatas=[record["metadata"] for record in records],
                    embeddings=[record["embedding"] for record in records],
                )
            except Exception:
                # Neu Chroma loi do trung id/schema, in-memory store van la source of truth.
                self._use_chroma = False

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k=top_k)

        # Loc metadata truoc khi tinh similarity de thu hep tap ung vien retrieval.
        filtered_records = [
            record
            for record in self._store
            if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        before = len(self._store)
        self._store = [
            record
            for record in self._store
            if record["id"] != doc_id and record["metadata"].get("doc_id") != doc_id
        ]
        removed = len(self._store) != before

        if removed and self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(where={"doc_id": doc_id})
                self._collection.delete(ids=[doc_id])
            except Exception:
                self._use_chroma = False
        return removed
