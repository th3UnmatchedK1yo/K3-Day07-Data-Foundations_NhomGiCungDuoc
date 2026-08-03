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
            import chromadb  # noqa: F401

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build a normalized stored record (id/content/metadata/embedding) for one document."""
        embedding = self._embedding_fn(doc.content)
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": dict(doc.metadata),
            "embedding": [float(value) for value in embedding],
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run an in-memory similarity search over the provided records."""
        if not records:
            return []

        query_embedding = self._embedding_fn(query)
        scored = [
            {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": _dot(query_embedding, record["embedding"]),
            }
            for record in records
        ]
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]

    def _all_records(self) -> list[dict[str, Any]]:
        """Return every stored record regardless of backend (Chroma or in-memory)."""
        if self._use_chroma and self._collection is not None:
            raw = self._collection.get(include=["documents", "metadatas", "embeddings"])
            ids = raw.get("ids") or []
            documents = raw.get("documents") or []
            metadatas = raw.get("metadatas") or []
            embeddings = raw.get("embeddings") or []
            records = []
            for index, record_id in enumerate(ids):
                records.append(
                    {
                        "id": record_id,
                        "content": documents[index] if index < len(documents) else "",
                        "metadata": dict(metadatas[index]) if index < len(metadatas) and metadatas[index] else {},
                        "embedding": list(embeddings[index]) if index < len(embeddings) else [],
                    }
                )
            return records
        return self._store

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        for doc in docs:
            record = self._make_record(doc)
            if self._use_chroma and self._collection is not None:
                self._collection.add(
                    ids=[record["id"]],
                    documents=[record["content"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"] or {}],
                )
            else:
                self._store.append(record)
            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        return self._search_records(query, self._all_records(), top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        records = self._all_records()
        if metadata_filter:
            records = [
                record
                for record in records
                if all(record["metadata"].get(key) == value for key, value in metadata_filter.items())
            ]
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        def _belongs_to(record: dict[str, Any]) -> bool:
            # Chunks created via ingest.py carry an explicit metadata['doc_id'];
            # standalone documents (no chunking) fall back to matching their own id.
            return record["metadata"].get("doc_id", record["id"]) == doc_id

        if self._use_chroma and self._collection is not None:
            matching_ids = [record["id"] for record in self._all_records() if _belongs_to(record)]
            if not matching_ids:
                return False
            self._collection.delete(ids=matching_ids)
            return True

        size_before = len(self._store)
        self._store = [record for record in self._store if not _belongs_to(record)]
        return len(self._store) < size_before
