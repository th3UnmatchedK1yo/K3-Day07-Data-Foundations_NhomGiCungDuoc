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

            # Khởi tạo ChromaDB nếu có.
            client = chromadb.Client()

            self._collection = client.get_or_create_collection(
                name=collection_name
            )

            self._use_chroma = True

        except Exception:
            # Nếu ChromaDB không hoạt động thì dùng
            # in-memory store.
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """
        Convert a Document into a normalized stored record.
        """

        embedding = self._embedding_fn(doc.content)

        metadata = dict(doc.metadata or {})

        # Lưu doc_id vào metadata để delete_document()
        # có thể tìm và xóa toàn bộ chunk của document.
        metadata.setdefault("doc_id", doc.id)

        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": metadata,
            "embedding": embedding,
        }

    def _search_records(
        self,
        query: str,
        records: list[dict[str, Any]],
        top_k: int
    ) -> list[dict[str, Any]]:
        """
        Run in-memory similarity search over provided records.
        """

        if not records or top_k <= 0:
            return []

        query_embedding = self._embedding_fn(query)

        scored_records = []

        for record in records:
            embedding = record["embedding"]

            # Với mock embedding của lab,
            # sử dụng dot product theo yêu cầu đề bài.
            score = _dot(
                query_embedding,
                embedding
            )

            result = {
                "id": record["id"],
                "content": record["content"],
                "metadata": record["metadata"],
                "score": score,
            }

            scored_records.append(result)

        # Điểm cao nhất đứng trước.
        scored_records.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(...)
        For in-memory: append dicts to self._store.
        """

        if not docs:
            return

        records = [
            self._make_record(doc)
            for doc in docs
        ]

        if self._use_chroma and self._collection is not None:
            try:
                self._collection.add(
                    ids=[
                        record["id"]
                        for record in records
                    ],
                    documents=[
                        record["content"]
                        for record in records
                    ],
                    embeddings=[
                        record["embedding"]
                        for record in records
                    ],
                    metadatas=[
                        record["metadata"]
                        for record in records
                    ],
                )

                return

            except Exception:
                # Nếu ChromaDB lỗi thì fallback
                # về in-memory.
                self._use_chroma = False
                self._collection = None

        self._store.extend(records)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.
        """

        if top_k <= 0:
            return []

        if self._use_chroma and self._collection is not None:
            try:
                results = self._collection.query(
                    query_embeddings=[
                        self._embedding_fn(query)
                    ],
                    n_results=top_k,
                )

                output = []

                ids = results.get("ids", [[]])[0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]

                for i in range(len(ids)):
                    # Chroma trả distance thay vì similarity.
                    # Chuyển thành score dễ hiểu hơn.
                    distance = (
                        distances[i]
                        if i < len(distances)
                        else 0.0
                    )

                    output.append({
                        "id": ids[i],
                        "content": documents[i],
                        "metadata": (
                            metadatas[i]
                            if i < len(metadatas)
                            else {}
                        ),
                        "score": 1.0 - distance,
                    })

                return output

            except Exception:
                # Fallback về in-memory
                pass

        return self._search_records(
            query,
            self._store,
            top_k
        )

    def get_collection_size(self) -> int:
        """
        Return the total number of stored chunks.
        """

        if self._use_chroma and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass

        return len(self._store)

    def search_with_filter(
        self,
        query: str,
        top_k: int = 3,
        metadata_filter: dict = None
    ) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter,
        then run similarity search.
        """

        # Không có filter → search bình thường.
        if not metadata_filter:
            return self.search(
                query,
                top_k=top_k
            )

        # Với ChromaDB, ta vẫn có thể dùng in-memory
        # vì self._store đảm bảo logic filter đơn giản
        # và ổn định cho test.
        filtered_records = []

        for record in self._store:
            metadata = record.get(
                "metadata",
                {}
            )

            matched = True

            for key, expected_value in metadata_filter.items():
                if metadata.get(key) != expected_value:
                    matched = False
                    break

            if matched:
                filtered_records.append(record)

        return self._search_records(
            query,
            filtered_records,
            top_k
        )

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed,
        False otherwise.
        """

        # Xóa trong in-memory store.
        original_size = len(self._store)

        self._store = [
            record
            for record in self._store
            if record.get("metadata", {}).get("doc_id") != doc_id
            and record.get("id") != doc_id
        ]

        removed = len(self._store) < original_size

        # Nếu đang dùng ChromaDB thì xóa theo id.
        if self._use_chroma and self._collection is not None:
            try:
                self._collection.delete(
                    ids=[doc_id]
                )
                removed = True

            except Exception:
                pass

        return removed
