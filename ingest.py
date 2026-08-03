"""
ingest.py — pipeline nạp dữ liệu ĐÃ CUNG CẤP cho Lab 7 (Giai đoạn 2/3).

Biến một thư mục file `.md`/`.txt` thành một `EmbeddingStore` tìm kiếm được:

    1. Parse YAML front matter  -> metadata cho từng tài liệu (xem docs/DATA_COLLECTION.md)
    2. Chia mỗi tài liệu thành chunks bằng chiến lược bạn chọn
    3. Gắn `doc_id` + toàn bộ metadata của tài liệu lên TỪNG chunk
    4. Nạp tất cả chunk vào EmbeddingStore

Phần "glue" này được cung cấp sẵn để bạn tập trung vào phần được chấm điểm:
thiết kế/lựa chọn chiến lược chunking, schema metadata, câu hỏi đánh giá và phân
tích truy xuất. Bạn vẫn truyền vào CHUNKER của mình (và có thể thêm metadata).

Front matter (dạng phẳng `key: value`), ví dụ:

    ---
    doc_id: library-renewal-policy
    audience: student
    ---
    <nội dung đã làm sạch>

Chạy `python3 ingest.py` để tự kiểm tra bộ parser front matter (không cần store —
chạy được cả trước khi bạn hoàn thành Giai đoạn 2).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from src.chunking import RecursiveChunker
from src.models import Document
from src.store import EmbeddingStore

TEXT_EXTENSIONS = {".md", ".txt"}


def parse_front_matter(text: str) -> tuple[dict, str]:
    """Tách một tài liệu thành ``(metadata, body)``.

    Nhận diện khối front matter kiểu YAML mở đầu bằng dòng ``---`` và kết thúc
    bằng dòng ``---``. Chỉ parse các cặp ``key: value`` phẳng (đủ cho lab này).
    Nếu có `pyyaml` thì dùng cho chắc chắn; nếu không, dùng parser tối giản.
    Trả về ``({}, text)`` khi không có front matter.
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    closing = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing = index
            break
    if closing is None:
        return {}, text

    fm_block = "\n".join(lines[1:closing])
    body = "\n".join(lines[closing + 1 :]).lstrip("\n")
    return _load_flat_yaml(fm_block), body


def _load_flat_yaml(block: str) -> dict:
    """Parse một khối front matter phẳng thành dict (pyyaml nếu có, không thì fallback)."""
    try:
        import yaml  # optional: dùng nếu môi trường có sẵn

        loaded = yaml.safe_load(block) or {}
        if isinstance(loaded, dict):
            return {str(key): value for key, value in loaded.items()}
    except Exception:
        pass

    metadata: dict = {}
    for raw in block.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.split(" #", 1)[0].strip()  # bỏ comment nội dòng " # ..."
        value = value.strip('"').strip("'")
        metadata[key.strip()] = value
    return metadata


def load_documents(data_dir: str | Path) -> list[Document]:
    """Đọc mọi file `.md`/`.txt` dưới `data_dir` thành `Document` (metadata từ front matter)."""
    data_path = Path(data_dir)
    documents: list[Document] = []

    # Cho phép truyền trực tiếp một file lẻ qua LAB_DATA_DIR, không chỉ thư mục.
    candidate_paths = [data_path] if data_path.is_file() else sorted(data_path.rglob("*"))

    for path in candidate_paths:
        if not path.is_file() or path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        doc_id = str(metadata.get("doc_id") or path.stem)
        metadata.setdefault("doc_id", doc_id)
        metadata.setdefault("source", str(path))
        documents.append(Document(id=doc_id, content=body, metadata=metadata))
    return documents


def chunk_document(doc: Document, chunker) -> list[Document]:
    """Chia một `Document` thành nhiều chunk-`Document`, giữ `doc_id` + metadata trên từng chunk."""
    chunk_docs: list[Document] = []
    for index, piece in enumerate(chunker.chunk(doc.content)):
        chunk_meta = dict(doc.metadata)
        chunk_meta["doc_id"] = doc.id           # để delete_document()/lọc theo doc_id hoạt động
        chunk_meta["chunk_index"] = index
        chunk_docs.append(
            Document(id=f"{doc.id}::chunk_{index}", content=piece, metadata=chunk_meta)
        )
    return chunk_docs


def build_knowledge_base(
    data_dir: str | Path,
    embedding_fn: Callable[[str], list[float]],
    chunker=None,
    collection_name: str = "lab7_kb",
) -> EmbeddingStore:
    """Đầu-cuối: file -> tài liệu đã parse -> chunk (kèm metadata) -> nạp vào store.

    Truyền CHUNKER bạn chọn (mặc định `RecursiveChunker`). Cần `EmbeddingStore`
    bạn đã hoàn thành ở Giai đoạn 2.
    """
    chunker = chunker or RecursiveChunker()
    chunk_docs: list[Document] = []
    for doc in load_documents(data_dir):
        chunk_docs.extend(chunk_document(doc, chunker))

    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedding_fn)
    store.add_documents(chunk_docs)
    return store


def _self_check() -> int:
    """Kiểm tra parser + gắn metadata mà KHÔNG cần EmbeddingStore."""
    sample = (
        "---\n"
        "doc_id: demo-policy\n"
        "audience: student\n"
        'document_version: "2026-08-01"  # comment nội dòng\n'
        "source_url: https://example.edu/demo\n"
        "---\n\n"
        "Câu nội dung thứ nhất. Câu nội dung thứ hai.\n"
    )
    meta, body = parse_front_matter(sample)
    assert meta.get("doc_id") == "demo-policy", meta
    assert meta.get("audience") == "student", meta
    assert meta.get("document_version") == "2026-08-01", meta
    assert meta.get("source_url") == "https://example.edu/demo", meta
    assert body.startswith("Câu nội dung thứ nhất."), repr(body)

    doc = Document(id="demo-policy", content=body * 20, metadata=meta)
    chunks = chunk_document(doc, RecursiveChunker(chunk_size=60))
    assert len(chunks) > 1, "kỳ vọng nhiều hơn 1 chunk"
    assert all(c.metadata["doc_id"] == "demo-policy" for c in chunks), chunks
    assert all("chunk_index" in c.metadata for c in chunks), chunks
    assert all(c.id.startswith("demo-policy::chunk_") for c in chunks), chunks

    print(
        f"ingest self-check OK: parse được {len(meta)} khóa metadata, "
        f"tạo {len(chunks)} chunk (mỗi chunk giữ doc_id + metadata)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_check())
