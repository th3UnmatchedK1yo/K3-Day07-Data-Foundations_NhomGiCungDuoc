from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from ingest import build_knowledge_base
from src.agent import KnowledgeBaseAgent
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)

# Thư mục dữ liệu mặc định cho demo = bộ khởi động cố định của lớp K3.
# Đổi bằng biến môi trường: LAB_DATA_DIR=data/<thu-muc-cua-nhom> python3 main.py
DEFAULT_DATA_DIR = "data/quy-trinh-danh-cho-sinh-vien.md"


def _select_embedder():
    """Chọn backend nhúng theo biến môi trường EMBEDDING_PROVIDER (mock | local | openai)."""
    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            return LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            print("Local embedder không sẵn sàng; tạm dùng mock.")
            return _mock_embed
    if provider == "openai":
        try:
            return OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            print("OpenAI embedder không sẵn sàng; tạm dùng mock.")
            return _mock_embed
    return _mock_embed


def demo_llm(prompt: str) -> str:
    """LLM giả lập đơn giản để thử RAG thủ công."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def run_manual_demo(question: str | None = None, data_dir: str | None = None) -> int:
    data_dir = data_dir or DEFAULT_DATA_DIR
    query = question or "Tóm tắt thông tin chính từ bộ tài liệu."

    print("=== Demo pipeline nạp dữ liệu (ingest.build_knowledge_base) ===")
    print(f"Thư mục dữ liệu: {data_dir}")
    if not Path(data_dir).exists():
        print(f"Không tìm thấy thư mục dữ liệu: {data_dir}")
        print("Thu thập tài liệu vào thư mục này (xem docs/DATA_COLLECTION.md) rồi chạy lại:")
        print("  python3 main.py")
        return 1

    embedder = _select_embedder()
    backend = getattr(embedder, "_backend_name", embedder.__class__.__name__)
    print(f"Backend nhúng: {backend}")
    if backend == "mock embeddings fallback":
        print(
            "Lưu ý: mock chỉ để chạy thử/unit test và KHÔNG phản ánh chất lượng ngữ nghĩa. "
            "Ở Giai đoạn 2, đặt EMBEDDING_PROVIDER=local để so sánh retrieval có ý nghĩa."
        )

    # Pipeline cung cấp sẵn: parse front matter -> chunk -> gắn metadata -> nạp store.
    store = build_knowledge_base(data_dir, embedding_fn=embedder)
    print(f"Đã nạp {store.get_collection_size()} chunk vào EmbeddingStore")

    print("\n=== Tìm kiếm (EmbeddingStore.search) ===")
    print(f"Câu hỏi: {query}")
    for index, result in enumerate(store.search(query, top_k=3), start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')}")
        print(f"   {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent ===")
    agent = KnowledgeBaseAgent(store=store, llm_fn=demo_llm)
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or None
    data_dir = os.getenv("LAB_DATA_DIR", DEFAULT_DATA_DIR)
    return run_manual_demo(question=question, data_dir=data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
