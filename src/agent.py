from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(
        self,
        store: EmbeddingStore,
        llm_fn: Callable[[str], str]
    ) -> None:
        # Lưu vector store
        self.store = store

        # Lưu hàm gọi LLM
        self.llm_fn = llm_fn

    def answer(
        self,
        question: str,
        top_k: int = 3
    ) -> str:
        # Bước 1: Retrieve các document/chunk liên quan
        results = self.store.search(
            question,
            top_k=top_k
        )

        # Bước 2: Xây dựng context từ các chunk được retrieve
        context_parts = []

        for result in results:
            content = result.get("content", "")

            if content:
                context_parts.append(content)

        context = "\n\n".join(context_parts)

        # Bước 3: Tạo prompt cho LLM
        prompt = f"""
You are a knowledge base assistant.

Answer the user's question using the provided context.

Context:
{context}

Question:
{question}

Answer:
""".strip()

        # Bước 4: Gọi LLM
        answer = self.llm_fn(prompt)

        # Đảm bảo kết quả luôn là string
        return str(answer)
