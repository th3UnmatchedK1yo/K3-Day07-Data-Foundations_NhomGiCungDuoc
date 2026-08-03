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

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        results = self._store.search(question, top_k=top_k)
        context = "\n\n".join(f"- {result['content']}" for result in results)
        prompt = (
            "Use the context below to answer the question. "
            "If the context does not contain the answer, say so.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return self._llm_fn(prompt)
