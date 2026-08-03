"""DeepSeek generation client (OpenAI-compatible Chat Completions API).

DeepSeek is the only LLM used for answer generation in this project and is
never used to produce embeddings (see ``src/rag_embeddings.py``). The API
key, base URL and model are always read from the environment -- never
hard-coded -- so importing this module never requires a configured API key
or network access; only calling :meth:`DeepSeekClient.generate` does.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"
DEEPSEEK_MODEL_ENV = "DEEPSEEK_MODEL"

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"

REFUSAL_MESSAGE = "Không tìm thấy thông tin này trong tài liệu UIT đã nạp."

SYSTEM_PROMPT = f"""Bạn là trợ lý trả lời câu hỏi về các quy trình dành cho sinh viên UIT.

QUY TẮC BẮT BUỘC (không được vi phạm):
1. Chỉ được trả lời dựa trên nội dung trong phần "Context" được cung cấp bên dưới. Không dùng kiến thức bên ngoài để tự suy diễn hoặc tự hoàn thiện quy định.
2. Nếu context không chứa đủ thông tin để trả lời câu hỏi, hãy trả lời đúng nguyên văn: "{REFUSAL_MESSAGE}"
3. Phải phân biệt rõ ràng trong câu trả lời: điều kiện, trình tự các bước, giới hạn, ngoại lệ, và thời hạn.
4. Không được làm mất hoặc đổi nghĩa các từ định lượng/giới hạn xuất hiện trong context, ví dụ: "tối đa", "tối thiểu", "chỉ", "không được", "dưới 30%", "chậm nhất", "trong vòng".
5. Mỗi ý trong câu trả lời phải có trích dẫn (citation) ngay sau nó theo đúng định dạng: [uit_student_procedures/{{procedure_slug}}/{{chunk_id}}] (lấy đúng procedure_slug và chunk_id ghi trong context).
6. Câu trả lời phải ngắn gọn, đi thẳng vào vấn đề, không suy đoán, không đưa ra lời khuyên ngoài phạm vi context."""


class DeepSeekConfigurationError(RuntimeError):
    """Raised when DeepSeek credentials/configuration are missing or invalid."""


@dataclass(frozen=True)
class DeepSeekConfig:
    api_key: str
    base_url: str
    model: str


def load_config_from_env() -> DeepSeekConfig:
    """Read DeepSeek configuration from the environment. Never hard-codes a key."""
    api_key = os.getenv(DEEPSEEK_API_KEY_ENV, "").strip()
    if not api_key:
        raise DeepSeekConfigurationError(
            f"{DEEPSEEK_API_KEY_ENV} is not set. Set it in your environment or in a local "
            ".env file (see .env.example). The key itself is never printed or logged."
        )
    base_url = os.getenv(DEEPSEEK_BASE_URL_ENV, DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    model = os.getenv(DEEPSEEK_MODEL_ENV, DEFAULT_MODEL).strip() or DEFAULT_MODEL
    return DeepSeekConfig(api_key=api_key, base_url=base_url, model=model)


class DeepSeekClient:
    """Thin OpenAI-compatible wrapper around the DeepSeek chat completions API.

    The underlying ``openai.OpenAI`` client is created lazily (never at
    import time, never at ``__init__`` time either) so this class can be
    instantiated and swapped out in tests without any network access.
    """

    def __init__(self, config: Optional[DeepSeekConfig] = None) -> None:
        self._config = config or load_config_from_env()
        self._client = None

    @property
    def model(self) -> str:
        return self._config.model

    @property
    def base_url(self) -> str:
        return self._config.base_url

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise DeepSeekConfigurationError(
                    "The 'openai' package is not installed. Run: "
                    "python -m pip install -r requirements-rag.txt"
                ) from exc
            self._client = OpenAI(api_key=self._config.api_key, base_url=self._config.base_url)
        return self._client

    def build_user_prompt(self, question: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Câu hỏi: {question}\n\n"
            "Hãy trả lời theo đúng các quy tắc bắt buộc ở trên."
        )

    def generate(self, question: str, context: str, temperature: float = 0.0) -> str:
        """Generate a grounded answer for ``question`` using ``context``.

        ``temperature=0.0`` is the benchmark default for reproducibility.
        Raises :class:`DeepSeekConfigurationError` (never silently swallowed)
        if the API call fails.
        """
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self._config.model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": self.build_user_prompt(question, context)},
                ],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with context, never swallowed
            raise DeepSeekConfigurationError(f"DeepSeek API call failed: {exc}") from exc
        return response.choices[0].message.content or ""


__all__ = [
    "DeepSeekClient",
    "DeepSeekConfig",
    "DeepSeekConfigurationError",
    "load_config_from_env",
    "SYSTEM_PROMPT",
    "REFUSAL_MESSAGE",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "DEEPSEEK_API_KEY_ENV",
    "DEEPSEEK_BASE_URL_ENV",
    "DEEPSEEK_MODEL_ENV",
]
