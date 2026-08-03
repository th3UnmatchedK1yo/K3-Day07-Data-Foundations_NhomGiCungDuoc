"""Tests for src/deepseek_client.py.

DeepSeek is NEVER called over the network in these tests: the OpenAI SDK
client is monkeypatched with an in-process fake before ``generate()`` runs.
"""

from __future__ import annotations

import pytest

from src.deepseek_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    DeepSeekClient,
    DeepSeekConfig,
    DeepSeekConfigurationError,
    load_config_from_env,
)


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = type("Message", (), {"content": content})()


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_call_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_call_kwargs = kwargs
        return _FakeCompletionResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeChatCompletions(content)


class _FakeOpenAIClient:
    """Stand-in for openai.OpenAI: records the request, never touches the network."""

    def __init__(self, content: str = "Đây là câu trả lời giả.") -> None:
        self.chat = _FakeChat(content)


def _make_client_with_fake_backend(monkeypatch, content: str = "Đây là câu trả lời giả.") -> tuple[DeepSeekClient, _FakeOpenAIClient]:
    client = DeepSeekClient(DeepSeekConfig(api_key="fake-key-not-real", base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL))
    fake_backend = _FakeOpenAIClient(content)
    monkeypatch.setattr(client, "_get_client", lambda: fake_backend)
    return client, fake_backend


def test_load_config_from_env_raises_when_api_key_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(DeepSeekConfigurationError):
        load_config_from_env()


def test_load_config_from_env_uses_defaults_when_only_key_is_set(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "fake-key-not-real")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
    config = load_config_from_env()
    assert config.base_url == DEFAULT_BASE_URL
    assert config.model == DEFAULT_MODEL


def test_load_config_from_env_never_hardcodes_a_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "my-custom-key")
    assert load_config_from_env().api_key == "my-custom-key"


def test_client_constructor_never_touches_network_or_requires_client_creation():
    # Constructing a DeepSeekClient must not build the openai.OpenAI client eagerly.
    client = DeepSeekClient(DeepSeekConfig(api_key="fake-key-not-real", base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL))
    assert client._client is None  # lazy


def test_generate_calls_mocked_backend_and_never_the_real_api(monkeypatch):
    client, fake_backend = _make_client_with_fake_backend(monkeypatch, content="Trả lời có trích dẫn [uit_student_procedures/dang_ky_hoc_phan/chunk_000]")
    answer = client.generate("Câu hỏi test?", context="Ngữ cảnh test", temperature=0.0)

    assert "trích dẫn" in answer
    assert fake_backend.chat.completions.last_call_kwargs["temperature"] == 0.0
    assert fake_backend.chat.completions.last_call_kwargs["model"] == DEFAULT_MODEL


def test_generate_sends_the_grounding_system_prompt(monkeypatch):
    client, fake_backend = _make_client_with_fake_backend(monkeypatch)
    client.generate("Câu hỏi test?", context="Ngữ cảnh test")

    messages = fake_backend.chat.completions.last_call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert REFUSAL_MESSAGE in SYSTEM_PROMPT


def test_system_prompt_requires_citation_format():
    assert "[uit_student_procedures/" in SYSTEM_PROMPT


def test_system_prompt_requires_preserving_quantifier_words():
    for quantifier in ("tối đa", "tối thiểu", "chậm nhất", "trong vòng"):
        assert quantifier in SYSTEM_PROMPT


def test_build_user_prompt_includes_context_and_question():
    client = DeepSeekClient(DeepSeekConfig(api_key="fake-key-not-real", base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL))
    prompt = client.build_user_prompt("Câu hỏi?", "Ngữ cảnh XYZ")
    assert "Ngữ cảnh XYZ" in prompt
    assert "Câu hỏi?" in prompt


def test_generate_wraps_backend_failures_without_swallowing_them(monkeypatch):
    client = DeepSeekClient(DeepSeekConfig(api_key="fake-key-not-real", base_url=DEFAULT_BASE_URL, model=DEFAULT_MODEL))

    class _BrokenBackend:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    raise ConnectionError("simulated network failure")

    monkeypatch.setattr(client, "_get_client", lambda: _BrokenBackend())
    with pytest.raises(DeepSeekConfigurationError):
        client.generate("Câu hỏi?", context="Ngữ cảnh")


def test_model_and_base_url_properties_reflect_config():
    config = DeepSeekConfig(api_key="fake-key-not-real", base_url="https://example.com", model="deepseek-v4-pro")
    client = DeepSeekClient(config)
    assert client.model == "deepseek-v4-pro"
    assert client.base_url == "https://example.com"
