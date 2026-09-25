"""Tests for Ghost's provider abstraction, rate limiting integration, and Groq implementation."""

from __future__ import annotations

import builtins
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import httpx
import pytest
import respx

try:
    import openai  # noqa: F401

    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

from ghost.config import GhostConfig
from ghost.errors import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
)
from ghost.providers import (
    POPULAR_MODELS,
    PROVIDER_MODELS,
    AnthropicProvider,
    BaseProvider,
    CustomProvider,
    GroqProvider,
    LMStudioProvider,
    OllamaProvider,
    OpenAIProvider,
    OpenRouterProvider,
    _get_async_anthropic_cls,
    _get_async_openai_cls,
    auto_detect_provider,
    get_provider,
    list_available_providers,
    list_providers,
)
from ghost.rate_limiter import RateLimiter


class MockCountingLimiter(RateLimiter):
    """RateLimiter instrumented to record acquire calls."""

    def __init__(self) -> None:
        super().__init__(rate_limit_rpm=6000)
        self.acquire_count = 0

    @override
    async def acquire(self) -> float:
        self.acquire_count += 1
        return 0.0


class MockRateLimitError(Exception):
    pass


class DummyProvider(BaseProvider):
    """Concrete provider for testing the template-method pattern."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(base_delay=0.01, **kwargs)  # type: ignore[arg-type]
        self.call_count = 0
        self.fail_times = 0

    @property
    @override
    def name(self) -> str:
        return "dummy"

    @override
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        self.call_count += 1
        if self.call_count <= self.fail_times:
            msg = "Rate limit 429: too many requests"
            raise MockRateLimitError(msg)
        return "dummy completion"

    @override
    async def list_models(self) -> list[str]:
        return ["dummy-model-v1"]

    @override
    async def is_available(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_base_provider_template_method_enforces_rate_limiting() -> None:
    """The base class enforces rate limiting; subclasses cannot accidentally bypass it."""
    limiter = MockCountingLimiter()
    provider = DummyProvider(rate_limiter=limiter)

    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}], model="dummy-model-v1"
    )
    assert result == "dummy completion"
    assert limiter.acquire_count == 1


@pytest.mark.asyncio
async def test_base_provider_template_method_retries_on_rate_limit() -> None:
    """Template method retries rate limit errors using exponential backoff."""
    provider = DummyProvider(max_retries=3)
    provider.fail_times = 2

    # Should fail twice with 429, then succeed on 3rd attempt
    result = await provider.chat(
        messages=[{"role": "user", "content": "hi"}], model="dummy-model-v1"
    )
    assert result == "dummy completion"
    assert provider.call_count == 3


@pytest.mark.asyncio
async def test_base_provider_retries_exhausted_raises_provider_rate_limit_error() -> None:
    """Retries exhausted results in ProviderRateLimitError."""
    provider = DummyProvider(max_retries=2)
    provider.fail_times = 5

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await provider.chat(messages=[{"role": "user", "content": "hi"}], model="dummy-model-v1")

    assert "dummy" in str(exc_info.value)
    assert provider.call_count == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_base_provider_non_rate_limit_error_fails_immediately() -> None:
    """Non-rate-limit exceptions propagate immediately without retry."""
    provider = DummyProvider(max_retries=5)

    class CustomNonRateLimitError(Exception):
        pass

    async def failing_api(messages: list[dict[str, str]], model: str, temperature: float) -> str:
        provider.call_count += 1
        msg = "invalid syntax in prompt"
        raise CustomNonRateLimitError(msg)

    provider._call_api = failing_api  # type: ignore[method-assign]

    with pytest.raises(CustomNonRateLimitError):
        await provider.chat(messages=[], model="dummy-model-v1")

    assert provider.call_count == 1


def test_groq_provider_missing_api_key_raises_auth_error() -> None:
    """GroqProvider raises ProviderAuthenticationError when api_key is missing."""
    provider = GroqProvider(api_key=None)

    with pytest.raises(ProviderAuthenticationError) as exc_info:
        provider._get_client()

    assert "GROQ_API_KEY" in str(exc_info.value)
    assert "groq" in str(exc_info.value)


@pytest.mark.asyncio
@respx.mock
async def test_groq_provider_list_models_live() -> None:
    """GroqProvider queries the /v1/models endpoint live."""
    mock_route = respx.get("https://api.groq.com/openai/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {"id": "openai/gpt-oss-120b"},
                    {"id": "openai/gpt-oss-20b"},
                ]
            },
        )
    )

    provider = GroqProvider(api_key="gsk_test_key_123")
    models = await provider.list_models()

    assert mock_route.called
    assert models == ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]


@pytest.mark.asyncio
@respx.mock
async def test_groq_provider_chat_completion() -> None:
    """GroqProvider calls chat completions and extracts content."""
    mock_route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "def test_answer(): assert 1 == 1",
                        }
                    }
                ]
            },
        )
    )

    provider = GroqProvider(api_key="gsk_test_key_123")
    result = await provider.chat(
        messages=[{"role": "user", "content": "write a test"}],
        model="openai/gpt-oss-120b",
    )

    assert mock_route.called
    assert "def test_answer():" in result


@pytest.mark.asyncio
@respx.mock
async def test_groq_provider_model_not_found_handling() -> None:
    """404 Model does not exist raises ModelNotFoundError."""
    respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            404,
            json={"error": {"message": "The model `llama-dead-model` does not exist"}},
        )
    )

    provider = GroqProvider(api_key="gsk_test_key_123")
    with pytest.raises(ModelNotFoundError) as exc_info:
        await provider.chat(
            messages=[{"role": "user", "content": "hi"}],
            model="llama-dead-model",
        )

    assert "llama-dead-model" in str(exc_info.value)
    assert "groq" in str(exc_info.value)


def test_provider_registry_and_get_provider() -> None:
    """Provider registry contains all supported providers."""
    providers = list_providers()
    assert "groq" in providers
    assert "openai" in providers
    assert "anthropic" in providers
    assert "ollama" in providers
    assert "lmstudio" in providers
    assert "openrouter" in providers
    assert "custom" in providers

    provider = get_provider("groq", api_key="test-key")
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"

    # Local providers don't enforce rate limits by default
    ollama_prov = get_provider("ollama", config=GhostConfig())
    assert ollama_prov.rate_limiter is None

    # Unknown provider
    with pytest.raises(ProviderError) as exc_info:
        get_provider("nonexistent-cloud")
    assert "unknown provider 'nonexistent-cloud'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_available_providers() -> None:
    """list_available_providers returns mapping of provider to configuration status."""
    cfg = GhostConfig()
    status = await list_available_providers(cfg)
    assert "groq" in status
    assert "openai" in status
    assert "anthropic" in status
    assert "ollama" in status
    assert "lmstudio" in status
    assert "openrouter" in status
    assert "custom" in status


def test_popular_models_registry_contains_default() -> None:
    """POPULAR_MODELS and PROVIDER_MODELS are correctly populated."""
    assert "openai/gpt-oss-120b" in POPULAR_MODELS
    assert POPULAR_MODELS["openai/gpt-oss-120b"].provider == "groq"
    assert "gpt-4o" in POPULAR_MODELS
    assert POPULAR_MODELS["gpt-4o"].provider == "openai"
    assert "claude-sonnet-4-20250514" in POPULAR_MODELS
    assert "llama3:latest" in POPULAR_MODELS

    for prov in ("groq", "openai", "anthropic", "ollama", "lmstudio", "openrouter", "custom"):
        assert prov in PROVIDER_MODELS


def test_openai_provider_missing_key() -> None:
    """OpenAIProvider raises ProviderAuthenticationError when api_key is missing."""
    provider = OpenAIProvider(api_key=None)
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        provider._get_client()
    assert "OPENAI_API_KEY" in str(exc_info.value)
    assert "openai" in str(exc_info.value)


@pytest.mark.asyncio
async def test_openai_provider_chat_and_models() -> None:
    """OpenAIProvider completes chat and lists models."""
    provider = OpenAIProvider(api_key="sk-test-openai")
    mock_client = AsyncMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "assert 2 == 2"
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    mock_m1 = MagicMock(id="gpt-4o")
    mock_m2 = MagicMock(id="gpt-4o-mini")
    mock_client.models.list.return_value = MagicMock(data=[mock_m1, mock_m2])
    provider._client = mock_client

    res = await provider.chat([{"role": "user", "content": "test"}], model="gpt-4o")
    assert "assert 2 == 2" in res

    models = await provider.list_models()
    assert models == ["gpt-4o", "gpt-4o-mini"]
    assert await provider.is_available() is True


def test_anthropic_provider_missing_key() -> None:
    """AnthropicProvider raises ProviderAuthenticationError when api_key is missing."""
    provider = AnthropicProvider(api_key=None)
    with pytest.raises(ProviderAuthenticationError) as exc_info:
        provider._get_client()
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    assert "anthropic" in str(exc_info.value)


@pytest.mark.asyncio
async def test_anthropic_provider_chat_and_system_separation() -> None:
    """AnthropicProvider converts system message and posts to /v1/messages."""
    provider = AnthropicProvider(api_key="sk-ant-test")
    mock_client = AsyncMock()
    mock_block = MagicMock()
    mock_block.text = "def test_anthropic(): pass"
    mock_client.messages.create.return_value = MagicMock(content=[mock_block])

    mock_m1 = MagicMock(id="claude-3-5-sonnet")
    mock_client.models.list.return_value = MagicMock(data=[mock_m1])
    provider._client = mock_client

    res = await provider.chat(
        [
            {"role": "system", "content": "You are a test writer."},
            {"role": "user", "content": "Write a test."},
        ],
        model="claude-sonnet-4-20250514",
    )
    assert "def test_anthropic(): pass" in res
    assert await provider.is_available() is True

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "You are a test writer."
    assert call_kwargs["messages"] == [{"role": "user", "content": "Write a test."}]
    assert call_kwargs["extra_body"] == {"temperature": 0.1}


@pytest.mark.asyncio
@respx.mock
async def test_ollama_provider_availability_and_models() -> None:
    """OllamaProvider checks /api/tags for live availability."""
    tags_route = respx.get("http://localhost:11434/api/tags").mock(
        return_value=httpx.Response(
            200,
            json={"models": [{"name": "llama3:latest"}, {"name": "codellama:latest"}]},
        )
    )

    provider = OllamaProvider()
    assert await provider.is_available() is True
    assert tags_route.called

    models = await provider.list_models()
    assert "codellama:latest" in models
    assert "llama3:latest" in models

    # Test unavailable when endpoint fails
    tags_route.side_effect = httpx.ConnectError("Connection refused")
    assert await provider.is_available() is False


@pytest.mark.asyncio
@respx.mock
async def test_lmstudio_provider_availability() -> None:
    """LMStudioProvider checks /v1/models for availability."""
    models_route = respx.get("http://localhost:1234/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "local-model"}]})
    )

    provider = LMStudioProvider()
    assert await provider.is_available() is True
    assert models_route.called

    models = await provider.list_models()
    assert "local-model" in models

    # Test unavailable when offline
    models_route.side_effect = httpx.ConnectError("Connection refused")
    assert await provider.is_available() is False


@pytest.mark.skipif(not HAS_OPENAI, reason="openai optional dependency is required")
def test_openrouter_provider_configuration() -> None:
    """OpenRouterProvider sets OpenRouter base URL and default headers."""
    provider = OpenRouterProvider(api_key="sk-or-test")
    assert provider.name == "openrouter"
    assert provider.default_headers["X-Title"] == "Ghost"
    client = provider._get_client()
    assert str(client.base_url) == "https://openrouter.ai/api/v1/"


@pytest.mark.skipif(not HAS_OPENAI, reason="openai optional dependency is required")
@pytest.mark.asyncio
async def test_custom_provider_validation() -> None:
    """CustomProvider requires explicit base_url."""
    provider_no_url = CustomProvider(api_key="test-key")
    with pytest.raises(ProviderError) as exc_info:
        provider_no_url._get_client()
    assert "base_url is required" in str(exc_info.value)

    provider_with_url = CustomProvider(base_url="https://custom-llm.example.com/v1")
    assert await provider_with_url.is_available() is True
    client = provider_with_url._get_client()
    assert str(client.base_url) == "https://custom-llm.example.com/v1/"


def test_missing_optional_dependency_raises_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing openai or anthropic raises clean ProviderError."""
    orig_import = builtins.__import__

    def fail_imports(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"openai", "anthropic"}:
            err_msg = f"No module named '{name}'"
            raise ImportError(err_msg)
        return orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_imports)

    with pytest.raises(ProviderError) as exc_openai:
        _get_async_openai_cls()
    assert "pip install 'autotest-ghost[openai]'" in str(exc_openai.value)

    with pytest.raises(ProviderError) as exc_anthropic:
        _get_async_anthropic_cls()
    assert "pip install 'autotest-ghost[anthropic]'" in str(exc_anthropic.value)


@pytest.mark.asyncio
async def test_auto_detect_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """auto_detect_provider tests local providers before cloud providers."""

    # When Ollama is available
    async def mock_ollama_avail(self: OllamaProvider) -> bool:
        return True

    monkeypatch.setattr(OllamaProvider, "is_available", mock_ollama_avail)
    detected = await auto_detect_provider()
    assert detected == "ollama"
