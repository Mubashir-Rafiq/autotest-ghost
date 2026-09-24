"""Tests for Ghost's provider abstraction, rate limiting integration, and Groq implementation."""

from __future__ import annotations

import sys

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import httpx
import pytest
import respx

from ghost.config import GhostConfig
from ghost.errors import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
)
from ghost.providers import (
    POPULAR_MODELS,
    BaseProvider,
    GroqProvider,
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
    """Provider registry allows lookup and registration."""
    assert "groq" in list_providers()

    provider = get_provider("groq", api_key="test-key")
    assert isinstance(provider, GroqProvider)
    assert provider.name == "groq"

    # Unknown provider
    with pytest.raises(ProviderError) as exc_info:
        get_provider("nonexistent-cloud")
    assert "unknown provider 'nonexistent-cloud'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_list_available_providers() -> None:
    """list_available_providers returns mapping of provider to configuration status."""
    cfg_with_key = GhostConfig()
    # Without API key configured
    status = await list_available_providers(cfg_with_key)
    assert "groq" in status


def test_popular_models_registry_contains_default() -> None:
    """POPULAR_MODELS contains the locked default model."""
    assert "openai/gpt-oss-120b" in POPULAR_MODELS
    default = POPULAR_MODELS["openai/gpt-oss-120b"]
    assert default.provider == "groq"
    assert default.context_length > 0
