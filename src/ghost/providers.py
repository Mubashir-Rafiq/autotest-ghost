"""Ghost's LLM provider abstraction and registry.

Owns:
- Abstract :class:`BaseProvider` establishing the template-method pattern so rate
  limiting and backoff retries cannot be bypassed by any concrete provider.
- Concrete :class:`GroqProvider` communicating with Groq via ``AsyncGroq``.
- Provider registry and live model discovery.

Does NOT:
- Construct test generation prompts (owned by ``prompts.py``).
- Parse verdicts or validate ASTs (owned by ``chat.py``).
- Present tables or banners to the terminal (owned by ``console.py`` / ``cli.py``).

Guarantees:
1. Template-method pattern on :meth:`BaseProvider.chat`: rate limiting and
   exponential-backoff retries are enforced by the base class. Subclasses implement
   the internal :meth:`BaseProvider._call_api` hook and cannot forget retry logic.
2. Model listings are fetched live from provider APIs rather than hardcoded.
3. No provider SDK details leak past this module.
"""

from __future__ import annotations

import asyncio
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from groq import AsyncGroq
from pydantic import BaseModel, ConfigDict

from ghost.errors import (
    ModelNotFoundError,
    ProviderAuthenticationError,
    ProviderError,
    ProviderRateLimitError,
    ProviderUnavailableError,
)
from ghost.rate_limiter import RateLimiter, calculate_backoff_delay, is_rate_limit_error

if TYPE_CHECKING:
    from ghost.config import GhostConfig

__all__ = [
    "POPULAR_MODELS",
    "BaseProvider",
    "GroqProvider",
    "ModelConfig",
    "get_provider",
    "list_available_providers",
    "list_providers",
    "register_provider",
]


class ModelConfig(BaseModel):
    """Metadata describing a popular or reference model."""

    model_config = ConfigDict(frozen=True)

    name: str
    provider: str
    context_length: int
    description: str


POPULAR_MODELS: Final[dict[str, ModelConfig]] = {
    "openai/gpt-oss-120b": ModelConfig(
        name="openai/gpt-oss-120b",
        provider="groq",
        context_length=131072,
        description="Best open-weights code generation model on Groq's current catalogue.",
    ),
    "openai/gpt-oss-20b": ModelConfig(
        name="openai/gpt-oss-20b",
        provider="groq",
        context_length=131072,
        description="Fast, low-latency code model on Groq.",
    ),
}


class BaseProvider(ABC):
    """Abstract base class for all LLM providers using the template-method pattern.

    Subclasses implement :meth:`_call_api`, :meth:`list_models`, and :meth:`is_available`.
    The public :meth:`chat` method coordinates rate limiting and retry backoff.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.rate_limiter = rate_limiter
        self.max_retries = max_retries
        self.base_delay = base_delay

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical name identifier for this provider."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.1,
    ) -> str:
        """Template method: enforces rate limiting and retry with backoff around _call_api."""
        if self.rate_limiter is not None:
            await self.rate_limiter.acquire()

        for attempt in range(self.max_retries + 1):
            try:
                return await self._call_api(messages, model, temperature)
            except Exception as exc:
                if not is_rate_limit_error(exc) or attempt >= self.max_retries:
                    if is_rate_limit_error(exc):
                        raise ProviderRateLimitError(
                            provider=self.name,
                            attempts=attempt + 1,
                            details=str(exc),
                        ) from exc
                    raise

                delay = calculate_backoff_delay(attempt, base_delay=self.base_delay)
                await asyncio.sleep(delay)

        raise ProviderRateLimitError(provider=self.name, attempts=self.max_retries)

    @abstractmethod
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        """Provider-specific chat completion API call."""

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Fetch available models live from the provider API."""

    @abstractmethod
    async def is_available(self) -> bool:
        """Check whether the provider is currently configured and available."""


class GroqProvider(BaseProvider):
    """Groq cloud provider backed by AsyncGroq."""

    _DEFAULT_ENV_VAR: Final[str] = "GROQ_API_KEY"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 5,
        base_delay: float = 2.0,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            rate_limiter=rate_limiter,
            max_retries=max_retries,
            base_delay=base_delay,
        )
        self._client: AsyncGroq | None = None

    @property
    @override
    def name(self) -> str:
        return "groq"

    def _get_client(self) -> AsyncGroq:
        """Lazily initialize and return the AsyncGroq client."""
        if not self.api_key:
            raise ProviderAuthenticationError(provider=self.name, env_var=self._DEFAULT_ENV_VAR)
        if self._client is None:
            self._client = AsyncGroq(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @override
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        client = self._get_client()
        try:
            response = await client.chat.completions.create(
                messages=messages,  # type: ignore[arg-type]
                model=model,
                temperature=temperature,
            )
        except Exception as exc:
            lowered = str(exc).lower()
            if "model" in lowered and ("not found" in lowered or "does not exist" in lowered):
                raise ModelNotFoundError(provider=self.name, model=model) from exc
            raise

        content = response.choices[0].message.content
        return content if content is not None else ""

    @override
    async def list_models(self) -> list[str]:
        client = self._get_client()
        try:
            response = await client.models.list()
            return sorted(m.id for m in response.data)
        except ProviderError:
            raise
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise ProviderRateLimitError(
                    provider=self.name,
                    attempts=1,
                    details=str(exc),
                ) from exc
            endpoint = str(self.base_url or "https://api.groq.com/openai/v1")
            raise ProviderUnavailableError(
                provider=self.name,
                endpoint=endpoint,
                details=str(exc),
            ) from exc

    @override
    async def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


_PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "groq": GroqProvider,
}


def register_provider(name: str, provider_cls: type[BaseProvider]) -> None:
    """Register a new concrete provider class."""
    _PROVIDER_REGISTRY[name.lower()] = provider_cls


def list_providers() -> list[str]:
    """Return names of all registered providers."""
    return sorted(_PROVIDER_REGISTRY.keys())


def get_provider(
    name: str,
    config: GhostConfig | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """Instantiate a provider by name, resolving credentials and rate limits from config."""
    lowered = name.lower()
    provider_cls = _PROVIDER_REGISTRY.get(lowered)
    if provider_cls is None:
        supported = ", ".join(list_providers())
        msg = f"unknown provider {name!r}. Supported providers: {supported}"
        raise ProviderError(name, msg)

    if config is not None:
        if "api_key" not in kwargs or kwargs["api_key"] is None:
            kwargs["api_key"] = config.resolve_api_key()
        if "base_url" not in kwargs or kwargs["base_url"] is None:
            kwargs["base_url"] = config.resolve_base_url()
        if "rate_limiter" not in kwargs or kwargs["rate_limiter"] is None:
            kwargs["rate_limiter"] = RateLimiter(rate_limit_rpm=config.ai.rate_limit_rpm)
        if "max_retries" not in kwargs:
            kwargs["max_retries"] = config.ai.max_retries

    return provider_cls(**kwargs)


async def list_available_providers(config: GhostConfig | None = None) -> dict[str, bool]:
    """Return availability status for all registered providers."""
    status_map: dict[str, bool] = {}
    for name in list_providers():
        try:
            provider = get_provider(name, config=config)
            status_map[name] = await provider.is_available()
        except ProviderError:
            status_map[name] = False
    return status_map
