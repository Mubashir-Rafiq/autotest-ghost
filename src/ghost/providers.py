"""Ghost's LLM provider abstraction and registry.

Owns:
- Abstract :class:`BaseProvider` establishing the template-method pattern so rate
  limiting and backoff retries cannot be bypassed by any concrete provider.
- Concrete providers: :class:`GroqProvider`, :class:`OpenAIProvider`,
  :class:`AnthropicProvider`, :class:`OllamaProvider`, :class:`LMStudioProvider`,
  :class:`OpenRouterProvider`, and :class:`CustomProvider`.
- Provider registry and live model discovery.

Does NOT:
- Construct test generation prompts (owned by ``prompts.py``).
- Parse verdicts or validate ASTs (owned by ``client.py``).
- Present tables or banners to the terminal (owned by ``console.py`` / ``cli.py``).

Guarantees:
1. Template-method pattern on :meth:`BaseProvider.chat`: rate limiting and
   exponential-backoff retries are enforced by the base class. Subclasses implement
   the internal :meth:`BaseProvider._call_api` hook and cannot forget retry logic.
2. Model listings are fetched live from provider APIs rather than hardcoded.
3. No provider SDK details leak past this module.
4. Missing optional provider dependencies (openai, anthropic) raise clear, actionable errors.
"""

from __future__ import annotations

import asyncio
import contextlib
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import httpx
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

_HTTP_OK: Final = 200

__all__ = [
    "POPULAR_MODELS",
    "PROVIDER_MODELS",
    "AnthropicProvider",
    "BaseProvider",
    "CustomProvider",
    "GroqProvider",
    "LMStudioProvider",
    "ModelConfig",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenRouterProvider",
    "auto_detect_provider",
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
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        provider="openai",
        context_length=128000,
        description="Flagship OpenAI omni model for complex coding and reasoning.",
    ),
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        provider="openai",
        context_length=128000,
        description="Fast, cost-efficient OpenAI model for test generation.",
    ),
    "claude-sonnet-4-20250514": ModelConfig(
        name="claude-sonnet-4-20250514",
        provider="anthropic",
        context_length=200000,
        description="Anthropic Claude Sonnet with strong coding intelligence.",
    ),
    "claude-3-5-haiku-20241022": ModelConfig(
        name="claude-3-5-haiku-20241022",
        provider="anthropic",
        context_length=200000,
        description="Fast, lightweight Claude model for rapid iteration.",
    ),
    "llama3:latest": ModelConfig(
        name="llama3:latest",
        provider="ollama",
        context_length=8192,
        description="Local Llama 3 running via Ollama without cloud telemetry.",
    ),
    "local-model": ModelConfig(
        name="local-model",
        provider="lmstudio",
        context_length=8192,
        description="Currently loaded model in local LM Studio instance.",
    ),
    "anthropic/claude-3.5-sonnet": ModelConfig(
        name="anthropic/claude-3.5-sonnet",
        provider="openrouter",
        context_length=200000,
        description="Claude 3.5 Sonnet routed via OpenRouter.",
    ),
}

PROVIDER_MODELS: Final[dict[str, list[str]]] = {
    "groq": [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama-3.3-70b-versatile",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "o1",
        "o3-mini",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514",
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
    ],
    "ollama": [
        "llama3:latest",
        "codellama:latest",
        "deepseek-coder-v2:latest",
    ],
    "lmstudio": [
        "local-model",
    ],
    "openrouter": [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o",
    ],
    "custom": [
        "default",
    ],
}


def _get_async_openai_cls() -> type[Any]:
    try:
        from openai import AsyncOpenAI  # noqa: PLC0415
    except ImportError as exc:
        p_name = "openai"
        msg = (
            "Package 'openai' is required for this provider. "
            "Install with: pip install 'autotest-ghost[openai]'"
        )
        raise ProviderError(p_name, msg) from exc
    else:
        return AsyncOpenAI


def _get_async_anthropic_cls() -> type[Any]:
    try:
        from anthropic import AsyncAnthropic  # noqa: PLC0415
    except ImportError as exc:
        p_name = "anthropic"
        msg = (
            "Package 'anthropic' is required for provider 'anthropic'. "
            "Install with: pip install 'autotest-ghost[anthropic]'"
        )
        raise ProviderError(p_name, msg) from exc
    else:
        return AsyncAnthropic


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


class OpenAICompatibleProvider(BaseProvider):
    """Base class for providers conforming to the OpenAI Chat Completions API schema."""

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
        self._client: Any = None

    @property
    def requires_api_key(self) -> bool:
        return True

    @property
    def default_api_key(self) -> str | None:
        return None

    @property
    def default_base_url(self) -> str | None:
        return None

    @property
    def default_env_var(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def default_headers(self) -> dict[str, str] | None:
        return None

    def _get_client(self) -> Any:
        effective_key = self.api_key or self.default_api_key
        if self.requires_api_key and not effective_key:
            raise ProviderAuthenticationError(provider=self.name, env_var=self.default_env_var)

        effective_base_url = self.base_url or self.default_base_url
        if self._client is None:
            async_openai_cls = _get_async_openai_cls()
            kwargs: dict[str, Any] = {
                "api_key": effective_key or "placeholder",
            }
            if effective_base_url:
                kwargs["base_url"] = effective_base_url
            if self.default_headers:
                kwargs["default_headers"] = self.default_headers
            self._client = async_openai_cls(**kwargs)
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
                messages=messages,
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
            endpoint = str(self.base_url or self.default_base_url or "https://api.openai.com/v1")
            raise ProviderUnavailableError(
                provider=self.name,
                endpoint=endpoint,
                details=str(exc),
            ) from exc

    @override
    async def is_available(self) -> bool:
        if self.requires_api_key:
            return bool(self.api_key and self.api_key.strip())
        return True


class OpenAIProvider(OpenAICompatibleProvider):
    """Official OpenAI cloud provider."""

    _DEFAULT_ENV_VAR: Final[str] = "OPENAI_API_KEY"

    @property
    @override
    def name(self) -> str:
        return "openai"

    @property
    @override
    def default_env_var(self) -> str:
        return self._DEFAULT_ENV_VAR


class OllamaProvider(OpenAICompatibleProvider):
    """Local Ollama instance via OpenAI compatibility layer."""

    _DEFAULT_BASE_URL: Final[str] = "http://localhost:11434/v1"

    @property
    @override
    def name(self) -> str:
        return "ollama"

    @property
    @override
    def requires_api_key(self) -> bool:
        return False

    @property
    @override
    def default_api_key(self) -> str:
        return "ollama"

    @property
    @override
    def default_base_url(self) -> str:
        return self._DEFAULT_BASE_URL

    @override
    async def list_models(self) -> list[str]:
        base = (self.base_url or self.default_base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        target = f"{base}/api/tags"
        with contextlib.suppress(Exception):
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(target)
                if res.status_code == _HTTP_OK:
                    data = res.json()
                    models = [m["name"] for m in data.get("models", [])]
                    if models:
                        return sorted(models)

        with contextlib.suppress(Exception):
            return await super().list_models()

        return sorted(PROVIDER_MODELS.get("ollama", ["llama3:latest"]))

    @override
    async def is_available(self) -> bool:
        base = (self.base_url or self.default_base_url).rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        target = f"{base}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(target)
                return res.status_code == _HTTP_OK
        except Exception:
            return False


class LMStudioProvider(OpenAICompatibleProvider):
    """Local LM Studio instance via OpenAI compatibility layer."""

    _DEFAULT_BASE_URL: Final[str] = "http://localhost:1234/v1"

    @property
    @override
    def name(self) -> str:
        return "lmstudio"

    @property
    @override
    def requires_api_key(self) -> bool:
        return False

    @property
    @override
    def default_api_key(self) -> str:
        return "lmstudio"

    @property
    @override
    def default_base_url(self) -> str:
        return self._DEFAULT_BASE_URL

    @override
    async def list_models(self) -> list[str]:
        with contextlib.suppress(Exception):
            return await super().list_models()
        return ["local-model"]

    @override
    async def is_available(self) -> bool:
        base = (self.base_url or self.default_base_url).rstrip("/")
        target = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                res = await client.get(target)
                return res.status_code == _HTTP_OK
        except Exception:
            return False


class OpenRouterProvider(OpenAICompatibleProvider):
    """OpenRouter provider routing requests across multiple models."""

    _DEFAULT_BASE_URL: Final[str] = "https://openrouter.ai/api/v1"
    _DEFAULT_ENV_VAR: Final[str] = "OPENROUTER_API_KEY"

    @property
    @override
    def name(self) -> str:
        return "openrouter"

    @property
    @override
    def default_base_url(self) -> str:
        return self._DEFAULT_BASE_URL

    @property
    @override
    def default_env_var(self) -> str:
        return self._DEFAULT_ENV_VAR

    @property
    @override
    def default_headers(self) -> dict[str, str]:
        return {
            "HTTP-Referer": "https://github.com/Mubashir-Rafiq/autotest-ghost",
            "X-Title": "Ghost",
        }


class CustomProvider(OpenAICompatibleProvider):
    """Custom OpenAI-compatible provider requiring explicit base_url."""

    @property
    @override
    def name(self) -> str:
        return "custom"

    @property
    @override
    def requires_api_key(self) -> bool:
        return False

    @property
    @override
    def default_api_key(self) -> str:
        return "custom"

    @override
    def _get_client(self) -> Any:
        if not self.base_url or not self.base_url.strip():
            raise ProviderError(
                self.name,
                "base_url is required for custom provider (set in ghost.toml or GHOST_BASE_URL).",
            )
        return super()._get_client()

    @override
    async def is_available(self) -> bool:
        return bool(self.base_url and self.base_url.strip())


class AnthropicProvider(BaseProvider):
    """Anthropic cloud provider backed by AsyncAnthropic."""

    _DEFAULT_ENV_VAR: Final[str] = "ANTHROPIC_API_KEY"

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
        self._client: Any = None

    @property
    @override
    def name(self) -> str:
        return "anthropic"

    def _get_client(self) -> Any:
        """Lazily initialize and return the AsyncAnthropic client."""
        if not self.api_key:
            raise ProviderAuthenticationError(provider=self.name, env_var=self._DEFAULT_ENV_VAR)
        if self._client is None:
            async_anthropic_cls = _get_async_anthropic_cls()
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = async_anthropic_cls(**kwargs)
        return self._client

    @override
    async def _call_api(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
    ) -> str:
        client = self._get_client()

        system_prompt: str | None = None
        anthropic_messages: list[dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_prompt = f"{system_prompt}\n{content}" if system_prompt else content
            else:
                anthropic_messages.append({"role": role, "content": content})

        if not anthropic_messages:
            anthropic_messages = [{"role": "user", "content": system_prompt or ""}]
            system_prompt = None

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": 4096,
            "extra_body": {"temperature": temperature},
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        try:
            response = await client.messages.create(**kwargs)
        except Exception as exc:
            lowered = str(exc).lower()
            if "not_found" in lowered or (
                "model" in lowered and ("not found" in lowered or "does not exist" in lowered)
            ):
                raise ModelNotFoundError(provider=self.name, model=model) from exc
            raise

        parts: list[str] = []
        for block in getattr(response, "content", []):
            if hasattr(block, "text"):
                parts.append(block.text)
        return "".join(parts)

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
            return sorted(
                PROVIDER_MODELS.get(
                    "anthropic",
                    [
                        "claude-sonnet-4-20250514",
                        "claude-3-7-sonnet-20250219",
                        "claude-3-5-sonnet-20241022",
                        "claude-3-5-haiku-20241022",
                    ],
                )
            )

    @override
    async def is_available(self) -> bool:
        return bool(self.api_key and self.api_key.strip())


_PROVIDER_REGISTRY: dict[str, type[BaseProvider]] = {
    "groq": GroqProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
    "lmstudio": LMStudioProvider,
    "openrouter": OpenRouterProvider,
    "custom": CustomProvider,
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
            if lowered in {"ollama", "lmstudio"}:
                kwargs["rate_limiter"] = None
            else:
                kwargs["rate_limiter"] = RateLimiter(rate_limit_rpm=config.ai.rate_limit_rpm)
        if "max_retries" not in kwargs:
            kwargs["max_retries"] = config.ai.max_retries

    return provider_cls(**kwargs)


async def auto_detect_provider(config: GhostConfig | None = None) -> str:
    """Detect the most appropriate provider based on local availability and credentials."""
    # 1. Check local zero-cost providers first
    for local_name in ("ollama", "lmstudio"):
        with contextlib.suppress(Exception):
            prov = get_provider(local_name, config=config)
            if await prov.is_available():
                return local_name

    # 2. Check cloud providers with configured credentials
    for cloud_name in ("groq", "openai", "anthropic", "openrouter"):
        with contextlib.suppress(Exception):
            prov = get_provider(cloud_name, config=config)
            if await prov.is_available():
                return cloud_name

    return "groq"


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
