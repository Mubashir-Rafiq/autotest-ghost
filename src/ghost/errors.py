"""Ghost's exception taxonomy.

Owns: every exception Ghost raises deliberately, rooted at a single base class.

Does NOT: format errors for display (that is ``console.py``'s job), decide exit
codes (``cli.py``'s job), or catch anything.

Why this is its own module rather than living beside the code that raises:
``providers.py`` imports ``rate_limiter.py``, and both need to raise and catch
the same types. A shared type that two layers depend on must sit *below* both of
them, or the import graph acquires a cycle.

Two standing rules for everything defined here:

1. **Every Ghost error derives from :class:`GhostError`.** That gives the CLI a
   single ``except GhostError`` boundary which distinguishes "a condition we
   anticipated and can explain" from "a bug in Ghost", which should crash loudly
   with a traceback.
2. **Every error names the offending input.** "Invalid config" is not a
   diagnostic; "unknown key 'auto_heel' in ghost.toml [tests]" is. Constructors
   below take the specific values and build the message, so a caller cannot
   forget to include them.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ConfigError",
    "GhostError",
    "ModelNotFoundError",
    "ProjectNotInitializedError",
    "ProviderAuthenticationError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderUnavailableError",
]


class GhostError(Exception):
    """Base class for every error Ghost raises on purpose.

    Catching this means "Ghost hit a condition it understands and can explain to
    the user." Anything that escapes as a non-``GhostError`` is a defect in
    Ghost and should surface with its full traceback rather than be swallowed.
    """


class ConfigError(GhostError):
    """``ghost.toml`` (or an environment override) is missing, malformed, or invalid."""


class ProjectNotInitializedError(GhostError):
    """No ``ghost.toml`` was found, so there is no project to operate on.

    Deliberately distinct from :class:`ConfigError`: this one is recoverable by
    running ``ghost init``, and the CLI says exactly that. A config that exists
    but is broken needs a human to edit it, which is a different conversation.
    """

    def __init__(self, searched_from: Path) -> None:
        self.searched_from = searched_from
        message = (
            f"no ghost.toml found in {searched_from} or any parent directory. "
            f"Run 'ghost init' in your project root to create one."
        )
        super().__init__(message)


class ProviderError(GhostError):
    """Base class for all errors originating from LLM providers."""

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        formatted = f"[{provider}] {message}"
        super().__init__(formatted)


class ProviderAuthenticationError(ProviderError):
    """Missing or invalid API key for a cloud provider."""

    def __init__(self, provider: str, env_var: str | None = None) -> None:
        hint = f" Set {env_var} in your environment or .env file." if env_var else ""
        message = f"no API key found for provider {provider!r}.{hint}"
        super().__init__(provider, message)


class ProviderRateLimitError(ProviderError):
    """Rate limit retries exhausted for a provider."""

    def __init__(self, provider: str, attempts: int, details: str | None = None) -> None:
        detail_msg = f": {details}" if details else "."
        message = f"rate limit exceeded after {attempts} attempt(s){detail_msg}"
        super().__init__(provider, message)


class ProviderUnavailableError(ProviderError):
    """The provider endpoint is offline or unreachable."""

    def __init__(self, provider: str, endpoint: str, details: str | None = None) -> None:
        detail_msg = f" ({details})" if details else ""
        message = f"endpoint {endpoint!r} unreachable{detail_msg}."
        super().__init__(provider, message)


class ModelNotFoundError(ProviderError):
    """The requested model is not supported or not available from this provider."""

    def __init__(
        self, provider: str, model: str, available_models: list[str] | None = None
    ) -> None:
        hint = (
            f" Available models: {', '.join(sorted(available_models))}" if available_models else ""
        )
        message = f"model {model!r} not found for provider {provider!r}.{hint}"
        super().__init__(provider, message)
