"""Ghost's configuration system.

Owns: ``ghost.toml``, ``.env``, and environment variable loading into typed,
validated, immutable :class:`GhostConfig` models; project root discovery;
default template generation; and API key resolution order.

Does NOT: execute tests (that is ``runner.py``'s job), call LLM APIs
(``chat.py`` / ``providers.py``), print to console (``console.py`` / ``cli.py``),
or watch filesystem events (``watcher.py``).

Guarantees:
1. Frozen models with ``extra="forbid"`` -- unrecognized keys produce an
   actionable error naming the offending key and section.
2. Layered precedence: defaults -> ``ghost.toml`` -> ``.env`` -> environment.
3. Exactly one template-writing function (:func:`write_default_config`),
   preventing drift between CLI and initialization paths.
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path
from typing import Any, Final, Literal

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    field_validator,
)
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

from ghost.errors import ConfigError, ProjectNotInitializedError

__all__ = [
    "DEFAULT_CONFIG_TEMPLATE",
    "AIConfig",
    "GhostConfig",
    "ProjectConfig",
    "ProviderName",
    "ScannerConfig",
    "TestConfig",
    "TestFramework",
    "WatcherConfig",
    "find_project_root",
    "generate_default_config",
    "get_api_key",
    "get_config",
    "load_config",
    "write_default_config",
]

ProviderName = Literal["groq", "openai", "ollama", "anthropic", "openrouter", "lmstudio", "custom"]
TestFramework = Literal["pytest", "unittest"]

_PROVIDER_ENV_VARS: Final[dict[str, tuple[str, ...]]] = {
    "groq": ("GROQ_API_KEY", "GROQ_API_KEY3", "GHOST_API_KEY"),
    "openai": ("OPENAI_API_KEY", "GHOST_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "GHOST_API_KEY"),
    "openrouter": ("OPENROUTER_API_KEY", "GHOST_API_KEY"),
    "ollama": ("GHOST_API_KEY",),
    "lmstudio": ("GHOST_API_KEY",),
    "custom": ("GHOST_API_KEY",),
}

DEFAULT_CONFIG_TEMPLATE: Final[str] = """[project]
name = "{name}"
language = "python"

[ai]
provider = "{provider}"
model = "{model}"
rate_limit_rpm = 30
# base_url = "http://localhost:11434/v1"  # optional, for local/custom endpoints

[scanner]
ignore_dirs = [
    ".venv", "venv", "node_modules", ".git", "__pycache__",
    "dist", "build", ".ghost", "tests", ".tox", ".pytest_cache", ".mypy_cache",
]
ignore_files = ["setup.py", "conftest.py", "__init__.py"]

[tests]
framework = "{framework}"
output_dir = "tests"
auto_heal = true
max_heal_attempts = 3
use_judge = true

[watcher]
debounce_seconds = 15
patterns = ["*.py"]
"""


class ProjectConfig(BaseModel):
    """Metadata describing the target project being tested."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "my-app"
    language: str = "python"

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        lowered = value.lower()
        if lowered != "python":
            msg = f"unsupported language {value!r}; Ghost currently only supports 'python'"
            raise ValueError(msg)
        return lowered


class AIConfig(BaseModel):
    """Settings controlling LLM interactions, provider selection, and rate limits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: ProviderName = "groq"
    model: str = "openai/gpt-oss-120b"
    rate_limit_rpm: int = Field(default=30, gt=0)
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_retries: int = Field(default=5, ge=0)


class ScannerConfig(BaseModel):
    """Rules controlling which directories and files Ghost indexes for AST context."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            ".venv",
            "venv",
            "node_modules",
            ".git",
            "__pycache__",
            "dist",
            "build",
            ".ghost",
            "tests",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
        ]
    )
    ignore_files: list[str] = Field(
        default_factory=lambda: [
            "setup.py",
            "conftest.py",
            "__init__.py",
        ]
    )


class TestConfig(BaseModel):
    """Settings controlling test execution, self-healing, and the judge safety valve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    framework: TestFramework = "pytest"
    output_dir: str = "tests"
    auto_heal: bool = True
    max_heal_attempts: int = Field(default=3, ge=0)
    use_judge: bool = True


class WatcherConfig(BaseModel):
    """Settings controlling filesystem debounce timing and path patterns."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    debounce_seconds: int = Field(default=15, ge=0)
    patterns: list[str] = Field(default_factory=lambda: ["*.py"])


class GhostConfig(BaseSettings):
    """The root configuration object for a Ghost-managed project.

    Combines defaults, ``ghost.toml``, ``.env``, and environment variables into
    a single frozen, validated model.
    """

    model_config = SettingsConfigDict(
        frozen=True,
        extra="forbid",
        env_nested_delimiter="__",
        env_prefix="GHOST_",
    )

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    tests: TestConfig = Field(default_factory=TestConfig)
    watcher: WatcherConfig = Field(default_factory=WatcherConfig)

    _env_values: dict[str, str] = PrivateAttr(default_factory=dict)

    def resolve_api_key(self) -> str | None:
        """Resolve the API key for the configured provider according to precedence."""
        if self.ai.api_key:
            return self.ai.api_key
        from_env = get_api_key(self.ai.provider)
        if from_env:
            return from_env
        var_names = _PROVIDER_ENV_VARS.get(self.ai.provider.lower(), ("GHOST_API_KEY",))
        for name in var_names:
            val = self._env_values.get(name)
            if val and val.strip():
                return val.strip()
        return None

    def resolve_base_url(self) -> str | None:
        """Resolve the provider base URL according to precedence."""
        if self.ai.base_url:
            return self.ai.base_url
        ghost_base_url = os.environ.get("GHOST_BASE_URL") or self._env_values.get("GHOST_BASE_URL")
        if ghost_base_url and ghost_base_url.strip():
            return ghost_base_url.strip()
        if self.ai.provider == "ollama":
            ollama_host = os.environ.get("OLLAMA_HOST") or self._env_values.get("OLLAMA_HOST")
            if ollama_host and ollama_host.strip():
                return ollama_host.strip()
        return None


def get_api_key(provider: str) -> str | None:
    """Return the first non-empty API key found in the environment for *provider*."""
    var_names = _PROVIDER_ENV_VARS.get(provider.lower(), ("GHOST_API_KEY",))
    for name in var_names:
        val = os.environ.get(name)
        if val and val.strip():
            return val.strip()
    return None


def generate_default_config(
    name: str = "my-app",
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
    framework: str = "pytest",
) -> str:
    """Generate the canonical ghost.toml template content."""
    return DEFAULT_CONFIG_TEMPLATE.format(
        name=name,
        provider=provider,
        model=model,
        framework=framework,
    )


def write_default_config(
    target_path: Path,
    *,
    name: str = "my-app",
    provider: str = "groq",
    model: str = "openai/gpt-oss-120b",
    framework: str = "pytest",
    overwrite: bool = False,
) -> Path:
    """Write the canonical ghost.toml template to *target_path*.

    If *target_path* is a directory, writes to ``target_path / "ghost.toml"``.
    """
    dest = target_path / "ghost.toml" if target_path.is_dir() else target_path
    if dest.exists() and not overwrite:
        msg = f"{dest} already exists. Use overwrite=True to replace it."
        raise ConfigError(msg)
    content = generate_default_config(
        name=name,
        provider=provider,
        model=model,
        framework=framework,
    )
    dest.write_text(content, encoding="utf-8")
    return dest


def find_project_root(start_path: Path | None = None) -> Path | None:
    """Search upward from *start_path* (or CWD) for the nearest directory containing ghost.toml."""
    current = (start_path or Path.cwd()).resolve()
    if current.is_file():
        current = current.parent
    for parent in (current, *current.parents):
        if (parent / "ghost.toml").is_file():
            return parent
    return None


def _format_validation_error(err: ValidationError, filename: str = "ghost.toml") -> str:
    """Format Pydantic ValidationError into an actionable diagnostic naming the key."""
    messages: list[str] = []
    for e in err.errors():
        loc = e["loc"]
        msg = e["msg"]
        err_type = e["type"]
        if err_type == "extra_forbidden":
            match loc:
                case (section, key, *_):
                    messages.append(f"unknown key {key!r} in {filename} [{section}]")
                case (section,):
                    messages.append(f"unknown section [{section}] in {filename}")
                case _:
                    key_path = ".".join(str(x) for x in loc)
                    messages.append(f"unknown key {key_path!r} in {filename}")
        else:
            field_name = ".".join(str(x) for x in loc)
            messages.append(f"invalid value for {field_name!r} in {filename}: {msg}")
    return "; ".join(messages)


class _FilteredDotEnvSource(DotEnvSettingsSource):
    """DotEnv settings source filtering out variables not matching model fields."""

    @override
    def __call__(self) -> dict[str, Any]:
        d = super().__call__()
        return {k: v for k, v in d.items() if k in self.settings_cls.model_fields}


def _resolve_config_paths(
    project_path: Path | None,
    *,
    must_exist: bool,
) -> tuple[Path | None, Path | None, str]:
    """Resolve (toml_path, env_path, filename) from project_path or CWD."""
    if project_path is not None:
        resolved = project_path.resolve()
        if resolved.is_file():
            return resolved, resolved.parent / ".env", resolved.name
        if (resolved / "ghost.toml").is_file():
            return resolved / "ghost.toml", resolved / ".env", "ghost.toml"
        root = find_project_root(resolved)
        if root is not None:
            return root / "ghost.toml", root / ".env", "ghost.toml"
        if must_exist:
            raise ProjectNotInitializedError(project_path)
        return None, None, "ghost.toml"

    root = find_project_root()
    if root is not None:
        return root / "ghost.toml", root / ".env", "ghost.toml"
    if must_exist:
        raise ProjectNotInitializedError(Path.cwd())
    return None, None, "ghost.toml"


def load_config(
    project_path: Path | None = None,
    *,
    must_exist: bool = False,
    **kwargs: Any,
) -> GhostConfig:
    """Load GhostConfig respecting precedence: defaults -> ghost.toml -> .env -> environment.

    Parameters
    ----------
    project_path:
        Optional path to a ``ghost.toml`` file or project directory. If omitted,
        searches upward from the current working directory.
    must_exist:
        If ``True``, raises :class:`~ghost.errors.ProjectNotInitializedError` if
        no ``ghost.toml`` is found. If ``False``, returns defaults when absent.
    **kwargs:
        Explicit overrides passed directly to ``GhostConfig``.
    """
    toml_path, env_path, filename = _resolve_config_paths(project_path, must_exist=must_exist)

    if toml_path is not None:
        try:
            with toml_path.open("rb") as f:
                tomllib.load(f)
        except tomllib.TOMLDecodeError as err:
            msg = f"malformed {filename}: {err}"
            raise ConfigError(msg) from err

    env_values: dict[str, str] = {}
    if env_path is not None and env_path.is_file():
        raw_env = dotenv.dotenv_values(env_path)
        env_values = {k: v for k, v in raw_env.items() if v is not None}

    class _ProjectGhostConfig(GhostConfig):
        def __init__(self, **init_kwargs: Any) -> None:
            super().__init__(**init_kwargs)
            self._env_values = env_values

        @classmethod
        @override
        def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
        ) -> tuple[PydanticBaseSettingsSource, ...]:
            _ = (dotenv_settings, file_secret_settings)
            sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
            if env_path is not None and env_path.is_file():
                sources.append(_FilteredDotEnvSource(settings_cls, env_file=env_path))
            if toml_path is not None and toml_path.is_file():
                sources.append(TomlConfigSettingsSource(settings_cls, toml_file=toml_path))
            return tuple(sources)

    try:
        config = _ProjectGhostConfig(**kwargs)
    except ValidationError as err:
        msg = _format_validation_error(err, filename=filename)
        raise ConfigError(msg) from err
    else:
        return config


get_config = load_config
