"""Tests for Ghost's configuration system.

Verifies:
- Frozen models with extra="forbid"
- Layered precedence: defaults -> ghost.toml -> .env -> environment
- Error diagnostics naming the exact offending key and section
- Canonical template generation and writing
- API key resolution hierarchy
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from ghost.config import (
    AIConfig,
    GhostConfig,
    find_project_root,
    generate_default_config,
    get_api_key,
    load_config,
    write_default_config,
)
from ghost.errors import ConfigError, ProjectNotInitializedError


def test_default_config_values() -> None:
    """GhostConfig defaults match the locked decisions in the specification."""
    config = GhostConfig()
    assert config.project.name == "my-app"
    assert config.project.language == "python"
    assert config.ai.provider == "groq"
    assert config.ai.model == "openai/gpt-oss-120b"
    assert config.ai.rate_limit_rpm == 30
    assert config.ai.temperature == 0.1
    assert config.ai.max_retries == 5
    assert config.ai.base_url is None
    assert config.ai.api_key is None
    assert ".venv" in config.scanner.ignore_dirs
    assert "setup.py" in config.scanner.ignore_files
    assert config.tests.framework == "pytest"
    assert config.tests.output_dir == "tests"
    assert config.tests.auto_heal is True
    assert config.tests.max_heal_attempts == 3
    assert config.tests.use_judge is True
    assert config.watcher.debounce_seconds == 15
    assert config.watcher.patterns == ["*.py"]


def test_config_models_are_frozen() -> None:
    """Config instances cannot be modified in place after creation."""
    config = GhostConfig()
    with pytest.raises(ValidationError):
        config.project.name = "mutated"

    with pytest.raises(ValidationError):
        config.ai.rate_limit_rpm = 999

    with pytest.raises(ValidationError):
        config.tests.auto_heal = False


def test_extra_keys_are_forbidden_top_level(tmp_path: Path) -> None:
    """Unrecognized top-level sections raise ConfigError naming the section."""
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("[extra_section]\nfoo = 'bar'\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file, must_exist=True)

    assert "unknown section [extra_section]" in str(exc_info.value)


def test_extra_keys_are_forbidden_nested_section(tmp_path: Path) -> None:
    """Unknown keys within sections name the exact key and section."""
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("[tests]\nauto_heel = true\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file, must_exist=True)

    # SPEC.md 4.1 / errors.py requirement: unknown key 'auto_heel' in ghost.toml [tests]
    assert "unknown key 'auto_heel' in ghost.toml [tests]" in str(exc_info.value)


def test_malformed_toml_syntax_raises_clear_error(tmp_path: Path) -> None:
    """Syntax errors in ghost.toml produce a malformed diagnostic, not a traceback."""
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("[project\nname = unquoted\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file, must_exist=True)

    assert "malformed ghost.toml" in str(exc_info.value)


def test_invalid_field_types_raise_clear_error(tmp_path: Path) -> None:
    """Non-integer rate_limit_rpm produces an actionable diagnostic."""
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("[ai]\nrate_limit_rpm = 'thirty'\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file, must_exist=True)

    assert "invalid value for 'ai.rate_limit_rpm'" in str(exc_info.value)


def test_unsupported_language_raises_clear_error(tmp_path: Path) -> None:
    """Language other than python produces an error."""
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("[project]\nlanguage = 'rust'\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file, must_exist=True)

    assert "unsupported language 'rust'" in str(exc_info.value)


def test_unsupported_framework_raises_clear_error(tmp_path: Path) -> None:
    """Test framework outside pytest/unittest is rejected."""
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("[tests]\nframework = 'jest'\n", encoding="utf-8")

    with pytest.raises(ConfigError) as exc_info:
        load_config(config_file, must_exist=True)

    assert "invalid value for 'tests.framework'" in str(exc_info.value)


def test_layered_precedence_defaults_to_toml_to_dotenv_to_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify exact 4-tier precedence: defaults -> ghost.toml -> .env -> environment."""
    # 1. Defaults
    default_cfg = load_config(tmp_path)
    assert default_cfg.ai.model == "openai/gpt-oss-120b"

    # 2. ghost.toml overrides defaults
    config_file = tmp_path / "ghost.toml"
    config_file.write_text('[ai]\nmodel = "from-toml"\n', encoding="utf-8")
    toml_cfg = load_config(tmp_path)
    assert toml_cfg.ai.model == "from-toml"

    # 3. .env overrides ghost.toml
    env_file = tmp_path / ".env"
    env_file.write_text("GHOST_AI__MODEL=from-dotenv\n", encoding="utf-8")
    dotenv_cfg = load_config(tmp_path)
    assert dotenv_cfg.ai.model == "from-dotenv"

    # 4. Environment variable overrides .env
    monkeypatch.setenv("GHOST_AI__MODEL", "from-environment")
    env_cfg = load_config(tmp_path)
    assert env_cfg.ai.model == "from-environment"


def test_find_project_root_walks_ancestors(tmp_path: Path) -> None:
    """find_project_root finds ghost.toml in parent directories."""
    project_dir = tmp_path / "my_project"
    sub_dir = project_dir / "src" / "pkg"
    sub_dir.mkdir(parents=True)
    (project_dir / "ghost.toml").write_text("[project]\nname='sub'\n", encoding="utf-8")

    assert find_project_root(sub_dir) == project_dir
    assert find_project_root(project_dir) == project_dir
    assert find_project_root(tmp_path) is None


def test_load_config_must_exist_behavior(tmp_path: Path) -> None:
    """load_config respects must_exist flag."""
    # When missing and must_exist=False -> returns defaults
    cfg = load_config(tmp_path, must_exist=False)
    assert cfg.project.name == "my-app"

    # When missing and must_exist=True -> raises ProjectNotInitializedError
    with pytest.raises(ProjectNotInitializedError) as exc_info:
        load_config(tmp_path, must_exist=True)

    assert "no ghost.toml found" in str(exc_info.value)
    assert "Run 'ghost init'" in str(exc_info.value)


def test_api_key_resolution_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """API key resolves via explicit config -> provider-specific env var -> GHOST_API_KEY."""
    # Clean out any ambient keys
    for var in ("GROQ_API_KEY", "GROQ_API_KEY3", "GHOST_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    # 1. No key set
    assert get_api_key("groq") is None

    # 2. GHOST_API_KEY fallback
    monkeypatch.setenv("GHOST_API_KEY", "ghost-key-123")
    assert get_api_key("groq") == "ghost-key-123"

    # 3. Legacy GROQ_API_KEY3 precedes GHOST_API_KEY
    monkeypatch.setenv("GROQ_API_KEY3", "groq-key-3")
    assert get_api_key("groq") == "groq-key-3"

    # 4. GROQ_API_KEY precedes GROQ_API_KEY3
    monkeypatch.setenv("GROQ_API_KEY", "groq-primary-key")
    assert get_api_key("groq") == "groq-primary-key"

    # 5. Config with explicit api_key in ghost.toml wins over environment
    config_file = tmp_path / "ghost.toml"
    config_file.write_text('[ai]\napi_key = "explicit-in-toml"\n', encoding="utf-8")
    cfg = load_config(config_file)
    assert cfg.resolve_api_key() == "explicit-in-toml"


def test_base_url_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """Base URL falls back to GHOST_BASE_URL or OLLAMA_HOST when provider is ollama."""
    monkeypatch.delenv("GHOST_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)

    cfg = GhostConfig()
    assert cfg.resolve_base_url() is None

    monkeypatch.setenv("GHOST_BASE_URL", "http://custom:8080/v1")
    assert cfg.resolve_base_url() == "http://custom:8080/v1"

    # Ollama provider uses OLLAMA_HOST if GHOST_BASE_URL is not set
    monkeypatch.delenv("GHOST_BASE_URL")
    monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
    ollama_cfg = GhostConfig(ai=AIConfig(provider="ollama"))
    assert ollama_cfg.resolve_base_url() == "http://localhost:11434"


def test_single_canonical_template_generates_valid_config(tmp_path: Path) -> None:
    """The template produced by generate_default_config parses with zero validation errors."""
    content = generate_default_config(
        name="test-app",
        provider="groq",
        model="openai/gpt-oss-120b",
        framework="pytest",
    )
    dest = tmp_path / "ghost.toml"
    dest.write_text(content, encoding="utf-8")

    loaded = load_config(dest, must_exist=True)
    assert loaded.project.name == "test-app"
    assert loaded.ai.provider == "groq"
    assert loaded.ai.model == "openai/gpt-oss-120b"
    assert loaded.tests.framework == "pytest"


def test_write_default_config_creates_file_and_protects_overwrite(tmp_path: Path) -> None:
    """write_default_config writes ghost.toml and refuses to overwrite without flag."""
    written = write_default_config(tmp_path, name="written-app")
    assert written.is_file()
    assert written.name == "ghost.toml"
    assert "written-app" in written.read_text(encoding="utf-8")

    # Second write without overwrite=True raises ConfigError
    with pytest.raises(ConfigError) as exc_info:
        write_default_config(tmp_path, overwrite=False)
    assert "already exists" in str(exc_info.value)

    # With overwrite=True, it succeeds
    write_default_config(tmp_path, name="updated-app", overwrite=True)
    assert "updated-app" in written.read_text(encoding="utf-8")
