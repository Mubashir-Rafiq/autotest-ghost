"""Tests for the command-line surface.

These use click's :class:`~click.testing.CliRunner`, which invokes a command
in-process and captures its output and exit code -- no subprocess, so the whole
file runs in milliseconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest
import respx
from click.testing import CliRunner

import ghost
from ghost.cli import cli, main
from ghost.errors import ConfigError, ProjectNotInitializedError


def test_bare_invocation_shows_help() -> None:
    result = CliRunner().invoke(cli, [])
    assert result.exit_code == 0
    assert "Usage:" in result.output
    assert "doctor" in result.output


def test_version_flag_reports_the_package_version() -> None:
    result = CliRunner().invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert ghost.__version__ in result.output


def test_version_command_reports_ghost_python_and_platform() -> None:
    result = CliRunner().invoke(cli, ["version"])
    assert result.exit_code == 0
    assert ghost.__version__ in result.output
    assert sys.executable in result.output


def test_doctor_reports_every_required_dependency() -> None:
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0, result.output
    for distribution in ("click", "pydantic", "httpx", "groq", "watchdog", "pytest"):
        assert distribution in result.output
    assert "MISSING" not in result.output


def test_doctor_lists_optional_providers_without_failing() -> None:
    """Optional extras are absent by default; that is normal, not an error."""
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "openai" in result.output
    assert "anthropic" in result.output


def test_unknown_command_exits_with_usage_error() -> None:
    result = CliRunner().invoke(cli, ["definitely-not-a-command"])
    assert result.exit_code != 0


def test_main_returns_zero_for_a_successful_command() -> None:
    """``main`` returns an exit code rather than raising ``SystemExit``."""
    assert main(["version"]) == 0


def test_main_returns_nonzero_for_an_unknown_command() -> None:
    assert main(["definitely-not-a-command"]) != 0


def test_config_command_outside_project_reports_uninitialized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running ghost config outside any project gives a clear ProjectNotInitializedError."""
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["config"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotInitializedError)
    assert "no ghost.toml found" in str(result.exception)
    assert "Run 'ghost init'" in str(result.exception)


def test_config_command_shows_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ghost config prints the contents of ghost.toml when valid."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[project]\nname = 'cli-test'\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["config"])
    assert result.exit_code == 0
    assert "name = 'cli-test'" in result.output


def test_config_command_with_show_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Both --show and -s output the config identically."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[project]\nname = 'show-flag-test'\n", encoding="utf-8")

    res_long = CliRunner().invoke(cli, ["config", "--show"])
    assert res_long.exit_code == 0
    assert "show-flag-test" in res_long.output

    res_short = CliRunner().invoke(cli, ["config", "-s"])
    assert res_short.exit_code == 0
    assert "show-flag-test" in res_short.output


def test_config_command_malformed_config_reports_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ghost config reports the exact malformed key and exits with code 1."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[tests]\nauto_heel = true\n", encoding="utf-8")
    result = CliRunner().invoke(cli, ["config"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ConfigError)
    assert "unknown key 'auto_heel' in ghost.toml [tests]" in str(result.exception)


def test_main_config_reports_error_to_stderr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() handles GhostError and prints Error: ... to stderr."""
    monkeypatch.chdir(tmp_path)
    code = main(["config"])
    assert code == 1
    captured = capsys.readouterr()
    assert "Error: no ghost.toml found" in captured.err


def test_doctor_reports_configuration_status() -> None:
    """ghost doctor includes configuration discovery in its report."""
    result = CliRunner().invoke(cli, ["doctor"])
    assert result.exit_code == 0
    assert "Configuration" in result.output
    assert "ghost.toml" in result.output


def test_providers_command_shows_providers_and_popular_models() -> None:
    """ghost providers lists supported providers and popular models table."""
    result = CliRunner().invoke(cli, ["providers"])
    assert result.exit_code == 0
    assert "Supported Providers" in result.output
    assert "groq" in result.output
    assert "Popular Models" in result.output
    assert "openai/gpt-oss-120b" in result.output


def test_models_command_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """ghost models fails with an authentication error when no key is set."""
    for var in ("GROQ_API_KEY", "GROQ_API_KEY3", "GHOST_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    result = CliRunner().invoke(cli, ["models"])
    assert result.exit_code != 0
    assert "no API key found" in str(result.exception)


def test_main_models_missing_key_reports_to_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main(["models"]) routes ProviderAuthenticationError through the error boundary."""
    for var in ("GROQ_API_KEY", "GROQ_API_KEY3", "GHOST_API_KEY"):
        monkeypatch.delenv(var, raising=False)

    code = main(["models"])
    assert code == 1
    captured = capsys.readouterr()
    assert "no API key found" in captured.err


@respx.mock
def test_models_command_success_with_live_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """ghost models outputs the live models list when authorized."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_key_123")
    respx.get("https://api.groq.com/openai/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "openai/gpt-oss-120b"}, {"id": "openai/gpt-oss-20b"}]},
        )
    )

    result = CliRunner().invoke(cli, ["models"])
    assert result.exit_code == 0
    assert "Available models" in result.output
    assert "openai/gpt-oss-120b" in result.output
    assert "openai/gpt-oss-20b" in result.output


def test_index_command_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def hello(name: str) -> str: pass\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["index"])
    assert result.exit_code == 0
    assert "Indexed 1 file(s)" in result.output
    assert (tmp_path / ".ghost" / "context.json").is_file()


def test_index_command_show(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text(
        "def greet() -> None:\n    '''Say hi.'''\n    pass\n", encoding="utf-8"
    )

    result = CliRunner().invoke(cli, ["index", "--show"])
    assert result.exit_code == 0
    assert "Ghost AST Project Index" in result.output
    assert "app.py" in result.output
    assert "greet() -> None" in result.output


def test_index_command_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "app.py").write_text("def add(a: int, b: int) -> int: pass\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["index", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "app.py" in data
    assert "add(a: int, b: int) -> int" in data["app.py"]


def test_prompt_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ghost prompt FILE prints the prompt without making an API call."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    src = tmp_path / "app.py"
    src.write_text("def multiply(x: int, y: int) -> int: return x * y\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["prompt", "app.py"])
    assert result.exit_code == 0
    assert "Output RAW PYTHON CODE ONLY" in result.output
    assert "SOURCE CODE UNDER TEST (`app.py`):" in result.output
    assert "def multiply(x: int, y: int) -> int:" in result.output
    assert "PROJECT TREE:" in result.output
