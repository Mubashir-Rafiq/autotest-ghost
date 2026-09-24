"""Tests for the command-line surface.

These use click's :class:`~click.testing.CliRunner`, which invokes a command
in-process and captures its output and exit code -- no subprocess, so the whole
file runs in milliseconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
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
