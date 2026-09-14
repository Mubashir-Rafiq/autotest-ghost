"""Tests for the command-line surface.

These use click's :class:`~click.testing.CliRunner`, which invokes a command
in-process and captures its output and exit code -- no subprocess, so the whole
file runs in milliseconds.
"""

from __future__ import annotations

import sys

from click.testing import CliRunner

import ghost
from ghost.cli import cli, main


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
