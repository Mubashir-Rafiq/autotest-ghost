"""Tests for the command-line surface.

These use click's :class:`~click.testing.CliRunner`, which invokes a command
in-process and captures its output and exit code -- no subprocess, so the whole
file runs in milliseconds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from click.testing import CliRunner

import ghost
from ghost.cli import cli, main
from ghost.daemon import DaemonStatus
from ghost.errors import ConfigError, ExitCode, ProjectNotInitializedError
from ghost.history import HistoryTracker, UsageTracker
from ghost.pipeline import PipelineResult, PipelineStatus, TestPipeline
from ghost.runner import TestRunResult
from ghost.watcher import FileWatcher


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


def test_run_tests_command_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    test_f = tmp_path / "test_ok.py"
    test_f.write_text("def test_ok(): assert True\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["run-tests", "test_ok.py"])
    assert result.exit_code == 0
    assert "PASS: test_ok.py" in result.output


def test_run_tests_command_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    test_f = tmp_path / "test_bad.py"
    test_f.write_text("def test_bad(): assert 1 == 2\n", encoding="utf-8")

    result = CliRunner().invoke(cli, ["run-tests", "test_bad.py"])
    assert result.exit_code != 0
    assert "FAIL (LOGIC): test_bad.py" in result.output
    assert "AssertionError" in result.output


def test_generate_command_help() -> None:
    result = CliRunner().invoke(cli, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--force" in result.output
    assert "--heal" in result.output
    assert "--judge" in result.output
    assert "--timeout" in result.output


def test_generate_command_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")

    code = main(["generate", "missing.py"])
    assert code != 0
    captured = capsys.readouterr()
    assert "does not exist" in captured.err


def test_generate_command_non_py_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("Hello\n", encoding="utf-8")

    code = main(["generate", "notes.txt"])
    assert code != 0
    captured = capsys.readouterr()
    assert "only Python (.py) files are supported" in captured.err


def test_generate_command_handwritten_test_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "# Human test\ndef test_add(): pass\n", encoding="utf-8"
    )

    code = main(["generate", "calc.py"])
    assert code != 0
    captured = capsys.readouterr()
    assert "refusing to overwrite hand-written test file" in captured.err


def test_generate_command_existing_ghost_test_prompt_declined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_calc.py").write_text(
        "# Generated at: 24-09-2026 12:00:00 | Source: calc.py\ndef test_add(): pass\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["generate", "calc.py"], input="n\n")
    assert result.exit_code == 0
    assert "Generation cancelled" in result.output


def test_generate_command_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

    test_file = tmp_path / "tests" / "test_calc.py"

    async def mock_run(self: TestPipeline, *args: Any, **kwargs: Any) -> PipelineResult:
        return PipelineResult(
            source_file=tmp_path / "calc.py",
            test_file=test_file,
            status=PipelineStatus.PASSED,
            passed=True,
            attempts=0,
        )

    monkeypatch.setattr(TestPipeline, "run", mock_run)

    result = CliRunner().invoke(cli, ["generate", "calc.py", "--force"])
    assert result.exit_code == 0
    assert "PASS:" in result.output
    assert str(test_file) in result.output


def test_generate_command_failure_exits_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "broken.py").write_text("x = 1\n", encoding="utf-8")

    test_file = tmp_path / "tests" / "test_broken.py"

    async def mock_run(self: TestPipeline, *args: Any, **kwargs: Any) -> PipelineResult:
        return PipelineResult(
            source_file=tmp_path / "broken.py",
            test_file=test_file,
            status=PipelineStatus.FAILED,
            passed=False,
            attempts=3,
            error_message="healing budget exhausted",
        )

    monkeypatch.setattr(TestPipeline, "run", mock_run)

    result = CliRunner().invoke(cli, ["generate", "broken.py", "--force"])
    assert result.exit_code == 1
    assert "FAIL:" in result.output
    assert "healing budget exhausted" in result.output


def test_generate_command_if_changed_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")
    (tmp_path / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

    test_file = tmp_path / "tests" / "test_calc.py"

    async def mock_run(self: TestPipeline, *args: Any, **kwargs: Any) -> PipelineResult:
        if kwargs.get("if_changed"):
            return PipelineResult(
                source_file=tmp_path / "calc.py",
                test_file=test_file,
                status=PipelineStatus.SKIPPED,
                passed=True,
                attempts=0,
            )
        return PipelineResult(
            source_file=tmp_path / "calc.py",
            test_file=test_file,
            status=PipelineStatus.PASSED,
            passed=True,
            attempts=0,
        )

    monkeypatch.setattr(TestPipeline, "run", mock_run)

    result = CliRunner().invoke(cli, ["generate", "calc.py", "--force", "--if-changed"])
    assert result.exit_code == 0
    assert "SKIPPED: calc.py is unchanged (--if-changed)." in result.output


def test_watch_command_help() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--help"])
    assert result.exit_code == 0
    assert "Watch Python source files for changes" in result.output
    assert "--verbose" in result.output
    assert "--heal / --no-heal" in result.output
    assert "--judge / --no-judge" in result.output


def test_watch_auto_initializes_ghost_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "ghost.toml"
    assert not config_file.exists()

    def mock_start(self: FileWatcher) -> None:
        # Immediately interrupt to terminate the watch command cleanly
        raise KeyboardInterrupt

    monkeypatch.setattr(FileWatcher, "start", mock_start)

    runner = CliRunner()
    result = runner.invoke(cli, ["watch"])
    assert result.exit_code == 0
    assert "Initialized ghost.toml" in result.output
    assert config_file.is_file()
    assert "Ghost watching" in result.output
    assert "Stopping watcher..." in result.output


def test_watch_command_runs_with_options(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")

    started = False

    def mock_start(self: FileWatcher) -> None:
        nonlocal started
        started = True
        assert self.config.tests.auto_heal is False
        assert self.config.tests.use_judge is False
        raise KeyboardInterrupt

    monkeypatch.setattr(FileWatcher, "start", mock_start)

    runner = CliRunner()
    result = runner.invoke(cli, ["watch", "--no-heal", "--no-judge", "--verbose"])
    assert result.exit_code == 0
    assert started
    assert "Ghost watching" in result.output
    assert "Watcher stopped." in result.output


def test_exit_code_constants() -> None:
    assert ExitCode.SUCCESS.value == 0
    assert ExitCode.ERROR.value == 1
    assert ExitCode.USAGE_ERROR.value == 2
    assert ExitCode.INTERRUPTED.value == 130


def test_init_command_help() -> None:
    result = CliRunner().invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "--provider" in result.output
    assert "--model" in result.output
    assert "--framework" in result.output


def test_init_fresh_project_with_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "app.py").write_text("def hello() -> str: return 'hi'\n", encoding="utf-8")

    # Inputs:
    # 1. AI provider prompt: accept default (Enter)
    # 2. API key prompt: skip (Enter)
    # 3. Model prompt: accept default (Enter)
    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="\n\n\n")
    assert result.exit_code == 0, result.output
    assert "Created " in result.output
    assert "Indexed 1 file(s)" in result.output
    assert "Ghost initialized. Run 'ghost watch' to start." in result.output

    config_file = tmp_path / "ghost.toml"
    assert config_file.is_file()
    assert 'provider = "groq"' in config_file.read_text(encoding="utf-8")

    context_file = tmp_path / ".ghost" / "context.json"
    assert context_file.is_file()
    data = json.loads(context_file.read_text(encoding="utf-8"))
    assert "app.py" in data


def test_init_with_options_and_api_key_saving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    # Provider passed via -p (openai), model via -m (gpt-4o), framework via -f (unittest)
    # Wizard still confirms provider; then prompts for API key and offers to save to .env
    # Inputs:
    # 1. AI provider confirmation: accept default openai (Enter)
    # 2. Enter API key: sk-secret-12345
    # 3. Save to .env?: y
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["init", "-p", "openai", "-m", "gpt-4o", "-f", "unittest"],
        input="\nsk-secret-12345\ny\n",
    )
    assert result.exit_code == 0, result.output

    config_file = tmp_path / "ghost.toml"
    assert config_file.is_file()
    text = config_file.read_text(encoding="utf-8")
    assert 'provider = "openai"' in text
    assert 'model = "gpt-4o"' in text
    assert 'framework = "unittest"' in text

    env_file = tmp_path / ".env"
    assert env_file.is_file()
    assert "OPENAI_API_KEY=sk-secret-12345" in env_file.read_text(encoding="utf-8")


def test_init_existing_config_declines_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("# preserved\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="n\n")
    assert result.exit_code == 0
    assert "Init cancelled." in result.output
    assert config_file.read_text(encoding="utf-8") == "# preserved\n"


def test_init_existing_config_confirms_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config_file = tmp_path / "ghost.toml"
    config_file.write_text("# preserved\n", encoding="utf-8")

    # Inputs:
    # 1. Overwrite: y
    # 2. Provider: Enter
    # 3. API key: Enter
    # 4. Model: Enter
    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="y\n\n\n\n")
    assert result.exit_code == 0, result.output
    assert config_file.read_text(encoding="utf-8") != "# preserved\n"
    assert "Created " in result.output


def test_init_existing_env_key_accepted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1234567890abcdef")

    # Inputs:
    # 1. Provider: Enter
    # 2. Use this key?: y (Enter)
    # 3. Model: Enter
    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="\n\n\n")
    assert result.exit_code == 0
    assert "Found API key in $GROQ_API_KEY: gsk_************" in result.output


def test_init_existing_env_key_declined_and_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_1234567890abcdef")

    # Inputs:
    # 1. Provider: Enter
    # 2. Use this key?: n
    # 3. Enter API key: gsk_custom999
    # 4. Save to .env?: y
    # 5. Model: Enter
    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="\nn\ngsk_custom999\ny\n\n")
    assert result.exit_code == 0

    env_file = tmp_path / ".env"
    assert env_file.is_file()
    assert "GROQ_API_KEY=gsk_custom999" in env_file.read_text(encoding="utf-8")


def test_init_ollama_local_provider_skips_key_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    # Inputs:
    # 1. Provider: ollama
    # 2. Model: Enter (llama3:latest)
    runner = CliRunner()
    result = runner.invoke(cli, ["init"], input="ollama\n\n")
    assert result.exit_code == 0, result.output
    assert "Enter " not in result.output

    config_file = tmp_path / "ghost.toml"
    assert 'provider = "ollama"' in config_file.read_text(encoding="utf-8")
    assert 'model = "llama3:latest"' in config_file.read_text(encoding="utf-8")


def test_start_command_help() -> None:
    result = CliRunner().invoke(cli, ["start", "--help"])
    assert result.exit_code == 0
    assert "--detach" in result.output
    assert "--foreground" in result.output


def test_start_uninitialized_project_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["start"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectNotInitializedError)
    assert "no ghost.toml found" in str(result.exception)
    assert main(["start"]) == ExitCode.ERROR


def test_start_foreground_invokes_watch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")

    called = False

    def mock_start(self: FileWatcher) -> None:
        nonlocal called
        called = True
        raise KeyboardInterrupt

    monkeypatch.setattr(FileWatcher, "start", mock_start)

    runner = CliRunner()
    result = runner.invoke(cli, ["start", "--foreground"])
    assert result.exit_code == 0
    assert called
    assert "Ghost watching" in result.output


def test_start_detach_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")

    monkeypatch.setattr("ghost.cli.start_daemon", lambda _root: 12345)

    runner = CliRunner()
    result = runner.invoke(cli, ["start"])
    assert result.exit_code == 0
    assert "Ghost daemon started (PID 12345)." in result.output


def test_start_detach_already_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[ai]\nprovider = 'groq'\n", encoding="utf-8")

    def mock_start(_root: Path) -> int:
        msg = "daemon already running (PID 9999)"
        raise RuntimeError(msg)

    monkeypatch.setattr("ghost.cli.start_daemon", mock_start)

    runner = CliRunner()
    result = runner.invoke(cli, ["start"])
    assert result.exit_code == 1
    assert "daemon already running (PID 9999)" in result.output


def test_stop_command_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ghost.cli.stop_daemon", lambda _root: True)

    runner = CliRunner()
    result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 0
    assert "Ghost daemon stopped." in result.output


def test_stop_command_not_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("ghost.cli.stop_daemon", lambda _root: False)

    runner = CliRunner()
    result = runner.invoke(cli, ["stop"])
    assert result.exit_code == 0
    assert "No daemon is running." in result.output


def test_status_command_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    status = DaemonStatus(
        pid=5555,
        running=True,
        log_tail=["watcher started", "processing test_calc.py"],
    )
    monkeypatch.setattr("ghost.cli.query_daemon_status", lambda _root: status)

    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Ghost daemon is running (PID 5555)." in result.output
    assert "Recent log:" in result.output
    assert "watcher started" in result.output
    assert "processing test_calc.py" in result.output


def test_status_command_not_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    status = DaemonStatus(pid=None, running=False, log_tail=[])
    monkeypatch.setattr("ghost.cli.query_daemon_status", lambda _root: status)

    runner = CliRunner()
    result = runner.invoke(cli, ["status"])
    assert result.exit_code == 0
    assert "Ghost daemon is not running." in result.output


def test_logs_command_no_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["logs"])
    assert result.exit_code == 0
    assert "No log file found." in result.output


def test_logs_command_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / ".ghost" / "daemon.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("line 1\nline 2\nline 3\n", encoding="utf-8")

    runner = CliRunner()
    result = runner.invoke(cli, ["logs", "--lines", "2"])
    assert result.exit_code == 0
    assert "line 2\nline 3" in result.output


def test_logs_command_follow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    log_file = tmp_path / ".ghost" / "daemon.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text("initial log\n", encoding="utf-8")

    def mock_follow(_root: Path, cb: Any) -> None:
        cb("streamed event\n")

    monkeypatch.setattr("ghost.cli.follow_log_stream", mock_follow)

    runner = CliRunner()
    result = runner.invoke(cli, ["logs", "--follow"])
    assert result.exit_code == 0
    assert "initial log" in result.output
    assert "streamed event" in result.output


def test_main_structured_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    # Success returns ExitCode.SUCCESS (0)
    assert main(["version"]) == ExitCode.SUCCESS

    # Usage error returns 2
    assert main(["nonexistent-command"]) == ExitCode.USAGE_ERROR

    # Anticipated GhostError returns ExitCode.ERROR (1)
    assert main(["generate", "nonexistent_file.py"]) == ExitCode.ERROR


def test_stats_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    tracker = UsageTracker(tmp_path)
    tracker.record_call(prompt_tokens=150, completion_tokens=75, is_heal=True)
    tracker.record_success()

    runner = CliRunner()
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "AI Usage & Token Statistics" in result.output
    assert "Prompt Tokens" in result.output
    assert "150" in result.output


def test_history_and_rollback_commands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[project]\nname = 'test'\n")
    src = tmp_path / "mod.py"
    src.write_text("x = 1\n")
    test_file = tmp_path / "tests" / "test_mod.py"
    test_file.parent.mkdir(parents=True)
    test_file.write_text("# initial\n")

    runner = CliRunner()

    # Empty history
    res0 = runner.invoke(cli, ["history"])
    assert res0.exit_code == 0
    assert "No test generation or healing history" in res0.output

    # Record two attempts
    tracker = HistoryTracker(tmp_path)
    tracker.record_attempt(src, attempt=0, test_code="# attempt 0\n", status="generated")
    tracker.record_attempt(
        src, attempt=1, test_code="# attempt 1\n", status="healed", classification="LOGIC"
    )

    # History command for project
    res1 = runner.invoke(cli, ["history"])
    assert res1.exit_code == 0
    assert "mod.py" in res1.output

    # History command for specific file
    res2 = runner.invoke(cli, ["history", str(src)])
    assert res2.exit_code == 0
    assert "History for mod.py" in res2.output
    assert "attempt_0.py" in res2.output

    # Rollback command to attempt 0
    test_file.write_text("# attempt 1\n")
    res_rb = runner.invoke(cli, ["rollback", str(src), "--attempt", "0"])
    assert res_rb.exit_code == 0
    assert "Successfully rolled back" in res_rb.output
    assert test_file.read_text() == "# attempt 0\n"

    # Rollback command using test_file path
    res_rb2 = runner.invoke(cli, ["rollback", str(test_file)])
    assert res_rb2.exit_code == 0
    assert test_file.read_text() == "# attempt 0\n"


def test_run_tests_with_coverage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_file = tmp_path / "test_sample.py"
    test_file.write_text("def test_ok(): pass\n")

    async def mock_run_test(*_args: Any, **_kwargs: Any) -> TestRunResult:
        return TestRunResult(
            test_file=test_file,
            passed=True,
            return_code=0,
            stdout="TOTAL 10 1 90%",
            stderr="",
            coverage_summary="TOTAL 10 1 90%",
        )

    monkeypatch.setattr("ghost.cli.run_test", mock_run_test)

    runner = CliRunner()
    result = runner.invoke(cli, ["run-tests", str(test_file), "--cov"])
    assert result.exit_code == 0
    assert "PASS:" in result.output
    assert "Coverage: TOTAL 10 1 90%" in result.output


def test_generate_batch_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "ghost.toml").write_text("[project]\nname = 'test'\n")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "mod_a.py").write_text("a = 1\n")
    (src_dir / "mod_b.py").write_text("b = 2\n")

    async def mock_pipeline_run(self: Any, source_file: Path, **_kwargs: Any) -> PipelineResult:
        return PipelineResult(
            source_file=source_file,
            test_file=tmp_path / "tests" / f"test_{source_file.name}",
            status=PipelineStatus.PASSED,
            passed=True,
            attempts=0,
        )

    monkeypatch.setattr("ghost.pipeline.TestPipeline.run", mock_pipeline_run)

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--all"])
    assert result.exit_code == 0
    assert "Batch Generation Summary" in result.output
    assert "mod_a.py" in result.output
    assert "mod_b.py" in result.output
