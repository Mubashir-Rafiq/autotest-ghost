"""Tests for the unified test generation, execution, healing, and judge pipeline."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import pytest

from ghost.client import (
    JudgeOutcome,
    LLMClient,
)
from ghost.config import GhostConfig
from ghost.errors import (
    HandwrittenTestOverwriteError,
    SourceFileNotFoundError,
    UnsupportedFileError,
)
from ghost.pipeline import (
    PipelineEvent,
    PipelineListener,
    PipelineStatus,
    TestPipeline,
    resolve_test_path,
)
from ghost.prompts import HEADER_PREFIX
from ghost.providers import BaseProvider
from ghost.runner import ErrorClassification, TestRunResult


class MockProvider(BaseProvider):
    """Mock LLM provider returning programmed responses."""

    def __init__(self, responses: list[str] | None = None) -> None:
        super().__init__()
        self.responses: list[str] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    @property
    @override
    def name(self) -> str:
        return "mock"

    @override
    async def _call_api(
        self, messages: list[dict[str, str]], model: str, temperature: float
    ) -> str:
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0)
        return (
            f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: dummy.py\n"
            "def test_dummy():\n    assert True\n"
        )

    @override
    async def list_models(self) -> list[str]:
        return ["mock-model"]

    @override
    async def is_available(self) -> bool:
        return True


class RecordingListener(PipelineListener):
    """Listener recording received events for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[PipelineEvent, dict[str, Any]]] = []

    @override
    def on_event(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        self.events.append((event, data))


def test_resolve_test_path_flat(tmp_path: Path) -> None:
    src = tmp_path / "maths.py"
    src.touch()
    resolved = resolve_test_path(src, tmp_path, output_dir="tests")
    assert resolved == tmp_path / "tests" / "test_maths.py"


def test_resolve_test_path_with_src_prefix(tmp_path: Path) -> None:
    src = tmp_path / "src" / "pkg" / "calculator.py"
    src.parent.mkdir(parents=True)
    src.touch()
    resolved = resolve_test_path(src, tmp_path, output_dir="tests")
    assert resolved == tmp_path / "tests" / "pkg" / "test_calculator.py"


def test_resolve_test_path_avoids_basename_collision(tmp_path: Path) -> None:
    """Regression test: identically named files in different subdirectories never collide."""
    auth_views = tmp_path / "auth" / "views.py"
    billing_views = tmp_path / "billing" / "views.py"
    auth_views.parent.mkdir()
    billing_views.parent.mkdir()
    auth_views.touch()
    billing_views.touch()

    path_auth = resolve_test_path(auth_views, tmp_path, output_dir="tests")
    path_billing = resolve_test_path(billing_views, tmp_path, output_dir="tests")

    assert path_auth == tmp_path / "tests" / "auth" / "test_views.py"
    assert path_billing == tmp_path / "tests" / "billing" / "test_views.py"
    assert path_auth != path_billing


def test_resolve_test_path_custom_output(tmp_path: Path) -> None:
    src = tmp_path / "app.py"
    src.touch()
    custom = tmp_path / "custom_tests" / "my_test.py"
    resolved = resolve_test_path(src, tmp_path, custom_output=custom)
    assert resolved == custom.resolve()


@pytest.mark.asyncio
async def test_pipeline_generates_and_passes(tmp_path: Path) -> None:
    src = tmp_path / "maths.py"
    src.write_text("def add(a, b): return a + b\n", encoding="utf-8")

    generated_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: maths.py\n"
        "import pytest\n"
        "from maths import add\n"
        "def test_add():\n"
        "    assert add(1, 2) == 3\n"
    )

    provider = MockProvider([generated_code])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    async def fake_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(
            test_file=test_file,
            passed=True,
            return_code=0,
            stdout="1 passed",
            stderr="",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=fake_runner)
    result = await pipeline.run(src)

    assert result.passed is True
    assert result.status == PipelineStatus.PASSED
    assert result.attempts == 0
    assert result.test_file.is_file()
    assert result.test_file.read_text(encoding="utf-8") == generated_code


@pytest.mark.asyncio
async def test_pipeline_passing_test_never_healed(tmp_path: Path) -> None:
    """Regression test: return_code == 0 is never misclassified or sent for healing."""
    src = tmp_path / "service.py"
    src.write_text("def run(): pass\n", encoding="utf-8")

    test_path = tmp_path / "tests" / "test_service.py"
    test_path.parent.mkdir()
    valid_test = f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: service.py\ndef test_run(): pass\n"
    test_path.write_text(valid_test, encoding="utf-8")

    provider = MockProvider()
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    # Runner returns return_code=0, but stdout contains scary strings
    async def passing_with_error_strings(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(
            test_file=test_file,
            passed=True,
            return_code=0,
            stdout="Warning: Exception in log, but test completed cleanly\n1 passed",
            stderr="AttributeError mentioned in debug output",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=passing_with_error_strings)
    result = await pipeline.run(src)

    assert result.passed is True
    assert result.status == PipelineStatus.PASSED
    assert result.attempts == 0
    # Provider should never have been invoked because test already existed and passed
    assert len(provider.calls) == 0


@pytest.mark.asyncio
async def test_pipeline_self_healing_success(tmp_path: Path) -> None:
    src = tmp_path / "calc.py"
    src.write_text("def multiply(a, b): return a * b\n", encoding="utf-8")

    first_broken_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: calc.py\n"
        "from calc import multiply\n"
        "def test_calc():\n"
        "    assert multip(2, 3) == 6\n"
    )
    healed_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:01:00 | Source: calc.py\n"
        "from calc import multiply\n"
        "def test_calc():\n"
        "    assert multiply(2, 3) == 6\n"
    )

    provider = MockProvider([first_broken_code, healed_code])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    run_counter = 0

    async def flaky_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        nonlocal run_counter
        run_counter += 1
        if run_counter == 1:
            return TestRunResult(
                test_file=test_file,
                passed=False,
                return_code=1,
                stdout="NameError: name 'multip' is not defined",
                stderr="",
                classification=ErrorClassification.RUNTIME,
                exception_type="NameError",
            )
        return TestRunResult(
            test_file=test_file,
            passed=True,
            return_code=0,
            stdout="1 passed",
            stderr="",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=flaky_runner)
    result = await pipeline.run(src)

    assert result.passed is True
    assert result.status == PipelineStatus.HEALED
    assert result.attempts == 1
    assert run_counter == 2
    assert result.test_file.read_text(encoding="utf-8") == healed_code


@pytest.mark.asyncio
async def test_pipeline_max_heal_attempts_exceeded(tmp_path: Path) -> None:
    src = tmp_path / "broken.py"
    src.write_text("x = 1\n", encoding="utf-8")

    test_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: broken.py\n"
        "def test_broken(): assert False\n"
    )

    provider = MockProvider([test_code, test_code, test_code])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    runs = 0

    async def always_fail_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        nonlocal runs
        runs += 1
        return TestRunResult(
            test_file=test_file,
            passed=False,
            return_code=1,
            stdout="",
            stderr="AttributeError: bad attr",
            classification=ErrorClassification.RUNTIME,
            exception_type="AttributeError",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=always_fail_runner)
    result = await pipeline.run(src, max_heal_attempts=2)

    assert result.passed is False
    assert result.status == PipelineStatus.FAILED
    assert result.attempts == 2
    assert runs == 3  # initial run (0) + 2 healed runs (1, 2)


@pytest.mark.asyncio
async def test_pipeline_auto_heal_false_runs_first_test(tmp_path: Path) -> None:
    """Regression test: auto_heal=False still executes the first test run unconditionally."""
    src = tmp_path / "module.py"
    src.write_text("val = 42\n", encoding="utf-8")

    test_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: module.py\n"
        "def test_module(): assert False\n"
    )

    provider = MockProvider([test_code])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    runs = 0

    async def fail_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        nonlocal runs
        runs += 1
        return TestRunResult(
            test_file=test_file,
            passed=False,
            return_code=1,
            stdout="",
            stderr="ImportError: missing",
            classification=ErrorClassification.SYNTAX,
            exception_type="ImportError",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=fail_runner)
    result = await pipeline.run(src, auto_heal=False)

    assert runs == 1  # The first run MUST have happened
    assert result.passed is False
    assert result.status == PipelineStatus.FAILED
    assert result.attempts == 0
    assert "auto_heal is disabled" in (result.error_message or "")


@pytest.mark.asyncio
async def test_pipeline_logic_error_use_judge_false(tmp_path: Path) -> None:
    src = tmp_path / "logic.py"
    src.write_text("def check(): return True\n", encoding="utf-8")

    test_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: logic.py\ndef test_logic(): assert 1 == 2\n"
    )

    provider = MockProvider([test_code])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    async def logic_fail_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(
            test_file=test_file,
            passed=False,
            return_code=1,
            stdout="AssertionError: assert 1 == 2",
            stderr="",
            classification=ErrorClassification.LOGIC,
            exception_type="AssertionError",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=logic_fail_runner)
    result = await pipeline.run(src, use_judge=False)

    assert result.passed is False
    assert result.status == PipelineStatus.FAILED
    assert result.attempts == 0
    assert "use_judge is disabled" in (result.error_message or "")


@pytest.mark.asyncio
async def test_pipeline_logic_error_judge_bug_in_code(tmp_path: Path) -> None:
    """Regression test: Judge BUG_IN_CODE halts immediately and never touches the test."""
    src = tmp_path / "buggy.py"
    src.write_text("def get_val(): return 0\n", encoding="utf-8")

    test_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: buggy.py\n"
        "from buggy import get_val\n"
        "def test_val(): assert get_val() == 1\n"
    )

    # First call: generate test. Second call: judge -> BUG_IN_CODE
    provider = MockProvider([test_code, "BUG_IN_CODE"])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    async def logic_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(
            test_file=test_file,
            passed=False,
            return_code=1,
            stdout="AssertionError: assert 0 == 1",
            stderr="",
            classification=ErrorClassification.LOGIC,
            exception_type="AssertionError",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=logic_runner)
    result = await pipeline.run(src, use_judge=True)

    assert result.passed is False
    assert result.status == PipelineStatus.BUG_IN_CODE
    assert result.judge_outcome == JudgeOutcome.BUG_IN_CODE
    assert result.attempts == 0
    # Invariant: test file was NOT modified by healing
    assert result.test_file.read_text(encoding="utf-8") == test_code


@pytest.mark.asyncio
async def test_pipeline_logic_error_judge_unclear(tmp_path: Path) -> None:
    """Regression test: Judge UNCLEAR halts immediately and never touches the test."""
    src = tmp_path / "complex.py"
    src.write_text("def func(): return 'foo'\n", encoding="utf-8")

    test_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: complex.py\n"
        "def test_complex(): assert False\n"
    )

    # Judge returns unclear text
    provider = MockProvider([test_code, "I am unsure whether code or test is wrong."])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    async def logic_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(
            test_file=test_file,
            passed=False,
            return_code=1,
            stdout="AssertionError: assert False",
            stderr="",
            classification=ErrorClassification.LOGIC,
            exception_type="AssertionError",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=logic_runner)
    result = await pipeline.run(src, use_judge=True)

    assert result.passed is False
    assert result.status == PipelineStatus.UNCLEAR
    assert result.judge_outcome == JudgeOutcome.UNCLEAR
    assert result.attempts == 0
    assert result.test_file.read_text(encoding="utf-8") == test_code


@pytest.mark.asyncio
async def test_pipeline_logic_error_judge_fix_test(tmp_path: Path) -> None:
    src = tmp_path / "fixable.py"
    src.write_text("def get_answer(): return 42\n", encoding="utf-8")

    initial_test = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: fixable.py\n"
        "from fixable import get_answer\n"
        "def test_answer(): assert get_answer() == 99\n"
    )
    healed_test = (
        f"{HEADER_PREFIX} 24-09-2026 12:01:00 | Source: fixable.py\n"
        "from fixable import get_answer\n"
        "def test_answer(): assert get_answer() == 42\n"
    )

    # 1. generate, 2. judge FIX_TEST, 3. heal_test
    provider = MockProvider([initial_test, "FIX_TEST", healed_test])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    call_num = 0

    async def runner_fn(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        nonlocal call_num
        call_num += 1
        if call_num == 1:
            return TestRunResult(
                test_file=test_file,
                passed=False,
                return_code=1,
                stdout="AssertionError: assert 42 == 99",
                stderr="",
                classification=ErrorClassification.LOGIC,
                exception_type="AssertionError",
            )
        return TestRunResult(
            test_file=test_file,
            passed=True,
            return_code=0,
            stdout="1 passed",
            stderr="",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=runner_fn)
    result = await pipeline.run(src, use_judge=True)

    assert result.passed is True
    assert result.status == PipelineStatus.HEALED
    assert result.attempts == 1
    assert result.test_file.read_text(encoding="utf-8") == healed_test


@pytest.mark.asyncio
async def test_pipeline_refuses_to_overwrite_handwritten_test(tmp_path: Path) -> None:
    """Regression test: Never overwrites a hand-written test file unless force=True."""
    src = tmp_path / "app.py"
    src.write_text("x = 10\n", encoding="utf-8")

    test_file = tmp_path / "tests" / "test_app.py"
    test_file.parent.mkdir()
    test_file.write_text("# Human written test without Ghost header\ndef test_app(): assert True\n")

    config = GhostConfig()
    pipeline = TestPipeline(config, tmp_path, client=LLMClient(MockProvider(), config))

    with pytest.raises(HandwrittenTestOverwriteError):
        await pipeline.run(src, force=False)

    # With force=True: permitted
    async def pass_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(test_file=test_file, passed=True, return_code=0, stdout="", stderr="")

    result = await pipeline.run(src, force=True)
    assert result.passed is True


@pytest.mark.asyncio
async def test_pipeline_source_not_found(tmp_path: Path) -> None:
    config = GhostConfig()
    pipeline = TestPipeline(config, tmp_path, client=LLMClient(MockProvider(), config))

    with pytest.raises(SourceFileNotFoundError):
        await pipeline.run(tmp_path / "nonexistent.py")


@pytest.mark.asyncio
async def test_pipeline_non_python_source(tmp_path: Path) -> None:
    non_py = tmp_path / "readme.txt"
    non_py.write_text("Hello\n", encoding="utf-8")

    config = GhostConfig()
    pipeline = TestPipeline(config, tmp_path, client=LLMClient(MockProvider(), config))

    with pytest.raises(UnsupportedFileError):
        await pipeline.run(non_py)


@pytest.mark.asyncio
async def test_pipeline_listener_records_lifecycle(tmp_path: Path) -> None:
    src = tmp_path / "logged.py"
    src.write_text("x = 1\n", encoding="utf-8")

    test_code = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: logged.py\ndef test_logged(): assert True\n"
    )
    provider = MockProvider([test_code])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    async def pass_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        return TestRunResult(test_file=test_file, passed=True, return_code=0, stdout="", stderr="")

    listener = RecordingListener()
    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=pass_runner)
    result = await pipeline.run(src, listener=listener)

    assert result.passed is True
    event_names = [ev[0] for ev in listener.events]
    assert PipelineEvent.GENERATING in event_names
    assert PipelineEvent.GENERATED in event_names
    assert PipelineEvent.RUNNING in event_names
    assert PipelineEvent.PASSED in event_names


@pytest.mark.asyncio
async def test_pipeline_timeout_healing(tmp_path: Path) -> None:
    src = tmp_path / "loop.py"
    src.write_text("def run(): pass\n", encoding="utf-8")

    bad_test = (
        f"{HEADER_PREFIX} 24-09-2026 12:00:00 | Source: loop.py\n"
        "def test_loop():\n"
        "    while True: pass\n"
    )
    healed_test = (
        f"{HEADER_PREFIX} 24-09-2026 12:01:00 | Source: loop.py\ndef test_loop(): assert True\n"
    )

    provider = MockProvider([bad_test, healed_test])
    config = GhostConfig()
    client = LLMClient(provider=provider, config=config)

    call_count = 0

    async def timeout_then_pass_runner(
        test_file: Path, project_root: Path, timeout_seconds: float = 30.0
    ) -> TestRunResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return TestRunResult(
                test_file=test_file,
                passed=False,
                return_code=-9,
                stdout="",
                stderr="",
                timed_out=True,
                classification=ErrorClassification.TIMEOUT,
                exception_type="TimeoutError",
            )
        return TestRunResult(
            test_file=test_file,
            passed=True,
            return_code=0,
            stdout="1 passed",
            stderr="",
        )

    pipeline = TestPipeline(config, tmp_path, client=client, runner_fn=timeout_then_pass_runner)
    result = await pipeline.run(src)

    assert result.passed is True
    assert result.status == PipelineStatus.HEALED
    assert result.attempts == 1
    assert result.test_file.read_text(encoding="utf-8") == healed_test
