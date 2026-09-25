"""The core test generation, execution, classification, healing, and judge pipeline.

Owns:
- The single unified state machine for generate -> run -> classify -> heal -> judge.
- Mapping source files to test target paths preserving subdirectory uniqueness.
- Safeguarding hand-written test files against accidental overwrite.
- Reporting decoupled lifecycle progress via pipeline listeners.

Does NOT:
- Implement direct subprocess invocation or process groups (runner.py owns this).
- Directly invoke LLM APIs or format raw prompt strings (client.py and prompts.py own this).
- Watch filesystem events or manage async job queues (watcher.py / job_queue.py own this).
- Render rich terminal animations (console.py / cli.py own this).

Guarantees:
- Single implementation of the state machine shared by CLI, foreground watcher, and daemon.
- The initial test run always executes, even when auto_heal is disabled.
- Passing tests (return_code == 0) are never sent to error classification or healing.
- Hand-written tests are never overwritten without explicit force=True.
- UNCLEAR judge outcomes always halt execution without modifying the test.
- BUG_IN_CODE judge outcomes halt immediately and never modify the test.
- Max healing attempts are strictly respected and shared across healing paths.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ghost.change_tracker import ChangeTracker
from ghost.client import (
    JudgeOutcome,
    LLMClient,
    ensure_can_overwrite_test,
)
from ghost.errors import (
    SourceFileNotFoundError,
    UnsupportedFileError,
)
from ghost.history import HistoryTracker, UsageTracker
from ghost.providers import get_provider
from ghost.runner import ErrorClassification, TestRunResult, run_test

if TYPE_CHECKING:
    from ghost.config import GhostConfig

__all__ = [
    "PipelineEvent",
    "PipelineListener",
    "PipelineResult",
    "PipelineStatus",
    "TestPipeline",
    "resolve_test_path",
]


class PipelineStatus(StrEnum):
    """Overall status outcome of running the test pipeline."""

    PASSED = "PASSED"
    HEALED = "HEALED"
    FAILED = "FAILED"
    BUG_IN_CODE = "BUG_IN_CODE"
    UNCLEAR = "UNCLEAR"
    SKIPPED = "SKIPPED"
    ABORTED = "ABORTED"


class PipelineEvent(StrEnum):
    """Events emitted during the pipeline lifecycle."""

    GENERATING = "generating"
    GENERATED = "generated"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    HEALING = "healing"
    HEALED = "healed"
    JUDGING = "judging"
    JUDGE_RESULT = "judge_result"
    GIVE_UP = "give_up"
    SKIPPED = "skipped"


class PipelineListener:
    """Base listener for pipeline lifecycle events. Subclass to observe."""

    def on_event(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        """Receive a pipeline lifecycle event."""


@dataclass(frozen=True)
class PipelineResult:
    """The structured result returned after a pipeline execution."""

    source_file: Path
    test_file: Path
    status: PipelineStatus
    passed: bool
    attempts: int
    last_run: TestRunResult | None = None
    judge_outcome: JudgeOutcome | None = None
    error_message: str | None = None


def resolve_test_path(
    source_path: Path,
    project_root: Path,
    output_dir: str | Path = "tests",
    custom_output: Path | None = None,
) -> Path:
    """Resolve the test destination path for a given source file.

    Preserves package hierarchy relative to *project_root* to prevent
    basename collisions between identically named files in different subdirectories.
    If *source_path* begins with a top-level ``src/`` directory, that component is
    stripped so ``src/pkg/mod.py`` maps to ``tests/pkg/test_mod.py`` instead of
    ``tests/src/pkg/test_mod.py``.
    """
    if custom_output is not None:
        return custom_output.resolve()

    resolved_src = source_path.resolve()
    resolved_root = project_root.resolve()

    try:
        rel = resolved_src.relative_to(resolved_root)
    except ValueError:
        rel = Path(resolved_src.name)

    parts = list(rel.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]

    if len(parts) > 1:
        parent_dir = Path(*parts[:-1])
        filename = f"test_{parts[-1]}"
        return resolved_root / output_dir / parent_dir / filename

    filename = f"test_{resolved_src.name}"
    return resolved_root / output_dir / filename


def _validate_source_file(source_path: Path) -> Path:
    resolved = source_path.resolve()
    if not resolved.is_file():
        raise SourceFileNotFoundError(resolved)
    if resolved.suffix != ".py":
        raise UnsupportedFileError(resolved)
    return resolved


def _emit(listener: PipelineListener | None, event: PipelineEvent, data: dict[str, Any]) -> None:
    if listener is not None:
        listener.on_event(event, data)


RunnerFn = Callable[..., Coroutine[Any, Any, TestRunResult]]


class TestPipeline:
    """The single shared state machine for generating, running, healing, and judging tests."""

    __test__ = False

    def __init__(
        self,
        config: GhostConfig,
        project_root: Path,
        *,
        client: LLMClient | None = None,
        runner_fn: RunnerFn | None = None,
        tracker: ChangeTracker | None = None,
        history: HistoryTracker | None = None,
        usage: UsageTracker | None = None,
    ) -> None:
        self.config = config
        self.project_root = project_root
        if client is not None:
            self.client = client
        else:
            provider = get_provider(config.ai.provider, config=config)
            self.client = LLMClient(provider=provider, config=config)
        self._runner_fn: RunnerFn = runner_fn or run_test
        self.tracker = tracker or ChangeTracker(project_root)
        self.history = history or HistoryTracker(project_root)
        self.usage = usage or UsageTracker(project_root)

    async def _ensure_test_file(
        self,
        source_path: Path,
        test_path: Path,
        *,
        force: bool,
        force_generate: bool,
        listener: PipelineListener | None,
    ) -> None:
        is_existing = await asyncio.to_thread(test_path.is_file)
        if is_existing and not force_generate:
            await asyncio.to_thread(ensure_can_overwrite_test, test_path, force=force)
            if not self.history.get_history(source_path):
                existing_code = await asyncio.to_thread(
                    test_path.read_text, encoding="utf-8", errors="replace"
                )
                await self.history.arecord_attempt(
                    source_path=source_path,
                    attempt=0,
                    test_code=existing_code,
                    status="existing",
                )
            return

        await asyncio.to_thread(ensure_can_overwrite_test, test_path, force=force)
        _emit(
            listener,
            PipelineEvent.GENERATING,
            {"source_file": source_path, "test_file": test_path},
        )

        test_code = await self.client.generate_test(source_path, self.project_root)

        await asyncio.to_thread(test_path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(test_path.write_text, f"{test_code.rstrip()}\n", encoding="utf-8")

        await self.history.arecord_attempt(
            source_path=source_path,
            attempt=0,
            test_code=test_code,
            status="generated",
        )
        src_text = await asyncio.to_thread(
            source_path.read_text, encoding="utf-8", errors="replace"
        )
        await self.usage.arecord_call(
            prompt_tokens=len(src_text) // 4,
            completion_tokens=len(test_code) // 4,
            is_heal=False,
        )

        _emit(
            listener,
            PipelineEvent.GENERATED,
            {"source_file": source_path, "test_file": test_path},
        )

    async def _handle_logic_failure(
        self,
        source_path: Path,
        test_path: Path,
        last_run: TestRunResult,
        attempt: int,
        *,
        use_judge: bool,
        listener: PipelineListener | None,
    ) -> tuple[PipelineStatus | None, JudgeOutcome | None, str | None]:
        if not use_judge:
            _emit(
                listener,
                PipelineEvent.GIVE_UP,
                {"reason": "assertion failed and use_judge is disabled"},
            )
            return (
                PipelineStatus.FAILED,
                None,
                "Assertion failed and use_judge is disabled",
            )

        _emit(
            listener,
            PipelineEvent.JUDGING,
            {"test_file": test_path, "attempt": attempt},
        )

        err_combined = f"{last_run.stdout}\n{last_run.stderr}"
        judge_outcome = await self.client.consult_the_judge(
            source_path=source_path,
            test_file=test_path,
            error_output=err_combined,
        )

        _emit(
            listener,
            PipelineEvent.JUDGE_RESULT,
            {"outcome": judge_outcome},
        )

        if judge_outcome == JudgeOutcome.BUG_IN_CODE:
            return (
                PipelineStatus.BUG_IN_CODE,
                JudgeOutcome.BUG_IN_CODE,
                "Judge determined defect is in source code; test was not modified",
            )

        if judge_outcome == JudgeOutcome.UNCLEAR:
            return (
                PipelineStatus.UNCLEAR,
                JudgeOutcome.UNCLEAR,
                "Judge outcome was UNCLEAR; test was not modified",
            )

        return (None, JudgeOutcome.FIX_TEST, None)

    def _check_heal_budget(
        self,
        attempt: int,
        max_attempts: int,
        *,
        auto_heal: bool,
        listener: PipelineListener | None,
    ) -> tuple[bool, str | None]:
        if not auto_heal:
            _emit(
                listener,
                PipelineEvent.GIVE_UP,
                {"reason": "auto_heal is disabled"},
            )
            return (False, "Test failed and auto_heal is disabled")

        if attempt >= max_attempts:
            _emit(
                listener,
                PipelineEvent.GIVE_UP,
                {"reason": f"maximum heal attempts reached ({max_attempts})"},
            )
            return (
                False,
                f"Test failed after reaching maximum healing attempts ({max_attempts})",
            )

        return (True, None)

    async def _heal_and_save(
        self,
        source_path: Path,
        test_path: Path,
        last_run: TestRunResult,
        attempt: int,
        *,
        force: bool,
        classification: ErrorClassification,
        listener: PipelineListener | None,
    ) -> tuple[bool, str | None]:
        _emit(
            listener,
            PipelineEvent.HEALING,
            {"attempt": attempt, "classification": classification},
        )

        err_combined = f"{last_run.stdout}\n{last_run.stderr}"
        try:
            healed_code = await self.client.heal_test(
                source_path=source_path,
                test_file=test_path,
                error_output=err_combined,
                project_root=self.project_root,
                return_code=last_run.return_code,
            )
        except Exception as err:
            _emit(
                listener,
                PipelineEvent.GIVE_UP,
                {"reason": f"healing failed: {err}"},
            )
            return (False, f"Healing failed: {err}")

        await asyncio.to_thread(ensure_can_overwrite_test, test_path, force=force)
        await asyncio.to_thread(test_path.write_text, f"{healed_code.rstrip()}\n", encoding="utf-8")

        cls_val = classification.value if hasattr(classification, "value") else str(classification)
        await self.history.arecord_attempt(
            source_path=source_path,
            attempt=attempt,
            test_code=healed_code,
            status="healed",
            classification=cls_val,
        )
        await self.usage.arecord_call(
            prompt_tokens=len(err_combined) // 4,
            completion_tokens=len(healed_code) // 4,
            is_heal=True,
        )

        _emit(
            listener,
            PipelineEvent.HEALED,
            {"attempt": attempt, "test_file": test_path},
        )

        return (True, None)

    async def _finish(
        self,
        result: PipelineResult,
        source_path: Path,
        source_code: str,
    ) -> PipelineResult:
        if result.status not in (PipelineStatus.SKIPPED, PipelineStatus.ABORTED):
            await self.tracker.amark_processed(source_path, source_code)
        return result

    async def _check_if_changed(
        self,
        source_path: Path,
        source_code: str,
        test_path: Path,
        listener: PipelineListener | None,
    ) -> PipelineResult | None:
        has_changed = await self.tracker.ahas_changed(source_path, source_code)
        if not has_changed:
            _emit(
                listener,
                PipelineEvent.SKIPPED,
                {"source_file": source_path, "test_file": test_path},
            )
            return PipelineResult(
                source_file=source_path,
                test_file=test_path,
                status=PipelineStatus.SKIPPED,
                passed=True,
                attempts=0,
                error_message="Source file unchanged; pipeline skipped.",
            )
        return None

    async def _execute_runner(
        self,
        test_path: Path,
        resolved_root: Path,
        timeout_seconds: float,
        *,
        coverage: bool,
        cov_source: str | None,
    ) -> TestRunResult:
        try:
            return await self._runner_fn(
                test_path,
                resolved_root,
                timeout_seconds=timeout_seconds,
                coverage=coverage,
                cov_source=cov_source,
            )
        except TypeError:
            return await self._runner_fn(
                test_path,
                resolved_root,
                timeout_seconds=timeout_seconds,
            )

    async def run(
        self,
        source_file: Path,
        *,
        custom_output: Path | None = None,
        force: bool = False,
        force_generate: bool = False,
        if_changed: bool = False,
        auto_heal: bool | None = None,
        use_judge: bool | None = None,
        max_heal_attempts: int | None = None,
        timeout_seconds: float | None = None,
        coverage: bool = False,
        cov_source: str | None = None,
        listener: PipelineListener | None = None,
    ) -> PipelineResult:
        """Execute the generate -> run -> classify -> heal -> judge state machine."""
        resolved_src = _validate_source_file(source_file)
        resolved_root = self.project_root.resolve()

        test_path = resolve_test_path(
            resolved_src,
            resolved_root,
            output_dir=self.config.tests.output_dir,
            custom_output=custom_output,
        )

        source_code = await asyncio.to_thread(resolved_src.read_text, encoding="utf-8")

        if if_changed:
            skipped = await self._check_if_changed(resolved_src, source_code, test_path, listener)
            if skipped is not None:
                return skipped

        effective_auto_heal = self.config.tests.auto_heal if auto_heal is None else auto_heal
        effective_use_judge = self.config.tests.use_judge if use_judge is None else use_judge
        effective_max_attempts = (
            self.config.tests.max_heal_attempts if max_heal_attempts is None else max_heal_attempts
        )
        effective_timeout = timeout_seconds if timeout_seconds is not None else 30.0

        await self._ensure_test_file(
            resolved_src,
            test_path,
            force=force,
            force_generate=force_generate,
            listener=listener,
        )

        attempt = 0
        last_run: TestRunResult | None = None
        judge_outcome: JudgeOutcome | None = None

        while True:
            _emit(
                listener,
                PipelineEvent.RUNNING,
                {"attempt": attempt, "test_file": test_path},
            )

            last_run = await self._execute_runner(
                test_path,
                resolved_root,
                effective_timeout,
                coverage=coverage,
                cov_source=cov_source,
            )

            if last_run.return_code == 0:
                _emit(
                    listener,
                    PipelineEvent.PASSED,
                    {"attempt": attempt, "test_file": test_path, "run_result": last_run},
                )
                if attempt > 0:
                    await self.usage.arecord_success()
                status = PipelineStatus.HEALED if attempt > 0 else PipelineStatus.PASSED
                return await self._finish(
                    PipelineResult(
                        source_file=resolved_src,
                        test_file=test_path,
                        status=status,
                        passed=True,
                        attempts=attempt,
                        last_run=last_run,
                        judge_outcome=judge_outcome,
                    ),
                    resolved_src,
                    source_code,
                )

            _emit(
                listener,
                PipelineEvent.FAILED,
                {"attempt": attempt, "test_file": test_path, "run_result": last_run},
            )

            classification = last_run.classification or ErrorClassification.UNKNOWN

            if classification == ErrorClassification.LOGIC:
                term_status, j_outcome, err_msg = await self._handle_logic_failure(
                    resolved_src,
                    test_path,
                    last_run,
                    attempt,
                    use_judge=effective_use_judge,
                    listener=listener,
                )
                judge_outcome = j_outcome
                if term_status is not None:
                    return await self._finish(
                        PipelineResult(
                            source_file=resolved_src,
                            test_file=test_path,
                            status=term_status,
                            passed=False,
                            attempts=attempt,
                            last_run=last_run,
                            judge_outcome=judge_outcome,
                            error_message=err_msg,
                        ),
                        resolved_src,
                        source_code,
                    )

            can_heal, reason = self._check_heal_budget(
                attempt,
                effective_max_attempts,
                auto_heal=effective_auto_heal,
                listener=listener,
            )
            if not can_heal:
                return await self._finish(
                    PipelineResult(
                        source_file=resolved_src,
                        test_file=test_path,
                        status=PipelineStatus.FAILED,
                        passed=False,
                        attempts=attempt,
                        last_run=last_run,
                        judge_outcome=judge_outcome,
                        error_message=reason,
                    ),
                    resolved_src,
                    source_code,
                )

            attempt += 1
            heal_ok, heal_err = await self._heal_and_save(
                resolved_src,
                test_path,
                last_run,
                attempt,
                force=force,
                classification=classification,
                listener=listener,
            )
            if not heal_ok:
                return await self._finish(
                    PipelineResult(
                        source_file=resolved_src,
                        test_file=test_path,
                        status=PipelineStatus.FAILED,
                        passed=False,
                        attempts=attempt,
                        last_run=last_run,
                        judge_outcome=judge_outcome,
                        error_message=heal_err,
                    ),
                    resolved_src,
                    source_code,
                )
