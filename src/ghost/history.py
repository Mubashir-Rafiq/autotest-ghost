"""Ghost's test healing history, snapshot rollback, and token usage tracking.

Owns:
- Archiving each test generation and healing attempt to ``.ghost/history/``.
- Tracking metadata (attempt index, timestamp, outcome, classification, diff).
- Recording cumulative LLM token usage and request counts in ``.ghost/usage.json``.
- Providing history inspection and usage reporting APIs for the CLI.

Does NOT:
- Execute subprocess tests (owned by ``runner.py``).
- Coordinate the self-healing state machine (owned by ``pipeline.py``).
- Communicate directly with AI providers (owned by ``providers.py``).

Guarantees:
- Snapshots are isolated per relative source file path, preventing collisions.
- History recording never raises unhandled exceptions to crash the pipeline (fail-soft).
- Usage metrics are persisted atomically with temp-file replacement.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AttemptRecord",
    "HistoryTracker",
    "UsageRecord",
    "UsageTracker",
    "get_history_tracker",
    "get_usage_tracker",
]

_HISTORY_DIR_NAME: Final[str] = ".ghost/history"
_USAGE_FILE_NAME: Final[str] = ".ghost/usage.json"


class AttemptRecord(BaseModel):
    """Metadata recorded for a single test generation or heal attempt."""

    model_config = ConfigDict(frozen=True)

    attempt: int
    timestamp: str
    status: str
    classification: str | None = None
    error_message: str | None = None
    snapshot_file: str


class UsageRecord(BaseModel):
    """Cumulative usage statistics for AI test generation."""

    model_config = ConfigDict(frozen=True)

    total_requests: int = Field(default=0)
    prompt_tokens: int = Field(default=0)
    completion_tokens: int = Field(default=0)
    total_tokens: int = Field(default=0)
    heal_attempts: int = Field(default=0)
    healed_tests: int = Field(default=0)


class HistoryTracker:
    """Manages snapshot archives and history records in .ghost/history."""

    def __init__(self, project_root: Path) -> None:
        self.project_root: Final[Path] = project_root.resolve()
        self.history_dir: Final[Path] = self.project_root / _HISTORY_DIR_NAME
        self._lock: Final[threading.Lock] = threading.Lock()

    def _relative_key(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return resolved.name

    def record_attempt(
        self,
        source_path: Path,
        attempt: int,
        test_code: str,
        status: str,
        *,
        classification: str | None = None,
        error_message: str | None = None,
    ) -> Path:
        """Snapshot a generated test attempt and record its metadata."""
        rel_key = self._relative_key(source_path)
        file_history_dir = self.history_dir / rel_key
        file_history_dir.mkdir(parents=True, exist_ok=True)

        snapshot_name = f"attempt_{attempt}.py"
        snapshot_path = file_history_dir / snapshot_name
        snapshot_path.write_text(test_code, encoding="utf-8")

        meta_file = file_history_dir / "meta.json"
        now_iso = datetime.now(UTC).isoformat()

        record = AttemptRecord(
            attempt=attempt,
            timestamp=now_iso,
            status=status,
            classification=classification,
            error_message=error_message,
            snapshot_file=snapshot_name,
        )

        with self._lock:
            entries: list[dict[str, Any]] = []
            if meta_file.is_file():
                try:
                    raw = json.loads(meta_file.read_text(encoding="utf-8"))
                    if isinstance(raw, list):
                        entries = raw
                except Exception:
                    entries = []

            entries.append(record.model_dump())

            # Atomic write
            with tempfile.NamedTemporaryFile(
                "w",
                dir=file_history_dir,
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(entries, indent=2))
                temp_path = Path(f.name)
            temp_path.replace(meta_file)

        return snapshot_path

    async def arecord_attempt(
        self,
        source_path: Path,
        attempt: int,
        test_code: str,
        status: str,
        *,
        classification: str | None = None,
        error_message: str | None = None,
    ) -> Path:
        """Asynchronously record an attempt snapshot."""
        return await asyncio.to_thread(
            self.record_attempt,
            source_path,
            attempt,
            test_code,
            status,
            classification=classification,
            error_message=error_message,
        )

    def get_history(self, source_path: Path) -> list[AttemptRecord]:
        """Return all attempt records for a specific source file."""
        rel_key = self._relative_key(source_path)
        meta_file = self.history_dir / rel_key / "meta.json"
        if not meta_file.is_file():
            return []

        try:
            raw = json.loads(meta_file.read_text(encoding="utf-8"))
            if not isinstance(raw, list):
                return []
            return [AttemptRecord.model_validate(item) for item in raw]
        except Exception:
            return []

    def list_tracked_files(self) -> list[str]:
        """Return relative paths of all source files that have recorded history."""
        if not self.history_dir.is_dir():
            return []
        tracked: list[str] = []
        for meta in self.history_dir.rglob("meta.json"):
            rel = meta.parent.relative_to(self.history_dir).as_posix()
            tracked.append(rel)
        tracked.sort()
        return tracked

    def rollback(
        self,
        source_path: Path,
        target_test_path: Path,
        *,
        attempt: int | None = None,
    ) -> Path:
        """Roll back target_test_path to a previous attempt snapshot.

        If *attempt* is None, the earliest recorded attempt (attempt 0) is restored.
        Raises FileNotFoundError if no history or snapshot exists.
        """
        records = self.get_history(source_path)
        if not records:
            msg = f"No history records found for {source_path}"
            raise FileNotFoundError(msg)

        if attempt is not None:
            matching = [r for r in records if r.attempt == attempt]
            if not matching:
                msg = f"No attempt #{attempt} found in history for {source_path}"
                raise FileNotFoundError(msg)
            target_record = matching[0]
        else:
            target_record = records[0]

        rel_key = self._relative_key(source_path)
        snapshot_file = self.history_dir / rel_key / target_record.snapshot_file
        if not snapshot_file.is_file():
            msg = f"Snapshot file {snapshot_file} does not exist."
            raise FileNotFoundError(msg)

        code = snapshot_file.read_text(encoding="utf-8")
        target_test_path.parent.mkdir(parents=True, exist_ok=True)
        target_test_path.write_text(code, encoding="utf-8")
        return target_test_path

    async def arollback(
        self,
        source_path: Path,
        target_test_path: Path,
        *,
        attempt: int | None = None,
    ) -> Path:
        """Asynchronously roll back target_test_path to a previous attempt snapshot."""
        return await asyncio.to_thread(
            self.rollback,
            source_path,
            target_test_path,
            attempt=attempt,
        )


class UsageTracker:
    """Tracks token metrics and generation counts atomically in .ghost/usage.json."""

    def __init__(self, project_root: Path) -> None:
        self.project_root: Final[Path] = project_root.resolve()
        self.usage_file: Final[Path] = self.project_root / _USAGE_FILE_NAME
        self._lock: Final[threading.Lock] = threading.Lock()

    def load_usage(self) -> UsageRecord:
        """Load current cumulative usage stats, returning zeroed stats on failure."""
        if not self.usage_file.is_file():
            return UsageRecord()
        with contextlib.suppress(Exception):
            raw = json.loads(self.usage_file.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return UsageRecord.model_validate(raw)
        return UsageRecord()

    def record_call(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        is_heal: bool = False,
        healed_success: bool = False,
    ) -> UsageRecord:
        """Record an LLM call and update cumulative counts."""
        with self._lock:
            current = self.load_usage()
            new_total_reqs = current.total_requests + 1
            new_prompt_tok = current.prompt_tokens + max(0, prompt_tokens)
            new_comp_tok = current.completion_tokens + max(0, completion_tokens)
            new_tot_tok = current.total_tokens + max(0, prompt_tokens + completion_tokens)
            new_heal_att = current.heal_attempts + (1 if is_heal else 0)
            new_healed = current.healed_tests + (1 if healed_success else 0)

            updated = UsageRecord(
                total_requests=new_total_reqs,
                prompt_tokens=new_prompt_tok,
                completion_tokens=new_comp_tok,
                total_tokens=new_tot_tok,
                heal_attempts=new_heal_att,
                healed_tests=new_healed,
            )

            self.usage_file.parent.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(Exception):
                with tempfile.NamedTemporaryFile(
                    "w",
                    dir=self.usage_file.parent,
                    delete=False,
                    encoding="utf-8",
                ) as f:
                    f.write(json.dumps(updated.model_dump(), indent=2))
                    temp_path = Path(f.name)
                temp_path.replace(self.usage_file)

        return updated

    async def arecord_call(
        self,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        is_heal: bool = False,
        healed_success: bool = False,
    ) -> UsageRecord:
        """Asynchronously record an LLM call."""
        return await asyncio.to_thread(
            self.record_call,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            is_heal=is_heal,
            healed_success=healed_success,
        )

    def record_success(self) -> UsageRecord:
        """Record that a test was successfully healed."""
        current = self.load_usage()
        updated = UsageRecord(
            total_requests=current.total_requests,
            prompt_tokens=current.prompt_tokens,
            completion_tokens=current.completion_tokens,
            total_tokens=current.total_tokens,
            heal_attempts=current.heal_attempts,
            healed_tests=current.healed_tests + 1,
        )
        self.usage_file.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(Exception):
            with tempfile.NamedTemporaryFile(
                "w",
                dir=self.usage_file.parent,
                delete=False,
                encoding="utf-8",
            ) as f:
                f.write(json.dumps(updated.model_dump(), indent=2))
                temp_path = Path(f.name)
            temp_path.replace(self.usage_file)
        return updated

    async def arecord_success(self) -> UsageRecord:
        """Asynchronously record that a test was successfully healed."""
        return await asyncio.to_thread(self.record_success)


def get_history_tracker(project_root: Path) -> HistoryTracker:
    """Factory creating a :class:`HistoryTracker` instance for a project."""
    return HistoryTracker(project_root)


def get_usage_tracker(project_root: Path) -> UsageTracker:
    """Factory creating a :class:`UsageTracker` instance for a project."""
    return UsageTracker(project_root)
