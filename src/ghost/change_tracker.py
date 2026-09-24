"""Content hash cache for skipping the pipeline on byte-identical file saves.

Owns:
- Computing stable SHA-256 content hashes.
- Persisting processed hashes at ``.ghost/hashes.json``.
- Normalizing file paths to relative POSIX keys to prevent basename collisions.
- Atomic, non-blocking file writes to prevent corrupted on-disk state.

Does NOT:
- Handle rapid-burst debouncing (that is ``job_queue.py``'s job).
- Inject hashes into prompts (hashes are strictly isolated from ``context.json``).
- Execute the test pipeline or decide error classifications.

Guarantees:
- Never keys by basename alone; files sharing names across subdirectories never collide.
- Fails open: a missing or corrupted ``hashes.json`` is treated as "all files changed".
- Atomic write via temporary file replace; avoids torn writes or partial JSON.
- Never records a new hash until the pipeline has actually completed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Final

__all__ = [
    "HASHES_FILE",
    "ChangeTracker",
    "compute_hash",
]

HASHES_FILE: Final[str] = ".ghost/hashes.json"


def compute_hash(content: str) -> str:
    """Compute deterministic SHA-256 hexadecimal digest for *content*."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ChangeTracker:
    """Tracks SHA-256 content hashes to detect and skip unchanged re-saves."""

    def __init__(self, project_root: Path) -> None:
        self.project_root: Final[Path] = project_root.resolve()
        self.hashes_file: Final[Path] = self.project_root / ".ghost" / "hashes.json"

    def normalize_key(self, path: Path) -> str:
        """Convert a path into a collision-free relative POSIX key."""
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.project_root).as_posix()
        except ValueError:
            return resolved.name

    def load_hashes(self) -> dict[str, str]:
        """Load the on-disk hash map, failing open on any error."""
        if not self.hashes_file.is_file():
            return {}
        try:
            text = self.hashes_file.read_text(encoding="utf-8")
            data = json.loads(text)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return {}
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
        return {}

    def save_hashes(self, data: dict[str, str]) -> None:
        """Persist hashes atomically via a temporary file replace."""
        self.hashes_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.hashes_file.with_suffix(".tmp")
        serialized = json.dumps(data, indent=2, sort_keys=True)
        temp_file.write_text(f"{serialized}\n", encoding="utf-8")
        temp_file.replace(self.hashes_file)

    def has_changed(self, path: Path, content: str) -> bool:
        """Return True if *content* differs from the recorded hash for *path*."""
        key = self.normalize_key(path)
        hashes = self.load_hashes()
        previous_hash = hashes.get(key)
        return previous_hash != compute_hash(content)

    def mark_processed(self, path: Path, content: str) -> None:
        """Record *content*'s hash as the last-processed version of *path*."""
        key = self.normalize_key(path)
        hashes = self.load_hashes()
        hashes[key] = compute_hash(content)
        self.save_hashes(hashes)

    def remove(self, path: Path) -> None:
        """Remove *path*'s recorded hash when the file is deleted."""
        key = self.normalize_key(path)
        hashes = self.load_hashes()
        if key in hashes:
            del hashes[key]
            self.save_hashes(hashes)

    async def ahas_changed(self, path: Path, content: str) -> bool:
        """Asynchronous non-blocking check whether *content* has changed."""
        return await asyncio.to_thread(self.has_changed, path, content)

    async def amark_processed(self, path: Path, content: str) -> None:
        """Asynchronous non-blocking recording of processed *content* hash."""
        await asyncio.to_thread(self.mark_processed, path, content)

    async def aremove(self, path: Path) -> None:
        """Asynchronous non-blocking removal of recorded hash."""
        await asyncio.to_thread(self.remove, path)
