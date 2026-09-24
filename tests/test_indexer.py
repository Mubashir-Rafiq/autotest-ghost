"""Tests for AST project indexing, context extraction, and budgeting."""

from __future__ import annotations

import json
from pathlib import Path

from ghost.config import ScannerConfig
from ghost.indexer import (
    budget_context,
    extract_imports,
    get_project_files,
    get_project_tree,
    index_file,
    index_source,
    should_ignore_path,
    walk_and_delete_json,
    walk_and_generate_json,
    walk_and_modify_json,
)


def test_index_source_functions_with_type_hints_and_defaults() -> None:
    source = '''"""Module docstring."""

def simple():
    pass

def typed_fn(
    a: int,
    b: str = "default",
    *,
    flag: bool = True,
    **kwargs: float,
) -> list[str]:
    """Summary of typed_fn."""
    return []

async def async_worker(task_id: str) -> None:
    """Async task worker."""
    pass
'''
    findex = index_source(source, file_path="sample.py")
    assert findex is not None
    assert findex.docstring == "Module docstring."
    assert len(findex.functions) == 3

    simple_fn = findex.functions[0]
    assert simple_fn.name == "simple"
    assert simple_fn.signature == "simple()"
    assert simple_fn.docstring is None
    assert not simple_fn.is_async

    typed_fn = findex.functions[1]
    assert typed_fn.name == "typed_fn"
    assert typed_fn.signature == (
        "typed_fn(a: int, b: str = 'default', *, flag: bool = True, **kwargs: float) -> list[str]"
    )
    assert typed_fn.docstring == "Summary of typed_fn."
    assert not typed_fn.is_async

    async_fn = findex.functions[2]
    assert async_fn.name == "async_worker"
    assert async_fn.signature == "async_worker(task_id: str) -> None"
    assert async_fn.is_async

    summary = findex.format_summary()
    assert "Functions:" in summary
    assert "typed_fn" in summary
    assert "Summary of typed_fn." in summary
    assert "Classes: None" in summary


def test_index_source_classes_methods_and_decorators() -> None:
    source = '''
class User(BaseModel, Serializable):
    """User account model."""

    def __init__(self, username: str, email: str | None = None) -> None:
        """Create a new user."""
        self.username = username

    @property
    def display_name(self) -> str:
        return self.username

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> User:
        """Construct user from dictionary."""
        return cls(data["username"])

    @staticmethod
    def validate_name(name: str) -> bool:
        return len(name) > 0

    async def fetch_avatar(self) -> bytes:
        return b""
'''
    findex = index_source(source, file_path="models.py")
    assert findex is not None
    assert len(findex.classes) == 1

    cls_info = findex.classes[0]
    assert cls_info.name == "User"
    assert cls_info.bases == ["BaseModel", "Serializable"]
    assert cls_info.docstring == "User account model."
    assert len(cls_info.methods) == 5

    method_map = {m.name: m for m in cls_info.methods}
    assert (
        method_map["__init__"].signature
        == "__init__(self, username: str, email: str | None = None) -> None"
    )
    assert method_map["__init__"].docstring == "Create a new user."

    prop = method_map["display_name"]
    assert "@property" in prop.format()

    cls_m = method_map["from_dict"]
    assert "@classmethod" in cls_m.format()

    stat_m = method_map["validate_name"]
    assert "@staticmethod" in stat_m.format()

    async_m = method_map["fetch_avatar"]
    assert async_m.is_async
    assert "async fetch_avatar(self) -> bytes" in async_m.format()

    summary = findex.format_summary()
    assert "Functions: None" in summary
    assert "Classes: User(BaseModel, Serializable): User account model. [Methods:" in summary


def test_syntax_error_returns_none() -> None:
    source = "def broken( incomplete syntax"
    assert index_source(source, "bad.py") is None


def test_index_file_nonexistent_or_binary(tmp_path: Path) -> None:
    assert index_file(tmp_path / "nonexistent.py") is None

    binary_file = tmp_path / "bad.py"
    binary_file.write_bytes(b"\x80\x81\x82\xff")
    assert index_file(binary_file) is None


def test_should_ignore_path_default_and_custom(tmp_path: Path) -> None:
    cfg = ScannerConfig()

    assert should_ignore_path(tmp_path / ".venv" / "lib.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / "venv" / "sub" / "lib.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / "node_modules" / "pkg" / "a.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / ".git" / "hooks.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / "__pycache__" / "mod.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / "tests" / "test_app.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / ".ghost" / "cache.py", tmp_path, cfg)

    # Ignored files
    assert should_ignore_path(tmp_path / "conftest.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / "setup.py", tmp_path, cfg)
    assert should_ignore_path(tmp_path / "__init__.py", tmp_path, cfg)

    # Valid non-ignored file
    assert not should_ignore_path(tmp_path / "src" / "app.py", tmp_path, cfg)
    assert not should_ignore_path(tmp_path / "service" / "worker.py", tmp_path, cfg)


def test_get_project_files_and_get_project_tree_consistency(tmp_path: Path) -> None:
    """Regression test: get_project_tree must use the exact same ignore rules as file indexing."""
    (tmp_path / "src" / "core").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / ".venv").mkdir(parents=True)

    (tmp_path / "src" / "core" / "engine.py").write_text("def run(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "core" / "util.py").write_text("def util(): pass\n", encoding="utf-8")
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "tests" / "test_engine.py").write_text("def test_run(): pass\n", encoding="utf-8")
    (tmp_path / ".venv" / "pip.py").write_text("pass\n", encoding="utf-8")

    files = get_project_files(tmp_path)
    # Only non-ignored files
    assert files == [Path("src/core/engine.py"), Path("src/core/util.py")]

    tree_str = get_project_tree(tmp_path)
    assert "src/" in tree_str
    assert "core/" in tree_str
    assert "engine.py" in tree_str
    assert "util.py" in tree_str
    # Ignored folders/files must NOT be in the tree
    assert "tests/" not in tree_str
    assert ".venv/" not in tree_str
    assert "__init__.py" not in tree_str


def test_same_basename_in_different_directories_no_collision(tmp_path: Path) -> None:
    """Regression test: paths must be keyed by project-relative path, not basename (SPEC §5)."""
    (tmp_path / "pkg_a").mkdir()
    (tmp_path / "pkg_b").mkdir()

    (tmp_path / "pkg_a" / "service.py").write_text(
        "def service_a() -> str: pass\n", encoding="utf-8"
    )
    (tmp_path / "pkg_b" / "service.py").write_text(
        "def service_b() -> int: pass\n", encoding="utf-8"
    )

    index_map = walk_and_generate_json(tmp_path)

    assert "pkg_a/service.py" in index_map
    assert "pkg_b/service.py" in index_map
    assert "service_a() -> str" in index_map["pkg_a/service.py"]
    assert "service_b() -> int" in index_map["pkg_b/service.py"]

    # Verify context.json on disk
    saved_json = json.loads((tmp_path / ".ghost" / "context.json").read_text(encoding="utf-8"))
    assert "pkg_a/service.py" in saved_json
    assert "pkg_b/service.py" in saved_json


def test_walk_and_modify_json_incremental(tmp_path: Path) -> None:
    (tmp_path / "service.py").write_text("def v1(): pass\n", encoding="utf-8")
    context_file = tmp_path / ".ghost" / "context.json"

    # Initial scan
    walk_and_generate_json(tmp_path, context_file)

    # Valid modification
    (tmp_path / "service.py").write_text("def v2() -> int: pass\n", encoding="utf-8")
    updated = walk_and_modify_json(tmp_path, tmp_path / "service.py", context_file)
    assert updated is not None
    assert "v2() -> int" in updated["service.py"]

    # Syntax error modification: must return None and NOT mutate context.json
    (tmp_path / "service.py").write_text("def broken(", encoding="utf-8")
    result = walk_and_modify_json(tmp_path, tmp_path / "service.py", context_file)
    assert result is None

    persisted = json.loads(context_file.read_text(encoding="utf-8"))
    assert "v2() -> int" in persisted["service.py"]


def test_walk_and_delete_json_incremental(tmp_path: Path) -> None:
    f1 = tmp_path / "a.py"
    f2 = tmp_path / "b.py"
    f1.write_text("def a(): pass\n", encoding="utf-8")
    f2.write_text("def b(): pass\n", encoding="utf-8")
    context_file = tmp_path / ".ghost" / "context.json"

    walk_and_generate_json(tmp_path, context_file)
    assert "a.py" in json.loads(context_file.read_text(encoding="utf-8"))

    walk_and_delete_json(tmp_path, f1, context_file)
    updated = json.loads(context_file.read_text(encoding="utf-8"))
    assert "a.py" not in updated
    assert "b.py" in updated


def test_extract_imports(tmp_path: Path) -> None:
    code = """
import os
import sys
from ghost.config import ScannerConfig
from ..models import User
"""
    f = tmp_path / "imports_test.py"
    f.write_text(code, encoding="utf-8")
    imports = extract_imports(f)
    assert "os" in imports
    assert "sys" in imports
    assert "ghost.config" in imports


def test_context_budgeting_prevents_overflow_and_prioritizes_relevance(tmp_path: Path) -> None:
    """Regression test: budget_context prevents context overflow and prioritizes modules."""
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "main.py"
    target.write_text(
        "import app.helper\nfrom core.util import calc\ndef run(): pass\n", encoding="utf-8"
    )

    index: dict[str, str] = {
        "app/helper.py": "Functions: help_fn() -> None; Classes: None",
        "core/util.py": "Functions: calc(x: int) -> int; Classes: None",
        "app/other.py": "Functions: other() -> None; Classes: None",
        "unrelated/deep/far.py": "Functions: far() -> None; Classes: None",
    }

    # When budget fits everything
    assert len(budget_context(index, target, max_chars=10000, root=tmp_path)) == 4

    # When budget is tight (e.g. only ~120 chars)
    budgeted = budget_context(index, target, max_chars=130, root=tmp_path)
    # Must prioritize app/helper.py and core/util.py over unrelated/deep/far.py
    assert len(budgeted) < 4
    assert "app/helper.py" in budgeted
    assert "unrelated/deep/far.py" not in budgeted
