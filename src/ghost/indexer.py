"""AST-based project indexing and context extraction.

Owns:
- Pure static analysis of Python source files using standard library ``ast``.
- Extracting top-level functions (with full type hints and default values),
  classes, methods (with type hints and decorators), and docstrings.
- Single shared file discovery and project tree generation respecting
  :class:`~ghost.config.ScannerConfig`.
- Serialization to ``.ghost/context.json`` keyed by project-relative path
  (preventing same-basename collisions).
- Context budgeting so large projects do not exceed LLM context window limits.
- Incremental index updates (modify/delete) for filesystem watchers.

Does NOT:
- Invoke LLMs or perform semantic/vector search (this is pure deterministic AST).
- Execute user code (safe against arbitrary code execution).
- Overwrite user test files.

Guarantees:
- Never crashes on syntax errors or encoding issues (skips and logs cleanly).
- Keyed strictly by relative POSIX paths to eliminate filename collisions.
- Shared ignore logic: ``get_project_files`` and ``get_project_tree`` always
  use the same :class:`~ghost.config.ScannerConfig`.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ghost.config import ScannerConfig

logger = logging.getLogger(__name__)

__all__ = [
    "ClassInfo",
    "FileIndex",
    "FunctionInfo",
    "MethodInfo",
    "budget_context",
    "extract_imports",
    "format_parameters",
    "get_project_files",
    "get_project_tree",
    "index_file",
    "index_source",
    "should_ignore_path",
    "walk_and_delete_json",
    "walk_and_generate_json",
    "walk_and_modify_json",
]


@dataclass(frozen=True)
class FunctionInfo:
    """Metadata for a top-level function extracted from AST."""

    name: str
    signature: str
    docstring: str | None = None
    is_async: bool = False

    def format(self, *, include_docstring: bool = True) -> str:
        prefix = "async " if self.is_async else ""
        base = f"{prefix}{self.signature}"
        if include_docstring and self.docstring:
            return f"{base} - {self.docstring}"
        return base


@dataclass(frozen=True)
class MethodInfo:
    """Metadata for a class method extracted from AST."""

    name: str
    signature: str
    docstring: str | None = None
    is_async: bool = False
    decorators: list[str] = field(default_factory=list)

    def format(self, *, include_docstring: bool = True) -> str:
        prefix = ""
        for dec in self.decorators:
            if dec in ("staticmethod", "classmethod", "property"):
                prefix = f"@{dec} "
                break
        if self.is_async:
            prefix = f"{prefix}async "
        base = f"{prefix}{self.signature}"
        if include_docstring and self.docstring:
            return f"{base} - {self.docstring}"
        return base


@dataclass(frozen=True)
class ClassInfo:
    """Metadata for a class extracted from AST."""

    name: str
    bases: list[str] = field(default_factory=list)
    docstring: str | None = None
    methods: list[MethodInfo] = field(default_factory=list)

    def format(self, *, include_docstrings: bool = True) -> str:
        bases_str = f"({', '.join(self.bases)})" if self.bases else ""
        doc_str = f": {self.docstring}" if (include_docstrings and self.docstring) else ""
        if self.methods:
            methods_str = ", ".join(
                m.format(include_docstring=include_docstrings) for m in self.methods
            )
        else:
            methods_str = "None"
        return f"{self.name}{bases_str}{doc_str} [Methods: {methods_str}]"


@dataclass(frozen=True)
class FileIndex:
    """AST index of a single Python file."""

    path: str
    docstring: str | None = None
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)

    def format_summary(self, *, include_docstrings: bool = True) -> str:
        if self.functions:
            fns_str = ", ".join(
                f.format(include_docstring=include_docstrings) for f in self.functions
            )
        else:
            fns_str = "None"

        if self.classes:
            cls_str = ", ".join(
                c.format(include_docstrings=include_docstrings) for c in self.classes
            )
        else:
            cls_str = "None"

        return f"Functions: {fns_str}; Classes: {cls_str}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "docstring": self.docstring,
            "functions": [
                {
                    "name": f.name,
                    "signature": f.signature,
                    "docstring": f.docstring,
                    "is_async": f.is_async,
                }
                for f in self.functions
            ],
            "classes": [
                {
                    "name": c.name,
                    "bases": c.bases,
                    "docstring": c.docstring,
                    "methods": [
                        {
                            "name": m.name,
                            "signature": m.signature,
                            "docstring": m.docstring,
                            "is_async": m.is_async,
                            "decorators": m.decorators,
                        }
                        for m in c.methods
                    ],
                }
                for c in self.classes
            ],
        }


def format_parameters(args: ast.arguments) -> str:
    """Format an ``ast.arguments`` node into a Python parameter list with type hints."""
    parts: list[str] = []
    pos_args = args.posonlyargs + args.args
    diff = len(pos_args) - len(args.defaults)

    for i, arg in enumerate(pos_args):
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        default = ""
        if i >= diff:
            default = f" = {ast.unparse(args.defaults[i - diff])}"
        parts.append(f"{arg.arg}{annotation}{default}")
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            parts.append("/")

    if args.vararg:
        ann = f": {ast.unparse(args.vararg.annotation)}" if args.vararg.annotation else ""
        parts.append(f"*{args.vararg.arg}{ann}")
    elif args.kwonlyargs:
        parts.append("*")

    for arg, default_node in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        annotation = f": {ast.unparse(arg.annotation)}" if arg.annotation else ""
        default = f" = {ast.unparse(default_node)}" if default_node is not None else ""
        parts.append(f"{arg.arg}{annotation}{default}")

    if args.kwarg:
        ann = f": {ast.unparse(args.kwarg.annotation)}" if args.kwarg.annotation else ""
        parts.append(f"**{args.kwarg.arg}{ann}")

    return ", ".join(parts)


def _format_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Format a function or method signature including parameters and return type."""
    params = format_parameters(node.args)
    returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
    return f"{node.name}({params}){returns}"


def _first_line_docstring(
    node: ast.AsyncFunctionDef | ast.FunctionDef | ast.ClassDef | ast.Module,
) -> str | None:
    """Return the first non-empty summary line of a node's docstring, if any."""
    raw = ast.get_docstring(node)
    if not raw:
        return None
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


def index_source(source: str, file_path: str = "") -> FileIndex | None:
    """Parse Python source code and extract functions, classes, and docstrings.

    Returns ``None`` if the source has a syntax error.
    """
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        logger.warning("Syntax error parsing %s; skipping index", file_path)
        return None

    module_doc = _first_line_docstring(tree)
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []

    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sig = _format_signature(stmt)
            doc = _first_line_docstring(stmt)
            is_async = isinstance(stmt, ast.AsyncFunctionDef)
            functions.append(
                FunctionInfo(
                    name=stmt.name,
                    signature=sig,
                    docstring=doc,
                    is_async=is_async,
                )
            )
        elif isinstance(stmt, ast.ClassDef):
            class_doc = _first_line_docstring(stmt)
            bases = [ast.unparse(b) for b in stmt.bases]
            methods: list[MethodInfo] = []
            for item in stmt.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    m_sig = _format_signature(item)
                    m_doc = _first_line_docstring(item)
                    m_async = isinstance(item, ast.AsyncFunctionDef)
                    decs = [ast.unparse(d) for d in item.decorator_list]
                    methods.append(
                        MethodInfo(
                            name=item.name,
                            signature=m_sig,
                            docstring=m_doc,
                            is_async=m_async,
                            decorators=decs,
                        )
                    )
            classes.append(
                ClassInfo(
                    name=stmt.name,
                    bases=bases,
                    docstring=class_doc,
                    methods=methods,
                )
            )

    return FileIndex(
        path=file_path,
        docstring=module_doc,
        functions=functions,
        classes=classes,
    )


def index_file(file_path: Path, root: Path | None = None) -> FileIndex | None:
    """Index a single Python file on disk.

    Returns ``None`` on syntax or decode errors.
    """
    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as err:
        logger.warning("Cannot read %s: %s", file_path, err)
        return None

    if root is not None:
        try:
            rel_path = file_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel_path = file_path.as_posix()
    else:
        rel_path = file_path.as_posix()
    return index_source(content, file_path=rel_path)


def should_ignore_path(
    path: Path,
    root: Path,
    scanner_config: ScannerConfig,
    *,
    is_dir: bool = False,
) -> bool:
    """Determine whether *path* should be ignored per *scanner_config*.

    Evaluates path components against ``ignore_dirs`` and file names against
    ``ignore_files``.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        rel = path

    parts = rel.parts
    if not parts or parts == (".",):
        return False

    check_dirs = parts if is_dir else parts[:-1]
    for part in check_dirs:
        if part in scanner_config.ignore_dirs:
            return True

    if not is_dir:
        filename = parts[-1]
        if filename in scanner_config.ignore_files:
            return True

    return False


def get_project_files(
    root: Path,
    scanner_config: ScannerConfig | None = None,
) -> list[Path]:
    """Discover all non-ignored Python source files under *root*.

    Returns paths relative to *root*, sorted lexicographically.
    """
    cfg = scanner_config or ScannerConfig()
    resolved_root = root.resolve()
    results: list[Path] = []

    for dirpath, dirnames, filenames in os.walk(resolved_root, followlinks=False):
        current_dir = Path(dirpath)
        # Prune ignored directories in-place so os.walk does not recurse into them
        dirnames[:] = [
            d
            for d in dirnames
            if not should_ignore_path(current_dir / d, resolved_root, cfg, is_dir=True)
        ]

        for fname in filenames:
            if not fname.endswith(".py"):
                continue
            file_path = current_dir / fname
            if not should_ignore_path(file_path, resolved_root, cfg, is_dir=False):
                results.append(file_path.relative_to(resolved_root))

    results.sort()
    return results


def get_project_tree(
    root: Path,
    scanner_config: ScannerConfig | None = None,
) -> str:
    """Generate a clean ASCII directory tree of non-ignored Python files.

    Uses the exact same :class:`~ghost.config.ScannerConfig` as file indexing,
    guaranteeing no drift between tree presentation and indexed files.
    """
    files = get_project_files(root, scanner_config)
    if not files:
        return f"{root.name}/\n(no Python files found)"

    tree_dict: dict[str, Any] = {}
    for rel_path in files:
        current = tree_dict
        for part in rel_path.parts:
            current = current.setdefault(part, {})

    lines: list[str] = [f"{root.name}/"]

    def _render(subtree: dict[str, Any], prefix: str = "") -> None:
        items = sorted(subtree.items(), key=lambda kv: (bool(kv[1]), kv[0]))
        count = len(items)
        for i, (name, children) in enumerate(items):
            is_last = i == count - 1
            connector = "└── " if is_last else "├── "
            display_name = f"{name}/" if children else name
            lines.append(f"{prefix}{connector}{display_name}")
            if children:
                extension = "    " if is_last else "│   "
                _render(children, prefix + extension)

    _render(tree_dict)
    return "\n".join(lines)


def walk_and_generate_json(
    root: Path,
    output_file: Path | None = None,
    scanner_config: ScannerConfig | None = None,
) -> dict[str, str]:
    """Scan the entire project and write the AST index to ``context.json``.

    Returns a mapping from project-relative file paths to their summary strings.
    """
    cfg = scanner_config or ScannerConfig()
    resolved_root = root.resolve()
    target_out = output_file or (resolved_root / ".ghost" / "context.json")

    files = get_project_files(resolved_root, cfg)
    index_map: dict[str, str] = {}

    for rel_path in files:
        full_path = resolved_root / rel_path
        findex = index_file(full_path, resolved_root)
        if findex is not None:
            index_map[rel_path.as_posix()] = findex.format_summary()

    _write_context_atomic(target_out, index_map)
    return index_map


def _write_context_atomic(target_context: Path, data: dict[str, str]) -> None:
    target_context.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=target_context.parent, delete=False, encoding="utf-8"
    ) as f:
        f.write(json.dumps(data, indent=2, sort_keys=True))
        temp_path = Path(f.name)
    temp_path.replace(target_context)


def walk_and_modify_json(
    root: Path,
    file_path: Path,
    context_file: Path | None = None,
    scanner_config: ScannerConfig | None = None,
) -> dict[str, str] | None:
    """Incrementally update ``context.json`` after a file is modified.

    Returns ``None`` and leaves ``context.json`` untouched if the file has a
    syntax error or should be ignored. Callers use a ``None`` return as a signal
    to skip generation/healing for this save event.
    """
    cfg = scanner_config or ScannerConfig()
    resolved_root = root.resolve()
    resolved_file = file_path.resolve()

    if should_ignore_path(resolved_file, resolved_root, cfg, is_dir=False):
        return None

    findex = index_file(resolved_file, resolved_root)
    if findex is None:
        return None

    target_context = context_file or (resolved_root / ".ghost" / "context.json")
    context_data: dict[str, str] = {}
    if target_context.is_file():
        try:
            context_data = json.loads(target_context.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            context_data = {}

    try:
        rel_key = resolved_file.relative_to(resolved_root).as_posix()
    except ValueError:
        rel_key = resolved_file.name

    context_data[rel_key] = findex.format_summary()

    _write_context_atomic(target_context, context_data)
    return context_data


def walk_and_delete_json(
    root: Path,
    file_path: Path,
    context_file: Path | None = None,
) -> dict[str, str]:
    """Incrementally remove a deleted file from ``context.json``."""
    resolved_root = root.resolve()
    target_context = context_file or (resolved_root / ".ghost" / "context.json")

    context_data: dict[str, str] = {}
    if target_context.is_file():
        try:
            context_data = json.loads(target_context.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            context_data = {}

    try:
        rel_key = file_path.resolve().relative_to(resolved_root).as_posix()
    except ValueError:
        rel_key = file_path.as_posix()

    if rel_key in context_data:
        del context_data[rel_key]
        _write_context_atomic(target_context, context_data)

    return context_data


def extract_imports(file_path: Path) -> set[str]:
    """Extract top-level imported module names from a Python source file.

    Returns a set of imported module prefixes (e.g. ``{"ghost.config", "os"}``).
    """
    try:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _score_file_relevance(
    candidate_path: str,
    target_path: str | None,
    imported_modules: set[str],
) -> int:
    """Compute a relevance score for a candidate file relative to a target file."""
    if target_path is None:
        return 0

    if candidate_path == target_path:
        return 200

    # Convert candidate path to module style (e.g. src/ghost/config.py -> ghost.config)
    cand_parts = Path(candidate_path).with_suffix("").parts
    cand_mod = ".".join(cand_parts)

    score = 0
    # Direct import match gets top score
    for imp in imported_modules:
        if imp == cand_mod or cand_mod.endswith(imp) or imp.endswith(cand_mod):
            score += 100
            break

    # Same directory gets proximity boost
    cand_dir = Path(candidate_path).parent
    target_dir = Path(target_path).parent
    if cand_dir == target_dir:
        score += 50
    else:
        # Common prefix depth
        cand_p = cand_dir.parts
        tgt_p = target_dir.parts
        common = 0
        for p1, p2 in zip(cand_p, tgt_p, strict=False):
            if p1 == p2:
                common += 1
            else:
                break
        score += common * 10

    return score


def budget_context(
    index: dict[str, str],
    target_file: str | Path | None = None,
    *,
    max_chars: int = 8000,
    max_tokens: int | None = None,
    root: Path | None = None,
) -> dict[str, str]:
    """Select a budgeted subset of the index to prevent LLM context overflow.

    Files are prioritized by relevance to *target_file*:
    1. Modules directly imported by *target_file*.
    2. Files located in the same directory.
    3. Files sharing a common ancestor directory.

    Guarantees the serialized JSON size of the returned dict does not exceed
    *max_chars* (or *max_tokens* * 4), while preserving complete entries.
    """
    effective_limit = max_tokens * 4 if max_tokens is not None else max_chars

    # If the full index already fits within budget, return it directly
    full_dump = json.dumps(index, indent=2)
    if len(full_dump) <= effective_limit:
        return dict(index)

    target_rel: str | None = None
    imported_modules: set[str] = set()

    if target_file is not None:
        target_p = Path(target_file)
        if root is not None and target_p.is_absolute():
            try:
                target_rel = target_p.resolve().relative_to(root.resolve()).as_posix()
            except ValueError:
                target_rel = target_p.as_posix()
        else:
            target_rel = target_p.as_posix()

        full_target_path = (root / target_rel) if root else target_p
        if full_target_path.is_file():
            imported_modules = extract_imports(full_target_path)

    # Score and sort candidate files by relevance descending, then by path
    scored_candidates = [
        (_score_file_relevance(path_str, target_rel, imported_modules), path_str)
        for path_str in index
    ]
    scored_candidates.sort(key=lambda item: (-item[0], item[1]))

    budgeted: dict[str, str] = {}
    for _score, path_str in scored_candidates:
        candidate_dict = {**budgeted, path_str: index[path_str]}
        dump_len = len(json.dumps(candidate_dict, indent=2))
        if dump_len <= effective_limit or not budgeted:
            budgeted[path_str] = index[path_str]
        else:
            # Reached budget capacity
            break

    return budgeted
