"""Ghost's command-line interface.

Owns: argument parsing, the mapping from a command to the one function that
implements it, and the process exit code.

Does NOT: contain business logic. Every command body should read as (1) resolve
the project root, (2) load config, (3) build one typed request object, (4) call
exactly one function from the module that owns the behaviour, (5) hand the
result to a reporter. Any command that grows a loop, a state machine, or a
``if provider == ...`` branch has logic that belongs in another module.

The size of this file is itself a design signal: the implementation this project
replaces had a 1,201-line ``cli.py``, roughly a quarter of the whole codebase,
because behaviour kept accumulating in command bodies. If this file passes ~350
lines, something has leaked into it.
"""

from __future__ import annotations

import asyncio
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import click

from ghost import __version__
from ghost.config import find_project_root, load_config
from ghost.errors import ConfigError, GhostError, ProjectNotInitializedError
from ghost.indexer import budget_context, get_project_tree, walk_and_generate_json
from ghost.prompts import build_generation_prompt
from ghost.providers import (
    POPULAR_MODELS,
    get_provider,
    list_available_providers,
)
from ghost.runner import TestRunResult, run_test

if TYPE_CHECKING:
    from collections.abc import Coroutine, Sequence

__all__ = ["cli", "main"]

# Distributions Ghost needs at runtime, checked by `ghost doctor`. Keep in sync
# with [project.dependencies] in pyproject.toml -- doctor exists to tell a user
# why Ghost is broken on their machine, so a stale list here defeats the point.
_RUNTIME_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "click",
    "pydantic",
    "pydantic-settings",
    "httpx",
    "groq",
    "rich",
    "watchdog",
    "pytest",
)

# Optional extras. Absence is normal and reported as such, never as a failure.
_OPTIONAL_DISTRIBUTIONS: Final[tuple[str, ...]] = (
    "openai",
    "anthropic",
)

_OK: Final = "ok"
_MISSING: Final = "MISSING"


def _installed_version(distribution: str) -> str | None:
    """Return the installed version of *distribution*, or ``None`` if absent."""
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None


@click.group(invoke_without_command=True)
@click.version_option(__version__, "--version", "-v", prog_name="ghost")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Ghost -- generate, run, and heal tests for your Python project."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cli.command()
def version() -> None:
    """Show Ghost, Python, and platform versions."""
    click.echo(f"ghost   {__version__}")
    click.echo(f"python  {platform.python_version()} ({sys.executable})")
    click.echo(f"system  {platform.system()} {platform.release()}")


@cli.command()
def doctor() -> None:
    """Check that Ghost's environment is healthy.

    Reports the interpreter in use and whether each dependency is importable.
    This command grows with the project: later stages add configuration
    discovery and provider credential checks as those concepts come into
    existence.
    """
    click.echo("Ghost environment check")
    click.echo("=" * 46)

    click.echo(f"\nghost       {__version__}")
    click.echo(f"python      {platform.python_version()}")
    click.echo(f"executable  {sys.executable}")

    click.echo("\nRequired dependencies")
    missing: list[str] = []
    for distribution in _RUNTIME_DISTRIBUTIONS:
        found = _installed_version(distribution)
        if found is None:
            missing.append(distribution)
        status = found if found is not None else _MISSING
        click.echo(f"  {distribution:<20} {status}")

    click.echo("\nOptional providers")
    for distribution in _OPTIONAL_DISTRIBUTIONS:
        found = _installed_version(distribution)
        status = found if found is not None else "not installed"
        click.echo(f"  {distribution:<20} {status}")

    click.echo("\nConfiguration")
    project_root = find_project_root()
    if project_root is not None:
        try:
            load_config(project_root, must_exist=True)
            click.echo(f"  ghost.toml           found ({project_root / 'ghost.toml'})")
        except ConfigError as err:
            click.echo(f"  ghost.toml           invalid ({err})")
    else:
        click.echo("  ghost.toml           not found")

    click.echo("\nAI Providers")
    config_for_doctor = load_config()
    provider_status: dict[str, bool] = _run_async(list_available_providers(config_for_doctor))
    for prov_name, is_avail in sorted(provider_status.items()):
        status_str = "available" if is_avail else "not configured"
        click.echo(f"  {prov_name:<20} {status_str}")

    click.echo("")
    if missing:
        click.echo(f"{len(missing)} required dependency/dependencies missing: {', '.join(missing)}")
        click.echo("Run 'uv sync' to install them.")
        raise SystemExit(1)
    click.echo(f"All required dependencies present ({_OK}).")


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """The single event-loop boundary in Ghost."""
    return asyncio.run(coro)


@cli.command("config")
@click.option(
    "--show",
    "-s",
    "_show",
    is_flag=True,
    default=False,
    help="Pretty-print the current ghost.toml configuration.",
)
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
def config_cmd(*, _show: bool = False, path: Path | None = None) -> None:
    """View and inspect Ghost project configuration."""
    target = path or Path.cwd()
    root = find_project_root(target)
    if target.is_file() and target.name == "ghost.toml":
        config_file = target
    elif root is not None:
        config_file = root / "ghost.toml"
    else:
        raise ProjectNotInitializedError(target)

    load_config(config_file, must_exist=True)
    click.echo(config_file.read_text(encoding="utf-8").strip())


@cli.command("providers")
def providers_cmd() -> None:
    """List supported LLM providers and their availability."""
    config = load_config()
    availability: dict[str, bool] = _run_async(list_available_providers(config))

    click.echo("Supported Providers")
    click.echo("=" * 40)
    for name, available in sorted(availability.items()):
        status = "available (configured)" if available else "not configured"
        click.echo(f"  {name:<15} {status}")

    click.echo("\nPopular Models")
    click.echo("=" * 40)
    for model_id, model_cfg in POPULAR_MODELS.items():
        click.echo(f"  {model_id:<25} ({model_cfg.provider}) - {model_cfg.description}")


@cli.command("models")
@click.option(
    "--provider",
    "-p",
    default=None,
    help="Provider to query for models (defaults to configured provider).",
)
def models_cmd(provider: str | None) -> None:
    """List models available live from the provider."""
    config = load_config()
    provider_name = provider or config.ai.provider
    prov = get_provider(provider_name, config=config)

    click.echo(f"Fetching models live from {prov.name}...")
    models: list[str] = _run_async(prov.list_models())

    click.echo(f"\nAvailable models ({len(models)}):")
    for model in models:
        prefix = "  * " if model == config.ai.model else "    "
        click.echo(f"{prefix}{model}")


@cli.command("index")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=True),
    required=False,
    default=None,
)
@click.option(
    "--show",
    "-s",
    "_show",
    is_flag=True,
    default=False,
    help="Display the project index and extracted AST symbols.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    default=False,
    help="Output the index as JSON.",
)
@click.option(
    "--budget",
    type=int,
    default=None,
    help="Limit output to a character budget (context budgeting).",
)
@click.option(
    "--for-file",
    type=click.Path(path_type=Path),
    default=None,
    help="Target file for context budgeting prioritization.",
)
def index_cmd(
    path: Path | None = None,
    *,
    _show: bool = False,
    as_json: bool = False,
    budget: int | None = None,
    for_file: Path | None = None,
) -> None:
    """Index project Python files and build AST context."""
    target = path or Path.cwd()
    project_root = find_project_root(target) or target
    config = load_config(project_root)

    index_data = walk_and_generate_json(project_root, scanner_config=config.scanner)

    if budget is not None or for_file is not None:
        effective_budget = budget if budget is not None else 8000
        index_data = budget_context(
            index_data,
            target_file=for_file,
            max_chars=effective_budget,
            root=project_root,
        )

    if as_json:
        click.echo(json.dumps(index_data, indent=2, sort_keys=True))
    elif _show:
        click.echo(f"Ghost AST Project Index: {project_root.name} ({len(index_data)} files)")
        click.echo("=" * 60)
        for file_key, summary in sorted(index_data.items()):
            click.echo(f"\n{file_key}\n  {summary}")
    else:
        click.echo(f"Indexed {len(index_data)} file(s) in {project_root} -> .ghost/context.json")
        click.echo("Use 'ghost index --show' to view the indexed symbols.")


@cli.command("prompt")
@click.argument("file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
def prompt_cmd(file: Path) -> None:
    """Print the test generation prompt for a source file without making an API call."""
    resolved_file = file.resolve()
    project_root = find_project_root(resolved_file) or resolved_file.parent
    config = load_config(project_root)

    source_code = resolved_file.read_text(encoding="utf-8")
    rel_path = (
        resolved_file.relative_to(project_root).as_posix()
        if project_root in resolved_file.parents
        else resolved_file.name
    )

    tree_str = get_project_tree(project_root, config.scanner)
    index = walk_and_generate_json(project_root, scanner_config=config.scanner)
    budgeted = budget_context(index, target_file=rel_path, root=project_root)

    prompt = build_generation_prompt(
        source_code=source_code,
        source_path=rel_path,
        project_tree=tree_str,
        context=budgeted,
        framework=config.tests.framework,
    )
    click.echo(prompt)


@cli.command("run-tests")
@click.argument("test_file", type=click.Path(path_type=Path, exists=True, dir_okay=False))
@click.option("--timeout", type=float, default=None, help="Execution timeout in seconds.")
def run_tests_cmd(test_file: Path, timeout: float | None = None) -> None:
    """Run a test file in an isolated subprocess and report structured results."""
    resolved_test = test_file.resolve()
    project_root = find_project_root(resolved_test) or resolved_test.parent
    config = load_config(project_root)

    effective_timeout = (
        timeout if timeout is not None else getattr(config.tests, "timeout_seconds", 30.0)
    )
    result: TestRunResult = _run_async(
        run_test(resolved_test, project_root, timeout_seconds=effective_timeout)
    )

    if result.passed:
        click.echo(f"PASS: {test_file}")
    else:
        click.echo(f"FAIL ({result.classification}): {test_file}")
        if result.timed_out:
            click.echo(f"  Execution timed out after {effective_timeout:.1f}s.")
        elif result.exception_type:
            click.echo(f"  {result.exception_type}: {result.message}")
        raise SystemExit(1)


def _system_exit_code(exc: SystemExit) -> int:
    """Normalise ``SystemExit.code``, which may be ``None`` or a non-integer."""
    if exc.code is None:
        return 0
    return exc.code if isinstance(exc.code, int) else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code rather than calling ``sys.exit``.

    ``standalone_mode=False`` stops click from exiting the process itself, which
    is what lets this function own the error boundary: an anticipated
    :class:`~ghost.errors.GhostError` becomes a one-line message, while anything
    else propagates with its traceback intact, because an unexpected exception
    is a bug in Ghost and hiding it would make that bug harder to find.
    """
    try:
        cli.main(args=argv, standalone_mode=False)
    except click.exceptions.Exit as exc:
        return exc.exit_code
    except click.exceptions.Abort:
        click.echo("Aborted.", err=True)
        return 130
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except GhostError as exc:
        click.echo(f"Error: {exc}", err=True)
        return 1
    except SystemExit as exc:  # raised by commands that fail their own checks
        return _system_exit_code(exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
