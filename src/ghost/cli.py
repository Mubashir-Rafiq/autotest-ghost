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
import contextlib
import json
import os
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

import click

from ghost import __version__
from ghost.client import is_ghost_managed_test
from ghost.config import (
    _PROVIDER_ENV_VARS,
    find_project_root,
    generate_default_config,
    load_config,
    write_default_config,
)
from ghost.console import (
    print_banner,
    print_batch_summary,
    print_history_table,
    print_providers_table,
    print_usage_stats,
)
from ghost.daemon import (
    daemon_log_path,
    follow_log_stream,
    query_daemon_status,
    start_daemon,
    stop_daemon,
    tail_log,
)
from ghost.errors import (
    ConfigError,
    ExitCode,
    GhostError,
    HandwrittenTestOverwriteError,
    ProjectNotInitializedError,
)
from ghost.history import HistoryTracker, UsageTracker
from ghost.indexer import (
    budget_context,
    get_project_files,
    get_project_tree,
    walk_and_generate_json,
)
from ghost.pipeline import (
    PipelineEvent,
    PipelineListener,
    PipelineResult,
    PipelineStatus,
    TestPipeline,
    resolve_test_path,
)
from ghost.prompts import build_generation_prompt
from ghost.providers import (
    POPULAR_MODELS,
    get_provider,
    list_available_providers,
)
from ghost.runner import TestRunResult, run_test
from ghost.watcher import FileWatcher, is_test_file

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

_INIT_PROVIDER_CHOICES: Final[tuple[str, ...]] = (
    "groq",
    "openai",
    "ollama",
    "anthropic",
    "openrouter",
    "lmstudio",
    "custom",
)

_PROVIDER_DEFAULT_MODELS: Final[dict[str, str]] = {
    "groq": "openai/gpt-oss-120b",
    "openai": "gpt-4o",
    "ollama": "llama3:latest",
    "anthropic": "claude-sonnet-4-20250514",
    "openrouter": "anthropic/claude-3.5-sonnet",
    "lmstudio": "local-model",
    "custom": "default",
}


def _installed_version(distribution: str) -> str | None:
    """Return the installed version of *distribution*, or ``None`` if absent."""
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return None


def _detect_default_provider() -> str:
    """Auto-detect the most likely configured provider from environment variables."""
    for prov in ("groq", "openai", "anthropic", "openrouter"):
        for var in _PROVIDER_ENV_VARS.get(prov, ()):
            if os.environ.get(var):
                return prov
    if os.environ.get("GHOST_API_KEY"):
        return "groq"
    return "groq"


def _offer_save_to_env(project_root: Path, var_name: str, key_value: str) -> None:
    """Offer to save an API key to the project's .env file."""
    if click.confirm(f"Save {var_name} to .env?", default=True):
        env_file = project_root / ".env"
        line = f"{var_name}={key_value}\n"
        if env_file.is_file():
            existing = env_file.read_text(encoding="utf-8")
            if var_name in existing:
                updated = re.sub(
                    rf"^{re.escape(var_name)}=.*$",
                    f"{var_name}={key_value}",
                    existing,
                    flags=re.MULTILINE,
                )
                env_file.write_text(updated, encoding="utf-8")
            else:
                with env_file.open("a", encoding="utf-8") as f:
                    f.write(line)
        else:
            env_file.write_text(line, encoding="utf-8")
        click.echo(f"Saved {var_name} to {env_file}")
    else:
        click.echo(f"You can set it manually: export {var_name}=<your-key>")


def _resolve_init_provider(provider: str) -> str:
    """Prompt the user for the provider choice, prefilled with auto/passed default."""
    default_provider = (
        _detect_default_provider() if provider.lower() == "auto" else provider.lower()
    )
    return click.prompt(
        "AI provider",
        type=click.Choice(_INIT_PROVIDER_CHOICES, case_sensitive=False),
        default=default_provider,
    ).lower()


def _handle_init_api_key(target: Path, chosen_provider: str) -> None:
    """Detect or prompt for provider API key during init."""
    env_vars = _PROVIDER_ENV_VARS.get(chosen_provider, ("GHOST_API_KEY",))
    existing_key: str | None = None
    existing_var: str | None = None
    for var in env_vars:
        val = os.environ.get(var)
        if val and val.strip():
            existing_key = val.strip()
            existing_var = var
            break

    if existing_key:
        masked = existing_key[:4] + "*" * max(0, len(existing_key) - 4)
        click.echo(f"Found API key in ${existing_var}: {masked}")
        if not click.confirm("Use this key?", default=True):
            new_key = click.prompt("Enter API key", hide_input=True)
            if new_key.strip():
                _offer_save_to_env(target, env_vars[0], new_key.strip())
    elif chosen_provider not in {"ollama", "lmstudio"}:
        new_key = click.prompt(
            f"Enter {env_vars[0]} (or press Enter to skip)",
            default="",
            show_default=False,
            hide_input=True,
        )
        if new_key.strip():
            _offer_save_to_env(target, env_vars[0], new_key.strip())


@click.group(invoke_without_command=True)
@click.version_option(__version__, "--version", "-v", prog_name="ghost")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """Ghost -- generate, run, and heal tests for your Python project."""
    if ctx.invoked_subcommand is None:
        print_banner()
        click.echo(ctx.get_help())


@cli.command()
def version() -> None:
    """Show Ghost, Python, and platform versions."""
    click.echo(f"ghost   {__version__}")
    click.echo(f"python  {platform.python_version()} ({sys.executable})")
    click.echo(f"system  {platform.system()} {platform.release()}")


@cli.command("init")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
@click.option(
    "--provider",
    "-p",
    type=click.Choice([*_INIT_PROVIDER_CHOICES, "auto"], case_sensitive=False),
    default="auto",
    help="AI provider to use (default: auto).",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="Model to use (default: provider default).",
)
@click.option(
    "--framework",
    "-f",
    type=click.Choice(["pytest", "unittest"], case_sensitive=False),
    default="pytest",
    help="Test framework (default: pytest).",
)
def init_cmd(
    path: Path | None = None,
    provider: str = "auto",
    model: str | None = None,
    framework: str = "pytest",
) -> None:
    """Initialize a new Ghost project with interactive wizard."""
    target = (path or Path.cwd()).resolve()
    config_file = target / "ghost.toml"

    if config_file.is_file() and not click.confirm(
        f"ghost.toml already exists at {config_file}. Overwrite?", default=False
    ):
        click.echo("Init cancelled.")
        return

    chosen_provider = _resolve_init_provider(provider)
    _handle_init_api_key(target, chosen_provider)

    default_model = _PROVIDER_DEFAULT_MODELS.get(chosen_provider, "openai/gpt-oss-120b")
    chosen_model = model or click.prompt("Model", default=default_model)

    content = generate_default_config(
        name=target.name,
        provider=chosen_provider,
        model=chosen_model,
        framework=framework,
    )
    config_file.write_text(content, encoding="utf-8")
    click.echo(f"Created {config_file}")

    ghost_dir = target / ".ghost"
    ghost_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(target)
    index = walk_and_generate_json(target, scanner_config=config.scanner)
    context_file = ghost_dir / "context.json"
    context_file.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    click.echo(f"Indexed {len(index)} file(s) -> {context_file}")
    click.echo("Ghost initialized. Run 'ghost watch' to start.")


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
    print_providers_table(availability, POPULAR_MODELS)


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
@click.option("--cov/--no-cov", default=False, help="Run with pytest-cov coverage reporting.")
@click.option(
    "--cov-source",
    type=str,
    default=None,
    help="Package or directory name to measure coverage for.",
)
def run_tests_cmd(
    test_file: Path,
    timeout: float | None = None,
    *,
    cov: bool = False,
    cov_source: str | None = None,
) -> None:
    """Run a test file in an isolated subprocess and report structured results."""
    resolved_test = test_file.resolve()
    project_root = find_project_root(resolved_test) or resolved_test.parent
    config = load_config(project_root)

    effective_timeout = (
        timeout if timeout is not None else getattr(config.tests, "timeout_seconds", 30.0)
    )
    result: TestRunResult = _run_async(
        run_test(
            resolved_test,
            project_root,
            timeout_seconds=effective_timeout,
            coverage=cov,
            cov_source=cov_source,
        )
    )

    if result.passed:
        click.echo(f"PASS: {test_file}")
        if result.coverage_summary:
            click.echo(f"Coverage: {result.coverage_summary}")
    else:
        click.echo(f"FAIL ({result.classification}): {test_file}")
        if result.coverage_summary:
            click.echo(f"Coverage: {result.coverage_summary}")
        if result.timed_out:
            click.echo(f"  Execution timed out after {effective_timeout:.1f}s.")
        elif result.exception_type:
            click.echo(f"  {result.exception_type}: {result.message}")
        raise SystemExit(1)


class CliPipelineListener(PipelineListener):
    """Terminal listener printing progress messages to stdout."""

    def __init__(self, *, verbose: bool = False, show_results: bool = False) -> None:
        self.verbose = verbose
        self.show_results = show_results

    def _handle_result(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        if event == PipelineEvent.PASSED and self.show_results:
            test_file = data.get("test_file")
            attempt = data.get("attempt", 0)
            if attempt > 0:
                click.echo(f"PASS (healed after {attempt} attempt(s)): {test_file}")
            else:
                click.echo(f"PASS: {test_file}")
        elif event == PipelineEvent.FAILED and self.show_results:
            test_file = data.get("test_file")
            click.echo(f"FAIL: {test_file}")
        elif event == PipelineEvent.SKIPPED and self.verbose:
            src = data.get("source_file")
            click.echo(f"SKIPPED: {Path(src).name if src else 'source'} (unchanged)")

    @override
    def on_event(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        if event == PipelineEvent.GENERATING:
            src = data.get("source_file")
            click.echo(f"Generating tests for {Path(src).name if src else 'source'}...")
        elif event == PipelineEvent.RUNNING:
            attempt = data.get("attempt", 0)
            if attempt > 0:
                click.echo(f"Re-running test (attempt {attempt})...")
            else:
                click.echo("Running tests...")
        elif event == PipelineEvent.HEALING:
            attempt = data.get("attempt", 1)
            cls_name = data.get("classification", "UNKNOWN")
            click.echo(f"Healing test failure ({cls_name}, attempt {attempt})...")
        elif event == PipelineEvent.JUDGING:
            click.echo("Assertion failure: consulting Judge...")
        elif event == PipelineEvent.JUDGE_RESULT:
            outcome = data.get("outcome")
            click.echo(f"Judge evaluated failure as: {outcome}")
        else:
            self._handle_result(event, data)


def _run_batch_generation(
    search_dir: Path,
    project_root: Path,
    config: Any,
    *,
    force: bool,
    if_changed: bool,
    auto_heal: bool | None,
    use_judge: bool | None,
    timeout: float | None,
    cov: bool,
    cov_source: str | None,
) -> None:
    candidate_rel_paths = get_project_files(search_dir, config.scanner)
    source_files = [
        (search_dir / rel).resolve()
        for rel in candidate_rel_paths
        if not is_test_file(search_dir / rel, project_root, config.tests.output_dir)
    ]
    if not source_files:
        click.echo(f"No Python source files found in {search_dir}.")
        return

    click.echo(f"Found {len(source_files)} source file(s) for batch test generation.")
    pipeline = TestPipeline(config=config, project_root=project_root)

    async def _run_batch() -> list[PipelineResult]:
        batch_results: list[PipelineResult] = []
        for src in source_files:
            listener = CliPipelineListener()
            res = await pipeline.run(
                src,
                force=force,
                force_generate=True,
                if_changed=if_changed,
                auto_heal=auto_heal,
                use_judge=use_judge,
                timeout_seconds=timeout,
                coverage=cov,
                cov_source=cov_source,
                listener=listener,
            )
            batch_results.append(res)
        return batch_results

    results = _run_async(_run_batch())
    print_batch_summary(results)

    has_failure = any(not r.passed and r.status != PipelineStatus.SKIPPED for r in results)
    if has_failure:
        raise SystemExit(1)


def _run_single_generation(
    resolved_file: Path,
    project_root: Path,
    config: Any,
    output: Path | None,
    *,
    force: bool,
    if_changed: bool,
    auto_heal: bool | None,
    use_judge: bool | None,
    timeout: float | None,
    cov: bool,
    cov_source: str | None,
) -> None:
    test_path = resolve_test_path(
        resolved_file,
        project_root,
        output_dir=config.tests.output_dir,
        custom_output=output,
    )

    if test_path.is_file() and not force and not if_changed:
        if not is_ghost_managed_test(test_path):
            raise HandwrittenTestOverwriteError(test_path)
        if not click.confirm(f"Test file '{test_path}' already exists. Overwrite?", default=False):
            click.echo("Generation cancelled.")
            return

    pipeline = TestPipeline(config=config, project_root=project_root)
    listener = CliPipelineListener()
    result: PipelineResult = _run_async(
        pipeline.run(
            resolved_file,
            custom_output=output,
            force=force,
            force_generate=True,
            if_changed=if_changed,
            auto_heal=auto_heal,
            use_judge=use_judge,
            timeout_seconds=timeout,
            coverage=cov,
            cov_source=cov_source,
            listener=listener,
        )
    )

    if result.status == PipelineStatus.SKIPPED:
        click.echo(f"SKIPPED: {resolved_file.name} is unchanged (--if-changed).")
        return

    if result.passed:
        if result.attempts > 0:
            click.echo(f"PASS (healed after {result.attempts} attempt(s)): {result.test_file}")
        else:
            click.echo(f"PASS: {result.test_file}")
        if result.last_run and result.last_run.coverage_summary:
            click.echo(f"Coverage: {result.last_run.coverage_summary}")
    else:
        click.echo(f"FAIL: {result.test_file} ({result.status.value})")
        if result.last_run and result.last_run.coverage_summary:
            click.echo(f"Coverage: {result.last_run.coverage_summary}")
        if result.error_message:
            click.echo(f"  {result.error_message}")
        raise SystemExit(1)


@cli.command("generate")
@click.argument(
    "file",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
@click.option(
    "--all",
    "all_files",
    is_flag=True,
    default=False,
    help="Batch generate tests for all source files in the project.",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Custom output path for the generated test file.",
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    default=False,
    help="Overwrite existing test file without confirmation.",
)
@click.option(
    "--heal/--no-heal",
    "auto_heal",
    default=None,
    help="Enable or disable self-healing of failing tests.",
)
@click.option(
    "--judge/--no-judge",
    "use_judge",
    default=None,
    help="Enable or disable the Judge safety valve for logic errors.",
)
@click.option(
    "--if-changed",
    "if_changed",
    is_flag=True,
    default=False,
    help="Skip test generation if source file content has not changed since last processed run.",
)
@click.option(
    "--timeout",
    type=float,
    default=None,
    help="Execution timeout in seconds.",
)
@click.option(
    "--cov/--no-cov",
    default=False,
    help="Run tests with pytest-cov coverage reporting.",
)
@click.option(
    "--cov-source",
    type=str,
    default=None,
    help="Package or directory name to measure coverage for.",
)
def generate_cmd(
    file: Path | None = None,
    output: Path | None = None,
    *,
    all_files: bool = False,
    force: bool = False,
    if_changed: bool = False,
    auto_heal: bool | None = None,
    use_judge: bool | None = None,
    timeout: float | None = None,
    cov: bool = False,
    cov_source: str | None = None,
) -> None:
    """Generate, run, and optionally heal tests for Python source files."""
    if file is None and not all_files:
        click.echo("Error: Please provide a source FILE or pass --all.", err=True)
        raise SystemExit(1)

    target_path = (file or Path.cwd()).resolve()
    project_root = find_project_root(target_path) or (
        target_path if target_path.is_dir() else target_path.parent
    )
    config = load_config(project_root)

    if all_files or target_path.is_dir():
        search_dir = target_path if target_path.is_dir() else project_root
        _run_batch_generation(
            search_dir,
            project_root,
            config,
            force=force,
            if_changed=if_changed,
            auto_heal=auto_heal,
            use_judge=use_judge,
            timeout=timeout,
            cov=cov,
            cov_source=cov_source,
        )
    else:
        _run_single_generation(
            target_path,
            project_root,
            config,
            output,
            force=force,
            if_changed=if_changed,
            auto_heal=auto_heal,
            use_judge=use_judge,
            timeout=timeout,
            cov=cov,
            cov_source=cov_source,
        )


@cli.command("watch")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
@click.option(
    "--verbose",
    "-V",
    is_flag=True,
    default=False,
    help="Enable verbose output.",
)
@click.option(
    "--heal/--no-heal",
    "auto_heal",
    default=None,
    help="Enable or disable self-healing of failing tests.",
)
@click.option(
    "--judge/--no-judge",
    "use_judge",
    default=None,
    help="Enable or disable the Judge safety valve for logic errors.",
)
def watch_cmd(
    path: Path | None = None,
    *,
    verbose: bool = False,
    auto_heal: bool | None = None,
    use_judge: bool | None = None,
) -> None:
    """Watch Python source files for changes and automatically generate/heal tests."""
    target = (path or Path.cwd()).resolve()
    project_root = find_project_root(target) or target

    config_file = project_root / "ghost.toml"
    if not config_file.is_file():
        write_default_config(project_root)
        click.echo(f"Initialized ghost.toml at {config_file}")

    config = load_config(project_root)
    test_updates: dict[str, Any] = {}
    if auto_heal is not None:
        test_updates["auto_heal"] = auto_heal
    if use_judge is not None:
        test_updates["use_judge"] = use_judge
    if test_updates:
        config = config.model_copy(update={"tests": config.tests.model_copy(update=test_updates)})

    click.echo(f"Ghost watching {project_root} (debounce: {config.watcher.debounce_seconds}s)")
    click.echo("Press Ctrl+C to stop.\n")

    listener = CliPipelineListener(verbose=verbose, show_results=True)

    async def _run() -> None:
        watcher = FileWatcher(
            project_root=project_root,
            config=config,
            listener=listener,
        )
        watcher.start()
        try:
            stop_event = asyncio.Event()
            await stop_event.wait()
        except (asyncio.CancelledError, KeyboardInterrupt):
            pass
        finally:
            await watcher.stop()

    try:
        _run_async(_run())
    except (KeyboardInterrupt, asyncio.CancelledError):
        click.echo("\nStopping watcher...")
        click.echo("Watcher stopped.")


@cli.command("start")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
@click.option(
    "--detach/--foreground",
    default=True,
    help="Run as a background daemon (default) or in foreground.",
)
def start_cmd(path: Path | None = None, *, detach: bool = True) -> None:
    """Start the Ghost watcher daemon."""
    target = (path or Path.cwd()).resolve()
    project_root = find_project_root(target) or target

    config_file = project_root / "ghost.toml"
    if not config_file.is_file():
        raise ProjectNotInitializedError(target)

    if not detach:
        ctx = click.get_current_context()
        ctx.invoke(watch_cmd, path=path)
        return

    try:
        pid = start_daemon(project_root)
    except RuntimeError as exc:
        click.echo(f"Error: {exc}", err=True)
        raise SystemExit(ExitCode.ERROR) from exc

    click.echo(f"Ghost daemon started (PID {pid}).")
    click.echo(f"Logs: {daemon_log_path(project_root)}")


@cli.command("stop")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
def stop_cmd(path: Path | None = None) -> None:
    """Stop the Ghost watcher daemon."""
    target = (path or Path.cwd()).resolve()
    project_root = find_project_root(target) or target

    stopped = stop_daemon(project_root)
    if stopped:
        click.echo("Ghost daemon stopped.")
    else:
        click.echo("No daemon is running.")


@cli.command("status")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
def status_cmd(path: Path | None = None) -> None:
    """Report whether the Ghost daemon is running."""
    target = (path or Path.cwd()).resolve()
    project_root = find_project_root(target) or target

    status = query_daemon_status(project_root)
    if status.running:
        click.echo(f"Ghost daemon is running (PID {status.pid}).")
    else:
        click.echo("Ghost daemon is not running.")

    if status.log_tail:
        click.echo("\nRecent log:")
        for line in status.log_tail:
            click.echo(f"  {line}")


@cli.command("logs")
@click.argument(
    "path",
    type=click.Path(path_type=Path, exists=False),
    required=False,
    default=None,
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Continuously follow the log file.",
)
@click.option(
    "--lines",
    "-n",
    type=int,
    default=20,
    help="Number of lines to show (default 20).",
)
def logs_cmd(
    path: Path | None = None,
    *,
    follow: bool = False,
    lines: int = 20,
) -> None:
    """Show Ghost daemon log output."""
    target = (path or Path.cwd()).resolve()
    project_root = find_project_root(target) or target

    log_file = daemon_log_path(project_root)
    if not log_file.is_file():
        click.echo("No log file found.")
        return

    tail = tail_log(project_root, lines=lines)
    for line in tail:
        click.echo(line)

    if not follow:
        return

    with contextlib.suppress(KeyboardInterrupt):
        follow_log_stream(project_root, lambda line: click.echo(line, nl=False))


@cli.command("history")
@click.argument("file", type=click.Path(path_type=Path), required=False, default=None)
def history_cmd(file: Path | None = None) -> None:
    """View self-healing history and snapshots."""
    target = (file or Path.cwd()).resolve()
    project_root = find_project_root(target) or (target if target.is_dir() else target.parent)
    tracker = HistoryTracker(project_root)

    if file is not None and not file.is_dir():
        records = tracker.get_history(target)
        if not records:
            click.echo(f"No healing history found for {file.name}.")
            return
        print_history_table(records, file.name)
    else:
        files = tracker.list_tracked_files()
        if not files:
            click.echo("No test generation or healing history recorded yet.")
            return
        click.echo(f"Files with healing history ({len(files)}):")
        for f in files:
            click.echo(f"  • {f}")


@cli.command("rollback")
@click.argument("file", type=click.Path(path_type=Path, exists=True))
@click.option(
    "--attempt",
    "-a",
    type=int,
    default=None,
    help="Snapshot attempt number to restore (default: attempt 0).",
)
def rollback_cmd(file: Path, attempt: int | None = None) -> None:
    """Roll back a test file to a previously snapshotted attempt."""
    resolved_file = file.resolve()
    project_root = (
        find_project_root(resolved_file)
        or find_project_root(Path.cwd())
        or (
            resolved_file.parent.parent
            if resolved_file.parent.name == "tests"
            else resolved_file.parent
        )
    )
    config = load_config(project_root)
    tracker = HistoryTracker(project_root)

    if is_test_file(resolved_file, project_root, config.tests.output_dir):
        test_path = resolved_file
        history_files = tracker.list_tracked_files()
        matching_source: Path | None = None
        for rel in history_files:
            candidate_source = project_root / rel
            expected_test = resolve_test_path(
                candidate_source, project_root, config.tests.output_dir
            )
            if expected_test.resolve() == test_path:
                matching_source = candidate_source
                break
        if matching_source is None:
            click.echo(
                f"Could not find source file history associated with test {file.name}.",
                err=True,
            )
            raise SystemExit(1)
        source_path = matching_source
    else:
        source_path = resolved_file
        test_path = resolve_test_path(source_path, project_root, config.tests.output_dir)

    try:
        tracker.rollback(source_path, test_path, attempt=attempt)
        target_label = f"attempt {attempt}" if attempt is not None else "original attempt 0"
        click.echo(f"Successfully rolled back {test_path.name} to {target_label}.")
    except Exception as err:
        click.echo(f"Rollback failed: {err}", err=True)
        raise SystemExit(1) from err


@cli.command("stats")
def stats_cmd() -> None:
    """View cumulative AI token usage and test generation metrics."""
    project_root = find_project_root(Path.cwd()) or Path.cwd()
    tracker = UsageTracker(project_root)
    usage = tracker.load_usage()
    print_usage_stats(usage)


def _system_exit_code(exc: SystemExit) -> int:
    """Normalise ``SystemExit.code``, which may be ``None`` or a non-integer."""
    if exc.code is None:
        return ExitCode.SUCCESS
    return int(exc.code) if isinstance(exc.code, int) else ExitCode.ERROR


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
        return ExitCode.INTERRUPTED
    except click.ClickException as exc:
        exc.show()
        return exc.exit_code
    except GhostError as exc:
        click.echo(f"Error: {exc}", err=True)
        return ExitCode.ERROR
    except SystemExit as exc:  # raised by commands that fail their own checks
        return _system_exit_code(exc)
    return ExitCode.SUCCESS


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
