"""Ghost's rich terminal presentation and formatting layer.

Owns: terminal styling, banners, panels, syntax-highlighted code output,
rich status reporting, and the ``RichPipelineListener`` progress reporter.

Does NOT: contain business logic, error decision-making (``errors.py`` / ``cli.py``),
or test pipeline orchestration (``pipeline.py``).

Stage 12 replaces custom ANSI escape codes with modern ``rich`` components:
Console, Panel, Syntax, and Table, giving automatic terminal capability detection
and consistent styling.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from ghost import __version__
from ghost.history import AttemptRecord, UsageRecord
from ghost.pipeline import (
    PipelineEvent,
    PipelineListener,
    PipelineResult,
    PipelineStatus,
)

if TYPE_CHECKING:
    from ghost.providers import ModelConfig

__all__ = [
    "BANNER",
    "RichPipelineListener",
    "get_console",
    "print_banner",
    "print_batch_summary",
    "print_error",
    "print_history_table",
    "print_info",
    "print_panel",
    "print_providers_table",
    "print_success",
    "print_syntax",
    "print_usage_stats",
    "print_warning",
]

BANNER: Final[str] = rf"""[bold cyan]
   ____ _               _
  / ___| |__   ___  ___| |_
 | |  _| '_ \ / _ \/ __| __|
 | |_| | | | | (_) \__ \ |_
  \____|_| |_|\___/|___/\__|  v{__version__}
[/bold cyan][dim]Self-healing test generation for Python[/dim]
"""


@functools.cache
def get_console() -> Console:
    """Return the shared default :class:`rich.console.Console` instance."""
    return Console()


def print_banner(c: Console | None = None) -> None:
    """Print the Ghost ASCII mini-banner."""
    out = c or get_console()
    out.print(BANNER.strip())


def print_success(message: str, *, c: Console | None = None) -> None:
    """Print a success message prefixed with a green checkmark."""
    out = c or get_console()
    out.print(f"[bold green]✔[/bold green] {message}")


def print_error(message: str, *, c: Console | None = None) -> None:
    """Print an error message prefixed with a red cross."""
    out = c or get_console()
    out.print(f"[bold red]✖[/bold red] [red]{message}[/red]")


def print_warning(message: str, *, c: Console | None = None) -> None:
    """Print a warning message prefixed with a yellow warning symbol."""
    out = c or get_console()
    out.print(f"[bold yellow]▲[/bold yellow] [yellow]{message}[/yellow]")


def print_info(message: str, *, c: Console | None = None) -> None:
    """Print an informational message prefixed with a cyan symbol."""
    out = c or get_console()
    out.print(f"[bold cyan]*[/bold cyan] {message}")


def print_panel(content: str, title: str | None = None, *, c: Console | None = None) -> None:
    """Print *content* enclosed in a styled Rich panel."""
    out = c or get_console()
    out.print(Panel(content, title=title, border_style="cyan"))


def print_syntax(
    code: str, language: str = "python", *, line_numbers: bool = True, c: Console | None = None
) -> None:
    """Print *code* with syntax highlighting."""
    out = c or get_console()
    syntax = Syntax(code.strip(), language, theme="monokai", line_numbers=line_numbers)
    out.print(syntax)


def print_providers_table(
    availability: dict[str, bool],
    popular_models: dict[str, ModelConfig],
    *,
    c: Console | None = None,
) -> None:
    """Render supported AI providers and popular models as rich tables."""
    out = c or get_console()

    prov_table = Table(title="Supported Providers", border_style="blue")
    prov_table.add_column("Provider", style="bold")
    prov_table.add_column("Type")
    prov_table.add_column("Status")

    local_providers = {"ollama", "lmstudio"}
    for name, is_avail in sorted(availability.items()):
        ptype = "Local" if name in local_providers else "Cloud"
        status = (
            "[green]available (configured)[/green]" if is_avail else "[dim]not configured[/dim]"
        )
        prov_table.add_row(name, ptype, status)

    out.print(prov_table)

    model_table = Table(title="Popular Models", border_style="magenta")
    model_table.add_column("Model ID", style="bold")
    model_table.add_column("Provider")
    model_table.add_column("Description")

    for model_id, model_cfg in popular_models.items():
        model_table.add_row(model_id, model_cfg.provider, model_cfg.description)

    out.print(model_table)


class RichPipelineListener(PipelineListener):
    """PipelineListener using Rich styling for terminal progress updates."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        verbose: bool = False,
        show_results: bool = True,
    ) -> None:
        self.console = console or get_console()
        self.verbose = verbose
        self.show_results = show_results

    def _handle_result(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        if event == PipelineEvent.PASSED and self.show_results:
            test_file = data.get("test_file", "test")
            attempt = data.get("attempt", 0)
            if attempt > 0:
                self.console.print(
                    f"[bold green]✔ PASS[/bold green] "
                    f"[dim](healed after {attempt} attempt(s)):[/dim] "
                    f"[green]{test_file}[/green]"
                )
            else:
                self.console.print(f"[bold green]✔ PASS[/bold green]: [green]{test_file}[/green]")
        elif event == PipelineEvent.FAILED and self.show_results:
            test_file = data.get("test_file", "test")
            self.console.print(f"[bold red]✖ FAIL[/bold red]: [red]{test_file}[/red]")
        elif event == PipelineEvent.SKIPPED and self.verbose:
            src = data.get("source_file", "")
            src_name = Path(src).name if src else "source"
            self.console.print(f"[dim]SKIPPED: {src_name} (unchanged)[/dim]")

    @override
    def on_event(self, event: PipelineEvent, data: dict[str, Any]) -> None:
        src = data.get("source_file", "")
        src_name = Path(src).name if src else "source"

        if event == PipelineEvent.GENERATING:
            self.console.print(f"[cyan]⚡ Generating tests for [bold]{src_name}[/bold]...[/cyan]")
        elif event == PipelineEvent.RUNNING:
            attempt = data.get("attempt", 0)
            if attempt > 0:
                self.console.print(f"[blue]▶ Re-running test (attempt {attempt})...[/blue]")
            else:
                self.console.print("[blue]▶ Running tests...[/blue]")
        elif event == PipelineEvent.HEALING:
            attempt = data.get("attempt", 1)
            cls_name = data.get("classification", "UNKNOWN")
            self.console.print(
                f"[yellow]↻ Healing failure ({cls_name}, attempt {attempt})...[/yellow]"
            )
        elif event == PipelineEvent.JUDGING:
            self.console.print("[magenta]⚖ Assertion failure: consulting Judge...[/magenta]")
        elif event == PipelineEvent.JUDGE_RESULT:
            outcome = data.get("outcome", "UNKNOWN")
            self.console.print(
                f"[magenta]⚖ Judge evaluated failure as: [bold]{outcome}[/bold][/magenta]"
            )
        else:
            self._handle_result(event, data)


def print_batch_summary(results: list[PipelineResult], *, c: Console | None = None) -> None:
    """Print a summary table of batch generation outcomes."""
    out = c or get_console()
    table = Table(title="Batch Generation Summary", border_style="cyan")
    table.add_column("Source File", style="cyan", no_wrap=True)
    table.add_column("Test File", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Attempts", justify="right")
    table.add_column("Details", style="dim")

    passed_cnt = sum(
        1 for r in results if r.status in (PipelineStatus.PASSED, PipelineStatus.HEALED)
    )
    healed_cnt = sum(1 for r in results if r.status == PipelineStatus.HEALED)
    failed_cnt = sum(
        1
        for r in results
        if r.status in (PipelineStatus.FAILED, PipelineStatus.BUG_IN_CODE, PipelineStatus.UNCLEAR)
    )
    skipped_cnt = sum(1 for r in results if r.status == PipelineStatus.SKIPPED)

    for r in results:
        if r.status == PipelineStatus.PASSED:
            status_style = "[bold green]PASS[/bold green]"
        elif r.status == PipelineStatus.HEALED:
            status_style = "[bold cyan]HEALED[/bold cyan]"
        elif r.status == PipelineStatus.SKIPPED:
            status_style = "[dim]SKIPPED[/dim]"
        else:
            status_style = f"[bold red]{r.status.value}[/bold red]"

        detail = r.error_message or ""
        if r.last_run and r.last_run.coverage_summary:
            detail = f"Cov: {r.last_run.coverage_summary}"

        table.add_row(
            r.source_file.name,
            r.test_file.name,
            status_style,
            str(r.attempts),
            detail,
        )

    out.print(table)
    summary_text = (
        f"[bold]Total:[/] {len(results)} | "
        f"[bold green]Passed:[/] {passed_cnt} "
        f"[dim]([cyan]Healed:[/] {healed_cnt})[/dim] | "
        f"[bold red]Failed:[/] {failed_cnt} | "
        f"[dim]Skipped:[/] {skipped_cnt}"
    )
    out.print(Panel(summary_text, border_style="blue"))


def print_history_table(
    records: list[AttemptRecord], source_file: str, *, c: Console | None = None
) -> None:
    """Print a styled table of heal history attempts for *source_file*."""
    out = c or get_console()
    table = Table(title=f"History for {source_file}", border_style="cyan")
    table.add_column("Attempt", justify="right")
    table.add_column("Timestamp", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Classification", style="yellow")
    table.add_column("Snapshot File", style="dim")

    for rec in records:
        status_styled = (
            f"[green]{rec.status}[/green]"
            if rec.status in ("passed", "healed", "generated")
            else f"[red]{rec.status}[/red]"
        )
        table.add_row(
            str(rec.attempt),
            rec.timestamp[:19].replace("T", " "),
            status_styled,
            rec.classification or "-",
            rec.snapshot_file,
        )

    out.print(table)


def print_usage_stats(usage: UsageRecord, *, c: Console | None = None) -> None:
    """Print cumulative LLM token usage and request statistics."""
    out = c or get_console()
    table = Table(title="AI Usage & Token Statistics", border_style="cyan")
    table.add_column("Metric", style="cyan bold")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Total Requests", f"{usage.total_requests:,}")
    table.add_row("Prompt Tokens", f"{usage.prompt_tokens:,}")
    table.add_row("Completion Tokens", f"{usage.completion_tokens:,}")
    table.add_row("Total Tokens", f"{usage.total_tokens:,}")
    table.add_row("Healing Attempts", f"{usage.heal_attempts:,}")
    table.add_row("Successfully Healed Tests", f"{usage.healed_tests:,}")

    out.print(table)
