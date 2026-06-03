# SPDX-License-Identifier: MIT
"""
Reusable UI components for Eluate CLI.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .theme import ELUATE_THEME, BoxChars, Colors


def create_console() -> Console:
    """Create a themed Rich console."""
    return Console(theme=ELUATE_THEME, highlight=False)


def info_panel(title: str, content: dict[str, str], style: str = Colors.PRIMARY) -> Panel:
    """
    Create an info panel with key-value pairs.

    Args:
        title: Panel title
        content: Dictionary of label -> value pairs
        style: Border color

    Example:
        info_panel("Video Info", {
            "📺 Title": "Wildlife Documentary",
            "⏱ Duration": "1:42:35",
            "📦 Size": "~2.1 GB"
        })
    """
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()

    for label, value in content.items():
        table.add_row(label, value)

    return Panel(
        table,
        title=f"[bold]{title}[/bold]",
        border_style=style,
        padding=(1, 2),
    )


def status_panel(
    stages: list[tuple[str, str, str]], current_stage: int, stage_progress: float = 0.0
) -> Panel:
    """
    Create a status panel showing pipeline stages.

    Args:
        stages: List of (id, name, emoji) tuples
        current_stage: Index of current stage (0-based)
        stage_progress: Progress within current stage (0.0 to 1.0)
    """
    lines = []

    for i, (stage_id, stage_name, emoji) in enumerate(stages):
        if i < current_stage:
            # Completed
            bullet = f"[{Colors.SUCCESS}]{BoxChars.BULLET_COMPLETE}[/]"
            text_style = Colors.TEXT_MUTED
        elif i == current_stage:
            # Active
            bullet = f"[{Colors.PRIMARY}]{BoxChars.BULLET_ACTIVE}[/]"
            text_style = Colors.TEXT_PRIMARY
        else:
            # Pending
            bullet = f"[{Colors.TEXT_MUTED}]{BoxChars.BULLET_PENDING}[/]"
            text_style = Colors.TEXT_MUTED

        lines.append(f"  {bullet} [{text_style}]{stage_name}[/]")

    content = "\n".join(lines)

    return Panel(
        content,
        title="[bold]Progress[/bold]",
        border_style=Colors.PRIMARY,
        padding=(1, 2),
    )


def input_panel(prompt: str, value: str = "") -> Panel:
    """
    Create a styled input prompt panel.

    Args:
        prompt: The prompt text
        value: Current input value (for display)
    """
    cursor = "█" if not value else ""
    content = f"{prompt}\n[bold white]> {value}{cursor}[/bold white]"

    return Panel(
        content,
        border_style=Colors.PRIMARY,
        padding=(1, 2),
    )


def success_panel(output_path: str, duration: str, processing_time: str) -> Panel:
    """
    Create the completion success panel.
    """
    content = Text()
    content.append("Saved to:\n", style="bold")
    content.append(f"{output_path}\n\n", style=Colors.PRIMARY_LIGHT)
    content.append(f"Original: {duration}  •  ", style=Colors.TEXT_SECONDARY)
    content.append(f"Processed in: {processing_time}", style=Colors.TEXT_SECONDARY)

    return Panel(
        content,
        title=f"[bold {Colors.SUCCESS}]✓ COMPLETE[/]",
        border_style=Colors.SUCCESS,
        padding=(1, 2),
    )


def error_panel(message: str, details: Optional[str] = None) -> Panel:
    """
    Create an error panel.
    """
    content = Text()
    content.append(f"{message}\n", style="bold")
    if details:
        content.append(f"\n{details}", style=Colors.TEXT_MUTED)

    return Panel(
        content,
        title=f"[bold {Colors.ERROR}]✗ ERROR[/]",
        border_style=Colors.ERROR,
        padding=(1, 2),
    )


def warning_panel(message: str, details: Optional[str] = None) -> Panel:
    """
    Create a warning panel.
    """
    content = Text()
    content.append(f"{message}\n", style="bold")
    if details:
        content.append(f"\n{details}", style=Colors.TEXT_MUTED)

    return Panel(
        content,
        title=f"[bold {Colors.WARNING}]⚠ WARNING[/]",
        border_style=Colors.WARNING,
        padding=(1, 2),
    )


def divider(width: int = 60, char: str = "─") -> str:
    """Create a horizontal divider."""
    return f"[{Colors.TEXT_MUTED}]{char * width}[/]"
