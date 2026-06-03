# SPDX-License-Identifier: MIT
"""
Unified progress display for Eluate pipeline.
Shows all stages with a single, clean progress interface.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from .theme import BoxChars, Colors, Stages


class StageStatus(Enum):
    """Status of a pipeline stage."""

    PENDING = auto()
    ACTIVE = auto()
    COMPLETE = auto()
    ERROR = auto()


@dataclass
class StageInfo:
    """Information about a pipeline stage."""

    id: str
    name: str
    emoji: str
    status: StageStatus = StageStatus.PENDING
    progress: float = 0.0
    detail: str = ""


class EluateProgress:
    """
    Unified progress display manager.

    Handles all three pipeline stages with a consistent UI.
    """

    def __init__(self, console: Console):
        self.console = console
        self.video_info: dict = {}

        # Initialize stages
        self.stages = [StageInfo(id=s[0], name=s[1], emoji=s[2]) for s in Stages.ALL]
        self.current_stage_idx = 0

        # Progress bar for current stage
        self.progress = Progress(
            SpinnerColumn(spinner_name="bouncingBall", style=Colors.PRIMARY),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=40,
                style=Colors.BG_PANEL,
                complete_style=Colors.PRIMARY,
                finished_style=Colors.PRIMARY_LIGHT,
            ),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console,
            expand=False,
        )

        self.task_id: Optional[TaskID] = None
        self.live: Optional[Live] = None

    def set_video_info(self, title: str, duration: str):
        """Set video information to display."""
        self.video_info = {
            "📺  Title": title,
            "⏱   Duration": duration,
        }

    def _build_stages_display(self) -> Table:
        """Build the stages status display."""
        table = Table.grid(padding=(0, 1))
        table.add_column(width=3)  # Bullet
        table.add_column()  # Stage name

        for i, stage in enumerate(self.stages):
            if stage.status == StageStatus.COMPLETE:
                bullet = f"[{Colors.SUCCESS}]{BoxChars.BULLET_COMPLETE}[/]"
                name_style = Colors.TEXT_MUTED
            elif stage.status == StageStatus.ACTIVE:
                bullet = f"[{Colors.PRIMARY}]{BoxChars.BULLET_ACTIVE}[/]"
                name_style = Colors.PRIMARY
            elif stage.status == StageStatus.ERROR:
                bullet = f"[{Colors.ERROR}]{BoxChars.BULLET_ERROR}[/]"
                name_style = Colors.ERROR
            else:
                bullet = f"[{Colors.TEXT_MUTED}]{BoxChars.BULLET_PENDING}[/]"
                name_style = Colors.TEXT_MUTED

            name = f"[{name_style}]{stage.name}[/]"
            table.add_row(bullet, name)

        return table

    def _build_display(self) -> Panel:
        """Build the complete progress display."""
        stages_display = self._build_stages_display()

        # Show the progress bar (with spinner) for the active stage even when
        # completed == 0 — the spinner conveys "working" during silent gaps
        # like model load before any progress tick arrives.
        current_stage = self.stages[self.current_stage_idx]
        content: Group | Table
        if current_stage.status == StageStatus.ACTIVE and self.task_id is not None:
            content = Group(stages_display, "", self.progress)
        else:
            content = stages_display

        return Panel(
            content,
            title="[bold]Progress[/bold]",
            border_style=Colors.PRIMARY,
            padding=(1, 2),
        )

    def start(self):
        """Start the live display."""
        # 20 Hz keeps the bouncingBall spinner fluid (50 ms < its 80 ms
        # native interval). Redraw of this small panel is negligible
        # alongside torch inference.
        self.live = Live(
            self._build_display(),
            console=self.console,
            refresh_per_second=20,
            transient=False,
        )
        self.live.start()

    def stop(self):
        """Stop the live display."""
        if self.live:
            self.live.stop()

    def _refresh(self):
        """Refresh the display."""
        if self.live:
            self.live.update(self._build_display())

    def begin_stage(self, stage_id: str, total: Optional[float] = None):
        """
        Begin a new stage.

        Args:
            stage_id: Stage identifier (extract, separate, compile)
            total: Total units for progress (percent for all stages)
        """
        # Find stage index
        for i, stage in enumerate(self.stages):
            if stage.id == stage_id:
                self.current_stage_idx = i
                stage.status = StageStatus.ACTIVE
                stage.detail = ""
                break

        # Remove previous task to prevent accumulation
        if self.task_id is not None:
            self.progress.remove_task(self.task_id)
            self.task_id = None

        # Create new progress task
        current_stage = self.stages[self.current_stage_idx]
        self.task_id = self.progress.add_task(
            description=current_stage.name,
            total=total or 100,
        )

        self._refresh()

    def update_progress(
        self,
        completed: Optional[float] = None,
        advance: Optional[float] = None,
        detail: str = "",
        total: Optional[float] = None,
    ):
        """
        Update current stage progress.

        Args:
            completed: Absolute completed amount
            advance: Amount to advance by
            detail: Optional detail text
            total: Update total
        """
        current_stage = self.stages[self.current_stage_idx]
        current_stage.detail = detail

        any_update = completed is not None or advance is not None or total is not None
        if self.task_id is not None and any_update:
            self.progress.update(
                self.task_id,
                completed=completed,
                advance=advance,
                total=total,
            )

        self._refresh()

    def complete_stage(self, stage_id: str):
        """Mark a stage as complete."""
        for stage in self.stages:
            if stage.id == stage_id:
                stage.status = StageStatus.COMPLETE
                stage.progress = 1.0
                break

        # Complete the progress task
        if self.task_id is not None:
            # Get task total using the internal tasks dict (task_id is not a list index)
            task = self.progress._tasks.get(self.task_id)
            if task:
                self.progress.update(self.task_id, completed=task.total)

        self._refresh()

    def error_stage(self, stage_id: str, message: str):
        """Mark a stage as failed."""
        for stage in self.stages:
            if stage.id == stage_id:
                stage.status = StageStatus.ERROR
                stage.detail = message
                break

        self._refresh()
