# SPDX-License-Identifier: MIT
"""
Eluate color theme and styling constants.
Black and white palette.
"""

from rich.style import Style
from rich.theme import Theme


class Colors:
    """Black and white UI palette for Eluate."""

    # Brand colors
    PRIMARY = "#FFFFFF"  # White
    PRIMARY_LIGHT = "#DDDDDD"  # Light gray (highlights)
    PRIMARY_DARK = "#AAAAAA"  # Mid gray (emphasis)

    # Status colors
    SUCCESS = "#FFFFFF"  # White (completion)
    ERROR = "#FF6B6B"  # Soft red (errors — kept for legibility)
    WARNING = "#CCCCCC"  # Light gray (warnings)
    INFO = "#AAAAAA"  # Mid gray (informational)

    # Background colors
    BG_DARK = "#1A1A1A"  # Dark background
    BG_PANEL = "#2D2D2D"  # Dark gray panel

    # Text colors
    TEXT_PRIMARY = "#FFFFFF"  # White
    TEXT_SECONDARY = "#AAAAAA"  # Mid gray
    TEXT_MUTED = "#666666"  # Dark gray


# Rich theme for console
ELUATE_THEME = Theme(
    {
        # Brand colors
        "eluate.primary": Style(color=Colors.PRIMARY, bold=True),
        "eluate.primary.light": Style(color=Colors.PRIMARY_LIGHT),
        "eluate.primary.dark": Style(color=Colors.PRIMARY_DARK),
        # Status colors
        "eluate.success": Style(color=Colors.SUCCESS, bold=True),
        "eluate.error": Style(color=Colors.ERROR, bold=True),
        "eluate.warning": Style(color=Colors.WARNING),
        "eluate.info": Style(color=Colors.INFO),
        # Text styles
        "eluate.text": Style(color=Colors.TEXT_PRIMARY),
        "eluate.text.secondary": Style(color=Colors.TEXT_SECONDARY),
        "eluate.text.muted": Style(color=Colors.TEXT_MUTED),
        # Progress bar styles
        "bar.back": Style(color=Colors.BG_PANEL),
        "bar.complete": Style(color=Colors.PRIMARY),
        "bar.finished": Style(color=Colors.PRIMARY_LIGHT),
        "bar.pulse": Style(color=Colors.PRIMARY_LIGHT),
        # Override Rich's default magenta/cyan progress colors so the
        # bar stays monochrome regardless of terminal theme.
        "progress.description": Style(color=Colors.TEXT_PRIMARY),
        "progress.percentage": Style(color=Colors.PRIMARY),
        "progress.remaining": Style(color=Colors.TEXT_SECONDARY),
        "progress.elapsed": Style(color=Colors.TEXT_SECONDARY),
        "progress.data.speed": Style(color=Colors.TEXT_SECONDARY),
        "progress.download": Style(color=Colors.TEXT_SECONDARY),
        "progress.filesize": Style(color=Colors.TEXT_SECONDARY),
        "progress.spinner": Style(color=Colors.PRIMARY),
        # Panel/box styles
        "eluate.panel.border": Style(color=Colors.PRIMARY),
        "eluate.panel.title": Style(color=Colors.PRIMARY, bold=True),
    }
)


class BoxChars:
    """Unicode box drawing characters in Eluate style."""

    TOP_LEFT = "╭"
    TOP_RIGHT = "╮"
    BOTTOM_LEFT = "╰"
    BOTTOM_RIGHT = "╯"
    HORIZONTAL = "─"
    VERTICAL = "│"

    # Status indicators
    BULLET_ACTIVE = "●"
    BULLET_PENDING = "○"
    BULLET_COMPLETE = "✓"
    BULLET_ERROR = "✗"


class Stages:
    """Pipeline stage definitions with display info."""

    EXTRACT = ("extract", "Extracting audio", "🎵")
    LOAD_MODEL = ("load_model", "Loading model", "📥")
    SEPARATE = ("separate", "Removing music", "🔇")
    COMPILE = ("compile", "Rebuilding video", "🎬")

    ALL = [EXTRACT, LOAD_MODEL, SEPARATE, COMPILE]

    @classmethod
    def get_by_id(cls, stage_id: str) -> tuple[str, str, str] | None:
        """Get stage tuple by ID."""
        for stage in cls.ALL:
            if stage[0] == stage_id:
                return stage
        return None
