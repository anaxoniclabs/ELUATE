# SPDX-License-Identifier: MIT
"""
Eluate ASCII art and decorative elements.
"""

from rich.align import Align
from rich.panel import Panel
from rich.text import Text

from .theme import Colors

# Main logo — ANSI Shadow figlet font
ELUATE_LOGO = """
███████╗██╗     ██╗   ██╗ █████╗ ████████╗███████╗
██╔════╝██║     ██║   ██║██╔══██╗╚══██╔══╝██╔════╝
█████╗  ██║     ██║   ██║███████║   ██║   █████╗
██╔══╝  ██║     ██║   ██║██╔══██║   ██║   ██╔══╝
███████╗███████╗╚██████╔╝██║  ██║   ██║   ███████╗
╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""


def get_styled_logo() -> Text:
    """Return the ELUATE logo with a white-to-gray fade."""
    text = Text()
    lines = ELUATE_LOGO.strip("\n").split("\n")

    gradient = [
        Colors.PRIMARY,
        Colors.PRIMARY,
        Colors.PRIMARY_LIGHT,
        Colors.PRIMARY_LIGHT,
        Colors.PRIMARY_DARK,
        Colors.PRIMARY_DARK,
    ]

    for i, line in enumerate(lines):
        color = gradient[min(i, len(gradient) - 1)]
        text.append(line + "\n", style=color)

    return text


def get_header_panel(console_width: int = 80) -> Panel:
    """Generate the complete header with logo only, centered."""
    logo = get_styled_logo()

    return Panel(
        Align.center(logo),
        border_style=Colors.PRIMARY,
        padding=(1, 2),
    )


# Smaller logo — thin box-drawing style
ELUATE_LOGO_SMALL = """
┏━╸╻  ╻ ╻┏━┓┏┳┓┏━╸
┣╸ ┃  ┃ ┃┣━┫ ┃ ┣╸
┗━╸┗━╸┗━┛╹ ╹ ╹ ┗━╸
"""


def get_small_logo() -> Text:
    """Return the small Eluate logo with styling."""
    text = Text()
    for line in ELUATE_LOGO_SMALL.strip().split("\n"):
        text.append(line + "\n", style=Colors.PRIMARY)
    return text


# Spinner frames for indeterminate progress
SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
