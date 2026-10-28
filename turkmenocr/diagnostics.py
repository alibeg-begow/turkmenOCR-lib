"""Plain-text diagnostics written to standard error.

Every failure the library can anticipate is raised as a :class:`TurkmenOCRError`.
The command-line interface renders it with :func:`format_error` using a fixed
three-paragraph layout: a one-line summary, an explanation of what was
detected, and instructions for resolving the problem.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Optional, Sequence, TextIO

WRAP_WIDTH = 79

EXIT_INPUT = 2
EXIT_WEIGHTS = 3
EXIT_ENVIRONMENT = 4
EXIT_WORKSPACE = 5
EXIT_RUNTIME = 6
EXIT_INTERNAL = 70
EXIT_INTERRUPTED = 130

_EXIT_CODES = {
    "INPUT": EXIT_INPUT,
    "WEIGHTS": EXIT_WEIGHTS,
    "MODEL": EXIT_WEIGHTS,
    "ENV": EXIT_ENVIRONMENT,
    "WORKSPACE": EXIT_WORKSPACE,
    "RUNTIME": EXIT_RUNTIME,
}


class TurkmenOCRError(Exception):
    """An anticipated failure carrying a user-facing explanation.

    Attributes:
        code: Category code such as ``ENV-003``. The prefix before the first
            hyphen selects the process exit status.
        summary: One sentence describing the failure.
        explanation: What was detected, with concrete figures and paths.
        resolution: Plain-language instructions for fixing the problem.
        steps: Optional numbered steps appended to the resolution paragraph.
    """

    def __init__(
        self,
        code: str,
        summary: str,
        explanation: str,
        resolution: str,
        steps: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(f"{code}: {summary}")
        self.code = code
        self.summary = summary
        self.explanation = explanation
        self.resolution = resolution
        self.steps = list(steps or [])

    @property
    def exit_code(self) -> int:
        return _EXIT_CODES.get(self.code.split("-", 1)[0], EXIT_INTERNAL)


def _wrap(text: str, initial: str = "", subsequent: str = "") -> str:
    return textwrap.fill(
        " ".join(text.split()),
        width=WRAP_WIDTH,
        initial_indent=initial,
        subsequent_indent=subsequent,
        break_long_words=False,
        break_on_hyphens=False,
    )


def format_error(error: TurkmenOCRError) -> str:
    """Render an error in the three-paragraph diagnostic layout."""
    resolution = _wrap(f"How to resolve this issue: {error.resolution}")
    if error.steps:
        numbered = [
            _wrap(step, initial=f"{index}. ", subsequent=" " * (len(str(index)) + 2))
            for index, step in enumerate(error.steps, start=1)
        ]
        resolution = resolution + "\n" + "\n".join(numbered)
    paragraphs = [
        _wrap(f"Error [{error.code}]: {error.summary}"),
        _wrap(error.explanation),
        resolution,
    ]
    return "\n\n".join(paragraphs) + "\n"


def format_size(num_bytes: float) -> str:
    """Format a byte count using binary units, for example ``11.6 GB``."""
    units = ["bytes", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "bytes":
                return f"{int(value)} bytes"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


class Reporter:
    """Writes progress notices and errors to standard error."""

    def __init__(self, quiet: bool = False, stream: Optional[TextIO] = None) -> None:
        self.quiet = quiet
        self._stream = stream

    @property
    def stream(self) -> TextIO:
        return self._stream if self._stream is not None else sys.stderr

    def info(self, message: str) -> None:
        if not self.quiet:
            self.stream.write(_wrap(message) + "\n")
            self.stream.flush()

    def warning(self, message: str) -> None:
        self.stream.write(_wrap(f"Warning: {message}") + "\n")
        self.stream.flush()

    def error(self, error: TurkmenOCRError) -> None:
        self.stream.write(format_error(error))
        self.stream.flush()


SILENT = Reporter(quiet=True)
