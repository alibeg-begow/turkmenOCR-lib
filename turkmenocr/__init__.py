"""Local TurkmenOCR text recognition on NVIDIA GPUs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.2.0"

__all__ = ["TurkmenOCR", "TurkmenOCRError", "recognize", "__version__"]

if TYPE_CHECKING:
    from .diagnostics import TurkmenOCRError
    from .engine import TurkmenOCR, recognize


def __getattr__(name: str) -> Any:
    if name in ("TurkmenOCR", "recognize"):
        from . import engine

        return getattr(engine, name)
    if name == "TurkmenOCRError":
        from .diagnostics import TurkmenOCRError

        return TurkmenOCRError
    raise AttributeError(f"module 'turkmenocr' has no attribute {name!r}")
