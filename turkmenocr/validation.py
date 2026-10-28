"""Validation of the weights folder and image paths supplied by the user."""

from __future__ import annotations

import os
from pathlib import Path

from .diagnostics import TurkmenOCRError

UNSUPPORTED_DOCUMENTS = {
    ".pdf": "a PDF document",
    ".heic": "an HEIC photo (the default format of iPhone cameras)",
    ".heif": "an HEIF photo",
    ".svg": "an SVG vector drawing",
    ".docx": "a Word document",
}


def normalize_path(raw: str) -> Path:
    """Turn a path typed on the command line into an absolute path.

    Stray quotes are removed because Windows shells pass a trailing quote
    through when a quoted path ends in a backslash, for example
    ``-path="C:\\models\\"``.
    """
    cleaned = raw.strip().strip('"').strip("'").strip()
    return Path(os.path.expandvars(os.path.expanduser(cleaned))).resolve()


def validate_weights_dir(path: Path) -> Path:
    if not path.exists():
        raise TurkmenOCRError(
            "INPUT-001",
            "The weights folder does not exist.",
            f"The folder given with -path was interpreted as {path}, but nothing "
            "exists at that location. The path may contain a typing mistake, or "
            "the folder may have been moved, renamed or stored on a drive that is "
            "not connected.",
            "Open the folder that contains the TurkmenOCR .safetensors files in "
            "your file manager, copy its full location from the address bar, and "
            'pass it in quotes, for example -path="D:\\models\\TurkmenOCR".',
        )
    if not path.is_dir():
        hint = (
            f' Pass the folder that contains it instead: -path="{path.parent}".'
            if path.suffix.lower() == ".safetensors"
            else ""
        )
        raise TurkmenOCRError(
            "INPUT-002",
            "The weights path points to a file, not a folder.",
            f"The location {path} is a single file. The -path option expects the "
            "folder that holds the model's .safetensors files.",
            "Run the command again with -path set to the folder that contains the "
            "weight files." + hint,
        )
    try:
        next(iter(path.iterdir()), None)
    except PermissionError as exc:
        raise TurkmenOCRError(
            "INPUT-003",
            "The weights folder cannot be opened.",
            f"The folder {path} exists, but your user account is not allowed to "
            "look inside it.",
            "Give your user account read access to the folder (on Windows: "
            "right-click it, choose Properties, then Security; on Linux or macOS: "
            f'run chmod -R a+rX "{path}"), or copy the folder to a location you '
            "own, such as your Documents folder, and pass that copy with -path.",
        ) from exc
    return path


def validate_image(path: Path) -> Path:
    from PIL import Image

    if not path.exists():
        raise TurkmenOCRError(
            "INPUT-004",
            "The image file does not exist.",
            f"The image given with -img was interpreted as {path}, but no file "
            "exists at that location.",
            "Check the spelling of the file name, including its extension such as "
            ".png or .jpg, and pass the full path in quotes, for example "
            '-img="C:\\Users\\you\\Pictures\\page.png".',
        )
    if path.is_dir():
        raise TurkmenOCRError(
            "INPUT-005",
            "The image path points to a folder, not a file.",
            f"The location {path} is a folder. The -img option expects a single "
            "image file.",
            "Run the command again with -img set to one image file inside that "
            "folder, for example a .png or .jpg file.",
        )
    kind = UNSUPPORTED_DOCUMENTS.get(path.suffix.lower())
    if kind:
        raise TurkmenOCRError(
            "INPUT-006",
            "The file is not a supported image format.",
            f"The file {path.name} is {kind}. TurkmenOCR reads ordinary raster "
            "images such as PNG, JPEG, BMP, TIFF and WebP.",
            "Save or export the page as a PNG or JPEG image (for a PDF, export "
            "each page as an image or take a screenshot of it), then pass that "
            "image with -img.",
        )
    try:
        with Image.open(path) as image:
            image.load()
    except PermissionError as exc:
        raise TurkmenOCRError(
            "INPUT-003",
            "The image file cannot be opened.",
            f"Your user account does not have permission to read {path}.",
            "Copy the image to a folder you own, such as your Documents or "
            "Pictures folder, and pass that copy with -img.",
        ) from exc
    except Image.DecompressionBombError as exc:
        raise TurkmenOCRError(
            "INPUT-008",
            "The image is too large to open safely.",
            f"The image {path.name} has more pixels than the image library allows "
            "for a single picture, which protects the computer from running out "
            "of memory.",
            "Reduce the image to at most 8000 pixels on its longest side with any "
            "image editor, save it, and run the command again with the smaller image.",
        ) from exc
    except Exception as exc:  # PIL raises many unrelated types for bad input.
        raise TurkmenOCRError(
            "INPUT-006",
            "The image file could not be read.",
            f"The file {path} ({path.stat().st_size:,} bytes) could not be decoded "
            "as an image. It may be damaged, only partly downloaded, or in a "
            "format other than its extension suggests.",
            "Open the file in an image viewer to confirm it displays correctly. "
            "If it does, save it again as PNG or JPEG and pass the new file with "
            "-img. If it does not, download or scan the image again.",
        ) from exc
    return path
