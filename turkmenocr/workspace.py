"""Non-destructive bridging of user weights with the bundled model files.

``transformers`` loads a model from a single folder that holds both the
configuration and the weights. The user's folder normally holds only the
weights and may be read-only, so a separate workspace folder is assembled
from the bundled configuration, tokenizer and modeling files plus links to
the user's ``.safetensors`` files. The following strategies are tried in
order until one succeeds:

1. A workspace in the per-user cache folder with symbolic links to the weights.
2. The same workspace with hard links (Windows without symlink rights).
3. A ``turkmenocr_workspace`` subfolder inside the weights folder with hard
   links, for weights stored on a different drive than the cache folder.
4. Copying only the missing support files directly into the weights folder.

Workspaces are rebuilt on every run and removed afterwards. Their folder
names are stable per weights folder, which keeps the dynamic-module cache of
``transformers`` from growing on every run.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional

from .diagnostics import Reporter, TurkmenOCRError
from .inspector import WeightsReport

INDEX_NAME = "model.safetensors.index.json"
MARKER_NAME = ".turkmenocr-workspace"
LOCAL_WORKSPACE_NAME = "turkmenocr_workspace"


@dataclass
class Workspace:
    """A folder that transformers can load the complete model from."""

    path: Path
    mode: str
    created: List[Path] = field(default_factory=list)

    def cleanup(self) -> None:
        """Remove everything this workspace added, leaving user files intact."""
        if self.mode == "in-place":
            for created in reversed(self.created):
                try:
                    created.unlink()
                except OSError:
                    pass
            self.created.clear()
        elif (self.path / MARKER_NAME).exists():
            shutil.rmtree(self.path, ignore_errors=True)

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()


def cache_root() -> Path:
    """Per-user cache folder, overridable with TURKMENOCR_CACHE_DIR."""
    override = os.environ.get("TURKMENOCR_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "turkmenocr"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "turkmenocr"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "turkmenocr"


def bundled_assets(asset_dir: str) -> Dict[str, bytes]:
    """Contents of the bundled support files for a model variant."""
    folder = resources.files("turkmenocr").joinpath("assets").joinpath(asset_dir)
    files: Dict[str, bytes] = {}
    for entry in folder.iterdir():
        name = entry.name
        if entry.is_file() and name != INDEX_NAME and not name.endswith(".pyc"):
            files[name] = entry.read_bytes()
    return files


def build_index(report: WeightsReport) -> bytes:
    """A shard index that maps every tensor to the user's actual file names."""
    index = {
        "metadata": {"total_size": report.memory_required(bytes_per_parameter=2)},
        "weight_map": dict(sorted(report.weight_map().items())),
    }
    return (json.dumps(index, indent=2) + "\n").encode("utf-8")


def workspace_name(directory: Path) -> str:
    key = str(directory).lower() if os.name == "nt" else str(directory)
    return "ws_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def prepare_workspace(report: WeightsReport, reporter: Optional[Reporter] = None) -> Workspace:
    """Assemble a loadable model folder for the inspected weights."""
    reporter = reporter or Reporter(quiet=True)
    assets = bundled_assets(report.variant.asset_dir)
    assets[INDEX_NAME] = build_index(report)

    cache_folder = cache_root() / "workspaces" / workspace_name(report.directory)
    attempts = [
        (cache_folder, os.symlink, "symlink"),
        (cache_folder, os.link, "hardlink"),
        (report.directory / LOCAL_WORKSPACE_NAME, os.link, "local-hardlink"),
    ]
    for folder, link, mode in attempts:
        workspace = _try_linked_workspace(folder, report, assets, link, mode)
        if workspace is not None:
            return workspace
    return _in_place_workspace(report, assets, reporter)


def _try_linked_workspace(
    folder: Path,
    report: WeightsReport,
    assets: Dict[str, bytes],
    link: Callable[[Path, Path], None],
    mode: str,
) -> Optional[Workspace]:
    try:
        _reset_folder(folder)
        (folder / MARKER_NAME).write_text("Managed by turkmenOCR-lib.\n", encoding="utf-8")
        for name, content in assets.items():
            (folder / name).write_bytes(content)
        for weight_file in report.files:
            link(weight_file.path, folder / weight_file.path.name)
    except OSError:
        if (folder / MARKER_NAME).exists():
            shutil.rmtree(folder, ignore_errors=True)
        return None
    return Workspace(path=folder, mode=mode)


def _reset_folder(folder: Path) -> None:
    if folder.exists():
        if not (folder / MARKER_NAME).exists():
            raise OSError(f"{folder} exists and is not a turkmenOCR-lib workspace")
        shutil.rmtree(folder)
    folder.mkdir(parents=True)


def _in_place_workspace(report: WeightsReport, assets: Dict[str, bytes], reporter: Reporter) -> Workspace:
    directory = report.directory
    if not _is_writable(directory):
        raise _not_writable_error(directory)

    existing_index = directory / INDEX_NAME
    if existing_index.exists():
        _check_existing_index(existing_index, report)

    workspace = Workspace(path=directory, mode="in-place")
    kept: List[str] = []
    try:
        for name, content in assets.items():
            target = directory / name
            if target.exists():
                if name != INDEX_NAME and target.read_bytes() != content:
                    kept.append(name)
                continue
            target.write_bytes(content)
            workspace.created.append(target)
    except OSError as exc:
        workspace.cleanup()
        raise _not_writable_error(directory) from exc
    if kept:
        reporter.warning(
            f"The weights folder already contains its own {', '.join(sorted(kept))}. "
            "Those files are used instead of the versions bundled with turkmenOCR-lib."
        )
    return workspace


def _check_existing_index(path: Path, report: WeightsReport) -> None:
    try:
        weight_map = json.loads(path.read_text(encoding="utf-8")).get("weight_map", {})
    except (OSError, ValueError, AttributeError):
        weight_map = None
    if weight_map == report.weight_map():
        return
    raise TurkmenOCRError(
        "WORKSPACE-002",
        f"The file {INDEX_NAME} in the weights folder does not match the weight files.",
        f"The folder {report.directory} contains a file named {INDEX_NAME}, which "
        "tells the loader which weight file holds each part of the model. It "
        "lists files or parts that do not match the .safetensors files actually "
        "present, so the model would fail to load. turkmenOCR-lib does not "
        "overwrite files it did not create.",
        f"Rename or move {INDEX_NAME} out of the folder (turkmenOCR-lib creates "
        "a correct one automatically), then run the command again.",
    )


def _is_writable(directory: Path) -> bool:
    probe = directory / f".turkmenocr-write-test-{os.getpid()}"
    try:
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def _not_writable_error(directory: Path) -> TurkmenOCRError:
    if os.name == "nt":
        steps = [
            "Recommended: turn on Developer Mode (Settings, then Privacy and "
            "security, then For developers, then Developer Mode). This lets "
            "turkmenOCR-lib link to the weights without changing the folder.",
            f"Alternatively, right-click the folder {directory}, choose Properties, "
            "open the Security tab, select your user name, click Edit and tick "
            "Allow next to Write.",
            "Alternatively, copy the weights folder into your Documents folder and "
            "pass that copy with -path.",
        ]
    else:
        steps = [
            f'Give your user write access to the folder: chmod u+w "{directory}"',
            "Alternatively, copy the weights folder into your home folder and pass "
            "that copy with -path.",
        ]
    return TurkmenOCRError(
        "WORKSPACE-001",
        "turkmenOCR-lib could not combine its model files with your weights.",
        f"The weights in {directory} must be loaded together with configuration "
        "files that ship with turkmenOCR-lib. Linking to the weights from the "
        f"cache folder {cache_root()} was not permitted, and writing the "
        "configuration files next to the weights failed because the folder is "
        "read-only for your user account.",
        "Use one of the options below, then run the same command again.",
        steps=steps,
    )
