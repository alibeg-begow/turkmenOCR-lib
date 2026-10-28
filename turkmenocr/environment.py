"""Hardware inspection and runtime dependency management.

The checks run in an order that avoids wasted work: the NVIDIA driver and the
GPU's total memory are queried with ``nvidia-smi`` before anything large is
downloaded, PyTorch is examined in a child process so that a CPU-only build
never gets imported into this one, and only then are ``transformers`` and the
remaining packages aligned with the versions the model code was written for.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .diagnostics import Reporter, TurkmenOCRError, format_size

GIB = 1024 ** 3

TRANSFORMERS_TARGET = "4.46.0"
TRANSFORMERS_SERIES = "4.46."

# 8 GB cards report slightly less than 8 GiB, so the gate sits a little lower.
MIN_TOTAL_VRAM = int(7.5 * GIB)
ACTIVATION_RESERVE = int(1.0 * GIB)

TORCH_VERSION = "2.7.1"
TORCHVISION_VERSION = "0.22.1"
TORCH_PYTHON_RANGE = ((3, 9), (3, 13))

SUPPORT_PACKAGES: Dict[str, str] = {
    "accelerate": "accelerate>=0.26.0",
    "einops": "einops",
    "addict": "addict",
    "easydict": "easydict",
    "requests": "requests",
    "numpy": "numpy",
    "tqdm": "tqdm",
}

PROBE_MARKER = "TURKMENOCR_PROBE:"

_PROBE_SCRIPT = f"""
import json
result = {{"installed": False}}
try:
    import torch
except Exception as exc:
    result["error"] = repr(exc)
else:
    result["installed"] = True
    result["version"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    try:
        result["arch_list"] = torch.cuda.get_arch_list() if result["cuda_available"] else []
    except Exception:
        result["arch_list"] = []
    try:
        import torchvision
        from torchvision import transforms
        result["torchvision"] = torchvision.__version__
    except Exception as exc:
        result["torchvision_error"] = repr(exc)
print("{PROBE_MARKER}" + json.dumps(result))
"""


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_bytes: int
    driver_version: str
    compute_capability: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class DriverInfo:
    gpus: List[GpuInfo]
    cuda_version: Optional[Tuple[int, int]]


@dataclass(frozen=True)
class TorchWheel:
    tag: str
    torch: str = TORCH_VERSION
    torchvision: str = TORCHVISION_VERSION

    @property
    def index_url(self) -> str:
        return f"https://download.pytorch.org/whl/{self.tag}"

    def requirements(self) -> List[str]:
        return [f"torch=={self.torch}+{self.tag}", f"torchvision=={self.torchvision}+{self.tag}"]

    def manual_command(self) -> str:
        return (
            f"python -m pip install torch=={self.torch} torchvision=={self.torchvision} "
            f"--index-url {self.index_url}"
        )


@dataclass
class TorchProbe:
    installed: bool = False
    version: Optional[str] = None
    cuda_build: Optional[str] = None
    cuda_available: bool = False
    arch_list: List[str] = field(default_factory=list)
    torchvision: Optional[str] = None
    torchvision_error: Optional[str] = None
    error: Optional[str] = None


@dataclass(frozen=True)
class RuntimeSettings:
    """What the engine needs to know to place the model on the GPU."""

    device_index: int
    dtype_name: str
    gpu_name: str
    total_bytes: int
    free_bytes: int

    @property
    def device(self) -> str:
        return f"cuda:{self.device_index}"


def _run(command: Sequence[str], timeout: Optional[float] = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        cwd=tempfile.gettempdir(),
    )


def find_nvidia_smi() -> Optional[str]:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    if os.name == "nt":
        candidates = [
            Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
            Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
            / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
    return None


def parse_cuda_version(text: str) -> Optional[Tuple[int, int]]:
    """Extract the highest CUDA version the driver supports from nvidia-smi output."""
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else None


def parse_gpu_csv(text: str) -> List[GpuInfo]:
    """Parse ``nvidia-smi --query-gpu`` CSV output (no header, no units)."""
    gpus: List[GpuInfo] = []
    for line in text.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        capability: Optional[Tuple[int, int]] = None
        if len(parts) >= 5:
            match = re.match(r"(\d+)\.(\d+)", parts[4])
            if match:
                capability = (int(match.group(1)), int(match.group(2)))
        try:
            total = int(float(parts[2])) * 1024 * 1024
            index = int(parts[0])
        except ValueError:
            continue
        gpus.append(GpuInfo(index, parts[1], total, parts[3], capability))
    return gpus


def query_driver() -> Optional[DriverInfo]:
    """Ask nvidia-smi about the installed GPUs. Returns None if it is unavailable."""
    executable = find_nvidia_smi()
    if executable is None:
        return None
    try:
        summary = _run([executable], timeout=60)
        if summary.returncode != 0:
            return None
        fields = "index,name,memory.total,driver_version"
        query = _run(
            [executable, f"--query-gpu={fields},compute_cap", "--format=csv,noheader,nounits"],
            timeout=60,
        )
        if query.returncode != 0:
            query = _run([executable, f"--query-gpu={fields}", "--format=csv,noheader,nounits"], timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = parse_gpu_csv(query.stdout) if query.returncode == 0 else []
    if not gpus:
        return None
    return DriverInfo(gpus=gpus, cuda_version=parse_cuda_version(summary.stdout))


def select_torch_wheel(
    cuda_version: Optional[Tuple[int, int]], capability: Optional[Tuple[int, int]]
) -> Optional[TorchWheel]:
    """Choose the PyTorch CUDA build that both the driver and the GPU support.

    Blackwell GPUs (compute capability 10.0 and newer) need the CUDA 12.8 build.
    Every older generation down to Maxwell is covered by the CUDA 12.6 build,
    or by the CUDA 11.8 build when the driver predates CUDA 12.6.
    """
    if cuda_version is None:
        return None
    if capability is not None and capability >= (10, 0):
        return TorchWheel("cu128") if cuda_version >= (12, 8) else None
    if cuda_version >= (12, 6):
        return TorchWheel("cu126")
    if cuda_version >= (11, 8):
        return TorchWheel("cu118")
    return None


def arch_supported(capability: Optional[Tuple[int, int]], arch_list: Sequence[str]) -> bool:
    """Whether a PyTorch build ships kernels that can run on a GPU generation."""
    if capability is None or not arch_list:
        return True
    major, minor = capability
    for arch in arch_list:
        match = re.match(r"(sm|compute)_(\d+)(\d)$", arch)
        if not match:
            continue
        kind, arch_major, arch_minor = match.group(1), int(match.group(2)), int(match.group(3))
        if kind == "sm" and arch_major == major and arch_minor <= minor:
            return True
        if kind == "compute" and (arch_major, arch_minor) <= (major, minor):
            return True
    return False


def probe_torch() -> TorchProbe:
    """Inspect the installed PyTorch in a child process."""
    try:
        completed = _run([sys.executable, "-c", _PROBE_SCRIPT], timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return TorchProbe(error=repr(exc))
    for line in completed.stdout.splitlines():
        if line.startswith(PROBE_MARKER):
            data = json.loads(line[len(PROBE_MARKER):])
            return TorchProbe(
                installed=data.get("installed", False),
                version=data.get("version"),
                cuda_build=data.get("cuda_build"),
                cuda_available=data.get("cuda_available", False),
                arch_list=list(data.get("arch_list") or []),
                torchvision=data.get("torchvision"),
                torchvision_error=data.get("torchvision_error"),
                error=data.get("error"),
            )
    return TorchProbe(error=(completed.stderr or "").strip()[-500:] or "no output")


def pip_install(arguments: Sequence[str]) -> subprocess.CompletedProcess:
    command = [
        sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--quiet",
        "--timeout", "120", "--retries", "10",
    ]
    return _run(command + list(arguments))


def _last_line(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    return lines[-1] if lines else "no details were reported"


def check_python_version() -> None:
    low, high = TORCH_PYTHON_RANGE
    current = sys.version_info[:2]
    if low <= current <= high:
        return
    raise TurkmenOCRError(
        "ENV-007",
        f"Python {current[0]}.{current[1]} is not supported.",
        f"turkmenOCR-lib is running on Python {sys.version.split()[0]}. The GPU "
        "build of PyTorch that TurkmenOCR depends on is published for Python "
        f"{low[0]}.{low[1]} through {high[0]}.{high[1]} only, so it cannot be "
        "installed into this Python.",
        "Install Python 3.12 from https://www.python.org/downloads/, create a "
        "fresh environment with it, install turkmenOCR-lib there, and run the "
        "command again from that environment.",
        steps=[
            "py -3.12 -m venv turkmenocr-env (Windows) or python3.12 -m venv turkmenocr-env (Linux).",
            "Activate it: turkmenocr-env\\Scripts\\activate (Windows) or source turkmenocr-env/bin/activate (Linux).",
            "pip install turkmenOCR-lib",
        ],
    )


def require_gpu(device_index: int) -> Tuple[DriverInfo, GpuInfo]:
    driver = query_driver()
    if driver is None:
        raise TurkmenOCRError(
            "ENV-001",
            "No working NVIDIA graphics card was found.",
            "turkmenOCR-lib asked the NVIDIA driver tool (nvidia-smi) which "
            "graphics cards are installed, but the tool is missing or could not "
            "talk to a graphics card. TurkmenOCR runs only on NVIDIA graphics "
            "cards with at least 8 GB of video memory, using NVIDIA's official "
            "driver.",
            "Check that this computer has an NVIDIA graphics card and that its "
            "driver is installed.",
            steps=[
                "On Windows, open Device Manager and expand Display adapters. On "
                "Linux, run lspci | grep -i nvidia. An entry that mentions NVIDIA "
                "must be listed.",
                "Download and install the latest driver for your card from "
                "https://www.nvidia.com/Download/index.aspx, then restart the computer.",
                "Open a new terminal, run nvidia-smi and confirm that it prints a "
                "table with your graphics card, then run the turkmenocr command again.",
            ],
        )
    by_index = {gpu.index: gpu for gpu in driver.gpus}
    if device_index not in by_index:
        available = ", ".join(f"{gpu.index} ({gpu.name})" for gpu in driver.gpus)
        raise TurkmenOCRError(
            "INPUT-007",
            f"Graphics card number {device_index} does not exist.",
            f"The --gpu option asked for card number {device_index}, but the "
            f"cards on this computer are numbered: {available}.",
            "Run the command again with --gpu set to one of the numbers listed "
            "above, or leave the option out to use card 0.",
        )
    return driver, by_index[device_index]


def check_total_vram(gpu: GpuInfo, total_bytes: Optional[int] = None) -> None:
    total = gpu.total_bytes if total_bytes is None else total_bytes
    if total >= MIN_TOTAL_VRAM:
        return
    raise TurkmenOCRError(
        "ENV-003",
        "The graphics card does not have enough memory for TurkmenOCR.",
        f"The graphics card {gpu.name} has {format_size(total)} of video memory "
        "in total. TurkmenOCR needs a card with at least 8 GB of video memory to "
        "hold the model and process a page, so it cannot run on this card.",
        "Run turkmenOCR-lib on a computer with an NVIDIA graphics card that has "
        "8 GB of video memory or more, for example a GeForce RTX 3060, RTX 3070, "
        "RTX 4060, RTX 4070 or newer. If this computer has more than one "
        "graphics card, select a larger one with the --gpu option.",
    )


def ensure_torch(driver: DriverInfo, gpu: GpuInfo, auto_install: bool, reporter: Reporter) -> None:
    """Make sure a PyTorch build that can use this GPU is installed."""
    probe = probe_torch()
    reason = _torch_problem(probe, gpu)
    if reason is None:
        return

    wheel = select_torch_wheel(driver.cuda_version, gpu.compute_capability)
    if wheel is None:
        cuda_text = (
            f"CUDA {driver.cuda_version[0]}.{driver.cuda_version[1]}"
            if driver.cuda_version
            else "an unknown CUDA version"
        )
        raise TurkmenOCRError(
            "ENV-002",
            "The NVIDIA driver is too old.",
            f"{reason} The installed NVIDIA driver (version {gpu.driver_version}) "
            f"supports {cuda_text}, which is older than any GPU-enabled PyTorch "
            f"release that works with the {gpu.name}.",
            "Install the latest NVIDIA driver for your graphics card from "
            "https://www.nvidia.com/Download/index.aspx, restart the computer and "
            "run the command again.",
        )
    if not auto_install:
        raise _install_disabled_error("PyTorch with GPU support", reason, wheel.manual_command())

    check_python_version()
    reporter.info(
        f"{reason} Installing the GPU-enabled PyTorch {wheel.torch} build "
        f"({wheel.tag}). This is a one-time download of about 3 GB and can take "
        "several minutes."
    )
    completed = pip_install(
        wheel.requirements()
        + ["--index-url", wheel.index_url, "--extra-index-url", "https://pypi.org/simple"]
    )
    if completed.returncode == 0:
        importlib.invalidate_caches()
        probe = probe_torch()
        if _torch_problem(probe, gpu) is None:
            return
        detail = probe.error or "PyTorch still cannot use the graphics card after installation"
    else:
        detail = _last_line(completed.stderr or completed.stdout)
    raise TurkmenOCRError(
        "ENV-005",
        "PyTorch with GPU support could not be installed automatically.",
        f"{reason} turkmenOCR-lib tried to install the GPU-enabled PyTorch "
        f"{wheel.torch} build ({wheel.tag}) for the {gpu.name}, but the "
        f"installation did not succeed. The installer reported: {detail}",
        "Check that the computer is connected to the internet and close every "
        "other Python program that might be using PyTorch, then install it "
        "manually by running the command below in this same terminal. When it "
        "finishes, run the turkmenocr command again.",
        steps=[wheel.manual_command()],
    )


def _torch_problem(probe: TorchProbe, gpu: GpuInfo) -> Optional[str]:
    if not probe.installed:
        return "PyTorch is not installed."
    if not probe.cuda_build:
        return f"The installed PyTorch {probe.version} is a CPU-only build that cannot use the graphics card."
    if not probe.cuda_available:
        return (
            f"The installed PyTorch {probe.version} was built for CUDA {probe.cuda_build}, "
            "which the current NVIDIA driver cannot run."
        )
    if not arch_supported(gpu.compute_capability, probe.arch_list):
        major, minor = gpu.compute_capability or (0, 0)
        return (
            f"The installed PyTorch {probe.version} does not include support for the "
            f"{gpu.name} (compute capability {major}.{minor})."
        )
    if probe.torchvision is None:
        return "The torchvision package is missing or does not match the installed PyTorch."
    return None


def ensure_support_packages(auto_install: bool, reporter: Reporter) -> None:
    missing = [spec for module, spec in SUPPORT_PACKAGES.items() if importlib.util.find_spec(module) is None]
    if not missing:
        return
    command = "python -m pip install " + " ".join(f'"{spec}"' for spec in missing)
    if not auto_install:
        raise _install_disabled_error("required helper packages", "Some helper packages are missing.", command)
    reporter.info(f"Installing missing helper packages: {', '.join(missing)}.")
    completed = pip_install(missing)
    importlib.invalidate_caches()
    if completed.returncode != 0:
        raise TurkmenOCRError(
            "ENV-005",
            "Required helper packages could not be installed.",
            f"The packages {', '.join(missing)} are needed by TurkmenOCR but are "
            "not installed, and the automatic installation failed. The installer "
            f"reported: {_last_line(completed.stderr or completed.stdout)}",
            "Check the internet connection, then run the command below in this "
            "same terminal and run the turkmenocr command again.",
            steps=[command],
        )


def installed_transformers_version() -> Optional[str]:
    try:
        return metadata.version("transformers")
    except metadata.PackageNotFoundError:
        return None


def ensure_transformers(auto_install: bool, reporter: Reporter) -> None:
    """Align transformers with the 4.46 series the model code depends on."""
    current = installed_transformers_version()
    if current is not None and current.startswith(TRANSFORMERS_SERIES):
        return
    manual = f"python -m pip install transformers=={TRANSFORMERS_TARGET}"
    found = f"version {current} is installed" if current else "it is not installed"
    if "transformers" in sys.modules:
        raise TurkmenOCRError(
            "ENV-006",
            f"transformers {TRANSFORMERS_TARGET} is required, but another version is already loaded.",
            f"TurkmenOCR relies on internal parts of transformers {TRANSFORMERS_TARGET}, "
            f"but {found} and it has already been loaded into the running Python "
            "program, so it cannot be replaced while the program is running.",
            "Run the command below, then restart Python (or your notebook kernel) "
            "and try again.",
            steps=[manual],
        )
    if not auto_install:
        raise _install_disabled_error(
            f"transformers {TRANSFORMERS_TARGET}",
            f"TurkmenOCR needs transformers {TRANSFORMERS_TARGET}, but {found}.",
            manual,
        )
    reporter.info(
        f"TurkmenOCR needs transformers {TRANSFORMERS_TARGET} ({found}), so it is being installed now."
    )
    completed = pip_install([f"transformers=={TRANSFORMERS_TARGET}"])
    importlib.invalidate_caches()
    installed = installed_transformers_version()
    if completed.returncode == 0 and installed and installed.startswith(TRANSFORMERS_SERIES):
        return
    raise TurkmenOCRError(
        "ENV-006",
        f"transformers {TRANSFORMERS_TARGET} could not be installed automatically.",
        f"TurkmenOCR relies on internal parts of transformers {TRANSFORMERS_TARGET}, "
        f"but {found}. The automatic installation failed, most likely because "
        "there is no internet connection or because this Python installation is "
        "write-protected. The installer reported: "
        f"{_last_line(completed.stderr or completed.stdout)}",
        "Run the command below in this same terminal. If it reports a permission "
        "problem, add --user to the end of it or use a virtual environment. Then "
        "run the turkmenocr command again.",
        steps=[manual],
    )


def _install_disabled_error(what: str, reason: str, command: str) -> TurkmenOCRError:
    return TurkmenOCRError(
        "ENV-008",
        f"{what[0].upper()}{what[1:]} must be installed before TurkmenOCR can run.",
        f"{reason} Automatic installation is turned off because the --no-install "
        "option was given.",
        "Run the command below in this same terminal, or run turkmenocr again "
        "without --no-install to let it install the package automatically.",
        steps=[command],
    )


def select_precision(device_index: int) -> str:
    """bfloat16 on Ampere and newer, float16 on older generations."""
    import torch

    major, _ = torch.cuda.get_device_capability(device_index)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "float16"


def check_memory(
    gpu: GpuInfo, device_index: int, required_bytes: int, ignore_vram_check: bool
) -> RuntimeSettings:
    import torch

    properties = torch.cuda.get_device_properties(device_index)
    free, total = torch.cuda.mem_get_info(device_index)
    if not ignore_vram_check:
        check_total_vram(gpu, total_bytes=properties.total_memory)
        needed = required_bytes + ACTIVATION_RESERVE
        if free < needed:
            raise TurkmenOCRError(
                "ENV-004",
                "Not enough free video memory to load TurkmenOCR right now.",
                f"The graphics card {properties.name} has {format_size(total)} of "
                f"video memory in total, but only {format_size(free)} is free at "
                f"the moment. Loading the model and processing a page needs about "
                f"{format_size(needed)}. Other programs are currently using the "
                f"remaining {format_size(total - free)}.",
                "Close programs that use the graphics card, such as web browsers, "
                "games, video editors, other AI tools or Python notebooks, then run "
                "the command again. Running nvidia-smi in a terminal lists the "
                "programs that are currently using video memory.",
            )
    return RuntimeSettings(
        device_index=device_index,
        dtype_name=select_precision(device_index),
        gpu_name=properties.name,
        total_bytes=total,
        free_bytes=free,
    )


def prepare_environment(
    required_bytes: int,
    device_index: int = 0,
    auto_install: bool = True,
    ignore_vram_check: bool = False,
    reporter: Optional[Reporter] = None,
) -> RuntimeSettings:
    """Verify the GPU and install or align runtime packages as needed."""
    reporter = reporter or Reporter(quiet=True)
    driver, gpu = require_gpu(device_index)
    if not ignore_vram_check:
        check_total_vram(gpu)
    reporter.info(f"Using graphics card {gpu.index}: {gpu.name} ({format_size(gpu.total_bytes)}).")
    ensure_torch(driver, gpu, auto_install, reporter)
    ensure_support_packages(auto_install, reporter)
    ensure_transformers(auto_install, reporter)
    importlib.invalidate_caches()
    return check_memory(gpu, device_index, required_bytes, ignore_vram_check)
