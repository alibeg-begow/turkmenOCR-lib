import subprocess

import pytest

from turkmenocr import environment
from turkmenocr.diagnostics import Reporter, TurkmenOCRError
from turkmenocr.environment import (
    DriverInfo,
    GpuInfo,
    TorchProbe,
    arch_supported,
    check_total_vram,
    parse_cuda_version,
    parse_gpu_csv,
    plan_gpu_budget,
    select_torch_wheel,
)

SMI_SUMMARY = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
+-----------------------------------------------------------------------------------------+
"""

GTX_1050 = GpuInfo(0, "NVIDIA GeForce GTX 1050 Ti", 4096 * 1024 * 1024, "582.28", (6, 1))
GTX_1060 = GpuInfo(0, "NVIDIA GeForce GTX 1060", 6144 * 1024 * 1024, "582.28", (6, 1))
RTX_3060 = GpuInfo(0, "NVIDIA GeForce RTX 3060", 12288 * 1024 * 1024, "560.94", (8, 6))
RTX_3070 = GpuInfo(0, "NVIDIA GeForce RTX 3070", 8192 * 1024 * 1024, "560.94", (8, 6))
QUIET = Reporter(quiet=True)


def completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


def test_parse_cuda_version():
    assert parse_cuda_version(SMI_SUMMARY) == (12, 6)
    assert parse_cuda_version("CUDA Version: 13.0") == (13, 0)
    assert parse_cuda_version("no driver") is None


def test_parse_gpu_csv_with_and_without_compute_capability():
    gpus = parse_gpu_csv("0, NVIDIA GeForce RTX 3060, 12288, 560.94, 8.6\n1, Tesla T4, 15360, 535.1, 7.5\n")
    assert gpus[0].total_bytes == 12288 * 1024 * 1024
    assert gpus[0].compute_capability == (8, 6)
    assert gpus[1].index == 1
    legacy = parse_gpu_csv("0, Quadro P6000, 24576, 470.1")
    assert legacy[0].compute_capability is None


@pytest.mark.parametrize(
    "cuda, capability, python, tag",
    [
        ((13, 0), (12, 0), (3, 12), "cu128"),
        ((12, 8), (10, 0), (3, 9), "cu128"),
        ((12, 4), (12, 0), (3, 12), None),
        ((13, 0), (8, 6), (3, 12), "cu126"),
        ((12, 6), (6, 1), (3, 13), "cu126"),
        ((12, 2), (7, 5), (3, 10), "cu118"),
        ((11, 8), None, (3, 11), "cu118"),
        ((11, 4), (7, 5), (3, 12), None),
        (None, (8, 6), (3, 12), None),
        ((13, 0), (6, 1), (3, 14), "cu126"),
        ((13, 0), (12, 0), (3, 14), "cu128"),
        ((12, 2), (7, 5), (3, 14), None),
        ((13, 0), (8, 6), (3, 15), None),
    ],
)
def test_select_torch_wheel(cuda, capability, python, tag):
    wheel = select_torch_wheel(cuda, capability, python)
    assert (wheel.tag if wheel else None) == tag


@pytest.mark.parametrize(
    "python, torch, torchvision",
    [((3, 9), "2.7.1", "0.22.1"), ((3, 13), "2.7.1", "0.22.1"), ((3, 14), "2.9.1", "0.24.1")],
)
def test_wheel_requirements_pin_local_cuda_version(python, torch, torchvision):
    wheel = select_torch_wheel((12, 6), (8, 6), python)
    assert wheel.requirements() == [f"torch=={torch}+cu126", f"torchvision=={torchvision}+cu126"]
    assert wheel.index_url.endswith("/cu126")


def test_every_python_from_3_9_to_3_14_has_a_release():
    for minor in range(9, 15):
        assert environment.torch_release_for((3, minor)) is not None
    assert environment.torch_release_for((3, 8)) is None


def test_arch_supported():
    assert arch_supported((8, 6), ["sm_80", "sm_86", "sm_90"])
    assert arch_supported((8, 9), ["sm_80", "sm_86"])
    assert not arch_supported((6, 1), ["sm_75", "sm_80", "sm_90", "sm_120"])
    assert arch_supported((12, 0), ["sm_90", "compute_90"])
    assert arch_supported(None, ["sm_80"])


def test_torch_problem_detection():
    assert "not installed" in environment._torch_problem(TorchProbe(), RTX_3060)
    cpu = TorchProbe(installed=True, version="2.7.1+cpu")
    assert "CPU-only" in environment._torch_problem(cpu, RTX_3060)
    old = TorchProbe(installed=True, version="2.8.0+cu128", cuda_build="12.8", cuda_available=True,
                     arch_list=["sm_75", "sm_80", "sm_120"], torchvision="0.23.0")
    assert "does not include support" in environment._torch_problem(old, GTX_1060)
    good = TorchProbe(installed=True, version="2.7.1+cu126", cuda_build="12.6", cuda_available=True,
                      arch_list=["sm_61", "sm_86"], torchvision="0.22.1")
    assert environment._torch_problem(good, RTX_3060) is None


def test_small_card_is_rejected():
    with pytest.raises(TurkmenOCRError) as excinfo:
        check_total_vram(GTX_1050)
    assert excinfo.value.code == "ENV-003"
    assert "4.0 GB" in excinfo.value.explanation
    assert "at least 6 GB" in excinfo.value.explanation
    check_total_vram(GTX_1060)
    check_total_vram(RTX_3060)
    check_total_vram(RTX_3070)


GIB = 1024 ** 3
WEIGHTS = int(6.2 * GIB)
MINIMUM = int(1.7 * GIB)


def test_whole_model_on_large_card():
    assert plan_gpu_budget("RTX 3060", int(11 * GIB), 12 * GIB, WEIGHTS, MINIMUM) is None


def test_partial_placement_on_six_gb_card():
    budget = plan_gpu_budget("GTX 1060", int(5.6 * GIB), 6 * GIB, WEIGHTS, MINIMUM)
    assert budget == int(5.6 * GIB) - environment.ACTIVATION_RESERVE
    assert MINIMUM <= budget < WEIGHTS


def test_too_little_free_memory():
    with pytest.raises(TurkmenOCRError) as excinfo:
        plan_gpu_budget("GTX 1060", int(2 * GIB), 6 * GIB, WEIGHTS, MINIMUM)
    assert excinfo.value.code == "ENV-004"
    assert "2.0 GB is free" in excinfo.value.explanation
    assert plan_gpu_budget("GTX 1060", int(2 * GIB), 6 * GIB, WEIGHTS, MINIMUM, ignore_vram_check=True) == MINIMUM


def test_missing_driver(monkeypatch):
    monkeypatch.setattr(environment, "query_driver", lambda: None)
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.require_gpu(0)
    assert excinfo.value.code == "ENV-001"


def test_unknown_gpu_index(monkeypatch):
    monkeypatch.setattr(environment, "query_driver", lambda: DriverInfo([RTX_3060], (12, 6)))
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.require_gpu(3)
    assert excinfo.value.code == "INPUT-007"


def test_old_driver_cannot_get_torch(monkeypatch):
    monkeypatch.setattr(environment, "probe_torch", lambda: TorchProbe())
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.ensure_torch(DriverInfo([RTX_3060], (11, 4)), RTX_3060, True, QUIET)
    assert excinfo.value.code == "ENV-002"


def test_torch_install_disabled(monkeypatch):
    monkeypatch.setattr(environment, "probe_torch", lambda: TorchProbe())
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.ensure_torch(DriverInfo([RTX_3060], (12, 6)), RTX_3060, False, QUIET)
    assert excinfo.value.code == "ENV-008"
    assert "download.pytorch.org/whl/cu126" in excinfo.value.steps[0]


def test_torch_is_installed_for_cpu_only_build(monkeypatch):
    probes = iter([
        TorchProbe(installed=True, version="2.7.1+cpu"),
        TorchProbe(installed=True, version="2.7.1+cu126", cuda_build="12.6", cuda_available=True,
                   arch_list=["sm_86"], torchvision="0.22.1"),
    ])
    calls = []
    monkeypatch.setattr(environment, "probe_torch", lambda: next(probes))
    monkeypatch.setattr(environment, "pip_install", lambda args: calls.append(list(args)) or completed())
    environment.ensure_torch(DriverInfo([RTX_3060], (12, 6)), RTX_3060, True, QUIET)
    release = environment.torch_release_for()
    assert calls[0][:2] == [f"torch=={release.torch}+cu126", f"torchvision=={release.torchvision}+cu126"]


def test_failed_torch_install(monkeypatch):
    monkeypatch.setattr(environment, "probe_torch", lambda: TorchProbe())
    monkeypatch.setattr(environment, "pip_install", lambda args: completed(1, stderr="ERROR: network down"))
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.ensure_torch(DriverInfo([RTX_3060], (12, 6)), RTX_3060, True, QUIET)
    assert excinfo.value.code == "ENV-005"
    assert "network down" in excinfo.value.explanation


@pytest.mark.parametrize(
    "python, series, target",
    [((3, 9), "4.46.", "4.46.0"), ((3, 13), "4.46.", "4.46.0"), ((3, 14), "4.47.", "4.47.1")],
)
def test_transformers_requirement_per_python(python, series, target):
    assert environment.transformers_requirement(python) == (series, target)


def test_transformers_patch_release_is_accepted(monkeypatch):
    series, _ = environment.transformers_requirement()
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: series + "9")
    monkeypatch.setattr(environment, "pip_install", lambda args: pytest.fail("must not install"))
    environment.ensure_transformers(True, QUIET)


def test_transformers_is_pinned(monkeypatch):
    _, target = environment.transformers_requirement()
    versions = iter(["4.57.1", target])
    calls = []
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: next(versions))
    monkeypatch.setattr(environment, "pip_install", lambda args: calls.append(list(args)) or completed())
    monkeypatch.delitem(environment.sys.modules, "transformers", raising=False)
    environment.ensure_transformers(True, QUIET)
    assert calls == [[f"transformers=={target}"]]


def test_transformers_install_failure(monkeypatch):
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: None)
    monkeypatch.setattr(environment, "pip_install", lambda args: completed(1, stderr="Permission denied"))
    monkeypatch.delitem(environment.sys.modules, "transformers", raising=False)
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.ensure_transformers(True, QUIET)
    assert excinfo.value.code == "ENV-006"
    assert excinfo.value.steps == [f"python -m pip install transformers=={environment.transformers_requirement()[1]}"]


def test_python_version_gate(monkeypatch):
    monkeypatch.setattr(environment.sys, "version_info", (3, 15, 0, "final", 0))
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.check_python_version()
    assert excinfo.value.code == "ENV-007"


def test_describe_gpu_reports_driver_cuda_and_capability():
    text = environment.describe_gpu(DriverInfo([GTX_1060], (13, 0)), GTX_1060)
    assert "GTX 1060 with 6.0 GB" in text
    assert "compute capability 6.1" in text
    assert "driver 582.28 supports CUDA up to 13.0" in text


def test_cuda_label():
    assert select_torch_wheel((12, 6), (8, 6), (3, 12)).cuda_label == "CUDA 12.6"
    assert select_torch_wheel((12, 2), (8, 6), (3, 12)).cuda_label == "CUDA 11.8"


def test_setup_runtime_summary(monkeypatch):
    good = TorchProbe(installed=True, version="2.7.1+cu126", cuda_build="12.6", cuda_available=True,
                      arch_list=["sm_86"], torchvision="0.22.1")
    monkeypatch.setattr(environment, "query_driver", lambda: DriverInfo([RTX_3060], (12, 6)))
    monkeypatch.setattr(environment, "probe_torch", lambda: good)
    monkeypatch.setattr(environment, "ensure_support_packages", lambda auto, reporter: None)
    monkeypatch.setattr(environment, "ensure_transformers", lambda auto, reporter: None)
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: "4.46.0")
    summary = environment.setup_runtime(0, True, QUIET)
    assert "PyTorch 2.7.1+cu126 (CUDA 12.6)" in summary
    assert "transformers 4.46.0" in summary


def test_pip_install_never_pulls_torch():
    # PyPI only offers CPU builds of torch on Windows, so neither torch nor a
    # package that depends on it may be an install-time dependency.
    import pathlib
    import re

    pyproject = (pathlib.Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", pyproject, re.S | re.M).group(1)
    names = {re.match(r'\s*"([A-Za-z0-9_.-]+)', line).group(1).lower() for line in block.splitlines() if '"' in line}
    assert not names & {"torch", "torchvision", "accelerate"}
