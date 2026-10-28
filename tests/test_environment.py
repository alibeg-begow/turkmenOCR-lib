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
    select_torch_wheel,
)

SMI_SUMMARY = """
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
+-----------------------------------------------------------------------------------------+
"""

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
    "cuda, capability, tag",
    [
        ((13, 0), (12, 0), "cu128"),
        ((12, 8), (10, 0), "cu128"),
        ((12, 4), (12, 0), None),
        ((13, 0), (8, 6), "cu126"),
        ((12, 6), (6, 1), "cu126"),
        ((12, 2), (7, 5), "cu118"),
        ((11, 8), None, "cu118"),
        ((11, 4), (7, 5), None),
        (None, (8, 6), None),
    ],
)
def test_select_torch_wheel(cuda, capability, tag):
    wheel = select_torch_wheel(cuda, capability)
    assert (wheel.tag if wheel else None) == tag


def test_wheel_requirements_pin_local_cuda_version():
    wheel = select_torch_wheel((12, 6), (8, 6))
    assert wheel.requirements() == ["torch==2.7.1+cu126", "torchvision==0.22.1+cu126"]
    assert wheel.index_url.endswith("/cu126")


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
        check_total_vram(GTX_1060)
    assert excinfo.value.code == "ENV-003"
    assert "6.0 GB" in excinfo.value.explanation
    check_total_vram(RTX_3060)
    check_total_vram(RTX_3070)


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
    assert calls[0][:2] == ["torch==2.7.1+cu126", "torchvision==0.22.1+cu126"]


def test_failed_torch_install(monkeypatch):
    monkeypatch.setattr(environment, "probe_torch", lambda: TorchProbe())
    monkeypatch.setattr(environment, "pip_install", lambda args: completed(1, stderr="ERROR: network down"))
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.ensure_torch(DriverInfo([RTX_3060], (12, 6)), RTX_3060, True, QUIET)
    assert excinfo.value.code == "ENV-005"
    assert "network down" in excinfo.value.explanation


def test_transformers_patch_release_is_accepted(monkeypatch):
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: "4.46.3")
    monkeypatch.setattr(environment, "pip_install", lambda args: pytest.fail("must not install"))
    environment.ensure_transformers(True, QUIET)


def test_transformers_is_pinned(monkeypatch):
    versions = iter(["4.57.1", "4.46.0"])
    calls = []
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: next(versions))
    monkeypatch.setattr(environment, "pip_install", lambda args: calls.append(list(args)) or completed())
    monkeypatch.delitem(environment.sys.modules, "transformers", raising=False)
    environment.ensure_transformers(True, QUIET)
    assert calls == [["transformers==4.46.0"]]


def test_transformers_install_failure(monkeypatch):
    monkeypatch.setattr(environment, "installed_transformers_version", lambda: None)
    monkeypatch.setattr(environment, "pip_install", lambda args: completed(1, stderr="Permission denied"))
    monkeypatch.delitem(environment.sys.modules, "transformers", raising=False)
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.ensure_transformers(True, QUIET)
    assert excinfo.value.code == "ENV-006"
    assert excinfo.value.steps == ["python -m pip install transformers==4.46.0"]


def test_python_version_gate(monkeypatch):
    monkeypatch.setattr(environment.sys, "version_info", (3, 14, 0, "final", 0))
    with pytest.raises(TurkmenOCRError) as excinfo:
        environment.check_python_version()
    assert excinfo.value.code == "ENV-007"
