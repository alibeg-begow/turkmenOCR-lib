# Changelog

All notable changes to this project are documented in this file. The format
follows Keep a Changelog, and the project uses Semantic Versioning.

## [Unreleased]

## [0.2.1] - 2026-10-28

### Fixed

- Fixed false `ENV-002` ("The NVIDIA driver is too old") errors on NVIDIA 590
  and newer driver branches (such as 617.42) and on Optimus laptop GPUs where
  `nvidia-smi` prints `CUDA UMD Version` instead of `CUDA Version` or reports
  `N/A`. CUDA support is now also queried directly from the NVIDIA CUDA driver
  library (`cuDriverGetVersion`), with a fallback to the driver release branch.

## [0.2.0] - 2026-10-28

### Added

- `turkmenocr --setup` detects the graphics card, its compute capability, the
  driver and the newest CUDA version it supports, and installs the matching
  GPU build of PyTorch before the first run.
- Python 3.14 support. On Python 3.14, PyTorch 2.9.1 and transformers 4.47.1
  are installed. On Python 3.9 to 3.13, PyTorch 2.7.1 and transformers 4.46
  remain in use.
- CONTRIBUTING.md, SECURITY.md, issue forms and a pull request template.

### Changed

- `pip install turkmenOCR-lib` no longer installs PyTorch. Previously,
  `accelerate` pulled in the CPU-only PyTorch build from PyPI on Windows,
  which then had to be replaced. `accelerate` is now installed at runtime,
  after the GPU build of PyTorch.
- Every run now reports the detected driver, CUDA version and compute
  capability.

## [0.1.0] - 2026-10-28

### Added

- `turkmenocr` command-line tool and `turkmenocr` Python package.
- Zero-copy safetensors inspection with shard, truncation and architecture checks.
- Automatic GPU detection, PyTorch CUDA installation and transformers 4.46 alignment.
- Non-destructive workspace that combines bundled model files with user weights.
- Plain-text, three-paragraph diagnostics for every anticipated failure.
- Graphics cards with 6 GB of video memory are supported: when the model does
  not fit, part of the mixture-of-experts weights runs on the CPU.

[0.2.1]: https://github.com/alibeg-begow/turkmenOCR-lib/releases/tag/v0.2.1
[0.2.0]: https://github.com/alibeg-begow/turkmenOCR-lib/releases/tag/v0.2.0
[0.1.0]: https://github.com/alibeg-begow/turkmenOCR-lib/releases/tag/v0.1.0
