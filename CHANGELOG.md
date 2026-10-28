# Changelog

All notable changes to this project are documented in this file. The format
follows Keep a Changelog, and the project uses Semantic Versioning.

## [0.1.0] - Unreleased

### Added

- `turkmenocr` command-line tool and `turkmenocr` Python package.
- Zero-copy safetensors inspection with shard, truncation and architecture checks.
- Automatic GPU detection, PyTorch CUDA installation and transformers 4.46 alignment.
- Non-destructive workspace that combines bundled model files with user weights.
- Plain-text, three-paragraph diagnostics for every anticipated failure.
- Graphics cards with 6 GB of video memory are supported: when the model does
  not fit, part of the mixture-of-experts weights runs on the CPU.
