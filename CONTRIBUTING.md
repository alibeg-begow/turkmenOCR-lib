# Contributing

Thank you for helping improve turkmenOCR-lib. This guide explains how to set
up a development environment, what a good change looks like, and how changes
reach a release.

## Reporting problems

Open an issue at https://github.com/alibeg-begow/turkmenOCR-lib/issues and
use the bug report form. The most useful reports include:

- the exact command you ran,
- the complete output of that command with `--debug` added,
- the output of `nvidia-smi`,
- your operating system and Python version.

Never attach model weights or private documents. A cropped sample image with
no personal information is enough to reproduce most reading problems.

## Development setup

```
git clone https://github.com/alibeg-begow/turkmenOCR-lib
cd turkmenOCR-lib
python -m venv .venv
.venv\Scripts\activate          (Windows)
source .venv/bin/activate       (Linux)
python -m pip install -e ".[dev]"
```

Run the checks before you open a pull request:

```
ruff check .
pytest
```

The test suite needs no graphics card, PyTorch or model weights. Tests that
require PyTorch, such as `tests/test_offload.py`, are skipped when PyTorch is
not installed.

## Guidelines for changes

- Keep every message the user sees in plain English. Report anticipated
  failures as a `TurkmenOCRError` with a code, a summary, an explanation that
  includes the concrete values that were detected, and resolution steps that
  someone without programming experience can follow.
- Standard output is reserved for recognised text. Progress messages and
  diagnostics go through `Reporter`, which writes to standard error.
- Do not import `torch` or `transformers` at module level outside
  `turkmenocr/offload.py`. The environment checks must be able to run, and
  replace these packages if needed, before either one is imported.
- Do not use emojis, decorative separators or colour codes in terminal output
  or documentation.
- Add or update tests for every behaviour change, and add an entry under
  "Unreleased" in `CHANGELOG.md`.
- The files in `turkmenocr/assets/turkmen_ocr/` are the model's own code and
  configuration. Change them only when the model itself requires it, and
  test the change on a real GPU.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for how the library is
organised and [docs/RELEASING.md](docs/RELEASING.md) for the release process.

## Pull requests

1. Create a branch from `main`.
2. Make your change and run `ruff check .` and `pytest`.
3. Open a pull request that describes what changed and why. If the change
   affects model loading or inference, say how you tested it on a GPU.

The CI checks (lint, tests on Linux and Windows for Python 3.9 to 3.13, and
the package build) must pass before a pull request can be merged.

## License

By contributing, you agree that your contributions are licensed under the
MIT license that covers this project.
