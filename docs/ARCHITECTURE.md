# Architecture

This document describes how turkmenOCR-lib is organised, the order in which
it performs its checks, and the reasons behind the main design decisions. It
is written for contributors and maintainers.

## Goals

1. Run TurkmenOCR from a folder that contains nothing but `.safetensors`
   weights. Every other file the model needs ships inside the package.
2. Detect every foreseeable problem before expensive work starts: before
   PyTorch is imported, before gigabytes are downloaded and before video
   memory is allocated.
3. Explain every problem in plain English that a person without programming
   experience can act on. Standard output carries only the recognised text.

## Package layout

- `turkmenocr/cli.py`: argument parsing, the top-level flow and the single
  place where exceptions become printed diagnostics and exit codes.
- `turkmenocr/diagnostics.py`: the `TurkmenOCRError` exception, the
  three-paragraph formatter, exit codes and the `Reporter` used for progress
  messages on standard error.
- `turkmenocr/validation.py`: path normalisation and validation of the
  weights folder and the image.
- `turkmenocr/inspector.py`: safetensors header parsing, shard verification,
  completeness checks and architecture fingerprinting.
- `turkmenocr/environment.py`: GPU discovery through `nvidia-smi`, the
  PyTorch probe and installer, `transformers` alignment, video memory checks
  and precision selection.
- `turkmenocr/workspace.py`: assembly of a loadable model folder from the
  bundled assets and the user's weights.
- `turkmenocr/engine.py`: the `TurkmenOCR` class, which loads the model and
  runs inference.
- `turkmenocr/assets/turkmen_ocr/`: bundled configuration, tokenizer, shard
  index and model code for the TurkmenOCR architecture.

Only `cli.py` writes to standard output. All other modules raise
`TurkmenOCRError` or report through a `Reporter`. Library users therefore
get exceptions, never prints.

## Execution order

1. Parse arguments. Usage errors are raised as `INPUT-000` instead of
   argparse's default message, so they share the diagnostic layout.
2. Normalise both paths. Surrounding whitespace and stray quotes are removed,
   because Windows shells keep a trailing quote when a quoted path ends in a
   backslash. Then `~` and environment variables are expanded and the path
   is resolved.
3. Validate the weights folder (exists, is a folder, can be listed) and the
   image (exists, is a file, is not a PDF or HEIC, decodes completely with
   Pillow).
4. Inspect the weights (described below). No third-party library is used.
5. Query `nvidia-smi`. A missing driver, a missing card index or a card below
   the video memory minimum stops the run here, before any download.
6. Probe PyTorch in a child process, and install the matching CUDA build if
   required.
7. Install any missing helper packages, then align `transformers` with the
   4.46 series.
8. Import PyTorch in this process, measure free video memory and choose the
   precision.
9. Build the workspace, load the tokenizer and the model, and remove the
   workspace again.
10. Prepare the image, run generation, clean the output and print it.

## Weight inspection

### Header parsing

A safetensors file is an 8-byte little-endian header length, a JSON header
and the raw tensor bytes. The header is parsed directly rather than through
`safetensors.safe_open`, for three reasons:

- `safe_open(..., framework="pt")` imports PyTorch. At this stage PyTorch may
  be missing or may be a CPU-only build that is about to be replaced.
- The header gives each tensor's byte range. Comparing the largest end
  offset with the file size reports exactly how many bytes an interrupted
  download is missing. `safe_open` does not detect this until the tensor
  data is read.
- Each byte range is checked against the size implied by the tensor's dtype
  and shape, which catches damaged headers.

### Shards

Names that match `<prefix>-<index>-of-<total>.safetensors` are grouped by
prefix and total. Every index from 1 to the total must be present. The
zero-padding of the reconstructed names is taken from the files on disk.
This matters because the published TurkmenOCR file uses different padding
widths for the index and the total (`model-00001-of-000001.safetensors`).

### Fingerprinting

The checks run in this order:

1. Files whose tensor names contain LoRA markers are adapters. If only
   adapters are present, the run stops with `MODEL-002`, which explains how
   to merge them. Otherwise they are ignored with a warning.
2. The same tensor name in two files means two models are mixed in one
   folder (`WEIGHTS-006`).
3. TurkmenOCR is recognised by its `model.sam_model.` and
   `model.vision_model.` tensors, or by an embedding table of 129280 x 1280.
   The recognised set is then compared with the 2710 tensor names in the
   bundled `model.safetensors.index.json`. Missing parts are reported by
   component, for example "high-resolution vision encoder" (`MODEL-003`).
   The embedding shape must also match (`MODEL-004`).
4. Anything else is named as precisely as possible for `MODEL-001`: Qwen-VL,
   LLaVA or Pixtral, other vision-language chat models, CLIP or SigLIP,
   diffusion models, Whisper, or text-only language models. Text models are
   further identified by vocabulary size where it is distinctive (LLaMA 2 or
   Mistral, LLaMA 3, Qwen, Gemma).

### Variants

`SUPPORTED_VARIANTS` currently holds one entry, `turkmen_ocr`. A new variant
needs three things: a `ModelVariant` describing its component prefixes and
dimensions, an asset folder with its configuration, tokenizer, model code
and full shard index, and a fingerprint rule in `identify_variant`.

## Environment management

### Why PyTorch is probed in a child process

Once a CPU-only PyTorch is imported, it cannot be replaced within the same
process. The probe runs `import torch` in a separate interpreter. It reports
the version, the CUDA build, `cuda.is_available()`, the compiled
architecture list and whether `torchvision` imports cleanly. The main
process imports PyTorch only after any reinstall has finished.

### Choosing a PyTorch build

The CUDA version supported by the driver comes from the `nvidia-smi` banner.
The compute capability comes from `--query-gpu=compute_cap` (with a fallback
query for old drivers that lack this field). The choice is made in
`select_torch_wheel`:

- Compute capability 10.0 or higher (Blackwell): `cu128`, which needs a
  driver that supports CUDA 12.8.
- Otherwise, a driver that supports CUDA 12.6 or newer: `cu126`. This build
  covers Maxwell through Hopper, including Pascal cards that newer CUDA 12.8
  builds no longer support.
- Otherwise, a driver that supports CUDA 11.8 or newer: `cu118`.
- Otherwise: the driver is too old (`ENV-002`).

The versions are pinned to `torch==2.7.1+<tag>` and
`torchvision==0.22.1+<tag>`. The local version suffix forces pip to replace
an installed CPU build of the same version number.

An installed CUDA build is still replaced when its architecture list cannot
run on the GPU, for example a CUDA 12.8 build on a Pascal card. In that case
`torch.cuda.is_available()` returns true, but every kernel launch fails.

### transformers

The model code uses internal APIs of the `transformers` 4.46 series.
`importlib.metadata` reads the installed version without importing the
package. Any 4.46.x release is accepted. Anything else is replaced with
4.46.0, and import caches are invalidated afterwards. If `transformers` has
already been imported (possible only in library use), it cannot be replaced
safely, and the user is asked to restart Python (`ENV-006`).

### Video memory and precision

- Total memory must be at least 7.5 GiB. Cards sold as 8 GB report slightly
  less than 8 GiB.
- Free memory must cover the weights at two bytes per parameter (about
  6.2 GiB) plus a 1 GiB reserve for activations and the key-value cache.
- bfloat16 is used on compute capability 8.0 and newer. float16 is used on
  older cards, where bfloat16 is emulated or missing. The bundled model code
  takes its input dtype from the vision encoder's parameters, so both
  precisions work.

## Workspace

`transformers` loads a model from one folder that holds both the
configuration and the weights. The workspace provides that folder without
modifying the user's folder:

1. `<cache>/workspaces/ws_<hash>/` with symbolic links to the weights.
2. The same folder with hard links. This covers Windows without Developer
   Mode, as long as the weights are on the same drive as the cache.
3. `<weights>/turkmenocr_workspace/` with hard links. This covers weights on
   a different drive, as long as the weights folder is writable.
4. The missing support files copied into the weights folder itself. Only
   files this run created are removed afterwards. Existing user files are
   never overwritten.

The folder name is derived from the weights path, so it is stable between
runs. This matters because `transformers` copies remote code into a module
cache keyed by folder name. A random name would add a new cache entry on
every run. A folder is deleted only if it contains the workspace marker
file.

The shard index is always generated from the inspected headers, so any file
naming works: one file, many shards or renamed files.

## Inference

- The tokenizer and model load with `trust_remote_code=True`,
  `use_safetensors=True`, `low_cpu_mem_usage=True`, the selected dtype,
  `device_map={"": <gpu>}` and `attn_implementation="eager"`. The model
  code defines no SDPA path, so eager attention is required.
- The image is loaded with its EXIF orientation applied. Transparency is
  composited onto white, and the longest side is capped at 4096 pixels.
- Prompts are `"<image>\nFree OCR. "` for text and
  `"<image>\n<|grounding|>Convert the document to markdown. "` for Markdown.
- The output is cleaned of grounding tags (`<|ref|>`, `<|det|>`) and special
  tokens.
- While the model loads and runs, standard output is redirected to standard
  error and warnings are silenced, so third-party prints cannot reach the
  OCR output.
- `torch.cuda.OutOfMemoryError` frees the CUDA cache and is reported as
  `RUNTIME-001` with current memory figures.

## Error codes

- `INPUT-000` to `INPUT-008`: command line, paths and image.
- `WEIGHTS-001` to `WEIGHTS-006`: discovery, shards and file integrity.
- `MODEL-001` to `MODEL-004`: architecture, adapters, completeness and
  dimensions.
- `ENV-001` to `ENV-008`: driver, GPU, video memory, PyTorch,
  `transformers`, Python version and disabled installs.
- `WORKSPACE-001` to `WORKSPACE-002`: links, permissions and a conflicting
  index.
- `RUNTIME-001` to `RUNTIME-003`: out of memory, load failure and inference
  failure.
- `INTERNAL-001`: anything unanticipated. Run again with `--debug` to see
  the traceback.

## Testing

The test suite needs only `pytest` and `pillow`. It builds small synthetic
safetensors files, fakes `nvidia-smi` and pip results with monkeypatching,
and checks the full set of 2710 bundled tensor names in memory. CI runs it
on Linux and Windows for Python 3.9 to 3.13. CI also builds the wheel,
checks that every bundled asset is inside it, and runs the installed
command.

Inference itself needs the real weights and a GPU. CI cannot cover it, so it
must be checked by hand before each release (see `RELEASING.md`).
