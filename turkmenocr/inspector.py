"""Zero-copy inspection of ``.safetensors`` weight directories.

A safetensors file starts with an 8-byte little-endian header length followed
by a JSON header that lists every tensor's dtype, shape and byte range. This
module reads only that header, so it needs neither PyTorch nor enough memory
to hold the weights, and it can tell a truncated download apart from a
complete one by comparing the byte ranges with the size of the file on disk.
"""

from __future__ import annotations

import json
import re
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .diagnostics import Reporter, TurkmenOCRError, format_size

HEADER_LIMIT = 100 * 1024 * 1024

DTYPE_SIZES: Dict[str, int] = {
    "F64": 8, "F32": 4, "F16": 2, "BF16": 2,
    "I64": 8, "I32": 4, "I16": 2, "I8": 1,
    "U64": 8, "U32": 4, "U16": 2, "U8": 1,
    "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
}

SHARD_PATTERN = re.compile(
    r"^(?P<prefix>.+)-(?P<index>\d+)-of-(?P<total>\d+)\.safetensors$", re.IGNORECASE
)

OTHER_WEIGHT_FORMATS = {
    ".bin": "PyTorch pickle (.bin)",
    ".pt": "PyTorch pickle (.pt)",
    ".pth": "PyTorch pickle (.pth)",
    ".ckpt": "checkpoint (.ckpt)",
    ".gguf": "GGUF (.gguf, used by llama.cpp and Ollama)",
    ".ggml": "GGML (.ggml)",
    ".onnx": "ONNX (.onnx)",
    ".h5": "Keras/TensorFlow (.h5)",
    ".msgpack": "Flax (.msgpack)",
}

PARTIAL_DOWNLOAD_SUFFIXES = (".part", ".incomplete", ".crdownload", ".download", ".tmp")

LORA_MARKERS = ("lora_A", "lora_B", "lora_up", "lora_down", "lora_embedding_A")


@dataclass(frozen=True)
class TensorInfo:
    dtype: str
    shape: Tuple[int, ...]

    @property
    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count


@dataclass
class WeightFile:
    path: Path
    size: int
    tensors: Dict[str, TensorInfo]

    @property
    def is_adapter(self) -> bool:
        return any(marker in key for key in self.tensors for marker in LORA_MARKERS)


@dataclass(frozen=True)
class ModelVariant:
    """A supported model layout and the bundled assets that describe it."""

    identifier: str
    display_name: str
    asset_dir: str
    hidden_size: int
    vocab_size: int
    component_prefixes: Tuple[Tuple[str, str], ...]


TURKMEN_OCR = ModelVariant(
    identifier="turkmen_ocr",
    display_name="TurkmenOCR",
    asset_dir="turkmen_ocr",
    hidden_size=1280,
    vocab_size=129280,
    component_prefixes=(
        ("model.sam_model.", "high-resolution vision encoder"),
        ("model.vision_model.", "semantic vision encoder"),
        ("model.projector.", "vision-to-text projector"),
        ("model.layers.", "text decoder layers"),
        ("model.embed_tokens.", "token embedding table"),
        ("lm_head.", "output head"),
    ),
)

SUPPORTED_VARIANTS: Tuple[ModelVariant, ...] = (TURKMEN_OCR,)


@dataclass
class WeightsReport:
    """Result of a successful inspection."""

    directory: Path
    files: List[WeightFile]
    variant: ModelVariant
    ignored_files: List[Path] = field(default_factory=list)

    @property
    def tensors(self) -> Dict[str, TensorInfo]:
        merged: Dict[str, TensorInfo] = {}
        for weight_file in self.files:
            merged.update(weight_file.tensors)
        return merged

    @property
    def parameter_count(self) -> int:
        return sum(info.numel for info in self.tensors.values())

    @property
    def total_file_size(self) -> int:
        return sum(weight_file.size for weight_file in self.files)

    def weight_map(self) -> Dict[str, str]:
        return {
            key: weight_file.path.name
            for weight_file in self.files
            for key in weight_file.tensors
        }

    def memory_required(self, bytes_per_parameter: int = 2) -> int:
        """Bytes needed to hold every tensor at the given precision."""
        return self.parameter_count * bytes_per_parameter


@dataclass(frozen=True)
class Detection:
    family: str
    purpose: str


def read_header(path: Path) -> WeightFile:
    """Parse the header of one safetensors file without reading tensor data."""
    size = path.stat().st_size
    if size == 0:
        raise TurkmenOCRError(
            "WEIGHTS-003",
            f"The weight file {path.name} is empty.",
            f"The file {path} exists but contains 0 bytes. This usually happens "
            "when a download or copy was interrupted before any data was written.",
            f"Delete {path.name} and download or copy it again from its original "
            "source, then run the same command again.",
        )
    try:
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) < 8:
                raise ValueError("file shorter than the 8-byte header length field")
            (header_length,) = struct.unpack("<Q", prefix)
            if header_length > HEADER_LIMIT or header_length > size - 8:
                raise ValueError("header length field is out of range")
            raw = handle.read(header_length)
        header = json.loads(raw.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError("header is not a JSON object")
        header.pop("__metadata__", None)
        tensors, data_end = _parse_entries(header)
    except PermissionError:
        raise
    except (ValueError, UnicodeDecodeError, struct.error) as exc:
        raise _corrupt_error(path, size, str(exc)) from exc

    expected = 8 + header_length + data_end
    if size < expected:
        raise TurkmenOCRError(
            "WEIGHTS-005",
            f"The weight file {path.name} is incomplete.",
            f"The header of {path} says the file should be {expected:,} bytes "
            f"({format_size(expected)}), but only {size:,} bytes "
            f"({format_size(size)}) are present on disk. The download or copy of "
            "this file stopped before it finished.",
            f"Delete {path.name} and download or copy it again. Make sure the "
            "transfer finishes completely and that the disk has enough free space "
            "before you retry.",
        )
    return WeightFile(path=path, size=size, tensors=tensors)


def _parse_entries(header: Dict[str, object]) -> Tuple[Dict[str, TensorInfo], int]:
    tensors: Dict[str, TensorInfo] = {}
    data_end = 0
    for key, entry in header.items():
        if not isinstance(entry, dict):
            raise ValueError(f"entry for tensor {key!r} is malformed")
        dtype = entry.get("dtype")
        shape = entry.get("shape")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
        ):
            raise ValueError(f"entry for tensor {key!r} is missing required fields")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start:
            raise ValueError(f"tensor {key!r} has an invalid byte range")
        info = TensorInfo(dtype=dtype, shape=tuple(int(dim) for dim in shape))
        itemsize = DTYPE_SIZES.get(dtype)
        if itemsize is not None and end - start != info.numel * itemsize:
            raise ValueError(f"tensor {key!r} byte range does not match its shape")
        tensors[key] = info
        data_end = max(data_end, end)
    return tensors, data_end


def _corrupt_error(path: Path, size: int, detail: str) -> TurkmenOCRError:
    return TurkmenOCRError(
        "WEIGHTS-004",
        f"The weight file {path.name} is damaged and cannot be read.",
        f"The file {path} is {size:,} bytes ({format_size(size)}) on disk, but its "
        f"internal table of contents could not be decoded ({detail}). This "
        "happens when a download was cut off early, when the file was saved "
        "incorrectly, or when a different kind of file was renamed to end in "
        ".safetensors.",
        f"Delete {path.name} and download or copy it again from its original "
        "source, then run the same command again.",
    )


def discover_weight_files(directory: Path) -> List[Path]:
    """Return the safetensors files in a directory, verifying shard sets."""
    entries = sorted(p for p in directory.iterdir() if p.is_file())
    weights = [p for p in entries if p.suffix.lower() == ".safetensors"]
    if not weights:
        raise _no_weights_error(directory, entries)
    _verify_shards(directory, weights)
    return weights


def _no_weights_error(directory: Path, entries: Sequence[Path]) -> TurkmenOCRError:
    partial = [p.name for p in entries if p.name.lower().endswith(PARTIAL_DOWNLOAD_SUFFIXES)]
    other = sorted(
        {OTHER_WEIGHT_FORMATS[p.suffix.lower()] for p in entries if p.suffix.lower() in OTHER_WEIGHT_FORMATS}
    )
    nested = sorted(
        sub.name
        for sub in directory.iterdir()
        if sub.is_dir() and any(sub.glob("*.safetensors"))
    )
    if partial:
        found = (
            "The folder contains files that look like unfinished downloads: "
            f"{_join(partial)}. A file only becomes usable once its download has "
            "finished and its name ends in .safetensors."
        )
        resolution = "Let the download finish completely, then run the same command again."
    elif other:
        found = (
            f"The folder contains model weights in a different file format: "
            f"{_join(other)}. turkmenOCR-lib only reads weights stored in the "
            "safetensors format, which is safer to open and faster to load."
        )
        resolution = (
            "Download the TurkmenOCR weights in safetensors format (files ending "
            "in .safetensors) and pass the folder that contains them with -path."
        )
    elif nested:
        found = (
            "No .safetensors files are stored directly in this folder, but these "
            f"subfolders contain some: {_join(nested)}."
        )
        resolution = (
            "Point -path at the subfolder that holds the .safetensors files, for "
            f'example -path="{directory / nested[0]}".'
        )
    else:
        found = "The folder does not contain any .safetensors files."
        resolution = (
            "Check that you passed the folder where the TurkmenOCR weight files "
            "(ending in .safetensors) are saved. If the files are still being "
            "downloaded or copied, wait for that to finish first."
        )
    return TurkmenOCRError(
        "WEIGHTS-001",
        "No safetensors weight files were found.",
        f"The folder {directory} was searched for files ending in .safetensors. "
        + found,
        resolution,
    )


def _verify_shards(directory: Path, weights: Sequence[Path]) -> None:
    groups: Dict[Tuple[str, str], List[Tuple[int, str]]] = {}
    for path in weights:
        match = SHARD_PATTERN.match(path.name)
        if match:
            key = (match.group("prefix"), match.group("total"))
            groups.setdefault(key, []).append((int(match.group("index")), match.group("index")))

    for (prefix, total_text), found in sorted(groups.items()):
        total = int(total_text)
        width = max(len(text) for _, text in found)
        present = {index for index, _ in found}
        missing = [
            f"{prefix}-{index:0{width}d}-of-{total_text}.safetensors"
            for index in range(1, total + 1)
            if index not in present
        ]
        if missing:
            raise TurkmenOCRError(
                "WEIGHTS-002",
                "Part of the model weights is missing.",
                f"The weights in {directory} are split into {total} parts named "
                f"{prefix}-NNNNN-of-{total_text}.safetensors, but "
                f"{len(missing)} of them could not be found: {_join(missing)}.",
                "Finish downloading or copying the missing part(s) into the same "
                "folder as the others, then run the same command again.",
            )


def inspect_directory(directory: Path, reporter: Optional[Reporter] = None) -> WeightsReport:
    """Inspect a weights directory and confirm it holds a supported model."""
    reporter = reporter or Reporter(quiet=True)
    paths = discover_weight_files(directory)
    files: List[WeightFile] = []
    for path in paths:
        try:
            files.append(read_header(path))
        except PermissionError as exc:
            raise TurkmenOCRError(
                "INPUT-003",
                f"The weight file {path.name} cannot be opened.",
                f"Your user account does not have permission to read {path}.",
                "Give your user account read access to this file (on Windows: "
                "right-click the file, choose Properties, then Security; on Linux "
                f'or macOS: run chmod a+r "{path}"), then run the command again.',
            ) from exc

    adapters = [f for f in files if f.is_adapter]
    full = [f for f in files if not f.is_adapter]
    if not full:
        raise _adapter_error(directory, adapters)
    for adapter in adapters:
        reporter.warning(
            f"Ignoring {adapter.path.name} because it is a fine-tuning adapter, not "
            "a full set of model weights."
        )

    _reject_duplicates(full)
    tensors: Dict[str, TensorInfo] = {}
    for weight_file in full:
        tensors.update(weight_file.tensors)

    variant = identify_variant(tensors, directory)
    return WeightsReport(
        directory=directory,
        files=full,
        variant=variant,
        ignored_files=[a.path for a in adapters],
    )


def _reject_duplicates(files: Sequence[WeightFile]) -> None:
    owner: Dict[str, Path] = {}
    for weight_file in files:
        for key in weight_file.tensors:
            if key in owner:
                raise TurkmenOCRError(
                    "WEIGHTS-006",
                    "The folder contains more than one copy of the model weights.",
                    f"The tensor {key} appears in both {owner[key].name} and "
                    f"{weight_file.path.name}. The folder probably holds two "
                    "different downloads or versions of a model side by side, and "
                    "it is not possible to tell which one should be used.",
                    "Keep only one complete set of .safetensors files in the "
                    "folder. Move the other files to a separate folder, then run "
                    "the same command again.",
                )
            owner[key] = weight_file.path


def _adapter_error(directory: Path, adapters: Sequence[WeightFile]) -> TurkmenOCRError:
    return TurkmenOCRError(
        "MODEL-002",
        "The folder contains only a fine-tuning adapter, not a full model.",
        f"The file(s) {_join([a.path.name for a in adapters])} in {directory} "
        "contain LoRA adapter weights. An adapter stores only the small changes "
        "made during fine-tuning and cannot run on its own without the full "
        "model it was trained on.",
        "Merge the adapter into the full TurkmenOCR weights and save the merged "
        "model as .safetensors files (with PEFT this is done with "
        "merge_and_unload() followed by save_pretrained()). Then point -path at "
        "the folder that contains the merged files.",
    )


@lru_cache(maxsize=None)
def expected_tensor_names(variant: ModelVariant) -> frozenset:
    """Tensor names listed in the bundled index for a variant."""
    index = resources.files("turkmenocr").joinpath("assets").joinpath(variant.asset_dir)
    data = json.loads(index.joinpath("model.safetensors.index.json").read_text(encoding="utf-8"))
    return frozenset(data["weight_map"])


def identify_variant(tensors: Dict[str, TensorInfo], directory: Path) -> ModelVariant:
    """Match tensors against supported variants or explain what was found."""
    if _looks_like_turkmen_ocr(tensors):
        variant = TURKMEN_OCR
        _check_completeness(tensors, variant, directory)
        _check_dimensions(tensors, variant, directory)
        return variant

    detection = detect_architecture(tensors)
    raise TurkmenOCRError(
        "MODEL-001",
        "The weights in this folder belong to a different kind of model.",
        f"The folder {directory} contains {len(tensors):,} tensors whose names and "
        f"shapes match {detection.family}. That kind of model is designed for "
        f"{detection.purpose}. turkmenOCR-lib is built specifically for the "
        "TurkmenOCR architecture and cannot run other models, because their "
        "internal structure and expected inputs are different.",
        "Locate the folder where your TurkmenOCR weights are saved (it contains "
        "a .safetensors file of roughly 6.7 GB) and run the command again with "
        "-path pointing to that folder.",
    )


def _looks_like_turkmen_ocr(tensors: Dict[str, TensorInfo]) -> bool:
    keys = tensors.keys()
    if any(k.startswith(("model.sam_model.", "model.vision_model.")) for k in keys):
        return True
    embed = tensors.get("model.embed_tokens.weight")
    return embed is not None and embed.shape == (TURKMEN_OCR.vocab_size, TURKMEN_OCR.hidden_size)


def _check_completeness(tensors: Dict[str, TensorInfo], variant: ModelVariant, directory: Path) -> None:
    expected = expected_tensor_names(variant)
    missing = sorted(expected.difference(tensors))
    if not missing:
        return
    absent_parts = [
        description
        for prefix, description in variant.component_prefixes
        if not any(k.startswith(prefix) for k in tensors)
    ]
    if absent_parts:
        what = "These parts of the model are absent entirely: " + _join(absent_parts) + "."
    else:
        what = "Examples of missing tensors: " + _join(missing[:5]) + "."
    raise TurkmenOCRError(
        "MODEL-003",
        f"The {variant.display_name} weights in this folder are incomplete.",
        f"The files in {directory} look like {variant.display_name} weights, but "
        f"{len(missing):,} of the {len(expected):,} tensors the model needs are "
        f"missing. {what} This happens when only part of a model was saved, for "
        "example the text decoder without the vision encoder.",
        f"Download or export the complete {variant.display_name} weights (about "
        "6.7 GB in total) into the folder and run the command again.",
    )


def _check_dimensions(tensors: Dict[str, TensorInfo], variant: ModelVariant, directory: Path) -> None:
    embed = tensors.get("model.embed_tokens.weight")
    if embed is None or len(embed.shape) != 2:
        return
    vocab, hidden = embed.shape
    if (vocab, hidden) == (variant.vocab_size, variant.hidden_size):
        return
    raise TurkmenOCRError(
        "MODEL-004",
        f"The weights do not match the {variant.display_name} model size.",
        f"The token embedding table in {directory} has {vocab:,} entries of width "
        f"{hidden:,}, while {variant.display_name} uses {variant.vocab_size:,} "
        f"entries of width {variant.hidden_size:,}. The weights were produced by "
        "a model with a different vocabulary or size, so the bundled "
        "configuration cannot describe them.",
        f"Use the standard {variant.display_name} weights, or fine-tuned weights "
        "that kept the original vocabulary, and run the command again.",
    )


def detect_architecture(tensors: Dict[str, TensorInfo]) -> Detection:
    """Name the model family that a set of tensors most likely belongs to."""
    keys = list(tensors)

    def has(*fragments: str) -> bool:
        return any(fragment in key for key in keys for fragment in fragments)

    if has("visual.blocks.", "visual.merger.") or any(k.startswith("transformer.visual.") for k in keys):
        return Detection(
            "a Qwen-VL / Qwen2-VL vision-language model",
            "general image understanding and conversational question answering",
        )
    if has("vision_tower") and has("multi_modal_projector"):
        return Detection(
            "a LLaVA or Pixtral vision-language model",
            "general image understanding and conversational question answering",
        )
    if (has("vision_model.") and has("aligner.") and has("language_model.")) or (
        any(k.startswith("vision.") for k in keys)
        and any(k.startswith("projector.") for k in keys)
        and any(k.startswith("language.") for k in keys)
    ):
        return Detection(
            "a general-purpose vision-language chat model",
            "describing images and answering questions about them in a chat",
        )
    if has("text_model.") and has("vision_model."):
        return Detection(
            "a CLIP or SigLIP image-text embedding model",
            "measuring how well images and captions match, not reading text",
        )
    if has("down_blocks.", "up_blocks.", "mid_block.", "time_embed", "diffusion_model.",
           "double_blocks.", "single_blocks.", "joint_blocks."):
        return Detection(
            "an image generation (diffusion) model such as Stable Diffusion, SDXL or Flux",
            "creating new images from text prompts",
        )
    if has("encoder.conv1.") and has("decoder.embed_positions"):
        return Detection("a Whisper speech recognition model", "transcribing audio recordings")
    if any(re.search(r"layers\.\d+\.self_attn\.", k) or re.search(r"\.h\.\d+\.attn\.", k) for k in keys):
        return Detection(_text_model_family(tensors), "generating and completing text from text prompts")
    return Detection(
        "an unrecognised model (top-level tensor groups: "
        + _join(sorted({k.split(".")[0] for k in keys})[:6])
        + ")",
        "a purpose other than TurkmenOCR text recognition",
    )


_VOCAB_FAMILIES: Dict[int, str] = {
    32000: "a LLaMA 2 or Mistral text-only language model",
    32768: "a Mistral text-only language model",
    128256: "a LLaMA 3 text-only language model",
    151936: "a Qwen text-only language model",
    152064: "a Qwen text-only language model",
    256000: "a Gemma text-only language model",
    262144: "a Gemma text-only language model",
}


def _text_model_family(tensors: Dict[str, TensorInfo]) -> str:
    for key, info in tensors.items():
        if key.endswith(("embed_tokens.weight", "wte.weight", "word_embeddings.weight")) and info.shape:
            return _VOCAB_FAMILIES.get(info.shape[0], "a text-only large language model")
    return "a text-only large language model"


def _join(items: Iterable[str]) -> str:
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]
