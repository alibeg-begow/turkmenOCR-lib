from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import pytest

from turkmenocr.inspector import DTYPE_SIZES

TensorSpec = Tuple[str, Sequence[int]]


def write_safetensors(
    path: Path,
    tensors: Dict[str, TensorSpec],
    truncate_by: int = 0,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """Write a minimal but valid safetensors file filled with zeros."""
    header: Dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        size = numel * DTYPE_SIZES[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if metadata is not None:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(raw)))
        handle.write(raw)
        handle.write(b"\0" * max(offset - truncate_by, 0))
    return path


@pytest.fixture
def make_weights():
    return write_safetensors


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("TURKMENOCR_CACHE_DIR", str(tmp_path_factory.mktemp("cache")))
