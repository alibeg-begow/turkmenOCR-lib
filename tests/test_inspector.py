import pytest

from turkmenocr import inspector
from turkmenocr.diagnostics import TurkmenOCRError
from turkmenocr.inspector import (
    TURKMEN_OCR,
    TensorInfo,
    detect_architecture,
    discover_weight_files,
    expected_tensor_names,
    identify_variant,
    inspect_directory,
    read_header,
)


def turkmen_tensors():
    tensors = {name: TensorInfo("BF16", (1,)) for name in expected_tensor_names(TURKMEN_OCR)}
    tensors["model.embed_tokens.weight"] = TensorInfo("BF16", (129280, 1280))
    return tensors


def error_code(excinfo):
    return excinfo.value.code


def test_bundled_index_lists_every_component():
    names = expected_tensor_names(TURKMEN_OCR)
    assert len(names) == 2710
    for prefix, _ in TURKMEN_OCR.component_prefixes:
        assert any(name.startswith(prefix) for name in names)


def test_read_header_parses_shapes(tmp_path, make_weights):
    tensors = {"w": ("F16", [2, 3]), "b": ("F32", [3])}
    path = make_weights(tmp_path / "a.safetensors", tensors, metadata={"format": "pt"})
    weight_file = read_header(path)
    assert weight_file.tensors["w"] == TensorInfo("F16", (2, 3))
    assert weight_file.tensors["w"].numel == 6
    assert "__metadata__" not in weight_file.tensors


def test_truncated_file_is_reported_with_sizes(tmp_path, make_weights):
    path = make_weights(tmp_path / "a.safetensors", {"w": ("F32", [100])}, truncate_by=10)
    with pytest.raises(TurkmenOCRError) as excinfo:
        read_header(path)
    assert error_code(excinfo) == "WEIGHTS-005"
    assert "a.safetensors" in excinfo.value.explanation


def test_garbage_file_is_reported_as_damaged(tmp_path):
    path = tmp_path / "a.safetensors"
    path.write_bytes(b"this is not a safetensors file at all")
    with pytest.raises(TurkmenOCRError) as excinfo:
        read_header(path)
    assert error_code(excinfo) == "WEIGHTS-004"


def test_empty_file(tmp_path):
    path = tmp_path / "a.safetensors"
    path.write_bytes(b"")
    with pytest.raises(TurkmenOCRError) as excinfo:
        read_header(path)
    assert error_code(excinfo) == "WEIGHTS-003"


def test_other_formats_are_explained(tmp_path):
    (tmp_path / "pytorch_model.bin").write_bytes(b"x")
    with pytest.raises(TurkmenOCRError) as excinfo:
        discover_weight_files(tmp_path)
    assert error_code(excinfo) == "WEIGHTS-001"
    assert "PyTorch pickle (.bin)" in excinfo.value.explanation


def test_nested_folder_is_suggested(tmp_path, make_weights):
    (tmp_path / "inner").mkdir()
    make_weights(tmp_path / "inner" / "model.safetensors", {"w": ("F16", [1])})
    with pytest.raises(TurkmenOCRError) as excinfo:
        discover_weight_files(tmp_path)
    assert "inner" in excinfo.value.resolution


def test_partial_download_is_explained(tmp_path):
    (tmp_path / "model.safetensors.part").write_bytes(b"x")
    with pytest.raises(TurkmenOCRError) as excinfo:
        discover_weight_files(tmp_path)
    assert "unfinished downloads" in excinfo.value.explanation


def test_missing_shard_is_named(tmp_path, make_weights):
    make_weights(tmp_path / "model-00001-of-00003.safetensors", {"a": ("F16", [1])})
    make_weights(tmp_path / "model-00003-of-00003.safetensors", {"c": ("F16", [1])})
    with pytest.raises(TurkmenOCRError) as excinfo:
        discover_weight_files(tmp_path)
    assert error_code(excinfo) == "WEIGHTS-002"
    assert "model-00002-of-00003.safetensors" in excinfo.value.explanation


def test_uneven_shard_padding_is_accepted(tmp_path, make_weights):
    make_weights(tmp_path / "model-00001-of-000001.safetensors", {"a": ("F16", [1])})
    assert len(discover_weight_files(tmp_path)) == 1


def test_turkmen_weights_are_identified(tmp_path):
    assert identify_variant(turkmen_tensors(), tmp_path) is TURKMEN_OCR


def test_incomplete_turkmen_weights(tmp_path):
    tensors = {k: v for k, v in turkmen_tensors().items() if not k.startswith("model.sam_model.")}
    with pytest.raises(TurkmenOCRError) as excinfo:
        identify_variant(tensors, tmp_path)
    assert error_code(excinfo) == "MODEL-003"
    assert "high-resolution vision encoder" in excinfo.value.explanation


def test_text_decoder_only_is_reported_as_incomplete(tmp_path):
    tensors = {k: v for k, v in turkmen_tensors().items() if k.startswith(("model.layers.", "model.embed_tokens"))}
    with pytest.raises(TurkmenOCRError) as excinfo:
        identify_variant(tensors, tmp_path)
    assert error_code(excinfo) == "MODEL-003"


def test_wrong_vocabulary_size(tmp_path):
    tensors = turkmen_tensors()
    tensors["model.embed_tokens.weight"] = TensorInfo("BF16", (130000, 1280))
    with pytest.raises(TurkmenOCRError) as excinfo:
        identify_variant(tensors, tmp_path)
    assert error_code(excinfo) == "MODEL-004"


@pytest.mark.parametrize(
    "keys, expected",
    [
        (["visual.blocks.0.attn.qkv.weight", "visual.merger.mlp.0.weight", "model.layers.0.self_attn.q_proj.weight"],
         "Qwen"),
        (["vision_tower.encoder.weight", "multi_modal_projector.linear_1.weight"], "LLaVA"),
        (["down_blocks.0.resnets.0.conv1.weight", "mid_block.attentions.0.proj.weight"], "diffusion"),
        (["double_blocks.0.img_attn.qkv.weight", "single_blocks.0.linear1.weight"], "diffusion"),
        (["text_model.encoder.layers.0.mlp.fc1.weight", "vision_model.encoder.layers.0.mlp.fc1.weight"], "CLIP"),
        (["foo.bar", "baz.qux"], "unrecognised"),
    ],
)
def test_other_architectures(keys, expected):
    tensors = {key: TensorInfo("F16", (4, 4)) for key in keys}
    assert expected in detect_architecture(tensors).family


def test_text_llm_family_from_vocabulary():
    tensors = {
        "model.embed_tokens.weight": TensorInfo("BF16", (128256, 4096)),
        "model.layers.0.self_attn.q_proj.weight": TensorInfo("BF16", (4096, 4096)),
    }
    assert "LLaMA 3" in detect_architecture(tensors).family


def test_unsupported_model_message(tmp_path, make_weights):
    make_weights(
        tmp_path / "model.safetensors",
        {
            "model.embed_tokens.weight": ("F16", [32000, 8]),
            "model.layers.0.self_attn.q_proj.weight": ("F16", [8, 8]),
        },
    )
    with pytest.raises(TurkmenOCRError) as excinfo:
        inspect_directory(tmp_path)
    error = excinfo.value
    assert error.code == "MODEL-001"
    assert "LLaMA 2 or Mistral" in error.explanation
    assert "TurkmenOCR architecture" in error.explanation
    assert "deepseek" not in (error.summary + error.explanation + error.resolution).lower()


def test_adapter_only(tmp_path, make_weights):
    adapter_key = "base_model.model.layers.0.q_proj.lora_A.weight"
    make_weights(tmp_path / "adapter_model.safetensors", {adapter_key: ("F16", [4, 8])})
    with pytest.raises(TurkmenOCRError) as excinfo:
        inspect_directory(tmp_path)
    assert error_code(excinfo) == "MODEL-002"


def test_duplicate_tensors(tmp_path, make_weights):
    make_weights(tmp_path / "a.safetensors", {"x": ("F16", [1])})
    make_weights(tmp_path / "b.safetensors", {"x": ("F16", [1])})
    with pytest.raises(TurkmenOCRError) as excinfo:
        inspect_directory(tmp_path)
    assert error_code(excinfo) == "WEIGHTS-006"


def test_full_inspection_of_sharded_turkmen_weights(tmp_path, make_weights, monkeypatch):
    names = ["model.sam_model.a", "model.vision_model.b", "model.projector.layers.weight", "model.layers.0.c"]
    monkeypatch.setattr(inspector, "expected_tensor_names", lambda variant: frozenset(names))
    make_weights(tmp_path / "model-00001-of-00002.safetensors", {names[0]: ("BF16", [2]), names[1]: ("BF16", [2])})
    make_weights(tmp_path / "model-00002-of-00002.safetensors", {names[2]: ("BF16", [2]), names[3]: ("BF16", [2])})
    report = inspect_directory(tmp_path)
    assert report.variant is TURKMEN_OCR
    assert report.parameter_count == 8
    assert report.weight_map()[names[3]] == "model-00002-of-00002.safetensors"
