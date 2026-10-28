import json
import os

import pytest

from turkmenocr import inspector, workspace
from turkmenocr.diagnostics import TurkmenOCRError
from turkmenocr.workspace import INDEX_NAME, LOCAL_WORKSPACE_NAME, prepare_workspace

NAMES = ["model.sam_model.a", "model.vision_model.b", "model.projector.layers.weight", "model.layers.0.c"]


@pytest.fixture
def report(tmp_path, make_weights, monkeypatch):
    monkeypatch.setattr(inspector, "expected_tensor_names", lambda variant: frozenset(NAMES))
    weights = tmp_path / "weights"
    weights.mkdir()
    make_weights(weights / "turkmen.safetensors", {name: ("BF16", [2]) for name in NAMES})
    return inspector.inspect_directory(weights)


def refuse(*args, **kwargs):
    raise OSError("links are not permitted")


def test_linked_workspace(report):
    with prepare_workspace(report) as ws:
        assert ws.mode in ("symlink", "hardlink")
        assert (ws.path / "config.json").is_file()
        assert (ws.path / "modeling_turkmenocr.py").is_file()
        assert (ws.path / "turkmen.safetensors").is_file()
        index = json.loads((ws.path / INDEX_NAME).read_text(encoding="utf-8"))
        assert set(index["weight_map"].values()) == {"turkmen.safetensors"}
        assert sorted(index["weight_map"]) == sorted(NAMES)
        folder = ws.path
    assert not folder.exists()
    assert (report.directory / "turkmen.safetensors").is_file()


def test_bundled_config_uses_turkmen_names(report):
    config = json.loads(workspace.bundled_assets("turkmen_ocr")["config.json"])
    assert config["auto_map"]["AutoModel"] == "modeling_turkmenocr.TurkmenOCRForCausalLM"


def test_hardlink_fallback(report, monkeypatch):
    monkeypatch.setattr(os, "symlink", refuse)
    with prepare_workspace(report) as ws:
        assert ws.mode == "hardlink"
        linked = (ws.path / "turkmen.safetensors").stat().st_size
        assert linked == (report.directory / "turkmen.safetensors").stat().st_size


def test_local_subfolder_fallback(report, monkeypatch):
    monkeypatch.setattr(os, "symlink", refuse)
    real_link = os.link

    def link_only_inside_weights(src, dst):
        if LOCAL_WORKSPACE_NAME not in str(dst):
            raise OSError("different drive")
        real_link(src, dst)

    monkeypatch.setattr(os, "link", link_only_inside_weights)
    with prepare_workspace(report) as ws:
        assert ws.mode == "local-hardlink"
        assert ws.path == report.directory / LOCAL_WORKSPACE_NAME
    assert not (report.directory / LOCAL_WORKSPACE_NAME).exists()


def test_in_place_fallback_is_non_destructive(report, monkeypatch):
    monkeypatch.setattr(os, "symlink", refuse)
    monkeypatch.setattr(os, "link", refuse)
    user_file = report.directory / "special_tokens_map.json"
    user_file.write_text("{}", encoding="utf-8")
    with prepare_workspace(report) as ws:
        assert ws.mode == "in-place"
        assert (report.directory / "config.json").is_file()
        assert user_file not in ws.created
    remaining = sorted(p.name for p in report.directory.iterdir())
    assert remaining == ["special_tokens_map.json", "turkmen.safetensors"]
    assert user_file.read_text(encoding="utf-8") == "{}"


def test_mismatched_existing_index(report, monkeypatch):
    monkeypatch.setattr(os, "symlink", refuse)
    monkeypatch.setattr(os, "link", refuse)
    (report.directory / INDEX_NAME).write_text(json.dumps({"weight_map": {"x": "other.safetensors"}}), encoding="utf-8")
    with pytest.raises(TurkmenOCRError) as excinfo:
        prepare_workspace(report)
    assert excinfo.value.code == "WORKSPACE-002"
    assert not (report.directory / "config.json").exists()


def test_read_only_folder(report, monkeypatch):
    monkeypatch.setattr(os, "symlink", refuse)
    monkeypatch.setattr(os, "link", refuse)
    monkeypatch.setattr(workspace, "_is_writable", lambda directory: False)
    with pytest.raises(TurkmenOCRError) as excinfo:
        prepare_workspace(report)
    assert excinfo.value.code == "WORKSPACE-001"
    assert excinfo.value.steps


def test_foreign_folder_is_never_deleted(report, tmp_path, monkeypatch):
    monkeypatch.setenv("TURKMENOCR_CACHE_DIR", str(tmp_path / "cache2"))
    target = workspace.cache_root() / "workspaces" / workspace.workspace_name(report.directory)
    target.mkdir(parents=True)
    keep = target / "keep.txt"
    keep.write_text("mine", encoding="utf-8")
    with prepare_workspace(report) as ws:
        assert ws.path != target
    assert keep.read_text(encoding="utf-8") == "mine"
