import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


def load_tool(name):
    path = Path(__file__).resolve().parents[1] / "tools" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepared_model_edits_are_rejected(tmp_path, monkeypatch):
    tool = load_tool("prepare_sources")
    monkeypatch.setattr(tool, "DEPENDENCIES", tmp_path)
    model = tmp_path / "miniLLAMA"
    model.mkdir()
    (model / "model.py").write_text("# prepared model\n")
    expected = {"miniLLAMA": {"revision": "pinned", "patch_sha256": "patch"}}
    manifest = {"inputs": expected,
                "files": {"miniLLAMA": tool.file_manifest(model)}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    tool.verify_prepared(expected)
    (model / "model.py").write_text("# unexpected edit\n")
    with pytest.raises(ValueError, match="has changed"):
        tool.verify_prepared(expected)


def test_prepare_applies_model_patch_inside_a_checkout_with_existing_cache(tmp_path, monkeypatch):
    tool = load_tool("prepare_sources")
    course = tool.MP3_ROOT
    original = tool.REPO_ROOT / "miniLLAMA"
    repo = tmp_path / "checkout"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", repo], check=True)
    (repo / "miniLLAMA").symlink_to(original.resolve(), target_is_directory=True)
    mp3 = repo / "mp3"
    (mp3 / "patches").mkdir(parents=True)
    shutil.copy2(course / "patches/miniLLAMA.patch", mp3 / "patches/miniLLAMA.patch")
    spec = json.loads((course / "vendor_versions.json").read_text())["miniLLAMA"]
    (mp3 / "vendor_versions.json").write_text(json.dumps({"miniLLAMA": spec}))
    deps = mp3 / ".deps"
    (deps / "rtl_build").mkdir(parents=True)
    (deps / "rtl_build/keep.txt").write_text("existing cache")
    monkeypatch.setattr(tool, "REPO_ROOT", repo)
    monkeypatch.setattr(tool, "MP3_ROOT", mp3)
    monkeypatch.setattr(tool, "DEPENDENCIES", deps)
    monkeypatch.setattr(sys, "argv", ["prepare_sources.py", "--only", "miniLLAMA"])
    assert tool.main() == 0
    model = (deps / "miniLLAMA/model.py").read_text()
    assert "from .utils import create_masks" in model
    assert "matmul_executor" in model
    assert (deps / "rtl_build/keep.txt").read_text() == "existing cache"
    assert tool.main() == 0
