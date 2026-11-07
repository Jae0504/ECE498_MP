from types import SimpleNamespace

import pytest

from test_prepare_sources import load_tool


def test_empty_submodule_directory_is_initialized(tmp_path, monkeypatch):
    tool = load_tool("setup_rtl")
    monkeypatch.setattr(tool, "REPO", tmp_path)
    (tmp_path / ".git").mkdir()
    path = tmp_path / "cocotb"
    path.mkdir()
    calls = []

    def initialize(args, cwd=None):
        calls.append((args, cwd))
        (path / ".git").write_text("gitdir: test\n")

    monkeypatch.setattr(tool, "run", initialize)
    monkeypatch.setattr(tool.subprocess, "check_output", lambda *args, **kwargs: tool.PINS["cocotb"][1])
    monkeypatch.setattr(tool.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0))
    assert tool.source("cocotb") == path
    assert calls == [(["git", "submodule", "update", "--init", "--", "cocotb"], tmp_path)]


def test_unversioned_tool_files_are_preserved(tmp_path, monkeypatch):
    tool = load_tool("setup_rtl")
    monkeypatch.setattr(tool, "REPO", tmp_path)
    (tmp_path / ".git").mkdir()
    path = tmp_path / "verilator"
    path.mkdir()
    (path / "local.txt").write_text("keep this")
    with pytest.raises(RuntimeError, match="Preserving"):
        tool.source("verilator")
    assert (path / "local.txt").read_text() == "keep this"
