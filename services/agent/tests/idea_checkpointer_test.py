"""
Unit tests for FileCheckpointer: round-trip and listing.
"""
from __future__ import annotations

import os
import tempfile

import pytest

from agent.app.idea_checkpointer import FileCheckpointer, create_checkpointer_from_env


@pytest.mark.asyncio
async def test_file_checkpointer_round_trip(tmp_path):
    cp = FileCheckpointer(root_dir=str(tmp_path))
    snapshot = {"graph": {"root_id": "r", "nodes": {}}, "current_id": "r"}
    await cp.save("run-1", 0, snapshot)
    loaded = await cp.load("run-1")
    assert loaded is not None
    assert loaded["run_id"] == "run-1"
    assert loaded["step_index"] == 0
    assert loaded["snapshot"]["current_id"] == "r"


@pytest.mark.asyncio
async def test_file_checkpointer_latest_wins(tmp_path):
    cp = FileCheckpointer(root_dir=str(tmp_path))
    await cp.save("run-2", 0, {"graph": {"v": 0}})
    await cp.save("run-2", 1, {"graph": {"v": 1}})
    await cp.save("run-2", 2, {"graph": {"v": 2}})
    loaded = await cp.load("run-2")
    assert loaded["step_index"] == 2
    assert loaded["snapshot"]["graph"]["v"] == 2


@pytest.mark.asyncio
async def test_file_checkpointer_load_missing_returns_none(tmp_path):
    cp = FileCheckpointer(root_dir=str(tmp_path))
    assert await cp.load("never-saved") is None


@pytest.mark.asyncio
async def test_file_checkpointer_list_runs(tmp_path):
    cp = FileCheckpointer(root_dir=str(tmp_path))
    await cp.save("a", 0, {"g": 1})
    await cp.save("b", 0, {"g": 2})
    runs = set(await cp.list_runs())
    assert {"a", "b"}.issubset(runs)


@pytest.mark.asyncio
async def test_file_checkpointer_delete_removes_run(tmp_path):
    cp = FileCheckpointer(root_dir=str(tmp_path))
    await cp.save("r", 0, {"g": 1})
    await cp.delete("r")
    assert await cp.load("r") is None


def test_create_checkpointer_disabled_by_default(monkeypatch):
    monkeypatch.delenv("IDEA_CHECKPOINT_ENABLED", raising=False)
    assert create_checkpointer_from_env() is None


def test_create_checkpointer_file_when_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("IDEA_CHECKPOINT_ENABLED", "1")
    monkeypatch.setenv("IDEA_CHECKPOINT_BACKEND", "file")
    monkeypatch.setenv("IDEA_CHECKPOINT_DIR", str(tmp_path))
    cp = create_checkpointer_from_env()
    assert isinstance(cp, FileCheckpointer)
    assert cp.root_dir == str(tmp_path)
