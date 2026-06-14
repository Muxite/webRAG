"""Unit tests for `prompts.loader` — no external deps required."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_loader():
    here = Path(__file__).resolve().parent
    target = here.parent / "ideaengine" / "prompts" / "loader.py"
    spec = importlib.util.spec_from_file_location("_prompts_loader_under_test", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_prompts_loader_under_test"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_loader = _load_loader()


def test_load_default_prompts_finds_all_keys():
    prompts = _loader.load_default_prompts()
    # Every key in PROMPT_KEYS should resolve to a file.
    for key in _loader.PROMPT_KEYS:
        assert key in prompts, f"missing default for {key}"
        assert len(prompts[key]) > 0


def test_apply_does_not_overwrite_existing():
    settings = {
        "expansion_system_prompt": "EXISTING",
        "final_system_prompt": "",
    }
    _loader.apply_default_prompts(settings)
    assert settings["expansion_system_prompt"] == "EXISTING"
    # Empty string is treated as missing.
    assert settings["final_system_prompt"] != ""


def test_apply_fills_missing_keys():
    settings = {}
    _loader.apply_default_prompts(settings)
    for key in _loader.PROMPT_KEYS:
        assert key in settings


def test_apply_uses_custom_directory(tmp_path):
    custom = tmp_path / "custom_prompts"
    custom.mkdir()
    (custom / "expansion_system_prompt.md").write_text("CUSTOM-EXPANSION", encoding="utf-8")
    settings = {}
    _loader.apply_default_prompts(settings, directory=custom)
    assert settings.get("expansion_system_prompt") == "CUSTOM-EXPANSION"
    # Keys not in the custom dir remain absent.
    assert "merge_system_prompt" not in settings
