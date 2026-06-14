"""
Default prompt loader.

Prompts that drive the Idea Engine (expansion, evaluation, merge, final) live
on disk as `.md` files under `prompts/defaults/`. The engine's settings dict
takes precedence: if a key is present in the loaded settings.json (or env
override), that wins. Only keys absent from settings fall back to the disk
defaults.

This indirection lets prompts be versioned as documents rather than buried
inside a JSON blob. The settings.json copies remain for backwards compat;
callers can drop them once the file-based source is the only one in use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional


DEFAULTS_DIR = Path(__file__).resolve().parent / "defaults"


PROMPT_KEYS = (
    "expansion_system_prompt",
    "expansion_user_prompt",
    "expansion_planning_addendum",
    "evaluation_system_prompt",
    "evaluation_user_prompt",
    "final_system_prompt",
    "final_user_prompt",
    "merge_system_prompt",
    "merge_user_prompt",
)


def _read_prompt(key: str, directory: Optional[Path] = None) -> Optional[str]:
    base = directory or DEFAULTS_DIR
    path = base / f"{key}.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def load_default_prompts(directory: Optional[Path] = None) -> Dict[str, str]:
    """Return all prompts found on disk under `directory` (default: package defaults)."""
    out: Dict[str, str] = {}
    for key in PROMPT_KEYS:
        text = _read_prompt(key, directory)
        if text is not None:
            out[key] = text
    return out


def apply_default_prompts(settings: Dict[str, object], directory: Optional[Path] = None) -> Dict[str, object]:
    """
    Mutate `settings` in place to fill in any missing prompt keys from disk.

    Existing values in `settings` are never overwritten. Returns the same dict
    for convenience.
    """
    defaults = load_default_prompts(directory)
    for key, value in defaults.items():
        if key not in settings or settings.get(key) is None or settings.get(key) == "":
            settings[key] = value
    return settings
