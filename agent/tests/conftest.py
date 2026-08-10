"""Shared pytest fixtures for agent/tests/."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def codebench_materialize_script() -> Path:
    """Path to codebench's materialize_task.py, shared so a future move only needs updating here."""
    return Path(__file__).resolve().parent.parent.parent / "codebench" / "materialize_task.py"
