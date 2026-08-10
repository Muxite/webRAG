#!/usr/bin/env python3
"""Render one agent/app/idea_code_tests/test_c*.py task module to an on-disk
fixture tree that run_agent_sandbox.sh / run_grade.sh can mount and consume.

Task module contract (see agent/app/idea_code_tests/test_c01_*.py for the first
concrete example):
    get_test_metadata() -> {"test_id": str, "category": "hard"|"soft", ...}
    get_task_statement() -> str                       # becomes public/prompt.md
    get_visibility() -> "visible" | "hidden"
    get_sandbox_fixture() -> {relpath: content}        # becomes public/repo/<relpath>
    get_grading_payload() -> {
        "tests": {relpath: content},                   # becomes private/tests/<relpath>
        "entrypoint": {...},                           # informational, written into meta.json
        "keystone_test_ids": [str, ...],
    }
    get_judge_rubric() -> dict | None                  # soft tasks only

Output layout:
    tasks/<id>/public/prompt.md      <- mounted read-only as /task in BOTH agent sandboxes
    tasks/<id>/public/repo/...       <- starter files (visible test included, if visible)
    tasks/<id>/private/tests/...     <- canonical tests. NEVER mounted into an agent sandbox.
    tasks/<id>/private/test_manifest.json  <- globs of agent-writable paths run_grade.sh must
                                               discard before trusting the submission
    tasks/<id>/meta.json             <- category/visibility/keystone_test_ids/entrypoint

Usage: PYTHONPATH=. python codebench/materialize_task.py c01 --out codebench/tasks
"""
from __future__ import annotations

import argparse
import importlib
import json
import pkgutil
import shutil
from pathlib import Path


def _resolve_module_name(test_id: str) -> str:
    """Task modules are test_<id>_<slug>.py (e.g. test_c01_nth_even_fib_sum.py),
    mirroring idea_tests/'s test_NNN_<slug>.py convention — glob rather than assume
    the suffix. Imported as agent.app.idea_code_tests.* (the PYTHONPATH=. host
    convention this repo's own test invocation already uses — see project memory
    "Tests need PYTHONPATH=."), NOT app.idea_code_tests.*, which is a container-only
    alias that only resolves via the Docker image's WORKDIR=/app/agent CWD trick."""
    import agent.app.idea_code_tests as pkg

    matches = [
        name
        for _, name, _ in pkgutil.iter_modules(pkg.__path__)
        if name == f"test_{test_id}" or name.startswith(f"test_{test_id}_")
    ]
    if not matches:
        raise ModuleNotFoundError(f"no idea_code_tests module for test_id={test_id!r}")
    if len(matches) > 1:
        raise ModuleNotFoundError(f"ambiguous test_id={test_id!r}: matches {matches}")
    return f"agent.app.idea_code_tests.{matches[0]}"


def _write_tree(root: Path, files: dict[str, str]) -> list[str]:
    written = []
    for rel, content in files.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
        written.append(rel)
    return written


def materialize(test_id: str, out_root: Path) -> Path:
    module = importlib.import_module(_resolve_module_name(test_id))

    meta = module.get_test_metadata()
    statement = module.get_task_statement()
    visibility = module.get_visibility()
    fixture = module.get_sandbox_fixture()
    grading = module.get_grading_payload()
    rubric = module.get_judge_rubric() if hasattr(module, "get_judge_rubric") else None

    task_dir = out_root / test_id
    if task_dir.exists():
        shutil.rmtree(task_dir)
    public_dir = task_dir / "public"
    private_dir = task_dir / "private"
    (public_dir / "repo").mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    (public_dir / "prompt.md").write_text(statement)
    fixture_paths = _write_tree(public_dir / "repo", fixture)

    # The compiled plan is safe to expose (offline-authored instructions, not answers) —
    # rendered as data so codebench_run_task.py never needs to import the task module
    # (and the module, with its canonical hidden-test content, is stripped from the agent
    # image entirely — see agents/badmodel/Dockerfile's SECURITY comment).
    (public_dir / "plan.json").write_text(json.dumps(module.get_compiled_plan(), indent=2))

    # grading["tests"] keys are already workdir-relative (e.g. "tests/test_fib_mod.py" — the
    # SAME namespace get_sandbox_fixture() uses), so write them directly under private_dir,
    # not private_dir/"tests" — that would double the "tests/" prefix (private/tests/tests/...)
    # and break both the canonical file's importability (pytest's own basedir would land two
    # levels off from where fib_mod.py needs to be on sys.path) and keystone_test_ids node-id
    # matching (score_and_record.py looks up the ORIGINAL "tests/test_fib_mod.py::..." ids).
    test_paths = _write_tree(private_dir, grading["tests"])

    # Any path the agent could plant a fake copy of a canonical test at — the visible
    # fixture's own copy (same relpath) plus the canonical test paths themselves.
    manifest_globs = sorted(set(test_paths) | {p for p in fixture_paths if p in test_paths})
    (private_dir / "test_manifest.json").write_text(
        json.dumps({"test_file_globs": manifest_globs}, indent=2)
    )

    meta_out = {
        **meta,
        "visibility": visibility,
        "keystone_test_ids": grading.get("keystone_test_ids", []),
        "entrypoint": grading.get("entrypoint", {}),
        "has_rubric": rubric is not None,
    }
    (task_dir / "meta.json").write_text(json.dumps(meta_out, indent=2))
    if rubric is not None:
        (task_dir / "rubric.json").write_text(json.dumps(rubric, indent=2))

    print(f"materialize_task: {test_id} -> {task_dir}")
    return task_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("test_id", help="e.g. c01 (module test_c01_*.py under idea_code_tests/)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    materialize(args.test_id, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
