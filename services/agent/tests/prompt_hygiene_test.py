"""Tests for ``app.prompt_hygiene`` — the engine-agnostic self-contradictory-instruction lint
and the ``strip_source_ask`` transform it's built around (ported from
``execution_compiled._strip_source_ask``, see ``execution_compiled_leaf_extract_test.py`` for the
transform's own byte-level regression coverage, unchanged by this port).

Two things are covered here that aren't covered elsewhere:
  1. ``find_bare_output_contradiction`` itself — the new lint primitive.
  2. A corpus-wide regression guard: every hand-authored ``idea_tests/*.py`` compiled-plan leaf
     instruction AND every ``plan_library/templates/*.json`` leaf instruction, after
     ``strip_source_ask``, no longer trips the contradiction against the compiled path's real
     ``_THIN_EXTRACT_SYS`` directive. This is deliberately a POST-strip invariant, not a raw-corpus
     assertion — the raw corpus is KNOWN to contain the source-ask pattern (that's exactly what the
     strip exists to remove); the guard is that the strip's coverage keeps up as new tasks/templates
     are added, so a future novel phrasing doesn't silently slip through uncaught.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agent.app import prompt_hygiene as ph
from agent.app.testing import execution_compiled as ec

_IDEA_TESTS_DIR = Path(__file__).resolve().parent.parent / "app" / "idea_tests"
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "app" / "plan_library" / "templates"


# ---------------------------------------------------------------------------
# find_bare_output_contradiction
# ---------------------------------------------------------------------------
_BARE_DIRECTIVE = ec._THIN_EXTRACT_SYS  # the real live directive, not a hand-copied string


def test_no_contradiction_when_directive_does_not_demand_bareness():
    assert ph.find_bare_output_contradiction(
        "Answer freely, citing your source.", "Report the value and the exact source URL."
    ) == []


def test_no_contradiction_when_instruction_has_no_source_ask():
    assert ph.find_bare_output_contradiction(_BARE_DIRECTIVE, "Report the value only.") == []


def test_contradiction_found_inline_shape():
    found = ph.find_bare_output_contradiction(
        _BARE_DIRECTIVE, "Report the length, and the exact source URL."
    )
    assert found and "source URL" in found[0]


def test_contradiction_found_enumerated_shape():
    found = ph.find_bare_output_contradiction(
        _BARE_DIRECTIVE, "Report: (a) the atomic number, (b) the source URL."
    )
    assert found


def test_contradiction_found_standalone_sentence_shape():
    found = ph.find_bare_output_contradiction(_BARE_DIRECTIVE, "Give the source URL.")
    assert found


def test_contradiction_survives_empty_or_none_inputs_safely():
    assert ph.find_bare_output_contradiction("", "Give the source URL.") == []
    assert ph.find_bare_output_contradiction(_BARE_DIRECTIVE, "") == []


# ---------------------------------------------------------------------------
# strip_source_ask parity with the pre-port execution_compiled behavior
# ---------------------------------------------------------------------------
def test_strip_source_ask_matches_execution_compiled_wrapper():
    raw = "Report the max depth in metres, and the exact source URL."
    assert ph.strip_source_ask(raw) == ec._strip_source_ask(raw)


def test_strip_source_ask_removes_what_find_bare_output_contradiction_flags():
    raw = "Report the atomic number. Give the source URL."
    assert ph.find_bare_output_contradiction(_BARE_DIRECTIVE, raw)
    stripped = ph.strip_source_ask(raw)
    assert ph.find_bare_output_contradiction(_BARE_DIRECTIVE, stripped) == []


def test_strip_source_ask_degenerate_all_source_ask_falls_back_and_stays_flagged():
    # A documented, intentional edge case: if the instruction IS ENTIRELY the source-ask sentence,
    # stripping it would leave nothing to ask, so strip_source_ask falls back to the untouched
    # original — the lint correctly still flags this (it's genuinely unresolved), rather than the
    # transform silently emptying the question.
    raw = "Give the source URL."
    stripped = ph.strip_source_ask(raw)
    assert stripped == raw
    assert ph.find_bare_output_contradiction(_BARE_DIRECTIVE, stripped) == [raw]


# ---------------------------------------------------------------------------
# Corpus-wide regression guard
# ---------------------------------------------------------------------------
def _iter_idea_tests_leaf_instructions():
    for path in sorted(_IDEA_TESTS_DIR.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        get_plan = getattr(module, "get_compiled_plan", None)
        if get_plan is None:
            continue
        plan = get_plan()
        for leaf in plan.get("leaves", []):
            instruction = leaf.get("instruction", "")
            if instruction:
                yield f"{path.name}:{leaf.get('id', '?')}", instruction


def _iter_template_leaf_instructions():
    for path in sorted(_TEMPLATES_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue  # _manifest.json etc. — not a template itself
        data = json.loads(path.read_text())
        for leaf in data.get("leaves", []):
            instruction = leaf.get("instruction", "")
            if instruction:
                yield f"{path.name}:{leaf.get('id_pattern', leaf.get('id', '?'))}", instruction


@pytest.mark.parametrize("label,instruction", list(_iter_idea_tests_leaf_instructions()))
def test_idea_tests_leaf_instruction_has_no_contradiction_after_strip(label, instruction):
    stripped = ph.strip_source_ask(instruction)
    found = ph.find_bare_output_contradiction(_BARE_DIRECTIVE, stripped)
    assert found == [], f"{label}: strip_source_ask left a contradiction: {found!r}"


@pytest.mark.parametrize("label,instruction", list(_iter_template_leaf_instructions()))
def test_plan_library_template_leaf_instruction_has_no_contradiction_after_strip(label, instruction):
    stripped = ph.strip_source_ask(instruction)
    found = ph.find_bare_output_contradiction(_BARE_DIRECTIVE, stripped)
    assert found == [], f"{label}: strip_source_ask left a contradiction: {found!r}"


def test_corpus_scan_actually_found_leaves_not_a_silently_empty_parametrize():
    # A guard against the parametrize collecting zero cases (e.g. a moved directory) and the two
    # tests above passing vacuously.
    assert len(list(_iter_idea_tests_leaf_instructions())) > 50
    assert len(list(_iter_template_leaf_instructions())) > 0
