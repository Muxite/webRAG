"""Tests for the ChromaDB init/warmup skip gate in ``idea_test_runner``.

``init_chroma()`` and ``_warmup_chroma()`` used to run unconditionally in the per-cell
subprocess setup even for execution variants that never touch ChromaDB (measured at
~5.2-5.6s per cell on the healthy path, dominated by the SentenceTransformer/CUDA cold
load). ``_execution_variants_need_chroma`` gates that setup on whether ANY requested
variant actually needs Chroma, derived from ``testing/runner.py``'s own dispatch tuples
(the single source of truth for which execution function a variant reaches) rather than
a fresh hand-maintained list here.

Contract under test:
  - A non-Chroma-only variant list -> False (skip init/warmup).
  - A Chroma-using variant list -> True (do init/warmup).
  - A MIXED list -> True (any(), not all() - one Chroma arm is enough).
  - An unrecognized/future variant name -> True (fail-safe: unknown means "needs chroma"
    so a new variant can never silently lose its memory just because this gate wasn't
    updated for it).
"""
from __future__ import annotations

from agent.app.idea_test_runner import (
    _execution_variants_need_chroma,
    _NO_CHROMA_VARIANTS,
)
from agent.app.testing.runner import BASELINE_VARIANTS, OFFTHESHELF_VARIANTS


def test_no_chroma_variants_matches_runner_baseline_and_offtheshelf():
    # Sanity: the allowlist is exactly the union of the two dispatch tuples it claims to
    # derive from, not a coincidentally-similar hand copy.
    assert _NO_CHROMA_VARIANTS == frozenset(BASELINE_VARIANTS) | frozenset(OFFTHESHELF_VARIANTS)


def test_non_chroma_only_variants_skip():
    for variant_list in (
        ["parametric"],
        ["naive_rag"],
        ["minimal"],
        ["langgraph_react"],
        ["parametric", "naive_rag", "minimal", "langgraph_react"],
    ):
        assert _execution_variants_need_chroma(variant_list) is False, variant_list


def test_chroma_using_variants_need_chroma():
    for variant_list in (
        ["graph"],
        ["sequential_react"],
        ["naive_discretion"],
        ["graph_compiled"],
        ["graph_compiled_code"],
    ):
        assert _execution_variants_need_chroma(variant_list) is True, variant_list


def test_mixed_matrix_needs_chroma():
    # any(), not all(): one Chroma-using arm anywhere in the matrix must still init Chroma.
    assert _execution_variants_need_chroma(["parametric", "graph"]) is True
    assert _execution_variants_need_chroma(["langgraph_react", "sequential_react"]) is True
    assert _execution_variants_need_chroma(["minimal", "naive_rag", "graph_compiled"]) is True


def test_unknown_variant_defaults_to_needs_chroma():
    # Fail-safe: an unrecognized variant name (e.g. a brand-new variant added to
    # testing/runner.py but not yet reflected in any allowlist here) must default to
    # "needs chroma" so it never silently loses its memory.
    assert _execution_variants_need_chroma(["some_future_variant_not_yet_known"]) is True
    # And it dominates even alongside a known non-Chroma variant.
    assert _execution_variants_need_chroma(["parametric", "some_future_variant"]) is True


def test_empty_variant_list_skips():
    assert _execution_variants_need_chroma([]) is False
