"""
Strategy library — retrievable, GENERALIZED prose advice for a task archetype.

The sibling of :mod:`agent.app.plan_library`, and deliberately a different artifact: that
package holds slot-parameterized DAG blueprints that become nodes; this one holds short
paragraphs of method advice that become *prompt text* on the ``graph_compiled`` path (the
authoring meta-prompt in ``testing/scaffold_compiler.py`` and that path's aggregation prompt).
A bad template authors a wrong subtree; a bad note only adds an unhelpful paragraph — which is
why this package can ride the proven, deterministic-composer-capable path with a much smaller
blast radius.

Four modules:

* :mod:`schema` — the ``StrategyNote`` artifact plus the PRE-REGISTERED promotion gate (a note
  is invisible to retrieval until a held-out uplift has actually been measured);
* :mod:`leak_gate` — the four-layer, automated "is this generalized or memorized" check every
  write passes and every read re-runs, with the ``c0dbc720`` leak and the ``mead``/``res_mead``
  word-boundary blind spot as required-reject regression fixtures;
* :mod:`authoring` — seed task statements -> one strong-model call -> gate -> ``notes/``;
* :mod:`retrieval` — top-1 Chroma match + one threshold + the read-time leak screen.

Everything is opt-in and default-off (``strategy_library_enabled``), so a run with the flag
unset is byte-identical to one from before this package existed.

Only the schema is re-exported here: :mod:`retrieval` imports :mod:`leak_gate`, which imports
task modules, so eager imports would put avoidable weight on every ``import agent.app``. Import
the rest explicitly, e.g.
``from agent.app.strategy_library.retrieval import StrategyLibrary``.
"""

from agent.app.strategy_library.schema import (
    MIN_HELD_OUT_N,
    MIN_HELD_OUT_UPLIFT,
    HeldOutMetrics,
    NoteValidationError,
    StrategyNote,
    generalization_ratio,
    is_active,
    normalize_note,
    note_to_dict,
    promotion_reason,
    validate_note,
)

__all__ = [
    "HeldOutMetrics",
    "MIN_HELD_OUT_N",
    "MIN_HELD_OUT_UPLIFT",
    "NoteValidationError",
    "StrategyNote",
    "generalization_ratio",
    "is_active",
    "normalize_note",
    "note_to_dict",
    "promotion_reason",
    "validate_note",
]
