"""
Plan library — a persistent, retrievable set of pre-authored composition strategies.

A weak/mid executor's real wall is inventing the right DECOMPOSITION STRATEGY, not formatting
it. This package holds parameterized ``PlanTemplate``s for the composition shapes that wall
shows up on (argmax over N page reads, N-hop entity chains, computed-ratio argmax, ...), so the
strategy can be *retrieved* instead of invented.

Schema, slot filling (:mod:`slot_fill`), retrieval (:mod:`retrieval`), its instrumentation
(:mod:`retrieval_log`) and the hand-authored corpus (``templates/``) live here. The engine
imports exactly two things from this package — :mod:`retrieval` (which template fits this
node, and the I/O to find out) and :mod:`retrieval_log` — and gets the *candidates* those
produce through :mod:`agent.app.idea_policies.plan_library`, the adapter that turns a filled
template into native Graph-of-Thought expansion candidates. Nothing here ever shapes engine
input on its own.

Only the schema is re-exported at package level: :mod:`retrieval` imports the adapter, which
imports the schema, so eagerly importing it here would put a cycle through this ``__init__``.
Import it as ``from agent.app.plan_library.retrieval import PlanLibrary``.
"""

from agent.app.plan_library.schema import (
    ExtractionStrategy,
    FilledLeaf,
    FilledPlan,
    LeafBlueprint,
    PlanTemplate,
    SlotKind,
    SlotSpec,
    TemplateValidationError,
    bind_template,
    fill_template,
    normalize_template,
    validate_template,
)

__all__ = [
    "ExtractionStrategy",
    "FilledLeaf",
    "FilledPlan",
    "LeafBlueprint",
    "PlanTemplate",
    "SlotKind",
    "SlotSpec",
    "TemplateValidationError",
    "bind_template",
    "fill_template",
    "normalize_template",
    "validate_template",
]
