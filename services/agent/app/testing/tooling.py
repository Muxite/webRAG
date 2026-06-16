"""
Tooling-ablation profiles for the cost-vs-accuracy benchmark.

Three rungs isolate the contribution of *tooling* while the model is held fixed,
so the benchmark can answer "how much accuracy does the graph buy a cheap model?":

- ``minimal``: search-only (no crawl). The model answers from search-result
  snippets plus its own parametric knowledge. This is the "strong model, minimal
  tools" baseline the thesis is measured against.
- ``partial``: one fixed search+visit round, no planner (the existing ``naive_rag``).
- ``full``: the full Graph-of-Thoughts engine with verify + grounding (``graph``).

A profile maps to an internal execution *variant* (the existing runner mechanism in
``testing/runner.py`` / ``testing/execution.py``) plus the ``allowed_actions`` the
engine path may use. The profile name is stamped onto every result
(``result["tooling_profile"]``) so ``scripts/recovery_curve.py`` and
``scripts/level_ladder.py`` can group by tooling rung.

This is a thin naming + mapping layer on top of the variant machinery — it does not
replace it — so existing ``parametric`` runs keep working as a "no tools" floor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ToolingProfile:
    """One rung of the tooling ladder."""

    name: str
    #: Internal execution variant the runner dispatches on.
    variant: str
    #: Actions the engine path is permitted to use (informative for baselines,
    #: enforced for the ``full`` graph path via ``idea_settings["allowed_actions"]``).
    allowed_actions: List[str]
    description: str


TOOLING_PROFILES: Dict[str, ToolingProfile] = {
    "minimal": ToolingProfile(
        name="minimal",
        variant="minimal",
        allowed_actions=["search", "think"],
        description="Search-only (no crawl); answer from snippets + parametric knowledge.",
    ),
    "partial": ToolingProfile(
        name="partial",
        variant="naive_rag",
        allowed_actions=["search", "visit", "think"],
        description="One fixed search+visit round, no planner.",
    ),
    "full": ToolingProfile(
        name="full",
        variant="graph",
        allowed_actions=["search", "visit", "save", "think", "merge", "verify"],
        description="Full Graph-of-Thoughts engine with verify + grounding.",
    ),
}

#: Reverse map: internal variant -> tooling-profile name (to back-fill the profile
#: name when a run was launched via the legacy ``IDEA_TEST_EXECUTION_VARIANTS`` axis).
#: ``sequential_react`` is an agent ARCHITECTURE (not a tooling rung) but shares the
#: full toolset, so it is labelled ``sequential`` for grouping in the analysis scripts.
_VARIANT_TO_PROFILE: Dict[str, str] = {p.variant: name for name, p in TOOLING_PROFILES.items()}
_VARIANT_TO_PROFILE["sequential_react"] = "sequential"


def resolve_profile(token: str) -> Optional[ToolingProfile]:
    """Return the :class:`ToolingProfile` for a profile name, or ``None``."""
    return TOOLING_PROFILES.get((token or "").strip().lower())


def profile_for_variant(variant: str) -> str:
    """Map an internal execution variant back to a tooling-profile name.

    Falls back to the variant string itself for non-laddered variants (e.g.
    ``parametric`` = the no-tools floor, ``sequential``), so the field is always set.
    """
    v = (variant or "").strip().lower()
    return _VARIANT_TO_PROFILE.get(v, v)
