"""
Strategy-note schema — GENERALIZED prose advice for one task archetype.

A *strategy note* is the second, deliberately different artifact class in this repo's
"retrieve strategy instead of inventing it" line of work. Its sibling,
:mod:`agent.app.plan_library.schema`, holds slot-parameterized DAG *blueprints* that are bound
to concrete entities and executed. A note holds no slots and no structure at all: it is a short
paragraph of METHOD advice ("write every candidate's value out before concluding; the winner is
not necessarily the famous one") that is spliced into a PROMPT — the ``graph_compiled``
authoring prompt (:mod:`agent.app.testing.scaffold_compiler`) and that path's aggregation
prompt — rather than turned into nodes.

The split is on purpose (see the Architecture Decisions in
``/home/muk/.claude/plans/agent-trace-execution-paths-kind-kahn.md``):

  * ``plan_library`` is consumed by the native engine's free-text ``MergeLeafAction``, which is
    where its one dogfooded result went negative (a 7B model eyeballing six numbers);
  * ``strategy_library`` rides the compiled path, which already owns the deterministic
    composers, and only ever *adds prose to a prompt* — so a bad note degrades a prompt, it
    can never author a wrong subtree.

Two invariants make a note a note and not a leak:

1. **No slots.** ``<<slot>>`` (``plan_library``'s placeholder syntax) is rejected outright: a
   note that needs binding is a template and belongs in the other package.
2. **No ground truth.** Enforced structurally, not by review, by
   :mod:`agent.app.strategy_library.leak_gate` — every write goes through it, and every read
   re-checks against the CURRENT task's ledger. This module only holds the *provenance and
   evaluation* fields that make that auditable.

Promotion is a separate, higher bar than validity: :func:`is_active` requires a measured
held-out uplift on task instances the note was NOT derived from (``held_out_n >= 2`` and
uplift past :data:`MIN_HELD_OUT_UPLIFT`, both pre-registered here BEFORE any note was
generated). A note that only validates is ``candidate`` and is never retrieved.

Pure module: no I/O, no LLM.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union


class NoteValidationError(ValueError):
    """Raised when a strategy note is structurally invalid."""


#: Status vocabulary. ``candidate`` is authored-and-gated but unproven; ``active`` has cleared
#: :func:`is_active`; ``retired`` is kept on disk for provenance but never retrieved.
STATUS_CANDIDATE = "candidate"
STATUS_ACTIVE = "active"
STATUS_RETIRED = "retired"
STATUSES = (STATUS_CANDIDATE, STATUS_ACTIVE, STATUS_RETIRED)

#: Provenance vocabulary, mirroring ``plan_library/schema.py``'s existing (currently unused)
#: ``provenance.source`` field shape so a later mining pass slots in with no schema change.
#: ``"mined"`` — methodology (ii), distilling notes from adaptive-engine run traces — is
#: DEFERRED and no code here produces it; it is listed only so the vocabulary is stable.
SOURCE_HAND_AUTHORED = "hand_authored"
SOURCE_MINED = "mined"
SOURCES = (SOURCE_HAND_AUTHORED, SOURCE_MINED)

# --- the promotion gate, PRE-REGISTERED ------------------------------------------------
# Written down before any note existed, which is the whole point: designing the bar after
# seeing the numbers is how "we measured an uplift" quietly becomes "we picked a threshold
# that made our number pass". Both are deliberately modest — this is a floor for "not noise
# and not negative", NOT a significance test, and nothing here is powered enough for a
# default flip (the same honesty bar ``TECHNIQUE_INVENTORY.md`` applies elsewhere).

#: Held-out task INSTANCES (not repeats) a note must have been measured on. Two, because one
#: held-out instance cannot distinguish "the advice generalizes" from "that one task liked it".
MIN_HELD_OUT_N = 2
#: Minimum held-out score delta (advice-on minus advice-off, same model/task/repeats) on the
#: keystone metric. 0.05 = five points of pass rate: below that, an A/B at the repeat counts
#: this project can afford is indistinguishable from sampling noise.
MIN_HELD_OUT_UPLIFT = 0.05

#: A note is a prompt addendum, so it competes for the same context budget as the meta-prompt
#: it is spliced into. Long "advice" is also a symptom of a note drifting toward a worked
#: example, which is exactly what the leak gate exists to catch.
MAX_ADVICE_CHARS = 1200

#: ``plan_library``'s placeholder syntax — forbidden here (see the module docstring).
_SLOT_RE = re.compile(r"<<\s*[A-Za-z_][A-Za-z0-9_.]*\s*>>")


@dataclass(frozen=True)
class HeldOutMetrics:
    """What ``scripts/eval_strategy_library_generalization.py`` measured for one note.

    ``seed_fit`` is the same delta computed on a task the note WAS derived from. The ratio of
    the two is the reportable "generalization ratio": near 1 means the note transfers to unseen
    instances as well as to its own seeds, and a high seed_fit with ~zero held-out uplift is an
    indirect leak signal — an entry that cleared the textual gate but is really memorizing.

    All three deltas are ``None`` until measured; ``held_out_n`` is 0. That "not measured yet"
    state is distinct from "measured and flat" (0.0), and :func:`is_active` treats only the
    latter as a decision.
    """

    held_out_uplift: Optional[float] = None
    seed_fit: Optional[float] = None
    generalization_ratio: Optional[float] = None
    held_out_n: int = 0
    #: Task ids the held-out delta was measured on — must be disjoint from
    #: ``provenance.based_on_tasks`` (checked by :func:`validate_note`).
    held_out_tasks: List[str] = field(default_factory=list)
    #: Free text: which model / variant / repeat count produced the numbers. A delta is
    #: meaningless without it, so it rides in the same block rather than in a commit message.
    measured_with: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyNote:
    """One retrievable piece of generalized, archetype-scoped method advice."""

    note_id: str
    schema_version: int = 1
    #: Open string, not a closed enum — same reasoning as ``PlanTemplate.archetype``, and
    #: deliberately the SAME vocabulary ("argmax", "chain", ...) so the two libraries can be
    #: cross-referenced offline.
    archetype: str = ""
    title: str = ""
    #: THE artifact: prose method advice, spliced verbatim into a prompt. No slots, no
    #: entities, no numbers that came from a solved instance.
    advice: str = ""
    #: What gets embedded for retrieval — the note's *trigger* phrasing (a manual HyDE anchor
    #: harvested from real task statements of this archetype), not the advice itself, exactly
    #: like ``PlanTemplate.embedding_text``. Falls back to title+advice when absent.
    embedding_text: str = ""
    #: ``{source, based_on_tasks, mined_from_trace_ids, authored_by?, author_prompt_sha?}``
    provenance: Dict[str, Any] = field(default_factory=dict)
    evaluation: HeldOutMetrics = field(default_factory=HeldOutMetrics)
    status: str = STATUS_CANDIDATE
    #: Recorded verdict of the WRITE-time leak gate: ``{passed, audited, findings: [...]}``.
    #: Kept on the artifact so "here is the automated gate every entry passed" is a claim a
    #: reader can check against the shipped corpus, not one they have to take on trust.
    leak_gate: Dict[str, Any] = field(default_factory=dict)

    @property
    def based_on_tasks(self) -> List[str]:
        return [str(t) for t in (self.provenance or {}).get("based_on_tasks") or []]

    @property
    def source(self) -> str:
        return str((self.provenance or {}).get("source") or SOURCE_HAND_AUTHORED)


# --------------------------------------------------------------------------------------
# normalization / validation
# --------------------------------------------------------------------------------------


def normalize_note(raw: Union[StrategyNote, Mapping[str, Any]]) -> StrategyNote:
    """Return a :class:`StrategyNote` from a JSON-ish mapping (or another note).

    Forgiving about missing optional fields, mirroring ``plan_library.normalize_template`` /
    ``compiled_plan.normalize_plan``; strictness lives in :func:`validate_note`.
    """
    if isinstance(raw, StrategyNote):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        raise NoteValidationError(f"note must be a mapping, got {type(raw).__name__}")

    return StrategyNote(
        note_id=str(raw.get("note_id") or "").strip(),
        schema_version=_as_int(raw.get("schema_version"), default=1),
        archetype=str(raw.get("archetype") or "").strip(),
        title=str(raw.get("title") or "").strip(),
        advice=str(raw.get("advice") or "").strip(),
        embedding_text=str(raw.get("embedding_text") or "").strip(),
        provenance=dict(raw.get("provenance") or {}),
        evaluation=_normalize_metrics(raw.get("evaluation")),
        status=str(raw.get("status") or STATUS_CANDIDATE).strip().lower(),
        leak_gate=dict(raw.get("leak_gate") or {}),
    )


def _normalize_metrics(raw: Any) -> HeldOutMetrics:
    if isinstance(raw, HeldOutMetrics):
        raw = asdict(raw)
    if not isinstance(raw, Mapping):
        return HeldOutMetrics()
    tasks = raw.get("held_out_tasks") or []
    if isinstance(tasks, str):
        tasks = [tasks]
    return HeldOutMetrics(
        held_out_uplift=_as_opt_float(raw.get("held_out_uplift")),
        seed_fit=_as_opt_float(raw.get("seed_fit")),
        generalization_ratio=_as_opt_float(raw.get("generalization_ratio")),
        held_out_n=_as_int(raw.get("held_out_n"), default=0),
        held_out_tasks=[str(t).strip() for t in tasks if str(t).strip()],
        measured_with=str(raw.get("measured_with") or "").strip(),
    )


def validate_note(raw: Union[StrategyNote, Mapping[str, Any]]) -> StrategyNote:
    """Normalize then strictly validate a note; return it or raise.

    Checks identity/archetype/advice presence, the no-slots invariant, the advice length cap,
    the status and provenance-source vocabularies, and that the held-out task set is DISJOINT
    from the seed set — a "held-out" score measured on a seed is not a generalization measure
    at all, and that mistake is silent unless something rejects it here.

    Deliberately does NOT check for leaks: that is :mod:`leak_gate`'s job, needs per-task
    ledgers this module has no business loading, and is recorded on the note itself.
    """
    note = normalize_note(raw)

    if not note.note_id:
        raise NoteValidationError("note_id is required")
    where = f"note '{note.note_id}'"
    if not note.archetype:
        raise NoteValidationError(f"{where}: archetype is required")
    if not note.advice:
        raise NoteValidationError(f"{where}: advice is required")
    if len(note.advice) > MAX_ADVICE_CHARS:
        raise NoteValidationError(
            f"{where}: advice is {len(note.advice)} chars, over the {MAX_ADVICE_CHARS} cap "
            "(a note is a prompt addendum, not a document)"
        )
    slot = _SLOT_RE.search(note.advice) or _SLOT_RE.search(note.embedding_text)
    if slot:
        raise NoteValidationError(
            f"{where}: contains the plan-library placeholder '{slot.group(0)}' — a strategy "
            "note is unparameterized prose; a slotted strategy belongs in plan_library/"
        )
    if note.status not in STATUSES:
        raise NoteValidationError(
            f"{where}: unknown status '{note.status}' (expected one of: {', '.join(STATUSES)})"
        )
    if note.source not in SOURCES:
        raise NoteValidationError(
            f"{where}: unknown provenance.source '{note.source}' "
            f"(expected one of: {', '.join(SOURCES)})"
        )

    seeds = {t.strip().lower() for t in note.based_on_tasks}
    overlap = sorted(t for t in note.evaluation.held_out_tasks if t.strip().lower() in seeds)
    if overlap:
        raise NoteValidationError(
            f"{where}: held_out_tasks {overlap} are also in provenance.based_on_tasks — a "
            "held-out uplift measured on a seed task is a seed fit, not a generalization"
        )
    return note


# --------------------------------------------------------------------------------------
# the promotion gate
# --------------------------------------------------------------------------------------


def is_active(
    note: StrategyNote,
    *,
    min_held_out_n: int = MIN_HELD_OUT_N,
    min_uplift: float = MIN_HELD_OUT_UPLIFT,
) -> bool:
    """True when ``note`` may be retrieved at all.

    Three conditions, all required: the note is not ``retired``; it was measured on at least
    ``min_held_out_n`` held-out task instances; and its measured held-out uplift is at least
    ``min_uplift``. ``status == "active"`` is NOT sufficient on its own — the metrics are the
    gate and the status field only records the last decision, so hand-editing a JSON file to
    say "active" cannot promote an unmeasured note.
    """
    if note.status == STATUS_RETIRED:
        return False
    metrics = note.evaluation
    if metrics.held_out_n < int(min_held_out_n):
        return False
    if metrics.held_out_uplift is None:
        return False
    return float(metrics.held_out_uplift) >= float(min_uplift)


def promotion_reason(
    note: StrategyNote,
    *,
    min_held_out_n: int = MIN_HELD_OUT_N,
    min_uplift: float = MIN_HELD_OUT_UPLIFT,
) -> str:
    """Human-readable verdict for :func:`is_active` — for the sync script and the eval CLI."""
    if note.status == STATUS_RETIRED:
        return "retired"
    metrics = note.evaluation
    if metrics.held_out_n < int(min_held_out_n):
        return (
            f"held_out_n={metrics.held_out_n} < {min_held_out_n} "
            "(not yet powered enough to promote)"
        )
    if metrics.held_out_uplift is None:
        return "held_out_uplift not measured"
    if float(metrics.held_out_uplift) < float(min_uplift):
        return f"held_out_uplift={metrics.held_out_uplift:+.3f} < {min_uplift:+.3f}"
    return (
        f"active: held_out_uplift={metrics.held_out_uplift:+.3f} over "
        f"{metrics.held_out_n} held-out instance(s)"
    )


def generalization_ratio(held_out_uplift: Optional[float], seed_fit: Optional[float]) -> Optional[float]:
    """``held_out_uplift / seed_fit``, or ``None`` when it is not a meaningful number.

    Undefined when either delta is missing or the seed fit is non-positive: a note that did not
    even help on its own seeds has no baseline to generalize *from*, and dividing by ~0 turns
    noise into a spectacular-looking ratio. Reported as ``None`` rather than fabricated.
    """
    if held_out_uplift is None or seed_fit is None:
        return None
    if float(seed_fit) <= 0.0:
        return None
    return float(held_out_uplift) / float(seed_fit)


# --------------------------------------------------------------------------------------
# serialization
# --------------------------------------------------------------------------------------


def note_to_dict(note: StrategyNote) -> Dict[str, Any]:
    """JSON-ready mapping for the on-disk corpus (stable key order comes from the dataclass)."""
    raw = asdict(note)
    raw["evaluation"] = note.evaluation.as_dict()
    return raw


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_opt_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

