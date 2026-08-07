"""
Strategy-note retrieval — which piece of generalized advice fits THIS task, if any.

Deliberately the *minimal* version of ``plan_library/retrieval.py``: top-1 by embedding
similarity, one threshold, no archetype-boost rerank and no runner-up margin. Those extra
stages exist there because picking the WRONG template authors a wrong subtree; picking the
wrong note only appends a paragraph of irrelevant advice to a prompt, so the machinery would
cost more than the failure it prevents. The ranking primitive itself
(:func:`~agent.app.plan_library.retrieval.similarity_from_distance`) is imported rather than
re-derived — it is artifact-agnostic, and the cosine/l2 conversion is exactly the kind of thing
that must not exist in two places.

Disk is the source of truth (``notes/*.json``); Chroma is only an index, so a missing, empty or
broken collection degrades to "no advice" — never to wrong advice.

Three gates, in order, each of which can only ever REMOVE advice:

1. **Promotion** (:func:`~agent.app.strategy_library.schema.is_active`) — a note is invisible
   to retrieval until a held-out uplift has actually been measured on task instances it was not
   derived from. Applied at load, so an unproven note cannot be reached by any code path.
2. **Similarity** — one threshold, :data:`APPLY_THRESHOLD`.
3. **Read-time leak re-check** — the retrieved note's text is run through
   :func:`~agent.app.strategy_library.leak_gate.check_text` against the CURRENT task's ledger,
   which is the one thing write time could not know. Any finding drops the note and the run
   proceeds with no advice: fail toward silence, the same philosophy
   ``plan_library/retrieval.py`` and ``contract_satisfaction.py`` state.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set
from uuid import uuid4

from agent.app.plan_library.retrieval import similarity_from_distance
from agent.app.strategy_library import leak_gate
from agent.app.strategy_library.schema import (
    NoteValidationError,
    StrategyNote,
    is_active,
    note_to_dict,
    promotion_reason,
    validate_note,
)

logger = logging.getLogger(__name__)

#: Versioned so a schema/embedding change rolls out as a new collection, not a migration.
COLLECTION_NAME = "strategy_library_notes_v1"
COLLECTION_METADATA: Dict[str, Any] = {"hnsw:space": "cosine"}

NOTES_DIR = Path(__file__).resolve().parent / "notes"
MANIFEST_NAME = "_manifest.json"

#: The one decision constant. UNCALIBRATED: unlike ``plan_library``'s thresholds (fitted to a
#: labelled positive/negative set), no such set exists for notes yet — the corpus is empty until
#: something clears the promotion gate. 0.50 is inherited as a starting value from the sibling
#: package's calibration on a *different* corpus and should be re-fitted by
#: ``scripts/eval_strategy_library_generalization.py`` before anyone reports a number that
#: depends on it. Erring high is the safe direction here: a miss costs today's behavior, a
#: false hit costs an irrelevant paragraph in a prompt.
APPLY_THRESHOLD = 0.50
TOP_K = 1

DECISION_APPLY = "apply"
DECISION_NO_MATCH = "fallthrough_no_match"
DECISION_LEAK_BLOCKED = "fallthrough_leak_gate"

#: The promotion gate's chicken-and-egg escape hatch, and the ONLY way past it.
#:
#: A note cannot be promoted until its held-out uplift is measured, and it cannot be measured
#: until a run can retrieve it. Rather than have the eval flow hand-fabricate metrics into a
#: committed artifact (which is exactly the kind of edit that gets committed by accident), this
#: env var makes UNPROMOTED notes retrievable for a measurement run. It promotes nothing on
#: disk, and the run's result JSON still records which note applied — so a measurement run is
#: distinguishable after the fact. Any number produced with it set is a measurement, never a
#: production configuration.
ENV_INCLUDE_UNPROMOTED = "IDEA_TEST_STRATEGY_LIBRARY_INCLUDE_UNPROMOTED"


@dataclass(frozen=True)
class NoteMatch:
    """One ranked candidate note."""

    note_id: str
    archetype: str
    similarity: float
    title: str = ""
    #: True when the id came back from Chroma but no note file declares it (a stale index
    #: entry, or a note that has since been un-promoted): ranked but never applicable.
    known: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "archetype": self.archetype,
            "similarity": round(self.similarity, 4),
            "known": self.known,
        }


@dataclass(frozen=True)
class AdviceResult:
    """The outcome of one retrieval attempt — always returned, never raised."""

    retrieval_id: str
    query_text: str
    decision: str
    candidates: List[NoteMatch] = field(default_factory=list)
    note_id: Optional[str] = None
    archetype: Optional[str] = None
    similarity: Optional[float] = None
    #: The text to splice. Empty unless ``decision == DECISION_APPLY``.
    advice: str = ""
    reason: str = ""
    #: The read-time gate's verdict, when it ran — kept for the log.
    leak_findings: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def applied(self) -> bool:
        return self.decision == DECISION_APPLY and bool(self.advice)

    def __bool__(self) -> bool:
        return self.applied

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_id": self.retrieval_id,
            "query_text": self.query_text,
            "decision": self.decision,
            "note_id": self.note_id,
            "archetype": self.archetype,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "reason": self.reason,
            "results": [m.as_dict() for m in self.candidates],
            "leak_findings": list(self.leak_findings),
        }


# --------------------------------------------------------------------------------------
# what gets embedded / hashed
# --------------------------------------------------------------------------------------


def note_hash(note: StrategyNote) -> str:
    """Content hash of a note — the sync lockfile's key (mirrors ``template_hash``)."""
    payload = json.dumps(note_to_dict(note), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def document_text(note: StrategyNote) -> str:
    """What gets embedded: the note's TRIGGER phrasing, not its advice.

    Same reasoning as ``plan_library.document_text`` — a MiniLM-class embedding is surface-form
    sensitive, and advice ("write every value out before concluding") shares almost no
    vocabulary with the task statements it should fire on. ``embedding_text`` is authored as a
    manual HyDE anchor harvested from real statements of the archetype; title+advice is only the
    fallback for a note that omits it.
    """
    text = (note.embedding_text or "").strip()
    if text:
        return text
    return " ".join(p for p in (note.title, note.advice) if p).strip()


def document_metadata(note: StrategyNote) -> Dict[str, Any]:
    provenance = note.provenance or {}
    return {
        "note_id": note.note_id,
        "archetype": note.archetype,
        "title": note.title,
        "schema_version": note.schema_version,
        "source": str(provenance.get("source") or "hand_authored"),
        "based_on_tasks": list(provenance.get("based_on_tasks") or []),
        "held_out_n": int(note.evaluation.held_out_n or 0),
        "content_hash": note_hash(note),
    }


# --------------------------------------------------------------------------------------
# the library
# --------------------------------------------------------------------------------------


class StrategyLibrary:
    """The in-process note corpus plus its Chroma-backed ranking."""

    def __init__(
        self,
        notes_dir: Optional[Path] = None,
        *,
        collection: str = COLLECTION_NAME,
        warn_on_drift: bool = True,
        include_inactive: Optional[bool] = None,
    ) -> None:
        self.notes_dir = Path(notes_dir) if notes_dir else NOTES_DIR
        self.collection = collection
        #: Every note on disk, keyed by id — including unpromoted ones, which the sync script
        #: and the eval CLI need to see.
        self.all_notes: Dict[str, StrategyNote] = {}
        self.load_errors: Dict[str, str] = {}
        #: ``include_inactive`` exists ONLY for the authoring/eval tooling. Retrieval always
        #: uses :attr:`notes`, which the promotion gate has already filtered — unless
        #: :data:`ENV_INCLUDE_UNPROMOTED` is set, which is the measurement escape hatch.
        self.include_inactive = (
            _env_include_unpromoted() if include_inactive is None else bool(include_inactive)
        )
        self._space: Optional[str] = None
        self._warned_empty_index = False
        self._load()
        if warn_on_drift:
            self._warn_on_drift()

    # -- corpus -------------------------------------------------------------------------

    def _load(self) -> None:
        if not self.notes_dir.is_dir():
            logger.debug("Strategy library: no notes directory at %s", self.notes_dir)
            return
        for path in sorted(self.notes_dir.glob("*.json")):
            if path.name.startswith("_"):  # _manifest.json and friends
                continue
            try:
                note = validate_note(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, NoteValidationError) as exc:
                self.load_errors[path.name] = str(exc)
                logger.warning("Strategy library: skipping invalid note %s: %s", path.name, exc)
                continue
            if note.note_id in self.all_notes:
                self.load_errors[path.name] = f"duplicate note_id '{note.note_id}'"
                logger.warning(
                    "Strategy library: duplicate note_id '%s' in %s — keeping the first",
                    note.note_id, path.name,
                )
                continue
            self.all_notes[note.note_id] = note

    @property
    def notes(self) -> Dict[str, StrategyNote]:
        """The RETRIEVABLE corpus: promoted notes only (see :func:`schema.is_active`)."""
        if self.include_inactive:
            return dict(self.all_notes)
        return {nid: n for nid, n in self.all_notes.items() if is_active(n)}

    def __len__(self) -> int:
        return len(self.notes)

    def get(self, note_id: Optional[str]) -> Optional[StrategyNote]:
        return self.notes.get(str(note_id)) if note_id else None

    def promotion_report(self) -> Dict[str, str]:
        """``{note_id: why it is (not) retrievable}`` — for the sync script and the eval CLI."""
        return {nid: promotion_reason(n) for nid, n in sorted(self.all_notes.items())}

    # -- manifest -----------------------------------------------------------------------

    def manifest_path(self) -> Path:
        return self.notes_dir / MANIFEST_NAME

    def read_manifest(self) -> Dict[str, str]:
        try:
            data = json.loads(self.manifest_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def drift(self) -> Dict[str, str]:
        """``{note_id: added|changed|removed}`` vs the manifest — Chroma-free.

        Computed over the RETRIEVABLE corpus: a note that loses its promotion (its metrics were
        re-measured and no longer clear the bar) must show up as ``removed`` so the sync script
        deletes it from the index instead of leaving it silently rankable.
        """
        manifest = self.read_manifest()
        notes = self.notes
        out: Dict[str, str] = {}
        for note_id, note in notes.items():
            recorded = manifest.get(note_id)
            if recorded is None:
                out[note_id] = "added"
            elif recorded != note_hash(note):
                out[note_id] = "changed"
        for note_id in manifest:
            if note_id not in notes:
                out[note_id] = "removed"
        return out

    def _warn_on_drift(self) -> None:
        drifted = self.drift()
        if drifted:
            logger.warning(
                "Strategy library: %d note(s) differ from %s (%s) — run "
                "scripts/sync_strategy_library.py to re-index",
                len(drifted), MANIFEST_NAME,
                ", ".join(f"{k}:{v}" for k, v in sorted(drifted.items())),
            )

    # -- index --------------------------------------------------------------------------

    async def ensure_collection(self, connector_chroma: Any) -> Any:
        """Get (creating it with the COSINE space) the ranking collection.

        Always before any query: ``query_chroma`` would otherwise create it with chroma's
        default ``l2`` space, which no later sync can change — silently miscalibrating
        :data:`APPLY_THRESHOLD`.
        """
        try:
            coll = await connector_chroma.get_or_create_collection(
                self.collection, metadata=dict(COLLECTION_METADATA)
            )
        except TypeError:
            coll = await connector_chroma.get_or_create_collection(self.collection)
        if coll is not None and self._space is None:
            self._space = _read_space(coll)
        return coll

    async def indexed_ids(
        self, connector_chroma: Any, note_ids: Sequence[str]
    ) -> Optional[Set[str]]:
        """Which of ``note_ids`` the TARGET collection holds; ``None`` when unknowable."""
        wanted = [str(n) for n in note_ids or ()]
        if not wanted:
            return set()
        getter = getattr(connector_chroma, "get_from_chroma", None)
        if getter is None:
            return None
        try:
            if await self.ensure_collection(connector_chroma) is None:
                return None
            raw = await getter(self.collection, wanted)
        except Exception as exc:  # noqa: BLE001 — a membership check never breaks a caller
            logger.warning("Strategy library: index membership check failed: %s", exc)
            return None
        found = raw.get("ids") if isinstance(raw, Mapping) else None
        if not isinstance(found, list):
            return None
        if found and isinstance(found[0], list):
            found = [i for row in found for i in row]
        return {str(i) for i in found}

    # -- retrieval ----------------------------------------------------------------------

    async def retrieve(
        self,
        connector_chroma: Any,
        query_text: str,
        *,
        top_k: int = TOP_K,
        apply_threshold: float = APPLY_THRESHOLD,
    ) -> AdviceResult:
        """Rank notes for ``query_text`` and decide whether the top hit may be spliced.

        Never raises: any missing/failed/empty piece returns a ``fallthrough_no_match`` result
        carrying the reason, so the caller simply proceeds with no advice.
        """
        retrieval_id = uuid4().hex
        query = " ".join(str(query_text or "").split())
        if not query:
            return _no_match(retrieval_id, query, "empty query")
        if not self.notes:
            return _no_match(retrieval_id, query, "no promoted notes in the strategy library")

        matches = await self._rank(connector_chroma, query, top_k=top_k)
        if not matches:
            self._warn_index_looks_unsynced()
            return _no_match(retrieval_id, query, "no index matches")

        top = matches[0]
        if not top.known:
            return _no_match(
                retrieval_id, query,
                f"top hit '{top.note_id}' is a stale index entry (no promoted note file)",
                matches,
            )
        if top.similarity < apply_threshold:
            return _no_match(
                retrieval_id, query,
                f"top similarity {top.similarity:.3f} < apply threshold {apply_threshold:.2f}",
                matches,
            )
        note = self.notes[top.note_id]
        return AdviceResult(
            retrieval_id=retrieval_id,
            query_text=query,
            decision=DECISION_APPLY,
            candidates=matches,
            note_id=note.note_id,
            archetype=note.archetype,
            similarity=top.similarity,
            advice=note.advice,
            reason=f"similarity {top.similarity:.3f}",
        )

    async def _rank(self, connector_chroma: Any, query: str, *, top_k: int) -> List[NoteMatch]:
        if connector_chroma is None:
            return []
        n_results = max(1, min(int(top_k or TOP_K), len(self.notes)))
        try:
            await self.ensure_collection(connector_chroma)
            raw = await connector_chroma.query_chroma(
                collection=self.collection, query_texts=[query], n_results=n_results
            )
        except Exception as exc:  # noqa: BLE001 — retrieval degrades, it never breaks a run
            logger.warning("Strategy library: retrieval query failed: %s", exc)
            return []
        return self._parse_matches(raw)

    def _parse_matches(self, raw: Any) -> List[NoteMatch]:
        if not isinstance(raw, Mapping):
            return []
        ids = _first_row(raw.get("ids"))
        distances = _first_row(raw.get("distances"))
        metadatas = _first_row(raw.get("metadatas"))
        space = self._space or "cosine"
        notes = self.notes
        out: List[NoteMatch] = []
        for index, note_id in enumerate(ids):
            note_id = str(note_id)
            distance = distances[index] if index < len(distances) else None
            meta = metadatas[index] if index < len(metadatas) else None
            meta = meta if isinstance(meta, Mapping) else {}
            note = notes.get(note_id)
            out.append(NoteMatch(
                note_id=note_id,
                archetype=str((note.archetype if note else None) or meta.get("archetype") or ""),
                similarity=similarity_from_distance(distance, space),
                title=str((note.title if note else None) or meta.get("title") or ""),
                known=note is not None,
            ))
        return out

    def _warn_index_looks_unsynced(self) -> None:
        if self._warned_empty_index:
            return
        self._warned_empty_index = True
        logger.warning(
            "Strategy library: collection '%s' ranked none of the %d promoted note(s) on disk "
            "— if this repeats it is probably unsynced (run scripts/sync_strategy_library.py "
            "against this chroma)",
            self.collection, len(self.notes),
        )

    # -- the read-time leak gate --------------------------------------------------------

    def screen(self, result: AdviceResult, task_source: Any) -> AdviceResult:
        """Re-check an applied note against the CURRENT task's ledger; drop it on any finding.

        This is the gate write time structurally cannot run: a note is authored against its seed
        tasks' ledgers, and the task it is later retrieved for has its own secrets. Deterministic
        layers only (no LLM call per run — see ``leak_gate.check_text``).

        Anything that goes wrong here — an unloadable task module, a broken ledger — also drops
        the note. A retrieval that cannot be screened is not a retrieval that may be used.
        """
        if not result.applied or task_source is None:
            return result
        try:
            ledger = leak_gate.build_ledger(task_source)
            verdict = leak_gate.check_text(result.advice, [ledger])
        except Exception as exc:  # noqa: BLE001 — fail toward silence
            logger.warning("Strategy library: read-time leak screen failed: %s", exc)
            return _downgrade(result, DECISION_LEAK_BLOCKED, f"leak screen errored: {exc}")
        if verdict.passed:
            return result
        logger.warning(
            "Strategy library: note '%s' dropped by the read-time leak gate for task %s: %s",
            result.note_id, ledger.task_id, verdict.summary(),
        )
        return _downgrade(
            result, DECISION_LEAK_BLOCKED,
            f"read-time leak gate: {verdict.summary()}",
            findings=[f.as_dict() for f in verdict.rejections],
        )

    async def advice_for_task(
        self,
        connector_chroma: Any,
        mandate: str,
        task_source: Any = None,
        *,
        top_k: int = TOP_K,
        apply_threshold: float = APPLY_THRESHOLD,
    ) -> AdviceResult:
        """:meth:`retrieve` then :meth:`screen` — the one call a consumer makes."""
        result = await self.retrieve(
            connector_chroma, mandate, top_k=top_k, apply_threshold=apply_threshold
        )
        return self.screen(result, task_source)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def _env_include_unpromoted() -> bool:
    """Whether :data:`ENV_INCLUDE_UNPROMOTED` is set (logged loudly — this is not a normal run)."""
    import os

    if os.environ.get(ENV_INCLUDE_UNPROMOTED, "") in ("", "0", "false", "False"):
        return False
    logger.warning(
        "Strategy library: %s is set — UNPROMOTED notes are retrievable. This is a measurement "
        "configuration, not a production one; do not report its output as a normal run.",
        ENV_INCLUDE_UNPROMOTED,
    )
    return True


def _first_row(value: Any) -> List[Any]:
    """Chroma returns one row per query text; this module always sends exactly one."""
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return list(value) if isinstance(value, list) else []


def _read_space(collection: Any) -> str:
    try:
        config = getattr(collection, "configuration_json", None)
        if isinstance(config, Mapping):
            space = (config.get("hnsw") or {}).get("space")
            if space:
                return str(space).lower()
        meta = getattr(collection, "metadata", None)
        if isinstance(meta, Mapping) and meta.get("hnsw:space"):
            return str(meta["hnsw:space"]).lower()
    except Exception:  # noqa: BLE001
        pass
    return "cosine"


def _no_match(
    retrieval_id: str,
    query: str,
    reason: str,
    matches: Optional[List[NoteMatch]] = None,
) -> AdviceResult:
    return AdviceResult(
        retrieval_id=retrieval_id,
        query_text=query,
        decision=DECISION_NO_MATCH,
        candidates=list(matches or []),
        similarity=matches[0].similarity if matches else None,
        reason=reason,
    )


def _downgrade(
    result: AdviceResult,
    decision: str,
    reason: str,
    findings: Optional[List[Dict[str, Any]]] = None,
) -> AdviceResult:
    """Post-hoc "no advice" that keeps the ranking for the log (fail toward silence)."""
    from dataclasses import replace

    return replace(
        result, decision=decision, advice="", reason=reason,
        leak_findings=list(findings or []),
    )
