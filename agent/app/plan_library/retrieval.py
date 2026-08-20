"""
Plan-library retrieval — which pre-authored composition strategy fits THIS node's intent.

Disk is the source of truth: :class:`PlanLibrary` loads every ``templates/*.json`` into an
in-process dict at construction. Chroma is only an INDEX — consulted to rank ``template_id``s
for a query, never to store a template. A missing/empty/broken collection therefore degrades to
"no match" (organic expansion), never to a wrong or partial template.

The pipeline is four steps, each independently loggable (``retrieval_log``):

1. :func:`build_query_text` — the node's *intent* (title, goal, parent goal, expect), which is
   the level a template's ``embedding_text`` is authored at. Deliberately NO gathered evidence:
   matching a plan shape against page content would drift with whatever was retrieved so far.
2. :meth:`PlanLibrary.retrieve` — top-K by embedding similarity from Chroma.
3. an archetype-hint *soft rerank* — ``shape_classifier.classify_shape()`` (deterministic,
   already fails open to ``None``) as a prior: a hit whose declared ``archetype`` matches the
   hint gets :data:`ARCHETYPE_BOOST` added before thresholding. A second, independent signal
   that nudges ties; it can never override embedding similarity outright.
4. :meth:`PlanLibrary.fill_from_query` — extract this query's slot values
   (:mod:`agent.app.plan_library.slot_fill`, where the pipeline's ONE LLM call lives, fired
   only now that a match has qualified), bind them, and hand the result to
   :mod:`agent.app.idea_policies.plan_library`. :meth:`PlanLibrary.fill` is the same step for
   a caller that already HAS the values. A slot-extraction or slot-fill failure downgrades an
   otherwise-adopted match to "no match" rather than raising: fail toward silence, exactly
   ``contract_satisfaction.py``'s stated philosophy.

**Similarity is cosine, and that is enforced, not assumed**: chroma's default HNSW space is
``l2`` (squared euclidean), so ``1 - distance`` would NOT be a cosine similarity on a
default-created collection. :meth:`PlanLibrary.ensure_collection` therefore creates the
collection with ``{"hnsw:space": "cosine"}`` and :func:`similarity_from_distance` converts
whatever space the live collection actually reports, so thresholds keep one meaning even
against a collection someone else created first.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Set
from uuid import uuid4

from agent.app.idea_policies.base import DetailKey
from agent.app.idea_policies.plan_library import (
    ORIGIN_AUTO,
    PlanLibraryExpansion,
    candidates_from_template,
)
from agent.app.idea_policies.shape_classifier import classify_shape
from agent.app.plan_library.schema import (
    PlanTemplate,
    TemplateValidationError,
    validate_template,
)

if TYPE_CHECKING:
    from agent.app.idea_dag import IdeaNode

logger = logging.getLogger(__name__)

#: Versioned so a future schema/embedding change can be rolled out as a new collection
#: rather than migrated in place.
COLLECTION_NAME = "plan_library_templates_v1"

#: Creation-time collection config. Applied only when the collection does not exist yet;
#: :func:`similarity_from_distance` covers the case where it already does, with another space.
COLLECTION_METADATA: Dict[str, Any] = {"hnsw:space": "cosine"}

#: Where the hand-authored corpus lives, and the lockfile the sync script maintains.
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
MANIFEST_NAME = "_manifest.json"

# --- decision constants ----------------------------------------------------------------
# CALIBRATED against the bootstrap eval set (``scripts/eval_plan_library_retrieval.py``:
# reachable-tier task statements as labelled positives with a known template_id, micro- and
# format-tier statements as negatives), not guessed. On the seed corpus that set measures:
# every positive's correct top-1 lands at >= 0.54 and every negative's best hit at <= 0.45,
# so the class boundary sits near 0.49 and these constants bracket it. Re-run that script
# after editing a template's ``embedding_text`` — the numbers move with the corpus.

TOP_K = 5
#: Below the weakest correct positive (0.54) with headroom, above every negative. The
#: asymmetry is deliberate: a missed hit costs only today's organic expansion, while a
#: false hit is still caught downstream by slot arity and the ``expect`` contract.
AUTO_APPLY_THRESHOLD = 0.50
#: The negatives' ceiling. Single-entity mandates (the micro/format tiers) sit just under it
#: because MiniLM barely encodes cardinality; the structural backstop for them is slot-fill,
#: where a list slot's ``min_arity`` rejects a one-entity mandate outright.
SUGGEST_THRESHOLD = 0.45
#: Minimum gap to the runner-up before a top hit may be applied — WAIVED when the runner-up
#: declares the same archetype (two phrasings of one shape are not a real ambiguity). Sized
#: just under the tightest correct positive margin on the eval set (task 072, +0.04 over the
#: argmax template, whose fan-out is identical and differs only in aggregation).
MIN_MARGIN = 0.04
#: Additive prior for a hit whose archetype matches the deterministic shape classifier.
ARCHETYPE_BOOST = 0.03

DECISION_AUTO_APPLY = "auto_apply"
DECISION_SUGGEST = "suggest"
DECISION_NO_MATCH = "fallthrough_no_match"
DECISION_LOW_MARGIN = "fallthrough_low_margin"

CALL_SITE_AUTO = "auto_pre_expansion"
CALL_SITE_ON_DEMAND = "on_demand"


@dataclass(frozen=True)
class TemplateMatch:
    """One ranked candidate template."""

    template_id: str
    archetype: str
    #: Post-boost score — what the thresholds are applied to.
    similarity: float
    #: Pre-boost cosine similarity, kept so the log can separate the two signals.
    raw_similarity: float
    archetype_boost: float = 0.0
    title: str = ""
    #: True when the id came back from Chroma but no template file declares it (a stale
    #: index entry): ranked but never applicable.
    known: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {
            "template_id": self.template_id,
            "archetype": self.archetype,
            "similarity": round(self.similarity, 4),
            "raw_similarity": round(self.raw_similarity, 4),
            "archetype_boost": round(self.archetype_boost, 4),
            "known": self.known,
        }


@dataclass(frozen=True)
class RetrievalResult:
    """The outcome of one retrieval attempt — always returned, never raised."""

    #: uuid4 join key: the retrieval log keys on it, and the contract log's
    #: ``template_id`` rows join back to it offline ("was the match good" vs "did the leaf
    #: deliver" are different questions, hence two logs).
    retrieval_id: str
    query_text: str
    decision: str
    candidates: List[TemplateMatch] = field(default_factory=list)
    template_id: Optional[str] = None
    archetype: Optional[str] = None
    similarity: Optional[float] = None
    margin_over_second: Optional[float] = None
    archetype_hint: Optional[str] = None
    reason: str = ""

    @property
    def applicable(self) -> bool:
        """True when a template was selected (auto-apply or suggest)."""
        return self.decision in (DECISION_AUTO_APPLY, DECISION_SUGGEST)

    @property
    def auto_apply(self) -> bool:
        return self.decision == DECISION_AUTO_APPLY

    def __bool__(self) -> bool:
        return self.applicable

    def as_dict(self) -> Dict[str, Any]:
        return {
            "retrieval_id": self.retrieval_id,
            "query_text": self.query_text,
            "decision": self.decision,
            "template_id": self.template_id,
            "archetype": self.archetype,
            "similarity": None if self.similarity is None else round(self.similarity, 4),
            "margin_over_second": (
                None if self.margin_over_second is None else round(self.margin_over_second, 4)
            ),
            "archetype_hint": self.archetype_hint,
            "reason": self.reason,
            "results": [m.as_dict() for m in self.candidates],
        }


@dataclass(frozen=True)
class FillOutcome:
    """What :meth:`PlanLibrary.fill` made of a match — expansion, or a downgraded result."""

    #: The GoT candidates, or ``None`` when the slot values did not fill the template.
    expansion: Optional[PlanLibraryExpansion]
    #: The retrieval result, DOWNGRADED to ``fallthrough_no_match`` on a fill failure so
    #: the caller logs (and acts on) one consistent decision.
    retrieval: RetrievalResult
    error: Optional[str] = None
    #: The slot values the template was bound with — carried so the caller can log them
    #: (``retrieval_log.record_retrieval(slot_values=...)``) without re-deriving them, which
    #: matters most when :meth:`PlanLibrary.fill_from_query` extracted them itself. ``None``
    #: when extraction never produced a mapping.
    slot_values: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        return self.expansion is not None

    def __bool__(self) -> bool:
        return self.ok


# --------------------------------------------------------------------------------------
# query text
# --------------------------------------------------------------------------------------


def build_query_text(node: "IdeaNode", is_root: bool = False) -> str:
    """The intent-level text a node is matched against templates with.

    The root node IS the mandate, so its title is the whole query. Deeper nodes carry their
    intent across several detail keys, joined by ``" | "`` — the goal (or the original goal it
    was rewritten from), the parent's goal for context, and the ``expect`` contract, which is
    often the single most template-shaped sentence on the node.
    """
    title = str(getattr(node, "title", "") or "").strip()
    if is_root:
        return title
    details = getattr(node, "details", None) or {}
    parts = [
        title,
        details.get(DetailKey.GOAL.value) or details.get(DetailKey.ORIGINAL_GOAL.value),
        details.get(DetailKey.PARENT_GOAL.value),
        details.get(DetailKey.EXPECT.value),
    ]
    return " | ".join(p.strip() for p in parts if isinstance(p, str) and p.strip())


# --------------------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------------------


def similarity_from_distance(distance: Any, space: str = "cosine") -> float:
    """Convert a Chroma distance into a cosine similarity in ``[-1, 1]``.

    ``cosine``/``ip`` distances are ``1 - cos``. Chroma's DEFAULT ``l2`` space returns the
    SQUARED euclidean distance, which for the unit-norm embeddings chroma's bundled MiniLM
    produces is exactly ``2 - 2cos`` — hence ``1 - d/2``. (That identity is why a
    default-created collection still ranks correctly; it is not why the thresholds are
    calibrated, which is what :data:`COLLECTION_METADATA` is for.)
    """
    try:
        d = float(distance)
    except (TypeError, ValueError):
        return 0.0
    sim = 1.0 - (d / 2.0) if str(space or "").lower() == "l2" else 1.0 - d
    return max(-1.0, min(1.0, sim))


def template_hash(template: PlanTemplate) -> str:
    """Content hash of a template — the sync lockfile's key.

    Hashes the NORMALIZED template (so cosmetic JSON edits — key order, whitespace, an
    omitted optional field — do not force a re-embed) but the WHOLE of it rather than just
    the embedded document, so the manifest is an honest record of "the library as synced"
    and the startup drift check can see any edit at all.
    """
    payload = json.dumps(_template_dict(template), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _template_dict(template: PlanTemplate) -> Dict[str, Any]:
    from dataclasses import asdict

    raw = asdict(template)
    for slot in raw.get("slots") or []:
        slot["kind"] = getattr(slot["kind"], "value", slot["kind"])
        slot["extraction"] = getattr(slot["extraction"], "value", slot["extraction"])
    return raw


def document_text(template: PlanTemplate) -> str:
    """What actually gets embedded for a template.

    ``embedding_text`` is authored directly (description + concrete trigger phrases harvested
    from the real task statements the template generalizes — a manual HyDE anchor), because
    MiniLM-class embeddings are surface-form sensitive and an abstract-only spec under-matches
    concrete phrasing. Falls back to title/description only if a template omits it.
    """
    text = (template.embedding_text or "").strip()
    if text:
        return text
    return " ".join(p for p in (template.title, template.description) if p).strip()


def document_metadata(template: PlanTemplate) -> Dict[str, Any]:
    """Metadata stored alongside the embedded document (used for the archetype rerank)."""
    provenance = template.provenance or {}
    return {
        "template_id": template.template_id,
        "archetype": template.archetype,
        "title": template.title,
        "schema_version": template.schema_version,
        "source": str(provenance.get("source") or "hand_authored"),
        "based_on_tasks": list(provenance.get("based_on_tasks") or []),
        "content_hash": template_hash(template),
    }


# --------------------------------------------------------------------------------------
# the library
# --------------------------------------------------------------------------------------


class PlanLibrary:
    """The in-process template corpus plus its Chroma-backed ranking."""

    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        *,
        collection: str = COLLECTION_NAME,
        warn_on_drift: bool = True,
    ) -> None:
        self.templates_dir = Path(templates_dir) if templates_dir else TEMPLATES_DIR
        self.collection = collection
        self.templates: Dict[str, PlanTemplate] = {}
        #: ``{path: error}`` for every file that failed to load — a broken template is
        #: skipped with a warning, never a crash of the whole library.
        self.load_errors: Dict[str, str] = {}
        self._space: Optional[str] = None
        self._warned_empty_index = False
        self._load()
        if warn_on_drift:
            self._warn_on_drift()

    # -- corpus -------------------------------------------------------------------------

    def _load(self) -> None:
        if not self.templates_dir.is_dir():
            logger.warning("Plan library: no templates directory at %s", self.templates_dir)
            return
        for path in sorted(self.templates_dir.glob("*.json")):
            if path.name.startswith("_"):  # _manifest.json and friends
                continue
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                template = validate_template(raw)
            except (OSError, ValueError, TemplateValidationError) as exc:
                self.load_errors[path.name] = str(exc)
                logger.warning("Plan library: skipping invalid template %s: %s", path.name, exc)
                continue
            if template.template_id in self.templates:
                self.load_errors[path.name] = f"duplicate template_id '{template.template_id}'"
                logger.warning(
                    "Plan library: duplicate template_id '%s' in %s — keeping the first",
                    template.template_id, path.name,
                )
                continue
            self.templates[template.template_id] = template

    def __len__(self) -> int:
        return len(self.templates)

    def get(self, template_id: Optional[str]) -> Optional[PlanTemplate]:
        return self.templates.get(str(template_id)) if template_id else None

    def manifest_path(self) -> Path:
        return self.templates_dir / MANIFEST_NAME

    def read_manifest(self) -> Dict[str, str]:
        """``{template_id: content_hash}`` as last synced (``{}`` when absent/unreadable)."""
        path = self.manifest_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}

    def drift(self) -> Dict[str, str]:
        """``{template_id: added|changed|removed}`` vs the manifest — Chroma-free."""
        manifest = self.read_manifest()
        out: Dict[str, str] = {}
        for template_id, template in self.templates.items():
            recorded = manifest.get(template_id)
            if recorded is None:
                out[template_id] = "added"
            elif recorded != template_hash(template):
                out[template_id] = "changed"
        for template_id in manifest:
            if template_id not in self.templates:
                out[template_id] = "removed"
        return out

    def _warn_on_drift(self) -> None:
        """Warn — never block — when the corpus has moved since the last sync.

        Sync is an explicit authoring step (edit template -> run sync -> commit both), NOT
        something an engine startup does: re-embedding from every worker subprocess is exactly
        the Chroma write contention this codebase already had to fix once.

        DELIBERATELY Chroma-free, unlike :meth:`indexed_ids`: this runs in ``__init__`` (which
        is synchronous, while every Chroma call is async) on the engine's startup path, once
        per worker subprocess — the one place an extra round-trip is NOT free. So it answers
        only "has the corpus changed since it was synced", never "is THIS collection
        populated". The latter is checked where it can be awaited and where it matters: the
        sync script gates on it, and :meth:`_warn_index_looks_unsynced` catches an unsynced
        collection at the first retrieval, at zero extra I/O.
        """
        drifted = self.drift()
        if drifted:
            logger.warning(
                "Plan library: %d template(s) differ from %s (%s) — run "
                "scripts/sync_plan_library.py to re-index",
                len(drifted), MANIFEST_NAME,
                ", ".join(f"{k}:{v}" for k, v in sorted(drifted.items())),
            )

    # -- index --------------------------------------------------------------------------

    async def ensure_collection(self, connector_chroma: Any) -> Any:
        """Get (creating it with the cosine space) the ranking collection.

        Always called BEFORE any query: ``query_chroma`` would otherwise create the
        collection itself with chroma's default ``l2`` space, and a later sync could not
        change it — silently miscalibrating every threshold in this module.
        """
        try:
            coll = await connector_chroma.get_or_create_collection(
                self.collection, metadata=dict(COLLECTION_METADATA)
            )
        except TypeError:
            # A connector/stub without the metadata kwarg: fall back, and let
            # ``read_collection_space`` report whatever space that collection actually has.
            coll = await connector_chroma.get_or_create_collection(self.collection)
        if coll is not None and self._space is None:
            self._space = read_collection_space(coll)
        return coll

    async def indexed_ids(
        self, connector_chroma: Any, template_ids: Sequence[str]
    ) -> Optional[Set[str]]:
        """Which of ``template_ids`` the TARGET collection actually holds.

        The manifest answers "was this template ever embedded", NOT "is it in THIS index":
        the manifest is committed to the repo, while the collection is per-environment (an
        embedded SQLite path, or wherever ``CHROMA_URL`` points). Trusting the manifest alone
        makes a fresh checkout aimed at an empty collection look permanently up to date — so
        anything that must know the index really matches the corpus asks here instead.

        Returns ``None`` when membership is UNKNOWABLE (no connector, no collection, a
        malformed reply), which a caller must distinguish from an empty set: "verified
        absent" means embed it, "could not check" means say so.
        """
        wanted = [str(t) for t in template_ids or ()]
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
            logger.warning("Plan library: index membership check failed: %s", exc)
            return None
        found = raw.get("ids") if isinstance(raw, Mapping) else None
        if not isinstance(found, list):
            return None
        if found and isinstance(found[0], list):
            # ``get`` returns a FLAT id list; only ``query`` nests one row per query text.
            found = [i for row in found for i in row]
        return {str(i) for i in found}

    def _warn_index_looks_unsynced(self) -> None:
        """Once per instance: a corpus on disk that the index ranks nothing from.

        The startup drift check cannot see this (it never touches Chroma — see
        :meth:`_warn_on_drift`), so this is where a never-synced collection becomes visible
        at runtime, instead of silently degrading every retrieval to "no match" forever.
        Costs no extra I/O: it fires on a query that already came back empty.
        """
        if self._warned_empty_index:
            return
        self._warned_empty_index = True
        logger.warning(
            "Plan library: collection '%s' ranked none of the %d template(s) on disk — if "
            "this repeats it is probably unsynced (run scripts/sync_plan_library.py against "
            "this chroma)",
            self.collection, len(self.templates),
        )

    async def retrieve(
        self,
        connector_chroma: Any,
        query_text: str,
        *,
        top_k: int = TOP_K,
        archetype_hint: Optional[str] = None,
        auto_apply_threshold: float = AUTO_APPLY_THRESHOLD,
        suggest_threshold: float = SUGGEST_THRESHOLD,
        min_margin: float = MIN_MARGIN,
    ) -> RetrievalResult:
        """Rank templates for ``query_text`` and decide what may be done with the top hit.

        Never raises: any missing/failed/empty piece of the pipeline returns a
        ``fallthrough_no_match`` result carrying the reason, so the caller falls back to
        organic expansion.
        """
        retrieval_id = uuid4().hex
        query = " ".join(str(query_text or "").split())
        if not query:
            return _no_match(retrieval_id, query, archetype_hint, "empty query")
        if not self.templates:
            return _no_match(retrieval_id, query, archetype_hint, "empty plan library")

        matches = await self._rank(connector_chroma, query, top_k=top_k)
        if not matches:
            self._warn_index_looks_unsynced()
            return _no_match(retrieval_id, query, archetype_hint, "no index matches")

        matches = self._apply_archetype_prior(matches, archetype_hint)
        return _decide(
            retrieval_id=retrieval_id,
            query=query,
            matches=matches,
            archetype_hint=archetype_hint,
            auto_apply_threshold=auto_apply_threshold,
            suggest_threshold=suggest_threshold,
            min_margin=min_margin,
        )

    async def _rank(
        self, connector_chroma: Any, query: str, *, top_k: int
    ) -> List[TemplateMatch]:
        """Top-K by embedding similarity. Returns ``[]`` on any Chroma failure."""
        if connector_chroma is None:
            return []
        n_results = max(1, min(int(top_k or TOP_K), len(self.templates)))
        try:
            await self.ensure_collection(connector_chroma)
            raw = await connector_chroma.query_chroma(
                collection=self.collection, query_texts=[query], n_results=n_results
            )
        except Exception as exc:  # noqa: BLE001 — retrieval degrades, it never breaks a run
            logger.warning("Plan library: retrieval query failed: %s", exc)
            return []
        return self._parse_matches(raw)

    def _parse_matches(self, raw: Any) -> List[TemplateMatch]:
        if not isinstance(raw, Mapping):
            return []
        ids = _first_row(raw.get("ids"))
        distances = _first_row(raw.get("distances"))
        metadatas = _first_row(raw.get("metadatas"))
        space = self._space or "cosine"
        out: List[TemplateMatch] = []
        for index, template_id in enumerate(ids):
            template_id = str(template_id)
            distance = distances[index] if index < len(distances) else None
            meta = metadatas[index] if index < len(metadatas) else None
            meta = meta if isinstance(meta, Mapping) else {}
            template = self.templates.get(template_id)
            archetype = str(
                (template.archetype if template else None) or meta.get("archetype") or ""
            )
            similarity = similarity_from_distance(distance, space)
            out.append(
                TemplateMatch(
                    template_id=template_id,
                    archetype=archetype,
                    similarity=similarity,
                    raw_similarity=similarity,
                    title=str((template.title if template else None) or meta.get("title") or ""),
                    known=template is not None,
                )
            )
        return out

    def _apply_archetype_prior(
        self, matches: Sequence[TemplateMatch], archetype_hint: Optional[str]
    ) -> List[TemplateMatch]:
        """Soft rerank: nudge hits whose archetype matches the deterministic shape hint."""
        hint = str(archetype_hint or "").strip().lower()
        if not hint:
            return list(matches)
        boosted = [
            replace(
                m,
                similarity=min(1.0, m.raw_similarity + ARCHETYPE_BOOST),
                archetype_boost=ARCHETYPE_BOOST,
            )
            if m.archetype.strip().lower() == hint
            else m
            for m in matches
        ]
        boosted.sort(key=lambda m: m.similarity, reverse=True)
        return boosted

    # -- filling ------------------------------------------------------------------------

    def fill(
        self,
        result: RetrievalResult,
        slot_values: Mapping[str, Any],
        *,
        origin: str = ORIGIN_AUTO,
        contracts: Any = None,
    ) -> FillOutcome:
        """Bind ``slot_values`` into the retrieved template and adapt it into GoT candidates.

        A slot-fill failure is NOT an error the caller has to handle: the outcome carries a
        retrieval result downgraded to ``fallthrough_no_match``, so a half-filled template can
        never reach the engine. (The adapter itself stays retrieval-unaware — this downgrade is
        retrieval's policy, not the adapter's.)

        Takes slot values the caller already has. To derive them FROM the query instead — the
        automatic path — use :meth:`fill_from_query`, which extracts and then calls this.
        """
        values = dict(slot_values) if isinstance(slot_values, Mapping) else None
        if not result.applicable:
            return FillOutcome(
                expansion=None,
                retrieval=result,
                error=result.reason or "no match",
                slot_values=values,
            )
        template = self.get(result.template_id)
        if template is None:
            return FillOutcome(
                expansion=None,
                retrieval=_downgrade(result, f"template '{result.template_id}' not in the library"),
                error="unknown template",
                slot_values=values,
            )
        try:
            expansion = candidates_from_template(
                template, slot_values, origin=origin, contracts=contracts
            )
        except TemplateValidationError as exc:
            logger.info(
                "Plan library: '%s' matched (%.3f) but slot-fill failed: %s",
                result.template_id, result.similarity or 0.0, exc,
            )
            return FillOutcome(
                expansion=None,
                retrieval=_downgrade(result, f"slot fill failed: {exc}"),
                error=str(exc),
                slot_values=values,
            )
        return FillOutcome(expansion=expansion, retrieval=result, slot_values=values)

    async def fill_from_query(
        self,
        result: RetrievalResult,
        *,
        mandate: str = "",
        query_text: str = "",
        io: Any = None,
        origin: str = ORIGIN_AUTO,
        contracts: Any = None,
        model_name: Optional[str] = None,
        fallback_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ) -> FillOutcome:
        """Step 4 end to end: extract the matched template's slot values from the query, then
        :meth:`fill` it.

        This is the entry point the engine wants — ``retrieve()`` then ``fill_from_query()`` —
        so a caller never has to know that slot extraction exists, let alone which strategy each
        slot declares. Slot extraction is where the ONE LLM call of this pipeline happens, which
        is why it lives behind an ``async`` method here and not inside the synchronous
        :meth:`fill`: it must never fire speculatively, only after a match has already qualified.

        Fails toward silence like every other step: an unfillable (or unextractable) template
        comes back as a downgraded ``fallthrough_no_match`` outcome, never an exception.
        """
        if not result.applicable:
            return FillOutcome(expansion=None, retrieval=result, error=result.reason or "no match")
        template = self.get(result.template_id)
        if template is None:
            return FillOutcome(
                expansion=None,
                retrieval=_downgrade(result, f"template '{result.template_id}' not in the library"),
                error="unknown template",
            )
        # Imported here, not at module scope: slot_fill reaches into the LLM stack, and the
        # sync script / eval script / offline tests use this module without ever filling.
        from agent.app.plan_library.slot_fill import extract_slot_values

        # ``timeout_seconds=None`` means "no timeout" to the connector, so an unset knob must
        # leave slot_fill's own default in place rather than overwrite it with None.
        overrides = {} if timeout_seconds is None else {"timeout_seconds": timeout_seconds}
        try:
            slot_values = await extract_slot_values(
                template,
                query_text or result.query_text,
                mandate,
                io,
                model_name=model_name,
                fallback_model=fallback_model,
                **overrides,
            )
        except TemplateValidationError as exc:
            logger.info(
                "Plan library: '%s' matched (%.3f) but slot extraction failed: %s",
                result.template_id, result.similarity or 0.0, exc,
            )
            return FillOutcome(
                expansion=None,
                retrieval=_downgrade(result, f"slot extraction failed: {exc}"),
                error=str(exc),
            )
        except Exception as exc:  # noqa: BLE001 — extraction degrades, it never breaks a run
            logger.warning("Plan library: slot extraction errored for '%s': %s", result.template_id, exc)
            return FillOutcome(
                expansion=None,
                retrieval=_downgrade(result, f"slot extraction errored: {exc}"),
                error=str(exc),
            )
        return self.fill(result, slot_values, origin=origin, contracts=contracts)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def archetype_hint_for(mandate: str) -> Optional[str]:
    """The deterministic shape prior for a mandate (``None`` when unrecognised).

    Thin wrapper over ``shape_classifier.classify_shape`` so callers do not have to know
    where the prior comes from — and so the two stay one import apart if it is ever widened.
    """
    try:
        return classify_shape(mandate or "")
    except Exception:  # noqa: BLE001 — a prior must never break retrieval
        return None


def _first_row(value: Any) -> List[Any]:
    """Chroma returns one row per query text; this module always sends exactly one."""
    if isinstance(value, list) and value and isinstance(value[0], list):
        return value[0]
    return list(value) if isinstance(value, list) else []


def read_collection_space(collection: Any) -> str:
    """The distance space a live collection actually uses (best effort, defaults cosine).

    Public because :mod:`agent.app.idea_memory` needs the same answer for the ``mem_*``
    collections, and the pairing with :func:`similarity_from_distance` must not be
    re-derived per subsystem.
    """
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
    retrieval_id: str, query: str, archetype_hint: Optional[str], reason: str
) -> RetrievalResult:
    return RetrievalResult(
        retrieval_id=retrieval_id,
        query_text=query,
        decision=DECISION_NO_MATCH,
        archetype_hint=archetype_hint,
        reason=reason,
    )


def _downgrade(result: RetrievalResult, reason: str) -> RetrievalResult:
    """Post-hoc "no match" that keeps the ranking for the log (fail toward silence)."""
    return replace(
        result, decision=DECISION_NO_MATCH, template_id=None, archetype=None, reason=reason
    )


def _decide(
    *,
    retrieval_id: str,
    query: str,
    matches: List[TemplateMatch],
    archetype_hint: Optional[str],
    auto_apply_threshold: float,
    suggest_threshold: float,
    min_margin: float,
) -> RetrievalResult:
    """Threshold + margin policy over a ranked list. Pure, so the tests can pin all four
    outcomes without any Chroma at all."""
    top = matches[0]
    runner_up = matches[1] if len(matches) > 1 else None
    margin = None if runner_up is None else top.similarity - runner_up.similarity

    def result(decision: str, reason: str, selected: bool = True) -> RetrievalResult:
        return RetrievalResult(
            retrieval_id=retrieval_id,
            query_text=query,
            decision=decision,
            candidates=matches,
            template_id=top.template_id if selected else None,
            archetype=top.archetype if selected else None,
            similarity=top.similarity,
            margin_over_second=margin,
            archetype_hint=archetype_hint,
            reason=reason,
        )

    if not top.known:
        return result(
            DECISION_NO_MATCH,
            f"top hit '{top.template_id}' is a stale index entry (no template file)",
            selected=False,
        )
    if top.similarity < suggest_threshold:
        return result(
            DECISION_NO_MATCH,
            f"top similarity {top.similarity:.3f} < suggest threshold {suggest_threshold:.2f}",
            selected=False,
        )
    if top.similarity < auto_apply_threshold:
        # No margin gate here on purpose: a suggestion is offered, not applied, so the
        # cost of a thin gap is a second option in a prompt rather than a wrong subtree.
        return result(
            DECISION_SUGGEST,
            f"top similarity {top.similarity:.3f} < auto-apply threshold "
            f"{auto_apply_threshold:.2f}",
        )
    # Margin is only meaningful between DIFFERENT shapes: two templates of the same
    # archetype are two phrasings of one strategy, so a thin gap there is not ambiguity.
    same_archetype = (
        runner_up is not None
        and bool(top.archetype)
        and top.archetype.strip().lower() == runner_up.archetype.strip().lower()
    )
    if margin is not None and margin < min_margin and not same_archetype:
        return result(
            DECISION_LOW_MARGIN,
            f"margin {margin:.3f} over '{runner_up.template_id}' < {min_margin:.2f}",
            selected=False,
        )
    return result(DECISION_AUTO_APPLY, f"similarity {top.similarity:.3f}")
