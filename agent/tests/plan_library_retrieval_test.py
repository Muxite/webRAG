"""Offline unit tests for plan-library retrieval — free, no Chroma, no embedding model.

Retrieval is four separable pieces and each is pinned here without any live service:

* :func:`build_query_text` — what a node is matched *on*;
* distance -> similarity, including chroma's DEFAULT ``l2`` space (where ``1 - distance``
  would be wrong, which is the whole reason the collection is created as cosine);
* the decision policy — all four outcomes (``auto_apply`` / ``suggest`` /
  ``fallthrough_no_match`` / ``fallthrough_low_margin``) plus the archetype-hint boost and
  the same-archetype margin waiver;
* fail-toward-silence — a slot-fill failure on an otherwise-adopted match downgrades to "no
  match" rather than raising, so a half-filled template can never reach the engine.

The Chroma stand-in is a fake in the shape of ``ConnectorChroma``'s two methods
(``get_or_create_collection`` / ``query_chroma``), following ``idea_dag_error_modes_test``'s
``MockChroma`` pattern: retrieval must degrade to "no match" on any failure, so the fake also
covers the raising / empty / stale-entry paths.

The corresponding LIVE checks (real embeddings, real Chroma, real task statements) live in
``scripts/eval_plan_library_retrieval.py`` — deliberately a script, so the offline suite keeps
zero live-service dependencies.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from agent.app.idea_dag import IdeaDag
from agent.app.idea_policies.base import DetailKey
from agent.app.plan_library import retrieval as R
from agent.app.plan_library import retrieval_log
from agent.app.plan_library.schema import (
    LeafBlueprint,
    PlanTemplate,
    SlotKind,
    SlotSpec,
)


# --------------------------------------------------------------------------------------
# fakes
# --------------------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self, space: str = "cosine"):
        self.configuration_json = {"hnsw": {"space": space}}
        self.metadata = {"hnsw:space": space}


class FakeChroma:
    """A ``ConnectorChroma``-shaped stand-in: only the three methods the library calls."""

    def __init__(self, hits=(), *, space: str = "cosine", raises: bool = False,
                 collection_is_none: bool = False, result=None, indexed=(),
                 get_raises: bool = False):
        #: ``[(template_id, distance, archetype), ...]`` in rank order.
        self.hits = list(hits)
        self.space = space
        self.raises = raises
        self.collection_is_none = collection_is_none
        self.result = result
        #: ids the collection actually HOLDS — independent of ``hits``, because the whole
        #: point of the membership check is that the index can disagree with the manifest.
        self.indexed = set(indexed)
        self.get_raises = get_raises
        self.queries = []
        self.created = []
        self.gets = []
        self.added = []
        self.deleted = []

    async def get_or_create_collection(self, collection, metadata=None):
        self.created.append((collection, metadata))
        return None if self.collection_is_none else _FakeCollection(self.space)

    async def get_from_chroma(self, collection, ids, include=None):
        self.gets.append({"collection": collection, "ids": list(ids)})
        if self.get_raises:
            raise RuntimeError("chroma is down")
        # Real chroma returns ONLY the ids it holds; an absent id is simply not in the reply.
        return {"ids": [i for i in ids if i in self.indexed]}

    # -- write side: only the sync script uses these ------------------------------------

    async def add_to_chroma(self, collection, ids, metadatas, documents):
        self.indexed.update(ids)
        self.added.extend(ids)
        return True

    async def delete_from_chroma(self, collection, ids):
        self.indexed.difference_update(ids)
        self.deleted.extend(ids)
        return True

    async def query_chroma(self, collection, query_texts, n_results=3, where=None):
        self.queries.append({"collection": collection, "q": list(query_texts), "n": n_results})
        if self.raises:
            raise RuntimeError("chroma is down")
        if self.result is not None:
            return self.result
        hits = self.hits[:n_results]
        return {
            "ids": [[h[0] for h in hits]],
            "distances": [[h[1] for h in hits]],
            "metadatas": [[{"template_id": h[0], "archetype": h[2]} for h in hits]],
        }


def _template(template_id: str, archetype: str, *, min_arity: int = 2) -> PlanTemplate:
    return PlanTemplate(
        template_id=template_id,
        archetype=archetype,
        title=template_id,
        embedding_text=f"{template_id} {archetype}",
        slots=[
            SlotSpec(name="candidates", kind=SlotKind.ENTITY_LIST, min_arity=min_arity),
            SlotSpec(name="field", kind=SlotKind.FIELD),
        ],
        leaves=[
            LeafBlueprint(
                id_pattern="<<item.key>>_leaf",
                for_each="candidates",
                instruction="Open the page for <<item.name>> and read its <<field>>.",
                expect="<<item.name>>: its <<field>> -- source URL",
            )
        ],
        aggregation="Compare every <<field>> and name the winner.",
    )


def _load_script(name: str):
    """The sync script is a script, not a package — load it by path (``benchmark_tooling_test``
    pattern) so its planning logic can be unit-tested without a live chroma."""
    root = Path(__file__).resolve()
    for _ in range(8):
        if (root / "scripts" / f"{name}.py").exists():
            break
        root = root.parent
    else:  # pragma: no cover - repo layout changed
        raise RuntimeError("repo root not found")
    spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


sync_script = _load_script("sync_plan_library")


def _library(tmp_path, templates=None) -> R.PlanLibrary:
    """A library backed by a temp corpus — never the real ``templates/`` directory."""
    templates = templates or [_template("argmax_t", "argmax"), _template("count_t", "count")]
    for template in templates:
        path = tmp_path / f"{template.template_id}.json"
        path.write_text(json.dumps(R._template_dict(template)), encoding="utf-8")
    return R.PlanLibrary(templates_dir=tmp_path, warn_on_drift=False)


# --------------------------------------------------------------------------------------
# query text
# --------------------------------------------------------------------------------------


def test_build_query_text_root_is_the_mandate():
    graph = IdeaDag("Which of these six peaks is the most prominent?", {})
    root = graph.get_node(graph.root_id())
    assert R.build_query_text(root, is_root=True) == "Which of these six peaks is the most prominent?"


def test_build_query_text_child_joins_intent_fields_only():
    graph = IdeaDag("root mandate", {})
    child = graph.add_child(
        graph.root_id(),
        "Read Kongur Tagh's prominence",
        {
            DetailKey.GOAL.value: "find the prominence of Kongur Tagh",
            DetailKey.PARENT_GOAL.value: "compare six peaks",
            DetailKey.EXPECT.value: "prominence in metres -- source URL",
            # evidence-level details must NOT leak into the query
            DetailKey.ACTION_RESULT.value: {"content": "Kongur Tagh is 7,649 m tall"},
        },
    )
    text = R.build_query_text(child, is_root=False)
    assert text == (
        "Read Kongur Tagh's prominence | find the prominence of Kongur Tagh | "
        "compare six peaks | prominence in metres -- source URL"
    )
    assert "7,649" not in text


def test_build_query_text_falls_back_to_original_goal_and_skips_blanks():
    graph = IdeaDag("root", {})
    child = graph.add_child(
        graph.root_id(),
        "  Titled  ",
        {DetailKey.ORIGINAL_GOAL.value: "the original goal", DetailKey.PARENT_GOAL.value: "   "},
    )
    assert R.build_query_text(child) == "Titled | the original goal"


# --------------------------------------------------------------------------------------
# similarity
# --------------------------------------------------------------------------------------


def test_similarity_cosine_is_one_minus_distance():
    assert R.similarity_from_distance(0.25, "cosine") == pytest.approx(0.75)
    assert R.similarity_from_distance(0.0, "cosine") == pytest.approx(1.0)


def test_similarity_l2_is_halved_because_chroma_returns_squared_distance():
    # chroma's DEFAULT space: for unit-norm embeddings d = 2 - 2cos, so cos = 1 - d/2.
    # Reading it as ``1 - d`` would put a 0.75-similar template at 0.5 and silently
    # miscalibrate every threshold in the module.
    assert R.similarity_from_distance(0.5, "l2") == pytest.approx(0.75)
    assert R.similarity_from_distance(0.5, "cosine") == pytest.approx(0.5)


def test_similarity_is_clamped_and_survives_garbage():
    assert R.similarity_from_distance(9.0, "cosine") == -1.0
    assert R.similarity_from_distance(None, "cosine") == 0.0
    assert R.similarity_from_distance("nope", "cosine") == 0.0


@pytest.mark.asyncio
async def test_l2_collection_is_read_from_the_live_collection(tmp_path):
    """A pre-existing l2 collection must still yield calibrated similarities."""
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.5, "argmax")], space="l2")
    result = await library.retrieve(chroma, "which peak is most prominent")
    assert result.similarity == pytest.approx(0.75)
    # and the cosine space is REQUESTED, so a fresh collection never gets chroma's l2 default
    assert chroma.created[0][1] == {"hnsw:space": "cosine"}


# --------------------------------------------------------------------------------------
# decision policy
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_apply_on_a_clear_winner(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.30, "argmax"), ("count_t", 0.60, "count")])
    result = await library.retrieve(chroma, "which of these six peaks is most prominent")
    assert result.decision == R.DECISION_AUTO_APPLY
    assert result.template_id == "argmax_t"
    assert result.archetype == "argmax"
    assert result.similarity == pytest.approx(0.70)
    assert result.margin_over_second == pytest.approx(0.30)
    assert result.applicable and result.auto_apply and bool(result)
    assert len(result.retrieval_id) == 32  # uuid4 hex — the join key for the logs


@pytest.mark.asyncio
async def test_suggest_between_the_two_thresholds(tmp_path):
    library = _library(tmp_path)
    mid = 1.0 - (R.SUGGEST_THRESHOLD + R.AUTO_APPLY_THRESHOLD) / 2
    chroma = FakeChroma([("argmax_t", mid, "argmax"), ("count_t", 0.95, "count")])
    result = await library.retrieve(chroma, "roughly argmax-ish")
    assert result.decision == R.DECISION_SUGGEST
    assert result.template_id == "argmax_t"  # selected, just not auto-applied
    assert result.applicable and not result.auto_apply


@pytest.mark.asyncio
async def test_no_match_below_the_suggest_threshold(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.99, "argmax")])
    result = await library.retrieve(chroma, "report the maximum depth of one lake")
    assert result.decision == R.DECISION_NO_MATCH
    assert result.template_id is None
    assert not result.applicable and not bool(result)
    assert result.candidates  # the ranking is still logged
    assert "suggest threshold" in result.reason


@pytest.mark.asyncio
async def test_low_margin_blocks_a_confident_but_ambiguous_hit(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.25, "argmax"), ("count_t", 0.26, "count")])
    result = await library.retrieve(chroma, "ambiguous between two shapes")
    assert result.decision == R.DECISION_LOW_MARGIN
    assert result.template_id is None
    assert result.similarity == pytest.approx(0.75)
    assert "count_t" in result.reason


@pytest.mark.asyncio
async def test_margin_is_waived_between_two_templates_of_one_archetype(tmp_path):
    """Two phrasings of ONE shape are not a real ambiguity — the margin gate must not fire."""
    library = _library(
        tmp_path, [_template("argmax_a", "argmax"), _template("argmax_b", "argmax")]
    )
    chroma = FakeChroma([("argmax_a", 0.25, "argmax"), ("argmax_b", 0.26, "argmax")])
    result = await library.retrieve(chroma, "argmax either way")
    assert result.decision == R.DECISION_AUTO_APPLY
    assert result.template_id == "argmax_a"


@pytest.mark.asyncio
async def test_archetype_hint_reorders_but_never_decides_on_its_own(tmp_path):
    """The prior is a nudge, not a vote: a boost-driven flip lands in the ambiguous band.

    A reorder means the two hits were within ``ARCHETYPE_BOOST`` of each other, which is
    inside ``MIN_MARGIN`` — so the hint can change WHICH template is on top without ever
    being enough, by itself, to auto-apply it.
    """
    library = _library(tmp_path)
    chroma = FakeChroma([("count_t", 0.29, "count"), ("argmax_t", 0.30, "argmax")])
    plain = await library.retrieve(chroma, "a query")
    assert plain.candidates[0].template_id == "count_t"
    assert plain.decision == R.DECISION_LOW_MARGIN

    hinted = await library.retrieve(chroma, "a query", archetype_hint="argmax")
    top = hinted.candidates[0]
    assert top.template_id == "argmax_t"
    assert top.archetype_boost == pytest.approx(R.ARCHETYPE_BOOST)
    assert top.similarity == pytest.approx(top.raw_similarity + R.ARCHETYPE_BOOST)
    assert hinted.decision == R.DECISION_LOW_MARGIN


@pytest.mark.asyncio
async def test_archetype_hint_can_lift_a_lone_hit_over_the_threshold(tmp_path):
    """With no close rival, the prior IS enough to move a borderline hit into auto-apply."""
    library = _library(tmp_path)
    just_under = 1.0 - (R.AUTO_APPLY_THRESHOLD - 0.01)
    chroma = FakeChroma([("argmax_t", just_under, "argmax")])
    assert (await library.retrieve(chroma, "a query")).decision == R.DECISION_SUGGEST

    hinted = await library.retrieve(chroma, "a query", archetype_hint="argmax")
    assert hinted.decision == R.DECISION_AUTO_APPLY
    assert hinted.template_id == "argmax_t"


@pytest.mark.asyncio
async def test_unmatched_hint_changes_nothing(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.30, "argmax"), ("count_t", 0.60, "count")])
    hinted = await library.retrieve(chroma, "a query", archetype_hint="chain")
    assert hinted.template_id == "argmax_t"
    assert all(m.archetype_boost == 0.0 for m in hinted.candidates)


def test_archetype_hint_for_reuses_the_deterministic_classifier():
    chain = (
        "Follow a research chain - each step can only be answered by reading the previous "
        "entity's page. 1. Identify the AUTHOR of the novel. 2. Read that author's page."
    )
    assert R.archetype_hint_for(chain) == "chain"
    assert R.archetype_hint_for("") is None
    assert R.archetype_hint_for("what is the capital of France") is None


# --------------------------------------------------------------------------------------
# degradation — every failure path is a silent "no match"
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "chroma,reason",
    [
        (FakeChroma(raises=True), "no index matches"),
        (FakeChroma([]), "no index matches"),
        (FakeChroma(result={"ids": [[]], "distances": [[]]}), "no index matches"),
    ],
)
async def test_chroma_failures_degrade_to_no_match(tmp_path, chroma, reason):
    library = _library(tmp_path)
    result = await library.retrieve(chroma, "which of these is deepest")
    assert result.decision == R.DECISION_NO_MATCH
    assert result.reason == reason
    assert result.candidates == []


@pytest.mark.asyncio
async def test_an_index_that_ranks_nothing_warns_once_that_it_looks_unsynced(tmp_path, caplog):
    """The silent-forever failure mode gets ONE line: a corpus on disk, an index that has
    nothing to say about it. Costs no extra I/O — the query already came back empty."""
    library = _library(tmp_path)
    chroma = FakeChroma([])
    with caplog.at_level("WARNING"):
        await library.retrieve(chroma, "which of these is deepest")
        await library.retrieve(chroma, "which of these is tallest")
    unsynced = [r for r in caplog.records if "probably unsynced" in r.getMessage()]
    assert len(unsynced) == 1  # once per instance, not once per node
    assert library.collection in unsynced[0].getMessage()


@pytest.mark.asyncio
async def test_empty_query_and_empty_library_never_touch_chroma(tmp_path):
    chroma = FakeChroma([("argmax_t", 0.1, "argmax")])
    library = _library(tmp_path)
    empty_query = await library.retrieve(chroma, "   ")
    assert empty_query.decision == R.DECISION_NO_MATCH
    assert empty_query.reason == "empty query"

    empty_library = R.PlanLibrary(templates_dir=tmp_path / "nope", warn_on_drift=False)
    result = await empty_library.retrieve(chroma, "a real query")
    assert result.reason == "empty plan library"
    assert chroma.queries == []  # neither path issued an I/O call


@pytest.mark.asyncio
async def test_stale_index_entry_is_never_applied(tmp_path):
    """An id in Chroma with no template file on disk: disk is the source of truth."""
    library = _library(tmp_path)
    chroma = FakeChroma([("deleted_t", 0.10, "argmax"), ("argmax_t", 0.60, "argmax")])
    result = await library.retrieve(chroma, "a query")
    assert result.decision == R.DECISION_NO_MATCH
    assert result.template_id is None
    assert result.candidates[0].known is False
    assert "stale index entry" in result.reason


@pytest.mark.asyncio
async def test_top_k_is_capped_by_the_corpus_size(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.10, "argmax")])
    await library.retrieve(chroma, "a query", top_k=50)
    assert chroma.queries[0]["n"] == 2  # two templates on disk


# --------------------------------------------------------------------------------------
# corpus loading
# --------------------------------------------------------------------------------------


def test_invalid_template_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.json").write_text('{"template_id": "broken"}', encoding="utf-8")
    (tmp_path / "garbage.json").write_text("{not json", encoding="utf-8")
    library = _library(tmp_path)
    assert set(library.templates) == {"argmax_t", "count_t"}
    assert set(library.load_errors) == {"broken.json", "garbage.json"}


def test_manifest_drift_is_reported_but_never_blocks(tmp_path):
    library = _library(tmp_path)
    assert set(library.drift()) == {"argmax_t", "count_t"}  # no manifest yet -> all added

    manifest = {tid: R.template_hash(t) for tid, t in library.templates.items()}
    manifest["gone_t"] = "deadbeef"
    library.manifest_path().write_text(json.dumps(manifest), encoding="utf-8")
    fresh = R.PlanLibrary(templates_dir=tmp_path, warn_on_drift=True)
    assert fresh.drift() == {"gone_t": "removed"}
    assert set(fresh.templates) == {"argmax_t", "count_t"}  # drift changed nothing


def test_template_hash_is_stable_and_content_sensitive(tmp_path):
    library = _library(tmp_path)
    template = library.templates["argmax_t"]
    assert R.template_hash(template) == R.template_hash(template)
    edited = R.validate_template(
        {**R._template_dict(template), "embedding_text": "something else entirely"}
    )
    assert R.template_hash(edited) != R.template_hash(template)


def test_document_text_and_metadata_carry_the_archetype(tmp_path):
    library = _library(tmp_path)
    template = library.templates["argmax_t"]
    assert R.document_text(template) == template.embedding_text
    meta = R.document_metadata(template)
    assert meta["template_id"] == "argmax_t" and meta["archetype"] == "argmax"
    assert meta["content_hash"] == R.template_hash(template)


# --------------------------------------------------------------------------------------
# index membership — the manifest says "embedded", only chroma says "embedded HERE"
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_indexed_ids_reports_what_the_collection_actually_holds(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma(indexed=("argmax_t",))
    assert await library.indexed_ids(chroma, ["argmax_t", "count_t"]) == {"argmax_t"}
    assert chroma.gets == [{"collection": library.collection, "ids": ["argmax_t", "count_t"]}]
    # the cosine space is still requested first: a membership check must never be the call
    # that creates the collection with chroma's default l2 space.
    assert chroma.created[0][1] == {"hnsw:space": "cosine"}


async def _returns(value):
    return value


@pytest.mark.asyncio
async def test_indexed_ids_is_none_when_membership_cannot_be_checked(tmp_path):
    """``None`` (unknowable) is NOT ``set()`` (verified absent) — callers act differently."""
    library = _library(tmp_path)
    assert await library.indexed_ids(None, ["argmax_t"]) is None
    assert await library.indexed_ids(object(), ["argmax_t"]) is None  # connector w/o the method
    assert await library.indexed_ids(FakeChroma(collection_is_none=True), ["argmax_t"]) is None
    assert await library.indexed_ids(FakeChroma(get_raises=True), ["argmax_t"]) is None
    bad_shape = FakeChroma()
    bad_shape.get_from_chroma = lambda *a, **k: _returns(None)
    assert await library.indexed_ids(bad_shape, ["argmax_t"]) is None


@pytest.mark.asyncio
async def test_indexed_ids_on_an_empty_request_issues_no_io(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma()
    assert await library.indexed_ids(chroma, []) == set()
    assert chroma.gets == [] and chroma.created == []


# --------------------------------------------------------------------------------------
# the sync plan — the manifest is necessary but NOT sufficient
# --------------------------------------------------------------------------------------


def _synced_manifest(library) -> None:
    """Write the manifest the sync script would leave behind — "all six embedded"."""
    library.manifest_path().write_text(
        json.dumps({tid: R.template_hash(t) for tid, t in library.templates.items()}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_sync_plan_embeds_templates_the_target_collection_never_got(tmp_path, capsys):
    """THE REGRESSION: ``_manifest.json`` is committed to the repo, the collection is not.

    A fresh checkout pointed at a new/empty chroma has a manifest claiming every template is
    synced. Trusting it reported "nothing to embed", exited 0 and left an index that every
    later retrieval silently missed on, forever.
    """
    library = _library(tmp_path)
    _synced_manifest(library)
    empty_collection = FakeChroma(indexed=())

    work = await sync_script._plan(library, empty_collection, force=False)

    assert work["added"] == ["argmax_t", "count_t"]  # added: nothing indexed to delete first
    assert work["unchanged"] == [] and work["changed"] == [] and work["removed"] == []
    assert "MISSING from" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sync_plan_skips_templates_the_collection_already_holds(tmp_path):
    """The property the fix must NOT break: a correct target still costs zero re-embedding."""
    library = _library(tmp_path)
    _synced_manifest(library)
    chroma = FakeChroma(indexed=("argmax_t", "count_t"))

    work = await sync_script._plan(library, chroma, force=False)

    assert work["unchanged"] == ["argmax_t", "count_t"]
    assert work["added"] == [] and work["changed"] == [] and work["removed"] == []
    assert len(chroma.gets) == 1  # one round-trip for the whole plan, not one per template


@pytest.mark.asyncio
async def test_sync_plan_reclassifies_only_the_ids_that_are_actually_absent(tmp_path):
    library = _library(tmp_path)
    _synced_manifest(library)
    half = FakeChroma(indexed=("count_t",))

    work = await sync_script._plan(library, half, force=False)

    assert work["added"] == ["argmax_t"] and work["unchanged"] == ["count_t"]


@pytest.mark.asyncio
async def test_sync_plan_leaves_changed_and_removed_ids_alone(tmp_path):
    """A hash mismatch already means re-embed, so it is never asked about — and it stays
    ``changed`` (delete-then-add), because chroma's ``add`` keeps the stale row otherwise."""
    library = _library(tmp_path)
    manifest = {tid: R.template_hash(t) for tid, t in library.templates.items()}
    manifest["argmax_t"] = "0000deadbeef0000"
    manifest["gone_t"] = "cafe"
    library.manifest_path().write_text(json.dumps(manifest), encoding="utf-8")
    chroma = FakeChroma(indexed=("argmax_t", "count_t", "gone_t"))

    work = await sync_script._plan(library, chroma, force=False)

    assert work["changed"] == ["argmax_t"] and work["removed"] == ["gone_t"]
    assert work["unchanged"] == ["count_t"] and work["added"] == []
    assert chroma.gets == [{"collection": library.collection, "ids": ["count_t"]}]


@pytest.mark.asyncio
async def test_sync_plan_forced_never_asks_the_index(tmp_path):
    library = _library(tmp_path)
    _synced_manifest(library)
    chroma = FakeChroma(indexed=("argmax_t", "count_t"))

    work = await sync_script._plan(library, chroma, force=True)

    assert work["changed"] == ["argmax_t", "count_t"] and work["unchanged"] == []
    assert chroma.gets == []  # nothing to verify: everything is being re-embedded


@pytest.mark.asyncio
async def test_sync_plan_says_so_when_the_index_cannot_be_read(tmp_path, capsys):
    """Unreadable index: fall back to the manifest, but never quietly — the run then fails on
    ``ensure_collection`` in ``_sync`` anyway, so the plan must not pretend it was verified."""
    library = _library(tmp_path)
    _synced_manifest(library)

    work = await sync_script._plan(library, FakeChroma(get_raises=True), force=False)

    assert work["unchanged"] == ["argmax_t", "count_t"] and work["added"] == []
    assert work["unverified"] == ["argmax_t", "count_t"]
    assert "could not read" in capsys.readouterr().err


def _args(tmp_path, **overrides):
    import argparse

    return argparse.Namespace(**{
        "templates_dir": str(tmp_path), "collection": R.COLLECTION_NAME,
        "force": False, "dry_run": False, **overrides,
    })


@pytest.mark.asyncio
async def test_sync_main_embeds_into_a_collection_the_manifest_lied_about(
    tmp_path, monkeypatch, capsys
):
    """End to end, the exact reported failure: fresh checkout + brand-new empty collection.

    Before the fix this printed "nothing to embed", exited 0 and embedded NOTHING.
    """
    library = _library(tmp_path)
    _synced_manifest(library)
    chroma = FakeChroma(indexed=())
    monkeypatch.setattr(sync_script, "ConnectorConfig", lambda *a, **k: None)
    monkeypatch.setattr(sync_script, "ConnectorChroma", lambda *a, **k: chroma)

    assert await sync_script.main_async(_args(tmp_path)) == 0

    assert sorted(chroma.added) == ["argmax_t", "count_t"]
    assert chroma.deleted == []  # nothing was indexed, so nothing needed deleting first
    assert "embedded 2 template(s)" in capsys.readouterr().out

    # ...and the second run over the now-correct collection is a no-op again.
    assert await sync_script.main_async(_args(tmp_path)) == 0
    assert sorted(chroma.added) == ["argmax_t", "count_t"]
    assert "index already up to date" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_sync_main_fails_instead_of_calling_an_unreadable_index_up_to_date(
    tmp_path, monkeypatch, capsys
):
    """An unreachable chroma exiting 0 with "up to date" is the same trap, one layer out."""
    library = _library(tmp_path)
    _synced_manifest(library)
    monkeypatch.setattr(sync_script, "ConnectorConfig", lambda *a, **k: None)
    monkeypatch.setattr(sync_script, "ConnectorChroma", lambda *a, **k: FakeChroma(get_raises=True))

    assert await sync_script.main_async(_args(tmp_path)) == 1
    assert "refusing to report it up to date" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# fill — fail toward silence
# --------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fill_produces_got_candidates_on_a_hit(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.30, "argmax"), ("count_t", 0.60, "count")])
    result = await library.retrieve(chroma, "which of these six peaks is most prominent")
    outcome = library.fill(
        result,
        {
            "candidates": [{"name": "Kongur Tagh"}, {"name": "Noshaq"}],
            "field": "topographic prominence in metres",
        },
    )
    assert outcome.ok and bool(outcome)
    assert outcome.retrieval.decision == R.DECISION_AUTO_APPLY
    assert len(outcome.expansion.candidates) == 2
    first = outcome.expansion.candidates[0]["details"]
    assert first[DetailKey.ACTION.value] == "search"
    assert first[DetailKey.QUERY.value] == "Kongur Tagh topographic prominence in metres"
    assert "Kongur Tagh" in first[DetailKey.EXPECT.value]


@pytest.mark.asyncio
async def test_slot_fill_failure_downgrades_to_no_match_instead_of_raising(tmp_path):
    """Fail toward silence: a match that cannot be filled must not reach the engine."""
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.30, "argmax"), ("count_t", 0.60, "count")])
    result = await library.retrieve(chroma, "which of these six peaks is most prominent")
    assert result.decision == R.DECISION_AUTO_APPLY

    outcome = library.fill(result, {"candidates": [{"name": "Kongur Tagh"}]})  # min_arity 2
    assert not outcome.ok and outcome.expansion is None
    assert outcome.retrieval.decision == R.DECISION_NO_MATCH
    assert outcome.retrieval.template_id is None
    assert "slot fill failed" in outcome.retrieval.reason
    assert outcome.retrieval.candidates  # ranking preserved for the log
    assert outcome.retrieval.retrieval_id == result.retrieval_id  # same join key


@pytest.mark.asyncio
async def test_fill_on_a_no_match_is_a_no_op(tmp_path):
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.99, "argmax")])
    result = await library.retrieve(chroma, "a single-entity lookup")
    outcome = library.fill(result, {"candidates": [{"name": "A"}, {"name": "B"}], "field": "x"})
    assert not outcome.ok
    assert outcome.retrieval is result


def test_fill_on_an_unknown_template_downgrades(tmp_path):
    library = _library(tmp_path)
    ghost = R.RetrievalResult(
        retrieval_id="x" * 32,
        query_text="q",
        decision=R.DECISION_AUTO_APPLY,
        template_id="not_on_disk",
        similarity=0.9,
    )
    outcome = library.fill(ghost, {"candidates": [{"name": "A"}, {"name": "B"}], "field": "x"})
    assert not outcome.ok
    assert outcome.retrieval.decision == R.DECISION_NO_MATCH
    assert "not in the library" in outcome.retrieval.reason


# --------------------------------------------------------------------------------------
# the shipped corpus
# --------------------------------------------------------------------------------------


def test_shipped_templates_all_validate_and_are_indexable():
    """The real ``templates/`` directory: every file loads, and nothing is silently skipped."""
    library = R.PlanLibrary(warn_on_drift=False)
    assert library.load_errors == {}
    assert set(library.templates) >= {
        "argmax_over_n_page_field",
        "computed_ratio_argmax",
        "entity_chain_resolution",
        "odd_one_out_negation",
        "bounded_subset_sum",
    }
    for template in library.templates.values():
        assert template.archetype and template.embedding_text and template.aggregation
        assert template.provenance.get("based_on_tasks")
        assert R.document_text(template) == template.embedding_text


def test_shipped_manifest_matches_the_shipped_templates():
    """Templates and ``_manifest.json`` are committed together — drift means a missed sync."""
    assert R.PlanLibrary(warn_on_drift=False).drift() == {}


# --------------------------------------------------------------------------------------
# the retrieval log — a SEPARATE stream from the contract log, joined offline
# --------------------------------------------------------------------------------------


def _enable_log(monkeypatch, tmp_path):
    out = tmp_path / "plan_retrievals.jsonl"
    monkeypatch.setenv(retrieval_log.ENV_FLAG, "1")
    monkeypatch.setenv(retrieval_log.ENV_PATH, str(out))
    return out


@pytest.mark.asyncio
async def test_retrieval_log_writes_nothing_when_the_flag_is_off(tmp_path, monkeypatch):
    monkeypatch.delenv(retrieval_log.ENV_FLAG, raising=False)
    monkeypatch.setenv(retrieval_log.ENV_PATH, str(tmp_path / "plan_retrievals.jsonl"))
    library = _library(tmp_path)
    result = await library.retrieve(FakeChroma([("argmax_t", 0.3, "argmax")]), "a query")
    assert retrieval_log.record_retrieval(result, call_site="auto_pre_expansion") is None
    assert not (tmp_path / "plan_retrievals.jsonl").exists()


@pytest.mark.asyncio
async def test_retrieval_log_row_carries_the_join_key_and_review_block(tmp_path, monkeypatch):
    out = _enable_log(monkeypatch, tmp_path)
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.30, "argmax"), ("count_t", 0.60, "count")])
    result = await library.retrieve(chroma, "which of these six peaks is most prominent")

    returned = retrieval_log.record_retrieval(
        result, call_site=R.CALL_SITE_AUTO, slot_fill="filled",
        applied_template_id="argmax_t", node_id="n1",
    )
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert returned == result.retrieval_id == row["retrieval_id"]
    assert row["decision"] == R.DECISION_AUTO_APPLY
    assert row["applied_template_id"] == "argmax_t" and row["slot_fill"] == "filled"
    assert [r["template_id"] for r in row["results"]] == ["argmax_t", "count_t"]
    assert row["results"][0]["rank"] == 1
    assert row["human_review"] == {
        "reviewed": False, "label": None, "reviewer": None, "notes": None
    }
    assert retrieval_log.read_events(out) == rows


@pytest.mark.asyncio
async def test_retrieval_log_records_a_downgraded_slot_fill_failure(tmp_path, monkeypatch):
    """"Matched but unusable" must stay visible: the decision downgrades, the ranking does not."""
    out = _enable_log(monkeypatch, tmp_path)
    library = _library(tmp_path)
    chroma = FakeChroma([("argmax_t", 0.30, "argmax"), ("count_t", 0.60, "count")])
    result = await library.retrieve(chroma, "which of these six peaks is most prominent")
    outcome = library.fill(result, {"candidates": [{"name": "only one"}]})

    retrieval_log.record_retrieval(
        outcome.retrieval, call_site="on_demand", slot_fill="failed", error=outcome.error
    )
    row = retrieval_log.read_events(out)[0]
    assert row["decision"] == R.DECISION_NO_MATCH
    assert row["applied_template_id"] is None and row["slot_fill"] == "failed"
    assert "at least 2 items" in row["error"]
    assert [r["template_id"] for r in row["results"]] == ["argmax_t", "count_t"]
