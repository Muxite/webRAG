"""
Offline unit tests for the reworked ODD-ONE-OUT / negation task (test 080, variant B) — free.

The retired variant used five Caucasus rivers with the Rioni (-> Black Sea) as the exception; that
"which sea does the Rioni drain into" fact turned out to be parametrically recallable, so a model
could guess the odd one without ever reading a page. The rework swaps to FIVE genuinely-obscure
rivers of New Guinea (all in Papua New Guinea) that all APPEAR to empty into the Gulf of Papua; the
Sepik is the exception (it empties into the Bismarck Sea on the OPPOSITE, north coast). New Guinea
river hydrography is deeply obscure -- there is no "everyone knows river X drains to sea Y" prior --
and the exception is the region's MOST prominent river, so any fame shortcut lumps it with the
dominant southern outlet and lands on the wrong answer.

Ground truth (verified against live English Wikipedia 2026-07-07):
  Fly / Kikori / Purari / Turama -> infobox "Mouth: Gulf of Papua"  (satisfiers, south coast)
  Sepik -> infobox "Mouth: Bismarck Sea"; lead "to the Bismarck Sea off northern Papua New Guinea"
           (ODD ONE OUT, north coast, near Wewak)

Covers:
  * a correct full answer scores 1.0 on every validator (single- AND multi-line layout);
  * the negation-flip decoy ("Sepik empties into the Gulf of Papua") gates the keystone to 0;
  * a wrong odd-one-out (a Gulf river named as the exception) gates the keystone AND short-circuits
    the gated secondaries (why / citation) to 0, while the UN-gated coverage diagnostic is retained;
  * a naive "all five empty into the Gulf" fails the keystone;
  * partial coverage yields the exact gathered fraction;
  * a no-visits run gates the visit process metric to 0;
  * the compiled plan is a well-formed 5-way independent fan-out (5 leaves, 0 edges), each leaf is
    SELF-DESCRIBING (repeats its river name), and the plan leaks neither the exception nor any
    per-river outlet.
"""
from agent.app.idea_tests import test_080_tier5_odd_one_out_b as t
from agent.app.testing import compiled_plan as cp


_OBS = {"visit": {"count": 5}}

# Body carrying all five rivers' outlets and all five source URLs.
_BODY_SINGLE = (
    "The Fly empties into the Gulf of Papua; the Kikori empties into the Gulf of Papua; "
    "the Purari empties into the Gulf of Papua; the Turama empties into the Gulf of Papua. "
    "The Sepik is the odd one out: it empties into the Bismarck Sea off the northern coast "
    "near Wewak, not the Gulf of Papua. "
    "Sources: https://en.wikipedia.org/wiki/Fly_River "
    "https://en.wikipedia.org/wiki/Kikori_River "
    "https://en.wikipedia.org/wiki/Purari_River "
    "https://en.wikipedia.org/wiki/Turama_River "
    "https://en.wikipedia.org/wiki/Sepik"
)

_BODY_MULTI = (
    "Outlet seas:\n"
    "  Fly: Gulf of Papua\n"
    "  Kikori: Gulf of Papua\n"
    "  Purari: Gulf of Papua\n"
    "  Turama: Gulf of Papua\n"
    "  Sepik: NOT the Gulf of Papua -- empties into the Bismarck Sea (north coast, near Wewak)\n"
    "Sources:\n"
    "  https://en.wikipedia.org/wiki/Fly_River\n"
    "  https://en.wikipedia.org/wiki/Kikori_River\n"
    "  https://en.wikipedia.org/wiki/Purari_River\n"
    "  https://en.wikipedia.org/wiki/Turama_River\n"
    "  https://en.wikipedia.org/wiki/Sepik\n"
)

_PRIMARY = "The odd one out is the Sepik -- it does NOT empty into the Gulf of Papua; it drains " \
           "north to the Bismarck Sea."


def _result(primary: str, body: str):
    """Build a result whose deliverables[0] is the keystone (odd-one-out) slot and deliverables[1]
    is the supporting body (outlets, URLs)."""
    return {
        "deliverables": [primary, body],
        "output": {"final_deliverable": f"{primary} {body}"},
    }


# ── correct full answer ────────────────────────────────────────────────────────

def test_full_answer_single_line_scores_all():
    r = _result(_PRIMARY, _BODY_SINGLE)
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_why_exception(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 1.0


def test_full_answer_multi_line_scores_all():
    r = _result(_PRIMARY, _BODY_MULTI)
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_why_exception(r, _OBS)["score"] == 1.0
    assert t.validate_citation(r, _OBS)["score"] == 1.0


def test_terse_primary_only_sepik_passes_keystone():
    # A one-word correct answer (just the entity) still passes the keystone gate.
    r = _result("Sepik", _BODY_MULTI)
    assert t.validate_keystone_odd_one_out(r, _OBS)["passed"] is True


# ── negation-flip decoy: right entity, wrong property value ─────────────────────

def test_negation_flip_sepik_into_gulf_gates_keystone():
    flip = "The odd one out is the Sepik, which empties into the Gulf of Papua."
    r = _result(flip, _BODY_MULTI)
    assert t.validate_keystone_odd_one_out(r, _OBS)["passed"] is False
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 0.0


# ── wrong odd-one-out: gate keystone + secondaries, keep coverage ───────────────

def test_wrong_entity_gates_keystone_and_secondaries_but_keeps_coverage():
    # Names a Gulf river (Fly) as the exception -> wrong-entity answer.
    wrong = "The odd one out is the Fly -- it is the exception that empties into the Bismarck Sea."
    r = _result(wrong, _BODY_MULTI)
    assert t.validate_keystone_odd_one_out(r, _OBS)["passed"] is False
    # Gated secondaries short-circuit to 0 despite the body carrying the reason + all URLs.
    assert t.validate_why_exception(r, _OBS)["score"] == 0.0
    assert t.validate_citation(r, _OBS)["score"] == 0.0
    # UN-gated coverage is retained: all five outlets were still gathered.
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_naive_all_gulf_fails_keystone():
    naive = "All five rivers -- Fly, Kikori, Purari, Turama and Sepik -- empty into the Gulf of Papua."
    r = _result(naive, _BODY_MULTI)
    assert t.validate_keystone_odd_one_out(r, _OBS)["passed"] is False


# ── partial coverage yields exact fraction ─────────────────────────────────────

def test_partial_coverage_scores_fraction():
    # Correct keystone, but only three of the five rivers' outlets reported in the body.
    partial = (
        "Fly: Gulf of Papua. Purari: Gulf of Papua. "
        "Sepik: Bismarck Sea (north coast)."
    )
    r = _result(_PRIMARY, partial)
    assert t.validate_keystone_odd_one_out(r, _OBS)["score"] == 1.0
    cov = t.validate_coverage(r, _OBS)
    assert abs(cov["score"] - 3.0 / 5.0) < 1e-9
    assert cov["passed"] is False


# ── process metric ─────────────────────────────────────────────────────────────

def test_no_visits_gates_visit_metric():
    r = _result(_PRIMARY, _BODY_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False
    # 3 of 5 visits -> partial credit, still below the >=4 pass bar? 3<4 -> fail.
    assert abs(t.validate_visits(r, {"visit": {"count": 3}})["score"] - 0.6) < 1e-9
    assert t.validate_visits(r, {"visit": {"count": 3}})["passed"] is False
    assert t.validate_visits(r, {"visit": {"count": 4}})["passed"] is True


# ── compiled plan ──────────────────────────────────────────────────────────────

def test_compiled_plan_is_well_formed_five_way_fanout():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 5
    assert struct["edge_count"] == 0  # five independent parallel leaves, no cross-entity hops


def test_compiled_plan_leaves_are_self_describing():
    # Every leaf must repeat its own river name so the fact survives ID-stripping in aggregation.
    plan = t.get_compiled_plan()
    for leaf, e in zip(plan["leaves"], t.ENTITIES):
        assert e["name"] in leaf["instruction"]
        assert e["name"] in leaf["expect"]


def test_compiled_plan_leaks_neither_exception_nor_outlets():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: the plan names the five GIVEN rivers and the GIVEN shared property, but must
    # not reveal which river is the exception nor pre-supply any per-river outlet determination.
    assert "bismarck" not in blob, "plan leaks the exception's outlet sea (Bismarck)"
    assert "wewak" not in blob, "plan leaks the exception's location (Wewak)"


def test_compiled_plan_leaves_are_symmetric_no_special_casing():
    # The five leaves must be the SAME neutral template modulo the river's own name -- proving the
    # plan special-cases none of them (the Sepik leaf is shaped identically to the four satisfiers),
    # so nothing reveals which river is the exception.
    plan = t.get_compiled_plan()
    templates = set()
    for leaf, e in zip(plan["leaves"], t.ENTITIES):
        templates.add(leaf["instruction"].replace(e["name"], "<RIVER>"))
    assert len(templates) == 1, "leaf instructions are not a single symmetric template"
