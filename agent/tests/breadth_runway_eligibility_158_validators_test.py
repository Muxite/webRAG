"""Adversarial offline tests for test 158 (genuine 7-way fan-out: Greek-island runway eligibility).

Everything here is offline string-in / score-out: no network, no LLM, no engine. The cases are the
ones that decide whether this task can serve as the mechanism suite's wide-fan-out HOLDOUT:

  * a complete correct answer in BOTH realistic layouts (one row per island, and an indented
    per-island block) scores 1.0 on every validator;
  * a wrong keystone (a second island falsely dropped via the Rhodes-Maritsa 1,200 m trap) gates
    the keystone and every secondary to 0 while the un-gated coverage axis is RETAINED;
  * partial coverage scores the exact fraction and fails the keystone's evidence floor even when
    the named island happens to be right;
  * a zero-visit recall answer banks nothing;
  * ONE branch carrying another island's runway figure (entity collision) fails exactly that
    branch and nothing else;
  * the compiled plan is well-formed, fully parallel, and leaks no fact.
"""

import re

import pytest

from agent.app.idea_tests import test_158_tier5_breadth_runway_eligibility as T


# --- fixtures -----------------------------------------------------------------------------

URLS = {
    "naxos":     "https://en.wikipedia.org/wiki/Naxos_Island_National_Airport",
    "skiathos":  "https://en.wikipedia.org/wiki/Skiathos_International_Airport",
    "mykonos":   "https://en.wikipedia.org/wiki/Mykonos_Airport",
    "samos":     "https://en.wikipedia.org/wiki/Samos_International_Airport",
    "santorini": "https://en.wikipedia.org/wiki/Santorini_(Thira)_International_Airport",
    "kos":       "https://en.wikipedia.org/wiki/Kos_International_Airport",
    "rhodes":    "https://en.wikipedia.org/wiki/Rhodes_International_Airport",
}

PRIMARY_OK = (
    "Naxos has to come off the shortlist: its runway is only 901 m, well under the 1,300 m "
    "your operator needs. The other six islands on the list are all fine."
)

ROW_REPORT = f"""Runway check against the 1,300 m minimum:
Naxos - Naxos Island National Airport - 901 m - too short, drop it - source: {URLS['naxos']}
Skiathos - Skiathos International Airport - 1,628 m - clears the minimum - source: {URLS['skiathos']}
Mykonos - Mykonos-Manto Mavrogenous Airport - 1,903 m - clears the minimum - source: {URLS['mykonos']}
Samos - Samos International Airport - 2,100 m - clears the minimum - source: {URLS['samos']}
Santorini - Santorini International Airport - 2,197 m - clears the minimum - source: {URLS['santorini']}
Kos - Kos International Airport - 2,400 m - clears the minimum - source: {URLS['kos']}
Rhodes - Rhodes International Airport - 3,305 m - clears the minimum - source: {URLS['rhodes']}
"""

BLOCK_REPORT = f"""Here is what each island's airport looks like.

Naxos
  Airport: Naxos Island National Airport
  Runway: 901 m
  Verdict: too short for the aircraft, we have to drop it
  Source: {URLS['naxos']}

Skiathos
  Airport: Skiathos International Airport
  Runway: 1,628 m
  Verdict: clears the minimum
  Source: {URLS['skiathos']}

Mykonos
  Airport: Mykonos-Manto Mavrogenous Airport
  Runway: 1,903 m
  Verdict: clears the minimum
  Source: {URLS['mykonos']}

Samos
  Airport: Samos International Airport
  Runway: 2,100 m
  Verdict: clears the minimum
  Source: {URLS['samos']}

Santorini
  Airport: Santorini International Airport
  Runway: 2,197 m
  Verdict: clears the minimum
  Source: {URLS['santorini']}

Kos
  Airport: Kos International Airport
  Runway: 2,400 m
  Verdict: clears the minimum
  Source: {URLS['kos']}

Rhodes
  Airport: Rhodes International Airport
  Runway: 3,305 m
  Verdict: clears the minimum
  Source: {URLS['rhodes']}
"""


def _result(primary, body):
    """The shape the harness really passes: ``result["output"]["final_deliverable"]``."""
    return {"output": {"final_deliverable": f"{primary}\n\n{body}"}}


def _obs(visits):
    return {"visit": {"count": visits}}


def _scores(result, observability):
    return {c["check"]: c for c in (fn(result, observability)
                                    for fn in T.get_validation_functions())}


def _mean(scored):
    return sum(c["score"] for c in scored.values()) / len(scored)


# --- 1. full correct answer, both layouts -------------------------------------------------

@pytest.mark.parametrize("body", [ROW_REPORT, BLOCK_REPORT], ids=["one_row_per_island", "block"])
def test_full_answer_scores_one(body):
    scored = _scores(_result(PRIMARY_OK, body), _obs(7))
    for name, check in scored.items():
        assert check["score"] == pytest.approx(1.0), f"{name}: {check['reason']}"
    assert _mean(scored) == pytest.approx(1.0)


def test_full_answer_with_a_separate_primary_deliverable_slot_still_passes():
    """Some variants also expose a per-deliverable list; the primary slot must resolve there too."""
    scored = _scores(
        {"output": {"final_deliverable": ROW_REPORT}, "deliverables": [PRIMARY_OK, ROW_REPORT]},
        _obs(7),
    )
    assert scored["keystone_dropped_island"]["score"] == 1.0
    assert _mean(scored) == pytest.approx(1.0)


# --- 2. wrong keystone: a second island falsely dropped (Rhodes-Maritsa 1,200 m trap) -------

WRONG_BODY = ROW_REPORT.replace(
    f"Rhodes - Rhodes International Airport - 3,305 m - clears the minimum - source: {URLS['rhodes']}",
    "Rhodes - Rhodes Maritsa Airport - 1,200 m - too short, drop it - source: "
    "https://en.wikipedia.org/wiki/Rhodes_Maritsa_Airport",
)
WRONG_PRIMARY = (
    "Two islands have to come off the shortlist: Naxos (901 m) and Rhodes (1,200 m) are both "
    "under the 1,300 m minimum. The rest are fine."
)


def test_wrong_keystone_gates_secondaries_but_keeps_coverage():
    scored = _scores(_result(WRONG_PRIMARY, WRONG_BODY), _obs(7))
    assert scored["keystone_dropped_island"]["score"] == 0.0
    # every GATED secondary short-circuits to 0 -> bimodal, never a constant part-score
    for gated in ("dropped_runway_value", "viable_islands", "citations"):
        assert scored[gated]["score"] == 0.0, scored[gated]["reason"]
    # the UN-gated breadth axes survive: six arms were genuinely gathered, one was not
    assert scored["runway_coverage"]["score"] == pytest.approx(6 / 7)
    assert scored["island_verdicts"]["score"] == pytest.approx(6 / 7)
    assert scored["visit_count"]["score"] == 1.0
    assert _mean(scored) < 0.75


def test_naming_no_island_at_all_fails_the_keystone():
    body = ROW_REPORT.replace("too short, drop it", "runway noted")
    scored = _scores(_result("All seven islands were checked.", body), _obs(7))
    assert scored["keystone_dropped_island"]["score"] == 0.0
    assert scored["runway_coverage"]["score"] == 1.0     # data still gathered


def test_wrong_island_named_fails_the_keystone():
    scored = _scores(_result("Skiathos is the island we must drop.", ROW_REPORT), _obs(7))
    assert scored["keystone_dropped_island"]["score"] == 0.0


# --- 3. partial coverage ------------------------------------------------------------------

PARTIAL_BODY = "\n".join(ROW_REPORT.splitlines()[1:4]) + "\n"  # Naxos, Skiathos, Mykonos only


def test_partial_coverage_scores_the_exact_fraction_and_fails_the_evidence_floor():
    scored = _scores(_result(PRIMARY_OK, PARTIAL_BODY), _obs(3))
    assert scored["runway_coverage"]["score"] == pytest.approx(3 / 7)
    assert scored["island_verdicts"]["score"] == pytest.approx(3 / 7)
    # right island named, but 3/7 gathered is below the keystone's evidence floor (5/7)
    assert scored["keystone_dropped_island"]["score"] == 0.0
    assert scored["citations"]["score"] == 0.0
    assert _mean(scored) < 0.75


def test_evidence_floor_is_exactly_five_of_seven():
    lines = ROW_REPORT.splitlines()
    four = "\n".join(lines[1:5]) + "\n"
    five = "\n".join(lines[1:6]) + "\n"
    assert _scores(_result(PRIMARY_OK, four), _obs(7))["keystone_dropped_island"]["score"] == 0.0
    assert _scores(_result(PRIMARY_OK, five), _obs(7))["keystone_dropped_island"]["score"] == 1.0


# --- 4. no visits: nothing is banked ------------------------------------------------------

def test_zero_visits_banks_nothing():
    scored = _scores(_result(PRIMARY_OK, ROW_REPORT), _obs(0))
    assert scored["visit_count"]["score"] == 0.0
    assert scored["keystone_dropped_island"]["score"] == 0.0
    assert scored["runway_coverage"]["score"] == 0.0      # capped by visit count
    assert scored["island_verdicts"]["score"] == 0.0
    assert _mean(scored) == 0.0


def test_coverage_is_capped_by_visit_count():
    scored = _scores(_result(PRIMARY_OK, ROW_REPORT), _obs(2))
    assert scored["runway_coverage"]["score"] == pytest.approx(2 / 7)


# --- 5. entity collision: one branch carries another island's figure ----------------------

COLLIDED_BODY = ROW_REPORT.replace(
    "Samos - Samos International Airport - 2,100 m",
    "Samos - Samos International Airport - 2,197 m",   # Santorini's runway, on the Samos row
)


def test_borrowed_figure_fails_exactly_that_branch():
    scored = _scores(_result(PRIMARY_OK, COLLIDED_BODY), _obs(7))
    assert scored["runway_coverage"]["score"] == pytest.approx(6 / 7)
    assert "Samos" not in scored["runway_coverage"]["reason"].split(";")[0]
    # the collision does not damage anything else: the verdict and the keystone still hold
    assert scored["island_verdicts"]["score"] == 1.0
    assert scored["keystone_dropped_island"]["score"] == 1.0


def test_neighbouring_island_airport_substitution_fails_that_branch():
    """Naxos answered with the neighbouring Paros airport figure (1,400 m) -> arm dead AND the
    keystone dies, because the substituted figure would clear the minimum."""
    body = ROW_REPORT.replace(
        "Naxos - Naxos Island National Airport - 901 m - too short, drop it",
        "Naxos - Paros National Airport - 1,400 m - clears the minimum",
    )
    scored = _scores(_result("No island has to be dropped.", body), _obs(7))
    assert scored["runway_coverage"]["score"] == pytest.approx(6 / 7)
    assert scored["island_verdicts"]["score"] == pytest.approx(6 / 7)
    assert scored["keystone_dropped_island"]["score"] == 0.0


# --- 6. compiled plan ---------------------------------------------------------------------

def test_compiled_plan_is_well_formed_and_fully_parallel():
    plan = T.get_compiled_plan()
    leaves = plan["leaves"]
    assert len(leaves) == len(T.ENTITIES) == 7
    ids = [leaf["id"] for leaf in leaves]
    assert len(set(ids)) == len(ids)
    for leaf in leaves:
        assert leaf["depends_on"] == []           # genuinely independent arms, one parallel wave
        assert leaf["instruction"].strip() and leaf["expect"].strip()
    comp = plan["composition"]
    assert plan["agg_mode"] == "computed"
    assert comp["op"] == "count_threshold" and comp["threshold"] == T.MIN_RUNWAY_M
    assert [item["leaf"] for item in comp["items"]] == ids
    assert plan["aggregation"].strip()


def test_compiled_plan_and_task_statement_leak_nothing():
    plan = T.get_compiled_plan()
    plan_text = " ".join(
        [plan["aggregation"]]
        + [leaf["instruction"] + " " + leaf["expect"] for leaf in plan["leaves"]]
        + [str(item.get("label", "")) for item in plan["composition"]["items"]]
    )
    statement = T.get_task_statement()
    for text, label in ((plan_text, "compiled plan"), (statement, "task statement")):
        for entity in T.ENTITIES:
            for value in entity["values"]:
                assert not re.search(rf"\b{value:,}\b".replace(",", "[, ]?"), text), \
                    f"{label} leaks {entity['island']}'s runway figure {value}"
            # airport identification is the agent's job: no official airport name may appear
            for token in entity["airport"].replace('"', " ").split():
                if token.lower() in {"airport", "international", "national", "island", "of"}:
                    continue
                if token.lower() == entity["island"].lower():
                    continue      # the ISLANDS are a given
                assert token.lower() not in text.lower(), f"{label} leaks airport token {token}"
        # no verdict about any island, and no hint that exactly one fails
        assert "901" not in text
        assert "only island" not in text.lower()


def test_deterministic_composer_output_scores_full_marks():
    """Regression guard on the ACTUAL render shape of ``_compose_count_threshold``: its rows read
    '(>=1,300 m? no)' and its second tally reads 'Islands not at or above 1,300 m (1): Naxos'.
    Both are negated cues, so the verdict parser has to invert them rather than read 'above' as a
    pass — otherwise the graph_compiled arm would score 0 on a perfectly correct answer."""
    from agent.app.testing.execution_compiled import _compose_count_threshold

    plan = T.get_compiled_plan()
    facts = {
        f"{e['key']}_runway": f"{e['runway_m']} m — source: {URLS[e['key']]}"
        for e in T.ENTITIES
    }
    composed = _compose_count_threshold(plan["leaves"], facts, plan["composition"])
    assert composed, "composer refused to render a fully-resolved fixture"
    scored = _scores({"output": {"final_deliverable": composed}}, _obs(7))
    for name, check in scored.items():
        assert check["score"] == pytest.approx(1.0), f"{name}: {check['reason']}"


def test_metadata_and_api_surface():
    meta = T.get_test_metadata()
    assert meta["test_id"] == "158"
    assert meta["level"] == "graph" and meta["weight"] == "long"
    assert len(T.get_required_deliverables()) >= 4
    assert len(T.get_success_criteria()) >= 5
    assert T.get_llm_validation_function() is None
    assert len(T.get_validation_functions()) == 7
