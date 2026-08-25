"""
Offline unit tests for the ENTITY COLLISION mechanism task (test 303) — free, no network.

Covers: the keystone gate (present Tay RAIL bridge length), the wrong-entity attribution defect the
task exists to catch (the Tay ROAD Bridge's 2,250 m pinned on the rail bridge), the UN-gated identity
coverage diagnostic, the visit gate, the keystone-gated collision-resolution and citation checks, and
that the compiled plan is well-formed and leaks neither the length nor any identity fact.
"""
import re

from agent.app.idea_tests import test_303_mech_entity_collision_tay_rail_bridge as t


def _r(text):
    return {"output": {"final_deliverable": text}}


def _score(text, obs):
    fns = t.get_validation_functions()
    return sum(f(_r(text), obs)["score"] for f in fns) / len(fns)


_OBS2 = {"visit": {"count": 2}}

_FULL_ONELINE = (
    "The present Tay Bridge (Tay Rail Bridge), which carries rail traffic between Dundee and Wormit, "
    "has a total length of 10,711 feet (2.0286 mi; 3,265 m). It opened on 20 June 1887, was designed "
    "by William Henry Barlow and built by William Arrol & Co, and has 85 spans. The earlier bridge on "
    "the site was designed by Thomas Bouch, opened in 1878 and collapsed in 1879. This figure is NOT "
    "the Tay Road Bridge's length (that separate structure is 2,250 m / 1.4 mi, opened 1966, designer "
    "William A Fairhurst, 42 spans). Source: https://en.wikipedia.org/wiki/Tay_Bridge"
)

_FULL_MULTILINE = (
    "(a) Total length of the present rail bridge:\n"
    "10,711 ft (3,265 m)\n\n"
    "(b) Identity evidence:\n"
    "- Opened: 20 June 1887 (present structure)\n"
    "- Engineer: William Henry Barlow; contractor William Arrol & Co\n"
    "- Spans: 85\n"
    "- Earlier bridge: Thomas Bouch, opened 1878, collapsed 1879\n\n"
    "(c) Collision note:\n"
    "This is not the Tay Road Bridge\n"
    "Tay Road Bridge (different structure): 2,250 m, opened 1966\n\n"
    "Source: https://en.wikipedia.org/wiki/Tay_Bridge\n"
)


def test_full_answer_single_line_scores_all():
    for f in t.get_validation_functions():
        res = f(_r(_FULL_ONELINE), _OBS2)
        assert res["score"] == 1.0, (res["check"], res["reason"])
    assert _score(_FULL_ONELINE, _OBS2) == 1.0


def test_full_answer_multiline_layout_scores_all():
    """Bullet/table layout must score identically — clause splitting is newline-tolerant and a
    properly labelled 'Tay Road Bridge ... 2,250 m' row must not read as a misattribution."""
    for f in t.get_validation_functions():
        res = f(_r(_FULL_MULTILINE), _OBS2)
        assert res["score"] == 1.0, (res["check"], res["reason"])
    assert _score(_FULL_MULTILINE, _OBS2) == 1.0


def test_wrong_entity_length_fails_keystone_but_keeps_coverage():
    """THE defect: the near-duplicate road bridge's length reported as the rail bridge's."""
    text = (
        "The Tay Rail Bridge is 2,250 m (1.4 miles) long. It opened in 1887, was engineered by "
        "William Henry Barlow with William Arrol & Co, has 85 spans, and replaced Thomas Bouch's "
        "1878 bridge. Source: https://en.wikipedia.org/wiki/Tay_Bridge"
    )
    ks = t.validate_keystone_length(_r(text), _OBS2)
    assert ks["score"] == 0.0 and not ks["passed"]
    assert "COLLISION" in ks["reason"]
    assert t.validate_identity_coverage(_r(text), _OBS2)["score"] == 1.0   # gathering retained
    assert t.validate_collision_resolution(_r(text), _OBS2)["score"] == 0.0
    assert t.validate_citations(_r(text), _OBS2)["score"] == 0.0
    assert t.validate_visits(_r(text), _OBS2)["score"] == 1.0
    assert abs(_score(text, _OBS2) - 0.4) < 1e-9                            # bimodal, not a 0.44 trap


def test_right_number_present_but_also_misattributed_still_fails():
    """Sneaky case: the correct figure appears somewhere, yet a clause still pins the road
    bridge's length on the rail bridge. The keystone must not be rescued by mere presence."""
    text = (
        "Total length: 10,711 ft (3,265 m).\n"
        "The Tay Rail Bridge measures 2,250 m end to end.\n"
        "Source: https://en.wikipedia.org/wiki/Tay_Bridge"
    )
    ks = t.validate_keystone_length(_r(text), _OBS2)
    assert not ks["passed"] and ks["misattributed"]


def test_decoy_identity_facts_attributed_halves_collision_resolution():
    """Length right, but the road bridge's opening year is pinned on the rail bridge."""
    text = (
        "The Tay Rail Bridge is 10,711 ft (3,265 m) long.\n"
        "The Tay Bridge opened in 1966 and has 85 spans; Barlow engineered it; Bouch built the 1878 one.\n"
        "It is not the Tay Road Bridge.\n"
        "https://en.wikipedia.org/wiki/Tay_Bridge"
    )
    assert t.validate_keystone_length(_r(text), _OBS2)["passed"]
    cr = t.validate_collision_resolution(_r(text), _OBS2)
    assert cr["score"] == 0.5 and not cr["passed"]


def test_partial_coverage_scores_exact_fraction():
    text = (
        "The present Tay Rail Bridge is 3,265 m long; it opened in 1887 and was engineered by "
        "William Henry Barlow. https://en.wikipedia.org/wiki/Tay_Bridge"
    )
    cov = t.validate_identity_coverage(_r(text), _OBS2)
    assert abs(cov["score"] - 2 / 4) < 1e-9 and not cov["passed"]
    assert t.validate_keystone_length(_r(text), _OBS2)["passed"]


def test_coverage_capped_by_visits():
    """Narrating four identity markers off one page read cannot bank full coverage."""
    obs1 = {"visit": {"count": 1}}
    cov = t.validate_identity_coverage(_r(_FULL_ONELINE), obs1)
    assert abs(cov["score"] - 2 / 4) < 1e-9
    assert t.validate_identity_coverage(_r(_FULL_ONELINE), {"visit": {"count": 0}})["score"] == 0.0


def test_no_visits_kills_keystone_and_visit_gate():
    obs0 = {"visit": {"count": 0}}
    ks = t.validate_keystone_length(_r(_FULL_ONELINE), obs0)
    assert not ks["passed"] and "ungrounded" in ks["reason"]
    assert t.validate_visits(_r(_FULL_ONELINE), obs0)["score"] == 0.0
    assert t.validate_citations(_r(_FULL_ONELINE), obs0)["score"] == 0.0


def test_road_bridge_page_citation_alone_is_not_credited():
    text = _FULL_ONELINE.replace(
        "https://en.wikipedia.org/wiki/Tay_Bridge",
        "https://en.wikipedia.org/wiki/Tay_Road_Bridge",
    )
    assert t.validate_keystone_length(_r(text), _OBS2)["passed"]
    assert t.validate_citations(_r(text), _OBS2)["score"] == 0.0


def test_body_text_length_reading_also_accepted():
    text = (
        "The second Tay Bridge has an overall length of 10,780 ft covered by 85 spans; it opened in "
        "1887 under William Henry Barlow, replacing Thomas Bouch's 1878 bridge. Not the Tay Road "
        "Bridge. https://en.wikipedia.org/wiki/Tay_Rail_Bridge"
    )
    assert t.validate_keystone_length(_r(text), _OBS2)["passed"]
    assert t.validate_citations(_r(text), _OBS2)["score"] == 1.0


def test_metadata_and_api_surface():
    md = t.get_test_metadata()
    assert md["test_id"] == "303" and md["level"] == "navigation"
    assert t.get_llm_validation_function() is None
    assert len(t.get_required_deliverables()) >= 3 and len(t.get_success_criteria()) >= 3
    assert "Firth of Tay" in t.get_task_statement()


def test_task_statement_does_not_leak_the_answer():
    stmt = t.get_task_statement().lower()
    for leak in ["10,711", "10711", "3,265", "3265", "10,780", "1887", "barlow", "arrol", "85 spans", "2,250"]:
        assert leak not in stmt, f"task statement leaks {leak}"


def test_compiled_plan_is_wellformed_and_leaks_nothing():
    plan = t.get_compiled_plan()
    leaves = plan["leaves"]
    ids = [leaf["id"] for leaf in leaves]
    assert len(leaves) == 4 and len(set(ids)) == 4
    assert ids[0] == "tay_rail_bridge" and leaves[0]["depends_on"] == []
    for leaf in leaves[1:]:
        assert leaf["depends_on"] == ["tay_rail_bridge"]
        assert "{tay_rail_bridge}" in leaf["instruction"]
    for leaf in leaves:
        assert set(leaf) == {"id", "instruction", "expect", "depends_on"}
        assert leaf["expect"] and leaf["instruction"]
    blob = " ".join(str(leaf) for leaf in leaves).lower() + " " + plan["aggregation"].lower()
    for leak in ["10,711", "10711", "3,265", "3265", "10,780", "10780", "1887", "1878", "1879",
                 "barlow", "arrol", "bouch", "fairhurst", "85", "2,250", "2250", "1966", "42"]:
        assert leak not in blob, f"compiled plan leaks {leak!r}"
    # id keys a GIVEN (the entity description), never the answer.
    assert not re.search(r"\d", ids[0])
