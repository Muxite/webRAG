"""
Offline unit tests for the breadth fan-out/aggregation task (test 052) — free.

Cover the argmin keystone gate (earliest-born = Austen), the UN-gated coverage diagnostic
(how many of the six author/year pairs were gathered), the visit gate, and the
keystone-gated citations. Also assert the compiled plan is well-formed and leaks no
answers, since that plan is the "expensive-model-authored scaffold" the graph_compiled
arm executes.
"""
from agent.app.idea_tests import test_052_tier5_breadth_aggregation as t5


def _r(text):
    return {"output": {"final_deliverable": text}}


_FULL = (
    "Six novels, by author birth year:\n"
    "- 'Pride and Prejudice' -> Jane Austen, born 1775 "
    "(https://en.wikipedia.org/wiki/Jane_Austen)\n"
    "- 'Crime and Punishment' -> Fyodor Dostoevsky, born 1821 "
    "(https://en.wikipedia.org/wiki/Fyodor_Dostoevsky)\n"
    "- 'Mrs Dalloway' -> Virginia Woolf, born 1882 "
    "(https://en.wikipedia.org/wiki/Virginia_Woolf)\n"
    "- 'The Great Gatsby' -> F. Scott Fitzgerald, born 1896 "
    "(https://en.wikipedia.org/wiki/F._Scott_Fitzgerald)\n"
    "- 'The Old Man and the Sea' -> Ernest Hemingway, born 1899 "
    "(https://en.wikipedia.org/wiki/Ernest_Hemingway)\n"
    "- 'Beloved' -> Toni Morrison, born 1931 "
    "(https://en.wikipedia.org/wiki/Toni_Morrison)\n"
    "The earliest-born author is Jane Austen (1775)."
)


def test_full_answer_scores_all():
    obs = {"visit": {"count": 6}}
    checks = {c["check"]: c for c in (
        t5.validate_visits(_r(_FULL), obs),
        t5.validate_keystone_earliest(_r(_FULL), obs),
        t5.validate_coverage(_r(_FULL), obs),
        t5.validate_citations(_r(_FULL), obs),
    )}
    assert checks["keystone_earliest"]["score"] == 1.0
    assert checks["coverage"]["score"] == 1.0       # all six pairs
    assert checks["citations"]["score"] == 1.0      # all six pages cited
    assert checks["visit_count"]["score"] == 1.0


def test_wrong_earliest_fails_keystone_but_keeps_coverage():
    """Naming the wrong earliest must fail the argmin keystone, yet coverage (breadth
    actually gathered) is un-gated and still credited."""
    text = _FULL.replace(
        "The earliest-born author is Jane Austen (1775).",
        "The earliest-born author is Fyodor Dostoevsky.",
    )
    obs = {"visit": {"count": 6}}
    assert not t5.validate_keystone_earliest(_r(text), obs)["passed"]
    assert t5.validate_coverage(_r(text), obs)["score"] == 1.0   # still gathered all six
    assert t5.validate_citations(_r(text), obs)["score"] == 0.0  # citations gated on keystone


def test_partial_coverage_scores_fraction():
    obs = {"visit": {"count": 3}}
    text = (
        "Jane Austen born 1775; Virginia Woolf born 1882; Toni Morrison born 1931. "
        "Austen was born earliest."
    )
    cov = t5.validate_coverage(_r(text), obs)
    assert abs(cov["score"] - 3 / 6) < 1e-9
    assert not cov["passed"]
    assert t5.validate_keystone_earliest(_r(text), obs)["passed"]   # Austen + earliest + 1775
    assert t5.validate_visits(_r(text), obs)["score"] == 0.5        # 3/6 visits


def test_no_visits_loses_visit_credit():
    obs = {"visit": {"count": 0}}
    assert t5.validate_keystone_earliest(_r(_FULL), obs)["passed"]  # parametric leak possible
    assert t5.validate_visits(_r(_FULL), obs)["score"] == 0.0       # but no evidence visits


def test_multiline_earliest_layout_still_credited():
    """Answer on the line after an 'Earliest-born author:' header must still score the keystone
    (proximity tolerates the line break)."""
    text = ("(a) Earliest-born author:\nJane Austen (born 1775)\n\n(b) All six:\n"
            "Jane Austen 1775; Fyodor Dostoevsky 1821")
    assert t5.validate_keystone_earliest(_r(text), {"visit": {"count": 6}})["passed"]


# ---- compiled plan (the offline-authored scaffold) ---------------------------

def test_compiled_plan_is_wellformed_and_leaks_no_answers():
    plan = t5.get_compiled_plan()
    leaves = plan["leaves"]
    assert len(leaves) == len(t5.ENTRIES) == 6
    assert isinstance(plan["aggregation"], str) and "earliest" in plan["aggregation"].lower()
    blob = " ".join(str(leaf) for leaf in leaves).lower()
    # The scaffold encodes STRUCTURE only — no authors, birth years or the argmin answer.
    for e in t5.ENTRIES:
        assert e["author"].lower() not in blob, f"plan leaks author {e['author']}"
        assert e["year"] not in blob, f"plan leaks birth year {e['year']}"
    assert "austen" not in blob and "1775" not in blob
    # Every novel from the breadth set is present as a leaf target.
    for e in t5.ENTRIES:
        assert e["novel"].lower() in blob, f"plan missing novel {e['novel']}"
