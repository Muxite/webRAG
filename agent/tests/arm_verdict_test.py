"""Unit tests for ``agent.app.testing.arm_verdict.derive_verdict``.

The problem this module fixes: only ``evidence_loop`` emits a verdict at all
(``execution.output.ledger_verdict``, derived by ``Ledger.verdict()``).
``sequential_react_extract`` writes only ``success = bool(deliverable)`` and
``langgraph_react`` writes ``final_deliverable`` / ``success`` / ``goal_achieved`` with no
verdict field. Comparing a real verdict against ``success=True`` is not a comparison -- on one
stored run ``sequential_react_extract`` reported ``success=True`` on 48 of 48 cells.

``derive_verdict`` computes ANSWER / PARTIAL / ABSTAIN from ONLY what every arm actually emits:
the final text (``idea_test_utils.extract_final_text``) and the arm-symmetric evidence source
(``idea_test_utils.visited_evidence``). It never reads ``ledger_verdict``, ``success`` or
``goal_achieved`` -- deliberately, so a disagreement with ``ledger_verdict`` on an evidence_loop
cell is a measurable finding rather than something papered over by deferring to the native field.
"""
from agent.app.testing.arm_verdict import (
    VERDICT_ANSWER,
    VERDICT_PARTIAL,
    VERDICT_ABSTAIN,
    derive_verdict,
)


def _result(text, pages=None, graph=None):
    output = {"final_deliverable": text}
    if pages is not None:
        output["pages"] = pages
    return {"output": output, "graph": graph or {}}


# ---------------------------------------------------------------------------------------------
# ABSTAIN: no evidence at all, or no text, or explicit non-answer language
# ---------------------------------------------------------------------------------------------

def test_no_evidence_visited_is_abstain_regardless_of_text():
    result = _result("The answer is definitely 42.")
    assert derive_verdict(result, observability=None) == VERDICT_ABSTAIN


def test_empty_final_text_is_abstain():
    result = _result("", pages=[{"url": "https://example.com/x", "text": "some content"}])
    assert derive_verdict(result, observability=None) == VERDICT_ABSTAIN


def test_explicit_non_answer_language_is_abstain_even_with_evidence():
    result = _result(
        "I could not determine the value from the available sources.",
        pages=[{"url": "https://example.com/x", "text": "unrelated page content"}],
    )
    assert derive_verdict(result, observability=None) == VERDICT_ABSTAIN


def test_no_rows_no_pages_and_empty_observability_abstains():
    result = _result("42")
    assert derive_verdict(result, observability={}) == VERDICT_ABSTAIN


# ---------------------------------------------------------------------------------------------
# ANSWER: every checkable numeric/quoted claim in the final text is grounded in visited content
# ---------------------------------------------------------------------------------------------

def test_fully_grounded_numeric_claim_is_answer():
    result = _result(
        "Denali's official elevation is 20310 feet.",
        pages=[{"url": "https://example.com/denali",
                "text": "In 2015 the USGS resurveyed Denali at 20310 feet."}],
    )
    assert derive_verdict(result, observability=None) == VERDICT_ANSWER


def test_fully_grounded_quoted_claim_is_answer():
    result = _result(
        'The site says "the tallest peak in North America".',
        pages=[{"url": "https://example.com/denali",
                "text": "Denali is the tallest peak in North America."}],
    )
    assert derive_verdict(result, observability=None) == VERDICT_ANSWER


def test_prose_with_no_checkable_claims_is_partial_not_answer():
    # No numbers, no quotes -- nothing mechanically checkable, so we don't credit ANSWER.
    result = _result(
        "The mountain is very tall and well known.",
        pages=[{"url": "https://example.com/denali", "text": "Denali is a mountain in Alaska."}],
    )
    assert derive_verdict(result, observability=None) == VERDICT_PARTIAL


# ---------------------------------------------------------------------------------------------
# PARTIAL: some but not all claims grounded, or a fabricated number with no support at all
# ---------------------------------------------------------------------------------------------

def test_one_grounded_one_fabricated_number_is_partial():
    result = _result(
        "Denali is 20310 feet tall and was climbed by 99999 people in 2020.",
        pages=[{"url": "https://example.com/denali", "text": "Denali stands at 20310 feet."}],
    )
    assert derive_verdict(result, observability=None) == VERDICT_PARTIAL


def test_entirely_fabricated_number_with_real_evidence_present_is_partial_not_abstain():
    # Evidence WAS gathered (so this is not "nothing obtained"), but the asserted number appears
    # nowhere in it -- reportable/fabricated, not an abstention.
    result = _result(
        "The answer is 12345.",
        pages=[{"url": "https://example.com/x", "text": "This page never mentions that number."}],
    )
    assert derive_verdict(result, observability=None) == VERDICT_PARTIAL


# ---------------------------------------------------------------------------------------------
# Arm-symmetry: identical inputs (text + visited_evidence) from different arm shapes must reach
# the same verdict. observability["evidence"]["visited"], result["graph"]["nodes"] and
# output["pages"] are the three sources visited_evidence already unifies (idea_test_utils.py);
# derive_verdict must not special-case any of them.
# ---------------------------------------------------------------------------------------------

def test_same_verdict_whether_evidence_comes_from_observability_or_graph_or_pages():
    text = "The tower is 324 meters tall."
    content = "The Eiffel Tower is 324 meters tall."

    via_observability = ({"output": {"final_deliverable": text}, "graph": {}},
                         {"evidence": {"visited": [{"url": "https://example.com/a",
                                                     "content": content}]}})
    via_graph = ({"output": {"final_deliverable": text},
                 "graph": {"nodes": {"n1": {"details": {"action_result": {
                     "action": "visit", "success": True, "url": "https://example.com/a",
                     "content": content}}}}}}, None)
    via_pages = ({"output": {"final_deliverable": text,
                             "pages": [{"url": "https://example.com/a", "text": content}]},
                 "graph": {}}, None)

    verdicts = {derive_verdict(result, observability) for result, observability in
               (via_observability, via_graph, via_pages)}
    assert verdicts == {VERDICT_ANSWER}


def test_does_not_read_native_ledger_verdict_field():
    # A ledger_verdict of ANSWER sitting right there in output must NOT be read: the derived
    # verdict is computed purely from text + evidence, and here neither exists.
    result = {"output": {"final_deliverable": "", "ledger_verdict": "ANSWER"}, "graph": {}}
    assert derive_verdict(result, observability=None) == VERDICT_ABSTAIN


def test_does_not_read_success_field():
    result = {"output": {"final_deliverable": "", "success": True}, "graph": {}}
    assert derive_verdict(result, observability=None) == VERDICT_ABSTAIN


# ---------------------------------------------------------------------------------------------
# derive_verdict_graded: delegates to the arm-blind auditor (claim_audit.audit), which can
# credit a RECOMPUTABLE claim -- a value reproduced by one whitelisted operation over operands
# the answer itself states -- not just an ON_PAGE literal match. This is the fix for the
# structural defect the module docstring measures: a leak-proofed derived keystone can never be
# ON_PAGE, so the old literal-match rule could never credit it.
# ---------------------------------------------------------------------------------------------

def test_graded_credits_a_correct_derived_answer_whose_keystone_is_on_no_page():
    from agent.app.testing.arm_verdict import derive_verdict_graded

    result = {
        "output": {
            "final_deliverable": "Tower A is 419.7 metres and Tower B is 330.0 metres, so the "
                                 "difference is 89.7 metres.",
            "pages": [{"url": "https://example.com/towers",
                      "text": "Tower A stands 419.7 metres. Tower B stands 330.0 metres."}],
        },
        "graph": {},
    }
    # The old, literal-match rule cannot credit this: 89.7 appears on no page.
    assert derive_verdict(result, observability=None) == VERDICT_PARTIAL
    assert derive_verdict_graded(result, observability=None) == VERDICT_ANSWER


def test_graded_does_not_automatically_answer_a_claim_poor_response_with_one_grounded_number():
    from agent.app.testing.arm_verdict import derive_verdict_graded

    result = {
        "output": {
            "final_deliverable": "Denali is 20310 feet tall and was climbed by 99999 people.",
            "pages": [{"url": "https://example.com/denali",
                      "text": "Denali stands at 20310 feet."}],
        },
        "graph": {},
    }
    # 20310 is on_page, 99999 is unsupported (not on any page, not recomputable) ->
    # support_rate 0.5, below answer_min_support -> not credited as ANSWER.
    assert derive_verdict_graded(result, observability=None) == VERDICT_PARTIAL


def test_graded_abstains_on_explicit_non_answer_language():
    from agent.app.testing.arm_verdict import derive_verdict_graded

    result = {
        "output": {
            "final_deliverable": "I could not determine the value from the available sources.",
            "pages": [{"url": "https://example.com/x", "text": "unrelated page content"}],
        },
        "graph": {},
    }
    assert derive_verdict_graded(result, observability=None) == VERDICT_ABSTAIN


def test_graded_abstains_when_no_pages_were_stored():
    from agent.app.testing.arm_verdict import derive_verdict_graded

    result = {"output": {"final_deliverable": "The answer is 42."}, "graph": {}}
    assert derive_verdict_graded(result, observability=None) == VERDICT_ABSTAIN


def test_graded_abstains_when_the_answer_has_no_checkable_claim():
    from agent.app.testing.arm_verdict import derive_verdict_graded

    result = {
        "output": {
            "final_deliverable": "The mountain is very tall and well known.",
            "pages": [{"url": "https://example.com/denali", "text": "Denali is a mountain."}],
        },
        "graph": {},
    }
    assert derive_verdict_graded(result, observability=None) == VERDICT_ABSTAIN


def test_derive_verdict_unchanged_by_the_presence_of_derive_verdict_graded():
    # derive_verdict itself must stay byte-identical in behaviour: the old rule must remain
    # computable so every report can show old-vs-new side by side.
    result = {
        "output": {
            "final_deliverable": "Denali's official elevation is 20310 feet.",
            "pages": [{"url": "https://example.com/denali",
                      "text": "In 2015 the USGS resurveyed Denali at 20310 feet."}],
        },
        "graph": {},
    }
    assert derive_verdict(result, observability=None) == VERDICT_ANSWER
