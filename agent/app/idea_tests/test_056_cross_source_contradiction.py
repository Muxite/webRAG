"""
Test 056: Cross-source contradiction resolution (fact-check / anti-hallucination)
Level: integration   Weight: long   Difficulty: 8/10

A single fact whose COMMONLY-CITED value is WRONG (outdated) and whose AUTHORITATIVE source
carries the correct value. The agent must report the CORRECT value AND name the incorrect
popular value, resolving the contradiction by consulting the primary source instead of memory.
This is the ``verify`` axis: a cheap model that answers parametrically confidently asserts the
popular-wrong number; a structured agent cross-checks the popular claim against the authoritative
page and corrects it.

The fact: the official height of Mount Everest.
  * Commonly-cited / popular value (WRONG, outdated): 8,848 m (29,029 ft) — the long-standing
    pre-2020 figure, still repeated almost everywhere and the default a weak model recalls.
  * Authoritative / correct current value (KEYSTONE): 8,848.86 m (29,031.7 ft) — established by
    the 2020 China-Nepal joint survey; this is the figure in the live Wikipedia infobox.

Scoring is gated/bimodal: the KEYSTONE is the authoritative-correct value (the precise ``.86`` is
page-only — a weak parametric model recalls the round 8,848, not 8,848.86). The secondary checks
(identifies the outdated popular value, cites the authoritative URL) short-circuit to 0 when the
keystone is absent. An UN-gated contradiction-coverage diagnostic measures whether BOTH sides of
the contradiction were surfaced — the axis on which a structured/verify agent beats a linear one
even when the final answer is botched.

Ground truth (verified against live sources, 2026-06-26):
  * Authoritative (correct): https://en.wikipedia.org/wiki/Mount_Everest — infobox
    "8,848.86 m (29,031.7 ft)"; "On 8 December 2020 ... the new official height is 8,848.86 metres
    (29,031.7 ft)."  Page also documents the prior value: "The 8,848 m (29,029 ft) height ... was
    officially recognised by Nepal and China."
  * Secondary (still cites the wrong popular value): https://www.altitudehimalaya.com/blog/mount-everest-height
    — "8,848 meters ... was the previously accepted height" (29,028/29,029 ft); the 2020 update to
    8,848.86 m is attributed to re-survey + tectonic uplift.
  * Margin: meters differ by 0.86 (the keystone hinges on the ``.86`` suffix, which the wrong
    value lacks); feet are fully distinct tokens (29,031/29,032 vs 29,028/29,029) — so a single
    noisy extraction of the popular figure fails the keystone (correct, bimodal) rather than
    silently flipping it.
"""

from typing import Dict, Any, List
import re
from agent.app.idea_test_utils import extract_final_text


AUTHORITATIVE_URL = "https://en.wikipedia.org/wiki/Mount_Everest"

# Correct / current authoritative value: 8,848.86 m OR 29,031.7 / 29,032 ft. The ``.86`` (or the
# precise feet 29,031/29,032) is the page-only crux that distinguishes it from the popular 8,848.
_CORRECT_RE = re.compile(r"8[,\s]?848\.86|\b29[,\s]?03[12]\b")

# Wrong / outdated popular value: bare 8,848 m (NOT followed by a decimal — so the negative
# lookahead excludes the ``8,848`` prefix inside the correct 8,848.86) OR 29,028 / 29,029 ft.
_WRONG_RE = re.compile(r"\b8[,\s]?848\b(?!\.\d)|\b29[,\s]?02[89]\b")

# A "this value is outdated / commonly-cited but superseded" marker, for the gated secondary that
# requires the agent to FLAG the popular figure (not merely echo it).
_OUTDATED_MARK = (
    r"(?:outdated|out\s*of\s*date|old(?:er)?|previous(?:ly)?|prior|former(?:ly)?|supersed\w+|"
    r"commonly|widely|long[-\s]?(?:cited|standing|held)|no\s+longer|incorrect|wrong|mistaken|"
    r"earlier|until\s+2020|before\s+2020|pre[-\s]?2020|used\s+to|was\s+the\s+official)"
)
_WRONG_TOK = r"(?:\b8[,\s]?848\b(?!\.\d)|\b29[,\s]?02[89]\b)"
# Proximity (newline-tolerant via [^.], sentence-bounded): the outdated marker within a sentence of
# the wrong value, in either order.
_IDENT_RE = re.compile(
    rf"{_OUTDATED_MARK}[^.]{{0,80}}{_WRONG_TOK}|{_WRONG_TOK}[^.]{{0,80}}{_OUTDATED_MARK}",
    re.IGNORECASE,
)


def get_test_metadata() -> Dict[str, Any]:
    return {
        "test_id": "056",
        "test_name": "Cross-source contradiction resolution (fact-check)",
        "difficulty_level": "8/10",
        "category": "Cross-source contradiction / verify",
        "level": "integration",
        "weight": "long",
    }


def get_task_statement() -> str:
    return (
        "You are a fact-checker. A popular figure is circulating that you must VERIFY against the "
        "AUTHORITATIVE source — do NOT answer from memory, because the commonly-cited value may be "
        "OUTDATED:\n\n"
        '  CLAIM (commonly cited): "The official height of Mount Everest is 8,848 metres '
        '(29,029 feet)."\n\n'
        "Consult the authoritative source for Mount Everest's elevation (the most recent official "
        "China-Nepal joint survey, as reflected in the authoritative article) and then:\n"
        "  (a) Report the CORRECT, CURRENT official height of Mount Everest as established by the "
        "most recent authoritative survey (this is your primary answer).\n"
        "  (b) Identify the commonly-cited figure that is now OUTDATED / superseded, and explain "
        "the discrepancy (when and why the official height changed).\n\n"
        "Cite the exact authoritative source URL you read the current official height from."
    )


def get_required_deliverables() -> List[str]:
    return [
        "The correct, CURRENT official height of Mount Everest (the keystone / primary answer)",
        "The commonly-cited figure now identified as outdated / superseded",
        "Explanation of the discrepancy (the 2020 China-Nepal joint re-survey)",
        "Authoritative source URL",
    ]


def get_success_criteria() -> List[str]:
    return [
        "At least 2 sources/pages consulted (the authoritative figure must be read, not recalled)",
        "Correct current official height reported (8,848.86 m / 29,031.7 ft)",
        "Commonly-cited outdated value (8,848 m / 29,029 ft) identified as superseded",
        "Authoritative source URL cited",
    ]


def _keystone_ok(result: Dict[str, Any], observability: Dict[str, Any] = None) -> bool:
    """KEYSTONE: the authoritative-correct current height (8,848.86 m / 29,031.7 ft) is present.

    Credit requires GROUNDING: the value string alone is insufficient — the agent must have
    actually visited at least one page (visit.count > 0), else an ungrounded parametric-memory
    guess of the precise ``.86`` figure would earn credit."""
    n_visits = int((observability or {}).get("visit", {}).get("count", 0) or 0)
    if n_visits <= 0:
        return False
    return bool(_CORRECT_RE.search(extract_final_text(result)))


def validate_visits(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    n = int(observability.get("visit", {}).get("count", 0) or 0)
    return {"check": "visit_count", "passed": n >= 2, "score": min(1.0, n / 3.0),
            "reason": f"{n} visit(s) (target >=2; the authoritative figure must be read, not recalled)"}


def validate_keystone_height(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """KEYSTONE: authoritative-correct current official height present. Hard 0/1."""
    passed = _keystone_ok(result, observability)
    return {"check": "keystone_height", "passed": passed, "score": 1.0 if passed else 0.0,
            "reason": "Current official height (8,848.86 m / 29,031.7 ft) present" if passed
                      else "Authoritative-correct current height (8,848.86 m / 29,031.7 ft) missing"}


def validate_contradiction_coverage(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """UN-gated coverage diagnostic: how much of the CONTRADICTION was actually surfaced.

    Counts whether BOTH sides were gathered — the authoritative current value AND the commonly-cited
    outdated value. Deliberately NOT gated on the keystone: a linear/parametric agent surfaces only
    the popular figure (0.5), while a structured/verify agent that read the authoritative page
    surfaces both (1.0) — the axis that separates the two even when the final answer is botched.
    """
    text = extract_final_text(result)
    has_correct = bool(_CORRECT_RE.search(text))
    has_wrong = bool(_WRONG_RE.search(text))
    hits = int(has_correct) + int(has_wrong)
    return {"check": "contradiction_coverage", "passed": hits == 2, "score": hits / 2.0,
            "reason": f"current_value={has_correct}, outdated_value={has_wrong}"}


def validate_identifies_wrong_value(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated): the agent identifies the commonly-cited 8,848 m / 29,029 ft as the
    OUTDATED / superseded figure (the wrong value flagged near an outdated marker), not merely
    echoed. Short-circuits to 0 when the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "identifies_wrong_value", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> wrong-value identification not credited"}
    flagged = bool(_IDENT_RE.search(extract_final_text(result)))
    return {"check": "identifies_wrong_value", "passed": flagged, "score": 1.0 if flagged else 0.0,
            "reason": f"outdated popular value (8,848 m / 29,029 ft) flagged={flagged}"}


def validate_authoritative_citation(result: Dict[str, Any], observability: Dict[str, Any]) -> Dict[str, Any]:
    """SECONDARY (gated): the authoritative Mount Everest page is cited. Short-circuits to 0 when
    the keystone is absent."""
    if not _keystone_ok(result, observability):
        return {"check": "authoritative_citation", "passed": False, "score": 0.0,
                "reason": "Keystone absent -> citation not credited"}
    cited = bool(re.search(r"wiki/mount_everest", extract_final_text(result).lower()))
    return {"check": "authoritative_citation", "passed": cited, "score": 1.0 if cited else 0.0,
            "reason": f"authoritative Mount Everest page cited={cited}"}


def get_validation_functions() -> List[callable]:
    return [
        validate_visits,
        validate_keystone_height,
        validate_contradiction_coverage,
        validate_identifies_wrong_value,
        validate_authoritative_citation,
    ]


def get_llm_validation_function() -> callable:
    return None


def get_compiled_plan() -> Dict[str, Any]:
    """Offline-authored contradiction-resolution scaffold for the ``graph_compiled`` variant.

    A parallel pair of visit leaves (the CURRENT authoritative elevation and the PREVIOUSLY
    recognised one) plus a dependent ``verify`` leaf that cross-checks the popular CLAIM against
    both gathered facts (``depends_on`` the two visit leaves, templating ``{current_height}`` and
    ``{previous_height}``). The verify leaf carries ``action: "verify"`` and ``details.claim`` (the
    popular claim) per the ``verify`` action contract; the executor normalizes leaves to
    id/instruction/expect/depends_on, so the self-contained ``instruction`` also drives the generic
    leaf loop.

    Leak-free: the popular CLAIM (8,848 m / 29,029 ft — the WRONG value to fact-check) is present by
    design, but the authoritative-correct value (8,848.86 m / 29,031.7 ft) appears NOWHERE — the
    cheap model must read it off the page.
    """
    return {
        "leaves": [
            {
                "id": "current_height",
                "instruction": (
                    "Open the AUTHORITATIVE Wikipedia article for Mount Everest and read the CURRENT "
                    "official elevation of the mountain — the height established by the most recent "
                    "official China-Nepal joint survey (the latest figure in the infobox / lead). "
                    "Do NOT answer from memory; report the exact figure with units."
                ),
                "expect": "CURRENT official elevation with units — source URL",
                "depends_on": [],
            },
            {
                "id": "previous_height",
                "instruction": (
                    "On the AUTHORITATIVE Wikipedia article for Mount Everest, read the elevation "
                    "that was the officially-recognised height BEFORE the most recent joint survey "
                    "(the long-standing figure recognised by Nepal and China prior to the latest "
                    "survey). Do NOT answer from memory; report the exact figure with units."
                ),
                "expect": "PREVIOUS officially-recognised elevation with units — source URL",
                "depends_on": [],
            },
            {
                "id": "verify_popular",
                "action": "verify",
                "details": {
                    "claim": "The official height of Mount Everest is 8,848 metres (29,029 feet).",
                    "optional_url": AUTHORITATIVE_URL,
                },
                "instruction": (
                    "Fact-check this commonly-cited CLAIM against the evidence already gathered: "
                    '"The official height of Mount Everest is 8,848 metres (29,029 feet)." '
                    "The CURRENT official elevation gathered is: {current_height}. The PREVIOUSLY "
                    "recognised elevation was: {previous_height}. Decide whether the claim states the "
                    "CURRENT official height or an OUTDATED one, and give a verdict (TRUE / OUTDATED). "
                    "Report the correct current official height, the figure that is now superseded, "
                    "and cite the authoritative source URL."
                ),
                "expect": "VERDICT (TRUE/OUTDATED) + correct current height + superseded figure — source URL",
                "depends_on": ["current_height", "previous_height"],
            },
        ],
        "aggregation": (
            "Report (a) the CURRENT official height of Mount Everest as established by the most "
            "recent authoritative survey (this is the primary answer / keystone), and (b) identify "
            "the commonly-cited figure that is now OUTDATED / superseded, briefly explaining the "
            "discrepancy. Cite the authoritative source URL for the current official height."
        ),
    }
