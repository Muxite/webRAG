"""Evidence-loop executor — ReAct's winning loop plus a ledger, typed extraction and quotes.

The measured comparator on this suite is the plain sequential ReAct loop
(``testing/execution_sequential.py``), which beats the Graph-of-Thoughts engine by a wide margin
at half the step budget. This arm therefore STARTS from that loop's shape rather than defending a
tree: one flat ``for step in range(max_steps)`` pass, one JSON decision per step
(``{thought, action, args}``), the same ``search`` / ``visit`` / ``verify`` / ``finish`` verbs, the
same normalized-query dedup nudge, the same whole-page observation under a 1500-character cap, and
the same forced synthesis when the model never calls ``finish``. No graph, no scores, no beam.

Four things ReAct structurally lacks are added, and nothing else:

1. **A ledger sidecar that never expires.** The scratchpad is a fixed 12-step SLIDING WINDOW, so a
   value read more than twelve steps ago is simply gone (at ~2 steps per item that is ~6 items of
   live memory). The ledger is ``(entity, field) -> status`` rows rendered into EVERY prompt,
   outside the window, at ~100 characters per row against the 1500-character observation it
   stands in for.
2. **Per-hop typed extraction.** After each successful visit, one bounded call turns that page
   into ``{entity, field, value, verdict, source_url, quote}`` records for the rows the page
   plausibly addresses. Records are append-only and keep a raw pointer (page id + character
   offsets) so a value is never summarised away.
3. **Quote-offset grounding, on its own axis.** A row records two independent things: RESOLUTION
   (a value was read off a named page) and VERIFICATION (a verbatim quote for it is literally
   present in that page's text). :func:`verify_quote` decides verification mechanically, with no
   model judgment and no fuzzy matching, so a paraphrase always fails it — that failure rate is
   the honest measure of how often a weak model invents its supporting text. Verification does
   NOT gate resolution: a value the model paraphrased is still reported, carrying an explicit
   ``resolved_unverified`` tier, because being unable to work while holding evidence for it is
   not a good outcome. Every visited page is frozen into the
   result (id, URL, SHA-256, text), so :func:`reverify_cell` re-audits a finished run's quotes
   offline from the saved artifact — a runtime check nobody can re-run is not a verifiable result.
4. **Table-first finalization.** The typed table renders FIRST and is never truncated, then quotes
   for contested cells, then raw excerpts under whatever budget is left. The
   ANSWER / PARTIAL / ABSTAIN verdict is derived from RESOLUTION in code — the model is never
   asked what it is missing — and the quote-backed fraction rides alongside it, so an answer with
   unverified provenance is reported as exactly that rather than as an abstention.

**Extraction-time value gate** (:func:`extraction_value_gate_enabled`, default OFF). Both this
arm and ``sequential_react_extract`` already compute, per record, whether the VALUE itself (not
just the quote) is literally on the page it cites — that is the ``value_verified`` field — and
both mint the extraction as a supported record regardless of what it says. Measured on the tuning
split of the stored ``ledgernum22r3`` campaign: 41.6% of this arm's extractions and 32.9% of
``sequential_react_extract``'s cite a page that does not contain the value, and a cell's
unsupported-extraction fraction anti-correlates with its ``validation.overall_score``. The gate
behind this flag stops a record failing that check from resolving a ledger row to SUPPORTED — the
record itself is still appended, unconditionally, to the append-only audit trail; only its power
to make a row look answered is withheld. IMPORTANT — read before citing a KPI number produced with
this flag ON: refusing an unsupported extraction improves the unsupported-claim KPI BY
CONSTRUCTION. That improvement is not evidence of a capability gain by itself; it only counts if
``validation.overall_score`` holds steady or improves alongside it on the same cells. A KPI-only
win here is not a result — see :func:`extraction_value_gate_enabled` for the full design rationale
and the over-refusal guard.

Row minting routes on COUNT, never on task shape (``classify_shape`` is disproven here): a mandate
enumerating >= 2 names mints one row per name (capped at 8), anything else mints exactly one. The
single-row case is this loop reduced to plain ReAct plus a one-line ledger, so a misrouted
aggregation task degrades to the winning baseline rather than to a wrong branch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional

from agent.app.agent_io import AgentIO
from agent.app.connector_chroma import ConnectorChroma
from agent.app.connector_http import ConnectorHttp
from agent.app.connector_llm import ConnectorLLM
from agent.app.connector_search import ConnectorSearch
from agent.app.idea_policies.candidate_coverage import (
    extract_named_candidates,
    strip_enumerated_items,
)
from agent.app.telemetry import TelemetrySession
from agent.app.testing import confidence_channel
from agent.app.testing import json_telemetry as _json_telemetry
from agent.app.testing.execution import _empty_graph
from agent.app.testing.test_module import IdeaTestModule
from agent.app.testing.utils import summarize_observability
from agent.app.trace_recorder import TraceRecorder, build_trace_path, traces_retained
from agent.app.prompted_tools import _json_candidates as _pt_json_candidates
from agent.app.prompted_tools import repair_json_text as _pt_repair_json_text

_logger = logging.getLogger(__name__)

STATUS_OPEN = "OPEN"
STATUS_SUPPORTED = "SUPPORTED"
STATUS_ABSENT = "ABSENT"
STATUS_CONFLICTED = "CONFLICTED"
STATUS_BLOCKED = "BLOCKED"

#: Per-row confidence tiers: the two axes crossed, in decreasing order of trust.
TIER_RESOLVED_VERIFIED = "resolved_verified"
TIER_RESOLVED_UNVERIFIED = "resolved_unverified"
TIER_UNRESOLVED = "unresolved"

VERDICT_ANSWER = "ANSWER"
VERDICT_PARTIAL = "PARTIAL"
VERDICT_ABSTAIN = "ABSTAIN"

#: Ledger rows this arm will carry by default. A hard stop on the roster, matching the sibling
#: variants; a mandate naming more candidates than this is TRUNCATED, which
#: :meth:`Ledger.roster` reports rather than hiding. Raise it with
#: ``IDEA_TEST_EVIDENCE_LOOP_MAX_ROWS`` for an N=16 / N=32 task.
_MAX_ROWS = 8
#: Scratchpad window, copied verbatim from the sequential control: the model sees the last N steps.
_SCRATCHPAD_WINDOW = 12
#: Per-step observation cap, copied verbatim from the sequential control.
_UNCAPPED_OBSERVATION_CHARS = 1500
#: Longest ``value`` / ``source_url`` fragment a ledger row renders, keeping a row near ~100 chars.
_ROW_VALUE_CHARS = 60
_ROW_URL_CHARS = 70
#: Longest field label derived from a mandate.
_FIELD_LABEL_CHARS = 80

def _flag(name: str, default: str = "1") -> bool:
    """True when environment variable ``name`` is not one of the off spellings."""
    return os.environ.get(name, default) not in ("0", "false", "False")


def max_rows_setting() -> int:
    """The ledger's row cap: ``IDEA_TEST_EVIDENCE_LOOP_MAX_ROWS``, else :data:`_MAX_ROWS`.

    :returns: the cap, never below 1.
    :raises: nothing — an unparsable override falls back to the default.
    """
    try:
        return max(1, int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_MAX_ROWS", str(_MAX_ROWS))))
    except (TypeError, ValueError):
        return _MAX_ROWS


def roster_gate_enabled() -> bool:
    """True when the closed-roster completeness gate may downgrade a verdict (default OFF).

    Opt-in because ``roster_resolved`` counts ledger extraction records, which lag the
    values a model states correctly in its finalised deliverable. Measured on 48 eligible
    stored cells the gate blocked 46, including two that scored 1.00, while the two it
    let through scored 0.40 — an inverted signal. The roster arithmetic is still reported
    and still reaches the prompt; only the verdict downgrade is gated by this flag.

    :returns: the value of ``IDEA_TEST_EVIDENCE_LOOP_ROSTER_GATE``, defaulting to disabled.
    :raises: nothing.
    """
    return _flag("IDEA_TEST_EVIDENCE_LOOP_ROSTER_GATE", default="0")


def unit_extraction_enabled() -> bool:
    """True when the extractor asks for the value WITH its unit (default ON).

    :returns: the value of ``IDEA_TEST_EVIDENCE_LOOP_UNIT_EXTRACT``, defaulting to enabled.
    :raises: nothing.
    """
    return _flag("IDEA_TEST_EVIDENCE_LOOP_UNIT_EXTRACT")


def extraction_value_gate_enabled() -> bool:
    """True when a record whose value is not located on the page it cites is EXCLUDED FROM
    SUPPORT rather than minted as a supported extraction (default OFF).

    Measured on the tuning split of the stored ``ledgernum22r3`` campaign (re-checking each
    extraction's ``value`` against the page it cites, offline, via
    ``evidence_graph.verify_value_against_stored_page``): ``evidence_loop`` mints 332
    extractions of which 138 (41.6%) cite a page that does NOT contain the value;
    ``sequential_react_extract`` mints 334 of which 110 (32.9%) do the same. Both arms already
    compute this per record -- it is the ``value_verified`` field set in :func:`extract_from_page`
    -- and mint the extraction as SUPPORTED anyway. That is not cosmetic: the unsupported
    fraction of a cell's extractions anti-correlates with ``validation.overall_score``
    (``sequential_react_extract`` pearson -0.372, cells <=25% unsupported score 0.625 vs cells
    >=60% score 0.385; ``evidence_loop`` pearson -0.084, 0.515 vs 0.426).

    Refusing an unverifiable extraction lowers the unsupported-claim KPI BY CONSTRUCTION -- that
    improvement, on its own, proves nothing about capability; it is just hiding evidence. This
    flag is validated ONLY by whether ``validation.overall_score`` holds steady (or improves)
    alongside the KPI move on the frozen-corpus A/B the coordinator runs; a KPI-only win here
    must not be reported as a capability gain.

    Design chosen over refusing the record outright: the extraction record itself is APPENDED
    UNCONDITIONALLY regardless of this flag (see :func:`extract_from_page`) -- this is the
    append-only ledger sidecar, and the model's claim is exactly the kind of miss an audit wants
    to keep. What the flag controls is narrower: whether that record is allowed to resolve a
    ledger row to SUPPORTED. The alternative (dropping the record) was rejected because it
    destroys the raw pointer (page id, quote, claimed value) that lets a later pass tell a
    mis-attributed citation (the value IS on some OTHER stored page in the same cell) apart from
    a genuine fabrication -- exactly the split this flag's own measurement report distinguishes.

    The gate never widens what counts as a match: it reads ``value_verified``, which
    ``evidence_graph.verify_value`` already resolved through ONE normalized pass (digit-group
    separators, unit spacing, dash unification, a unit reported in a separate field but spelled
    differently in the value) before returning anything other than True. A value present on the
    page under one of those surface forms is ``value_verified is True`` and is therefore never
    gated -- over-refusing a real citation because of formatting would trade real accuracy for a
    flattered metric, which is the failure mode this module exists to avoid.

    :returns: the value of ``IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE``, defaulting to disabled.
    :raises: nothing.
    """
    return _flag("IDEA_TEST_EVIDENCE_LOOP_VALUE_GATE", default="0")


def derivation_gate_enabled() -> bool:
    """True when a KNOWN-INVALID derivation may downgrade the verdict (default OFF).

    Opt-in on the same precedent as :func:`roster_gate_enabled`: the roster gate, shipped ON,
    blocked 46 of 48 eligible cells including two that scored 1.00. A derivation is marked
    invalid when the model's proposed figure disagrees with the Python recomputation, which is
    exactly the signal this layer exists to surface — but whether that should cost the whole
    verdict is an empirical question, and nothing has measured it yet. The graph, the invalid
    flag and the detail are all reported regardless; only the DOWNGRADE is behind this flag.

    :returns: the value of ``LEDGER_DERIVATION_GATE``, defaulting to disabled.
    :raises: nothing.
    """
    return _flag("LEDGER_DERIVATION_GATE", default="0")


#: Operations ``derive`` accepts, mapped to the graph method that RECOMPUTES each one. The
#: vocabulary is closed on purpose and is a strict subset of the graph's: ``lookup`` /
#: ``assert_verbatim`` / ``judgment`` have no recomputation, so exposing them would let a model
#: assert a number under the appearance of having derived it.
#: Most recent SOURCE handles rendered into a step prompt. The analogue of
#: ``_SCRATCHPAD_WINDOW`` for evidence: a bound on what one step can show, not on what is kept.
_EVIDENCE_BLOCK_SOURCES = 40

_DERIVE_ARITH_OPS = ("sum", "difference", "product", "quotient", "ratio")
_DERIVE_EXTREMUM_OPS = ("max", "min")
_DERIVE_COMPARE_MODES = ("gt", "lt", "ge", "le", "eq", "ne")

#: A weak model paraphrases the closed vocabulary rather than quoting it verbatim, even when the
#: system prompt spells it out (`multiply` and `convert_to_numeric` both surfaced live, and got
#: rejected as ``UNKNOWN_OPERATION`` — see `ledgernum22`). Each key maps a paraphrase onto the ONE
#: canonical op it unambiguously means; applied once, in ``_handle_derive``, before dispatch, so
#: every other code path (dedup key, ``graph.add_arith``, the emitted node's ``operation`` field)
#: only ever sees the canonical spelling. ``minimum``/``maximum`` are unambiguous the same way
#: ``multiply`` is. ``average``/``mean`` are deliberately NOT aliased: an arithmetic mean is
#: ``sum / count``, a two-step composition this system has no single recomputable op for, and
#: guessing which one the model meant (sum? a bare average-of-two as quotient?) would silently
#: substitute a DIFFERENT computation for the one asked for — exactly the failure mode this whole
#: derive layer exists to prevent. Left unmapped, `average`/`mean` fall through to the existing
#: ``UNKNOWN_OPERATION`` refusal, whose advice already spells out the real vocabulary.
_DERIVE_OP_ALIASES = {
    "multiply": "product",
    "add": "sum",
    "subtract": "difference",
    "divide": "quotient",
    "minimum": "min",
    "maximum": "max",
}

#: Conversion-shaped requests (`convert_to_numeric` live; a couple of obvious synonyms added
#: defensively) are refused by a DEDICATED message, never aliased onto an arithmetic op — the
#: module this loop sits on top of (``evidence_graph``) has no unit-conversion table and
#: deliberately never will, so mapping "convert" onto e.g. `quotient` would make the loop silently
#: answer a question it was never asked. See :data:`_DERIVE_ADVICE`'s ``CONVERSION_UNAVAILABLE``.
_DERIVE_CONVERSION_OPS = ("convert_to_numeric", "convert", "unit_convert", "to_numeric")

#: What the model should DO about each refusal. A code alone is a dead step; the loop's whole
#: premise is that an observation has to be actionable.
_DERIVE_ADVICE = {
    "UNIT_MISMATCH": ("This system does not convert between units. Find both values in the same "
                      "unit, or report that they cannot be combined."),
    "MISSING_OPERAND": ("Use a handle from the EVIDENCE VALUES list. Only values already verified "
                        "on a fetched page can be combined; visit a page to obtain one first."),
    "NON_NUMERIC": "One operand carries no number. Extract a numeric value before combining.",
    "DIVISION_BY_ZERO": "The denominator recomputes to zero. Check which operand you divided by.",
    "UNKNOWN_OPERATION": ("Supported operations: sum, difference, product, quotient, ratio, max, "
                          "min, count, compare_gt/lt/ge/le/eq/ne (multiply/add/subtract/divide/"
                          "minimum/maximum are also accepted as aliases)."),
    "WRONG_ARITY": ("Wrong number of operands: difference/quotient/ratio/compare_* take exactly "
                    "two; sum/product/max/min/count take one or more."),
    "CONVERSION_UNAVAILABLE": ("Unit conversion is not available here. Find both values already "
                               "expressed in the SAME unit on a page, or report that they cannot "
                               "be combined; do not ask this system to convert."),
}


class _ConversionUnavailable(Exception):
    """A `derive` request shaped like a unit conversion (`convert_to_numeric` and its synonyms).

    Not an :class:`~agent.app.testing.evidence_graph.DerivationError` subclass — this module owns
    it rather than ``evidence_graph`` because it is a REFUSAL BY POLICY (no conversion table,
    deliberately) rather than a property of the operands ``evidence_graph`` could check. It still
    carries a ``code`` attribute so :meth:`EvidenceGraph.record_refusal` records it exactly like
    any other typed refusal.
    """

    code = "CONVERSION_UNAVAILABLE"


_SYSTEM = (
    "You are a web-research agent solving a TASK with tools. Work ONE step at a time: "
    "think, then call exactly one tool. Tools:\n"
    "- search(query): web search; returns titles+URLs+snippets. Keep the query SHORT — a few "
    "focused keywords (max ~400 characters / 50 words). Do NOT put whole sentences, your reasoning, "
    "or the full task text into the query; over-long queries are rejected by the search API.\n"
    "- visit(url): read a page's full text. Use EXACT URLs from search results.\n"
    "- verify(claim): cross-check a claim against the pages you have already read.\n"
    "- finish(answer): output the final answer. Cite the source URLs you used.\n"
    "You are also given a LEDGER: one row per fact the task requires. It persists for the whole "
    "run, so a row you resolved long ago is still shown even after its page scrolled out of the "
    "scratchpad. Work the OPEN rows: search for one, visit its authoritative page, and read the "
    "value straight off that page. Never answer a row from memory, and do not finish while rows "
    "are still OPEN unless you have exhausted the sources for them.\n"
    "- derive(operation, input_refs, proposed_value?, expected_unit?): combine values ALREADY "
    "verified on a page. Refer to them by the handles in EVIDENCE VALUES (E1, E2, ...); the "
    "result gets its own handle (D1, D2, ...). The arithmetic is recomputed in Python from the "
    "verified spans, so the figure you get back is the figure to state — never recompute it in "
    "your head, and never state a computed number you did not derive here. Operations: sum, "
    "difference, product, quotient, ratio, max, min, count, compare_gt/lt/ge/le/eq/ne. Units are "
    "NOT converted: combining metres with feet is refused, and the correct response to that "
    "refusal is to find both figures in one unit or to report that they cannot be combined.\n"
    "Each step, return ONLY JSON: {\"thought\": \"...\", \"action\": "
    "\"search|visit|derive|verify|finish\", \"args\": {...}}."
)

_EXTRACT_SYSTEM_BARE = (
    "Extract typed evidence from ONE page. For each ledger row this page actually addresses, emit "
    "a record. Quote EXACTLY from the page text — a quote that is not literally present is "
    "discarded. Never use memory or another page.\n"
    "Return ONLY JSON: {\"extractions\": [{\"entity\": \"...\", \"field\": \"...\", "
    "\"value\": \"...\", \"verdict\": \"SUPPORTED|ABSENT|BLOCKED\", \"quote\": \"...\"}]}\n"
    "Use SUPPORTED with the value and its verbatim quote when the page states it; ABSENT when the "
    "page is on-topic but does not state it; BLOCKED when the page refuses to show the content "
    "(paywall, login, bot check). Emit no record for a row this page says nothing about."
)

_EXTRACT_SYSTEM_UNIT = (
    "Extract typed evidence from ONE page. For each ledger row this page actually addresses, emit "
    "a record. Quote EXACTLY from the page text — a quote that is not literally present is "
    "discarded. Never use memory or another page.\n"
    "Return ONLY JSON: {\"extractions\": [{\"entity\": \"...\", \"field\": \"...\", "
    "\"value\": \"...\", \"unit\": \"...\", \"verdict\": \"SUPPORTED|ABSENT|BLOCKED\", "
    "\"quote\": \"...\"}]}\n"
    "Write the value WITH ITS UNIT exactly as the page spells it (\"590 m\", \"1,991 metres\", "
    "\"824 °C\"), and repeat the unit alone in the \"unit\" field. A page usually lists SEVERAL "
    "figures for the same quantity in different units (an infobox lists feet before metres), so a "
    "bare number silently becomes the wrong one. When the value genuinely has no unit (a name, a "
    "year, a count), give the value as the page writes it and leave \"unit\" empty — never skip "
    "a record over a missing unit.\n"
    "Use SUPPORTED with the value and its verbatim quote when the page states it; ABSENT when the "
    "page is on-topic but does not state it; BLOCKED when the page refuses to show the content "
    "(paywall, login, bot check). Emit no record for a row this page says nothing about."
)


def extract_system_prompt(unit_bearing: bool = True) -> str:
    """The extraction system prompt.

    :param unit_bearing: when True (the shipped default) the model is asked for the value WITH its
        unit plus the unit in its own field, which is what makes a figure verifiable as one span;
        when False the legacy bare-``value`` prompt is used.
    :returns: the prompt text.
    :raises: nothing.
    """
    return _EXTRACT_SYSTEM_UNIT if unit_bearing else _EXTRACT_SYSTEM_BARE


_SYNTHESIS_SYSTEM = (
    "Write the FINAL answer using ONLY the evidence below. The EVIDENCE TABLE is authoritative: "
    "report every SUPPORTED row's value with the source URL it came from, and state plainly that "
    "an ABSENT / BLOCKED / OPEN row could not be established. For a CONFLICTED row, give both "
    "values and their sources. A row marked UNVERIFIED QUOTE still has a value read off its "
    "source — report it, and say explicitly that its supporting quote could not be matched to "
    "the page. Never add a fact that is not in the evidence."
)


#: The quote was checked against a page in hand and its words are not there (paraphrase,
#: composed sentence, or a quote from somewhere else). This is the finding the check exists for.
QUOTE_FAIL_ABSENT = "absent"
#: No page text was available, so nothing could be checked. Never a claim about the quote.
QUOTE_FAIL_NO_PAGE = "no_page"
#: The model emitted no quote (or only wrapper punctuation), so there was nothing to check.
QUOTE_FAIL_EMPTY = "empty"

#: Wrapper pairs a model puts AROUND a quote: straight, curly and guillemet, opener to closer.
_QUOTE_WRAPPERS = (('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"),
                   ("\u00ab", "\u00bb"))


class QuoteMatch(NamedTuple):
    """Where a quote was found in a page, mechanically.

    :param verified: True when the quote is literally present in the page text, False when the
        page was in hand and the quote is not in it, None when nothing could be checked.
    :param start: Start offset in the RAW page text, or -1.
    :param end: End offset (exclusive) in the RAW page text, or -1.
    :param fail_reason: ``absent`` / ``no_page`` / ``empty`` when ``verified`` is not True,
        else None. This is what makes a low pass rate interpretable.
    """

    verified: Optional[bool]
    start: int
    end: int
    fail_reason: Optional[str] = None


def _collapse_whitespace(text: str):
    """``text`` with whitespace runs collapsed to single spaces, plus a raw-offset map.

    :param text: any string.
    :returns: ``(collapsed, offsets)`` where ``offsets[i]`` is the index in ``text`` of
        ``collapsed[i]`` — the map that lets :func:`verify_quote` report RAW page offsets for a
        match found in collapsed space.
    :raises: nothing.
    """
    collapsed: List[str] = []
    offsets: List[int] = []
    in_space = False
    for index, char in enumerate(text):
        if char.isspace():
            if collapsed and not in_space:
                collapsed.append(" ")
                offsets.append(index)
            in_space = True
            continue
        in_space = False
        collapsed.append(char)
        offsets.append(index)
    while collapsed and collapsed[-1] == " ":
        collapsed.pop()
        offsets.pop()
    return "".join(collapsed), offsets


def strip_quote_wrapper(quote: str) -> str:
    """``quote`` with the punctuation the MODEL wrapped around it removed, content untouched.

    Surrounding whitespace and MATCHED wrapping quote characters (straight, curly, guillemet,
    single and double, nested) come off. An unmatched quote character is content and stays: a
    lone ``"`` at one end is part of what the page said, and removing it would invent a match.

    :param quote: the model's claimed quote.
    :returns: the quote's content, possibly empty when the input was only wrapper punctuation.
    :raises: nothing.
    """
    text = (quote or "").strip()
    while len(text) >= 2:
        for opener, closer in _QUOTE_WRAPPERS:
            if text[0] == opener and text[-1] == closer:
                text = text[1:-1].strip()
                break
        else:
            break
    return text


def _locate(page_text: str, quote: str) -> Optional[QuoteMatch]:
    """``quote`` located in ``page_text`` exactly, else with whitespace runs collapsed on both
    sides; None when its words are not there."""
    if not quote:
        return None
    start = page_text.find(quote)
    if start >= 0:
        return QuoteMatch(True, start, start + len(quote))
    collapsed_page, offsets = _collapse_whitespace(page_text)
    collapsed_quote, _ = _collapse_whitespace(quote)
    if not collapsed_quote:
        return None
    index = collapsed_page.find(collapsed_quote)
    if index < 0:
        return None
    return QuoteMatch(True, offsets[index], offsets[index + len(collapsed_quote) - 1] + 1)


def verify_quote(page_text: str, quote: str) -> QuoteMatch:
    """Check mechanically that ``quote`` is literally present in ``page_text``.

    The quote is tried verbatim first, then with the wrapper punctuation the model added stripped
    (:func:`strip_quote_wrapper`); each attempt is matched exactly and then with whitespace runs
    collapsed on both sides, so a quote copied across a line break still verifies. Only what the
    model wrapped around the quote is normalized — a quote whose WORDS differ from the page never
    verifies, which is exactly how a paraphrase is caught. No model judgment is involved, and no
    fuzzy, token-overlap or edit-distance matching is used or wanted here.

    :param page_text: the fetched page text, exactly as it was handed to the model.
    :param quote: the model's claimed verbatim quote.
    :returns: a :class:`QuoteMatch`. ``verified`` is True with RAW page offsets on a hit, False
        with ``(-1, -1)`` and ``absent`` when the page was in hand and the quote is not in it, and
        None with ``empty`` / ``no_page`` when there was nothing to check.
    :raises: nothing — non-string input is simply unchecked.
    """
    if not isinstance(quote, str) or not strip_quote_wrapper(quote):
        return QuoteMatch(None, -1, -1, QUOTE_FAIL_EMPTY)
    if not isinstance(page_text, str) or not page_text.strip():
        return QuoteMatch(None, -1, -1, QUOTE_FAIL_NO_PAGE)
    for candidate in (quote.strip(), strip_quote_wrapper(quote)):
        match = _locate(page_text, candidate)
        if match is not None:
            return match
    return QuoteMatch(False, -1, -1, QUOTE_FAIL_ABSENT)


def hash_page_text(text: str) -> str:
    """The SHA-256 hex digest of ``text``, so drift or tampering in a stored page is detectable.

    :param text: the fetched page text as the extraction step saw it.
    :returns: a 64-character lowercase hex digest.
    :raises: nothing — non-string input is hashed as the empty string.
    """
    return hashlib.sha256((text if isinstance(text, str) else "").encode("utf-8")).hexdigest()


def store_page(page_id: str, url: str, text: str, max_chars: int) -> Dict[str, Any]:
    """Freeze ONE visited page into the result payload so its quotes stay checkable forever.

    The stored window is a head slice, and a head slice can silently put a late-page quote out of
    reach — so ``truncated`` is recorded and :func:`verify_against_stored_page` downgrades a miss
    on a truncated page to *unverifiable* rather than reporting a false ``absent``. The hash always
    covers the WHOLE fetched text, so a shortened window still detects drift.

    :param page_id: the id extraction records point back to.
    :param url: the canonical URL, taken from the fetch.
    :param text: the fetched page text as the extraction step saw it.
    :param max_chars: cap on the STORED window; the fetched text is unchanged.
    :returns: ``{"page_id", "url", "content_hash", "chars", "stored_chars", "truncated", "text"}``.
    :raises: nothing.
    """
    full = text if isinstance(text, str) else ""
    stored = full[:max(0, int(max_chars))]
    return {
        "page_id": page_id, "url": url, "content_hash": hash_page_text(full),
        "chars": len(full), "stored_chars": len(stored),
        "truncated": len(stored) < len(full), "text": stored,
    }


def verify_against_stored_page(page: Optional[Dict[str, Any]], quote: str) -> QuoteMatch:
    """Re-check ``quote`` against a page frozen by :func:`store_page`, offline.

    A quote absent from a page stored IN FULL is a real ``absent`` — the model paraphrased. A quote
    absent from a TRUNCATED page is merely unverifiable: it may live past the stored window, and
    calling that a failure would manufacture evidence of fabrication. Same for a missing page.

    :param page: a stored page dict, or None when the extraction's page was not persisted.
    :param quote: the model's claimed quote.
    :returns: a :class:`QuoteMatch` with the same tri-state contract as :func:`verify_quote`.
    :raises: nothing.
    """
    if not isinstance(quote, str) or not strip_quote_wrapper(quote):
        return QuoteMatch(None, -1, -1, QUOTE_FAIL_EMPTY)
    if not isinstance(page, dict) or not str(page.get("text") or ""):
        return QuoteMatch(None, -1, -1, QUOTE_FAIL_NO_PAGE)
    match = verify_quote(str(page.get("text")), quote)
    if match.verified is False and page.get("truncated"):
        return QuoteMatch(None, -1, -1, QUOTE_FAIL_NO_PAGE)
    return match


@dataclass
class Extraction:
    """One typed record read off ONE page for ONE ledger row. Append-only, never rewritten.

    ``page_id`` / ``quote_start`` / ``quote_end`` are the raw pointer back into the fetched text,
    kept so that a later summarisation step can never destroy the only copy of a value.
    """

    entity: str
    field: str
    value: str
    verdict: str
    source_url: str
    quote: str
    quote_verified: Optional[bool]
    page_id: str
    quote_start: int = -1
    quote_end: int = -1
    quote_fail_reason: Optional[str] = None
    unit: str = ""
    value_verified: Optional[bool] = None
    value_unit_bearing: bool = False
    value_shape: str = ""
    value_fail_reason: Optional[str] = None
    #: Id of the SOURCE node this record's value was admitted as, when it was located on the
    #: page. Empty when the value never became a node, which is the same thing as saying it was
    #: never mechanically found — the record survives either way.
    evidence_node_id: str = ""
    #: True when :func:`extraction_value_gate_enabled` was ON and this record's value was NOT
    #: ``value_verified is True`` against the page it cites, so it was withheld from resolving a
    #: ledger row to SUPPORTED. The record is appended regardless -- this only marks that its
    #: value claim did not earn support. Always False when the gate is OFF (the shipped default).
    excluded_from_support: bool = False

    def as_dict(self) -> Dict[str, Any]:
        """This record as a JSON-serializable dict for the result payload."""
        return {
            "entity": self.entity, "field": self.field, "value": self.value,
            "verdict": self.verdict, "source_url": self.source_url, "quote": self.quote,
            "quote_verified": self.quote_verified, "page_id": self.page_id,
            "quote_start": self.quote_start, "quote_end": self.quote_end,
            "quote_fail_reason": self.quote_fail_reason, "unit": self.unit,
            "value_verified": self.value_verified,
            "value_unit_bearing": self.value_unit_bearing, "value_shape": self.value_shape,
            "value_fail_reason": self.value_fail_reason,
            "evidence_node_id": self.evidence_node_id,
            "excluded_from_support": self.excluded_from_support,
        }


@dataclass
class LedgerRow:
    """One ``(entity, field)`` obligation on TWO axes.

    ``status`` is the RESOLUTION axis: ``SUPPORTED`` means a value was read off a named page,
    ``CONFLICTED`` that two equally trusted pages disagreed, ``ABSENT`` / ``BLOCKED`` that a page
    was consulted and could not supply it, ``OPEN`` that nothing has been read yet.
    ``quote_verified`` is the VERIFICATION axis, decided only by :func:`verify_quote`. The two are
    reported together as :attr:`confidence_tier` and never collapsed into one another.
    """

    entity: str
    field: str
    status: str = STATUS_OPEN
    value: str = ""
    source_url: str = ""
    quote: str = ""
    quote_verified: bool = False
    unit: str = ""
    #: The pin back into the fetched text, carried over from the :class:`Extraction` this row was
    #: written from. These four fields already existed on the record; a consumer reading a row had
    #: to re-find its record by fuzzy ``(entity, field)`` string matching to get them, which is
    #: guesswork about the provenance of a claim the product exists to pin exactly. Empty / ``-1``
    #: on an unresolved row, and on a resolved row whose record never located its span.
    page_id: str = ""
    quote_start: int = -1
    quote_end: int = -1
    #: Id of the SOURCE node this row's value was admitted as, or ``""`` when the value never
    #: became a node — which is the same thing as saying it was never mechanically located.
    evidence_node_id: str = ""

    @property
    def resolved(self) -> bool:
        """True when this row holds ONE value read off a page, quote or no quote.

        ``CONFLICTED`` is not resolved: two values that disagree are not an answer.
        """
        return self.status == STATUS_SUPPORTED and bool(self.value)

    @property
    def confidence_tier(self) -> str:
        """``resolved_verified`` / ``resolved_unverified`` / ``unresolved`` — the two axes crossed."""
        if not self.resolved:
            return TIER_UNRESOLVED
        return TIER_RESOLVED_VERIFIED if self.quote_verified else TIER_RESOLVED_UNVERIFIED

    def render(self) -> str:
        """This row as one compact prompt line (~100 chars), value and URL truncated."""
        head = f"[{self.status}] {self.entity} | {self.field}"
        if self.status in (STATUS_SUPPORTED, STATUS_CONFLICTED) and self.value:
            head += f" = {self.value[:_ROW_VALUE_CHARS]}"
        if self.source_url:
            head += f" <{self.source_url[:_ROW_URL_CHARS]}>"
        if self.confidence_tier == TIER_RESOLVED_UNVERIFIED:
            head += " (unverified quote)"
        return head

    def as_dict(self) -> Dict[str, Any]:
        """This row as a JSON-serializable dict for the result payload, both axes explicit."""
        return {
            "entity": self.entity, "field": self.field, "status": self.status,
            "value": self.value, "source_url": self.source_url, "quote": self.quote,
            "quote_verified": self.quote_verified, "resolved": self.resolved,
            "confidence_tier": self.confidence_tier, "unit": self.unit,
            "page_id": self.page_id, "quote_start": self.quote_start,
            "quote_end": self.quote_end, "evidence_node_id": self.evidence_node_id,
        }


def derive_field_label(mandate: str) -> str:
    """The short FIELD label every minted row shares: what the mandate asks about each entity.

    Taken from the mandate's own prose with the enumerated roster blanked out (so a candidate's
    own wording cannot answer for the question), truncated to a prompt-sized label.

    :param mandate: the task statement.
    :returns: a non-empty label; ``"requested value"`` when the mandate carries no usable prose.
    :raises: nothing.
    """
    prose = re.sub(r"\s+", " ", strip_enumerated_items(mandate or "")).strip()
    if not prose:
        return "requested value"
    sentence = re.split(r"(?<=[.?!])\s", prose)[0].strip()
    return (sentence or prose)[:_FIELD_LABEL_CHARS]


def _condense(text: str, limit: int = 80) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()[:limit]


def mint_rows(mandate: str, max_rows: Optional[int] = None) -> List[LedgerRow]:
    """Mint the ledger's rows from ``mandate``, routing on COUNT and never on shape.

    :param mandate: the task statement.
    :param max_rows: hard cap on the roster; ``None`` takes :func:`max_rows_setting`. A mandate
        naming more candidates than the cap is truncated, and :meth:`Ledger.roster` reports that.
    :returns: one row per enumerated named candidate when the mandate names >= 2 of them
        (capped at ``max_rows``), otherwise exactly ONE row covering the whole mandate — the
        chain case, in which this loop is plain ReAct plus a one-line ledger.
    :raises: nothing — an empty mandate still mints its single row.
    """
    cap = max_rows_setting() if max_rows is None else max(1, int(max_rows))
    label = derive_field_label(mandate)
    names = [name for name in extract_named_candidates(mandate or "") if name.strip()]
    if len(names) >= 2:
        return [LedgerRow(entity=_condense(name), field=label) for name in names[:cap]]
    return [LedgerRow(entity=_condense(mandate) or "(task)", field=label)]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


@dataclass
class Ledger:
    """The never-expiring sidecar: typed rows plus the append-only extraction records."""

    rows: List[LedgerRow]
    extractions: List[Extraction] = dataclass_field(default_factory=list)
    #: The mandate the rows were minted from, kept so the CLOSED roster it names stays countable.
    mandate: str = ""
    #: The run's :class:`~agent.app.testing.evidence_graph.EvidenceGraph`, built lazily. Typed
    #: ``Any`` deliberately: ``evidence_graph`` imports THIS module, so it can only be imported
    #: inside a function body here, exactly as :func:`_check_value` already does.
    graph: Any = None
    #: Prompt handle -> node id, in admission order. A node id is a content hash the model never
    #: sees; ``E1``/``D1`` are what it can actually refer to in a ``derive`` call.
    handles: Dict[str, str] = dataclass_field(default_factory=dict)

    def ensure_graph(self) -> Any:
        """The run's evidence graph, created on first use."""
        if self.graph is None:
            from agent.app.testing.evidence_graph import EvidenceGraph
            self.graph = EvidenceGraph()
        return self.graph

    def handle_for(self, node: Any) -> str:
        """This node's stable prompt handle, assigning one on first sight.

        ``E``-prefixed for a located SOURCE span, ``D``-prefixed for a recomputed DERIVED value,
        numbered per prefix in admission order. Re-admitting a content-identical node returns the
        SAME node object (the graph is content-addressed), so a handle is never duplicated.
        """
        from agent.app.testing.evidence_graph import KIND_DERIVED
        for handle, node_id in self.handles.items():
            if node_id == node.id:
                return handle
        prefix = "D" if node.kind == KIND_DERIVED else "E"
        handle = f"{prefix}{1 + sum(1 for h in self.handles if h.startswith(prefix))}"
        self.handles[handle] = node.id
        return handle

    def resolve_ref(self, ref: Any) -> str:
        """A model-supplied reference resolved to a node id.

        A handle (``"E1"``, case-insensitively) maps through :attr:`handles`; anything else is
        passed through unchanged so a raw node id still works and an unknown reference reaches
        the graph, which refuses it as ``MISSING_OPERAND`` rather than being silently dropped.
        """
        key = str(ref or "").strip()
        return self.handles.get(key) or self.handles.get(key.upper()) or key

    @staticmethod
    def _render_value(node: Any) -> str:
        """``value`` with its unit, appending the unit only when the value lacks one.

        A SOURCE value is usually already unit-bearing ("590 m") because that is what makes it
        verifiable as ONE span, while a DERIVED value is bare ("24") with the composed unit
        alongside. Appending unconditionally renders "590 m m", which is the figure the model is
        then asked to state.
        """
        from agent.app.testing.evidence_graph import is_unit_bearing
        value = str(node.value)
        if node.unit and not is_unit_bearing(value):
            return f"{value} {node.unit}".strip()
        return value

    def _render_node(self, handle: str, node: Any) -> str:
        from agent.app.testing.evidence_graph import KIND_DERIVED
        value = self._render_value(node)
        if node.kind != KIND_DERIVED:
            return f"{handle} = {value}  [{node.page_id}]"
        inputs = ", ".join(self.handle_for(self.graph.node(i)) if self.graph.node(i) else i
                           for i in node.input_ids)
        detail = f"  WARNING: {node.derivation_detail}" if node.derivation_valid is False else ""
        return f"{handle} = {value}  ({node.operation} of {inputs}){detail}"

    def evidence_lines(self, max_sources: int = _EVIDENCE_BLOCK_SOURCES) -> List[str]:
        """The per-step evidence block: EVERY derived value, plus the most recent sources.

        Unlike the ledger, whose rows are bounded by the minted roster, admitted SOURCE nodes are
        unbounded — a 32-page fan-out task yielding several values per page would put hundreds of
        lines into every step's prompt and crowd out the scratchpad. Sources are therefore capped
        at the most recent ``max_sources`` with an explicit count of what was elided.

        DERIVED values are NEVER elided: they are the locked figures the model is required to
        state rather than recompute, and they are few by construction.
        """
        if self.graph is None:
            return []
        from agent.app.testing.evidence_graph import KIND_DERIVED
        sources, derived = [], []
        for handle, node_id in self.handles.items():
            node = self.graph.node(node_id)
            if node is None:
                continue
            (derived if node.kind == KIND_DERIVED else sources).append((handle, node))
        elided = max(0, len(sources) - int(max_sources))
        shown = sources[elided:]
        lines = [self._render_node(handle, node) for handle, node in shown]
        if elided:
            lines.insert(0, f"... {elided} older verified value(s) not shown; they remain in the "
                            f"evidence graph and can still be referenced by handle.")
        lines.extend(self._render_node(handle, node) for handle, node in derived)
        return lines

    def derived_lines(self) -> List[str]:
        """One line per RECOMPUTED value, for the finalization context."""
        if self.graph is None:
            return []
        from agent.app.testing.evidence_graph import KIND_DERIVED
        lines = []
        for handle, node_id in self.handles.items():
            node = self.graph.node(node_id)
            if node is not None and node.kind == KIND_DERIVED:
                lines.append(self._render_node(handle, node))
        return lines

    def has_invalid_derivation(self) -> bool:
        """True when any DERIVED node's postcondition is KNOWN to have failed.

        ``None`` (unassessed) is not counted: "unknown" is not the same claim as "wrong", the
        same rule :meth:`EvidenceGraph._inputs_valid` applies one layer down.
        """
        if self.graph is None:
            return False
        from agent.app.testing.evidence_graph import KIND_DERIVED
        return any(node.kind == KIND_DERIVED and node.derivation_valid is False
                   for node in self.graph.nodes())

    @classmethod
    def mint(cls, mandate: str, max_rows: Optional[int] = None) -> "Ledger":
        """Build a ledger for ``mandate`` (see :func:`mint_rows`)."""
        return cls(rows=mint_rows(mandate, max_rows=max_rows), mandate=mandate or "")

    @staticmethod
    def _names(row: LedgerRow, entity: str) -> bool:
        """True when ``entity`` names ``row``: normalized equality, else containment either way."""
        target, row_norm = _norm(entity), _norm(row.entity)
        if not target or not row_norm:
            return False
        return target == row_norm or target in row_norm or row_norm in target

    def find(self, entity: str) -> Optional[LedgerRow]:
        """The row ``entity`` names, or ``None``.

        Exact normalized match first, then containment either way (a page may name a row's entity
        more or less specifically than the mandate did). A single-row ledger always matches, which
        is what keeps the N=1 case from silently dropping its only value.
        """
        if len(self.rows) == 1:
            return self.rows[0]
        target = _norm(entity)
        if not target:
            return None
        for row in self.rows:
            if _norm(row.entity) == target:
                return row
        for row in self.rows:
            if self._names(row, entity):
                return row
        return None

    def _is_wildcard(self, row: LedgerRow, entity: str) -> bool:
        """True when ``row`` came from the single-row catch-all rather than from being named.

        A single-row ledger's entity IS the condensed mandate, so only a record naming it exactly
        is about that row specifically; anything else reached it through the catch-all and is
        about whatever the page happened to state. Differing values from such records are not
        evidence of a disagreement.
        """
        return len(self.rows) == 1 and _norm(entity) != _norm(row.entity)

    def open_rows(self) -> List[LedgerRow]:
        """Rows still awaiting a grounded value."""
        return [row for row in self.rows if row.status == STATUS_OPEN]

    def apply(self, record: Extraction) -> None:
        """Fold one extraction record into the ledger, then keep the record forever.

        Transitions, in code and not by model judgment. A ``SUPPORTED`` record CARRYING A VALUE
        resolves an ``OPEN`` / ``ABSENT`` / ``BLOCKED`` row whether or not its quote verified — the
        quote decides how much the row is trusted, not whether the value exists. Between records
        for an already-resolved row: a verified record supersedes an unverified one; an unverified
        record never overturns a verified one; two records at the SAME tier carrying different
        values make the row ``CONFLICTED`` at that tier, and a conflict between UNVERIFIED values
        is itself superseded by a later verified record — otherwise a weak model's paraphrases
        could permanently deadlock a row it went on to prove. Records that reached a single-row
        ledger through its catch-all match (:meth:`_is_wildcard`) never conflict it: they are
        about different facts, not about the same one twice. ``ABSENT`` and ``BLOCKED`` mark only a row that is
        still ``OPEN``, so a value is never demoted by a later page that simply lacks it.
        A record :attr:`~Extraction.excluded_from_support` (set by :func:`extract_from_page` under
        :func:`extraction_value_gate_enabled`) is still appended below, but never resolves a row —
        that is the whole effect of the gate, and the ONLY place it takes effect.

        :param record: one typed extraction, appended to :attr:`extractions` unconditionally.
        :returns: None.
        :raises: nothing — a record naming no known row is kept and changes nothing.
        """
        self.extractions.append(record)
        row = self.find(record.entity)
        if row is None:
            return
        verdict = (record.verdict or "").strip().upper()
        if verdict == STATUS_SUPPORTED and record.value and not record.excluded_from_support:
            self._resolve(row, record, self._is_wildcard(row, record.entity))
        elif verdict in (STATUS_ABSENT, STATUS_BLOCKED) and row.status == STATUS_OPEN:
            row.status = verdict

    @staticmethod
    def _write(row: LedgerRow, record: Extraction) -> None:
        row.status = STATUS_SUPPORTED
        row.value = record.value
        row.source_url = record.source_url
        row.quote = record.quote
        row.quote_verified = record.quote_verified is True
        row.unit = record.unit
        # The pin travels with the value it pins. Writing it anywhere but here would let a row's
        # value and its offsets come from two different records.
        row.page_id = record.page_id
        row.quote_start = record.quote_start
        row.quote_end = record.quote_end
        row.evidence_node_id = record.evidence_node_id

    def _resolve(self, row: LedgerRow, record: Extraction, wildcard: bool = False) -> None:
        """Fold a value-carrying ``SUPPORTED`` record into ``row`` (see :meth:`apply`)."""
        verified = record.quote_verified is True
        if row.status == STATUS_CONFLICTED:
            if verified and not row.quote_verified:
                self._write(row, record)
            return
        if not row.resolved:
            self._write(row, record)
            return
        if _norm(row.value) == _norm(record.value):
            if verified and not row.quote_verified:
                self._write(row, record)
            return
        if verified and not row.quote_verified:
            self._write(row, record)
        elif verified == row.quote_verified and not wildcard:
            row.status = STATUS_CONFLICTED
            row.quote_verified = verified

    def status_counts(self) -> Dict[str, int]:
        """Row count per status."""
        counts: Dict[str, int] = {}
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return counts

    def resolution_counts(self) -> Dict[str, int]:
        """The two axes, counted separately over the rows.

        :returns: ``{"rows", "resolved", "resolved_verified", "resolved_unverified",
            "unresolved"}``. ``resolved`` answers "did we obtain the values"; the
            ``resolved_verified`` / ``resolved_unverified`` split answers "and can we prove them
            with a verbatim quote" — an analyst reads both without either hiding the other.
        :raises: nothing.
        """
        tiers = [row.confidence_tier for row in self.rows]
        verified = tiers.count(TIER_RESOLVED_VERIFIED)
        unverified = tiers.count(TIER_RESOLVED_UNVERIFIED)
        return {
            "rows": len(self.rows), "resolved": verified + unverified,
            "resolved_verified": verified, "resolved_unverified": unverified,
            "unresolved": tiers.count(TIER_UNRESOLVED),
        }

    def roster(self) -> Dict[str, Any]:
        """Completeness of the CLOSED roster the mandate names, as arithmetic on two integers.

        These tasks state every candidate in the prompt, so "did we work them all" needs no
        discovery and no judgment: it is ``resolved rows >= names in the mandate``. A count,
        extremum or comparison over a roster one member short is simply a different question —
        dropping one lake flips a "deeper than 480 m" count.

        :returns: ``{"roster_named", "roster_resolved", "roster_rows", "roster_truncated",
            "roster_complete"}``. ``roster_named`` is 0 when the mandate enumerates no roster, and
            the gate is then inert (``roster_complete`` True). ``roster_truncated`` is True when
            the mandate names more candidates than the row cap admitted — a truncated roster can
            never complete, which is what keeps an N=16 task from being silently capped.
        :raises: nothing.
        """
        named = len(extract_named_candidates(self.mandate or ""))
        resolved = sum(1 for row in self.rows if row.resolved)
        truncated = named > len(self.rows)
        return {
            "roster_named": named, "roster_resolved": resolved, "roster_rows": len(self.rows),
            "roster_truncated": truncated,
            "roster_complete": named < 2 or (resolved >= named and not truncated),
        }

    def roster_line(self) -> str:
        """The banner an INCOMPLETE closed roster earns, for the prompt and the deliverable.

        Emitted whenever the roster is provably short, independent of
        :func:`roster_gate_enabled` — the banner is advice to the model, while the flag governs
        only whether :meth:`verdict` may downgrade on it.

        :returns: the warning text, or ``""`` when the roster is complete or names no candidates.
        :raises: nothing.
        """
        status = self.roster()
        if status["roster_complete"]:
            return ""
        line = (f"ROSTER INCOMPLETE: {status['roster_resolved']} of the "
                f"{status['roster_named']} candidates named in the task have a resolved value. "
                "Do NOT assert a count, extremum or comparison over this roster — report the "
                "resolved values, and name the candidates that were not established.")
        if status["roster_truncated"]:
            line += (f"\nROSTER TRUNCATED: the ledger carries only {status['roster_rows']} of "
                     f"those {status['roster_named']} candidates, so the rest were never worked.")
        return line

    def verdict(self) -> str:
        """ANSWER / PARTIAL / ABSTAIN, derived from RESOLUTION alone.

        ``ANSWER`` — every row holds ONE value read off a page. ``PARTIAL`` — some rows do and
        some do not, ``CONFLICTED`` rows included, since two disagreeing values are reportable but
        are not an answer. ``ABSTAIN`` — no row obtained anything at all, which is the only honest
        reason to decline: "we did not obtain the values", never "we obtained them but the model
        paraphrased its quote" and never "we obtained two of them and they disagree". How many
        of the resolved rows are quote-backed is reported separately by
        :meth:`resolution_counts`, and a run answering with unverified provenance says so in its
        deliverable rather than abstaining.

        :returns: one of :data:`VERDICT_ANSWER` / :data:`VERDICT_PARTIAL` / :data:`VERDICT_ABSTAIN`.
        :raises: nothing — a ledger with no rows abstains.
        """
        resolved = sum(1 for row in self.rows if row.resolved)
        obtained = sum(1 for row in self.rows
                       if row.resolved or row.status == STATUS_CONFLICTED)
        if not self.rows or obtained == 0:
            return VERDICT_ABSTAIN
        if resolved != len(self.rows):
            return VERDICT_PARTIAL
        if roster_gate_enabled() and not self.roster()["roster_complete"]:
            return VERDICT_PARTIAL
        if derivation_gate_enabled() and self.has_invalid_derivation():
            return VERDICT_PARTIAL
        return VERDICT_ANSWER

    def render(self) -> str:
        """The prompt-side ledger block: every row, one compact line each, never truncated."""
        lines = [f"{i}. {row.render()}" for i, row in enumerate(self.rows, 1)]
        return "\n".join(lines)

    def render_table(self) -> str:
        """The finalization table: every row with value, status and source. Never truncated.

        An incomplete closed roster appends :meth:`roster_line`, so the same warning reaches the
        synthesis prompt and the deliverable.
        """
        counts = self.resolution_counts()
        header = (
            f"EVIDENCE TABLE ({counts['rows']} rows; {counts['resolved']} resolved, of which "
            f"{counts['resolved_verified']} are backed by a verbatim quote found on the page):"
        )
        lines = [header]
        for i, row in enumerate(self.rows, 1):
            value = row.value or "(none)"
            source = row.source_url or "(no source)"
            mark = {TIER_RESOLVED_VERIFIED: " [verified quote]",
                    TIER_RESOLVED_UNVERIFIED: " [UNVERIFIED QUOTE]"}.get(row.confidence_tier, "")
            lines.append(
                f"{i}. [{row.status}] {row.entity} | {row.field} = {value} — {source}{mark}")
        banner = self.roster_line()
        if banner:
            lines.append(banner)
        return "\n".join(lines)


def compose_user_prompt(mandate: str, ledger: Ledger, scratchpad: List[str]) -> str:
    """The per-step user message: task, the FULL ledger, and the last 12 scratchpad steps.

    :param mandate: the task statement.
    :param ledger: the run's ledger — rendered in full every step, which is the whole mechanism:
        a row costs ~100 characters and outlives the window that drops its 1500-character
        observation.
    :param scratchpad: every step entry so far; only the last ``_SCRATCHPAD_WINDOW`` are shown.
    :returns: the user message text.
    :raises: nothing.
    """
    history = "\n\n".join(scratchpad[-_SCRATCHPAD_WINDOW:]) if scratchpad else "(no actions yet)"
    evidence = ledger.evidence_lines()
    evidence_block = (
        "EVIDENCE VALUES (verified on a page; refer to these handles in a derive call):\n"
        + "\n".join(evidence) + "\n\n") if evidence else ""
    return (
        f"TASK:\n{mandate}\n\n"
        f"LEDGER (persists for the whole run):\n{ledger.render()}\n\n"
        f"{evidence_block}"
        f"SCRATCHPAD (your last {_SCRATCHPAD_WINDOW} steps only — older steps are gone; the "
        f"ledger above is what survives):\n{history}\n\n"
        "Return the next step as JSON."
    )


#: Same `````json`` fence prompted_tools.extract_decision strips before looking for JSON.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _loads_first_object(raw: Any) -> Any:
    """Parse ``raw`` as JSON, tolerating a code fence, surrounding prose, or a small
    deterministic defect (trailing comma, single quotes, unterminated string, unbalanced
    brackets). ``None`` on failure.

    Reuses ``prompted_tools``'s string-aware balanced-span candidate finder and repair, so this
    is no longer the greedy ``re.search(r"[\\[{].*[\\]}]")`` that spanned from the FIRST opener to
    the LAST closer (over-capturing across unrelated JSON-looking fragments). It deliberately does
    NOT go through ``extract_decision`` itself: that function collapses a bare JSON list to its
    first dict item, which is right for a single-decision reply but wrong here — one of this
    module's two callers (:func:`extract_from_page`) legitimately expects a bare list of MANY
    extraction records when a model drops the ``{"extractions": [...]}`` wrapper (see
    ``_records_from_payload``), so the parsed shape (dict, list, or scalar) must pass through
    unchanged.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    fenced_match = _FENCE_RE.search(text)
    if fenced_match:
        text = fenced_match.group(1).strip()

    candidates = _pt_json_candidates(text)
    whole = candidates[0]

    try:
        return json.loads(whole)
    except (json.JSONDecodeError, TypeError):
        pass

    repaired_whole = _pt_repair_json_text(whole)
    if repaired_whole is not None:
        try:
            return json.loads(repaired_whole)
        except (json.JSONDecodeError, TypeError):
            pass

    for candidate in candidates[1:]:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    for candidate in candidates[1:]:
        repaired = _pt_repair_json_text(candidate)
        if repaired is None:
            continue
        try:
            return json.loads(repaired)
        except (json.JSONDecodeError, TypeError):
            continue

    return None


def _check_value(page_text: str, value: str, unit: str):
    """Locate ``value`` (with ``unit``, as one span) in ``page_text``.

    ``evidence_graph`` imports this module, so its verifier is imported HERE rather than at module
    scope; the check itself is not reimplemented.

    :param page_text: the fetched page text.
    :param value: the value the record claims to have read off it.
    :param unit: the unit the record reported alongside it, possibly empty.
    :returns: ``(ValueMatch, shape)`` where shape is ``evidence_graph.value_shape(value, unit)`` —
        the unit FIELD is passed through so a bare number reported alongside a separate unit
        (never embedded in ``value`` itself) still counts as unit-bearing coverage.
    :raises: nothing.
    """
    from agent.app.testing.evidence_graph import value_shape, verify_value

    return verify_value(page_text, value, unit=unit), value_shape(value, unit)


def _records_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict):
        items = payload.get("extractions")
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        return [payload] if payload.get("entity") else []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


async def extract_from_page(agent_io: AgentIO, model_name: str, mandate: str, ledger: Ledger,
                            page_id: str, page_url: str, page_text: str,
                            max_tokens: int = 900) -> List[Extraction]:
    """Run ONE bounded typed-extraction call over a freshly visited page and fold it in.

    :param agent_io: the run's IO facade.
    :param model_name: executor model.
    :param mandate: the task statement (context for what a value means).
    :param ledger: the run's ledger; matched rows are updated via :meth:`Ledger.apply`.
    :param page_id: stable id of the fetched page, kept on every record as a raw pointer.
    :param page_url: the visited URL. Taken from the FETCH, never from the model, so a record
        can't cite a page it did not come from.
    :param page_text: the fetched page text the quote must literally appear in.
    :param max_tokens: cap for the extraction call.
    Each record's VALUE is additionally located on the page by ``evidence_graph.verify_value``,
    together with the unit the model reported, and the record carries what that found
    (``value_verified`` / ``value_unit_bearing`` / ``value_shape``). A unit-bearing span is the
    point: a bare number is far easier to hit by accident, since a page listing feet before metres
    contains both figures. The check is reported, never a gate — a value that fails it is still a
    value the model read off a named page.

    :returns: the records produced, in emission order (also appended to ``ledger.extractions``).
    :raises: nothing — a malformed or failed extraction yields ``[]`` and leaves the ledger as-is.
    """
    if not (page_text or "").strip():
        return []
    # The page is frozen into the graph BEFORE the extraction call, so a page that yields no
    # usable record still leaves an auditable trace of having been fetched and read.
    graph = ledger.ensure_graph()
    if graph.page(page_id) is None:
        graph.add_page(page_id, page_url, page_text)
    rows_block = "\n".join(f"- entity: {row.entity} | field: {row.field} | status: {row.status}"
                           for row in ledger.rows)
    payload = agent_io.build_llm_payload(
        messages=[
            {"role": "system", "content": extract_system_prompt(unit_extraction_enabled())},
            {"role": "user", "content": (
                f"TASK:\n{mandate}\n\nLEDGER ROWS:\n{rows_block}\n\n"
                f"SOURCE {page_url}\nPAGE TEXT:\n{page_text}"
            )},
        ],
        json_mode=True, model_name=model_name, temperature=0.0, max_tokens=max_tokens,
    )
    try:
        raw = await agent_io.query_llm(payload, model_name=model_name)
    except Exception as exc:  # noqa: BLE001. A failed extraction must not end the run.
        _logger.warning(f"[EVIDENCE-LOOP] extraction failed for {page_url}: {exc}")
        return []
    parsed = _loads_first_object(raw)
    _json_telemetry.record(model_name, raw, True, parsed is not None, phase="evidence_loop_extract")

    records: List[Extraction] = []
    for item in _records_from_payload(parsed):
        quote = str(item.get("quote", "") or "")
        match = verify_quote(page_text, quote)
        value = str(item.get("value", "") or "")
        unit = str(item.get("unit", "") or "").strip()
        value_match, shape = _check_value(page_text, value, unit)
        record = Extraction(
            entity=str(item.get("entity", "") or ""),
            field=str(item.get("field", "") or ""),
            value=value,
            verdict=str(item.get("verdict", "") or "").strip().upper(),
            source_url=page_url,
            quote=quote,
            quote_verified=match.verified,
            page_id=page_id,
            quote_start=match.start,
            quote_end=match.end,
            quote_fail_reason=match.fail_reason,
            unit=unit,
            value_verified=value_match.verified,
            value_unit_bearing=value_match.unit_bearing,
            value_shape=shape,
            value_fail_reason=value_match.fail_reason,
            # Gated (default OFF, see extraction_value_gate_enabled): a value that is not
            # `value_verified is True` against the page it cites -- confirmed absent, or never
            # even a checkable value (empty/junk) -- is withheld from resolving a ledger row.
            # `value_verified` already carries one normalized pass (digit grouping, unit
            # spacing, a differently-spelled declared unit), so this never punishes a value that
            # is genuinely on the page under a different surface form.
            excluded_from_support=(extraction_value_gate_enabled()
                                   and value_match.verified is not True),
        )
        # The graph is the single arbiter of admission, so `add_source` is called for EVERY
        # record rather than only for ones `_check_value` already liked: a value it admits gets a
        # prompt handle and becomes referable in a later `derive` call, and a value it refuses is
        # appended to `graph.rejections` WITH its reason. Refusing silently would lose the more
        # interesting half -- the graph's premise is that every operand traces back to a literal
        # span, so which values had nothing to trace to is exactly what an audit wants to read.
        node = graph.add_source(page_id, value, quote=quote, unit=unit or None,
                                label=record.field or None)
        if node is not None:
            ledger.handle_for(node)
            record.evidence_node_id = node.id
        ledger.apply(record)
        records.append(record)
    return records


def render_finalization_context(ledger: Ledger, pages: List[Dict[str, str]],
                                char_budget: int) -> str:
    """Assemble the finalization context TABLE-FIRST.

    The table renders in full and is never truncated; quotes for contested cells come next; raw
    page excerpts get only whatever budget remains, split evenly across pages. That ordering is
    the point — the graph finalizer's page-first assembly truncates at a per-page cap and then
    breaks out of its loop, so late evidence silently disappears. Here only the raw excerpts can
    ever be squeezed.

    :param ledger: the run's ledger.
    :param pages: ``[{"page_id", "url", "text"}, ...]`` in visit order.
    :param char_budget: total context budget; the table may exceed it, nothing else may.
    :returns: the assembled context text.
    :raises: nothing.
    """
    table = ledger.render_table()
    contested = [row for row in ledger.rows
                 if row.status == STATUS_CONFLICTED or (row.value and not row.quote_verified)]
    quote_lines = [f"- {row.entity} | {row.field}: \"{row.quote}\" — {row.source_url}"
                   for row in contested if row.quote]
    for record in ledger.extractions:
        if record.quote and any(_norm(record.entity) == _norm(row.entity) for row in contested):
            line = f"- {record.entity} = {record.value}: \"{record.quote}\" — {record.source_url}"
            if line not in quote_lines:
                quote_lines.append(line)
    quotes_block = "CONTESTED QUOTES:\n" + ("\n".join(quote_lines) if quote_lines else "(none)")
    derived = ledger.derived_lines()
    derived_block = ("DERIVED VALUES (recomputed in Python from the verified spans — state these "
                     "figures as given; do NOT recompute them):\n" + "\n".join(derived)
                     ) if derived else ""

    residual = max(0, int(char_budget) - len(table) - len(quotes_block) - len(derived_block))
    excerpt_lines: List[str] = []
    if pages and residual > 0:
        per_page = max(1, residual // len(pages))
        for page in pages:
            text = str(page.get("text", "") or "")[:per_page]
            excerpt_lines.append(f"SOURCE {page.get('url', '')}\n{text}")
    excerpts_block = "RAW EXCERPTS (truncated to the residual budget):\n" + (
        "\n\n".join(excerpt_lines) if excerpt_lines else "(none)")
    blocks = [table] + ([derived_block] if derived_block else []) + [quotes_block, excerpts_block]
    return "\n\n".join(blocks)


def _empty_quote_counts() -> Dict[str, int]:
    return {"verified": 0, "failed": 0, "unchecked": 0, "absent": 0, "no_page": 0, "empty": 0}


def _tally(counts: Dict[str, int], verified: Optional[bool], reason: Optional[str]) -> None:
    if verified is True:
        counts["verified"] += 1
        return
    counts["failed" if verified is False else "unchecked"] += 1
    if reason in counts:
        counts[reason] += 1


def _cell_output(cell: Dict[str, Any]) -> Dict[str, Any]:
    node = cell if isinstance(cell, dict) else {}
    for key in ("execution", "output"):
        if isinstance(node.get(key), dict):
            node = node[key]
    return node


def quote_verification_counts(ledger: Ledger) -> Dict[str, int]:
    """Split the run's extractions by what quote verification actually found.

    The three causes of a non-verified quote are reported separately because they mean different
    things: ``absent`` is the model paraphrasing or fabricating (the finding this check exists
    for), while ``empty`` and ``no_page`` are extractions nothing could be said about. A pass rate
    without this split is uninterpretable.

    :param ledger: the run's ledger, carrying every append-only extraction record.
    :returns: ``{"verified", "failed", "unchecked", "absent", "no_page", "empty"}`` counts;
        ``verified + failed + unchecked`` is the total number of extractions.
    :raises: nothing.
    """
    counts = _empty_quote_counts()
    for record in ledger.extractions:
        _tally(counts, record.quote_verified, record.quote_fail_reason)
    return counts


def reverify_cell(cell: Dict[str, Any]) -> Dict[str, Any]:
    """Re-check every stored extraction of a finished cell against its PERSISTED page, offline.

    This is the audit path: it needs no network, no model and no GPU, and it turns a pass rate into
    something anyone can reproduce from the artifact alone. Extractions resolve to pages by
    ``page_id``; an extraction whose page was not persisted, or whose quote falls past a truncated
    window, is reported unverifiable rather than failed.

    :param cell: a result-cell dict — the full saved JSON, its ``execution`` node, or the
        ``output`` node itself; all three resolve to the same payload.
    :returns: ``{"pages": n, "extractions": [{page_id, url, quote, quote_verified,
        quote_fail_reason}], "counts": {...}}`` with counts as in
        :func:`quote_verification_counts`.
    :raises: nothing — a cell with no ``extractions`` reports zeroes.
    """
    output = _cell_output(cell)
    pages = {str(page.get("page_id")): page
             for page in output.get("pages", []) if isinstance(page, dict)}
    counts = _empty_quote_counts()
    rows: List[Dict[str, Any]] = []
    for record in output.get("extractions", []):
        if not isinstance(record, dict):
            continue
        page = pages.get(str(record.get("page_id")))
        match = verify_against_stored_page(page, str(record.get("quote", "") or ""))
        _tally(counts, match.verified, match.fail_reason)
        rows.append({
            "page_id": record.get("page_id"), "url": (page or {}).get("url", ""),
            "quote": record.get("quote", ""), "quote_verified": match.verified,
            "quote_fail_reason": match.fail_reason,
        })
    return {"pages": len(pages), "extractions": rows, "counts": counts}


@dataclass
class EvidenceLoopResult:
    """What one run of :func:`run_evidence_loop` produced."""

    deliverable: str
    ledger: Ledger
    scratchpad: List[str]
    verdict: str
    pages: List[Dict[str, Any]] = dataclass_field(default_factory=list)
    #: The run's evidence graph. Taken from the ledger rather than passed at each of the loop's
    #: exit points, so no return path can forget it.
    graph: Any = None

    def __post_init__(self) -> None:
        if self.graph is None:
            self.graph = self.ledger.graph


def _fmt_search(results: List[Dict[str, str]], k: int) -> str:
    lines = []
    for i, item in enumerate((results or [])[:k], 1):
        lines.append(f"{i}. {item.get('title','')} — {item.get('url','')}\n   "
                     f"{item.get('description','')}")
    return "SEARCH RESULTS:\n" + ("\n".join(lines) if lines else "(none)")


async def _verify_claim(agent_io: AgentIO, claim: str, evidence: str, model_name: str) -> str:
    messages = [
        {"role": "system", "content": (
            "Judge whether the CLAIM is supported by the EVIDENCE (text from the pages already "
            "visited). Reply in one line: TRUE / PARTIALLY_TRUE / FALSE / UNVERIFIABLE, then the "
            "supporting-or-contradicting source URL and a brief reason.")},
        {"role": "user", "content": f"CLAIM: {claim}\n\nEVIDENCE:\n{evidence[:8000] or '(no evidence gathered yet)'}"},
    ]
    payload = agent_io.build_llm_payload(messages=messages, json_mode=False,
                                         model_name=model_name, temperature=0.0, max_tokens=300)
    return (await agent_io.query_llm(payload, model_name=model_name)) or "UNVERIFIABLE"


def _handle_derive(ledger: Ledger, args: Dict[str, Any],
                   refusal_repeats: Optional[Dict[Any, Dict[str, Any]]] = None) -> str:
    """Run ONE typed derivation and render it as an observation the model can act on.

    This is the first action with a real argument schema rather than a bare string: the other
    four coerce one key out of ``args`` and nudge in prose when it is missing, which costs a whole
    step and teaches the model nothing. Here the arguments are named
    (``operation`` / ``input_refs`` / ``proposed_value`` / ``expected_unit``) and every refusal
    comes back with its typed code AND what to do about it.

    Declarative by design — ``input_refs`` are handles, never a code string. A model that can send
    an expression to be evaluated can send an expression that fabricates its own inputs, which is
    precisely the failure this layer exists to prevent.

    ``operation`` is normalized through :data:`_DERIVE_OP_ALIASES` before ANYTHING else — dedup
    key, dispatch, the node's stored ``operation`` — so ``multiply`` and ``product`` over the same
    operands are the exact same attempt, not two.

    Repeat-refusal guard (mirrors the ``search`` action's ``seen_queries`` dedup — see the same
    shape at the top of :func:`run_evidence_loop`): on a live campaign one cell retried the SAME
    impossible derivation 10 times, burning a step on an identical ``NON_NUMERIC`` refusal each
    time. ``refusal_repeats`` keys on ``(operation, tuple(input_ids))`` AFTER alias normalisation
    and ref resolution; a repeat short-circuits before the graph is touched, returning a canned
    ``ALREADY REFUSED`` observation that still carries the ORIGINAL typed code and advice — the
    model needs to know what was wrong, not merely that it repeated itself, or the guard just
    trades one dead step for another.

    Design call: a repeat is NOT recorded again on ``graph.derivation_refusals``. Those rows are
    the measurement substrate for the refusal-rate endpoint (see the module note on
    `ledgernum22`'s incompatible-unit tasks); a model stuck in a 10-attempt loop would inflate
    that rate 10x for what is mechanically one refused derivation, corrupting the very endpoint
    this recording exists to make measurable. The looping behaviour itself is not lost — it is
    visible in ``refusal_repeats``' per-key count (available to a caller that wants it), in the
    scratchpad's repeated ``action=derive`` steps, and in the ``ALREADY REFUSED`` text itself — a
    caller can always recover "this was retried N times", just not by inflating a count that
    other code treats as "N independent refusals occurred."

    :param ledger: the run's ledger, carrying the graph and the handle map.
    :param args: the decision's ``args`` object.
    :param refusal_repeats: mutable dedup state, one dict shared across the run's steps (like
        ``seen_queries``); ``None`` disables the guard entirely (``IDEA_TEST_EVIDENCE_LOOP_DEDUP_DERIVE=0``).
    :returns: the observation text, either the recomputed value or a typed refusal.
    :raises: nothing — every refusal is an observation, never an exception out of the loop.
    """
    from agent.app.testing.evidence_graph import (DerivationError, UnknownOperation, WrongArity)

    graph = ledger.ensure_graph()
    operation = str(args.get("operation", "") or "").strip().lower()
    operation = _DERIVE_OP_ALIASES.get(operation, operation)
    refs = args.get("input_refs")
    if isinstance(refs, (str, int)):
        refs = [refs]
    input_ids = [ledger.resolve_ref(ref) for ref in refs] if isinstance(refs, list) else []
    proposed = args.get("proposed_value")
    expected_unit = str(args.get("expected_unit", "") or "").strip()

    dedup_key = (operation, tuple(input_ids))
    if refusal_repeats is not None and dedup_key in refusal_repeats:
        prior = refusal_repeats[dedup_key]
        prior["count"] += 1
        return (f"DERIVE ALREADY REFUSED [{prior['code']}]: same operation and operands as a "
                f"prior attempt ({prior['message']}). {_DERIVE_ADVICE.get(prior['code'], '')} "
                "Retrying will not change the outcome -- try a different approach.").strip()

    if operation in _DERIVE_CONVERSION_OPS:
        exc: Exception = _ConversionUnavailable(
            f"{operation!r} requests a unit conversion, which this system refuses by design")
        graph.record_refusal(operation, input_ids, exc)
        if refusal_repeats is not None:
            refusal_repeats[dedup_key] = {"code": exc.code, "message": str(exc), "count": 1}
        return f"DERIVE REFUSED [{exc.code}]: {exc}. {_DERIVE_ADVICE.get(exc.code, '')}".strip()

    try:
        if operation in _DERIVE_ARITH_OPS:
            node = graph.add_arith(operation, input_ids, proposed_value=proposed)
        elif operation in _DERIVE_EXTREMUM_OPS:
            node = graph.add_extremum(input_ids, operation)
        elif operation == "count":
            node = graph.add_count(input_ids)
        elif operation.startswith("compare_") and operation[8:] in _DERIVE_COMPARE_MODES:
            if len(input_ids) != 2:
                raise WrongArity(f"{operation} needs exactly 2 inputs, got {len(input_ids)}")
            node = graph.add_compare(input_ids[0], input_ids[1], operation[8:])
        else:
            raise UnknownOperation(f"unknown derive operation: {operation!r}")
    except DerivationError as exc:
        # Record it on the ARTIFACT as well as telling the model. A refusal creates no node by
        # design and the scratchpad is not persisted in a result cell, so without this row a
        # refused derivation is indistinguishable from one that was never attempted -- which is
        # exactly what made the incompatible-unit endpoint unmeasurable on `ledgernum22`.
        graph.record_refusal(operation, input_ids, exc)
        if refusal_repeats is not None:
            refusal_repeats[dedup_key] = {"code": exc.code, "message": str(exc), "count": 1}
        return f"DERIVE REFUSED [{exc.code}]: {exc}. {_DERIVE_ADVICE.get(exc.code, '')}".strip()

    handle = ledger.handle_for(node)
    value = Ledger._render_value(node)
    inputs = ", ".join(ledger.handle_for(graph.node(i)) if graph.node(i) else str(i)
                       for i in node.input_ids)
    parts = [f"DERIVED {handle} = {value} ({operation} of {inputs}). Recomputed in Python from "
             f"the verified spans — state THIS figure and do not recompute it yourself."]
    if node.derivation_valid is False and node.derivation_detail:
        parts.append(f"WARNING: {node.derivation_detail}. The recomputed figure stands; your "
                     f"proposed one does not.")
    if expected_unit and node.unit and normalize_for_derive(expected_unit) != normalize_for_derive(node.unit):
        parts.append(f"NOTE: you expected unit {expected_unit!r}; the operands compose to "
                     f"{node.unit!r}.")
    return " ".join(parts)


def normalize_for_derive(unit: Any) -> str:
    """Lowercased, whitespace-stripped unit text, for comparing an EXPECTED unit to a composed one.

    Deliberately not ``evidence_graph.normalize_for_match``: that pass exists to make a value
    locatable in page text and does far more than this comparison wants.
    """
    return re.sub(r"\s+", "", str(unit or "")).lower()


def _decorate(answer: str, ledger: Ledger) -> str:
    """The deliverable: the model's prose, the full table, and BOTH axes stated in one line.

    :param answer: the model's prose answer.
    :param ledger: the run's ledger.
    :returns: the decorated deliverable. The verdict line reports rows resolved and, separately,
        rows backed by a verified quote; when those differ the answer carries an explicit
        ``UNVERIFIED PROVENANCE`` flag, which is more useful than declining the answer outright.
    :raises: nothing.
    """
    counts = ledger.resolution_counts()
    total = counts["rows"]
    verdict_line = (f"VERDICT: {ledger.verdict()} "
                    f"({counts['resolved']}/{total} ledger rows resolved; "
                    f"{counts['resolved_verified']}/{total} backed by a verified verbatim quote)")
    if counts["resolved_unverified"]:
        verdict_line += (f"\nUNVERIFIED PROVENANCE: {counts['resolved_unverified']} resolved "
                         "row(s) cite a quote that is not literally present on the cited page; "
                         "their values are reported, their supporting text is not proven.")
    return f"{answer}\n\n{ledger.render_table()}\n{verdict_line}".strip()


async def run_evidence_loop(agent_io: AgentIO, mandate: str, model_name: str, max_steps: int,
                            max_tokens: int) -> EvidenceLoopResult:
    """Run the flat ReAct loop with the ledger, per-hop extraction and quote grounding.

    :param agent_io: the run's IO facade (search / visit / query_llm / build_llm_payload).
    :param mandate: the task statement.
    :param model_name: executor model.
    :param max_steps: hard step budget for the flat loop.
    :param max_tokens: cap for the final synthesis call.
    :returns: an :class:`EvidenceLoopResult` carrying the deliverable, the ledger (rows and
        append-only extraction records), the full scratchpad and the derived verdict.
    :raises: nothing from the tool surface — a failed search/visit becomes an observation, and a
        malformed decision becomes an empty decision, exactly as in the sequential control.
    """
    page_chars = int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_PAGE_CHARS", "6000"))
    search_k = int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_SEARCH_K", "6"))
    step_max_tokens = int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_STEP_MAX_TOKENS", "4096"))
    final_context_chars = int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_FINAL_CHARS", "12000"))
    store_chars = int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_STORE_CHARS", str(page_chars)))
    dedup_search = os.environ.get("IDEA_TEST_EVIDENCE_LOOP_DEDUP_SEARCH", "1") not in (
        "0", "false", "False")
    dedup_derive = os.environ.get("IDEA_TEST_EVIDENCE_LOOP_DEDUP_DERIVE", "1") not in (
        "0", "false", "False")

    ledger = Ledger.mint(mandate)
    scratchpad: List[str] = []
    pages: List[Dict[str, Any]] = []
    # What the MODEL sees is always the full fetched text: the storage cap must never silently
    # shrink the finalization context.
    context_pages: List[Dict[str, str]] = []
    seen_queries: set = set()
    # (operation, input_ids) -> {"code", "message", "count"} for the LAST refused attempt at that
    # key. Shared across steps exactly like ``seen_queries``.
    derive_refusals: Dict[Any, Dict[str, Any]] = {}
    last_answer = ""

    for step in range(max_steps):
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": compose_user_prompt(mandate, ledger, scratchpad)},
        ]
        payload = agent_io.build_llm_payload(messages=messages, json_mode=True,
                                             model_name=model_name, temperature=0.1,
                                             max_tokens=step_max_tokens)
        raw = await agent_io.query_llm(payload, model_name=model_name)
        decision = _loads_first_object(raw)
        _json_telemetry.record(model_name, raw, True, decision is not None, phase="evidence_loop")
        if isinstance(decision, list):
            decision = next((item for item in decision if isinstance(item, dict)), {})
        if not isinstance(decision, dict):
            decision = {}
        action = str(decision.get("action", "")).strip().lower()
        args = decision.get("args")
        if not isinstance(args, dict):
            args = {}
        thought = str(decision.get("thought", ""))[:300]

        if action == "finish" or step == max_steps - 1:
            last_answer = str(args.get("answer", "")) or last_answer
            if last_answer:
                return EvidenceLoopResult(_decorate(last_answer, ledger), ledger, scratchpad,
                                          ledger.verdict(), pages)
            context = render_finalization_context(ledger, context_pages, final_context_chars)
            messages = [
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user", "content": f"TASK:\n{mandate}\n\n{context}"},
            ]
            payload = agent_io.build_llm_payload(messages=messages, json_mode=False,
                                                 model_name=model_name, temperature=0.3,
                                                 max_tokens=max_tokens)
            answer = (await agent_io.query_llm(payload, model_name=model_name)) or ""
            return EvidenceLoopResult(_decorate(answer, ledger), ledger, scratchpad,
                                      ledger.verdict(), pages)

        if action == "search":
            query = str(args.get("query", ""))
            norm = re.sub(r"\s+", " ", query).strip().lower()
            if dedup_search and norm and norm in seen_queries:
                obs = (f"ALREADY SEARCHED '{query[:80]}'. Its results are in your scratchpad above — "
                       "VISIT one of those result URLs to read it, or FINISH if you have enough. "
                       "Do not repeat a search you have already run.")
            else:
                if norm:
                    seen_queries.add(norm)
                try:
                    results = await agent_io.search(query, count=search_k, timeout_seconds=20)
                    obs = _fmt_search(results or [], search_k)
                except Exception as exc:  # noqa: BLE001
                    obs = f"SEARCH ERROR: {exc}"
        elif action == "visit":
            url = str(args.get("url", "")).strip()
            try:
                content = (await agent_io.visit(url, timeout_seconds=30) or "")[:page_chars]
                error = None
            except Exception as exc:  # noqa: BLE001
                content, error = "", exc
            if error is not None:
                obs = f"VISIT ERROR for {url}: {error}"
            elif not content.strip():
                obs = f"VISIT ERROR for {url}: no extractable page text"
            else:
                page_id = f"p{len(pages) + 1}"
                pages.append(store_page(page_id, url, content, store_chars))
                context_pages.append({"page_id": page_id, "url": url, "text": content})
                await extract_from_page(agent_io, model_name, mandate, ledger,
                                        page_id=page_id, page_url=url, page_text=content)
                obs = f"PAGE {url}:\n{content}"
        elif action == "derive":
            obs = _handle_derive(ledger, args, derive_refusals if dedup_derive else None)
        elif action == "verify":
            claim = str(args.get("claim", ""))
            evidence = "\n\n".join(f"SOURCE {p['url']}\n{p['text']}" for p in context_pages)
            verdict = await _verify_claim(agent_io, claim, evidence, model_name)
            obs = f"VERIFY '{claim[:80]}': {verdict}"
        else:
            obs = "INVALID ACTION. Use search/visit/derive/verify/finish."

        scratchpad.append(
            f"STEP {step+1}: thought={thought}\naction={action} args={json.dumps(args)[:200]}\n"
            f"observation={obs[:_UNCAPPED_OBSERVATION_CHARS]}")

    return EvidenceLoopResult(_decorate(last_answer, ledger), ledger, scratchpad, ledger.verdict(),
                              pages)


async def run_evidence_loop_execution(
    test_module: IdeaTestModule,
    model_name: str,
    connector_llm: ConnectorLLM,
    connector_search: ConnectorSearch,
    connector_http: ConnectorHttp,
    connector_chroma: ConnectorChroma,
    run_stamp: str,
    cell_tag: str = "",
    summarize_observability_func=summarize_observability,
    connector_browser=None,
    idea_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the evidence-loop variant; same result shape as every other execution variant.

    ``output`` carries the standard ``final_deliverable`` / ``success`` / ``action_summary`` keys
    the analysis scripts read, PLUS this arm's own fields — additive, so nothing downstream has to
    know about them. They report the two axes separately: RESOLUTION in ``ledger_verdict``,
    ``ledger_resolution_counts``, ``rows_resolved``, ``rows_quote_backed``, ``rows_unresolved``,
    ``unverified_provenance`` and each row's ``confidence_tier``; ROSTER COMPLETENESS in
    ``roster_named`` / ``roster_resolved`` / ``roster_rows`` / ``roster_truncated`` /
    ``roster_complete`` / ``roster_gate``, so a count asserted over a partial closed roster is
    visible in the artifact; VERIFICATION in
    ``quote_verified_count``, ``quote_unverified_count``, ``quote_unchecked_count`` and
    ``quote_fail_reasons``, counted over extractions and unaffected by how rows resolved. ``pages`` freezes every visited page
    (id, URL, content hash, text) so :func:`reverify_cell` can re-audit the run's quotes offline
    from the saved artifact alone.

    :param test_module: the task under test.
    :param model_name: executor model.
    :param connector_llm: LLM connector (its model is set from ``model_name``).
    :param connector_search: search connector.
    :param connector_http: HTTP fetch connector.
    :param connector_chroma: vector-store connector.
    :param run_stamp: the run's timestamp, used in the correlation id and trace path.
    :param cell_tag: disambiguating suffix shared with the result JSON's filename.
    :param summarize_observability_func: observability summarizer (visit counts etc.).
    :param connector_browser: optional headless-Chrome fallback, wired like every other arm.
    :param idea_settings: accepted for signature parity and currently UNREAD — this arm has no
        graph knobs, and pretending otherwise would suggest it can be tuned when it cannot.
    :returns: the standard execution-result dict.
    :raises: nothing — a crashing loop is reported as an unsuccessful, well-formed result.
    """
    connector_llm.set_model(model_name)
    test_id = test_module.metadata.get("test_id", "unknown")
    correlation_id = f"idea_test_{test_id}_{model_name}_evidence_loop_{run_stamp}"

    results_dir = Path(__file__).resolve().parent.parent.parent / "idea_test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    trace_path = build_trace_path(results_dir, run_stamp, test_id, model_name, "evidence_loop",
                                  cell_tag)
    tracer = TraceRecorder(trace_path)

    mandate = test_module.get_task_statement()
    mandate_suffix = os.environ.get("IDEA_TEST_MANDATE_SUFFIX", "").strip()
    if mandate_suffix:
        mandate = f"{mandate}\n\n{mandate_suffix}"

    telemetry = TelemetrySession(enabled=True, mandate=mandate, correlation_id=correlation_id,
                                 trace_path=trace_path)
    agent_io = AgentIO(
        connector_llm=connector_llm, connector_search=connector_search,
        connector_http=connector_http, connector_chroma=connector_chroma,
        connector_browser=connector_browser,
        telemetry=telemetry, collection_name=f"idea_test_{test_id}_{run_stamp}",
    )

    max_steps = int(os.environ.get("IDEA_TEST_EVIDENCE_LOOP_MAX_STEPS", "25"))
    max_tokens = int(os.environ.get("IDEA_TEST_BASELINE_MAX_TOKENS", "8192"))
    started = time.perf_counter()
    result: Optional[EvidenceLoopResult] = None
    try:
        result = await run_evidence_loop(agent_io, mandate, model_name, max_steps, max_tokens)
    except Exception as exc:  # noqa: BLE001. Same failure contract as the sibling variants.
        _logger.error(f"Evidence loop failed: {exc}", exc_info=True)

    ledger = result.ledger if result else Ledger.mint(mandate)
    deliverable = result.deliverable if result else ""
    quote_counts = quote_verification_counts(ledger)
    resolution_counts = ledger.resolution_counts()
    roster = ledger.roster()
    output = {
        "final_deliverable": deliverable,
        "success": bool(deliverable),
        "goal_achieved": None,
        "action_summary": "evidence_loop",
        "ledger": [row.as_dict() for row in ledger.rows],
        "ledger_verdict": result.verdict if result else ledger.verdict(),
        "ledger_status_counts": ledger.status_counts(),
        "ledger_resolution_counts": resolution_counts,
        "rows_resolved": resolution_counts["resolved"],
        "rows_quote_backed": resolution_counts["resolved_verified"],
        "rows_unresolved": resolution_counts["unresolved"],
        "unverified_provenance": bool(resolution_counts["resolved_unverified"]),
        "extractions": [record.as_dict() for record in ledger.extractions],
        **roster,
        "roster_gate": roster_gate_enabled(),
        "pages": result.pages if result else [],
        "quote_verified_count": quote_counts["verified"],
        "quote_unverified_count": quote_counts["failed"],
        "quote_unchecked_count": quote_counts["unchecked"],
        "quote_fail_reasons": {reason: quote_counts[reason]
                               for reason in (QUOTE_FAIL_ABSENT, QUOTE_FAIL_NO_PAGE,
                                              QUOTE_FAIL_EMPTY)},
        # The derivation graph rides in `output`, NOT in the cell's top-level "graph" key: that
        # one is taken by the link-graph null object (`testing/execution._empty_graph`). Living
        # in `output` is also what makes it free to audit offline -- `reverify_cell` already
        # reads from there, and `evidence_graph.reverify_graph` re-checks this payload with no
        # model, no network and no GPU.
        #
        # `include_page_text=False`: the fetched page text already lives verbatim in `pages`
        # (above), so embedding it a second time here was pure duplication -- measured at 5.7MB
        # across 247 stored cells, the single largest redundancy in the artifact. Standalone audit
        # (no cell around it, just this `evidence_graph` payload) still works: `reverify_graph`
        # takes an optional `pages=` argument to supply the text back in from this cell's own
        # `pages` list, and `scripts/reverify.py` does exactly that.
        "evidence_graph": (ledger.graph.to_dict(include_page_text=False)
                           if ledger.graph is not None else None),
        "derivation_gate": derivation_gate_enabled(),
        "derivation_validity": (ledger.graph.derivation_validity()
                                if ledger.graph is not None else None),
        # Refused derivations, tallied by typed code. The endpoint the incompatible-unit tasks
        # (222-224) exist to measure: a correct refusal and a never-attempted derivation both
        # leave zero DERIVED nodes, and only this distinguishes them.
        "derivation_refusals": (ledger.graph.refusal_counts()
                                if ledger.graph is not None else None),
    }
    # The confidence/abstain channel every arm reports (see `confidence_channel` module docstring):
    # for this arm it is a direct re-export of the ledger verdict already computed above.
    channel = confidence_channel.from_evidence_loop_verdict(output["ledger_verdict"])
    if channel is not None:
        output.update(channel)
    telemetry.finish(success=output["success"])
    tracer.close()

    observability = summarize_observability_func({"output": output}, telemetry, model_name)
    telemetry_summary = telemetry.summary()
    ended = time.perf_counter()

    if not traces_retained():
        try:
            if trace_path.exists():
                trace_path.unlink()
        except Exception as exc:  # noqa: BLE001
            _logger.warning(f"Failed to delete trace file {trace_path}: {exc}")

    return {
        "output": output,
        "graph": _empty_graph(),
        "observability": observability,
        "duration_seconds": round(max(0.0, ended - started), 2),
        "telemetry": {
            "correlation_id": correlation_id,
            "trace_file": str(trace_path),
            "events_count": len(telemetry.events),
            "timings_count": len(telemetry.timings),
        },
        "telemetry_raw": telemetry_summary,
    }
