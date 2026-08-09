"""
Offline tests for the thin-leaf extraction fixes — free, no network.

Fix 1 (default ON, ``IDEA_TEST_COMPILED_STRIP_SOURCE_ASK``): the plan-authored "...and the exact
source URL" ask is removed from the EXTRACTION QUESTION only. ``_THIN_EXTRACT_SYS`` says "no
source" while ~2/3 of the hand-authored leaf instructions end with that ask (they are reused
verbatim as the question); live ablation on the real task-072 Sarez Lake leaf showed qwen2.5:7b
resolves the contradiction by abstaining (10/10 UNKNOWN), and that removing the clause flips it to
the correct value. ``_run_leaf_thin`` appends the real URL itself.

Fix 3 (default ON, ``IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY``): ONE extra vote per candidate page,
with the directive ``_THIN_EXTRACT_SYS_RETRY`` prompt, only when the first vote was inconclusive.
Flipped from default-OFF after a live calibration run (reachable tier, R=3) showed a real,
reproduced lift (avg 0.941->0.959) with no regressions on any task.

The load-bearing guarantees asserted here:
  * the RAW instruction (not the stripped one) still drives ``_target_entity``/``_leaf_search_query``;
  * a non-source-ask instruction survives ``_strip_source_ask`` byte-for-byte;
  * with the retry disabled (the default) the call sequence is exactly what it was before.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.testing import execution_compiled as ec
from agent.app.testing import consol_pilot


# --- Fix 1: _strip_source_ask ---------------------------------------------------------------
# Every LHS below is a real (or realistically shaped) string from the idea_tests plan corpus.
STRIP_CASES = [
    # inline, the dominant shape (54 leaves): "... and the source URL."
    ("Report the tower's height and the source URL. Do not guess from memory.",
     "Report the tower's height. Do not guess from memory."),
    ("Do NOT answer from memory; report the exact figure with units and the source URL.",
     "Do NOT answer from memory; report the exact figure with units."),
    ("Report ONLY that single population number and the source URL. Do not guess from memory.",
     "Report ONLY that single population number. Do not guess from memory."),
    # inline with an "exact" qualifier and a preceding comma
    ("Report the surviving warship, its displacement in tons, and the exact source URL. "
     "Do not guess from memory.",
     "Report the surviving warship, its displacement in tons. Do not guess from memory."),
    # inline "and cite ..." variants
    ("Report the correct architectural height, identify the tip figure as a different scope, "
     "and cite the source URL.",
     "Report the correct architectural height, identify the tip figure as a different scope."),
    ("Report the correct current official height, the figure that is now superseded, and cite the "
     "authoritative source URL.",
     "Report the correct current official height, the figure that is now superseded."),
    # inline possessive ("cite each <noun>'s source URL")
    ("List the five lakes and their maximum-depth figures, and cite each lake's source URL.",
     "List the five lakes and their maximum-depth figures."),
    # inline preceded by an em-dash aside (the dangling connective must go with it)
    ("Report the DECLARED figure in feet — NOT the exact rounded computed value and NOT any modern "
     "re-surveyed height — and the source URL. Do not guess from memory.",
     "Report the DECLARED figure in feet — NOT the exact rounded computed value and NOT any modern "
     "re-surveyed height. Do not guess from memory."),
    # enumerated-list form (the gap the design flagged: not matched by the two sketched patterns)
    ("Report: (a) the atomic number, (b) the Kelvin melting point, (c) the source URL.",
     "Report: (a) the atomic number, (b) the Kelvin melting point."),
    ("Report (a) the winner, (b) the runner-up, and (d) the exact source URL of every page you read.",
     "Report (a) the winner, (b) the runner-up."),
    ("Report the dam name and (d) the source URL for each dam.", "Report the dam name."),
    ("Give (e) each institution's source URL.", "Give."),
    # standalone "Cite ... source URL." sentence
    ("Read the figure off the infobox. Cite the exact authoritative source URL you read the "
     "figures from. Do not answer from memory.",
     "Read the figure off the infobox. Do not answer from memory."),
    ("Cite the source URLs you used. Do not answer from memory.", "Do not answer from memory."),
    # standalone "Give the source URL." sentence — the odd-one-out tasks' (069/080) real phrasing,
    # a different imperative verb hitting the exact same self-contradiction as "Cite ..."
    ("Answer with the country's landlocked status. Give the source URL. Do not guess from memory.",
     "Answer with the country's landlocked status. Do not guess from memory."),
]

# Instructions with nothing to strip — these must come back BYTE-IDENTICAL.
UNTOUCHED_CASES = [
    "Read the value off the infobox. Do not guess from memory.",
    "Report ONLY that single population number.",
    "Report the source of the Nile and its length in km.",          # 'source' but not a source-ask
    "Report the 'Source' field of the river infobox.",
    "Deliverable: (a) a markdown table with columns: Bridge | Main span (m) | Source URL.",
    "(c) which station it was; citing every source URL.",           # asks for the station too
    "Open the Wikipedia article for Lake Baikal and read its maximum depth in metres.",
    "",
]


@pytest.mark.parametrize("raw,expected", STRIP_CASES)
def test_strip_source_ask_removes_the_url_ask(raw, expected):
    assert ec._strip_source_ask(raw) == expected


@pytest.mark.parametrize("raw,_expected", STRIP_CASES)
def test_strip_source_ask_leaves_no_dangling_connective(raw, _expected):
    """No trailing ', and' / ' —' / doubled space / ' .' left where the clause was cut out."""
    out = ec._strip_source_ask(raw)
    assert "source URL" not in out.lower()
    assert "  " not in out
    assert " ." not in out and " ," not in out
    assert not out.rstrip(".").rstrip().endswith((",", ";", "and", "—", "–", ":"))


@pytest.mark.parametrize("raw", UNTOUCHED_CASES)
def test_strip_source_ask_leaves_unrelated_text_byte_identical(raw):
    assert ec._strip_source_ask(raw) == raw


def test_strip_source_ask_on_the_real_072_leaf():
    """The design's §4 before/after example, verbatim from test_072."""
    raw = ("Open the Wikipedia article for Sarez Lake (Tajikistan) and read, directly from the "
           "infobox, its MAXIMUM DEPTH in metres — the 'Max. depth' field. Report ONLY that single "
           "depth figure (a whole number in metres) and the exact source URL. Do NOT report "
           "surface elevation or average depth. Do not guess from memory.")
    assert ec._strip_source_ask(raw) == (
        "Open the Wikipedia article for Sarez Lake (Tajikistan) and read, directly from the "
        "infobox, its MAXIMUM DEPTH in metres — the 'Max. depth' field. Report ONLY that single "
        "depth figure (a whole number in metres). Do NOT report "
        "surface elevation or average depth. Do not guess from memory.")


def test_strip_source_ask_on_the_real_069_leaf():
    """069/080's real odd-one-out leaf ending — 'Give the source URL.' is a standalone-sentence
    self-contradiction, exactly like 072's 'Cite ...' shape but with a different verb; a corpus
    grep confirmed this is the only other real instance of the standalone-sentence shape."""
    raw = ("Open the Wikipedia page for Austria and determine, from the lead sentence and the "
           "infobox, whether Austria is a LANDLOCKED country (it has NO sea coastline) or whether "
           "it HAS a sea coastline. Answer strictly with 'Austria: landlocked' or 'Austria: not "
           "landlocked (coastline on <sea>)' -- ALWAYS repeat the country name 'Austria' at the "
           "start of your answer so the fact is self-contained even if read out of context. Give "
           "the source URL. Do not guess from memory.")
    assert ec._strip_source_ask(raw) == (
        "Open the Wikipedia page for Austria and determine, from the lead sentence and the "
        "infobox, whether Austria is a LANDLOCKED country (it has NO sea coastline) or whether "
        "it HAS a sea coastline. Answer strictly with 'Austria: landlocked' or 'Austria: not "
        "landlocked (coastline on <sea>)' -- ALWAYS repeat the country name 'Austria' at the "
        "start of your answer so the fact is self-contained even if read out of context. Do not "
        "guess from memory.")


def test_strip_source_ask_never_returns_an_empty_question():
    """A degenerate instruction that is ONLY a source-ask would leave nothing to extract — fall
    back to the original rather than sending an empty QUESTION."""
    raw = "Cite the exact authoritative source URL you read the figures from."
    assert ec._strip_source_ask(raw) == raw


def test_strip_source_ask_enabled_by_default_with_an_escape_hatch(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_COMPILED_STRIP_SOURCE_ASK", raising=False)
    assert ec._strip_source_ask_enabled() is True          # deliberate default-ON
    monkeypatch.setenv("IDEA_TEST_COMPILED_STRIP_SOURCE_ASK", "0")
    assert ec._strip_source_ask_enabled() is False
    monkeypatch.setenv("IDEA_TEST_COMPILED_STRIP_SOURCE_ASK", "1")
    assert ec._strip_source_ask_enabled() is True


# --- Fix 1 wiring: the question is trimmed, the search path is NOT --------------------------
_LEAF_INSTRUCTION = (
    "Open the Wikipedia article for Sarez Lake (Tajikistan) and read its MAXIMUM DEPTH in metres "
    "from the infobox. Report ONLY that single depth figure and the exact source URL. "
    "Do not guess from memory."
)


def _thin_io(answers, page="Max. depth 505 m"):
    """AgentIO mock for a _run_leaf_thin run whose extractions return ``answers`` in order."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=list(answers))
    io.search = AsyncMock(return_value=[
        {"title": "Sarez Lake", "url": "https://en.wikipedia.org/wiki/Sarez_Lake", "description": ""},
        {"title": "Usoi Dam", "url": "https://en.wikipedia.org/wiki/Usoi_Dam", "description": ""},
    ])
    io.visit = AsyncMock(return_value=page)
    return io


def _questions(io):
    """The QUESTION text of every extraction call made through build_llm_payload."""
    out = []
    for call in io.build_llm_payload.call_args_list:
        messages = call.kwargs["messages"]
        content = messages[-1]["content"]
        if "QUESTION: " in content:
            out.append(content.split("QUESTION: ", 1)[1])
    return out


def _system_prompts(io):
    return [c.kwargs["messages"][0]["content"] for c in io.build_llm_payload.call_args_list]


def test_run_leaf_thin_passes_the_raw_instruction_to_the_search_path(monkeypatch):
    """LOAD-BEARING: Fix 1 must never touch the query/page-pick path. ``_target_entity`` and
    ``_leaf_search_query`` keep seeing the FULL original instruction (they are tuned on it — the
    'article for <SUBJECT>' cue, the parenthetical gloss), only the extraction QUESTION is trimmed.
    """
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_STRIP_SOURCE_ASK", "1")
    seen_target, seen_query = [], []
    real_target, real_query = ec._target_entity, ec._leaf_search_query

    def spy_target(instruction):
        seen_target.append(instruction)
        return real_target(instruction)

    def spy_query(instruction, target):
        seen_query.append(instruction)
        return real_query(instruction, target)

    monkeypatch.setattr(ec, "_target_entity", spy_target)
    monkeypatch.setattr(ec, "_leaf_search_query", spy_query)

    io = _thin_io(["505"])
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))

    assert "505" in out and "Sarez_Lake" in out
    assert seen_target and all(seen == _LEAF_INSTRUCTION for seen in seen_target)   # incl. _pick_pages
    assert seen_query == [_LEAF_INSTRUCTION]
    assert io.search.await_args.args[0] == "Sarez Lake Tajikistan"                  # unchanged query
    # ... while the extraction question DID lose the ask.
    assert _questions(io) == [ec._strip_source_ask(_LEAF_INSTRUCTION)]
    assert "source URL" not in _questions(io)[0]


def test_run_leaf_thin_question_keeps_the_ask_when_disabled(monkeypatch):
    """``IDEA_TEST_COMPILED_STRIP_SOURCE_ASK=0`` restores the pre-change question byte-for-byte."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_STRIP_SOURCE_ASK", "0")
    io = _thin_io(["505"])
    asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert _questions(io) == [_LEAF_INSTRUCTION]


def test_run_leaf_thin_still_appends_the_real_url(monkeypatch):
    """The URL the model no longer has to produce is appended by the harness, as before."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    io = _thin_io(["505"])
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert out == "505 — source: https://en.wikipedia.org/wiki/Sarez_Lake"


# --- Fix 2 wiring: the infobox block is opt-in ----------------------------------------------
def test_infobox_block_flag_default_off_leaves_visit_call_unchanged(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.delenv("IDEA_TEST_COMPILED_INFOBOX_BLOCK", raising=False)
    assert ec._infobox_block_enabled() is False
    io = _thin_io(["505"])
    asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert io.visit.await_args.kwargs == {"timeout_seconds": 30}    # no new kwarg on the default path


def test_infobox_block_flag_threads_prepend_infobox_into_visit(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_INFOBOX_BLOCK", "1")
    assert ec._infobox_block_enabled() is True
    io = _thin_io(["505"])
    asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert io.visit.await_args.kwargs == {"timeout_seconds": 30, "prepend_infobox": True}


# --- Fix 3: sys_prompt override + the bounded retry ------------------------------------------
def test_thin_extract_once_uses_the_default_system_prompt():
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(return_value="505")
    asyncio.run(ec._thin_extract_once(io, "page", "q", "m", 0.0))
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS]


def test_thin_extract_once_honours_a_sys_prompt_override():
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(return_value="505")
    asyncio.run(ec._thin_extract_once(io, "page", "q", "m", 0.0, ec._THIN_EXTRACT_SYS_RETRY))
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS_RETRY]


def _vote_with_prompt(answers, k=None, sys_prompt=None):
    """``_vote_extract`` over canned per-sample answers, optionally with an alternate prompt.
    Returns ``(answer, io)`` so callers can inspect the system prompts that were sent."""
    io = MagicMock()
    io.build_llm_payload = MagicMock(return_value={})
    io.query_llm = AsyncMock(side_effect=list(answers))
    kwargs = {} if sys_prompt is None else {"sys_prompt": sys_prompt}
    ans = asyncio.run(ec._vote_extract(io, "p", "q", "m", k if k is not None else len(answers), **kwargs))
    return ans, io


def test_vote_extract_sys_prompt_reaches_every_sample():
    ans, io = _vote_with_prompt(["505", "505", "505"], sys_prompt=ec._THIN_EXTRACT_SYS_RETRY)
    assert "505" in ans
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS_RETRY] * 3


def test_vote_extract_quorum_rules_are_unchanged_under_a_sys_prompt_override():
    """Same fixtures as the quorum tests, run through the alternate prompt: the lone-survivor
    rejection, the corroborated minority, the anchor tie-break and the k<=1 short-circuit all
    behave identically — the prompt swap only changes the wording sent to the model."""
    retry = ec._THIN_EXTRACT_SYS_RETRY
    assert _vote_with_prompt(["UNKNOWN", "UNKNOWN", "1,320 m", "UNKNOWN", "UNKNOWN"],
                             sys_prompt=retry)[0] == ""
    assert "1,642" in _vote_with_prompt(["UNKNOWN", "1,642 m", "UNKNOWN", "1,642 m", "UNKNOWN"],
                                        sys_prompt=retry)[0]
    assert "1,642" in _vote_with_prompt(["1,642 m", "1,700 m", "1,700 m", "1,642 m"],
                                        sys_prompt=retry)[0]     # anchor wins the 2-2 tie
    assert "1,642" in _vote_with_prompt(["1,642 m"], k=1, sys_prompt=retry)[0]
    assert _vote_with_prompt(["UNKNOWN"], k=1, sys_prompt=retry)[0] == ""


def test_consol_path_does_not_drop_the_sys_prompt(monkeypatch):
    """The opt-in SPRT sampler must not silently fall back to the default prompt."""
    monkeypatch.setattr(consol_pilot, "consol_enabled", lambda: True)

    async def fake_consol_vote(sample, *, k, key_fn, **kwargs):
        await sample(0.0)
        return None                      # -> caller keeps fixed-k, which samples again

    monkeypatch.setattr(consol_pilot, "consol_vote", fake_consol_vote)
    ans, io = _vote_with_prompt(["505", "505", "505", "505"], k=3,
                                sys_prompt=ec._THIN_EXTRACT_SYS_RETRY)
    assert "505" in ans
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS_RETRY] * 4    # 1 ConSol sample + 3 fixed-k


def test_leaf_extract_retry_budget_env(monkeypatch):
    monkeypatch.delenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", raising=False)
    assert ec._leaf_extract_retry_budget() == 1                      # deliberate default-ON
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "0")
    assert ec._leaf_extract_retry_budget() == 0                      # explicit escape hatch
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "-3")
    assert ec._leaf_extract_retry_budget() == 0                      # <= 0 disables
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "nonsense")
    assert ec._leaf_extract_retry_budget() == 0


def test_retry_enabled_by_default_call_sequence(monkeypatch):
    """Default (unset -> ``1``): a quorum-inconclusive page gets one retry pass with the directive
    prompt before moving on to the next candidate."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.delenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", raising=False)
    io = _thin_io(["UNKNOWN", "505"])
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert out == "505 — source: https://en.wikipedia.org/wiki/Sarez_Lake"
    assert io.visit.await_count == 1                                  # same page, no extra fetch
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS, ec._THIN_EXTRACT_SYS_RETRY]


def test_no_retry_when_explicitly_disabled_call_sequence_is_unchanged(monkeypatch):
    """``0`` (the escape hatch): an inconclusive page moves straight on to the next candidate —
    exactly one extraction per page, two pages, then UNKNOWN. Byte-identical to the pre-retry
    behavior."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "0")
    io = _thin_io(["UNKNOWN", "UNKNOWN"])
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert out == "UNKNOWN — https://en.wikipedia.org/wiki/Sarez_Lake"
    assert io.visit.await_count == 2 and len(_questions(io)) == 2
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS] * 2         # never the retry prompt


def test_bounded_retry_runs_once_on_the_same_page_and_succeeds(monkeypatch):
    """Enabled: the first vote on page 1 is inconclusive, the retry vote on the SAME page (already
    fetched) succeeds — no second page is visited and no third extraction is made."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "1")
    io = _thin_io(["UNKNOWN", "505"])
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert out == "505 — source: https://en.wikipedia.org/wiki/Sarez_Lake"
    assert io.visit.await_count == 1                                  # same page, no extra fetch
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS, ec._THIN_EXTRACT_SYS_RETRY]


def test_bounded_retry_is_exactly_one_pass_per_page(monkeypatch):
    """Never more than one retry per page: 2 pages x (1 vote + 1 retry) = 4 extractions, then
    UNKNOWN — the retry never loops and never runs a third time on a page."""
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "5")   # value is NOT a loop count
    io = _thin_io(["UNKNOWN"] * 6)
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert out == "UNKNOWN — https://en.wikipedia.org/wiki/Sarez_Lake"
    assert io.visit.await_count == 2
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS, ec._THIN_EXTRACT_SYS_RETRY] * 2


def test_bounded_retry_never_runs_on_a_page_that_succeeded(monkeypatch):
    monkeypatch.setenv("IDEA_TEST_COMPILED_VOTES", "1")
    monkeypatch.setenv("IDEA_TEST_COMPILED_LEAF_EXTRACT_RETRY", "1")
    io = _thin_io(["505", "UNKNOWN"])
    out = asyncio.run(ec._run_leaf_thin(io, _LEAF_INSTRUCTION, "depth", "m", 6000, 6))
    assert out == "505 — source: https://en.wikipedia.org/wiki/Sarez_Lake"
    assert _system_prompts(io) == [ec._THIN_EXTRACT_SYS]               # zero added cost on success
