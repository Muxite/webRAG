"""
Offline tests for the strategy-library LEAK GATE — free, no LLM, no Chroma, no network.

The gate's whole claim is "here is the automated check every library entry passed, and here is
the proof it catches the exact leak class that got through before". That proof is the first
section of this file and it is not optional:

* the ACTUAL pre-fix text of ``test_c30_line_diff_patch.py`` / ``test_c32_email_fsm_validator.py``
  (commit ``c0dbc720``, vendored in ``fixtures/c0dbc720_prefix_leaks.json`` and re-derived from
  git whenever git is available) MUST be rejected;
* the post-fix text of the same two plans MUST pass, or the gate is not discriminating between
  "leaks" and "is about the same task" — it is just refusing everything;
* the ``mead``/``res_mead`` case from task 146's own leak test MUST be rejected, which a
  ``\\b``-anchored blocklist provably cannot do (asserted here alongside).

Then the layers individually (ledger construction, normalized overlap, the worked-example lint,
the batched LLM auditor through a canned stub) and the write-time policy that an unavailable
auditor is a rejection rather than a silent three-layer pass.
"""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from agent.app.strategy_library import leak_gate as LG

_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "c0dbc720_prefix_leaks.json"
_REPO_ROOT = Path(__file__).resolve().parents[3]

_CASES = json.loads(_FIXTURE.read_text(encoding="utf-8"))["cases"]

#: A genuinely GENERALIZED note of the argmax archetype (a paraphrase of
#: ``plan_library/templates/argmax_over_n_page_field.json``'s own aggregation guidance — the
#: text the live kill-switch A/B actually measured). It must pass against every task's ledger:
#: a gate that rejects this rejects the entire mechanism.
GENERALIZED_ADVICE = (
    "When several candidates must be compared on one looked-up quantity, write every "
    "candidate's value out explicitly, one per line, BEFORE naming a winner. The winner is not "
    "necessarily the most famous or the largest candidate: compare the figures you actually "
    "read, not reputation. If a value is missing for a candidate, say which one is missing "
    "rather than guessing it."
)


# --------------------------------------------------------------------------------------
# THE REGRESSION FIXTURES — the only real proof the gate isn't theatre
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("task_id", sorted(_CASES))
def test_c0dbc720_pre_fix_text_is_rejected(task_id):
    """The exact text commit c0dbc720 had to remove must fail the gate.

    Both leaks were natural-language paraphrase, not source code, so the check that existed for
    that domain (``"def test_" not in plan_text``) could not see them — asserted below.
    """
    ledger = LG.build_ledger(task_id)
    verdict = LG.check_text(_CASES[task_id]["pre_fix"], [ledger])
    assert not verdict.passed, f"c0dbc720's pre-fix {task_id} text passed the gate"
    assert verdict.rejections
    assert "def test_" not in _CASES[task_id]["pre_fix"], (
        "the pre-fix text contains no pytest source — which is exactly why the old "
        '`"def test_" not in plan_text` check missed it'
    )


def test_c30_pre_fix_leaks_the_hidden_middle_change_vectors():
    """c30's leak was the hidden keystone case's INPUT LISTS, re-punctuated.

    ``old=['a','b','c','d','e'], new=['a','x','c','d','f']`` in the plan vs
    ``["a", "b", "c", "d", "e"]`` in the hidden test: no literal substring survives that
    rewrite, which is why sequence entries are matched as consecutive whole word RUNS.
    """
    ledger = LG.build_ledger("c30")
    hits = LG.overlap_findings(_CASES["c30"]["pre_fix"], ledger)
    evidence = {f.evidence for f in hits}
    assert "a x c d f" in evidence, f"expected the hidden input vector among {evidence}"
    assert _CASES["c30"]["pre_fix"].count("'a','x','c','d','f'") == 1, (
        "fixture drift: the pre-fix text no longer carries the re-punctuated vector"
    )


def test_c32_pre_fix_leaks_hidden_addresses_but_not_the_published_ones():
    """c32's leak was five HIDDEN invalid addresses; three others in the same sentence are
    published in the task statement and must NOT be flagged.

    That asymmetry is the ledger's public-subtraction doing its job: flagging a solver for
    repeating a *given* would be noise, and noise is what makes a gate get switched off.
    """
    ledger = LG.build_ledger("c32")
    values = {e.value for e in ledger.entries}
    for hidden in (".user@example.com", "user@examplecom", "john.doe@example.com"):
        assert LG.normalize_tight(hidden) in values, f"{hidden} should be a secret"
    for published in ("a@b.co", "user@example..com", "user@example.c0m"):
        assert LG.normalize_tight(published) not in values, (
            f"{published} is in the task statement — it is a given, not a secret"
        )


@pytest.mark.parametrize("task_id", sorted(_CASES))
def test_c0dbc720_post_fix_text_passes(task_id):
    """The FIXED text must pass. Without this the pre-fix assertions prove nothing: a gate that
    rejects every string trivially rejects a leak too."""
    ledger = LG.build_ledger(task_id)
    verdict = LG.check_text(_CASES[task_id]["post_fix"], [ledger])
    assert verdict.passed, f"c0dbc720's FIXED {task_id} text was rejected: {verdict.summary()}"


def test_post_fix_worked_examples_survive_only_via_the_construct_your_own_framing():
    """c32's fix kept three published examples and added "construct your own" framing.

    The lint downgrades that span to a warning rather than clearing it silently — the framing
    exemption is trivially satisfiable by an author, so it is recorded, not forgotten.
    """
    ledger = LG.build_ledger("c32")
    verdict = LG.check_text(_CASES["c32"]["post_fix"], [ledger])
    assert verdict.passed
    assert any(f.layer == LG.LAYER_WORKED_EXAMPLE for f in verdict.warnings)


def test_mead_inside_res_mead_is_caught_where_a_word_boundary_regex_cannot():
    """Task 146's documented blind spot: ``\\bmead\\b`` does not match inside ``res_mead``.

    That is not a hypothetical — 146's own leak test carries a comment about it, and its plan's
    leaf ids used to be ``res_mead``/``res_manic``/``res_williston``, spelling out three of the
    four hop-1 answers. Normalized SUBSTRING matching is the fix, and it is the reason this
    module never uses a word-boundary regex anywhere.
    """
    leaky = "Leaf ids for this plan: res_mead, area_hoover, res_manic."
    assert re.search(r"\bmead\b", leaky, re.I) is None, "the blind spot itself"

    ledger = LG.build_ledger("146")
    verdict = LG.check_text(leaky, [ledger])
    assert not verdict.passed
    assert {f.evidence for f in verdict.rejections} >= {"mead", "manic"}


@pytest.mark.parametrize("task_id", ["062", "077", "084", "091", "146", "c30", "c32"])
def test_generalized_advice_passes_every_ledger(task_id):
    """The positive control: real generalized advice must survive every task's ledger.

    A noisy ledger is worse than a small one — it makes the READ-time gate drop every note, so
    the whole mechanism silently disables itself while still reporting "enabled".
    """
    verdict = LG.check_text(GENERALIZED_ADVICE, [LG.build_ledger(task_id)])
    assert verdict.passed, f"generalized advice rejected against {task_id}: {verdict.summary()}"


@pytest.mark.skipif(shutil.which("git") is None, reason="git not available")
def test_fixture_matches_git_history():
    """Re-derive the vendored pre-fix text from git, so the fixture cannot rot into fiction."""
    for task_id, case in sorted(_CASES.items()):
        try:
            raw = subprocess.run(
                ["git", "show", f"c0dbc720^:{case['path']}"],
                cwd=_REPO_ROOT, capture_output=True, text=True, timeout=30, check=True,
            ).stdout
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pytest.skip("commit c0dbc720^ is not in this checkout (shallow clone?)")
        namespace: dict = {}
        exec(compile(raw, case["path"], "exec"), namespace)  # noqa: S102 — a repo-owned blob
        plan = namespace["get_compiled_plan"]()
        text = "\n".join(str(leaf.get("instruction", "")) for leaf in plan["leaves"])
        text += "\n" + plan["aggregation"]
        assert text == case["pre_fix"], f"{task_id} fixture has drifted from git history"


# --------------------------------------------------------------------------------------
# (a) the ledger
# --------------------------------------------------------------------------------------


def test_wiki_ledger_harvests_answers_and_decoys_but_not_public_names():
    """084 publishes all six lake NAMES in its statement and hides all six DEPTHS."""
    ledger = LG.build_ledger("084")
    assert ledger.kind == LG.KIND_WIKI
    values = {e.value for e in ledger.entries}
    assert {"836", "594", "590", "514", "511", "425"} <= values, "every depth is ground truth"
    assert "craterlake" not in values, "the statement lists the candidates — they are givens"
    assert any("winning entity" in f.lower() for f in ledger.facts), (
        "the winner is a RELATION, invisible to substring matching when every name is public — "
        "it must reach the LLM auditor as a fact"
    )


def test_wiki_ledger_reads_answer_tokens_out_of_validator_regexes():
    """A wiki task's answer tokens live in its validator regexes; ``\\bmead\\b`` -> ``mead``,
    not ``bmeadb`` (the escape class's letter is syntax, not content)."""
    values = {e.value for e in LG.build_ledger("146").entries}
    assert {"mead", "smallwood", "williston", "manicouagan", "manic"} <= values
    assert "bmeadb" not in values
    assert "wiki" not in values, "slug regexes are scaffolding, not ground truth"


def test_code_ledger_covers_hidden_test_names_inputs_and_keystone_assertions():
    ledger = LG.build_ledger("c30")
    assert ledger.kind == LG.KIND_CODE
    values = {e.value for e in ledger.entries}
    assert "testmiddlechangefindsrealcommonsubsequence" in values
    assert "a x c d f" in values
    assert any("op_counts(ops) == (3, 2, 2)" in f for f in ledger.facts), (
        "the graded assertion must reach the auditor: its numbers are too short to be entries"
    )


def test_ledger_lookup_by_id_and_by_path_agree():
    by_id = LG.build_ledger("084")
    path = LG.find_task_file("084")
    by_path = LG.build_ledger(path)
    assert {e.value for e in by_id.entries} == {e.value for e in by_path.entries}
    assert LG.find_task_file("nope") is None


def test_unknown_task_raises_rather_than_returning_an_empty_ledger():
    """An empty ledger would silently certify anything as clean."""
    with pytest.raises(LG.LeakGateError):
        LG.build_ledger("definitely-not-a-task")


def test_build_ledgers_skips_what_it_cannot_load():
    assert [l.task_id for l in LG.build_ledgers(["084", "nope"])] == ["084"]


# --------------------------------------------------------------------------------------
# (b) normalization / overlap
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("res_mead", "resmead"),          # underscore is a separator -> contains "mead"
        ("6,527", "6527"),                # thousands separator
        ("Atatürk", "ataturk"),           # accents fold
        ("Lake O'Higgins / San Martín", "lakeo'higgins/sanmartin"),
    ],
)
def test_tight_normalization(raw, expected):
    assert LG.normalize_tight(raw) == expected


def test_tight_normalization_keeps_punctuation_that_distinguishes_cases():
    """``user@example..com`` (published) and ``user@examplecom`` (hidden) are DIFFERENT test
    cases; a normalization that strips all punctuation collapses them into one and silently
    un-protects the hidden one."""
    assert LG.normalize_tight("user@example..com") != LG.normalize_tight("user@examplecom")


def test_sequence_entries_match_consecutive_runs_not_glued_prose():
    """``["delete","a"]`` must not match "emitting 'delete' (advance old only)".

    Gluing every character together would; matching whole runs in order does not — while still
    matching the same list written as ``old=['delete','a']``.
    """
    ledger = LG.build_ledger("c30")
    prose = "emitting 'delete' (advance old only) or 'insert' (advance new only)"
    assert not LG.overlap_findings(prose, ledger)
    assert LG.overlap_findings("ops = [['delete','a'],['delete','b'],['delete','c']]", ledger)


# --------------------------------------------------------------------------------------
# (d) the worked-example lint
# --------------------------------------------------------------------------------------


def test_worked_example_needs_both_a_cue_and_a_literal():
    assert not LG.worked_example_findings("Compare the values you read, not their fame.")
    assert not LG.worked_example_findings("For example, be careful with units.")  # no literal
    findings = LG.worked_example_findings("For example, 'abc' -> 'xyz'.")
    assert [f.severity for f in findings] == [LG.SEVERITY_REJECT]


def test_worked_example_framing_downgrades_only_when_no_ledger_token_is_present():
    ledger = LG.build_ledger("c32")
    framed_clean = "Check the examples above ('a@b.co' -> True), then build several of your own."
    assert all(f.severity == LG.SEVERITY_WARN for f in LG.worked_example_findings(framed_clean, [ledger]))

    framed_leaky = "Check e.g. 'user@examplecom' -> False, then build several of your own."
    assert any(
        f.severity == LG.SEVERITY_REJECT
        for f in LG.worked_example_findings(framed_leaky, [ledger])
    ), "framing must not launder an example that prints a secret next to it"


# --------------------------------------------------------------------------------------
# (c) the LLM auditor
# --------------------------------------------------------------------------------------


class _StubIO:
    """An ``AgentIO``-shaped stub: canned completion, and it counts calls."""

    def __init__(self, reply: str = '{"leaks": false}'):
        self.reply = reply
        self.calls = []

    def build_llm_payload(self, **kwargs):
        return dict(kwargs)

    async def query_llm_with_fallback(self, payload, **kwargs):
        self.calls.append(payload)
        return self.reply


def test_auditor_sends_exactly_one_batched_call_for_every_fact():
    ledgers = [LG.build_ledger("084"), LG.build_ledger("091")]
    io = _StubIO()
    verdict = asyncio.run(LG.llm_audit(GENERALIZED_ADVICE, ledgers, io))
    assert verdict.passed and verdict.audited
    assert len(io.calls) == 1, "one batched call, like plan_library's slot fill"
    user = io.calls[0]["messages"][-1]["content"]
    assert "PROTECTED FACTS" in user and "1." in user


def test_auditor_catches_a_paraphrase_the_deterministic_layers_cannot():
    """c30's "the common lines are a, c, d -- 3 'equal' ops" restates a hidden assertion whose
    numbers are too short to be ledger entries. Layer (c) is the layer that sees it."""
    paraphrase = "Remember that for a five-line middle-change pair exactly three lines stay equal."
    ledger = LG.build_ledger("c30")
    assert LG.check_text(paraphrase, [ledger]).passed, "deterministic layers cannot see this"

    io = _StubIO('{"leaks": true, "fact_numbers": [1], "reason": "restates the graded op count"}')
    verdict = asyncio.run(LG.llm_audit(paraphrase, [ledger], io))
    assert not verdict.passed and verdict.audited
    assert verdict.rejections[0].layer == LG.LAYER_LLM_AUDIT


@pytest.mark.parametrize("io", [None, _StubIO("not json at all")])
def test_auditor_reports_not_audited_rather_than_a_false_clean(io):
    verdict = asyncio.run(LG.llm_audit("anything", [LG.build_ledger("084")], io))
    assert verdict.passed and not verdict.audited


def test_write_gate_treats_an_unavailable_auditor_as_a_rejection():
    """A corpus must never claim an audit it never had."""
    ledgers = [LG.build_ledger("084")]
    strict = asyncio.run(LG.audit_text(GENERALIZED_ADVICE, ledgers, None))
    assert not strict.passed and not strict.audited
    assert strict.rejections[0].layer == LG.LAYER_LLM_AUDIT

    lenient = asyncio.run(
        LG.audit_text(GENERALIZED_ADVICE, ledgers, None, require_llm_audit=False)
    )
    assert lenient.passed and not lenient.audited


def test_write_gate_runs_all_four_layers():
    io = _StubIO()
    verdict = asyncio.run(
        LG.audit_text(_CASES["c32"]["pre_fix"], [LG.build_ledger("c32")], io)
    )
    assert not verdict.passed and verdict.audited
    assert {f.layer for f in verdict.rejections} >= {LG.LAYER_OVERLAP, LG.LAYER_WORKED_EXAMPLE}
