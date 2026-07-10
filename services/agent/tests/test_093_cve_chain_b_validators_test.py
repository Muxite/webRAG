"""
Offline unit tests for the CVE advisory -> C source root-cause chain (test 093) — free.

Cover the source-only keystone gate (the exact flag variable `socks5_resolve_local`), the
UN-gated cross-domain coverage diagnostic (function/file/threshold/overflow/fix-version,
retained even when the exact identifier is wrong), the keystone-gated citation secondary, the
correct answer in both single- and multi-line layout, and the adversarial failure modes the
task is built to expose:
  * a plausible-but-wrong flag identifier (a summary-only paraphrase like `resolve_locally`) ->
    keystone gate 0, coverage retained, citations gated 0;
  * a confident from-memory answer that names the CVE + function but not the exact variable ->
    keystone 0 while coverage still credits the source-level facts;
  * partial coverage (only some supporting facts) -> exact fraction;
  * the no-visits floor.
Plus the compiled plan is a well-formed 2-leaf dependent chain (advisory -> source) whose
dependent leaf templates the upstream hop and which leaks NO answer.
"""
from agent.app.idea_tests import test_093_tier5_cve_chain_b as t
from agent.app.testing import compiled_plan as cp


def _r(text):
    return {"output": {"final_deliverable": text}}


_OBS = {"visit": {"count": 2}}


_FULL_SINGLE = (
    "Root cause of CVE-2023-38545: in lib/socks.c the function do_SOCKS5 declares a boolean "
    "local flag `socks5_resolve_local` that decides whether the SOCKS5 proxy resolves the host "
    "name. During a slow handshake it can hold the wrong value, so a host name longer than the "
    "255-byte RFC1928 limit is copied with memcpy into a fixed heap buffer, overflowing it. "
    "Fixed in curl 8.4.0. Sources: "
    "https://raw.githubusercontent.com/curl/curl/curl-8_3_0/lib/socks.c and "
    "https://curl.se/docs/CVE-2023-38545.html"
)

_FULL_MULTI = (
    "CVE-2023-38545 — curl SOCKS5 heap overflow.\n"
    "Flag variable:\n"
    "  socks5_resolve_local\n"
    "Function / file:\n"
    "  do_SOCKS5 in socks.c\n"
    "Threshold:\n"
    "  255\n"
    "  bytes (max SOCKS5 domain-name length).\n"
    "Overflow primitive: memcpy (heap buffer overflow).\n"
    "Fixed in: 8.4.0\n"
    "Refs: https://curl.se/docs/CVE-2023-38545.html\n"
    "      https://raw.githubusercontent.com/curl/curl/curl-8_3_0/lib/socks.c\n"
)


def test_full_answer_single_line_scores_all():
    r = _r(_FULL_SINGLE)
    assert t.validate_keystone_variable(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0
    assert t.validate_visits(r, _OBS)["score"] == 2 / 3


def test_full_answer_multi_line_scores_all():
    # Same answer in a newline-heavy report layout: keystone token, all five coverage facts and
    # the citations must still register across line breaks.
    r = _r(_FULL_MULTI)
    assert t.validate_keystone_variable(r, _OBS)["score"] == 1.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 1.0


def test_wrong_flag_identifier_gates_to_zero_but_keeps_coverage():
    # Summary-only slip: the agent read the advisory paraphrase and invented a plausible identifier
    # (`resolve_locally`) instead of reading the source for the exact `socks5_resolve_local`.
    # Keystone must gate 0; the un-gated coverage still credits the source-level facts it did get;
    # the gated citation zeroes out.
    wrong = _FULL_SINGLE.replace("socks5_resolve_local", "resolve_locally")
    r = _r(wrong)
    assert t.validate_keystone_variable(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0


def test_from_memory_names_function_not_variable_gates_zero():
    # A confident parametric answer: correct CVE, function and file, but never the exact flag
    # identifier (which is source-only). Keystone 0, citations gated 0; coverage still high
    # because the function/file/threshold/overflow/fix are all named.
    guess = (
        "From memory: CVE-2023-38545 is the curl SOCKS5 heap overflow in do_SOCKS5 (socks.c). "
        "When the host name exceeds 255 bytes a state-machine flag is mishandled and memcpy "
        "overflows the buffer. Fixed in 8.4.0. https://curl.se/docs/CVE-2023-38545.html"
    )
    r = _r(guess)
    assert t.validate_keystone_variable(r, _OBS)["score"] == 0.0
    assert t.validate_citations(r, _OBS)["score"] == 0.0
    assert t.validate_coverage(r, _OBS)["score"] == 1.0


def test_partial_coverage_scores_fraction():
    # Only two of the five supporting facts present (function + file), and no exact variable.
    text = (
        "The bug lives in do_SOCKS5 in socks.c, but I could not determine the exact flag name, "
        "the length threshold or the fix version."
    )
    r = _r(text)
    assert abs(t.validate_coverage(r, _OBS)["score"] - 0.4) < 1e-9
    assert t.validate_keystone_variable(r, _OBS)["score"] == 0.0


def test_no_visits_scores_fraction_and_gate():
    r = _r(_FULL_SINGLE)
    assert t.validate_visits(r, {"visit": {"count": 3}})["score"] == 1.0
    assert t.validate_visits(r, {"visit": {"count": 2}})["passed"] is True
    assert t.validate_visits(r, {"visit": {"count": 0}})["score"] == 0.0
    assert t.validate_visits(r, {"visit": {"count": 0}})["passed"] is False


def test_keystone_case_insensitive_but_exact_token():
    # Exact identifier, any case, wins; a near-miss token (extra/missing segment) does not.
    assert t.validate_keystone_variable(_r("SOCKS5_RESOLVE_LOCAL"), _OBS)["passed"] is True
    assert t.validate_keystone_variable(_r("socks_resolve_local"), _OBS)["passed"] is False
    assert t.validate_keystone_variable(_r("socks5_local_resolve"), _OBS)["passed"] is False


def test_compiled_plan_validates_and_is_a_chain_dag():
    plan = t.get_compiled_plan()
    cp.validate_plan(plan)  # must not raise (well-formed, acyclic, deps resolve)
    struct = cp.plan_structure(plan)
    assert struct["leaf_count"] == 2
    assert struct["edge_count"] == 1
    assert struct["waves"] == [["advisory_context"], ["source_flag_identifier"]]
    assert struct["edges"] == ["advisory_context->source_flag_identifier"]
    assert struct["is_dag_chain"] is True
    # The dependent leaf templates its upstream hop (a real edge, not duplicated text).
    by_id = {l["id"]: l for l in plan["leaves"]}
    assert "{advisory_context}" in by_id["source_flag_identifier"]["instruction"]
    # Each leaf self-describes what it is answering (guards against aggregation binding loss).
    assert "answers" in by_id["advisory_context"]["instruction"].lower()
    assert "answers" in by_id["source_flag_identifier"]["instruction"].lower()


def test_compiled_plan_leaks_nothing():
    plan = t.get_compiled_plan()
    blob = " ".join(str(l) for l in plan["leaves"]).lower() + " " + plan["aggregation"].lower()
    # STRUCTURE only: names the GIVEN CVE + seeded URLs but no answer — not the keystone variable,
    # the function, the byte threshold, the overflow primitive or the fix version.
    for leak in ("socks5_resolve_local", "do_socks5", "curl_socks5", "8.4.0", "255", "memcpy"):
        assert leak not in blob, f"plan leaks {leak!r}"
