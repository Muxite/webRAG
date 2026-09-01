"""One shared confidence/abstain channel for every execution arm.

Risk-coverage (selective prediction) needs a confidence signal from every arm it compares, but
today only ``evidence_loop`` reports one (``output.ledger_verdict``). ``sequential_react_extract``
writes only ``success = bool(deliverable)`` and ``langgraph_react`` emits no verdict at all — so a
claim that the evidence-first arm is "better calibrated" was partly an artifact of being the only
arm that reports confidence. This module levels that: every arm gets a same-shaped
``{"confidence": "ANSWER" | "PARTIAL" | "ABSTAIN", "confidence_basis": {"source", "detail"}}``
pair, computed from state that arm already has — no new LLM call.

Each function below reads ONLY the one arm's own state:

* :func:`from_evidence_loop_verdict` reads ``evidence_loop``'s own ``Ledger.verdict()`` string.
  Free — the verdict already exists.
* :func:`from_sequential_extractions` reads ``sequential_react_extract``'s own
  ``extractions[].value_verified`` records. Free and deterministic: ANSWER when the verified
  fraction is at least the frozen ``graded_verdict.answer_min_support`` threshold, ABSTAIN when
  nothing verified, PARTIAL otherwise.
* :func:`from_langgraph_audit` is DIFFERENT IN KIND from the other two, and that difference is the
  honest framing of this whole comparison. ``langgraph_react`` emits no typed extractions and
  keeps no ledger, so it has no private state of its own to report a confidence tier from. Its
  channel is instead AUDIT-DERIVED: it runs the arm-blind auditor
  (``agent.app.testing.claim_audit.audit``) over the pages and final text this arm itself
  produced and stored, then grades the resulting ``support_rate`` against the same frozen
  threshold. The question this sets up is whether keeping a ledger while you run beats auditing
  your own artifacts afterwards — not whether one arm "has" confidence and the other does not.

None of these three functions is arm-blind in the way ``claim_audit`` is (each is scoped to one
arm's own state on purpose), but none of them reads a field another arm exclusively owns:
``from_evidence_loop_verdict`` never sees ``extractions`` or ``pages``, and
``from_sequential_extractions`` never sees a ``ledger``.

Every tolerance and threshold read here comes from the frozen ``scripts/ledger_kpi_spec.json``
(hash-checked by ``agent/tests/ledger_kpi_spec_frozen_test.py``) — never a literal in this module.

Absent is never zero: when an arm's own state cannot support a tier at all (no extractions were
ever produced, or the input is not the right shape), the function returns ``None`` rather than a
default tier, and the caller omits ``confidence`` / ``confidence_basis`` from ``output`` entirely.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

# `claim_audit` imports `evidence_graph`, which imports `execution_evidence_loop` (for
# QUOTE_FAIL_* and QuoteMatch), which imports THIS module to build its own channel -- so the
# import is deferred into each function body, exactly like `Ledger.ensure_graph` defers its own
# import of `evidence_graph` for the same reason. Nothing here is imported at module scope from
# either `claim_audit` or `evidence_graph`.

#: The three confidence tiers every arm's channel is graded into, in increasing order of trust.
ABSTAIN = "ABSTAIN"
PARTIAL = "PARTIAL"
ANSWER = "ANSWER"

_VALID_TIERS = (ABSTAIN, PARTIAL, ANSWER)

#: Longest ``confidence_basis.detail`` string, per the contract.
_DETAIL_CHARS = 200


def _channel(tier: str, source: str, detail: str) -> Dict[str, Any]:
    """Build the shared two-key payload every arm's channel returns."""
    return {
        "confidence": tier,
        "confidence_basis": {"source": source, "detail": str(detail)[:_DETAIL_CHARS]},
    }


def _answer_min_support() -> float:
    """The frozen ``graded_verdict.answer_min_support`` threshold, never a literal here."""
    from agent.app.testing.claim_audit import load_spec
    return float(load_spec()["graded_verdict"]["answer_min_support"])


def from_evidence_loop_verdict(ledger_verdict: Optional[str]) -> Optional[Dict[str, Any]]:
    """``evidence_loop``'s channel: a direct re-export of its own ``Ledger.verdict()``.

    :param ledger_verdict: the value of ``output["ledger_verdict"]`` — one of ``ANSWER`` /
        ``PARTIAL`` / ``ABSTAIN`` when ``run_evidence_loop_execution`` ran, else something falsy
        or malformed.
    :returns: the ``{"confidence", "confidence_basis"}`` pair, or ``None`` when ``ledger_verdict``
        is not one of the three known tiers (absent is never zero).
    :raises: nothing.
    """
    tier = str(ledger_verdict or "").strip().upper()
    if tier not in _VALID_TIERS:
        return None
    return _channel(tier, "ledger_verdict",
                    f"Ledger.verdict() (RESOLUTION over the run's ledger rows) returned {tier}.")


def from_sequential_extractions(
        extractions: Optional[Sequence[Mapping[str, Any]]]) -> Optional[Dict[str, Any]]:
    """``sequential_react_extract``'s channel, derived from its own ``extractions[].value_verified``.

    Rule: ANSWER when the verified fraction is at least the frozen
    ``graded_verdict.answer_min_support`` threshold (0.8 as of this writing), ABSTAIN when not one
    single extraction verified, PARTIAL otherwise. Measured monotone over the 198 stored
    ``ledgernum22r3`` cells: 0.242 / 0.544 / 0.678 (ABSTAIN / PARTIAL / ANSWER).

    :param extractions: ``output["extractions"]`` — a list of this arm's own extraction dicts, each
        carrying ``value_verified`` (``True`` / ``False`` / ``None``).
    :returns: the ``{"confidence", "confidence_basis"}`` pair, or ``None`` when ``extractions`` is
        not a usable sequence or is empty — an arm that extracted nothing has no basis to report a
        tier from, and reporting ABSTAIN there would conflate "we checked and found nothing" with
        "we never checked" (absent is never zero).
    :raises: nothing.
    """
    if not isinstance(extractions, Sequence) or isinstance(extractions, (str, bytes)):
        return None
    total = len(extractions)
    if total == 0:
        return None
    verified = sum(1 for item in extractions
                   if isinstance(item, Mapping) and item.get("value_verified") is True)
    ratio = verified / total
    threshold = _answer_min_support()
    if verified == 0:
        tier = ABSTAIN
    elif ratio >= threshold:
        tier = ANSWER
    else:
        tier = PARTIAL
    detail = (f"{verified}/{total} extractions had value_verified=True "
              f"({ratio:.3f}; ANSWER requires >= {threshold}).")
    return _channel(tier, "extractions_value_verified", detail)


def from_langgraph_audit(execution: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """``langgraph_react``'s channel: AUDIT-DERIVED, not native (see module docstring).

    ``langgraph_react`` keeps no ledger and emits no typed extractions, so this runs the
    arm-blind auditor (:func:`agent.app.testing.claim_audit.audit`) over this arm's OWN stored
    pages and final text and grades the resulting ``support_rate``.

    Rule: ABSTAIN when no page was stored or the answer carried no checkable claim (nothing to
    audit); ANSWER when ``support_rate`` is at least the frozen ``graded_verdict.answer_min_support``
    threshold; PARTIAL otherwise. Measured monotone over the 198 stored ``ledgernum22r3`` cells:
    0.062 / 0.642 / 0.671 (ABSTAIN / PARTIAL / ANSWER).

    :param execution: this arm's own execution-result shape, i.e. ``{"output": ..., "telemetry_raw":
        ...}`` (or a whole stored cell) — never another arm's fields, since :func:`claim_audit.audit`
        itself never reads an arm identifier or an arm-exclusive key.
    :returns: the ``{"confidence", "confidence_basis"}`` pair, or ``None`` when ``execution`` is not
        a mapping (absent is never zero).
    :raises: nothing — :func:`claim_audit.audit` never raises on a malformed cell.
    """
    if not isinstance(execution, Mapping):
        return None
    from agent.app.testing.claim_audit import audit as claim_audit
    record = claim_audit(execution)
    pages_stored = int(record.get("pages_stored") or 0)
    checkable = int(record.get("checkable_claims") or 0)
    support_rate = record.get("support_rate")
    threshold = _answer_min_support()
    if not pages_stored or not checkable or support_rate is None:
        return _channel(
            ABSTAIN, "claim_audit.support_rate",
            f"pages_stored={pages_stored}, checkable_claims={checkable}: no auditable evidence.")
    if support_rate >= threshold:
        tier = ANSWER
    else:
        tier = PARTIAL
    detail = (f"support_rate={support_rate:.3f} over {checkable} checkable claim(s) on "
              f"{pages_stored} stored page(s) (ANSWER requires >= {threshold}).")
    return _channel(tier, "claim_audit.support_rate", detail)
