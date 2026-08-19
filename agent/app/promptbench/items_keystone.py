"""The ``keystone_claim`` family: check a datum against supplied evidence.

This is the family that most closely mirrors what ``VerifyLeafAction`` does at
runtime -- a claim, a block of retrieved text, and a verdict about whether the
text supports the claim -- on items where the right answer is known by
construction rather than by hand-labelling.

WHERE THE GROUND TRUTH COMES FROM
---------------------------------
Each source task module already owns a compiled ``KEYSTONE_RX``: the regex its
own validator runs to decide whether a live agent recovered the hard datum. That
regex is reused here as the item's **falsity oracle**. A claim is true when the
pattern matches it and false when it does not, so no judgement enters the label.

THE TRAP THIS FAMILY HAD TO AVOID
---------------------------------
The obvious construction -- take a two-alternative pattern and call alternative
#1 true and #2 false -- is wrong::

    KEYSTONE_RX = re.compile(r"\\b565\\b|\\b165\\b")   # Garabit viaduct

565 ft *is* 165 m. The alternation spells one value in two units, and four of the
fifteen keystone modules have that shape. Reading it as a right/wrong pair builds
a family whose entire negative population is true.

So the false twin is **generated** and then **proved** false:

1. perturb the digits of a true literal,
2. reject the result if ``KEYSTONE_RX`` still matches it,
3. reject it if the perturbed string appears anywhere in the evidence -- a
   number printed in the walkthrough for some other reason is a datum a careful
   reader could legitimately confirm, which would make the item's own label wrong.

Only a candidate surviving all three becomes an item. If a module yields none, it
is dropped and the census says so.

LEAK GUARD
----------
Deliberately not the one in ``items.py``. That guard rejects *asymmetric candidate
mention*, which is the contaminating shape for a selection item and is meaningless
here. This family's contaminating shape is the statement stating the answer, so
the guard is ``KEYSTONE_RX.search(statement)``.

The module docstring is used as evidence here and would be poison in ``select``,
where the same text annotates the answer outright. Which text leaks depends on
what the family measures, not on the text.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from agent.app.promptbench.availability import Item, Label

# The engine's own verify vocabulary. Unlike ``verify``, which asks
# SATISFIES/VIOLATES and therefore needed a post-hoc alias table before its
# SHIPPED arm could be graded at all, this family speaks the language the shipped
# prompt already asks for, so SHIPPED needs no re-grading to be coherent.
CLAIM_CHOICES = ("TRUE", "FALSE")

EVIDENCE_CHARS = 2600
STATEMENT_CHARS = 700

# Authoring metadata, not evidence: the banner every task module opens with.
_HEADER_RX = re.compile(
    r"^[ \t]*(?:Test\s+\d+\s*:.*|Level:.*Difficulty:.*)$", re.IGNORECASE | re.MULTILINE
)
# "[KEYSTONE]" marks the payoff line for a human reader. Stripping the bare marker
# keeps the evidence reading as prose; the "= <value>" form is rewritten rather
# than deleted, because the value inside it is the evidence.
_KEYSTONE_MARKER_RX = re.compile(r"\[\s*KEYSTONE\s*=\s*([^\]]+)\]", re.IGNORECASE)
_KEYSTONE_BARE_RX = re.compile(r"\[\s*KEYSTONE[^\]]*\]", re.IGNORECASE)


def _compiled(spec: Dict[str, Any]) -> Optional[re.Pattern]:
    pattern = spec.get("keystone_pattern") or ""
    if not pattern:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def _clean_evidence(raw: str) -> str:
    text = _KEYSTONE_MARKER_RX.sub(r"(\1)", raw)
    text = _KEYSTONE_BARE_RX.sub("", text)
    text = _HEADER_RX.sub("", text)
    return " ".join(text.split())


def _evidence_window(text: str, rx: re.Pattern) -> str:
    """Trim to a window that still contains the keystone.

    Truncating from the front is not safe here: several walkthroughs state the
    datum in their final paragraph, and an evidence block missing the datum makes
    the TRUE item unanswerable while leaving the FALSE item answerable -- a
    polarity-dependent difficulty gap, which is exactly the asymmetry that would
    manufacture an effect.
    """
    if len(text) <= EVIDENCE_CHARS:
        return text
    m = rx.search(text)
    if not m:
        return text[:EVIDENCE_CHARS]
    centre = (m.start() + m.end()) // 2
    half = EVIDENCE_CHARS // 2
    start = max(0, centre - half)
    return text[start:start + EVIDENCE_CHARS]


def _perturbations(literal: str) -> Iterator[str]:
    """Candidate false twins, cheapest and least suspicious first.

    Every perturbation keeps the literal's shape -- same digit count, same unit
    suffix -- so the false claim is not identifiable by looking at it. A twin that
    reads as obviously malformed would be rejected on form rather than on evidence,
    and the family would measure formatting sense instead of verification.
    """
    digits = [(i, c) for i, c in enumerate(literal) if c.isdigit()]
    if not digits:
        return

    # Adjacent transposition, right to left: 29,002 -> 29,020.
    for k in range(len(digits) - 1, 0, -1):
        (i, a), (j, b) = digits[k - 1], digits[k]
        if a == b:
            continue
        chars = list(literal)
        chars[i], chars[j] = b, a
        yield "".join(chars)

    # Single-digit substitution, least significant first: 763 -> 768.
    for i, c in reversed(digits):
        for delta in (1, 2, 3, 5):
            chars = list(literal)
            chars[i] = str((int(c) + delta) % 10)
            candidate = "".join(chars)
            if candidate != literal and not candidate.lstrip().startswith("0"):
                yield candidate


def _false_twin(literal: str, rx: re.Pattern, evidence: str) -> Optional[str]:
    """A perturbation proved false, or ``None`` if the module cannot supply one."""
    for candidate in _perturbations(literal):
        if rx.search(candidate):
            continue                       # still matches the oracle -- still true
        if re.search(rf"(?<!\d){re.escape(candidate)}(?!\d)", evidence):
            continue                       # printed in the evidence for another reason
        return candidate
    return None


def _canonical_literal(spec: Dict[str, Any]) -> str:
    """The literal the true claim is built from.

    Longest-then-first, so a unit-bearing alternative ("258 kg") is preferred over
    a bare number. A claim carrying its unit is harder to confirm by coincidence.
    """
    return sorted(spec["keystone_literals"], key=lambda s: (-len(s), s))[0]


def _pick(options: Sequence[str], seed: str) -> str:
    digest = hashlib.sha256(seed.encode()).hexdigest()
    return options[int(digest, 16) % len(options)]


def usable(spec: Dict[str, Any]) -> Tuple[bool, str]:
    """Whether a spec can produce a balanced claim pair, and why not if it cannot."""
    if not spec.get("keystone_literals"):
        return False, "no digit-bearing keystone literal"
    rx = _compiled(spec)
    if rx is None:
        return False, "keystone pattern does not compile"
    if spec.get("statement_leaks_keystone"):
        return False, "statement states the keystone"
    evidence = _clean_evidence(spec.get("evidence") or "")
    if not evidence:
        return False, "no evidence text"
    if not rx.search(evidence):
        return False, "evidence does not contain the keystone"
    literal = _canonical_literal(spec)
    if _false_twin(literal, rx, evidence) is None:
        return False, "no perturbation could be proved false"
    return True, ""


def build_keystone_claim_items(specs: Sequence[Dict[str, Any]]) -> List[Item]:
    items: List[Item] = []
    for spec in specs:
        ok, _ = usable(spec)
        if not ok:
            continue
        rx = _compiled(spec)
        evidence = _evidence_window(_clean_evidence(spec["evidence"]), rx)
        literal = _canonical_literal(spec)
        twin = _false_twin(literal, rx, evidence)
        if twin is None:
            # The window is a subset of the text ``usable`` cleared, so a twin that
            # was provable against the whole docstring can fail against the window.
            continue
        statement = " ".join((spec.get("statement") or "").split())[:STATEMENT_CHARS]
        for value, truth in ((literal, "TRUE"), (twin, "FALSE")):
            items.append(Item(
                item_id=f"keystone-{spec['test_id'] or spec['module']}-{truth.lower()}",
                cluster=spec["module"],
                runtime={
                    "statement": statement,
                    "evidence": evidence,
                    "claim": f"The value this task asks for is {value}.",
                    "choices": list(CLAIM_CHOICES),
                },
                posthoc={"test_id": spec["test_id"], "polarity": truth, "value": value},
                label=Label(
                    value=truth,
                    derived_from=f"task_module.{spec['module']}.KEYSTONE_RX",
                ),
            ))
    return items


def census(specs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    dropped: Dict[str, List[str]] = {}
    for spec in specs:
        if not spec.get("keystone_literals"):
            continue                       # never a candidate; not a drop worth reporting
        ok, why = usable(spec)
        if not ok:
            dropped.setdefault(why, []).append(spec["module"])
    items = build_keystone_claim_items(specs)
    pos = sum(1 for i in items if i.posthoc["polarity"] == "TRUE")
    return {
        "keystone_specs_seen": sum(1 for s in specs if s.get("keystone_literals")),
        "dropped": dropped,
        "keystone_claim_items": len(items),
        "keystone_claim_clusters": len({i.cluster for i in items}),
        "keystone_claim_positive": pos,
        "keystone_claim_negative": len(items) - pos,
    }


def main() -> int:
    path = Path("agent/tests/fixtures/promptbench/task_specs.json")
    specs = json.loads(path.read_text())["specs"]
    print(json.dumps(census(specs), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
