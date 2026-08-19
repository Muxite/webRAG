"""The factor ladder: one decision, rendered as each prompt shape.

PRIMARY (pre-registered) -- answer position and verbosity.

    A0  answer only
    A1  answer, then one sentence of justification   <-- the engine's convention
    A2  one sentence of reasoning, then the answer
    A3  think step by step, then the answer
    A4  reason in <=40 words, then the answer
    SHIPPED  the engine's literal instruction text, resolved from settings

CALIBRATION -- the same question, asked of a confidence number.

    C_A1  confidence, then reason          <-- the engine's {confidence, reason} order
    C_A2  reason, then confidence
    C_verbal  a verbal band, mapped to a number by a table declared in calibration.py
    C_expected  expected, then observed, then confidence

``C_expected`` is P2 from ``CONFIDENCE_JUDGE_MISCALIBRATION.md`` §4 in its cheap
one-call form: require the expectation to be written BEFORE the content is scored,
so it cannot be written backwards from the content. That document's diagnosis is
that the judge rates presence-of-plausible-text -- 65% of search steps score >=0.8
for +0.004 lift -- and every sampled reason keys on "clearly states" / "directly
matches".

SECONDARY (Holm-corrected)

    F_json         output format     -- bare token vs strict JSON (A1 body held fixed)
    G_nostatement  goal restatement  -- validity control: withhold the task statement
    G_noevidence   evidence          -- validity control: withhold the evidence block

The G arms are controls, not hypotheses. If withholding the thing the item is
supposed to require does NOT collapse accuracy toward chance, the items were
answerable from model priors and every other number on that family is suspect.
v1's ``G_nostatement`` worked exactly this way: -0.421 on ``qwen2.5:7b``,
Holm p = 0.007.

Every shape asks the SAME question about the SAME evidence. Only the
answer-position and verbosity instruction changes, so a difference in accuracy is
attributable to shape rather than to content.

The SHIPPED arm is resolved through ``families.shipped_prompt`` -- settings first,
exactly as the engine resolves it -- rather than retyped, so an arm cannot silently
drift from what the engine sends. ``promptbench_shipped_parity_test.py`` fails if
that resolution stops working or the sources diverge.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from agent.app.promptbench.availability import PromptContext

PRIMARY_VARIANTS = ("A0", "A1", "A2", "A3", "A4", "SHIPPED")
CALIBRATION_VARIANTS = ("C_A1", "C_A2", "C_verbal", "C_expected")
SECONDARY_VARIANTS = ("F_json", "G_nostatement", "G_noevidence")
ALL_VARIANTS = PRIMARY_VARIANTS + CALIBRATION_VARIANTS + SECONDARY_VARIANTS

# Shape instructions. Only the answer-position and verbosity instruction changes.

_SHAPES: Dict[str, str] = {
    "A0": "Reply with exactly one of the options above and nothing else.",
    # The engine's convention: commit to the answer, then rationalise it.
    "A1": "Reply with your answer first, then one sentence of justification.",
    "A2": "Give one sentence of reasoning, then state your answer on the final line.",
    "A3": "Think step by step, then state your answer on the final line.",
    "A4": "Reason in at most 40 words, then state your answer on the final line.",
}

_CALIBRATION_SHAPES: Dict[str, str] = {
    # The engine's literal order: the number, then the prose.
    "C_A1": (
        'Reply with strict JSON and nothing else: {"answer": "<one of the options above>", '
        '"confidence": <number between 0 and 1>, "reason": "<one sentence>"}'
    ),
    "C_A2": (
        'Reply with strict JSON and nothing else: {"reason": "<one sentence>", '
        '"answer": "<one of the options above>", "confidence": <number between 0 and 1>}'
    ),
    "C_verbal": (
        'Reply with strict JSON and nothing else: {"answer": "<one of the options above>", '
        '"certainty": "<one of: certain, likely, unsure, guessing>"}'
    ),
    # Expectation before observation, so it cannot be written backwards from the content.
    "C_expected": (
        'Reply with strict JSON and nothing else: {"expected": "<what a correct answer '
        'would have to be supported by, stated before you look>", "observed": "<what the '
        'material above actually supports>", "answer": "<one of the options above>", '
        '"confidence": <number between 0 and 1>}'
    ),
}


def shipped_instruction(family: str = "verify") -> str:
    """The engine's own instruction for this decision point, resolved not retyped."""
    from agent.app.promptbench.families import REGISTRY

    return REGISTRY[family].shipped()


# ---------------------------------------------------------------------------
# Bodies -- one per family. Each renders the SAME question under every shape.
# ---------------------------------------------------------------------------

def _statement_block(runtime: Mapping[str, Any], include: bool) -> str:
    if include:
        return f"TASK CONTEXT:\n{runtime['statement']}"
    return "TASK CONTEXT:\n(withheld)"


def _verify_body(runtime, *, include_statement=True, include_evidence=True) -> str:
    return "\n\n".join([
        _statement_block(runtime, include_statement),
        f"CANDIDATE:\n{runtime['candidate']}",
        "QUESTION: Does this candidate satisfy the requirement stated in the task?",
        "OPTIONS: SATISFIES | VIOLATES",
    ])


def _select_body(runtime, *, include_statement=True, include_evidence=True) -> str:
    return "\n\n".join([
        _statement_block(runtime, include_statement),
        f"CANDIDATES:\n{runtime['candidates']}",
        "QUESTION: Which ONE candidate satisfies the requirement stated in the task?",
        "OPTIONS: " + " | ".join(runtime["choices"]),
    ])


def _keystone_body(runtime, *, include_statement=True, include_evidence=True) -> str:
    evidence = runtime["evidence"] if include_evidence else "(withheld)"
    return "\n\n".join([
        _statement_block(runtime, include_statement),
        f"EVIDENCE:\n{evidence}",
        f"CLAIM:\n{runtime['claim']}",
        "QUESTION: Is the claim supported by the evidence above? "
        "Rely only on the evidence, never on prior knowledge.",
        "OPTIONS: TRUE | FALSE",
    ])


def _followup_body(runtime, *, include_statement=True, include_evidence=True) -> str:
    return "\n\n".join([
        _statement_block(runtime, include_statement),
        f"SUB-TASKS THAT ALREADY EXIST:\n{runtime['siblings']}",
        f"SUB-TASK JUST COMPLETED:\n{runtime['completed_task']}\n"
        f"RESULT: {runtime['completed_result']}",
        "QUESTION: Is a further sub-task required to satisfy the task, one that is not "
        "already covered by the sub-tasks listed above?",
        "OPTIONS: YES | NO",
    ])


def _goal_body(runtime, *, include_statement=True, include_evidence=True) -> str:
    return "\n\n".join([
        _statement_block(runtime, include_statement),
        f"SYNTHESIS PRODUCED SO FAR:\n{runtime['synthesis']}",
        "QUESTION: Has the task's goal been achieved by this synthesis?",
        "OPTIONS: ACHIEVED | NOT_ACHIEVED",
    ])


_BODIES = {
    "verify": _verify_body,
    "select": _select_body,
    "keystone_claim": _keystone_body,
    "followup": _followup_body,
    "goal_achieved": _goal_body,
    # Same items and same question as verify; the arms differ, not the body.
    "calibration": _verify_body,
}

# Which arms are coherent for which family.
#
# ``select`` has no SHIPPED arm: a four-way truth verdict cannot name one of five
# candidates. v1 ran those cells and discarded every one in analysis; here they are
# never placed. ``G_noevidence`` exists only where there is an evidence block to
# withhold, and the C arms only on the family whose metric reads a confidence.
_A_ARMS = ("A0", "A1", "A2", "A3", "A4")
_APPLICABLE: Dict[str, Tuple[str, ...]] = {
    "verify": _A_ARMS + ("SHIPPED", "F_json", "G_nostatement"),
    "select": _A_ARMS + ("F_json", "G_nostatement"),
    "keystone_claim": _A_ARMS + ("SHIPPED", "F_json", "G_nostatement", "G_noevidence"),
    "followup": _A_ARMS + ("SHIPPED", "F_json", "G_nostatement"),
    "goal_achieved": _A_ARMS + ("SHIPPED", "F_json", "G_nostatement"),
    "calibration": CALIBRATION_VARIANTS + ("SHIPPED",),
}


def applicable_variants(family: str) -> Tuple[str, ...]:
    return _APPLICABLE[family]


def is_applicable(family: str, variant: str) -> bool:
    return variant in _APPLICABLE.get(family, ())


def build_prompt(runtime: Mapping[str, Any], ctx: PromptContext) -> str:
    """Render one item under one variant.

    Signature is deliberately ``(runtime, ctx)``: ``ctx`` is the frozen
    PromptContext, which carries no label and no item, so ground truth cannot
    reach this function without going outside the signature on purpose.
    """
    body_fn = _BODIES[ctx.family]
    variant = ctx.variant

    if variant == "SHIPPED":
        return f"{shipped_instruction(ctx.family)}\n\n{body_fn(runtime)}"

    if variant == "F_json":
        instruction = (
            'Reply with strict JSON and nothing else: '
            '{"answer": "<one of the options above>", "reason": "<one sentence>"}'
        )
        return f"{body_fn(runtime)}\n\n{instruction}"

    if variant == "G_nostatement":
        return f"{body_fn(runtime, include_statement=False)}\n\n{_SHAPES['A1']}"

    if variant == "G_noevidence":
        return f"{body_fn(runtime, include_evidence=False)}\n\n{_SHAPES['A1']}"

    if variant in _CALIBRATION_SHAPES:
        return f"{body_fn(runtime)}\n\n{_CALIBRATION_SHAPES[variant]}"

    return f"{body_fn(runtime)}\n\n{_SHAPES[variant]}"


def choices_for(family: str, runtime: Mapping[str, Any]) -> List[str]:
    return list(runtime["choices"])
