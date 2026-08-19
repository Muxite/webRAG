"""The factor ladder: one decision, rendered as each prompt shape.

PRIMARY (pre-registered) -- answer position and verbosity.

    A0  answer only
    A1  answer, then one sentence of justification   <-- the engine's convention
    A2  one sentence of reasoning, then the answer
    A3  think step by step, then the answer
    A4  reason in <=40 words, then the answer
    SHIPPED  the engine's literal instruction text, imported from source

SECONDARY (Holm-corrected)

    F   output format     -- bare token vs strict JSON (A1 body held fixed)
    G   goal restatement  -- constraint only vs constraint + full task statement

Every shape asks the SAME question about the SAME evidence. Only the
answer-position and verbosity instruction changes, so a difference in accuracy
is attributable to shape rather than to content.

The SHIPPED arm imports ``VerifyLeafAction._DEFAULT_SYSTEM_PROMPT`` rather than
retyping it, so the arm cannot silently drift away from what the engine sends.
``promptbench_shipped_parity_test.py`` fails if that import stops resolving.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from agent.app.promptbench.availability import PromptContext

PRIMARY_VARIANTS = ("A0", "A1", "A2", "A3", "A4", "SHIPPED")
SECONDARY_VARIANTS = ("F_json", "G_nostatement")
ALL_VARIANTS = PRIMARY_VARIANTS + SECONDARY_VARIANTS

# ---------------------------------------------------------------------------
# Shape instructions. The ONLY thing that varies across the primary ladder.
# ---------------------------------------------------------------------------

_SHAPES: Dict[str, str] = {
    "A0": "Reply with exactly one of the options above and nothing else.",
    # The engine's convention: commit to the answer, then rationalise it.
    "A1": "Reply with your answer first, then one sentence of justification.",
    "A2": "Give one sentence of reasoning, then state your answer on the final line.",
    "A3": "Think step by step, then state your answer on the final line.",
    "A4": "Reason in at most 40 words, then state your answer on the final line.",
}


def shipped_instruction() -> str:
    """The engine's own verify instruction, imported rather than retyped."""
    from agent.app.idea_policies.actions import VerifyLeafAction

    return VerifyLeafAction._DEFAULT_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Family bodies
# ---------------------------------------------------------------------------

def _verify_body(runtime: Mapping[str, Any], *, include_statement: bool = True) -> str:
    parts = []
    if include_statement:
        parts.append(f"TASK CONTEXT:\n{runtime['statement']}")
    else:
        parts.append("TASK CONTEXT:\n(withheld)")
    parts.append(f"CANDIDATE:\n{runtime['candidate']}")
    parts.append("QUESTION: Does this candidate satisfy the requirement stated in the task?")
    parts.append("OPTIONS: SATISFIES | VIOLATES")
    return "\n\n".join(parts)


def _select_body(runtime: Mapping[str, Any], *, include_statement: bool = True) -> str:
    parts = []
    if include_statement:
        parts.append(f"TASK CONTEXT:\n{runtime['statement']}")
    else:
        parts.append("TASK CONTEXT:\n(withheld)")
    parts.append(f"CANDIDATES:\n{runtime['candidates']}")
    parts.append("QUESTION: Which ONE candidate satisfies the requirement stated in the task?")
    parts.append("OPTIONS: " + " | ".join(runtime["choices"]))
    return "\n\n".join(parts)


_BODIES = {"verify": _verify_body, "select": _select_body}


def build_prompt(runtime: Mapping[str, Any], ctx: PromptContext) -> str:
    """Render one item under one variant.

    Signature is deliberately ``(runtime, ctx)``: ``ctx`` is the frozen
    PromptContext, which carries no label and no item, so ground truth cannot
    reach this function without going outside the signature on purpose.
    """
    body_fn = _BODIES[ctx.family]
    variant = ctx.variant

    if variant == "SHIPPED":
        return f"{shipped_instruction()}\n\n{body_fn(runtime)}"

    if variant == "F_json":
        instruction = (
            'Reply with strict JSON and nothing else: '
            '{"answer": "<one of the options above>", "reason": "<one sentence>"}'
        )
        return f"{body_fn(runtime)}\n\n{instruction}"

    if variant == "G_nostatement":
        return f"{body_fn(runtime, include_statement=False)}\n\n{_SHAPES['A1']}"

    return f"{body_fn(runtime)}\n\n{_SHAPES[variant]}"


def choices_for(family: str, runtime: Mapping[str, Any]) -> List[str]:
    return list(runtime["choices"])
