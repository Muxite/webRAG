"""Engine-agnostic prompt-hygiene lint + transform.

Shared downward by both the compiled-scaffold test harness (``testing/execution_compiled.py``)
and, defensively, the native adaptive engine's own micro-prompt construction (``idea_policies/*``)
— neither engine imports the other; both may import this module, mirroring ``model_tiers.py``'s
"shared, engine-agnostic, imported downward by both" shape.

Covers one bug class so far: a leaf/micro-prompt instruction that separately asks a model to
answer with a BARE value ("nothing else", "no source") while also asking it to cite/give the
source URL. This self-contradiction measurably drives abstention on weak models (badmodel-lab:
hard-tier avg 0.60->0.906 from fixing exactly this on the compiled path) but is a general
prompt-authoring correctness bug, not something scoped to weak models or to one engine — any
leaf/micro-prompt author should be able to lint new instructions against it.

As of this module's creation, no live native prompt-construction call site pairs a bare-value
directive with a source-ask (checked: ``_select_links_with_llm``'s micro-prompt never asks for a
citation; ``verify_system_prompt`` asks for a quote/URL but never says "nothing else"). Its native
value today is defensive/future-proofing, not an active rewire.
"""
from __future__ import annotations

import re
from typing import List

# The three source-ask SHAPES a leaf instruction can contain, ported verbatim from
# ``execution_compiled.py`` — a pure text transform (not a benchmark-specific heuristic), so the
# regexes themselves are portable, unlike per-task validator regex.
_SOURCE_ASK_HEAD = (
    r"(?:cite\s+)?"
    r"(?:the|its|their|each\s+[a-z' ]{1,30}?'s|each|every)?\s*"
    r"(?:exact\s+|authoritative\s+|full\s+|complete\s+)*"
)
_INLINE_SOURCE_CLAUSE = re.compile(
    r"(?:[ \t]*[,;—–])?[ \t]*and\s+" + _SOURCE_ASK_HEAD + r"source URLs?\b[^.\n]*(?=[.\n]|$)",
    re.I,
)
_ENUM_SOURCE_ITEM = re.compile(
    r"(?:[ \t]*[,;])?\s*(?:and\s+)?\(\s*(?:[a-z]|[ivx]{1,4}|\d{1,2})\s*\)[ \t]*"
    + _SOURCE_ASK_HEAD + r"source URLs?\b[^.\n]*(?=[.\n]|$)",
    re.I,
)
_CITE_SENTENCE = re.compile(
    r"(?<![^\s])(?:Cite|Give)\b[^.\n]*\bsource URLs?\b[^.\n]*\.[ \t]*", re.I,
)
_DANGLING_PUNCT = re.compile(r"[ \t]*[,;:—–][ \t]*(?=\.)")

_SOURCE_ASK_PATTERNS = (_INLINE_SOURCE_CLAUSE, _ENUM_SOURCE_ITEM, _CITE_SENTENCE)

# A directive counts as demanding a BARE value when it contains one of these phrases — the exact
# wording the compiled path's ``_THIN_EXTRACT_SYS`` uses. Kept as a tuple (not one regex) so a
# caller composing its own directive text can match either signal independently.
_BARE_OUTPUT_SIGNALS = (
    re.compile(r"\bnothing else\b", re.I),
    re.compile(r"\bno source\b", re.I),
)


def find_bare_output_contradiction(output_directive: str, instruction: str) -> List[str]:
    """Return the source-ask clause(s) found in ``instruction`` IF ``output_directive`` also
    demands a bare value (no source) — the self-contradictory pattern this module exists to catch.

    Returns ``[]`` when the directive doesn't demand bareness, or the instruction has no
    source-ask (i.e. no contradiction is present). A LINT — never mutates either string.
    """
    if not any(sig.search(output_directive or "") for sig in _BARE_OUTPUT_SIGNALS):
        return []
    found: List[str] = []
    for pattern in _SOURCE_ASK_PATTERNS:
        found.extend(m.group(0) for m in pattern.finditer(instruction or ""))
    return found


def strip_source_ask(instruction: str) -> str:
    """Drop a plan-authored source-ask clause ('and the exact source URL' / '(c) the source URL'
    / standalone 'Cite ... source URL.' / 'Give the source URL.') from ``instruction``.

    Ported from ``execution_compiled._strip_source_ask`` — see that module's git history for the
    live-evidence rationale (qwen2.5:7b resolved the bare-value/cite-source contradiction by
    abstaining until the clause was textually removed from the question, not just addressed via a
    system-prompt override).

    An instruction with no source-ask is returned BYTE-IDENTICAL; an instruction that would be
    emptied entirely by the removal falls back to the original (an empty question extracts
    nothing).
    """
    out = _INLINE_SOURCE_CLAUSE.sub("", instruction)
    out = _ENUM_SOURCE_ITEM.sub("", out)
    out = _CITE_SENTENCE.sub("", out)
    if out == instruction:
        return instruction
    out = _DANGLING_PUNCT.sub("", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([.,;])", r"\1", out).strip()
    return out if re.search(r"[A-Za-z0-9]", out) else instruction
