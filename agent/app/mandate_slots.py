"""Lexical slot parser for the derivation mandates (tasks 210-221).

Phase 2a of ``docs/superpowers/plans/2026-09-08-ledger-dag-replan.md``. A host-side
``host_derive`` hook needs, from the mandate text alone, the list of *operand slots* the task
asks the agent to fill: which entity, and which field of it. The two existing parsers cannot
supply that pair --

* ``candidate_coverage.extract_named_candidates`` is digit-only and vetoes any item whose body
  begins with an imperative verb, so it returns ``[]`` on 210-217's ``A. Open the Wikipedia page
  for ... and read ...`` rosters. It has 8 consumers and is deliberately NOT widened here;
* ``execution_evidence_loop.derive_field_label`` derives ONE shared field label for a whole
  roster, which is right for 218-221 and wrong for 210-217 (whose two items ask for two
  different fields, sometimes of the same entity -- 215/216).

This module adds the missing lexical read, and only that. It is a host-side mechanism: pure
regex over the mandate, no model call, no network. Every entry point fails OPEN -- an
unrecognised mandate yields ``[]`` rather than a guess, and nothing here raises.

Two item shapes are recognised, both as lettered (``A.``, ``(a)``) or numbered (``1.``, ``1)``)
line-initial runs:

* **directed items** -- ``"Open the Wikipedia page for <entity> and read <field phrase>"``
  (210-217). Entity and field phrase come from the item itself, so the slots of one mandate may
  carry different field phrases. A URL written into the item is captured too.
* **bare-name items** -- ``"1. Mekong"`` (218-221). The entity is the item; the field phrase is
  the one the surrounding prose states once for the whole roster, taken from the mandate with
  the roster blanked out (``strip_enumerated_items``) so a candidate's own wording cannot answer
  for the question -- the same discipline ``derive_field_label`` follows, and its fallback.

The bare-name branch must not fire on an INSTRUCTION roster ("1. The exact name of that
vulnerable C function." -- tasks 044/093, whose items begin with an article rather than an
imperative verb and so slip past ``_INSTRUCTION_VERBS``). A name-shape test rejects the whole
list unless every item reads like a proper name: short, capitalised, no trailing full stop, no
leading determiner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from agent.app.idea_policies.candidate_coverage import (
    _INSTRUCTION_VERBS,
    _extract_name,
    strip_enumerated_items,
)

#: A lettered or numbered item at line start: ``"  A. body"`` / ``"(a) body"`` / ``"1) body"``.
_ITEM_LINE = re.compile(r"^[ \t]*\(?([A-Za-z]|\d{1,2})[.)]\s+(\S.*?)\s*$", re.MULTILINE)

#: ``Open the [Wikipedia] page for <entity> and read <field phrase>`` -- the 210-217 item body.
_DIRECTED_BODY = re.compile(
    r"^open\s+(?:the\s+)?(?:\w+\s+)?page\s+(?:for|of)\s+(?P<entity>.+?)"
    r"\s+and\s+read\s+(?P<field>.+)$",
    re.IGNORECASE | re.DOTALL,
)

#: ``Open <url> and read <field phrase>`` -- the same shape with the page named by URL.
_DIRECTED_URL_BODY = re.compile(
    r"^open\s+(?:the\s+page\s+at\s+)?(?P<url>https?://\S+?)\s+and\s+read\s+(?P<field>.+)$",
    re.IGNORECASE | re.DOTALL,
)

_URL = re.compile(r"https?://[^\s<>()\[\]\"']+")

#: The field phrase the surrounding prose states once for a bare-name roster.
_PROSE_READ_CUE = re.compile(r"\band\s+read\s+(?P<field>.+)$", re.IGNORECASE)

#: A trailing parenthetical qualifier on an entity ("Lake Tahoe (California/Nevada, USA)").
_TRAILING_PAREN = re.compile(r"\s*\([^()]*\)\s*$")

#: Determiners / interrogatives that begin a REPORT item, never a proper name. Number words are
#: deliberately absent: "One World Trade Center" (task 221) is a real roster entry.
_NAME_STOPWORDS = frozenset(
    {
        "the", "a", "an", "all", "any", "each", "every", "both", "this", "that", "these",
        "those", "its", "their", "his", "her", "which", "what", "whether", "how", "why",
        "when", "where", "who", "whose",
    }
)

#: A bare-name roster entry is short; the longest real one is 4 words ("One World Trade Center").
_MAX_NAME_WORDS = 6

#: Field phrases are prompt-sized labels. Wider than ``derive_field_label``'s 80 so the real
#: two-field prose of 218-221 survives whole ("... LENGTH (in kilometres) and its DRAINAGE ...").
_MAX_FIELD_CHARS = 200


@dataclass(frozen=True)
class Slot:
    """One operand the mandate asks for: a field of an entity, optionally at a named URL.

    :param entity: the thing to look up ("Lake Baikal"), trailing qualifier parenthetical removed.
    :param field_phrase: what to read off it ("its MAXIMUM DEPTH, in meters"), unit hints kept.
    :param url: the page URL when the item names one, else ``None``.
    :param index: 0-based position of this slot in the mandate's item run.
    """

    entity: str
    field_phrase: str
    url: Optional[str]
    index: int


def _marker(raw: str) -> Tuple[str, int]:
    """``("num"|"alpha", ordinal)`` for an item marker; ordinals are 1-based."""
    if raw.isdigit():
        return "num", int(raw)
    return "alpha", ord(raw.lower()) - ord("a") + 1


def _item_run(text: str) -> List[str]:
    """The bodies of ``text``'s longest consecutive line-initial item run, or ``[]``.

    Lettered and numbered runs are scanned independently and the longer one wins; each must start
    at ``1`` / ``A`` and increment by one, so a stray "(b)" in later prose cannot extend a run.
    """
    raw = [(m.group(1), m.group(2)) for m in _ITEM_LINE.finditer(text)]
    if not raw:
        return []
    best: List[str] = []
    for kind in ("alpha", "num"):
        run: List[str] = []
        expected = 1
        for marker, body in raw:
            mkind, ordinal = _marker(marker)
            if mkind != kind:
                continue
            if ordinal == expected:
                run.append(body)
                expected += 1
            elif ordinal == 1:
                run = [body]
                expected = 2
            else:
                if len(run) >= 2:
                    break
                run = []
                expected = 1
        if len(run) > len(best):
            best = run
    return best if len(best) >= 2 else []


def _clean_entity(text: str) -> str:
    """An entity name with its trailing qualifier parenthetical and punctuation removed."""
    name = _TRAILING_PAREN.sub("", re.sub(r"\s+", " ", text or "").strip())
    return name.strip().strip("'\"").strip(" .,;:").strip()


def _clean_field(text: str) -> str:
    """A field phrase: whitespace collapsed, trailing sentence punctuation removed, capped."""
    field = re.sub(r"\s+", " ", text or "").strip()
    return field.strip(" .;:,")[:_MAX_FIELD_CHARS].strip()


def _first_url(text: str) -> Optional[str]:
    m = _URL.search(text or "")
    return m.group(0).rstrip(".,;") if m else None


def _entity_from_url(url: str) -> str:
    """A readable entity name from a wiki-style URL slug ("/wiki/Lake_Tahoe" -> "Lake Tahoe")."""
    slug = (url or "").rstrip("/").rsplit("/", 1)[-1].split("#", 1)[0].split("?", 1)[0]
    return re.sub(r"[_+]+", " ", slug).strip()


def _parse_directed(body: str) -> Optional[Tuple[str, str, Optional[str]]]:
    """``(entity, field_phrase, url)`` for an "Open ... and read ..." item body, else ``None``."""
    flat = re.sub(r"\s+", " ", body or "").strip()
    url = _first_url(flat)
    m = _DIRECTED_BODY.match(flat)
    if m:
        entity = _clean_entity(m.group("entity"))
        if entity.lower().startswith(("http://", "https://")):
            entity = _entity_from_url(entity)
        field = _clean_field(m.group("field"))
        return (entity, field, url) if entity and field else None
    m = _DIRECTED_URL_BODY.match(flat)
    if m:
        entity = _entity_from_url(m.group("url"))
        field = _clean_field(m.group("field"))
        return (entity, field, m.group("url")) if entity and field else None
    return None


def _looks_like_a_name(body: str, name: str) -> bool:
    """Whether ``name`` (extracted from item ``body``) reads as a proper name, not a report ask."""
    if not name or len(name.split()) > _MAX_NAME_WORDS:
        return False
    if body.rstrip().endswith(".") or name.endswith("."):
        return False
    if not name[0].isalnum():
        return False
    first = re.match(r"[A-Za-z']+", name)
    token = first.group(0).lower() if first else ""
    return bool(token) and token not in _NAME_STOPWORDS and token not in _INSTRUCTION_VERBS


def shared_field_phrase(mandate: str) -> str:
    """The one field phrase a bare-name roster's surrounding prose states for every entity.

    Read from the mandate with the enumerated roster blanked out, so a candidate's own wording
    cannot answer for the question. Falls back to
    :func:`~agent.app.testing.execution_evidence_loop.derive_field_label` when the prose carries
    no "... and read <phrase>" cue, which is that function's own contract (never empty).
    """
    prose = strip_enumerated_items(mandate or "")
    for line in prose.splitlines():
        m = _PROSE_READ_CUE.search(re.sub(r"\s+", " ", line).strip())
        if m:
            field = _clean_field(m.group("field"))
            if field:
                return field
    # Imported here, not at module scope: ``execution_evidence_loop`` pulls in the whole
    # evidence-loop host, and this module is imported by it via ``ledger_tools``.
    from agent.app.testing.execution_evidence_loop import derive_field_label

    return derive_field_label(mandate or "")


def parse_slots(mandate: str) -> List[Slot]:
    """The operand slots ``mandate`` enumerates, in item order.

    :param mandate: the task statement.
    :returns: one :class:`Slot` per item of the mandate's lettered/numbered run -- directed items
        carry their own field phrase, bare-name items share :func:`shared_field_phrase`. ``[]``
        for anything unrecognised: no item run, a run shorter than 2, a mixed run (some items
        directed and some not), or a roster of instruction/report asks rather than names.
    :raises: nothing, ever -- every failure mode returns ``[]``.
    """
    try:
        if not isinstance(mandate, str) or not mandate.strip():
            return []
        bodies = _item_run(mandate)
        if len(bodies) < 2:
            return []

        directed = [_parse_directed(body) for body in bodies]
        if all(d is not None for d in directed):
            return [
                Slot(entity=d[0], field_phrase=d[1], url=d[2], index=i)
                for i, d in enumerate(directed)
                if d is not None
            ]
        if any(d is not None for d in directed):
            return []  # mixed roster: not one shape, so not parsed

        names = [(body, _clean_entity(_extract_name(body))) for body in bodies]
        if not all(_looks_like_a_name(body, name) for body, name in names):
            return []
        field = shared_field_phrase(mandate)
        return [
            Slot(entity=name, field_phrase=field, url=_first_url(body), index=i)
            for i, (body, name) in enumerate(names)
        ]
    except Exception:  # noqa: BLE001 -- a parser that raises would take the host down
        return []


def slot_field_phrases(slots: List[Slot]) -> List[str]:
    """The distinct field phrases of ``slots``, in first-appearance order."""
    out: List[str] = []
    for slot in slots or []:
        phrase = getattr(slot, "field_phrase", "")
        if phrase and phrase not in out:
            out.append(phrase)
    return out
