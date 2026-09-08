"""Operand attribution: which quantity on this page is the operand a mandate slot asks for.

The derivation host has to turn "read the height of its flue-gas chimney/stack, in meters" into
one specific :class:`~agent.app.quantity_index.QuantityRef` out of the forty a Wikipedia infobox
yields. Today that choice is made by first-match-wins / fixed-preference heuristics scattered
through ``ledger_tools`` (``_locate``, ``_find_backed_match``, ``_best_explanation``); this module
replaces the choice with an explicit **ranking** whose inputs are named, whose weights live in a
diffable JSON file, and whose ablation is a one-line swap.

Three properties are deliberate:

* **Pure lexical / structural.** No encoder, no model call, no I/O beyond reading the weight file.
  Every feature is computable from the slot, the entry, the page URL and the page text, so the
  ranker can never block a host on a model download and never costs a token. (An earlier draft of
  this work carried two sentence-encoder cosine features; they are cut. Nothing here imports
  ``sentence_transformers``.)
* **A JSON weight artifact, not code.** :data:`DEFAULT_MODEL_PATH` holds
  ``{"version", "kind", "feature_names", "weights", "intercept"}``. The shipped artifact is the
  hand rule; a fitted logistic regression over the same :data:`FEATURE_NAMES` drops in by
  overwriting the file, with no change here. The weights are a dict keyed by feature NAME rather
  than a positional coefficient list, so a reordering of :data:`FEATURE_NAMES` can never silently
  re-map a fitted vector onto the wrong features.
* **A named negative control ships alongside.** :func:`document_order_ranker` returns entries in
  index order with a constant score. The offline replay uses it as the ablation arm: if the hand
  rule does not beat document order, the ranking machinery is not earning its place and a trained
  ranker certainly is not.

Determinism: the scan order of ``build_index`` is the tie-break. Two entries with identical
features come back in the order they were passed, always.

The slot is **duck-typed** — anything with ``entity`` and ``field_phrase`` string attributes works
(``agent.app.mandate_slots.Slot`` is the intended producer). This module does not import that one:
the ranker is usable from a test, a script or a host that has only a pair of strings.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import unquote, urlparse

from agent.app.quantity_index import QuantityRef, _UNIT_WHITELIST
from agent.app.testing.evidence_graph import canonical_unit, normalize_for_match

#: The weight artifact shipped next to this module.
DEFAULT_MODEL_PATH = Path(__file__).with_name("operand_attribution_model.json")

#: The feature vector, in order. Explicit and closed: :meth:`OperandRanker.load` refuses an
#: artifact whose ``feature_names`` differ, so a stale fitted file fails loudly instead of being
#: silently mis-applied.
FEATURE_NAMES: List[str] = [
    # Does the entry's infobox label say what the field phrase asks for?
    "label_token_overlap",
    # Is the slot's entity named near this entry on the page? (the page may cover many entities)
    "entity_in_window",
    # Is the slot's entity the subject of this page at all? (constant per page; cross-page signal)
    "entity_in_url_slug",
    # Infobox rows are labelled and tabular; prose numbers are not.
    "is_infobox",
    # A unit-bearing number is a measurement; a bare one may be anything.
    "unit_present",
    # The phrase says "in meters" and the entry is written in metres.
    "unit_hint_match",
    # A small unitless integer ("2 flues", "3 runways") is almost never the asked-for operand.
    "is_trivial_bare_int",
    # A bare 1900-2099 integer is a date, not a measurement.
    "value_is_year_like",
]

#: The hand rule. Only five features carry weight; the other three are measured and reported but
#: not trusted to move an ordering without evidence. ``entity_in_url_slug`` and ``unit_present``
#: are near-constant within one page (they cannot re-order it), and ``value_is_year_like`` overlaps
#: ``is_trivial_bare_int`` on exactly the rows that matter, so giving it its own weight would
#: double-count the same evidence.
HAND_WEIGHTS: Dict[str, float] = {
    "label_token_overlap": 4.0,
    "entity_in_window": 2.0,
    "entity_in_url_slug": 0.0,
    "is_infobox": 1.0,
    "unit_present": 0.0,
    "unit_hint_match": 1.5,
    "is_trivial_bare_int": -3.0,
    "value_is_year_like": 0.0,
}
HAND_INTERCEPT = -2.0

#: Characters either side of an entry that count as "near" for :data:`entity_in_window`.
ENTITY_WINDOW_CHARS = 300

#: Tokens shorter than this are not treated as identifying parts of an entity name ("of", "de",
#: "2" in "GRES-2"); a name made only of such tokens falls back to using all of them.
_SIGNIFICANT_TOKEN_LEN = 4

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_INT_RE = re.compile(r"-?\d+")

#: Connectives that appear in every field phrase and every other infobox label, so counting them
#: as overlap would make "its height, in meters" match "Date of completion".
_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "it", "its", "of", "on", "or",
    "read", "s", "that", "the", "then", "this", "to", "with",
})


def _tokens(text: Any) -> List[str]:
    """Lowercased alphanumeric tokens of ``text``, in order, stopwords kept."""
    return _TOKEN_RE.findall(str(text or "").lower())


def _content_tokens(text: Any) -> Set[str]:
    """Token set of ``text`` with stopwords and single characters dropped."""
    return {token for token in _tokens(text) if len(token) > 1 and token not in _STOPWORDS}


def _significant_tokens(name: Any) -> Set[str]:
    """The identifying tokens of an entity name, falling back to all of them when none qualify."""
    content = _content_tokens(name)
    significant = {token for token in content if len(token) >= _SIGNIFICANT_TOKEN_LEN}
    return significant or content


def _slot_fields(slot: Any) -> Tuple[str, str]:
    """``(entity, field_phrase)`` off a duck-typed slot; missing attributes read as ``""``."""
    return (str(getattr(slot, "entity", "") or ""),
            str(getattr(slot, "field_phrase", "") or ""))


def label_token_overlap(field_phrase: str, label: str) -> float:
    """Symmetric containment of the field phrase's tokens and the entry label's, in [0, 1].

    ``max`` of the two containments rather than Jaccard: a one-word label (``"Height"``) inside a
    ten-word field phrase is a perfect match on the evidence available, and Jaccard would score it
    0.1 purely for the phrase's verbosity. The cost is that a label nested in another
    (``"Length"`` inside ``"Total length"``) can tie; ties fall through to document order, which is
    the honest answer when the label text alone cannot separate them.
    """
    field = _content_tokens(field_phrase)
    entry_label = _content_tokens(label)
    if not field or not entry_label:
        return 0.0
    shared = len(field & entry_label)
    if not shared:
        return 0.0
    return max(shared / len(field), shared / len(entry_label))


def _entity_in_window(entity: str, entry: QuantityRef, page_text: str) -> float:
    if not page_text or not entity:
        return 0.0
    low = max(0, int(entry.start) - ENTITY_WINDOW_CHARS)
    high = min(len(page_text), int(entry.end) + ENTITY_WINDOW_CHARS)
    window = set(_tokens(page_text[low:high]))
    return float(bool(_significant_tokens(entity) & window))


def _entity_in_url_slug(entity: str, page_url: str) -> float:
    """Fraction of the entity's identifying tokens present in the URL's last path segment."""
    if not page_url or not entity:
        return 0.0
    slug = unquote(urlparse(page_url).path).rsplit("/", 1)[-1]
    slug_tokens = set(_tokens(slug))
    wanted = _significant_tokens(entity)
    if not wanted:
        return 0.0
    return len(wanted & slug_tokens) / len(wanted)


def _unit_hints(field_phrase: str) -> Set[str]:
    """Canonical units named by the field phrase itself (``"in meters"`` -> ``{"m"}``).

    Unigrams and bigrams are both checked against ``quantity_index``'s unit whitelist, because the
    spelled-out multiword forms the phrases use (``"square kilometres"``) are single units.
    """
    words = _tokens(field_phrase)
    hints: Set[str] = set()
    for index, word in enumerate(words):
        for candidate in (word, " ".join(words[index:index + 2])):
            if normalize_for_match(candidate) in _UNIT_WHITELIST:
                hints.add(canonical_unit(candidate))
    return hints


def _bare_int(entry: QuantityRef) -> Optional[int]:
    """The entry's value as an integer when it is unitless and integral, else ``None``."""
    if str(entry.unit or "").strip():
        return None
    raw = str(entry.value or "").replace(",", "").replace(" ", "").strip()
    if not _INT_RE.fullmatch(raw):
        return None
    return int(raw)


def features(slot: Any, entry: QuantityRef, *, page_url: str = "",
             page_text: str = "") -> List[float]:
    """The :data:`FEATURE_NAMES` vector for one ``(slot, entry)`` pair on one page.

    :param slot: anything with ``entity`` and ``field_phrase`` string attributes.
    :param entry: one :class:`~agent.app.quantity_index.QuantityRef` from ``build_index``.
    :param page_url: the URL the page was fetched from; ``""`` zeroes the slug feature.
    :param page_text: the RAW text ``build_index`` was given, so ``entry.start``/``end`` index it;
        ``""`` zeroes the window feature.
    :returns: a list of floats, one per :data:`FEATURE_NAMES` entry, in that order.
    :raises: nothing — a missing attribute, a ``None`` field or an out-of-range offset all
        degrade to a zero feature rather than an exception, because a host calls this on whatever
        a weak model's page fetch produced.
    """
    entity, field_phrase = _slot_fields(slot)
    text = str(page_text or "")
    bare = _bare_int(entry)
    return [
        label_token_overlap(field_phrase, entry.label),
        _entity_in_window(entity, entry, text),
        _entity_in_url_slug(entity, str(page_url or "")),
        float(entry.source == "infobox"),
        float(bool(str(entry.unit or "").strip())),
        float(canonical_unit(entry.unit) in _unit_hints(field_phrase)
              if str(entry.unit or "").strip() else False),
        float(bare is not None and abs(bare) < 100),
        float(bare is not None and 1900 <= bare <= 2099),
    ]


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp = math.exp(logit)
    return exp / (1.0 + exp)


class OperandRanker:
    """A linear scorer over :data:`FEATURE_NAMES`: ``sigmoid(w . x + b)``.

    The weights come from :data:`DEFAULT_MODEL_PATH` (hand rule today, a fitted vector later);
    nothing about this class knows which, which is the point.
    """

    def __init__(self, weights: Mapping[str, float], intercept: float, *, version: int = 1,
                 kind: str = "hand_rule") -> None:
        missing = [name for name in FEATURE_NAMES if name not in weights]
        if missing:
            raise ValueError(f"operand attribution weights missing features: {missing}")
        self.weights = {name: float(weights[name]) for name in FEATURE_NAMES}
        self.intercept = float(intercept)
        self.version = int(version)
        self.kind = str(kind)

    @property
    def name(self) -> str:
        """Short label for reports — the arm this ranker represents."""
        return self.kind

    @classmethod
    def load(cls, path: Any = DEFAULT_MODEL_PATH) -> "OperandRanker":
        """Read a weight artifact.

        :raises FileNotFoundError: the artifact is absent.
        :raises ValueError: its ``feature_names`` are not exactly this module's, or a weight is
            missing — a fitted file from an older feature set must fail, not be reinterpreted.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"operand attribution model not found: {path}")
        data = json.loads(path.read_text())
        if list(data.get("feature_names") or []) != FEATURE_NAMES:
            raise ValueError(
                f"artifact feature_names do not match this module's FEATURE_NAMES: {path}")
        return cls(data.get("weights") or {}, data.get("intercept", 0.0),
                   version=data.get("version", 1), kind=data.get("kind", "hand_rule"))

    def to_dict(self) -> Dict[str, Any]:
        """The artifact form of this ranker — what :meth:`load` reads back."""
        return {
            "version": self.version,
            "kind": self.kind,
            "feature_names": list(FEATURE_NAMES),
            "weights": dict(self.weights),
            "intercept": self.intercept,
        }

    def score(self, slot: Any, entry: QuantityRef, *, page_url: str = "",
              page_text: str = "") -> float:
        """The probability-shaped score for one entry, in (0, 1)."""
        vector = features(slot, entry, page_url=page_url, page_text=page_text)
        logit = self.intercept + sum(self.weights[name] * value
                                     for name, value in zip(FEATURE_NAMES, vector))
        return _sigmoid(logit)

    def rank(self, slot: Any, entries: Sequence[QuantityRef], *, page_url: str = "",
             page_text: str = "") -> List[Tuple[float, QuantityRef]]:
        """``entries`` scored and sorted best-first; equal scores keep the input order.

        :returns: ``[(score, entry), ...]``, one per input entry, never reordered non-
            deterministically. An empty ``entries`` returns ``[]``.
        """
        scored = [
            (self.score(slot, entry, page_url=page_url, page_text=page_text), index, entry)
            for index, entry in enumerate(entries)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [(score, entry) for score, _, entry in scored]


class DocumentOrderRanker:
    """The negative control: document order, one constant score, no features consulted.

    This is the ablation arm of the offline replay. Any accuracy the hand rule shows over this is
    attributable to the features; anything it does not show is attributable to the fact that the
    right operand is usually printed first.
    """

    #: The score every entry gets. Constant on purpose: a report that groups by score sees one bin.
    CONSTANT_SCORE = 0.5

    name = "document_order"
    kind = "document_order"

    def rank(self, slot: Any, entries: Sequence[QuantityRef], *, page_url: str = "",
             page_text: str = "") -> List[Tuple[float, QuantityRef]]:
        """``entries`` unchanged, each paired with :data:`CONSTANT_SCORE`."""
        return [(self.CONSTANT_SCORE, entry) for entry in entries]


@lru_cache(maxsize=1)
def default_ranker() -> OperandRanker:
    """The shipped ranker, read once from :data:`DEFAULT_MODEL_PATH`.

    Cached because it is stateless and immutable in practice; a caller that wants a different
    artifact calls :meth:`OperandRanker.load` directly.
    """
    return OperandRanker.load(DEFAULT_MODEL_PATH)


def document_order_ranker() -> DocumentOrderRanker:
    """The negative-control ranker — see :class:`DocumentOrderRanker`."""
    return DocumentOrderRanker()
