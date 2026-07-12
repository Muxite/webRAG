"""Approximator-stripped answer voting — shared, pure helpers.

Ports the compiled-path robustness lever (``testing/execution_compiled._strip_approximators`` /
``_vote_key`` / ``_vote_extract``, which stays a frozen read-only reference) for reuse by the
NATIVE synthesis path. The idea: k independent extractions of a terminal answer are more robust
to one bad read than a single extraction — but only if superficial phrasing variants of the SAME
value ("approximately 646 m", "~646", "646m") land in ONE vote bucket. These helpers normalize a
vote KEY only; they NEVER round or fuzzy-match the underlying value, so genuinely different answers
(646 vs 564) always stay distinct.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable, Sequence, TypeVar

# Superficial phrasing wrappers a model sprinkles around the SAME answer ("approximately 265 m",
# "about 265", "~265m"). Stripping them keeps one real answer in ONE vote bucket instead of
# splitting the majority across format variants — WITHOUT touching the answer value itself (the
# digits are never rounded or fuzzy-matched: "265" and "275" stay distinct). Conservative on purpose.
_VOTE_APPROX_WRAPPERS = re.compile(
    r"\b(?:approximately|approx|about|around|roughly|nearly|almost|estimated|est|circa|ca|"
    r"over|under|approx\.)\b\.?",
    re.I,
)
_VOTE_TILDE = re.compile(r"[~≈]")  # '~' and '≈'


def strip_approximators(text: str) -> str:
    """Remove leading/inline approximator wrappers and the '~'/'≈' prefixes ("approximately 265m",
    "~265") so a value and its hedged phrasings share a vote bucket. Whitespace-collapsed; the
    numeric/entity value is untouched — only the surrounding noise words go."""
    cleaned = _VOTE_TILDE.sub(" ", text)
    cleaned = _VOTE_APPROX_WRAPPERS.sub(" ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def vote_key(ans: str) -> str:
    """Normalized voting key so equivalent answers vote together. Strips superficial phrasing noise
    (approximators, '~', units/whitespace/case) FIRST, then prefers the longest number token when
    present ("approximately 1,642 m", "1642 metres", "Max depth: ~1,642" -> "1642"); else a cleaned
    text key. Never rounds or fuzzy-matches the value, so genuinely different answers stay distinct."""
    low = strip_approximators(ans.strip().lower())
    nums = re.findall(r"\d[\d,]*", low)
    if nums:
        return max((n.replace(",", "") for n in nums), key=len)
    return re.sub(r"[^a-z0-9]+", " ", low).strip()


T = TypeVar("T")


def majority_vote(
    items: Sequence[T],
    key_fn: Callable[[T], str],
    anchor_index: int = 0,
) -> T:
    """Pick the majority item by ``key_fn``, tie-breaking toward the anchor.

    Mirrors the compiled ``_vote_extract`` selection: count normalized keys, take the top count,
    and among the tied keys prefer the ANCHOR's key (``items[anchor_index]`` — the deterministic
    temp-0 read) when it is one of them; otherwise the most common key. Returns the FIRST item
    carrying the chosen key (stable). ``items`` must be non-empty.
    """
    if not items:
        raise ValueError("majority_vote requires at least one item")
    keys = [key_fn(it) for it in items]
    counts = Counter(keys)
    top_count = counts.most_common(1)[0][1]
    tied = {k for k, c in counts.items() if c == top_count}
    anchor_key = keys[anchor_index] if 0 <= anchor_index < len(keys) else None
    chosen = anchor_key if anchor_key in tied else counts.most_common(1)[0][0]
    for it, k in zip(items, keys):
        if k == chosen:
            return it
    return items[anchor_index]
