"""
Mandate requirements parser — the single source of truth for "what does this mandate
require the agent to do".

Both the post-expansion enforcement hooks and the finalization grounding gate consume
``parse_mandate_requirements`` so there is exactly one definition of the phrase sets that
distinguish "must visit", "must search", "navigate by following links", and "ground the
answer in opened pages". Keeping this in one place avoids the drift that happens when the
same phrase lists are copied across hooks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


# --- phrase sets ------------------------------------------------------------------

_VISIT_PHRASES = (
    "must visit",
    "you must visit",
    "required to visit",
    "need to visit",
    "should visit",
    "visit the url",
    "visit the page",
    "visit only",
    "open that page",
    "open the page",
)
_SEARCH_PHRASES = (
    "must search",
    "you must search",
    "search for",
    "find and visit",
)
_NAVIGATION_PHRASES = (
    "follow the link",
    "follow links",
    "following links",
    "follow hyperlinks",
    "following hyperlinks",
    "follow the hyperlink",
    "navigate by following",
    "navigate to",
    "do not use web search",
    "do not use search",
    "without using search",
    "only hyperlinks",
    "chain of wikipedia",
    "link chain",
    "wiki-race",
    "wikipedia game",
    "reach the target",
    "from start to target",
)
# Phrases that mark an enumerated candidate list as requiring INDIVIDUAL DISPOSITION: the
# mandate does not merely ask which candidate ranks highest, it asks the agent to establish a
# per-candidate verdict ("exactly one of these ...", "eliminate", "the survivor"). Measured
# over the 165 task modules in ``agent/app/idea_tests``: 77 mandates yield an enumerated
# candidate list at all, and requiring one of these cues narrows that to 39 -- every
# branch-eliminate / survivor / AND-filter task and nothing else (no breadth-argmax, no
# question list, no logic-constraint list). Kept short on purpose: a longer list buys nothing
# and each added phrase risks pulling in a shape where "disqualify each candidate" is not
# what the mandate asks for.
_DISPOSITION_PHRASES = (
    "exactly one",
    "only one of",
    "eliminate",
    "survivor",
    "disqualif",
    "rule out",
    "which single one",
    "which one of",
)

# B8: cues that the asked-for superlative is TIME-INDEXED, so a candidate's status has to be
# established as of a date rather than recalled. The motivating case is task 122: "largest ...
# CURRENTLY IN OPERATION", where the fame-anchored candidate (Arecibo) collapsed in December
# 2020 -- straddling most training-data cutoffs, so parametric memory answers it wrong with
# full confidence. Deliberately not exhaustive; a missed cue only means the extra year-token
# requirement is not imposed.
#
# Matched on WORD boundaries rather than as bare substrings: "active" is a substring of
# "radioactive" and "inactive", either of which would otherwise make a chemistry mandate
# time-indexed by accident.
_TIME_INDEXED_PHRASES = (
    "currently",
    "still",
    "as of",
    "in operation",
    "active",
    "today",
)
_TIME_INDEXED_RE = re.compile(
    r"\b(?:" + "|".join(p.replace(" ", r"\s+") for p in _TIME_INDEXED_PHRASES) + r")\b"
)

_GROUNDING_PHRASES = (
    "do not guess",
    "do not rely on memory",
    "do not answer from memory",
    "not from memory",
    "base the",            # "base the height on the page you open"
    "based on the page",
    "from the page you open",
    "from the pages you visit",
    "verify against",
    "verify against the page",
    "substantiate",
    "do not rely only on",
)

# Phrases that introduce a described navigation destination; the captured tail is used as a
# semantic ``link_idea`` for the visit action's link-following.
_GENERIC_TARGETS = frozenset({
    "next one", "the next one", "next", "the next", "it", "that page", "the page",
    "the target", "target", "target article", "the target article", "next article",
    "the next article", "following one", "the following",
})

_NAV_TARGET_PATTERNS = (
    r"follow the link to (?:the )?([^,.\n;]+?)(?:,|\.|\n| then | and | open | then,|$)",
    r"follow the hyperlink to (?:the )?([^,.\n;]+?)(?:,|\.|\n| then | and |$)",
    r"link to (?:the )?(?:wikipedia page of )?(?:the )?([^,.\n;]+?)(?:,|\.|\n| then | and |$)",
    r"navigate to (?:the )?([^,.\n;]+?)(?:,|\.|\n| then | and |$)",
    r"reach the (?:wikipedia )?(?:page (?:about|of|for) )?(?:the )?([^,.\n;]+?)(?:,|\.|\n| then | and |$)",
)


@dataclass
class MandateRequirements:
    """Structured view of what a mandate asks the agent to do."""

    named_urls: List[str] = field(default_factory=list)
    must_visit: bool = False
    must_search: bool = False
    navigation: bool = False
    grounding: bool = False
    nav_targets: List[str] = field(default_factory=list)
    roster_candidates: List[str] = field(default_factory=list)
    individual_disposition: bool = False
    time_indexed: bool = False

    @property
    def needs_substantiation(self) -> bool:
        """True when the answer must be backed by actually-visited pages."""
        return self.grounding or self.navigation

    @property
    def requires_candidate_roster(self) -> bool:
        """True when every enumerated candidate needs its own recorded verdict.

        Both halves are required: an enumerated list ALONE is the breadth-argmax shape (pick
        the biggest of six lakes), where a per-candidate disqualifier is not something the
        mandate asks for; a disposition cue alone has no roster to enforce against.
        """
        return bool(self.roster_candidates) and self.individual_disposition


def clean_extracted_url(url: str) -> str:
    """Strip trailing punctuation while preserving balanced parens (Wikipedia URLs)."""
    strip_chars = ".,;:!?"
    url = url.rstrip(strip_chars)
    while url.endswith(")") and url.count("(") < url.count(")"):
        url = url[:-1]
    url = url.rstrip(strip_chars)
    return url


def extract_urls(text: str) -> List[str]:
    """Pull cleaned URLs out of free text, order-preserving and de-duplicated."""
    raw = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text or "")
    out: List[str] = []
    seen = set()
    for u in raw:
        c = clean_extracted_url(u)
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _slug_phrase(url: str) -> str:
    """Turn a wiki URL's trailing slug into a human phrase ('Roman_Empire' -> 'roman empire')."""
    from urllib.parse import unquote, urlparse

    try:
        path = (urlparse(url).path or "").rstrip("/")
    except ValueError:
        return ""
    slug = path.rsplit("/", 1)[-1] if path else ""
    return unquote(slug).replace("_", " ").replace("-", " ").strip().lower()


#: A roster entry has to be usable as a NAME -- something a page title can be matched against
#: and a sentence can be found mentioning. ``extract_named_candidates`` was built for a
#: narrower job (branch-eliminate coverage) and happily returns whole sentences when the
#: numbered list is a question list ("1. What class of attack is it, and ...") or a logic
#: puzzle's constraints ("1. Kestrel is not tested in slot 2."). Those are not names, so the
#: whole roster is rejected: a short token count plus "no terminal sentence punctuation" plus
#: "not a URL" separates every real candidate list in the corpus from every list that is not one.
_MAX_ROSTER_NAME_WORDS = 6


def _is_name_like(name: str) -> bool:
    text = str(name or "").strip()
    if not text or len(text.split()) > _MAX_ROSTER_NAME_WORDS:
        return False
    if text.lower().startswith(("http://", "https://", "www.")):
        return False
    return not re.search(r"[.?!]$", text)


def _roster_candidates(mandate: str) -> List[str]:
    """The enumerated candidate names, or ``[]`` when the list is not a roster of names.

    Fails CLOSED on any item that is not name-like: a partially-parsed roster would demand a
    disposition record for a candidate no page can be matched to, which is a false positive
    the caller has no way to recover from.
    """
    from agent.app.idea_policies.candidate_coverage import extract_named_candidates

    names = extract_named_candidates(mandate)
    if not names or not all(_is_name_like(n) for n in names):
        return []
    return names


def parse_mandate_requirements(mandate: str) -> MandateRequirements:
    """Parse a mandate string into a :class:`MandateRequirements`."""
    req = MandateRequirements()
    if not mandate:
        return req
    low = mandate.lower()

    req.named_urls = extract_urls(mandate)
    req.must_visit = any(p in low for p in _VISIT_PHRASES)
    req.must_search = any(p in low for p in _SEARCH_PHRASES)
    req.navigation = any(p in low for p in _NAVIGATION_PHRASES)
    req.grounding = any(p in low for p in _GROUNDING_PHRASES)
    req.individual_disposition = any(p in low for p in _DISPOSITION_PHRASES)
    req.time_indexed = bool(_TIME_INDEXED_RE.search(low))
    req.roster_candidates = _roster_candidates(mandate)

    targets: List[str] = []
    seen = set()
    for pat in _NAV_TARGET_PATTERNS:
        for m in re.finditer(pat, low):
            phrase = m.group(1).strip()
            # keep short, descriptive phrases usable as a link_idea; drop generic anaphora
            # ("the next one", "that page") that can't be matched against a page's links.
            if 2 <= len(phrase) <= 80 and phrase not in seen and phrase not in _GENERIC_TARGETS:
                seen.add(phrase)
                targets.append(phrase)
    # Wiki-race style: a TARGET: <url> line names the destination by slug.
    if req.navigation:
        m = re.search(r"target:\s*(\S+)", low)
        if m:
            slug = _slug_phrase(m.group(1))
            if slug and slug not in seen:
                seen.add(slug)
                targets.append(slug)
    req.nav_targets = targets
    return req
