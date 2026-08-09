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

    @property
    def needs_substantiation(self) -> bool:
        """True when the answer must be backed by actually-visited pages."""
        return self.grounding or self.navigation


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
