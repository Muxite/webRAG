"""Unit tests for `_url_slug_tokens` and the semantic dedup matching predicate.

We test the pure-logic helper without importing the engine (which pulls in
bs4 / chromadb). The helper is a staticmethod, so we re-implement the same
algorithm inline and assert on identical inputs we'd send through the engine.
This guards the match predicate, which is the only failure-prone piece.
"""

from __future__ import annotations

from urllib.parse import unquote, urlparse


def url_slug_tokens(url: str) -> list[str]:
    """Mirror of `IdeaDagEngine._url_slug_tokens` for isolated testing."""
    if not isinstance(url, str) or not url:
        return []
    try:
        parsed = urlparse(url)
    except ValueError:
        return []
    path = (parsed.path or "").rstrip("/")
    slug = path.rsplit("/", 1)[-1] if path else ""
    if not slug:
        return []
    slug = unquote(slug).replace("_", " ").replace("-", " ").lower()
    slug = slug.replace("(", " ").replace(")", " ")
    return [t for t in slug.split() if len(t) >= 3]


def matches(tokens: list[str], text: str) -> bool:
    """Return True when every slug token appears in the text."""
    text_lower = text.lower()
    return bool(tokens) and all(tok in text_lower for tok in tokens)


# ---- slug extraction ----

def test_slug_simple_wikipedia():
    assert url_slug_tokens("https://en.wikipedia.org/wiki/Axolotl") == ["axolotl"]


def test_slug_multi_word_underscore():
    assert url_slug_tokens("https://en.wikipedia.org/wiki/Voynich_manuscript") == ["voynich", "manuscript"]


def test_slug_with_parens():
    tokens = url_slug_tokens("https://en.wikipedia.org/wiki/Pando_(tree)")
    assert tokens == ["pando", "tree"]


def test_slug_with_hyphen():
    tokens = url_slug_tokens("https://example.com/some-thing-special")
    assert tokens == ["some", "thing", "special"]


def test_slug_percent_encoded():
    tokens = url_slug_tokens("https://en.wikipedia.org/wiki/%C3%89tat")
    # "État" lowercases to "état" via unquote
    assert "état" in tokens


def test_slug_root_path_empty():
    assert url_slug_tokens("https://example.com/") == []
    assert url_slug_tokens("https://example.com") == []


def test_slug_filters_short_tokens():
    # "a" and "is" would be dropped by the >=3-char filter.
    tokens = url_slug_tokens("https://example.com/a_is_in")
    assert "in" not in tokens  # too short
    assert tokens == []  # all tokens <3 chars


def test_slug_invalid_input():
    assert url_slug_tokens("") == []
    assert url_slug_tokens(None) == []  # type: ignore[arg-type]


# ---- match predicate ----

def test_match_axolotl_in_title():
    title = "Visit Axolotl Wikipedia page and extract scientific name"
    assert matches(["axolotl"], title) is True


def test_match_multi_word_requires_all_tokens():
    title = "Visit Voynich manuscript Wikipedia page"
    assert matches(["voynich", "manuscript"], title) is True
    # Only one word present → no match
    assert matches(["voynich", "missingword"], title) is False


def test_match_pando_tree():
    title = "Visit Pando clonal colony Wikipedia page"
    # Token "tree" not in the title → no match (even though "pando" is)
    assert matches(["pando", "tree"], title) is False


def test_match_pando_with_full_text():
    # The match should succeed if the slug appears in any of (title, goal,
    # parent_goal) concatenated. parent_goal often contains the URL.
    parent_goal = "https://en.wikipedia.org/wiki/Pando_(tree) - what state?"
    title = "Visit Pando clonal colony Wikipedia"
    combined = title + " " + parent_goal
    assert matches(["pando", "tree"], combined) is True


def test_match_no_tokens_returns_false():
    assert matches([], "anything") is False


def test_match_empty_text_returns_false():
    assert matches(["axolotl"], "") is False


def test_match_case_insensitive():
    assert matches(["axolotl"], "VISIT AXOLOTL WIKIPEDIA") is True


# ---- Fix #1: hook-only gate predicate ----
# Mirror of `_is_hook_injected` in idea_engine.py so we can test the predicate
# in isolation. The real engine reads source_node.details[JUSTIFICATION];
# we replicate the literal check here.

def is_hook_injected(justification: str) -> bool:
    return (justification or "").startswith("Mandate requires visiting")


def test_gate_allows_when_justification_matches_url_hook():
    # MandateUrlInjectionHook writes exactly this string at post_expansion_hooks.py:113
    assert is_hook_injected("Mandate requires visiting this URL") is True


def test_gate_blocks_when_justification_is_planner_generated():
    # Planner candidates carry varied justifications, none starting with the marker.
    assert is_hook_injected("Visit the Wikipedia page about X") is False
    assert is_hook_injected("") is False
    assert is_hook_injected(None) is False  # type: ignore[arg-type]


def test_gate_blocks_when_justification_is_phrase_hook():
    # MandatePhraseEnforcementHook writes a different prefix and typically
    # creates a link_idea visit (no URL) anyway, but be explicit.
    phrase_hook_just = "Mandate explicitly requires visit action - will extract URL from search results or use link_idea"
    assert is_hook_injected(phrase_hook_just) is False


def test_gate_blocks_when_justification_is_partial_match():
    # Suffix-only matches must fail; we anchor on the leading marker.
    assert is_hook_injected("Other text. Mandate requires visiting this URL") is False
