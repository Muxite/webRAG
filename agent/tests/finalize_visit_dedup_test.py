"""F1: the finalize evidence block must not repeat a page it already emitted.

``_collect_all_visit_content`` walked every successful VISIT node and emitted one section
each, up to a hard 80,000-char total, then stopped. It carried no seen-URL set, while
``_visited_sources`` twelve lines below deduped the very same selection by normalized URL.
On aggregation-shaped runs (the same hub page opened from several branches) most of the
emitted bytes were repeats, and because the walk fills depth-first to the cap and then
breaks, those repeats evicted distinct pages that were never reached at all.

The fix extracts the normalization both selections share into ``_visit_url_key`` so they
cannot drift, and adds the seen-set to the content collector. ``_visited_sources`` output
must stay byte-identical.

Offline: fake graph/node shells only (`finalize_visit_content_test`'s pattern), no engine.
"""
from agent.app.idea_finalize import (
    _collect_all_visit_content,
    _visit_url_key,
    _visited_sources,
)
from agent.app.idea_policies.base import DetailKey, IdeaActionType


class _Node:
    def __init__(self, action, result=None):
        self.details = {DetailKey.ACTION.value: action}
        if result is not None:
            self.details[DetailKey.ACTION_RESULT.value] = result


class _Graph:
    def __init__(self, nodes):
        self._nodes = nodes

    def iter_depth_first(self):
        return iter(self._nodes)


def _visit(url, **result):
    return _Node(IdeaActionType.VISIT.value, {"success": True, "url": url, **result})


def _legacy_key(url):
    """The inline expression `_visited_sources` used before `_visit_url_key` existed."""
    return url.split("#", 1)[0].rstrip("/").lower()


# --------------------------------------------------------------------------- key


def test_key_collapses_fragment_trailing_slash_and_case():
    variants = [
        "https://en.wikipedia.org/wiki/Dam",
        "https://en.wikipedia.org/wiki/Dam/",
        "https://en.wikipedia.org/wiki/Dam#History",
        "HTTPS://EN.WIKIPEDIA.ORG/wiki/Dam",
        "  https://en.wikipedia.org/wiki/Dam  ",
    ]
    assert len({_visit_url_key(u) for u in variants}) == 1


def test_key_keeps_distinct_pages_distinct():
    assert _visit_url_key("https://e.x/a") != _visit_url_key("https://e.x/b")


def test_key_tolerates_none_and_empty():
    assert _visit_url_key(None) == ""
    assert _visit_url_key("") == ""


# --------------------------------------------------------------------------- F1 dedup


def test_duplicate_urls_collapse_to_one_section():
    g = _Graph([
        _visit("https://e.x/a", content="ALPHA BODY"),
        _visit("https://e.x/a", content="ALPHA BODY"),
        _visit("https://e.x/b", content="BETA BODY"),
    ])

    out = _collect_all_visit_content(g)

    assert out.count("--- URL: https://e.x/a") == 1
    assert out.count("--- URL: https://e.x/b") == 1


def test_fragment_slash_and_case_variants_collapse():
    g = _Graph([
        _visit("https://e.x/a", content="FIRST"),
        _visit("https://e.x/a/", content="SECOND"),
        _visit("https://e.x/a#part-2", content="THIRD"),
        _visit("HTTPS://E.X/a", content="FOURTH"),
    ])

    out = _collect_all_visit_content(g)

    assert out.count("--- URL:") == 1
    assert "FIRST" in out
    for later in ("SECOND", "THIRD", "FOURTH"):
        assert later not in out


def test_first_seen_body_wins_like_visited_sources():
    g = _Graph([
        _visit("https://e.x/a", content="EARLIER"),
        _visit("https://e.x/a", content="LATER"),
    ])

    out = _collect_all_visit_content(g)

    assert "EARLIER" in out and "LATER" not in out


def test_a_visit_without_a_url_is_still_emitted():
    """Empty keys are not deduped: two unnamed pages are not evidence of one page."""
    g = _Graph([_visit("", content="BODY ONE"), _visit("", content="BODY TWO")])

    out = _collect_all_visit_content(g)

    assert "BODY ONE" in out and "BODY TWO" in out


def test_duplicates_no_longer_evict_distinct_pages_at_the_total_cap():
    """Five copies of one page used to consume the 80k budget and break before page B."""
    dup_body = "A" * 15000
    g = _Graph(
        [_visit("https://e.x/a", content=dup_body) for _ in range(5)]
        + [_visit("https://e.x/b", content="B" * 15000)]
    )

    out = _collect_all_visit_content(g)

    assert "--- URL: https://e.x/b" in out
    assert out.count("--- URL: https://e.x/a") == 1
    assert "truncated due to total size limit" not in out


def test_the_total_cap_still_applies_to_distinct_pages():
    g = _Graph([_visit(f"https://e.x/{i}", content="C" * 15000) for i in range(10)])

    out = _collect_all_visit_content(g)

    assert len(out) <= 80000 + 200
    assert "truncated due to total size limit" in out


# --------------------------------------------------------------------------- byte-identity


_MIXED = [
    _visit("https://e.x/a", title="A"),
    _visit("https://e.x/a/", title="A again"),
    _visit("https://e.x/a#frag", title="A fragment"),
    _visit("HTTPS://E.X/A", title="A upper"),
    _visit("  https://e.x/b  ", title=" B "),
    _visit("", title="no url"),
    _Node(IdeaActionType.VISIT.value, {"success": False, "url": "https://e.x/c"}),
    _Node(IdeaActionType.SEARCH.value, {"success": True, "url": "https://e.x/d"}),
]


def _legacy_visited_sources(nodes, cap=25):
    from agent.app.idea_policies.action_constants import ActionResultExtractor

    sources = []
    seen = set()
    for node in nodes:
        if node.details.get(DetailKey.ACTION.value) != IdeaActionType.VISIT.value:
            continue
        ar = node.details.get(DetailKey.ACTION_RESULT.value)
        if not ar or not isinstance(ar, dict) or not ActionResultExtractor.is_success(ar):
            continue
        url = (ar.get("url") or "").strip()
        if not url:
            continue
        key = _legacy_key(url)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"url": url, "title": (ar.get("title") or "").strip()})
        if len(sources) >= cap:
            break
    return sources


def test_visited_sources_output_is_byte_identical_to_the_inline_version():
    assert _visited_sources(_Graph(_MIXED)) == _legacy_visited_sources(_MIXED)


def test_visited_sources_still_dedupes_and_keeps_first_seen_order():
    assert _visited_sources(_Graph(_MIXED)) == [
        {"url": "https://e.x/a", "title": "A"},
        {"url": "https://e.x/b", "title": "B"},
    ]


def test_visited_sources_cap_is_honoured():
    nodes = [_visit(f"https://e.x/{i}") for i in range(40)]
    assert len(_visited_sources(_Graph(nodes), cap=25)) == 25


def test_superseded_nodes_stay_out_of_content_but_stay_in_sources():
    """Pins the documented asymmetry this change deliberately does NOT touch."""
    from agent.app.idea_policies.base import DetailKey as DK

    node = _visit("https://e.x/superseded", content="GUESS BODY")
    node.details[DK.FALLBACK_SUPERSEDED.value] = True
    g = _Graph([node])

    assert "GUESS BODY" not in _collect_all_visit_content(g)
    assert _visited_sources(g) == [{"url": "https://e.x/superseded", "title": ""}]


def test_an_empty_refetch_does_not_block_the_copy_that_has_content():
    """A page is only 'seen' once a section for it was actually emitted."""
    g = _Graph([
        _visit("https://e.x/a", content=""),
        _visit("https://e.x/a", content="REAL BODY"),
    ])

    assert "REAL BODY" in _collect_all_visit_content(g)
