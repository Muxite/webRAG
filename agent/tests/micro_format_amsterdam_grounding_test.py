"""
Offline unit tests for the m02/f02 (bad-model lab micro + format-stress tier) grounding checks.

Both tasks pin the same obscure fact on the same page, and both were capped at a
zero-variance score (m02 0.50, f02 0.67) because their ``grounding`` check only matched
one rendering of the page URL while Wikipedia canonicalises ``Amsterdam_Island`` to the
redirect target ``Île Amsterdam``. The redirect slug reaches the validator in three
renderings -- percent-encoded, literal, and JSON-escaped (``extract_final_text`` dumps
dict deliverables with ``ensure_ascii=True``) -- and all three are a correct visit, so all
three must score. Hallucinated non-Wikipedia sources must still fail.
"""
import pytest

from agent.app.idea_tests import test_f02_amsterdam_structured as f02
from agent.app.idea_tests import test_m02_amsterdam_area as m02

_OBS = {"visit": {"count": 1}}

# Every rendering of the page URL a correct visit can produce.
_CITED_OK = [
    "https://en.wikipedia.org/wiki/Amsterdam_Island",       # the slug the task hands out
    "https://en.wikipedia.org/wiki/%C3%8Ele_Amsterdam",     # canonical redirect, percent-encoded
    "https://en.wikipedia.org/wiki/Île_Amsterdam",          # canonical redirect, literal
    "https://en.wikipedia.org/wiki/\\u00cele_Amsterdam",    # canonical redirect, JSON-escaped
    "en.wikipedia.org/wiki/%C3%8Ele_Amsterdam#Geography",   # fragment-suffixed
]

# Real hallucinated sources observed from qwen2.5:7b on f02 -- correct number, invented page.
_CITED_BAD = [
    "https://www.worldatlas.com/articles/which-is-the-smallest-capital-city.html",
    "http://www.gla.ac.uk/research/postgraduate/wgsjdata/amsterdamandmasquaradelaindex/",
    "https://www.statcan.gc.ca/rural-urban/canada-provinces-territories/amsterdam-island.aspx",
]


def _prose(url: str) -> dict:
    return {"output": {"final_deliverable": f"The land area of Amsterdam Island is 56.6 km2. Source: {url}"}}


def _structured(url: str) -> dict:
    # Mirrors the real f02 path: a dict deliverable, ASCII-escaped by extract_final_text.
    return {"output": {"final_deliverable": {
        "entity": "Île Amsterdam", "value": 56.6, "unit": "km2",
        "source_url": url, "is_estimate": False,
    }}}


@pytest.mark.parametrize("url", _CITED_OK)
def test_m02_grounding_accepts_every_rendering_of_the_page_url(url):
    got = m02.validate_grounding(_prose(url), _OBS)
    assert got["passed"] and got["score"] == 1.0, f"{url} -> {got['reason']}"


@pytest.mark.parametrize("url", _CITED_OK)
def test_f02_grounding_accepts_every_rendering_including_escaped_dict_deliverables(url):
    got = f02.validate_grounding(_structured(url), _OBS)
    assert got["passed"] and got["score"] == 1.0, f"{url} -> {got['reason']}"


@pytest.mark.parametrize("url", _CITED_BAD)
def test_grounding_still_rejects_hallucinated_sources(url):
    assert not m02.validate_grounding(_prose(url), _OBS)["passed"]
    assert not f02.validate_grounding(_structured(url), _OBS)["passed"]


def test_grounding_short_circuits_when_keystone_absent():
    """A cited page must not earn grounding credit without the keystone fact."""
    for mod, res in ((m02, _prose(_CITED_OK[0])), (f02, _structured(_CITED_OK[0]))):
        text = res["output"]["final_deliverable"]
        res = {"output": {"final_deliverable": (
            text.replace("56.6", "55") if isinstance(text, str) else {**text, "value": 55})}}
        got = mod.validate_grounding(res, _OBS)
        assert not got["passed"] and got["score"] == 0.0
        assert "keystone" in got["reason"].lower()


def test_m02_and_f02_score_the_same_citation_identically():
    """The two tasks share one fact and one page; their grounding verdicts must not diverge."""
    for url in _CITED_OK + _CITED_BAD:
        assert m02.validate_grounding(_prose(url), _OBS)["passed"] == \
            f02.validate_grounding(_structured(url), _OBS)["passed"], url
