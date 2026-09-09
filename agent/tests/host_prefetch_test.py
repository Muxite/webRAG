"""`agent/app/host_prefetch.py` -- host-side prefetch of a mandate's uncovered slot entities.

Everything here is offline: `http` and `search` are fakes with the two connector methods the
module is allowed to call (`request(method, url, **kw)` -> `.data`/`.error`; `query_search(q,
count)`), and the mandates are the REAL task statements (via `host_derive_test.statement`) so a
prefetch that satisfies `LedgerToolkit.host_derive` here satisfies it live.

The picker cases at the bottom reproduce two LIVE Wikipedia search-API probes (2026-09-08):
`"Mississippi basin area"` ranks the river first, but the bare entity ranks the STATE first;
`"Amazon basin size"` ranks `Amazon` and `Amazon basin` above `Amazon River`. A title-only picker
gets both wrong; the verification picker reads the fetched infoboxes and gets both right.
"""
from __future__ import annotations

import asyncio
import os
import sys
from urllib.parse import parse_qs, quote, urlparse

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from host_derive_test import RIVERS, _river_page, _wiki, statement  # noqa: E402

from agent.app import host_prefetch as hp  # noqa: E402
from agent.app.host_prefetch import PREFETCH_SOURCE, Resolution, host_prefetch  # noqa: E402
from agent.app.ledger_tools import LedgerToolkit  # noqa: E402
from agent.app.mandate_slots import parse_slots  # noqa: E402
from agent.app.testing.execution_evidence_loop import hash_page_text  # noqa: E402


# --------------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------------

class _Result:
    def __init__(self, status, data, error=False):
        self.status, self.data, self.error = status, data, error


class FakeHttp:
    """`connector_http.request` stand-in: Wikipedia search-API JSON per query, HTML per URL."""

    def __init__(self, pages=None, api=None, raise_on=()):
        self.pages = dict(pages or {})       # url -> html
        self.api = dict(api or {})           # srsearch query -> [(title, snippet)]
        self.raise_on = set(raise_on)
        self.calls = []

    async def request(self, method, url, retries=2, **kwargs):
        self.calls.append((method, url))
        if url in self.raise_on:
            raise RuntimeError("boom")
        if url.startswith(hp.WIKI_API):
            query = parse_qs(urlparse(url).query).get("srsearch", [""])[0]
            hits = self.api.get(query, [])
            return _Result(200, {"query": {"search": [{"title": t, "snippet": s}
                                                      for t, s in hits]}})
        if url in self.pages:
            return _Result(200, self.pages[url])
        return _Result(404, "not found", error=True)

    @property
    def api_calls(self):
        return [u for _, u in self.calls if u.startswith(hp.WIKI_API)]

    @property
    def page_calls(self):
        return [u for _, u in self.calls if not u.startswith(hp.WIKI_API)]


class FakeSearch:
    def __init__(self, results=None):
        self.results = list(results or [])
        self.calls = []

    async def query_search(self, query, count=10):
        self.calls.append((query, count))
        return self.results


def _article(title):
    return hp.WIKI_ARTICLE + quote(title.replace(" ", "_"), safe="()_,'-.:")


def _river_html(name, length, basin, lead=None):
    return (f"<html><body><table class='infobox'>"
            f"<tr><th>Length</th><td>{length} km</td></tr>"
            f"<tr><th>Basin size</th><td>{basin} km<sup>2</sup></td></tr></table>"
            f"<p>{lead or f'The {name} is a major river.'}</p>"
            f"<p>It flows through several countries.</p></body></html>")


def _infobox_html(rows, lead="Some lead paragraph."):
    body = "".join(f"<tr><th>{label}</th><td>{value}</td></tr>" for label, value in rows)
    return f"<html><body><table class='infobox'>{body}</table><p>{lead}</p></body></html>"


def _run(coro):
    return asyncio.run(coro)


def _pages(kit):
    return kit.artifact()["pages"]


def _prefetched(kit):
    return [p for p in _pages(kit) if p.get("source") == PREFETCH_SOURCE]


# --------------------------------------------------------------------------------------------
# Structural budget
# --------------------------------------------------------------------------------------------

def test_every_slot_entity_is_fetched_even_when_the_model_already_visited_its_page():
    """G1 forensics: the model's registered page is the flattened window and can be missing a
    dual-unit cell's second unit, so the host copy is registered anyway -- as a NEW page, leaving
    the model's page byte-identical -- and the row says `model_visited`."""
    kit = LedgerToolkit()
    for name, slug, length, basin in RIVERS:
        kit.register_page(_wiki(slug), _river_page(name, length, basin))
    before = kit.artifact()["pages"]
    api, pages = _rivers_world()
    http, search = FakeHttp(pages=pages, api=api), FakeSearch()

    out = _run(host_prefetch(kit, statement("218"), http=http, search=search))

    assert out["uncovered_before"] == 0
    assert out["searches"] == 5 and out["registered"] == 5
    assert out["fetches"] == 10   # every acceptable candidate is read: two per entity
    assert search.calls == [] and out["error"] is None
    assert [row["status"] for row in out["entities"]] == ["prefetched"] * 5
    assert all(row["model_visited"] is True for row in out["entities"])
    after = kit.artifact()["pages"]
    assert after[:5] == before                      # model-read pages untouched
    assert [p["source"] for p in after[5:]] == [PREFETCH_SOURCE] * 5
    assert kit.host_derive(statement("218"))["winner_entity"] == "Mekong"


def test_a_mandate_with_fewer_than_two_slots_returns_early_without_touching_anything():
    kit = LedgerToolkit()
    http = FakeHttp()
    out = _run(host_prefetch(kit, "What is the height of the Eiffel Tower?", http=http,
                             search=FakeSearch()))
    assert out["entities"] == [] and http.calls == []
    assert out["registered"] == 0 and out["error"] is None
    assert _pages(kit) == []


def _rivers_world():
    """Fake web for 218: bare-entity API hits (the top hit IS the article, as the live probe
    showed for Yangtze/Nile/Mekong) and one river article per entity."""
    api, pages = {}, {}
    for name, slug, length, basin in RIVERS:
        title = slug.replace("_", " ")
        api[name] = [(title, f"The {name} is a river"), (f"{title} Delta", "delta")]
        pages[_article(title)] = _river_html(name, length, basin)
        pages[_article(f"{title} Delta")] = _infobox_html([("Area", "40,000 km<sup>2</sup>")])
    return api, pages


def test_218_with_only_mekong_registered_prefetches_all_five_and_host_derive_computes():
    kit = LedgerToolkit()
    name, slug, length, basin = RIVERS[0]
    kit.register_page(_wiki(slug), _river_page(name, length, basin))   # model-visited Mekong
    api, pages = _rivers_world()
    http, search = FakeHttp(pages=pages, api=api), FakeSearch()

    out = _run(host_prefetch(kit, statement("218"), http=http, search=search))

    assert out["error"] is None
    assert out["uncovered_before"] == 4
    assert [row["status"] for row in out["entities"]] == ["prefetched"] * 5
    assert [row["model_visited"] for row in out["entities"]] == [True, False, False, False, False]
    assert out["searches"] == 5 and len(http.api_calls) == 5   # exactly one resolve per entity
    # Both API hits per entity are read (no exact-title short-circuit -- see the resolver
    # docstring for the Mississippi-state replay that removed it); the better infobox wins.
    assert out["fetches"] == 10 and len(http.page_calls) == 10
    assert out["registered"] == 5 and search.calls == []
    assert [p["url"] for p in _prefetched(kit)] == [_wiki(s) for _, s, _, _ in RIVERS]
    assert all(p["source"] == PREFETCH_SOURCE for p in _prefetched(kit))
    assert "source" not in _pages(kit)[0]   # the model-read page is byte-identical

    # Availability -- this lane's deliverable: every one of the ten operands now resolves, on the
    # host copies for the four rivers the model never visited.
    result = kit.host_derive(statement("218"))
    selected = [row for row in result["slots"] if row["reason"] == "selected"]
    assert len(selected) == 10 and len({row["entity"] for row in selected}) == 5
    assert result["reason"] not in ("incomplete_roster", "operand_not_found", "no_pages")


def test_218_mixed_model_window_and_host_copies_computes_the_right_winner():
    kit = LedgerToolkit()
    name, slug, length, basin = RIVERS[0]
    kit.register_page(_wiki(slug), _river_page(name, length, basin))
    api, pages = _rivers_world()
    _run(host_prefetch(kit, statement("218"), http=FakeHttp(pages=pages, api=api),
                       search=FakeSearch()))

    result = kit.host_derive(statement("218"))
    assert result["reason"] == "computed"
    assert result["winner_entity"] == "Mekong"
    assert result["value"] == pytest.approx(4909 / 795_000, abs=1e-6)  # host rounds to 6 dp


def test_the_stored_page_is_the_full_text_and_its_hash_matches():
    kit = LedgerToolkit(max_page_chars=50)   # a tiny default window must not clip a prefetch
    api, pages = _rivers_world()
    long_lead = "The Yangtze is the longest river in Asia. " * 40
    pages[_article("Yangtze")] = _river_html("Yangtze", "6,300", "1,800,000", lead=long_lead)
    http = FakeHttp(pages=pages, api=api)

    out = _run(host_prefetch(kit, statement("218"), http=http, search=FakeSearch()))

    page = next(p for p in _pages(kit) if p["url"].endswith("/Yangtze"))
    row = next(r for r in out["entities"] if r["entity"] == "Yangtze")
    assert page["truncated"] is False
    assert page["stored_chars"] == page["chars"] == len(page["text"]) == row["chars"]
    assert page["chars"] > 50
    assert page["content_hash"] == hash_page_text(page["text"])
    assert page["text"].startswith("Length: 6,300 km\nBasin size: 1,800,000 km²\n")


def test_infobox_entries_are_first_in_the_page_index_and_their_offsets_resolve():
    kit = LedgerToolkit()
    api, pages = _rivers_world()
    http = FakeHttp(pages=pages, api=api)
    _run(host_prefetch(kit, statement("218"), http=http, search=FakeSearch()))

    page_id, page = next((pid, p) for pid, p in
                         ((f"p{i + 1}", p) for i, p in enumerate(_pages(kit)))
                         if p["url"].endswith("/Nile"))
    rendered = kit.page_index_text(page_id)
    first_two = rendered.split("\n")[:2]
    assert first_two[0].endswith("Length = 6,650 km")
    assert first_two[1].endswith("Basin size = 3,254,555 km²")
    entries = kit._indexes[page_id]
    assert entries[0].source == "infobox" and entries[0].label == "Length"
    assert page["text"][entries[0].start:entries[0].end] == "6,650"
    assert page["text"][entries[1].start:entries[1].end] == "3,254,555"


def test_two_slots_of_one_entity_cost_one_resolve_and_one_fetch():
    """216 asks two fields of the Tōkaidō Shinkansen: one entity, one row, one page."""
    kit = LedgerToolkit()
    slots = parse_slots(statement("216"))
    assert len(slots) == 2 and len({s.entity for s in slots}) == 1
    entity = slots[0].entity
    html = _infobox_html([("Line length", "515.4 km"), ("Journey time", "2.35 h")],
                         lead=f"The {entity} is a high-speed rail line.")
    http = FakeHttp(pages={_article(entity): html}, api={entity: [(entity, "rail line")]})

    out = _run(host_prefetch(kit, statement("216"), http=http, search=FakeSearch()))

    assert len(out["entities"]) == 1 and out["entities"][0]["status"] == "prefetched"
    assert out["searches"] == 1 and out["fetches"] == 1 and out["registered"] == 1


# --------------------------------------------------------------------------------------------
# Failure modes -- rows, never exceptions
# --------------------------------------------------------------------------------------------

def test_no_acceptable_page_anywhere_is_a_no_hit_row_after_the_web_fallback():
    kit = LedgerToolkit()
    http, search = FakeHttp(), FakeSearch(results=[{"url": "https://example.com/x"}])

    out = _run(host_prefetch(kit, statement("210"), http=http, search=search))

    assert [r["status"] for r in out["entities"]] == ["no_hit", "no_hit"]
    # "GRES-2 Power Station chimney": full, 3-token, 2-token API queries + web = 4;
    # "Inco Superstack": full API query (already 2 tokens, nothing to trim) + web = 2.
    assert out["searches"] == 6
    assert out["fetches"] == 0 and out["registered"] == 0 and out["error"] is None
    assert len(search.calls) == 2 and all(q.endswith(" wikipedia") for q, _ in search.calls)


def test_http_raising_never_raises_and_error_is_not_set_for_a_per_entity_failure():
    kit = LedgerToolkit()

    class Boom:
        async def request(self, *a, **kw):
            raise RuntimeError("network down")

    out = _run(host_prefetch(kit, statement("210"), http=Boom(), search=FakeSearch()))

    assert out["error"] is None
    assert [r["status"] for r in out["entities"]] == ["no_hit", "no_hit"]
    assert out["registered"] == 0


def test_a_failure_inside_the_kits_own_hook_is_the_outer_error():
    class BrokenKit(LedgerToolkit):
        def entities_without_page(self, slots):
            raise ValueError("kit broke")

    out = _run(host_prefetch(BrokenKit(), statement("210"), http=FakeHttp(), search=FakeSearch()))
    assert out["error"].startswith("ValueError: kit broke")
    assert out["registered"] == 0


def test_a_resolver_url_that_fails_to_fetch_is_a_fetch_failed_row():
    kit = LedgerToolkit()

    async def resolver(entity, phrase, *, http, search, **kw):
        return "https://en.wikipedia.org/wiki/Missing_Page"

    out = _run(host_prefetch(kit, statement("210"), http=FakeHttp(), search=FakeSearch(),
                             resolver=resolver))
    assert [r["status"] for r in out["entities"]] == ["fetch_failed", "fetch_failed"]
    assert out["fetches"] == 2 and out["registered"] == 0


def test_a_fetched_page_with_neither_infobox_nor_the_entity_in_its_lead_is_rejected():
    kit = LedgerToolkit()
    url = "https://en.wikipedia.org/wiki/Something_Else"

    async def resolver(entity, phrase, *, http, search, **kw):
        return url

    http = FakeHttp(pages={url: "<html><body><p>An unrelated error page.</p></body></html>"})
    out = _run(host_prefetch(kit, statement("210"), http=http, search=FakeSearch(),
                             resolver=resolver))
    assert [r["status"] for r in out["entities"]] == ["fetch_failed", "fetch_failed"]
    assert _pages(kit) == []


def test_two_entities_resolving_to_the_same_url_register_one_page():
    kit = LedgerToolkit()
    url = "https://en.wikipedia.org/wiki/Shared_Article"
    html = _infobox_html([("Height", "419.7 m")], lead="Inco Superstack and GRES-2 chimney.")

    async def resolver(entity, phrase, *, http, search, **kw):
        return Resolution(url=url, html=html, searches=1, fetches=1)

    out = _run(host_prefetch(kit, statement("210"), http=FakeHttp(), search=FakeSearch(),
                             resolver=resolver))
    statuses = [r["status"] for r in out["entities"]]
    assert statuses == ["prefetched", "duplicate_url"]
    assert out["registered"] == 1 and len(_pages(kit)) == 1


def test_a_slot_that_names_a_wikipedia_url_is_fetched_directly_without_a_search():
    kit = LedgerToolkit()
    mandate = ("Read two pages and compute the absolute difference in HEIGHT, in metres:\n"
               "  1. GRES-2 Power Station chimney (https://en.wikipedia.org/wiki/Ekibastuz_GRES-2_Power_Station)\n"
               "  2. Inco Superstack (https://en.wikipedia.org/wiki/Inco_Superstack)\n")
    slots = parse_slots(mandate)
    assert [s.url for s in slots] == ["https://en.wikipedia.org/wiki/Ekibastuz_GRES-2_Power_Station",
                                      "https://en.wikipedia.org/wiki/Inco_Superstack"]
    pages = {slots[0].url: _infobox_html([("Height", "419.7 m")], lead="GRES-2 Power Station."),
             slots[1].url: _infobox_html([("Height", "380 m")], lead="Inco Superstack.")}
    http = FakeHttp(pages=pages)

    out = _run(host_prefetch(kit, mandate, http=http, search=FakeSearch()))

    assert [r["status"] for r in out["entities"]] == ["prefetched", "prefetched"]
    assert out["searches"] == 0 and http.api_calls == [] and out["fetches"] == 2


# --------------------------------------------------------------------------------------------
# The resolver's picker -- verification, not titles
# --------------------------------------------------------------------------------------------

#: The real state infobox shape: a title row, then an `Area` section header over `• Total`, so
#: the quantity-bearing label is "Area • Total" -- which shares the token `area` with the 218
#: field phrase. That is exactly what fooled the exact-title short-circuit live.
STATE_HTML = ("<html><body><table class='infobox'>"
              "<tr><th class='infobox-above' colspan='2'>Mississippi</th></tr>"
              "<tr><th colspan='2'>Area</th></tr>"
              "<tr><th>• Total</th><td>48,430 sq mi (125,443 km<sup>2</sup>)</td></tr>"
              "<tr><th colspan='2'>Population</th></tr>"
              "<tr><th>• Total</th><td>2,961,279</td></tr>"
              "</table><p>Mississippi is a state.</p></body></html>")
MS_RIVER_HTML = _infobox_html([("Length", "2,340 mi (3,766 km)"),
                               ("Basin size", "1,151,000 sq mi (2,980,000 km<sup>2</sup>)")],
                              lead="The Mississippi River is the primary river of the largest "
                                   "drainage basin in the United States.")
MS_SYSTEM_HTML = _infobox_html([("Length", "3,766 km")], lead="The Mississippi River System.")
AMAZON_HTML = "<html><body><p>Amazon most often refers to: the Amazon River; the Amazon rainforest; Amazon (company).</p></body></html>"
AMAZON_BASIN_HTML = _infobox_html([("Area", "7,000,000 km<sup>2</sup>")], lead="The Amazon basin.")
RAINFOREST_HTML = _infobox_html([("Area", "5,500,000 km<sup>2</sup>")], lead="The Amazon rainforest.")
AMAZON_RIVER_HTML = _infobox_html([("Length", "6,400 km"), ("Basin size", "7,000,000 km<sup>2</sup>")],
                                  lead="The Amazon River in South America.")
CONGO_HTML = _infobox_html([("Area", "4,000,000 km<sup>2</sup>")], lead="The Congo Basin.")


def _resolve(entity, phrase, http, search=None):
    return _run(hp.resolve_entity_page(entity, phrase, http=http, search=search or FakeSearch()))


def test_live_probe_mississippi_basin_area_picks_the_river_not_a_basin_or_list_page():
    """`srsearch="Mississippi basin area"` -> River, River System, Atchafalaya Basin, Nitrate...,
    List of drainage basins by area (live, 2026-09-08)."""
    hits = [("Mississippi River", "primary river of the largest drainage basin"),
            ("Mississippi River System", "system"), ("Atchafalaya Basin", "basin"),
            ("Nitrate in the Mississippi River Basin", "nitrate"),
            ("List of drainage basins by area", "list")]
    http = FakeHttp(api={"Mississippi": hits},
                    pages={_article("Mississippi River"): MS_RIVER_HTML,
                           _article("Mississippi River System"): MS_SYSTEM_HTML})
    resolved = _resolve("Mississippi", "basin area", http)
    assert resolved.url == _article("Mississippi River")
    assert not any("List_of" in u or "Atchafalaya" in u for u in http.page_calls)


def test_bare_entity_ranking_the_state_first_still_resolves_to_the_river_by_reading_infoboxes():
    """The bare-name API order (live): Mississippi (state), Mississippi River, (disambiguation),
    Jackson, Mississippi, ... The state's infobox has no row naming the field, so it is skipped."""
    hits = [("Mississippi", "state"), ("Mississippi River", "river"),
            ("Mississippi (disambiguation)", "may refer to"), ("Jackson, Mississippi", "city")]
    http = FakeHttp(api={"Mississippi": hits},
                    pages={_article("Mississippi"): STATE_HTML,
                           _article("Mississippi River"): MS_RIVER_HTML,
                           _article("Jackson, Mississippi"): _infobox_html([("Area", "113 sq mi")])})
    # With the REAL 218 phrase (length + drainage basin size / basin area): the state's
    # "Area • Total" covers `area` (1), the river's Length + Basin size cover 3. Every acceptable
    # candidate is read (the disambiguation page never is). Note the bare phrase "basin area"
    # alone would be a 1-1 tie (`area` vs `basin`) that the exact-title tie-break gives to the
    # state -- which is why the host passes every phrase the mandate asks of the entity.
    phrase = parse_slots(statement("218"))[3].field_phrase
    resolved = _resolve("Mississippi", phrase, http)
    assert resolved.url == _article("Mississippi River")
    assert resolved.fetches == 3 and resolved.searches == 1
    # Order among the three is the snippet tie-break (the river's "river" snippet shares a token
    # with the phrase); what matters is that all three were read and the disambiguation never.
    assert set(http.page_calls) == {_article("Mississippi"), _article("Mississippi River"),
                                    _article("Jackson, Mississippi")}


def test_live_probe_amazon_basin_size_picks_amazon_river_over_amazon_and_amazon_basin():
    """`srsearch="Amazon basin size"` -> Amazon, Congo Basin, Amazon basin, Amazon rainforest,
    Amazon River (live, 2026-09-08). Title-only picking takes `Amazon`; verification does not."""
    hits = [("Amazon", "may refer to"), ("Congo Basin", "congo"), ("Amazon basin", "basin"),
            ("Amazon rainforest", "forest"), ("Amazon River", "river")]
    http = FakeHttp(api={"Amazon": hits},
                    pages={_article("Amazon"): AMAZON_HTML, _article("Congo Basin"): CONGO_HTML,
                           _article("Amazon basin"): AMAZON_BASIN_HTML,
                           _article("Amazon rainforest"): RAINFOREST_HTML,
                           _article("Amazon River"): AMAZON_RIVER_HTML})
    resolved = _resolve("Amazon", "basin size", http)
    assert resolved.url == _article("Amazon River")
    assert _article("Congo_Basin".replace("_", " ")) not in http.page_calls  # no entity token


def test_a_rival_page_sharing_one_field_token_loses_to_the_page_covering_more():
    """With the real 218 phrase ("... LENGTH ... DRAINAGE BASIN SIZE / basin area ...") the
    rainforest's `Area` row shares a token, but the river covers length + basin + size."""
    phrase = parse_slots(statement("218"))[4].field_phrase
    hits = [("Amazon rainforest", "forest"), ("Amazon River", "river")]
    http = FakeHttp(api={"Amazon": hits},
                    pages={_article("Amazon rainforest"): RAINFOREST_HTML,
                           _article("Amazon River"): AMAZON_RIVER_HTML})
    resolved = _resolve("Amazon", phrase, http)
    assert resolved.url == _article("Amazon River")
    assert resolved.fetches == 2   # both were read; the better one won, not the first one


def test_an_exact_title_only_breaks_a_tie_it_never_short_circuits_the_sweep():
    hits = [("Nile", "river"), ("Nile Delta", "delta"), ("Nile crocodile", "croc")]
    http = FakeHttp(api={"Nile": hits},
                    pages={_article("Nile"): _river_html("Nile", "6,650", "3,254,555"),
                           _article("Nile Delta"): _river_html("Nile Delta", "1", "240")})
    resolved = _resolve("Nile", "basin area", http)
    assert resolved.url == _article("Nile") and resolved.fetches == 3   # all three read
    # Reverse the rank: the exact title still wins the tie over an equally-scored earlier hit.
    http = FakeHttp(api={"Nile": list(reversed(hits))}, pages=http.pages)
    assert _resolve("Nile", "basin area", http).url == _article("Nile")


def test_the_state_beats_nothing_when_the_mandate_asks_two_fields_the_river_page_carries():
    """The coordinator's replay case, with the entity's TWO field phrases passed together:
    Mississippi (state) covers `area` once; Mississippi River covers `length` and `basin`."""
    hits = [("Mississippi", "state"), ("Mississippi River", "river")]
    http = FakeHttp(api={"Mississippi": hits},
                    pages={_article("Mississippi"): STATE_HTML,
                           _article("Mississippi River"): MS_RIVER_HTML})
    resolved = _run(hp.resolve_entity_page(
        "Mississippi", http=http, search=FakeSearch(),
        field_phrases=["length in METRES", "basin area in km^2"]))
    assert resolved.url == _article("Mississippi River")
    # And with the single real 218 phrase, the same outcome.
    phrase = parse_slots(statement("218"))[3].field_phrase
    http = FakeHttp(api={"Mississippi": hits}, pages=http.pages)
    assert _resolve("Mississippi", phrase, http).url == _article("Mississippi River")


def test_disambiguation_list_and_namespace_titles_are_never_fetched():
    hits = [("Nile (disambiguation)", "may refer to"), ("List of rivers by length", "list"),
            ("Category:Nile", "cat"), ("File:Nile.jpg", "file"), ("Nile", "river")]
    http = FakeHttp(api={"Nile": hits},
                    pages={_article("Nile"): _river_html("Nile", "6,650", "3,254,555")})
    resolved = _resolve("Nile", "basin area", http)
    assert resolved.url == _article("Nile")
    assert http.page_calls == [_article("Nile")]


def test_web_search_fallback_applies_the_same_picker_to_en_wikipedia_urls():
    http = FakeHttp(api={"Amazon": []},
                    pages={_article("Amazon River"): AMAZON_RIVER_HTML,
                           _article("Amazon rainforest"): RAINFOREST_HTML})
    search = FakeSearch(results=[
        {"url": "https://www.britannica.com/place/Amazon-River", "description": "x"},
        {"url": "https://en.wikipedia.org/wiki/Amazon_rainforest", "description": "forest"},
        {"url": "https://en.wikipedia.org/wiki/Amazon_River", "description": "river"},
    ])
    resolved = _resolve("Amazon", "basin size", http, search)
    assert resolved.url == _article("Amazon River")
    assert resolved.searches == 2
    assert search.calls == [("Amazon basin size wikipedia", 5)]


def test_resolver_is_deterministic_for_identical_inputs():
    hits = [("Amazon rainforest", "forest"), ("Amazon River", "river")]
    pages = {_article("Amazon rainforest"): RAINFOREST_HTML,
             _article("Amazon River"): AMAZON_RIVER_HTML}
    first = _resolve("Amazon", "basin size", FakeHttp(api={"Amazon": hits}, pages=pages))
    second = _resolve("Amazon", "basin size", FakeHttp(api={"Amazon": hits}, pages=pages))
    assert first == second
    api_url = FakeHttp(api={"Amazon": hits}, pages=pages)
    _resolve("Amazon", "basin size", api_url)
    assert api_url.api_calls[0] == (
        hp.WIKI_API + "?action=query&list=search&srsearch=Amazon&format=json&srlimit=10")


def test_resolver_reports_a_miss_as_an_empty_url_with_its_cost_counted():
    miss = _resolve("Amazon", "basin size", FakeHttp(api={"Amazon": []}))
    assert miss.url == "" and miss.html == ""
    assert miss.searches == 2 and miss.fetches == 0   # API, then the web fallback
    assert _resolve("Amazon", "basin size", None).url == ""


def test_entity_is_trimmed_of_its_last_token_when_the_full_name_finds_nothing():
    """210: the mandate names "GRES-2 Power Station chimney"; the article is the power station."""
    html = _infobox_html([("Height", "419.7 m (1,377 ft)")], lead="The Ekibastuz GRES-2 Power Station.")
    http = FakeHttp(api={"GRES-2 Power Station chimney": [],
                         "GRES-2 Power Station": [("Ekibastuz GRES-2 Power Station", "station")]},
                    pages={_article("Ekibastuz GRES-2 Power Station"): html})
    search = FakeSearch()
    resolved = _resolve("GRES-2 Power Station chimney", "its HEIGHT, in meters", http, search)
    assert resolved.url == _article("Ekibastuz GRES-2 Power Station")
    assert resolved.searches == 2 and resolved.fetches == 1
    assert search.calls == []
    assert [parse_qs(urlparse(u).query)["srsearch"][0] for u in http.api_calls] == [
        "GRES-2 Power Station chimney", "GRES-2 Power Station"]


def test_trimming_stops_at_two_tokens_then_falls_back_to_the_web_search():
    http = FakeHttp()
    search = FakeSearch()
    resolved = _resolve("GRES-2 Power Station chimney", "its HEIGHT, in meters", http, search)
    assert resolved.url == ""
    assert [parse_qs(urlparse(u).query)["srsearch"][0] for u in http.api_calls] == [
        "GRES-2 Power Station chimney", "GRES-2 Power Station", "GRES-2 Power"]
    assert resolved.searches == 4 and len(search.calls) == 1


# -- 221 replay: nested rows on the real page vs flat rows on a lesser page -------------------

def _tower_html(title, nested):
    height = ("<tr><th colspan='2'>Height</th></tr><tr><th>Architectural</th><td>828 m (2,717 ft)</td></tr>"
              "<tr><th>Tip</th><td>829.8 m</td></tr>" if nested else
              "<tr><th>Height</th><td>725 m (2,379 ft)</td></tr>")
    return (f"<html><body><table class='infobox'><tr><th class='infobox-above' colspan='2'>{title}</th></tr>"
            f"{height}<tr><th colspan='2'>Technical details</th></tr>"
            f"<tr><th>Floor count</th><td>163</td></tr></table><p>{title} is a skyscraper.</p></body></html>")


def test_burj_khalifa_exact_title_with_nested_height_beats_burj_azizi_flat_height():
    hits = [("Burj Azizi", "tower"), ("Burj Khalifa", "tower")]
    http = FakeHttp(api={"Burj Khalifa": hits},
                    pages={_article("Burj Azizi"): _tower_html("Burj Azizi", nested=False),
                           _article("Burj Khalifa"): _tower_html("Burj Khalifa", nested=True)})
    resolved = _run(hp.resolve_entity_page(
        "Burj Khalifa", http=http, search=FakeSearch(),
        field_phrases=["its ARCHITECTURAL HEIGHT, in metres", "its FLOOR COUNT"]))
    assert resolved.url == _article("Burj Khalifa")
    assert resolved.fetches == 2   # the exact-title candidate is read, never pruned


def test_shanghai_tower_exact_title_beats_jin_mao_tower_with_a_flat_height_row():
    hits = [("Jin Mao Tower", "tower"), ("Shanghai Tower", "tower"), ("Shanghai World Financial Center", "x")]
    http = FakeHttp(api={"Shanghai Tower": hits},
                    pages={_article("Jin Mao Tower"): _tower_html("Jin Mao Tower", nested=False),
                           _article("Shanghai Tower"): _tower_html("Shanghai Tower", nested=True),
                           _article("Shanghai World Financial Center"): _tower_html("SWFC", nested=False)})
    resolved = _run(hp.resolve_entity_page(
        "Shanghai Tower", http=http, search=FakeSearch(),
        field_phrases=["its ARCHITECTURAL HEIGHT, in metres", "its FLOOR COUNT"]))
    assert resolved.url == _article("Shanghai Tower")


def test_a_candidate_covering_strictly_more_field_phrases_beats_the_exact_title():
    """The Mississippi rule, spelled out on the phrase count: the state covers only the `area`
    phrase, the river covers both."""
    hits = [("Mississippi", "state"), ("Mississippi River", "river")]
    http = FakeHttp(api={"Mississippi": hits},
                    pages={_article("Mississippi"): STATE_HTML,
                           _article("Mississippi River"): MS_RIVER_HTML})
    assert hp._field_coverage(STATE_HTML, ["length in METRES", "basin area in km^2"]) == (1, 1)
    assert hp._field_coverage(MS_RIVER_HTML, ["length in METRES", "basin area in km^2"]) == (2, 2)
    resolved = _run(hp.resolve_entity_page("Mississippi", http=http, search=FakeSearch(),
                                           field_phrases=["length in METRES", "basin area in km^2"]))
    assert resolved.url == _article("Mississippi River")


# ---------------------------------------------------------------------------------------------
# Field coverage ranks candidates; it must not veto them (2026-09-09, task 210).
# `_field_coverage` reads INFOBOX rows only, so an article whose asked-for fact is prose scores
# (0, 0) however plainly it is the right page. Requiring coverage returned `no_hit` for every
# task-210 cell even though `Ekibastuz GRES-2 Power Station` was the top-ranked candidate.
# ---------------------------------------------------------------------------------------------

_PROSE_ONLY_HTML = (
    "<html><body><p>The Ekibastuz GRES-2 Power Station has the world's tallest flue-gas "
    "stack at 419.7 metres (1,377 ft) tall.</p></body></html>"
)
_COVERED_HTML = (
    "<html><body><table class='infobox'><tr><th>Height</th><td>380 m</td></tr></table>"
    "<p>A smokestack.</p></body></html>"
)


def _fake_http(pages):
    class _Http:
        async def request(self, method, url, **kwargs):
            return type("R", (), {"status": 200, "text": pages.get(url, "")})()
    return _Http()


def test_the_best_named_candidate_is_taken_when_no_candidate_shows_field_coverage(monkeypatch):
    """The 210 fix: a prose-only right page beats returning nothing at all."""
    import asyncio
    from agent.app import host_prefetch as HP

    cands = HP._candidates([("Ekibastuz GRES-2 Power Station", "")],
                           "GRES-2 Power Station", "the height of its flue-gas chimney, in meters")
    assert cands, "candidate list must be non-empty for this test to mean anything"

    async def fake_fetch(http, url):
        return _PROSE_ONLY_HTML

    monkeypatch.setattr(HP, "_fetch", fake_fetch)
    picked, _fetches = asyncio.run(
        HP._verify(None, cands, ["the height of its flue-gas chimney, in meters"]))
    assert picked is not None, "a prose-only page must resolve, not return no_hit"
    assert "Ekibastuz" in picked[0]


def test_a_candidate_with_field_coverage_still_beats_one_without(monkeypatch):
    """The fallback must change ONLY the all-zero case: coverage still decides when it exists."""
    import asyncio
    from agent.app import host_prefetch as HP

    covered = HP._Candidate(title="Inco Superstack", snippet="", rank=1, coverage=1.0,
                            parenthetical=False, snippet_hits=0, exact=True)
    bare = HP._Candidate(title="Some Other Stack", snippet="", rank=0, coverage=1.0,
                         parenthetical=False, snippet_hits=0, exact=False)

    async def fake_fetch(http, url):
        return _COVERED_HTML if "Inco" in url else _PROSE_ONLY_HTML

    monkeypatch.setattr(HP, "_fetch", fake_fetch)
    picked, _ = asyncio.run(HP._verify(None, [bare, covered], ["its height, in meters"]))
    assert picked is not None and "Inco" in picked[0]


def test_no_candidates_at_all_is_still_a_miss():
    import asyncio
    from agent.app import host_prefetch as HP

    picked, fetches = asyncio.run(HP._verify(None, [], ["its height, in meters"]))
    assert picked is None and fetches == 0


def test_a_title_naming_the_whole_entity_beats_a_verbose_impostor(monkeypatch):
    """Task 221's wrong-building bug (2026-09-09). Both pages cover the same one field phrase, so
    the tie fell to the TOKEN SUM -- and Burj Azizi's flat `Height` + `Observatory height` summed
    4 matching tokens against Burj Khalifa's nested `Height -> Architectural` 3. "Burj Khalifa"
    therefore resolved to wiki/Burj_Azizi, and 221 computed a floor height off the wrong building
    while still naming the right winner, so nothing downstream noticed."""
    import asyncio
    from agent.app import host_prefetch as HP

    khalifa = _infobox_html([("Architectural", "828 m"), ("Floor count", "163")])
    azizi = _infobox_html([("Height", "725 m"), ("Observatory height", "700 m"),
                           ("Floor count", "131")])
    hits = [("Burj Azizi", ""), ("Burj Khalifa", "")]
    cands = HP._candidates(hits, "Burj Khalifa", "its architectural HEIGHT and its FLOOR COUNT")

    async def fake_fetch(http, url):
        return khalifa if "Khalifa" in url else azizi

    monkeypatch.setattr(HP, "_fetch", fake_fetch)
    picked, _ = asyncio.run(HP._verify(
        None, cands, ["its architectural HEIGHT and its FLOOR COUNT"]))
    assert picked is not None and "Burj_Khalifa" in picked[0], picked
