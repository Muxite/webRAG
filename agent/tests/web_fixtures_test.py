"""Regression test for the JSON-body fixture-key fix (2026-08-08).

Before this fix, `make_key` only hashed `(method, url, params)` — a request that carries its query
in a POST JSON body (e.g. Serper's `{"q": ...}`) rather than the querystring had NO way to
differentiate two different queries: every such request hashed to the same key (identical
method+url, empty params), so a fixture-replay run would silently serve the FIRST query's cached
response to every subsequent different query. This is exactly the kind of silent-corruption bug
this project has repeatedly hunted down live; it's caught here offline instead.
"""
from agent.app import web_fixtures


def test_make_key_differs_by_json_body_not_just_params():
    k1 = web_fixtures.make_key("POST", "https://google.serper.dev/search", None, {"q": "apple inc"})
    k2 = web_fixtures.make_key("POST", "https://google.serper.dev/search", None, {"q": "orange inc"})
    assert k1 != k2


def test_make_key_same_body_same_key_regardless_of_dict_order():
    k1 = web_fixtures.make_key("POST", "https://x.test/search", None, {"q": "a", "gl": "us"})
    k2 = web_fixtures.make_key("POST", "https://x.test/search", None, {"gl": "us", "q": "a"})
    assert k1 == k2


def test_make_key_backward_compatible_when_json_body_omitted():
    # Existing (GET+params) callers never pass json_body — key must be identical to calling with
    # json_body explicitly None, so no historical fixture file is invalidated by this change.
    k1 = web_fixtures.make_key("GET", "https://api.search.brave.com/res/v1/web/search", {"q": "x"})
    k2 = web_fixtures.make_key("GET", "https://api.search.brave.com/res/v1/web/search", {"q": "x"}, None)
    assert k1 == k2


def test_make_key_distinguishes_params_from_json_body_requests():
    # A GET+params request and a POST+json request to the same URL/query text must not collide.
    k_params = web_fixtures.make_key("POST", "https://x.test/search", {"q": "a"}, None)
    k_body = web_fixtures.make_key("POST", "https://x.test/search", None, {"q": "a"})
    assert k_params != k_body


def test_save_and_load_roundtrip_with_json_body(tmp_path, monkeypatch):
    monkeypatch.setenv("IDEA_TEST_FIXTURES_DIR", str(tmp_path))
    from shared.request_result import RequestResult

    key = web_fixtures.make_key("POST", "https://google.serper.dev/search", None, {"q": "apple inc"})
    result = RequestResult(status=200, error=False, data={"organic": [{"title": "Apple"}]})
    web_fixtures.save(key, "POST", "https://google.serper.dev/search", None, result, {"q": "apple inc"})

    loaded = web_fixtures.load(key)
    assert loaded is not None
    assert loaded.status == 200
    assert loaded.data == {"organic": [{"title": "Apple"}]}
