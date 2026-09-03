"""Wrapped tool-argument coercion at the AgentIO seam.

The bug: both benchmark arms stringify whatever the model put in the `query`/`url` slot before
it reaches this module (`str(query)` in `idea_policies/actions.py`'s `SearchLeafAction`;
pydantic's `str` coercion at the LangGraph `@tool` boundary in `langgraph_solver.py`), so a
model that wraps a correct argument in a list or dict never arrives at `AgentIO.search()`/
`.visit()` as an actual list/dict object -- it arrives as that object's Python repr baked into a
string, e.g. `"['Lake Matano maximum depth']"`. A search then searches that literal string; a
visit tries to fetch it as a URL and can never succeed.

Reproduced by scanning every `search`/`visit` timing's payload in every stored cell under
`agent/idea_test_results/` (7828 result files):

    158  search queries whose recorded `query` text is shaped like a Python list/dict/tuple
         repr (125 list/tuple, 31 dict, 2 unparseable-but-bracketed).
    61   visit urls whose recorded `url` text unwraps (single-element list/tuple, or a dict's
         `url`/`link`/`href` key) to a syntactically valid http(s) URL.
    89   visit urls recorded as an empty string.

(A tighter "single-element list/tuple only, no multi-element" count for visit gives 54, not 61;
70 counts every wrapped visit url regardless of shape. 61 is reproduced by unwrapping the FIRST
element of the wrapper, whatever its length, and checking that value alone is a valid URL --
which is also exactly this fix's own `_coerce_visit_url`/`is_malformed_url_text` behavior for a
single-element wrapper, and the same "take the first" policy `_coerce_search_query` applies to a
multi-element search list. The search figure needs list/tuple (125) + dict (31) + unparseable-
but-still-bracketed (2) = 158 to match exactly; "searched literally" in the task brief was an
illustrative example, not the full shape of what was measured.)

`AgentIO.search()` and `AgentIO.visit()` are the one seam every arm reaches (graph engine,
sequential ReAct, LangGraph react, evidence-loop), so unwrapping happens here, once. It is
UNWRAPPING ONLY -- never repair: a query/url this cannot confidently resolve to a single value
is passed through completely unchanged, exactly as the model wrote it.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.app.agent_io import (
    AgentIO,
    _coerce_search_query,
    _coerce_visit_url,
    is_malformed_url_text,
)


# --- pure unit tests: _coerce_search_query -------------------------------------------------

def test_coerce_search_query_unwraps_single_element_list_repr_string():
    query, coerced, dropped = _coerce_search_query("['Lake Matano maximum depth']")
    assert query == "Lake Matano maximum depth"
    assert coerced == "list"
    assert dropped is None


def test_coerce_search_query_unwraps_single_element_list_object():
    """Defensive: also handles an actual (unstringified) list, not just its repr text, in case
    a future caller passes the object through directly instead of `str(query)`-ing it first."""
    query, coerced, dropped = _coerce_search_query(["Lake Matano maximum depth"])
    assert query == "Lake Matano maximum depth"
    assert coerced == "list"


def test_coerce_search_query_takes_first_of_multi_element_list_and_records_the_rest():
    query, coerced, dropped = _coerce_search_query(
        "['Lake Matano depth', 'Hornindalsvatnet depth', 'Quesnel Lake depth']"
    )
    assert query == "Lake Matano depth"
    assert coerced == "list_multi"
    assert dropped == ["Hornindalsvatnet depth", "Quesnel Lake depth"]


def test_coerce_search_query_unwraps_dict_with_query_key():
    query, coerced, dropped = _coerce_search_query("{'query': 'mont blanc height'}")
    assert query == "mont blanc height"
    assert coerced == "dict"


def test_coerce_search_query_unwraps_dict_with_q_key():
    query, coerced, dropped = _coerce_search_query("{'q': 'mont blanc height'}")
    assert query == "mont blanc height"
    assert coerced == "dict"


def test_coerce_search_query_dict_without_known_key_stays_as_is():
    raw = "{'topic': 'mont blanc height'}"
    query, coerced, dropped = _coerce_search_query(raw)
    assert query == raw
    assert coerced is None


def test_coerce_search_query_plain_string_unchanged():
    query, coerced, dropped = _coerce_search_query("mont blanc height")
    assert query == "mont blanc height"
    assert coerced is None
    assert dropped is None


def test_coerce_search_query_unparseable_bracketed_text_stays_as_is():
    raw = "['unterminated"
    query, coerced, dropped = _coerce_search_query(raw)
    assert query == raw
    assert coerced is None


# --- pure unit tests: _coerce_visit_url ----------------------------------------------------

def test_coerce_visit_url_unwraps_single_element_list_repr_string():
    url, coerced = _coerce_visit_url("['https://en.wikipedia.org/wiki/Victoria_Falls']")
    assert url == "https://en.wikipedia.org/wiki/Victoria_Falls"
    assert coerced == "list"


def test_coerce_visit_url_unwraps_dict_with_url_key():
    url, coerced = _coerce_visit_url("{'url': 'https://en.wikipedia.org/wiki/Gotthard_Base_Tunnel'}")
    assert url == "https://en.wikipedia.org/wiki/Gotthard_Base_Tunnel"
    assert coerced == "dict"


def test_coerce_visit_url_unwraps_dict_with_link_and_href_keys():
    url, coerced = _coerce_visit_url("{'link': 'https://example.com/a'}")
    assert url == "https://example.com/a"
    assert coerced == "dict"
    url, coerced = _coerce_visit_url("{'href': 'https://example.com/b'}")
    assert url == "https://example.com/b"
    assert coerced == "dict"


def test_coerce_visit_url_multi_element_list_is_not_guessed_at():
    """Unlike search, a visit fetches exactly one page -- there is no "take one, note the rest"
    available, so a multi-url wrapper is left completely alone rather than guessed at."""
    raw = "['https://a.example.com', 'https://b.example.com']"
    url, coerced = _coerce_visit_url(raw)
    assert url == raw
    assert coerced is None


def test_coerce_visit_url_dict_without_known_key_stays_as_is():
    raw = "{'title': 'Victoria Falls'}"
    url, coerced = _coerce_visit_url(raw)
    assert url == raw
    assert coerced is None


def test_coerce_visit_url_plain_string_unchanged():
    url, coerced = _coerce_visit_url("https://example.com")
    assert url == "https://example.com"
    assert coerced is None


def test_coerce_visit_url_empty_string_unchanged():
    url, coerced = _coerce_visit_url("")
    assert url == ""
    assert coerced is None


# --- is_malformed_url_text -------------------------------------------------------------

def test_is_malformed_url_text_empty():
    assert is_malformed_url_text("") is True
    assert is_malformed_url_text("   ") is True


def test_is_malformed_url_text_bracketed_repr():
    assert is_malformed_url_text("['https://example.com']") is True
    assert is_malformed_url_text("{'url': 'https://example.com'}") is True


def test_is_malformed_url_text_no_scheme_or_netloc():
    assert is_malformed_url_text("example.com") is True
    assert is_malformed_url_text("not a url at all") is True


def test_is_malformed_url_text_valid_http_https():
    assert is_malformed_url_text("https://example.com/a") is False
    assert is_malformed_url_text("http://example.com") is False


def test_is_malformed_url_text_non_string_input():
    assert is_malformed_url_text(None) is True
    assert is_malformed_url_text(["https://example.com"]) is True


# --- integration: AgentIO.search / AgentIO.visit --------------------------------------------

class _FakeTelemetry:
    def __init__(self):
        self.timings = []

    def record_timing(self, name, started_at=None, success=True, payload=None, error=None):
        self.timings.append({"name": name, "payload": payload or {}, "success": success, "error": error})

    def record_document_seen(self, source, document):
        pass


def _make_io(connector_search=None, connector_http=None, connector_browser=None, telemetry=None):
    mock_llm = MagicMock()
    mock_llm.set_telemetry = MagicMock()
    mock_chroma = MagicMock()
    mock_chroma.set_telemetry = MagicMock()

    connector_search = connector_search or MagicMock()
    connector_search.set_telemetry = MagicMock()
    connector_http = connector_http or MagicMock()
    connector_http.set_telemetry = MagicMock()
    if connector_browser is not None:
        connector_browser.set_telemetry = MagicMock()

    io = AgentIO(
        connector_llm=mock_llm,
        connector_search=connector_search,
        connector_http=connector_http,
        connector_chroma=mock_chroma,
        connector_browser=connector_browser,
    )
    if telemetry is not None:
        io.set_telemetry(telemetry)
    return io


@pytest.mark.asyncio
async def test_search_unwraps_wrapped_query_before_hitting_the_backend():
    connector_search = MagicMock()
    connector_search.query_search = AsyncMock(return_value=[])
    telemetry = _FakeTelemetry()
    io = _make_io(connector_search=connector_search, telemetry=telemetry)

    await io.search("['Lake Matano maximum depth']")

    connector_search.query_search.assert_awaited_once_with("Lake Matano maximum depth", count=10)
    timing = telemetry.timings[-1]
    assert timing["payload"]["query"] == "Lake Matano maximum depth"
    assert timing["payload"]["arg_coerced"] == "list"


@pytest.mark.asyncio
async def test_search_multi_element_list_uses_first_and_records_dropped():
    connector_search = MagicMock()
    connector_search.query_search = AsyncMock(return_value=[])
    telemetry = _FakeTelemetry()
    io = _make_io(connector_search=connector_search, telemetry=telemetry)

    await io.search("['Lake Matano depth', 'Hornindalsvatnet depth']")

    connector_search.query_search.assert_awaited_once_with("Lake Matano depth", count=10)
    timing = telemetry.timings[-1]
    assert timing["payload"]["arg_coerced"] == "list_multi"
    assert timing["payload"]["dropped_queries"] == ["Hornindalsvatnet depth"]


@pytest.mark.asyncio
async def test_search_plain_query_records_null_arg_coerced_not_a_missing_key():
    """`arg_coerced` must be present (as null) even when nothing was coerced, so the frequency
    of coercion stays measurable rather than requiring a `.get(..., default)` guess downstream."""
    connector_search = MagicMock()
    connector_search.query_search = AsyncMock(return_value=[])
    telemetry = _FakeTelemetry()
    io = _make_io(connector_search=connector_search, telemetry=telemetry)

    await io.search("plain query")

    timing = telemetry.timings[-1]
    assert "arg_coerced" in timing["payload"]
    assert timing["payload"]["arg_coerced"] is None


@pytest.mark.asyncio
async def test_visit_unwraps_wrapped_but_valid_url_and_succeeds():
    from shared.request_result import RequestResult

    connector_http = MagicMock()
    connector_http.request = AsyncMock(
        return_value=RequestResult(status=200, data="<html><body>ok</body></html>", error=False)
    )
    telemetry = _FakeTelemetry()
    io = _make_io(connector_http=connector_http, telemetry=telemetry)

    text = await io.visit("['https://en.wikipedia.org/wiki/Victoria_Falls']")

    assert text
    connector_http.request.assert_awaited_once()
    called_url = connector_http.request.await_args.args[1]
    assert called_url == "https://en.wikipedia.org/wiki/Victoria_Falls"
    timing = telemetry.timings[-1]
    assert timing["success"] is True
    assert timing["payload"]["arg_coerced"] == "list"


@pytest.mark.asyncio
async def test_visit_unwraps_dict_wrapped_url():
    from shared.request_result import RequestResult

    connector_http = MagicMock()
    connector_http.request = AsyncMock(
        return_value=RequestResult(status=200, data="<html><body>ok</body></html>", error=False)
    )
    io = _make_io(connector_http=connector_http)

    await io.visit("{'url': 'https://en.wikipedia.org/wiki/Gotthard_Base_Tunnel'}")

    called_url = connector_http.request.await_args.args[1]
    assert called_url == "https://en.wikipedia.org/wiki/Gotthard_Base_Tunnel"


@pytest.mark.asyncio
async def test_visit_empty_url_fails_immediately_without_a_network_call():
    connector_http = MagicMock()
    connector_http.request = AsyncMock()
    telemetry = _FakeTelemetry()
    io = _make_io(connector_http=connector_http, telemetry=telemetry)

    with pytest.raises(RuntimeError):
        await io.visit("")

    connector_http.request.assert_not_awaited()
    timing = telemetry.timings[-1]
    assert timing["success"] is False
    assert timing["payload"]["failure_class"] == "model"
    assert timing["payload"]["status"] is None


@pytest.mark.asyncio
async def test_visit_unresolved_wrapper_fails_immediately_without_guessing():
    """A multi-url list, or a dict without a recognized key, is never dispatched -- `visit`
    fails loudly on the unresolved wrapper text rather than guessing at a URL."""
    connector_http = MagicMock()
    connector_http.request = AsyncMock()
    telemetry = _FakeTelemetry()
    io = _make_io(connector_http=connector_http, telemetry=telemetry)

    with pytest.raises(RuntimeError):
        await io.visit("['https://a.example.com', 'https://b.example.com']")

    connector_http.request.assert_not_awaited()
    timing = telemetry.timings[-1]
    assert timing["payload"]["failure_class"] == "model"
    assert timing["payload"]["arg_coerced"] is None


@pytest.mark.asyncio
async def test_visit_never_repairs_a_url_that_merely_fails_over_the_network():
    """Coercion is unwrapping only. A url that is syntactically a real http(s) URL (not empty,
    not a recognized wrapper shape) is passed straight to the network layer unchanged -- never
    guessed at or repaired -- and its outcome is decided by the network call itself, not by
    this method inventing content for it."""
    from shared.request_result import RequestResult

    connector_http = MagicMock()
    connector_http.request = AsyncMock(
        return_value=RequestResult(status=None, data="Request failed: bad url", error=True)
    )
    telemetry = _FakeTelemetry()
    io = _make_io(connector_http=connector_http, telemetry=telemetry)

    with pytest.raises(RuntimeError):
        await io.visit("http://this-host-does-not-exist.invalid/page")

    connector_http.request.assert_awaited_once()
