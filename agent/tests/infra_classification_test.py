"""`_is_infra_timing`'s model-vs-infra classification for `visit`/`search` failures.

The bug: `_is_infra_timing` treated EVERY failed `visit`/`search`/`http_request` timing with no
HTTP status as infrastructure trouble. A model-invented or wrapped URL fails BEFORE any HTTP
status exists (see `agent/tests/tool_argument_coercion_test.py`), so it landed in that branch
too -- meaning the more a model mangled a URL, the more of its own failures were quarantined
out of scoring as "not the model's fault".

Per-model infra-failed rate BEFORE this fix (cells >= 20, from `agent/idea_test_results/`, per
the task brief):

    anthropic/claude-sonnet-5   35 cells   25.7% infra_failed   100% visit-only
    qwen2.5:0.5b                93 cells   18.3% infra_failed   100% visit-only
    phi3:mini                  150 cells   11.3% infra_failed    94% visit-only
    qwen2.5:7b                4112 cells    1.3% infra_failed    61% visit-only

These tests pin the new two-layer classification `_is_infra_timing` now does (see its own
docstring for the full rationale):

1. An explicit `payload["failure_class"]` (`"model"`/`"infra"`/`"unknown"`), set going forward
   by `AgentIO.search`/`.visit` (agent_io.py).
2. For a `visit` timing with no such field (every cell recorded before this fix), a retroactive
   check of the one signal such a cell already persisted -- the raw `url` text -- via
   `is_malformed_url_text` (shared with agent_io.py): empty, or still shaped like a Python
   list/dict repr, is unambiguously the model's fault regardless of what happened on the wire.

Everything else -- a syntactically valid url that still failed with status=None and no richer
signal, `search`'s status=None case, and the pre-existing `infra_failed`/status-coded paths --
is deliberately UNCHANGED: this fix only SUBTRACTS confirmed-model-fault cases from the infra
count, never adds ambiguity-driven guesses in either direction.
"""
from __future__ import annotations

from agent.app.testing.utils import _is_infra_timing, _summarize_infra


def _timing(name, success, payload=None, error=None):
    entry = {"name": name, "duration": 0.1, "success": success, "payload": payload or {}}
    if error:
        entry["error"] = error
    return entry


# --- explicit failure_class signal (new data, going forward) --------------------------------

def test_failure_class_model_overrides_status_none_to_not_infra():
    t = _timing("visit", False, {"url": "", "status": None, "failure_class": "model"})
    assert _is_infra_timing(t) is False


def test_failure_class_infra_is_infra_even_without_status():
    t = _timing("visit", False, {"url": "http://real-host.example.com", "status": None, "failure_class": "infra"})
    assert _is_infra_timing(t) is True


def test_failure_class_unknown_falls_back_to_old_heuristic():
    """`"unknown"` must behave exactly like the field being absent -- see the retroactive tests
    below -- never a silent downgrade to "not infra"."""
    t = _timing("visit", False, {"url": "http://real-host.example.com", "status": None, "failure_class": "unknown"})
    assert _is_infra_timing(t) is True


# --- retroactive reclassification from stored `url` text (old data, no failure_class) -------

def test_retroactive_empty_url_is_not_infra():
    """The 89-measured empty-url case: a model failure regardless of when it was recorded."""
    t = _timing("visit", False, {"url": "", "status": None})
    assert _is_infra_timing(t) is False


def test_retroactive_wrapped_list_url_is_not_infra():
    t = _timing("visit", False, {"url": "['https://en.wikipedia.org/wiki/X']", "status": None})
    assert _is_infra_timing(t) is False


def test_retroactive_wrapped_dict_url_is_not_infra():
    t = _timing("visit", False, {"url": "{'url': 'https://en.wikipedia.org/wiki/X'}", "status": None})
    assert _is_infra_timing(t) is False


def test_retroactive_valid_url_with_no_status_stays_infra_unchanged():
    """A syntactically real URL that still failed with no status: cannot tell a DNS failure on
    an invented host apart from a genuine transient network failure from this signal alone --
    stays on the pre-existing (and, for genuine infra, correct) heuristic. Confirms this fix
    never widens the definition to shrink the rate by guessing "model" here."""
    t = _timing("visit", False, {"url": "https://en.wikipedia.org/wiki/Victoria_Falls", "status": None})
    assert _is_infra_timing(t) is True


def test_missing_url_key_entirely_stays_on_old_heuristic():
    """A `visit` timing that never carried a `url` payload key at all (e.g. constructed by an
    older/unrelated code path) is genuinely unknown, not a confirmed bad url -- must not be
    treated the same as an explicit empty string."""
    t = _timing("visit", False, {"status": None})
    assert _is_infra_timing(t) is True


def test_search_status_none_is_unaffected_by_the_url_based_check():
    """The retroactive url-text check is `visit`-specific (a `search` timing's payload key is
    `query`, not `url`, and a wrapped/garbled query does not usually make the search call fail
    at all -- see tool_argument_coercion_test.py). `search`'s status=None classification is
    deliberately left exactly as it was -- note `"search"` (unlike `"visit"`) was never in
    `_INFRA_TIMING_NAMES` to begin with (only `http_request`/`search_query`/`visit` are), so a
    failed search with no status was already `False` before this fix and stays `False`."""
    t = _timing("search", False, {"query": "['a', 'b']", "status": None})
    assert _is_infra_timing(t) is False


# --- pre-existing behavior must be unchanged -------------------------------------------------

def test_success_is_never_infra():
    assert _is_infra_timing(_timing("visit", True, {"url": "", "status": None})) is False


def test_explicit_infra_failed_flag_still_wins():
    t = _timing("llm_call", False, {"infra_failed": True})
    assert _is_infra_timing(t) is True
    t = _timing("llm_call", False, {"infra_failed": False})
    assert _is_infra_timing(t) is False


def test_402_429_5xx_status_codes_still_infra():
    assert _is_infra_timing(_timing("http_request", False, {"status": 402})) is True
    assert _is_infra_timing(_timing("search_query", False, {"status": 429})) is True
    assert _is_infra_timing(_timing("http_request", False, {"status": 503})) is True


def test_bot_block_status_codes_still_not_infra():
    """401/403/404 stay a genuine failure, not infra -- unaffected by this fix, and a real url
    (not empty/wrapped) so the new retroactive check would not fire even if status weren't set."""
    assert _is_infra_timing(_timing("visit", False, {"url": "https://example.com", "status": 403})) is False
    assert _is_infra_timing(_timing("http_request", False, {"status": 404})) is False


def test_malformed_url_with_a_real_status_uses_status_path_not_url_path():
    """If a status code DID come back (e.g. a browser fallback reached a real page and only a
    later step failed), the url-text override never fires -- only the status=None branch is in
    scope for this fix."""
    t = _timing("visit", False, {"url": "", "status": 500})
    assert _is_infra_timing(t) is True


# --- _summarize_infra: the severity gate stays intact ----------------------------------------

def test_summarize_infra_drops_confirmed_model_fault_from_the_failed_gate():
    """A cell whose only failed op is a confirmed bad-url visit must no longer trip the
    severity-gated `failed` flag that quarantines cells from scoring (see
    `_INFRA_FAILURE_RATE_THRESHOLD`'s docstring for why that gate exists and must stay)."""
    timings = (
        [_timing("visit", True, {"url": "https://example.com", "status": 200}) for _ in range(3)]
        + [_timing("visit", False, {"url": "", "status": None})]
    )
    summary = _summarize_infra(timings)
    assert summary["failed"] is False
    assert summary["failure_count"] == 0


def test_summarize_infra_still_flags_a_majority_genuine_infra_op():
    timings = [_timing("visit", False, {"url": "https://real.example.com", "status": None}) for _ in range(3)] + [
        _timing("visit", True, {"url": "https://real.example.com", "status": 200})
    ]
    summary = _summarize_infra(timings)
    assert summary["failed"] is True
    assert summary["failure_count"] == 3
