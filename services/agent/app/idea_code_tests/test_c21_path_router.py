"""
codebench task c21 — hard/hidden, HTTP-shaped API logic: method+path-pattern request routing.

First task in the API/HTTP-server-logic gap (the existing suite's c16 "todo API" is the only
prior task that even brushes this space, and it's a soft/open-ended CRUD build with no status
codes or dispatch logic — this is a genuinely different, deterministic shape). No real network
server runs anywhere: "a request" is just a (method, path) pair passed straight to
``Router.dispatch``, which is what makes this fully offline-gradeable with plain pytest.

The task is a small, self-contained ``Router`` class (register method+path-pattern -> handler,
then dispatch a (method, path) pair to the right handler and return an HTTP-style status code).
The two edge cases that make this genuinely hard for a weak model — and that a naive
implementation plausibly gets wrong — are:
  1. Static-vs-dynamic PRECEDENCE: when a concrete path could match more than one registered
     pattern (e.g. both `/users/me` and `/users/{id}` match the path "/users/me"), the more
     specific (static) pattern must win. A router that just returns the first structurally
     matching pattern in registration order, or that has no precedence rule at all, gets this
     wrong on this exact case.
  2. 404 vs 405: a path that matches NO registered pattern is 404 (Not Found); a path that
     matches a registered pattern's shape but not for the requested METHOD is 405 (Method Not
     Allowed) — a distinct outcome from 404. Naive implementations frequently collapse both into
     404 (see the "naive" mutant referenced below).
  3. A classic ``zip(pattern_segments, path_segments)`` bug: zip() silently truncates to the
     shorter sequence, so a pattern `/users/{id}` (2 segments) naively matched against a path
     `/users/1/2` (3 segments) via zip() alone incorrectly "matches" (capturing id="1" and
     silently ignoring the trailing "/2") unless segment COUNT is checked first.

Ground truth for every case embedded below was independently confirmed two ways before being
locked in: (a) hand-traced through the spec's own stated algorithm, and (b) proven discriminating
by running the exact same canonical test file against BOTH a correct reference implementation
(all pass) and a deliberately naive one — first-match-wins with no specificity rule, no
segment-length check, and 404-for-everything-that-isn't-an-exact-hit — which failed exactly the
precedence, segment-count, method-mismatch, and duplicate-registration cases (5 of 10 tests),
confirming those cases actually discriminate a correct implementation from a plausible-looking
broken one rather than being redundant with the easier cases.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_mini_router.py"

_TEST_FILE_CONTENT = '''\
from mini_router import Router


def _recording_handler(tag, calls):
    def handler(**kwargs):
        calls.append((tag, kwargs))
        return tag
    return handler


def test_static_route_exact_match():
    r = Router()
    calls = []
    r.add_route("GET", "/health", _recording_handler("health", calls))
    assert r.dispatch("GET", "/health") == (200, "health")


def test_single_param_capture():
    r = Router()
    calls = []
    r.add_route("GET", "/users/{user_id}", _recording_handler("user", calls))
    status, body = r.dispatch("GET", "/users/42")
    assert (status, body) == (200, "user")
    assert calls[-1] == ("user", {"user_id": "42"})


def test_multi_param_capture():
    r = Router()
    calls = []
    r.add_route("GET", "/orgs/{org}/repos/{repo}", _recording_handler("repo", calls))
    status, body = r.dispatch("GET", "/orgs/acme/repos/widgets")
    assert (status, body) == (200, "repo")
    assert calls[-1] == ("repo", {"org": "acme", "repo": "widgets"})


def test_specificity_precedence():
    r = Router()
    calls = []
    r.add_route("GET", "/users/{id}", _recording_handler("dynamic", calls))
    r.add_route("GET", "/users/me", _recording_handler("static", calls))
    status, body = r.dispatch("GET", "/users/me")
    assert (status, body) == (200, "static")
    assert calls[-1] == ("static", {})

    status2, body2 = r.dispatch("GET", "/users/42")
    assert (status2, body2) == (200, "dynamic")
    assert calls[-1] == ("dynamic", {"id": "42"})


def test_method_mismatch_returns_405():
    r = Router()
    calls = []
    r.add_route("GET", "/widgets/{id}", _recording_handler("widget", calls))
    assert r.dispatch("DELETE", "/widgets/5") == (405, None)


def test_unknown_path_returns_404():
    r = Router()
    calls = []
    r.add_route("GET", "/widgets/{id}", _recording_handler("widget", calls))
    assert r.dispatch("GET", "/gadgets/5") == (404, None)


def test_segment_count_mismatch_returns_404():
    r = Router()
    calls = []
    r.add_route("GET", "/users/{id}", _recording_handler("user", calls))
    assert r.dispatch("GET", "/users/1/2") == (404, None)


def test_empty_segment_does_not_satisfy_param():
    r = Router()
    calls = []
    r.add_route("GET", "/users/{id}/profile", _recording_handler("profile", calls))
    assert r.dispatch("GET", "/users//profile") == (404, None)


def test_method_case_insensitive():
    r = Router()
    calls = []
    r.add_route("get", "/ping", _recording_handler("ping", calls))
    assert r.dispatch("GET", "/ping") == (200, "ping")
    assert r.dispatch("get", "/ping") == (200, "ping")


def test_route_overwrite_on_duplicate_registration():
    r = Router()
    calls = []
    r.add_route("GET", "/thing", _recording_handler("first", calls))
    r.add_route("GET", "/thing", _recording_handler("second", calls))
    assert r.dispatch("GET", "/thing") == (200, "second")
'''

# The four cases below are the ones a plausible-but-wrong router (first structural match wins,
# no specificity rule, no segment-length check) gets wrong — see the module docstring's naive-
# mutant note. Basic exact-match, unknown-path 404, method case-insensitivity, empty-segment
# rejection, and duplicate-registration overwrite are supporting/bonus credit: real behavior a
# correct router must have, but not the sharp edge that actually separates a correct
# implementation from a naive one.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_single_param_capture",
    f"{_TEST_FILE_PATH}::test_multi_param_capture",
    f"{_TEST_FILE_PATH}::test_specificity_precedence",
    f"{_TEST_FILE_PATH}::test_method_mismatch_returns_405",
    f"{_TEST_FILE_PATH}::test_segment_count_mismatch_returns_404",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c21",
        "title": "http-path-router",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `mini_router.py` implementing a small HTTP-style request "
        "router. No real network server is involved — \"a request\" is just a "
        "(method, path) pair passed directly to your router's dispatch method.\n\n"
        "Define a class `Router` with this API:\n\n"
        "- `Router()` — construct an empty router.\n"
        "- `add_route(self, method: str, path_pattern: str, handler)` — register `handler` "
        "for `method` (case-insensitive — store it normalized) and `path_pattern`. "
        "`path_pattern` always starts with `/` and is made of `/`-separated segments; a "
        "segment written as `{name}` is a WILDCARD that matches exactly one non-empty path "
        "segment and captures it under `name`; every other segment must match that exact "
        "literal text. Registering the same (method, path_pattern) pair twice REPLACES the "
        "earlier handler (last write wins).\n"
        "- `dispatch(self, method: str, path: str) -> tuple` — given a concrete `method` and "
        "`path` (also always starting with `/`), decide the response as follows:\n\n"
        "  1. Find every registered route whose PATTERN structurally matches `path` — same "
        "number of `/`-separated segments, every literal segment equal, and every `{name}` "
        "segment matched against a NON-EMPTY path segment (an empty segment, e.g. from a "
        "double slash, does not satisfy a `{name}` wildcard) — regardless of that route's "
        "registered method.\n"
        "  2. If no route matches structurally at all, return `(404, None)`.\n"
        "  3. Otherwise pick the single BEST structural match: prefer the route with more "
        "literal (non-`{}`) segments (a more specific pattern beats a more general one at "
        "the same path — e.g. `/users/me` beats `/users/{id}` for the path \"/users/me\"); "
        "break ties by whichever route was registered first.\n"
        "  4. If that best-matching route's registered method equals the requested `method` "
        "(case-insensitively), call its handler with the captured `{name}` values passed as "
        "KEYWORD arguments (e.g. `handler(id=\"42\")`), and return `(200, <handler's return "
        "value>)`.\n"
        "  5. Otherwise (the best structural match exists but is registered under a "
        "different method) return `(405, None)` — this is a DIFFERENT outcome from step 2's "
        "404, and callers must be able to tell the two apart.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "own work (register a few routes including two that could both match the same path, "
        "e.g. a static route and a wildcard route with an overlapping shape; dispatch a few "
        "(method, path) pairs and confirm the status codes and captured params match your "
        "understanding of the rules above, including at least one 404 case and one 405 "
        "case) before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "mini_router",
            "class": "Router",
            "methods": ["add_route", "dispatch"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one cohesive class, no independent sub-parts worth decomposing across."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write mini_router.py defining a class Router with add_route(self, "
                    "method, path_pattern, handler) and dispatch(self, method, path) -> "
                    "tuple. path_pattern segments written as {name} are wildcards matching "
                    "exactly one non-empty path segment (captured under that name); other "
                    "segments must match literally. dispatch must: (1) find every registered "
                    "route whose pattern structurally matches path (same segment count, "
                    "literal segments equal, {name} segments matched against a NON-EMPTY "
                    "path segment) regardless of that route's method; (2) if none match "
                    "structurally, return (404, None); (3) otherwise pick the route with the "
                    "MOST literal (non-wildcard) segments as the best match (a static segment "
                    "beats a wildcard at the same position — e.g. a registered '/users/me' "
                    "must win over a registered '/users/{id}' when dispatching path "
                    "'/users/me'), breaking ties by earliest registration; (4) if that best "
                    "match's registered method equals the requested method "
                    "(case-insensitive), call its handler with captured {name} values as "
                    "keyword arguments and return (200, handler's return value); (5) "
                    "otherwise return (405, None) — a DIFFERENT result from the (404, None) "
                    "in step 2. Also make sure a pattern with N segments never matches a path "
                    "with a different number of segments — do not rely on zip() alone without "
                    "checking segment counts first, since zip() silently truncates to the "
                    "shorter sequence. Use write_file to create mini_router.py. Then use "
                    "run_python to sanity-check it yourself: register at least one static "
                    "route and one overlapping wildcard route (e.g. '/users/me' and "
                    "'/users/{id}') and confirm the static one wins when you dispatch the "
                    "exact static path; register a route under one method and dispatch a "
                    "different method against the same path to confirm you get 405, not 404; "
                    "dispatch a completely unregistered path to confirm you get 404. Fix "
                    "issues with patch_file/run_python until confident, then finish."
                ),
                "expect": (
                    "mini_router.py written; Router.dispatch correctly implements "
                    "specificity precedence (static beats wildcard), 404-vs-405 "
                    "distinction, and segment-count-aware structural matching"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm mini_router.py exists and report the agent's own sanity-check results "
            "(precedence, 404, and 405 cases)."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["mini_router.py"]},
    }
