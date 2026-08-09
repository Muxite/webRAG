"""
codebench task c22 — hard/hidden, HTTP-shaped API logic: conditional-request (ETag) resource
store.

REPLACEMENT NOTE (2026-08-06): this task used to be a token-bucket rate limiter. Across THREE
separate live-calibration rounds (see git history of this file for the original), qwen2.5:14b via
Aider scored 1.0/1.0 every single time — including the third round, which required deriving a
brand-new inverse-refill `wait_time()` method with its own sharp edge case (a cost exceeding
capacity must raise `ValueError` since no amount of waiting can ever satisfy it) rather than just
transcribing another textual gotcha on `allow`/`available`. Token-bucket rate limiting is one of
the most over-represented toy exercises in existence — thousands of near-identical
implementations across tutorials, blog posts, and interview-prep repos — and a strong coding
model evidently pattern-matches the entire shape from memory regardless of how many explicit
edge-case footguns get spelled out in prose, because "spell it out clearly enough for hidden
grading to be fair" and "a strong model can then faithfully implement it" turned out to be the
same lever pulling in the same direction every time. Adding another footgun to `allow`/`available`
was not going to break that; the task needed a genuinely different shape, not another edge case.
This is a full replacement, same id/slug/category conventions, same "hard hidden HTTP-shaped API
logic" family as its former self and as siblings c21 (path routing) and c23 (request parsing).

The new task: a `ResourceStore` that models the conditional-request (`If-Match` /
`If-None-Match`) machinery real HTTP APIs use for optimistic-concurrency writes and cache
validation (RFC 7232's ETag comparison rules, without any actual RFC-recitation being sufficient
to solve it — see below). No real network server, no real time — `put`/`get` take plain string
arguments and return plain tuples, fully offline-gradeable with plain pytest.

Why this shape resists memorized pattern-matching where token-bucket did not: it is a real,
well-documented HTTP mechanism, but it is NOT a canonical from-scratch toy coding exercise the way
LRU caches, rate limiters, or basic routers are — in practice almost everyone either delegates
this to a framework/library or never implements the strong-vs-weak ETag comparison distinction by
hand at all. Getting it right requires READING the two rules below and applying them in
combination, not recalling a memorized reference solution:
  1. Every stored ETag this store generates is a STRONG validator: `'"' + sha256_hex(body) + '"'`
     (quoted lowercase hex of the SHA-256 digest of the UTF-8-encoded body — deterministic and
     content-addressed, so writing identical content twice yields the identical ETag).
  2. `If-Match` (used on `put`) is compared using STRONG comparison: two validators are equal
     only if NEITHER is weak-marked (`W/` prefix) AND their opaque (unquoted) parts are equal.
  3. `If-None-Match` (used on `get`) is compared using WEAK comparison: two validators are equal
     whenever their opaque parts are equal, REGARDLESS of whether either side carries a `W/`
     prefix.
  4. `*` is a wildcard matching "any current representation exists" — for `put`, this means a
     bare `*` on a resource that does not yet exist must still fail (there is no current
     representation to match against), a genuinely easy rule to get backwards (treating `*` as
     "always succeeds unconditionally" instead of "succeeds only if something is already there").
  5. Either header may carry a comma-separated LIST of validators; a match against ANY listed
     validator (via the header's applicable comparison rule) is sufficient.

The genuinely hard part is not any single rule above in isolation — it is that rules 2 and 3
INTERACT: since every stored ETag is strong (never weak-marked), the strong-vs-weak comparison
distinction only has observable consequences when a CLIENT-supplied validator carries a `W/`
prefix. A plausible, sophisticated-looking implementation can get the STRUCTURAL part completely
right — two distinctly-named comparison functions/branches, `If-Match` correctly routed to one,
`If-None-Match` correctly routed to the other, matching the prose exactly — while still copying
the weak-comparison function's body into the strong one (i.e., forgetting the "reject if either
side is weak-marked" clause specifically inside the strong path), because that's the natural
default shape: "compare the opaque parts" is the obvious thing both functions do, and it is easy
to write that once and route both call sites to it while believing the routing alone satisfies
the spec. This mirrors the c39/c40/c41 debugging-task insight (a fix that looks complete and
handles the "obvious" cases can still silently fail a case that only manifests when two
requirements interact) rather than the old token-bucket task's pattern of one additive edge case
after another.

Ground truth verified three ways: (a) hand-traced against the spec's own stated rules for every
case below (all SHA-256 digests are computed by the canonical test file itself via `hashlib`, not
hand-typed hex, so there is no transcription-error surface); (b) a reference implementation run
against the canonical 16-test suite passes all 16 (see idea_code_test_c22_test.py, which
re-derives this independently); a completely naive baseline that ignores `if_match`/
`if_none_match` entirely (unconditional reads/writes always succeed) fails 7 of 16, including 5 of
6 keystone tests, confirming the conditional machinery is not redundant with the "does it work at
all" cases; and, most importantly, the targeted "structurally-correct-but-mechanically-wrong"
mutant described above — routes `If-Match`/`If-None-Match` to correctly-named, correctly-called
strong/weak comparison functions, but implements the strong function's body identically to the
weak one — fails EXACTLY ONE test
(`test_put_if_match_weak_validator_never_satisfies_strong_comparison`, a keystone case) out of 16,
demonstrating this is a genuine, narrow, structurally-interacting trap rather than a broad
correctness gap a model would stumble into by accident; a second, independent mutant that treats
`*` as "always matches" instead of "matches only if a current representation exists" fails a
DIFFERENT single keystone test (`test_put_if_match_wildcard_fails_on_nonexistent_resource`),
confirming that edge is independently load-bearing too; (c) this replacement has not yet been
through a live-calibration round against qwen2.5:14b (offline mutation testing above is the
authoring-time evidence; live calibration against both the badmodel and aider harnesses is the
next step, owned by the coordinating session per its own instructions, not run from here).
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_resource_store.py"

_TEST_FILE_CONTENT = '''\
import hashlib

import pytest
from resource_store import ResourceStore


def _expected_etag(body):
    return '"' + hashlib.sha256(body.encode("utf-8")).hexdigest() + '"'


def test_put_new_resource_returns_201_with_content_hash_etag():
    store = ResourceStore()
    status, etag = store.put("/widgets/1", "hello")
    assert status == 201
    assert etag == _expected_etag("hello")


def test_put_overwrite_unconditional_returns_200_with_new_etag():
    store = ResourceStore()
    store.put("/widgets/1", "hello")
    status, etag = store.put("/widgets/1", "world")
    assert status == 200
    assert etag == _expected_etag("world")


def test_get_nonexistent_returns_404():
    store = ResourceStore()
    assert store.get("/nope") == (404, None, None)


def test_get_existing_returns_current_body_and_etag():
    store = ResourceStore()
    store.put("/widgets/1", "hello")
    status, etag, body = store.get("/widgets/1")
    assert status == 200
    assert body == "hello"
    assert etag == _expected_etag("hello")


def test_same_body_produces_same_etag_different_body_produces_different_etag():
    store = ResourceStore()
    _, etag_a1 = store.put("/x", "same")
    store.put("/x", "changed")
    _, etag_a2 = store.put("/x", "same")
    assert etag_a1 == etag_a2 == _expected_etag("same")


def test_put_if_match_exact_strong_etag_succeeds():
    store = ResourceStore()
    _, etag1 = store.put("/x", "v1")
    status, etag2 = store.put("/x", "v2", if_match=etag1)
    assert status == 200
    assert etag2 == _expected_etag("v2")


def test_put_if_match_stale_etag_returns_412_and_leaves_resource_unchanged():
    store = ResourceStore()
    store.put("/x", "v1")
    status, etag = store.put("/x", "v2", if_match='"not-the-real-etag"')
    assert (status, etag) == (412, None)
    assert store.get("/x") == (200, _expected_etag("v1"), "v1")


def test_put_if_match_wildcard_fails_on_nonexistent_resource():
    store = ResourceStore()
    status, etag = store.put("/brand-new", "v1", if_match="*")
    assert (status, etag) == (412, None)
    assert store.get("/brand-new") == (404, None, None)


def test_put_if_match_wildcard_succeeds_on_existing_resource():
    store = ResourceStore()
    store.put("/x", "v1")
    status, etag = store.put("/x", "v2", if_match="*")
    assert status == 200
    assert etag == _expected_etag("v2")


def test_put_if_match_weak_validator_never_satisfies_strong_comparison():
    store = ResourceStore()
    _, etag1 = store.put("/x", "v1")
    weak_token = "W/" + etag1
    status, etag = store.put("/x", "v2", if_match=weak_token)
    assert (status, etag) == (412, None)
    assert store.get("/x") == (200, etag1, "v1")


def test_get_if_none_match_weak_validator_satisfies_weak_comparison():
    store = ResourceStore()
    _, etag1 = store.put("/x", "v1")
    weak_token = "W/" + etag1
    status, etag, body = store.get("/x", if_none_match=weak_token)
    assert (status, etag, body) == (304, etag1, None)


def test_get_if_none_match_exact_strong_etag_also_satisfies_weak_comparison():
    store = ResourceStore()
    _, etag1 = store.put("/x", "v1")
    status, etag, body = store.get("/x", if_none_match=etag1)
    assert (status, etag, body) == (304, etag1, None)


def test_get_if_none_match_stale_etag_returns_200_with_body():
    store = ResourceStore()
    store.put("/x", "v1")
    status, etag, body = store.get("/x", if_none_match='"not-the-real-etag"')
    assert status == 200
    assert body == "v1"
    assert etag == _expected_etag("v1")


def test_get_if_none_match_wildcard_matches_existing_resource():
    store = ResourceStore()
    store.put("/x", "v1")
    status, etag, body = store.get("/x", if_none_match="*")
    assert (status, body) == (304, None)


def test_put_if_match_multi_value_list_matches_any_listed_etag():
    store = ResourceStore()
    _, etag1 = store.put("/x", "v1")
    header = '"decoy-one", ' + etag1 + ', "decoy-two"'
    status, etag2 = store.put("/x", "v2", if_match=header)
    assert status == 200
    assert etag2 == _expected_etag("v2")


def test_get_if_none_match_multi_value_list_matches_any_listed_etag():
    store = ResourceStore()
    _, etag1 = store.put("/x", "v1")
    header = '"decoy-one", "decoy-two", ' + etag1
    status, etag, body = store.get("/x", if_none_match=header)
    assert (status, etag, body) == (304, etag1, None)
'''

# Keystone set: the reject-without-existing wildcard rule, the basic conditional-write
# protection (stale If-Match must not write), and — the sharpest edge, see module docstring —
# the strong-vs-weak comparison distinction in BOTH directions (a weak client validator must
# never satisfy If-Match's strong comparison; the same weak validator MUST satisfy
# If-None-Match's weak comparison) plus both directions of comma-separated multi-value list
# matching. Basic unconditional put/get, the content-hash-determinism property, exact-match
# success cases, and the "stale etag still returns 200 with body" case are supporting/bonus
# credit: real behavior, but not the discriminating edge — see module docstring's mutant
# analysis for exactly what a plausible-but-incomplete implementation gets wrong.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_put_if_match_stale_etag_returns_412_and_leaves_resource_unchanged",
    f"{_TEST_FILE_PATH}::test_put_if_match_wildcard_fails_on_nonexistent_resource",
    f"{_TEST_FILE_PATH}::test_put_if_match_weak_validator_never_satisfies_strong_comparison",
    f"{_TEST_FILE_PATH}::test_get_if_none_match_weak_validator_satisfies_weak_comparison",
    f"{_TEST_FILE_PATH}::test_put_if_match_multi_value_list_matches_any_listed_etag",
    f"{_TEST_FILE_PATH}::test_get_if_none_match_multi_value_list_matches_any_listed_etag",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c22",
        "title": "conditional-etag-resource-store",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `resource_store.py` implementing a small in-memory HTTP-style "
        "resource store with CONDITIONAL-REQUEST support (`If-Match` / `If-None-Match`) — the "
        "same mechanism a real API server uses for optimistic-concurrency writes and cache "
        "validation. No real network I/O or real time is involved: everything is plain "
        "strings and dict lookups.\n\n"
        "Define a class `ResourceStore` with this API:\n\n"
        "- `ResourceStore()` — construct an empty store (no resources exist yet).\n"
        "- `put(self, path: str, body: str, if_match: str | None = None) -> tuple` — write "
        "`body` to `path`. First, compute this resource's ETag as `'\"' + sha256_hex(body) + "
        "'\"'` — a double-quoted string containing the lowercase hexadecimal SHA-256 digest "
        "of `body` encoded as UTF-8 (Python's `hashlib.sha256` and `.hexdigest()` do the "
        "digest computation for you). This is a STRONG validator: it never carries a `W/` "
        "prefix, and writing the exact same body content always produces the exact same "
        "ETag.\n"
        "  - If `if_match` is `None`: the write is unconditional. Store `body` under `path` "
        "(replacing anything already there) and return `(status, new_etag)`, where `status` "
        "is `201` if `path` did not already exist before this call, or `200` if it did.\n"
        "  - If `if_match` is not `None`: it is a conditional-write precondition, using ONE "
        "of these two shapes: the literal string `\"*\"`, or a comma-separated list of ETag "
        "tokens (optional whitespace around each comma), where each token is either a STRONG "
        "validator `'\"<opaque>\"'` or a WEAK validator `'W/\"<opaque>\"'` (the literal two "
        "characters `W/` followed by a quoted opaque string).\n"
        "    - If `path` does not currently exist, the write MUST fail regardless of what "
        "`if_match` says — even `\"*\"` — because there is no current representation for "
        "anything to match against. Return `(412, None)` and do not store anything.\n"
        "    - Otherwise, compare `if_match` against the resource's CURRENT stored ETag using "
        "STRONG comparison: `\"*\"` always matches (the resource exists, so there is *some* "
        "current representation); for a comma-separated list, the write proceeds if ANY "
        "listed token satisfies STRONG comparison against the current ETag. Two validators "
        "satisfy STRONG comparison only if NEITHER of them carries the `W/` weak prefix AND "
        "their opaque (unquoted, prefix-stripped) contents are equal — a weak-marked token "
        "must NEVER be treated as a strong match, even if its opaque content is identical to "
        "the current ETag's.\n"
        "    - If the comparison matches: proceed exactly as the unconditional case above "
        "(store `body`, return `(200, new_etag)` — since the resource already existed, this "
        "is always `200`, never `201`). If it does not match: return `(412, None)` and do not "
        "store anything.\n"
        "- `get(self, path: str, if_none_match: str | None = None) -> tuple` — read `path`.\n"
        "  - If `path` does not exist: return `(404, None, None)`, regardless of "
        "`if_none_match`.\n"
        "  - If `if_none_match` is `None`: return `(200, current_etag, current_body)`.\n"
        "  - If `if_none_match` is not `None` (same `\"*\"` / comma-separated-list shape as "
        "`if_match` above): compare it against the current ETag using WEAK comparison this "
        "time — `\"*\"` always matches; for a list, matching against ANY listed token is "
        "sufficient. Two validators satisfy WEAK comparison whenever their opaque "
        "(unquoted, prefix-stripped) contents are equal, REGARDLESS of whether either side "
        "carries the `W/` prefix (unlike strong comparison above, a weak-marked token here "
        "counts as a match as long as the opaque content matches). If it matches: return "
        "`(304, current_etag, None)` — no body. If it does not match: return `(200, "
        "current_etag, current_body)`.\n\n"
        "The critical distinction to get right: `if_match` (used by `put`) always uses STRONG "
        "comparison; `if_none_match` (used by `get`) always uses WEAK comparison. These are "
        "genuinely different comparison RULES, not just different call sites for the same "
        "rule — a weak-marked client validator (`W/\"...\"`) must be REJECTED by `if_match`'s "
        "strong comparison even when its opaque content exactly matches the current ETag, but "
        "that exact same weak-marked validator must be ACCEPTED by `if_none_match`'s weak "
        "comparison. Implementing both comparisons with the same underlying logic (e.g. only "
        "comparing opaque content and never checking the `W/` prefix at all) will satisfy the "
        "easy exact-match cases but silently get the weak-validator cases backwards.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your own "
        "work before finishing. At minimum, confirm: (1) putting a new path returns 201, and "
        "putting an existing one again returns 200; (2) the same body content always produces "
        "the same ETag; (3) get on a path that was never written returns (404, None, None); "
        "(4) a put with `if_match` set to some ETag string that does NOT match the current one "
        "returns (412, None) and leaves the stored body/ETag unchanged; (5) a put with "
        "`if_match=\"*\"` on a path that does not exist yet ALSO returns (412, None) — the "
        "wildcard still requires something to already be there; (6) once that same path does "
        "exist, `if_match=\"*\"` then succeeds; (7) take an existing resource's real ETag "
        "string, prepend the two characters `W/` to it to build a weak-marked token with the "
        "identical opaque content, and confirm a `put` using that weak token as `if_match` "
        "still returns (412, None) even though the opaque content matches; (8) confirm a `get` "
        "using that SAME weak-marked token as `if_none_match` returns (304, <etag>, None) — "
        "the opposite outcome from (7) despite using the identical token, because "
        "`if_none_match` uses weak comparison; (9) build a comma-separated `if_match` (or "
        "`if_none_match`) string containing a couple of clearly-wrong decoy tokens plus the "
        "real current ETag somewhere in the list, and confirm the match still succeeds."
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
            "module": "resource_store",
            "class": "ResourceStore",
            "methods": ["put", "get"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one small, cohesive stateful class — nothing here decomposes into
    independent sub-parts worth a separate leaf."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write resource_store.py defining a class ResourceStore with put(self, "
                    "path, body, if_match=None) -> tuple and get(self, path, "
                    "if_none_match=None) -> tuple, modeling HTTP conditional-request "
                    "semantics (If-Match / If-None-Match) for optimistic-concurrency writes "
                    "and cache validation. Every stored ETag is a STRONG validator computed "
                    "as a quoted lowercase hex SHA-256 digest of the UTF-8-encoded body "
                    "(content-addressed: identical content always yields the identical "
                    "ETag). put(): with if_match=None, write unconditionally and return "
                    "(201, new_etag) if the path was new or (200, new_etag) if it already "
                    "existed. With if_match set (either the literal '*' or a comma-separated "
                    "list of ETag tokens, each either a strong '\"opaque\"' token or a weak "
                    "'W/\"opaque\"' token), the write must fail with (412, None) — no state "
                    "change — if the path does not currently exist at all (even '*' cannot "
                    "match a nonexistent resource, since there is no current representation "
                    "to match against), OR if none of the supplied tokens satisfy STRONG "
                    "comparison against the current ETag ('*' itself always counts as a "
                    "match once the resource exists). Strong comparison: two validators are "
                    "equal only if NEITHER carries the 'W/' prefix AND their opaque "
                    "(unquoted, prefix-stripped) contents are equal — a weak-marked token "
                    "must never satisfy it. On a successful conditional match, proceed like "
                    "the unconditional case (always 200, since the resource already "
                    "existed). get(): if the path was never written, return (404, None, "
                    "None) regardless of if_none_match. With if_none_match=None, return "
                    "(200, current_etag, current_body). With if_none_match set (same '*' / "
                    "comma-list shape), use WEAK comparison instead: two validators are "
                    "equal whenever their opaque contents match REGARDLESS of either side's "
                    "'W/' prefix status. If any supplied token weak-matches (or the token is "
                    "'*'), return (304, current_etag, None) — no body; otherwise return "
                    "(200, current_etag, current_body). The load-bearing distinction: "
                    "if_match always uses strong comparison, if_none_match always uses weak "
                    "comparison — these are different comparison RULES that must be kept "
                    "genuinely separate in your implementation (not just called from two "
                    "different places using one shared 'compare opaque parts' helper that "
                    "never checks the W/ prefix anywhere), since a weak-marked client token "
                    "must be rejected by if_match's strong check but accepted by "
                    "if_none_match's weak check even when its opaque content is identical to "
                    "the current ETag in both cases. Use write_file to create "
                    "resource_store.py. Then use run_python to sanity check it: create a "
                    "resource, confirm 201-then-200 on repeated unconditional puts, confirm "
                    "a stale if_match returns 412 without changing anything, confirm "
                    "if_match='*' fails on a not-yet-created path but succeeds once it "
                    "exists, take a real current ETag string and prepend 'W/' to build a "
                    "weak-marked token with the same opaque content and confirm that token "
                    "fails as if_match but succeeds as if_none_match, and confirm a "
                    "comma-separated list of decoy tokens plus the real ETag still matches. "
                    "Fix issues with patch_file/run_python until confident, then finish."
                ),
                "expect": (
                    "resource_store.py written; ResourceStore.put/get correctly implement "
                    "content-addressed ETags, the 201-vs-200 existence distinction, the "
                    "wildcard-requires-existing-resource rule, comma-separated multi-value "
                    "matching, and — the sharp edge — genuinely different strong (if_match) "
                    "vs weak (if_none_match) comparison semantics for weak-marked validators"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": (
            "Confirm resource_store.py exists and report the agent's own sanity-check "
            "results (unconditional put/get, stale if_match rejection, wildcard-on-absent-"
            "resource rejection, and the weak-validator strong-vs-weak comparison cases)."
        ),
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["resource_store.py"]},
    }
