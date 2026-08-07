"""
codebench task c37 — hard/hidden, data structure implementation (a fixed-lane round-robin
dispatcher with a promotion operation that must NOT disturb whose turn is next).

SECOND REPLACEMENT NOTE (2026-08-07): the previous version of this task (LFU-style frequency
cache with an immutable-original-insertion-order tie-break, replacing an even earlier binary-
min-heap-with-FIFO-tie-break design) was itself a full replacement chosen specifically to move
away from an over-recognized textbook shape. It STILL scored 1.0/1.0 for qwen2.5:14b on Aider in
a third live-calibration round (badmodel scored 0.0, having implemented the tie-break rule
BACKWARDS — evicting the most-recently-inserted of a tied pair instead of the oldest). Inspecting
Aider's actual submitted code revealed something more interesting than "it got lucky": its
tie-break bookkeeping is
```python
self.insertion_time[key] = len(self.cache)   # set AFTER inserting into self.cache
```
— i.e. "insertion time" is really just "cache size at the moment of insertion," which is bounded
between 1 and `capacity` and therefore collapses to the SAME value (`capacity`) for essentially
every insertion after the first fill cycle. This looks broken, and in isolation it is — except
that Python's `dict` never reorders existing keys on update and always appends brand-new keys at
the end, so `self.frequency.items()` (which this submission scans to build its candidate list)
iterates the currently-resident keys in EXACTLY true chronological insertion order regardless of
how many evictions have happened, and `min()` with a key function is stable — on a tie it returns
the FIRST candidate in iteration order. The result: no matter how degenerate `insertion_time`
becomes, ties always fall back to the dict's own iteration order, which is ALWAYS correct for
this task's tie-break rule. This is not a coincidence that better test cases could have caught —
it is a provable structural fact about Python `dict` semantics that makes "frequency + a
per-key scalar you *think* is an insertion-order counter, tie-broken via `min()`" UNFALSIFIABLE
by any sequence of `get`/`put` operations for THIS tie-break axis, however the scalar is
computed, as long as it is never inverted between two keys relative to their true order (which a
"cache-size-at-insertion" scalar structurally cannot be, since it is non-decreasing across the
insertion sequence). In short: a single-axis frequency-plus-insertion-order LFU variant is
fundamentally too tractable in Python for competent dict-and-min()-reaching-for implementers,
regardless of how the tie-break rule is dressed up — hardening the SAME axis further cannot fix
this, so this is a full domain replacement rather than a third revision of the same shape (same
precedent as c39/c41's full replacements after two failed hardening rounds).

The new task: `LaneDispatcher` — a small, fixed set of named "lanes" (think: priority queues for
a print server, a support-ticket router, a network scheduler), each holding its own FIFO of
pending items, served in ROUND-ROBIN order across lanes so no single busy lane starves the
others. This is NOT a famous, single-canonical-implementation algorithm the way LFU/LRU/binary
search/priority queues are — "round robin across queues" is a real, useful pattern (network
schedulers, OS process scheduling) but has no one memorized reference shape for this exact,
bespoke micro-API, and — critically — round-robin state here is tracked via a MUTABLE LIST whose
order can be explicitly rearranged (`promote`), so the "Python dict iteration order does the
correctness work for you" trick that defeated the previous LFU design does not apply: there is
no dict here doing any of the work, only whatever bookkeeping the implementation deliberately
chooses for "which lane is due next."

Full contract (see get_task_statement() for the version shown to the agent):
  - `LaneDispatcher(lanes: list[str])` — fixed set of uniquely-named lanes, each starting with an
    empty FIFO queue, in a given initial cycle order.
  - `enqueue(lane, item)` — append to the back of that lane's queue; `KeyError` if unknown.
  - `dispatch()` — serve exactly one item: scan lanes starting from whichever lane is currently
    "due" (continuing across calls, not restarting at the first lane every time), wrapping
    around the cycle, skipping any empty lane, and popping the front item of the first non-empty
    lane found. The lane immediately AFTER the one just served (in the CURRENT cycle order at
    the moment of THIS call) becomes due for the next call. If every lane is empty, raise
    `IndexError` and leave the due lane unchanged.
  - `promote(lane)` — move `lane` to the FRONT of the cycle order (relative order of the other
    lanes unchanged). This changes the order future FULL-CYCLE scans will visit lanes in, but it
    must NOT change which lane is currently due next — promoting some other lane can never let
    it cut in line ahead of whichever lane `dispatch()` was already about to serve.
  - `pending_count(lane)` — items currently queued in that lane; `KeyError` if unknown.

Why this resists the two most natural implementation shapes, MEASURED directly (not asserted from
authorial confidence — see services/agent/tests/idea_code_test_c37_test.py, which actually
executes all three variants against the real canonical suite via pytest):
  (1) tracking "due lane" as a raw NUMERIC INDEX into the lane list is completely correct for
      round-robin scanning as long as the lane list never changes — it is the "obvious," and for
      plain round-robin (no `promote` calls) genuinely CORRECT, implementation. But `promote`
      physically reorders the list, silently invalidating any numeric index recorded before the
      reorder — it now may point at a DIFFERENT lane than the one that was actually due. This
      mutant passes every plain round-robin scenario (no promote) and even survives some promote
      scenarios where the stale index happens to land on an empty lane and searches forward to
      the right answer anyway by luck — but fails decisively (returns an item from the WRONG
      lane, out of turn) once promote() is called while the lane at the stale index still has a
      pending item of its own. Verified: 6/14 canonical tests fail, and a 300-step randomized
      fuzz cross-check against an independently-coded oracle diverges within the first handful of
      operations on every one of 30 fixed seeds.
  (2) reading "promote" as "prioritize THIS lane starting right now" (setting the due lane to the
      promoted lane immediately, not merely reordering future scans) is a very natural misreading
      of the word "promote" — it is wrong precisely because the contract explicitly forbids a
      promoted lane from cutting in line ahead of whichever lane was already due. Verified: 3/14
      canonical tests fail (every scenario that calls `promote` on a lane other than the one
      already due), and the same fuzz cross-check diverges on all 30 seeds.
Both mutants pass every bonus/basic test (FIFO within a lane, missing-lane errors, empty-lane
skipping, plain round robin with no promote calls at all) — neither one is a broken, non-running
implementation; both are genuine, independently plausible near-misses.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_lane_dispatcher.py"

_TEST_FILE_CONTENT = '''\
import random

import pytest
from lane_dispatcher import LaneDispatcher


def test_basic_fifo_order_within_a_single_lane():
    d = LaneDispatcher(["a"])
    d.enqueue("a", 1)
    d.enqueue("a", 2)
    d.enqueue("a", 3)
    assert d.dispatch() == 1
    assert d.dispatch() == 2
    assert d.dispatch() == 3


def test_dispatch_raises_index_error_when_every_lane_is_empty():
    d = LaneDispatcher(["a", "b"])
    with pytest.raises(IndexError):
        d.dispatch()


def test_unknown_lane_name_raises_key_error_everywhere():
    d = LaneDispatcher(["a"])
    with pytest.raises(KeyError):
        d.enqueue("z", 1)
    with pytest.raises(KeyError):
        d.pending_count("z")
    with pytest.raises(KeyError):
        d.promote("z")


def test_plain_round_robin_with_no_promote_calls():
    d = LaneDispatcher(["a", "b", "c"])
    d.enqueue("a", "a1")
    d.enqueue("b", "b1")
    d.enqueue("c", "c1")
    d.enqueue("a", "a2")
    d.enqueue("b", "b2")
    assert d.dispatch() == "a1"
    assert d.dispatch() == "b1"
    assert d.dispatch() == "c1"
    assert d.dispatch() == "a2"
    assert d.dispatch() == "b2"


def test_empty_lane_is_skipped_over_in_the_scan():
    d = LaneDispatcher(["a", "b", "c"])
    d.enqueue("a", "a1")
    d.enqueue("c", "c1")
    assert d.dispatch() == "a1"
    assert d.dispatch() == "c1"  # "b" has nothing queued and must be skipped, not error


def test_pending_count_tracks_state_through_enqueue_and_dispatch():
    d = LaneDispatcher(["a", "b"])
    assert d.pending_count("a") == 0
    d.enqueue("a", 1)
    d.enqueue("a", 2)
    assert d.pending_count("a") == 2
    d.dispatch()
    assert d.pending_count("a") == 1


def test_promote_does_not_let_a_lane_cut_in_line_ahead_of_the_lane_already_due():
    d = LaneDispatcher(["a", "b", "c"])
    d.enqueue("a", "a1")
    d.enqueue("b", "b1")
    d.enqueue("c", "c1")
    assert d.dispatch() == "a1"  # "b" is now due next
    d.promote("c")  # must NOT jump ahead of "b"
    assert d.dispatch() == "b1"


def test_promote_reorders_future_scans_without_disturbing_the_currently_due_lane():
    d = LaneDispatcher(["a", "b", "c", "d"])
    d.enqueue("a", "a1")
    d.enqueue("b", "b1")
    d.enqueue("c", "c1")
    d.enqueue("d", "d1")
    assert d.dispatch() == "a1"  # "b" now due
    d.promote("d")  # cycle order becomes [d, a, b, c] -- "b" is still due, unaffected
    assert d.dispatch() == "b1"
    # future scanning now follows the NEW order [d, a, b, c], continuing after "b" -> c, d, a
    assert d.dispatch() == "c1"
    assert d.dispatch() == "d1"


def test_promote_with_a_pending_item_still_sitting_in_the_reordered_lane():
    # "a" has TWO pending items; after the first is served, "b" becomes due. Promoting "c" must
    # not disturb that -- "b" must still be served next, NOT a's remaining item, even though a
    # bug that tracks "due lane" as a raw list index (rather than by lane identity) would, after
    # the reorder, have that stale index land squarely on "a" (which still has a pending item of
    # its own, so a naive implementation would NOT even notice anything looks wrong).
    d = LaneDispatcher(["a", "b", "c"])
    d.enqueue("a", 1)
    d.enqueue("a", 2)
    d.enqueue("b", 10)
    d.enqueue("c", 100)
    assert d.dispatch() == 1        # "a" served once, "b" now due
    d.promote("c")                  # cycle order -> [c, a, b]; due lane must STILL be "b"
    assert d.dispatch() == 10       # "b", not a's leftover item
    assert d.dispatch() == 100      # "c" next in the new cycle order
    assert d.dispatch() == 2        # wraps back around to a's remaining item
    with pytest.raises(IndexError):
        d.dispatch()


def test_promoting_a_lane_already_at_the_front_is_a_harmless_no_op():
    d = LaneDispatcher(["a", "b"])
    d.enqueue("a", 1)
    d.enqueue("b", 2)
    d.promote("a")  # already first in the cycle
    assert d.dispatch() == 1
    assert d.dispatch() == 2


def test_promoting_the_currently_due_lane_itself_changes_nothing_observable():
    d = LaneDispatcher(["a", "b", "c"])
    d.enqueue("a", 1)
    d.enqueue("b", 2)
    d.enqueue("c", 3)
    assert d.dispatch() == 1  # "b" now due
    d.promote("b")  # promoting the lane that's ALREADY due must not skip its own turn
    assert d.dispatch() == 2
    assert d.dispatch() == 3


def test_dispatch_after_all_lanes_drain_then_refill_still_works():
    d = LaneDispatcher(["a", "b"])
    d.enqueue("a", 1)
    assert d.dispatch() == 1
    with pytest.raises(IndexError):
        d.dispatch()
    d.enqueue("b", 2)
    assert d.dispatch() == 2


def test_zero_configured_lanes_dispatch_and_enqueue_both_error():
    d = LaneDispatcher([])
    with pytest.raises(IndexError):
        d.dispatch()
    with pytest.raises(KeyError):
        d.enqueue("x", 1)


def test_long_randomized_sequence_matches_an_independent_oracle():
    # A from-scratch oracle -- rotating a SLICE of the lane-order list on every dispatch call
    # rather than doing modulo index arithmetic the way the task module's own reference does --
    # is fuzzed against the submitted LaneDispatcher over a long, fixed-seed sequence of
    # enqueue/dispatch/promote/pending_count operations.
    class _Oracle:
        def __init__(self, lanes):
            self.order = list(lanes)
            self.queues = {lane: [] for lane in lanes}
            self.next_lane = self.order[0] if self.order else None

        def enqueue(self, lane, item):
            if lane not in self.queues:
                raise KeyError(lane)
            self.queues[lane].append(item)

        def dispatch(self):
            if not self.order:
                raise IndexError("no lanes")
            start = self.order.index(self.next_lane)
            rotated = self.order[start:] + self.order[:start]
            for lane in rotated:
                if self.queues[lane]:
                    item = self.queues[lane].pop(0)
                    served_idx = self.order.index(lane)
                    self.next_lane = self.order[(served_idx + 1) % len(self.order)]
                    return item
            raise IndexError("all lanes empty")

        def promote(self, lane):
            if lane not in self.queues:
                raise KeyError(lane)
            self.order.remove(lane)
            self.order.insert(0, lane)

        def pending_count(self, lane):
            if lane not in self.queues:
                raise KeyError(lane)
            return len(self.queues[lane])

    seed = 20260807
    rng = random.Random(seed)
    lanes = ["a", "b", "c", "d"]
    dispatcher = LaneDispatcher(list(lanes))
    oracle = _Oracle(list(lanes))
    for step in range(300):
        op = rng.choice(["enqueue", "dispatch", "promote", "pending_count"])
        if op == "enqueue":
            lane = rng.choice(lanes)
            dispatcher.enqueue(lane, step)
            oracle.enqueue(lane, step)
        elif op == "dispatch":
            d_exc = o_exc = None
            try:
                d_val = dispatcher.dispatch()
            except IndexError:
                d_exc = "IndexError"
                d_val = None
            try:
                o_val = oracle.dispatch()
            except IndexError:
                o_exc = "IndexError"
                o_val = None
            assert d_exc == o_exc
            assert d_val == o_val
        elif op == "promote":
            lane = rng.choice(lanes)
            dispatcher.promote(lane)
            oracle.promote(lane)
        else:
            lane = rng.choice(lanes)
            assert dispatcher.pending_count(lane) == oracle.pending_count(lane)
'''

# The interaction between `promote` and "which lane is due next" is the entire point of this
# task -- so the tests that isolate that interaction gate the score: not-cutting-in-line, the
# reordering-without-disturbing-due-lane test, the worked scenario with a pending item sitting in
# the reordered lane a stale-index bug would land on, promoting the already-due lane being
# harmless, and the long randomized oracle cross-check. Basic FIFO-within-a-lane, missing-lane
# errors, plain round robin with no promote calls, empty-lane skipping, pending_count bookkeeping,
# promoting an already-first lane, drain-then-refill, and zero lanes are supporting/bonus credit
# -- see the module docstring's mutant analysis for exactly what a plausible-but-incomplete
# implementation gets wrong and what it still gets right by accident.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_promote_does_not_let_a_lane_cut_in_line_ahead_of_the_lane_already_due",
    f"{_TEST_FILE_PATH}::test_promote_reorders_future_scans_without_disturbing_the_currently_due_lane",
    f"{_TEST_FILE_PATH}::test_promote_with_a_pending_item_still_sitting_in_the_reordered_lane",
    f"{_TEST_FILE_PATH}::test_promoting_the_currently_due_lane_itself_changes_nothing_observable",
    f"{_TEST_FILE_PATH}::test_long_randomized_sequence_matches_an_independent_oracle",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c37",
        "title": "lane-dispatcher",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `lane_dispatcher.py` that defines a class `LaneDispatcher` — a "
        "fixed set of named \"lanes\" (think: priority queues for a print server or a "
        "support-ticket router), each holding its own first-in-first-out queue of pending "
        "items, served in ROUND-ROBIN order across lanes so that one constantly-busy lane can "
        "never starve the others.\n\n"
        "Implement:\n"
        "- `LaneDispatcher(lanes: list[str])` — construct a dispatcher with the given lane "
        "names (all unique), each starting with an empty queue. `lanes` also establishes the "
        "INITIAL cycle order used for round-robin scanning.\n"
        "- `enqueue(self, lane, item) -> None` — append `item` to the back of `lane`'s queue. "
        "Raise `KeyError` if `lane` is not one of the configured lane names.\n"
        "- `dispatch(self)` — serve and return exactly ONE item. Conceptually, there is always "
        "a single lane that is currently \"due\" to be checked first (this carries over "
        "between calls — it does NOT reset to the first lane every time `dispatch` is called). "
        "Starting from the due lane and moving forward through the CURRENT cycle order "
        "(wrapping around after the last lane back to the first), find the first lane that "
        "currently has at least one item queued, and pop+return the item from the FRONT of "
        "that lane's queue. Whichever lane is immediately AFTER the one just served, in the "
        "cycle order as it stands at the moment of THIS call, becomes the due lane for the "
        "NEXT call to `dispatch`. If every lane is empty, raise `IndexError` and do not change "
        "which lane is due.\n"
        "- `promote(self, lane) -> None` — move `lane` to the FRONT of the cycle order, "
        "without changing the relative order of the other lanes (e.g. cycle `[a, b, c, d]`, "
        "`promote(\"c\")` produces `[c, a, b, d]`). Raise `KeyError` if `lane` is not "
        "configured. Promoting a lane changes the order that FUTURE full-cycle scans will "
        "visit lanes in — but it must NEVER change which lane is currently due for the very "
        "next `dispatch` call. In particular, promoting some OTHER lane can never let it cut "
        "in line ahead of whichever lane was already due; and promoting the lane that is "
        "already due changes nothing observable at all (it was already going to be checked "
        "first regardless of its position in the cycle order).\n"
        "- `pending_count(self, lane) -> int` — the number of items currently queued in "
        "`lane`. Raise `KeyError` if `lane` is not configured.\n\n"
        "Worked example of the trickiest part — what `promote` does and does NOT affect (read "
        "carefully):\n"
        "```\n"
        "d = LaneDispatcher([\"north\", \"south\", \"east\"])\n"
        "d.enqueue(\"north\", \"n1\")\n"
        "d.enqueue(\"south\", \"s1\")\n"
        "d.enqueue(\"east\", \"e1\")\n"
        "d.dispatch()              # -> \"n1\"  (serves \"north\"; \"south\" now due)\n"
        "d.promote(\"east\")        # cycle order becomes [east, north, south] -- but\n"
        "                           # \"south\" is STILL due, unaffected by this call\n"
        "d.dispatch()              # -> \"s1\"  (\"south\", not disturbed by the reorder)\n"
        "d.dispatch()              # -> \"e1\"  (\"east\", next in the NEW cycle order,\n"
        "                           # right after \"south\")\n"
        "```\n"
        "Notice `promote` only changed the order lanes are considered in for scans that "
        "start AFTER the call — it never changed which specific lane was already due at the "
        "moment `promote` was called.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your own "
        "work before finishing. At minimum, confirm: (1) plain round-robin across several "
        "lanes with no `promote` calls serves items in the correct cycling order and correctly "
        "skips any lane that's currently empty; (2) `dispatch` raises `IndexError` when every "
        "lane is empty, and this does not corrupt which lane becomes due once new items "
        "arrive; (3) `KeyError` on any unconfigured lane name, for every method that takes "
        "one; and (4) go further than the worked example above — build a scenario where the "
        "lane a `promote` call moves is NOT the currently-due lane, and where some OTHER lane "
        "(not the one being promoted, and not the one already due) still has a leftover item "
        "of its own sitting around. Confirm the lane that was already due still gets served "
        "next in that case too, regardless of whatever position the promoted lane's move might "
        "shift things into — think carefully about whether your bookkeeping for \"which lane "
        "is due\" refers to a specific lane, or to a position/index that a reorder could quietly "
        "make point at a different lane instead."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "lane_dispatcher", "classes": ["LaneDispatcher"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one small, cohesive stateful class — nothing here decomposes into
    independent sub-parts worth a separate leaf, same rationale as c08/c37's predecessors."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write lane_dispatcher.py implementing class LaneDispatcher with "
                    "__init__(self, lanes) (fixed set of uniquely-named lanes, each an empty "
                    "FIFO queue, establishing an initial cycle order), enqueue(self, lane, "
                    "item) -> None (append to that lane's queue, KeyError if lane unknown), "
                    "dispatch(self) (serve exactly one item: there is always a 'due' lane that "
                    "persists across calls rather than resetting to the first lane every time; "
                    "scan forward from the due lane through the CURRENT cycle order, wrapping "
                    "around, skipping empty lanes, pop+return the front item of the first "
                    "non-empty lane found; the lane immediately after the one just served, in "
                    "the cycle order AS OF this call, becomes due for next time; IndexError if "
                    "every lane is empty, leaving the due lane unchanged), promote(self, lane) "
                    "-> None (move lane to the front of the cycle order without disturbing the "
                    "relative order of the others; KeyError if unknown; this changes future "
                    "full-cycle scan order but must NEVER change which lane is due for the "
                    "very next dispatch call -- a promoted lane can never cut in line ahead of "
                    "whatever lane was already due), and pending_count(self, lane) -> int "
                    "(KeyError if unknown). The single trickiest part of this task is "
                    "promote(): it is easy to accidentally implement it in a way that changes "
                    "which lane gets served next even though the contract explicitly forbids "
                    "that -- whatever internal bookkeeping you use to track 'which lane is "
                    "currently due,' make sure a call to promote() (which physically reorders "
                    "your lane list) can never cause that bookkeeping to end up referring to a "
                    "different lane than the one that was actually due before the reorder "
                    "happened. Use write_file to create lane_dispatcher.py, then use "
                    "run_python to sanity check it: confirm plain round-robin behavior across "
                    "several lanes with no promote calls, confirm empty lanes are skipped and "
                    "IndexError is raised only when every lane is truly empty, confirm KeyError "
                    "on unconfigured lane names, and specifically build a scenario where you "
                    "promote a lane OTHER than the one currently due while a different lane "
                    "still has an item queued -- confirm the lane that was already due still "
                    "gets served next, not whatever item your bookkeeping might mistakenly "
                    "land on after the reorder. Fix issues with patch_file/run_python until "
                    "confident, then finish."
                ),
                "expect": "lane_dispatcher.py written, defining LaneDispatcher with correct "
                          "round-robin dispatch AND a promote() that reorders future scans "
                          "without ever changing which lane is currently due, sanity-checked "
                          "with run_python",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm lane_dispatcher.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["lane_dispatcher.py"]},
    }
