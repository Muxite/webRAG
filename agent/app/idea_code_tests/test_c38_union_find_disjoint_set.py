"""
codebench task c38 — hard/hidden, data structure implementation (union-find / disjoint-set).

Data-structure-implementations category, beyond c08's LRU-cache coverage and distinct from
c36 (trie) and c37 (priority queue) — this one's discriminating edge case is proving PATH
COMPRESSION is a genuine structural mutation, not an invisible optimization detail, via a raw
``parent_of``/``rank_of`` accessor pair that must never itself compress anything.

The exact union-by-rank tie-break convention below (equal ranks -> x's root absorbs y's root,
x's root's rank increments) is FIXED by the task statement, same spirit as c08 pinning exact
LRU recency semantics — set membership (``connected``) is convention-independent, but the raw
``parent_of``/``rank_of`` values are not, so the contract nails down one specific, deterministic
tree shape to assert against.

Ground truth verified independently by RUNNING a throwaway reference implementation (iterative
find with path compression, NOT hand-traced) — see the offline test file's own reimplementation
(built with recursive find + a plain dict instead of a list, structurally unrelated to the
iterative/list-array approach) for the second, independent check. Verified trace over 8
elements 0..7 (every union() call also returns the shown bool):
    union(0,1)->True  union(2,3)->True  union(0,2)->True
    union(4,5)->True  union(6,7)->True  union(4,6)->True
    union(0,4)->True
    # raw parent pointers right before find(7) is ever called: 7->6->4->0 (three real links)
    parent_of(7)==6   parent_of(6)==4   parent_of(4)==0
    find(7) -> 0
    # path compression must flatten EVERY node visited during that walk, not just 7 itself:
    parent_of(7)==0   parent_of(6)==0
    parent_of(3)==2   (untouched -- not on the find(7) path)
    find(3) -> 0       # its own, still-uncompressed 3->2->0 chain flattens on this call
    parent_of(3)==0   parent_of(2)==0
    connected(3,7)==True   connected(0,5)==True
    union(3,7) -> False   (already the same set -- a real no-op, must not raise or corrupt state)

Hardened 2026-08-06 after live calibration showed a strong coding agent (Aider) acing this
task on the first try: the previous canonical suite only ever exercised union-by-rank on
SINGLETON-vs-singleton or already-a-root cases, so it never actually forced an implementation
to resolve `find(x)` to x's CURRENT root before comparing ranks -- a spec-compliant-LOOKING
but subtly wrong `union` that compares `rank[x]`/`rank[y]` directly (the ranks of the
ARGUMENTS as given, rather than `rank[find(x)]`/`rank[find(y)]`, the ranks of their resolved
ROOTS) passed every existing test, because in every existing test `x` and `y` happened to
already BE roots whenever union() was called on them. Continuing the same 8-element forest
from above (working out `rank_of(0)` by RUNNING the sequence, not assuming it: after
union(4,6), node 4's root has rank 2 -- same as node 0's root's rank 2 at that point -- so the
final union(0,4) is itself an equal-rank tie, bringing `rank_of(0)` to 3; find(7) and find(3)
afterward only compress paths, they never touch rank) and adding one fresh singleton element 8
(rank 0), verified with the same independent reference (re-run fresh, not hand-traced):
    union(3, 8) -> True
        (3's CURRENT root is 0, rank 3; 8 is its own root, rank 0 -- ranks 3 vs 0 are NOT
        tied, so 8 attaches under 0 and rank_of(0) stays 3. A buggy implementation that
        mistakenly compares node 3's own raw rank (0, stale/irrelevant since 3 hasn't been a
        root since long before this point) against node 8's rank (also 0) sees a FALSE tie,
        takes the equal-rank branch, and incorrectly increments rank_of(0) to 4.)
    rank_of(0) == 3 (unchanged)   parent_of(8) == 0   parent_of(3) == 0 (unaffected)

Hardened AGAIN 2026-08-07 after round-2 live calibration showed Aider still acing this task
(1.0) even with the 2026-08-06 resolved-root-rank case above in place: that case's SECOND
argument (`y=8`) is a FRESH singleton that IS its own root at the moment of the call, so
`rank_of(8)` (raw, as stored) and the rank of `find(8)` (resolved) are numerically IDENTICAL --
the case is only actually capable of catching a union() that forgets to resolve its FIRST
argument (`x`) before comparing ranks (confirmed by RUNNING three separate mutants against it:
one that uses raw `rank[x]`/raw `rank[y]` for the comparison, and one that uses raw `rank[x]`
paired with a correctly-resolved `rank[find(y)]`, are BOTH still caught by the 2026-08-06 case
above -- but a THIRD mutant that correctly resolves `rank[find(x)]` and only forgets to resolve
`y` (comparing `rank[find(x)]` against `y`'s raw, un-resolved `rank[y]` instead of
`rank[find(y)]`) passes it, because for that specific case swapping `y`'s raw rank for its
resolved rank changes nothing). This is exactly the kind of copy-paste "fixed one occurrence,
not the other" half-fix a plausible implementation could produce after correctly internalizing
"resolve arguments via find() before comparing ranks" as a general principle but only actually
applying it to one of the two comparison sites. The fix: a SECOND scenario where the roles are
reversed -- the FIRST argument is already a root (so mishandling it would be invisible) and the
SECOND argument is the one that needs resolving. Continuing the same 8-element forest (root 0,
rank 3, after find(7) and find(3) have both already compressed their paths) plus a fresh pair
(8, 9) unioned into its own 2-element tree (root 8, rank 1), verified with the same independent
reference (re-run fresh):
    union(8, 6) -> True
        (8 is already its own root, rank 1 -- resolving it changes nothing, so this scenario
        gives no signal about whether `x` was resolved. 6 is NOT a root: its raw stored rank
        is 1 (set once, at union(6,7), never touched again), but 6's CURRENT resolved root is
        0, with rank 3. Ranks 1 vs 3 are NOT tied, so root 8 attaches under root 0, and
        rank_of(8) stays 1. A buggy implementation that resolves `x=8` correctly but compares
        `rank_of(8)` against node 6's own raw rank (1, stale -- 6 hasn't been a root since
        union(4,6)) instead of `rank_of(find(6))` (3) sees a FALSE tie (1 == 1), takes the
        equal-rank branch, and gets the attachment BACKWARDS -- root 0's entire tree ends up
        absorbed under root 8, and rank_of(8) is wrongly incremented to 2.)
    parent_of(8) == 0   parent_of(0) == 0 (0 remains the overall root)
    rank_of(8) == 1 (unchanged)   rank_of(0) == 3 (unchanged)
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_union_find.py"

_TEST_FILE_CONTENT = '''\
from union_find import DisjointSet


def test_initial_state_all_singletons():
    ds = DisjointSet(5)
    for i in range(5):
        assert ds.find(i) == i
    assert ds.connected(0, 1) is False


def test_union_merges_two_singletons_and_reports_true():
    ds = DisjointSet(3)
    assert ds.union(0, 1) is True
    assert ds.connected(0, 1) is True


def test_union_on_already_connected_pair_returns_false_and_is_a_noop():
    ds = DisjointSet(3)
    ds.union(0, 1)
    assert ds.union(0, 1) is False
    assert ds.connected(0, 1) is True


def test_union_by_rank_tie_break_convention():
    ds = DisjointSet(2)
    assert ds.union(0, 1) is True
    assert ds.parent_of(1) == 0
    assert ds.parent_of(0) == 0
    assert ds.rank_of(0) == 1


def _build_eight_element_forest():
    ds = DisjointSet(8)
    ds.union(0, 1)
    ds.union(2, 3)
    ds.union(0, 2)
    ds.union(4, 5)
    ds.union(6, 7)
    ds.union(4, 6)
    ds.union(0, 4)
    return ds


def test_set_membership_after_sequence_of_unions():
    ds = _build_eight_element_forest()
    assert ds.connected(0, 3) is True
    assert ds.connected(1, 2) is True
    assert ds.connected(4, 7) is True
    assert ds.connected(3, 7) is True
    assert ds.connected(1, 5) is True


def test_path_compression_flattens_the_tree_on_find():
    ds = _build_eight_element_forest()
    # Before any find() on 7, the raw parent chain is genuinely multi-level: 7 -> 6 -> 4 -> 0.
    assert ds.parent_of(7) == 6
    assert ds.parent_of(6) == 4
    assert ds.parent_of(4) == 0
    root = ds.find(7)
    assert root == 0
    # Path compression must repoint every node visited during find(7) directly to the root.
    assert ds.parent_of(7) == 0
    assert ds.parent_of(6) == 0
    # A sibling branch not on the find(7) path must be untouched.
    assert ds.parent_of(3) == 2


def test_path_compression_is_transitive_on_the_next_find_too():
    ds = _build_eight_element_forest()
    ds.find(7)
    assert ds.parent_of(3) == 2
    root = ds.find(3)
    assert root == 0
    assert ds.parent_of(3) == 0
    assert ds.parent_of(2) == 0


def test_union_returns_false_and_structure_stable_when_called_on_already_merged_roots():
    ds = _build_eight_element_forest()
    ds.find(7)
    ds.find(3)
    assert ds.union(3, 7) is False
    assert ds.connected(3, 7) is True


def test_union_by_rank_resolves_arguments_to_their_current_roots_before_comparing_ranks():
    # Continue the same 8-element forest, but with 9 elements so there is a fresh singleton
    # (8) to union against. After find(7) and find(3) have both already run (compressing their
    # paths), node 3 is no longer a root -- its OWN stored rank is still 0 (untouched since
    # init), even though its CURRENT root (0) has rank 3 (union(4,6) brings node 4's root to
    # rank 2, matching node 0's root's rank 2 at that point, so the final union(0,4) is itself
    # an equal-rank tie that brings rank_of(0) to 3). Union-by-rank must compare the ranks of
    # the RESOLVED ROOTS (rank_of(0)=3 vs rank_of(8)=0 -- not tied), never the raw rank_of(3)
    # (0) vs rank_of(8) (0), which would look like a tie but is not the real comparison the
    # spec calls for.
    ds = DisjointSet(9)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds.union(a, b)
    ds.find(7)
    ds.find(3)
    assert ds.rank_of(0) == 3
    assert ds.rank_of(3) == 0
    assert ds.union(3, 8) is True
    assert ds.rank_of(0) == 3  # unchanged: true root ranks (3 vs 0) were not tied
    assert ds.parent_of(8) == 0
    assert ds.parent_of(3) == 0


def test_union_by_rank_also_resolves_the_second_argument_not_just_the_first():
    # Same 8-element forest, plus a fresh pair (8, 9) unioned into its own small tree (root 8,
    # rank 1). This time the FIRST argument to union() (8) is already a root -- resolving it
    # changes nothing, so this scenario gives no signal about the FIRST argument. The SECOND
    # argument (6) is NOT a root: its own raw stored rank is 1 (set once, at union(6, 7), never
    # touched again), but its CURRENT resolved root is 0, with rank 3 -- resolving it changes
    # everything. A union() that correctly resolves its first argument but forgets to resolve
    # its second (comparing rank_of(8) against node 6's raw rank_of(6)=1 instead of its
    # resolved root's rank_of(0)=3) sees a FALSE tie (1 == 1) and gets the attachment
    # completely backwards.
    ds = DisjointSet(10)
    for a, b in [(0, 1), (2, 3), (0, 2), (4, 5), (6, 7), (4, 6), (0, 4)]:
        ds.union(a, b)
    ds.find(7)
    ds.find(3)
    ds.union(8, 9)
    assert ds.rank_of(0) == 3
    assert ds.rank_of(6) == 1
    assert ds.rank_of(8) == 1
    assert ds.union(8, 6) is True
    assert ds.parent_of(8) == 0   # root(8) attaches UNDER root(0), never the reverse
    assert ds.parent_of(0) == 0   # 0 remains the overall root
    assert ds.rank_of(8) == 1     # unchanged: ranks (1 vs 3) were not tied
    assert ds.rank_of(0) == 3     # unchanged
'''

# The exact tie-break convention and the path-compression flattening behavior ARE the task
# (set membership alone is table-stakes for any union-find, correct or not) -- those, plus the
# already-connected no-op contract, the resolved-root-rank case added 2026-08-06, AND its
# 2026-08-07 companion (a union() that resolves its FIRST argument correctly but forgets to
# resolve its SECOND -- the 2026-08-06 case alone can't catch this, since that case's second
# argument was already a root, so resolving it changes nothing; this one's first argument is
# the trivial one instead, isolating the second-argument gap), gate the score. Initial-singleton
# state, the simple two-element merge, and the second-find/second-union confirmations are
# supporting/bonus credit that a partially-correct implementation could still pass.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_union_on_already_connected_pair_returns_false_and_is_a_noop",
    f"{_TEST_FILE_PATH}::test_union_by_rank_tie_break_convention",
    f"{_TEST_FILE_PATH}::test_set_membership_after_sequence_of_unions",
    f"{_TEST_FILE_PATH}::test_path_compression_flattens_the_tree_on_find",
    f"{_TEST_FILE_PATH}::test_union_by_rank_resolves_arguments_to_their_current_roots_before_comparing_ranks",
    f"{_TEST_FILE_PATH}::test_union_by_rank_also_resolves_the_second_argument_not_just_the_first",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c38",
        "title": "union-find-disjoint-set",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Write a Python module `union_find.py` that defines a class `DisjointSet` (a "
        "union-find / disjoint-set structure over the integers `0..n-1`) with:\n"
        "- `DisjointSet(n: int)` — construct `n` singleton sets, one per integer `0..n-1`, "
        "each initially its own representative with rank 0.\n"
        "- `find(self, x: int) -> int` — return the representative (root) of the set "
        "containing `x`. Every call must apply PATH COMPRESSION: after `find(x)` returns, "
        "every node visited while walking from `x` up to the root must have its internal "
        "parent pointer repointed directly to that root (not just `x` itself — the entire "
        "path).\n"
        "- `union(self, x: int, y: int) -> bool` — merge the sets containing `x` and `y`, "
        "using UNION BY RANK. Let `rx = find(x)` and `ry = find(y)`. If `rx == ry`, they are "
        "already the same set: do nothing and return `False`. Otherwise: if `rank[rx] < "
        "rank[ry]`, attach `rx` under `ry`; if `rank[rx] > rank[ry]`, attach `ry` under "
        "`rx`; if the ranks are EQUAL, attach `ry` under `rx` (`rx`'s root wins ties) and "
        "increment `rank[rx]` by 1. Return `True`. IMPORTANT: `rank[rx]`/`rank[ry]` always "
        "means the rank of the RESOLVED ROOT returned by `find`, never the rank stored "
        "directly against `x`/`y` themselves — if `x` (or `y`) is not currently a root, its "
        "own stored rank is stale and irrelevant to this comparison; always resolve to `rx`/"
        "`ry` via `find` first and compare THOSE ranks.\n"
        "- `connected(self, x: int, y: int) -> bool` — True iff `x` and `y` are currently in "
        "the same set (i.e. `find(x) == find(y)`).\n"
        "- `parent_of(self, x: int) -> int` — return `x`'s CURRENT immediate parent pointer "
        "exactly as stored internally (`x` itself if `x` is currently a root), WITHOUT "
        "walking to the root and WITHOUT applying any path compression. This is a raw "
        "accessor used only to inspect internal structure — unlike `find`, it must never "
        "mutate anything.\n"
        "- `rank_of(self, x: int) -> int` — return `x`'s current rank exactly as stored "
        "internally.\n\n"
        "Path compression is the trickiest part: it is a genuine structural change, not just "
        "an invisible optimization detail — `parent_of` exists specifically so it can be "
        "checked from outside. After enough unions, some elements sit several links below "
        "the root (e.g. `x`'s parent is `y`, `y`'s parent is `z`, `z`'s parent is the root) "
        "BEFORE any `find` has ever been called on them; calling `find(x)` once must flatten "
        "that ENTIRE chain, not just move `x` one level up.\n\n"
        "Worked example:\n"
        "```\n"
        "ds = DisjointSet(4)\n"
        "ds.union(0, 1)     # -> True; ranks equal (0,0) -> parent_of(1)==0, rank_of(0)==1\n"
        "ds.union(2, 3)     # -> True; parent_of(3)==2, rank_of(2)==1\n"
        "ds.union(0, 2)     # -> True; ranks equal (1,1) -> parent_of(2)==0, rank_of(0)==2\n"
        "ds.parent_of(3)    # -> 2   (raw pointer: 3 -> 2 -> 0, NOT yet compressed)\n"
        "ds.find(3)         # -> 0   (walks 3 -> 2 -> 0, and compresses along the way)\n"
        "ds.parent_of(3)    # -> 0   (now points directly at the root)\n"
        "ds.connected(1, 3) # -> True\n"
        "ds.union(0, 1)     # -> False (already in the same set; a real no-op)\n"
        "```\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "implementation yourself — reproduce the worked example above, and also build a "
        "longer union chain (at least 8 elements) so that some element sits 2+ links below "
        "the root before you ever call `find` on it, then confirm `parent_of` shows the "
        "flattened result immediately afterward. Also specifically test `union` where ONE "
        "argument is NOT currently a root (e.g. an element you already `find`-compressed "
        "earlier, so its own stored rank is old/low) against an argument that already IS a "
        "root — and do this with the non-root element in BOTH the first and second argument "
        "position (i.e. test both `union(non_root, root)` and `union(root, non_root)`), since "
        "a bug that only resolves one of the two arguments before comparing ranks can pass a "
        "test that only exercises one position and still be wrong. Confirm the tie-break "
        "decision and `rank_of` afterward reflect the RESOLVED roots' ranks in every case, not "
        "the raw stored rank of whichever argument you passed in — before finishing."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    """Hidden task: no starter files, no visible test — the agent works from the spec alone."""
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "union_find", "classes": ["DisjointSet"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    """Single leaf: one class, no I/O, no multi-file structure — same rationale as c08."""
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Write union_find.py implementing class DisjointSet over integers "
                    "0..n-1: __init__(self, n) (n singleton sets, each its own root, rank "
                    "0), find(self, x) -> int (returns the root of x's set, and MUST apply "
                    "path compression -- every node visited on the walk from x up to the "
                    "root gets its parent pointer repointed directly to that root, not "
                    "just x), union(self, x, y) -> bool (merge by rank: let rx=find(x), "
                    "ry=find(y); if rx==ry return False (no-op, already same set); "
                    "otherwise if rank[rx]<rank[ry] attach rx under ry, if rank[rx]>"
                    "rank[ry] attach ry under rx, if ranks are EQUAL attach ry under rx "
                    "and increment rank[rx] by 1; return True -- CRITICAL: rank[rx]/"
                    "rank[ry] must be the rank of the RESOLVED root from find(), never the "
                    "raw stored rank of x or y themselves if they are not currently roots; "
                    "always call find() first and compare the roots' ranks), "
                    "connected(self, x, y) -> "
                    "bool (find(x)==find(y)), parent_of(self, x) -> int (RAW immediate "
                    "parent pointer exactly as stored, x itself if x is currently a root, "
                    "must NOT walk to the root or compress anything -- purely a read), and "
                    "rank_of(self, x) -> int (raw stored rank). Path compression is the "
                    "hard part and must be a REAL structural mutation visible via "
                    "parent_of, not just an internal optimization -- after building a "
                    "multi-level chain via several unions, calling find() once on the "
                    "deepest element must flatten every node on that path, confirmed by "
                    "reading parent_of() afterward. Use write_file to create "
                    "union_find.py, then use run_python to reproduce this worked example: "
                    "ds=DisjointSet(4); ds.union(0,1) then ds.union(2,3) then ds.union(0,2) "
                    "(each returns True); confirm ds.parent_of(3)==2 (raw, uncompressed) "
                    "BEFORE calling find; then call ds.find(3), confirm it returns 0, and "
                    "confirm ds.parent_of(3)==0 immediately after (compression happened); "
                    "also confirm ds.union(0,1) now returns False since they're already "
                    "connected. Build a longer 8+ element union chain too and re-check the "
                    "same before/after parent_of pattern on a deeper node. Then specifically "
                    "test union() where one argument is NOT currently a root (already "
                    "find()-compressed earlier, so its own raw rank is stale/low) against an "
                    "argument that already IS a root -- and try this with the non-root element "
                    "in BOTH argument positions (union(non_root, root) AND union(root, "
                    "non_root)), since a bug that only resolves ONE of the two arguments before "
                    "comparing ranks can look correct if you only ever test one position. "
                    "Confirm the tie-break decision uses the resolved roots' ranks in every "
                    "case, not the raw stored rank of whichever argument was passed in. Fix "
                    "issues with patch_file/run_python until confident, then finish."
                ),
                "expect": "union_find.py written, defining DisjointSet with find/union/"
                          "connected/parent_of/rank_of, union-by-rank and path compression "
                          "both sanity-checked with run_python",
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm union_find.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["union_find.py"]},
    }
