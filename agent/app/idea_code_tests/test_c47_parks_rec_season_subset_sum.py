"""
codebench task c47 — hard/hidden, HYBRID web-search + code (sub-pattern 1: ported computation
shape). Reframes the QA suite's bounded-subset-sum-with-aggregate-distractor shape
(idea_tests/test_070_tier5_subset_sum_distractor.py — sum a per-item integer across a DEFINED
SUBSET while a bigger, more salient printed aggregate sits right next to it as a decoy) for the
codebench sandbox. Uses a DIFFERENT show than 070's Chuck (freshly chosen and freshly verified, per
the assignment's instruction not to blindly port an existing task's entities) so the entities here
are independently sourced: *Parks and Recreation* (NBC, 2009-2015).

The agent must sum the episode counts of the FIRST FOUR seasons, while the show's own Wikipedia
infobox prints a WHOLE-SERIES total (all seven seasons) as a single prominent number nearby -- that
total is the decoy, since it answers a different (bigger) question than the one asked.

Ground truth (verified live 2026-08-06 via WebFetch against each season's own Wikipedia infobox
'No. of episodes' field, and the main show article's infobox for the whole-series total):
    Season 1:  6 episodes   en.wikipedia.org/wiki/Parks_and_Recreation_(season_1)
    Season 2: 24 episodes   en.wikipedia.org/wiki/Parks_and_Recreation_season_2
    Season 3: 16 episodes   en.wikipedia.org/wiki/Parks_and_Recreation_season_3
    Season 4: 22 episodes   en.wikipedia.org/wiki/Parks_and_Recreation_season_4
    (not required) Season 5: 22, Season 6: 22, Season 7: 13

    KEYSTONE = SUM of seasons 1-4 = 6 + 24 + 16 + 22 = 68
    DISTRACTOR = main article infobox 'No. of episodes' = 126 (all seven seasons; the per-season
      figures above sum to 125, one short of the infobox's 126 -- a real, observed discrepancy
      between Wikipedia's own per-season and whole-series pages, most likely a clip/special
      episode counted in one place and not the other; irrelevant to this task, since the keystone
      only depends on the four VERIFIED per-season infobox figures above, not on reconciling them
      against the whole-series number).

    Tolerance band: keystone accepted in [67, 69] (68 +/- 1, one-slip tolerance, same convention
    as 070's Chuck task). Every season count here is >= 6, and three of the four are >= 16, so
    dropping any single season from the sum moves the total by at least 6 -- comfortably outside
    the +/-1 band (verified in the offline validator's drop-one collision check). The distractor
    (126) misses the band by 57, an enormous margin.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_parks_rec.py"

_TEST_FILE_CONTENT = '''\
from parks_rec import season_episode_counts, first_four_seasons_total


def test_subset_sum_logic_on_synthetic_data():
    assert first_four_seasons_total({1: 5, 2: 5, 3: 5, 4: 5}) == 20


def test_subset_sum_logic_ignores_extra_keys():
    # A dict with extra season keys (5, 6, ...) must still only sum seasons 1-4.
    assert first_four_seasons_total({1: 5, 2: 5, 3: 5, 4: 5, 5: 999, 6: 999}) == 20


def test_season_1_episode_count():
    counts = season_episode_counts()
    assert 5 <= counts[1] <= 7


def test_season_2_episode_count():
    counts = season_episode_counts()
    assert 23 <= counts[2] <= 25


def test_season_3_episode_count():
    counts = season_episode_counts()
    assert 15 <= counts[3] <= 17


def test_season_4_episode_count():
    counts = season_episode_counts()
    assert 21 <= counts[4] <= 23


def test_final_first_four_seasons_total():
    total = first_four_seasons_total()
    assert 67 <= total <= 69


def test_final_total_is_not_the_whole_series_distractor():
    total = first_four_seasons_total()
    assert abs(total - 126) > 30
'''

# The two synthetic-data logic tests plus the final tolerance-banded total gate the score
# (mirrors 070's convention: the computed SUM is the keystone). The four per-season reads are
# graduated, ungated credit.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_subset_sum_logic_on_synthetic_data",
    f"{_TEST_FILE_PATH}::test_subset_sum_logic_ignores_extra_keys",
    f"{_TEST_FILE_PATH}::test_final_first_four_seasons_total",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c47",
        "title": "parks-and-recreation-season-subset-sum",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "The NBC sitcom *Parks and Recreation* ran for seven seasons. You need the SUM of the "
        "episode counts of ONLY its FIRST FOUR seasons (seasons 1 through 4) -- not the whole "
        "series.\n\n"
        "Careful: if you search for the show's main article, you will likely see a prominent "
        "'No. of episodes' figure in its infobox -- that figure counts ALL SEVEN seasons, which "
        "is a DIFFERENT (bigger) number than what is being asked here. To get the right answer "
        "you need each of the first four seasons' OWN episode count (each season generally has "
        "its own Wikipedia article/infobox, or is listed season-by-season in an episode list "
        "article) and then add exactly those four numbers together yourself -- do not use the "
        "whole-series total.\n\n"
        "Use search_web to find each of the first four seasons' individual episode count.\n\n"
        "Then write a Python module `parks_rec.py` defining exactly two functions:\n\n"
        "  `season_episode_counts() -> dict` — returns a dict with exactly four keys, the "
        "integers 1, 2, 3, 4 (one per season), each mapped to that season's episode count as an "
        "int.\n\n"
        "  `first_four_seasons_total(counts: dict = None) -> int` — returns the SUM of the "
        "values at keys 1, 2, 3, 4 in `counts` (default: the return value of "
        "`season_episode_counts()`). If `counts` has extra keys beyond 1-4, they must be "
        "IGNORED — this must be a generic \"sum keys 1 through 4\" helper, not hardcoded to "
        "these particular numbers.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check both "
        "functions before finishing: call `first_four_seasons_total` on a small made-up dict "
        "(including one with extra keys beyond 4) to confirm it only sums keys 1-4, then call "
        "`season_episode_counts()` and add up the four values by hand to confirm the total makes "
        "sense and is clearly NOT the same as the whole-series total you may have seen on the "
        "main article, before you finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "parks_rec",
            "functions": ["season_episode_counts", "first_four_seasons_total"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "You need the episode count of EACH of the first four seasons (seasons 1, "
                    "2, 3, 4) of the NBC sitcom Parks and Recreation, individually -- NOT the "
                    "whole-series total (the show ran seven seasons; the main article's infobox "
                    "'No. of episodes' figure counts all seven, which is a different, bigger "
                    "number than what you need). Use search_web (e.g. \"Parks and Recreation "
                    "season 1 episode count\", repeated for seasons 2, 3, 4) to find each "
                    "season's own episode count -- do not use the whole-series total. Then write "
                    "parks_rec.py defining two functions: (1) season_episode_counts() -> dict, "
                    "returning {1: <int>, 2: <int>, 3: <int>, 4: <int>} for seasons 1-4; (2) "
                    "first_four_seasons_total(counts: dict = None) -> int, which sums ONLY the "
                    "values at keys 1, 2, 3, 4 of `counts` (defaulting to "
                    "season_episode_counts() when not given), ignoring any other keys present -- "
                    "this must be a generic helper, not hardcoded to particular numbers. Use "
                    "write_file to create parks_rec.py, then use run_python to check: (a) "
                    "first_four_seasons_total({1: 1, 2: 1, 3: 1, 4: 1, 5: 999}) returns 4 (extra "
                    "keys ignored); (b) season_episode_counts() has plausible per-season episode "
                    "counts and their sum is clearly different from any whole-series total "
                    "figure you may have seen. Fix any issues with patch_file/run_python, then "
                    "finish."
                ),
                "expect": (
                    "parks_rec.py written, defining season_episode_counts() -> {1: int, 2: int, "
                    "3: int, 4: int} and first_four_seasons_total(dict) -> int (generic sum of "
                    "keys 1-4)"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm parks_rec.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["parks_rec.py"]},
    }
