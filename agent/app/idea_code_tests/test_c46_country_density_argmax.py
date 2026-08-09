"""
codebench task c46 — hard/hidden, HYBRID web-search + code (sub-pattern 1: ported computation
shape). Reframes the QA suite's computed-ratio-argmax shape
(idea_tests/test_064_tier5_computed_ratio_argmax_wide.py — "argmax over a ratio neither page
prints, five-way fan-out, double-decoy against the biggest raw quantity") for the codebench
sandbox's search-snippet-only web tool (see c45's docstring for why the entity/fact choice had to
change from 064's page-infobox design: no page-visit tool here, only search_web SNIPPETS).
Population and land area both reliably surface in a plain search snippet for any country (unlike
064's lake volume/area, which needed an opened infobox), so the ratio-argmax SHAPE ports cleanly.

Among five countries the agent must compute POPULATION DENSITY (population / area, in people per
km^2) and determine which has the highest density. Verified live 2026-08-06 (WebSearch tool):

    country        population         area (km^2)      density (people/km^2)
    Rwanda         ~14.39 million        26,338              ~546.5   <- ARGMAX (keystone)
    Philippines    ~112.73 million      298,170              ~378.1   <- 2nd place
    Vietnam        ~99.50 million       310,070              ~320.9
    Nigeria        ~232.68 million      910,770              ~255.5   <- biggest population AND area
    Cambodia       ~17.12 million       176,520               ~97.0

    DENSITY ARGMAX = Rwanda (~546.5/km^2). Runner-up = Philippines (~378.1/km^2), a margin of
    roughly +45% -- wide enough that no plausible single mis-read source flips the winner (see the
    robustness check in the offline validator: even the LOWEST plausible Rwanda density against the
    HIGHEST plausible density of every other country still leaves Rwanda on top).

    DOUBLE-DECOY (matching 064's shape): Nigeria is the biggest country here by BOTH raw population
    AND raw area, and is NOT the density winner (Nigeria's own density is only ~255.5/km^2, roughly
    half of Rwanda's) -- "pick the biggest country" fails on both axes. Rwanda, the smallest country
    of the five by area, wins because it is small AND has a large population for its size.

Area figures for some of these countries are cited two ways in different sources (LAND area only vs
TOTAL area including inland water, up to roughly +6% apart for Vietnam/Philippines/Nigeria) -- the
per-country test bands below are wide enough to accept either convention; the argmax verdict is
unaffected either way (see the offline validator's robustness check).
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_country_density.py"

_TEST_FILE_CONTENT = '''\
from country_density import country_stats, density_argmax


def test_argmax_logic_on_synthetic_data():
    synthetic = {
        "X": {"population": 100, "area_km2": 10},   # density 10
        "Y": {"population": 50, "area_km2": 1},      # density 50 <- winner
        "Z": {"population": 1000, "area_km2": 1000}, # density 1
    }
    assert density_argmax(synthetic) == "Y"


def test_argmax_logic_biggest_raw_numbers_is_not_always_winner():
    synthetic = {
        "Big": {"population": 900, "area_km2": 900},   # density 1
        "Small": {"population": 90, "area_km2": 9},     # density 10 <- winner despite smaller
    }
    assert density_argmax(synthetic) == "Small"


def test_stats_rwanda():
    stats = country_stats()
    assert 13_000_000 <= stats["Rwanda"]["population"] <= 15_500_000
    assert 25_800 <= stats["Rwanda"]["area_km2"] <= 26_900


def test_stats_vietnam():
    stats = country_stats()
    assert 95_000_000 <= stats["Vietnam"]["population"] <= 103_000_000
    assert 300_000 <= stats["Vietnam"]["area_km2"] <= 335_000


def test_stats_philippines():
    stats = country_stats()
    assert 108_000_000 <= stats["Philippines"]["population"] <= 118_000_000
    assert 295_000 <= stats["Philippines"]["area_km2"] <= 302_000


def test_stats_nigeria():
    stats = country_stats()
    assert 220_000_000 <= stats["Nigeria"]["population"] <= 245_000_000
    assert 905_000 <= stats["Nigeria"]["area_km2"] <= 930_000


def test_stats_cambodia():
    stats = country_stats()
    assert 16_000_000 <= stats["Cambodia"]["population"] <= 18_000_000
    assert 175_000 <= stats["Cambodia"]["area_km2"] <= 182_000


def test_final_density_argmax_is_rwanda():
    assert density_argmax() == "Rwanda"
'''

# The two synthetic-data logic tests plus the final real-data argmax gate the score (mirrors
# 064's convention: the winning ENTITY is the keystone). The ten individual population/area reads
# are graduated, ungated credit.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_argmax_logic_on_synthetic_data",
    f"{_TEST_FILE_PATH}::test_argmax_logic_biggest_raw_numbers_is_not_always_winner",
    f"{_TEST_FILE_PATH}::test_final_density_argmax_is_rwanda",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c46",
        "title": "country-population-density-argmax",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You need the population AND land area of five countries, then must determine which one "
        "has the highest POPULATION DENSITY (population divided by area, in people per km^2). "
        "No single source states this ranking directly -- you must look up two numbers per "
        "country and compute the ratio yourself. Do not assume the country with the biggest "
        "population, or the biggest area, is the densest -- check.\n\n"
        "The five countries: Rwanda, Vietnam, Philippines, Nigeria, Cambodia.\n\n"
        "For each one, use search_web to find (a) its total population and (b) its total land "
        "area in km^2 (any recent reputable source is fine).\n\n"
        "Then write a Python module `country_density.py` defining exactly two functions:\n\n"
        "  `country_stats() -> dict` — returns a dict with exactly these five keys: \"Rwanda\", "
        "\"Vietnam\", \"Philippines\", \"Nigeria\", \"Cambodia\", each mapped to a dict with two "
        "keys: \"population\" (the raw headcount, e.g. 14000000, NOT millions) and \"area_km2\" "
        "(the land area in km^2).\n\n"
        "  `density_argmax(stats: dict = None) -> str` — returns the KEY (country name) of the "
        "entry in `stats` (default: the return value of `country_stats()`) with the highest "
        "population / area_km2 ratio. This must be a generic argmax helper that works for ANY "
        "dict shaped like `country_stats()`'s return value, not hardcoded to these five "
        "countries.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check both "
        "functions before finishing: call `density_argmax` on a small made-up stats dict where "
        "the entry with the smaller raw numbers has the higher ratio, and confirm it correctly "
        "picks that entry (not the one with the bigger raw population/area) — then call "
        "`country_stats()` and manually compute each of the five ratios to confirm which one is "
        "highest before you finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "country_density",
            "functions": ["country_stats", "density_argmax"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "You need the population and land area (km^2) of five countries: Rwanda, "
                    "Vietnam, Philippines, Nigeria, Cambodia. Use search_web once per country "
                    "(e.g. \"Rwanda population area km2\") to find both numbers for each -- do "
                    "not guess from memory. Then write country_density.py defining two "
                    "functions: (1) country_stats() -> dict, returning {\"Rwanda\": "
                    "{\"population\": <int>, \"area_km2\": <float>}, \"Vietnam\": {...}, "
                    "\"Philippines\": {...}, \"Nigeria\": {...}, \"Cambodia\": {...}} — "
                    "population as the raw headcount (e.g. 14000000, not 14 million written as "
                    "14.0); (2) density_argmax(stats: dict = None) -> str, which returns the KEY "
                    "of the entry in `stats` (defaulting to country_stats() when not given) with "
                    "the highest population/area_km2 ratio. This must be a generic argmax helper "
                    "that works on any dict shaped like country_stats()'s output, not hardcoded "
                    "to these five countries. Use write_file to create country_density.py, then "
                    "use run_python to check: (a) density_argmax on a small made-up dict where "
                    "the entry with SMALLER raw numbers has the higher ratio picks that entry, "
                    "not the one with bigger raw numbers; (b) country_stats() has all five keys "
                    "with plausible population/area values. Fix any issues with patch_file/"
                    "run_python, then finish."
                ),
                "expect": (
                    "country_density.py written, defining country_stats() -> dict (five "
                    "countries, population + area_km2) and density_argmax(dict) -> str (generic "
                    "highest-ratio-key helper)"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm country_density.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["country_density.py"]},
    }
