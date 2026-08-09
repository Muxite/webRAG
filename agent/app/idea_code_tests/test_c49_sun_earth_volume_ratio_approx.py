"""
codebench task c49 — hard/hidden, HYBRID web-search + code (sub-pattern 2: approximation with an
execution budget). Sibling of c48: "get the closest approximation you can" of how many Earths
would fit inside the Sun BY VOLUME, deliberately open to either --
  (a) search directly (a commonly published "fun fact" figure), OR
  (b) search for the Sun's and Earth's RADII (two of the most reliably published numbers in all of
      astronomy) and COMPUTE the ratio yourself via (R_sun / R_earth)^3, a formula given directly
      in the task statement.

The task statement explicitly disambiguates against a DIFFERENT, lower, also-commonly-cited figure
(sphere-PACKING-adjusted, ~930,000-960,000) that live search turned up alongside the pure-volume
figure during verification -- the task asks for the pure volume ratio (Sun's volume / Earth's
volume), not a packing-efficiency-adjusted count of whole solid spheres.

Ground truth (verified live 2026-08-06 against NASA fact sheets and NASA's own "Sun: Facts" page):
    R_sun   = 696,000 km   (volumetric mean radius)
    R_earth = 6,371 km     (volumetric mean radius)
    ratio = (R_sun / R_earth)^3 = (696000/6371)^3 = 109.245^3 = ~1,303,800

    This matches NASA's own commonly-published "1.3 million Earths fit inside the Sun" figure
    (by pure volume) -- both derivation strategies converge on the same value.

    TARGET = 1,300,000. Tolerance bands (relative error): 50%, 25%, 10%, 5%, 2%.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_sun_earth_ratio.py"
_TRUE_RATIO = 1_300_000.0

_TEST_FILE_CONTENT = '''\
from sun_earth_ratio import earths_in_sun_by_volume

_TRUE = 1_300_000.0


def _rel_error(v):
    return abs(v - _TRUE) / _TRUE


def test_within_50_percent():
    assert _rel_error(earths_in_sun_by_volume()) <= 0.50


def test_within_25_percent():
    assert _rel_error(earths_in_sun_by_volume()) <= 0.25


def test_within_10_percent():
    assert _rel_error(earths_in_sun_by_volume()) <= 0.10


def test_within_5_percent():
    assert _rel_error(earths_in_sun_by_volume()) <= 0.05


def test_within_2_percent():
    assert _rel_error(earths_in_sun_by_volume()) <= 0.02
'''

KEYSTONE_TEST_IDS = [f"{_TEST_FILE_PATH}::test_within_50_percent"]


def get_test_metadata() -> dict:
    return {
        "test_id": "c49",
        "title": "sun-earth-volume-ratio-approximation",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Get the CLOSEST APPROXIMATION you can of how many Earths would fit inside the Sun, BY "
        "VOLUME (i.e. the Sun's volume divided by the Earth's volume -- treat this as a pure "
        "geometric ratio of two spheres, assuming Earth-sized volumes pack with NO wasted space; "
        "do NOT apply any real-world sphere-packing-efficiency correction, since you may see a "
        "different, lower figure online for that variant of the question).\n\n"
        "You have two valid ways to get there -- pick whichever you trust more, or do both and "
        "cross-check:\n\n"
        "  1. Search directly for how many Earths fit inside the Sun by volume.\n\n"
        "  2. Search for the Sun's RADIUS and the Earth's RADIUS (in the same units, e.g. both "
        "in km), then COMPUTE the ratio yourself using: ratio = (R_sun / R_earth) ** 3 (since "
        "the volume of a sphere scales with the cube of its radius, the ratio of two spheres' "
        "volumes equals the cube of the ratio of their radii).\n\n"
        "Precision matters here -- you will be scored on how close your final number is to the "
        "true value, with more credit for tighter accuracy, so do not round aggressively or "
        "guess.\n\n"
        "Then write a Python module `sun_earth_ratio.py` defining exactly one function:\n\n"
        "  `earths_in_sun_by_volume() -> float` — returns your best estimate of how many "
        "Earth-volumes fit inside the Sun's volume.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "function before finishing: call it and print the result -- it should be in the "
        "millions, not the thousands or the billions; if you used the formula approach, double "
        "check you used the SAME units for both radii before cubing the ratio, before you finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "sun_earth_ratio", "functions": ["earths_in_sun_by_volume"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Get the closest approximation you can of how many Earths would fit inside "
                    "the Sun by VOLUME (pure geometric ratio, no sphere-packing-efficiency "
                    "correction -- a different, lower figure exists online for the packing "
                    "variant of this question; that is NOT what is being asked). Two valid "
                    "strategies: (1) search_web directly for how many Earths fit inside the Sun "
                    "by volume; or (2) search_web for the Sun's radius and the Earth's radius "
                    "(same units), then compute ratio = (R_sun / R_earth) ** 3. You will be "
                    "scored on closeness to the true value with graduated credit for tighter "
                    "accuracy, so do not round aggressively. Write sun_earth_ratio.py defining "
                    "one function: earths_in_sun_by_volume() -> float, returning your best "
                    "estimate. Use write_file to create it, then use run_python to call the "
                    "function and print the result -- it should be in the millions; if you used "
                    "the formula, double-check both radii used the SAME units before cubing the "
                    "ratio. Fix any issues with patch_file/run_python, then finish."
                ),
                "expect": (
                    "sun_earth_ratio.py written, defining earths_in_sun_by_volume() -> float "
                    "returning a close approximation of the Sun-to-Earth volume ratio"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm sun_earth_ratio.py exists and report the sanity-check result.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["sun_earth_ratio.py"]},
    }
