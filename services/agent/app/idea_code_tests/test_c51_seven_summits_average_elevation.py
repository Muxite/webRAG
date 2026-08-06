"""
codebench task c51 — hard/hidden, HYBRID web-search + code (sub-pattern 3: fresh hybrid task,
derived statistic = AVERAGE). Look up seven real data points via search, compute their mean via
code -- a weak model should not trust itself to add seven 4-5 digit numbers and divide by 7 in its
head; the natural move is a one-line `sum(...) / len(...)`.

To sidestep the well-known "Seven Summits" list AMBIGUITY (two competing canonical lists exist --
the Bass list uses Mount Kosciuszko for Oceania, the Messner list uses Puncak Jaya/Carstensz
Pyramid instead -- differing by roughly 2,650 metres on that one entry), the task statement names
all seven peaks EXPLICITLY by name rather than saying "the Seven Summits", so there is no ambiguity
about which list is meant.

Ground truth (verified live 2026-08-06, WebSearch, cross-checked against each peak's commonly
cited elevation):
    Mount Everest    (Asia)          8,849 m
    Aconcagua        (South America) 6,961 m
    Denali           (North America) 6,190 m
    Kilimanjaro      (Africa)        5,895 m
    Mount Elbrus     (Europe)        5,642 m
    Vinson Massif    (Antarctica)    4,892 m
    Puncak Jaya       (Oceania)       4,884 m

    sum = 8849 + 6961 + 6190 + 5895 + 5642 + 4892 + 4884 = 43,313
    KEYSTONE = average = 43313 / 7 = 6,187.57 m

These are all extremely well-documented, physically fixed elevations (routinely cross-checked
survey figures, unlike a socially/economically drifting quantity), so the per-peak test bands can
be reasonably tight (+/-3%) while the final average allows a slightly wider +/-4% band to absorb
the compounding of up to seven small independent read errors.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_seven_summits.py"

_TEST_FILE_CONTENT = '''\
from seven_summits import seven_summits_elevations_m, average_elevation_m


def test_average_logic_on_synthetic_data():
    assert average_elevation_m({"A": 100, "B": 200, "C": 300}) == 200.0


def test_average_logic_is_not_hardcoded_to_seven_entries():
    assert average_elevation_m({"A": 10, "B": 20}) == 15.0


def test_elevation_everest():
    elevations = seven_summits_elevations_m()
    assert 8700 <= elevations["Mount Everest"] <= 9000


def test_elevation_aconcagua():
    elevations = seven_summits_elevations_m()
    assert 6800 <= elevations["Aconcagua"] <= 7100


def test_elevation_denali():
    elevations = seven_summits_elevations_m()
    assert 6050 <= elevations["Denali"] <= 6350


def test_elevation_kilimanjaro():
    elevations = seven_summits_elevations_m()
    assert 5750 <= elevations["Kilimanjaro"] <= 6050


def test_elevation_elbrus():
    elevations = seven_summits_elevations_m()
    assert 5500 <= elevations["Mount Elbrus"] <= 5800


def test_elevation_vinson_massif():
    elevations = seven_summits_elevations_m()
    assert 4700 <= elevations["Vinson Massif"] <= 5150


def test_elevation_puncak_jaya():
    elevations = seven_summits_elevations_m()
    assert 4700 <= elevations["Puncak Jaya"] <= 5150


def test_final_average_elevation():
    avg = average_elevation_m()
    assert 5940 <= avg <= 6435
'''

# The two synthetic-data logic tests plus the final average gate the score. The seven individual
# elevation reads are graduated, ungated credit.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_average_logic_on_synthetic_data",
    f"{_TEST_FILE_PATH}::test_average_logic_is_not_hardcoded_to_seven_entries",
    f"{_TEST_FILE_PATH}::test_final_average_elevation",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c51",
        "title": "seven-summits-average-elevation",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You need the elevation (in metres) of seven specific mountains -- the highest peak on "
        "each continent -- and then their AVERAGE elevation.\n\n"
        "The seven peaks: Mount Everest, Aconcagua, Denali, Kilimanjaro, Mount Elbrus, Vinson "
        "Massif, Puncak Jaya.\n\n"
        "For each one, use search_web to find its elevation in metres.\n\n"
        "Then write a Python module `seven_summits.py` defining exactly two functions:\n\n"
        "  `seven_summits_elevations_m() -> dict` — returns a dict with exactly these seven "
        "keys: \"Mount Everest\", \"Aconcagua\", \"Denali\", \"Kilimanjaro\", \"Mount Elbrus\", "
        "\"Vinson Massif\", \"Puncak Jaya\", each mapped to that peak's elevation in metres (a "
        "number).\n\n"
        "  `average_elevation_m(elevations: dict = None) -> float` — returns the AVERAGE (mean) "
        "of the values in `elevations` (default: the return value of "
        "`seven_summits_elevations_m()`). This must be a generic averaging helper that works "
        "correctly for a dict of ANY size, not hardcoded to divide by 7.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check both "
        "functions before finishing: call `average_elevation_m` on a small made-up 2- or "
        "3-entry dict to confirm it computes a real mean (sum divided by count, not a hardcoded "
        "divisor), then call `seven_summits_elevations_m()` and confirm all seven values look "
        "like plausible mountain elevations in metres before you finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "seven_summits",
            "functions": ["seven_summits_elevations_m", "average_elevation_m"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "You need the elevation in metres of seven specific mountains: Mount "
                    "Everest, Aconcagua, Denali, Kilimanjaro, Mount Elbrus, Vinson Massif, "
                    "Puncak Jaya (the highest peak on each continent). Use search_web once per "
                    "peak (e.g. \"Aconcagua elevation metres\") to find each one's elevation -- "
                    "do not guess from memory. Then write seven_summits.py defining two "
                    "functions: (1) seven_summits_elevations_m() -> dict, returning {\"Mount "
                    "Everest\": <number>, \"Aconcagua\": <number>, \"Denali\": <number>, "
                    "\"Kilimanjaro\": <number>, \"Mount Elbrus\": <number>, \"Vinson Massif\": "
                    "<number>, \"Puncak Jaya\": <number>}, each value that peak's elevation in "
                    "metres; (2) average_elevation_m(elevations: dict = None) -> float, "
                    "returning the mean of the values in `elevations` (defaulting to "
                    "seven_summits_elevations_m() when not given) -- this must be a generic "
                    "sum-divided-by-count helper that works for a dict of any size, not "
                    "hardcoded to divide by 7. Use write_file to create seven_summits.py, then "
                    "use run_python to check: (a) average_elevation_m on a small made-up 2- or "
                    "3-entry dict returns the correct mean; (b) seven_summits_elevations_m() has "
                    "all seven keys with plausible elevation values. Fix any issues with "
                    "patch_file/run_python, then finish."
                ),
                "expect": (
                    "seven_summits.py written, defining seven_summits_elevations_m() -> dict "
                    "(seven named peaks, elevation in metres) and average_elevation_m(dict) -> "
                    "float (generic mean helper)"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm seven_summits.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["seven_summits.py"]},
    }
