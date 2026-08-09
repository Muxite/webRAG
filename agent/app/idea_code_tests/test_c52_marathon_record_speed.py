"""
codebench task c52 — hard/hidden, HYBRID web-search + code (sub-pattern 3: fresh hybrid task,
derived statistic = RATE). Look up one real historical fact (a specific, named, closed race result
-- not a "current record" that could later be superseded, so the ground truth here never drifts)
via search, then compute a RATE (average speed in km/h) via code -- converting "2 hours, 0 minutes,
35 seconds" into a speed is exactly the kind of arithmetic a weak model botches in its head (mixed
-radix time, then a division) but a two-line script gets right every time.

Ground truth (verified live 2026-08-06 via WebSearch, multiple independent sources: Olympics.com,
LetsRun.com, Bank of America Newsroom, Sportico):
    Kelvin Kiptum ran the men's marathon world record at the 2023 Chicago Marathon (8 October
    2023) in 2:00:35 (2 hours, 0 minutes, 35 seconds) = 2*3600 + 0*60 + 35 = 7,235 seconds.
    Standard marathon distance = 42.195 km (a fixed, defined sports-rule constant, given directly
    in the task statement -- not something to search for).

    KEYSTONE = average speed = 42.195 km / (7235 s / 3600 s-per-hour) = 42.195 / 2.009722 hours
             = 20.995 km/h  (~21.0 km/h)

This is a named, closed, historical event (Kiptum's specific run on that specific date), not
"the current marathon world record" -- so even though a faster record could in principle be set
after this task was authored, the ground truth for THIS question never changes.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_marathon_speed.py"
_TRUE_TIME_SECONDS = 7235  # 2:00:35
_TRUE_SPEED_KMH = 20.995

_TEST_FILE_CONTENT = '''\
from marathon_speed import speed_kmh, kiptum_record_time_seconds, kiptum_record_speed_kmh


def test_speed_logic_one_hour_flat():
    assert abs(speed_kmh(distance_km=10, time_seconds=3600) - 10.0) < 1e-9


def test_speed_logic_two_hours_flat():
    # 42.195 km in exactly 2 hours (7200s) -> 21.0975 km/h; independent of the real record time.
    assert abs(speed_kmh(distance_km=42.195, time_seconds=7200) - 21.0975) < 0.01


def test_record_time_seconds():
    # 2:00:35 = 7235 seconds; generous +/-30s band for a mm:ss-to-seconds conversion slip.
    t = kiptum_record_time_seconds()
    assert 7205 <= t <= 7265


def test_final_speed_within_10_percent():
    speed = kiptum_record_speed_kmh()
    assert abs(speed - 20.995) / 20.995 <= 0.10


def test_final_speed_within_3_percent():
    speed = kiptum_record_speed_kmh()
    assert abs(speed - 20.995) / 20.995 <= 0.03


def test_final_speed_within_1_percent():
    speed = kiptum_record_speed_kmh()
    assert abs(speed - 20.995) / 20.995 <= 0.01
'''

# The two logic tests (pure arithmetic, no real-world fact needed) plus the 10%-band final-speed
# test gate the score -- reasonably achievable, not the loosest-of-many-bands convention used for
# the pure approximation tasks (c48-c50), since the underlying fact here is a precise historical
# constant, not an open-ended "closest approximation" target.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_speed_logic_one_hour_flat",
    f"{_TEST_FILE_PATH}::test_speed_logic_two_hours_flat",
    f"{_TEST_FILE_PATH}::test_final_speed_within_10_percent",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c52",
        "title": "marathon-record-average-speed",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Kelvin Kiptum set the men's marathon world record at the 2023 Chicago Marathon. You "
        "need his exact finishing TIME for that specific race, and then his AVERAGE SPEED in "
        "km/h over the marathon distance.\n\n"
        "Use search_web to find Kelvin Kiptum's exact finishing time (hours:minutes:seconds) at "
        "the 2023 Chicago Marathon.\n\n"
        "The standard marathon distance is 42.195 km (you do not need to search for this — it "
        "is a fixed, defined distance used for every standard marathon).\n\n"
        "Then write a Python module `marathon_speed.py` defining exactly three functions:\n\n"
        "  `speed_kmh(distance_km: float, time_seconds: float) -> float` — a GENERIC helper that "
        "returns average speed in km/h given a distance in km and a duration in seconds. This "
        "must work for ANY distance/time pair, not just the marathon record.\n\n"
        "  `kiptum_record_time_seconds() -> int` — returns Kiptum's exact finishing time at the "
        "2023 Chicago Marathon, converted to TOTAL SECONDS (e.g. a time of 1:02:03 would be "
        "1*3600 + 2*60 + 3 = 3723).\n\n"
        "  `kiptum_record_speed_kmh() -> float` — returns Kiptum's average speed in km/h for "
        "that race. This should be computed by calling `speed_kmh(42.195, "
        "kiptum_record_time_seconds())` — do not hardcode a separately-typed speed value.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check all three "
        "functions before finishing: call `speed_kmh` with a couple of easy round-number inputs "
        "(e.g. 10 km in exactly 1 hour should give 10.0 km/h) to confirm the conversion logic is "
        "right, then call `kiptum_record_speed_kmh()` and confirm the result is a plausible "
        "elite-marathon speed (world-class marathoners run at roughly 19-21 km/h, not 5 km/h or "
        "50 km/h) before you finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "marathon_speed",
            "functions": ["speed_kmh", "kiptum_record_time_seconds", "kiptum_record_speed_kmh"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "You need Kelvin Kiptum's exact finishing time at the 2023 Chicago Marathon "
                    "(where he set the men's marathon world record), then his average speed in "
                    "km/h. Use search_web (e.g. \"Kelvin Kiptum 2023 Chicago Marathon time\") to "
                    "find his exact time (hours:minutes:seconds) -- do not guess from memory. "
                    "The standard marathon distance is 42.195 km (a fixed constant, no need to "
                    "search for it). Write marathon_speed.py defining three functions: (1) "
                    "speed_kmh(distance_km: float, time_seconds: float) -> float, a GENERIC "
                    "helper computing average speed in km/h from any distance/time pair -- "
                    "distance_km / (time_seconds / 3600); (2) kiptum_record_time_seconds() -> "
                    "int, Kiptum's exact finishing time converted to total seconds (hours*3600 + "
                    "minutes*60 + seconds); (3) kiptum_record_speed_kmh() -> float, computed by "
                    "calling speed_kmh(42.195, kiptum_record_time_seconds()) -- do not hardcode "
                    "a separately-typed number here. Use write_file to create "
                    "marathon_speed.py, then use run_python to check: (a) speed_kmh(10, 3600) "
                    "returns 10.0; (b) kiptum_record_speed_kmh() returns a plausible elite-"
                    "marathon speed (roughly 19-21 km/h -- not 5 or 50). Fix any issues with "
                    "patch_file/run_python, then finish."
                ),
                "expect": (
                    "marathon_speed.py written, defining speed_kmh(distance_km, time_seconds) -> "
                    "float (generic), kiptum_record_time_seconds() -> int, and "
                    "kiptum_record_speed_kmh() -> float composing the two"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm marathon_speed.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["marathon_speed.py"]},
    }
