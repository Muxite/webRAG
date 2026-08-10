"""
codebench task c50 — hard/hidden, HYBRID web-search + code (sub-pattern 2: approximation with an
execution budget).

REVISED 2026-08-06 after live calibration on qwen2.5:14b (run_id=coordinator_batch2) showed the
original framing ("how many Earth days does Mercury take to orbit the Sun") was too easy: Mercury's
sidereal orbital period ("about 88 days") is one of the most commonly memorized planetary trivia
figures there is, so a model can ace all five tolerance bands purely from training-data
memorization, with ZERO web search and ZERO computation. The aider transcript
(codebench/results/runs/coordinator_batch2/c50__aider__qwen2.5_14b/submission/
.aider.chat.history.md) shows qwen2.5:14b writing the Kepler's-law computation in code AND stating
the correct semi-major axis, then discarding all of that to `return direct_value` (a hardcoded
"87.97" recited from memory) — scoring 5/5 (100%) without ever calling search_web. Since our own
ground truth WAS that same commonly-cited rounded figure, no amount of tolerance-band tightening
could have caught this (relative error was ~0%): the fix has to be a different target quantity.

New target: Mercury's SYNODIC period, NOT its sidereal/orbital period. The sidereal period (~88
days, the famous figure) is how long Mercury takes to go once around the Sun. The synodic period
(~116 days) is how long it takes for Mercury to return to the SAME apparent configuration relative
to Earth and the Sun (e.g. successive greatest elongations, or successive inferior conjunctions) --
a different, less commonly memorized quantity, because Earth is also moving. It genuinely requires
COMBINING two orbital periods (Mercury's own, which must still be looked up/derived, and Earth's,
given directly below) via the standard synodic-period formula -- so neither a lazily recalled "88
days" nor a search that stops at the first "Mercury ... orbital period ... days" result (which
returns the sidereal figure, the WRONG quantity for this question) suffices; the agent has to
recognize which period is actually being asked for and combine facts correctly.

Ground truth (verified live 2026-08-06 against NASA's Mercury Fact Sheet, nssdc.gsfc.nasa.gov, and
Wikipedia's Mercury (planet) infobox, which lists the synodic period directly):
    Mercury's sidereal orbital period T_mercury = 87.9691 days (the "famous" ~88-day figure --
        derivable via Kepler's third law from semi-major axis a = 0.38710 AU: T_years = a^1.5,
        T_days = T_years * 365.25)
    Earth's sidereal orbital period    T_earth    = 365.25 days (given directly to the agent, same
        constant already used for the AU-years-to-days conversion)

    Synodic period formula (standard astronomical formula for an inferior planet, i.e. one closer
    to the Sun than Earth): 1 / S = 1 / T_mercury - 1 / T_earth

    S = 1 / (1/87.9691 - 1/365.25) = 115.878 days

    This matches Wikipedia's own directly-published synodic period for Mercury, 115.88 days.

    A model that instead recites the famous SIDEREAL/orbital period (87.97 days, answering a
    different question than the one asked) is off by |87.97 - 115.878| / 115.878 = 24.1% -- it
    clears the loose 50% and 25% bands but fails the tighter 10%/5%/2% bands.

    TARGET = 115.88 days. Tolerance bands (relative error): 50%, 25%, 10%, 5%, 2%.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_mercury_orbit.py"
_TRUE_DAYS = 115.88

_TEST_FILE_CONTENT = '''\
from mercury_orbit import mercury_orbital_period_days

_TRUE = 115.88


def _rel_error(v):
    return abs(v - _TRUE) / _TRUE


def test_within_50_percent():
    assert _rel_error(mercury_orbital_period_days()) <= 0.50


def test_within_25_percent():
    assert _rel_error(mercury_orbital_period_days()) <= 0.25


def test_within_10_percent():
    assert _rel_error(mercury_orbital_period_days()) <= 0.10


def test_within_5_percent():
    assert _rel_error(mercury_orbital_period_days()) <= 0.05


def test_within_2_percent():
    assert _rel_error(mercury_orbital_period_days()) <= 0.02
'''

KEYSTONE_TEST_IDS = [f"{_TEST_FILE_PATH}::test_within_50_percent"]


def get_test_metadata() -> dict:
    return {
        "test_id": "c50",
        "title": "mercury-synodic-period-approximation",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Get the CLOSEST APPROXIMATION you can of Mercury's SYNODIC period -- how many EARTH DAYS "
        "pass between two successive times Mercury returns to the SAME apparent position relative "
        "to Earth and the Sun (e.g. two successive greatest elongations, or two successive "
        "inferior conjunctions, as seen from Earth).\n\n"
        "IMPORTANT: this is a DIFFERENT quantity from Mercury's ORBITAL (sidereal) period -- the "
        "well-known \"Mercury takes about 88 days to go around the Sun\" figure. That 88-day figure "
        "is how long Mercury takes to complete one full orbit; it is NOT the answer to this "
        "question, because Earth is also moving around the Sun during that time, so it takes "
        "LONGER than one Mercury orbit for the two planets to return to the same relative "
        "configuration. Do not just look up or recall Mercury's ~88-day orbital period and use "
        "that -- it will score poorly here.\n\n"
        "You have two valid ways to get the right number -- pick whichever you trust more, or do "
        "both and cross-check:\n\n"
        "  1. Search directly for Mercury's SYNODIC period (not its orbital/sidereal period -- "
        "make sure whatever source you use is answering the synodic-period question).\n\n"
        "  2. Find Mercury's own ORBITAL (sidereal) period in days -- either search for it "
        "directly, or search for Mercury's SEMI-MAJOR AXIS in AU and apply KEPLER'S THIRD LAW: "
        "T_years = a_AU ** 1.5, then multiply by 365.25 to convert years to days. Then COMBINE "
        "that with Earth's own orbital period (365.25 days -- the same constant, given to you) "
        "using the SYNODIC PERIOD FORMULA for a planet closer to the Sun than Earth: "
        "1 / S = 1 / T_mercury - 1 / T_earth, where T_mercury is what you just found and "
        "T_earth = 365.25. Solve for S (in days) -- that is your answer.\n\n"
        "Precision matters here -- you will be scored on how close your final number is to the "
        "true value, with more credit for tighter accuracy, so do not round aggressively or "
        "guess.\n\n"
        "Then write a Python module `mercury_orbit.py` defining exactly one function:\n\n"
        "  `mercury_orbital_period_days() -> float` — returns your best estimate of Mercury's "
        "SYNODIC period in Earth days (despite the module/function name, this is the SYNODIC "
        "period, not the sidereal/orbital period -- see above).\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "function before finishing: call it and print the result -- it should be noticeably "
        "MORE than Mercury's ~88-day orbital period (since the synodic period is always longer "
        "than the shorter planet's own orbital period), but still well under one Earth year (365 "
        "days); if your answer come out to roughly 88, you have probably answered the sidereal-"
        "period question by mistake -- go back and apply the synodic formula."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {"module": "mercury_orbit", "functions": ["mercury_orbital_period_days"]},
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Get the closest approximation you can of Mercury's SYNODIC period, in Earth "
                    "days -- the time between two successive returns to the same apparent "
                    "configuration relative to Earth and the Sun. This is NOT the same as "
                    "Mercury's well-known ~88-day orbital/sidereal period (do not just recite that "
                    "number). Two valid strategies: (1) search_web directly for Mercury's synodic "
                    "period specifically (not orbital/sidereal period); or (2) search_web for "
                    "Mercury's own orbital (sidereal) period in days (directly, or via semi-major "
                    "axis + Kepler's third law: T_years = a_AU ** 1.5, times 365.25), then combine "
                    "it with Earth's orbital period (365.25 days, given) via the synodic-period "
                    "formula: 1 / S = 1 / T_mercury - 1 / T_earth, and solve for S. You will be "
                    "scored on closeness to the true synodic-period value with graduated credit "
                    "for tighter accuracy, so do not round aggressively. Write mercury_orbit.py "
                    "defining one function: mercury_orbital_period_days() -> float, returning your "
                    "best estimate of the SYNODIC period (not sidereal period) in days. Use "
                    "write_file to create it, then use run_python to call the function and print "
                    "the result -- it should be noticeably more than 88 days (if it comes out "
                    "close to 88, you've answered the sidereal-period question by mistake) but "
                    "well under 365 days. Fix any issues with patch_file/run_python, then finish."
                ),
                "expect": (
                    "mercury_orbit.py written, defining mercury_orbital_period_days() -> float "
                    "returning a close approximation of Mercury's real SYNODIC period, distinct "
                    "from and longer than its sidereal/orbital period"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm mercury_orbit.py exists and report the sanity-check result.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["mercury_orbit.py"]},
    }
