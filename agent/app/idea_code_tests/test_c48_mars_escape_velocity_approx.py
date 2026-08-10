"""
codebench task c48 — hard/hidden, HYBRID web-search + code (sub-pattern 2: approximation with an
execution budget). First of the three "get the closest approximation you can" tasks.

REVISED 2026-08-06 after live calibration on qwen2.5:14b (run_id=coordinator_batch2) showed the
original framing ("Mars's escape velocity") was too easy: escape velocity is one of the single most
commonly published planetary trivia figures (NASA's own Mars Fact Sheet states it directly as
"5.03 km/s"), so a model can ace all five tolerance bands purely from training-data memorization,
with ZERO web search and ZERO computation — which is exactly what happened. The aider transcript
(codebench/results/runs/coordinator_batch2/c48__aider__qwen2.5_14b/submission/
.aider.chat.history.md) shows qwen2.5:14b writing the correct v_esc = sqrt(2GM/R) formula in code,
correctly computing it internally, and then discarding that computation to `return published_esc_vel`
— i.e. reciting the memorized trivia figure instead of trusting its own derivation, and scoring 5/5
(100%) without ever calling search_web. Since our own ground truth WAS that same commonly-published
rounded figure, no amount of tolerance-band tightening could have caught this (relative error was
~0%): the fix has to be a different target quantity, not a tighter band. First revision: retarget
Mars's escape velocity to Mars's CIRCULAR ORBITAL velocity (v_orbit = sqrt(GM/R), no factor of 2),
same body, different formula.

REVISED AGAIN 2026-08-07: round-2 live calibration showed the sqrt(2)-formula retargeting above
was STILL not enough — Aider scored 1.0/1.0 again. Its round-3 submission
(codebench/results/runs/coordinator_batch3/c48__aider__qwen2.5_14b/submission/
mars_escape.py) never calls search_web at all (confirmed by grepping its own
.aider.chat.history.md — the only mentions of "search" in the whole transcript are the literal
instructions copy-pasted FROM the task prompt); it just recites Mars's mass and radius from
training-data memory (M=6.39e23 kg, R=3389.5e3 m — both close to, but not exactly, the standard
NASA/Wikipedia figures used below) and plugs them into the given v_orbit=sqrt(GM/R) formula,
landing at 3.5471 km/s — 0.22% relative error, comfortably inside even the tightest 2% band. The
lesson: retargeting the FORMULA (escape vs. orbital velocity) does nothing if the underlying FACTS
(a planet's mass and radius) are themselves so ubiquitously repeated in training data that a 14B
model can recite them to 4 significant figures without ever needing to look anything up — the
model doesn't need search_web because it doesn't need NEW information, only arithmetic it can
already do from memory. Also worth flagging in the OLD framing specifically: the task statement
used to hand over the exact relationship "circular orbital velocity is escape velocity divided by
sqrt(2)" — for MARS this was a live, exploitable shortcut in principle (Mars's escape velocity,
5.03 km/s, IS one of the most memorized planetary trivia figures there is; dividing it by sqrt(2)
gives 3.557 km/s, itself within 2% of the true value) even though Aider's actual submission didn't
happen to use that particular path this round.

The fix this time is a different LEVER than before: switch the target BODY away from Mars (whose
mass and radius are near-universally memorized to high precision) to the dwarf planet CERES —
still a real, well-documented body (largest object in the asteroid belt, visited by NASA's Dawn
mission), but one whose mass and radius are FAR less commonly repeated/memorized standalone trivia,
so reciting them from parametric memory to the precision needed to clear a 2% relative-error band
is much less reliable than it is for Mars. The formula and general framing are otherwise unchanged
(same v_orbit = sqrt(G*M/R), no factor of 2, same "don't confuse with escape velocity" caveat) —
Ceres's escape velocity (~0.516 km/s) isn't itself a commonly memorized figure either, so keeping
that explanatory relationship in the prompt is not a new exploitable shortcut the way it was for
Mars's famous escape-velocity trivia.

Ground truth (verified live 2026-08-07 against Ceres's Wikipedia infobox, cross-checked against
NASA Science's Ceres fact page and a general web search for "Ceres orbital velocity"):
    mass M   = 9.38392e20 kg        (Wikipedia infobox, precise, Dawn-mission-era figure)
    radius R = 469,700 m            (469.7 km, mean/volumetric radius, Wikipedia infobox)
    G (universal gravitational constant, as given to the agent) = 6.674e-11 m^3 kg^-1 s^-2

    v_orbit = sqrt(G*M/R) = sqrt(6.674e-11 * 9.38392e20 / 469700) = sqrt(1.263413e11 / 469700)
            = sqrt(268981.5) ... (m/s, before final sqrt) = 365.15 m/s = 0.36515 km/s

    Cross-check via Wikipedia's OWN published escape velocity (0.516 km/s) and the exact sqrt(2)
    relationship: 0.516 / sqrt(2) = 0.3649 km/s — within 0.07% of the direct M/R derivation above,
    confirming M and R are mutually consistent with Wikipedia's independently-stated escape
    velocity (itself cross-derived from the same M and R by Wikipedia's own editors/sources).
    Also cross-checked: computing surface gravity from the same M, R (g = GM/R^2 = 0.2839 m/s^2)
    matches Wikipedia's independently stated 0.284 m/s^2 to 4 significant figures.

    Sensitivity check across other reasonably-sourced M/R combinations (an older, pre-Dawn mass
    estimate of 9.1e20 kg paired with NASA's rounded 476 km radius, etc.) stayed within roughly
    2.3% of the value above — reasonable source variance is real at this scale but modern,
    easily-found sources (Wikipedia's infobox, the first hit for "Ceres mass" or "Ceres radius")
    converge tightly on the M and R used here.

    A model that instead recites the famous MARS orbital velocity from the previous version of
    this task (3.555 km/s) is off by |3.555 - 0.36515| / 0.36515 ≈ 8.7x — nowhere close to any
    band. A model that computes Ceres's ESCAPE velocity instead of orbital velocity (forgets to
    omit the factor of 2) gets 0.5164 km/s, off by 41.4% — clears only the loosest (50%) band,
    the same margin the old Mars-escape-vs-orbital mixup produced.

    TARGET = 0.3652 km/s. Tolerance bands (relative error): 50%, 25%, 10%, 5%, 2%.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_ceres_orbital_velocity.py"
_TRUE_KMS = 0.3652

_TEST_FILE_CONTENT = '''\
from ceres_orbital_velocity import ceres_orbital_velocity_kms

_TRUE = 0.3652


def _rel_error(v):
    return abs(v - _TRUE) / _TRUE


def test_within_50_percent():
    assert _rel_error(ceres_orbital_velocity_kms()) <= 0.50


def test_within_25_percent():
    assert _rel_error(ceres_orbital_velocity_kms()) <= 0.25


def test_within_10_percent():
    assert _rel_error(ceres_orbital_velocity_kms()) <= 0.10


def test_within_5_percent():
    assert _rel_error(ceres_orbital_velocity_kms()) <= 0.05


def test_within_2_percent():
    assert _rel_error(ceres_orbital_velocity_kms()) <= 0.02
'''

# Approximation task: only the LOOSEST band gates the score (a "plausible, on-topic, right order
# of magnitude" floor) -- real discrimination between a rough guess and a precise derivation lives
# entirely in the graduated (ungated) tighter bands, per the assignment's grading guidance.
KEYSTONE_TEST_IDS = [f"{_TEST_FILE_PATH}::test_within_50_percent"]


def get_test_metadata() -> dict:
    return {
        "test_id": "c48",
        "title": "ceres-orbital-velocity-approximation",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "Get the CLOSEST APPROXIMATION you can of the speed a spacecraft would need to maintain a "
        "stable, idealized CIRCULAR ORBIT right at the surface (mean radius) of the dwarf planet "
        "CERES -- in km/s. Ceres is the largest object in the asteroid belt and was visited by "
        "NASA's Dawn mission. This speed is sometimes called a body's "
        "\"first cosmic velocity\" or circular orbital speed.\n\n"
        "IMPORTANT: this is NOT the same thing as Ceres's ESCAPE velocity, and the two are "
        "commonly confused for any body -- escape velocity is the speed to leave a body's gravity "
        "permanently and never come back; circular orbital velocity (what this task asks for) is "
        "the slower speed to stay in a stable loop around it forever. Circular orbital velocity is "
        "escape velocity divided by sqrt(2) -- roughly 71% of it, not the same number. Do not just "
        "look up or recall Ceres's escape velocity and use that directly -- it is a different, "
        "larger quantity and will score poorly here.\n\n"
        "You have two valid ways to get the right number -- pick whichever you trust more, or do "
        "both and cross-check:\n\n"
        "  1. Search directly for Ceres's circular orbital velocity (also called its \"first "
        "cosmic velocity\").\n\n"
        "  2. Search for Ceres's MASS (in kg) and RADIUS (in km or m), then COMPUTE the circular "
        "orbital velocity yourself using: v_orbit = sqrt(G * M / R) -- note there is NO factor of "
        "2 here (that formula is for escape velocity, which is a different quantity) -- where "
        "G = 6.674e-11 (SI units: m^3 / (kg * s^2)), M is Ceres's mass in kg, and R is Ceres's "
        "radius in METERS (convert if you found it in km). The result is in m/s; divide by 1000 "
        "to get km/s.\n\n"
        "Ceres is a much less famous body than a major planet, so do not rely on a vague memory "
        "of its mass or radius -- these are not commonly memorized figures, and even a mildly "
        "inaccurate recollection can put your final answer outside every useful tolerance band. "
        "Look the numbers up rather than guessing from memory.\n\n"
        "Precision matters here -- you will be scored on how close your final number is to the "
        "true value, with more credit for tighter accuracy, so do not round aggressively or "
        "guess.\n\n"
        "Then write a Python module `ceres_orbital_velocity.py` defining exactly one function:\n\n"
        "  `ceres_orbital_velocity_kms() -> float` — returns your best estimate of Ceres's "
        "circular orbital (surface) velocity in km/s (NOT escape velocity -- see above).\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check your "
        "function before finishing: call it and print the result, and if you used the formula "
        "approach, double check (a) you used v_orbit = sqrt(G*M/R) with NO factor of 2, and (b) "
        "your unit conversion (km to m for the radius; m/s to km/s for the final answer) before "
        "you finish -- a units mistake here is the single most common way to be off by a factor of "
        "1000, and using the escape-velocity formula by mistake is the single most common way to "
        "be off by a factor of sqrt(2)."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "ceres_orbital_velocity",
            "functions": ["ceres_orbital_velocity_kms"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "Get the closest approximation you can of the speed needed to maintain a "
                    "stable circular orbit right at the surface (mean radius) of the dwarf "
                    "planet CERES (the largest object in the asteroid belt) -- its \"first "
                    "cosmic velocity\" -- in km/s. This is NOT escape velocity (escape velocity "
                    "is sqrt(2) times bigger); do not substitute a recalled escape-velocity "
                    "figure for this. Ceres is much less famous than a planet, so do not guess "
                    "its mass or radius from vague memory -- look them up. Two valid "
                    "strategies: (1) search_web directly for Ceres's circular orbital / "
                    "first-cosmic velocity; or (2) search_web for Ceres's mass (kg) and radius "
                    "(km or m), then compute v_orbit = sqrt(G * M / R) -- NO factor of 2 -- "
                    "with G = 6.674e-11 (SI units), M in kg, R in METERS -- the result is in "
                    "m/s, divide by 1000 for km/s. You will be scored on closeness to the true "
                    "value with graduated credit for tighter accuracy, so do not round "
                    "aggressively. Write ceres_orbital_velocity.py defining one function: "
                    "ceres_orbital_velocity_kms() -> float, returning your best estimate of "
                    "the ORBITAL (not escape) velocity in km/s. Use write_file to create it, "
                    "then use run_python to call the function and print the result -- if you "
                    "used the formula, double-check (a) you did NOT include the factor of 2 "
                    "that belongs to escape velocity, and (b) your km-to-m and m/s-to-km/s unit "
                    "conversions. Fix any issues with patch_file/run_python, then finish."
                ),
                "expect": (
                    "ceres_orbital_velocity.py written, defining ceres_orbital_velocity_kms() "
                    "-> float returning a close approximation of Ceres's real circular orbital "
                    "(surface) velocity, distinct from and slower than its escape velocity"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm ceres_orbital_velocity.py exists and report the sanity-check result.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["ceres_orbital_velocity.py"]},
    }
