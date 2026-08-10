"""
codebench task c45 — hard/hidden, HYBRID web-search + code (sub-pattern 1: ported computation
shape). First of the c45-c52 hybrid batch: genuinely requires BOTH search_web (real facts are not
baked into the prompt or a visible test) AND run_python/write_file (the count is a COMPUTATION over
several looked-up numbers, not a single recallable fact). Reframes the QA suite's
count-with-condition shape (idea_tests/test_072_tier5_count_with_condition.py,
test_078_tier5_count_with_condition_b.py — "how many of N entities clear a numeric threshold") for
the codebench sandbox, which has NO page-visit tool (only ``search_web``, returning search-engine
SNIPPETS, not full pages — see connector_sandbox.py's ``search_web``). Population is a fact that
reliably surfaces in a plain search snippet (unlike e.g. topographic prominence, which needs an
opened infobox), so this shape ports cleanly.

REVISED 2026-08-06 after live calibration on qwen2.5:14b (run_id=coordinator_batch2) showed the
original six-country set was too easy: every country sat 11-49 million clear of the 50-million
threshold, and the per-country tolerance bands were generous enough (25-50% wide) that a model
answering purely from training-data memory, with ZERO web search, still landed inside every band
and got the classification count exactly right (10/10, 100%) — see the aider transcript
(codebench/results/runs/coordinator_batch2/c45__aider__qwen2.5_14b/submission/
.aider.chat.history.md), which opens with "### Population Estimates (as of 2023):" and a table
recited from memory, never calling search_web. Its Tanzania figure (61.0M) was ~11-15% stale
relative to the current (2024-2026) figure, and its South Africa figure (59.0M) was ~6% stale, but
both still cleared the old generous bands and — more importantly — both were nowhere near the
threshold, so staleness never risked the actual count.

The fix: add TWO more countries whose CURRENT population sits close enough to the 50-million line
(a 3-6% margin, not the 25-100% margins of the other six) that a rough or stale recollection
genuinely risks landing on the wrong side, while the correct current figure — verified live,
2026-08-06, against multiple reputable current sources — is unambiguous. Colombia is a
particularly sharp trap: its population crossed 50 million around 2018 and has been ~52-53 million
for the past several years, but it was very commonly described as "about 48-50 million" for a long
time before that and still often gets rounded down from stale general knowledge — exactly the kind
of figure a model would recite confidently and incorrectly without actually checking. South Korea
adds a second, less dramatic but still tight (51.7-51.9M, ~3.5% margin) close call. The set is now
EIGHT countries (up from six), which also raises plain aggregation risk (more independent facts
that must each be gotten right).

Ground truth (verified live 2026-08-06, several independent sources per country, current
2024-2026 estimates; every country's LOWEST plausible cited figure and HIGHEST plausible cited
figure both land on the same side of the 50-million threshold — see the per-country notes below):
    Tanzania       ~68.6-70.5 million   (UN 2024/2025 estimates)        -> ABOVE 50M (margin ~19M)
    Thailand       ~66-72 million       (registered-vs-resident spread) -> ABOVE 50M (margin ~16M)
    South Africa   ~63.0 million        (Stats SA 2024 mid-year est.)   -> ABOVE 50M (margin ~13M)
    Vietnam        ~100.1-102.3 million (World Bank/UN 2024/2025)       -> ABOVE 50M (margin ~50M)
    Colombia       ~52.1-53.0 million   (DANE/UN/Worldometer 2024/2025) -> ABOVE 50M (margin ~2.1M)
    South Korea    ~51.7-51.9 million   (Statistics Korea census 2024)  -> ABOVE 50M (margin ~1.7M)
    Poland         ~37.5-38.5 million   (GUS/UN 2024/2025, declining)   -> BELOW 50M (margin ~11.5M)
    Peru           ~34.0-34.2 million   (INEI 2024/2025 census)         -> BELOW 50M (margin ~15.8M)

    KEYSTONE = count of the 8 countries with population > 50 million = 6
    (Tanzania, Thailand, South Africa, Vietnam, Colombia, South Korea clear it; Poland, Peru do
    not.)

Colombia and South Korea both clear the threshold by a real but comparatively thin margin (~2-4%
of the threshold value) — every reputable current source still agrees they are above it, so the
classification itself is not ambiguous, but it demands an actually-current lookup rather than a
guess. The other six retain generous margins. Per-country test bands below remain generously wide
(well beyond the actual observed current-source spread) so a legitimately different — but still
roughly correct — figure the agent's own live search turns up still passes; they are NOT tight
precision checks (population is not a fixed physical constant) — the classification itself, via the
keystone count, is where real difficulty now lives.
"""
from __future__ import annotations

_TEST_FILE_PATH = "tests/test_country_pop.py"

_TEST_FILE_CONTENT = '''\
from country_pop import country_populations_millions, count_over_threshold


def test_counting_logic_on_synthetic_data():
    # Counting logic must be correct independent of any real-world fact: strictly GREATER THAN
    # the threshold (50.0 itself does not count), duplicates/extra keys handled fine.
    synthetic = {"A": 10, "B": 51, "C": 200, "D": 49.9, "E": 50.0}
    assert count_over_threshold(synthetic, threshold=50.0) == 2


def test_counting_logic_all_below():
    assert count_over_threshold({"A": 1, "B": 2, "C": 3}, threshold=50.0) == 0


def test_counting_logic_all_above():
    assert count_over_threshold({"A": 100, "B": 200}, threshold=50.0) == 2


def test_population_tanzania():
    pops = country_populations_millions()
    assert 58.0 <= pops["Tanzania"] <= 82.0


def test_population_thailand():
    pops = country_populations_millions()
    assert 60.0 <= pops["Thailand"] <= 82.0


def test_population_south_africa():
    pops = country_populations_millions()
    assert 55.0 <= pops["South Africa"] <= 72.0


def test_population_vietnam():
    pops = country_populations_millions()
    assert 88.0 <= pops["Vietnam"] <= 112.0


def test_population_colombia():
    pops = country_populations_millions()
    assert 50.5 <= pops["Colombia"] <= 58.0


def test_population_south_korea():
    pops = country_populations_millions()
    assert 50.5 <= pops["South Korea"] <= 57.0


def test_population_poland():
    pops = country_populations_millions()
    assert 30.0 <= pops["Poland"] <= 45.0


def test_population_peru():
    pops = country_populations_millions()
    assert 28.0 <= pops["Peru"] <= 40.0


def test_final_count_over_50_million():
    assert count_over_threshold() == 6
'''

# Discrete final answer + the counting logic itself gate the score (mirrors 072/078's "the count
# is the keystone" convention); the eight per-country population reads are graduated, ungated
# credit — a model that gets most countries right but drops one or two still earns most of the
# score.
KEYSTONE_TEST_IDS = [
    f"{_TEST_FILE_PATH}::test_counting_logic_on_synthetic_data",
    f"{_TEST_FILE_PATH}::test_final_count_over_50_million",
]


def get_test_metadata() -> dict:
    return {
        "test_id": "c45",
        "title": "country-population-threshold-count",
        "category": "hard",
    }


def get_task_statement() -> str:
    return (
        "You need the CURRENT (most recent available) population of eight countries, then a COUNT "
        "of how many of them exceed a threshold. Do not answer from memory — population figures "
        "drift (some of these countries are close enough to the line that an out-of-date or "
        "roughly-remembered figure can genuinely put you on the wrong side of it) — look each one "
        "up.\n\n"
        "The eight countries: Tanzania, Thailand, South Africa, Vietnam, Colombia, South Korea, "
        "Poland, Peru.\n\n"
        "For each one, use search_web to find its most recent population estimate, in millions "
        "(any recent reputable source is fine — UN, World Bank, Worldometer, national statistics "
        "office). Most of these eight are comfortably clear of the line either way, but a couple "
        "are close enough that you should double-check you're using a genuinely current figure, "
        "not an old or rounded one.\n\n"
        "Then write a Python module `country_pop.py` defining exactly two functions:\n\n"
        "  `country_populations_millions() -> dict` — returns a dict with exactly these eight "
        "keys: \"Tanzania\", \"Thailand\", \"South Africa\", \"Vietnam\", \"Colombia\", "
        "\"South Korea\", \"Poland\", \"Peru\", each mapped to that country's population in "
        "MILLIONS (a float, e.g. 12.0 for 12 million people — not the raw headcount).\n\n"
        "  `count_over_threshold(populations: dict = None, threshold: float = 50.0) -> int` — "
        "returns how many entries in `populations` (default: the return value of "
        "`country_populations_millions()`) are STRICTLY GREATER than `threshold`. This function "
        "must work correctly for ANY dict passed to it, not only the eight countries above — it "
        "is a generic counting helper, not something that special-cases these countries.\n\n"
        "There is no visible test file for this task. Use run_python to sanity-check both "
        "functions yourself before finishing: call `count_over_threshold` on a small made-up "
        "dict to confirm the counting logic (strictly greater-than, not >=) is right, and call "
        "`country_populations_millions()` to confirm all eight keys are present with sensible "
        "values before you finish."
    )


def get_visibility() -> str:
    return "hidden"


def get_sandbox_fixture() -> dict:
    return {}


def get_grading_payload() -> dict:
    return {
        "tests": {_TEST_FILE_PATH: _TEST_FILE_CONTENT},
        "entrypoint": {
            "module": "country_pop",
            "functions": ["country_populations_millions", "count_over_threshold"],
        },
        "keystone_test_ids": KEYSTONE_TEST_IDS,
    }


def get_compiled_plan() -> dict:
    return {
        "leaves": [
            {
                "id": "implement",
                "instruction": (
                    "You need the current population (in millions) of eight countries: Tanzania, "
                    "Thailand, South Africa, Vietnam, Colombia, South Korea, Poland, Peru. Use "
                    "search_web once per country (e.g. \"Colombia population 2025\") to find each "
                    "one's most recent population estimate — do not guess from memory, some of "
                    "these are close enough to typical round-number thresholds that an "
                    "out-of-date recollection can be genuinely wrong. Then write country_pop.py "
                    "defining two functions: (1) country_populations_millions() -> dict, "
                    "returning {\"Tanzania\": <float>, \"Thailand\": <float>, \"South Africa\": "
                    "<float>, \"Vietnam\": <float>, \"Colombia\": <float>, \"South Korea\": "
                    "<float>, \"Poland\": <float>, \"Peru\": <float>}, each value the country's "
                    "population IN MILLIONS (e.g. 12.0, not 12000000); (2) "
                    "count_over_threshold(populations: dict = None, threshold: float = 50.0) -> "
                    "int, which counts how many entries in `populations` (defaulting to "
                    "country_populations_millions() when not given) are STRICTLY greater than "
                    "`threshold` — this must be a generic counting helper that works on any dict "
                    "you pass it, not hardcoded to the eight countries. Use write_file to create "
                    "country_pop.py, then use run_python to check: (a) "
                    "count_over_threshold({\"a\": 10, \"b\": 60}, threshold=50) returns 1; (b) "
                    "country_populations_millions() has all eight keys with plausible values. Fix "
                    "any issues with patch_file/run_python, then finish."
                ),
                "expect": (
                    "country_pop.py written, defining country_populations_millions() -> dict "
                    "(eight countries, population in millions) and count_over_threshold(dict, "
                    "threshold) -> int (generic strictly-greater-than counter)"
                ),
                "depends_on": [],
            }
        ],
        "aggregation": "Confirm country_pop.py exists and report the sanity-check results.",
        "agg_mode": "sandbox_submit",
        "composition": {"op": "submit_files", "files": ["country_pop.py"]},
    }
