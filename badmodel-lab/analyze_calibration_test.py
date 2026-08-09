"""Tests for the calibration-aware secondary metric in analyze.py (abstention
classification, Brier score, 2-point AURC).

analyze.py had no existing test file anywhere in the repo before this one — kept
minimal, following the plain-pytest / sys.path-injection convention used by
agent/tests/adaptive_analyzer_rigor_test.py for other standalone scripts
that live outside the services.* package tree.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

# ---- 4 real, hand-labeled final_deliverable texts, pulled verbatim (via repr())
# from agent/idea_test_results/bml__qwen2.5-7b__m1_thin__hard_*.json ----

HARD_071_R1 = 'The gathered facts do not include the depths of the five candidate lakes. Therefore, it is impossible to compute the absolute differences as required in the task.\n\nPlease provide the depth information for each lake from credible sources such as Wikipedia or other reliable encyclopedias and databases. Once you have that data, I can proceed with the necessary calculations and determinations.\n\nSources actually gathered:\n- https://en.wikipedia.org/wiki/Lake_Matano'

HARD_071_R2 = "Subtracting the depths of the candidate lakes from Lake Matano's depth (590 metres):\n\n- Lake Tahoe: |501 - 590| = 89 metres\n- Lake Superior's depth is unknown, so no subtraction can be performed.\n- Lake Kivu's depth is unknown, so no subtraction can be performed.\n- Great Slave Lake's depth is unknown, so no subtraction can be performed.\n- Lake Malawi's depth is unknown, so no subtraction can be performed.\n\nSince we are unable to perform all the necessary subtractions because of missing data for four candidate lakes (Lake Superior, Lake Kivu, Great Slave Lake, and Lake Malawi), we cannot accurately determine which candidate lake's maximum depth is closest to that of Lake Matano. Only Lake Tahoe has a valid difference: 89 metres from Lake Matano's maximum depth.\n\nUnfortunately, the data required to draw any further conclusions is incomplete. Here are the depths known for all lakes referenced:\n\n- Lake Matano: 590 metres - [https://en.wikipedia.org/wiki/Lake_Matano]\n- Lake Tahoe: 501 metres - [https://en.wikipedia.org/wiki/Lake_Tahoe]"

HARD_077_R1 = 'Tatara Bridge: 890 m\nPont de Normandie: 856 m\nRio–Antirrio Bridge: 2,214 m\nEdong Yangtze River Bridge: 1,750 m\nRussky Bridge: 1,104 m\nHelgeland Bridge: 425 m\n\nThe Rio–Antirrio Bridge has the longest main span at 2,214 m. \n\nSources:\n- https://en.wikipedia.org/wiki/Tatara_Bridge\n- https://en.wikipedia.org/wiki/Pont_de_Normandie\n- https://en.wikipedia.org/wiki/Rio%E2%80%93Antirrio_Bridge\n- https://en.wikipedia.org/wiki/Edong_Yangtze_River_Bridge\n- https://en.wikipedia.org/wiki/Russky_Bridge\n- https://en.wikipedia.org/wiki/Helgeland_Bridge'

HARD_063_R2 = 'element,atomic_number,melting_point_k\nSamarium,62,1072\nGadolinium,64,1802\nErbium,68,1925\nLutetium,71,1802'


def test_is_abstained_genuine_refusal():
    # 071 r1: "impossible to compute" — a genuine global refusal, no data gathered at all.
    assert analyze._is_abstained(HARD_071_R1) is True


def test_is_abstained_hedges_but_still_computes():
    # 071 r2: hedges on 4/5 lakes ("Lake Superior's depth is unknown...") but still
    # commits a computed answer for Lake Tahoe — must NOT be classified as abstained.
    assert analyze._is_abstained(HARD_071_R2) is False


def test_is_abstained_fully_confident_answer():
    # 077 r1: a fully confident bridge-length answer with a trailing citation block.
    assert analyze._is_abstained(HARD_077_R1) is False


def test_is_abstained_confident_data_table():
    # 063 r2: a confident CSV data table, no hedging language anywhere.
    assert analyze._is_abstained(HARD_063_R2) is False


def test_strip_citation_tail_drops_trailing_sources_block():
    text = "The answer is 42.\n\nSources:\n- https://example.com/a\n- https://example.com/b"
    assert analyze._strip_citation_tail(text) == "The answer is 42."


def test_strip_citation_tail_noop_when_no_citation_block():
    text = "The answer is 42, computed from the gathered facts."
    assert analyze._strip_citation_tail(text) == text


def test_brier_term_correct_is_zero():
    assert analyze._brier_term("correct") == 0.0


def test_brier_term_confident_wrong_is_one():
    assert analyze._brier_term("confident_wrong") == 1.0


def test_brier_term_abstained_is_quarter():
    assert analyze._brier_term("abstained") == 0.25


def _row(bucket):
    return {"bucket": bucket, "brier_term": analyze._brier_term(bucket)}


def test_cell_calibration_synthetic_mixed_cell():
    # 3 correct, 1 confident_wrong, 1 abstained (n=5) — hand-computed:
    #   abstain_rate=0.2, correct_rate=0.6, confident_wrong_rate=0.2
    #   risk_full = 0.2 + 0.2 = 0.4
    #   risk_selective = 1 / (5-1) = 0.25
    #   aurc = 0.5*(0.25+0.4)*0.2 + 0.25*0.8 = 0.065 + 0.2 = 0.265
    #   brier = (0*3 + 1*1 + 0.25*1) / 5 = 0.25
    rows = [_row("correct")] * 3 + [_row("confident_wrong")] + [_row("abstained")]
    cal = analyze._cell_calibration(rows)
    assert cal["n"] == 5
    assert cal["abstain_rate"] == 0.2
    assert cal["correct_rate"] == 0.6
    assert cal["confident_wrong_rate"] == 0.2
    assert cal["risk_full"] == 0.4
    assert cal["risk_selective"] == 0.25
    assert cal["aurc"] == 0.265
    assert cal["brier"] == 0.25


def test_cell_calibration_all_abstained_no_nan():
    # All-abstained: risk_selective would be 0/0 — must be guarded to None, and aurc
    # must collapse to risk_full (1.0), never NaN.
    rows = [_row("abstained")] * 4
    cal = analyze._cell_calibration(rows)
    assert cal["n"] == 4
    assert cal["abstain_rate"] == 1.0
    assert cal["correct_rate"] == 0.0
    assert cal["risk_selective"] is None
    assert cal["risk_full"] == 1.0
    assert cal["aurc"] == 1.0
    assert cal["aurc"] == cal["aurc"]  # not NaN (NaN != NaN)
    assert cal["brier"] == 0.25
