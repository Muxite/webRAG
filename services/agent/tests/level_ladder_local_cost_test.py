"""Regression tests for scripts/level_ladder.py's local-model cost reporting.

Before this fix, ``row["usd"] = row.get("usd") or 0.0`` coerced an unpriced/local model's
missing cost to a literal ``0.0``, which the report printed as a real ``$0.00000`` figure
(indistinguishable from an actually-measured near-zero cost). ``_fmt_usd`` must instead
report "local" for an all-local group and compute the mean over priced rows only when a
group mixes local and API models.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
import level_ladder as ladder  # noqa: E402


def _row(usd=None, origin=None, model="x"):
    return {"usd": usd, "origin": origin, "model": model}


def test_fmt_usd_all_priced_rows_reports_mean():
    rows = [_row(usd=0.01, origin="api"), _row(usd=0.03, origin="api")]
    assert ladder._fmt_usd(rows) == "0.02000"


def test_fmt_usd_all_local_rows_reports_local_not_zero():
    rows = [_row(usd=None, origin="local"), _row(usd=None, origin="local")]
    assert ladder._fmt_usd(rows) == "local"


def test_fmt_usd_mixed_local_and_priced_excludes_local_from_mean():
    rows = [_row(usd=0.02, origin="api"), _row(usd=None, origin="local")]
    assert ladder._fmt_usd(rows) == "0.02000"


def test_fmt_usd_empty_group_is_na():
    assert ladder._fmt_usd([]) == "n/a"


def test_fmt_usd_unpriced_non_local_is_na():
    # A paid model with a pricing-table gap (origin="api", usd=None) is a real data gap,
    # not "free" — must not be reported as "local".
    rows = [_row(usd=None, origin="api")]
    assert ladder._fmt_usd(rows) == "n/a"


def test_is_local_falls_back_to_unpriced_model_when_origin_absent():
    # Legacy result files predating the `origin` field fall back to model_costs.is_local_row's
    # own fallback, which is model-name-based (unpriced -> local), not usd-based.
    assert ladder._is_local(_row(usd=None, origin=None, model="qwen2.5:14b")) is True
    assert ladder._is_local(_row(usd=0.01, origin=None, model="gpt-5-mini")) is False
