"""Tests for decide() with made-up numbers. Run from the repo root:  pytest partB -v"""
import pytest

from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 2.0}
GOOD = {"mae": 8.0, "mae_rush": 9.0, "mae_rain": 8.5, "p95_ms": 4.0}


def test_clear_win_promotes():
    out = decide(PROD, GOOD)
    assert out["promote"] is True
    assert all(out["criteria"].values())
    assert set(out["criteria"]) == {"mae_gain", "latency_p95", "rush_slice", "rain_slice"}


def test_returns_bools_and_is_pure():
    before_p, before_c = dict(PROD), dict(GOOD)
    out = decide(PROD, GOOD)
    assert all(isinstance(v, bool) for v in out["criteria"].values())
    assert isinstance(out["promote"], bool)
    assert PROD == before_p and GOOD == before_c     # inputs not modified
    assert decide(PROD, GOOD) == out                 # same input -> same output


@pytest.mark.parametrize("cand_mae, expected", [
    (9.70, True),    # exactly 3.0 % better: "at least" -> promote
    (9.69, True),    # just inside
    (9.71, False),   # 2.9 %: just outside
    (10.5, False),   # worse
])
def test_mae_gain_boundary(cand_mae, expected):
    out = decide(PROD, {**GOOD, "mae": cand_mae})
    assert out["criteria"]["mae_gain"] is expected
    assert out["promote"] is expected


@pytest.mark.parametrize("p95, expected", [(10.0, True), (9.99, True), (10.01, False)])
def test_latency_boundary(p95, expected):
    out = decide(PROD, {**GOOD, "p95_ms": p95})
    assert out["criteria"]["latency_p95"] is expected
    assert out["promote"] is expected


@pytest.mark.parametrize("slice_key, prod_val", [("mae_rush", 12.0), ("mae_rain", 11.0)])
def test_slice_regression_boundary(slice_key, prod_val):
    name = {"mae_rush": "rush_slice", "mae_rain": "rain_slice"}[slice_key]
    at_limit = decide(PROD, {**GOOD, slice_key: prod_val * 1.05})          # exactly +5 %: allowed
    inside = decide(PROD, {**GOOD, slice_key: prod_val * 1.049})
    outside = decide(PROD, {**GOOD, slice_key: prod_val * 1.051})
    assert at_limit["criteria"][name] and inside["criteria"][name]
    assert outside["criteria"][name] is False and outside["promote"] is False


def test_slice_regression_blocks_even_with_big_overall_gain():
    # 40 % better overall but much worse when it rains -> must not promote
    cand = {**GOOD, "mae": 6.0, "mae_rain": 14.0}
    out = decide(PROD, cand)
    assert out["criteria"]["mae_gain"] and out["criteria"]["latency_p95"]
    assert out["criteria"]["rain_slice"] is False
    assert out["promote"] is False


def test_custom_thresholds():
    assert decide(PROD, GOOD, min_rel_gain=0.20)["promote"] is True     # exactly 20 % better
    assert decide(PROD, GOOD, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, GOOD, max_p95_ms=3.0)["criteria"]["latency_p95"] is False
    assert decide(PROD, {**GOOD, "mae_rush": 13.0}, max_slice_regression=0.10)["criteria"]["rush_slice"] is True
