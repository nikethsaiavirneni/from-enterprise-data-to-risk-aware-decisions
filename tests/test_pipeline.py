"""Unit tests: metrics, chronological split and sampling, leakage-safe features, decision rules."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import (LAGS, assign_split, mae, naive_scale, on_shelf_mask, price_features,
                  rmse, rmsse, rolling, sales_features, sample_items, shift, split_bounds, wape)
from decision import (RULES, adjustment, cost_parts, evaluate, fit_calibration, recommend,
                      risk_strata, strata_table)


# ---- metrics ---------------------------------------------------------------


def test_basic_metrics_hand_computed():
    y = np.array([0.0, 2.0, 4.0, 0.0])
    p = np.array([1.0, 2.0, 1.0, 0.0])
    assert mae(y, p) == 1.0                       # (1+0+3+0)/4
    assert np.isclose(rmse(y, p), np.sqrt(10 / 4))
    assert np.isclose(wape(y, p), 4 / 6)          # sum|e| / sum|y|


def test_wape_is_finite_with_zero_actuals_where_mape_is_not():
    y = np.array([0.0, 0.0, 5.0])
    p = np.array([1.0, 0.0, 5.0])
    assert np.isfinite(wape(y, p))


def test_naive_scale_and_rmsse():
    s = [np.array([1, 3, 3, 7]), np.array([5, 5, 5])]
    sc = naive_scale(s)
    assert np.isclose(sc[0], np.mean([4, 0, 16])) and sc[1] == 0
    y = np.array([2.0, 4.0, 9.0])
    p = np.array([2.0, 2.0, 9.0])
    sid = np.array([0, 0, 1])
    # series 0: mse = (0+4)/2 = 2, scale 20/3 ; series 1 has scale 0 -> skipped
    assert np.isclose(rmsse(y, p, sid, sc), np.sqrt(2 / (20 / 3)))


def test_perfect_forecast_gives_zero():
    y = np.arange(6.0)
    assert mae(y, y) == 0 and rmse(y, y) == 0 and wape(y, y) == 0


# ---- split and sampling ----------------------------------------------------


def test_split_is_contiguous_and_covers_all_days():
    b = split_bounds(1941, 91, 91)
    assert b["train"][0] == 1 and b["test"][1] == 1941
    assert b["train"][1] + 1 == b["val"][0]
    assert b["val"][1] + 1 == b["test"][0]
    assert b["val"][1] - b["val"][0] + 1 == 91
    assert b["test"][1] - b["test"][0] + 1 == 91


def test_assign_split_is_chronological_and_never_shuffles():
    b = split_bounds(100, 10, 10)
    s = assign_split(np.arange(1, 101), b)
    assert (np.diff(s) >= 0).all()                       # train -> val -> test only
    assert (s == 0).sum() == 80 and (s == 1).sum() == 10 and (s == 2).sum() == 10


def test_on_shelf_mask_drops_weeks_without_price():
    price_wide = np.array([[np.nan, 2.0, 3.0], [1.0, np.nan, 3.0]], dtype=np.float32)
    week_pos = np.array([0, 0, 1, 2])
    m = on_shelf_mask(price_wide, week_pos)
    assert m.tolist() == [[False, False, True, True], [True, True, False, True]]


def test_sample_items_is_reproducible_and_stratified():
    items = pd.DataFrame({"item_id": [f"i{k}" for k in range(60)],
                          "dept_id": ["A"] * 40 + ["B"] * 20})
    a = sample_items(items, 12, seed=1)
    assert a == sample_items(items, 12, seed=1)
    assert a != sample_items(items, 12, seed=2)
    depts = items.set_index("item_id").loc[a, "dept_id"]
    assert set(depts) == {"A", "B"} and (depts == "A").sum() == 8


# ---- features (no leakage) -------------------------------------------------


@pytest.mark.parametrize("h", [1, 28])
def test_sales_features_never_use_data_after_t_minus_h(h):
    rng = np.random.default_rng(0)
    y = rng.poisson(1.5, size=(5, 200)).astype(np.int16)
    base = sales_features(y, h)
    t0 = 120                              # corrupt every sale from day t0 onward
    y2 = y.copy()
    y2[:, t0:] = 999
    alt = sales_features(y2, h)
    for name in base:
        # day t may only see sales up to t-h, so columns t < t0 + h must be identical
        a, b = base[name][:, : t0 + h], alt[name][:, : t0 + h]
        assert np.allclose(a, b, equal_nan=True), f"{name} leaks at h={h}"


@pytest.mark.parametrize("h", [1, 28])
def test_all_lags_are_at_least_horizon(h):
    assert min(LAGS[h]) >= h


def test_lag_values_are_correct():
    y = np.arange(100, dtype=np.int16)[None, :]
    f = sales_features(y, 28)
    assert f["lag_28"][0, 60] == 32 and f["lag_56"][0, 60] == 4
    assert np.isnan(f["lag_56"][0, 55])
    assert f["roll_mean_7"][0, 60] == np.mean(np.arange(26, 33))  # days 26..32 = t-28-6..t-28


def test_shift_and_rolling_basic():
    a = np.arange(10, dtype=float)[None, :]
    assert np.isnan(shift(a, 3)[0, :3]).all() and shift(a, 3)[0, 5] == 2
    r = rolling(a, 4, "mean")
    assert np.isnan(r[0, 2]) and r[0, 3] == 1.5 and r[0, 9] == 7.5
    assert np.isclose(rolling(a, 4, "std")[0, 9], np.std([6, 7, 8, 9]))


def test_price_ratio_uses_price_seven_days_earlier():
    p = np.tile(np.arange(1.0, 31.0), (2, 1))
    f = price_features(p, 1)
    assert np.isclose(f["price_ratio_7"][0, 10], 11 / 4)
    assert np.isnan(f["price_ratio_7"][0, 3])


# ---- decision layer --------------------------------------------------------


def test_cost_parts_asymmetric():
    u, o = cost_parts(np.array([5.0, 1.0]), np.array([3.0, 4.0]), cu=4.0, co=1.0)
    assert u.tolist() == [8.0, 0.0] and o.tolist() == [0.0, 3.0]


def test_point_policy_adds_nothing_and_high_tau_raises_quantity():
    rng = np.random.default_rng(0)
    f = rng.uniform(0, 5, 2000)
    y = rng.poisson(f).astype(float)
    cal = fit_calibration(f, y)
    assert not adjustment(cal, f, 0.9, "point").any()
    assert adjustment(cal, f, 0.95, "global")[0] > adjustment(cal, f, 0.5, "global")[0]


def test_risk_aware_beats_point_when_understock_is_costly():
    rng = np.random.default_rng(1)
    f_va = rng.uniform(0.1, 6, 20000)
    y_va = rng.poisson(f_va).astype(float)
    f_te = rng.uniform(0.1, 6, 20000)
    y_te = rng.poisson(f_te).astype(float)
    cal = fit_calibration(f_va, y_va)
    res = evaluate(np.arange(20000) % 200, y_te, f_te, cal, ratios=(8,), boot=50)
    c = res.set_index("policy").cost_per_unit
    assert c["risk_aware"] < c["point"]
    assert res.set_index("policy").loc["risk_aware", "reduction_vs_point"] > 0


def test_thresholds_depend_only_on_validation_data():
    rng = np.random.default_rng(2)
    f_va = rng.uniform(0, 5, 5000)
    y_va = rng.poisson(f_va).astype(float)
    cal = fit_calibration(f_va, y_va)
    f1, f2 = rng.uniform(0, 5, 100), rng.uniform(0, 5, 100)
    _, cuts1, _ = risk_strata(cal, f1, f_va)
    _, cuts2, _ = risk_strata(cal, f2, f_va)
    assert np.allclose(cuts1, cuts2)          # cut-offs unchanged by the test cases


def test_recommend_rules_cover_every_cell_and_strata_table_shape():
    assert set(RULES) == {(a, b) for a in ("High", "Low") for b in ("Low", "Moderate", "High")}
    f_va = np.linspace(0, 6, 300)
    lvl, pr, act = recommend(np.array([0.1, 5.9]), f_va, np.array(["Low", "High"]))
    assert lvl.tolist() == ["Low", "High"] and pr[0] == "P4" and "human review" in act[1]
    st = strata_table(np.array(["Low", "High", "High"]), np.array([1.0, 2.0, 0.0]),
                      np.array([1.0, 1.0, 1.0]))
    assert list(st.index) == ["Low", "Moderate", "High"] and st.loc["High", "cases"] == 2
