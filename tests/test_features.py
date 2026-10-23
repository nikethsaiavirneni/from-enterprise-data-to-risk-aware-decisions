import numpy as np
import pytest

from feature_engineering import LAGS, price_features, rolling, sales_features, shift


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
