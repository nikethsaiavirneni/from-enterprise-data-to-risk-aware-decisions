import numpy as np

from metrics import mae, naive_scale, rmse, rmsse, wape


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
