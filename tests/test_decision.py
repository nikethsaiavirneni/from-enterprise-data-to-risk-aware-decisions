import numpy as np

from decision_engine import (RULES, adjustment, cost_parts, evaluate, fit_calibration,
                             recommend, risk_strata, strata_table)


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
