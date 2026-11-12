import numpy as np
import pandas as pd
import pytest

from credit_core.binning import WoeEncoder, woe_table
from credit_core.economics import cutoff_profit, expected_loss
from credit_core.metrics import gini, ks_statistic, psi
from credit_core.scorecard import Scorecard, points_scaling


def _toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    p_good = 1 / (1 + np.exp(-(1.5 * x)))
    y = pd.Series((rng.uniform(size=n) < p_good).astype(int))
    X = pd.DataFrame({"x": x, "cat": rng.choice(["a", "b", "c"], size=n)})
    return X, y


def test_woe_sign_follows_target_convention():
    # Target 1 = good. Bin with more goods must have positive WOE.
    bins = np.array(["lo"] * 100 + ["hi"] * 100)
    y = pd.Series([0] * 80 + [1] * 20 + [1] * 80 + [0] * 20)
    table = woe_table(bins, y)
    assert table.loc["hi", "woe"] > 0
    assert table.loc["lo", "woe"] < 0


def test_iv_is_non_negative_and_zero_for_uninformative_feature():
    rng = np.random.default_rng(1)
    n = 4000
    y = pd.Series(rng.integers(0, 2, size=n))
    X = pd.DataFrame({"noise": rng.normal(size=n)})
    enc = WoeEncoder(numeric=["noise"], categorical=[], iv_min=0.0, iv_max=10).fit(X, y)
    assert enc.iv_["noise"] >= 0
    assert enc.iv_["noise"] < 0.01


def test_encoder_fits_on_train_only_and_maps_unseen_bins_to_zero():
    X, y = _toy()
    enc = WoeEncoder(numeric=["x"], categorical=["cat"], iv_min=0.0, iv_max=10).fit(X, y)
    X_new = X.copy()
    X_new.loc[X_new.index[0], "cat"] = "never_seen"
    out = enc.transform(X_new)
    assert out.loc[X_new.index[0], "cat"] == 0.0


def test_iv_filter_drops_out_of_range_features():
    X, y = _toy()
    enc = WoeEncoder(numeric=["x"], categorical=["cat"], iv_min=0.5, iv_max=10).fit(X, y)
    assert "cat" in enc.dropped_
    assert "x" not in enc.dropped_ or enc.iv_["x"] < 0.5


def test_points_doubling_odds_adds_pdo():
    factor, offset = points_scaling(pdo=20, base_score=600, base_odds=50)
    s_50 = offset + factor * np.log(50)
    s_100 = offset + factor * np.log(100)
    assert s_100 - s_50 == pytest.approx(20)
    assert s_50 == pytest.approx(600)


def test_scorecard_scores_increase_with_good_odds():
    X, y = _toy()
    enc = WoeEncoder(numeric=["x"], categorical=["cat"], iv_min=0.0, iv_max=10)
    sc = Scorecard(enc, C=1.0).fit(X, y)
    pd_ = sc.predict_pd(X)
    assert np.all((pd_ >= 0) & (pd_ <= 1))
    # Higher x means higher good probability, so lower PD and higher score.
    lo = X["x"].idxmin()
    hi = X["x"].idxmax()
    assert sc.score(X.loc[[hi]])[0] > sc.score(X.loc[[lo]])[0]


def test_psi_zero_for_identical_samples():
    rng = np.random.default_rng(2)
    a = rng.normal(size=5000)
    assert psi(a, a) == pytest.approx(0.0, abs=1e-9)


def test_psi_large_for_shifted_sample():
    rng = np.random.default_rng(3)
    a = rng.normal(size=5000)
    b = rng.normal(loc=2.0, size=5000)
    assert psi(a, b) > 0.25


def test_gini_and_ks_sanity():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert gini(y, perfect) == pytest.approx(1.0)
    assert ks_statistic(y, perfect) == pytest.approx(1.0)


def test_expected_loss_scales_with_pd_lgd_ead():
    el = expected_loss([0.1, 0.2], lgd=0.5, ead=[1000, 2000])
    assert el.tolist() == pytest.approx([50.0, 200.0])


def test_cutoff_profit_approves_only_low_pd():
    pd_ = np.array([0.05, 0.5, 0.9])
    defaulted = np.array([0, 1, 1])
    ead = np.array([1000.0, 1000.0, 1000.0])
    table = cutoff_profit(pd_, defaulted, ead, margin=0.1, lgd=0.5, cutoffs=[0.1, 1.0])
    low = table.set_index("cutoff_pd").loc[0.1]
    assert low["approved"] == 1
    assert low["realized_profit"] == pytest.approx(100.0)
    full = table.set_index("cutoff_pd").loc[1.0]
    assert full["approved"] == 3
