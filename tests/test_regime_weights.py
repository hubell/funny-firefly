# SPDX-License-Identifier: GPL-3.0-only
"""C8's weights, ``taxiout.regime_weights``, on a constructed frame."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("lightgbm")

from taxiout.calendar import HOLIDAY_FEATURES  # noqa: E402
from taxiout.congestion import CONGESTION_FEATURES  # noqa: E402
from taxiout.features import TIME_FEATURES  # noqa: E402
from taxiout.metar import WEATHER_FEATURES  # noqa: E402
from taxiout.regime_weights import (  # noqa: E402
    CLASSIFIER_FEATURES,
    classifier_params,
    clip_bounds,
    regime_weights,
)


def test_no_calendar_or_date_field_is_a_classifier_feature():
    assert CLASSIFIER_FEATURES == ("icao",) + WEATHER_FEATURES + CONGESTION_FEATURES
    assert not set(CLASSIFIER_FEATURES) & (set(TIME_FEATURES) | set(HOLIDAY_FEATURES))
    words = ("day", "month", "date", "dow", "doy", "week", "year", "holiday", "hour")
    assert [c for c in CLASSIFIER_FEATURES if any(w in c for w in words)] == []


def test_the_classifier_is_the_amendments():
    p = classifier_params()
    assert (p["objective"], p["n_estimators"], p["learning_rate"]) == (
        "binary",
        200,
        0.05,
    )
    assert (p["num_leaves"], p["min_child_samples"], p["random_state"]) == (
        31,
        500,
        20260902,
    )
    assert p["deterministic"] and p["force_col_wise"] and p["n_jobs"] == 8
    assert clip_bounds() == (0.1, 10.0)


def _frame(n=6000, seed=5):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({c: rng.normal(0, 1, n) for c in CLASSIFIER_FEATURES[1:]})
    frame["icao"] = rng.choice(["EDDF", "LSZH"], n)
    takeoff = pd.Series(
        pd.Timestamp("2025-01-01T00:00:00Z")
        + pd.to_timedelta(rng.integers(0, 60 * 86400, n), unit="s")
    )
    target = pd.Series(rng.random(n) < 0.2)
    # the target month is colder: the classifier has something to find
    frame.loc[target.to_numpy(), "wx_freezing"] += 2.0
    train = pd.Series(~target.to_numpy() & (rng.random(n) < 0.9))
    return frame, takeoff, train, target


def test_weights_are_clipped_then_scaled_to_a_mean_of_one():
    frame, takeoff, train, target = _frame()
    got = regime_weights(frame, takeoff, train, target)
    assert len(got.weights) == int(train.sum())
    assert got.weights.mean() == pytest.approx(1.0)
    low, high = clip_bounds()
    ratio = got.weights.max() / got.weights.min()
    assert ratio <= high / low + 1e-9
    raw = got.p / (1 - got.p)
    assert got.clipped_low == int((raw < low).sum())
    assert got.clipped_high == int((raw > high).sum())
    d = got.distribution()
    assert d["min"] <= d["q1"] <= d["median"] <= d["q3"] <= d["max"]


def test_rows_like_the_target_weigh_more():
    frame, takeoff, train, target = _frame()
    got = regime_weights(frame, takeoff, train, target)
    cold = frame.loc[train.to_numpy(), "wx_freezing"].to_numpy() > 1.5
    assert got.weights[cold].mean() > got.weights[~cold].mean()


def test_a_training_rows_probability_is_from_the_other_parity():
    """Poisoning the even days' target rows moves no even-day training row's p.

    The even-day training rows are scored by the classifier fitted on the odd
    days, which never sees an even-day row; the odd-day rows by the one fitted
    on the even days, which the poisoning moves.
    """
    frame, takeoff, train, target = _frame()
    before = regime_weights(frame, takeoff, train, target)
    odd = (pd.to_datetime(takeoff, utc=True).dt.day % 2 == 1).to_numpy()
    poisoned = frame.copy()
    poisoned.loc[~odd & target.to_numpy(), "wx_vis_band"] = 50.0
    after = regime_weights(poisoned, takeoff, train, target)
    odd_train = odd[train.to_numpy()]
    np.testing.assert_array_equal(before.p[~odd_train], after.p[~odd_train])
    assert not np.array_equal(before.p[odd_train], after.p[odd_train])


def test_deterministic():
    frame, takeoff, train, target = _frame()
    a = regime_weights(frame, takeoff, train, target)
    b = regime_weights(frame, takeoff, train, target)
    np.testing.assert_array_equal(a.weights, b.weights)


def test_a_row_cannot_be_both():
    frame, takeoff, train, target = _frame()
    with pytest.raises(ValueError):
        regime_weights(frame, takeoff, train | target, target)
