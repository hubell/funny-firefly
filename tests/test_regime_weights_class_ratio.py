# SPDX-License-Identifier: GPL-3.0-only
"""C8c's factor, ``regime_weights(..., class_ratio=True)``, on constructed frames.

C8's own tests are ``tests/test_regime_weights.py``, unedited; these hold the
one change the amendment of 6 October 2026 makes: each training row's odds
multiplied, before the clip, by n0 / n1 of the fold whose classifier scored
it.
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("lightgbm")

from taxiout.regime_weights import CLASSIFIER_FEATURES, regime_weights  # noqa: E402


def _balanced(n=8000, seed=11):
    """Equal counts of each label in each parity, and a regime to find."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    frame = pd.DataFrame({c: rng.normal(0, 1, n) for c in CLASSIFIER_FEATURES[1:]})
    frame["icao"] = rng.choice(["EDDF", "LSZH"], n)
    day = 1 + 2 * rng.integers(0, 14, n) + idx % 2
    takeoff = pd.Series(pd.to_datetime([f"2025-01-{d:02d}T10:00:00Z" for d in day]))
    target = pd.Series((idx // 2) % 2 == 1)
    frame.loc[target.to_numpy(), "wx_freezing"] += 2.0
    return frame, takeoff, ~target, target


def _inseparable(counts):
    """Identical features on every row, so no classifier can separate the labels.

    ``counts`` maps a day of January to (training rows, target rows).
    """
    days, labels = [], []
    for day, (n0, n1) in counts.items():
        days += [day] * (n0 + n1)
        labels += [False] * n0 + [True] * n1
    n = len(days)
    frame = pd.DataFrame({c: np.zeros(n) for c in CLASSIFIER_FEATURES[1:]})
    frame["icao"] = "EDDF"
    takeoff = pd.Series(pd.to_datetime([f"2025-01-{d:02d}T10:00:00Z" for d in days]))
    target = pd.Series(labels)
    return frame, takeoff, ~target, target


def test_with_balanced_classes_the_factor_is_one_and_the_weights_are_c8s():
    frame, takeoff, train, target = _balanced()
    c8 = regime_weights(frame, takeoff, train, target)
    c8c = regime_weights(frame, takeoff, train, target, class_ratio=True)
    assert [f["factor"] for f in c8c.folds] == [1.0, 1.0]
    assert [(f["n0"], f["n1"]) for f in c8c.folds] == [(2000, 2000), (2000, 2000)]
    np.testing.assert_array_equal(c8c.p, c8.p)
    np.testing.assert_array_equal(c8c.weights, c8.weights)
    assert (c8c.clipped_low, c8c.clipped_high) == (c8.clipped_low, c8.clipped_high)


def test_the_option_is_off_by_default_and_c8s_distribution_is_unchanged():
    frame, takeoff, train, target = _balanced()
    c8 = regime_weights(frame, takeoff, train, target)
    assert c8.class_ratio is None
    assert set(c8.distribution()) == {
        "rows",
        "min",
        "q1",
        "median",
        "q3",
        "max",
        "clipped_low",
        "clipped_high",
        "share_clipped",
        "raw_mean",
    }


def test_one_in_ten_and_no_separation_give_every_weight_one_before_scaling():
    # n1 / n0 = 0.1 in each fold, the classifier left with nothing to split on
    frame, takeoff, train, target = _inseparable({2: (5000, 500), 3: (5000, 500)})
    c8c = regime_weights(frame, takeoff, train, target, class_ratio=True)
    assert [f["factor"] for f in c8c.folds] == [10.0, 10.0]
    before_scaling = c8c.p / (1.0 - c8c.p) * 10.0
    np.testing.assert_allclose(before_scaling, 1.0, rtol=1e-9)
    assert c8c.raw_mean == pytest.approx(1.0, rel=1e-9)
    np.testing.assert_allclose(c8c.weights, 1.0, rtol=1e-9)
    assert (c8c.clipped_low, c8c.clipped_high) == (0, 0)
    # C8 on the same rows: every odds at the floor's edge, the prior and not the regime
    c8 = regime_weights(frame, takeoff, train, target)
    np.testing.assert_allclose(c8.p / (1.0 - c8.p), 0.1, rtol=1e-9)


def test_the_factor_is_the_fitting_folds_own():
    """Even days 1 in 10, odd days 1 in 4: each row takes the ratio of the classifier that scored it.

    The even-day rows are scored by the classifier fitted on the odd days, and
    the odd-day rows by the one fitted on the even days. With nothing to
    separate, each classifier returns its own fold's prior, so only that
    fold's n0 / n1 brings every weight to 1; the pooled ratio, or the scored
    row's own fold's, would not.
    """
    frame, takeoff, train, target = _inseparable({2: (5000, 500), 3: (4000, 1000)})
    c8c = regime_weights(frame, takeoff, train, target, class_ratio=True)
    by = {f["fitted_on"]: f for f in c8c.folds}
    assert (by["even days"]["n0"], by["even days"]["n1"]) == (5000, 500)
    assert (by["odd days"]["n0"], by["odd days"]["n1"]) == (4000, 1000)
    assert (by["even days"]["factor"], by["odd days"]["factor"]) == (10.0, 4.0)
    assert by["even days"]["scored"] == 4000 and by["odd days"]["scored"] == 5000
    np.testing.assert_allclose(c8c.weights, 1.0, rtol=1e-9)
    assert c8c.raw_mean == pytest.approx(1.0, rel=1e-9)
    odd = (pd.to_datetime(takeoff, utc=True).dt.day % 2 == 1).to_numpy()[train.to_numpy()]
    odds = c8c.p / (1.0 - c8c.p)
    np.testing.assert_allclose(odds[odd], 0.1, rtol=1e-9)
    np.testing.assert_allclose(odds[~odd], 0.25, rtol=1e-9)


def test_the_distribution_carries_each_folds_counts():
    frame, takeoff, train, target = _inseparable({2: (5000, 500), 3: (4000, 1000)})
    d = regime_weights(frame, takeoff, train, target, class_ratio=True).distribution()
    assert d["class_ratio"] is True
    assert [(f["n0"], f["n1"], f["factor"]) for f in d["folds"]] == [
        (5000, 500, 10.0),
        (4000, 1000, 4.0),
    ]


def test_a_fold_without_a_target_row_has_no_factor():
    frame, takeoff, train, target = _inseparable({2: (5000, 0), 3: (4000, 1000)})
    with pytest.raises(ValueError):
        regime_weights(frame, takeoff, train, target, class_ratio=True)
