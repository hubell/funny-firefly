# SPDX-License-Identifier: GPL-3.0-only
"""``Booster`` and ``SeedAverage``, and what a row weight does to them.

No test file exercised ``Booster`` before this one: it is fitted in scripts
and read through reports, and the ladder's stored residuals were the only
check that it did what it says. That was enough while it took one shape of
call. It now takes a second, ``sample_weight``, and the property the whole
recency experiment rests on is that the first shape did not change: with
``None`` no weight argument reaches the backend, so an unweighted fit is bit
for bit the fit it was. That is asserted here on a synthetic frame and again,
on the real data and against the stored residuals, by
``scripts/preregistered.py``'s reproduction check.

The frame is synthetic and small. These tests are about the plumbing of the
weight argument and not about what a weighted model predicts, which is a
question for a leg and not for a unit test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.models import Booster, SeedAverage, booster_backend  # noqa: E402

pytestmark = pytest.mark.skipif(
    booster_backend() == "none",
    reason="neither lightgbm nor catboost is installed",
)

FEATURES = ("a", "b", "c")


def _frame(n: int = 240, seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
    """A frame with a signal in it, so two fits have something to disagree on."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    c = rng.integers(0, 4, size=n).astype(float)
    y = 600.0 + 90.0 * a - 40.0 * b + 25.0 * c + rng.normal(scale=15.0, size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c}), pd.Series(y)


def _params() -> dict:
    """Small and fixed: this is plumbing, not a model."""
    return {
        "objective": "regression",
        "n_estimators": 30,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "min_child_samples": 5,
        "random_state": 20260902,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": 1,
        "verbose": -1,
    }


def test_no_weight_is_the_fit_it_was():
    """``None`` twice gives identical predictions, to the bit."""
    X, y = _frame()
    first = Booster(FEATURES, _params()).fit(X, y).predict(X)
    second = Booster(FEATURES, _params()).fit(X, y, sample_weight=None).predict(X)
    assert np.array_equal(first.to_numpy(), second.to_numpy())


def test_a_weight_of_the_right_length_is_accepted():
    X, y = _frame()
    weight = np.full(len(X), 0.5)
    model = Booster(FEATURES, _params()).fit(X, y, sample_weight=weight)
    assert len(model.predict(X)) == len(X)


def test_a_weight_of_the_wrong_length_is_refused():
    """One row short, and one row long: both are a mistake, not a broadcast."""
    X, y = _frame()
    for n in (len(X) - 1, len(X) + 1):
        with pytest.raises(ValueError, match="sample_weight"):
            Booster(FEATURES, _params()).fit(X, y, sample_weight=np.ones(n))


def test_the_weight_array_is_not_kept_on_the_model():
    """It belongs to one fitting sample and to no later one."""
    X, y = _frame()
    weight = np.linspace(0.25, 1.0, len(X))
    model = Booster(FEATURES, _params()).fit(X, y, sample_weight=weight)
    held = [
        name
        for name, value in vars(model).items()
        if isinstance(value, np.ndarray) and value.shape == weight.shape
    ]
    assert held == []
    assert not hasattr(model, "sample_weight")


def test_a_weight_that_is_not_flat_moves_the_fit():
    """Otherwise the argument could be silently dropped and nothing would say so."""
    X, y = _frame()
    flat = Booster(FEATURES, _params()).fit(X, y).predict(X).to_numpy()
    tilted = np.where(np.arange(len(X)) < len(X) // 2, 1.0, 0.01)
    weighted = (
        Booster(FEATURES, _params()).fit(X, y, sample_weight=tilted).predict(X).to_numpy()
    )
    assert not np.array_equal(flat, weighted)


def test_seed_average_takes_the_same_argument():
    X, y = _frame()
    plain = SeedAverage(FEATURES, _params(), seeds=2).fit(X, y).predict(X)
    none = SeedAverage(FEATURES, _params(), seeds=2).fit(X, y, sample_weight=None)
    assert np.array_equal(plain.to_numpy(), none.predict(X).to_numpy())

    weight = np.linspace(0.25, 1.0, len(X))
    average = SeedAverage(FEATURES, _params(), seeds=2).fit(X, y, sample_weight=weight)
    assert len(average.models) == 2
    for member in average.models:
        assert not hasattr(member, "sample_weight")


def test_seed_average_refuses_a_wrong_length_through_its_members():
    X, y = _frame()
    with pytest.raises(ValueError, match="sample_weight"):
        SeedAverage(FEATURES, _params(), seeds=2).fit(
            X, y, sample_weight=np.ones(len(X) - 1)
        )
