# SPDX-License-Identifier: GPL-3.0-only
"""C4's columns, ``taxiout.flight_history``, on a constructed frame."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

from taxiout.encoding import INNER_FOLDS, SEED, _inner_folds  # noqa: E402
from taxiout.features import FITTED_COLUMNS, source_phase  # noqa: E402
from taxiout.flight_history import (  # noqa: E402
    SOURCES,
    airport_prior,
    fit_flight_history,
    flight_history,
)


def _frame(n=2000, seed=3):
    rng = np.random.default_rng(seed)
    icao = rng.choice(["EDDF", "LSZH"], n)
    flight = rng.choice(["LH1", "LH2", "LX3", "LX4", None], n)
    y = (
        600
        + 60 * (pd.Series(flight).fillna("").str.len().to_numpy())
        + rng.normal(0, 90, n)
    )
    features = pd.DataFrame({"icao": icao, "taxi_m": rng.uniform(800, 4000, n)})
    fit = pd.Series(rng.random(n) < 0.7)
    return features, pd.Series(flight, dtype=object), pd.Series(y), fit


def test_a_row_without_a_flight_number_takes_the_prior():
    features, flight, y, fit = _frame()
    got = flight_history(features, flight, y, fit)
    prior = airport_prior(features["icao"], y, fit)
    none = flight.isna().to_numpy()
    np.testing.assert_allclose(got["fn_mean"].to_numpy()[none], prior.to_numpy()[none])
    assert (got["fn_n"].to_numpy()[none] == 0).all()


def test_a_fitting_rows_own_target_is_not_in_its_own_group_mean():
    """The row's own target reaches its value only through the airport prior.

    Section 3 fixes the prior as the airport's mean over the fitting rows, so a
    fitting row's own target is in it, at one part in the airport's rows. The
    group's share is out of fold: shifting every residual by the prior's move
    moves the value by (1 - n / (n + k)) of that move, never more.
    """
    features, flight, y, fit = _frame()
    before = flight_history(features, flight, y, fit)
    victim = int(np.flatnonzero(fit.to_numpy() & flight.notna().to_numpy())[0])
    poisoned = y.copy()
    poisoned.iloc[victim] += 3600.0
    after = flight_history(features, flight, poisoned, fit)
    moved = abs(
        airport_prior(features["icao"], poisoned, fit).iloc[victim]
        - airport_prior(features["icao"], y, fit).iloc[victim]
    )
    assert 0 < moved < 3600.0 / 50
    assert (
        abs(after["fn_mean"].iloc[victim] - before["fn_mean"].iloc[victim])
        <= moved + 1e-9
    )


def test_rows_outside_the_fitting_sample_read_no_target_of_their_own():
    features, flight, y, fit = _frame()
    before = flight_history(features, flight, y, fit)
    poisoned = y.copy()
    poisoned[~fit] += 7200.0
    after = flight_history(features, flight, poisoned, fit)
    pd.testing.assert_frame_equal(before, after)


def test_the_count_is_out_of_fold_for_fitting_rows():
    features, flight, y, fit = _frame()
    got = flight_history(features, flight, y, fit)
    has = flight.notna().to_numpy()
    key = features["icao"].astype(str) + "|" + flight.astype(str)
    reads = fit.to_numpy() & has
    folds = np.zeros(len(y), dtype=int)
    folds[reads] = _inner_folds(int(reads.sum()), INNER_FOLDS, SEED)
    for i in range(len(y)):
        if not has[i]:
            continue
        same = reads & (key == key.iloc[i]).to_numpy()
        if reads[i]:
            same &= folds != folds[i]
        assert got["fn_n"].iloc[i] == same.sum()


def test_the_fitted_lookup_matches_the_whole_sample():
    features, flight, y, fit = _frame()
    fitted = fit_flight_history(features, flight, y, fit)
    got = fitted.transform(features, flight)
    has = flight.notna().to_numpy()
    key = features["icao"].astype(str) + "|" + flight.astype(str)
    reads = fit.to_numpy() & has
    for i in np.flatnonzero(has)[:20]:
        assert got["fn_n"].iloc[i] == (reads & (key == key.iloc[i]).to_numpy()).sum()
    prior = airport_prior(features["icao"], y, fit).to_numpy()
    np.testing.assert_allclose(got["fn_mean"].to_numpy()[~has], prior[~has])


def test_the_sources_declare_the_fitted_target_and_exposed_columns():
    for name, sources in SOURCES.items():
        for src in sources:
            phase, column = source_phase(src)
            assert phase in (None, "TRAIN"), (name, src)
            if phase == "TRAIN":
                assert column in FITTED_COLUMNS
            else:
                assert column in ("ADEP_mvt", "FLIGHT_mvt")
