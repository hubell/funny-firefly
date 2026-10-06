# SPDX-License-Identifier: GPL-3.0-only
"""The flight's own history: a shrunk out-of-fold mean per airport and flight number.

Candidate C4 of ``reports/preregistered-2026-10-04.md``, section 3:

- ``fn_mean``: the shrunk out-of-fold mean recorded taxi-out per airport and
  ``FLIGHT_mvt``, over the fitting rows, in the fold scheme and shrinkage form
  ``taxiout.encoding`` uses for the pair, the prior being the airport's mean
  over the fitting rows;
- ``fn_n``: the rows behind it.

A row with no ``FLIGHT_mvt`` takes the prior.

**The same arithmetic, not a copy of it.** ``taxiout.encoding.encode_level``
is called as it is for the pair, on a shallow copy of the frame whose pair key
is replaced by the airport and the flight number and whose prior is the
airport's mean: the same five inner folds dealt over the fitting rows by the
same seed, the same per-airport ``k`` with its sampling correction and its
floors, the same out-of-fold sums. Rows without a flight number are kept out
of the rows it reads, so their group holds no row and they encode to the
prior exactly. ``fn_n`` is counted from the same folds: a fitting row's own
fold is left out of its count, as it is left out of its mean, and every other
row counts all the fitting rows of its group.

For rows outside the fitting sample (a ranking month), ``FittedFlightHistory``
looks the group up over the whole sample, as ``FittedEncoding`` does for the
pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxiout.encoding import (
    INNER_FOLDS,
    SEED,
    _group_moments,
    _inner_folds,
    encode_level,
    estimate_shrinkage,
)

FLIGHT = "FLIGHT_mvt"
FLIGHT_HISTORY_FEATURES: tuple[str, ...] = ("fn_mean", "fn_n")

SOURCES: dict[str, tuple[str, ...]] = {
    c: ("ADEP_mvt", FLIGHT, "TRAIN.TAXITIME_SEC_mvt") for c in FLIGHT_HISTORY_FEATURES
}


def flight_key(icao: pd.Series, flight: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """The group key, airport and flight number, and whether the row has one."""
    f = flight.astype(object)
    has = f.notna().to_numpy() & (f.astype(str).str.strip() != "").to_numpy()
    key = icao.astype(str) + "|" + f.astype(str).str.strip()
    key = key.where(has, icao.astype(str) + "|")
    return key, has


def airport_prior(icao: pd.Series, y: pd.Series, fit_rows: pd.Series) -> pd.Series:
    """The airport's mean recorded taxi-out over the fitting rows, for every row."""
    target = pd.to_numeric(y, errors="coerce")
    mask = fit_rows.to_numpy(dtype=bool) & np.isfinite(target.to_numpy(dtype=float))
    means = target[mask].groupby(icao[mask].astype(str)).mean()
    pooled = float(target[mask].mean())
    return icao.astype(str).map(means).astype(float).fillna(pooled)


def _frame(features: pd.DataFrame, key: pd.Series) -> pd.DataFrame:
    out = features[["icao", "taxi_m"]].copy(deep=False)
    out["pair"] = key.to_numpy()
    return out


def flight_history(
    features: pd.DataFrame,
    flight: pd.Series,
    y: pd.Series,
    fit_rows: pd.Series,
) -> pd.DataFrame:
    """``fn_mean`` and ``fn_n`` for every row of ``features``, one split's fitting rows.

    ``features`` carries ``icao`` and ``taxi_m`` (the route length the
    encoding's within-variance reads); ``flight`` is ``FLIGHT_mvt`` aligned
    with it; ``fit_rows`` the split's fitting rows, the only rows a target is
    read from.
    """
    key, has = flight_key(features["icao"], flight)
    prior = airport_prior(features["icao"], y, fit_rows)
    reads = pd.Series(fit_rows.to_numpy(dtype=bool) & has, index=features.index)
    encoded, _ = encode_level(_frame(features, key), y, prior, reads, "pair")
    # The rows behind each value, from the very folds encode_level dealt.
    target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ghat = prior.to_numpy(dtype=float)
    fit = reads.to_numpy(dtype=bool) & np.isfinite(target) & np.isfinite(ghat)
    codes, levels = pd.factorize(key, use_na_sentinel=False)
    folds = _inner_folds(int(fit.sum()), INNER_FOLDS, SEED)
    total = np.bincount(codes[fit], minlength=len(levels)).astype(float)
    cell = np.bincount(
        codes[fit] * INNER_FOLDS + folds, minlength=len(levels) * INNER_FOLDS
    ).reshape(len(levels), INNER_FOLDS)
    n = total[codes]
    own = np.zeros(len(codes), dtype=int)
    own[fit] = folds
    n = n - np.where(fit, cell[codes, own], 0.0)
    n = np.where(has, n, 0.0)
    out = pd.DataFrame(index=features.index)
    out["fn_mean"] = encoded.to_numpy(dtype=float)
    out["fn_n"] = n
    return out[list(FLIGHT_HISTORY_FEATURES)]


@dataclass(frozen=True)
class FittedFlightHistory:
    """The whole fitting sample's groups, for rows outside it."""

    means: pd.Series
    pooled: float
    n: pd.Series
    sum_d: pd.Series
    shrinkage: object

    def transform(self, features: pd.DataFrame, flight: pd.Series) -> pd.DataFrame:
        key, has = flight_key(features["icao"], flight)
        icao = features["icao"].astype(str)
        prior = (
            icao.map(self.means).astype(float).fillna(self.pooled).to_numpy(dtype=float)
        )
        n = np.nan_to_num(key.map(self.n).to_numpy(dtype=float), nan=0.0)
        sum_d = np.nan_to_num(key.map(self.sum_d).to_numpy(dtype=float), nan=0.0)
        n = np.where(has, n, 0.0)
        length = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
        k = self.shrinkage.k_for(icao.to_numpy(), length)
        with np.errstate(divide="ignore", invalid="ignore"):
            shift = np.where(n > 0, n / (n + k) * (sum_d / np.where(n > 0, n, 1.0)), 0.0)
        out = pd.DataFrame(index=features.index)
        out["fn_mean"] = prior + shift
        out["fn_n"] = n
        return out[list(FLIGHT_HISTORY_FEATURES)]


def fit_flight_history(
    features: pd.DataFrame, flight: pd.Series, y: pd.Series, fit_rows: pd.Series
) -> FittedFlightHistory:
    """The lookup a ranking month reads, from every fitting row at once."""
    key, has = flight_key(features["icao"], flight)
    prior = airport_prior(features["icao"], y, fit_rows)
    target = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ghat = prior.to_numpy(dtype=float)
    fit = fit_rows.to_numpy(dtype=bool) & has & np.isfinite(target) & np.isfinite(ghat)
    codes, levels = pd.factorize(key, use_na_sentinel=False)
    length = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
    moments = _group_moments(
        codes[fit],
        len(levels),
        target[fit] - ghat[fit],
        np.nan_to_num(length[fit], nan=0.0),
    )
    icao = features["icao"].astype(str).to_numpy()
    airport_of_group = np.empty(len(levels), dtype=object)
    airport_of_group[:] = ""
    first = np.full(len(levels), len(codes))
    np.minimum.at(first, codes, np.arange(len(codes)))
    seen = first < len(codes)
    airport_of_group[seen] = icao[first[seen]]
    shrink = estimate_shrinkage("pair", airport_of_group, moments)
    target_s = pd.Series(target)
    mask = fit_rows.to_numpy(dtype=bool) & np.isfinite(target)
    means = target_s[mask].groupby(features["icao"].astype(str).to_numpy()[mask]).mean()
    return FittedFlightHistory(
        means=means,
        pooled=float(target_s[mask].mean()),
        n=pd.Series(moments["n"], index=levels),
        sum_d=pd.Series(moments["sum_d"], index=levels),
        shrinkage=shrink,
    )


__all__ = [
    "FLIGHT_HISTORY_FEATURES",
    "FittedFlightHistory",
    "SOURCES",
    "airport_prior",
    "fit_flight_history",
    "flight_history",
]
