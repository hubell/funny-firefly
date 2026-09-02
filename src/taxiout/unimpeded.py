# SPDX-License-Identifier: GPL-3.0-only
"""The unimpeded term: the time a departure would take with nothing in its way.

Two estimates of the same quantity, and the whole point of the entry is the
relation between them.

**The PRU reference** is the empirical one, and it is the organisers' own. The
Performance Review Unit's additional-taxi-out-time indicator defines unimpeded
taxi-out as a low percentile of the observed distribution for a (stand,
runway) pair -- the 10th here -- over pairs observed often enough for a
percentile to mean anything. It is exact where it exists and undefined where
it does not, and "where it does not" is a large part of the problem: a year of
departures leaves thousands of pairs with a handful of observations or none.

**The geometric version** is the same quantity as a function of the route. A
per-airport speed turns ``taxi_m`` into seconds, and the route descriptors
modulate it -- turn angle, junctions, junction-free running, backtrack -- in
the sense of Ravizza et al., who find route shape nearly as predictive as
route length. It is defined for every pair the matrix can route, seen or not,
and that is its entire value: it extends the reference rather than competing
with it.

So the fit is stated as a reproduction, not as a regression on taxi-out. The
speed is fitted on the unflagged lower tail of training rows -- the rows at or
below their own pair's reference, which are the rows the reference is built
from -- with each row's *target* being its pair's PRU value. Least squares
then gives the speed whose ``taxi_m / speed`` best reproduces the reference
where both exist, weighted by how often each pair was observed. How well it
manages that is not assumed: ``calibration`` measures it per airport on the
seen pairs, and ``reports/baseline.md`` prints the result. If the geometry
cannot reproduce the reference on pairs where the reference is known, it has
no business extending it to pairs where it is not, and the report says so.

The flagged rows are excluded from the fit and not from the evaluation. A
record-matching artefact is not an aircraft taxiing slowly (*Tail handling* in
``docs/modelling_notes.md``); fitting a physical constant on one would be
fitting a physical constant on a clerical error. Flags come from
``taxiout.tails`` and arrive here as a boolean mask, so this module has no
opinion about how a row gets flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# The PRU indicator's own percentile. Not a tuning parameter: changing it would
# change which published quantity this reproduces.
PRU_PERCENTILE = 0.10

# Below this many observations a pair's 10th percentile is the minimum of a
# handful of draws rather than a percentile. Chosen from the shape of the
# problem, not from a validation score: at 30 observations the 10th percentile
# sits between the third and fourth order statistic, which is the smallest
# sample where it is an interpolation rather than an extremum.
MIN_PAIR_COUNT = 30

# The descriptors that modulate the speed. Route *shape*, normalised so that a
# 400 m taxi and a 4 km one are comparable -- the raw descriptors scale with
# length, which the distance term already carries.
MODULATORS: tuple[str, ...] = (
    "turn_deg_per_km",
    "junctions_per_km",
    "straight_frac",
    "backtrack_frac",
    "detour_ratio",
)

# The modulators may scale the distance-implied time by at most this factor
# either way. A guard, not a fitted quantity: a cell with a defective
# descriptor should degrade to the plain distance model, not to an absurdity.
MAX_LOG_ADJUSTMENT = float(np.log(2.0))

# No unimpeded taxi is shorter than this. The data has 388 training departures
# at or below zero seconds; they are records, not taxis.
MIN_UNIMPEDED_S = 30.0


def pair_key(frame: pd.DataFrame) -> pd.Series:
    """``icao|stand|designator`` as a string, the PRU pair identifier."""
    if "pair" in frame.columns:
        return frame["pair"].astype(str)
    return (
        frame["icao"].astype(str)
        + "|"
        + frame["stand"].astype(str)
        + "|"
        + frame["designator"].astype(str)
    )


def runway_key(frame: pd.DataFrame) -> pd.Series:
    """``icao|designator``: the coarser key the cascade falls back to."""
    return frame["icao"].astype(str) + "|" + frame["designator"].astype(str)


@dataclass
class PruReference:
    """10th percentile of taxi-out per (airport, stand, runway), with fallbacks.

    ``predict`` cascades: the pair's percentile where the pair was seen at
    least ``min_count`` times, then the (airport, runway) percentile, then the
    airport's. ``level`` says which rung answered, so a report can separate
    the reference proper from its fallbacks -- they are different claims and
    the ablation ladder treats them as one rung only because a submission has
    to predict every row.
    """

    pair: pd.Series
    runway: pd.Series
    airport: pd.Series
    counts: pd.Series
    min_count: int = MIN_PAIR_COUNT
    percentile: float = PRU_PERCENTILE
    overall: float = float("nan")

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        p = self.pair.reindex(pair_key(frame)).to_numpy()
        r = self.runway.reindex(runway_key(frame)).to_numpy()
        a = self.airport.reindex(frame["icao"].astype(str)).to_numpy()
        out = np.where(np.isnan(p), np.where(np.isnan(r), a, r), p)
        return pd.Series(
            np.where(np.isnan(out), self.overall, out), index=frame.index, dtype=float
        )

    def level(self, frame: pd.DataFrame) -> pd.Series:
        p = self.pair.reindex(pair_key(frame)).to_numpy()
        r = self.runway.reindex(runway_key(frame)).to_numpy()
        a = self.airport.reindex(frame["icao"].astype(str)).to_numpy()
        return pd.Series(
            np.select(
                [~np.isnan(p), ~np.isnan(r), ~np.isnan(a)],
                ["pair", "runway", "airport"],
                default="none",
            ),
            index=frame.index,
        )

    def pair_counts(self, frame: pd.DataFrame) -> pd.Series:
        """Training observations of each row's pair; 0 for a pair never seen."""
        return pd.Series(
            self.counts.reindex(pair_key(frame)).fillna(0).to_numpy(),
            index=frame.index,
            dtype=float,
        )


def fit_pru_reference(
    frame: pd.DataFrame,
    target: pd.Series,
    *,
    mask: pd.Series | None = None,
    min_count: int = MIN_PAIR_COUNT,
    percentile: float = PRU_PERCENTILE,
) -> PruReference:
    """Fit the reference on training rows.

    ``mask`` selects the rows to fit on -- the training side of a split, with
    flagged rows dropped where ``taxiout.tails`` is available to flag them.
    Everything the reference knows comes from those rows and nothing else.
    """
    sel = pd.Series(True, index=frame.index) if mask is None else mask.astype(bool)
    keys = pd.DataFrame(
        {
            "pair": pair_key(frame)[sel],
            "runway": runway_key(frame)[sel],
            "icao": frame["icao"].astype(str)[sel],
            "t": pd.to_numeric(target[sel], errors="coerce"),
        }
    ).dropna(subset=["t"])

    counts = keys.groupby("pair", sort=True)["t"].size()
    pair_q = keys.groupby("pair", sort=True)["t"].quantile(percentile)
    pair_q = pair_q[counts >= min_count]
    return PruReference(
        pair=pair_q,
        runway=keys.groupby("runway", sort=True)["t"].quantile(percentile),
        airport=keys.groupby("icao", sort=True)["t"].quantile(percentile),
        counts=counts,
        min_count=min_count,
        percentile=percentile,
        overall=float(keys["t"].quantile(percentile)) if len(keys) else float("nan"),
    )


def lower_tail(
    frame: pd.DataFrame,
    target: pd.Series,
    reference: PruReference,
    *,
    mask: pd.Series | None = None,
) -> pd.Series:
    """Rows at or below their own pair's reference: the unimpeded observations.

    This is the sample the speed is fitted on. It is a subset of the rows the
    reference was fitted on by construction, so no row enters the speed fit
    whose pair the reference has not seen.
    """
    sel = pd.Series(True, index=frame.index) if mask is None else mask.astype(bool)
    t = pd.to_numeric(target, errors="coerce")
    return sel & t.notna() & (t <= reference.predict(frame)) & (t > 0)


@dataclass
class GeometricUnimpeded:
    """``taxi_m / speed``, per airport, modulated by route shape.

    ``speed`` is metres per second per airport; ``default_speed`` answers for
    an airport not in the fit. ``coef`` are the modulator coefficients on
    standardised descriptors, in log space, so the prediction is

        unimpeded = (taxi_m / speed) * exp(clip(intercept + coef . z))

    with ``z`` the standardised descriptors and the clip at
    ``MAX_LOG_ADJUSTMENT``. Five pooled coefficients across ten airports and
    thousands of pairs: too few to memorise a pair, which is the property that
    makes the held-out-pair split meaningful.
    """

    speed: dict[str, float]
    default_speed: float
    coef: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    centre: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    n_fit: int = 0

    def _standardised(self, features: pd.DataFrame) -> np.ndarray:
        cols = list(self.coef)
        if not cols:
            return np.zeros(len(features))
        z = np.zeros(len(features))
        for c in cols:
            v = pd.to_numeric(features[c], errors="coerce").to_numpy(dtype=float)
            v = (v - self.centre[c]) / self.scale[c]
            z += self.coef[c] * np.nan_to_num(v, nan=0.0)
        return z + self.intercept

    def predict(self, features: pd.DataFrame) -> pd.Series:
        d = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
        v = (
            features["icao"]
            .astype(str)
            .map(self.speed)
            .fillna(self.default_speed)
            .to_numpy(dtype=float)
        )
        adj = np.clip(
            self._standardised(features), -MAX_LOG_ADJUSTMENT, MAX_LOG_ADJUSTMENT
        )
        out = (d / v) * np.exp(adj)
        out = np.where(np.isfinite(out), out, np.nan)
        return pd.Series(np.maximum(out, MIN_UNIMPEDED_S), index=features.index)


def fit_geometric(
    features: pd.DataFrame,
    reference: PruReference,
    fit_rows: pd.Series,
    *,
    modulators: tuple[str, ...] = MODULATORS,
) -> GeometricUnimpeded:
    """Fit the per-airport speed and the modulators to reproduce ``reference``.

    ``fit_rows`` is the unflagged lower tail. Rows are further restricted to
    those whose geometry is a real matrix cell (``geom_imputed == 0``) and
    whose pair has a reference of its own -- an airport-level fallback target
    would fit the speed to an airport mean, which is the rung below.

    The speed is a least-squares fit of ``reference ~ taxi_m / speed``, which
    has the closed form ``speed = sum(d^2) / sum(d * r)``. The modulators are
    then an ordinary least squares on the log residual, pooled across airports.
    """
    rows = fit_rows.to_numpy(dtype=bool)
    rows &= reference.level(features).to_numpy() == "pair"
    rows &= pd.to_numeric(features["geom_imputed"], errors="coerce").to_numpy() == 0.0
    d = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
    r = reference.predict(features).to_numpy(dtype=float)
    rows &= np.isfinite(d) & (d > 0) & np.isfinite(r) & (r > 0)
    if not rows.any():
        raise ValueError("no rows to fit the unimpeded speed on")

    icao = features["icao"].astype(str).to_numpy()
    speed: dict[str, float] = {}
    for ap in np.unique(icao[rows]):
        m = rows & (icao == ap)
        num, den = float(np.sum(d[m] ** 2)), float(np.sum(d[m] * r[m]))
        if den > 0:
            speed[ap] = num / den
    default = float(np.sum(d[rows] ** 2) / np.sum(d[rows] * r[rows]))

    # Log residual of the plain distance model, then OLS on standardised
    # descriptors. Standardising is for interpretability of the printed
    # coefficients and for conditioning; it changes no prediction.
    v = np.array([speed.get(a, default) for a in icao[rows]])
    resid = np.log(r[rows]) - np.log(d[rows] / v)
    cols = [c for c in modulators if c in features.columns]
    centre: dict[str, float] = {}
    scale: dict[str, float] = {}
    design = [np.ones(int(rows.sum()))]
    for c in cols:
        x = pd.to_numeric(features[c], errors="coerce").to_numpy(dtype=float)[rows]
        mu = float(np.nanmean(x))
        sd = float(np.nanstd(x))
        centre[c], scale[c] = mu, sd if sd > 0 else 1.0
        design.append(np.nan_to_num((x - mu) / scale[c], nan=0.0))
    beta, *_ = np.linalg.lstsq(np.column_stack(design), resid, rcond=None)
    return GeometricUnimpeded(
        speed=speed,
        default_speed=default,
        coef={c: float(b) for c, b in zip(cols, beta[1:])},
        intercept=float(beta[0]),
        centre=centre,
        scale=scale,
        n_fit=int(rows.sum()),
    )


def calibration(
    features: pd.DataFrame,
    reference: PruReference,
    model: GeometricUnimpeded,
    *,
    weights: pd.Series | None = None,
) -> pd.DataFrame:
    """How well the geometric version recovers the PRU reference on seen pairs.

    One row per airport plus an ``all`` row, over the distinct pairs that have
    both a pair-level reference and a real matrix cell. Cell-weighted, so a
    busy pair does not outvote a quiet one, with a movement-weighted RMSE
    beside it because the submission is scored per movement.

    This is the calibration check the entry rests on. Both quantities are
    fitted on training rows only, so it says nothing about held-out
    performance -- it says whether the function the geometry defines can
    express the reference at all.
    """
    ok = (reference.level(features).to_numpy() == "pair") & (
        pd.to_numeric(features["geom_imputed"], errors="coerce").to_numpy() == 0.0
    )
    w = pd.Series(1.0, index=features.index) if weights is None else weights.astype(float)
    frame = pd.DataFrame(
        {
            "icao": features["icao"].astype(str),
            "pair": pair_key(features),
            "ref": reference.predict(features),
            "geo": model.predict(features),
            "w": w,
        }
    )[ok]
    frame = frame[np.isfinite(frame["ref"]) & np.isfinite(frame["geo"])]
    cells = (
        frame.groupby(["icao", "pair"], sort=True)
        .agg(ref=("ref", "first"), geo=("geo", "first"), w=("w", "sum"))
        .reset_index()
    )

    rows = []
    for label, g in list(cells.groupby("icao", sort=True)) + [("all", cells)]:
        err = g["geo"] - g["ref"]
        ratio = g["geo"] / g["ref"]
        rows.append(
            {
                "icao": label,
                "pairs": len(g),
                "ref_median_s": float(g["ref"].median()),
                "rmse_s": float(np.sqrt((err**2).mean())),
                "mae_s": float(err.abs().mean()),
                "bias_s": float(err.mean()),
                "ratio_p10": float(ratio.quantile(0.10)),
                "ratio_p50": float(ratio.quantile(0.50)),
                "ratio_p90": float(ratio.quantile(0.90)),
                "corr": (
                    float(np.corrcoef(g["geo"], g["ref"])[0, 1])
                    if len(g) > 2
                    else float("nan")
                ),
                "rmse_movement_weighted_s": float(
                    np.sqrt((g["w"] * err**2).sum() / g["w"].sum())
                ),
            }
        )
    return pd.DataFrame(rows)
