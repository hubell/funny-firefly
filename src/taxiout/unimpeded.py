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
term is fitted on the unflagged lower tail of training rows -- the rows at or
below their own pair's reference, which are the rows the reference is built
from -- with each row's *target* being its pair's PRU value. One least
squares gives the intercept per (airport, wake class), the speed per airport
and the shape modulators that best reproduce the reference where it exists,
weighted by how often each pair was observed. The intercept is there because
``reports/geometry_gaps.md`` found one: a fixed pushback-to-line-up component
of several hundred seconds that no distance explains and that differs by
wake class at the same stand and runway. How well the fit manages is not
assumed: ``calibration`` measures it per airport on the seen pairs, and
``reports/baseline.md`` prints the result. If the geometry cannot reproduce
the reference on pairs where the reference is known, it has no business
extending it to pairs where it is not, and the report says so.

The flagged rows are excluded from the fit and not from the evaluation. A
record-matching artefact is not an aircraft taxiing slowly (*Tail handling* in
``docs/modelling_notes.md``); fitting a physical constant on one would be
fitting a physical constant on a clerical error. Flags come from
``taxiout.tails`` and arrive here as a boolean mask, so this module has no
opinion about how a row gets flagged.
"""

from __future__ import annotations

from collections import Counter
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

# The modulators may move an airport's inverse speed by at most this factor
# either way. A guard, not a fitted quantity: a cell with a defective
# descriptor should degrade to the plain speed model, not to an absurdity.
MAX_SPEED_FACTOR = 2.0

# The wake-turbulence class the per-class intercept reads, where the frame
# carries it; ``None`` fits one intercept per airport. ``ALL_CLASSES`` is the
# label every row gets in that case, ``NO_CLASS`` the label of a null class
# -- the unjoined flight-side block, which is a class of its own -- and
# ``OTHER_CLASS`` the pool for classes too thin to carry their own intercept.
CLASS_COLUMN = "wake_class"
ALL_CLASSES = "all"
NO_CLASS = "none"
OTHER_CLASS = "other"

# Below this many lower-tail rows an (airport, class) intercept is pooled
# into the airport's ``OTHER_CLASS``. The PRU floor for a pair, reused.
MIN_CLASS_ROWS = 30

# Below this many distinct pairs among its fit rows an airport gets the pooled
# intercept and speed rather than its own. Two pairs identify a line; five is
# the smallest number at which a line through the fastest deciles of real
# pairs is a fit rather than a construction. Every real airport has hundreds;
# this exists so that a subsample cannot hand an airport one pair and a
# collinear design, which is what a 2% development run did.
MIN_AIRPORT_PAIRS = 5

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
    """An intercept per airport and wake class, plus distance over a speed.

        unimpeded = a[airport, class] + taxi_m * clip(s[airport] + gamma . z)

    ``a`` is seconds: the fixed pushback, start-up and line-up component that
    no distance explains. ``reports/geometry_gaps.md`` measured it at 313 to
    740 s per airport on the training year, and found it differs by wake
    class by 35 to 281 s at the same stand and runway -- a heavy pushes,
    starts and lines up more slowly than a medium, whatever the route. A
    speed fitted through the origin folds all of that into the slope, which
    is why the Gate 1 speeds came out at 3 to 6 m/s: below any aircraft's
    taxi speed, because they were carrying the pushback.

    ``s`` is the inverse speed per airport, seconds per metre. ``gamma`` are
    the modulators, pooled across airports and in the same units, on
    standardised route descriptors; the clip keeps the modulated inverse
    speed within a factor of ``MAX_SPEED_FACTOR`` of the airport's, so a
    cell with a defective descriptor degrades to the plain speed model.

    The whole thing is one least squares per fit, so every printed parameter
    is the fitted one and not a stage of it. A (airport, class) intercept
    rests on at least ``MIN_CLASS_ROWS`` lower-tail rows; a class with fewer
    is pooled into the airport's ``OTHER_CLASS``. Ten airports, a handful of
    classes and five modulators: too few parameters to memorise a pair,
    which is the property that makes the held-out-pair split meaningful.
    """

    inverse_speed: dict[str, float]
    default_inverse_speed: float
    class_intercept: dict[tuple[str, str], float] = field(default_factory=dict)
    airport_intercept: dict[str, float] = field(default_factory=dict)
    default_intercept: float = 0.0
    coef: dict[str, float] = field(default_factory=dict)
    centre: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    class_column: str | None = CLASS_COLUMN
    pooled_airports: tuple[str, ...] = ()
    n_fit: int = 0

    @property
    def speed(self) -> dict[str, float]:
        """Metres per second per airport."""
        return {k: 1.0 / v for k, v in self.inverse_speed.items()}

    @property
    def default_speed(self) -> float:
        return 1.0 / self.default_inverse_speed

    def classes(self, features: pd.DataFrame) -> np.ndarray:
        """The class label per row, or ``ALL_CLASSES`` where none is carried."""
        return _classes(features, self.class_column)

    def intercepts(self, features: pd.DataFrame) -> np.ndarray:
        """The additive term per row, cascading class -> other -> airport -> all."""
        icao = features["icao"].astype(str).to_numpy()
        cls = self.classes(features)
        out = np.empty(len(features))
        for i, (a, c) in enumerate(zip(icao, cls)):
            v = self.class_intercept.get((a, c))
            if v is None:
                v = self.class_intercept.get((a, OTHER_CLASS))
            if v is None:
                v = self.airport_intercept.get(a, self.default_intercept)
            out[i] = v
        return out

    def _modulation(self, features: pd.DataFrame) -> np.ndarray:
        z = np.zeros(len(features))
        for c, g in self.coef.items():
            v = pd.to_numeric(features[c], errors="coerce").to_numpy(dtype=float)
            v = (v - self.centre[c]) / self.scale[c]
            z += g * np.nan_to_num(v, nan=0.0)
        return z

    def predict(self, features: pd.DataFrame) -> pd.Series:
        d = pd.to_numeric(features["taxi_m"], errors="coerce").to_numpy(dtype=float)
        s = (
            features["icao"]
            .astype(str)
            .map(self.inverse_speed)
            .fillna(self.default_inverse_speed)
            .to_numpy(dtype=float)
        )
        inv = np.clip(
            s + self._modulation(features), s / MAX_SPEED_FACTOR, s * MAX_SPEED_FACTOR
        )
        out = self.intercepts(features) + d * inv
        out = np.where(np.isfinite(out), out, np.nan)
        return pd.Series(np.maximum(out, MIN_UNIMPEDED_S), index=features.index)


def _classes(features: pd.DataFrame, class_column: str | None) -> np.ndarray:
    if class_column is None or class_column not in features.columns:
        return np.full(len(features), ALL_CLASSES, dtype=object)
    col = features[class_column]
    return np.where(col.isna(), NO_CLASS, col.astype(str)).astype(object)


def fit_geometric(
    features: pd.DataFrame,
    reference: PruReference,
    fit_rows: pd.Series,
    *,
    modulators: tuple[str, ...] = MODULATORS,
    class_column: str | None = CLASS_COLUMN,
    min_class_rows: int = MIN_CLASS_ROWS,
    min_airport_pairs: int = MIN_AIRPORT_PAIRS,
) -> GeometricUnimpeded:
    """Fit the intercepts, the speeds and the modulators to reproduce ``reference``.

    ``fit_rows`` is the unflagged lower tail. Rows are further restricted to
    those whose geometry is a real matrix cell (``geom_imputed == 0``) and
    whose pair has a reference of its own -- an airport-level fallback target
    would fit the speed to an airport mean, which is the rung below.

    One ordinary least squares over every fit row, of the reference on: a
    dummy per (airport, class) group, ``taxi_m`` per airport, and ``taxi_m``
    times each standardised descriptor, pooled. The modulators multiply
    distance rather than sit in a log residual so that the fit is a single
    linear problem with the stated objective, and a planted parameter is
    recovered exactly. A class with fewer than ``min_class_rows`` fit rows at
    an airport is pooled into that airport's ``OTHER_CLASS`` group.

    An airport takes the pooled intercept and speed instead of its own, and
    is listed in ``pooled_airports``, in two cases: fewer than
    ``min_airport_pairs`` distinct pairs among its fit rows, which cannot
    identify a line; or a fitted slope that is not positive, which would
    predict a taxi that gets shorter with distance. The second is found by
    fitting, so the fit repeats without that airport's rows until every slope
    that remains is positive -- an airport whose slope fails must not leave
    its intercept behind, and must not have shaped the shared modulators.
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
    cls = _classes(features, class_column)
    pairs = pair_key(features).to_numpy()
    cols = [c for c in modulators if c in features.columns]
    raw = {
        c: pd.to_numeric(features[c], errors="coerce").to_numpy(dtype=float) for c in cols
    }

    # The pooled line comes from every fit row, the pooled airports included;
    # it is the answer for them and for any airport the fit never saw.
    pooled, *_ = np.linalg.lstsq(
        np.column_stack([np.ones(int(rows.sum())), d[rows]]), r[rows], rcond=None
    )
    if pooled[1] > 0:
        default_inverse, default_intercept = float(pooled[1]), float(pooled[0])
    else:
        default_inverse, default_intercept = (
            float(np.mean(r[rows]) / np.mean(d[rows])),
            0.0,
        )

    per_airport = {a: len(set(pairs[rows & (icao == a)])) for a in np.unique(icao[rows])}
    pooled_airports = {a for a, k in per_airport.items() if k < min_airport_pairs}

    while True:
        own = rows & ~np.isin(icao, sorted(pooled_airports))
        if not own.any():
            return GeometricUnimpeded(
                inverse_speed={},
                default_inverse_speed=default_inverse,
                default_intercept=default_intercept,
                class_column=class_column,
                pooled_airports=tuple(sorted(pooled_airports)),
                n_fit=int(rows.sum()),
            )
        counts = Counter(zip(icao[own], cls[own]))
        group = [
            (a, c if counts.get((a, c), 0) >= min_class_rows else OTHER_CLASS)
            for a, c in zip(icao[own], cls[own])
        ]
        groups = sorted(set(group))
        airports = sorted(set(icao[own]))
        n = int(own.sum())
        dr = d[own]
        g_index = {g: i for i, g in enumerate(groups)}
        a_index = {a: len(groups) + i for i, a in enumerate(airports)}
        design = np.zeros((n, len(groups) + len(airports) + len(cols)))
        for i, (a, g) in enumerate(zip(icao[own], group)):
            design[i, g_index[g]] = 1.0
            design[i, a_index[a]] = dr[i]
        centre: dict[str, float] = {}
        scale: dict[str, float] = {}
        for j, c in enumerate(cols):
            x = raw[c][own]
            mu, sd = float(np.nanmean(x)), float(np.nanstd(x))
            centre[c], scale[c] = mu, sd if sd > 0 else 1.0
            design[:, len(groups) + len(airports) + j] = dr * np.nan_to_num(
                (x - mu) / scale[c], nan=0.0
            )
        beta, *_ = np.linalg.lstsq(design, r[own], rcond=None)
        inverse = {a: float(beta[i]) for a, i in a_index.items()}
        failed = {a for a, s in inverse.items() if not s > 0}
        if not failed:
            break
        pooled_airports |= failed

    intercept = {g: float(beta[i]) for g, i in g_index.items()}
    weight = Counter(group)
    airport_intercept = {
        a: float(
            sum(intercept[g] * weight[g] for g in groups if g[0] == a)
            / sum(weight[g] for g in groups if g[0] == a)
        )
        for a in airports
    }
    return GeometricUnimpeded(
        inverse_speed=inverse,
        default_inverse_speed=default_inverse,
        class_intercept=intercept,
        airport_intercept=airport_intercept,
        default_intercept=default_intercept,
        coef={
            c: float(beta[len(groups) + len(airports) + j]) for j, c in enumerate(cols)
        },
        centre=centre,
        scale=scale,
        class_column=class_column,
        pooled_airports=tuple(sorted(pooled_airports)),
        n_fit=n,
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
