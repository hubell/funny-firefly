# SPDX-License-Identifier: GPL-3.0-only
"""The ablation ladder, and the estimators each of its rungs uses.

The ladder is the decomposition of ``docs/modelling_notes.md`` read bottom-up.
Each rung adds one term, and the rung-to-rung difference is what the write-up
reports:

1. **airport mean** -- the null model the data review already measured.
2. **PRU reference** -- the organisers' own empirical unimpeded time, the
   10th percentile per (stand, runway), with its cascade for pairs it has not
   seen often enough.
3. **geometric unimpeded** -- the same quantity from route length and shape,
   alone, with no learned model above it. It is on the ladder as a predictor
   in its own right rather than only as a feature, because a physical term
   that cannot beat an airport mean unaided is not a physical term.
4. **identifiers and time** -- a gradient-boosted model on the airport, stand
   and runway as categoricals plus the take-off clock and the flight plan's
   delay signal. This is the saturated baseline: on a pair it has seen, it
   already contains everything the geometry could tell it.
5. **identifiers, time and geometry** -- the same model with the matrix block
   and its quality flags added. Rung 4 to rung 5 is the claim, and it is only
   readable on the held-out-pair split, where rung 4's advantage of having
   memorised the pair is taken away.
6. **plus the aircraft** -- wake class and type as categoricals. On a seen
   pair the stand absorbs the class, because heavies park at wide-body
   stands; on a new pair nothing does, so this too is read on the held-out
   split. The geometric rung has a twin for the same reason: one intercept
   per airport, then one per (airport, wake class).
7. **plus the excess term, one block at a time** -- runway throughput,
   arrivals, scheduled demand, runway configuration, and the shared corridor
   (Gate 2). Each is its own rung so the report shows what each is worth,
   and each is built from what the ranking set exposes about surrounding
   movements: never a departure's block time or taxi time. The corridor sits
   last, after throughput, so that its rung says whether it is a new column
   or the take-off count renamed.
8. **plus weather** -- the regime flags per airport-hour (Gate 2b), the one
   block from outside the challenge data. Skipped by the runner when the
   archive is not cached, and said so, rather than fitted on a missing column.

Each rung is scored twice, once plain and once with the record-quality term on
top. That term lives in ``taxiout.tails`` and is deliberately not reimplemented
here: it is not a model of taxiing but the conditional mean of a cell that
mixes ordinary departures with defaulted block times, which is the
RMSE-optimal prediction under the stated scoring rule and nothing else
(*Tail handling*). It is applied by ``taxiout.tails.combine`` as an overlay on
the rows it covers rather than fitted jointly, so the physical model
underneath stays legible and the two can never be confused for one another.
Every rung here is fitted with those rows excluded and evaluated with them in.

**No submission-driven tuning.** The booster's hyperparameters are written out
below and were chosen once, for a compute budget, before any RMSE existed.
There is no early stopping on the validation set, because stopping on the set
you report is tuning on it; the number of rounds is fixed. Nothing here has
been or will be chosen by a leaderboard score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from taxiout.congestion import (
    ARRIVAL_FEATURES,
    CONFIGURATION_FEATURES,
    COVERAGE_FEATURES,
    DEMAND_FEATURES,
    THROUGHPUT_FEATURES,
)
from taxiout.corridor import CORRIDOR_FEATURES
from taxiout.metar import WEATHER_FEATURES
from taxiout.features import (
    AIRCRAFT_FEATURES,
    CATEGORICAL_FEATURES,
    GEOMETRY_FEATURES,
    PLAN_FEATURES,
    QUALITY_FEATURES,
    TIME_FEATURES,
)

# The identifier block the ladder's saturated rung uses. ``pair`` is
# deliberately not here: as a categorical it has tens of thousands of levels,
# which no boosted tree splits usefully, and the airport-stand-runway triple
# recovers the interaction through successive splits. It stays in the frame as
# the key the PRU reference and the held-out-pair split are built on.
IDENTIFIER_BLOCK: tuple[str, ...] = ("icao", "stand", "designator")

# Fixed once, for a compute budget, before any RMSE existed. L2 because the
# metric is RMSE. No early stopping: see the module docstring.
BOOSTER_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 500,
    "learning_rate": 0.06,
    "num_leaves": 63,
    "min_child_samples": 200,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "subsample_freq": 1,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "random_state": 20260902,
    "n_jobs": -1,
    "verbose": -1,
}


@dataclass(frozen=True)
class Rung:
    """One step of the ladder."""

    key: str
    label: str
    kind: str  # "airport-mean" | "pru" | "geometric" | "booster"
    features: tuple[str, ...] = ()
    note: str = ""
    # Keyword options for the rung's fitter, as pairs so the dataclass stays
    # frozen and hashable. The geometric rungs use it to say which class
    # column, if any, the intercept reads.
    options: tuple[tuple[str, object], ...] = ()

    @property
    def uses_geometry(self) -> bool:
        return self.kind == "geometric" or bool(
            set(self.features) & set(GEOMETRY_FEATURES)
        )


LADDER: tuple[Rung, ...] = (
    Rung(
        key="airport-mean",
        label="airport mean",
        kind="airport-mean",
        note="the null model; one number per airport",
    ),
    Rung(
        key="pru",
        label="PRU reference",
        kind="pru",
        note=(
            "10th percentile of taxi-out per (airport, stand, runway), "
            "cascading to (airport, runway) then airport below the minimum count"
        ),
    ),
    Rung(
        key="geometric",
        label="geometric unimpeded",
        kind="geometric",
        options=(("class_column", None),),
        note=(
            "an intercept per airport plus taxi_m over a per-airport speed, "
            "modulated by route shape; no learned model above it"
        ),
    ),
    Rung(
        key="geometric-class",
        label="geometric unimpeded + wake class",
        kind="geometric",
        options=(("class_column", "wake_class"),),
        note="the same, with the intercept fitted per (airport, wake class)",
    ),
    Rung(
        key="identifiers",
        label="identifiers + time",
        kind="booster",
        features=IDENTIFIER_BLOCK + TIME_FEATURES + PLAN_FEATURES,
        note="the saturated baseline: airport, stand and runway as categoricals",
    ),
    Rung(
        key="identifiers-geometry",
        label="identifiers + time + geometry",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES,
        note="the same model with the matrix block and its join-quality flags",
    ),
    Rung(
        key="identifiers-geometry-aircraft",
        label="identifiers + time + geometry + aircraft",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES,
        note="the same model with the wake class and the aircraft type",
    ),
    # Gate 2: the excess term, one block at a time, each on top of the last.
    # The coverage flags ride with the first block that needs them.
    Rung(
        key="throughput",
        label="+ runway throughput",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES,
        note="take-offs from the subject's runway and from all runways, trailing",
    ),
    Rung(
        key="arrivals",
        label="+ arrivals",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES,
        note="landings, on-blocks, and arrivals' taxi-in times, trailing",
    ),
    Rung(
        key="demand",
        label="+ demand",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES,
        note="scheduled departures around the subject's schedule",
    ),
    Rung(
        key="configuration",
        label="+ configuration",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES
        + CONFIGURATION_FEATURES,
        note="the runways in use in the trailing hour, and how long they have held",
    ),
    Rung(
        key="corridor",
        label="+ shared corridor",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES
        + CONFIGURATION_FEATURES
        + CORRIDOR_FEATURES,
        note=(
            "other movements in the window weighted by how much of the subject's "
            "own taxiway route theirs shares"
        ),
    ),
    Rung(
        key="weather",
        label="+ weather",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + THROUGHPUT_FEATURES
        + COVERAGE_FEATURES
        + ARRIVAL_FEATURES
        + DEMAND_FEATURES
        + CONFIGURATION_FEATURES
        + CORRIDOR_FEATURES
        + WEATHER_FEATURES,
        note="visibility band, precipitation and freezing, per airport-hour (Gate 2b)",
    ),
)

LADDER_BY_KEY: dict[str, Rung] = {r.key: r for r in LADDER}


class AirportMean:
    """Mean taxi-out per airport, with the pooled mean for an unseen airport."""

    def __init__(self) -> None:
        self.by_airport: pd.Series = pd.Series(dtype=float)
        self.overall: float = float("nan")

    def fit(self, icao: pd.Series, y: pd.Series) -> "AirportMean":
        t = pd.to_numeric(y, errors="coerce")
        self.by_airport = t.groupby(icao.astype(str), sort=True).mean()
        self.overall = float(t.mean())
        return self

    def predict(self, icao: pd.Series) -> pd.Series:
        return icao.astype(str).map(self.by_airport).fillna(self.overall).astype(float)


def booster_backend() -> str:
    """``"lightgbm"``, ``"catboost"`` or ``"none"``, whichever is installed."""
    try:
        import lightgbm  # noqa: F401

        return "lightgbm"
    except ImportError:
        pass
    try:
        import catboost  # noqa: F401

        return "catboost"
    except ImportError:
        return "none"


class Booster:
    """Gradient-boosted trees over a named feature block.

    LightGBM by preference, CatBoost as the fallback, with the same fixed
    parameters translated across. Categoricals are passed as categoricals
    rather than expanded: for a tree model that is what a one-hot buys, at a
    fraction of the memory.
    """

    def __init__(self, features: tuple[str, ...], params: dict | None = None) -> None:
        self.features = tuple(features)
        self.params = dict(BOOSTER_PARAMS if params is None else params)
        self.backend = booster_backend()
        self.model = None
        self.categorical = tuple(c for c in self.features if c in CATEGORICAL_FEATURES)
        self._levels: dict[str, pd.Index | None] = {}

    def _design(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[list(self.features)].copy()
        for c in self.categorical:
            out[c] = out[c].astype("category")
        return out

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Booster":
        design = self._design(X)
        if self.backend == "lightgbm":
            import lightgbm as lgb

            self.model = lgb.LGBMRegressor(**self.params)
            self.model.fit(
                design,
                np.asarray(y, dtype=float),
                categorical_feature=list(self.categorical),
            )
        elif self.backend == "catboost":
            from catboost import CatBoostRegressor

            cb = CatBoostRegressor(
                iterations=int(self.params["n_estimators"]),
                learning_rate=float(self.params["learning_rate"]),
                depth=8,
                loss_function="RMSE",
                random_seed=int(self.params["random_state"]),
                verbose=False,
            )
            for c in self.categorical:
                design[c] = design[c].astype(str)
            cb.fit(
                design, np.asarray(y, dtype=float), cat_features=list(self.categorical)
            )
            self.model = cb
        else:
            raise RuntimeError(
                "neither lightgbm nor catboost is installed; "
                'pip install -e ".[modelling]"'
            )
        # The categories seen in training, so prediction cannot silently
        # re-code a level to a different integer on a differently-shaped frame.
        self._levels = {
            c: design[c].cat.categories if hasattr(design[c], "cat") else None
            for c in self.categorical
        }
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.model is None:
            raise RuntimeError("predict before fit")
        design = X[list(self.features)].copy()
        for c in self.categorical:
            levels = self._levels.get(c)
            if self.backend == "catboost":
                design[c] = design[c].astype(str)
            elif levels is not None:
                design[c] = pd.Categorical(design[c].astype(str), categories=levels)
            else:
                design[c] = design[c].astype("category")
        return pd.Series(self.model.predict(design), index=X.index, dtype=float)

    def importances(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        raw = getattr(self.model, "feature_importances_", None)
        if raw is None:
            return pd.Series(dtype=float)
        return pd.Series(raw, index=list(self.features)).sort_values(ascending=False)


__all__ = [
    "BOOSTER_PARAMS",
    "IDENTIFIER_BLOCK",
    "LADDER",
    "LADDER_BY_KEY",
    "AirportMean",
    "Booster",
    "Rung",
    "booster_backend",
]
