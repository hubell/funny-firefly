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
7. **plus the plan lags** -- take-off minus a planning value fixed before
   departure (the first and last off-block estimates and the schedule), so
   taxi plus the pushback delay against the plan; the subject's own and the
   mean over the other departures on its runway within fifteen minutes. It
   sits before the congestion blocks so that what they are worth is read net
   of it, and the congestion rungs are kept without it as a second chain at
   the foot of the ladder, so the reader sees both.
8. **plus the excess term, one block at a time** -- runway throughput,
   arrivals, scheduled demand, runway configuration, and the shared corridor
   (Gate 2). Each is its own rung so the report shows what each is worth,
   and each is built from what the ranking set exposes about surrounding
   movements: never a departure's block time or taxi time. The corridor sits
   last, after throughput, so that its rung says whether it is a new column
   or the take-off count renamed.
9. **plus weather** -- the regime flags per airport-hour (Gate 2b), the one
   block from outside the challenge data. Skipped by the runner when the
   archive is not cached, and said so, rather than fitted on a missing column.
10. **plus the shrunk identifier encoding**, in three arms read against the
   weather rung rather than against each other. The pair and the stand as
   empirical-Bayes encodings shrunk toward the geometric unimpeded prediction
   (``taxiout.encoding``): added alongside the identifier block, replacing the
   stand within it, and replacing both the stand and the runway. The control
   is the weather rung itself, unchanged. The second and third arms exist
   because removing the model's capacity to memorise a thin pair may
   generalise better than giving it a smoothed version of the same thing, and
   because "replacing the pair categorical" has two readings in a frame where
   ``pair`` is not itself a feature -- so both are run rather than one being
   guessed at.
11. **plus runway saturation** -- the cheap Gate 3: from the other departures'
   take-off times on the subject's runway, the share of consecutive gaps
   under ninety seconds, the smallest gap, and the span of the last three.
   Measured against the ceiling in ``reports/congestion.md``, found to be
   worth nothing, and therefore **not** in the submission: ``TOP_RUNG`` is
   the weather rung below it. The rung stays on the ladder because a cheap
   queue proxy that was built, measured and bought nothing is a result.

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
below. They were set once, for a compute budget, before any RMSE existed, and
then chosen once more by the one permitted sweep: fitted on January to
September 2025 and scored on October 2025, a month none of the ladder's
splits reads, over one grid, with the October number recorded beside them.
There is no early stopping on the validation set, because stopping on the set
you report is tuning on it; the number of rounds is fixed. Nothing here has
been or will be chosen by a leaderboard score, and nothing was chosen on
December, July or the held-out pairs.
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
from taxiout.encoding import ENCODING_FEATURES
from taxiout.metar import WEATHER_FEATURES
from taxiout.planlags import PLAN_LAG_FEATURES
from taxiout.saturation import SATURATION_FEATURES
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

# What a model may keep at an airport it has never seen: everything in the top
# rung that is not an identifier. The airport, the stand and the runway all go
# -- at an unseen airport none of the three has a level the model has met --
# and what remains is fitted across the other airports and applied through the
# held-out airport's own geometry. ``scripts/report_transfer.py`` reads this.
TRANSFERABLE_EXCLUDED: tuple[str, ...] = ("icao", "stand", "designator", "pair")

# Fixed once, for a compute budget, before any RMSE existed. L2 because the
# metric is RMSE. No early stopping: see the module docstring.
# Chosen once by the one permitted sweep (scripts/sweep_params.py): fit
# January to September 2025, score October 2025 -- a month the ladder never
# reads -- over a grid of trees (300, 600, 1000), learning rate (0.03, 0.06,
# 0.1), leaves (31, 63, 127) and minimum child samples (50, 200, 500), on
# the saturation rung's features -- the top rung when the sweep ran, before
# that block was dropped from the submission -- with the RMSE taken over the
# rows the record-quality term leaves to the physical model.
# ``scripts/sweep_params.py`` pins that rung by name so the numbers below stay
# reproducible now that ``TOP_RUNG`` has moved. The cell below scored 235.0 s on
# October; the grid ran from 235.0 to 244.8 s and its top eight cells
# lie within 0.8 s of each other. The parameters that stood before it
# (500 trees, 0.06, 63 leaves, 200) sit between the 238.2 s of their
# 300-tree and the 236.8 s of their 600-tree neighbours in the grid.
# The grid's numbers stay under data/processed/, gitignored, and are not
# reported anywhere else.
BOOSTER_PARAMS: dict[str, object] = {
    "objective": "regression",
    "n_estimators": 1000,
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_child_samples": 50,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "subsample_freq": 1,
    "max_cat_threshold": 64,
    "cat_smooth": 20.0,
    "random_state": 20260902,
    "n_jobs": -1,
    "verbose": -1,
}


# The top rung's feature block, named once so the rungs that build on it say
# so rather than repeating fifteen tuples and drifting from one another.
WEATHER_BLOCK: tuple[str, ...] = (
    IDENTIFIER_BLOCK
    + TIME_FEATURES
    + PLAN_FEATURES
    + GEOMETRY_FEATURES
    + QUALITY_FEATURES
    + AIRCRAFT_FEATURES
    + PLAN_LAG_FEATURES
    + THROUGHPUT_FEATURES
    + COVERAGE_FEATURES
    + ARRIVAL_FEATURES
    + DEMAND_FEATURES
    + CONFIGURATION_FEATURES
    + CORRIDOR_FEATURES
    + WEATHER_FEATURES
)


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
    # The rung this one is read against. ``None`` is the rung before it on
    # the ladder; a key names another, for the rungs kept as a second chain.
    base: str | None = None

    @property
    def uses_geometry(self) -> bool:
        """Whether this rung reads the geometry at all, however indirectly.

        ``ENCODING_FEATURES`` counts. The shrunk encodings are not in
        ``GEOMETRY_FEATURES`` -- they are their own block -- but the quantity
        they shrink toward is ``g_hat``, so a rung that carries them carries the
        geometry with an extra step in front of it. Classifying them as
        geometry-free would let an ablation call itself "geometry removed" while
        the geometry was still arriving through the prior, and the contrast it
        reported would shrink as an artefact of that definition rather than as a
        finding. The claim contrast is measured between two rungs that carry no
        encoding either way, so nothing currently printed moves; this keeps that
        true if the claim is ever re-anchored above the arms.
        """
        return self.kind == "geometric" or bool(
            set(self.features) & (set(GEOMETRY_FEATURES) | set(ENCODING_FEATURES))
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
    Rung(
        key="plan-lags",
        label="+ plan lags",
        kind="booster",
        features=IDENTIFIER_BLOCK
        + TIME_FEATURES
        + PLAN_FEATURES
        + GEOMETRY_FEATURES
        + QUALITY_FEATURES
        + AIRCRAFT_FEATURES
        + PLAN_LAG_FEATURES,
        note=(
            "take-off minus a planning value fixed before departure -- the first "
            "and last off-block estimates and the schedule -- so taxi plus the "
            "pushback delay against the plan; the subject's own and the mean over "
            "the other departures on its runway within fifteen minutes"
        ),
    ),
    # Gate 2: the excess term, one block at a time, each on top of the last,
    # read net of the plan lags. The coverage flags ride with the first block
    # that needs them.
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
        + PLAN_LAG_FEATURES
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
        + PLAN_LAG_FEATURES
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
        + PLAN_LAG_FEATURES
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
        + PLAN_LAG_FEATURES
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
        + PLAN_LAG_FEATURES
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
        features=WEATHER_BLOCK,
        note="visibility band, precipitation and freezing, per airport-hour (Gate 2b)",
    ),
    # The three encoding arms. Each is read against the weather rung and not
    # against the arm before it: they are alternatives, not steps, and reading
    # one against another would report the difference between two ways of
    # spending the same block as though it were a block of its own.
    Rung(
        key="encoding-added",
        label="+ shrunk encoding",
        kind="booster",
        features=WEATHER_BLOCK + ENCODING_FEATURES,
        base="weather",
        note=(
            "the pair and the stand as empirical-Bayes encodings shrunk toward "
            "the geometric unimpeded prediction, added alongside the identifier "
            "block"
        ),
    ),
    Rung(
        key="encoding-for-stand",
        label="+ shrunk encoding, replacing the stand",
        kind="booster",
        features=tuple(c for c in WEATHER_BLOCK if c != "stand") + ENCODING_FEATURES,
        base="weather",
        note=(
            "the same, with the stand categorical removed: the encodings replace "
            "the level the model would memorise a thin pair through, while the "
            "runway stays because it is an operational variable and not a way of "
            "identifying one cell"
        ),
    ),
    Rung(
        key="encoding-for-identifiers",
        label="+ shrunk encoding, replacing stand and runway",
        kind="booster",
        features=tuple(c for c in WEATHER_BLOCK if c not in ("stand", "designator"))
        + ENCODING_FEATURES,
        base="weather",
        note=(
            "the wider reading of the same arm: both halves of the pair removed, "
            "leaving the airport, so nothing but the encodings carries the cell"
        ),
    ),
    Rung(
        key="saturation",
        label="+ runway saturation",
        kind="booster",
        features=WEATHER_BLOCK + SATURATION_FEATURES,
        # Explicit, because the encoding arms now sit between this rung and the
        # one it was measured against. Without it this rung would silently be
        # re-read against an arm, and a measured negative would change meaning
        # without anybody editing the number.
        base="weather",
        note=(
            "the share of consecutive take-off gaps under ninety seconds, the "
            "smallest gap, and the span of the last three, from other departures' "
            "take-off times on the subject's runway (the cheap Gate 3)"
        ),
    ),
)

# The arms, by key: alternatives to one another rather than steps of the
# ladder, and excluded from the chain kept without the plan lags below.
ENCODING_ARMS: tuple[str, ...] = (
    "encoding-added",
    "encoding-for-stand",
    "encoding-for-identifiers",
)


def _without_plan_lags(rung: Rung, base: str) -> Rung:
    return Rung(
        key=f"{rung.key}-nolag",
        label=f"{rung.label} (without plan lags)",
        kind=rung.kind,
        features=tuple(c for c in rung.features if c not in PLAN_LAG_FEATURES),
        note=rung.note,
        options=rung.options,
        base=base,
    )


# The ladder proper, before the chain kept without the plan lags is appended.
LADDER_PROPER: tuple[Rung, ...] = LADDER

# The congestion and weather rungs kept without the plan lags, each read
# against the one before it in this chain and the first against the aircraft
# rung, so the report shows what each block is worth both gross and net. The
# encoding arms are excluded: they are alternatives rather than steps, and a
# chain through them would read one arm against another.
_NET_CHAIN = tuple(
    r
    for r in LADDER
    if PLAN_LAG_FEATURES[0] in r.features
    and r.key != "plan-lags"
    and r.key not in ENCODING_ARMS
)
_WITHOUT: list[Rung] = []
_base = "identifiers-geometry-aircraft"
for _r in _NET_CHAIN:
    _WITHOUT.append(_without_plan_lags(_r, _base))
    _base = _WITHOUT[-1].key
LADDER = LADDER + tuple(_WITHOUT)

LADDER_BY_KEY: dict[str, Rung] = {r.key: r for r in LADDER}

# The rung the submission is fitted from: the shrunk encoding replacing the
# stand, chosen by the rule in ``reports/baseline.md``'s encoding section.
#
# Two rungs of the ladder proper sit above it, for two different reasons.
#
# ``+ shrunk encoding, replacing stand and runway`` is the arm that goes one
# step further and replaces the designator too. It is out on a mechanism
# rather than on its score, which is indistinguishable from this arm's:
# ``reports/geometry_gaps.md`` measures an intersection signature of up to 96 s
# per designator that the routed distance cannot compute, so the runway
# categorical carries something no encoding shrunk toward that distance can.
#
# The saturation rung is the other case, and the older one: a measured
# negative that costs half a second in December and eight tenths in July,
# takes none of the ceiling, and belongs in the ladder and the write-up rather
# than in the model.
#
# Named here so that ``scripts/make_submission.py`` and the report's
# per-airport and importance sections all read the same rung. Moving it here
# is what puts the encoding into the submitted model: ``make_submission.py``
# computes the encoding once over the training year -- out of fold for the
# rows it fits on, by lookup for the rows it predicts -- as
# ``taxiout.encoding`` describes.
SUBMISSION_KEY = "encoding-for-stand"

# The rung the encoding arms were read against, and it is *not* whichever rung
# the submission happens to use. Before the arms shipped these were the same
# string, and writing the encoding section against ``SUBMISSION_KEY`` was
# harmless; the moment an arm displaced the control they came apart, and a
# section that took its control from the shipped rung would have compared the
# chosen arm against itself and reported a difference of zero. The control is a
# property of the experiment, fixed when it was designed, so it is named
# separately and does not move when the model does.
ENCODING_CONTROL = "weather"
TOP_RUNG: Rung = LADDER_BY_KEY[SUBMISSION_KEY]

# The rungs of the ladder proper above it: blocks that were built, measured,
# and left out of the model. The report measures each against the ceiling of
# the rung it was added to.
ABOVE_SUBMISSION: tuple[Rung, ...] = LADDER_PROPER[LADDER_PROPER.index(TOP_RUNG) + 1 :]

# The top rung with every identifier removed. Fitted across nine airports and
# applied to the tenth, this is what a model has left at an airport it has
# never seen: the geometry of that airport's own taxiways, the clock, the
# aircraft, the plan lags, the traffic around the movement and the weather --
# and no level of any categorical that names a place. The shrunk encodings are
# absent by the same rule: at an unseen airport every group has ``n = 0`` and
# the encoding is the geometric prior with an extra step in front of it.
TRANSFER_FEATURES: tuple[str, ...] = tuple(
    c
    for c in TOP_RUNG.features
    if c not in TRANSFERABLE_EXCLUDED and c not in ENCODING_FEATURES
)


def base_of(rung: Rung) -> Rung | None:
    """The rung ``rung`` is read against: its ``base``, else the one before it."""
    if rung.base is not None:
        return LADDER_BY_KEY[rung.base]
    i = LADDER.index(rung)
    return LADDER[i - 1] if i > 0 else None


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
    "ABOVE_SUBMISSION",
    "ENCODING_ARMS",
    "LADDER_PROPER",
    "ENCODING_CONTROL",
    "SUBMISSION_KEY",
    "TOP_RUNG",
    "TRANSFERABLE_EXCLUDED",
    "TRANSFER_FEATURES",
    "WEATHER_BLOCK",
    "base_of",
    "IDENTIFIER_BLOCK",
    "LADDER",
    "LADDER_BY_KEY",
    "AirportMean",
    "Booster",
    "Rung",
    "booster_backend",
]
