# SPDX-License-Identifier: GPL-3.0-only
"""The record-quality term: what to predict for rows no aircraft produced.

The target has a heavy tail, and most of it is not taxiing. Profiled in
``reports/tails.md``: the rows whose recorded taxi-out exceeds two hours are
82% Rome Fiumicino, and for nine in ten of those the recorded off-block time
sits within a minute of the *scheduled* departure time while the take-off is
hours later. The block time was defaulted to the schedule, so the recorded
"taxi-out" is ``MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt`` -- the schedule lag,
which the ranking set exposes for every test departure. A smaller population
in the same cells carries a block time from the previous day, so the recorded
value is a day plus an ordinary taxi.

The ranking page states the scoring rule in these words: teams "will be
ranked based on the best Root Mean Square Error (RMSE) achieved across their
submissions", and the ranking script "will issue an error if ... any rows
are missing", so every row of the template is scored and none is excluded.
Under that rule the optimal prediction for such a row is its conditional
mean, and this module computes that conditional mean for what it is: a
number no aircraft ever produced, optimal under the stated rule and nothing
else. It is kept apart from the physical model so the two cannot be
confused, and the physical model is reported with and without it.

Three fields the ranking set exposes locate the rows:

- whether the flight-side join found a record (``LOBT_flt`` null);
- the *lag*, take-off minus scheduled time, in one of four bands
  (``< -1 h``, ``[-1 h, 4 h]``, ``(4 h, 12 h]``, ``> 12 h`` -- the edges of
  the lag table in ``reports/data_review.md``);
- the airport, because the defaulting is a property of one airport's data
  feed and of no other.

Together they define a *cell*. Fitting measures, per cell on training rows,
how often the block time was defaulted (``share``) and what the remaining
rows average (``rest_mean``); prediction for a row in a covered cell is the
mixture ``share * lag + (1 - share) * rest_mean``. A cell is covered when it
has enough rows and a material defaulted share; every other row -- including
every row at an airport whose cells are clean -- gets no prediction here and
falls through to the physical model. There is no pooled fallback across
airports: pooling would carry Fiumicino's artifact rate to airports that have
none, and the report measures what that costs.

``kind="mean"`` replaces the mixture with the plain cell mean, the form the
plan first proposed; it is kept so the ablation can show what knowing the
mechanism is worth.

One class of departure the cells cannot reach is predicted beside them: the
*plan-shifted* departures, ``PlanShiftModel``. A departure the term does not
cover whose take-off falls more than ``TAIL_EXCLUSION_ADOPTED_H`` after both of
its flight plan's off-block estimates, ``EOBT_1_flt`` and ``LOBT_flt``, both
present. The cells key on the schedule lag and on the absence of a record, and
these departures have a record. A fit counts the class's departures and how
many of them recorded a taxi-out over the same four hours, and the prediction
is the term's own form: ``share * lag + (1 - share) * rest_mean``, with the
add-one share ``(long + 1) / (n + 2)``, the lag taken from ``LOBT_flt``, and the
rest mean of the airport's departures with a record and a taxi-out of four
hours or less. The class was added on 2 October 2026, after it was seen in the
ranking set's exposed fields, on the evidence of three training departures,
two of them long. ``reports/preregistered-2026-10-02.md`` fixed the rule before
it was evaluated or built, and is the source of its definition.

Interface, fixed between the two Gate 1 streams, and widened by one class::

    flag(frame)                -> DataFrame[flt_missing, lag_s, tail_bucket]
    TailModel().fit(frame)     -> self          (training rows, target present)
    TailModel().predict(frame) -> Series        (NaN where not covered)
    combine(physical, tail)    -> Series        (tail where present, else physical)

    PlanShiftModel(term).fit(frame)     -> self    (the rows the term was fitted on)
    PlanShiftModel(term).predict(frame) -> Series  (NaN off the plan-shifted rows)
    combine(physical, tail, plan_shift) -> Series  (tail, else plan_shift, else physical)

``flag`` and ``predict`` read only ``LOBT_flt``, ``MVT_TIME_UTC_mvt``,
``SCHED_TIME_UTC_mvt`` and ``ADEP_mvt``; ``fit`` additionally reads the two
fields the ranking set hides, which is what a fit on training rows is for.
``PlanShiftModel.predict`` reads ``MVT_TIME_UTC_mvt``, ``EOBT_1_flt``,
``LOBT_flt`` and ``ADEP_mvt``, and leaves the term's rows to ``combine``, which
puts the term first; its ``fit`` additionally reads the recorded taxi-out, and
asks the fitted term which of the rows it is given the term covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
import pandas as pd

from taxiout.models import TAIL_EXCLUSION_ADOPTED_H

HOUR = 3600.0

# What may be read where. ``FLAG_INPUTS`` are exposed for every ranking
# departure; ``FIT_ONLY_INPUTS`` are exactly the fields the ranking set hides,
# and are read on training rows only. ``tests/test_tails.py`` asserts both
# against ``taxiout.challenge``.
FLAG_INPUTS: tuple[str, ...] = ("LOBT_flt", "MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt")
PREDICT_INPUTS: tuple[str, ...] = FLAG_INPUTS + ("ADEP_mvt",)
FIT_ONLY_INPUTS: tuple[str, ...] = ("BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt")

# Lag bands, with the edges of the lag table in reports/data_review.md:
# ``> 12 h`` is open below, ``4-12 h`` is (4 h, 12 h], ``< -1 h`` is open
# above, and everything else -- [-1 h, 4 h] -- is ``normal``.
LAG_BANDS: tuple[str, ...] = ("normal", "4-12h", ">12h", "<-1h")
BUCKETS: tuple[str, ...] = tuple(
    f"{rec}:{band}" for rec in ("flt", "noflt") for band in LAG_BANDS
)
UNFLAGGED_BUCKET = "flt:normal"

# A block time this close to the schedule, on a departure this long after
# it, is a defaulted block time rather than a pushback. The 60 s absorbs the
# schedule's minute resolution; 83% of the rows it catches are within 5 s.
# Two hours is above every plausible taxi-out and below every artifact.
NEAR_S = 60.0
LONG_S = 2 * HOUR

# Coverage rule. Below five rows a cell's defaulted share is a coin toss;
# above ten the rule discards the cells the term exists for (the Fiumicino
# ``> 12 h`` cell has 29 rows in a year). Five percent separates cells whose
# artifact share is a property of the feed from cells with a stray row.
MIN_COUNT = 5
MIN_SHARE = 0.05

CELL_COLUMNS: tuple[str, ...] = ("n", "share", "mean", "rest_mean", "covered")


# --------------------------------------------------------------------------
# flags


def _utc(s: pd.Series) -> pd.Series:
    """Timestamps as tz-aware UTC, whatever they arrived as."""
    return pd.to_datetime(s, utc=True)


def lag_seconds(frame: pd.DataFrame) -> pd.Series:
    """Take-off minus scheduled time, in seconds. The *lag*."""
    delta = _utc(frame["MVT_TIME_UTC_mvt"]) - _utc(frame["SCHED_TIME_UTC_mvt"])
    return delta.dt.total_seconds()


def lag_band(lag_s: pd.Series) -> pd.Series:
    """Which of ``LAG_BANDS`` a lag falls in. A missing lag reads as normal."""
    lag = pd.to_numeric(lag_s, errors="coerce")
    return pd.Series(
        np.select(
            [lag > 12 * HOUR, lag > 4 * HOUR, lag < -HOUR],
            [">12h", "4-12h", "<-1h"],
            default="normal",
        ),
        index=lag_s.index,
    )


def flag(frame: pd.DataFrame) -> pd.DataFrame:
    """The record-quality flags a test departure exposes.

    Returns ``flt_missing`` (no flight-side record), ``lag_s`` and
    ``tail_bucket`` (a categorical over ``BUCKETS``), indexed like ``frame``.
    Reads ``FLAG_INPUTS`` and nothing else.
    """
    missing = [c for c in FLAG_INPUTS if c not in frame.columns]
    if missing:
        raise KeyError(f"flag() needs {missing}")
    flt_missing = frame["LOBT_flt"].isna()
    lag_s = lag_seconds(frame)
    prefix = np.where(flt_missing.to_numpy(), "noflt:", "flt:")
    label = prefix + lag_band(lag_s).to_numpy()
    return pd.DataFrame(
        {
            "flt_missing": flt_missing.astype(bool),
            "lag_s": lag_s.astype(float),
            "tail_bucket": pd.Categorical(label, categories=BUCKETS),
        },
        index=frame.index,
    )


def defaulted_block(
    frame: pd.DataFrame, near_s: float = NEAR_S, long_s: float = LONG_S
) -> pd.Series:
    """Training rows whose block time was defaulted to the schedule.

    ``|BLOCK - SCHED| <= near_s`` on a departure with lag above ``long_s``.
    Reads the hidden block time, so this is for training rows only.
    """
    gap = _utc(frame["BLOCK_TIME_UTC_mvt"]) - _utc(frame["SCHED_TIME_UTC_mvt"])
    near = gap.dt.total_seconds().abs() <= near_s
    return near & (lag_seconds(frame) > long_s)


# --------------------------------------------------------------------------
# the term


def _cell_keys(frame: pd.DataFrame, flags: pd.DataFrame) -> pd.MultiIndex:
    return pd.MultiIndex.from_arrays(
        [
            frame["ADEP_mvt"].astype(str).to_numpy(),
            flags["tail_bucket"].astype(str).to_numpy(),
        ],
        names=["icao", "tail_bucket"],
    )


@dataclass
class TailModel:
    """Conditional mean per (airport, bucket) cell, for covered cells only.

    ``cells`` is the fitted table: one row per (airport, bucket) seen in
    training, with ``n``, ``share`` (defaulted block times), ``mean`` (of the
    target), ``rest_mean`` (of the target over rows not defaulted) and
    ``covered``. Aggregates only; safe to print.
    """

    kind: str = "mixture"
    min_count: int = MIN_COUNT
    min_share: float = MIN_SHARE
    near_s: float = NEAR_S
    long_s: float = LONG_S
    cells: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        if self.kind not in ("mixture", "mean"):
            raise ValueError(f"kind must be 'mixture' or 'mean', not {self.kind!r}")

    # -- fitting

    def fit(self, frame: pd.DataFrame) -> "TailModel":
        """Measure every (airport, bucket) cell on training rows."""
        need = PREDICT_INPUTS + FIT_ONLY_INPUTS
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise KeyError(f"fit() needs {missing}")
        flags = flag(frame)
        y = pd.to_numeric(frame["TAXITIME_SEC_mvt"], errors="coerce").astype(float)
        art = defaulted_block(frame, self.near_s, self.long_s)
        art = art.fillna(False).astype(bool)
        keep = y.notna().to_numpy()
        keys = _cell_keys(frame, flags)[keep]
        t = pd.DataFrame(
            {"y": y.to_numpy()[keep], "art": art.to_numpy()[keep]}, index=keys
        )
        levels = ["icao", "tail_bucket"]
        g = t.groupby(level=levels, sort=True)
        cells = pd.DataFrame(
            {
                "n": g.size().astype(int),
                "share": g["art"].mean().astype(float),
                "mean": g["y"].mean().astype(float),
            }
        )
        rest = t.loc[~t["art"], "y"].groupby(level=levels).mean()
        cells["rest_mean"] = rest.reindex(cells.index).astype(float)
        cells["covered"] = (cells["n"] >= self.min_count) & (
            cells["share"] >= self.min_share
        )
        self.cells = cells[list(CELL_COLUMNS)]
        return self

    @property
    def covered(self) -> pd.DataFrame:
        """The cells the term predicts for."""
        self._check_fitted()
        return self.cells[self.cells["covered"]]

    # -- prediction

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """The term for rows in covered cells; NaN elsewhere.

        Reads ``PREDICT_INPUTS`` only, so it runs unchanged on the ranking set.
        """
        self._check_fitted()
        missing = [c for c in PREDICT_INPUTS if c not in frame.columns]
        if missing:
            raise KeyError(f"predict() needs {missing}")
        flags = flag(frame)
        cells = self.cells.reindex(_cell_keys(frame, flags))
        covered = cells["covered"].to_numpy(dtype=object)
        covered = np.where(pd.isna(covered), False, covered).astype(bool)
        if self.kind == "mean":
            value = cells["mean"].to_numpy()
        else:
            share = cells["share"].to_numpy()
            # A cell with every row defaulted has no ``rest_mean``; its weight
            # is zero, and 0 * NaN must not poison the lag.
            rest = cells["rest_mean"].to_numpy()
            rest = np.where(np.isnan(rest), 0.0, rest)
            value = share * flags["lag_s"].to_numpy() + (1.0 - share) * rest
        out = np.where(covered, value, np.nan)
        return pd.Series(out, index=frame.index, name="tail", dtype=float)

    def flagged(self, frame: pd.DataFrame) -> pd.Series:
        """Boolean mask of the rows the term predicts for."""
        return self.predict(frame).notna().rename("flagged")

    def _check_fitted(self) -> None:
        if "covered" not in self.cells.columns:
            raise RuntimeError("TailModel is not fitted")


# --------------------------------------------------------------------------
# the plan-shifted departures

# What the rule reads, in the words of ``reports/preregistered-2026-10-02.md``:
# the take-off time, the two off-block estimates, the departure airport and, at
# fit time, the recorded taxi-out. Never ``AOBT_3_flt``, which stays excluded
# by decision. ``tests/test_tails.py`` asserts both against ``taxiout.challenge``.
PLAN_SHIFT_PREDICT_INPUTS: tuple[str, ...] = (
    "MVT_TIME_UTC_mvt",
    "EOBT_1_flt",
    "LOBT_flt",
    "ADEP_mvt",
)
PLAN_SHIFT_FIT_ONLY_INPUTS: tuple[str, ...] = ("TAXITIME_SEC_mvt",)

# The add-one estimate of the class's long share, ``(long + 1) / (n + 2)``: one
# long and one ordinary departure added to the class a fit is given, so that a
# class of one or three rows cannot assert a share of none or all. The form is
# the estimator's, fixed with the rest of the rule by
# reports/preregistered-2026-10-02.md, and nothing about it was measured.
PLAN_SHIFT_ADD_ONE: tuple[int, int] = (1, 2)


def plan_shift_threshold_s() -> float:
    """``TAIL_EXCLUSION_ADOPTED_H`` in seconds, read where it is fixed.

    The pre-registration defines the class, *long* and the rest mean by the
    four hours ``taxiout.models`` already fixed on the mechanism that no
    aircraft taxis for four hours, so there is no class without it.
    """
    if TAIL_EXCLUSION_ADOPTED_H is None:
        raise RuntimeError(
            "the plan-shifted class is defined by TAIL_EXCLUSION_ADOPTED_H,"
            " which is not adopted"
        )
    return float(TAIL_EXCLUSION_ADOPTED_H) * HOUR


def plan_shifted(frame: pd.DataFrame) -> pd.Series:
    """Rows whose take-off falls past the threshold after both estimates.

    Both ``EOBT_1_flt`` and ``LOBT_flt`` present, and take-off minus each one
    strictly above ``plan_shift_threshold_s``. The class is these rows less the
    term's; ``PlanShiftModel.in_class`` takes the term's out.
    """
    need = ("MVT_TIME_UTC_mvt", "EOBT_1_flt", "LOBT_flt")
    missing = [c for c in need if c not in frame.columns]
    if missing:
        raise KeyError(f"plan_shifted() needs {missing}")
    threshold = plan_shift_threshold_s()
    takeoff = _utc(frame["MVT_TIME_UTC_mvt"])
    after_eobt = (takeoff - _utc(frame["EOBT_1_flt"])).dt.total_seconds()
    after_lobt = (takeoff - _utc(frame["LOBT_flt"])).dt.total_seconds()
    present = frame["EOBT_1_flt"].notna() & frame["LOBT_flt"].notna()
    out = present & (after_eobt > threshold) & (after_lobt > threshold)
    return out.fillna(False).astype(bool).rename("plan_shifted")


@dataclass
class PlanShiftModel:
    """The rule of ``reports/preregistered-2026-10-02.md``, fitted.

    ``term`` is the fitted record-quality term, and the class is defined
    against it: a departure the term does not cover, with both lags past the
    threshold. ``fit`` reads the rows the term was fitted on. ``n`` and
    ``long`` count the class and its departures recorded over the threshold,
    ``share`` is the add-one estimate, and ``rest_mean`` is the mean recorded
    taxi-out per airport over the departures with a flight-plan record
    (``LOBT_flt`` present, as ``flag`` reads it) and a taxi-out of at most the
    threshold, with nothing left out for being the term's or the class's, since
    the pre-registration names no such exclusion. Aggregates only; safe to print.
    """

    PREDICT_INPUTS: ClassVar[tuple[str, ...]] = PLAN_SHIFT_PREDICT_INPUTS
    FIT_ONLY_INPUTS: ClassVar[tuple[str, ...]] = PLAN_SHIFT_FIT_ONLY_INPUTS

    term: TailModel
    n: int | None = None
    long: int | None = None
    share: float | None = None
    rest_mean: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def in_class(self, frame: pd.DataFrame) -> pd.Series:
        """The class: the rows the term leaves, of those ``plan_shifted`` picks."""
        covered = self.term.flagged(frame).to_numpy(dtype=bool)
        return (plan_shifted(frame) & ~covered).rename("plan_shift")

    def fit(self, frame: pd.DataFrame) -> "PlanShiftModel":
        """Count the class, its long departures and the rest means on ``frame``."""
        need = PLAN_SHIFT_PREDICT_INPUTS + PLAN_SHIFT_FIT_ONLY_INPUTS
        missing = [c for c in need if c not in frame.columns]
        if missing:
            raise KeyError(f"fit() needs {missing}")
        threshold = plan_shift_threshold_s()
        y = pd.to_numeric(frame["TAXITIME_SEC_mvt"], errors="coerce").astype(float)
        members = self.in_class(frame)
        self.n = int(members.sum())
        self.long = int((members & (y > threshold)).sum())
        plus_long, plus_n = PLAN_SHIFT_ADD_ONE
        self.share = (self.long + plus_long) / (self.n + plus_n)
        rest = (frame["LOBT_flt"].notna() & y.notna() & (y <= threshold)).to_numpy()
        icao = frame["ADEP_mvt"].astype(str).to_numpy()[rest]
        self.rest_mean = (
            pd.Series(y.to_numpy()[rest], index=pd.Index(icao, name="icao"))
            .groupby(level="icao", sort=True)
            .mean()
            .astype(float)
            .rename("rest_mean")
        )
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        """The rule on the rows ``plan_shifted`` picks; NaN elsewhere.

        Reads ``PREDICT_INPUTS`` only, so it runs unchanged on the ranking set,
        and does not ask the term: a row the term covers is the term's, and
        ``combine`` puts the term first. A plan-shifted row at an airport the
        fit holds no rest mean for is refused with a ``ValueError``: the
        pre-registration does not define its prediction, so this neither
        drops the rest mean nor leaves the row to the model, either of which
        would be a choice made here. No departure of the training year or of
        the ranking set is such a row.
        """
        self._check_fitted()
        missing = [c for c in PLAN_SHIFT_PREDICT_INPUTS if c not in frame.columns]
        if missing:
            raise KeyError(f"predict() needs {missing}")
        picked = plan_shifted(frame).to_numpy(dtype=bool)
        lag = (
            (_utc(frame["MVT_TIME_UTC_mvt"]) - _utc(frame["LOBT_flt"]))
            .dt.total_seconds()
            .to_numpy(dtype=float)
        )
        icao = frame["ADEP_mvt"].astype(str)
        rest = icao.map(self.rest_mean).to_numpy(dtype=float)
        undefined = picked & np.isnan(rest)
        if undefined.any():
            raise ValueError(
                "the plan-shifted departures' rule has no rest mean for "
                f"{sorted(set(icao[undefined]))}, and the pre-registration does not"
                " say what it predicts there"
            )
        value = self.share * lag + (1.0 - self.share) * rest
        out = np.where(picked, value, np.nan)
        return pd.Series(out, index=frame.index, name="plan_shift", dtype=float)

    def _check_fitted(self) -> None:
        if self.share is None:
            raise RuntimeError("PlanShiftModel is not fitted")


def combine(physical, tail: pd.Series, plan_shift: pd.Series | None = None) -> pd.Series:
    """The submitted prediction, in order: the term where it exists, then the
    plan-shifted departures' rule on the rows the term leaves, else the
    physical model. Without ``plan_shift`` it is the term and the model alone.
    """
    tail = pd.Series(tail, dtype=float)
    if isinstance(physical, pd.Series):
        # A Series is aligned by index, and a mismatch is an error rather
        # than a silent NaN; an array is taken positionally.
        if not physical.index.equals(tail.index):
            raise ValueError("physical and tail must share an index")
        phys = pd.to_numeric(physical, errors="coerce").to_numpy(dtype=float)
    else:
        phys = np.asarray(physical, dtype=float)
        if phys.shape != (len(tail),):
            raise ValueError("physical and tail differ in length")
    if plan_shift is not None:
        plan_shift = pd.Series(plan_shift, dtype=float)
        if not plan_shift.index.equals(tail.index):
            raise ValueError("tail and plan_shift must share an index")
        phys = np.where(plan_shift.notna().to_numpy(), plan_shift.to_numpy(), phys)
    out = np.where(tail.notna().to_numpy(), tail.to_numpy(), phys)
    return pd.Series(out, index=tail.index, name="prediction", dtype=float)


# --------------------------------------------------------------------------
# C6: the term's rows predicted from the schedule
#
# Section 2 of ``reports/preregistered-2026-10-04-tails.md``, beside the term
# and leaving it unchanged: the rows are the term's, in the cells it covers,
# and the prediction keeps the term's form, share * lag + (1 - share) * rest
# mean. Two things change. A row is *at the schedule* when its block time lies
# within ``NEAR_S`` of its scheduled time, with no condition on the lag; the
# share is the share of rows at the schedule and the rest mean is over the rows
# that are not. And ``noflt:normal`` is split by lag into four sub-cells. That
# is C6a. C6b, adopted by ``reports/c6.md`` under the section's rule, replaces
# each row's share with a classifier's probability that the row is at the
# schedule, and keeps C6a's rest mean for the row's sub-cell.
#
# Readings the section leaves open, fixed before any figure existed and kept
# here as they were fixed: a missing ``STAND_mvt``, ``AIRCRAFT_TYPE_mvt`` or
# callsign prefix is a level of its own under the ten-row rule; the bands are
# [-1 h, 0], (0, 1 h], (1 h, 2 h] and (2 h, 4 h]; the scheduled hour, the
# take-off hour and the weekday enter the classifier as integers, the sub-cell
# and the three fields as LightGBM categoricals under its default categorical
# settings; and a level the fit never saw is read as ``other`` at prediction.

# The three fields the classifier reads beside the term's inputs, all exposed
# for ranking departures; at fit time C6 reads the term's two hidden fields.
C6_PREDICT_INPUTS: tuple[str, ...] = PREDICT_INPUTS + (
    "FLIGHT_mvt",
    "STAND_mvt",
    "AIRCRAFT_TYPE_mvt",
)
C6_FIT_ONLY_INPUTS: tuple[str, ...] = FIT_ONLY_INPUTS

# "A sub-cell with fewer than five fitting rows takes the values C6a computes
# for its whole cell"; "each level with fewer than ten fitting rows read as one
# level, `other`". Both the section's words.
C6_SUB_MIN = 5
C6_LEVEL_MIN = 10
C6_OTHER = "other"
C6_MISSING = "<missing>"
C6_SPLIT_BUCKET = "noflt:normal"
# The four sub-cells of ``noflt:normal``, edges in hours as the section gives
# them; the first band is closed below, every band closed above.
C6_SUB_EDGES_H: tuple[tuple[int, int], ...] = ((-1, 0), (0, 1), (1, 2), (2, 4))
C6_SUB_NAMES: tuple[str, ...] = ("[-1h,0]", "(0,1h]", "(1h,2h]", "(2h,4h]")
C6_CATEGORICAL: tuple[str, ...] = ("prefix", "STAND_mvt", "AIRCRAFT_TYPE_mvt")

# "objective binary, 300 trees, learning rate 0.03, 15 leaves, minimum child
# 20, no bagging, deterministic, column-wise, eight threads, seed 20260902".
C6_CLASSIFIER_PARAMS: dict[str, object] = {
    "objective": "binary",
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 20,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "feature_fraction": 1.0,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 8,
    "seed": 20260902,
    "verbose": -1,
}
C6_CLASSIFIER_ROUNDS = 300


def at_schedule(frame: pd.DataFrame) -> pd.Series:
    """Block time within ``NEAR_S`` of the scheduled time, no lag condition."""
    gap = _utc(frame["BLOCK_TIME_UTC_mvt"]) - _utc(frame["SCHED_TIME_UTC_mvt"])
    return (gap.dt.total_seconds().abs() <= NEAR_S).fillna(False).astype(bool)


def c6_cell(frame: pd.DataFrame) -> pd.Series:
    """The term's cell of each row, ``airport|bucket``."""
    flags = flag(frame)
    return frame["ADEP_mvt"].astype(str) + "|" + flags["tail_bucket"].astype(str)


def c6_band(lag_s: np.ndarray) -> np.ndarray:
    """The four bands of ``noflt:normal``; '' outside them (a missing lag included)."""
    out = np.full(lag_s.shape, "", dtype=object)
    for (lo_h, hi_h), name in zip(C6_SUB_EDGES_H, C6_SUB_NAMES):
        lo, hi = lo_h * HOUR, hi_h * HOUR
        inside = (lag_s > lo) & (lag_s <= hi)
        if (lo_h, hi_h) == C6_SUB_EDGES_H[0]:
            inside = (lag_s >= lo) & (lag_s <= hi)
        out[inside] = name
    return out


def c6_sub_cell(frame: pd.DataFrame) -> pd.Series:
    """The cell, with ``noflt:normal`` split into its four lag sub-cells."""
    cell = c6_cell(frame).to_numpy(dtype=object)
    lag = lag_seconds(frame).to_numpy(dtype=float)
    split = np.array([c.endswith("|" + C6_SPLIT_BUCKET) for c in cell], dtype=bool)
    sub = c6_band(lag)
    if (split & (sub == "")).any():
        raise ValueError("a noflt:normal row lies in no sub-cell (missing lag?)")
    out = np.where(split, cell + "|" + sub, cell)
    return pd.Series(out, index=frame.index, name="sub_cell")


def _share_rest(at: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    share = float(at.mean())
    rest = float(y[~at].mean()) if (~at).any() else float("nan")
    return share, rest


def c6_mix(share: np.ndarray, lag: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """share x lag + (1 - share) x rest; a NaN rest (every row at the schedule)
    is read as 0 under weight 0, as the term reads it."""
    rest = np.where(np.isnan(rest), 0.0, rest)
    return share * lag + (1.0 - share) * rest


class C6a:
    """Share at the schedule and rest mean, per sub-cell, with the fallback.

    ``cells`` holds ``(share, rest mean, n)`` per cell and ``subs``
    ``(share, rest mean, n, fell back)`` per sub-cell. Aggregates only.
    """

    def fit(self, frame: pd.DataFrame) -> "C6a":
        """``frame``: the fitting rows (the term's covered cells, target present)."""
        y = pd.to_numeric(frame["TAXITIME_SEC_mvt"]).to_numpy(dtype=float)
        at = at_schedule(frame).to_numpy()
        cell = c6_cell(frame).to_numpy(dtype=object)
        sub = c6_sub_cell(frame).to_numpy(dtype=object)
        self.cells = {}
        for c in sorted(set(cell)):
            m = cell == c
            self.cells[c] = (*_share_rest(at[m], y[m]), int(m.sum()))
        self.subs = {}
        for s in sorted(set(sub)):
            m = sub == s
            c = s.rsplit("|", 1)[0] if s.count("|") == 2 else s
            n = int(m.sum())
            if n < C6_SUB_MIN:
                share, rest, _ = self.cells[c]
                self.subs[s] = (share, rest, n, True)
            else:
                self.subs[s] = (*_share_rest(at[m], y[m]), n, False)
        return self

    def values(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """(share, rest mean) per row; a sub-cell unseen in the fit takes its cell's."""
        sub = c6_sub_cell(frame).to_numpy(dtype=object)
        cell = c6_cell(frame).to_numpy(dtype=object)
        share = np.empty(len(frame))
        rest = np.empty(len(frame))
        for i, (s, c) in enumerate(zip(sub, cell)):
            v = self.subs.get(s)
            sh, re = (v[0], v[1]) if v is not None else self.cells[c][:2]
            share[i], rest[i] = sh, re
        return share, rest

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        share, rest = self.values(frame)
        lag = lag_seconds(frame).to_numpy(dtype=float)
        return c6_mix(share, lag, rest)


def c6_levels(frame: pd.DataFrame) -> pd.DataFrame:
    """The three categorical fields as strings, a missing value its own level."""
    flight = frame["FLIGHT_mvt"].astype(object)
    prefix = flight.where(flight.isna(), flight.astype(str).str[:3])
    out = pd.DataFrame(index=frame.index)
    out["prefix"] = prefix
    out["STAND_mvt"] = frame["STAND_mvt"].astype(object)
    out["AIRCRAFT_TYPE_mvt"] = frame["AIRCRAFT_TYPE_mvt"].astype(object)
    for c in C6_CATEGORICAL:
        v = out[c]
        out[c] = np.where(v.isna().to_numpy(), C6_MISSING, v.astype(str).to_numpy())
    return out


def c6_fold_levels(
    levels: pd.Series, kept: list[str] | None = None
) -> tuple[pd.Series, list[str]]:
    """Levels with fewer than ``C6_LEVEL_MIN`` rows (in the fit) read as ``other``."""
    if kept is None:
        counts = levels.value_counts()
        kept = sorted(counts.index[counts >= C6_LEVEL_MIN].tolist())
    out = levels.where(levels.isin(kept), C6_OTHER)
    return out, kept


class C6b:
    """The classifier's probability as the share; C6a's rest mean."""

    def __init__(self, c6a: C6a):
        self.c6a = c6a

    def features(self, frame: pd.DataFrame) -> pd.DataFrame:
        sched = _utc(frame["SCHED_TIME_UTC_mvt"])
        takeoff = _utc(frame["MVT_TIME_UTC_mvt"])
        X = pd.DataFrame(index=frame.index)
        X["lag"] = lag_seconds(frame).astype(float)
        X["sub_cell"] = pd.Categorical(c6_sub_cell(frame), categories=self.sub_levels)
        X["sched_hour"] = sched.dt.hour.astype(int)
        X["takeoff_hour"] = takeoff.dt.hour.astype(int)
        X["weekday"] = sched.dt.dayofweek.astype(int)
        raw = c6_levels(frame)
        for c in C6_CATEGORICAL:
            folded, _ = c6_fold_levels(raw[c], self.kept[c])
            X[c] = pd.Categorical(folded, categories=self.kept[c] + [C6_OTHER])
        return X

    def fit(self, frame: pd.DataFrame) -> "C6b":
        import lightgbm as lgb

        raw = c6_levels(frame)
        self.kept = {c: c6_fold_levels(raw[c])[1] for c in C6_CATEGORICAL}
        self.sub_levels = sorted(set(c6_sub_cell(frame)))
        X = self.features(frame)
        if X["sub_cell"].isna().any():
            raise ValueError("sub-cell outside the fitted levels")
        y = at_schedule(frame).astype(int).to_numpy()
        cats = ["sub_cell", *C6_CATEGORICAL]
        ds = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
        self.booster = lgb.train(
            C6_CLASSIFIER_PARAMS, ds, num_boost_round=C6_CLASSIFIER_ROUNDS
        )
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        X = self.features(frame)
        p = self.booster.predict(X)
        _, rest = self.c6a.values(frame)
        return c6_mix(p, X["lag"].to_numpy(dtype=float), rest)


def c6_fitting_rows(term: TailModel, frame: pd.DataFrame) -> np.ndarray:
    """The term's fitting rows in its covered cells: in a covered cell, target present."""
    flags = flag(frame)
    keys = _cell_keys(frame, flags)
    cov = set(term.covered.index.tolist())
    in_cov = np.fromiter((k in cov for k in keys), bool, len(frame))
    y = pd.to_numeric(frame["TAXITIME_SEC_mvt"], errors="coerce")
    return in_cov & y.notna().to_numpy()


@dataclass
class C6Model:
    """C6b fitted on every fitting row of the term's cells, predicting the term's rows.

    ``term`` is the fitted record-quality term; ``fit`` reads the rows the term
    was fitted on and ``predict`` returns C6b's value on exactly the rows the
    term predicts, NaN elsewhere, raw: the floor at zero and the rounding are
    the file's, applied to every row alike, as for the term.
    """

    PREDICT_INPUTS: ClassVar[tuple[str, ...]] = C6_PREDICT_INPUTS
    FIT_ONLY_INPUTS: ClassVar[tuple[str, ...]] = C6_FIT_ONLY_INPUTS

    term: TailModel
    c6a: C6a | None = None
    c6b: C6b | None = None
    fitting_rows: int | None = None

    def fit(self, frame: pd.DataFrame) -> "C6Model":
        rows = frame[c6_fitting_rows(self.term, frame)]
        self.fitting_rows = int(len(rows))
        self.c6a = C6a().fit(rows)
        self.c6b = C6b(self.c6a).fit(rows)
        return self

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        if self.c6b is None:
            raise RuntimeError("C6Model is not fitted")
        covered = self.term.predict(frame).notna().to_numpy()
        out = np.full(len(frame), np.nan)
        if covered.any():
            out[covered] = self.c6b.predict(frame[covered])
        return pd.Series(out, index=frame.index, name="c6b", dtype=float)


__all__ = [
    "BUCKETS",
    "C6_CATEGORICAL",
    "C6_CLASSIFIER_PARAMS",
    "C6_CLASSIFIER_ROUNDS",
    "C6_FIT_ONLY_INPUTS",
    "C6_LEVEL_MIN",
    "C6_PREDICT_INPUTS",
    "C6_SUB_EDGES_H",
    "C6_SUB_MIN",
    "C6_SUB_NAMES",
    "C6Model",
    "C6a",
    "C6b",
    "CELL_COLUMNS",
    "at_schedule",
    "c6_band",
    "c6_cell",
    "c6_fitting_rows",
    "c6_fold_levels",
    "c6_levels",
    "c6_mix",
    "c6_sub_cell",
    "FIT_ONLY_INPUTS",
    "FLAG_INPUTS",
    "LAG_BANDS",
    "LONG_S",
    "MIN_COUNT",
    "MIN_SHARE",
    "NEAR_S",
    "PLAN_SHIFT_ADD_ONE",
    "PLAN_SHIFT_FIT_ONLY_INPUTS",
    "PLAN_SHIFT_PREDICT_INPUTS",
    "PREDICT_INPUTS",
    "UNFLAGGED_BUCKET",
    "PlanShiftModel",
    "TailModel",
    "combine",
    "defaulted_block",
    "flag",
    "lag_band",
    "lag_seconds",
    "plan_shift_threshold_s",
    "plan_shifted",
]
