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


__all__ = [
    "BUCKETS",
    "CELL_COLUMNS",
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
