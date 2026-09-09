# SPDX-License-Identifier: GPL-3.0-only
"""Plan lags: take-off against the planning values fixed before departure.

Three planning times are exposed for every ranking departure and fixed
before the aircraft moves: the flight plan's first estimated off-block time
(``EOBT_1_flt``), its last (``LOBT_flt``), and the schedule
(``SCHED_TIME_UTC_mvt``). Take-off minus any of them is the taxi-out plus
the pushback delay against that plan -- how late the aircraft left the
stand, and then how long it taxied. Neither part is known on its own from
what the ranking set exposes; the sum is, and the sum is what this block
carries.

Two readings of each lag:

- the subject's own, in seconds, NaN where the planning value is missing,
  with one flag for a missing record;
- the mean of the same lag over the *other* departures on the subject's
  runway whose take-off falls within fifteen minutes either side of the
  subject's, built by the window machinery of ``taxiout.windows`` from the
  ``DEP`` phase, the subject removed by identity. That is the lag the
  runway is running at around the subject.

The three lags are all take-off minus a planning value, so their pairwise
differences cancel the take-off and are properties of the plan alone.
``eobt_minus_sched_s`` is how far the filed off-block sat from the
schedule and ``lobt_minus_eobt_s`` how far the Network Manager moved it
afterwards; ``lobt_iobt_s`` in the feature frame, computed in
``taxiout.features``, is the third such difference, so these two complete
the set rather than starting one. The booster sees both lags and reaches
their difference only by splitting on each in turn, which spends depth on
one subtraction. Both are NaN where either planning value is missing, and
there is no second missing flag: ``lag_missing`` is already set whenever
any of the three planning values is absent, and a flag saying that again
could only disagree with it. The block is expected to be worth about a
second. It is here because it is free, and because a quantity the model
has to reconstruct is better given to it.

The sources are exposed fields of the subject's row and of other departures'
rows -- never a departure's block time or taxi time, and never
``AOBT_3_flt`` -- and the registry says which. The ladder places this block
after the aircraft and before the congestion blocks, so that what the
congestion blocks are worth is read net of it; the ladder also keeps the
congestion rungs without it, so the reader sees both.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.windows import (
    Sorted,
    centred_mean,
    coverage_end,
    coverage_start,
    truncated_centred,
)

# The planning values, in the order the features name them.
PLANS: tuple[tuple[str, str], ...] = (
    ("eobt", "EOBT_1_flt"),
    ("lobt", "LOBT_flt"),
    ("sched", "SCHED_TIME_UTC_mvt"),
)

HALF_WINDOW_MIN = 15

OWN_FEATURES: tuple[str, ...] = tuple(f"lag_{k}_s" for k, _ in PLANS) + ("lag_missing",)
RUNWAY_FEATURES: tuple[str, ...] = tuple(
    f"lag_{k}_rwy{HALF_WINDOW_MIN}m" for k, _ in PLANS
) + ("lag_truncated",)
PLAN_LAG_FEATURES: tuple[str, ...] = OWN_FEATURES + RUNWAY_FEATURES
PLAN_DIFF_FEATURES: tuple[str, ...] = ("eobt_minus_sched_s", "lobt_minus_eobt_s")

_SELF = ("MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_ID_mvt")
_OTHERS = (
    "DEP.MVT_TIME_UTC_mvt",
    "DEP.ADEP_mvt",
    "DEP.RUNWAY_mvt",
    "DEP.MVT_ID_mvt",
)

SOURCES: dict[str, tuple[str, ...]] = {
    **{f"lag_{k}_s": ("MVT_TIME_UTC_mvt", col) for k, col in PLANS},
    "lag_missing": ("MVT_TIME_UTC_mvt",) + tuple(col for _, col in PLANS),
    **{
        f"lag_{k}_rwy{HALF_WINDOW_MIN}m": _SELF + (col,) + _OTHERS + (f"DEP.{col}",)
        for k, col in PLANS
    },
    "lag_truncated": ("MVT_TIME_UTC_mvt",),
}

MOVEMENT_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {"PHASE_mvt"}
        | {
            c.split(".", 1)[1]
            for sources in SOURCES.values()
            for c in sources
            if "." in c
        }
    )
)

# The differences keep their own registry, so that the lags' one and the
# movement columns it feeds stay exactly what the ladder measured them at.
# They read the subject's own row, so they add no movement column at all.
DIFF_SOURCES: dict[str, tuple[str, ...]] = {
    "eobt_minus_sched_s": ("EOBT_1_flt", "SCHED_TIME_UTC_mvt"),
    "lobt_minus_eobt_s": ("LOBT_flt", "EOBT_1_flt"),
}


def _lag_seconds(takeoff: pd.Series, plan: pd.Series) -> np.ndarray:
    """One timestamp minus another in seconds, NaN where either is missing.

    ``total_seconds`` reads the timedelta's own unit, so the answer is the
    same whether the stamps arrive at the nanoseconds the tests build or the
    microseconds the challenge parquet files carry. An integer view divided
    by a fixed billion would not be: on microsecond stamps every lag and
    every plan difference would come out a thousand times too small, all
    together, with nothing about the numbers to say so.
    ``taxiout.windows.seconds`` buys the same resolution safety the other
    way, by forcing the unit before it takes its integer view; both blocks
    here feed that function too, so the pair has to hold at once.
    """
    lag = pd.to_datetime(takeoff, utc=True) - pd.to_datetime(plan, utc=True)
    return lag.dt.total_seconds().to_numpy(dtype=float)


def planlag_features(departures: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """The plan-lag block for ``departures``, from their rows and their runway's.

    ``movements`` carries ``MOVEMENT_COLUMNS``; only its ``DEP`` rows are
    read. The subjects are normally among them and are removed from their
    own runway mean by identifier whether or not they are. The result is
    indexed like ``departures`` and carries exactly ``PLAN_LAG_FEATURES``.
    """
    for c in MOVEMENT_COLUMNS:
        if c not in movements.columns:
            raise KeyError(f"movements need {c!r}")
    out = pd.DataFrame(index=departures.index)
    t = departures["MVT_TIME_UTC_mvt"]
    own = {k: _lag_seconds(t, departures[col]) for k, col in PLANS}
    for k, _ in PLANS:
        out[f"lag_{k}_s"] = own[k]
    out["lag_missing"] = (
        np.isnan(np.column_stack(list(own.values()))).any(axis=1).astype(float)
    )

    dep = movements[movements["PHASE_mvt"].astype(str) == "DEP"]
    dep_t = dep["MVT_TIME_UTC_mvt"]
    events = Sorted.build(
        dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str),
        dep_t,
        values={k: _lag_seconds(dep_t, dep[col]) for k, col in PLANS},
    )
    key = departures["ADEP_mvt"].astype(str) + "|" + departures["RUNWAY_mvt"].astype(str)
    in_dep = departures["MVT_ID_mvt"].isin(dep["MVT_ID_mvt"]).to_numpy()
    for k, _ in PLANS:
        mean, _n = centred_mean(
            events,
            k,
            key,
            t,
            60 * HALF_WINDOW_MIN,
            self_value=np.where(in_dep, own[k], np.nan),
        )
        out[f"lag_{k}_rwy{HALF_WINDOW_MIN}m"] = mean

    start = coverage_start(t)
    end = coverage_end(t)
    out["lag_truncated"] = truncated_centred(t, 60 * HALF_WINDOW_MIN, start, end).astype(
        float
    )
    return out[list(PLAN_LAG_FEATURES)]


def plandiff_features(departures: pd.DataFrame) -> pd.DataFrame:
    """The plan-difference block for ``departures``, from their own rows alone.

    No movement frame, because the take-off cancels: each column is one
    planning value minus another and nothing outside the subject's row is
    read. Indexed like ``departures`` and carrying exactly
    ``PLAN_DIFF_FEATURES``, NaN where either planning value is missing and
    with no flag of its own -- ``lag_missing`` already says that.
    """
    out = pd.DataFrame(index=departures.index)
    # The registry's pair is in subtraction order, the column name's order.
    for name in PLAN_DIFF_FEATURES:
        minuend, subtrahend = DIFF_SOURCES[name]
        out[name] = _lag_seconds(departures[minuend], departures[subtrahend])
    return out[list(PLAN_DIFF_FEATURES)]
