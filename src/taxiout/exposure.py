# SPDX-License-Identifier: GPL-3.0-only
"""Regime exposure: how a ranking month differs from its training-year twin.

The plan-lag block found that the ranking months' departures take off later
against their plans than the same calendar month of 2025 did. Take-off minus
a planning value is taxi plus the pushback delay against that plan, and the
exposed fields cannot say which of the two moved: the block time is hidden
for a ranking departure, so the split is unobservable. They can, however,
**bound** it, and that is all this module does. Nothing here is a feature;
it is read only by ``scripts/report_congestion.py``.

Three quantities per airport, per ranking month against the same calendar
month of 2025:

1. **The plan shift.** The change in the mean of take-off minus each of the
   three planning values -- the flight plan's first and last off-block
   estimates and the schedule.
2. **The control.** Two quantities that involve no departure plan and are
   exposed on both sides: the change in arrivals' mean taxi-in at the same
   airport-month, and the change in arrivals' mean landing minus schedule.
   Taxi-in is the surface running slower or not; arrivals' lateness is the
   day running late or not. If the plan shift is the surface, taxi-in moves
   with it; if it is the day, arrivals' lateness moves with it.
3. **The worst case.** The airport's training-year slope of taxi-out on
   take-off minus the first off-block estimate, times the plan shift: what
   the model would mispredict if the *entire* shift were pushback delay and
   none of it taxi. That bias added in quadrature to the airport's own
   ex-flagged RMSE is the worst-case cost. It is a bound, not an estimate --
   the true shift is a mixture, and the control says which way.

The row set: departures unflagged by the record-quality term and carrying a
flight-plan record, with each lag then windowed to ``[LAG_LO_S, LAG_HI_S]``
**on its own**. A departure taking off half an hour before its own off-block
estimate is a record artefact rather than a fast taxi, and one taking off two
hours after it is the record-quality tail, which has its own term.

The windowing is per lag and not across the three together, which matters
more than it looks. A row whose schedule lag runs past two hours is usually a
row whose take-off ran late, so requiring all three lags in the window drops
late rows -- and it drops more of them on the side that ran later, which is
the side whose shift is being measured. Each lag is therefore measured on the rows
where that lag is defined and in range, and the report prints both windows'
keep rates and what the joint one would have reported, so the reader can see
the size of the difference rather than take it on faith.

Arrivals carry their own windows: a taxi-in between ``TAXI_IN_LO_S`` and
``TAXI_IN_HI_S``, and a lateness within ``ARRIVAL_LATE_S`` either way --
symmetric, because an arrival landing early is ordinary, where a departure
taking off before its off-block estimate is not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.validation import month_of

# Each ranking month against the same calendar month of the training year.
MONTH_PAIRS: tuple[tuple[str, str], ...] = (
    ("2026-01", "2025-01"),
    ("2026-07", "2025-07"),
)

# The planning values, and the name each lag carries.
PLANS: tuple[tuple[str, str], ...] = (
    ("eobt", "EOBT_1_flt"),
    ("lobt", "LOBT_flt"),
    ("sched", "SCHED_TIME_UTC_mvt"),
)
# What "has a flight-plan record" means: the two flight-side planning values.
PLAN_RECORD: tuple[str, ...] = ("EOBT_1_flt", "LOBT_flt")

LAG_LO_S = -30 * 60.0
LAG_HI_S = 2 * 3600.0
TAXI_IN_LO_S = 60.0
TAXI_IN_HI_S = 3600.0
ARRIVAL_LATE_S = 2 * 3600.0

LAG_COLUMNS: tuple[str, ...] = tuple(f"lag_{name}_s" for name, _ in PLANS)
ARRIVAL_COLUMNS: tuple[str, ...] = ("arr_taxi_in_s", "arr_late_s")

# A cell with fewer rows than this on either side is not compared.
MIN_ROWS = 100


def _utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True)


# --------------------------------------------------------------------------
# the rows


def departure_rows(
    frame: pd.DataFrame, flagged: pd.Series | np.ndarray | None = None
) -> pd.DataFrame:
    """One row per departure: airport, month, whether it is in the record set, its lags.

    ``flagged`` is the record-quality term's mask, fitted on the training
    side and predicted on both; ``None`` keeps every row and the report says
    so. ``in_record`` is unflagged and carrying a flight-plan record. Each
    lag column is that lag where the row is in the record set *and* that lag
    is in range, and NaN otherwise, so every mean is over the rows where its
    own quantity is defined and the count of each is the count of those rows.
    """
    for name, col in PLANS:
        if col not in frame.columns:
            raise KeyError(f"departures need {col!r}")
    if "ADEP_mvt" not in frame.columns or "MVT_TIME_UTC_mvt" not in frame.columns:
        raise KeyError("departures need 'ADEP_mvt' and 'MVT_TIME_UTC_mvt'")

    out = pd.DataFrame(index=frame.index)
    out["icao"] = frame["ADEP_mvt"].astype(str)
    out["month"] = month_of(frame["MVT_TIME_UTC_mvt"]).to_numpy()
    t = _utc(frame["MVT_TIME_UTC_mvt"])

    base = np.ones(len(frame), dtype=bool)
    if flagged is not None:
        base &= ~np.asarray(flagged, dtype=bool)
    for col in PLAN_RECORD:
        base &= frame[col].notna().to_numpy()
    out["in_record"] = base
    for name, col in PLANS:
        lag = (t - _utc(frame[col])).dt.total_seconds().to_numpy(dtype=float)
        inside = base & np.isfinite(lag) & (lag >= LAG_LO_S) & (lag <= LAG_HI_S)
        out[f"lag_{name}_s"] = np.where(inside, lag, np.nan)
    return out


def window_losses(rows: pd.DataFrame, *, lag: str = "lag_eobt_s") -> pd.DataFrame:
    """Per (airport, month): the record set, what the window kept, what a joint one would.

    ``in_window`` is the rows this lag's own window keeps; ``in_joint`` is the
    rows a window across all three lags together would keep. The difference
    is the trap described in the module docstring, and the report prints both
    so the reader can see the filter is not eating the shift.
    """
    r = rows.copy()
    r["_joint"] = r[list(LAG_COLUMNS)].notna().all(axis=1)
    g = r.groupby(["icao", "month"], sort=True)
    return pd.DataFrame(
        {
            "in_record": g["in_record"].sum().astype(float),
            "in_window": g[lag].count().astype(float),
            # the worst of the three lags, so a claim about "each lag's own
            # window" is measured over each lag and not only over one of them
            "in_window_min": g[list(LAG_COLUMNS)].count().min(axis=1).astype(float),
            "in_joint": g["_joint"].sum().astype(float),
        }
    )


def joint_window_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """``rows`` as a single window across all three lags together would leave them.

    Every lag NaN unless all three are in range. This is the row set the
    module docstring argues against, kept so the report can print what it
    would have reported rather than asserting the difference.
    """
    out = rows.copy()
    joint = out[list(LAG_COLUMNS)].notna().all(axis=1)
    for c in LAG_COLUMNS:
        out[c] = out[c].where(joint, np.nan)
    return out


def arrival_rows(movements: pd.DataFrame) -> pd.DataFrame:
    """One row per arrival: its airport, month, taxi-in and lateness, windowed.

    Both quantities are exposed on the ranking side, and neither reads a
    departure's plan or a departure's hidden fields. A value outside its
    window is NaN rather than dropped, so each column keeps its own count.
    """
    arr = movements[movements["PHASE_mvt"].astype(str) == "ARR"]
    out = pd.DataFrame(index=arr.index)
    out["icao"] = arr["ADES_mvt"].astype(str)
    out["month"] = month_of(arr["MVT_TIME_UTC_mvt"]).to_numpy()
    txi = pd.to_numeric(arr["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(dtype=float)
    out["arr_taxi_in_s"] = np.where(
        (txi >= TAXI_IN_LO_S) & (txi <= TAXI_IN_HI_S), txi, np.nan
    )
    late = (
        (_utc(arr["MVT_TIME_UTC_mvt"]) - _utc(arr["SCHED_TIME_UTC_mvt"]))
        .dt.total_seconds()
        .to_numpy(dtype=float)
    )
    out["arr_late_s"] = np.where(np.abs(late) <= ARRIVAL_LATE_S, late, np.nan)
    return out


# --------------------------------------------------------------------------
# the shift


def cell_means(rows: pd.DataFrame, columns) -> pd.DataFrame:
    """Mean and count of each column per (airport, month); missing values ignored."""
    g = rows.groupby(["icao", "month"], sort=True)
    out = {}
    for c in columns:
        out[f"{c}_mean"] = g[c].mean()
        out[f"{c}_n"] = g[c].count().astype(float)
    return pd.DataFrame(out)


def shift_table(
    train_rows: pd.DataFrame,
    rank_rows: pd.DataFrame,
    columns,
    *,
    pairs: tuple[tuple[str, str], ...] = MONTH_PAIRS,
    min_rows: int = MIN_ROWS,
) -> pd.DataFrame:
    """Each column's mean on both sides and the shift, per airport and month pair.

    One row per (ranking month, airport). A column whose cell holds fewer
    than ``min_rows`` rows on either side is NaN for that airport rather than
    reported from a handful of movements.
    """
    columns = list(columns)
    a = cell_means(train_rows, columns)
    b = cell_means(rank_rows, columns)
    out = []
    for rank_month, train_month in pairs:
        icaos = sorted(
            {i for i, m in b.index if m == rank_month}
            & {i for i, m in a.index if m == train_month}
        )
        for icao in icaos:
            rec = {
                "icao": icao,
                "ranking_month": rank_month,
                "training_month": train_month,
            }
            for c in columns:
                n_a = float(a.loc[(icao, train_month), f"{c}_n"])
                n_b = float(b.loc[(icao, rank_month), f"{c}_n"])
                enough = n_a >= min_rows and n_b >= min_rows
                m_a = float(a.loc[(icao, train_month), f"{c}_mean"]) if enough else np.nan
                m_b = float(b.loc[(icao, rank_month), f"{c}_mean"]) if enough else np.nan
                rec[f"{c}_train"] = m_a
                rec[f"{c}_rank"] = m_b
                rec[f"{c}_shift"] = m_b - m_a
                rec[f"{c}_n_train"] = n_a
                rec[f"{c}_n_rank"] = n_b
            out.append(rec)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# the worst case


def lag_slopes(
    rows: pd.DataFrame,
    target: pd.Series | np.ndarray,
    *,
    lag: str = "lag_eobt_s",
    min_rows: int = MIN_ROWS,
) -> pd.DataFrame:
    """Per airport, the ordinary least squares slope of ``target`` on ``lag``.

    Fitted on the training year's rows only, within airport, over exactly the
    rows the shift is measured on: the lag column is already NaN outside the
    record set and outside its window. One number per airport -- how many
    seconds of taxi-out go with a second of plan lag.
    """
    y = pd.to_numeric(pd.Series(np.asarray(target), index=rows.index), errors="coerce")
    keep = np.isfinite(rows[lag].to_numpy(dtype=float)) & np.isfinite(
        y.to_numpy(dtype=float)
    )
    r = rows.loc[keep, ["icao", lag]].copy()
    r["y"] = y[keep].to_numpy(dtype=float)
    out = []
    for icao, g in r.groupby("icao", sort=True):
        x = g[lag].to_numpy(dtype=float)
        yy = g["y"].to_numpy(dtype=float)
        n = len(x)
        if n < min_rows or not np.isfinite(x).all() or x.var() <= 0:
            out.append({"icao": str(icao), "n": float(n), "slope": np.nan})
            continue
        xm, ym = x.mean(), yy.mean()
        slope = float(((x - xm) * (yy - ym)).sum() / ((x - xm) ** 2).sum())
        out.append({"icao": str(icao), "n": float(n), "slope": slope})
    return pd.DataFrame(out)


def worst_case(shift_s: float, slope: float, rmse_s: float) -> dict:
    """The bias if the whole shift were pushback delay, and what it would cost.

    The bias is a constant per airport, so it enters the RMSE in quadrature:
    an unbiased error of ``rmse_s`` plus a constant ``bias`` gives
    ``hypot(rmse_s, bias)``. The cost is the difference.
    """
    bias = float(slope) * float(shift_s)
    if not np.isfinite(bias) or not np.isfinite(rmse_s):
        return {
            "bias_s": bias,
            "rmse_s": rmse_s,
            "rmse_with_bias_s": np.nan,
            "cost_s": np.nan,
        }
    with_bias = float(np.hypot(float(rmse_s), bias))
    return {
        "bias_s": bias,
        "rmse_s": float(rmse_s),
        "rmse_with_bias_s": with_bias,
        "cost_s": with_bias - float(rmse_s),
    }


def pooled_cost(rows: pd.DataFrame) -> dict:
    """The same worst case pooled over airports, weighted by rows.

    ``rows`` carries ``n``, ``rmse_s`` and ``bias_s`` per airport. Biases of
    opposite sign do not cancel: each enters its own airport's squared error.
    """
    r = rows.dropna(subset=["n", "rmse_s", "bias_s"])
    if r.empty:
        return {"n": 0.0, "rmse_s": np.nan, "rmse_with_bias_s": np.nan, "cost_s": np.nan}
    n = r["n"].to_numpy(dtype=float)
    rmse = r["rmse_s"].to_numpy(dtype=float)
    bias = r["bias_s"].to_numpy(dtype=float)
    before = float(np.sqrt((n * rmse**2).sum() / n.sum()))
    after = float(np.sqrt((n * (rmse**2 + bias**2)).sum() / n.sum()))
    return {
        "n": float(n.sum()),
        "rmse_s": before,
        "rmse_with_bias_s": after,
        "cost_s": after - before,
    }
