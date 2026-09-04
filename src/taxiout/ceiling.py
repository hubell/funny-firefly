# SPDX-License-Identifier: GPL-3.0-only
"""The ceiling: how much of a rung's residual sits in the runway sequence.

An oracle diagnostic, never a feature. It reads the residuals of the
*other* departures on the subject's runway -- which are their taxi times,
minus what the model already knew -- and asks two things of them on the
forward splits' validation rows:

- **lag-1 autocorrelation**: the correlation between a departure's residual
  and the residual of the departure before it on the same runway, by the
  take-off gap between them. A queue is a run of departures late together;
  if the residuals of consecutive take-offs at ninety-second spacing are
  uncorrelated, there is no queue signal left to find.
- **the oracle regression**: the RMSE after regressing the residual on the
  mean residual of the other same-runway departures within fifteen minutes
  either side, the subject excluded. That is what a model could reach if it
  knew the neighbours' outcomes exactly, which no admissible feature does:
  it bounds what any queue model built from the sequence could add.

Both are computed by ``scripts/report_congestion.py`` from residuals the
ladder writes under ``data/processed/residuals/``, gitignored because they
carry the target.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.windows import Sorted, centred_mean, seconds

GAP_BUCKETS_MIN: tuple[tuple[float, float], ...] = ((0, 2), (2, 5), (5, 15))
HALF_WINDOW_S = 15 * 60
MIN_ROWS = 30

REQUIRED = ("icao", "runway", "takeoff", "residual")


def _check(res: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED:
        if c not in res.columns:
            raise KeyError(f"residuals need {c!r}")
    out = res[list(REQUIRED)].copy()
    out["takeoff"] = pd.to_datetime(out["takeoff"], utc=True)
    out["residual"] = pd.to_numeric(out["residual"], errors="coerce")
    return out[out["residual"].notna()].reset_index(drop=True)


def bucket_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g} min"


def lag1_autocorrelation(res: pd.DataFrame, *, by_airport: bool = False) -> pd.DataFrame:
    """Correlation of consecutive same-runway residuals, by the gap between them.

    One row per bucket (and per airport when asked): the number of pairs and
    the Pearson correlation, NaN with fewer than ``MIN_ROWS`` pairs.
    """
    r = _check(res).sort_values(["icao", "runway", "takeoff"], kind="stable")
    same = (r["icao"] == r["icao"].shift()) & (r["runway"] == r["runway"].shift())
    prev = r["residual"].shift()
    gap_min = (seconds(r["takeoff"]) - seconds(r["takeoff"].shift().bfill())) / 60.0
    pairs = pd.DataFrame(
        {
            "icao": r["icao"].to_numpy(),
            "residual": r["residual"].to_numpy(),
            "previous": prev.to_numpy(),
            "gap_min": np.where(same.to_numpy(), gap_min, np.nan),
        }
    )
    pairs = pairs[np.isfinite(pairs["gap_min"])]
    rows = []
    groups = pairs.groupby("icao", sort=True) if by_airport else [("all", pairs)]
    for icao, g in groups:
        for lo, hi in GAP_BUCKETS_MIN:
            sel = g[(g["gap_min"] >= lo) & (g["gap_min"] < hi)]
            n = int(len(sel))
            corr = (
                float(np.corrcoef(sel["residual"], sel["previous"])[0, 1])
                if n >= MIN_ROWS
                else np.nan
            )
            rows.append(
                {"icao": icao, "bucket": bucket_label(lo, hi), "n": n, "corr": corr}
            )
    return pd.DataFrame(rows)


def neighbour_mean(
    res: pd.DataFrame, *, half_window_s: float = HALF_WINDOW_S
) -> np.ndarray:
    """Mean residual of the other same-runway departures within the window; NaN with none."""
    r = _check(res)
    key = r["icao"].astype(str) + "|" + r["runway"].astype(str)
    events = Sorted.build(key, r["takeoff"], values={"res": r["residual"]})
    mean, _n = centred_mean(
        events,
        "res",
        key,
        r["takeoff"],
        half_window_s,
        self_value=r["residual"].to_numpy(dtype=float),
    )
    return mean


def oracle(res: pd.DataFrame, *, half_window_s: float = HALF_WINDOW_S) -> dict:
    """The RMSE before and after regressing the residual on its neighbours' mean.

    One ordinary least squares fit, pooled, in sample -- this is a bound, not
    a model. Rows with no neighbour keep their residual. Returns the RMSE
    before and after, the slope, the intercept, the share of rows with a
    neighbour, and the per-airport RMSE before and after from the same fit.
    """
    r = _check(res)
    m = neighbour_mean(r, half_window_s=half_window_s)
    y = r["residual"].to_numpy(dtype=float)
    has = np.isfinite(m)
    if has.sum() >= MIN_ROWS:
        a = np.column_stack([np.ones(int(has.sum())), m[has]])
        coef, *_ = np.linalg.lstsq(a, y[has], rcond=None)
        intercept, slope = float(coef[0]), float(coef[1])
    else:
        intercept, slope = 0.0, 0.0
    after = y.copy()
    after[has] = y[has] - (intercept + slope * m[has])
    per_airport = []
    for icao, idx in r.groupby("icao", sort=True).indices.items():
        per_airport.append(
            {
                "icao": str(icao),
                "n": int(len(idx)),
                "with_neighbour": float(has[idx].mean()),
                "rmse_before": float(np.sqrt(np.mean(y[idx] ** 2))),
                "rmse_after": float(np.sqrt(np.mean(after[idx] ** 2))),
            }
        )
    return {
        "n": int(len(y)),
        "with_neighbour": float(has.mean()) if len(y) else 0.0,
        "slope": slope,
        "intercept": intercept,
        "rmse_before": float(np.sqrt(np.mean(y**2))) if len(y) else np.nan,
        "rmse_after": float(np.sqrt(np.mean(after**2))) if len(y) else np.nan,
        "per_airport": pd.DataFrame(per_airport),
    }
