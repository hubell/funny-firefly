# SPDX-License-Identifier: GPL-3.0-only
"""Thunderstorms at the airport, from the METAR already in the deposit.

Candidate C7 of the amendment of 4 October 2026 to the board pre-registration
(``preregistration_2026-10-04_c7_c8.md``). A METAR's present weather observes
a thunderstorm at the airport itself, and ``METAR.wxcodes`` is already fetched
and registered, so this reads it alone and fetches nothing. Per airport-hour,
each the worst reading in the hour, carried forward into hours without a
report and marked missing beyond that exactly as ``taxiout.metar`` carries
the regime flags (up to ``taxiout.metar.MAX_CARRY_H`` hours):

- ``wx_ts``: 1 if a present-weather group at the aerodrome (not prefixed
  ``VC``) contains ``TS``;
- ``wx_vcts``: 1 if a group is ``VCTS`` and ``wx_ts`` is 0;
- ``wx_heavy``: 1 if a group carries the heavy-intensity prefix ``+``;
- ``wx_ts_prev3``: how many of the three hours before had ``wx_ts`` 1.

A departure reads the hour its take-off falls in, as for the regime flags,
and ``wx_ts_prev_hour``, the ``wx_ts`` of the hour before it. An hour whose
reading is missing is NaN in every column read from it; ``wx_ts_prev3``
counts the hours that had ``wx_ts`` 1, so a missing hour counts as not one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.metar import MAX_CARRY_H

THUNDER_FEATURES: tuple[str, ...] = (
    "wx_ts",
    "wx_vcts",
    "wx_heavy",
    "wx_ts_prev3",
    "wx_ts_prev_hour",
)

SOURCES: dict[str, tuple[str, ...]] = {
    c: ("MVT_TIME_UTC_mvt", "ADEP_mvt", "METAR.station", "METAR.valid", "METAR.wxcodes")
    for c in THUNDER_FEATURES
}


def report_flags(metar: pd.DataFrame) -> pd.DataFrame:
    """The three flags of each report, before the hour is taken."""
    ts, vcts, heavy = [], [], []
    for v in metar["wxcodes"].astype(object).where(metar["wxcodes"].notna(), ""):
        groups = str(v).split()
        here = [g for g in groups if not g.startswith("VC")]
        t = any("TS" in g for g in here)
        ts.append(t)
        vcts.append(any(g == "VCTS" for g in groups))
        heavy.append(any(g.startswith("+") for g in groups))
    return pd.DataFrame(
        {
            "station": metar["station"].astype(str).to_numpy(),
            "hour": pd.to_datetime(metar["valid"], utc=True).dt.floor("h").to_numpy(),
            "ts": np.asarray(ts, dtype=float),
            "vcts": np.asarray(vcts, dtype=float),
            "heavy": np.asarray(heavy, dtype=float),
        }
    )


def thunder_by_hour(
    metar: pd.DataFrame, *, max_carry_h: int = MAX_CARRY_H
) -> pd.DataFrame:
    """The flags per (station, hour), worst in the hour, carried forward.

    ``wx_vcts`` is set only where the hour's ``wx_ts`` is 0, after the hour
    is taken: a thunderstorm at the aerodrome in any report of the hour
    outranks one in the vicinity. ``wx_ts_prev3`` is over the carried-forward
    hours. Hours beyond the carry are absent.
    """
    f = report_flags(metar)
    hourly = (
        f.groupby(["station", "hour"], sort=True)[["ts", "vcts", "heavy"]]
        .max()
        .reset_index()
    )
    out = []
    for station, g in hourly.groupby("station", sort=True):
        g = g.set_index("hour").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
        r = g.reindex(full)
        present = r["ts"].notna().to_numpy()
        idx = np.arange(len(r))
        last = pd.Series(np.where(present, idx, np.nan)).ffill().to_numpy()
        age = idx - last
        r = r.ffill(limit=max_carry_h)
        keep = np.isfinite(age) & (age <= max_carry_h)
        ts = np.where(keep, r["ts"].to_numpy(dtype=float), np.nan)
        vcts = np.where(keep, r["vcts"].to_numpy(dtype=float), np.nan)
        vcts = np.where(ts == 1.0, 0.0, vcts)
        heavy = np.where(keep, r["heavy"].to_numpy(dtype=float), np.nan)
        one = np.nan_to_num(ts, nan=0.0)
        prev = np.zeros(len(r))
        for lag in (1, 2, 3):
            prev[lag:] += one[:-lag]
        prev_hour = np.full(len(r), np.nan)
        prev_hour[1:] = ts[:-1]
        out.append(
            pd.DataFrame(
                {
                    "station": station,
                    "hour": full,
                    "wx_ts": ts,
                    "wx_vcts": vcts,
                    "wx_heavy": heavy,
                    "wx_ts_prev3": prev,
                    "wx_ts_prev_hour": prev_hour,
                    "keep": keep,
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=["station", "hour", *THUNDER_FEATURES])
    got = pd.concat(out, ignore_index=True)
    return got.drop(columns="keep")


def thunder_features(departures: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """The C7 block for ``departures``, by the hour their take-off falls in.

    An hour beyond the carry is NaN in the three flags of the hour itself;
    ``wx_ts_prev3`` and ``wx_ts_prev_hour`` are read from the hours before,
    and are NaN only where the station's record does not reach the hour.
    """
    key = pd.DataFrame(
        {
            "station": departures["ADEP_mvt"].astype(str).to_numpy(),
            "hour": pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
            .dt.floor("h")
            .to_numpy(),
        }
    )
    got = key.merge(hourly, on=["station", "hour"], how="left")
    out = pd.DataFrame(index=departures.index)
    for c in THUNDER_FEATURES:
        out[c] = got[c].to_numpy(dtype=float)
    return out[list(THUNDER_FEATURES)]


__all__ = ["SOURCES", "THUNDER_FEATURES", "thunder_by_hour", "thunder_features"]
