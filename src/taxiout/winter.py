# SPDX-License-Identifier: GPL-3.0-only
"""Winter classes from the METAR, and a layer on the prediction that reads them.

Candidate C2a of ``reports/preregistered-2026-10-04.md``, section 3. From the
METAR already in the deposit (``taxiout.metar``), each station-hour takes one
class, in this order of priority:

1. ``fz_precip``: FZRA, FZDZ or PL in the present weather;
2. ``snow_persistent``: SN, SG or GS in the hour and in at least four of the
   previous twelve hours;
3. ``snow``: SN, SG or GS;
4. ``fz_fog``: FZFG;
5. ``cold_wet``: RA or DZ at or below +3 C;
6. ``frost_risk``: at or below +1 C, a dew-point spread of at most 2 C, and
   no precipitation code;
7. ``other``.

Codes are matched on the present-weather groups at the aerodrome, the groups
prefixed ``VC`` dropped, exactly as ``taxiout.metar`` matches its regime
flags, and each condition is the worst reading in the hour: an hour is in a
class when any of its reports is. "The previous twelve hours" are the twelve
clock hours before the hour, counted over the hours that have a report of
their own. The class is carried forward into hours without a report for up
to ``taxiout.metar.MAX_CARRY_H`` hours, as the regime flags are; beyond that
the hour has no class, and a departure in it is treated as ``other`` and left
unchanged. A departure reads the hour its take-off falls in.

**The layer.** For a row of airport a in class c other than ``other``, the
prediction gains Z * m(a, c) + (1 - Z) * m(c), with Z = n(a, c) / (n(a, c) +
500), where m is the mean residual (recorded minus predicted) of the rows the
layer is fitted on and n counts them. Which rows it is fitted on is the
caller's: the pre-registration names the submission rung's out-of-fold
predictions over the fitting months' rows that neither tail rule covers and
whose recorded taxi-out is 4 h or less.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from taxiout.metar import MAX_CARRY_H, PRECIP_CODES

CLASSES: tuple[str, ...] = (
    "fz_precip",
    "snow_persistent",
    "snow",
    "fz_fog",
    "cold_wet",
    "frost_risk",
    "other",
)
OTHER = "other"

WINTER_CLASS = "winter_class"
SOURCES: dict[str, tuple[str, ...]] = {
    WINTER_CLASS: (
        "MVT_TIME_UTC_mvt",
        "ADEP_mvt",
        "METAR.station",
        "METAR.valid",
        "METAR.tmpf",
        "METAR.dwpf",
        "METAR.wxcodes",
    )
}


def thresholds() -> dict[str, float]:
    """The numbers section 3 fixes, in its words.

    "RA or DZ at or below +3 C"; "at or below +1 C, dew-point spread at most
    2 C"; "in at least four of the previous twelve hours"; "Z = n(a, c) /
    (n(a, c) + 500)". Held in a function so that a new module carries no
    module-level number before the commit that registers it.
    """
    return {
        "cold_wet_c": 3.0,
        "frost_c": 1.0,
        "frost_spread_c": 2.0,
        "persistent_hours": 4,
        "persistent_window_h": 12,
        "prior_rows": 500.0,
    }


def _matches(wx: pd.Series, codes: tuple[str, ...]) -> np.ndarray:
    """Whether a group at the aerodrome (not prefixed ``VC``) carries a code."""
    pattern = re.compile("|".join(re.escape(c) for c in codes))
    out = np.zeros(len(wx), dtype=bool)
    for i, v in enumerate(wx.astype(object).where(wx.notna(), "")):
        groups = [g for g in str(v).split() if not g.startswith("VC")]
        out[i] = any(pattern.search(g) for g in groups)
    return out


def report_conditions(metar: pd.DataFrame) -> pd.DataFrame:
    """Each report's conditions, before the hour and the priority are applied."""
    th = thresholds()
    wx = metar["wxcodes"]
    tmpc = (pd.to_numeric(metar["tmpf"], errors="coerce") - 32.0) * 5.0 / 9.0
    dwpc = (pd.to_numeric(metar["dwpf"], errors="coerce") - 32.0) * 5.0 / 9.0
    rain = _matches(wx, ("RA", "DZ"))
    any_precip = _matches(wx, PRECIP_CODES)
    cold = (tmpc <= th["cold_wet_c"]).fillna(False).to_numpy()
    frost = (
        ((tmpc <= th["frost_c"]) & ((tmpc - dwpc) <= th["frost_spread_c"]))
        .fillna(False)
        .to_numpy()
    )
    return pd.DataFrame(
        {
            "station": metar["station"].astype(str).to_numpy(),
            "hour": pd.to_datetime(metar["valid"], utc=True).dt.floor("h").to_numpy(),
            "fz_precip": _matches(wx, ("FZRA", "FZDZ", "PL")),
            "snow": _matches(wx, ("SN", "SG", "GS")),
            "fz_fog": _matches(wx, ("FZFG",)),
            "cold_wet": rain & cold,
            "frost_risk": frost & ~any_precip,
        }
    )


def classes_by_hour(
    metar: pd.DataFrame, *, max_carry_h: int = MAX_CARRY_H
) -> pd.DataFrame:
    """One class per (station, hour), carried forward into hours without a report.

    Columns ``station``, ``hour``, ``winter_class`` and ``age_h``; hours more
    than ``max_carry_h`` past the last report are absent.
    """
    th = thresholds()
    r = report_conditions(metar)
    flags = ["fz_precip", "snow", "fz_fog", "cold_wet", "frost_risk"]
    hourly = r.groupby(["station", "hour"], sort=True)[flags].max().reset_index()
    out = []
    window = int(th["persistent_window_h"])
    for station, g in hourly.groupby("station", sort=True):
        g = g.set_index("hour").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
        h = g.reindex(full)
        present = h["snow"].notna().to_numpy()
        snow = h["snow"].eq(True).to_numpy()
        # snow hours among the twelve before, the hour itself excluded
        csum = np.concatenate([[0], np.cumsum(snow.astype(int))])
        idx = np.arange(len(h))
        before = csum[idx] - csum[np.maximum(idx - window, 0)]
        persistent = snow & (before >= int(th["persistent_hours"]))
        cls = np.select(
            [
                h["fz_precip"].eq(True).to_numpy(),
                persistent,
                snow,
                h["fz_fog"].eq(True).to_numpy(),
                h["cold_wet"].eq(True).to_numpy(),
                h["frost_risk"].eq(True).to_numpy(),
            ],
            list(CLASSES[:6]),
            default=OTHER,
        ).astype(object)
        cls[~present] = None
        last = pd.Series(np.where(present, idx, np.nan)).ffill().to_numpy()
        age = idx - last
        carried = pd.Series(cls).ffill(limit=max_carry_h).to_numpy()
        keep = np.isfinite(age) & (age <= max_carry_h)
        out.append(
            pd.DataFrame(
                {
                    "station": station,
                    "hour": full[keep],
                    WINTER_CLASS: carried[keep],
                    "age_h": age[keep],
                }
            )
        )
    if not out:
        return pd.DataFrame(columns=["station", "hour", WINTER_CLASS, "age_h"])
    return pd.concat(out, ignore_index=True)


def winter_class(departures: pd.DataFrame, hourly: pd.DataFrame) -> pd.Series:
    """The class of the hour each take-off falls in; ``other`` where the hour has none."""
    key = pd.DataFrame(
        {
            "station": departures["ADEP_mvt"].astype(str).to_numpy(),
            "hour": pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
            .dt.floor("h")
            .to_numpy(),
        }
    )
    got = key.merge(
        hourly[["station", "hour", WINTER_CLASS]], on=["station", "hour"], how="left"
    )
    values = got[WINTER_CLASS].where(got[WINTER_CLASS].notna(), OTHER).to_numpy()
    return pd.Series(values, index=departures.index, dtype=object, name=WINTER_CLASS)


@dataclass
class WinterLayer:
    """m(a, c), n(a, c) and m(c), fitted on residuals, applied as section 3 says."""

    by_cell: dict = field(default_factory=dict)  # (airport, class) -> (mean, n)
    by_class: dict = field(default_factory=dict)  # class -> mean

    def fit(self, airport, cls, residual) -> "WinterLayer":
        frame = pd.DataFrame(
            {
                "a": np.asarray(airport, dtype=object),
                "c": np.asarray(cls, dtype=object),
                "r": np.asarray(residual, dtype=float),
            }
        )
        frame = frame[frame["c"] != OTHER]
        if frame["r"].isna().any():
            raise ValueError("a residual the layer is fitted on is missing")
        cell = frame.groupby(["a", "c"], sort=True)["r"].agg(["mean", "size"])
        self.by_cell = {k: (float(v["mean"]), int(v["size"])) for k, v in cell.iterrows()}
        self.by_class = {
            k: float(v) for k, v in frame.groupby("c", sort=True)["r"].mean().items()
        }
        return self

    def adjustment(self, airport, cls) -> np.ndarray:
        """What the layer adds to each row; 0 in ``other`` and in a class it never saw."""
        k = thresholds()["prior_rows"]
        a = pd.Series(np.asarray(airport, dtype=object)).astype(str)
        c = pd.Series(np.asarray(cls, dtype=object)).astype(str)
        key = a + "\x1f" + c
        cell_mean = {f"{ka}\x1f{kc}": v[0] for (ka, kc), v in self.by_cell.items()}
        cell_n = {f"{ka}\x1f{kc}": v[1] for (ka, kc), v in self.by_cell.items()}
        m_ac = key.map(cell_mean).fillna(0.0).to_numpy(dtype=float)
        n_ac = key.map(cell_n).fillna(0.0).to_numpy(dtype=float)
        m_c = c.map(self.by_class)
        seen = m_c.notna().to_numpy() & (c != OTHER).to_numpy()
        z = n_ac / (n_ac + k)
        out = z * m_ac + (1.0 - z) * m_c.fillna(0.0).to_numpy(dtype=float)
        return np.where(seen, out, 0.0)

    def apply(self, pred, airport, cls) -> np.ndarray:
        return np.asarray(pred, dtype=float) + self.adjustment(airport, cls)


__all__ = [
    "CLASSES",
    "OTHER",
    "SOURCES",
    "WINTER_CLASS",
    "WinterLayer",
    "classes_by_hour",
    "report_conditions",
    "thresholds",
    "winter_class",
]
