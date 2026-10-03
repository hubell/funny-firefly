# SPDX-License-Identifier: GPL-3.0-only
"""Weather as a regime indicator, from the Iowa State Mesonet's ASOS archive.

Gate 2b of ``docs/plan.md``. The organisers name airport procedures and load
among the drivers of taxi-out, and the ranking set's first week of January
2026 at Schiphol has the signature of winter operations. What weather does to
taxi-out is not continuous -- a knot of wind does nothing -- but a regime: low
visibility procedures space departures out, precipitation slows the surface,
freezing conditions add de-icing and its queue. So the weather enters as
three flags per airport-hour and nothing else, per the literature position in
``docs/prior_art.md``:

- **visibility band**: 0 at 8 km or more, 1 between 1.5 and 8 km, 2 below
  1.5 km, the range where low-visibility procedures are in force;
- **precipitation**: any present-weather code that is rain, snow, drizzle,
  showers, thunderstorm, hail or unknown precipitation, or a measured
  accumulation in the hour;
- **freezing**: an air temperature at or below 0 C, or a freezing or frozen
  present-weather code.

Each is the worst reading in the hour, and a subject reads the flags of the
hour its take-off falls in. An hour with no report inherits the previous
report for up to ``MAX_CARRY_H`` hours and says how old it is; beyond that
the flags are missing and a flag says so.

The data are METARs, which the airports themselves publish; the Mesonet
serves an archive of them, and ``scripts/fetch_metar.py`` records the request
it made and the terms it read in the cache beside the data. The cache is
under ``data/raw/metar/`` and gitignored with the OpenStreetMap cache, for
the same reason: a reader regenerates it from the recorded request rather
than trusting a copy.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# The ASOS archive's own column names, as fetched. These are the only
# external columns a feature may read, and the registry names them under the
# ``METAR.`` phase so the audit can hold them to this list.
FIELDS: tuple[str, ...] = (
    "station",
    "valid",
    "tmpf",
    "dwpf",
    "vsby",
    "p01i",
    "wxcodes",
    "skyc1",
    "skyl1",
)

WEATHER_FEATURES: tuple[str, ...] = (
    "wx_vis_band",
    "wx_precip",
    "wx_freezing",
    "wx_age_h",
    "wx_missing",
)

SOURCES: dict[str, tuple[str, ...]] = {
    c: (
        "MVT_TIME_UTC_mvt",
        "ADEP_mvt",
        # The archive's station code, the key the weather is joined on to
        # ``ADEP_mvt``.
        "METAR.station",
        "METAR.valid",
        "METAR.vsby",
        "METAR.tmpf",
        "METAR.p01i",
        "METAR.wxcodes",
    )
    for c in WEATHER_FEATURES
}

# Visibility bands, metres. 1,500 m is where low-visibility procedures are
# typically in force; 8 km is ordinary daylight visibility.
VIS_LOW_M = 1500.0
VIS_HIGH_M = 8000.0
STATUTE_MILE_M = 1609.344

# Present-weather codes. Precipitation is anything that falls; freezing is
# anything frozen or freezing. ``UP`` is unknown precipitation.
PRECIP_CODES = ("RA", "SN", "DZ", "SH", "TS", "GR", "GS", "PL", "SG", "IC", "UP")
FREEZING_CODES = ("FZ", "SN", "SG", "IC", "PL", "GR", "GS")

# How many hours a report may be carried forward into hours without one.
MAX_CARRY_H = 3


def load_metar(root: Path) -> pd.DataFrame:
    """Every cached ASOS file under ``root``, as one frame with UTC stamps."""
    frames = []
    for f in sorted(Path(root).glob("*.csv")):
        df = pd.read_csv(f, na_values=["M"], keep_default_na=True)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(FIELDS))
    out = pd.concat(frames, ignore_index=True)
    out["valid"] = pd.to_datetime(out["valid"], utc=True)
    return out.drop_duplicates(["station", "valid"]).sort_values(["station", "valid"])


def _codes_match(wx: pd.Series, codes: tuple[str, ...]) -> np.ndarray:
    """Whether any present-weather group at the aerodrome carries one of ``codes``.

    A METAR's present weather is a list of groups; a group prefixed ``VC``
    is weather in the vicinity, eight to sixteen kilometres out, and is not
    weather on the surface the subject is taxiing on -- ``VCSH`` is showers
    nearby, not rain here -- so those groups are dropped before matching.
    """
    pattern = re.compile("|".join(re.escape(c) for c in codes))
    out = np.zeros(len(wx), dtype=bool)
    for i, v in enumerate(wx.astype(object).where(wx.notna(), "")):
        groups = [g for g in str(v).split() if not g.startswith("VC")]
        out[i] = any(pattern.search(g) for g in groups)
    return out


def flags(metar: pd.DataFrame) -> pd.DataFrame:
    """The three flags per report, from the fields as fetched."""
    vis_m = pd.to_numeric(metar["vsby"], errors="coerce") * STATUTE_MILE_M
    band = np.select([vis_m < VIS_LOW_M, vis_m < VIS_HIGH_M], [2.0, 1.0], default=0.0)
    band = np.where(vis_m.isna(), np.nan, band)
    p01 = pd.to_numeric(metar["p01i"], errors="coerce")
    precip = _codes_match(metar["wxcodes"], PRECIP_CODES) | (p01.fillna(0) > 0).to_numpy()
    tmpc = (pd.to_numeric(metar["tmpf"], errors="coerce") - 32.0) * 5.0 / 9.0
    freezing = (
        _codes_match(metar["wxcodes"], FREEZING_CODES)
        | (tmpc <= 0).fillna(False).to_numpy()
    )
    return pd.DataFrame(
        {
            "station": metar["station"].astype(str).to_numpy(),
            "valid": pd.to_datetime(metar["valid"], utc=True).to_numpy(),
            "vis_band": band,
            "precip": precip.astype(float),
            "freezing": freezing.astype(float),
        }
    )


def regime_by_hour(
    metar: pd.DataFrame, *, max_carry_h: int = MAX_CARRY_H
) -> pd.DataFrame:
    """The worst flags per (station, hour), carried forward into empty hours.

    Indexed by ``(station, hour)``; ``age_h`` is how many hours old the
    reading is, 0 for an hour with its own report; hours beyond
    ``max_carry_h`` from the last report are absent, so a join on them yields
    missing flags.
    """
    f = flags(metar)
    f["hour"] = pd.to_datetime(f["valid"], utc=True).dt.floor("h")
    hourly = (
        f.groupby(["station", "hour"], sort=True)
        .agg(
            vis_band=("vis_band", "max"),
            precip=("precip", "max"),
            freezing=("freezing", "max"),
        )
        .reset_index()
    )
    out = []
    for station, g in hourly.groupby("station", sort=True):
        g = g.set_index("hour").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq="h", tz="UTC")
        r = g.reindex(full)
        present = r["vis_band"].notna() | r["precip"].notna()
        # age: hours since the last present hour
        idx = np.arange(len(r))
        last = pd.Series(np.where(present, idx, np.nan)).ffill().to_numpy()
        age = idx - last
        r = r.ffill(limit=max_carry_h)
        r["age_h"] = age
        r["station"] = station
        r = r[np.isfinite(age) & (age <= max_carry_h)]
        out.append(r.rename_axis("hour").reset_index())
    if not out:
        return pd.DataFrame(
            columns=["station", "hour", "vis_band", "precip", "freezing", "age_h"]
        )
    return pd.concat(out, ignore_index=True)[
        ["station", "hour", "vis_band", "precip", "freezing", "age_h"]
    ]


def weather_features(
    departures: pd.DataFrame, hourly: pd.DataFrame | None
) -> pd.DataFrame:
    """The weather block for ``departures``, by the hour their take-off falls in.

    ``hourly`` is ``regime_by_hour``'s output. ``None`` yields a block that is
    NaN with ``wx_missing`` set, which is what a caller without the cache gets
    and what the ladder skips the weather rung on.
    """
    out = pd.DataFrame(index=departures.index)
    if hourly is None or hourly.empty:
        for c in WEATHER_FEATURES:
            out[c] = np.nan
        out["wx_missing"] = 1.0
        return out[list(WEATHER_FEATURES)]
    key = pd.DataFrame(
        {
            "station": departures["ADEP_mvt"].astype(str).to_numpy(),
            "hour": pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
            .dt.floor("h")
            .to_numpy(),
        },
        index=departures.index,
    )
    got = key.merge(hourly, on=["station", "hour"], how="left")
    got.index = departures.index
    out["wx_vis_band"] = got["vis_band"].to_numpy()
    out["wx_precip"] = got["precip"].to_numpy()
    out["wx_freezing"] = got["freezing"].to_numpy()
    out["wx_age_h"] = got["age_h"].to_numpy()
    out["wx_missing"] = got["vis_band"].isna().astype(float).to_numpy()
    return out[list(WEATHER_FEATURES)]
