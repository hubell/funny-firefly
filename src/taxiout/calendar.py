# SPDX-License-Identifier: GPL-3.0-only
"""Public holidays at the departure aerodrome, as a regime flag on the date.

The cheapest regime variable there is. A public holiday changes what an
airport is doing: the traffic mix moves from business to leisure, the banks
of departures move with it, and the ground staffing that pushes the aircraft
back is on a holiday roster. The day either side carries the leisure peak
that the holiday itself does not. Two flags per departure, from the date of
its take-off and the country of its departure aerodrome:

- ``holiday``: that date is a public holiday in that country;
- ``holiday_adjacent``: the day before or the day after is one. It is
  computed from the neighbouring dates alone, so a holiday is not adjacent
  to itself, and a holiday next to another holiday -- Christmas Day beside
  Boxing Day -- carries both flags.

It is worth about half a second in December and nothing in July. The block
is here because it is trivially cheap and correctly licensed, not because it
is large.

**A code dependency, not a dataset.** ``holidays`` (python-holidays) is
MIT-licensed and computes each country's calendar from published rules.
Nothing is fetched and nothing is cached, so unlike the METAR archive and the
OurAirports file there is no manifest to record and no external prefix in the
registry: the sources are the subject's own exposed fields and nothing else.

The calendars are read at country level. Germany, Spain and Switzerland
settle most of their public holidays by subdivision -- Land, comunidad
autónoma, canton -- and none of that is read here. A subdivision would be a
better feature and a worse-understood one, and the block is half a second
either way.

The date is the UTC date of the take-off; no timezone table is carried, for
the reason the time block gives. At Antalya and Istanbul, three hours ahead,
the first three hours of a local holiday therefore fall on the previous UTC
date, where the adjacency flag catches them and the holiday flag does not.
An eighth of each Turkish holiday lands on the wrong side of midnight, in
the quietest hours of it, and that is the price of not carrying a table.

Where ``holidays`` is not importable both columns are NaN throughout, not
zero, so the ladder skips the rung and the submission writer refuses --
exactly as they do without the METAR cache. An aerodrome with no country in
``COUNTRY`` is NaN for the same reason: an unmapped airport has no calendar,
and a zero would assert that its dates are ordinary.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

# Absolute imports, so this is the third-party package and not a sibling of
# this module, and an unrelated ``import calendar`` still finds the standard
# library.
try:
    import holidays
except ImportError:
    holidays = None

HOLIDAY_FEATURES: tuple[str, ...] = ("holiday", "holiday_adjacent")

SOURCES: dict[str, tuple[str, ...]] = {
    c: ("ADEP_mvt", "MVT_TIME_UTC_mvt") for c in HOLIDAY_FEATURES
}

# ISO 3166-1 alpha-2, one per aerodrome in ``taxiout.airports.AIRPORTS``.
COUNTRY: dict[str, str] = {
    "EDDF": "DE",
    "EDDM": "DE",
    "EGLL": "GB",
    "EHAM": "NL",
    "LEBL": "ES",
    "LEMD": "ES",
    "LFPG": "FR",
    "LIRF": "IT",
    "LSZH": "CH",
    "LTAI": "TR",
    "LTFM": "TR",
}


@lru_cache(maxsize=None)
def _holiday_days(country: str, first_year: int, last_year: int) -> np.ndarray:
    """The country's holidays over the inclusive year range, as days since the epoch.

    Cached because the ladder rebuilds the feature frame once per fold and a
    published calendar does not change between them.
    """
    dates = holidays.country_holidays(country, years=range(first_year, last_year + 1))
    return np.array(sorted(dates), dtype="datetime64[D]").astype(np.int64)


def holiday_features(departures: pd.DataFrame) -> pd.DataFrame:
    """The holiday block for ``departures``, by the date of their take-off.

    Indexed like ``departures``; carries exactly ``HOLIDAY_FEATURES``. NaN
    throughout without ``holidays``, and NaN on any row whose aerodrome is
    not in ``COUNTRY`` or whose take-off time is missing.
    """
    out = pd.DataFrame(index=departures.index)
    n = len(departures)
    holiday = np.full(n, np.nan)
    adjacent = np.full(n, np.nan)
    if holidays is not None and n:
        t = pd.to_datetime(departures["MVT_TIME_UTC_mvt"], utc=True)
        dated = ~t.isna().to_numpy()
        # the day comes from a datetime64 unit cast, which knows the
        # resolution it is given, and not from dividing an integer view by a
        # nanosecond day: the parquet files carry microseconds, and that
        # division would put every date in January 1970 -- both columns zero
        # rather than NaN, which the ladder reads as a block that is present
        day = (
            t.dt.tz_convert(None)
            .to_numpy(dtype="datetime64[ns]")
            .astype("datetime64[D]")
            .astype(np.int64)
        )
        country = departures["ADEP_mvt"].astype(str).map(COUNTRY).to_numpy()
        if dated.any():
            years = t.dt.year[dated]
            # a year either side, so a lookup on 31 December or 1 January
            # sees the neighbouring date in the calendar next door
            span = (int(years.min()) - 1, int(years.max()) + 1)
            for code in pd.unique(country[dated & pd.notna(country)]):
                rows = dated & (country == code)
                days = _holiday_days(str(code), *span)
                d = day[rows]
                holiday[rows] = np.isin(d, days).astype(float)
                adjacent[rows] = (np.isin(d - 1, days) | np.isin(d + 1, days)).astype(
                    float
                )
    out["holiday"] = holiday
    out["holiday_adjacent"] = adjacent
    return out[list(HOLIDAY_FEATURES)]
