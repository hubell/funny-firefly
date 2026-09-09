# SPDX-License-Identifier: GPL-3.0-only
"""The operator, read off the callsign rather than off the flight record.

Airlines differ in pushback and start-up practice, in which apron they park
on, and in what they do with a delay, and the challenge exposes who is
flying twice: as ``AIRCRAFT_OPERATOR_flt`` on the flight side, and as the
leading three letters of ``FLIGHT_mvt``, which are the ICAO airline
designator -- THY, AFR, DLH and BAW the commonest in the training year. The
two carry the same information and are worth the same on the residual, +2.4 s
against +2.1 s on December (``docs/review_headroom.md``); both together are
worth no more than either.

The callsign is the one this entry reads, for the reason that decides it.
The flight-side record is missing for 1.08% of training departures and 1.48%
of ranking departures (``reports/data_review.md``), and those rows are the
worst in the dataset: Heathrow's no-record departures run at 1,216 s RMSE
against a pooled 233, because a plan cancelled and refiled after a long
delay loses its join, so the missing record is itself a delay signal. An
operator feature keyed on the flight side goes null on exactly those rows --
it abstains where it is most needed. ``FLIGHT_mvt`` is null for 0.01% of
ranking departures.

The prefix is not the operator, and the gap between them shows in three
places. A callsign that does not follow the airline convention -- a
registration, a military or a general-aviation identifier -- has no
designator to read, and either falls to ``NA`` or yields three letters that
are a registration prefix rather than an airline. A designator says who a
flight is flown as and not who holds the aircraft, so a wet lease reads as
the carrier whose flight number it wears. And an airline that files its
subsidiaries under one designator is one level here whatever its handling
arrangements are.

Null-ness is a level and not a hole: no match, a null flight identifier
included, is the level ``NA``, exactly as ``taxiout.features._level`` treats
a null wake class. A level seen fewer than ``MIN_LEVEL_ROWS`` times in the
fitting sample is pooled into ``other``, the way wake classes are pooled in
the geometric fit (``taxiout.unimpeded``), so that the ranking set cannot
present a level the model never saw and a designator with three flights in
the year does not get a split of its own. That threshold is counted on the
fitting sample, which makes the pooling a property of the fitted estimator
rather than of the feature frame: the frame carries the raw prefix, and the
two pieces of the pooling are exported here for an estimator to wire in.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

# A categorical: the designator is a name and not a quantity, and nothing
# orders THY against AFR.
CALLSIGN_FEATURES: tuple[str, ...] = ("cs_prefix",)

SOURCES: dict[str, tuple[str, ...]] = {"cs_prefix": ("FLIGHT_mvt",)}

# What no designator is called, and what a level too thin to stand alone is
# pooled into. Neither is ever a real designator: ICAO designators are three
# letters, so a two-letter label and a lower-case one cannot collide with one.
NA_LEVEL = "NA"
OTHER_LEVEL = "other"

# Below this many rows in the fitting sample a level is pooled. The floor the
# geometric fit pools wake classes at (``taxiout.unimpeded.MIN_CLASS_ROWS``),
# reused: the argument is the same one, that a group this thin is a handful of
# flights and not an operator's practice.
MIN_LEVEL_ROWS = 30

_DESIGNATOR = r"^([A-Z]{3})"


def callsign_features(departures: pd.DataFrame) -> pd.DataFrame:
    """The callsign block for ``departures``: the designator, unpooled.

    Indexed like ``departures`` and carrying exactly ``CALLSIGN_FEATURES``,
    as strings and never a NaN. The prefix is raw: pooling belongs to the
    fitted estimator, through ``frequent_levels`` and ``pool_rare``.
    """
    flight = departures["FLIGHT_mvt"]
    # ``FLIGHT_mvt`` arrives dictionary-encoded, so the column is a pandas
    # categorical, which refuses ``fillna`` with a value that is not already
    # one of its levels. Nulls go out through ``np.where`` instead, as
    # ``taxiout.features._level`` does -- and they have to go before the
    # pattern runs rather than after, because ``astype(str)`` writes a null as
    # "nan" (as "None" on an object column), and either upper-cases into three
    # letters that match.
    text = pd.Series(
        np.where(flight.isna(), "", flight.astype(str)),
        index=departures.index,
        dtype=object,
    )
    prefix = text.str.upper().str.extract(_DESIGNATOR, expand=False)
    out = pd.DataFrame(index=departures.index)
    out["cs_prefix"] = prefix.fillna(NA_LEVEL).astype(object)
    return out[list(CALLSIGN_FEATURES)]


def frequent_levels(
    values: pd.Series, *, min_rows: int = MIN_LEVEL_ROWS
) -> frozenset[str]:
    """The levels of ``values`` occurring at least ``min_rows`` times.

    Counted on the fitting sample and nowhere else. The estimator keeps the
    set, and every other side of the split is mapped through it rather than
    counted again -- a set counted on the ranking rows would let a level the
    model never fitted reach it under its own name.
    """
    counts = pd.Series(values).dropna().astype(str).value_counts()
    return frozenset(counts.index[counts >= min_rows])


def pool_rare(values: pd.Series, levels: Iterable[str]) -> pd.Series:
    """``values`` with everything outside ``levels`` mapped to ``OTHER_LEVEL``.

    Strings, in an object column, rather than a categorical: which levels
    exist is the caller's decision, and a categorical fixed here would carry
    a level set built from one side of one split into every use of it. A null
    is outside every set and pools with the rest; the feature never presents
    one, and a caller pooling some other column gets ``other`` and not a hole.
    """
    keep = frozenset(levels)
    text = pd.Series(
        np.where(values.isna(), OTHER_LEVEL, values.astype(str)),
        index=values.index,
        dtype=object,
    )
    return text.where(text.isin(keep), OTHER_LEVEL)
