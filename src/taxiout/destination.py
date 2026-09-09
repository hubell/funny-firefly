# SPDX-License-Identifier: GPL-3.0-only
"""The destination as geometry: which way the aircraft leaves, and how far it goes.

The destination aerodrome is exposed for every ranking departure and is fixed
long before the aircraft moves. It decides which standard instrument departure
the flight is cleared for, and the SID decides which runway entry the aircraft
is sent to and which holding point it queues at. Two departures off the same
runway to opposite corners of the continent do not share a queue; two to the
same corner do. That is the mechanism, and it is the whole of what this block
claims.

It enters as two continuous quantities rather than as a categorical of the 914
destinations the data names: the initial great-circle bearing from the
departure aerodrome to the destination, and the great-circle distance. The
bearing is what a departure gate approximates -- destinations that leave by the
same gate share one -- and unlike an identifier it means the same thing at
every airport, so a bearing learnt where the data is thick is still read where
it is thin. The distance stands in for the departure weight, and for how far
round the field a long-haul departure is sent to reach the full-length entry.
Measured on the residual of the submission rung before the block was built,
the pair is worth about 1.4 s of RMSE in December and 1.6 s in July; the
categorical was worth 1.5 and 1.9, which does not buy 914 levels.

Three limits. The bearing runs between aerodrome reference points, not along
the SID: a departure turns within a few miles of the field, and it is that
first turn the queue is organised by, so this is the direction the flight
leaves in and not the track it flies. Neither column says anything about the
day -- both are properties of the city pair and identical for every departure
to that destination -- so what the model reads here is a per-flight constant
taken jointly with the stand and the runway, never a congestion signal. And
the coordinates come from outside the challenge data: a null destination, or
one the OurAirports file does not name, has no geometry at all. The data
review puts ``ADES_mvt`` at 0.01% null on the training side and 0.00% on the
ranking side, so almost all of that is the second case: 0.7% of ranking
departures name a destination the cached file does not, which is the rate
``docs/review_headroom.md`` estimated at 0.4 to 0.7%. Those rows carry NaN in
both columns and ``dest_missing`` set, never a zero -- a zero bearing is due
north and a zero distance is the field itself, and the model would read either
as fact.

OurAirports releases the file to the public domain, and
``scripts/fetch_ourairports.py`` caches it with the request and the terms
recorded beside it, as the METAR fetch does. It is the second external dataset
the model reads and is registered under its own ``OURAIRPORTS.`` prefix, so
the leakage audit holds it to that file's own field list rather than to either
challenge phase's.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# The archive's own column names, as fetched. These are the only external
# columns a feature may read, and the registry names them under the
# ``OURAIRPORTS.`` phase so the audit can hold them to this list.
FIELDS: tuple[str, ...] = ("ident", "latitude_deg", "longitude_deg")

DESTINATION_FEATURES: tuple[str, ...] = (
    "dest_bearing_deg",
    "dest_km",
    "dest_missing",
)

SOURCES: dict[str, tuple[str, ...]] = {
    c: (
        "ADEP_mvt",
        "ADES_mvt",
        "OURAIRPORTS.ident",
        "OURAIRPORTS.latitude_deg",
        "OURAIRPORTS.longitude_deg",
    )
    for c in DESTINATION_FEATURES
}

# The sphere the distances are measured on: the IUGG mean radius, (2a + b) / 3.
# ``taxiout.geo`` fits its radius to the midpoint latitude and to the azimuth
# the line runs at, which is exact enough to ignore under fifteen kilometres
# and drifts on a leg of hundreds, where the radius the line passes over is not
# the radius at its midpoint. The two therefore disagree by up to a few tenths
# of a percent, which is the spread of the local radius about the mean and not
# an error either of them can remove. On a column whose job is to order
# destinations by how far away they are, that is nothing.
EARTH_RADIUS_KM = 6371.0088


def bearings_deg(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Initial true bearings, degrees clockwise from north in ``[0, 360)``.

    The vectorised form of ``taxiout.geo.bearing_deg``, term for term; nothing
    but the array arithmetic differs, and ``tests/test_destination.py`` holds
    the two to exact agreement so that a transcription slip cannot pass.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    y = np.sin(dlam) * np.cos(phi2)
    x = np.cos(phi1) * np.sin(phi2) - np.sin(phi1) * np.cos(phi2) * np.cos(dlam)
    return np.degrees(np.arctan2(y, x)) % 360.0


def distances_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """Great-circle distances in kilometres, on the mean-radius sphere."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    h = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(h, 1.0)))


def _empty_airports() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ident": pd.Series(dtype=object),
            "latitude_deg": pd.Series(dtype=float),
            "longitude_deg": pd.Series(dtype=float),
        }
    )


def load_airports(path: Path) -> pd.DataFrame:
    """The cached OurAirports table, reduced to ``FIELDS`` and one row per ident.

    Rows without an identifier or without a finite pair of coordinates are
    dropped: they cannot answer the only question this module asks of the
    file. A missing file is an empty table rather than an error, so a caller
    without the cache gets the missing block instead of a traceback.
    """
    path = Path(path)
    if not path.exists():
        return _empty_airports()
    out = pd.read_csv(path, usecols=list(FIELDS))[list(FIELDS)]
    ident = out["ident"]
    out["ident"] = ident.where(ident.notna(), "").astype(str).str.strip()
    for c in ("latitude_deg", "longitude_deg"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    keep = (
        (out["ident"] != "")
        & np.isfinite(out["latitude_deg"])
        & np.isfinite(out["longitude_deg"])
    )
    out = out[keep].drop_duplicates("ident", keep="first")
    return out.reset_index(drop=True)


def _idents(column: pd.Series) -> np.ndarray:
    """A column of aerodrome codes as strings, a null becoming the empty string.

    The empty string is in no airport table, so a null lands where an unknown
    code lands and the two need no separate handling.
    """
    return np.where(column.isna(), "", column.astype(str)).astype(object)


def destination_features(
    departures: pd.DataFrame, airports: pd.DataFrame | None
) -> pd.DataFrame:
    """The destination block for ``departures``, from the cached coordinates.

    ``airports`` is ``load_airports``'s output, one row per identifier. ``None``
    or empty yields a block that is NaN with ``dest_missing`` set throughout,
    which is what a caller without the cache gets and what lets the ladder skip
    the rung rather than fit on a column of nothing. Indexed like
    ``departures``; carries exactly ``DESTINATION_FEATURES``.
    """
    out = pd.DataFrame(index=departures.index)
    if airports is None or airports.empty:
        out["dest_bearing_deg"] = np.nan
        out["dest_km"] = np.nan
        out["dest_missing"] = 1.0
        return out[list(DESTINATION_FEATURES)]

    # A sentinel row of NaN past the end of the coordinates, so a code the
    # table does not carry indexes it and the arithmetic runs over NaN rather
    # than over whichever aerodrome happens to sit at index -1.
    lat = np.append(airports["latitude_deg"].to_numpy(dtype=float), np.nan)
    lon = np.append(airports["longitude_deg"].to_numpy(dtype=float), np.nan)
    sentinel = len(lat) - 1

    lookup = pd.Index(airports["ident"].astype(str))
    adep = lookup.get_indexer(_idents(departures["ADEP_mvt"]))
    ades = lookup.get_indexer(_idents(departures["ADES_mvt"]))
    a = np.where(adep < 0, sentinel, adep)
    b = np.where(ades < 0, sentinel, ades)

    # A departure returning to its own aerodrome reads zero kilometres, which
    # is the truth about the pair, and a bearing of zero, which is what
    # ``arctan2(0, 0)`` gives and is arbitrary. There is nothing better to put
    # there and the case is too rare to spend a flag on.
    out["dest_bearing_deg"] = bearings_deg(lat[a], lon[a], lat[b], lon[b])
    out["dest_km"] = distances_km(lat[a], lon[a], lat[b], lon[b])
    # The flag marks every row without a pair of positions, which is the
    # destination in all but a hypothetical case: the departure aerodrome is
    # one of the eleven and the file names all of them. Written as the union
    # anyway, so that ``dest_missing`` is zero exactly where the two columns
    # are finite and a reader need not check both.
    out["dest_missing"] = ((adep < 0) | (ades < 0)).astype(float)
    return out[list(DESTINATION_FEATURES)]
