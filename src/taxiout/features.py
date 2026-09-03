# SPDX-License-Identifier: GPL-3.0-only
"""The Gate 1 feature frame, and a registry of what every column reads.

One frame per movement, built from ``taxiout.challenge``'s loaders and
``taxiout.join.join_stands``. Ten blocks:

- **Identifiers.** Airport, stand, runway, and the pair. These are what the
  ablation ladder's saturated rung uses and what the held-out-pair split
  groups on.
- **Time, anchored on take-off.** ``MVT_TIME_UTC_mvt`` is the only actual time
  a test departure carries (``reports/data_review.md``), so it is the anchor
  for every time feature. Times are UTC: the ten airports span UTC+0 to UTC+3
  and no timezone table is carried, so the offset is left for the airport
  identifier to absorb rather than being hard-coded here.
- **Geometry.** The distance-matrix cell for this (airport, stand, runway),
  plus four shape ratios that make the descriptors comparable between a
  400 m taxi and a 4 km one.
- **Record quality of the join.** ``join_stage``, ``n_ids``, ``id_spread_m``,
  ``spread_m``, ``snap_max_m`` and ``n_features`` say how much to trust the
  geometry block on this row: which normalisation stage matched, how many OSM
  identifiers the cell stands for and how far their routes disagree, how far
  the duplicate features behind one identifier disagree, and how far the stand
  had to be snapped to reach mapped pavement. They are inputs, not filters.
- **The flight plan's own delay signal.** ``LOBT_flt - IOBT_flt``: how far the
  planned off-block moved between the first estimate the Network Manager held
  and the last, from ATFM or the airline or the airport without saying which
  (*Deliberate non-goals* in ``docs/modelling_notes.md``).
- **The aircraft.** Wake-turbulence class and type, as categoricals. A
  heavy pushes back, starts and lines up more slowly than a medium on the
  same route; on a seen pair the stand identifier absorbs that, because
  heavies park at wide-body stands, and on a new pair nothing does.
- **Congestion.** What the ranking set exposes about the movements around
  the subject, in windows anchored on its take-off: runway throughput,
  arrivals and their taxi-in times, scheduled demand, the runway
  configuration in force. Built in ``taxiout.congestion`` from the whole
  movement frame, and the one block whose sources name another row's phase.
- **Shared corridor.** The other movements in the window weighted by how
  much of the subject's own taxiway route theirs shares, from the graph's
  edge sets (``taxiout.corridor``). The one feature here no other entry has.
- **Plan lags.** Take-off minus a planning value fixed before departure --
  the first and last off-block estimates and the schedule -- so taxi plus
  the pushback delay against the plan; the subject's own and the mean over
  the other departures on its runway within fifteen minutes
  (``taxiout.planlags``).
- **Weather.** Three regime flags per airport-hour -- visibility band,
  precipitation, freezing -- from the Mesonet's METAR archive
  (``taxiout.metar``), the one block that reads outside the challenge data,
  and registered under its own prefix so the audit can say so.

Stands that do not join fall back to airport level rather than carrying nulls
-- 3.7% of training departures and 3.3% of ranking departures are stands
OpenStreetMap does not have. The fallback is the median of the matrix cells
for that (airport, runway), or for that airport where the runway is a
sentinel, and ``geom_imputed`` marks every row it fired on. It is computed
from the matrix alone and never from the target, so it is identical on both
sides of every split and needs no fitting.

``FEATURE_SOURCES`` maps every feature column to the raw columns it reads.
It exists so the leakage rule in ``docs/modelling_notes.md`` can be tested
rather than asserted. A source is a bare column name for the subject's own
row, or ``DEP.<column>`` / ``ARR.<column>`` for another movement's; the rule
is phase-specific -- a feature may read what the ranking set exposes *for the
rows it reads from* -- and the registry states the phase so the test can hold
each to its own list. ``tests/test_features.py`` checks that no bare or
``DEP.`` source is in ``taxiout.challenge.FORBIDDEN_FEATURE_INPUTS``, that
every ``ARR.`` source is in ``AVAILABLE_FOR_ARRIVALS``, that ``AOBT_3_flt``
appears under no prefix, and that the registry and the built frame agree
column for column, so a feature added without a source entry fails the suite
instead of slipping past it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.congestion import (
    CATEGORICAL as CONGESTION_CATEGORICAL,
    CONGESTION_FEATURES,
    SOURCES as CONGESTION_SOURCES,
    congestion_features,
)
from taxiout.corridor import (
    CORRIDOR_FEATURES,
    SOURCES as CORRIDOR_SOURCES,
    corridor_features,
)
from taxiout.join import UNMATCHED, join_stands
from taxiout.metar import (
    SOURCES as WEATHER_SOURCES,
    WEATHER_FEATURES,
    weather_features,
)
from taxiout.planlags import (
    PLAN_LAG_FEATURES,
    SOURCES as PLAN_LAG_SOURCES,
    planlag_features,
)
from taxiout.routing import AGGREGATED_FIELDS

TARGET = "TAXITIME_SEC_mvt"

# The join keys, named once. Every geometry and quality column reads exactly
# these three raw columns, through the matrix.
_KEYS: tuple[str, ...] = ("ADEP_mvt", "STAND_mvt", "RUNWAY_mvt")
_TAKEOFF = "MVT_TIME_UTC_mvt"

IDENTIFIER_FEATURES: tuple[str, ...] = ("icao", "stand", "designator", "pair")

TIME_FEATURES: tuple[str, ...] = (
    "tko_minute_of_day",
    "tko_hour_sin",
    "tko_hour_cos",
    "tko_dow",
    "tko_is_weekend",
    "tko_month",
    "tko_doy_sin",
    "tko_doy_cos",
)

# The matrix cell as routed, then four ratios. The raw descriptors scale with
# route length, so a model reading them alone cannot tell a twisty short taxi
# from a straight long one; the ratios say shape independently of size.
GEOMETRY_FEATURES: tuple[str, ...] = AGGREGATED_FIELDS + (
    "straight_frac",
    "backtrack_frac",
    "turn_deg_per_km",
    "junctions_per_km",
)

QUALITY_FEATURES: tuple[str, ...] = (
    "join_stage",
    "n_ids",
    "id_spread_m",
    "spread_m",
    "snap_max_m",
    "n_features",
    "geom_imputed",
)

PLAN_FEATURES: tuple[str, ...] = ("lobt_iobt_s", "lobt_iobt_missing")

# The aircraft. Both are exposed for every ranking departure and neither
# depends on the phase. ``reports/geometry_gaps.md`` found the wake class
# worth 35 to 281 s on the fastest decile at the same stand and runway, which
# the stand identifier absorbs on a seen pair and nothing absorbs on a new
# one; the type is the finer version of the same thing. A null class is the
# unjoined flight-side block and is carried as a class of its own.
AIRCRAFT_FEATURES: tuple[str, ...] = ("wake_class", "aircraft_type")

FEATURE_COLUMNS: tuple[str, ...] = (
    IDENTIFIER_FEATURES
    + TIME_FEATURES
    + GEOMETRY_FEATURES
    + QUALITY_FEATURES
    + PLAN_FEATURES
    + AIRCRAFT_FEATURES
    + PLAN_LAG_FEATURES
    + CONGESTION_FEATURES
    + CORRIDOR_FEATURES
    + WEATHER_FEATURES
)

# Every feature column against the raw columns it reads. Checked against
# ``FORBIDDEN_FEATURE_INPUTS`` by the test suite; see the module docstring.
FEATURE_SOURCES: dict[str, tuple[str, ...]] = {
    **{c: _KEYS for c in IDENTIFIER_FEATURES},
    **{c: (_TAKEOFF,) for c in TIME_FEATURES},
    # Geometry and join quality are functions of the triple alone: the matrix
    # is keyed on it, and the fallback is keyed on (airport, runway).
    **{c: _KEYS for c in GEOMETRY_FEATURES},
    **{c: _KEYS for c in QUALITY_FEATURES},
    **{c: ("LOBT_flt", "IOBT_flt") for c in PLAN_FEATURES},
    "wake_class": ("WK_TBL_CAT_flt",),
    "aircraft_type": ("AIRCRAFT_TYPE_mvt",),
    # The plan lags read the subject's own planning values and, under
    # ``DEP.``, the other departures' on its runway.
    **PLAN_LAG_SOURCES,
    # The congestion block reads other movements' rows, and says so: its
    # sources carry the phase they are read from (``DEP.`` or ``ARR.``), and
    # a bare name is the subject's own row, as everywhere above.
    **CONGESTION_SOURCES,
    **CORRIDOR_SOURCES,
    # The weather block reads an external dataset, and says so: its sources
    # carry the ``METAR.`` prefix and name the archive's own fields.
    **WEATHER_SOURCES,
}

# Prefixes that name an external dataset rather than a phase of the challenge
# data. A source under one of these is held to that dataset's field list
# rather than to either challenge list.
EXTERNAL_SOURCES: tuple[str, ...] = ("METAR",)


def source_phase(source: str) -> tuple[str | None, str]:
    """``("ARR", "BLOCK_TIME_UTC_mvt")`` for ``"ARR.BLOCK_TIME_UTC_mvt"``; ``(None, name)`` bare."""
    if "." in source:
        phase, column = source.split(".", 1)
        return phase, column
    return None, source


# What a caller has to load to build the frame: the subject's own columns.
# The target is not here: it is read by the validation harness, never by a
# feature.
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {
            source_phase(c)[1]
            for sources in FEATURE_SOURCES.values()
            for c in sources
            if source_phase(c)[0] is None
        }
    )
)

# What the movement frame has to carry: every prefixed source of every block,
# stripped, plus the phase that says which prefix a row answers to.
MOVEMENT_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {"PHASE_mvt"}
        | {
            source_phase(c)[1]
            for sources in FEATURE_SOURCES.values()
            for c in sources
            if source_phase(c)[0] not in (None, *EXTERNAL_SOURCES)
        }
    )
)

# Columns the model treats as categorical rather than ordered.
CATEGORICAL_FEATURES: tuple[str, ...] = (
    IDENTIFIER_FEATURES + AIRCRAFT_FEATURES + CONGESTION_CATEGORICAL
)

# What a null categorical is called. Null-ness is information here -- a null
# wake class is the unjoined block -- and a label keeps it a level rather
# than a hole.
NULL_LEVEL = "none"

_DAY = 86400.0
_MINUTE = 60.0


def _finite_ratio(num: pd.Series, den: pd.Series, floor: float = 1.0) -> pd.Series:
    """``num / den`` with a floor on the denominator, propagating nulls.

    The floor is in the denominator's own units and exists because a cell can
    route zero metres -- a stand attached directly to the runway entry -- and
    an infinity in a feature column is worse than a null.
    """
    return num / den.where(den.abs() >= floor)


def _level(column: pd.Series) -> pd.Series:
    """A categorical's values as strings, with nulls as ``NULL_LEVEL``.

    Through ``np.where`` rather than ``fillna`` because a pandas categorical
    refuses a fill value that is not already one of its levels.
    """
    values = np.where(column.isna(), NULL_LEVEL, column.astype(str))
    return pd.Series(values, index=column.index, dtype=object)


def time_features(takeoff: pd.Series) -> pd.DataFrame:
    """Calendar and clock features of the take-off time.

    The anchor is deliberate. Operationally taxi-out is predicted at pushback
    and the take-off time is unknown; the challenge poses the reverse and
    blanks the off-block time, so the take-off time is the only actual stamp a
    test departure carries. See *Leakage watch* in ``docs/modelling_notes.md``.
    """
    t = pd.to_datetime(takeoff, utc=True)
    minute = t.dt.hour * 60 + t.dt.minute + t.dt.second / _MINUTE
    angle = 2 * np.pi * minute / (_DAY / _MINUTE)
    # Day of year over 366 whatever the year, so the phase does not shift by a
    # day between a leap year and the next -- the training year and the ranking
    # months are different years and the seasonal term has to line up.
    doy = 2 * np.pi * (t.dt.dayofyear - 1) / 366.0
    return pd.DataFrame(
        {
            "tko_minute_of_day": minute.astype("float32"),
            "tko_hour_sin": np.sin(angle).astype("float32"),
            "tko_hour_cos": np.cos(angle).astype("float32"),
            "tko_dow": t.dt.dayofweek.astype("float32"),
            "tko_is_weekend": (t.dt.dayofweek >= 5).astype("float32"),
            "tko_month": t.dt.month.astype("float32"),
            "tko_doy_sin": np.sin(doy).astype("float32"),
            "tko_doy_cos": np.cos(doy).astype("float32"),
        },
        index=takeoff.index,
    )


def airport_fallback(matrix: pd.DataFrame) -> pd.DataFrame:
    """Airport-level geometry for stands the join could not place.

    Two levels, coarsest last: the median matrix cell for each (airport,
    runway), and the median for each airport over every runway. A row whose
    runway is known but whose stand is not gets the first, which is the typical
    taxi to that runway; a row whose runway names no runway at all (``NA`` and
    the two helicopter codes) gets the second.

    Medians are over the *cells* of the matrix, not over movements, so this
    reads nothing from the challenge data and nothing from the target. It is
    therefore the same table in training and at inference, and carries no split.
    """
    cols = [c for c in AGGREGATED_FIELDS if c in matrix.columns]
    by_runway = (
        matrix.groupby(["icao", "designator"], sort=True)[cols].median().reset_index()
    )
    by_airport = matrix.groupby("icao", sort=True)[cols].median().reset_index()
    by_airport["designator"] = pd.NA
    return pd.concat([by_runway, by_airport], ignore_index=True)


def _apply_fallback(
    frame: pd.DataFrame, fallback: pd.DataFrame, matched: pd.Series
) -> pd.DataFrame:
    """Fill the geometry block of unmatched rows from ``fallback``, in place."""
    cols = [c for c in AGGREGATED_FIELDS if c in frame.columns]
    need = ~matched
    if not need.any():
        return frame
    keys = pd.DataFrame(
        {
            "icao": frame.loc[need, "icao"].astype(str),
            "designator": frame.loc[need, "designator"].astype(str),
        }
    )
    per_runway = fallback[fallback["designator"].notna()].astype({"designator": str})
    per_airport = fallback[fallback["designator"].isna()].drop(columns=["designator"])
    filled = keys.merge(per_runway, on=["icao", "designator"], how="left")
    coarse = keys[["icao"]].merge(per_airport, on="icao", how="left")
    filled.index = coarse.index = keys.index
    for c in cols:
        frame.loc[need, c] = filled[c].fillna(coarse[c]).to_numpy()
    return frame


def build_features(
    frame: pd.DataFrame,
    matrix: pd.DataFrame,
    *,
    fallback: pd.DataFrame | None = None,
    movements: pd.DataFrame | None = None,
    edges: pd.DataFrame | None = None,
    weather: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The feature frame for ``frame``, one row in, one row out, order kept.

    ``frame`` is departures as the challenge loaders return them, carrying at
    least ``REQUIRED_COLUMNS``. ``matrix`` is the concatenated distance
    matrices. ``fallback`` defaults to ``airport_fallback(matrix)``.

    ``movements`` is every movement of the same side, both phases, carrying
    ``MOVEMENT_COLUMNS``; the congestion block is counted from it. A caller
    that has the data always passes it. Left as ``None``, the frame itself
    stands in -- departures only, so no arrival is ever counted -- which is
    what a unit test of the other blocks wants and what nothing else should.

    ``edges`` is the per-cell taxiway edge table from ``<ICAO>_edges.csv``
    (``taxiout.corridor.load_edges``); the shared-corridor block is computed
    from it. Left as ``None``, that block is NaN throughout -- not zero,
    because "not computed" is not "no traffic" -- and the callers that have
    the data always pass it.

    ``weather`` is the hourly regime table from ``taxiout.metar.regime_by_hour``
    over the cached archive. Left as ``None``, the weather block is missing
    throughout with its flag set, and the ladder skips its rung.

    The result carries exactly ``FEATURE_COLUMNS`` and the original index, so
    it can be set beside the target and the split masks without a join.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"feature frame needs {missing}")
    if fallback is None:
        fallback = airport_fallback(matrix)
    if movements is None:
        movements = frame.assign(PHASE_mvt="DEP")
        for c in MOVEMENT_COLUMNS:
            if c not in movements.columns:
                movements[c] = None
    missing = [c for c in MOVEMENT_COLUMNS if c not in movements.columns]
    if missing:
        raise KeyError(f"movement frame needs {missing}")

    joined = join_stands(frame[list(_KEYS)], matrix)
    out = pd.DataFrame(index=frame.index)

    out["icao"] = frame["ADEP_mvt"].astype(str)
    out["stand"] = frame["STAND_mvt"].astype(str)
    out["designator"] = frame["RUNWAY_mvt"].astype(str)
    # The pair is the saturated identifier the geometry claims to replace. It
    # is built from the raw triple rather than the matched cell so that
    # unmatched stands still form groups -- the held-out-pair split has to be
    # able to hold out a pair the geometry could not place.
    out["pair"] = out["icao"] + "|" + out["stand"] + "|" + out["designator"]

    for c, values in time_features(frame[_TAKEOFF]).items():
        out[c] = values

    matched = pd.Series(joined["join_stage"].to_numpy() != UNMATCHED, index=frame.index)
    for c in AGGREGATED_FIELDS:
        out[c] = pd.to_numeric(joined[c], errors="coerce").to_numpy()
    out["geom_imputed"] = (~matched).astype("float32")
    _apply_fallback(out, fallback, matched)

    route = out["route_m"]
    out["straight_frac"] = _finite_ratio(out["straight_m"], route)
    out["backtrack_frac"] = _finite_ratio(out["backtrack_m"], route)
    out["turn_deg_per_km"] = 1000.0 * _finite_ratio(out["turn_deg"], route)
    out["junctions_per_km"] = 1000.0 * _finite_ratio(out["junctions"], route)

    # Join quality. An unmatched row has no cell, so it has no ambiguity to
    # report either: the flags go to zero and ``geom_imputed`` carries the
    # fact that the geometry is an airport-level guess.
    for c in (
        "join_stage",
        "n_ids",
        "id_spread_m",
        "spread_m",
        "snap_max_m",
        "n_features",
    ):
        out[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0.0).to_numpy()

    delay = (
        pd.to_datetime(frame["LOBT_flt"], utc=True)
        - pd.to_datetime(frame["IOBT_flt"], utc=True)
    ).dt.total_seconds()
    out["lobt_iobt_missing"] = delay.isna().astype("float32")
    out["lobt_iobt_s"] = delay.fillna(0.0).astype("float32")

    out["wake_class"] = _level(frame["WK_TBL_CAT_flt"])
    out["aircraft_type"] = _level(frame["AIRCRAFT_TYPE_mvt"])

    for c, values in planlag_features(frame, movements).items():
        out[c] = values

    for c, values in congestion_features(frame, movements).items():
        out[c] = values

    if edges is None:
        for c in CORRIDOR_FEATURES:
            out[c] = np.nan
    else:
        for c, values in corridor_features(frame, movements, matrix, edges).items():
            out[c] = values

    for c, values in weather_features(frame, weather).items():
        out[c] = values

    for c in FEATURE_COLUMNS:
        if c in CATEGORICAL_FEATURES:
            out[c] = out[c].astype("category")
        else:
            out[c] = out[c].astype("float32")
    return out[list(FEATURE_COLUMNS)]


def feature_block(names: tuple[str, ...] | list[str]) -> list[str]:
    """Feature columns for a named subset, in ``FEATURE_COLUMNS`` order."""
    wanted = set(names)
    unknown = wanted - set(FEATURE_COLUMNS)
    if unknown:
        raise KeyError(f"not feature columns: {sorted(unknown)}")
    return [c for c in FEATURE_COLUMNS if c in wanted]
