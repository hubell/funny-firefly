# SPDX-License-Identifier: GPL-3.0-only
"""The Network Manager's off-block time, and the queue it makes countable.

Candidate C1 of ``reports/preregistered-2026-10-04.md``, section 3, and
nothing else. ``AOBT_3_flt`` is in ``taxiout.challenge.EXCLUDED_BY_DECISION``
and stays there: this module is the one place it is read, for a measurement
fixed before any figure existed, and its admission to anything that ships
needs the rule's verdict and the owner's originality decision in writing.
``tests/test_aobt3.py`` checks that no other module's ``SOURCES`` names it.

The features, computed the same way on both sides and missing where the
subject's ``AOBT_3_flt`` is missing:

- ``aobt3_lag``: ``MVT_TIME_UTC_mvt`` minus ``AOBT_3_flt``, seconds;
- ``aobt3_vs_lobt``: ``AOBT_3_flt`` minus ``LOBT_flt``; ``aobt3_vs_eobt``:
  ``AOBT_3_flt`` minus ``EOBT_1_flt``; ``aobt3_missing``, 0 or 1 (never
  missing);
- at the subject's airport, over the other departures j with ``AOBT_3_flt``
  present (a_j its off-block time, t_j its take-off; a and t the subject's):
  ``q_surface_off``, the count of j with a_j <= a < t_j, all runways, and
  ``q_surface_off_rwy`` on the subject's runway; ``q_takeoffs_during``, the
  count of j with a <= t_j < t, all runways, and ``q_takeoffs_during_rwy``;
  ``q_overtakes_rwy``, the count of j on the subject's runway with a_j > a
  and t_j < t.

"The other departures" are the rows of the frame passed in, the subject
removed by identity (its own row), so a training year counts the training
year and a ranking month counts the ranking month. Every count is exact: no
window bounds how far apart two departures may be.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AOBT = "AOBT_3_flt"

AOBT3_OWN: tuple[str, ...] = (
    "aobt3_lag",
    "aobt3_vs_lobt",
    "aobt3_vs_eobt",
    "aobt3_missing",
)
QUEUE_FEATURES: tuple[str, ...] = (
    "q_surface_off",
    "q_surface_off_rwy",
    "q_takeoffs_during",
    "q_takeoffs_during_rwy",
    "q_overtakes_rwy",
)
AOBT3_FEATURES: tuple[str, ...] = AOBT3_OWN + QUEUE_FEATURES

_QUEUE_SELF = ("MVT_TIME_UTC_mvt", AOBT, "ADEP_mvt", "RUNWAY_mvt", "MVT_ID_mvt")
_QUEUE_OTHERS = (
    "DEP.MVT_TIME_UTC_mvt",
    f"DEP.{AOBT}",
    "DEP.ADEP_mvt",
    "DEP.RUNWAY_mvt",
    "DEP.MVT_ID_mvt",
)
SOURCES: dict[str, tuple[str, ...]] = {
    "aobt3_lag": ("MVT_TIME_UTC_mvt", AOBT),
    "aobt3_vs_lobt": (AOBT, "LOBT_flt"),
    "aobt3_vs_eobt": (AOBT, "EOBT_1_flt"),
    "aobt3_missing": (AOBT,),
    **{c: _QUEUE_SELF + _QUEUE_OTHERS for c in QUEUE_FEATURES},
}
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    sorted({s for v in SOURCES.values() for s in v if "." not in s})
)


def _seconds(values: pd.Series) -> np.ndarray:
    """UTC timestamps as float seconds since the epoch, NaN where missing."""
    t = pd.to_datetime(values, utc=True)
    out = (t - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    return out.to_numpy(dtype=float)


def _count_le(sorted_values: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, x, side="right")


def _count_lt(sorted_values: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_values, x, side="left")


def _surface_and_during(a: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Within one group of rows, all with both times present, the two counts.

    ``surface``: j with a_j <= a < t_j. The j with a_j <= a are counted, less
    those that also have t_j <= a, which are exactly the j with
    max(a_j, t_j) <= a. ``during``: j with a <= t_j < t, the j with t_j < t
    less those with t_j < a, and none where a >= t. The row itself is in
    neither sorted array's complement: it is counted once in ``surface`` when
    a < t, and never in ``during`` (t_j < t fails for j = the row), so it is
    removed from ``surface`` by its own condition.
    """
    sa = np.sort(a)
    sm = np.sort(np.maximum(a, t))
    st = np.sort(t)
    surface = _count_le(sa, a) - _count_le(sm, a) - (a < t).astype(np.int64)
    during = np.where(a < t, _count_lt(st, t) - _count_lt(st, a), 0)
    return surface.astype(float), during.astype(float)


def _overtakes(a: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Within one group, for each row the count of j with a_j > a and t_j < t.

    Offline dominance counting: rows visited in decreasing a, each tie of a
    queried before any of it is inserted, so only strictly later off-blocks
    are counted; a Fenwick tree over the take-off ranks counts those taking
    off strictly earlier. The row itself has a_j = a and is never counted.
    """
    n = len(a)
    out = np.zeros(n, dtype=np.int64)
    if n == 0:
        return out.astype(float)
    t_rank = np.searchsorted(np.unique(t), t, side="left")  # 0-based, ties equal
    size = int(t_rank.max()) + 1
    tree = [0] * (size + 1)
    order = np.argsort(-a, kind="stable")
    a_sorted = a[order]
    i = 0
    while i < n:
        j = i
        while j < n and a_sorted[j] == a_sorted[i]:
            j += 1
        for k in order[i:j]:
            # count of inserted ranks strictly below this row's rank
            r = int(t_rank[k])
            s = 0
            while r > 0:
                s += tree[r]
                r -= r & -r
            out[k] = s
        for k in order[i:j]:
            r = int(t_rank[k]) + 1
            while r <= size:
                tree[r] += 1
                r += r & -r
        i = j
    return out.astype(float)


def aobt3_features(departures: pd.DataFrame) -> pd.DataFrame:
    """The C1 block for every row of ``departures``, indexed like it.

    ``departures`` carries ``REQUIRED_COLUMNS``; every row is both a subject
    and, where its ``AOBT_3_flt`` is present, one of the others.
    """
    for c in REQUIRED_COLUMNS:
        if c not in departures.columns:
            raise KeyError(f"departures need {c!r}")
    a = _seconds(departures[AOBT])
    t = _seconds(departures["MVT_TIME_UTC_mvt"])
    present = np.isfinite(a) & np.isfinite(t)
    out = pd.DataFrame(index=departures.index)
    out["aobt3_lag"] = t - a
    out["aobt3_vs_lobt"] = a - _seconds(departures["LOBT_flt"])
    out["aobt3_vs_eobt"] = a - _seconds(departures["EOBT_1_flt"])
    out["aobt3_missing"] = (~np.isfinite(a)).astype(float)
    for c in QUEUE_FEATURES:
        out[c] = np.nan
    airport = departures["ADEP_mvt"].astype(str).to_numpy()
    runway = departures["RUNWAY_mvt"].astype(str).to_numpy()
    idx = np.flatnonzero(present)
    values = {c: np.full(len(departures), np.nan) for c in QUEUE_FEATURES}
    for ap in np.unique(airport[idx]):
        rows = idx[airport[idx] == ap]
        surface, during = _surface_and_during(a[rows], t[rows])
        values["q_surface_off"][rows] = surface
        values["q_takeoffs_during"][rows] = during
        for rw in np.unique(runway[rows]):
            sub = rows[runway[rows] == rw]
            surface, during = _surface_and_during(a[sub], t[sub])
            values["q_surface_off_rwy"][sub] = surface
            values["q_takeoffs_during_rwy"][sub] = during
            values["q_overtakes_rwy"][sub] = _overtakes(a[sub], t[sub])
    for c, v in values.items():
        out[c] = v
    lagged = ~np.isfinite(a)
    out.loc[lagged, ["aobt3_lag", "aobt3_vs_lobt", "aobt3_vs_eobt"]] = np.nan
    return out[list(AOBT3_FEATURES)]


__all__ = [
    "AOBT3_FEATURES",
    "AOBT3_OWN",
    "QUEUE_FEATURES",
    "REQUIRED_COLUMNS",
    "SOURCES",
    "aobt3_features",
]
