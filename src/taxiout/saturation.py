# SPDX-License-Identifier: GPL-3.0-only
"""Runway saturation, from the other departures' take-off times alone.

The cheap Gate 3. A departure queue shows in the take-off times of the
aircraft ahead: when the runway is saturated, consecutive take-offs sit at
the minimum separation, the gaps close to the wake-turbulence floor, and the
last few take-offs span no more time than the separations demand. Three
readings of that, for the subject's runway, from other departures' take-off
times only, the subject removed by identity:

- ``sat_gap90_{5,15}m``: over the trailing five and fifteen minutes, the
  share of consecutive take-off gaps under ninety seconds; NaN with fewer
  than two other take-offs in the window;
- ``sat_min_gap_5m``: the smallest consecutive gap in the trailing five
  minutes, seconds; NaN with fewer than two;
- ``sat_span3_s``: the time spanned by the last three gaps -- the subject's
  take-off minus the third take-off before it on that runway -- capped at
  thirty minutes, with ``sat_span3_short`` set when fewer than three precede
  it (the span then sits at the cap).

Windows are cut by ``taxiout.windows.Sorted`` and are closed, ``[t - w, t]``;
a take-off at the subject's own second is one of the others. The trailing
sixty-minute coverage flag of the congestion block already marks every
subject whose fifteen-minute window reaches before its covered block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.windows import Sorted, seconds

GAP_S = 90.0
WINDOWS_MIN: tuple[int, ...] = (5, 15)
MIN_GAP_WINDOW_MIN = 5
SPAN_GAPS = 3
SPAN_CAP_S = 1800.0

SATURATION_FEATURES: tuple[str, ...] = tuple(f"sat_gap90_{w}m" for w in WINDOWS_MIN) + (
    f"sat_min_gap_{MIN_GAP_WINDOW_MIN}m",
    "sat_span3_s",
    "sat_span3_short",
)

_SELF = ("MVT_TIME_UTC_mvt", "ADEP_mvt", "RUNWAY_mvt", "MVT_ID_mvt")
_OTHERS = ("DEP.MVT_TIME_UTC_mvt", "DEP.ADEP_mvt", "DEP.RUNWAY_mvt", "DEP.MVT_ID_mvt")
SOURCES: dict[str, tuple[str, ...]] = {c: _SELF + _OTHERS for c in SATURATION_FEATURES}
MOVEMENT_COLUMNS: tuple[str, ...] = tuple(
    sorted({"PHASE_mvt"} | {c.split(".", 1)[1] for c in _OTHERS})
)

_FAR_S = 10 * 366 * 86400  # further back than any covered block reaches


class _RangeMin:
    """Minimum over ``[lo, hi]`` of an array, by a sparse table of the levels needed."""

    def __init__(self, values: np.ndarray, max_len: int):
        self.levels = [values]
        k = 1
        while (1 << k) <= max(max_len, 1):
            prev = self.levels[-1]
            half = 1 << (k - 1)
            cur = np.minimum(prev[:-half], prev[half:]) if len(prev) > half else prev[:0]
            self.levels.append(cur)
            k += 1

    def query(self, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        """Min over the closed index range; +inf where ``hi < lo``."""
        out = np.full(len(lo), np.inf)
        length = hi - lo + 1
        ok = length > 0
        if not ok.any():
            return out
        k = np.floor(np.log2(np.where(ok, length, 1))).astype(int)
        for level in range(len(self.levels)):
            sel = ok & (k == level)
            if not sel.any():
                continue
            tab = self.levels[level]
            a = tab[lo[sel]]
            b = tab[hi[sel] - (1 << level) + 1]
            out[sel] = np.minimum(a, b)
        return out


def saturation_features(
    departures: pd.DataFrame, movements: pd.DataFrame
) -> pd.DataFrame:
    """The saturation block for ``departures``, from their runway's other take-offs.

    ``movements`` carries ``MOVEMENT_COLUMNS``; only its ``DEP`` rows are
    read. The subjects are normally among them and are removed from their
    own runway's sequence by identifier whether or not they are. Indexed
    like ``departures``; carries exactly ``SATURATION_FEATURES``.
    """
    for c in MOVEMENT_COLUMNS:
        if c not in movements.columns:
            raise KeyError(f"movements need {c!r}")
    out = pd.DataFrame(index=departures.index)
    dep = movements[movements["PHASE_mvt"].astype(str) == "DEP"].reset_index(drop=True)
    events = Sorted.build(
        dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str),
        dep["MVT_TIME_UTC_mvt"],
    )
    n_ev = len(dep)
    times = seconds(dep["MVT_TIME_UTC_mvt"])[events.order].astype(float)
    codes_sorted = events.codes["_lookup"][events.order]
    # consecutive gaps within a group; +inf at a group boundary and at the front
    gap = np.full(n_ev, np.inf)
    if n_ev > 1:
        same = codes_sorted[1:] == codes_sorted[:-1]
        gap[1:] = np.where(same, times[1:] - times[:-1], np.inf)
    short = np.concatenate(
        [[0.0], np.cumsum((gap < GAP_S).astype(float))]
    )  # short[j+1] = count over 0..j

    # the subject's own position in the sorted sequence, -1 where it is not there
    pos_of_row = np.empty(n_ev, dtype=np.int64)
    pos_of_row[events.order] = np.arange(n_ev)
    lookup = pd.Series(pos_of_row, index=dep["MVT_ID_mvt"].to_numpy())
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    p = (
        lookup.reindex(departures["MVT_ID_mvt"].to_numpy())
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )

    key = departures["ADEP_mvt"].astype(str) + "|" + departures["RUNWAY_mvt"].astype(str)
    code = events.group_codes(key)
    t = departures["MVT_TIME_UTC_mvt"]
    tt = seconds(t)

    def _window(window_s: float):
        lo, hi = events.bounds(code, tt - int(window_s), tt)
        inside = (p >= lo) & (p < hi)
        return lo, hi, inside

    def _short_share(window_s: float) -> np.ndarray:
        lo, hi, inside = _window(window_s)
        n = hi - lo
        # gaps wholly inside the window: j = lo+1 .. hi-1
        count = short[np.maximum(hi, 1)] - short[np.minimum(lo + 1, np.maximum(hi, 1))]
        n_gaps = (n - 1).astype(float)
        # take the subject out: its two gaps go, the joined gap comes in
        left = inside & (p - 1 >= lo)
        right = inside & (p + 1 < hi)
        count = count - np.where(left, (gap[np.clip(p, 0, n_ev - 1)] < GAP_S), 0.0)
        count = count - np.where(right, (gap[np.clip(p + 1, 0, n_ev - 1)] < GAP_S), 0.0)
        joined = left & right
        j_gap = np.where(
            joined,
            times[np.clip(p + 1, 0, n_ev - 1)] - times[np.clip(p - 1, 0, n_ev - 1)],
            np.inf,
        )
        count = count + np.where(joined, (j_gap < GAP_S), 0.0)
        n_gaps = n_gaps - inside.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n_gaps >= 1, count / n_gaps, np.nan)

    for w in WINDOWS_MIN:
        out[f"sat_gap90_{w}m"] = _short_share(60 * w)

    # the smallest gap in the trailing window, the subject taken out
    lo, hi, inside = _window(60 * MIN_GAP_WINDOW_MIN)
    max_len = int((hi - lo).max()) if len(lo) else 1
    rmq = _RangeMin(gap, max_len)
    left = inside & (p - 1 >= lo)
    right = inside & (p + 1 < hi)
    first = np.where(inside, np.minimum(p, hi), hi)  # gaps lo+1 .. p-1, then p+2 .. hi-1
    a = rmq.query(lo + 1, np.where(inside, p - 1, hi - 1))
    b = rmq.query(np.where(inside, p + 2, hi), hi - 1)
    joined = left & right
    j_gap = np.where(
        joined,
        times[np.clip(p + 1, 0, n_ev - 1)] - times[np.clip(p - 1, 0, n_ev - 1)],
        np.inf,
    )
    del first
    mn = np.minimum(np.minimum(a, b), j_gap)
    out[f"sat_min_gap_{MIN_GAP_WINDOW_MIN}m"] = np.where(np.isfinite(mn), mn, np.nan)

    # the span of the last three gaps: the subject minus the third take-off before it
    g_lo, hi, inside = _window(_FAR_S)
    third = hi - SPAN_GAPS - np.where(inside & (p >= hi - SPAN_GAPS), 1, 0)
    enough = (third >= g_lo) & (code >= 0)
    ref = times[np.clip(third, 0, max(n_ev - 1, 0))] if n_ev else np.zeros(len(tt))
    span = np.where(enough, tt - ref, SPAN_CAP_S)
    out["sat_span3_s"] = np.minimum(span, SPAN_CAP_S)
    out["sat_span3_short"] = (~enough).astype(float)
    return out[list(SATURATION_FEATURES)]
