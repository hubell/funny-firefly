# SPDX-License-Identifier: GPL-3.0-only
"""Counts and sums of surrounding movements in windows anchored on a time.

Every congestion feature in Gate 2 is one of three shapes, and this module
is those three shapes and nothing about what they count:

- a **trailing** window: events in ``[t - w, t]`` for a subject anchored at
  ``t``, its own take-off;
- a **centred** window: events in ``[s - w, s + w]`` around a subject's
  scheduled time ``s``;
- a **trailing sum**: the sum and count of some value over the events of a
  trailing window, from which a mean follows.

Three rules are fixed here once, so that every feature built on them is
constructed identically on the training side and the ranking side.

**The window is closed at both ends.** An event at exactly ``t - w`` counts,
and an event at exactly ``t`` counts. Timestamps are recorded to the second
and two movements at one airport in the same second are ordinary, so the
boundary has to be decided, and *closed* is the decision: a take-off in the
same second as the subject's is a take-off that has happened.

**The subject never counts itself.** Where the subject is one of the events
-- a take-off among take-offs, a schedule among schedules -- it is inside
its own window by construction and is removed by identity, not by time. So
the tie-break for equal timestamps is not a tie-break at all: nothing is
ordered by identifier, and the count is the same whatever order equal
timestamps sort in.

**A window that reaches outside the data is flagged, not silently short.**
Each side of the data covers whole months, and a window near the start of a
covered block reaches into a month that is not there: the first hour of
January 2025 on the training side, the first hour of January and of July
2026 on the ranking side. ``coverage_start`` finds each row's block from the
months its own frame contains, and ``truncated`` says whether a window
crosses it. The count is still returned -- it is what the data supports --
and the flag is a feature beside it.

Everything is vectorised through ``searchsorted`` on a composite key of
group and time, so a year of movements costs a sort and a few binary
searches per window rather than a pass per subject.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Seconds in a group key's time part. Well above any span the data has and
# below what an int64 leaves once a few thousand groups are multiplied in.
_TIME_SPAN = 1 << 40


def seconds(t: pd.Series) -> np.ndarray:
    """Timestamps as int64 seconds since the epoch; NaT becomes the minimum.

    The unit is forced to nanoseconds before the integer view is taken. The
    challenge parquet files carry microsecond timestamps, which pandas 2
    keeps, and an integer view of those divided as if they were nanoseconds
    puts a whole year into thirty thousand "seconds" -- every hour-long
    window then spans six weeks, and every row is truncated. The unit tests
    build their stamps at nanosecond resolution and never saw it; the sanity
    check on the real year did.
    """
    ts = pd.to_datetime(t, utc=True).dt.as_unit("ns")
    out = (ts.astype("int64") // 1_000_000_000).to_numpy()
    return np.where(ts.isna().to_numpy(), np.iinfo(np.int64).min // 4, out)


@dataclass
class Sorted:
    """Events sorted by (group, time), ready for windows to be cut from.

    ``key`` is ``group * _TIME_SPAN + t`` so that a single sorted array serves
    every group; ``cumulative`` are prefix sums of any values attached, with a
    leading zero, so a window's sum is one subtraction.
    """

    key: np.ndarray
    order: np.ndarray
    n_groups: int
    codes: dict[str, np.ndarray]
    cumulative: dict[str, np.ndarray]

    @classmethod
    def build(
        cls,
        group: pd.Series | np.ndarray,
        t: pd.Series,
        values: dict[str, pd.Series | np.ndarray] | None = None,
    ) -> "Sorted":
        codes, uniques = pd.factorize(pd.Series(np.asarray(group)).astype(str))
        tt = seconds(t)
        valid = codes >= 0
        key = np.where(
            valid, codes.astype(np.int64) * _TIME_SPAN + tt, np.iinfo(np.int64).min
        )
        order = np.argsort(key, kind="stable")
        cumulative = {}
        for name, v in (values or {}).items():
            raw = pd.to_numeric(pd.Series(np.asarray(v)), errors="coerce").to_numpy(
                dtype=float
            )
            present = valid & np.isfinite(raw)
            arr = np.where(present, raw, 0.0)
            cumulative[name] = np.concatenate([[0.0], np.cumsum(arr[order])])
            # a second prefix sum of "value present", so a mean divides by the
            # rows that carried one rather than by the rows in the window
            cumulative[name + "__n"] = np.concatenate(
                [[0.0], np.cumsum(present[order].astype(float))]
            )
        return cls(
            key=key[order],
            order=order,
            n_groups=len(uniques),
            codes={"_lookup": codes, "_uniques": uniques},
            cumulative=cumulative,
        )

    def group_codes(self, group: pd.Series | np.ndarray) -> np.ndarray:
        """Codes for another frame's groups, -1 where the group was never seen."""
        lookup = pd.Index(self.codes["_uniques"].astype(str))
        return lookup.get_indexer(pd.Series(np.asarray(group)).astype(str))

    def bounds(self, code: np.ndarray, lo_t: np.ndarray, hi_t: np.ndarray) -> tuple:
        """Index range ``[lo, hi)`` of events in ``[lo_t, hi_t]`` per anchor."""
        base = code.astype(np.int64) * _TIME_SPAN
        lo = np.searchsorted(self.key, base + lo_t, side="left")
        hi = np.searchsorted(self.key, base + hi_t, side="right")
        missing = code < 0
        lo = np.where(missing, 0, lo)
        hi = np.where(missing, 0, hi)
        return lo, hi


def trailing_count(
    events: Sorted,
    group: pd.Series | np.ndarray,
    t: pd.Series,
    window_s: float,
    *,
    subtract_self: bool = False,
) -> np.ndarray:
    """Events of the subject's group in ``[t - w, t]``, closed at both ends.

    ``subtract_self`` removes one for the subject's own event, for the case
    where the anchors are themselves among the events. Never conditional on
    time: a subject is inside its own closed window by construction.
    """
    code = events.group_codes(group)
    tt = seconds(t)
    lo, hi = events.bounds(code, tt - int(window_s), tt)
    out = (hi - lo).astype(float)
    if subtract_self:
        out = out - 1.0
    return np.where(code < 0, 0.0, out)


def centred_count(
    events: Sorted,
    group: pd.Series | np.ndarray,
    t: pd.Series,
    half_window_s: float,
    *,
    subtract_self: bool = False,
) -> np.ndarray:
    """Events of the subject's group in ``[t - w, t + w]``, closed at both ends."""
    code = events.group_codes(group)
    tt = seconds(t)
    lo, hi = events.bounds(code, tt - int(half_window_s), tt + int(half_window_s))
    out = (hi - lo).astype(float)
    if subtract_self:
        out = out - 1.0
    return np.where(code < 0, 0.0, out)


def trailing_mean(
    events: Sorted,
    name: str,
    group: pd.Series | np.ndarray,
    t: pd.Series,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and count of a value over the events of a trailing window.

    The count is of events that carried the value; the mean is NaN where
    that count is zero. Anchors are never among these events -- the values
    are arrivals', read by departures -- so nothing is subtracted.
    """
    code = events.group_codes(group)
    tt = seconds(t)
    lo, hi = events.bounds(code, tt - int(window_s), tt)
    total = events.cumulative[name][hi] - events.cumulative[name][lo]
    n = events.cumulative[name + "__n"][hi] - events.cumulative[name + "__n"][lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, total / n, np.nan)
    return mean, np.where(code < 0, 0.0, n)


# --------------------------------------------------------------------------
# coverage


def coverage_start(t: pd.Series) -> np.ndarray:
    """Start of each row's block of contiguous covered months, as seconds.

    The months are read from the frame itself: whichever calendar months it
    contains, grouped into runs of consecutive months, and each row assigned
    the first instant of the run its own month belongs to. The training year
    is one run; the ranking set is two, January and July 2026.
    """
    ts = pd.to_datetime(t, utc=True)
    period = ts.dt.tz_convert(None).dt.to_period("M")
    months = sorted(period.dropna().unique())
    if not months:
        return np.full(len(t), np.iinfo(np.int64).min // 4)
    start_of: dict = {}
    run_start = months[0]
    prev = months[0]
    for m in months:
        if (m - prev).n > 1:
            run_start = m
        start_of[m] = run_start
        prev = m
    starts = period.map(start_of)
    first = pd.to_datetime(starts.astype(str), utc=True)
    return seconds(first)


def truncated(t: pd.Series, window_s: float, block_start: np.ndarray) -> np.ndarray:
    """Whether ``[t - w, t]`` reaches before the row's covered block."""
    return (seconds(t) - int(window_s)) < block_start


def truncated_centred(
    t: pd.Series, half_window_s: float, block_start: np.ndarray, block_end: np.ndarray
) -> np.ndarray:
    """Whether ``[t - w, t + w]`` reaches outside the row's covered block."""
    tt = seconds(t)
    return ((tt - int(half_window_s)) < block_start) | (
        (tt + int(half_window_s)) > block_end
    )


def coverage_end(t: pd.Series) -> np.ndarray:
    """First instant after each row's block of contiguous covered months."""
    ts = pd.to_datetime(t, utc=True)
    period = ts.dt.tz_convert(None).dt.to_period("M")
    months = sorted(period.dropna().unique())
    if not months:
        return np.full(len(t), np.iinfo(np.int64).max // 4)
    end_of: dict = {}
    i = 0
    while i < len(months):
        j = i
        while j + 1 < len(months) and (months[j + 1] - months[j]).n == 1:
            j += 1
        end = months[j] + 1
        for k in range(i, j + 1):
            end_of[months[k]] = end
        i = j + 1
    ends = period.map(end_of)
    return seconds(pd.to_datetime(ends.astype(str), utc=True))


# --------------------------------------------------------------------------
# runway configuration


def configuration(
    group: pd.Series | np.ndarray,
    t: pd.Series,
    designator: pd.Series | np.ndarray,
    *,
    window_s: float = 3600.0,
    min_movements: int = 2,
    max_since_s: float = 6 * 3600.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The set of runways in use at each movement, and how long it has held.

    Per group, movements are walked in time order with a two-pointer window
    of ``window_s``; the configuration at a movement is the sorted set of
    designators with at least ``min_movements`` movements in the trailing
    window -- the floor suppresses the flicker of a single stray movement on
    an otherwise idle runway. A change is a movement whose set differs from
    the previous movement's; ``since`` is the seconds since the last change,
    capped at ``max_since_s``.

    Returns, per input row: the configuration as a ``|``-joined string, the
    seconds since it last changed, and the number of runways in it. Rows with
    no designator contribute nothing to the sets and receive the current one.

    The set is kept incrementally -- a designator enters it when its tally
    crosses the floor upwards and leaves when it crosses downwards -- so the
    string is rebuilt only at a change, and four million movements walk in
    seconds rather than minutes.
    """
    g = pd.Series(np.asarray(group)).astype(str).to_numpy()
    tt = seconds(t)
    des = pd.Series(np.asarray(designator)).astype(object)
    des = np.where(des.isna() | (des.astype(str) == "nan"), None, des.astype(str))
    n = len(g)
    order = np.lexsort((tt, g))
    config = np.empty(n, dtype=object)
    since = np.full(n, np.nan)
    count = np.zeros(n)
    w = int(window_s)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and g[order[j + 1]] == g[order[i]]:
            j += 1
        idx = order[i : j + 1]
        times = tt[idx]
        tally: dict[str, int] = {}
        active: set[str] = set()
        lo = 0
        current = ""
        n_active = 0
        previous = None
        changed_at = times[0] if len(times) else 0
        dirty = False
        for k in range(len(idx)):
            d = des[idx[k]]
            if d is not None:
                c = tally.get(d, 0) + 1
                tally[d] = c
                if c == min_movements:
                    active.add(d)
                    dirty = True
            while times[lo] < times[k] - w:
                old = des[idx[lo]]
                if old is not None:
                    c = tally[old] - 1
                    if c:
                        tally[old] = c
                    else:
                        del tally[old]
                    if c == min_movements - 1:
                        active.discard(old)
                        dirty = True
                lo += 1
            if dirty:
                current = "|".join(sorted(active))
                n_active = len(active)
                dirty = False
            if previous is not None and current != previous:
                changed_at = times[k]
            previous = current
            config[idx[k]] = current
            since[idx[k]] = min(float(times[k] - changed_at), max_since_s)
            count[idx[k]] = n_active
        i = j + 1
    return config, since, count
