# SPDX-License-Identifier: GPL-3.0-only
"""Stand occupancy: the arrival that took the stand the subject had left.

A stand holds one aircraft. If an arrival on-blocks at the subject's stand
before the subject takes off, then the subject was already off the stand,
and its taxi-out is at least the gap between that on-block and its take-off.
The ranking set gives every arrival's stand and on-block time in full, so
that is a physical lower bound on the hidden field built entirely out of
exposed rows, and it is the operational quantity A-CDM calls stand
availability. At Heathrow in particular aircraft are pushed back to free the
stand and then held.

It is a feature and never a floor. Measured on the December validation rows,
by minutes from the last arrival's on-block at the stand to the subject's
take-off: below twenty minutes the bound holds in 99 to 100% of rows and the
model already respects it.

**The twenty-to-forty-five-minute band is where the bound is the mechanism.**
It is about 1% of the rows, the model under-predicted them by +185 s at
twenty-to-thirty and +120 s at thirty-to-forty-five, and the previous
arrival's aircraft type and operator say which of two things happened: the
subject's own inbound after a fast turnaround, or a different aircraft that
took the stand once the subject had pushed back. Where the type differs the
bias is +239 s and where it matches it is +55 s, and it is the differing-type
rows that the bound actually speaks about. Above forty-five minutes the bound
collapses even when type and operator both differ, because a stand identifier
can denote a position that holds two aircraft and because aircraft are towed.
Applied as a hard floor it was measured to lose 1 to 20 s; carried as a
feature, the booster learns the band it holds in.

**Most of what the block buys is not the bound.** The band above carries 22%
of the block's reduction in squared error in December and 36% on the three
of ``JULY_AIRPORTS`` -- a minority on every leg measured, and the
figures are in ``reports/occupancy.md`` rather than here. What the rest is,
is the stand's recent history: the previous arrival's lateness and taxi-in,
and whether it was the subject's own aircraft returning. At gaps past an hour
that is a turnaround-pattern signal about the stand's day and it bounds
nothing at all -- in December those rows are 48% of the reduction. Both halves
are admissible on the same footing, and this block should be described as the
stand's recent history with a bound inside it rather than as a bound.

There is no re-occupation flag among the columns. ``occ_gap1_s`` is that
flag with the threshold left in it, and the threshold is the booster's to
find rather than this module's to fix.

One limit is stated here rather than left to be noticed. The arrival this
block reads on-blocked *after* the subject's own off-block, so the block is
admissible under the challenge's post-hoc framing and not under an
operational one -- the same standing as every take-off-anchored window the
model already carries. It does bound the hidden field from inside the
challenge data.

The key is the airport and the stand, the stand folded for case and
whitespace and nothing else. Both sides of this match come from the movement
feed, where a stand is written one way; the staged normalisation in
``taxiout.join`` exists to bridge that feed to OpenStreetMap's, which is a
different problem and would here merge stands that the feed distinguishes. A
null stand or a null airport is no key and matches nothing. Windows follow
the repository's convention and are closed at the subject's take-off second,
so an arrival on-blocking in that second has happened. An arrival with no
on-block time is not an occupation of a stand at all and is dropped before
the sequence is built, since ``taxiout.windows.seconds`` would otherwise
sort it to the front of its stand and hand out the gap to a sentinel.

Where the stand received no arrival before the take-off -- 0.09% of the
training year and 0.007% of the December validation rows, so the block
abstains on almost nothing -- two flags say which of the two reasons applies:
``occ_missing`` for a take-off far enough into its covered block that the
absence is a fact about the stand, and
``occ_truncated`` for one near enough to the block's start that the previous
arrival may simply lie before the data, exactly as ``win_truncated`` says
for the congestion windows. A first-day departure whose stand did receive an
arrival has a real gap and carries neither flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.windows import Sorted, coverage_start, seconds

# The stand's previous arrival is looked for over the whole covered block
# rather than a window: the gap is itself the feature, and a window would fix
# here the threshold the docstring leaves to the booster.
_FAR_S = 10 * 366 * 86400

# How far into its covered block a take-off has to sit before "no arrival at
# this stand" is a fact about the stand rather than about the data's edge.
COVERAGE_MARGIN_S = 86400

OCCUPANCY_FEATURES: tuple[str, ...] = (
    "occ_gap1_s",
    "occ_gap2_s",
    "occ_gap1_sched_s",
    "occ_prev_late_s",
    "occ_prev_txi_s",
    "occ_prev_same_type",
    "occ_prev_same_op",
    "occ_prev_dep_gap_s",
    "occ_missing",
    "occ_truncated",
)

# The subject's own row: the take-off every gap is measured back from, and
# the stand it is measured at.
_ANCHOR = ("MVT_TIME_UTC_mvt", "ADEP_mvt", "STAND_mvt")
# What decides which arrival held the stand last, and when it took it.
_PREVIOUS = _ANCHOR + ("ARR.ADES_mvt", "ARR.STAND_mvt", "ARR.BLOCK_TIME_UTC_mvt")
_STAND_DEPARTURES = (
    "DEP.ADEP_mvt",
    "DEP.STAND_mvt",
    "DEP.MVT_TIME_UTC_mvt",
    "DEP.MVT_ID_mvt",
)

SOURCES: dict[str, tuple[str, ...]] = {
    "occ_gap1_s": _PREVIOUS,
    "occ_gap2_s": _PREVIOUS,
    "occ_gap1_sched_s": _PREVIOUS + ("SCHED_TIME_UTC_mvt",),
    "occ_prev_late_s": _PREVIOUS + ("ARR.SCHED_TIME_UTC_mvt",),
    "occ_prev_txi_s": _PREVIOUS + ("ARR.TAXITIME_SEC_mvt",),
    "occ_prev_same_type": _PREVIOUS + ("AIRCRAFT_TYPE_mvt", "ARR.AIRCRAFT_TYPE_mvt"),
    "occ_prev_same_op": _PREVIOUS
    + ("AIRCRAFT_OPERATOR_flt", "ARR.AIRCRAFT_OPERATOR_flt"),
    "occ_prev_dep_gap_s": _ANCHOR + ("MVT_ID_mvt",) + _STAND_DEPARTURES,
    "occ_missing": _PREVIOUS,
    "occ_truncated": _PREVIOUS,
}

MOVEMENT_COLUMNS: tuple[str, ...] = tuple(
    sorted(
        {"PHASE_mvt"}
        | {
            c.split(".", 1)[1]
            for sources in SOURCES.values()
            for c in sources
            if "." in c
        }
    )
)


def _phase(movements: pd.DataFrame, phase: str) -> pd.DataFrame:
    return movements[movements["PHASE_mvt"].astype(str) == phase]


def _stand_key(icao: pd.Series, stand: pd.Series) -> np.ndarray:
    """``AIRPORT|STAND`` folded for case and whitespace; empty where there is none."""
    folded = stand.astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    named = icao.notna() & stand.notna() & (folded.str.len() > 0)
    key = icao.astype(str) + "|" + folded
    return key.where(named, "").to_numpy()


def _delta_seconds(later: pd.Series, earlier: pd.Series) -> np.ndarray:
    """``later - earlier`` in seconds, NaN where either is missing."""
    delta = pd.to_datetime(later, utc=True) - pd.to_datetime(earlier, utc=True)
    return delta.dt.total_seconds().to_numpy(dtype=float)


def _labels(values: pd.Series) -> np.ndarray:
    """Values as strings, nulls as ``None``, so a comparison against one is unknown."""
    s = values.astype(object)
    return np.where(s.isna(), None, s.astype(str))


def _gather(values: np.ndarray, index: np.ndarray, ok: np.ndarray, fill=np.nan):
    """``values`` at ``index`` where ``ok``, ``fill`` elsewhere and throughout if empty."""
    if not len(values):
        return np.full(len(index), fill)
    return np.where(ok, values[np.clip(index, 0, len(values) - 1)], fill)


OCCUPANCY_TFREE_FEATURES: tuple[str, ...] = (
    "occt_gap1_sched_s",
    "occt_prev_late_s",
    "occt_prev_txi_s",
    "occt_prev_same_type",
    "occt_prev_same_op",
    "occt_missing",
    "occt_truncated",
)

# The schedule-anchored twin's anchor: the subject's scheduled time, and the
# stand it is measured at. The subject's take-off appears nowhere.
_TFREE_ANCHOR = ("SCHED_TIME_UTC_mvt", "ADEP_mvt", "STAND_mvt")
_TFREE_PREVIOUS = _TFREE_ANCHOR + (
    "ARR.ADES_mvt",
    "ARR.STAND_mvt",
    "ARR.BLOCK_TIME_UTC_mvt",
)

TFREE_SOURCES: dict[str, tuple[str, ...]] = {
    "occt_gap1_sched_s": _TFREE_PREVIOUS,
    "occt_prev_late_s": _TFREE_PREVIOUS + ("ARR.SCHED_TIME_UTC_mvt",),
    "occt_prev_txi_s": _TFREE_PREVIOUS + ("ARR.TAXITIME_SEC_mvt",),
    "occt_prev_same_type": _TFREE_PREVIOUS
    + ("AIRCRAFT_TYPE_mvt", "ARR.AIRCRAFT_TYPE_mvt"),
    "occt_prev_same_op": _TFREE_PREVIOUS
    + ("AIRCRAFT_OPERATOR_flt", "ARR.AIRCRAFT_OPERATOR_flt"),
    "occt_missing": _TFREE_PREVIOUS,
    "occt_truncated": _TFREE_PREVIOUS,
}


def occupancy_tfree_features(
    departures: pd.DataFrame, movements: pd.DataFrame
) -> pd.DataFrame:
    """The stand-occupancy block re-anchored on the SCHEDULE, not the take-off.

    A diagnostic twin of ``occupancy_features``, and the answer to one question:
    how much of what the stand-occupancy block buys survives if the block is
    forbidden the subject's take-off time entirely?

    Dropping the three columns that carry ``MVT_TIME_UTC_mvt`` in their
    arithmetic does **not** answer it. ``occ_gap1_sched_s`` cancels the take-off
    algebraically -- it is ``SCHED - onblock`` -- but the *selection* of which
    arrival is the stand's last runs ``bounds(code, tt - _FAR_S, tt)``, so the
    take-off decides which on-block the subtraction is against, and every
    ``_gather``ed column inherits that choice. Moving the take-off ten minutes
    with the schedule and both on-blocks held fixed moves ``occ_gap1_sched_s``
    by eighty and flips both match flags.

    So the anchor moves instead. The stand's last arrival is the last one to
    on-block at or before the subject's **scheduled** time, every column is
    read from that arrival, and the coverage flag is judged against the
    schedule too. What comes out is admissible under an operational framing --
    at the moment the subject is due off the stand, all of it is known.

    Read against the same rung as the full block, the difference between the two
    is what the take-off anchor is worth. Diagnostic only: these columns are not
    on the ladder's chain and not in the submission.
    """
    for c in MOVEMENT_COLUMNS:
        if c not in movements.columns:
            raise KeyError(f"movements need {c!r}")
    out = pd.DataFrame(index=departures.index)
    sched = departures["SCHED_TIME_UTC_mvt"]
    ss = seconds(sched)
    key = _stand_key(departures["ADEP_mvt"], departures["STAND_mvt"])

    arr = _phase(movements, "ARR")
    arr_key = _stand_key(arr["ADES_mvt"], arr["STAND_mvt"])
    onblock = pd.to_datetime(arr["BLOCK_TIME_UTC_mvt"], utc=True)
    held = (arr_key != "") & onblock.notna().to_numpy()
    arr, arr_key, onblock = arr[held], arr_key[held], onblock[held]
    events = Sorted.build(arr_key, onblock)
    order = events.order
    onb = seconds(onblock)[order].astype(float)

    code = events.group_codes(key)
    # The one line that makes this block take-off free: the search is bounded
    # at the schedule.
    lo, hi = events.bounds(code, ss - _FAR_S, ss)
    last = hi - 1
    found = last >= lo

    # ``SCHED - onblock`` directly, rather than as a difference of two
    # take-off-anchored quantities. Positive where the inbound reached the
    # stand before the outbound was due off it.
    out["occt_gap1_sched_s"] = ss.astype(float) - _gather(onb, last, found)
    out["occt_prev_late_s"] = _gather(
        _delta_seconds(onblock, arr["SCHED_TIME_UTC_mvt"])[order], last, found
    )
    out["occt_prev_txi_s"] = _gather(
        pd.to_numeric(arr["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(float)[order],
        last,
        found,
    )
    for column, name in (
        ("AIRCRAFT_TYPE_mvt", "occt_prev_same_type"),
        ("AIRCRAFT_OPERATOR_flt", "occt_prev_same_op"),
    ):
        mine = _labels(departures[column])
        theirs = _gather(_labels(arr[column])[order], last, found, fill=None)
        known = ~pd.isna(mine) & ~pd.isna(theirs)
        out[name] = np.where(known, (mine == theirs).astype(float), np.nan)

    early = ss < coverage_start(sched) + COVERAGE_MARGIN_S
    out["occt_missing"] = (~found & ~early).astype(float)
    out["occt_truncated"] = (~found & early).astype(float)
    return out[list(OCCUPANCY_TFREE_FEATURES)]


def occupancy_features(departures: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """The stand-occupancy block for ``departures``, from their own stand's traffic.

    ``movements`` is every movement of the same side -- both phases --
    carrying ``MOVEMENT_COLUMNS``. Arrivals give the stand's last two
    on-blocks before the subject's take-off and what the last of them was;
    departures give the previous take-off from the same stand, the subject
    removed by identifier whether or not it is among them. The result is
    indexed like ``departures`` and carries exactly ``OCCUPANCY_FEATURES``.
    """
    for c in MOVEMENT_COLUMNS:
        if c not in movements.columns:
            raise KeyError(f"movements need {c!r}")
    out = pd.DataFrame(index=departures.index)
    t = departures["MVT_TIME_UTC_mvt"]
    tt = seconds(t)
    key = _stand_key(departures["ADEP_mvt"], departures["STAND_mvt"])

    # -- the arrivals that held the stand
    arr = _phase(movements, "ARR")
    arr_key = _stand_key(arr["ADES_mvt"], arr["STAND_mvt"])
    onblock = pd.to_datetime(arr["BLOCK_TIME_UTC_mvt"], utc=True)
    held = (arr_key != "") & onblock.notna().to_numpy()
    arr, arr_key, onblock = arr[held], arr_key[held], onblock[held]
    events = Sorted.build(arr_key, onblock)
    order = events.order
    onb = seconds(onblock)[order].astype(float)

    code = events.group_codes(key)
    lo, hi = events.bounds(code, tt - _FAR_S, tt)
    last, second = hi - 1, hi - 2
    found, two = last >= lo, second >= lo

    gap1 = tt.astype(float) - _gather(onb, last, found)
    out["occ_gap1_s"] = gap1
    out["occ_gap2_s"] = tt.astype(float) - _gather(onb, second, two)
    # the same on-block against the subject's own plan rather than its
    # take-off: negative where the inbound reached the stand after the
    # outbound was due off it
    out["occ_gap1_sched_s"] = gap1 - _delta_seconds(t, departures["SCHED_TIME_UTC_mvt"])
    out["occ_prev_late_s"] = _gather(
        _delta_seconds(onblock, arr["SCHED_TIME_UTC_mvt"])[order], last, found
    )
    out["occ_prev_txi_s"] = _gather(
        pd.to_numeric(arr["TAXITIME_SEC_mvt"], errors="coerce").to_numpy(float)[order],
        last,
        found,
    )

    # An aircraft of the subject's own type and operator on the stand is the
    # subject's inbound after a turnaround; a different one took the stand
    # after the pushback. Unknown where either side has no record.
    for column, name in (
        ("AIRCRAFT_TYPE_mvt", "occ_prev_same_type"),
        ("AIRCRAFT_OPERATOR_flt", "occ_prev_same_op"),
    ):
        mine = _labels(departures[column])
        theirs = _gather(_labels(arr[column])[order], last, found, fill=None)
        known = ~pd.isna(mine) & ~pd.isna(theirs)
        out[name] = np.where(known, (mine == theirs).astype(float), np.nan)

    # -- the previous departure off the same stand
    dep = _phase(movements, "DEP")
    dep_key = _stand_key(dep["ADEP_mvt"], dep["STAND_mvt"])
    dep_t = pd.to_datetime(dep["MVT_TIME_UTC_mvt"], utc=True)
    flew = (dep_key != "") & dep_t.notna().to_numpy()
    dep, dep_key, dep_t = dep[flew], dep_key[flew], dep_t[flew]
    stand_deps = Sorted.build(dep_key, dep_t)
    n_dep = len(dep)
    dep_times = seconds(dep_t)[stand_deps.order].astype(float)

    # the subject's own place in its stand's sequence, -1 where the movement
    # frame does not hold it: it is removed by identity, never by time
    pos_of_row = np.empty(n_dep, dtype=np.int64)
    pos_of_row[stand_deps.order] = np.arange(n_dep)
    lookup = pd.Series(pos_of_row, index=dep["MVT_ID_mvt"].to_numpy())
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    p = (
        lookup.reindex(departures["MVT_ID_mvt"].to_numpy())
        .fillna(-1)
        .to_numpy(dtype=np.int64)
    )
    d_lo, d_hi = stand_deps.bounds(stand_deps.group_codes(key), tt - _FAR_S, tt)
    previous = np.where(p == d_hi - 1, d_hi - 2, d_hi - 1)
    out["occ_prev_dep_gap_s"] = tt.astype(float) - _gather(
        dep_times, previous, previous >= d_lo
    )

    # -- no arrival at all: a fact about the stand, or about the data's edge
    early = tt < coverage_start(t) + COVERAGE_MARGIN_S
    out["occ_missing"] = (~found & ~early).astype(float)
    out["occ_truncated"] = (~found & early).astype(float)
    return out[list(OCCUPANCY_FEATURES)]
