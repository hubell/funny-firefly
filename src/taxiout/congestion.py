# SPDX-License-Identifier: GPL-3.0-only
"""The excess term's inputs: what the ranking set exposes about surrounding traffic.

Gate 2 of ``docs/plan.md``. The operational literature builds congestion
from other departures' pushbacks, and the ranking set blanks every
departure's off-block time, so none of that is available at inference. What
the ranking set does expose for the movements around a subject, and what
every feature here is built from:

- **Throughput.** Take-offs from the subject's runway, and from every runway
  at its airport, in trailing windows before its own take-off. Under the
  post-hoc framing (*Leakage watch* in ``docs/modelling_notes.md``) the
  subject's take-off time is given, so the runway's realised service rate
  ahead of it is a legitimate input.
- **Arrivals.** Landings and on-blocks in the same windows, and the mean and
  count of taxi-in times of the arrivals that reached their stand in the
  window. The ranking set gives arrivals' block times and taxi-in times in
  full, and a slow taxi-in is a direct measurement of the surface the subject
  is taxiing on.
- **Demand.** Scheduled departures in windows around the subject's own
  schedule. Causal under either framing.
- **Configuration.** The set of runways in use in the trailing hour, from
  every movement's designator, and how long that set has held; taxi times
  are elevated after a switch.

Every feature has an explicit as-of time -- the subject's take-off, or its
schedule for demand -- and is registered in ``SOURCES`` with the phase each
raw column is read from, so ``tests/test_features.py`` can hold departures'
sources to ``FORBIDDEN_FEATURE_INPUTS`` and arrivals' to
``AVAILABLE_FOR_ARRIVALS``. The subject is removed from its own count by
identity. Windows that reach before the covered block are flagged, not
silently short (``taxiout.windows``).

Counts need no coverage normalisation. January 2026 is a census of all ten
airports and July 2026 of the three it contains, each in full
(``reports/data_review.md``), so a count of movements in a window means the
same thing on the ranking side as on the training side within any
(airport, month) that is present at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from taxiout.challenge import NON_RUNWAY_VALUES
from taxiout.windows import (
    Sorted,
    centred_count,
    configuration,
    coverage_end,
    coverage_start,
    trailing_count,
    trailing_mean,
    truncated,
    truncated_centred,
)

# Trailing windows, minutes. Five is the runway's immediate queue, sixty is
# the regime; the two between are where the literature's counts sit.
WINDOWS_MIN: tuple[int, ...] = (5, 15, 30, 60)

# Half-widths of the centred demand windows, minutes.
DEMAND_HALF_MIN: tuple[int, ...] = (15, 30, 60)

# Configuration: the trailing hour, and the floor below which a designator
# is a stray rather than a runway in use.
CONFIGURATION_WINDOW_S = 3600.0
CONFIGURATION_MIN_MOVEMENTS = 2
CONFIGURATION_MAX_SINCE_S = 6 * 3600.0

THROUGHPUT_FEATURES: tuple[str, ...] = tuple(
    f"tko_rwy_{w}m" for w in WINDOWS_MIN
) + tuple(f"tko_all_{w}m" for w in WINDOWS_MIN)
ARRIVAL_FEATURES: tuple[str, ...] = tuple(
    name
    for w in WINDOWS_MIN
    for name in (f"ldg_{w}m", f"onb_{w}m", f"txi_n_{w}m", f"txi_mean_{w}m")
)
DEMAND_FEATURES: tuple[str, ...] = tuple(f"sched_pm{h}m" for h in DEMAND_HALF_MIN)
CONFIGURATION_FEATURES: tuple[str, ...] = (
    "cfg_n_runways",
    "cfg_since_change_s",
)
COVERAGE_FEATURES: tuple[str, ...] = ("win_truncated", "sched_truncated")

CONGESTION_FEATURES: tuple[str, ...] = (
    THROUGHPUT_FEATURES
    + ARRIVAL_FEATURES
    + DEMAND_FEATURES
    + CONFIGURATION_FEATURES
    + COVERAGE_FEATURES
)
# The raw configuration set -- the designators in use, as one categorical --
# was a feature until the ladder showed it taking 12% of the top rung's
# splits and buying nothing on any split: a high-cardinality categorical
# memorizing airport-hours. It is now a diagnostic only, exposed by
# ``configuration_sets`` for the report; the model reads the count of
# runways in use and the seconds since the set last changed.
CATEGORICAL: tuple[str, ...] = ()

# The subject's own columns every feature here anchors on.
_SELF = ("MVT_TIME_UTC_mvt", "ADEP_mvt", "MVT_ID_mvt")
_SELF_SCHED = ("SCHED_TIME_UTC_mvt", "ADEP_mvt", "MVT_ID_mvt")

# Sources, phase-qualified. A bare name is the subject's row; ``DEP.`` and
# ``ARR.`` are other movements' rows, read from the movement frame.
SOURCES: dict[str, tuple[str, ...]] = {
    **{
        f"tko_rwy_{w}m": _SELF
        + (
            "RUNWAY_mvt",
            "DEP.MVT_TIME_UTC_mvt",
            "DEP.ADEP_mvt",
            "DEP.RUNWAY_mvt",
            "DEP.MVT_ID_mvt",
        )
        for w in WINDOWS_MIN
    },
    **{
        f"tko_all_{w}m": _SELF
        + ("DEP.MVT_TIME_UTC_mvt", "DEP.ADEP_mvt", "DEP.MVT_ID_mvt")
        for w in WINDOWS_MIN
    },
    **{
        f"ldg_{w}m": _SELF + ("ARR.MVT_TIME_UTC_mvt", "ARR.ADES_mvt") for w in WINDOWS_MIN
    },
    **{
        f"onb_{w}m": _SELF + ("ARR.BLOCK_TIME_UTC_mvt", "ARR.ADES_mvt")
        for w in WINDOWS_MIN
    },
    **{
        name: _SELF + ("ARR.BLOCK_TIME_UTC_mvt", "ARR.TAXITIME_SEC_mvt", "ARR.ADES_mvt")
        for w in WINDOWS_MIN
        for name in (f"txi_n_{w}m", f"txi_mean_{w}m")
    },
    **{
        f"sched_pm{h}m": _SELF_SCHED
        + ("DEP.SCHED_TIME_UTC_mvt", "DEP.ADEP_mvt", "DEP.MVT_ID_mvt")
        for h in DEMAND_HALF_MIN
    },
    **{
        c: _SELF
        # The subject's own runway: a subject the movement frame does not hold is
        # appended to it as a movement of its own, from its bare columns, in
        # ``_configuration``. Every subject is in the frame at inference, so the
        # branch is latent there, but it reads the column.
        + ("RUNWAY_mvt",)
        + (
            "DEP.MVT_TIME_UTC_mvt",
            "DEP.RUNWAY_mvt",
            "DEP.ADEP_mvt",
            "DEP.MVT_ID_mvt",
            "ARR.MVT_TIME_UTC_mvt",
            "ARR.RUNWAY_mvt",
            "ARR.ADES_mvt",
            "ARR.MVT_ID_mvt",
        )
        for c in CONFIGURATION_FEATURES
    },
    "win_truncated": ("MVT_TIME_UTC_mvt",),
    # The take-off as well as the schedule: the covered block the schedule is
    # judged against starts and ends with the take-off's calendar month.
    "sched_truncated": ("MVT_TIME_UTC_mvt", "SCHED_TIME_UTC_mvt"),
}

# What the movement frame has to carry: every prefixed source, stripped, plus
# the phase that says which prefix a row answers to.
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


def _designators(values: pd.Series) -> np.ndarray:
    """Designators as strings, with the sentinels that name no runway as None."""
    s = values.astype(object)
    s = np.where(s.isna(), None, s.astype(str))
    return np.array(
        [None if (v is None or v in NON_RUNWAY_VALUES) else v for v in s], dtype=object
    )


def congestion_features(
    departures: pd.DataFrame, movements: pd.DataFrame
) -> pd.DataFrame:
    """The congestion block for ``departures``, from the movements around them.

    ``departures`` are the subjects, carrying the bare sources in ``SOURCES``.
    ``movements`` is every movement of the same side -- both phases -- carrying
    ``MOVEMENT_COLUMNS``; the subjects are normally among them, and are
    removed from their own counts by identifier whether or not they are.
    The result is indexed like ``departures`` and carries exactly
    ``CONGESTION_FEATURES``.
    """
    for c in MOVEMENT_COLUMNS:
        if c not in movements.columns:
            raise KeyError(f"movements need {c!r}")
    out = pd.DataFrame(index=departures.index)
    dep = _phase(movements, "DEP")
    arr = _phase(movements, "ARR")
    icao = departures["ADEP_mvt"].astype(str)
    t = departures["MVT_TIME_UTC_mvt"]
    ids = departures["MVT_ID_mvt"]

    # -- throughput
    self_in_dep = ids.isin(dep["MVT_ID_mvt"]).to_numpy().astype(float)
    by_runway = Sorted.build(
        dep["ADEP_mvt"].astype(str) + "|" + dep["RUNWAY_mvt"].astype(str),
        dep["MVT_TIME_UTC_mvt"],
    )
    by_airport = Sorted.build(dep["ADEP_mvt"].astype(str), dep["MVT_TIME_UTC_mvt"])
    rwy_key = icao + "|" + departures["RUNWAY_mvt"].astype(str)
    for w in WINDOWS_MIN:
        out[f"tko_rwy_{w}m"] = trailing_count(by_runway, rwy_key, t, 60 * w) - self_in_dep
        out[f"tko_all_{w}m"] = trailing_count(by_airport, icao, t, 60 * w) - self_in_dep

    # -- arrivals: landings by landing time, on-blocks and taxi-in by block time
    landings = Sorted.build(arr["ADES_mvt"].astype(str), arr["MVT_TIME_UTC_mvt"])
    onblocks = Sorted.build(
        arr["ADES_mvt"].astype(str),
        arr["BLOCK_TIME_UTC_mvt"],
        values={"txi": arr["TAXITIME_SEC_mvt"]},
    )
    for w in WINDOWS_MIN:
        out[f"ldg_{w}m"] = trailing_count(landings, icao, t, 60 * w)
        out[f"onb_{w}m"] = trailing_count(onblocks, icao, t, 60 * w)
        mean, n = trailing_mean(onblocks, "txi", icao, t, 60 * w)
        out[f"txi_n_{w}m"] = n
        out[f"txi_mean_{w}m"] = mean

    # -- demand: schedules around the subject's schedule
    s = departures["SCHED_TIME_UTC_mvt"]
    schedules = Sorted.build(dep["ADEP_mvt"].astype(str), dep["SCHED_TIME_UTC_mvt"])
    for h in DEMAND_HALF_MIN:
        out[f"sched_pm{h}m"] = centred_count(schedules, icao, s, 60 * h) - self_in_dep

    # -- configuration, at every movement, read back by the subject's identifier
    got = _configuration(departures, movements)
    out["cfg_n_runways"] = got["n"].fillna(0.0).to_numpy()
    out["cfg_since_change_s"] = got["since"].to_numpy()

    # -- coverage flags, from the subjects' own months
    start = coverage_start(t)
    end = coverage_end(t)
    out["win_truncated"] = truncated(t, 60 * max(WINDOWS_MIN), start).astype(float)
    out["sched_truncated"] = truncated_centred(
        s, 60 * max(DEMAND_HALF_MIN), start, end
    ).astype(float)
    return out[list(CONGESTION_FEATURES)]


def configuration_sets(departures: pd.DataFrame, movements: pd.DataFrame) -> pd.Series:
    """The designator set in use at each subject's take-off, ``"09|27"`` style.

    A diagnostic for the report, not a feature: see ``CATEGORICAL``. Empty
    where no runway had two movements in the trailing hour.
    """
    got = _configuration(departures, movements)
    return pd.Series(got["cfg"].fillna("").to_numpy(), index=departures.index)


def _configuration(departures: pd.DataFrame, movements: pd.DataFrame) -> pd.DataFrame:
    """The configuration at every movement, read back by the subjects' identifiers."""
    ids = departures["MVT_ID_mvt"]
    events = movements[
        [
            "MVT_ID_mvt",
            "PHASE_mvt",
            "ADEP_mvt",
            "ADES_mvt",
            "MVT_TIME_UTC_mvt",
            "RUNWAY_mvt",
        ]
    ]
    absent = ~ids.isin(events["MVT_ID_mvt"])
    if absent.any():
        # A subject the movement frame does not hold still gets a configuration:
        # it is added as a movement of its own, which is what it is.
        extra = departures.loc[
            absent, ["MVT_ID_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]
        ].copy()
        extra["PHASE_mvt"] = "DEP"
        extra["ADES_mvt"] = None
        events = pd.concat([events, extra[events.columns]], ignore_index=True)
    ev_airport = np.where(
        events["PHASE_mvt"].astype(str).to_numpy() == "DEP",
        events["ADEP_mvt"].astype(str).to_numpy(),
        events["ADES_mvt"].astype(str).to_numpy(),
    )
    cfg, since, n_rwy = configuration(
        ev_airport,
        events["MVT_TIME_UTC_mvt"],
        _designators(events["RUNWAY_mvt"]),
        window_s=CONFIGURATION_WINDOW_S,
        min_movements=CONFIGURATION_MIN_MOVEMENTS,
        max_since_s=CONFIGURATION_MAX_SINCE_S,
    )
    lookup = pd.DataFrame(
        {"cfg": cfg, "since": since, "n": n_rwy}, index=events["MVT_ID_mvt"].to_numpy()
    )
    lookup = lookup[~lookup.index.duplicated(keep="first")]
    return lookup.reindex(ids.to_numpy())
