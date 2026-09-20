# SPDX-License-Identifier: GPL-3.0-only
"""The tails report's callsign-prefix section, on a synthetic year.

The section exists to settle a review note, so what is worth holding to a test
is the thing the note turned on: that ``|taxi - lag| <= NEAR_S`` and
``|block - schedule| <= NEAR_S`` are one test and not two, and that what
separates the note's reading from ``taxiout.tails.defaulted_block`` is the
two-hour condition alone. Beside it, the guard every published report carries:
the section prints operator designators and cell labels, and must not print a
stand identifier it was handed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import report_tails as rt  # noqa: E402
from taxiout.tails import NEAR_S, TailModel, flag, lag_seconds  # noqa: E402

HOUR = 3600
T0 = pd.Timestamp("2025-03-01T06:00:00Z")


def _dep(rows) -> pd.DataFrame:
    """Departures as ``(prefix, lag_s, record, block_offset_s, taxi_s)``.

    ``block_offset_s`` is the block time against the schedule; ``None`` derives
    it from the taxi, so ``taxi = take-off - block`` holds as it does in the
    data. Every row is at ``rt.PREFIX_AIRPORT`` and on the same stand, which is
    what the guard looks for.
    """
    recs = []
    for i, (prefix, lag_s, record, block_off, taxi) in enumerate(rows):
        sched = T0 + pd.Timedelta(minutes=13 * i)
        mvt = sched + pd.Timedelta(seconds=lag_s)
        block = (
            mvt - pd.Timedelta(seconds=taxi)
            if block_off is None
            else sched + pd.Timedelta(seconds=block_off)
        )
        recs.append(
            {
                "ADEP_mvt": rt.PREFIX_AIRPORT,
                "MVT_TIME_UTC_mvt": mvt,
                "SCHED_TIME_UTC_mvt": sched,
                "BLOCK_TIME_UTC_mvt": block,
                "LOBT_flt": sched if record else pd.NaT,
                "TAXITIME_SEC_mvt": np.int32(taxi),
                "STAND_mvt": "A12",
                "RUNWAY_mvt": "16R",
                "FLIGHT_mvt": f"{prefix}{100 + i}",
            }
        )
    return _derive(pd.DataFrame(recs))


def _derive(dep: pd.DataFrame) -> pd.DataFrame:
    """The columns ``rt.load`` adds, added the same way."""
    from taxiout.callsign import callsign_features
    from taxiout.tails import defaulted_block

    dep = dep.copy()
    dep["y"] = pd.to_numeric(dep["TAXITIME_SEC_mvt"], errors="coerce").astype(float)
    dep["lag"] = lag_seconds(dep)
    dep["month"] = dep["MVT_TIME_UTC_mvt"].dt.month
    dep["art"] = defaulted_block(dep).fillna(False).astype(bool)
    dep["bucket"] = flag(dep)["tail_bucket"].astype(str)
    dep["prefix"] = callsign_features(dep)["cs_prefix"].to_numpy()
    dep["near_lag"] = ((dep["y"] - dep["lag"]).abs() <= NEAR_S).fillna(False)
    dep["near_block"] = (
        (dep["BLOCK_TIME_UTC_mvt"] - dep["SCHED_TIME_UTC_mvt"])
        .dt.total_seconds()
        .abs()
        .le(NEAR_S)
        .fillna(False)
    )
    return dep


# A punctual operator: pushes back on schedule, ordinary taxi, so the taxi
# equals the lag in every row and none of it is a defaulted block time. And a
# defaulted one: block time on the schedule, take-off many hours later.
PUNCTUAL = [("PUN", 1_200, True, 0, 1_200)] * 8
DEFAULTED = [("DEF", 5 * HOUR, False, 0, 5 * HOUR)] * 8
ORDINARY = [("ORD", 1_800, True, 600, 1_200)] * 8


def test_the_two_readings_are_one_test_plus_the_two_hour_condition():
    dep = _dep(PUNCTUAL + DEFAULTED + ORDINARY)
    assert (dep["near_lag"] == dep["near_block"]).all()
    # the punctual operator is caught by the note's reading and by nothing else
    pun = dep[dep["prefix"] == "PUN"]
    assert pun["near_lag"].all() and not pun["art"].any()
    # the defaulted one is caught by both
    dfl = dep[dep["prefix"] == "DEF"]
    assert dfl["near_lag"].all() and dfl["art"].all()
    # and an aircraft that pushed back late is caught by neither
    ordinary = dep[dep["prefix"] == "ORD"]
    assert not ordinary["near_lag"].any() and not ordinary["art"].any()


def test_the_section_publishes_no_stand_it_was_handed():
    dep = _dep(PUNCTUAL + DEFAULTED + ORDINARY)
    rk = dep.head(4).copy()
    text = "\n".join(rt.section_prefixes(dep, rk, TailModel().fit(dep)))
    rt.check_no_identifiers(text, dep)
    assert "A12" not in text
    # what it does print is operator designators and cell labels
    assert "PUN" in text and "DEF" in text


def test_the_section_separates_the_punctual_operator_from_the_defaulted_one():
    dep = _dep(PUNCTUAL + DEFAULTED + ORDINARY)
    text = "\n".join(rt.section_prefixes(dep, dep.head(0), TailModel().fit(dep)))
    rows = [line for line in text.splitlines() if line.startswith("| PUN |")]
    assert rows, text
    # rows, defaulted under the term, caught by the note
    assert rows[0].split("|")[3].strip() == "0.0%"
    assert rows[0].split("|")[4].strip() == "100.0%"


def test_a_cell_inside_a_covered_cell_opens_nothing():
    """Splitting a covered cell by prefix moves no row: it is covered already."""
    dep = _dep(DEFAULTED * 2)
    model = TailModel().fit(dep)
    assert len(model.covered), "the synthetic year should cover its one cell"
    text = "\n".join(rt.section_prefixes(dep, dep.head(0), model))
    assert "would open nothing and touch no ranking row" in text


def test_the_section_says_so_where_the_airport_is_absent():
    dep = _dep(ORDINARY)
    dep["ADEP_mvt"] = "EDDF"
    text = "\n".join(rt.section_prefixes(dep, dep.head(0), TailModel().fit(dep)))
    assert f"No {rt.PREFIX_AIRPORT} departures" in text
