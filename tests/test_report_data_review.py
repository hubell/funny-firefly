# SPDX-License-Identifier: GPL-3.0-only
"""The data review's disruption-day scan, on a synthetic year.

The scan is read on both sides of the data at one rule, and the report sets
the two tables against each other, so the rule has to mean the same thing on
each: the same threshold, the same per-airport baseline, and a run of
consecutive days counted the same way. Those are what is held to a test here,
together with the guard every published report carries.
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

import review_challenge_data as rcd  # noqa: E402

HOUR = 3600


def _frame(spec) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Departures as ``(icao, day, rows, unjoined, late)`` per airport-day."""
    recs = []
    for icao, day, rows, unjoined, late in spec:
        base = pd.Timestamp(f"{day}T05:00:00Z")
        for i in range(rows):
            recs.append(
                {
                    "ADEP_mvt": icao,
                    "MVT_TIME_UTC_mvt": base + pd.Timedelta(minutes=i),
                    "SCHED_TIME_UTC_mvt": base
                    + pd.Timedelta(minutes=i)
                    - pd.Timedelta(seconds=5 * HOUR if i < late else 600),
                    "LOBT_flt": pd.NaT if i < unjoined else base,
                    "STAND_mvt": "A12",
                }
            )
    frame = pd.DataFrame(recs)
    lag = (frame["MVT_TIME_UTC_mvt"] - frame["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    return frame, frame["LOBT_flt"].isna(), lag


def _rate(unj: float, late: float, airports) -> pd.DataFrame:
    return pd.DataFrame({"unj": unj, "late": late}, index=list(airports))


def test_a_day_needs_both_the_count_and_the_multiple():
    """Twenty rows at the airport's own rate is not a disruption."""
    # 400 departures, 20 unjoined: a 5% share. At a 5% baseline that is the
    # airport's ordinary day however many rows it holds.
    frame, unj, lag = _frame([("EHAM", "2025-01-02", 400, 20, 0)])
    ordinary = rcd.disruption_days(frame, unj, lag, _rate(0.05, 0.01, ["EHAM"]))
    assert len(ordinary) == 0
    # the same day against a baseline a hundred times lower is one
    odd = rcd.disruption_days(frame, unj, lag, _rate(0.0005, 0.01, ["EHAM"]))
    assert len(odd) == 1 and int(odd.iloc[0]["unj"]) == 20


def test_nineteen_flagged_departures_is_below_the_count_whatever_the_share():
    frame, unj, lag = _frame([("EHAM", "2025-01-02", 40, 19, 0)])
    got = rcd.disruption_days(frame, unj, lag, _rate(0.0001, 0.0001, ["EHAM"]))
    assert len(got) == 0
    assert rcd.DISRUPTION_MIN_ROWS == 20


def test_either_flag_alone_is_enough():
    frame, unj, lag = _frame([("LFPG", "2025-01-02", 400, 0, 25)])
    got = rcd.disruption_days(frame, unj, lag, _rate(0.05, 0.0005, ["LFPG"]))
    assert len(got) == 1 and int(got.iloc[0]["late"]) == 25


def test_the_longest_run_counts_consecutive_days_at_one_airport():
    days = pd.DataFrame(
        {
            "ADEP_mvt": ["EHAM", "EHAM", "EHAM", "EDDM", "EDDM"],
            "day": [
                pd.Timestamp("2026-01-02").date(),
                pd.Timestamp("2026-01-03").date(),
                pd.Timestamp("2026-01-04").date(),
                pd.Timestamp("2026-01-12").date(),
                pd.Timestamp("2026-01-20").date(),
            ],
        }
    )
    assert rcd.longest_run(days) == ("EHAM", 3)
    one = days.head(1)
    assert rcd.longest_run(one) == ("EHAM", 1)


def test_the_regime_shares_are_over_the_hours_that_have_a_reading():
    hourly = pd.DataFrame(
        {
            "station": ["EDDM"] * 4,
            "hour": pd.date_range("2025-12-24T00:00:00Z", periods=4, freq="h"),
            "vis_band": [2.0, 0.0, 0.0, 0.0],
            "precip": [1.0, 1.0, 0.0, 0.0],
            "freezing": [1.0, 1.0, 1.0, 1.0],
        }
    )
    day = pd.Timestamp("2025-12-24").date()
    assert rcd.regime_shares(hourly, "EDDM", day) == ["4", "25%", "50%", "100%"]
    # a station-day with no reading, and no cache at all, each say so
    assert rcd.regime_shares(hourly, "EGLL", day)[0] == "0"
    assert rcd.regime_shares(None, "EDDM", day) == ["n/a"] * 4


def test_the_section_publishes_no_stand_it_was_handed():
    frame, _unj, _lag = _frame(
        [("EHAM", "2025-01-02", 60, 25, 0), ("EHAM", "2026-01-02", 60, 25, 0)]
    )
    train = frame[frame["MVT_TIME_UTC_mvt"].dt.year == 2025].copy()
    rank = frame[frame["MVT_TIME_UTC_mvt"].dt.year == 2026].copy()
    for f in (train, rank):
        # the section opens on the over-two-hour rows, so the frame needs one
        f["TAXITIME_SEC_mvt"] = 900.0
        f.iloc[0, f.columns.get_loc("TAXITIME_SEC_mvt")] = 3 * HOUR
    lines, _facts = rcd.section_artefacts(train, rank, None)
    text = "\n".join(lines)
    matrix = pd.DataFrame({"stand_id": ["A12"], "designator": ["16R"]})
    rcd.check_no_identifiers(text, train, rank, matrix)
    assert "A12" not in text
