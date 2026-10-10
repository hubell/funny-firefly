# SPDX-License-Identifier: GPL-3.0-only
"""``taxiout.opdi`` on synthetic frames: the files, the reduction, the tie, the match.

No network and no OPDI or challenge data; every row is written here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from taxiout import opdi
from taxiout.adsb import REFERENCE_POINTS

pa = pytest.importorskip("pyarrow")


def ts(s):
    return pd.Timestamp(s, tz="UTC")


def test_the_ten_day_grid_and_the_monthly_lists():
    match = opdi.match_months(("2026-01", "2026-07"))
    ev = opdi.event_files(match)
    assert len(ev) == 45
    assert ev[0] == "flight_events_20241226_20250105.parquet"
    assert "flight_events_20251231_20260110.parquet" in ev
    assert ev[-5] == "flight_events_20260130_20260209.parquet"
    assert ev[-1] == "flight_events_20260729_20260808.parquet"
    # a file wholly in June 2026 is not read
    assert "flight_events_20260619_20260629.parquet" not in ev
    lists = opdi.list_files(opdi.list_months(match))
    assert len(lists) == 16
    assert (
        lists[0] == "flight_list_202412.parquet"
        and lists[-1] == "flight_list_202607.parquet"
    )
    assert opdi.url_of(ev[0]).endswith("/v002/flight_events/" + ev[0])
    assert opdi.url_of(lists[0]).endswith("/v002/flight_list/" + lists[0])


def test_the_two_ranking_months_give_the_lists_they_were_named_with():
    match = opdi.match_months(("2026-01", "2026-07"))
    assert match == tuple(f"2025{m:02d}" for m in range(1, 13)) + ("202601", "202607")
    assert opdi.match_months(("202607", "202601")) == match
    assert opdi.list_months(match) == (
        ("202412",) + match[:12] + ("202601", "202606", "202607")
    )


def test_four_ranking_months_read_their_files_and_the_month_before_each_run():
    match = opdi.match_months(("2026-01", "2026-02", "2026-06", "2026-07"))
    assert match[-4:] == ("202601", "202602", "202606", "202607")
    lists = opdi.list_months(match)
    # December 2024 before the training year and January and February 2026,
    # May 2026 before June and July
    assert set(lists) - set(match) == {"202412", "202605"}
    ev = opdi.event_files(match)
    two = opdi.event_files(opdi.match_months(("2026-01", "2026-07")))
    assert set(two) <= set(ev)
    assert sorted(set(ev) - set(two)) == [
        "flight_events_20260209_20260219.parquet",
        "flight_events_20260219_20260301.parquet",
        "flight_events_20260530_20260609.parquet",
        "flight_events_20260609_20260619.parquet",
        "flight_events_20260619_20260629.parquet",
    ]
    # a file wholly in March or in May 2026 is not read
    assert "flight_events_20260301_20260311.parquet" not in ev
    assert "flight_events_20260520_20260530.parquet" not in ev


def test_ids_are_the_same_64_bits_whether_stored_unsigned_or_signed():
    big = np.array([2**63 + 5, 7], dtype=np.uint64)
    got = opdi.id64(pa.array(big)).to_numpy()
    assert got.dtype == np.int64
    assert got[0] == np.array([2**63 + 5], dtype=np.uint64).view(np.int64)[0]
    assert got[1] == 7


def test_times_without_a_zone_and_strings_are_utc():
    naive = pa.array([pd.Timestamp("2025-01-01 10:00")], pa.timestamp("us"))
    s = pa.array(["2025-01-01T10:00:00"])
    for col in (naive, s):
        assert (
            opdi.utc_times(col).to_pylist()[0] == ts("2025-01-01 10:00").to_pydatetime()
        )


def _events(rows):
    keys = [
        "flight_id",
        "type",
        "event_time",
        "latitude",
        "longitude",
        "altitude",
        "info",
    ]
    return pa.table(
        {
            "flight_id": pa.array([r[0] for r in rows], pa.uint64()),
            "type": pa.array([r[1] for r in rows]),
            "event_time": pa.array([r[2] for r in rows], pa.timestamp("us")),
            "latitude": pa.array([r[3] for r in rows], pa.float64()),
            "longitude": pa.array([r[4] for r in rows], pa.float64()),
            "altitude": pa.array([r[5] for r in rows], pa.float64()),
            "info": pa.array(
                [json.dumps(r[6], separators=(",", ":")) if r[6] else None for r in rows]
            ),
        }
    ).select(keys)


def test_the_reduction_keeps_what_section_2_reads():
    lat, lon = REFERENCE_POINTS["LSZH"]
    t = pd.Timestamp("2025-01-01 10:00")
    surface = {
        "osm_airport": "LSZH",
        "osm_ref": "A1",
        "opdi_time_in_use_s": 600,
        "osn_icao24": "abc123",
        "osn_flight_id": "SWR 12",
    }
    rows = [
        (1, "first-xing-fl50", t, 47.0, 8.0, 5000.0, None),
        (1, "first_seen", t, lat, lon, None, None),  # within 4 km
        (2, "first_seen", t, lat + 0.1, lon, None, None),  # 11 km away
        (3, "first_seen", t, lat, lon, 3000.0, None),  # airborne
        (1, "exit-parking_position", t, lat, lon, None, surface),
        (1, "entry-taxiway", t, lat, lon, None, {**surface, "osm_airport": "LOWW"}),
        (1, "landing", t, lat, lon, None, surface),
    ]
    r = opdi.reduce_group(_events(rows)).to_pandas()
    assert sorted(r["type"]) == ["exit-parking_position", "first-xing-fl50", "first_seen"]
    fs = r[r["type"] == "first_seen"].iloc[0]
    assert fs["airport"] == "LSZH" and fs["flight_id"] == 1
    park = r[r["type"] == "exit-parking_position"].iloc[0]
    assert park["osm_ref"] == "A1" and park["time_in_use_s"] == 600.0
    assert park["osn_callsign"] == "SWR12" and park["osn_icao24"] == "abc123"


def _event_flights(rows):
    return pd.DataFrame(
        rows, columns=["flight_id", "osn_icao24", "osn_callsign", "anchor"]
    )


def _list(rows):
    return pd.DataFrame(rows, columns=["id", "icao24", "callsign", "first_seen"])


def test_the_tie_takes_the_nearest_within_three_hours_and_never_a_tie():
    e = _event_flights(
        [
            (10, "aaa", "X1", ts("2025-12-01 10:00")),
            (11, "bbb", "Y1", ts("2025-12-01 10:00")),
            (12, "ccc", "Z1", ts("2025-12-01 10:00")),
        ]
    )
    rows = _list(
        [
            (100, "aaa", "X1", ts("2025-12-01 09:00")),
            (101, "aaa", "X1", ts("2025-12-01 10:30")),  # the nearest
            (102, "bbb", "Y1", ts("2025-12-01 09:30")),
            (103, "bbb", "Y1", ts("2025-12-01 10:30")),  # equally near: no tie
            (104, "ccc", "Z1", ts("2025-12-01 13:01")),  # beyond 3 hours
        ]
    )
    got = opdi.tie(e, rows)
    assert got.loc[10, "tied_id"] == 101
    assert pd.isna(got.loc[11, "tied_id"])
    assert 12 not in got.index


def test_an_event_flight_keys_by_its_commonest_pair_and_anchors_on_fl50():
    t = ts("2025-12-01 10:00")
    red = pd.DataFrame(
        {
            "flight_id": [1, 1, 1, 1],
            "type": ["entry-taxiway", "exit-apron", "entry-runway", "first-xing-fl50"],
            "event_time": [
                t,
                t + pd.Timedelta(minutes=1),
                t + pd.Timedelta(minutes=5),
                t + pd.Timedelta(minutes=9),
            ],
            "osn_icao24": ["aaa", "bbb", "bbb", None],
            "osn_callsign": ["X1", "Y1", "Y1", None],
        }
    )
    e = opdi.event_flights(red).set_index("flight_id")
    assert (e.loc[1, "osn_icao24"], e.loc[1, "osn_callsign"]) == ("bbb", "Y1")
    assert e.loc[1, "anchor"] == t + pd.Timedelta(minutes=9)


def test_the_months_around():
    assert opdi.months_around("202501") == ["202412", "202501", "202502"]
    assert opdi.months_around("202607") == ["202606", "202607", "202608"]


def test_a_departure_matches_by_callsign_and_time_and_is_covered(monkeypatch):
    take = ts("2025-03-10 12:00")
    dep = pd.DataFrame(
        {
            "MVT_ID_mvt": [1.0, 2.0],
            "ADEP_mvt": pd.array(["LSZH", "LSZH"], dtype="string"),
            "ADES_mvt": pd.array(["EDDF", "EDDF"], dtype="string"),
            "MVT_TIME_UTC_mvt": [take, take],
            "BLOCK_TIME_UTC_mvt": [take - pd.Timedelta(minutes=10)] * 2,
            "SCHED_TIME_UTC_mvt": [take] * 2,
            "STAND_mvt": pd.array(["E1", "E2"], dtype="string"),
            "CALLSIGN_flt": pd.array(["SWR 12", "SWR99"], dtype="string"),
            "FLIGHT_mvt": pd.array(["LX12", "LX99"], dtype="string"),
        }
    )
    arr = dep.iloc[:0].copy()
    monkeypatch.setattr(
        opdi, "movements", lambda root, ym, phase: dep if phase == "DEP" else arr
    )
    flights = pd.DataFrame(
        {
            "id": [7, 8],
            "icao24": ["aaa", "bbb"],
            "callsign": ["SWR12", "SWR99"],
            "adep": ["LSZH", "LSZH"],
            "ades": ["EDDF", "EDDF"],
            "first_seen": [
                take - pd.Timedelta(minutes=12),
                take + pd.Timedelta(seconds=601),
            ],
            "last_seen": [take + pd.Timedelta(hours=1)] * 2,
            "is_dep": [True, True],
            "is_arr": [False, False],
            "opdi_tied": [False, False],
        }
    )
    for c in opdi.EVENT_COLS:
        timed = c.endswith("time") or c == "fl50"
        flights[c] = (
            pd.Series([pd.NaT] * 2, dtype="datetime64[us, UTC]") if timed else np.nan
        )
    flights["O1_time"] = pd.Series(
        [take - pd.Timedelta(minutes=8), pd.NaT], dtype="datetime64[us, UTC]"
    )
    flights["fl50_missing"] = False
    out, info = opdi.match_month("202503", None, flights)
    out = out.set_index("MVT_ID_mvt")
    assert out.loc[1.0, "opdi_id"] == 7 and str(out["opdi_id"].dtype) == "Int64"
    assert bool(out.loc[1.0, "covered"])
    # first_seen more than 600 s after take-off: no match
    assert pd.isna(out.loc[2.0, "opdi_id"]) and not bool(out.loc[2.0, "matched"])
    assert info["matched"] == 1 and info["covered"] == 1
