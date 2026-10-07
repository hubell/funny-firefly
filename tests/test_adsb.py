# SPDX-License-Identifier: GPL-3.0-only
"""``taxiout.adsb`` on synthetic traces: where a leg starts, its events, the table.

No network and no adsb.lol data: each trace is written here, in the globe
history's layout (seconds after the file's timestamp, latitude, longitude,
altitude or "ground", ground speed, and a details object at index 8).
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import pytest

from taxiout import adsb

REPO = Path(__file__).resolve().parents[1]
LAT, LON = adsb.REFERENCE_POINTS["EDDF"]
FAR = (LAT + 1.0, LON)  # well outside the 8 km circle


def point(t, alt, gs, where=(LAT, LON), flight="DLH1"):
    return [t, where[0], where[1], alt, gs, None, None, None, {"flight": flight}]


def take_off(t):
    """A ground position at ``t`` followed by a climb of more than 300 ft."""
    return [
        point(t, "ground", 140.0),
        point(t + 10, 500, 160.0),
        point(t + 40, 1200, 170.0),
    ]


def legs(trace):
    t0 = 1_700_000_000
    t = [t0 + p[0] for p in trace]
    inc = [adsb.haversine_km(p[1], p[2], LAT, LON) < adsb.RADIUS_KM for p in trace]
    return adsb.legs_at(
        trace, t, inc, "EDDF", {"icao": "abc123", "r": "D-ABCD", "t": "A320"}
    )


def test_a_leg_starts_after_a_long_silence_and_reads_e1_e2_e3():
    trace = (
        [point(0, "ground", 10.0)]
        # 58 minutes of silence, rule (ii)
        + [
            point(3480, "ground", 0.0),
            point(3600, "ground", 0.5),
            point(3660, "ground", 0.0),
        ]
        + [point(3700, "ground", 6.0), point(3800, "ground", 15.0)]
        + take_off(3900)
    )
    (leg,) = legs(trace)
    t0 = 1_700_000_000
    assert leg["start_rule"] == "ii"
    assert leg["e1"] == t0 + 3480
    assert leg["e2"] == t0 + 3700
    # the last run below 1 kt of 120 s or more that ends before E2
    assert leg["e3"] == t0 + 3660
    assert leg["takeoff"] == t0 + 3900
    assert leg["n_ground"] == 6
    assert leg["flight"] == "DLH1"


def test_a_long_stationary_run_starts_the_leg_at_its_last_position():
    still = [point(60 * k, "ground", 0.0) for k in range(25)]  # 24 minutes below 1 kt
    trace = still + [point(1500, "ground", 8.0)] + take_off(1600)
    (leg,) = legs(trace)
    assert leg["start_rule"] == "iii"
    assert leg["e1"] == 1_700_000_000 + 1440


def test_a_landing_earlier_ends_the_previous_leg():
    trace = [point(0, 800, 140.0), point(30, "ground", 120.0), point(200, "ground", 10.0)]
    trace += take_off(600)
    (leg,) = legs(trace)
    assert leg["start_rule"] == "i"
    assert leg["e1"] == 1_700_000_000 + 30


def test_no_take_off_without_the_climb():
    trace = [point(0, "ground", 10.0), point(10, 500, 160.0), point(40, 600, 160.0)]
    assert legs(trace) == []


def test_e3_and_the_stationary_time_are_missing_without_e2():
    trace = [point(0, "ground", 0.5), point(200, "ground", 0.2)] + take_off(300)
    (leg,) = legs(trace)
    assert leg["e2"] is not None  # the take-off roll is 5 kt or more
    trace = [point(0, "ground", 0.5), point(200, "ground", None)] + [
        point(300, "ground", None),
        point(310, 500, 160.0),
        point(340, 1200, 170.0),
    ]
    (leg,) = legs(trace)
    assert leg["e2"] is None and leg["e3"] is None and leg["stationary_s"] is None


def test_a_trace_away_from_every_airport_is_not_read():
    adsb.init_worker(adsb.REFERENCE_POINTS, adsb.RADIUS_KM)
    raw = gzip.compress(
        json.dumps(
            {"icao": "abc123", "timestamp": 0, "trace": [point(0, "ground", 0.0, FAR)]}
        ).encode()
    )
    assert adsb.process_trace(raw) == ([], False)


def test_the_table_carries_the_notice_and_is_sorted(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    row = {k: None for k, _ in adsb.SCHEMA_FIELDS}
    rows = []
    for ap, t in (("EHAM", 20.0), ("EDDF", 30.0), ("EDDF", 10.0)):
        r = dict(row, day="2025-01-01", airport=ap, address="a", takeoff=t, e1=t - 300)
        r.update(n_ground=1, max_gap_s=0.0, n_positions=1)
        rows.append(r)
    p = tmp_path / "2025-01-01.parquet"
    assert adsb.write_parquet(rows, p, "2025-01-01", adsb.tag_of("2025-01-01")) == 3
    assert adsb.has_notice(p)
    t = pq.read_table(p)
    assert t.column("airport").to_pylist() == ["EDDF", "EDDF", "EHAM"]
    md = t.schema.metadata
    assert (
        md[b"source"]
        == b"adsblol/globe_history_2025 release v2025.01.01-planes-readsb-prod-0"
    )


def test_the_release_tag():
    assert adsb.tag_of("2025-12-31") == "v2025.12.31-planes-readsb-prod-0"


def test_the_reference_points_are_ourairports_where_the_table_is_cached():
    csv = REPO / "data" / "raw" / "ourairports_airports.csv"
    if not csv.exists():
        pytest.skip("the OurAirports table is not cached (make ourairports)")
    import csv as _csv

    found = {}
    with csv.open(encoding="utf-8") as fh:
        for r in _csv.DictReader(fh):
            if r["ident"] in adsb.REFERENCE_POINTS:
                found[r["ident"]] = (float(r["latitude_deg"]), float(r["longitude_deg"]))
    for icao, (lat, lon) in adsb.REFERENCE_POINTS.items():
        assert math.isclose(found[icao][0], lat, abs_tol=1e-6), icao
        assert math.isclose(found[icao][1], lon, abs_tol=1e-6), icao
