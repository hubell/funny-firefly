# SPDX-License-Identifier: GPL-3.0-only
"""Runway assembly and threshold resolution."""

from __future__ import annotations

import pytest

from taxiout.geo import geodesic_m
from taxiout.runways import (
    build_runways,
    designator_heading,
    parse_designators,
)


def _way(way_id: int, ref: str | None, coords: list[tuple[float, float]]) -> dict:
    tags = {"aeroway": "runway"}
    if ref is not None:
        tags["ref"] = ref
    return {
        "type": "way",
        "id": way_id,
        "tags": tags,
        "geometry": [{"lat": a, "lon": b} for a, b in coords],
    }


@pytest.mark.parametrize(
    ("ref", "expected"),
    [
        ("09L/27R", ["09L", "27R"]),
        ("9L/27R", ["09L", "27R"]),
        ("18", ["18"]),  # departure-only: do not invent the reciprocal
        ("07C/25C", ["07C", "25C"]),
        ("16R-34L", ["16R", "34L"]),
        ("Q07/Q25", []),  # leading letter: a helipad ref, not a runway
        ("", []),
        (None, []),
        ("37/19", ["19"]),  # 37 is not a designator; 19 is
        ("taxiway A", []),
    ],
)
def test_parse_designators(ref, expected):
    assert parse_designators(ref) == expected


def test_designator_heading():
    assert designator_heading("27L") == 270.0
    assert designator_heading("09") == 90.0
    assert designator_heading("36R") == 360.0


def test_merges_ways_sharing_a_ref():
    """Two halves of one runway are one runway, and its length spans both."""
    ways = [
        _way(1, "09/27", [(50.0, 8.000), (50.0, 8.010)]),
        _way(2, "09/27", [(50.0, 8.010), (50.0, 8.020)]),
    ]
    runways, unusable = build_runways("TEST", ways)
    assert len(runways) == 1
    assert unusable == []
    assert runways[0].way_ids == [1, 2]
    assert runways[0].length_m == pytest.approx(
        geodesic_m((50.0, 8.0), (50.0, 8.02)), rel=1e-6
    )


def test_merge_survives_unordered_and_overlapping_ways():
    """Endpoint chaining would fail here; the diameter does not."""
    ways = [
        _way(1, "09/27", [(50.0, 8.010), (50.0, 8.020)]),
        _way(2, "09/27", [(50.0, 8.000), (50.0, 8.012)]),  # overlaps way 1
    ]
    runways, _ = build_runways("TEST", ways)
    assert len(runways) == 1
    assert runways[0].length_m == pytest.approx(
        geodesic_m((50.0, 8.0), (50.0, 8.02)), rel=1e-6
    )


def test_threshold_is_the_end_you_depart_from():
    """A 09 departure starts at the west end and rolls east."""
    west, east = (50.0, 8.000), (50.0, 8.020)
    runways, _ = build_runways("TEST", [_way(1, "09/27", [west, east])])
    rw = runways[0]
    assert rw.threshold("09") == west
    assert rw.threshold("27") == east


def test_threshold_is_stable_under_geometry_reversal():
    west, east = (50.0, 8.000), (50.0, 8.020)
    a, _ = build_runways("TEST", [_way(1, "09/27", [west, east])])
    b, _ = build_runways("TEST", [_way(1, "09/27", [east, west])])
    assert a[0].threshold("09") == b[0].threshold("09")


def test_heading_error_is_small_for_a_true_aligned_runway():
    runways, _ = build_runways("TEST", [_way(1, "09/27", [(50.0, 8.000), (50.0, 8.020)])])
    assert runways[0].heading_error_deg("09") < 1.0
    assert runways[0].heading_error_deg("27") < 1.0


def test_unrefd_and_non_designator_ways_are_excluded_with_a_reason():
    """Excluded ways must be reported, not silently dropped."""
    ways = [
        _way(1, "09/27", [(50.0, 8.000), (50.0, 8.020)]),
        _way(2, None, [(50.0, 8.000), (50.0, 8.005)]),
        _way(3, "Q07/Q25", [(50.0, 8.000), (50.0, 8.001)]),
    ]
    runways, unusable = build_runways("TEST", ways)
    assert len(runways) == 1
    assert {u["way_id"] for u in unusable} == {2, 3}
    assert all(u["reason"] for u in unusable)


def test_untagged_geometry_cannot_move_a_threshold():
    """The whole point of ref-only threshold derivation.

    An untagged way running far past the runway end must not extend its axis,
    because a mis-tagged taxiway would otherwise relocate the take-off point.
    """
    ways = [
        _way(1, "09/27", [(50.0, 8.000), (50.0, 8.020)]),
        _way(2, None, [(50.0, 8.020), (50.0, 8.090)]),
    ]
    runways, _ = build_runways("TEST", ways)
    assert runways[0].threshold("27") == (50.0, 8.020)


def test_nodes_tagged_as_runway_are_ignored():
    node = {
        "type": "node",
        "id": 9,
        "lat": 50.0,
        "lon": 8.0,
        "tags": {"aeroway": "runway", "ref": "09/27"},
    }
    runways, unusable = build_runways("TEST", [node])
    assert runways == [] and unusable == []


def _displaced(way_id: int, coords: list[tuple[float, float]]) -> dict:
    return {
        "type": "way",
        "id": way_id,
        "tags": {"aeroway": "runway", "runway": "displaced_threshold"},
        "geometry": [{"lat": a, "lon": b} for a, b in coords],
    }


def test_displaced_threshold_is_reattached_to_its_runway():
    """A departure uses the full take-off run, so this pavement counts.

    Excluding it made the same column mean different things at different
    airports, depending only on whether a mapper had split it out.
    """
    ways = [
        _way(1, "09/27", [(50.0, 8.010), (50.0, 8.030)]),
        _displaced(2, [(50.0, 8.000), (50.0, 8.010)]),
    ]
    runways, unusable = build_runways("TEST", ways)
    assert len(runways) == 1
    assert unusable == []
    rw = runways[0]
    assert rw.displaced_m > 0
    assert rw.threshold("09") == (50.0, 8.000)  # the physical end, not the marked one
    assert rw.length_m == pytest.approx(
        geodesic_m((50.0, 8.000), (50.0, 8.030)), rel=1e-6
    )


def test_displaced_threshold_on_a_different_axis_is_not_absorbed():
    """Alignment and adjacency are both required, or a crossing runway is eaten."""
    ways = [
        _way(1, "09/27", [(50.0, 8.010), (50.0, 8.030)]),
        _displaced(2, [(49.990, 8.010), (50.010, 8.010)]),  # perpendicular
    ]
    runways, unusable = build_runways("TEST", ways)
    assert runways[0].displaced_m == 0.0
    assert [u["way_id"] for u in unusable] == [2]


def test_displaced_threshold_far_from_any_runway_is_not_absorbed():
    ways = [
        _way(1, "09/27", [(50.0, 8.010), (50.0, 8.030)]),
        _displaced(2, [(50.0, 8.100), (50.0, 8.110)]),  # aligned but remote
    ]
    runways, unusable = build_runways("TEST", ways)
    assert runways[0].displaced_m == 0.0
    assert [u["way_id"] for u in unusable] == [2]
