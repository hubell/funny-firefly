# SPDX-License-Identifier: GPL-3.0-only
"""Geodesy checks against independently known quantities."""

from __future__ import annotations

import math

import pytest

from taxiout.geo import (
    _radii,
    simplify,
    LocalFrame,
    angle_diff_deg,
    bearing_deg,
    farthest_pair,
    geodesic_m,
    project_on_segment,
    resample,
    sharp_turns,
    turning_deg,
)


def test_one_degree_of_latitude():
    """A degree of latitude near 50N is about 111.2 km."""
    d = geodesic_m((50.0, 0.0), (51.0, 0.0))
    assert 111_100 < d < 111_400


def test_degenerate_distance_is_zero():
    assert geodesic_m((51.47, -0.45), (51.47, -0.45)) == 0.0


def test_known_runway_length():
    """EGLL 09R/27L endpoints from OSM measure ~3.3 km; published 3660 m paved.

    The OSM way is shorter than the published runway because it is traced to
    the marked ends rather than the full paved surface. This pins the geodesy,
    not the runway: the value must stay stable, and must be in the right
    kilometre.
    """
    d = geodesic_m((51.4649463, -0.4340796), (51.4647867, -0.4823454))
    assert 3300 < d < 3400


def test_bearing_cardinals():
    assert bearing_deg((51.0, 0.0), (52.0, 0.0)) == pytest.approx(0.0, abs=0.1)
    assert bearing_deg((51.0, 0.0), (51.0, 1.0)) == pytest.approx(90.0, abs=0.5)
    assert bearing_deg((51.0, 0.0), (50.0, 0.0)) == pytest.approx(180.0, abs=0.1)


def test_angle_diff_wraps():
    assert angle_diff_deg(350.0, 10.0) == pytest.approx(20.0)
    assert angle_diff_deg(10.0, 350.0) == pytest.approx(20.0)
    assert angle_diff_deg(0.0, 180.0) == pytest.approx(180.0)


def test_local_frame_round_trips():
    frame = LocalFrame(50.0333, 8.5706)
    p = (50.0400, 8.5800)
    x, y = frame.to_xy(p)
    back = frame.to_latlon(x, y)
    assert geodesic_m(p, back) < 0.01


def test_local_frame_agrees_with_geodesic():
    """Planar distance in the frame matches the geodesic at airport scale."""
    frame = LocalFrame(50.0333, 8.5706)
    a, b = (50.0333, 8.5706), (50.0500, 8.6000)
    ax, ay = frame.to_xy(a)
    bx, by = frame.to_xy(b)
    planar = math.hypot(bx - ax, by - ay)
    assert planar == pytest.approx(geodesic_m(a, b), rel=1e-3)


def test_projection_onto_segment():
    frame = LocalFrame(50.0, 8.0)
    a, b = (50.0, 8.0), (50.0, 8.001)
    foot, dist, t = project_on_segment(frame, (50.0005, 8.0005), a, b)
    assert 0.0 < t < 1.0
    assert dist == pytest.approx(geodesic_m((50.0005, 8.0), (50.0, 8.0)), rel=1e-2)


def test_projection_clamps_beyond_the_ends():
    frame = LocalFrame(50.0, 8.0)
    a, b = (50.0, 8.0), (50.0, 8.001)
    _, _, t = project_on_segment(frame, (50.0, 7.999), a, b)
    assert t == 0.0
    _, _, t = project_on_segment(frame, (50.0, 8.002), a, b)
    assert t == 1.0


def test_projection_on_degenerate_segment():
    frame = LocalFrame(50.0, 8.0)
    foot, dist, t = project_on_segment(frame, (50.001, 8.0), (50.0, 8.0), (50.0, 8.0))
    assert foot == (50.0, 8.0)
    assert t == 0.0
    assert dist > 0


def test_farthest_pair_finds_the_ends():
    pts = [(50.0, 8.0), (50.0, 8.005), (50.0, 8.010), (50.0001, 8.003)]
    a, b = farthest_pair(pts)
    assert {a, b} == {(50.0, 8.0), (50.0, 8.010)}


def test_farthest_pair_needs_two_points():
    with pytest.raises(ValueError):
        farthest_pair([(50.0, 8.0)])


def test_resample_spacing_and_endpoints():
    line = [(50.0, 8.0), (50.0, 8.02)]
    pts = resample(line, 100.0)
    assert pts[0] == line[0] and pts[-1] == line[-1]
    gaps = [geodesic_m(p, q) for p, q in zip(pts, pts[1:])]
    assert all(g <= 105.0 for g in gaps)


def test_turning_is_zero_on_a_straight_line():
    line = [(50.0, 8.0), (50.0, 8.01), (50.0, 8.02)]
    assert turning_deg(line) == pytest.approx(0.0, abs=0.5)


def test_turning_measures_a_right_angle():
    """East then north is a 90 degree turn, however finely it is drawn."""
    coarse = [(50.0, 8.0), (50.0, 8.006), (50.006, 8.006)]
    assert turning_deg(coarse) == pytest.approx(90.0, abs=2.0)


def test_turning_ignores_tracing_noise():
    """A jittered straight line must not out-turn a real corner.

    This is what the simplification exists for: without it, vertex density
    controls the statistic and airports become incomparable.
    """
    jittered = [
        (50.0 + (0.000004 if i % 2 else -0.000004), 8.0 + i * 0.0001) for i in range(60)
    ]
    corner = [(50.0, 8.0), (50.0, 8.006), (50.006, 8.006)]
    assert turning_deg(jittered) < turning_deg(corner)


def test_sharp_turns_counts_corners_not_arcs():
    corner = [(50.0, 8.0), (50.0, 8.006), (50.006, 8.006)]
    assert sharp_turns(corner) == 1
    straight = [(50.0, 8.0), (50.0, 8.01), (50.0, 8.02)]
    assert sharp_turns(straight) == 0


def test_distance_does_not_depend_on_direction():
    """The bug this replaced was a bias aligned with runway orientation.

    Using the Gaussian mean radius for every azimuth makes north-south lines
    read ~0.13% long and east-west lines ~0.13% short. Two legs of equal true
    length at right angles must measure equal; with the mean radius they differ
    by ~9 m in 3.5 km, which is exactly the scale of a taxi route.
    """
    lat0, lon0 = 51.47, -0.45
    # The radius the function implies must be the meridional one going north
    # and the prime-vertical one going east -- not the same number for both.
    m, n = _radii(lat0)
    assert n > m
    ns = geodesic_m((lat0, lon0), (lat0 + 0.001, lon0))
    ew = geodesic_m((lat0, lon0), (lat0, lon0 + 0.001))
    implied_ns = ns / math.radians(0.001)
    implied_ew = ew / (math.radians(0.001) * math.cos(math.radians(lat0)))
    assert implied_ns == pytest.approx(m, rel=1e-4)
    assert implied_ew == pytest.approx(n, rel=1e-4)


def test_turning_is_symmetric_under_reversal():
    """Total turning cannot depend on which way you walk the path.

    Anchoring the simplification frame at the first vertex broke this by up to
    20 degrees on real routes, because the frame shift flipped keep/drop
    decisions for vertices sitting at the tolerance.
    """
    import random

    rng = random.Random(11)
    path = [(50.0, 8.0)]
    for i in range(1, 120):
        path.append(
            (
                50.0 + i * 2e-5 + rng.uniform(-3e-5, 3e-5),
                8.0 + i * 3e-5 + rng.uniform(-3e-5, 3e-5),
            )
        )
    # Exact to floating point: the residual is ~4e-5 deg on a 2137 deg total,
    # against the 20 deg the first-vertex anchor produced.
    assert turning_deg(path) == pytest.approx(turning_deg(path[::-1]), abs=1e-3)
    assert sharp_turns(path) == sharp_turns(path[::-1])


def test_turning_does_not_depend_on_vertex_density():
    """The property the simplification exists for.

    Splitting every segment in half adds vertices and no geometry; the
    statistic must not move.
    """
    coarse = [(50.0, 8.0), (50.0, 8.006), (50.006, 8.006), (50.006, 8.0)]
    dense: list[tuple[float, float]] = [coarse[0]]
    for a, b in zip(coarse, coarse[1:]):
        for k in (1, 2, 3, 4):
            dense.append((a[0] + (b[0] - a[0]) * k / 4, a[1] + (b[1] - a[1]) * k / 4))
    assert turning_deg(dense) == pytest.approx(turning_deg(coarse), abs=0.5)


def test_simplify_keeps_real_geometry_and_drops_jitter():
    corner = [(50.0, 8.0), (50.0, 8.006), (50.006, 8.006)]
    assert len(simplify(corner)) == 3
    jittered = [(50.0 + (2e-6 if i % 2 else -2e-6), 8.0 + i * 1e-4) for i in range(40)]
    assert len(simplify(jittered)) == 2
