# SPDX-License-Identifier: GPL-3.0-only
"""Local geodesy for airport-scale geometry.

Everything here operates over distances below ~15 km, where a tangent-plane
approximation about a local origin is accurate to well under a metre. Two
primitives are provided:

``geodesic_m``
    Great-circle distance on a sphere fitted to the ellipsoid in the direction
    the line runs. Getting the direction dependence right matters more than it
    sounds: the naive alternative biases east-west and north-south distances in
    opposite directions by ~0.13% each, which at these airports means a bias
    aligned with runway orientation.

``LocalFrame``
    An equirectangular projection tangent at a reference point, used for the
    planar maths that snapping and simplification need (point-to-segment
    projection, perpendicular offsets). Planar geometry on a sphere is only
    defensible because the airport movement area is small; the frame is always
    constructed from the aerodrome reference point so the error is bounded by
    the size of the airport, not by position on the globe.

No external geodesy dependency: the maths is short, the accuracy requirement is
modest and explicit, and one fewer dependency is one fewer reproducibility risk.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# WGS84 defining parameters
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)

Point = tuple[float, float]  # (lat, lon) in degrees


def _radii(lat_deg: float) -> tuple[float, float]:
    """Meridional (M) and prime-vertical (N) radii of curvature at a latitude."""
    sin_lat = math.sin(math.radians(lat_deg))
    w = 1.0 - WGS84_E2 * sin_lat * sin_lat
    n = WGS84_A / math.sqrt(w)
    m = WGS84_A * (1.0 - WGS84_E2) / (w * math.sqrt(w))
    return m, n


def geodesic_m(a: Point, b: Point) -> float:
    """Distance in metres between two ``(lat, lon)`` points.

    The sphere is fitted to the ellipsoid in the direction the line actually
    runs, via Euler's theorem: ``R = MN / (N cos^2 a + M sin^2 a)`` for azimuth
    ``a``, which reduces to the meridional radius north-south and the
    prime-vertical radius east-west.

    Using the Gaussian mean ``sqrt(MN)`` for every azimuth instead -- the
    obvious simplification, and the first thing written here -- costs about
    1300 ppm, *signed by direction*: east-west lines come out ~0.13% short and
    north-south lines ~0.13% long. That is not noise. It is a systematic bias
    aligned with runway orientation, so it would have shortened every taxi
    distance at the east-west airports (EGLL, EDDF, EDDM, LFPG) and lengthened
    them at the north-south ones (LTAI, LTFM, half of EHAM and LEMD), by around
    9 m on a 3.5 km route. The correction costs one extra line and removes it.
    """
    lat1, lon1 = a
    lat2, lon2 = b
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlam = math.radians(lon2 - lon1)
    h = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2.0) ** 2
    )
    lat_mid = (lat1 + lat2) / 2.0
    m, n = _radii(lat_mid)
    # Local displacement components, used only to recover the azimuth.
    dy = dphi * m
    dx = dlam * n * math.cos(math.radians(lat_mid))
    denom = n * dy * dy + m * dx * dx
    r = math.sqrt(m * n) if denom <= 0.0 else m * n * (dx * dx + dy * dy) / denom
    return 2.0 * r * math.asin(min(1.0, math.sqrt(h)))


def bearing_deg(a: Point, b: Point) -> float:
    """Initial true bearing from ``a`` to ``b``, degrees clockwise from north."""
    phi1, phi2 = math.radians(a[0]), math.radians(b[0])
    dlam = math.radians(b[1] - a[1])
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return math.degrees(math.atan2(y, x)) % 360.0


def angle_diff_deg(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in ``[0, 180]``."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def path_length_m(points: list[Point]) -> float:
    return sum(geodesic_m(p, q) for p, q in zip(points, points[1:]))


def resample(points: list[Point], step_m: float) -> list[Point]:
    """Resample a polyline to roughly uniform spacing, keeping both ends.

    Vertex density in OSM reflects how carefully somebody traced a curve, not
    anything about the pavement. Any per-vertex statistic is therefore a
    measure of the mapper, and comparing one across airports compares mapping
    styles: Zurich carries a vertex every 9 m of pavement, Istanbul one every
    39 m. Resampling to a fixed step removes that dependence.
    """
    if step_m <= 0 or len(points) < 2:
        return list(points)

    out = [points[0]]
    carry = 0.0
    for a, b in zip(points, points[1:]):
        seg = geodesic_m(a, b)
        if seg <= 0.0:
            continue
        pos = step_m - carry
        while pos <= seg:
            t = pos / seg
            out.append((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t))
            pos += step_m
        carry = (carry + seg) % step_m
    if out[-1] != points[-1]:
        out.append(points[-1])
    return out


# Tolerance for the simplification that precedes any turn statistic: below it a
# vertex is tracing noise, above it a deliberate bend.
#
# Measured rather than guessed. Sweeping the tolerance and watching turn angle
# per kilometre, every airport falls steeply while noise is still being removed
# and then flattens once only real geometry is left. Six of the seven airports
# sampled are flat by 2.5-3.0 m (EDDF 115 -> 114 -> 112 deg/km at 2.5, 3, 4;
# LEBL 183 -> 181 -> 181; EHAM, LTFM and LTAI flat from 2.0). Going lower is
# what produced the anomaly that prompted the check: at 1.5 m, Zurich and
# Barcelona read 289 and 337 deg/km against Schiphol's 183, which is a claim
# about how carefully they were traced, not about their taxiways.
#
# Zurich is the honest exception -- it keeps declining past 5 m, so its turn
# statistic remains partly scale-dependent and should be read with that in mind.
SIMPLIFY_M = 3.0


def simplify(points: list[Point], tol_m: float = SIMPLIFY_M) -> list[Point]:
    """Douglas-Peucker simplification, tolerance in metres.

    Iterative rather than recursive: a routed path can carry several hundred
    vertices and the recursive form is one pathological input away from a stack
    overflow for no benefit.

    The polyline is projected to the tangent plane once and the whole search
    runs on plain floats. Re-projecting inside the inner loop -- the obvious
    way to write it -- made a full build take ten minutes instead of thirty
    seconds, because it converts three points and computes two curvature radii
    per candidate vertex, tens of millions of times.
    """
    n = len(points)
    if n < 3:
        return list(points)
    # Anchor the projection at the midpoint of the two ends, which is the same
    # point whichever way the polyline is read. Anchoring at points[0] -- the
    # obvious choice -- shifts the frame by ~4e-4 in the curvature terms under
    # reversal, which flips keep/drop decisions for vertices sitting exactly at
    # the tolerance and made the "total turning does not depend on direction of
    # travel" invariant false by up to 20 degrees on real routes.
    frame = LocalFrame(
        (points[0][0] + points[-1][0]) / 2.0, (points[0][1] + points[-1][1]) / 2.0
    )
    xy = [frame.to_xy(p) for p in points]

    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi <= lo + 1:
            continue
        ax, ay = xy[lo]
        bx, by = xy[hi]
        dx, dy = bx - ax, by - ay
        span = dx * dx + dy * dy
        worst, worst_d = lo, -1.0
        for i in range(lo + 1, hi):
            px, py = xy[i]
            ux, uy = px - ax, py - ay
            if span <= 0.0:
                d = math.hypot(ux, uy)
            else:
                t = (ux * dx + uy * dy) / span
                t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
                d = math.hypot(ux - t * dx, uy - t * dy)
            if d > worst_d:
                worst, worst_d = i, d
        if worst_d > tol_m:
            keep[worst] = True
            stack.append((lo, worst))
            stack.append((worst, hi))
    return [p for p, k in zip(points, keep) if k]


def turn_profile(
    points: list[Point], tol_m: float = SIMPLIFY_M, threshold_deg: float = 30.0
) -> tuple[float, int]:
    """``(total_turn_deg, sharp_turn_count)`` from a single simplification.

    Both statistics want the same simplified polyline, and simplification is
    the expensive part, so they are produced together.
    """
    headings = _headings(simplify(points, tol_m))
    total = 0.0
    sharp = 0
    for h1, h2 in zip(headings, headings[1:]):
        delta = angle_diff_deg(h1, h2)
        total += delta
        if delta >= threshold_deg:
            sharp += 1
    return total, sharp


def _headings(points: list[Point]) -> list[float]:
    return [bearing_deg(p, q) for p, q in zip(points, points[1:]) if p != q]


def turning_deg(points: list[Point], tol_m: float = SIMPLIFY_M) -> float:
    """Total absolute heading change along a polyline, in degrees.

    Ravizza et al. (2013) find total turn angle almost as predictive of taxi
    speed as distance, so it is computed alongside every route rather than
    bolted on later.

    The polyline is simplified first, and the choice of *how* took two
    attempts. Summing raw vertex-to-vertex heading changes measures the mapper,
    not the pavement: Zurich carries a vertex every 9 m and Istanbul one every
    39 m, so tracing jitter alone made the two incomparable. Fixed-step
    resampling removed that but introduced its own artefact -- it discards the
    original vertices, so any feature shorter than the step is cut by a chord,
    and an S-bend narrower than the step cancels itself. Measured on real
    routes, that left the statistic dependent on which end you started from, by
    up to 150 degrees.

    Simplification has neither problem. Douglas-Peucker keeps every vertex that
    departs from its own chord by more than the tolerance and drops the rest,
    so sub-metre jitter disappears, real geometry survives at any density, and
    the result is symmetric under reversal -- which it must be, since the total
    turning of a path does not depend on the direction of travel.
    """
    return turn_profile(points, tol_m)[0]


def sharp_turns(
    points: list[Point], tol_m: float = SIMPLIFY_M, threshold_deg: float = 30.0
) -> int:
    """Count of heading changes beyond ``threshold_deg`` along a simplified path.

    Total turn angle cannot tell a long gentle arc from a sequence of corners,
    and the two cost an aircraft very different amounts of time. This counts
    the corners.
    """
    return turn_profile(points, tol_m, threshold_deg)[1]


@dataclass(frozen=True)
class LocalFrame:
    """Tangent-plane frame for planar geometry near a reference point."""

    lat0: float
    lon0: float

    def to_xy(self, p: Point) -> tuple[float, float]:
        m, n = _radii(self.lat0)
        x = math.radians(p[1] - self.lon0) * n * math.cos(math.radians(self.lat0))
        y = math.radians(p[0] - self.lat0) * m
        return x, y

    def to_latlon(self, x: float, y: float) -> Point:
        m, n = _radii(self.lat0)
        lon = self.lon0 + math.degrees(x / (n * math.cos(math.radians(self.lat0))))
        lat = self.lat0 + math.degrees(y / m)
        return lat, lon


def project_on_segment(
    frame: LocalFrame, p: Point, a: Point, b: Point
) -> tuple[Point, float, float]:
    """Project ``p`` onto segment ``a``-``b``.

    Returns ``(foot, distance_m, t)`` where ``t`` in ``[0, 1]`` is the position
    of the foot along the segment. Degenerate segments return the ``a`` end.
    """
    px, py = frame.to_xy(p)
    ax, ay = frame.to_xy(a)
    bx, by = frame.to_xy(b)
    dx, dy = bx - ax, by - ay
    denom = dx * dx + dy * dy
    if denom <= 0.0:
        return a, geodesic_m(p, a), 0.0
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denom))
    foot = frame.to_latlon(ax + t * dx, ay + t * dy)
    return foot, geodesic_m(p, foot), t


def centroid(points: list[Point]) -> Point:
    """Arithmetic mean of a small set of nearby points."""
    if not points:
        raise ValueError("centroid of empty point list")
    return (
        sum(p[0] for p in points) / len(points),
        sum(p[1] for p in points) / len(points),
    )


def farthest_pair(points: list[Point]) -> tuple[Point, Point]:
    """The two most distant points in a set (brute force; sets here are small).

    Used to recover the two ends of a runway from its constituent OSM ways,
    which is robust to the ways being split, unordered, or partially
    overlapping in a way that endpoint chaining is not.
    """
    if len(points) < 2:
        raise ValueError("need at least two points")
    best = (points[0], points[1])
    best_d = -1.0
    for i, p in enumerate(points):
        for q in points[i + 1 :]:
            d = geodesic_m(p, q)
            if d > best_d:
                best_d, best = d, (p, q)
    return best
