# SPDX-License-Identifier: GPL-3.0-only
"""A worked airport: a fictional field on which every number is known.

The geometry pipeline is validated on the real airports by internal
consistency -- routes no shorter than the straight line, pavement conserved
across repairs, fixed points reached -- and by cross-checks against
OurAirports. What it cannot do on a real airport is compare a routed distance
against the *right answer*, because nobody has it. This module supplies a
field where somebody does: ``ZZZZ``, drawn in a local metric frame so that
every distance is a sum of round numbers, every turn is a right angle, and
every repair the graph builder performs fires exactly once, at a known
place, for a known reason.

The elements are OSM-shaped (the same dicts the Overpass cache holds) and go
through ``build_graph``, ``attach_stands``, ``build_runways``, ``route_all``
and ``aggregate`` unchanged. ``expected_*`` hold the hand derivation, with
the arithmetic written out; ``tests/test_synthetic_airport.py`` asserts the
pipeline reproduces it, and ``scripts/report_synthetic_airport.py`` prints
both side by side so a reader can check without running anything.

Nothing here is derived from OpenStreetMap or from the challenge data. The
field is authored, the identifiers are small integers, and the coordinates
are a tangent plane at the equator. The latitude is not decoration: at 50 N,
a parallel 350 m north of the reference point is 55 parts per million
shorter than the plane says, so a 1,750 m route came out 6 cm short of its
hand-computed length -- the convergence of meridians, which is geometry
rather than error, but which would put a discrepancy in every table that
claims to be exact. On the equator the tangent plane agrees with the
pipeline's geodesic to a tenth of a millimeter along the axes and to about
a centimeter on a two-kilometer diagonal, the latter being the ellipsoid's
second-order term, so the tests compare distances to two centimeters.

The field
---------

Meters east (x) and north (y). Runway ``09/27`` runs along y = 0 from x = 0
to 2400; runway ``18/36`` runs along x = 2000 from y = -900 to 100, crossing
it. Taxiway A parallels the runway at y = 150 with links down to it at
x = 0, 600, 1500 and 2400. Apron taxilane B at y = 350 (x = 300 to 1500)
reaches A by one link at x = 600, so a stand's shortest route depends on
where it sits: the apron has a single exit. A south apron taxilane C at
y = -200 (x = 1200 to 1800) reaches the runway by one link at x = 1500, so
its stands must cross ``09/27`` to reach A -- or run along it, which is the
runway penalty's decision to make, at a penalty that is hand-computable.

Nose-in stands stand 70 m off their taxilane. Eleven parking positions
exercise the attachment rules: four ordinary line stands on B (one drawn
stop-to-entry rather than entry-to-stop, one identified by ``name`` rather
than ``ref``), two further features sharing stand 2's identifier with 75 m
and 95 m lead-ins, a stand drawn as a bare node 120 m from B, a stand drawn
as a footprint polygon, a stand 350 m from any pavement that must *not*
attach, and two line stands on C.

The three repairs, once each: the B-to-A link ends 0.2 m from A's vertex
at x = 600 (coincident nodes, merged); the link at x = 1500 stops 0.8 m
short of A where A has no vertex (node on an edge, edge split -- it must
stop short, because a link that overshoots A by the same 0.8 m *crosses*
it and is repaired as a crossing instead, which is how the first draft of
this field found out the two repairs are distinguished by geometry, not by
intent); ``18/36`` crosses ``09/27`` with no shared vertex (crossing,
junction inserted). Threshold 36 has no taxiway near it, so every
departure on 36 backtracks 900 m along the runway: the "missing taxiway"
case the routing report flags, with a spread of zero because the offset
is the same for every stand.

The merge and the split sit on different edges of A on purpose. A merged
node takes the coordinates of the last way that referenced it, and an edge
split later at such a node is recomputed from the shifted position while
its other half keeps the length of the way it was read from; on the first
draft, with both on one edge, 0.2 m of pavement went missing and every
route through it came out 0.2 m short. ``probe_merge_shift`` reproduces
that on a three-way graph so the report can state it as a measurement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geo import LocalFrame, Point

ICAO = "ZZZZ"
REF: Point = (0.0, 0.0)  # the equator: see the module docstring
FRAME = LocalFrame(*REF)

# -- the layout, in meters ---------------------------------------------------

RWY_Y = 0.0
RWY_X0, RWY_X1 = 0.0, 2400.0
RWY_VERTICES = (0.0, 600.0, 1500.0, 2400.0)  # shared with the links below
CROSS_X = 2000.0  # runway 18/36
CROSS_Y0, CROSS_Y1 = -900.0, 100.0

A_Y = 150.0
A_VERTICES = (0.0, 600.0, 900.0, 2400.0)  # no vertex at 1500: repair 2 fires there
LINK_XS = (0.0, 600.0, 1500.0, 2400.0)  # runway <-> A
LINK_150_TOP_Y = A_Y - 0.8  # the link at x = 1500 stops 0.8 m short of A's line

B_Y = 350.0
B_X0, B_X1 = 300.0, 1500.0
B_LINK_X = 600.0
B_LINK_BOTTOM_X = 600.2  # 0.2 m from A's vertex at (600, 150): repair 1

C_Y = -200.0
C_X0, C_X1 = 1200.0, 1800.0
C_LINK_X = 1500.0

LEAD_IN = 70.0
NORTH_STOP_Y = B_Y + LEAD_IN
SOUTH_STOP_Y = C_Y - LEAD_IN

NORTH_STANDS = {"1": 400.0, "2": 700.0, "3": 1000.0, "4": 1300.0}
# Stand 2 is drawn three times. With two features the closest-to-mean rule
# in ``aggregate`` is a tie -- the mean is their midpoint -- decided by
# floating-point rounding; three features make the rule's choice unique.
DUPLICATE_OF = "2"
DUPLICATE_LEAD_INS = (75.0, 95.0)
DUPLICATE_PICK = 75.0  # nearest the mean of 70, 75 and 95
NODE_STAND = ("9", 600.0, 470.0)  # bare node, 120 m north of B's junction
POLYGON_STAND = ("10", 250.0, 400.0)  # footprint centered here; nearest node (300, 350)
FAR_STAND = ("11", 1000.0, 700.0)  # 350 m from B: must not attach
SOUTH_STANDS = {"12": 1300.0, "13": 1700.0}

DESIGNATORS = ("09", "27", "18", "36")
THRESHOLDS = {
    "09": (RWY_X0, RWY_Y),
    "27": (RWY_X1, RWY_Y),
    "18": (CROSS_X, CROSS_Y1),
    "36": (CROSS_X, CROSS_Y0),
}

# Element order is fixed so the repair counts are a property of the field:
# the merge counts a *near* coincidence once, and which node is canonical
# is decided by read order. Taxiways before the defective links.
_WAY_IDS = {
    "runway": 100,
    "cross": 101,
    "A": 200,
    "B": 201,
    "C": 202,
    "link0": 210,
    "link600": 211,
    "link1500": 212,
    "link2400": 213,
    "linkB": 220,
    "linkC": 221,
}
_STAND_ID_BASE = 300


def xy(x: float, y: float) -> Point:
    """Field coordinates to ``(lat, lon)``."""
    return FRAME.to_latlon(x, y)


def _way(way_id: int, kind: str, coords: list[tuple[float, float]], **tags) -> dict:
    return {
        "type": "way",
        "id": way_id,
        "tags": {"aeroway": kind, **tags},
        "geometry": [{"lat": lat, "lon": lon} for lat, lon in (xy(*c) for c in coords)],
    }


def _node(node_id: int, kind: str, x: float, y: float, **tags) -> dict:
    lat, lon = xy(x, y)
    return {
        "type": "node",
        "id": node_id,
        "lat": lat,
        "lon": lon,
        "tags": {"aeroway": kind, **tags},
    }


def elements() -> list[dict]:
    """The field as Overpass-shaped elements, in a fixed order."""
    els = [
        _way(
            _WAY_IDS["runway"], "runway", [(x, RWY_Y) for x in RWY_VERTICES], ref="09/27"
        ),
        _way(
            _WAY_IDS["cross"],
            "runway",
            [(CROSS_X, CROSS_Y0), (CROSS_X, CROSS_Y1)],
            ref="18/36",
        ),
        _way(_WAY_IDS["A"], "taxiway", [(x, A_Y) for x in A_VERTICES], ref="A"),
        _way(
            _WAY_IDS["B"],
            "taxilane",
            [(B_X0, B_Y), (B_LINK_X, B_Y), (B_X1, B_Y)],
            ref="B",
        ),
        _way(
            _WAY_IDS["C"],
            "taxilane",
            [(C_X0, C_Y), (C_LINK_X, C_Y), (C_X1, C_Y)],
            ref="C",
        ),
        _way(_WAY_IDS["link0"], "taxiway", [(0.0, RWY_Y), (0.0, A_Y)], ref="A1"),
        _way(_WAY_IDS["link600"], "taxiway", [(600.0, RWY_Y), (600.0, A_Y)], ref="A2"),
        # Repair 2: the top end sits 0.8 m off A's line, where A has no vertex.
        _way(
            _WAY_IDS["link1500"],
            "taxiway",
            [(1500.0, RWY_Y), (1500.0, LINK_150_TOP_Y)],
            ref="A3",
        ),
        _way(_WAY_IDS["link2400"], "taxiway", [(2400.0, RWY_Y), (2400.0, A_Y)], ref="A4"),
        # Repair 1: the bottom end is 0.2 m from A's vertex at (900, 150).
        _way(
            _WAY_IDS["linkB"],
            "taxilane",
            [(B_LINK_BOTTOM_X, A_Y), (B_LINK_X, B_Y)],
            ref="B1",
        ),
        _way(
            _WAY_IDS["linkC"], "taxiway", [(C_LINK_X, C_Y), (C_LINK_X, RWY_Y)], ref="C1"
        ),
    ]
    sid = _STAND_ID_BASE
    for ref, x in NORTH_STANDS.items():
        coords = [(x, B_Y), (x, NORTH_STOP_Y)]
        if ref == "4":
            coords = coords[::-1]  # drawn stop-to-entry; the end is decided by measuring
        tags = {"name": ref} if ref == "3" else {"ref": ref}
        els.append(_way(sid, "parking_position", coords, **tags))
        sid += 1
    x = NORTH_STANDS[DUPLICATE_OF]
    for lead in DUPLICATE_LEAD_INS:
        els.append(
            _way(sid, "parking_position", [(x, B_Y), (x, B_Y + lead)], ref=DUPLICATE_OF)
        )
        sid += 1
    ref, x, y = NODE_STAND
    els.append(_node(sid, "parking_position", x, y, ref=ref))
    sid += 1
    ref, cx, cy = POLYGON_STAND
    square = [
        (cx - 20, cy - 15),
        (cx + 20, cy - 15),
        (cx + 20, cy + 15),
        (cx - 20, cy + 15),
    ]
    els.append(_way(sid, "parking_position", square + square[:1], ref=ref, area="yes"))
    sid += 1
    ref, x, y = FAR_STAND
    els.append(_node(sid, "parking_position", x, y, ref=ref))
    sid += 1
    for ref, x in SOUTH_STANDS.items():
        els.append(_way(sid, "parking_position", [(x, C_Y), (x, SOUTH_STOP_Y)], ref=ref))
        sid += 1
    return els


# -- running the real pipeline ----------------------------------------------


@dataclass
class Built:
    tg: object
    runways: list
    routes: list
    entries: list
    route_stats: dict
    cells: list


def build(runway_penalty: float = 4.0, max_snap_m: float = 150.0) -> Built:
    """The field through the pipeline, exactly as ``scripts/build_taxi_graph.py`` runs it."""
    from .graph import attach_stands, build_graph
    from .routing import aggregate, route_all
    from .runways import build_runways

    els = elements()
    tg = build_graph(ICAO, els, REF)
    attach_stands(tg, els, max_snap_m=max_snap_m)
    runways, _unusable = build_runways(ICAO, els)
    routes, entries, rstats = route_all(tg, runways, runway_penalty=runway_penalty)
    return Built(tg, runways, routes, entries, rstats, aggregate(routes))


# -- the hand derivation -----------------------------------------------------

# Degrees of the graph nodes a route can pass through, after every repair and
# attachment. Needed for two descriptors: ``junctions`` counts interior path
# nodes of degree three or more, and ``straight_m`` sums edges whose two
# endpoints both have degree two (junction-free running, in Ravizza's sense).
#
# Along B, west to east: the west end (degree 1), the stand entries (each
# splits B, degree 2), the link junction at 600 (degree 3), the east end
# (degree 1). The node stand attaches *at* the 600 junction and the polygon
# stand at the west end, so neither adds a node.
_B_NODES: tuple[tuple[float, int], ...] = (
    (300.0, 1),
    (400.0, 2),
    (600.0, 3),
    (700.0, 2),
    (1000.0, 2),
    (1300.0, 2),
    (1500.0, 1),
)
# Along C: the west end, stand 12, the link junction, stand 13, the east end.
_C_NODES: tuple[tuple[float, int], ...] = (
    (1200.0, 1),
    (1300.0, 2),
    (1500.0, 3),
    (1700.0, 2),
    (1800.0, 1),
)


def _straight_along(
    nodes: tuple[tuple[float, int], ...], x_from: float, x_to: float
) -> float:
    """Junction-free length of a taxilane between two of its nodes."""
    lo, hi = sorted((x_from, x_to))
    run = [(x, d) for x, d in nodes if lo <= x <= hi]
    return sum(b[0] - a[0] for a, b in zip(run, run[1:]) if a[1] == 2 and b[1] == 2)


# Two segment lengths carry the 0.8 m defect at x = 1500: A is cut and
# rejoined through the link's end node, so its two halves are hypotenuses.
A_900_TO_1500 = math.hypot(1500.0 - 900.0, LINK_150_TOP_Y - A_Y)  # 600.0005
A_1500_TO_2400 = math.hypot(2400.0 - 1500.0, LINK_150_TOP_Y - A_Y)  # 900.0004
LINK_1500 = LINK_150_TOP_Y - RWY_Y  # 149.2

# The jog is below the 3 m simplification tolerance, so a route running
# straight through (1500, 149.2) records no turn there. A route that turns
# there, or at the merged node (600.2, 150), has its right angle perturbed
# by under a tenth of a degree per corner -- 0.08 deg for 0.8 m over 600 m,
# 0.06 deg for 0.2 m over 200 m -- which is why the tests compare turn
# totals to half a degree rather than exactly.
RIGHT_ANGLE = 90.0
DISTANCE_TOL_M = 0.02
TURN_TOL_DEG = 0.5


def _north(
    stand_id: str, s: float, lead_in: float, snap: float, stop: tuple[float, float]
):
    """Cells for a stand whose route starts on B at x = s and leaves B at x = 600.

    The exit junction (600, 150) has degree four: A both ways, link A2 down
    to the runway, B's link up. A's vertex at 900 has degree two, so it is
    neither a junction nor a corner.
    """
    on_b = abs(s - B_LINK_X)
    b_straight = _straight_along(_B_NODES, s, B_LINK_X)
    to_a = on_b + (B_Y - A_Y)  # along B, then down the link
    # Turns and junctions before reaching A: the corner onto the link (only
    # if the stand is not already at the junction) and the corner onto A.
    corner_b = 0 if s == B_LINK_X else 1
    out = {}
    # 09: west along A to x = 0, down link A1. Junctions: (600,350) if used,
    # (600,150); the corner at (0,150) has degree 2. A1's two ends both have
    # degree 2, so its 150 m is junction-free.
    route = to_a + (B_LINK_X - 0.0) + (A_Y - RWY_Y)
    out["09"] = dict(
        route_m=route,
        turn_deg=RIGHT_ANGLE * (corner_b + 2),
        sharp_turns=corner_b + 2,
        junctions=corner_b + 1,
        straight_m=b_straight + 150.0,
        backtrack_m=0.0,
        cross_runway_m=0.0,
        runway_crossings=0,
    )
    # 27: east along A -- 300 to the 900 vertex, then through the 1500 jog
    # -- to x = 2400, down link A4.
    east = (900.0 - B_LINK_X) + A_900_TO_1500 + A_1500_TO_2400
    route = to_a + east + (A_Y - RWY_Y)
    out["27"] = dict(
        route_m=route,
        turn_deg=RIGHT_ANGLE * (corner_b + 2),  # the jog is below the 3 m simplification
        sharp_turns=corner_b + 2,
        junctions=corner_b + 2,  # (600,150) and (1500,149.2); (2400,150) has degree 2
        straight_m=b_straight + 150.0,
        backtrack_m=0.0,
        cross_runway_m=0.0,
        runway_crossings=0,
    )
    # 18: east along A to the 1500 link, down it, 500 m east along 09/27 --
    # a foreign runway, one crossing -- then 100 m north along 18/36. Corners
    # at (600,150), (1500,149.2), (1500,0) and (2000,0), plus the apron one.
    to_cross = (900.0 - B_LINK_X) + A_900_TO_1500 + LINK_1500 + (CROSS_X - 1500.0)
    route = to_a + to_cross + (CROSS_Y1 - RWY_Y)
    out["18"] = dict(
        route_m=route,
        turn_deg=RIGHT_ANGLE * (corner_b + 4),
        sharp_turns=corner_b + 4,
        junctions=corner_b + 4,  # (600,150), (1500,149.2), (1500,0), (2000,0)
        straight_m=b_straight,
        backtrack_m=CROSS_Y1 - RWY_Y,
        cross_runway_m=CROSS_X - 1500.0,
        runway_crossings=1,
    )
    # 36: as 18 to the crossing, then 900 m south along 18/36 -- the
    # backtrack -- with the same five corners, the last one turning south.
    route = to_a + to_cross + (RWY_Y - CROSS_Y0)
    out["36"] = dict(
        route_m=route,
        turn_deg=RIGHT_ANGLE * (corner_b + 4),
        sharp_turns=corner_b + 4,
        junctions=corner_b + 4,
        straight_m=b_straight,
        backtrack_m=RWY_Y - CROSS_Y0,
        cross_runway_m=CROSS_X - 1500.0,
        runway_crossings=1,
    )
    return _finish(stand_id, out, lead_in, snap, stop)


def _south(stand_id: str, s: float):
    """Cells for a stand on C at x = s; every route leaves C at x = 1500."""
    on_c = abs(s - C_LINK_X)
    to_rwy = on_c + (RWY_Y - C_Y)  # along C, then up the link to (1500, 0)
    out = {}
    # 09: up link A3 to A, west along A to x = 0, down A1. At the default
    # penalty the taxiway route (1799.2 m from the runway) beats running
    # 1500 m west along the runway (6000 m at four times length).
    route = to_rwy + LINK_1500 + A_900_TO_1500 + (900.0 - 0.0) + (A_Y - RWY_Y)
    out["09"] = dict(
        route_m=route,
        turn_deg=RIGHT_ANGLE * 3,
        sharp_turns=3,
        junctions=4,  # (1500,-200), (1500,0), (1500,149.2), (600,150); 900 has degree 2
        straight_m=150.0,  # A1 only
        backtrack_m=0.0,
        cross_runway_m=0.0,
        runway_crossings=0,  # 09/27 is the departure runway, so passing (1500,0) is not foreign
    )
    # 27: up A3, east along A, down A4 (1199.2 m) against 900 m of runway.
    route = to_rwy + LINK_1500 + A_1500_TO_2400 + (A_Y - RWY_Y)
    out["27"] = dict(
        route_m=route,
        turn_deg=RIGHT_ANGLE * 3,
        sharp_turns=3,
        junctions=3,  # (1500,-200), (1500,0), (1500,149.2)
        straight_m=150.0,  # A4 only
        backtrack_m=0.0,
        cross_runway_m=0.0,
        runway_crossings=0,
    )
    # 18: east along 09/27 from (1500,0) -- 2,400 m of penalized cost
    # against 3,200.8 by the taxiways -- then north.
    route = to_rwy + (CROSS_X - 1500.0) + (CROSS_Y1 - RWY_Y)
    out["18"] = dict(
        route_m=route,
        turn_deg=270.0,
        sharp_turns=3,
        junctions=3,  # (1500,-200), (1500,0), (2000,0)
        straight_m=0.0,
        backtrack_m=CROSS_Y1 - RWY_Y,
        cross_runway_m=CROSS_X - 1500.0,
        runway_crossings=1,
    )
    route = to_rwy + (CROSS_X - 1500.0) + (RWY_Y - CROSS_Y0)
    out["36"] = dict(
        route_m=route,
        turn_deg=270.0,
        sharp_turns=3,
        junctions=3,
        straight_m=0.0,
        backtrack_m=RWY_Y - CROSS_Y0,
        cross_runway_m=CROSS_X - 1500.0,
        runway_crossings=1,
    )
    return _finish(stand_id, out, LEAD_IN, 0.0, (s, SOUTH_STOP_Y))


def _finish(stand_id, out, lead_in, snap, stop):
    cells = {}
    for designator, d in out.items():
        tx, ty = THRESHOLDS[designator]
        gc = math.hypot(stop[0] - tx, stop[1] - ty)
        taxi = lead_in + snap + d["route_m"] + 0.0
        cells[(stand_id, designator)] = {
            **d,
            "lead_in_m": lead_in,
            "snap_m": snap,
            "entry_offset_m": 0.0,
            "taxi_m": taxi,
            "gc_m": gc,
            "detour_ratio": taxi / gc,
        }
    return cells


def expected_cells() -> dict[tuple[str, str], dict]:
    """Every (stand, designator) cell the matrix should contain, by hand."""
    cells: dict[tuple[str, str], dict] = {}
    for ref, x in NORTH_STANDS.items():
        lead = DUPLICATE_PICK if ref == DUPLICATE_OF else LEAD_IN
        cells.update(_north(ref, x, lead, 0.0, (x, B_Y + lead)))
    ref, x, y = NODE_STAND
    cells.update(_north(ref, x, 0.0, y - B_Y, (x, y)))
    ref, cx, cy = POLYGON_STAND
    snap = math.hypot(B_X0 - cx, B_Y - cy)  # centroid to B's west end node
    cells.update(_north(ref, B_X0, 0.0, snap, (cx, cy)))
    for ref, x in SOUTH_STANDS.items():
        cells.update(_south(ref, x))
    return cells


def expected_cell_meta() -> dict[str, dict]:
    """Per stand identifier: how many features, their spread, the worst snap."""
    meta = {ref: dict(n_features=1, spread_m=0.0, snap_max_m=0.0) for ref in NORTH_STANDS}
    # Stand 2 is drawn three times, with lead-ins of 70, 75 and 95 m: the
    # spread is 25 m and the representative is the feature nearest the mean
    # of 80, the 75 m one, so the cell carries a 75 m lead-in.
    meta[DUPLICATE_OF] = dict(
        n_features=1 + len(DUPLICATE_LEAD_INS),
        spread_m=max(DUPLICATE_LEAD_INS) - LEAD_IN,
        snap_max_m=0.0,
    )
    meta[NODE_STAND[0]] = dict(n_features=1, spread_m=0.0, snap_max_m=NODE_STAND[2] - B_Y)
    ref, cx, cy = POLYGON_STAND
    meta[ref] = dict(
        n_features=1, spread_m=0.0, snap_max_m=math.hypot(B_X0 - cx, B_Y - cy)
    )
    for ref in SOUTH_STANDS:
        meta[ref] = dict(n_features=1, spread_m=0.0, snap_max_m=0.0)
    return meta


def expected_stats() -> dict:
    """What the build should report about the field."""
    pavement = (
        (RWY_X1 - RWY_X0)
        + (CROSS_Y1 - CROSS_Y0)
        + (900.0 - 0.0)
        + A_900_TO_1500
        + A_1500_TO_2400  # A, cut and rejoined through the 0.8 m jog
        + (B_X1 - B_X0)
        + (C_X1 - C_X0)
        + 3 * (A_Y - RWY_Y)  # links at 0, 600, 2400
        + LINK_1500
        + (B_Y - A_Y)  # B's link, whose 0.2 m offset merges away
        + (RWY_Y - C_Y)
    )
    n_features = (
        len(NORTH_STANDS) + len(DUPLICATE_LEAD_INS) + 1 + 1 + 1 + len(SOUTH_STANDS)
    )
    attached = n_features - 1  # the far stand
    return {
        "ways": 11,
        "nodes_merged": 1,
        "edges_split": 1,
        "crossings_split": 1,
        "components_before_split": 2,  # 18/36 touches nothing until the crossing is repaired
        "components_after_split": 1,
        "repair_rounds": 2,  # one round finds the split, the next finds nothing
        "repair_converged": True,
        "pavement_km": pavement / 1000.0,
        "pavement_deduped_km": 0.0,
        "stands_total": n_features,
        "stands_attached": attached,
        "stands_with_id": n_features,
        "stands_direct": attached - 2,  # all but the node stand and the polygon
        "stands_bridged": 2,  # the node stand (120 m) and the polygon (70.7 m)
        "stands_polygon": 1,
        "stands_point": 2,  # the node stand and the far stand
        "stand_snap_max_m": NODE_STAND[2] - B_Y,
        "designators": 4,
        "designators_without_entry": 0,
        "pairs_possible": attached * 4,
        "pairs_routed": attached * 4,
        "pairs_unreachable": 0,
        "cells": 32,  # 8 identifiers x 4 designators; stand 11 never attaches
        "backtracking_designators": {"36": {"median_m": 900.0, "spread_m": 0.0}},
    }


def penalty_crossovers() -> dict[str, float]:
    """The runway penalty below which a south stand runs along the runway.

    From (1500, 0), departing 27: the taxiway route is 1199.2 m and the
    runway route 900 m, so the runway wins once ``900 p < 1199.2``. Departing
    09: 1799.2 m against 1500 m. The default penalty of 4 is well above both;
    the sweep in the report shows ``backtrack_m`` switching at these values.
    """
    taxi_27 = LINK_1500 + A_1500_TO_2400 + (A_Y - RWY_Y)
    taxi_09 = LINK_1500 + A_900_TO_1500 + (900.0 - 0.0) + (A_Y - RWY_Y)
    # A north stand departing 27 has the same choice at the 1500 link: down
    # it and 900 m along the runway (149.2 + 900 p) against staying on A
    # (900.0004 + 150). The taxiway wins only above 1.0009 -- a penalty of
    # exactly one sends every stand on the field down the runway. Below
    # 0.9991 the router goes further still: down link A2 at x = 600 and
    # 1,800 m along the runway (150 + 1800 p) beats the 1500 link's
    # 300 + 600.0005 + 149.2 + 900 p. With runway pavement cheaper than
    # taxiway, more of it is better, which is the whole case for a penalty.
    north_27 = (A_1500_TO_2400 + (A_Y - RWY_Y) - LINK_1500) / (RWY_X1 - 1500.0)
    north_27_a2 = ((900.0 - B_LINK_X) + A_900_TO_1500 + LINK_1500 - (A_Y - RWY_Y)) / (
        (RWY_X1 - B_LINK_X) - (RWY_X1 - 1500.0)
    )
    return {
        "south 27": taxi_27 / (RWY_X1 - 1500.0),
        "south 09": taxi_09 / (1500.0 - RWY_X0),
        "north 27": north_27,
        "north 27 via A2": north_27_a2,
    }


def probe_merge_shift() -> dict[str, float]:
    """Measure what a merged node's position does to a later split.

    Three ways: a straight taxiway from x = 0 to 1000 with a vertex at 500;
    a link ending 0.2 m from that vertex, read last, so the merged node takes
    the link's coordinates; and a stub ending 0.8 m off the taxiway at
    x = 800, so the taxiway's eastern edge is split at a node adjacent to the
    shifted one. Returns the pavement the graph reports against the sum of
    the ways as drawn; the difference is the shift, booked once.
    """
    from .graph import build_graph

    frame = FRAME
    ll = frame.to_latlon
    els = [
        {
            "type": "way",
            "id": 1,
            "tags": {"aeroway": "taxiway"},
            "geometry": [
                {"lat": la, "lon": lo} for la, lo in (ll(0, 0), ll(500, 0), ll(1000, 0))
            ],
        },
        {
            "type": "way",
            "id": 2,
            "tags": {"aeroway": "taxiway"},
            "geometry": [
                {"lat": la, "lon": lo} for la, lo in (ll(800, -100), ll(800, -0.8))
            ],
        },
        {
            "type": "way",
            "id": 3,
            "tags": {"aeroway": "taxiway"},
            "geometry": [
                {"lat": la, "lon": lo} for la, lo in (ll(500.2, 0), ll(500, 200))
            ],
        },
    ]
    tg = build_graph("PROB", els, REF)
    drawn = 1000.0 + (100.0 - 0.8) + math.hypot(0.2, 200.0)
    return {
        "drawn_m": drawn,
        "graph_m": tg.stats["pavement_km"] * 1000.0,
        "missing_m": drawn - tg.stats["pavement_km"] * 1000.0,
        "nodes_merged": float(tg.stats["nodes_merged"]),
        "edges_split": float(tg.stats["edges_split"]),
    }


def probe_two_feature_tie(lead_ins: tuple[float, float] = (70.0, 75.0)) -> dict:
    """Which feature ``aggregate`` keeps when a stand is drawn exactly twice.

    The representative is the feature whose ``taxi_m`` is nearest the group
    mean, with the lower OSM id breaking ties. For two features the mean is
    their midpoint, so the two distances to it are equal in exact arithmetic
    and the tie-break should decide; in floating point they can differ in the
    last bit, and then rounding decides instead. Returns the lead-in the
    pipeline kept, the one the tie-break would have kept, and whether the two
    distances came out bit-for-bit equal.
    """
    from .graph import attach_stands, build_graph
    from .routing import aggregate, route_all
    from .runways import build_runways

    ll = FRAME.to_latlon

    def way(wid, kind, coords, **tags):
        return {
            "type": "way",
            "id": wid,
            "tags": {"aeroway": kind, **tags},
            "geometry": [{"lat": la, "lon": lo} for la, lo in (ll(*c) for c in coords)],
        }

    els = [
        way(1, "runway", [(0, 0), (1000, 0)], ref="09/27"),
        way(2, "taxiway", [(500, 0), (500, 200)]),
        way(3, "taxilane", [(300, 200), (700, 200)]),
    ]
    for i, lead in enumerate(lead_ins):
        els.append(
            way(10 + i, "parking_position", [(400, 200), (400, 200 + lead)], ref="1")
        )
    tg = build_graph("PROB", els, REF)
    attach_stands(tg, els)
    runways, _ = build_runways("PROB", els)
    routes, _, _ = route_all(tg, runways)
    cell = next(c for c in aggregate(routes) if c.designator == "09")
    group = [r for r in routes if r.designator == "09"]
    mean = sum(r.taxi_m for r in group) / len(group)
    gaps = [abs(r.taxi_m - mean) for r in group]
    return {
        "kept_lead_in_m": cell.values["lead_in_m"],
        "tie_break_lead_in_m": min(group, key=lambda r: r.osm_id).lead_in_m,
        "gaps_equal": gaps[0] == gaps[1],
        "gap_difference_m": abs(gaps[0] - gaps[1]),
    }


# Worked examples for the report: the same arithmetic as ``expected_cells``,
# spelled out for a reader.
WORKED: tuple[tuple[str, str, str], ...] = (
    (
        "1",
        "09",
        "lead-in 70; along B from x=400 east to the apron exit at 600: 200; "
        "down the link 200; west along A from 600 to 0: 600; down A1 to the "
        "threshold: 150. Route 1,150 m, taxi 1,220 m. Three right-angle "
        "turns; junctions at (600,350) and (600,150) -- the corner at (0,150) "
        "has degree two; junction-free running: A1 only, 150, because B from "
        "400 to 600 ends at the degree-three exit.",
    ),
    (
        "4",
        "27",
        "lead-in 70, drawn stop-to-entry and resolved by measuring; along B "
        "from 1300 west to the exit at 600: 700; down the link 200; east along "
        "A: 300 to the vertex at 900, then 600.0005 and 900.0004 through the "
        "repaired jog at 1500; down A4: 150. Route 2,850.0 m, taxi 2,920.0 m. "
        "Three turns (the 0.8 m jog is below the 3 m simplification); "
        "junctions (600,350), (600,150), (1500,149.2); junction-free: B from "
        "1300 to 700 between degree-two stand nodes, 600, plus A4, 150.",
    ),
    (
        "9",
        "27",
        "a bare node 120 m north of the apron exit: no lead-in, snap 120 to "
        "the junction at (600,350); down the link 200; east along A: 300, "
        "600.0005, 900.0004; down A4: 150. Route 2,150.0 m, taxi 2,270.0 m. "
        "Two turns; junctions (600,150) and (1500,149.2).",
    ),
    (
        "12",
        "36",
        "lead-in 70; along C from 1300 to the link at 1500: 200; up to the "
        "runway: 200; east along 09/27 to the crossing at 2000: 500, a foreign "
        "runway, so cross_runway 500 and one crossing; south along 18/36 to "
        "the threshold: 900 of backtrack. Route 1,800 m, taxi 1,870 m; three "
        "turns; junctions (1500,-200), (1500,0), (2000,0).",
    ),
    (
        "13",
        "27",
        "lead-in 70; west along C from 1700 to 1500: 200; up to the runway: "
        "200; on up A3 to A: 149.2; east along A: 900.0004; down A4: 150. "
        "Route 1,599.2 m. The alternative -- 900 m east along the runway -- "
        "costs 3,600 at the default penalty and wins only below 1.332.",
    ),
)


# ==========================================================================
# Worked traffic: movements on the field, from a stated process
# ==========================================================================
#
# The challenge data cannot leave the machine it is on, so nothing that reads
# it can run in CI, and nothing that reads it can be checked against a known
# answer. This generates a year of movements on ``ZZZZ`` in the challenge's
# own schema and file layout, from a process every parameter of which is
# written here: an off-block delay, an unimpeded taxi at a fixed speed over
# the field's own distances, a first-in-first-out runway queue, noise, and a
# feed that defaults the block time to the schedule for a stated share of
# departures without a flight-plan record. ``taxiout.challenge`` loads it
# unchanged; the record-quality term, the submission writer and the geometry
# pipeline can be run end to end on it, and the target's components are kept
# beside the ranking rows so a prediction can be scored against what it was
# supposed to recover.
#
# It is a fixture, not a simulation of any airport. Its one purpose is that
# the answer is known.

# The challenge schema, as (column, arrow type) in file order. The synthetic
# frames carry exactly these so the loaders and the feature layer see no
# difference from the real files.
CHALLENGE_SCHEMA: tuple[tuple[str, str], ...] = (
    ("MVT_ID_mvt", "double"),
    ("FLIGHT_ID_mvt", "double"),
    ("FLIGHT_mvt", "string"),
    ("FLIGHT_RULE_mvt", "string"),
    ("ADEP_mvt", "string"),
    ("ADES_mvt", "string"),
    ("PHASE_mvt", "string"),
    ("MVT_TIME_UTC_mvt", "timestamp"),
    ("BLOCK_TIME_UTC_mvt", "timestamp"),
    ("SCHED_TIME_UTC_mvt", "timestamp"),
    ("AIRCRAFT_TYPE_mvt", "string"),
    ("RUNWAY_mvt", "string"),
    ("STAND_mvt", "string"),
    ("TAXITIME_SEC_mvt", "int32"),
    ("LOBT_flt", "timestamp"),
    ("CALLSIGN_flt", "string"),
    ("ADEP_flt", "string"),
    ("ADES_flt", "string"),
    ("ADES_FILED_flt", "string"),
    ("MARKET_SEGMENT_flt", "string"),
    ("IOBT_flt", "timestamp"),
    ("FLIGHT_RULE_flt", "string"),
    ("FLIGHT_TYPE_flt", "string"),
    ("AIRCRAFT_TYPE_flt", "string"),
    ("WK_TBL_CAT_flt", "string"),
    ("AIRCRAFT_OPERATOR_flt", "string"),
    ("EOBT_1_flt", "timestamp"),
    ("ARVT_1_flt", "timestamp"),
    ("AOBT_3_flt", "timestamp"),
    ("ARVT_3_flt", "timestamp"),
)
FLT_COLUMNS: tuple[str, ...] = tuple(c for c, _ in CHALLENGE_SCHEMA if c.endswith("_flt"))

# The process. Seconds and meters throughout.
SPEED_MPS = 6.0  # unimpeded taxi speed over the field's routed distance
PUSHBACK_S = 45.0  # fixed component: the push itself
SERVICE_S = 60.0  # runway occupancy per departure, the queue's service time
NOISE_SD_S = 30.0  # what the process does not explain
DELAY_MEAN_S, DELAY_SD_S = 180.0, 300.0  # off-block relative to schedule
MIN_TAXI_S = 60.0
P_NO_RECORD = 0.03  # departures whose flight-plan join fails
P_DEFAULTED_IF_NO_RECORD = 0.5  # ... of which this share have a defaulted block time
P_LATE_WITH_RECORD = 0.01  # ordinary departures hours behind schedule
P_UNKNOWN_STAND = 0.02  # a stand identifier the field does not have
P_UNATTACHED_STAND = 0.01  # stand 11: on the field, attached to nothing
RUNWAY_DAY_WEIGHTS = {"09": 0.55, "27": 0.40, "36": 0.05}  # one runway in use per day
BANK_CENTRES_S = (
    7 * 3600.0,
    10.5 * 3600.0,
    14 * 3600.0,
    18.5 * 3600.0,
)  # four departure banks
BANK_SD_S = 8 * 60.0
DESTINATION = "ZZZY"
ARRIVAL_TAXI_NOISE_SD_S = 30.0
MVT_ID_BASE = 100_000_000.0


@dataclass
class Traffic:
    """A year of movements on the field, plus what the ranking rows hide."""

    training: object  # DataFrame in the challenge schema, 2025, DEP and ARR
    ranking: object  # DataFrame, January and July 2026, departures blanked as the challenge blanks them
    template: object  # DataFrame[MVT_ID_mvt, TAXITIME_SEC_mvt]
    truth: object  # DataFrame per ranking departure: the target and its components
    matrix: object  # DataFrame of the field's (stand, designator) cells, as the join would see them


def _cells_frame():
    import pandas as pd

    from .routing import AGGREGATED_FIELDS

    built = build()
    rows = []
    for c in built.cells:
        rows.append(
            {
                "stand_id": c.stand_id,
                "designator": c.designator,
                **{f: c.values[f] for f in AGGREGATED_FIELDS},
            }
        )
    return pd.DataFrame(rows)


def traffic(seed: int = 0, per_day: int = 40) -> Traffic:
    """Generate the fixture. Deterministic in ``seed``; ``per_day`` scales it."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    matrix = _cells_frame()
    taxi_m = {(r.stand_id, r.designator): r.taxi_m for r in matrix.itertuples()}
    stand_ids = sorted({s for s, _ in taxi_m}, key=int)
    mean_taxi_m = {
        d: float(np.mean([v for (s, dd), v in taxi_m.items() if dd == d]))
        for d in DESIGNATORS
    }

    def month_days(year: int, month: int) -> list[pd.Timestamp]:
        start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
        end = start + pd.offsets.MonthBegin(1)
        return list(pd.date_range(start, end, freq="D", inclusive="left"))

    train_days = [d for m in range(1, 13) for d in month_days(2025, m)]
    rank_days = month_days(2026, 1) + month_days(2026, 7)

    next_id = [MVT_ID_BASE]

    def ids(n: int):
        out = np.arange(next_id[0], next_id[0] + n, dtype=float)
        next_id[0] += n
        return out

    def departures(day: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
        n = per_day
        runway = rng.choice(list(RUNWAY_DAY_WEIGHTS), p=list(RUNWAY_DAY_WEIGHTS.values()))
        # Schedules come in banks, so the runway queue has something to do.
        bank = rng.choice(BANK_CENTRES_S, n)
        sched_s = np.sort(
            np.clip(bank + rng.normal(0, BANK_SD_S, n), 5 * 3600, 22.5 * 3600)
        )
        sched = day + pd.to_timedelta(np.round(sched_s / 60) * 60, unit="s")
        stand = rng.choice(stand_ids, n)
        kind = rng.choice(
            ["ok", "unknown", "unattached"],
            n,
            p=[
                1 - P_UNKNOWN_STAND - P_UNATTACHED_STAND,
                P_UNKNOWN_STAND,
                P_UNATTACHED_STAND,
            ],
        )
        stand = np.where(
            kind == "unknown", "77", np.where(kind == "unattached", FAR_STAND[0], stand)
        )
        dist = np.array([taxi_m.get((s, runway), mean_taxi_m[runway]) for s in stand])
        unimpeded = PUSHBACK_S + dist / SPEED_MPS
        delay = np.clip(rng.normal(DELAY_MEAN_S, DELAY_SD_S, n), -300, None)
        block_s = sched_s + delay
        # First-in-first-out runway queue in order of arrival at the runway.
        at_runway = block_s + unimpeded
        order = np.argsort(at_runway, kind="stable")
        takeoff_s = np.empty(n)
        free_at = -np.inf
        for i in order:
            free_at = max(free_at, at_runway[i]) + SERVICE_S
            takeoff_s[i] = free_at
        excess = takeoff_s - at_runway
        noise = rng.normal(0.0, NOISE_SD_S, n)
        taxi = np.maximum(MIN_TAXI_S, unimpeded + excess + noise)
        takeoff_s = block_s + taxi
        # The flight-plan join, and the feed that defaults block times.
        no_record = rng.random(n) < P_NO_RECORD
        defaulted = no_record & (rng.random(n) < P_DEFAULTED_IF_NO_RECORD)
        late = ~no_record & (rng.random(n) < P_LATE_WITH_RECORD)
        sched_s = sched_s.copy()
        # A late departure keeps its real pushback; its schedule was hours ago.
        sched_s[late] = takeoff_s[late] - rng.uniform(3600, 6 * 3600, late.sum())
        # A defaulted one: the schedule is hours before the take-off, and the
        # recorded block time is the schedule plus a few seconds, so the
        # recorded taxi-out is the lag.
        sched_s[defaulted] = takeoff_s[defaulted] - rng.uniform(
            2.5 * 3600, 15 * 3600, defaulted.sum()
        )
        sched_s = np.round(sched_s / 60) * 60
        recorded_block_s = block_s.copy()
        recorded_block_s[defaulted] = sched_s[defaulted] + rng.integers(
            0, 6, defaulted.sum()
        )
        recorded_taxi = np.round(takeoff_s - recorded_block_s)
        sched = day + pd.to_timedelta(sched_s, unit="s")
        block = day + pd.to_timedelta(np.round(recorded_block_s), unit="s")
        takeoff = day + pd.to_timedelta(np.round(takeoff_s), unit="s")
        mvt_id = ids(n)
        frame = pd.DataFrame(
            {
                "MVT_ID_mvt": mvt_id,
                "FLIGHT_ID_mvt": np.where(no_record, np.nan, mvt_id + 0.5),
                "FLIGHT_mvt": [f"ZZ{int(i % 9000) + 1000}" for i in mvt_id],
                "FLIGHT_RULE_mvt": "I",
                "ADEP_mvt": ICAO,
                "ADES_mvt": DESTINATION,
                "PHASE_mvt": "DEP",
                "MVT_TIME_UTC_mvt": takeoff,
                "BLOCK_TIME_UTC_mvt": block,
                "SCHED_TIME_UTC_mvt": sched,
                "AIRCRAFT_TYPE_mvt": rng.choice(
                    ["ZT1", "ZT2", "ZT3"], n, p=[0.6, 0.3, 0.1]
                ),
                "RUNWAY_mvt": runway,
                "STAND_mvt": stand,
                "TAXITIME_SEC_mvt": recorded_taxi.astype("int32"),
            }
        )
        frame = _add_flight_side(frame, rng, no_record, sched, block, takeoff)
        truth = pd.DataFrame(
            {
                "MVT_ID_mvt": mvt_id,
                "taxi_true_s": np.round(taxi),
                "recorded_s": recorded_taxi,
                "unimpeded_s": unimpeded,
                "excess_s": excess,
                "queue_ahead": 0,
                "defaulted": defaulted,
                "no_record": no_record,
                "late": late,
                "stand_kind": kind,
            }
        )
        return frame, truth

    def arrivals(day: pd.Timestamp) -> pd.DataFrame:
        n = per_day
        runway = rng.choice(list(RUNWAY_DAY_WEIGHTS), p=list(RUNWAY_DAY_WEIGHTS.values()))
        landing_s = np.sort(rng.uniform(5 * 3600, 23 * 3600, n))
        stand = rng.choice(stand_ids, n)
        dist = np.array([taxi_m.get((s, runway), mean_taxi_m[runway]) for s in stand])
        taxi_in = np.maximum(
            MIN_TAXI_S, dist / SPEED_MPS + rng.normal(0, ARRIVAL_TAXI_NOISE_SD_S, n)
        )
        onblock_s = landing_s + taxi_in
        sched_s = np.round((landing_s - rng.normal(0, 600, n)) / 60) * 60
        no_record = rng.random(n) < P_NO_RECORD / 2
        mvt_id = ids(n)
        landing = day + pd.to_timedelta(np.round(landing_s), unit="s")
        onblock = day + pd.to_timedelta(np.round(onblock_s), unit="s")
        sched = day + pd.to_timedelta(sched_s, unit="s")
        frame = pd.DataFrame(
            {
                "MVT_ID_mvt": mvt_id,
                "FLIGHT_ID_mvt": np.where(no_record, np.nan, mvt_id + 0.5),
                "FLIGHT_mvt": [f"ZZ{int(i % 9000) + 1000}" for i in mvt_id],
                "FLIGHT_RULE_mvt": "I",
                "ADEP_mvt": DESTINATION,
                "ADES_mvt": ICAO,
                "PHASE_mvt": "ARR",
                "MVT_TIME_UTC_mvt": landing,
                "BLOCK_TIME_UTC_mvt": onblock,
                "SCHED_TIME_UTC_mvt": sched,
                "AIRCRAFT_TYPE_mvt": rng.choice(
                    ["ZT1", "ZT2", "ZT3"], n, p=[0.6, 0.3, 0.1]
                ),
                "RUNWAY_mvt": runway,
                "STAND_mvt": stand,
                "TAXITIME_SEC_mvt": np.round(onblock_s - landing_s).astype("int32"),
            }
        )
        # For an arrival the flight-side times describe the departure from
        # the origin; they are filled so the columns are present, and are not
        # meaningful beyond that.
        origin_block = landing - pd.Timedelta(hours=2)
        return _add_flight_side(
            frame, rng, no_record, sched - pd.Timedelta(hours=2), origin_block, landing
        )

    def month_frames(days):
        deps, truths, arrs = [], [], []
        for day in days:
            d, t = departures(day)
            deps.append(d)
            truths.append(t)
            arrs.append(arrivals(day))
        frame = pd.concat(deps + arrs, ignore_index=True)
        frame = frame.sort_values("MVT_TIME_UTC_mvt", kind="stable").reset_index(
            drop=True
        )
        return frame, pd.concat(truths, ignore_index=True)

    training, _ = month_frames(train_days)
    ranking, truth = month_frames(rank_days)
    is_dep = ranking["PHASE_mvt"] == "DEP"
    ranking.loc[is_dep, "BLOCK_TIME_UTC_mvt"] = pd.NaT
    ranking.loc[is_dep, "TAXITIME_SEC_mvt"] = np.nan
    template = pd.DataFrame(
        {
            "MVT_ID_mvt": ranking.loc[is_dep, "MVT_ID_mvt"].to_numpy(),
            "TAXITIME_SEC_mvt": np.full(int(is_dep.sum()), np.nan),
        }
    )
    truth = truth.set_index("MVT_ID_mvt").loc[template["MVT_ID_mvt"]].reset_index()
    return Traffic(training, ranking, template, truth, matrix)


def _add_flight_side(frame, rng, no_record, sched, block, takeoff):
    """The ``_flt`` columns: present where the join succeeded, null as a block where it did not."""
    import numpy as np
    import pandas as pd

    n = len(frame)

    def minute(t, sd):
        """``t`` jittered by ``sd`` seconds and floored to the minute."""
        jitter = pd.to_timedelta(rng.normal(0, sd, n), unit="s")
        return (pd.DatetimeIndex(t) + jitter).floor("min")

    flt = pd.DataFrame(
        {
            "LOBT_flt": minute(block, 240),
            "CALLSIGN_flt": frame["FLIGHT_mvt"].to_numpy(),
            "ADEP_flt": frame["ADEP_mvt"].to_numpy(),
            "ADES_flt": frame["ADES_mvt"].to_numpy(),
            "ADES_FILED_flt": frame["ADES_mvt"].to_numpy(),
            "MARKET_SEGMENT_flt": rng.choice(
                ["Mainline", "Lowcost", "Regional"], n, p=[0.6, 0.25, 0.15]
            ),
            "IOBT_flt": minute(sched, 120),
            "FLIGHT_RULE_flt": "I",
            "FLIGHT_TYPE_flt": "S",
            "AIRCRAFT_TYPE_flt": frame["AIRCRAFT_TYPE_mvt"].to_numpy(),
            "WK_TBL_CAT_flt": np.where(
                frame["AIRCRAFT_TYPE_mvt"].to_numpy() == "ZT3", "H", "M"
            ),
            "AIRCRAFT_OPERATOR_flt": rng.choice(["ZZA", "ZZB", "ZZC"], n),
            "EOBT_1_flt": minute(sched, 300),
            "ARVT_1_flt": minute(sched + pd.Timedelta(hours=2), 300),
            "AOBT_3_flt": minute(block, 90),
            "ARVT_3_flt": minute(takeoff + pd.Timedelta(hours=2), 60),
        }
    )
    for c in flt.columns:
        if flt[c].dtype == object:
            flt[c] = flt[c].astype(object)
            flt.loc[no_record, c] = None
        else:
            flt.loc[no_record, c] = pd.NaT
    out = pd.concat([frame, flt], axis=1)
    return out[[c for c, _ in CHALLENGE_SCHEMA]]


def arrow_schema():
    import pyarrow as pa

    types = {
        "double": pa.float64(),
        "string": pa.string(),
        "timestamp": pa.timestamp("us", tz="UTC"),
        "int32": pa.int32(),
    }
    return pa.schema([(c, types[t]) for c, t in CHALLENGE_SCHEMA])


def write_traffic(t: Traffic, root) -> dict:
    """Write the fixture in the challenge's file layout under ``root``.

    ``root/movements/training_<month>_<next>.parquet`` for each month of
    2025, ``root/ranking.parquet`` and ``root/submitting.parquet``, with the
    challenge's arrow schema, so ``taxiout.challenge`` reads them as it reads
    the real files. Returns the paths written.
    """
    from pathlib import Path

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    (root / "movements").mkdir(parents=True, exist_ok=True)
    schema = arrow_schema()
    written = {}
    stamp = t.training["MVT_TIME_UTC_mvt"]
    month = stamp.dt.year * 100 + stamp.dt.month
    for ym, part in t.training.groupby(month, sort=True):
        start = pd.Timestamp(year=int(ym) // 100, month=int(ym) % 100, day=1)
        end = start + pd.offsets.MonthBegin(1)
        path = root / "movements" / f"training_{start:%Y-%m-%d}_{end:%Y-%m-%d}.parquet"
        pq.write_table(
            pa.Table.from_pandas(part, schema=schema, preserve_index=False), path
        )
        written[path.name] = len(part)
    path = root / "ranking.parquet"
    pq.write_table(
        pa.Table.from_pandas(t.ranking, schema=schema, preserve_index=False), path
    )
    written[path.name] = len(t.ranking)
    path = root / "submitting.parquet"
    tpl = pa.table(
        {
            "MVT_ID_mvt": pa.array(t.template["MVT_ID_mvt"].to_numpy(), pa.float64()),
            "TAXITIME_SEC_mvt": pa.array([None] * len(t.template), pa.int32()),
        }
    )
    pq.write_table(tpl, path)
    written[path.name] = len(t.template)
    return written


__all__ = [
    "CHALLENGE_SCHEMA",
    "DESIGNATORS",
    "DISTANCE_TOL_M",
    "FLT_COLUMNS",
    "FRAME",
    "ICAO",
    "REF",
    "SPEED_MPS",
    "THRESHOLDS",
    "TURN_TOL_DEG",
    "WORKED",
    "Built",
    "Traffic",
    "arrow_schema",
    "build",
    "elements",
    "expected_cell_meta",
    "expected_cells",
    "expected_stats",
    "penalty_crossovers",
    "probe_merge_shift",
    "probe_two_feature_tie",
    "traffic",
    "write_traffic",
    "xy",
]
