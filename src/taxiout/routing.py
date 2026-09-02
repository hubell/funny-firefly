# SPDX-License-Identifier: GPL-3.0-only
"""Stand to runway routing, and the route descriptors that come with it.

The output is deliberately not a single number. Ravizza et al. (2013) found
that decomposing a taxi route -- distance, total turn angle, junction-free
running -- explains substantially more of the variation in taxi speed than
distance alone, and that total turn angle on its own is nearly as predictive as
distance. So every route is reported as a small vector of physical descriptors,
and the choice of which ones earn their place is left to the modelling stage
where it can be measured against the target.

Distance is reported in four separable pieces rather than one total:

``lead_in_m``      stand stop position to the point where the stand meets the
                   network -- the pushback, which is slow and tug-limited.
``snap_m``         the gap the attachment had to bridge. Zero where the stand
                   genuinely touches a mapped taxiway; large where OSM has no
                   apron lead-in and this is a straight-line stand-in for
                   pavement that exists in reality but not in the data. Note
                   that a snapped stand loses more than distance: the apron
                   manoeuvre it replaces contributed turns and junctions, so
                   ``turn_deg`` is biased low for exactly those stands.
``route_m``        the routed distance over mapped pavement.
``entry_offset_m`` the gap between the threshold and the graph. Zero by
                   construction, since the threshold is a vertex of the runway
                   centreline and the centreline is in the graph; retained as a
                   guard, so that a future change which stops that being true
                   shows up as a number rather than as silently lost distance.
                   The along-runway component a full-length departure pays --
                   which an intersection departure does not -- is
                   ``backtrack_m``, not this.

Keeping them apart means a later modelling decision -- whether pushback counts,
whether to trust a snapped stand, whether departures line up at the threshold or
at an intersection -- can be made without recomputing geometry, and means a
route with a large fabricated component can be identified rather than averaged
into the honest ones.

Runway pavement is traversable but penalised. Aircraft do cross runways, and
occasionally backtrack along one, but they do not use a runway as a taxiway;
an unpenalised graph happily routes several kilometres down the nearest runway
because it is straight and unobstructed. The penalty is a modelling assumption,
stated here and exposed as a parameter so its effect can be measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

from .geo import Point, geodesic_m, turn_profile
from .graph import TaxiGraph, attach_point
from .runways import Runway

# Multiplier on runway-edge length when choosing a route. Crossing a runway
# costs a few tens of metres and stays cheap; running a kilometre along one
# does not. See docs/modelling_notes.md for the sensitivity check.
RUNWAY_PENALTY = 4.0


@dataclass
class RunwayEntry:
    """Where a departure on one designator joins the runway."""

    ref: str
    designator: str
    threshold: Point
    node: int | None
    entry_offset_m: float
    heading_error_deg: float


@dataclass
class Route:
    stand_id: str | None
    id_tag: str | None
    osm_id: int
    osm_type: str
    designator: str
    runway_ref: str
    lead_in_m: float
    snap_m: float
    route_m: float
    entry_offset_m: float
    turn_deg: float
    sharp_turns: int
    junctions: int
    straight_m: float
    backtrack_m: float
    cross_runway_m: float
    runway_crossings: int
    gc_m: float
    stand_lat: float
    stand_lon: float
    thr_lat: float
    thr_lon: float

    @property
    def taxi_m(self) -> float:
        """Total stand-to-threshold distance."""
        return self.lead_in_m + self.snap_m + self.route_m + self.entry_offset_m

    @property
    def detour_ratio(self) -> float:
        """Routed distance over straight-line distance.

        Below 1.0 is impossible for a real route and means the geometry is
        wrong; above roughly 3 means the route is wandering and is worth an
        eyeball. This is the cheapest sanity check available without ground
        truth, which is why it is computed for every pair rather than sampled.
        """
        return self.taxi_m / self.gc_m if self.gc_m > 0 else float("nan")


def _edge_cost(data: dict, runway_penalty: float) -> float:
    return data["length_m"] * (runway_penalty if data.get("kind") == "runway" else 1.0)


def runway_entries(
    tg: TaxiGraph, runways: list[Runway], *, threshold_snap_m: float = 250.0
) -> list[RunwayEntry]:
    """Place each designator's take-off point on the graph.

    The threshold is attached to the nearest pavement rather than matched to a
    pre-existing junction. Matching junctions was the first approach and it
    failed in a way worth recording: OSM frequently stops a taxiway at the
    runway edge instead of joining the centreline, so a runway can have no
    junction at all near one of its ends. The nearest junction was then at the
    far end, and Madrid's 32L departures were routed 1.5 km up the runway to
    reach a threshold 280 m from the stand.

    Attaching directly puts the entry node on the runway centreline at the
    threshold, and leaves the routing question -- taxi round to a nearer entry,
    or backtrack along the runway -- to the cost function, where the runway
    penalty can express it. Backtracking then shows up in ``backtrack_m``
    instead of hiding inside a threshold offset.
    """
    entries: list[RunwayEntry] = []
    for rw in runways:
        for designator in rw.designators:
            thr = rw.threshold(designator)
            node, snap = attach_point(tg, thr, max_snap_m=threshold_snap_m)
            entries.append(
                RunwayEntry(
                    ref=rw.ref,
                    designator=designator,
                    threshold=thr,
                    node=node,
                    entry_offset_m=snap if node is not None else float("nan"),
                    heading_error_deg=rw.heading_error_deg(designator),
                )
            )
    return entries


def label_runway_edges(tg: TaxiGraph, runways: list[Runway]) -> int:
    """Give every runway edge the ref of the runway it belongs to.

    ``build_runways`` re-attaches displaced-threshold pavement to the runway it
    extends, but those OSM ways carry no ``ref`` of their own, so the graph
    labels their edges ``ref=None``. That pavement is then invisible to
    ``runway_membership``, which needs a ref -- and the consequence is not
    cosmetic: at Fiumicino it hid a runway crossing from one route in six, and
    everywhere it occurred the length was booked as a backtrack along the
    departure runway rather than as a crossing of another one.

    Matching is by OSM way id rather than by geometry, so it inherits whatever
    ``_attach_displaced`` decided and cannot disagree with it. Returns the
    number of edges relabelled.
    """
    by_way = {wid: rw.ref for rw in runways for wid in rw.way_ids}
    relabelled = 0
    for _u, _v, d in tg.g.edges(data=True):
        if d.get("kind") != "runway" or d.get("ref"):
            continue
        ref = by_way.get(d.get("way_id"))
        if ref:
            d["ref"] = ref
            relabelled += 1
    return relabelled


def runway_membership(g: nx.Graph) -> dict[int, frozenset[str]]:
    """For each node, the runways whose centreline it lies *on*.

    A node is on runway ``R`` when it has two or more incident edges of that
    runway: that makes it interior to the centreline rather than an endpoint
    brushing past it.

    This exists because counting crossings from edges does not work, and fails
    silently. Repairing a taxiway/runway crossing inserts a node at the
    intersection and splits both ways there, so a route crossing the runway
    goes taxiway-edge, node, taxiway-edge -- it never traverses a single edge
    labelled runway. Counting foreign runway *edges* therefore reported zero
    crossings for every route at Paris CDG, which reads as a finding rather
    than a bug. Crossings are a property of the nodes a route passes through.
    """
    incident: dict[int, dict[str, int]] = {}
    for u, v, d in g.edges(data=True):
        ref = d.get("ref")
        if d.get("kind") != "runway" or not ref:
            continue
        for n in (u, v):
            incident.setdefault(n, {})
            incident[n][ref] = incident[n].get(ref, 0) + 1
    return {
        n: frozenset(r for r, c in refs.items() if c >= 2) for n, refs in incident.items()
    }


def _describe(
    tg: TaxiGraph,
    path: list[int],
    target_ref: str,
    membership: dict[int, frozenset[str]],
    *,
    turns: bool = True,
) -> dict:
    """Physical descriptors of one routed path.

    ``turns=False`` skips the polyline simplification, which dominates the cost
    and is not needed by callers that only want distances -- the penalty sweep
    re-routes every stand at every airport several times over, and paying for
    turn angles it discards made that check take three quarters of an hour.
    """
    g = tg.g
    points: list[Point] = [tg.point(n) for n in path]
    route_m = straight_m = backtrack_m = cross_runway_m = 0.0

    for u, v in zip(path, path[1:]):
        d = g.edges[u, v]
        length = d["length_m"]
        route_m += length
        # Runway pavement splits into two physically different things. Running
        # along the departure runway itself is a backtrack or line-up, part of
        # the approach to the take-off roll. Running along a *different* runway
        # is unusual and worth separating, because it predicts a different
        # delay. Edges with no ref are unattributed pavement, not foreign.
        if d.get("kind") == "runway":
            ref = d.get("ref")
            if ref and ref != target_ref:
                cross_runway_m += length
            else:
                backtrack_m += length
        # Junction-free running, in Ravizza's sense: pavement between
        # decision points. Note it is *not* straightness -- a degree-2 vertex
        # can be a bend, so a curve with no junction on it counts in full.
        if g.degree(u) == 2 and g.degree(v) == 2:
            straight_m += length

    # A crossing is a run of consecutive path nodes lying on the same foreign
    # runway. Counting runs makes one crossing count once however many segments
    # it was drawn in; tracking which runway makes two adjacent crossings of
    # different runways count twice.
    crossings = 0
    previous: frozenset[str] = frozenset()
    for n in path:
        foreign = membership.get(n, frozenset()) - {target_ref}
        if foreign and not (foreign & previous):
            crossings += 1
        previous = foreign

    junctions = sum(1 for n in path[1:-1] if g.degree(n) >= 3)
    turn_deg, sharp = turn_profile(points) if turns else (float("nan"), 0)
    return {
        "route_m": route_m,
        "turn_deg": turn_deg,
        "sharp_turns": sharp,
        "junctions": junctions,
        "straight_m": straight_m,
        "backtrack_m": backtrack_m,
        "cross_runway_m": cross_runway_m,
        "runway_crossings": crossings,
    }


def route_all(
    tg: TaxiGraph,
    runways: list[Runway],
    *,
    runway_penalty: float = RUNWAY_PENALTY,
    turns: bool = True,
) -> tuple[list[Route], list[RunwayEntry], dict]:
    """Route every attached stand to every runway designator.

    One Dijkstra per designator over the whole graph, rather than one per
    stand-designator pair: there are a handful of designators and hundreds of
    stands, so the cheap direction is outward from the runway.
    """
    g = tg.g
    # Entries are placed before the stand list is fixed: attaching a threshold
    # can split an edge, and a stand node must not be read across that change.
    relabelled = label_runway_edges(tg, runways)
    entries = runway_entries(tg, runways)
    stands = [s for s in tg.stands if s.node is not None]
    routes: list[Route] = []
    unreachable = 0

    def weight(_u: int, _v: int, data: dict) -> float:
        return _edge_cost(data, runway_penalty)

    membership = runway_membership(g)

    for entry in entries:
        if entry.node is None:
            unreachable += len(stands)
            continue
        _, paths = nx.single_source_dijkstra(g, entry.node, weight=weight)
        for stand in stands:
            path = paths.get(stand.node)
            if path is None:
                unreachable += 1
                continue
            # Dijkstra ran outward from the runway; the aircraft goes the other
            # way. Every descriptor here is symmetric under reversal, so this is
            # for readability of the stored path rather than for correctness.
            desc = _describe(tg, path[::-1], entry.ref, membership, turns=turns)
            routes.append(
                Route(
                    stand_id=stand.stand_id,
                    id_tag=stand.id_tag,
                    osm_id=stand.osm_id,
                    osm_type=stand.osm_type,
                    designator=entry.designator,
                    runway_ref=entry.ref,
                    lead_in_m=stand.lead_in_m,
                    snap_m=stand.snap_m,
                    entry_offset_m=entry.entry_offset_m,
                    gc_m=geodesic_m(stand.stop, entry.threshold),
                    stand_lat=stand.stop[0],
                    stand_lon=stand.stop[1],
                    thr_lat=entry.threshold[0],
                    thr_lon=entry.threshold[1],
                    **desc,
                )
            )

    # A designator whose departures routinely run a long way up their own
    # runway is not a modelling nuance, it is a missing taxiway: OSM has no
    # connection near that threshold, so the router reaches it the only way it
    # can. Surfacing it per designator makes the gap fixable upstream.
    per_designator: dict[str, list[float]] = {}
    for r in routes:
        per_designator.setdefault(r.designator, []).append(r.backtrack_m)
    backtracking = {}
    for d, v in per_designator.items():
        med = _median(v)
        if med <= 300.0:
            continue
        # Whether the offset is the same for every stand decides how much it
        # costs. A constant is absorbed by any model that knows which runway
        # was used; one that varies distorts the distances between stands,
        # which is the signal the whole feature exists to carry.
        backtracking[d] = {"median_m": med, "spread_m": max(v) - min(v)}

    n_pairs = len(stands) * len(entries)
    stats = {
        "designators": len(entries),
        "designators_without_entry": sum(1 for e in entries if e.node is None),
        "stands_routable": len({r.osm_id for r in routes}),
        "pairs_possible": n_pairs,
        "pairs_routed": len(routes),
        "pairs_unreachable": unreachable,
        "routable_share": len(routes) / n_pairs if n_pairs else 0.0,
        "runway_edges_relabelled": relabelled,
        "backtracking_designators": backtracking,
    }
    return routes, entries, stats


@dataclass
class StandRunwayCell:
    """One joinable ``(stand_id, designator)`` cell of the distance matrix.

    Several OSM features can share a stand identifier -- a stand drawn twice,
    or a wide stand split into left and right positions. One of them is chosen
    as the representative and every field is taken from it, so each row remains
    a real route.

    Taking a per-column median instead was the first approach, and it produced
    rows that did not describe anything: ``taxi_m`` stopped equalling the sum of
    its own components in 22% of Istanbul's cells, and integer counts came out
    as halves. The spread between features is kept separately, along with the
    largest snap among them, because a stand identifier whose duplicates
    disagree is a data-quality signal worth more than the metre or two that
    averaging would have saved.
    """

    stand_id: str
    designator: str
    runway_ref: str
    n_features: int
    spread_m: float
    snap_max_m: float
    values: dict = field(default_factory=dict)


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


AGGREGATED_FIELDS = (
    "taxi_m",
    "lead_in_m",
    "snap_m",
    "route_m",
    "entry_offset_m",
    "turn_deg",
    "sharp_turns",
    "junctions",
    "straight_m",
    "backtrack_m",
    "cross_runway_m",
    "runway_crossings",
    "gc_m",
    "detour_ratio",
)


def aggregate(routes: list[Route]) -> list[StandRunwayCell]:
    """Collapse per-feature routes into one row per identified stand.

    The representative is the real feature whose ``taxi_m`` lies closest to the
    group mean. Taking the lower of the two middle values was the first choice
    and it is not neutral: with two features -- much the commonest case -- it
    always picks the shorter route, biasing Barcelona's cells 154 m low on
    average. Closest-to-mean keeps the row an actual route while removing the
    direction from the error.
    """
    buckets: dict[tuple[str, str], list[Route]] = {}
    for r in routes:
        if not r.stand_id:
            continue
        buckets.setdefault((r.stand_id, r.designator), []).append(r)

    cells: list[StandRunwayCell] = []
    for (stand_id, designator), group in sorted(buckets.items()):
        ordered = sorted(group, key=lambda r: r.taxi_m)
        mean = sum(r.taxi_m for r in ordered) / len(ordered)
        pick = min(ordered, key=lambda r: (abs(r.taxi_m - mean), r.osm_id))
        cells.append(
            StandRunwayCell(
                stand_id=stand_id,
                designator=designator,
                runway_ref=pick.runway_ref,
                n_features=len(group),
                spread_m=ordered[-1].taxi_m - ordered[0].taxi_m,
                snap_max_m=max(r.snap_m for r in group),
                values={f: float(getattr(pick, f)) for f in AGGREGATED_FIELDS},
            )
        )
    return cells
