# SPDX-License-Identifier: GPL-3.0-only
"""A routable surface-movement graph built from OSM aeroway geometry.

The topology OSM gives you is not the topology an aircraft uses. Three defects
show up consistently and each is repaired explicitly here, because each one
silently changes taxi distance if left alone:

1. **Coincident nodes.** Two ways drawn to the same corner can end at two
   different node IDs a few centimetres apart. Topologically they are separate;
   physically they are one junction. Unmerged, a taxiway pair that visibly meets
   on the map is unreachable in the graph.

2. **Crossings without junctions.** A taxiway drawn across a runway shares no
   node with it. The map looks right and the graph is disconnected.

3. **Stands hanging off the network.** ``aeroway=parking_position`` is often a
   short stub that stops short of the taxiway centreline it feeds.

Defect 1 is repaired by merging nodes that share a position. Defect 2 needs two
passes, because it has two shapes: where a node sits on an edge it is not joined
to, the edge is split at that node; where two ways cross with no vertex at the
crossing at all, the intersection is computed and a junction inserted, since
there is nothing there to split. Defect 3 is repaired by projecting the stand's
entry end onto the nearest edge, with the projection distance recorded for every
stand so the report can show how much of the network is real and how much was
invented by snapping.

Tolerances are deliberately tight. A generous snap would connect everything and
report perfect coverage while quietly routing aircraft through buildings; the
point of this module is to produce a number that can be trusted, and that means
preferring an honest gap to a fabricated link.

Prior art: ``traffic`` (Olive, MIT) ships an ``AirportGraph`` that performs the
same class of repair on OSM aeroway data, and Olive, Waltert & Schultz (ATRD
2025) document these defects. This is an independent implementation -- kept
in-repo to hold the dependency surface small and the geometry auditable -- not a
port; see ``docs/modelling_notes.md`` for the reasoning and the cross-check.

Data source: OpenStreetMap, (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from .geo import (
    LocalFrame,
    Point,
    centroid,
    geodesic_m,
    path_length_m,
    project_on_segment,
)

# Way kinds that carry aircraft under their own power. `parking_position` is
# handled separately: it is an attachment point, not a through route.
ROUTABLE_KINDS: tuple[str, ...] = ("taxiway", "taxilane", "runway")

# Two OSM nodes closer than this are the same physical point. 0.5 m is below
# the positional accuracy of the imagery these ways are traced from, so merging
# within it cannot join two genuinely distinct junctions.
DEDUP_M = 0.5

# A node this close to an unrelated segment was meant to sit on it. Kept tight:
# taxiway centrelines at a hub can pass within a few metres without connecting.
SPLIT_M = 1.5

# Largest projection accepted when linking a stand to the network, and the tier
# boundary below which the link is a genuine lead-in rather than a stand-in for
# unmapped apron.
#
# Chosen from the measured distribution, not picked: where OSM maps the apron
# lead-in the stand touches the network outright (median 0.0 m at EGLL, LSZH,
# EDDF). Where it does not, the gap jumps straight to ~105-137 m -- at EGLL
# every one of the 70 node-form parking positions sits in that band, with no
# stands in between. So there is no tolerance that trades the two off; there is
# only a choice about whether to carry the second group at all.
#
# They are carried, because a stand with an approximate distance is worth more
# to a model than a stand with none, and dropped precision is recoverable while
# a dropped stand is not. Every attachment reports ``snap_m`` so the fabricated
# component stays visible and filterable downstream.
MAX_SNAP_M = 150.0

# Attachments at or below this are real: the stand meets mapped pavement.
DIRECT_SNAP_M = 60.0

# Grid cell for the segment index, in metres. Any value comfortably above
# SPLIT_M and MAX_SNAP_M works; it only affects speed.
CELL_M = 50.0


@dataclass
class StandNode:
    """A parking position resolved onto the graph."""

    osm_id: int
    osm_type: str
    stand_id: str | None
    id_tag: str | None
    stop: Point  # where the aircraft parks
    entry: Point  # where the stand meets the network
    node: int | None  # graph node, None if unattached
    snap_m: float  # distance the entry end was moved to attach
    lead_in_m: float  # length of the parking_position way itself
    geometry: str = "line"  # line | polygon | point


class SegmentIndex:
    """Uniform grid over graph edges, for nearest-segment queries.

    Entries are append-only; edges removed by a split stay in the grid as stale
    pairs and are filtered on read by checking the graph. That keeps insertion
    cheap during the many small mutations that snapping performs.
    """

    def __init__(self, frame: LocalFrame) -> None:
        self.frame = frame
        self.cells: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)

    def _span(self, a: Point, b: Point) -> list[tuple[int, int]]:
        ax, ay = self.frame.to_xy(a)
        bx, by = self.frame.to_xy(b)
        i0, i1 = sorted((int(math.floor(ax / CELL_M)), int(math.floor(bx / CELL_M))))
        j0, j1 = sorted((int(math.floor(ay / CELL_M)), int(math.floor(by / CELL_M))))
        return [(i, j) for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)]

    def add(self, u: int, v: int, a: Point, b: Point) -> None:
        for cell in self._span(a, b):
            self.cells[cell].append((u, v))

    def near(self, p: Point, radius_m: float) -> set[tuple[int, int]]:
        px, py = self.frame.to_xy(p)
        span = int(math.ceil(radius_m / CELL_M))
        i0, j0 = int(math.floor(px / CELL_M)), int(math.floor(py / CELL_M))
        out: set[tuple[int, int]] = set()
        for i in range(i0 - span, i0 + span + 1):
            for j in range(j0 - span, j0 + span + 1):
                out.update(self.cells.get((i, j), ()))
        return out


@dataclass
class TaxiGraph:
    icao: str
    frame: LocalFrame
    g: nx.Graph
    index: SegmentIndex
    stands: list[StandNode] = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    _next_node: int = 0
    _deduped_m: float = 0.0  # pavement collapsed because it was drawn twice

    def point(self, node: int) -> Point:
        d = self.g.nodes[node]
        return d["lat"], d["lon"]

    def new_node_id(self) -> int:
        self._next_node += 1
        return self._next_node

    def candidate_edges(self, p: Point, radius_m: float) -> list[tuple[int, int]]:
        return [uv for uv in self.index.near(p, radius_m) if self.g.has_edge(*uv)]

    def nearest_edge(
        self, p: Point, radius_m: float
    ) -> tuple[tuple[int, int] | None, Point | None, float, float]:
        """Nearest edge to ``p``: ``(edge, foot, distance_m, t)``. Read-only."""
        best: tuple[int, int] | None = None
        best_foot: Point | None = None
        best_d, best_t = float("inf"), 0.0
        for u, v in self.candidate_edges(p, radius_m):
            a = self.point(u)
            b = self.point(v)
            foot, d, t = project_on_segment(self.frame, p, a, b)
            if d < best_d:
                best, best_foot, best_d, best_t = (u, v), foot, d, t
        return best, best_foot, best_d, best_t


class _NodeMerger:
    """Assigns one canonical graph node per physical position."""

    def __init__(self, frame: LocalFrame, tol_m: float = DEDUP_M) -> None:
        self.frame = frame
        self.tol = tol_m
        self.cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.points: dict[int, Point] = {}
        self.merged = 0
        self.next_id = 0

    def canonical(self, p: Point) -> int:
        x, y = self.frame.to_xy(p)
        i0, j0 = int(math.floor(x / self.tol)), int(math.floor(y / self.tol))
        for i in range(i0 - 1, i0 + 2):
            for j in range(j0 - 1, j0 + 2):
                for node in self.cells.get((i, j), ()):
                    gap = geodesic_m(p, self.points[node])
                    if gap <= self.tol:
                        # Only a *near* coincidence is a repair. An exact
                        # coordinate match is ordinary way-sharing -- two ways
                        # meeting at a node they both reference -- and counting
                        # it would report Frankfurt as having a thousand OSM
                        # defects when most of those are the format working.
                        if gap > 0.0:
                            self.merged += 1
                        return node
        node = self.next_id
        self.next_id += 1
        self.points[node] = p
        self.cells[(i0, j0)].append(node)
        return node


def _way_points(el: dict) -> list[Point]:
    return [(g["lat"], g["lon"]) for g in el.get("geometry", [])]


def _is_closed(pts: list[Point], tags: dict) -> bool:
    """Is this way a footprint polygon rather than a centreline?

    Antalya draws 168 of its 576 stands this way and no other airport in scope
    draws any -- a local mapping convention that has to be detected rather than
    assumed away.
    """
    return len(pts) >= 3 and (pts[0] == pts[-1] or tags.get("area") == "yes")


def build_graph(
    icao: str,
    elements: list[dict],
    ref_point: Point,
    *,
    split_m: float = SPLIT_M,
    dedup_m: float = DEDUP_M,
) -> TaxiGraph:
    """Assemble the routable graph for one airport.

    ``elements`` is the raw Overpass element list; ``ref_point`` is the
    aerodrome reference point, used as the origin of the local frame.
    """
    frame = LocalFrame(*ref_point)
    merger = _NodeMerger(frame, dedup_m)
    g = nx.Graph()

    ways = [
        el
        for el in elements
        if el.get("type") == "way"
        and (el.get("tags") or {}).get("aeroway") in ROUTABLE_KINDS
    ]

    for el in ways:
        tags = el.get("tags") or {}
        kind = tags["aeroway"]
        pts = _way_points(el)
        nodes = [merger.canonical(p) for p in pts]
        for (u, p), (v, q) in zip(zip(nodes, pts), zip(nodes[1:], pts[1:])):
            if u == v:
                continue  # collapsed to zero length by merging
            g.add_node(u, lat=p[0], lon=p[1])
            g.add_node(v, lat=q[0], lon=q[1])
            existing = g.get_edge_data(u, v)
            if existing is not None:
                # Runway wins the label: shared pavement must be costed and
                # counted as runway, whichever way was read first. A ref is only
                # ever upgraded, never cleared -- an untagged runway way read
                # after a tagged one must not erase the runway's identity.
                if kind == "runway":
                    existing["kind"] = "runway"
                    if tags.get("ref"):
                        existing["ref"] = tags["ref"]
                continue
            g.add_edge(
                u,
                v,
                length_m=geodesic_m(p, q),
                kind=kind,
                ref=tags.get("ref"),
                way_id=el["id"],
            )

    index = SegmentIndex(frame)
    for u, v in g.edges():
        index.add(
            u,
            v,
            (g.nodes[u]["lat"], g.nodes[u]["lon"]),
            (g.nodes[v]["lat"], g.nodes[v]["lon"]),
        )

    tg = TaxiGraph(
        icao=icao,
        frame=frame,
        g=g,
        index=index,
        _next_node=merger.next_id,
        stats={
            "ways": len(ways),
            "nodes_merged": merger.merged,
            "nodes_before_split": g.number_of_nodes(),
            "edges_before_split": g.number_of_edges(),
            "components_before_split": nx.number_connected_components(g),
        },
    )

    splits, crossings = repair_topology(tg, split_m=split_m)
    tg.stats["edges_split"] = splits
    tg.stats["crossings_split"] = crossings
    tg.stats["components_after_split"] = nx.number_connected_components(g)
    tg.stats["nodes"] = g.number_of_nodes()
    tg.stats["edges"] = g.number_of_edges()
    tg.stats["pavement_km"] = (
        sum(d["length_m"] for _, _, d in g.edges(data=True)) / 1000.0
    )
    # Splitting reveals pavement that was drawn twice: once the two versions
    # share endpoints they are one edge, and the duplicate length is dropped.
    # Reported so the shrinkage is visible rather than looking like lost data.
    tg.stats["pavement_deduped_km"] = tg._deduped_m / 1000.0

    if g.number_of_nodes():
        largest = max(nx.connected_components(g), key=len)
        tg.stats["largest_component_nodes"] = len(largest)
        tg.stats["largest_component_share"] = len(largest) / g.number_of_nodes()
    else:
        tg.stats["largest_component_nodes"] = 0
        tg.stats["largest_component_share"] = 0.0

    return tg


def _put_edge(tg: TaxiGraph, u: int, v: int, a: Point, b: Point, data: dict) -> None:
    """Add an edge without overwriting one that is already there.

    Splitting can land a new half on top of an edge that already exists, when
    the same pavement was drawn twice. A plain ``add_edge`` would silently
    replace that edge's attributes -- which quietly deleted several percent of
    the network's measured length before this was caught. The existing edge is
    kept, taking the shorter of the two lengths (they describe the same
    pavement) and the stronger label, since runway pavement must stay costed as
    runway however it was first read.
    """
    g = tg.g
    old = g.get_edge_data(u, v)
    if old is not None:
        tg._deduped_m += data["length_m"]
        old["length_m"] = min(old["length_m"], data["length_m"])
        if data.get("kind") == "runway":
            old["kind"] = "runway"
            if data.get("ref"):
                old["ref"] = data["ref"]
        return
    g.add_edge(u, v, **data)
    tg.index.add(u, v, a, b)


def _split_edge(tg: TaxiGraph, u: int, v: int, at: int) -> None:
    """Cut edge ``u-v`` and rejoin it through node ``at``, which lies on it.

    Edge attributes carry over to both halves: the two pieces are the same
    pavement, so they keep the same kind and ref.
    """
    g = tg.g
    a, b, p = tg.point(u), tg.point(v), tg.point(at)
    data = dict(g.edges[u, v])
    g.remove_edge(u, v)
    _put_edge(tg, u, at, a, p, {**data, "length_m": geodesic_m(a, p)})
    _put_edge(tg, at, v, p, b, {**data, "length_m": geodesic_m(p, b)})


def snap_and_split(
    tg: TaxiGraph,
    tol_m: float = SPLIT_M,
    nodes: set[int] | None = None,
    attempted: set[tuple[int, int, int]] | None = None,
) -> tuple[int, set[int]]:
    """Split edges at nodes that lie on them but are not connected to them.

    This is what turns a drawn crossing into a routable junction. Only exact
    coincidences within ``tol_m`` are repaired, and a split never moves the
    node -- the edge is cut and rejoined through the node's existing position,
    so total pavement length is preserved to within the tolerance.

    Returns ``(splits, touched)``, where ``touched`` is the neighbourhood of
    every split. A later round only needs to look there: splitting perturbs the
    two halves of one edge and can only bring nodes near *that* edge newly into
    tolerance. Rescanning the whole airport each round instead was correct and
    took ten minutes; this takes seconds and finds the same repairs.

    ``attempted`` records every (node, edge) combination already considered, so
    that no combination is examined twice across rounds. This is what makes the
    repair terminate. Without it the process has no fixed point: splitting an
    edge and collapsing a duplicate can restore an earlier configuration, and
    Zurich settled into a cycle of fifteen splits per round forever, which made
    the output a function of how many rounds were permitted rather than of the
    airport.
    """
    splits = 0
    touched: set[int] = set()
    if attempted is None:
        attempted = set()
    # Iterate in position order, not insertion order. Splitting at a node up to
    # tol_m off the chord perturbs the two halves, so the result depends on the
    # order nodes are visited -- and insertion order is the order OSM ways
    # happened to be read. Shuffling it moved Zurich's reported pavement by
    # 0.4%. Sorting by position makes the output a function of the airport.
    scan = tg.g.nodes() if nodes is None else [n for n in nodes if n in tg.g]
    for node in sorted(scan, key=tg.point):
        p = tg.point(node)
        for u, v in tg.candidate_edges(p, tol_m):
            # Re-check: an earlier split in this same pass may have consumed
            # the edge that the candidate list was built from.
            if node in (u, v) or not tg.g.has_edge(u, v):
                continue
            key = (node, *sorted((u, v)))
            if key in attempted:
                continue
            attempted.add(key)
            _, dist, t = project_on_segment(tg.frame, p, tg.point(u), tg.point(v))
            if dist > tol_m:
                continue
            # Margins in metres, not in t. This was the one place the parametric
            # test survived, and it did real damage: in dense clusters where
            # edges are a couple of metres long -- comparable to tol_m itself --
            # it split off half-edges of a few millimetres, whose midpoints then
            # sat on their neighbours, which split those. The repair never
            # reached a fixed point; it entered a limit cycle, splitting and
            # unsplitting the same edges forever, and the output became a
            # function of how many rounds were allowed rather than of the
            # airport.
            edge_m = tg.g.edges[u, v]["length_m"]
            if t * edge_m <= DEDUP_M or (1.0 - t) * edge_m <= DEDUP_M:
                continue
            _split_edge(tg, u, v, node)
            splits += 1
            touched.update((node, u, v))
            for a, b in tg.candidate_edges(p, CELL_M):
                touched.update((a, b))
    return splits, touched


def _segment_intersection(
    frame: LocalFrame, a: Point, b: Point, c: Point, d: Point
) -> Point | None:
    """Where segments ``a-b`` and ``c-d`` properly cross, if they do.

    Proper means strictly interior to both: touching at an endpoint is already
    a shared node, or a near-miss that ``snap_and_split`` handles. Only a true
    crossing is repaired here, so parallel taxiways and T-junctions drawn
    correctly are left alone.
    """
    ax, ay = frame.to_xy(a)
    bx, by = frame.to_xy(b)
    cx, cy = frame.to_xy(c)
    dx, dy = frame.to_xy(d)
    rx, ry = bx - ax, by - ay
    sx, sy = dx - cx, dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) < 1e-12:
        return None  # parallel or degenerate
    t = ((cx - ax) * sy - (cy - ay) * sx) / denom
    u = ((cx - ax) * ry - (cy - ay) * rx) / denom
    # Interior margins expressed in metres rather than in t and u, whose scale
    # depends on the length of the segment they parameterise: a fixed 1e-9 is
    # 3 micrometres on a 3 km runway edge and a nanometre on a 1 m taxiway one.
    len_r = math.hypot(rx, ry)
    len_s = math.hypot(sx, sy)
    if len_r <= 0.0 or len_s <= 0.0:
        return None
    margin_r = DEDUP_M / len_r
    margin_s = DEDUP_M / len_s
    if not (margin_r < t < 1.0 - margin_r and margin_s < u < 1.0 - margin_s):
        return None
    return frame.to_latlon(ax + t * rx, ay + t * ry)


def split_crossings(tg: TaxiGraph) -> int:
    """Insert a junction wherever two ways cross without sharing a node.

    OSM records a taxiway crossing a runway as two lines that intersect on the
    map and nowhere in the data. ``snap_and_split`` cannot repair it, because
    there is no node at the crossing to split anything at -- the vertex simply
    does not exist. Without this step a runway can be reachable only where a
    mapper happened to place a shared node, which is not a property of the
    airport.

    Candidate pairs come from the grid index, so only edges that already share
    a cell are tested against each other.
    """
    g = tg.g
    splits = 0
    for cell in sorted(tg.index.cells):
        # Converge cell by cell. A split replaces the edge the remaining
        # candidates were being tested against, so a runway crossed by four
        # taxiways needs four sweeps of its cell -- but only of its cell.
        # Restarting the whole airport instead was correct and about fifteen
        # times slower.
        while True:
            hit_found = False
            edges = [e for e in tg.index.cells[cell] if g.has_edge(*e)]
            for i, (u, v) in enumerate(edges):
                if not g.has_edge(u, v):
                    continue
                for p, q in edges[i + 1 :]:
                    if not g.has_edge(p, q) or {u, v} & {p, q}:
                        continue
                    hit = _segment_intersection(
                        tg.frame, tg.point(u), tg.point(v), tg.point(p), tg.point(q)
                    )
                    if hit is None:
                        continue
                    new_node = tg.new_node_id()
                    g.add_node(new_node, lat=hit[0], lon=hit[1])
                    _split_edge(tg, u, v, new_node)
                    _split_edge(tg, p, q, new_node)
                    splits += 1
                    hit_found = True
                    break
                if hit_found:
                    break
            if not hit_found:
                break
    return splits


def repair_topology(
    tg: TaxiGraph, *, split_m: float = SPLIT_M, max_rounds: int = 40
) -> tuple[int, int]:
    """Run both repairs to a fixed point. Returns ``(node_splits, crossings)``.

    One pass of each is not enough, because each repair creates edges that can
    themselves need repairing: splitting at a node up to ``split_m`` off the
    chord moves the two halves, which can bring further nodes into tolerance,
    and inserting a crossing junction replaces the edge that the next crossing
    would have been found against. Running each once found 42 crossings at
    Munich where there are 70.

    ``max_rounds`` is a guard rather than an expectation. Termination is
    guaranteed by the ``attempted`` set -- there are finitely many (node, edge)
    combinations and none is considered twice -- and every airport in scope
    settles well inside the cap. ``repair_rounds`` and ``repair_converged`` are
    recorded so a build that hits the cap says so instead of looking finished.
    """
    total_splits, total_crossings = 0, split_crossings(tg)
    frontier: set[int] | None = None
    attempted: set[tuple[int, int, int]] = set()
    rounds = 0
    for rounds in range(1, max_rounds + 1):
        splits, touched = snap_and_split(tg, split_m, frontier, attempted)
        total_splits += splits
        if splits == 0:
            break
        frontier = touched
    tg.stats["repair_rounds"] = rounds
    tg.stats["repair_converged"] = splits == 0
    return total_splits, total_crossings


def attach_point(
    tg: TaxiGraph, p: Point, *, max_snap_m: float = MAX_SNAP_M
) -> tuple[int | None, float]:
    """Attach an arbitrary point to the graph, returning ``(node, snap_m)``.

    A point coinciding with an existing node reuses it. Otherwise the nearest
    edge within ``max_snap_m`` is split at the projection foot and a node
    inserted there. Returns ``(None, inf)`` when nothing is close enough: an
    unattached point is reported, never forced on.
    """
    edge, foot, dist, t = tg.nearest_edge(p, max_snap_m)
    if edge is None or dist > max_snap_m:
        return None, float("inf")

    u, v = edge
    if geodesic_m(p, tg.point(u)) <= DEDUP_M:
        return u, geodesic_m(p, tg.point(u))
    if geodesic_m(p, tg.point(v)) <= DEDUP_M:
        return v, geodesic_m(p, tg.point(v))

    # Clamp in metres, not in the parameter t. A foot at t=1e-4 on a 500 m edge
    # is 5 cm from the endpoint: splitting there creates a sub-decimetre edge
    # that is pure numerical noise in the graph. Reuse the endpoint instead.
    edge_m = tg.g.edges[u, v]["length_m"]
    if t * edge_m <= DEDUP_M:
        return u, geodesic_m(p, tg.point(u))
    if (1.0 - t) * edge_m <= DEDUP_M:
        return v, geodesic_m(p, tg.point(v))

    assert foot is not None
    new_node = tg.new_node_id()
    tg.g.add_node(new_node, lat=foot[0], lon=foot[1])
    _split_edge(tg, u, v, new_node)
    return new_node, dist


def probe_distance(tg: TaxiGraph, p: Point, *, radius_m: float = MAX_SNAP_M) -> float:
    """Distance from ``p`` to the nearest edge, without touching the graph."""
    _, _, dist, _ = tg.nearest_edge(p, radius_m)
    return dist


def attach_stands(
    tg: TaxiGraph, elements: list[dict], *, max_snap_m: float = MAX_SNAP_M
) -> None:
    """Resolve every ``aeroway=parking_position`` onto the graph.

    A parking position drawn as a way runs between the stop position and the
    taxiway that feeds it; which end is which is decided by measuring, not
    assumed, since the drawing direction is not a convention anyone follows.
    The end nearer the network is the entry, the far end is where the aircraft
    parks, and the way's own length is kept as ``lead_in_m`` -- it is the part
    of the movement covered under tow, which taxi time may or may not include.
    """
    from .coverage import stand_identifier  # local import: avoids a cycle

    out: list[StandNode] = []
    for el in elements:
        tags = el.get("tags") or {}
        if tags.get("aeroway") != "parking_position":
            continue
        found = stand_identifier(tags)
        stand_id = found[1] if found else None
        id_tag = found[0] if found else None

        if el.get("type") == "node":
            stop = entry = (el["lat"], el["lon"])
            lead_in = 0.0
            geometry = "point"
        else:
            pts = _way_points(el)
            if len(pts) < 2:
                continue
            if _is_closed(pts, tags):
                # A stand drawn as its footprint rather than its centreline.
                # It has no head and no tail, so there is no lead-in to measure
                # and no entry end to choose: treating its perimeter as a
                # lead-in produced 270 m pushbacks at Antalya before this case
                # was handled. The centroid is the only defensible reduction.
                stop = entry = centroid(pts[:-1] if pts[0] == pts[-1] else pts)
                lead_in = 0.0
                geometry = "polygon"
            else:
                head, tail = pts[0], pts[-1]
                # Probe both ends read-only; the nearer one feeds the network.
                if probe_distance(tg, head, radius_m=max_snap_m) <= probe_distance(
                    tg, tail, radius_m=max_snap_m
                ):
                    entry, stop = head, tail
                else:
                    entry, stop = tail, head
                lead_in = path_length_m(pts)
                geometry = "line"

        node, snap = attach_point(tg, entry, max_snap_m=max_snap_m)
        out.append(
            StandNode(
                osm_id=el["id"],
                osm_type=el["type"],
                stand_id=stand_id,
                id_tag=id_tag,
                stop=stop,
                entry=entry,
                node=node,
                snap_m=snap,
                lead_in_m=lead_in,
                geometry=geometry,
            )
        )

    tg.stands = out
    attached = [s for s in out if s.node is not None]
    snaps = sorted(s.snap_m for s in attached)
    tg.stats["stands_total"] = len(out)
    tg.stats["stands_attached"] = len(attached)
    tg.stats["stands_with_id"] = sum(1 for s in out if s.stand_id)
    tg.stats["stands_direct"] = sum(1 for s in attached if s.snap_m <= DIRECT_SNAP_M)
    tg.stats["stands_bridged"] = sum(1 for s in attached if s.snap_m > DIRECT_SNAP_M)
    tg.stats["stands_polygon"] = sum(1 for s in out if s.geometry == "polygon")
    tg.stats["stands_point"] = sum(1 for s in out if s.geometry == "point")
    tg.stats["stand_snap_max_m"] = snaps[-1] if snaps else 0.0
    tg.stats["stand_snap_median_m"] = snaps[len(snaps) // 2] if snaps else 0.0
