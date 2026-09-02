# SPDX-License-Identifier: GPL-3.0-only
"""Graph construction, topology repair, and stand attachment.

The fixtures are small hand-built airports, each isolating one OSM defect that
was observed in the real cache. They are deliberately tiny: the point is to pin
one behaviour per test, not to re-measure the real airports, which
``scripts/build_taxi_graph.py`` reports on.
"""

from __future__ import annotations

import networkx as nx
import pytest

from taxiout.geo import geodesic_m
from taxiout.graph import (
    attach_point,
    attach_stands,
    build_graph,
    probe_distance,
    snap_and_split,
)
from taxiout.routing import aggregate, label_runway_edges, route_all
from taxiout.runways import build_runways

REF = (50.0, 8.0)


def way(way_id: int, kind: str, coords, **tags) -> dict:
    return {
        "type": "way",
        "id": way_id,
        "tags": {"aeroway": kind, **tags},
        "geometry": [{"lat": a, "lon": b} for a, b in coords],
    }


def node(node_id: int, kind: str, lat: float, lon: float, **tags) -> dict:
    return {
        "type": "node",
        "id": node_id,
        "lat": lat,
        "lon": lon,
        "tags": {"aeroway": kind, **tags},
    }


# One degree of longitude at 50N is ~71.6 km, so 1e-5 deg is ~0.7 m.
def lon(metres: float) -> float:
    return 8.0 + metres / 71_600.0


def lat(metres: float) -> float:
    return 50.0 + metres / 111_200.0


def test_edges_carry_true_length():
    g = build_graph("TEST", [way(1, "taxiway", [(50.0, lon(0)), (50.0, lon(100))])], REF)
    assert g.stats["edges"] == 1
    length = next(iter(g.g.edges(data=True)))[2]["length_m"]
    assert length == pytest.approx(100.0, rel=0.01)


def test_coincident_nodes_are_merged_into_one_junction():
    """Two ways ending 10 cm apart describe one junction, not two.

    Left unmerged they leave the graph disconnected while the map looks right.
    """
    els = [
        way(1, "taxiway", [(50.0, lon(0)), (50.0, lon(100))]),
        way(2, "taxiway", [(lat(0.1), lon(100)), (lat(100), lon(100))]),
    ]
    g = build_graph("TEST", els, REF)
    assert g.stats["nodes_merged"] >= 1
    assert nx.number_connected_components(g.g) == 1


def test_distinct_junctions_are_not_merged():
    """The merge tolerance must not fuse genuinely separate taxiways."""
    els = [
        way(1, "taxiway", [(50.0, lon(0)), (50.0, lon(100))]),
        way(2, "taxiway", [(lat(20), lon(0)), (lat(20), lon(100))]),
    ]
    g = build_graph("TEST", els, REF)
    assert g.stats["nodes_merged"] == 0
    assert nx.number_connected_components(g.g) == 2


def test_crossing_without_a_shared_node_becomes_a_junction():
    """A taxiway drawn across a runway shares no node with it in OSM."""
    els = [
        way(1, "runway", [(50.0, lon(-500)), (50.0, lon(500))], ref="09/27"),
        way(2, "taxiway", [(lat(-100), lon(0)), (lat(100), lon(0))]),
    ]
    g = build_graph("TEST", els, REF)
    assert g.stats["crossings_split"] >= 1
    assert nx.number_connected_components(g.g) == 1
    # the junction is where the lines actually meet
    assert any(
        abs(d["lat"] - 50.0) < 1e-6 and abs(d["lon"] - lon(0)) < 1e-6
        for _, d in g.g.nodes(data=True)
    )


def test_near_miss_is_not_joined():
    """Taxiways passing within a few metres must stay unconnected."""
    els = [
        way(1, "taxiway", [(50.0, lon(-500)), (50.0, lon(500))]),
        way(2, "taxiway", [(lat(-100), lon(0)), (lat(-5), lon(0))]),
    ]
    g = build_graph("TEST", els, REF)
    assert nx.number_connected_components(g.g) == 2


def test_splitting_preserves_pavement_length():
    els = [
        way(1, "taxiway", [(50.0, lon(-500)), (50.0, lon(500))]),
        way(2, "taxiway", [(lat(-100), lon(0)), (lat(100), lon(0))]),
    ]
    plain = build_graph("TEST", els, REF, split_m=0.0)
    split = build_graph("TEST", els, REF)
    assert split.stats["pavement_km"] == pytest.approx(
        plain.stats["pavement_km"], rel=1e-6
    )


def test_duplicated_pavement_is_deduped_not_lost():
    """Pavement drawn twice collapses once, and the amount is reported."""
    coords = [(50.0, lon(0)), (50.0, lon(100))]
    g = build_graph("TEST", [way(1, "taxiway", coords), way(2, "taxiway", coords)], REF)
    assert g.stats["edges"] == 1
    assert g.stats["pavement_km"] == pytest.approx(0.1, rel=0.01)


def test_runway_label_wins_on_shared_pavement():
    coords = [(50.0, lon(0)), (50.0, lon(100))]
    els = [way(1, "taxiway", coords), way(2, "runway", coords, ref="09/27")]
    g = build_graph("TEST", els, REF)
    assert next(iter(g.g.edges(data=True)))[2]["kind"] == "runway"


def test_stand_attaches_at_its_entry_end_and_keeps_the_lead_in():
    els = [
        way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))]),
        way(2, "parking_position", [(lat(-80), lon(0)), (50.0, lon(0))], ref="A1"),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    (stand,) = g.stands
    assert stand.stand_id == "A1"
    assert stand.geometry == "line"
    assert stand.node is not None
    assert stand.snap_m == pytest.approx(0.0, abs=0.6)
    assert stand.lead_in_m == pytest.approx(80.0, rel=0.02)
    # the parked end is the far one, not the one that meets the taxiway
    assert geodesic_m(stand.stop, (50.0, lon(0))) == pytest.approx(80.0, rel=0.02)


def test_stand_entry_end_is_chosen_by_measurement_not_drawing_order():
    forward = [(lat(-80), lon(0)), (50.0, lon(0))]
    els_a = [
        way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))]),
        way(2, "parking_position", forward, ref="A1"),
    ]
    els_b = [
        way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))]),
        way(2, "parking_position", forward[::-1], ref="A1"),
    ]
    a = build_graph("TEST", els_a, REF)
    attach_stands(a, els_a)
    b = build_graph("TEST", els_b, REF)
    attach_stands(b, els_b)
    assert a.stands[0].stop == b.stands[0].stop


def test_polygon_stand_is_reduced_to_its_centroid():
    """Antalya draws 168 stands as footprints; a perimeter is not a lead-in."""
    ring = [
        (lat(-60), lon(-20)),
        (lat(-60), lon(20)),
        (lat(-20), lon(20)),
        (lat(-20), lon(-20)),
        (lat(-60), lon(-20)),
    ]
    els = [
        way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))]),
        way(2, "parking_position", ring, ref="106"),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    (stand,) = g.stands
    assert stand.geometry == "polygon"
    assert stand.lead_in_m == 0.0
    assert stand.stop == stand.entry
    assert stand.stop[0] == pytest.approx(lat(-40), abs=1e-5)


def test_node_stand_attaches_and_records_its_gap():
    els = [
        way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))]),
        node(2, "parking_position", lat(-100), lon(0), ref="Z9"),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    (stand,) = g.stands
    assert stand.geometry == "point"
    assert stand.node is not None
    assert stand.snap_m == pytest.approx(100.0, rel=0.02)


def test_distant_stand_is_left_unattached_rather_than_dragged_on():
    els = [
        way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))]),
        node(2, "parking_position", lat(-400), lon(0), ref="FAR"),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    assert g.stands[0].node is None
    assert g.stats["stands_attached"] == 0


def test_probe_does_not_mutate_the_graph():
    els = [way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))])]
    g = build_graph("TEST", els, REF)
    before = (g.g.number_of_nodes(), g.g.number_of_edges())
    probe_distance(g, (lat(-50), lon(0)))
    assert (g.g.number_of_nodes(), g.g.number_of_edges()) == before


def test_attach_point_splits_and_conserves_length():
    els = [way(1, "taxiway", [(50.0, lon(-200)), (50.0, lon(200))])]
    g = build_graph("TEST", els, REF)
    before = sum(d["length_m"] for *_, d in g.g.edges(data=True))
    n, snap = attach_point(g, (lat(-50), lon(0)))
    assert n is not None
    assert snap == pytest.approx(50.0, rel=0.02)
    after = sum(d["length_m"] for *_, d in g.g.edges(data=True))
    assert after == pytest.approx(before, rel=1e-6)


def test_end_to_end_route_has_credible_parts():
    """A stand 80 m off a taxiway, 500 m from the 27 threshold."""
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1000))], ref="09/27"),
        way(2, "taxiway", [(lat(-50), lon(0)), (lat(-50), lon(1000))]),
        way(3, "taxiway", [(lat(-50), lon(1000)), (50.0, lon(1000))]),
        way(
            4, "parking_position", [(lat(-130), lon(200)), (lat(-50), lon(200))], ref="A1"
        ),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    routes, entries, stats = route_all(g, runways)

    assert stats["routable_share"] == 1.0
    by_designator = {r.designator: r for r in routes}
    assert set(by_designator) == {"09", "27"}

    to_27 = by_designator["27"]
    # 80 m lead-in, 800 m along the parallel taxiway, 50 m up to the threshold
    assert to_27.lead_in_m == pytest.approx(80.0, rel=0.02)
    assert to_27.taxi_m == pytest.approx(930.0, rel=0.05)
    assert to_27.detour_ratio >= 1.0
    assert to_27.backtrack_m == pytest.approx(0.0, abs=1.0)

    # 09 is at the far end, so its route is longer
    assert by_designator["09"].taxi_m > to_27.taxi_m


def test_no_route_is_ever_shorter_than_the_straight_line():
    """The cheapest global sanity check there is, asserted rather than eyeballed."""
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1000))], ref="09/27"),
        way(2, "taxiway", [(lat(-50), lon(0)), (lat(-50), lon(1000))]),
        way(3, "taxiway", [(lat(-50), lon(1000)), (50.0, lon(1000))]),
        way(
            4, "parking_position", [(lat(-130), lon(200)), (lat(-50), lon(200))], ref="A1"
        ),
        way(
            5, "parking_position", [(lat(-130), lon(700)), (lat(-50), lon(700))], ref="A2"
        ),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    routes, _, _ = route_all(g, runways)
    assert routes
    for r in routes:
        assert r.taxi_m >= r.gc_m - 1e-6


def test_a_route_crossing_a_runway_is_counted_as_a_crossing():
    """The regression that mattered most.

    Repairing the crossing inserts a junction node, so the route crosses via
    that node and traverses no runway edge. Counting foreign runway *edges*
    therefore reported zero crossings for every route at CDG -- which reads as a
    finding about the airport rather than as a bug.
    """
    els = [
        way(1, "runway", [(50.0, lon(-800)), (50.0, lon(800))], ref="09/27"),
        way(2, "runway", [(lat(-700), lon(400)), (lat(700), lon(400))], ref="18/36"),
        # stand south of 09/27, reached by a taxiway that crosses 18/36
        way(3, "taxiway", [(lat(-200), lon(-600)), (lat(-200), lon(700))]),
        way(4, "taxiway", [(lat(-200), lon(-600)), (50.0, lon(-800))]),
        way(
            5,
            "parking_position",
            [(lat(-280), lon(600)), (lat(-200), lon(600))],
            ref="A1",
        ),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    routes, _, _ = route_all(g, runways)
    to_09 = next(r for r in routes if r.designator == "09")
    assert to_09.runway_crossings >= 1
    # and crossing costs no runway distance, because it goes through a junction
    assert to_09.cross_runway_m == pytest.approx(0.0, abs=1.0)


def test_a_route_that_never_meets_another_runway_counts_none():
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1000))], ref="09/27"),
        way(2, "taxiway", [(lat(-50), lon(0)), (lat(-50), lon(1000))]),
        way(3, "taxiway", [(lat(-50), lon(1000)), (50.0, lon(1000))]),
        way(
            4, "parking_position", [(lat(-130), lon(200)), (lat(-50), lon(200))], ref="A1"
        ),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    routes, _, _ = route_all(g, runways)
    assert all(r.runway_crossings == 0 for r in routes)


def test_aggregated_rows_stay_internally_consistent():
    """Every matrix row must be a real route, not an average of two.

    Median-per-column broke the documented identity ``taxi_m = lead_in + snap +
    route + entry_offset`` in a fifth of Istanbul's cells, and turned integer
    counts into halves.
    """
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1200))], ref="09/27"),
        way(2, "taxiway", [(lat(-50), lon(0)), (lat(-50), lon(1200))]),
        way(3, "taxiway", [(lat(-50), lon(1200)), (50.0, lon(1200))]),
        # same identifier drawn three times, in different places
        way(
            4, "parking_position", [(lat(-130), lon(200)), (lat(-50), lon(200))], ref="A1"
        ),
        way(
            5, "parking_position", [(lat(-140), lon(500)), (lat(-50), lon(500))], ref="A1"
        ),
        way(
            6, "parking_position", [(lat(-120), lon(900)), (lat(-50), lon(900))], ref="A1"
        ),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    routes, _, _ = route_all(g, runways)
    cells = aggregate(routes)

    assert cells
    for c in cells:
        v = c.values
        assert c.n_features == 3
        assert v["taxi_m"] == pytest.approx(
            v["lead_in_m"] + v["snap_m"] + v["route_m"] + v["entry_offset_m"],
            abs=1e-6,
        )
        assert v["detour_ratio"] == pytest.approx(v["taxi_m"] / v["gc_m"], rel=1e-9)
        for count in ("sharp_turns", "junctions", "runway_crossings"):
            assert float(v[count]).is_integer()
        assert c.spread_m > 0


def test_aggregate_reports_the_worst_snap_not_the_middle_one():
    """A stand whose duplicates include a snapped feature must show it."""
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1200))], ref="09/27"),
        way(2, "taxiway", [(lat(-50), lon(0)), (lat(-50), lon(1200))]),
        way(3, "taxiway", [(lat(-50), lon(1200)), (50.0, lon(1200))]),
        way(
            4, "parking_position", [(lat(-130), lon(200)), (lat(-50), lon(200))], ref="A1"
        ),
        # 100 m clear of the taxiway: attached, but across a gap
        node(5, "parking_position", lat(-150), lon(600), ref="A1"),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    cells = aggregate(route_all(g, runways)[0])
    assert cells
    assert max(c.snap_max_m for c in cells) > 60.0


def test_runway_penalty_discourages_taxiing_along_a_runway():
    """With a free runway the router uses it; penalised, it goes round."""
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1000))], ref="09/27"),
        # a slightly longer parallel taxiway joining both runway ends
        way(
            2,
            "taxiway",
            [
                (50.0, lon(0)),
                (lat(-60), lon(0)),
                (lat(-60), lon(1000)),
                (50.0, lon(1000)),
            ],
        ),
        way(3, "parking_position", [(lat(-120), lon(0)), (lat(-60), lon(0))], ref="A1"),
    ]
    runways, _ = build_runways("TEST", els)

    free = build_graph("TEST", els, REF)
    attach_stands(free, els)
    cheap, _, _ = route_all(free, runways, runway_penalty=1.0)

    strict = build_graph("TEST", els, REF)
    attach_stands(strict, els)
    dear, _, _ = route_all(strict, runways, runway_penalty=20.0)

    cheap_27 = next(r for r in cheap if r.designator == "27")
    dear_27 = next(r for r in dear if r.designator == "27")
    assert cheap_27.backtrack_m > dear_27.backtrack_m


def test_topology_repair_reaches_a_fixed_point():
    """It has to terminate, and it has to say whether it did.

    The first version had no minimum-length guard: it split off millimetre
    half-edges whose midpoints sat on their neighbours, and settled into a
    cycle that ran forever. The output then depended on the round cap rather
    than on the airport.
    """
    els = [
        way(1, "taxiway", [(50.0, lon(-500)), (50.0, lon(500))]),
        way(2, "taxiway", [(lat(-100), lon(0)), (lat(100), lon(0))]),
        way(3, "taxiway", [(lat(-100), lon(120)), (lat(100), lon(118))]),
        way(4, "taxiway", [(lat(1.2), lon(-300)), (lat(1.2), lon(300))]),
    ]
    g = build_graph("TEST", els, REF)
    assert g.stats["repair_converged"] is True
    # and a second pass from scratch finds nothing left to do
    again, _ = snap_and_split(g)
    assert again == 0


def test_split_positions_are_bounded_away_from_the_ends():
    """No edge shorter than the merge tolerance may be created by splitting."""
    els = [
        way(1, "taxiway", [(50.0, lon(-500)), (50.0, lon(500))]),
        way(2, "taxiway", [(lat(-80), lon(0)), (lat(80), lon(0))]),
        way(3, "taxiway", [(lat(-80), lon(0.2)), (lat(80), lon(0.2))]),
    ]
    g = build_graph("TEST", els, REF)
    short = [d["length_m"] for *_, d in g.g.edges(data=True) if d["length_m"] < 0.5]
    assert not short, f"created {len(short)} sub-tolerance edges: {short}"


def test_repair_is_independent_of_way_read_order():
    base = [
        way(1, "taxiway", [(50.0, lon(-400)), (50.0, lon(400))]),
        way(2, "taxiway", [(lat(-90), lon(-100)), (lat(90), lon(-100))]),
        way(3, "taxiway", [(lat(-90), lon(150)), (lat(90), lon(150))]),
        way(4, "runway", [(lat(60), lon(-400)), (lat(60), lon(400))], ref="09/27"),
    ]
    a = build_graph("TEST", base, REF)
    b = build_graph("TEST", list(reversed(base)), REF)
    assert a.stats["pavement_km"] == pytest.approx(b.stats["pavement_km"], rel=1e-9)
    assert a.g.number_of_edges() == b.g.number_of_edges()


def test_exact_coordinate_sharing_is_not_counted_as_a_repair():
    """Two ways meeting at the same coordinate is the format working."""
    els = [
        way(1, "taxiway", [(50.0, lon(0)), (50.0, lon(100))]),
        way(2, "taxiway", [(50.0, lon(100)), (lat(100), lon(100))]),
    ]
    g = build_graph("TEST", els, REF)
    assert g.stats["nodes_merged"] == 0
    assert nx.number_connected_components(g.g) == 1


def test_an_untagged_runway_way_cannot_clear_a_ref():
    coords = [(50.0, lon(0)), (50.0, lon(100))]
    els = [
        way(1, "runway", coords, ref="09/27"),
        way(2, "runway", coords),  # same pavement, no ref
    ]
    g = build_graph("TEST", els, REF)
    assert next(iter(g.g.edges(data=True)))[2]["ref"] == "09/27"


def test_displaced_threshold_pavement_is_labelled_in_the_graph():
    """Otherwise it is invisible to crossing detection and books as backtrack."""
    els = [
        way(1, "runway", [(50.0, lon(300)), (50.0, lon(1500))], ref="09/27"),
        {
            "type": "way",
            "id": 2,
            "tags": {"aeroway": "runway", "runway": "displaced_threshold"},
            "geometry": [
                {"lat": 50.0, "lon": lon(0)},
                {"lat": 50.0, "lon": lon(300)},
            ],
        },
    ]
    g = build_graph("TEST", els, REF)
    runways, _ = build_runways("TEST", els)
    unlabelled = [
        d
        for *_, d in g.g.edges(data=True)
        if d.get("kind") == "runway" and not d.get("ref")
    ]
    assert unlabelled, "fixture should start with unlabelled runway pavement"
    assert label_runway_edges(g, runways) == len(unlabelled)
    assert all(
        d.get("ref") == "09/27"
        for *_, d in g.g.edges(data=True)
        if d.get("kind") == "runway"
    )


def test_aggregate_representative_is_not_biased_short():
    """With two features the lower-middle pick always took the shorter route."""
    els = [
        way(1, "runway", [(50.0, lon(0)), (50.0, lon(1400))], ref="09/27"),
        way(2, "taxiway", [(lat(-50), lon(0)), (lat(-50), lon(1400))]),
        way(3, "taxiway", [(lat(-50), lon(1400)), (50.0, lon(1400))]),
        way(
            4, "parking_position", [(lat(-130), lon(200)), (lat(-50), lon(200))], ref="A1"
        ),
        way(
            5,
            "parking_position",
            [(lat(-130), lon(1000)), (lat(-50), lon(1000))],
            ref="A1",
        ),
    ]
    g = build_graph("TEST", els, REF)
    attach_stands(g, els)
    runways, _ = build_runways("TEST", els)
    routes, _, _ = route_all(g, runways)
    cells = aggregate(routes)
    for c in cells:
        group = [r for r in routes if r.designator == c.designator and r.stand_id == "A1"]
        mean = sum(r.taxi_m for r in group) / len(group)
        nearest = min(abs(r.taxi_m - mean) for r in group)
        assert abs(c.values["taxi_m"] - mean) == pytest.approx(nearest, abs=1e-6)
