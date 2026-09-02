#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Run the pipeline on the worked airport and write the report and figure.

``taxiout.synthetic`` draws a field on which every distance, turn, junction
count and repair is known before the pipeline runs. This script runs it,
prints the hand derivation beside the pipeline's output for every cell, and
draws the field. A reader can check the arithmetic on paper; the tests in
``tests/test_synthetic_airport.py`` check it on every run.

Writes ``reports/synthetic_airport.md`` and ``reports/synthetic_airport.svg``.
Needs no network, no OpenStreetMap cache and no challenge data; it is the
one geometry report that runs on a clean checkout.

Usage
-----
    python scripts/report_synthetic_airport.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout import synthetic as syn  # noqa: E402
from taxiout.routing import RUNWAY_PENALTY  # noqa: E402

OUT_MD = REPO / "reports" / "synthetic_airport.md"
OUT_SVG = REPO / "reports" / "synthetic_airport.svg"

INTEGER_FIELDS = ("sharp_turns", "junctions", "runway_crossings")


# --------------------------------------------------------------------------
# helpers


def num(x: float, digits: int = 1) -> str:
    return f"{x:,.{digits}f}"


def table(
    header: list[str], rows: list[list[str]], align: str | None = None
) -> list[str]:
    align = align or ("-" * len(header))
    sep = ["---:" if a == "r" else "---" for a in align]
    return [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def cell_key(k: tuple[str, str]) -> tuple[int, str]:
    return int(k[0]), k[1]


# --------------------------------------------------------------------------
# sections


def section_repairs(built: syn.Built) -> list[str]:
    stats = {**built.tg.stats, **built.route_stats, "cells": len(built.cells)}
    want = syn.expected_stats()
    rows = []
    labels = {
        "ways": "routable ways read",
        "nodes_merged": "coincident nodes merged (repair 1)",
        "edges_split": "edges split at a node lying on them (repair 2)",
        "crossings_split": "junctions inserted at crossings (repair 3)",
        "components_before_split": "connected components before repair",
        "components_after_split": "connected components after repair",
        "repair_rounds": "repair rounds to the fixed point",
        "pavement_km": "pavement, km",
        "stands_total": "parking positions",
        "stands_attached": "attached to the graph",
        "stands_direct": "attached at 60 m or less",
        "stands_bridged": "attached across a larger gap",
        "stands_polygon": "drawn as a footprint polygon",
        "stands_point": "drawn as a bare node",
        "stand_snap_max_m": "largest attachment gap, m",
        "designators": "runway designators",
        "pairs_routed": "(feature, designator) routes",
        "cells": "(stand, designator) matrix cells",
    }
    for key, label in labels.items():
        w, g = want[key], stats[key]
        if isinstance(w, float):
            rows.append([label, num(w, 4), num(g, 4)])
        else:
            rows.append([label, str(w), str(g)])
    bt = stats["backtracking_designators"]
    rows.append(
        [
            "designators flagged for backtracking (median, spread)",
            "36 (900.0, 0.0)",
            "; ".join(
                f"{d} ({num(v['median_m'])}, {num(v['spread_m'])})" for d, v in bt.items()
            ),
        ]
    )
    return [
        "## What fires",
        "",
        "Each repair the graph builder performs was planted once. The build",
        "reports what it did; the left column is what was planted.",
        "",
        *table(["quantity", "planted", "reported"], rows, "-rr"),
    ]


def section_stands(built: syn.Built) -> list[str]:
    rows = []
    for s in sorted(built.tg.stands, key=lambda s: (int(s.stand_id), s.osm_id)):
        rows.append(
            [
                s.stand_id,
                s.id_tag,
                s.geometry,
                num(s.lead_in_m),
                "--" if s.node is None else num(s.snap_m, 2),
                "yes" if s.node is not None else "**no** (350 m from pavement)",
            ]
        )
    return [
        "## The stands",
        "",
        "How each parking position resolved onto the graph. Stand 2 is drawn",
        "three times; stand 4 is drawn stop-to-entry and the entry end is",
        "chosen by measuring, so it reads like the others; stand 9 is a bare",
        "node 120 m from the taxilane and stand 10 a footprint polygon, both",
        "attached by a straight line whose length is recorded; stand 11 is too",
        "far from any pavement and is reported rather than dragged on.",
        "",
        *table(
            ["stand", "identified by", "drawn as", "lead-in m", "snap m", "attached"],
            rows,
            "---rr-",
        ),
    ]


def section_matrix(built: syn.Built) -> tuple[list[str], dict]:
    want = syn.expected_cells()
    got = {(c.stand_id, c.designator): c for c in built.cells}
    dist_rows, desc_rows = [], []
    worst_m, worst_deg = 0.0, 0.0
    for key in sorted(want, key=cell_key):
        w, g = want[key], got[key].values
        d_taxi = g["taxi_m"] - w["taxi_m"]
        d_gc = g["gc_m"] - w["gc_m"]
        worst_m = max(worst_m, abs(d_taxi), abs(d_gc), abs(g["route_m"] - w["route_m"]))
        worst_deg = max(worst_deg, abs(g["turn_deg"] - w["turn_deg"]))
        dist_rows.append(
            [
                f"{key[0]} to {key[1]}",
                num(w["lead_in_m"]),
                num(w["snap_m"]),
                num(w["route_m"], 4),
                num(g["route_m"], 4),
                num(w["taxi_m"], 4),
                num(g["taxi_m"], 4),
                f"{1000 * d_taxi:+.1f}",
                num(w["gc_m"], 3),
                num(g["gc_m"], 3),
                num(g["detour_ratio"], 3),
            ]
        )
        ints_ok = all(int(g[f]) == int(w[f]) for f in INTEGER_FIELDS)
        desc_rows.append(
            [
                f"{key[0]} to {key[1]}",
                num(w["turn_deg"]),
                num(g["turn_deg"], 2),
                str(int(w["sharp_turns"])),
                str(int(w["junctions"])),
                num(w["straight_m"]),
                num(g["straight_m"]),
                num(w["backtrack_m"]),
                num(g["backtrack_m"]),
                num(w["cross_runway_m"]),
                num(g["cross_runway_m"]),
                str(int(w["runway_crossings"])),
                "yes" if ints_ok else "**no**",
            ]
        )
    facts = {"worst_m": worst_m, "worst_deg": worst_deg, "cells": len(want)}
    lines = [
        "## The matrix, both ways",
        "",
        f"All {len(want)} cells. *Hand* is the derivation in `taxiout.synthetic`;",
        "*pipeline* is what `route_all` and `aggregate` produced. Distances in",
        "meters; the difference column is in millimeters.",
        "",
        *table(
            [
                "cell",
                "lead-in",
                "snap",
                "route, hand",
                "route, pipeline",
                "taxi, hand",
                "taxi, pipeline",
                "diff mm",
                "straight line, hand",
                "pipeline",
                "detour",
            ],
            dist_rows,
            "-rrrrrrrrrr",
        ),
        "",
        f"The largest distance discrepancy over every cell is {num(1000 * worst_m, 1)} mm, on",
        "a diagonal straight-line distance: the ellipsoid's second-order term,",
        "which the tangent-plane derivation does not carry. Along the axes the",
        "two agree to a tenth of a millimeter.",
        "",
        "Route shape. Turns are right angles by construction; the pipeline's",
        "totals differ from multiples of ninety by the tenth-of-a-degree the",
        "two planted defects put into two corners. Integer descriptors -- sharp",
        "turns, junctions, runway crossings -- must match exactly, and the last",
        "column says whether all three did.",
        "",
        *table(
            [
                "cell",
                "turn, hand",
                "pipeline",
                "sharp",
                "junctions",
                "junction-free m, hand",
                "pipeline",
                "backtrack, hand",
                "pipeline",
                "other runway, hand",
                "pipeline",
                "crossings",
                "integers match",
            ],
            desc_rows,
            "-rrrrrrrrrrr-",
        ),
        "",
        f"The largest turn discrepancy is {num(worst_deg, 2)} degrees.",
    ]
    return lines, facts


def section_worked(built: syn.Built) -> list[str]:
    got = {(c.stand_id, c.designator): c.values for c in built.cells}
    lines = [
        "## Four cells, worked",
        "",
        "The same arithmetic as the table, spelled out, with the pipeline's",
        "figures after each.",
        "",
    ]
    for stand_id, designator, text in syn.WORKED:
        g = got[(stand_id, designator)]
        lines += [
            f"**Stand {stand_id} to {designator}.** {text}",
            "",
            f"Pipeline: route {num(g['route_m'], 4)} m, taxi {num(g['taxi_m'], 4)} m,",
            f"turn {num(g['turn_deg'], 2)} deg, {int(g['sharp_turns'])} sharp,",
            f"{int(g['junctions'])} junctions, junction-free {num(g['straight_m'])} m,",
            f"backtrack {num(g['backtrack_m'])} m, other runway {num(g['cross_runway_m'])} m,",
            f"{int(g['runway_crossings'])} crossings.",
            "",
        ]
    return lines


def section_penalty() -> list[str]:
    x = syn.penalty_crossovers()
    probes = sorted(
        {1.0}
        | {round(v - 0.005, 4) for v in x.values()}
        | {round(v + 0.005, 4) for v in x.values()}
        | {RUNWAY_PENALTY}
    )
    rows = []
    for p in probes:
        b = syn.build(runway_penalty=p)
        bt = {(c.stand_id, c.designator): c.values["backtrack_m"] for c in b.cells}
        rows.append(
            [
                num(p, 4),
                num(bt[("12", "09")]),
                num(bt[("12", "27")]),
                num(bt[("1", "27")]),
            ]
        )
    return [
        "## The runway penalty, on a field where the answer is known",
        "",
        "Runway pavement is traversable at a multiple of its length. On the",
        "real airports the multiple was chosen by a sweep (`penalty_sensitivity.md`)",
        "and defended as measured rather than tuned. Here the crossover is",
        "arithmetic. A south stand at (1500, 0) departing 27 can run 900 m",
        "east along the runway or taxi 1,199.2 m round by A; the runway wins",
        f"while `900 p < 1199.2`, so below p = {num(x['south 27'], 4)}. Departing 09 it",
        f"is 1,500 m against 1,799.2 m, so below p = {num(x['south 09'], 4)}. A north",
        "stand departing 27 has the same choice at the 1500 link -- 149.2 + 900 p",
        f"against 1,050.0004 -- so below p = {num(x['north 27'], 4)}; and below",
        f"p = {num(x['north 27 via A2'], 4)} it is cheaper still to drop to the runway",
        "at x = 600 and run 1,800 m along it. A penalty of exactly one sends",
        "every stand on the field down a runway. `backtrack_m` at penalties",
        "bracketing each crossover:",
        "",
        *table(
            ["penalty", "stand 12 to 09", "stand 12 to 27", "stand 1 to 27"],
            rows,
            "rrrr",
        ),
        "",
        f"At the default penalty of {num(RUNWAY_PENALTY, 0)} no stand backtracks on 09 or 27; on 36",
        "every stand backtracks 900 m, because no taxiway reaches that",
        "threshold, and the routing report's per-designator flag reports it",
        "with a spread of zero -- the same offset for every stand, which a",
        "model that knows the runway absorbs.",
    ]


def section_findings() -> list[str]:
    shift = syn.probe_merge_shift()
    ties = [
        syn.probe_two_feature_tie(p) for p in ((70.0, 75.0), (70.0, 80.0), (70.0, 76.0))
    ]
    tie_rows = []
    for (a, b), t in zip(((70.0, 75.0), (70.0, 80.0), (70.0, 76.0)), ties):
        tie_rows.append(
            [
                f"{num(a, 0)} and {num(b, 0)}",
                "equal" if t["gaps_equal"] else f"differ by {t['gap_difference_m']:.1e}",
                num(t["tie_break_lead_in_m"], 0),
                num(t["kept_lead_in_m"], 0),
            ]
        )
    # Both findings are stated from the measurement, so the report stays
    # true after the fixes land: the probes then show nothing missing and
    # the tie-break's choice kept in every row.
    shift_fixed = abs(shift["missing_m"]) < 0.001
    tie_fixed = all(
        abs(t["kept_lead_in_m"] - t["tie_break_lead_in_m"]) < 1e-6 for t in ties
    )
    shift_state = (
        "The build now keeps the canonical node's coordinates: the probe reports "
        "nothing missing."
        if shift_fixed
        else "The fix is one line -- keep the canonical node's coordinates -- and "
        "belongs to `graph.py`."
    )
    tie_state = (
        "The pipeline now compares the gaps at a resolution the tie-break can see, "
        "and keeps the tie-break's choice in every row above."
        if tie_fixed
        else "The fix -- compare gaps rounded to the millimeter before the tie-break -- "
        "belongs to `routing.py`."
    )
    return [
        "## What the field found",
        "",
        "Three things the first draft of this field turned up in the pipeline,",
        "none visible on a real airport because none exceeds the tolerances",
        "the pipeline works to. They are recorded because a worked example",
        "that found nothing would be a weaker argument than one that did.",
        "",
        "**Overshoot and undershoot are different repairs.** The node-on-edge",
        "defect was first drawn as a link ending 0.8 m *beyond* taxiway A. That",
        "is a crossing -- two segments that properly intersect -- and the",
        "crossing repair took it, inserting a junction at the intersection and",
        "leaving a 0.8 m stub. Drawn 0.8 m *short*, it is a node near an edge",
        "and the split repair takes it. Both are correct; the field's",
        "defects had to be drawn as the specific defect intended.",
        "",
        "**A merged node moves to the last way that referenced it.** The graph",
        "builder adds each way's nodes with that way's coordinates, and an",
        "existing node is updated rather than kept, so a node merged across a",
        "0.2 m gap ends up wherever the last way put it. Edge lengths are taken",
        "from the way as drawn, but an edge split later is re-measured from",
        "node positions -- so a split on an edge at a shifted node books the",
        f"shift once. Measured on a three-way probe: {num(shift['drawn_m'], 4)} m drawn,",
        f"{num(shift['graph_m'], 4)} m in the graph, {num(1000 * shift['missing_m'], 0)} mm missing.",
        "Within the 0.5 m merge tolerance by construction, so no route on a",
        "real airport is wrong by more than that; the field keeps its merge",
        f"and its split on different edges so the tables above stay exact. {shift_state}",
        "",
        "**Two features sharing a stand are a coin toss.** `aggregate` keeps the",
        "feature nearest the group's mean taxi distance, lower OSM id on a tie.",
        "For two features the mean is their midpoint, so the two gaps are equal",
        "in exact arithmetic and the tie-break should decide; in floating point",
        "they can differ in the last bit and rounding decide instead. Measured on a",
        "two-feature probe:",
        "",
        *table(
            ["lead-ins, m", "gaps to the mean", "tie-break would keep", "pipeline kept"],
            tie_rows,
            "-r-r",
        ),
        "",
        "Two features is the commonest case on the real airports. The field",
        f"draws stand 2 three times so the rule's choice is unique. {tie_state}",
    ]


def section_traffic() -> list[str]:
    """A year of movements on the field, and what the term recovers from them."""
    import numpy as np

    from taxiout.tails import TailModel, combine

    t = syn.traffic(seed=0, per_day=40)
    train = t.training[t.training["PHASE_mvt"] == "DEP"]
    rk = t.ranking[t.ranking["PHASE_mvt"] == "DEP"].reset_index(drop=True)
    truth = t.truth.set_index("MVT_ID_mvt").loc[rk["MVT_ID_mvt"]]
    y = truth["recorded_s"].to_numpy(dtype=float)
    tm = TailModel().fit(train)
    dist = {(r.stand_id, r.designator): r.taxi_m for r in t.matrix.itertuples()}
    known = np.array(
        [
            syn.PUSHBACK_S + dist[(s, r)] / syn.SPEED_MPS + syn.SERVICE_S
            if (s, r) in dist
            else np.nan
            for s, r in zip(rk["STAND_mvt"], rk["RUNWAY_mvt"])
        ]
    )
    fallback = float(train.loc[~tm.flagged(train), "TAXITIME_SEC_mvt"].mean())
    known = np.where(np.isnan(known), fallback, known)
    with_term = combine(known, tm.predict(rk)).to_numpy()

    def rmse(p):
        return float(np.sqrt(np.mean((y - p) ** 2)))

    no_record = train["LOBT_flt"].isna()
    lag = (train["MVT_TIME_UTC_mvt"] - train["SCHED_TIME_UTC_mvt"]).dt.total_seconds()
    rows = [
        ["departures, 2025", f"{len(train):,}", "--"],
        ["departures, January and July 2026", f"{len(rk):,}", "--"],
        [
            "no flight-plan record",
            num(100 * no_record.mean(), 2) + "%",
            num(100 * syn.P_NO_RECORD, 2) + "%",
        ],
        [
            "defaulted block time, of those",
            num(100 * t.truth["defaulted"].sum() / max(1, t.truth["no_record"].sum()), 1)
            + "%",
            num(100 * syn.P_DEFAULTED_IF_NO_RECORD, 1) + "%",
        ],
        [
            "recorded taxi-out above two hours",
            num(100 * (train["TAXITIME_SEC_mvt"] > 7200).mean(), 2) + "%",
            "--",
        ],
        [
            "take-off more than four hours after schedule",
            num(100 * (lag > 4 * 3600).mean(), 2) + "%",
            "--",
        ],
    ]
    cells = []
    for (icao, bucket), r in tm.cells.iterrows():
        cells.append(
            [
                f"`{bucket}`",
                f"{int(r['n']):,}",
                num(100 * r["share"], 1) + "%",
                num(r["mean"], 0),
                "--" if np.isnan(r["rest_mean"]) else num(r["rest_mean"], 0),
                "yes" if r["covered"] else "no",
            ]
        )
    keep = np.isnan(tm.predict(rk).to_numpy())
    scores = [
        ["airport mean", num(rmse(np.full_like(y, fallback)), 1)],
        ["the process, without the term", num(rmse(known), 1)],
        ["the process, with the term", num(rmse(with_term), 1)],
        [
            "the process, on the rows the term leaves alone",
            num(float(np.sqrt(np.mean((y[keep] - known[keep]) ** 2))), 1),
        ],
    ]
    return [
        "## Worked traffic",
        "",
        "The challenge data cannot leave the machine it is on, so nothing that",
        "reads it runs in continuous integration and nothing that reads it can",
        "be checked against a known answer. `taxiout.synthetic.traffic` generates",
        "a year of movements on this field in the challenge's own schema and",
        "file layout, from a process every parameter of which is stated in the",
        "module: an off-block delay, an unimpeded taxi at",
        f"{num(syn.SPEED_MPS, 1)} m/s over the field's routed distance plus a",
        f"{num(syn.PUSHBACK_S, 0)} s push, a first-in-first-out runway queue with",
        f"{num(syn.SERVICE_S, 0)} s of occupancy per departure, {num(syn.NOISE_SD_S, 0)} s of noise, and a",
        "feed that fails the flight-plan join for a stated share of departures",
        "and defaults the block time to the schedule for a stated share of",
        "those -- the mechanism `reports/tails.md` found at one real airport.",
        "The ranking months are blanked exactly as the challenge blanks them,",
        "and the target's components are kept beside them. `taxiout.challenge`",
        "loads it unchanged; `tests/test_synthetic_airport.py` runs the",
        "record-quality term, the submission writer and the loaders on it end",
        "to end. It is a fixture, not a simulation of any airport.",
        "",
        *table(["quantity", "generated", "planted"], rows, "-rr"),
        "",
        "What the record-quality term measures on it, fitted on 2025:",
        "",
        *table(
            ["cell", "rows", "defaulted share", "mean s", "rest mean s", "covered"],
            cells,
            "-rrrr-",
        ),
        "",
        "And what a model that knows the process scores on the 2026 rows,",
        "against the recorded values -- the same comparison `reports/tails.md`",
        "makes with a stand-in on the real data, here with the answer known:",
        "",
        *table(["prediction", "RMSE s"], scores, "-r"),
        "",
        "The gap between the second and third rows is the scoring rule",
        "counting the defaulted rows; the fourth row is the process's own noise",
        "and queue variance, which is what a physical model is answerable for.",
    ]


def section_limits() -> list[str]:
    return [
        "## What the field does not exercise",
        "",
        "Displaced-threshold ways re-attached to a runway; a runway drawn as",
        "several OSM ways; curved taxiways, and with them the simplification",
        "tolerance that turn angles depend on; stands whose lead-in is mapped",
        "as a taxilane rather than a parking-position way; apron polygons.",
        "Each has a unit test on a fixture of its own in `tests/test_graph.py`",
        "and `tests/test_runways.py`; this field is the end-to-end case, not",
        "the union of every case.",
    ]


# --------------------------------------------------------------------------
# the figure


def _route_path(
    built: syn.Built, stand_id: str, designator: str, penalty: float
) -> list[int]:
    g = built.tg.g
    entry = next(e for e in built.entries if e.designator == designator)
    stand = next(
        s for s in built.tg.stands if s.stand_id == stand_id and s.node is not None
    )

    def weight(_u, _v, d):
        return d["length_m"] * (penalty if d.get("kind") == "runway" else 1.0)

    return nx.shortest_path(g, stand.node, entry.node, weight=weight)


def draw_svg(built: syn.Built) -> str:
    frame = syn.FRAME
    g = built.tg.g
    # Field extents in meters, then a scale to pixels with y flipped.
    x0, x1, y0, y1 = -150.0, 2550.0, -1000.0, 780.0
    scale = 0.3
    width, height = (x1 - x0) * scale, (y1 - y0) * scale

    def px(x: float, y: float) -> tuple[float, float]:
        return (x - x0) * scale, (y1 - y) * scale

    def line(a, b, **attrs) -> str:
        (ax, ay), (bx, by) = px(*a), px(*b)
        style = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return f'<line x1="{ax:.1f}" y1="{ay:.1f}" x2="{bx:.1f}" y2="{by:.1f}" {style}/>'

    def text(x, y, s, **attrs) -> str:
        tx, ty = px(x, y)
        style = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
        return f'<text x="{tx:.1f}" y="{ty:.1f}" {style}>{s}</text>'

    def node_xy(n: int) -> tuple[float, float]:
        return frame.to_xy(built.tg.point(n))

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'width="{width:.0f}" height="{height:.0f}" font-family="sans-serif" font-size="11">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        f"<title>{syn.ICAO}: the worked airport</title>",
    ]
    # Pavement from the graph, so the figure shows what was routed on.
    for u, v, d in g.edges(data=True):
        a, b = node_xy(u), node_xy(v)
        if d.get("kind") == "runway":
            out.append(
                line(a, b, stroke="#9a9a9a", stroke_width=12, stroke_linecap="butt")
            )
        elif d.get("kind") == "taxilane":
            out.append(line(a, b, stroke="#5c8fbf", stroke_width=3))
        else:
            out.append(line(a, b, stroke="#3b6ea5", stroke_width=4))
    # Two routes, drawn from the graph at the default penalty.
    for stand_id, designator, color in (("1", "09", "#c0392b"), ("12", "36", "#e67e22")):
        path = _route_path(built, stand_id, designator, RUNWAY_PENALTY)
        for u, v in zip(path, path[1:]):
            out.append(
                line(
                    node_xy(u),
                    node_xy(v),
                    stroke=color,
                    stroke_width=2.5,
                    stroke_dasharray="6 4",
                )
            )
    # Stands: stop position, lead-in, label.
    for s in built.tg.stands:
        stop, entry = frame.to_xy(s.stop), frame.to_xy(s.entry)
        if s.geometry == "line":
            out.append(line(stop, entry, stroke="#2c3e50", stroke_width=2))
        else:
            sx, sy = px(*stop)
            fill = "#ffffff" if s.node is not None else "#e74c3c"
            out.append(
                f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="4" stroke="#2c3e50" fill="{fill}"/>'
            )
        if s.node is not None and s.snap_m > 0.0:
            out.append(
                line(
                    entry,
                    node_xy(s.node),
                    stroke="#2c3e50",
                    stroke_width=1,
                    stroke_dasharray="2 3",
                )
            )
        lx, ly = stop
        out.append(text(lx + 12, ly + 6, s.stand_id, fill="#2c3e50"))
    # Thresholds.
    for d, (tx, ty) in syn.THRESHOLDS.items():
        dx = -40 if d == "09" else (16 if d == "27" else 14)
        dy = 24 if d == "18" else (-30 if d == "36" else 4)
        out.append(text(tx + dx, ty + dy, d, font_weight="bold", fill="#333333"))
    # The three repairs, numbered where they fired.
    for n, (x, y) in ((1, (600.0, 150.0)), (2, (1500.0, 149.2)), (3, (2000.0, 0.0))):
        cx, cy = px(x, y)
        out.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="9" stroke="#8e44ad" stroke_width="2" fill="none"/>'
        )
        out.append(
            f'<text x="{cx + 12:.1f}" y="{cy - 10:.1f}" fill="#8e44ad" font-weight="bold">{n}</text>'
        )
    # Legend and taxiway names.
    for name, (x, y) in (
        ("A", (1200.0, 165.0)),
        ("B", (1450.0, 365.0)),
        ("C", (1750.0, -185.0)),
    ):
        out.append(text(x, y + 6, name, fill="#3b6ea5", font_weight="bold"))
    legend = [
        ("#c0392b", "stand 1 to 09"),
        ("#e67e22", "stand 12 to 36"),
        ("#8e44ad", "repairs: 1 merge, 2 split, 3 crossing"),
    ]
    ly = y0 + 230
    for color, label in legend:
        out.append(
            line(
                (x0 + 40, ly),
                (x0 + 100, ly),
                stroke=color,
                stroke_width=2.5,
                stroke_dasharray="6 4",
            )
        )
        out.append(text(x0 + 115, ly - 4, label, fill="#333333"))
        ly -= 40
    out.append(
        text(
            x0 + 40,
            y0 + 30,
            "meters; runways gray, taxiways blue; stand 11 (red) does not attach",
            fill="#666666",
        )
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------


def main() -> int:
    built = syn.build()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    matrix_lines, facts = section_matrix(built)
    head = [
        "# The worked airport",
        "",
        "A fictional field, `ZZZZ`, on which every distance, turn, junction",
        "count and repair is known before the pipeline runs, pushed through",
        "`build_graph`, `attach_stands`, `build_runways`, `route_all` and",
        "`aggregate` unchanged. Generated by `scripts/report_synthetic_airport.py`",
        f"on {generated}; every figure below is computed there, and",
        "`tests/test_synthetic_airport.py` asserts the agreement on every run.",
        "The field is authored in `taxiout.synthetic`: no OpenStreetMap data,",
        "no challenge data, no network, so this is the one geometry report a",
        "clean checkout can regenerate.",
        "",
        "## In one paragraph",
        "",
        "The real airports validate the geometry by consistency -- pavement",
        "conserved, fixed points reached, no route shorter than the straight",
        "line -- because nobody has the right answer for a real stand. This",
        "field has one. Every planted repair fired once where it was planted;",
        f"every one of the {facts['cells']} matrix cells matches its hand derivation to",
        f"within {num(1000 * facts['worst_m'], 0)} mm and {num(facts['worst_deg'], 2)} degrees, with the",
        "integer descriptors exact; the runway-penalty crossovers fall where",
        "the arithmetic puts them. The first draft of the field also found",
        "three things in the pipeline, each below its working tolerance and",
        "each recorded below.",
        "",
        "![The worked airport](synthetic_airport.svg)",
        "",
        "Runway 09/27 along the bottom, 18/36 crossing it; taxiway A parallel",
        "to the runway with four links; apron B above with a single exit, apron",
        "C below the runway with a single link. Numbered circles mark where",
        "each repair fired. Two routes are drawn at the default penalty: stand",
        "1 to 09, and stand 12 to 36, which crosses 09/27 and backtracks the",
        "length of 18/36 because no taxiway reaches that threshold.",
        "",
    ]
    body = (
        head
        + section_repairs(built)
        + [""]
        + section_stands(built)
        + [""]
        + matrix_lines
        + [""]
        + section_worked(built)
        + section_penalty()
        + [""]
        + section_findings()
        + [""]
        + section_traffic()
        + [""]
        + section_limits()
        + [
            "",
            "No OpenStreetMap data and no challenge data were used for this",
            "report; the field is authored. The repository's distance matrices",
            "elsewhere are derived from OpenStreetMap",
            "(data (c) OpenStreetMap contributors, ODbL 1.0).",
        ]
    )
    OUT_SVG.write_text(draw_svg(built), encoding="utf-8")
    OUT_MD.write_text("\n".join(body) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD.relative_to(REPO)} and {OUT_SVG.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
