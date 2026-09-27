#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Build the taxiway graph and stand->runway distance matrix for each airport.

Reads the cached Overpass responses under ``data/raw/`` and writes, per airport:

``data/processed/taxi_distance/<ICAO>.csv``
    the joinable matrix -- one row per (stand identifier, runway designator).

``data/processed/taxi_distance/<ICAO>_features.csv``
``data/processed/taxi_distance/<ICAO>_edges.csv``
    one row per (OSM parking position, designator), before identifiers are
    aggregated. Keeps the unidentified stands, which the matrix drops, and lets
    a suspicious matrix cell be traced back to the feature that produced it.

``reports/taxi_routing.md``
    what actually resolved, and what did not.

Needs no challenge data: everything here is derived from OpenStreetMap and can
be built and checked before the movement buckets arrive.

Usage
-----
    python scripts/build_taxi_graph.py
    python scripts/build_taxi_graph.py --airports EGLL LSZH --runway-penalty 1.0
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS, BY_ICAO  # noqa: E402
from taxiout.geo import angle_diff_deg, bearing_deg  # noqa: E402
from taxiout.graph import (  # noqa: E402
    DEDUP_M,
    DIRECT_SNAP_M,
    MAX_SNAP_M,
    attach_stands,
    build_graph,
)
from taxiout.osm import cache_path  # noqa: E402
from taxiout.provenance import report_inputs_footer  # noqa: E402
from taxiout.routing import (  # noqa: E402
    AGGREGATED_FIELDS,
    RUNWAY_PENALTY,
    aggregate,
    encode_edges,
    route_all,
)
from taxiout.runways import build_runways  # noqa: E402

CACHE = REPO / "data" / "raw"
OUT_DIR = REPO / "data" / "processed" / "taxi_distance"
REPORT = REPO / "reports" / "taxi_routing.md"

FEATURE_COLUMNS = (
    "icao stand_id id_tag osm_id osm_type runway_ref designator "
    "taxi_m lead_in_m snap_m route_m entry_offset_m gc_m detour_ratio "
    "turn_deg sharp_turns junctions straight_m backtrack_m cross_runway_m "
    "runway_crossings stand_lat stand_lon thr_lat thr_lon"
).split()

MATRIX_COLUMNS = (
    "icao",
    "stand_id",
    "designator",
    "runway_ref",
    "n_features",
    "spread_m",
    "snap_max_m",
    *AGGREGATED_FIELDS,
)

EDGE_COLUMNS = ("icao", "stand_id", "designator", "n_edges", "edges")


def _round(value: object) -> object:
    return round(value, 2) if isinstance(value, float) else value


def process(icao: str, penalty: float, max_snap_m: float) -> dict:
    ap = BY_ICAO[icao]
    with gzip.open(cache_path(CACHE, icao), "rt", encoding="utf-8") as fh:
        elements = json.load(fh)["elements"]

    tg = build_graph(icao, elements, (ap.lat, ap.lon))
    attach_stands(tg, elements, max_snap_m=max_snap_m)
    runways, unusable = build_runways(icao, elements)
    routes, entries, rstats = route_all(tg, runways, runway_penalty=penalty)
    cells = aggregate(routes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / f"{icao}_features.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(FEATURE_COLUMNS)
        for r in routes:
            w.writerow(
                [
                    _round(getattr(r, c, None) if c != "icao" else icao)
                    for c in FEATURE_COLUMNS
                ]
            )

    with (OUT_DIR / f"{icao}.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(MATRIX_COLUMNS)
        for c in cells:
            row = [
                icao,
                c.stand_id,
                c.designator,
                c.runway_ref,
                c.n_features,
                _round(c.spread_m),
                _round(c.snap_max_m),
            ]
            row += [_round(c.values[f]) for f in AGGREGATED_FIELDS]
            w.writerow(row)

    # The representative route's taxiway edges per cell, for the shared-corridor
    # feature. Node ids are this build's own and mean nothing outside it, which
    # is fine: the overlap between two cells is computed within one file.
    with (OUT_DIR / f"{icao}_edges.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(EDGE_COLUMNS)
        for c in cells:
            w.writerow(
                [icao, c.stand_id, c.designator, len(c.edges), encode_edges(c.edges)]
            )

    # Report from the matrix, not from the per-feature table. The matrix is what
    # will actually be joined, so summarising the feature table instead would
    # describe how many times OSM happens to draw each stand -- and would
    # include stands with no identifier, which can never join at all. At Antalya
    # those are 59% of the feature rows.
    taxi = sorted(c.values["taxi_m"] for c in cells)
    detours = sorted(c.values["detour_ratio"] for c in cells if c.values["gc_m"] > 50.0)

    def pct(values: list[float], q: float) -> float:
        return values[int(q * (len(values) - 1))] if values else float("nan")

    stats = {
        **tg.stats,
        **rstats,
        "icao": icao,
        "name": ap.name,
        "runways": len(runways),
        "displaced_m": sum(r.displaced_m for r in runways),
        "runway_ways_excluded": len(unusable),
        "matrix_rows": len(cells),
        "matrix_stands": len({c.stand_id for c in cells}),
        "ambiguous_stands": len({c.stand_id for c in cells if c.spread_m > 250.0}),
        "max_spread_m": max((c.spread_m for c in cells), default=0.0),
        "snapped_stands": len(
            {c.stand_id for c in cells if c.snap_max_m > DIRECT_SNAP_M}
        ),
        "taxi_median_m": pct(taxi, 0.5),
        "taxi_p95_m": pct(taxi, 0.95),
        "detour_median": pct(detours, 0.5),
        "detour_p95": pct(detours, 0.95),
        "detour_below_one": sum(1 for d in detours if d < 1.0),
        "detour_above_three": sum(1 for d in detours if d > 3.0),
        "no_crossing_data": all(c.values["runway_crossings"] == 0 for c in cells),
        "max_heading_error": max((e.heading_error_deg for e in entries), default=0.0),
        "axes_deg": {r.ref: bearing_deg(*r.ends) for r in runways if r.ends},
        "designator_medians": {
            d: _median_of([c.values["taxi_m"] for c in cells if c.designator == d])
            for d in sorted({c.designator for c in cells})
        },
        "entries": entries,
        "unusable_ways": unusable,
    }
    return stats


def _median_of(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2] if s else float("nan")


def verdict(s: dict) -> str:
    """Coarse usability band for the distance feature at one airport."""
    direct_share = s["stands_direct"] / max(1, s["stands_total"])
    if s["routable_share"] >= 0.95 and direct_share >= 0.90:
        return "STRONG"
    if s["routable_share"] >= 0.90:
        return "PARTIAL"
    if s["routable_share"] > 0.0:
        return "WEAK"
    return "UNUSABLE"


def write_report(rows: list[dict], penalty: float, max_snap_m: float) -> None:
    head = (
        "| ICAO | rwy | desig | stands | attached | direct | joinable IDs | "
        "routable pairs | median taxi | p95 | detour p50 | verdict |\n"
        "| ---- | --: | ----: | -----: | -------: | -----: | -----------: | "
        "-------------: | ----------: | --: | ---------: | ------- |"
    )
    table = [
        f"| {s['icao']} | {s['runways']} | {s['designators']} | {s['stands_total']} "
        f"| {s['stands_attached']} ({s['stands_attached'] / s['stands_total']:.0%}) "
        f"| {s['stands_direct']} ({s['stands_direct'] / s['stands_total']:.0%}) "
        f"| {s['matrix_stands']} ({s['matrix_stands'] / s['stands_total']:.0%}) "
        f"| {s['routable_share']:.1%} | {s['taxi_median_m']:.0f} m "
        f"| {s['taxi_p95_m']:.0f} m | {s['detour_median']:.2f} | {verdict(s)} |"
        for s in rows
    ]

    graph_head = (
        "| ICAO | ways | nodes | edges | merged | split | crossed | deduped | "
        "components | largest | pavement |\n"
        "| ---- | ---: | ----: | ----: | -----: | ----: | ------: | ------: | "
        "---------: | ------: | -------: |"
    )
    graph_rows = [
        f"| {s['icao']} | {s['ways']} | {s['nodes']} | {s['edges']} "
        f"| {s['nodes_merged']} | {s['edges_split']} | {s['crossings_split']} "
        f"| {s['pavement_deduped_km']:.1f} km | {s['components_after_split']} "
        f"| {s['largest_component_share']:.1%} | {s['pavement_km']:.0f} km |"
        for s in rows
    ]

    flags: list[str] = []
    for s in rows:
        notes = []
        if s["stands_bridged"]:
            notes.append(
                f"{s['stands_bridged']} stands attached across a gap of "
                f"{DIRECT_SNAP_M:.0f}-{max_snap_m:.0f} m (no mapped apron lead-in)"
            )
        if s["stands_total"] - s["stands_attached"]:
            notes.append(
                f"{s['stands_total'] - s['stands_attached']} stands beyond "
                f"{max_snap_m:.0f} m, unattached"
            )
        if s["designators_without_entry"]:
            notes.append(
                f"{s['designators_without_entry']} designators with no entry node"
            )
        if s["ambiguous_stands"]:
            notes.append(
                f"{s['ambiguous_stands']} stand IDs spread over >250 m of "
                "taxi distance between duplicate OSM features"
            )
        if s["detour_below_one"]:
            notes.append(f"{s['detour_below_one']} routes shorter than straight line")
        if s["stands_polygon"]:
            notes.append(
                f"{s['stands_polygon']} stands drawn as footprint polygons, "
                "reduced to centroids"
            )
        if s["backtracking_designators"]:
            listed = ", ".join(
                f"{d} ({v['median_m']:.0f} m"
                + (
                    f", varying by {v['spread_m']:.0f} m)"
                    if v["spread_m"] > 100
                    else ", constant)"
                )
                for d, v in sorted(
                    s["backtracking_designators"].items(),
                    key=lambda kv: -kv[1]["median_m"],
                )
            )
            notes.append(f"threshold reached along the runway itself for {listed}")
        if s["max_heading_error"] > 15.0:
            notes.append(
                f"runway axis up to {s['max_heading_error']:.0f} deg off its "
                "designator (see note below)"
            )
        if s["displaced_m"]:
            notes.append(
                f"{s['displaced_m']:.0f} m of displaced-threshold pavement "
                "re-attached to its runway"
            )
        if s["no_crossing_data"]:
            notes.append(
                "no route crosses a runway — plausible for a midfield layout, "
                "but check it is not a geometry failure"
            )
        if s["detour_above_three"]:
            notes.append(
                f"{s['detour_above_three']} pairs with a detour ratio above 3, "
                "worth an eyeball"
            )
        if s["runway_ways_excluded"]:
            notes.append(
                f"{s['runway_ways_excluded']} runway ways excluded from threshold "
                "derivation (no ref, or not a designator)"
            )
        if notes:
            flags.append(f"- **{s['icao']}** — " + "; ".join(notes))

    # Spread of taxi distance between designators *within* one airport, which is
    # larger than the spread between airports and is the reason the headline
    # medians must not be read as operational figures.
    between = max(s["taxi_median_m"] for s in rows) / min(
        s["taxi_median_m"] for s in rows
    )
    exceeding = 0
    spread_lines = []
    for s in rows:
        meds = {d: m for d, m in s["designator_medians"].items() if m == m}
        if len(meds) < 2:
            continue
        lo = min(meds, key=meds.get)
        hi = max(meds, key=meds.get)
        ratio = meds[hi] / meds[lo]
        exceeding += ratio > between
        spread_lines.append(
            f"| {s['icao']} | {lo} | {meds[lo]:.0f} m | {hi} | {meds[hi]:.0f} m "
            f"| {ratio:.1f}x |"
        )

    worst_id_loss = min(
        rows, key=lambda s: s["matrix_stands"] / max(1, s["stands_total"])
    )
    by_rounds = sorted(rows, key=lambda s: -s["repair_rounds"])
    named, rest = by_rounds[:4], by_rounds[4:]
    rounds_summary = ", ".join(
        f"{s['icao']} {s['repair_rounds']}"
        + ("" if s["repair_converged"] else " (round cap hit)")
        for s in named
    )
    if rest:
        lo = min(s["repair_rounds"] for s in rest)
        hi = max(s["repair_rounds"] for s in rest)
        rounds_summary += f", the rest {lo}" + (f"-{hi}" if hi != lo else "")
    dedup_m = DEDUP_M

    ambiguous_total = sum(s["ambiguous_stands"] for s in rows)
    worst_spread = max(rows, key=lambda s: s["max_spread_m"])

    # Istanbul's heading flag, read off the runway geometry this run routed: the
    # largest angle between the axis of a 16 and the axis of a 17, stated as the
    # bound it sits within at the one decimal the note prints -- "within" is a
    # bound, and rounding the angle to the nearest tenth would state a tighter
    # one than holds -- and how far a 16's axis sits off its nominal heading.
    # Written only when LTFM was routed and both pairs have an axis.
    ltfm_section = ""
    istanbul = next((s for s in rows if s["icao"] == "LTFM"), None)
    axes = istanbul["axes_deg"] if istanbul else {}
    sixteens = [b for ref, b in axes.items() if ref.startswith("16")]
    seventeens = [b for ref, b in axes.items() if ref.startswith("17")]
    if sixteens and seventeens:
        apart = max(
            min(angle_diff_deg(a, b), angle_diff_deg(a, b + 180.0))
            for a in sixteens
            for b in seventeens
        )
        within = math.ceil(apart * 10) / 10
        off = max(
            e.heading_error_deg
            for e in istanbul["entries"]
            if e.designator.startswith("16")
        )
        ltfm_section = f"""### On the LTFM heading flag

Istanbul's 16L/34R and 16R/34L sit within {within:.1f} deg of the same physical axis as
17L/35R and 17R/35L, which reads as a {off:.0f} deg mismatch against the nominal
heading of a 16. It is not a data error: with five parallel strips there are
only three position letters to go round, so the designator number is offset
instead. OurAirports independently lists the same six designators
([`runway_crosscheck.md`](runway_crosscheck.md)) -- so the heading check must
flag rather than reject.

"""

    # Every cached answer this run routed, with its digest, in the form every
    # other report carries.
    footer = "\n".join(
        report_inputs_footer(REPO, [cache_path(CACHE, s["icao"]) for s in rows])
    )

    backtrack_count = sum(len(s["backtracking_designators"]) for s in rows)
    backtrack_airports = sum(1 for s in rows if s["backtracking_designators"])
    constant = [
        f"{s['icao']} {d}"
        for s in rows
        for d, v in s["backtracking_designators"].items()
        if v["spread_m"] <= 100
    ]
    varying = [
        f"{s['icao']} {d} (by {v['spread_m']:.0f} m)"
        for s in rows
        for d, v in s["backtracking_designators"].items()
        if v["spread_m"] > 100
    ]

    body = f"""# Stand to runway routing (OpenStreetMap)

Second gate check for the physically-grounded feature set. The first asked
whether a `STAND_mvt` value can be resolved to a location at all
([`stand_coverage.md`](stand_coverage.md)); this one asks the harder question:
**once resolved, can it actually be routed to a runway, and is the resulting
distance credible?**

Generated by `scripts/build_taxi_graph.py`. No challenge data is involved --
every number here comes from OpenStreetMap and was computed before the movement
buckets arrived, so it constrains the modelling plan rather than confirming it
after the fact.

Parameters: runway-edge penalty {penalty:g}, maximum stand attachment
{max_snap_m:.0f} m, of which the first {DIRECT_SNAP_M:.0f} m counts as a real
mapped lead-in.

## Routability

`stands` counts OSM parking positions. `direct` is the subset touching mapped
pavement within {DIRECT_SNAP_M:.0f} m; the remainder were attached across a gap
and carry a straight-line allowance in `snap_m`. `routable pairs` is the share
of (attached stand x designator) pairs with a path.

`joinable IDs` is the column that matters most and is the least flattering:
distinct stand identifiers in the matrix. It is well below the stand count
everywhere, because several OSM features routinely share one identifier -- so
the coverage report's "share of features carrying a tag" overstates what can
actually be joined against `STAND_mvt`. The worst case in scope is
{worst_id_loss["icao"]}: {worst_id_loss["stands_total"]} mapped stands,
{worst_id_loss["matrix_stands"]} joinable identifiers.

{head}
{chr(10).join(table)}

Verdict bands: STRONG >=95% of pairs routable and >=90% of stands directly
attached; PARTIAL >=90% routable; WEAK any.

## Graph construction

Three repairs run before anything is routed, and their sizes measure how much
of the raw OSM topology could not have been used as it stands. `merged` counts
nodes collapsed into a *nearby but distinct* node -- ordinary way-sharing at an
identical coordinate is the format working correctly and is not counted. `split`
counts edges cut to admit a node that lay on them without being connected to
them. `crossed` counts junctions inserted where two ways intersect on the map
and share no vertex at all -- the case where there is no node to split at.

All three run to a fixed point rather than once, which is not a detail: every
repair creates edges that can themselves need repairing. A runway crossed by
several taxiways needs one round per crossing, because each split replaces the
edge the remaining crossings were being tested against. Rounds taken, by
airport: {rounds_summary}.

Reaching a fixed point at all took a second attempt. The first version had no
minimum-length guard, so in densely mapped areas it split off half-edges of a
few millimetres whose midpoints then sat on their neighbours; the repair
settled into a cycle, splitting and unsplitting the same edges indefinitely,
and the output became a function of how many rounds were allowed rather than of
the airport. Split positions are now required to be at least
{dedup_m:.1f} m from either end, and no (node, edge) combination is considered
twice.

The crossing repair is kept even where it fires rarely, because its absence is
silent. A taxiway crossing a runway with no shared node makes that runway
reachable only where somebody happened to be thorough -- a property of the
mapping, not of the airport -- and the resulting distances would be wrong with
nothing looking wrong.

`deduped` is pavement that turned out to have been drawn twice: once splitting
gives two versions the same endpoints they are the same edge, and the duplicate
length is dropped. This is a reduction in double-counting, not a loss of network
-- component counts are unchanged by it -- but it is reported because an
unexplained shrinking total is indistinguishable from a bug, and was mistaken
for one during development.

{graph_head}
{chr(10).join(graph_rows)}

## Flags

{chr(10).join(flags) if flags else "None."}

### Taxi distance varies more within an airport than between airports

The headline medians above are unweighted over the full stand x designator
cross-product, which includes designators that never see a departure. That
makes them a data-quality statistic, not an operational one, and the size of
the effect is worth seeing:

| ICAO | shortest designator | | longest designator | | ratio |
| ---- | ------------------- | --: | ----------------- | --: | ----: |
{chr(10).join(spread_lines)}

At {exceeding} of {len(rows)} airports the within-airport spread exceeds the
{between:.1f}x spread between airport medians in the headline table. Part of
that is real layout and part is the mapping defect flagged above -- the extreme
designators are often the ones reached along the runway. Once `RUNWAY_mvt`
frequencies are known, this table should be reissued movement-weighted; until
then, read the medians as "did the geometry resolve", not as "how far do
aircraft taxi here".

### On thresholds reached along the runway

{backtrack_count} designators across {backtrack_airports} airports have no
taxiway joining their runway near the threshold in OSM, though the same runway
is connected at its other end. The router reaches those thresholds the only way
the data allows, by running up the runway itself, and that distance is reported
as `backtrack_m` rather than buried in `route_m`.

The cost is smaller than it looks, and the reason is worth stating. For
{", ".join(constant) if constant else "none of them"} the added distance is
*identical for every stand*: a constant offset on one designator, which any
model that knows the runway absorbs. {("Only " + ", ".join(varying) + " varies across stands, and only there does the gap distort the distances between stands -- which is the signal the feature exists to carry.") if varying else "None of them varies across stands."}
It is a defect in OSM that can be fixed upstream rather than worked around here.

### On the runway-edge penalty

Runway pavement is traversable at {penalty:g}x its length. The value was chosen
by measuring rather than tuning, and the measurement is reproducible:
`scripts/check_penalty_sensitivity.py` sweeps it and writes
[`penalty_sensitivity.md`](penalty_sensitivity.md). The short version is that
total distance barely depends on the penalty while the amount of foreign runway
pavement used collapses with it, so the value sits on a plateau rather than at
an optimum.

{ltfm_section}## What this does not establish

**That these identifiers join to `STAND_mvt` and `RUNWAY_mvt`.** Coverage and
routability both measure the OSM side alone. The join rate is still the first
thing to test when the buckets land.

**That a matrix cell describes the stand its identifier names.** Where several
OSM features share one identifier the cell takes the feature closest to their
mean, and `spread_m` records how far apart they were. Usually that is metres.
Sometimes it is not: {worst_spread["icao"]} has an identifier whose features
disagree by {worst_spread["max_spread_m"]:.0f} m against a median taxi distance
of {worst_spread["taxi_median_m"]:.0f} m, and {ambiguous_total} identifiers across all
airports disagree by more than 250 m. Those cells are not uncertain by a metre
or two; they are ambiguous about which piece of pavement the identifier means.
`spread_m` and `snap_max_m` exist so such cells can be down-weighted or dropped
rather than trusted equally, and nothing here does that for you.

**That the descriptors on a snapped cell describe real pavement.** Where a stand
had no mapped lead-in, part of its route is a straight line standing in for
apron that exists in reality and not in the data. That segment contributes
distance but no turns and no junctions.

**That the shortest path is the route flown.** Aircraft follow assigned routes,
one-way taxiways and configuration-dependent flows; the graph is undirected and
knows none of that. Shortest-path distance is a proxy, and no published study
validates it against actual European taxi routes at scale. The detour ratio is
reported per pair so implausible routes surface rather than averaging in.

**That thresholds are displaced thresholds.** OSM runway ways span the full
paved surface, so a runway with a displaced threshold has its take-off point
placed early by that displacement. `backtrack_m` is reported separately from
`route_m` so the along-runway component can be adjusted or dropped without
recomputing anything.

**That a departure taxis to the threshold at all.** Intersection departures are
routine and shorten the taxi materially. The matrix gives full-length
distances; `backtrack_m` is the part that an intersection departure would not
fly, and is the lever for modelling it once the movement data shows which
runways are used that way.

{footer}

Data (c) OpenStreetMap contributors, ODbL 1.0. Runway cross-check data from
OurAirports (public domain).
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--airports", nargs="*", default=[a.icao for a in AIRPORTS])
    parser.add_argument("--runway-penalty", type=float, default=RUNWAY_PENALTY)
    parser.add_argument("--max-snap-m", type=float, default=MAX_SNAP_M)
    args = parser.parse_args()

    rows: list[dict] = []
    for icao in args.airports:
        if not cache_path(CACHE, icao).exists():
            print(f"{icao}: no cache, skipping (run scripts/fetch_osm.py)")
            continue
        s = process(icao, args.runway_penalty, args.max_snap_m)
        rows.append(s)
        print(
            f"{icao}: {s['matrix_rows']:5} matrix rows, "
            f"{s['routable_share']:.1%} of pairs routable, "
            f"median taxi {s['taxi_median_m']:.0f} m, "
            f"detour {s['detour_median']:.2f}"
        )

    if rows:
        write_report(rows, args.runway_penalty, args.max_snap_m)
        print(f"\nwritten: {REPORT.relative_to(REPO)}")
        print(f"written: {OUT_DIR.relative_to(REPO)}/<ICAO>[_features].csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
