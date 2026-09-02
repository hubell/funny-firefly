# Stand to runway distance matrices

Derived from OpenStreetMap by `scripts/build_taxi_graph.py`. Regenerate with:

```bash
python scripts/fetch_osm.py          # cache OSM geometry (resumable)
python scripts/build_taxi_graph.py   # rebuild these files
```

## Licence

**These files are not committed, and that is a licensing decision rather than a
size one.** They are a Derivative Database of OpenStreetMap, so they carry ODbL
share-alike — which is not this project's to waive — and this repository ships
GPLv3 in totality. Full reasoning in
[`ODbL-NOTICE.md`](ODbL-NOTICE.md).

Regenerate them with `make data && make reports`. The reports under `reports/`
*are* committed: those are Produced Works under ODbL, so they need attribution
and nothing more, and they carry every summary figure a reader would want to
check without running anything.

## Files

`<ICAO>.csv` — the joinable matrix, one row per (stand identifier, runway
designator). Distances in metres, angles in degrees.

| column | meaning |
| ------ | ------- |
| `icao` | airport, so the per-airport files concatenate cleanly |
| `stand_id` | OSM stand identifier, to be joined against `STAND_mvt` |
| `designator` | runway designator, e.g. `27L`, to be joined against `RUNWAY_mvt` |
| `runway_ref` | the OSM runway this designator belongs to, e.g. `09R/27L` |
| `n_features` | OSM parking positions sharing this identifier |
| `spread_m` | range of `taxi_m` across those features — ambiguity indicator |
| `snap_max_m` | largest snap among them; survives aggregation, unlike `snap_m` |
| `taxi_m` | total stand to threshold distance: `lead_in + snap + route + entry_offset` |
| `lead_in_m` | stop position to where the stand meets the network (the pushback) |
| `snap_m` | gap bridged to attach the stand; 0 where it touches mapped pavement |
| `route_m` | routed distance over mapped pavement |
| `entry_offset_m` | threshold to its attachment point; 0 by construction |
| `turn_deg` | cumulative absolute heading change, on a 3 m simplified path |
| `sharp_turns` | heading changes of 30 degrees or more |
| `junctions` | nodes of degree 3 or more traversed |
| `straight_m` | distance on junction-free running |
| `backtrack_m` | distance along the departure runway itself |
| `cross_runway_m` | distance along other runways |
| `runway_crossings` | number of other runways crossed |
| `gc_m` | straight-line stand to threshold distance |
| `detour_ratio` | `taxi_m / gc_m` |

`<ICAO>_features.csv` — one row per (OSM parking position, designator), before
identifiers are aggregated. Keeps the unidentified stands that the matrix drops
and carries `osm_id` / `osm_type` so a suspect matrix cell can be traced back to
the feature that produced it. **Not committed**: it is a debugging
artefact roughly twice the size of the matrices, regenerated alongside them,
holding nothing they do not.

## Before using these

Read [`reports/taxi_routing.md`](../../../reports/taxi_routing.md) first. It
records what resolved, what did not, and six things these numbers do not
establish — chief among them that `stand_id` actually joins to `STAND_mvt`,
which is unverified until the challenge data lands, and that a cell with a large
`spread_m` describes the stand its identifier names at all.
