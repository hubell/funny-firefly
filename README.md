# prc-taxiout-2026

Taxi-out time prediction for the [EUROCONTROL PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/).

Predicts `TAXITIME_SEC_mvt` for departures at 11 major European airports.
Ranked by RMSE over January and July 2026 movements.

**Licence: GNU GPL v3.0 only.** Copyright (C) 2026 Hunter Bell. Every file in
this repository is GPLv3, from the first commit, as prize eligibility requires
— there is no second licence anywhere in it. The distance matrices the pipeline
*generates* are a Derivative Database of OpenStreetMap and carry ODbL, which is
why they are gitignored rather than committed; see
[Data sources](#data-sources).

## Approach

The challenge supplies `STAND_mvt` and `RUNWAY_mvt` as opaque identifiers. The
obvious move is to one-hot them and let a gradient-boosted tree memorise the
mean taxi time per pair. That works, but it throws away the fact that taxi-out
is a *physical* process: an aircraft traverses actual distance across an actual
taxiway network, then queues for an actual runway.

This entry recovers that structure from OpenStreetMap — stand and runway
geometry, taxiway topology — and derives distance and layout features that
generalise across stand/runway pairs the model has never seen together. Review
feedback from the 2025 edition consistently rewarded domain-grounded feature
engineering over pure ML, so the physical layer is the core of the approach
rather than an add-on.

Distance is not the only thing the geometry gives. Ravizza et al. (2013) found
that decomposing a taxi route — distance, cumulative turn angle, junction-free
running — explains materially more of the variation in taxi speed than distance
alone, and that turn angle by itself is nearly as predictive as distance. Each
route is therefore described by a small vector of physical quantities rather
than a single number, and which of them earn their place is a question for the
modelling stage, measured against the target.

## Status

- [x] OSM movement-area extraction for all 11 airports
- [x] Stand-identifier coverage audit — [`reports/stand_coverage.md`](reports/stand_coverage.md)
- [x] Routable taxiway graph + stand→runway distance matrices —
      [`reports/taxi_routing.md`](reports/taxi_routing.md)
- [ ] Join rate of OSM identifiers against `STAND_mvt` / `RUNWAY_mvt` (needs challenge data)
- [ ] Congestion features (departure queue depth, concurrent movements)
- [ ] Baseline model
- [ ] Submission pipeline

## Stand identifier coverage

Before building distance features, one question decides whether the approach is
viable: **can a `STAND_mvt` value be resolved to a physical location?** OSM
records stand identifiers on `aeroway=parking_position`, but coverage varies by
airport. Measured rather than assumed:

| verdict | airports |
| ------- | -------- |
| STRONG (≥90%) | EGLL, LEBL, LEMD, LFPG, LIRF, LTFM, LSZH |
| PARTIAL (≥60%) | EDDF, EDDM, EHAM |
| WEAK (>0%) | LTAI |

Full table with per-airport counts and identifier *formats*:
[`reports/stand_coverage.md`](reports/stand_coverage.md). Formats are given as
shapes rather than examples — a shape describes what the join will need, where a
list of examples would put an extract of OSM into a GPLv3 repository.

Antalya (LTAI) is the one genuine gap at 45%, and needs either a fallback to
apron-centroid geometry or an explicit airport-level treatment. Everything else
is workable.

## Stand to runway routing

The second gate check: once a stand resolves to a location, can it be routed to
a runway, and is the distance credible? Full results in
[`reports/taxi_routing.md`](reports/taxi_routing.md); the matrices themselves
are regenerated into `data/processed/taxi_distance/` by the build — not
committed, for the licensing reason in [Data sources](#data-sources).

Across the 11 airports, 95–100% of (attached stand × runway designator) pairs
route — 91–100% counted against every mapped stand, including those too far from
any mapped taxiway to attach at all. Median taxi distances run 2.6–5.5 km, with
a median detour ratio of 1.26–1.42 against the straight line.

The raw OSM topology does not support this directly. Three defects have to be
repaired first: nodes that are coincident but not shared (up to 166 per
airport), nodes lying on edges they are not connected to (7–1912 edge splits),
and ways that cross on the map with no vertex at the crossing at all (up to 89
junctions inserted). Each repair creates edges that can themselves need
repairing, so they run to a fixed point.

Two caveats belong next to those numbers rather than buried in the report. The
medians are unweighted over the full cross-product, so they include designators
that never see a departure — within a single airport the spread between
designators is larger than the spread between airports, and these figures should
be read as "did the geometry resolve", not "how far do aircraft taxi here". And
the joinable identifier count is well below the stand count everywhere (484 of
768 at CDG, 149 of 576 at Antalya), because several OSM features routinely share
one identifier; the coverage table above, which counts tagged *features*,
overstates what can actually be joined.

Antalya is the outlier again, for reasons independent of the identifier coverage
above — its apron lead-ins are largely unmapped, and 168 of its 576 stands are
drawn as footprint polygons rather than centrelines, a convention no other
airport in scope uses.

## Reproducing

```bash
make data      # cache OSM geometry from Overpass (network, resumable)
make reports   # coverage, distance matrices, routing + sensitivity reports
make test      # 72 tests, no network, no challenge data
make crosscheck  # optional: verify runway geometry against OurAirports
make licence-check  # verify no OSM-derived content is tracked
```

`make test` works on a clean checkout with nothing installed but `pytest`. For
the rest, `pip install -e .` — the only runtime dependency is `networkx`.

| stage | reads | writes |
| ----- | ----- | ------ |
| `scripts/fetch_osm.py` | Overpass API | `data/raw/<ICAO>.json.gz` |
| `scripts/report_coverage.py` | `data/raw/` | `reports/stand_coverage.md` |
| `scripts/build_taxi_graph.py` | `data/raw/` | `data/processed/taxi_distance/`, `reports/taxi_routing.md` |
| `scripts/check_penalty_sensitivity.py` | `data/raw/` | `reports/penalty_sensitivity.md` |
| `scripts/check_runways_ourairports.py` | `data/raw/`, OurAirports | `reports/runway_crosscheck.md` |

Only the first and last touch the network. `build_taxi_graph.py` runs all 11
airports in about two minutes from the cache and takes `--airports`,
`--runway-penalty` and `--max-snap-m` for sensitivity checks. Output is
deterministic: repeated runs produce byte-identical files.

`scripts/fetch_osm.py` is resumable and caches one gzipped Overpass response per
airport under `data/raw/`. Overpass mirrors fail transiently and often; the
client rotates mirrors and backs off. Delete a cache file to refetch that
airport. The cache is gitignored — the repository stays reproducible from source
rather than shipping a copy of OSM.

`--date 2026-01-01T00:00:00Z` pins a fetch to a moment in OSM's history, which
is what makes a rebuild months from now reproduce these numbers rather than
whatever the map says then. The figures in the committed reports were built from
an unpinned fetch; not every Overpass mirror answers attic queries, so a pinned
fetch may exercise the mirror rotation harder.

## Data sources

- **Challenge datasets** — EUROCONTROL PRC / OpenSky Network, under the terms of
  the challenge. Not redistributed here.
- **OpenStreetMap** — © OpenStreetMap contributors, available under the
  [Open Database License](https://www.openstreetmap.org/copyright). Retrieved
  via the Overpass API.
- **OurAirports** — released to the public domain; used to cross-check
  OSM-derived runway designators and lengths.

ODbL sorts derived artefacts into two kinds, and the difference decides what
ships. A **Derivative Database** carries share-alike; a **Produced Work** — a
report, an image, an analysis in another medium — needs attribution only.

The matrices the build writes are a Derivative Database, so they carry ODbL.
That obligation is not this project's to waive, and a second licence inside a
GPLv3 repository is exactly what this entry does not want, so they are
gitignored and regenerated rather than committed. The reports are Produced
Works, so they are committed under GPLv3 with OSM attribution, and they carry
the summary figures. Nothing about a model fitted on these features, or about
the predictions submitted, is encumbered either way.

Reasoning in full:
[`data/processed/taxi_distance/ODbL-NOTICE.md`](data/processed/taxi_distance/ODbL-NOTICE.md).
`make licence-check` enforces it: no OSM-derived artefact tracked, no literal
OSM stand identifier in any tracked file, attribution present on every report.
It runs in CI and should run before every commit.

## Prior art

Building a routable graph from OSM `aeroway` geometry is not new. `traffic`
(Olive, MIT) ships an `AirportGraph` that performs the same class of topology
repair, and Olive, Waltert & Schultz (ATRD 2025) document the same OSM defects
this pipeline works around. The graph builder here is an independent
implementation rather than a port — it keeps the dependency surface small and
every geometric decision auditable in-repo — and the reasoning is recorded in
[`docs/modelling_notes.md`](docs/modelling_notes.md).

What appears not to have been done before is the downstream half: using these
recovered quantities as regression features for taxi-out prediction in this
competition. No entry in the 2024 or 2025 editions appears to have used OSM at
all.

Full literature summary, the licence position of every external source
considered — including the ones ruled out — and the error magnitudes published
work actually achieves: [`docs/prior_art.md`](docs/prior_art.md).

## Layout

```
src/taxiout/
  airports.py    11 challenge airports + reference points
  osm.py         Overpass client: mirror rotation, backoff, disk cache
  coverage.py    stand-identifier coverage audit
  geo.py         local geodesy: distance, bearing, projection, simplification
  runways.py     assemble runways from OSM ways; resolve designator thresholds
  graph.py       routable graph: node merging, crossing repair, stand attachment
  routing.py     stand->runway routes and their physical descriptors
scripts/
  fetch_osm.py                 cache OSM geometry (resumable, network)
  report_coverage.py           reports/stand_coverage.md
  build_taxi_graph.py          distance matrices + reports/taxi_routing.md
  check_penalty_sensitivity.py reports/penalty_sensitivity.md
  check_runways_ourairports.py reports/runway_crosscheck.md (network)
  check_licence_hygiene.py     verifies no OSM content is tracked
  commit_geometry.sh           one-shot commit batch
docs/
  modelling_notes.md  decisions, caveats, open questions, leakage rules
  prior_art.md        literature, external data licences, benchmarks
  plan.md             gated plan to the 11 October deadline
reports/              generated; every number in them is computed, not typed
data/
  raw/                OSM cache (gitignored)
  processed/          distance matrices (generated, ODbL, gitignored)
tests/                geodesy, runway assembly, topology repair, routing
```
