# prc-taxiout-2026

Taxi-out time prediction for the [EUROCONTROL PRC Data Challenge 2026](https://ansperformance.eu/study/data-challenge/dc2026/).

Predicts `TAXITIME_SEC_mvt` for departures at major European airports — the
challenge lists 11, the data supplied contains 10 (Antalya has no rows; see
[`reports/data_review.md`](reports/data_review.md)). Ranked by RMSE over
January and July 2026 movements.

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
geometry, taxiway topology — and derives distance and layout features whose
purpose is to carry over to stand/runway pairs the model has never seen
together. Whether they do is the hypothesis the held-out-pair validation in
[`docs/modelling_notes.md`](docs/modelling_notes.md) exists to test, and it is
reported as measured, not assumed.

The model is stated as a decomposition, because that is how the organisers
frame the use of a taxi-out estimate — post-operations analysis of "excess
fuel burnt … compared to unconstrained status"
([rationale](https://ansperformance.eu/study/data-challenge/dc2026/rationale.html)):

    taxi-out = unimpeded(geometry) + excess(congestion, regime) + record-quality term

The unimpeded term is what the geometry supplies, and it generalises the PRU's
own reference — the 10th percentile per (stand, runway) pair — to pairs never
observed. The excess term is built from what the ranking set exposes about
surrounding traffic and conditions. The record-quality term exists because the
data carries day-long record-matching artefacts that the stated scoring rule
counts like any other row; it is kept apart so it cannot be mistaken for
modelling. Each term is a rung of the ablation ladder.

The 2025 selection committee's published reviews commended the winning entry
for a methodology "strongly informed by domain knowledge rather than relying
exclusively on machine-learning techniques" and its "near-complete JOAS-style
paper", and described the third-placed entry as "predominantly
machine-learning–driven, with less emphasis on domain-specific feature
engineering"
([2025 outcome](https://ansperformance.eu/study/data-challenge/dc2025/outcome.html)).
The physical layer is therefore the core of the approach rather than an
add-on, and the write-up is a deliverable, not an afterthought.

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
- [x] Challenge data review: what the ranking set exposes, target tails,
      identifier join — [`reports/data_review.md`](reports/data_review.md)
- [x] Join of OSM identifiers against `STAND_mvt` / `RUNWAY_mvt`
      (`taxiout.join`; 87–99.9% of departures by airport, 96% overall)
- [x] Baseline model, the ablation ladder and a valid submission —
      [`reports/baseline.md`](reports/baseline.md)
- [x] The record-quality term — [`reports/tails.md`](reports/tails.md)
- [x] A worked airport where every number is known —
      [`reports/synthetic_airport.md`](reports/synthetic_airport.md)
- [x] The unimpeded term measured against the training year —
      [`reports/geometry_gaps.md`](reports/geometry_gaps.md)
- [x] Congestion features from what the ranking set exposes (take-offs,
      schedules, arrivals, configuration, the shared corridor — not
      off-block times, which are blanked) —
      [`reports/congestion.md`](reports/congestion.md)
- [x] Weather as a regime indicator, from a scripted, cached, attributed
      METAR fetch
- [ ] Write-up

## What the data turned out to contain

The challenge data landed on 2 September and was profiled before any feature
was written; [`reports/data_review.md`](reports/data_review.md) is generated
from it and carries every figure. Four findings shape everything after:

- **Ten airports, not eleven.** Antalya (LTAI) has no rows in training or
  ranking, though the organisers list it and the total row count matches
  theirs. The geometry for it stays in the pipeline and is unused.
- **The ranking set blanks the off-block time, not just the target.** A test
  departure carries its take-off time, its schedule and the flight plan's
  estimates, and nothing else that is actual — and because the block time is
  blanked for *every* departure in the ranking months, no feature may count
  other departures' pushbacks either. Congestion has to be built from
  take-offs, schedules and arrivals, which the ranking set does expose.
  `AOBT_3_flt` survives but is a minute-resolution estimate that misses the
  blanked field by minutes; it is excluded by decision.
- **The target has a heavy tail, and the scoring rule counts every row.**
  Under 0.1% of training departures carry half the squared error of the
  airport-mean predictor, and the mechanism behind most of it turned out to
  be one airport's data feed rather than a scatter of bad records: 82% of the
  departures recorded above two hours are at Rome Fiumicino, and for nine in
  ten of those the recorded off-block time sits within a minute of the
  *scheduled* departure while the take-off is hours later. The block time was
  defaulted to the schedule, so the recorded taxi-out is the schedule lag —
  which every ranking departure exposes. The ranking page states that the
  score is RMSE over the January and July 2026 movements, with no row
  excluded; five such rows predicted as ordinary taxi-outs would raise a
  300 s model to 512 s. [`reports/tails.md`](reports/tails.md) has the
  mechanism, the cells it covers, and what the term is worth.
- **The identifiers join.** Runways at ~100%; stands at 87% (Munich) to 99.9%
  (Istanbul) of departures after a staged normalisation chosen by join rate
  and inspection, never by validation error.

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

Antalya (LTAI) is the one genuine gap at 45% — moot in practice, since the
challenge data contains no Antalya movements. Everything else is workable, and
the join against the real identifiers confirmed it:
[`reports/data_review.md`](reports/data_review.md).

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
make metar     # cache METARs from the Iowa State Mesonet (network, resumable)
make reports   # coverage, distance matrices, routing + sensitivity reports
make synthetic # the worked airport; reads nothing, so it runs anywhere
make test      # unit tests, no network, no challenge data
make crosscheck  # optional: verify runway geometry against OurAirports
make data-review  # needs the challenge data locally; writes reports/data_review.md
make tails        # needs the challenge data; writes reports/tails.md
make geometry-gaps  # needs the challenge data and the OSM cache; writes reports/geometry_gaps.md
make congestion   # needs the challenge data; writes reports/congestion.md
make baseline     # needs the challenge data; writes reports/baseline.md
make submission TEAM=<name>   # needs the challenge data; writes a submission
make licence-check  # verify no OSM-derived or challenge content is tracked
```

`make test` works on a clean checkout with nothing installed but `pytest`
(the join tests additionally need `pandas`, and skip without it). For the
geometry, `pip install -e .` — the only runtime dependency is `networkx`. The
data review and everything after it need `pip install -e ".[modelling]"`.

| stage | reads | writes |
| ----- | ----- | ------ |
| `scripts/fetch_osm.py` | Overpass API | `data/raw/<ICAO>.json.gz` |
| `scripts/fetch_metar.py` | Iowa State Mesonet ASOS archive | `data/raw/metar/<ICAO>_<period>.csv`, `MANIFEST.json` |
| `scripts/report_coverage.py` | `data/raw/` | `reports/stand_coverage.md` |
| `scripts/build_taxi_graph.py` | `data/raw/` | `data/processed/taxi_distance/`, `reports/taxi_routing.md` |
| `scripts/check_penalty_sensitivity.py` | `data/raw/` | `reports/penalty_sensitivity.md` |
| `scripts/check_runways_ourairports.py` | `data/raw/`, OurAirports | `reports/runway_crosscheck.md` |
| `scripts/report_synthetic_airport.py` | nothing — the field is authored in `taxiout.synthetic` | `reports/synthetic_airport.md`, `reports/synthetic_airport.svg` |
| `scripts/review_challenge_data.py` | `data/challenge/`, `data/processed/taxi_distance/` | `reports/data_review.md` |
| `scripts/report_tails.py` | `data/challenge/` | `reports/tails.md` |
| `scripts/report_geometry_gaps.py` | `data/challenge/`, `data/processed/taxi_distance/`, `data/raw/` | `reports/geometry_gaps.md` |
| `scripts/report_congestion.py` | `data/challenge/`, `data/processed/taxi_distance/` | `reports/congestion.md` |
| `scripts/run_baselines.py` | `data/challenge/`, `data/processed/taxi_distance/` | `reports/baseline.md` |
| `scripts/make_submission.py` | `data/challenge/`, `data/processed/taxi_distance/` | `data/submissions/<team>_v<n>.parquet` (gitignored) |

Only `fetch_osm.py`, `fetch_metar.py` and `check_runways_ourairports.py` touch the network. `build_taxi_graph.py` runs all 11
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
  the challenge. Not redistributable under any licence, so they live in
  `data/challenge/` and are gitignored in full; see
  [`data/challenge/README.md`](data/challenge/README.md).
- **OpenStreetMap** — © OpenStreetMap contributors, available under the
  [Open Database License](https://www.openstreetmap.org/copyright). Retrieved
  via the Overpass API.
- **OurAirports** — released to the public domain; used to cross-check
  OSM-derived runway designators and lengths.
- **Iowa Environmental Mesonet, Iowa State University** — the ASOS/METAR
  archive, fetched by `scripts/fetch_metar.py` for the training year and the
  two ranking months at every listed airport and cached, gitignored, under
  `data/raw/metar/` with the request parameters recorded in
  `MANIFEST.json`. This is the one additional dataset the model reads, and
  it enters only as three regime flags per airport-hour (visibility band,
  precipitation, freezing). The IEM's disclaimer page states the terms: the
  materials "are in the public domain and may be used freely by anyone for
  any lawful purpose", provided "without any warranty of accuracy", and
  "Attributing the Iowa Environmental Mesonet of Iowa State University would
  be appreciated" — which this line does. The observations themselves are
  METARs the airports publish; the archive is where a reader can re-fetch
  them from the recorded request.

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
`make licence-check` enforces it: no OSM-derived artefact or challenge data
tracked, no literal stand identifier from either source in any tracked file,
attribution present on every report. It runs in CI and should run before every
commit.

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
  challenge.py   challenge data access; which fields are hidden or excluded
  join.py        staged join of STAND_mvt / RUNWAY_mvt to the matrices
  windows.py     trailing and centred windows: closed, subject removed, truncation flagged
  congestion.py  throughput, arrivals, demand and configuration from the movements around a subject
  corridor.py    the shared-corridor load, from the routes' taxiway edge sets
  metar.py       weather as three regime flags per airport-hour
  synthetic.py   the worked airport: a field where every number is known, and
                 worked traffic in the challenge schema
  features.py    the feature frame, and a registry of what every column reads
  unimpeded.py   the PRU reference and its geometric counterpart
  validation.py  the splits and the metrics, fixed before any model
  models.py      the ablation ladder and the estimators its rungs use
  tails.py       the record-quality term
  submission.py  validate and write a submission file
scripts/
  fetch_osm.py                 cache OSM geometry (resumable, network)
  fetch_metar.py               cache METARs (resumable, network)
  report_coverage.py           reports/stand_coverage.md
  build_taxi_graph.py          distance matrices + reports/taxi_routing.md
  check_penalty_sensitivity.py reports/penalty_sensitivity.md
  check_runways_ourairports.py reports/runway_crosscheck.md (network)
  report_synthetic_airport.py  reports/synthetic_airport.md (+ .svg; reads nothing)
  review_challenge_data.py     reports/data_review.md (needs challenge data)
  report_tails.py              reports/tails.md (needs challenge data)
  report_geometry_gaps.py      reports/geometry_gaps.md (needs challenge data + OSM cache)
  report_congestion.py         reports/congestion.md (needs challenge data)
  run_baselines.py             reports/baseline.md (needs challenge data)
  make_submission.py           a submission file (needs challenge data)
  check_licence_hygiene.py     verifies no OSM or challenge content is tracked
  commit_geometry.sh           one-shot commit batch (geometry stage)
  commit_gate0.sh              one-shot commit batch (data review stage)
  commit_tails.sh              one-shot commit batch (record-quality term)
  commit_synthetic.sh          one-shot commit batch (worked airport)
  commit_gate1.sh              one-shot commit batch (modelling stage)
  commit_geometry_gaps.sh      one-shot commit batch (unimpeded-term diagnostics)
docs/
  modelling_notes.md  decisions, caveats, open questions, leakage rules
  prior_art.md        literature, external data licences, benchmarks
  plan.md             gated plan to the 11 October deadline
reports/              generated; every number in them is computed, not typed
data/
  raw/                OSM cache (gitignored)
  processed/          distance matrices (generated, ODbL, gitignored)
  challenge/          challenge data goes here (gitignored, not redistributable)
tests/                geodesy, runway assembly, topology repair, routing, join
```
