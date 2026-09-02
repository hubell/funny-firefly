# Plan to the deadline

Written 2 September 2026. The challenge closes **11 October 2026**, so this is a
five-and-a-half week plan, and the challenge data has not arrived — team
approval and the OpenSky bucket keys are still outstanding. Everything below is
therefore split by that dependency, and the expensive items are gated rather
than committed to.

The frame is a timeboxed probe with decision gates, not an open-ended build.
Judging weighs originality and reproducibility alongside RMSE, so a measured,
well-documented entry that places mid-table is worth more than an unexplained
one that places higher.

## Where things stand

Done, and reported rather than asserted:

- OSM movement-area geometry for all 11 airports, cached and regenerable.
- Stand-identifier coverage audit — [`../reports/stand_coverage.md`](../reports/stand_coverage.md).
- Routable surface graph with three classes of topology repair, converging to a
  fixed point at every airport.
- Stand→runway distance matrices with route descriptors, 95–100% of attached
  pairs routing — [`../reports/taxi_routing.md`](../reports/taxi_routing.md).
- Runway geometry cross-checked against OurAirports: 39 runways, zero material
  discrepancies — [`../reports/runway_crosscheck.md`](../reports/runway_crosscheck.md).
- Runway-penalty sensitivity swept — [`../reports/penalty_sensitivity.md`](../reports/penalty_sensitivity.md).
- 72 tests, no network, no challenge data. Deterministic build.

Not done, and the honest list: nothing has been validated against a taxi time.
Every number so far describes geometry, not prediction.

## Licensing, and why it shapes the plan

This repository ships **GPLv3 in totality**. OpenStreetMap is ODbL, and its
share-alike obligation is not this project's to waive, so no OSM content is
committed — not the matrices, not the cache, and not literal stand identifiers.
The reasoning is in
[`../data/processed/taxi_distance/ODbL-NOTICE.md`](../data/processed/taxi_distance/ODbL-NOTICE.md);
`make licence-check` enforces it.

Two consequences that reach into the modelling stage and are easy to forget:

- **A feature frame containing the OSM-derived columns is itself a Derivative
  Database.** Cached training frames stay out of the repository exactly as the
  matrices do.
- **The model needs an attribution line; the predictions do not.** OSMF's
  Attribution Guidelines (board-adopted 2021) hold that training sets which are
  substantial extractions are Derivative Databases, that models must be
  attributed in their documentation, and that predictions are not implicated by
  ODbL at all.

### Three licensing tasks, before or alongside the modelling

| task | effort | why |
| ---- | -----: | --- |
| Zenodo deposit of the matrices under ODbL, with DOI, referenced from the README | ~3 h | Puts the encumbered artefact under its correct licence in its correct place, rather than arguing it is not encumbered. Zenodo supports mixed-licence uploads; figshare does not offer ODbL at all. Needed for the paper regardless. |
| Synthetic worked airport in-repo | ~6–10 h | A fictional field where every distance is hand-computable and each repair visibly fires. Fully authored, no licence question, and a stronger correctness argument than 24,000 rows a reader must take on trust. Also the figure a paper wants. |
| Richer aggregates in the reports | ~2–4 h | Per-designator distributions and snap-tier breakdowns, staying inside what a report may carry. Recovers some of what the matrices would have shown. |

## Before the data lands

Buildable now, and all of it needed later:

1. **Leakage test harness** (~3–4 h). Assertions that no feature reads
   `TAXITIME_SEC_mvt`, and that every congestion feature has an explicit as-of
   time. Testable on synthetic data, so it exists before it can be violated.
2. **Validation machinery** (~4–6 h). Blocked forward splits, `GroupKFold` on
   (stand, designator) and on stand alone, and the frequency-stratified error
   report. See *Validation* in [`modelling_notes.md`](modelling_notes.md).
3. **Unimpeded taxi-time structure** (~3–4 h). Distance to time needs a speed,
   which needs data — but the code and its tests do not.
4. The three licensing tasks above.

That is roughly 20–30 hours of work with no dependency on the buckets, and it is
the difference between having a week of scramble after the data arrives and
having a pipeline waiting for it.

## When the data lands

**Gate 0 — does the join work?** (~4–6 h, and it blocks everything.)

Test the OSM identifier against `STAND_mvt` and the designators against
`RUNWAY_mvt`. The shapes in the coverage report say what normalisation to
expect: zero-padding, suffix stripping, case folding. Also confirm
`RUNWAY_mvt` and an off-block timestamp are exposed for ranking rows — if the
runway is not, the distance matrix, the corridor feature and the configuration
feature all fail together, and the fallback is to marginalise over a
runway-assignment model.

*If the join rate is poor at an airport, that airport falls back to an
airport-level constant and the write-up says so.* Antalya is the expected case
at 45% identifier coverage and 149 joinable identifiers against 576 stands.

**Gate 1 — a working end-to-end submission** (~10–15 h). Baseline, geometry
features, time-of-day, one-hot identifiers. Submit something valid early; a
scored mediocre entry beats an unscored good one.

**Gate 2 — cheap congestion** (~4–6 h). Trailing-window off-block counts:
departures with off-block in [t−Δ, t) for several Δ, and the same restricted to
the subject's runway and pier. Causal and available on both sides. This proxies
*demand*, not queue.

**Gate 3 — the queueing layer, only if Gate 2's ablation says congestion is
binding** (~20–30 h). This is the piece where the geometry stops being a side
feature.

The literature's strongest congestion variable — takeoff queue size — is an
oracle: it is defined over an interval ending at the subject's own take-off.
Its causally-clean replacement, counting aircraft that have pushed back but not
taken off, fails a second test: it subtracts other movements' taxi times, which
are the hidden target in the ranking months. Neither is usable.

What is usable is reconstructing the queue by forward simulation, after
Simaiakis & Balakrishnan (2016). Fit a runway service rate conditioned on
configuration, then walk each airport-day's departure sequence:

    estimated_takeoff = max(off_block + unimpeded_taxi, previous_takeoff + service_interval)

Queue position falls out as the number of aircraft ahead on that runway when
this one reaches the threshold. The inputs are off-block time, runway and stand
— all given — and no realised taxi time is touched, so it satisfies both
leakage conditions. `unimpeded_taxi` is what the distance matrix supplies,
which is the point: the physical layer becomes an input to the dominant term
rather than a decoration beside it.

Risks worth naming now. The service rate must be fitted on training data only
and applied identically at inference, with few enough parameters not to
memorise airport-hours. Window truncation at the month boundary leaves the first
hour of January and July with an incomplete lookback, which needs a flag rather
than a silent shortfall. And if the ranking set is a subset of movements rather
than a census, counts are systematically smaller at inference than in training —
normalise by an airport-hour baseline rather than using raw counts.

**Gate 4 — the write-up** (~10–15 h, and it must not be squeezed). The 2025
jury commended a near-complete JOAS-style report as the single strongest
artefact of the winning entry, and marked down entries that were
"predominantly ML-driven". The reports already carry most of the material.

## Ordering, and what gets cut

Gates 0 and 1 are mandatory. Gate 4 is mandatory and is reserved for, not
squeezed by, whatever precedes it. Gates 2 and 3 are the discretionary middle,
and Gate 3 is the one to drop if the schedule slips — with the ablation
recorded either way, because "the physical layer did not move RMSE once
congestion was controlled for" is a publishable finding and an honest one.

The failure mode to avoid is spending the timebox on Gate 3, arriving at
11 October with an unvalidated model and no paper. The gate exists to prevent
exactly that.

## Standing rules

- **Measure before asserting.** Coverage, leakage, routability and baselines get
  checked and written to `reports/`, not claimed.
- **Every number in a report is computed.** Hand-typed statistics drift from the
  tables beside them; this has already happened twice and been fixed twice.
- **`make licence-check` before every commit.** The repository stays GPLv3 in
  totality, and that is a checkable property rather than a habit.
- **No submission-driven feature selection.** Join normalisation is chosen by
  join rate and inspection, never by leaderboard RMSE. If a public leaderboard
  exists on any part of the ranking months, geometry decisions freeze before it
  is consulted.
