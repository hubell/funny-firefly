# Plan to the deadline

Written 2 September 2026 before the challenge data arrived, and revised the
same day once it had, because the data changed the plan in three places: there
are ten airports rather than eleven, the ranking set blanks the off-block
time of every departure, and the target's tail decides the metric. The
original text is kept where it still holds; the revisions are marked. The
challenge closes **11 October 2026**, so this is a five-and-a-half week plan,
and the expensive items are gated rather than committed to.

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
- A unit-test suite with no network and no challenge data. Deterministic build.

*Added 2 September, after the data:*

- Gate 0 passed and written up — [`../reports/data_review.md`](../reports/data_review.md).
  Runways join at ~100%, stands at 87–99.9% by airport after a staged
  normalisation; the target is not cleanly reconstructible from what the
  ranking set exposes; runway and stand are present for every test row.
- Join layer (`taxiout.join`), challenge-data access and the hidden-field
  constants (`taxiout.challenge`), with tests on synthetic frames.

Not done, and the honest list: nothing has been validated against a taxi time.
Every number so far describes geometry and data, not prediction.

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

## What was meant to be built before the data, and was not

The original plan listed four things as buildable before the buckets arrived:
a leakage test harness, the validation machinery, the unimpeded taxi-time
structure, and the three licensing tasks above. The data arrived the same
day the plan was written and none of them exists yet. Recorded rather than
quietly folded in, because a reader of the history should not infer they were
done: the first three are now Gate 1 work, with the harness asserting against
`taxiout.challenge.FORBIDDEN_FEATURE_INPUTS` rather than against one column;
the licensing tasks sit in the freeze window below, the synthetic airport
first if time allows and the Zenodo deposit regardless.

## When the data lands

**Gate 0 — does the join work?** (~4–6 h, and it blocks everything.)

Test the OSM identifier against `STAND_mvt` and the designators against
`RUNWAY_mvt`. The shapes in the coverage report say what normalisation to
expect: zero-padding, suffix stripping, case folding. Also confirm
`RUNWAY_mvt` and an off-block timestamp are exposed for ranking rows — if the
runway is not, the distance matrix, the corridor feature and the configuration
feature all fail together, and the fallback is to marginalise over a
runway-assignment model.

*Outcome, 2 September.* The join works: runways at ~100%, stands at 87%
(Munich) to 99.9% (Istanbul) of departures, 96% overall, with the three
normalisations the shapes predicted and one they did not (whitespace inside
OSM refs at Frankfurt). Runway and stand are present for every test row.
Antalya is not in the data at all, so its fallback is moot. The off-block
timestamp is **not** exposed — see Gate 2 — and `AOBT_3_flt`, which is, was
measured and excluded. Everything in
[`../reports/data_review.md`](../reports/data_review.md).

**Gate 1 — a working end-to-end submission** (~10–15 h). Baseline, geometry
features, time-of-day, one-hot identifiers, and — *added 2 September* — the
record-quality component for the artefact rows, because without it no
submission's RMSE means anything (*Tail handling* in
[`modelling_notes.md`](modelling_notes.md)). Submit something valid early; a
scored mediocre entry beats an unscored good one.

**Gate 2 — cheap congestion** (~4–6 h). *Rewritten 2 September.* The
original plan counted other departures' off-block events in a trailing window.
The ranking set blanks the off-block time of every departure, so that feature
is unavailable at inference — it fails the same availability test the plan
had already applied to the queue-size feature, and nobody noticed until the
file was opened. What the ranking set does expose for other movements, and
what Gate 2 is now built from:

- take-offs from the subject's runway, and from all runways, in trailing
  windows before its own take-off — realised runway throughput;
- arrivals' landings and on-blocks in the same windows, and the taxi-in times
  of arrivals that reached their stand in the window, which is a direct
  surface-congestion measurement the ranking set gives in full;
- scheduled departures in windows around the subject's schedule — demand;
- the runway configuration in force, from other movements' designators.

The anchor is the subject's take-off time, which the challenge gives and the
operational problem would not; the framing and its consequences are set out
under *Leakage watch* in [`modelling_notes.md`](modelling_notes.md).

**Gate 2b — weather as a regime indicator** (~4–6 h; *added 2 September on
review*). The organisers name airport procedures and load among the drivers,
and the ranking set's first week of January 2026 at Schiphol has the
signature of winter operations. METAR from the Iowa State Mesonet archive is
public domain and covers all ten airports; it enters as per-airport-hour
regime flags — visibility band, precipitation, freezing conditions — not as a
continuous driver, per the literature position in
[`prior_art.md`](prior_art.md). The fetch is scripted and cached exactly as
the OSM one is, because an external dataset that a reader cannot re-fetch
fails the reproduction clause of the prize rules; it is documented as an
additional open dataset in the README.

**Gate 3 — the queueing layer, only if Gate 2's ablation says congestion is
binding** (~20–30 h). This is the piece where the geometry stops being a side
feature. *Revised 2 September:* the forward simulation as first written —
walking each day's departure sequence from off-block times — has no off-block
times to walk in the ranking months. Two replacements are possible, and the
choice is deferred until Gate 2 has said whether congestion is binding at all.
The first is to run the simulation on *scheduled* off-block times, which are
available and causal, accepting that a schedule is a poorer start point than
a pushback. The second inverts it: with every take-off time known, the runway's
realised service sequence is observed exactly, and the queue a departure
joined can be estimated backwards from the throughput ahead of it and the
unimpeded taxi the distance matrix supplies. The second is more novel and more
dependent on post-hoc framing; the first is closer to Simaiakis & Balakrishnan
(2016) and easier to defend. Either way the inputs are stand, runway,
schedule and other movements' take-offs, and no realised taxi time of any
departure is touched. `unimpeded_taxi` is what the distance matrix supplies,
which is the point: the physical layer becomes an input to the dominant term
rather than a decoration beside it.

Risks worth naming now. The service rate must be fitted on training data only
and applied identically at inference, with few enough parameters not to
memorise airport-hours. Window truncation at the month boundary leaves the first
hour of January and July with an incomplete lookback, which needs a flag rather
than a silent shortfall. The census question is settled: January 2026 covers
all ten airports and July 2026 three, each in full, so raw counts are
comparable between training and inference within an (airport, month).

**Gate 3b — validating the physical layer against surface tracks** (~10–15 h,
gated on OpenSky access being granted; *added 2 September, scope fixed on
review the same day*). Historical ADS-B for 2025 at the ten airports — never
the ranking months — would test four things the matrices assume: whether OSM
shortest paths are the routes flown, what unimpeded speed looks like per
airport and route segment, which runways take intersection departures and
from which entry, and where queues accrue.

Its outputs are **diagnostics, not model inputs**. The prize rules require
additional datasets to be "openly available under an open source license";
OpenSky's historical data is research-use and application-gated, which is
not that. So nothing derived from it enters a feature or a fitted parameter.
What it yields goes into the paper as validation of the geometry — route-match
rates, speed distributions, evidence on intersection departures — and the
model stands on OSM, OurAirports, METAR and the challenge data alone. If
OpenSky confirms in writing that derived aggregates may be published under an
open licence, that decision can be revisited; until then it is not. Access is
pending; the coverage audit comes first and goes to `reports/`, and if access
does not arrive in time this gate is dropped without touching the others.

**Gate 4 — the write-up** (~10–15 h, and it must not be squeezed). The 2025
selection committee's published reviews
([outcome page](https://ansperformance.eu/study/data-challenge/dc2025/outcome.html))
called the winning entry's "near-complete JOAS-style paper" a major strength
and its methodology "strongly informed by domain knowledge rather than relying
exclusively on machine-learning techniques"; the third-placed entry was
"predominantly machine-learning–driven, with less emphasis on domain-specific
feature engineering". The same page states that only repositories scoring
below an RMSE of 400 were reviewed, and that reaching the final required
"the (documented) code and docs in a Github repository". The reports already
carry most of the material. The paper's outline gives the artefact analysis
of the target its own section — the 2025 reviews singled out "a thoughtful
analysis of noise in the data" for praise — and states the record-quality
term for what it is: the prediction the stated scoring rule makes optimal for
rows whose recorded value no aircraft produced, kept apart from the model.

## Ordering, and what gets cut

Gates 0 and 1 are mandatory. Gate 4 is mandatory and is reserved for, not
squeezed by, whatever precedes it. Gates 2 and 3 are the discretionary middle,
and Gate 3 is the one to drop if the schedule slips — with the ablation
recorded either way, because "the physical layer did not move RMSE once
congestion was controlled for" is a publishable finding and an honest one.

The failure mode to avoid is spending the timebox on Gate 3, arriving at
11 October with an unvalidated model and no paper. The gate exists to prevent
exactly that.

**Freeze: 4 October 2026.** Fixed on review, 2 September. Modelling stops on
that date whatever gate it is in; the final week is for the paper, the
repository the organisers will fork, and the final submission. Inside that
week, in order: a pinned rebuild of the OSM cache and every geometry report
(`scripts/fetch_osm.py --date`), with the pin recorded in each report's
header, because the committed figures come from an unpinned fetch and a
rebuild in 2027 would otherwise disagree with them for no stated reason; the
Zenodo deposit of the matrices under ODbL; the paper; a clean-checkout
reproduction of every report from the committed instructions; the final
submission and its documentation.

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
