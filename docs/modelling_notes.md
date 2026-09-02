# Modelling notes

Running record of decisions, caveats and open questions. Written as work
proceeds so the reasoning behind the feature set is recoverable later — for the
write-up, and for anyone reproducing the result.

## Geometry extraction

**Aprons excluded from the OSM query.** `aeroway=apron` polygons are large and
slow to serve, and contribute nothing to stand→runway routing. Revisit only if
an apron-area feature earns its place.

**Runway counts in the coverage report are segments, not runways.** OSM splits
runways into multiple ways; Schiphol reads as 10 and Zurich as 8 against 6 and 3
physical runways. The graph builder must merge segments by `ref` before treating
a runway as a routing target, and before deriving any per-runway statistic.

**Stand identifiers live in more than one tag.** `ref` dominates, but Frankfurt
has 2 stands and CDG 6 identified only by `name`. The resolver tries
`ref`, `aeroway:ref`, `local_ref`, `name` in that order — see
`taxiout.coverage.STAND_ID_TAGS`.

**Antalya draws stands as polygons; nobody else does.** 168 of LTAI's 576
parking positions are closed ways — the stand footprint rather than the nose-in
centreline. All ten other airports use centrelines exclusively. Treated as
polygons this is harmless (centroid, no lead-in); treated as lines it produces
270 m pushbacks from a perimeter and 40 stand identifiers whose duplicate
features disagree by more than 250 m. Handling it dropped LTAI's ambiguous
identifiers from 40 to 1.

## Graph construction

**Written in-repo rather than taken from `traffic`.** `traffic` (Olive, MIT) has
an `AirportGraph` doing the same class of OSM topology repair, and its licence
would permit use. Three reasons not to: the dependency is large relative to what
is needed; the route descriptors here (turn angle, backtrack, crossing counts)
are not in its API and would be bolted on anyway; and the geometric decisions
are the substance of the contribution, so having them auditable in this
repository is worth more than the code saved. It remains the natural
cross-check, and the prior art is cited rather than quietly reimplemented.

**Three OSM defects have to be repaired before anything routes.** Nodes that are
coincident but not shared (up to 166 per airport — an exact coordinate match is
ordinary way-sharing and is not counted); nodes lying on an edge they are not
connected to (7–1912 edge splits); and ways that cross on the map with no shared
vertex at all (up to 89). The third fires rarely, because mappers usually do
place a node at a runway crossing — it is kept because its absence is silent,
and would make a runway reachable only where somebody was thorough.

**Reaching a fixed point took two attempts, and the failure was instructive.**
Without a minimum-length guard the repair split off half-edges of a few
millimetres in densely mapped areas; their midpoints then sat on their
neighbours, which split those. There was no fixed point — Zurich settled into a
cycle of fifteen splits per round indefinitely, so the output was a function of
how many rounds were permitted rather than of the airport, and the claim that it
converged was simply false. Split positions are now bounded away from both ends
in metres, and no (node, edge) combination is considered twice. All eleven
airports now converge, in two to twenty-five rounds, and the build records which
and whether.

**Splitting reveals pavement drawn twice.** Once two versions of a segment share
endpoints they are one edge, so the total shrinks — most at the densely mapped
airports.
This looked like a bug for long enough to be worth recording that it is not —
component counts are unchanged, and the amount is now reported as
`pavement_deduped_km` rather than being left to look like lost data.

**Turn angle has to be measured on a simplified path, and the first fix was
wrong.** Vertex density is a property of the mapper, not the pavement: LSZH
carries a vertex every 9 m, LTFM every 39 m, so cumulative heading change over
raw vertices counts tracing noise and is not comparable across airports.
Fixed-step resampling removed that but introduced a worse artefact — it discards
the original vertices, so any feature narrower than the step is cut by a chord
and an S-bend inside one step cancels itself. The statistic then depended on
which end of the path it was measured from, by up to 150 degrees on real
routes, which is not a property a total turning angle can have.

Douglas–Peucker simplification has neither problem: it keeps every vertex
deviating from its own chord by more than the tolerance, so jitter disappears,
real geometry survives at any density, and the result is exactly symmetric under
reversal. The tolerance (3 m) was picked from a sweep — every airport falls
steeply while noise is still being removed, then flattens; six of seven are flat
by 2.5–3 m. Zurich keeps declining past 5 m and so remains partly
scale-dependent, which is recorded rather than smoothed over.

**The runway penalty was measured, not tuned.** Runway pavement is traversable
at 4x length. Median taxi distance barely moves across a twentyfold range of
that parameter, while the amount of *other* runways' pavement the routes use
collapses — which is the behaviour the penalty exists to suppress. Turn angle
does move across the full range, by around a fifth, because at a penalty of 1
routes run along runways and runways are straight; above 4 it is flat. The full
sweep, regenerated rather than quoted, is in
[`../reports/penalty_sensitivity.md`](../reports/penalty_sensitivity.md).

**Thresholds are attached to the graph, not matched to a junction.** Matching
the nearest runway/taxiway junction was the first approach and it failed
instructively: OSM often stops a taxiway at the runway edge rather than joining
the centreline, so a runway can have no junction near one end. Madrid's 32L
departures were being routed 1.5 km up the runway to reach a threshold 280 m
from the stand. Attaching directly leaves the taxi-round-or-backtrack question
to the cost function, where the penalty can express it, and puts the backtrack
in `backtrack_m` where it can be seen.

## What the challenge data turned out to contain

Settled on 2 September 2026, the day the data landed, by
[`../reports/data_review.md`](../reports/data_review.md); the figures below
are quoted from it and it is the one that is regenerated.

**The OSM stand refs do match `STAND_mvt`, after a staged normalisation.**
Exact matching gives 57–99% by airport. Folding case and whitespace, then
stripping leading zeros in digit runs, then one trailing letter after a
digit — each stage applied to both sides and tried only for rows the previous
stage left unmatched — gives 87% (Munich) to 99.9% (Istanbul), 96% of
departures overall, in both the training year and the ranking set. Each
stage was explained by inspecting what it fixed, never by what it did to a
validation error: the whitespace is inside OSM's refs at Frankfurt, Schiphol's
movement data zero-pads where OSM does not, and Munich and Istanbul map split
stands as lettered halves where the movement data names the whole. The
remainder is mostly stands OSM does not have — whole numeric blocks at Munich
and Barcelona — and those rows fall back to airport-level features.
Implementation and the ambiguity bookkeeping it adds (a coarser key can fold
several OSM identifiers onto one; the collapsed cell records how many and how
far apart) are in `taxiout.join`.

**Antalya is not in the data.** LTAI has no rows in training or ranking,
though the organisers list it and the row total matches theirs. The 45%
coverage question is moot; the geometry stays in the pipeline unused.

**Runway assignment is given.** `RUNWAY_mvt` and `STAND_mvt` are present for
every ranking departure, so the matrix applies directly and nothing is
marginalised.

**The ranking set is a census where it is present at all.** January 2026
covers all ten airports at the same daily volumes as January 2025; July 2026
covers Frankfurt, Heathrow and Schiphol only. Congestion counts need no
coverage normalisation, and those three airports are 53% of the test rows.

**The off-block time is blanked, and that changes the feature rules.** See
*Leakage watch* below, which was rewritten for it.

## Open questions

**Do departures taxi to the threshold?** The matrix gives full-length
distances. Intersection departures are routine and shorten the taxi materially,
and nothing in the OSM data says which runways are used that way.
`backtrack_m` isolates the part an intersection departure would not fly, so the
question can be settled empirically once movement data exists — if taxi times
for a runway sit systematically below the full-length prediction, that is the
signature.

**A handful of designators have no mapped taxiway near their threshold**, so the
router reaches them along the runway itself. The current set and the one case
that varies across stands are listed in
[`../reports/taxi_routing.md`](../reports/taxi_routing.md) and regenerated with
it rather than repeated here, because this list changed twice while the geometry
was being fixed and a hand-copied version was wrong both times. The distinction
that matters: where the added distance is the same for every stand it is a
constant offset on one designator, which any model knowing the runway absorbs;
where it varies, it distorts the distances between stands, which is the signal
the feature exists to carry.

**A snapped stand loses turns, not just distance.** Where OSM has no apron
lead-in, the stand is attached by a straight line and `snap_m` records the gap.
The distance error is small — under 2.2% of `taxi_m` at every airport — and that
framing understates the problem. A straight line contributes zero `turn_deg`,
zero `sharp_turns` and zero `junctions`, so the apron manoeuvre disappears from
exactly the descriptors the literature says are nearly as predictive as
distance. At LFPG that is 265 of 768 stands, clustered by apron rather than
scattered.

That is a mechanism, not a measurement, and the obvious check does not support
it: comparing turn angle per kilometre for snapped against unsnapped cells, six
of ten airports move the *wrong* way, LFPG among them. The counterfactual is
genuinely unobservable — the apron manoeuvre that is missing was never mapped —
so the honest position is that the bias is expected on physical grounds and the
available proxy cannot detect it. `snap_max_m` is carried into the matrix so
those cells can be identified and the interaction tested against the target,
which is the only test that will settle it.

**The committed matrices were built from an unpinned OSM snapshot.** Pinning is
now available — `scripts/fetch_osm.py --date 2026-01-01T00:00:00Z` issues an
attic query and records the pin inside the cache — but the shipped cache predates
it. Two costs remain until it is rebuilt: a mild anachronism where a stand or
taxiway was remapped mid-period, and, worse for a challenge judged on
reproducibility, a rebuild months from now disagreeing with the committed
matrices with nothing to explain why. Not every Overpass mirror answers attic
queries, so a pinned fetch may exercise the mirror rotation harder.

**Should aprons be re-added to the Overpass query?** They were excluded as
slow and useless for routing. The routing measurements are the first evidence
against that: at EGLL every node-form parking position sits well over 100 m from the
nearest mapped taxiway, and at LTAI the gap has a tail beyond 400 m — in both cases
because the apron lead-in is not mapped as a taxilane. Apron polygons might
bridge those gaps, but routing across an area needs a navmesh, which is a large
step up in complexity for a timeboxed probe. Recorded, measured, deferred; the
straight-line `snap_m` allowance is the interim answer and is reported per stand
so it can be filtered rather than trusted.

## Where the geometry could go next

Ordered by expected value, and recorded now because the graph already contains
what each of them needs.

**Shared-corridor load.** The graph knows which edges every (stand, runway)
route uses. Counting how many *other* departures in a trailing window are on
routes that share edges with this one gives a congestion measure that is
geometric rather than aggregate — two aircraft leaving opposite ends of the
field at the same moment do not interact, and a count of concurrent movements
cannot tell the difference. It was to be built from off-block times and runway
assignments; the off-block times turned out to be blanked, so it is built from
take-off times and runway assignments instead — the other departures on
overlapping routes in a window before the subject's take-off — which the
ranking set does expose (see *Leakage watch*). This is the strongest idea
latent in the work so far and the one most likely to be genuinely new.

**Intersection-departure offsets.** `backtrack_m` isolates the along-runway
component only where the mapping is defective. For a correctly mapped runway
there is currently no feature expressing how much runway an intersection
departure skips. A per-(runway, entry junction) offset table is derivable from
the same graph and would make the question answerable from data.

**Runway configuration and its persistence.** Which designators appear in the
preceding hour is a strong regime variable, computable from other movements'
given fields alone. "Minutes since the configuration last changed" is the
second half of it: taxi times are elevated after a switch. This also resolves
the arrival-versus-departure designator problem that inflates the unweighted
medians in the routing report.

**Pushback orientation.** The stand centreline bearing against the initial route
bearing separates a straight-ahead push from a 180-degree turn. Both are already
in the geometry and neither is currently used.

**Holding-point convergence.** How many stands' routes funnel through the same
holding point is a static proxy for where queueing accrues.

## Deliberate non-goals

Not attempting to reconstruct the ATFM slot or regulation state. The rationale
page names ATFM as a driver of taxi-out variability, but the challenge data
does not expose regulation identifiers or slot times. What it does expose is
the delay a regulation causes before departure: the flight side carries the
initial off-block estimate (`IOBT_flt`), the filed one (`EOBT_1_flt`) and the
last one the Network Manager held (`LOBT_flt`), all present for test rows,
and `LOBT − IOBT` is how far the planned off-block moved before the aircraft
left — from ATFM, from the airline, from the airport, without saying which.
It is used as a regime indicator, and the write-up says it is a proxy for
pre-departure delay of any cause, not for ATFM specifically. Congestion itself
is approached through observable movements.

## The model, stated as a decomposition

Fixed on review, 2 September 2026. The organisers describe the use of a
taxi-out estimate as post-operations analysis: identifying constrained
intervals and measuring "excess fuel burnt/CO2 production compared to
unconstrained status"
([rationale](https://ansperformance.eu/study/data-challenge/dc2026/rationale.html)).
That is a decomposition, and the model is stated as one:

    taxi-out = unimpeded(geometry) + excess(congestion, regime) + record-quality term

**Unimpeded.** The time an aircraft would take from this stand to this runway
with nothing in its way. The PRU's additional-taxi-out-time indicator defines
it empirically, as the 10th percentile of observed taxi-out per (stand,
runway) pair — which exists only for pairs that have been observed often
enough. The geometry supplies the same quantity as a function of route
distance and shape, and so extends the reference to pairs seen rarely or
never. Distance becomes time through a speed fitted on training data; the
route descriptors (turn angle, junctions, junction-free running) modulate it,
per Ravizza et al.

**Excess.** Everything above unimpeded: runway queueing, surface congestion,
de-icing and low-visibility procedures, pre-departure holds. Built from what
the ranking set exposes about surrounding traffic — take-offs, arrivals'
landings, on-blocks and taxi-in times, schedules, runway configuration —
and, as regime indicators, weather flags per airport-hour and the flight
plan's own delay signal (see *Deliberate non-goals*).

**Record-quality term.** A separate component for rows the data flags as
record-matching artefacts, whose recorded taxi-out no aircraft produced. It
exists because the stated scoring rule counts those rows; it is kept apart
so it cannot be mistaken for either of the other two (*Tail handling*).

The ablation ladder (*Validation*) is the decomposition read bottom-up: each
rung adds one term, and the held-out-pair split measures the one claim that
distinguishes this entry — that the unimpeded term carries over to pairs the
model never saw.

## Leakage watch

Rewritten on 2 September 2026 once the ranking set had been read, because the
version written before it assumed the wrong anchor. What the ranking set
exposes for a departure — measured in
[`../reports/data_review.md`](../reports/data_review.md) — is its take-off
time (`MVT_TIME_UTC_mvt`, to the second), its scheduled time, the flight
plan's estimated off-block times, and its stand and runway. `BLOCK_TIME_UTC_mvt`
is blanked along with the target, for every departure in the ranking months.

That settles what kind of problem this is. Operationally, taxi-out is predicted
at pushback, when the actual off-block time is known and the take-off time is
not. The challenge poses the reverse: the take-off time is given and the
off-block time is what is missing. It is a post-hoc estimation task — and
that is the use the organisers state, "post-operations analysis"
([rationale](https://ansperformance.eu/study/data-challenge/dc2026/rationale.html)),
not a departure-time forecast. The rules below are written for it; the
write-up states both framings and builds to the challenge's. The alternative
— anchoring only on schedules and filed estimates and reading no actual
take-off time at all — is principled for a pushback-time forecaster and was
considered and set aside: it answers a question the challenge did not ask,
and forgoes the strongest signal the data exposes.

**One rule with two halves.** A feature may read any field that the ranking
set exposes for the rows it reads from, and no field that it hides or that is
a noisy copy of one it hides. Concretely:

- *Hidden:* `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt` of any **departure**
  in the same month — the subject's or a neighbour's. This is the condition
  that bites. A trailing window of other departures' pushbacks, the feature
  the first version of this plan called Gate 2, is unavailable at inference
  for exactly the reason the queue-size feature was: the field it counts is
  blanked for every departure in January and July 2026. So is anything that
  subtracts other departures' taxi times. Both were oracle-quality in training
  and would have been noise on the leaderboard.
- *Excluded by decision:* `AOBT_3_flt`. It survives in the ranking set, but it
  is a minute-resolution Network Manager off-block that sits a median 51 s
  from the blanked field with an interquartile range of five minutes; as a
  single predictor it is worth no more than the filed EOBT lag. Not an oracle,
  but a noisy copy of a hidden field, and an entry judged on originality has
  nothing to gain from it. The organisers say they monitor for attempts to
  exploit the ranking process; this is the one place it could be alleged.
- *Available, and used:* every departure's take-off time, scheduled time and
  filed estimates; every arrival's landing time, on-block time and taxi-in
  time, which the ranking set gives in full; stand, runway, aircraft type,
  operator, and the flight-side categoricals; and whether the flight-side
  join exists at all, which turns out to matter.

The constants are in `taxiout.challenge` (`HIDDEN_FOR_TEST_DEPARTURES`,
`EXCLUDED_BY_DECISION`, `FORBIDDEN_FEATURE_INPUTS`) so that the feature layer
can be tested against them rather than against a paragraph.

**What congestion is built from, then.** Take-offs from the subject's runway
and from all runways in trailing windows before its own take-off — the
runway's realised throughput, which under post-hoc rules is legitimate and
under operational rules would not be. Arrivals: landings and on-blocks in the
same windows, and the taxi-in times of arrivals that reached their stand in
the window, which is a direct measurement of surface congestion available on
both sides. Scheduled departures in windows around the subject's schedule,
which is demand and is causal under either framing. The runway configuration
in force, from other movements' designators. Each is constructed identically
on both sides, with the subject excluded from its own count and the tie-break
for an equal timestamp fixed explicitly.

**What is still forbidden even under post-hoc rules.** Reading the target of
any departure; reading `AOBT_3_flt`; and recovering the blanked off-block time
from outside the challenge data — ADS-B surface tracks for the ranking months
would do it, and using them would be reconstructing a hidden variable through
a side channel, whatever their licence. External data is for calibrating the
physical layer on 2025 and nothing else.

**The literature's benchmark figures do not transfer.** Idris's takeoff queue
size counts departures between the subject's pushback and its take-off, a
function of the answer; published RMSEs resting on it are not a fair
comparison in either framing.

Every congestion feature needs an explicit as-of time and a test that asserts
it reads nothing in `FORBIDDEN_FEATURE_INPUTS`.

## Tail handling

Decided 2 September 2026, and revised the same day once the term was built
and the tail profiled for it. The figures below are quoted from
[`../reports/tails.md`](../reports/tails.md), which is the one that is
regenerated.

The target has a heavy tail: under 0.1% of training departures carry half
the airport-mean squared error. The data review read it as record-matching
artifacts — a block time from an earlier day matched to a later take-off —
and flagged it by two fields the ranking set exposes, no flight-side record
and a take-off hours after schedule. Building the term found a different
mechanism behind most of it. The departures recorded above two hours are
82% Fiumicino, and for nine in ten of those the recorded off-block time sits
within a minute of the *scheduled* departure time — over eight in ten within
five seconds — while the take-off is hours later. The block time was
defaulted to the schedule, so the recorded taxi-out is take-off minus
schedule: the lag, which every ranking departure exposes. The day-shift is
real but small, a dozen rows a year, a day plus an ordinary taxi, in the same
cells. No other airport's cells carry either at a material rate.

The scoring rule counts these rows. The ranking page says teams "will be
ranked based on the best Root Mean Square Error (RMSE) achieved across their
submissions", and the ranking script "will issue an error if ... any rows are
missing", so every template row is scored and none is excluded. Five such
rows predicted as ordinary taxi-outs turn a 300 s model into a 512 s one.

The decision is to model them explicitly, as the third term of the
decomposition, and to say what the term is. `taxiout.tails` measures, per
(airport, flight-record presence, lag band) cell on training rows, how often
the block time was defaulted and what the remaining rows average, and
predicts for a row in a covered cell `share × lag + (1 − share) × rest_mean`
— the conditional mean, computed knowing the mechanism. A cell is covered
when it has at least five rows and a defaulted share of at least five
percent; at the full-year fit that is three cells, all at Fiumicino, and 107
of the 215,876 ranking rows, all in January. Every other row falls through
to the physical model, which is fitted with the covered rows excluded and
validated on every row, with RMSE reported with and without the term.

Two forms were measured on the forward splits and set aside. A plain cell
mean with thin cells pooled over airports — the first proposal — carries one
airport's artifact rate to airports that have none and is the worst form
tried on every split. The plain cell mean without pooling loses 12 s on
December and 33 s on July to the mixture, because the mixture predicts each
artifact's own value rather than the cell's average of them. An extra lag
edge at two hours helped one split and hurt the other and is not adopted.
Leaving thin and clean cells to the physical model rather than pooling is
the choice with an exposure: an airport whose feed starts defaulting block
times in 2026 is not covered, and no rule fitted on 2025 could cover it. The
report states the arithmetic.

It must be presented for what it is. The term is the RMSE-optimal
prediction under the stated scoring rule for rows whose recorded value no
aircraft produced; it is a conditional mean, not a taxi time, and the paper
says so in those words, before a reviewer does, gives the artifact analysis
its own section rather than a footnote, and reports the physical model
without the term wherever it reports it with. The alternative, predicting
only physically plausible taxi-outs and arguing the point in the paper, was
set aside: it would place mid-table regardless of model quality, and the
argument can be made just as well in a paper that also scores.

## Validation, before any model exists

**The geometry block is an injective encoding of (stand, runway).** Every
feature in the matrix is a deterministic function of that pair, and the vectors
are unique for every cell at EDDF, LFPG and EHAM. So the block adds nothing that
a saturated one-hot(stand) x one-hot(runway) interaction does not already
contain. Its entire value is that it behaves sensibly on rare and unseen pairs
and uses far fewer effective parameters.

That value is invisible to any cross-validation that puts the same (stand,
runway) pair on both sides of the split. With roughly a million departures a
year against a few thousand used cells, there are hundreds of training rows per
cell — ample to fit a per-cell mean and read `taxi_m` as a continuous stand
identifier. A random K-fold would show the geometry "working" while testing
nothing, and the README's central claim would go unexamined.

The validation therefore needs to be, at minimum:

- **Blocked and forward in time.** Two splits, fixed on 2 September 2026 once
  the ranking set's shape was known. Train January–November 2025, validate
  December, at all ten airports: the winter proxy for a January test month
  that the training year cannot hold out forward (its only January is its
  first month). Train January–June, validate July, reported for Frankfurt,
  Heathrow and Schiphol separately, because those three are the whole of the
  July test set and 53% of the test rows. Never random K-fold on rows:
  congestion is autocorrelated over hours, so neighbouring departures share
  queue state. Block by airport-day at least.
- **A held-out-pair diagnostic.** `GroupKFold` on (stand_id, designator) within
  airport, holding out whole pairs. This is the only split that measures what
  the geometry is for, and the gap between it and the temporal split *is* the
  contribution.
- **Error stratified by pair frequency.** Bucket test rows by how often their
  pair appeared in training: 0, 1–10, 11–100, >100. Geometry should buy most in
  the first two buckets. If it does not, the physical layer is decoration, and
  the write-up should say so — that is still an honest and publishable result.
- **Per-airport RMSE reported alongside the pooled figure.** The test set is
  weighted towards the three July airports; a pooled-optimal model could be a
  Frankfurt–Heathrow–Schiphol-optimal model.
- **RMSE with and without the flagged artifact rows**, per *Tail handling*
  above, so the physical model's contribution is visible underneath the
  record-quality component's.

The ablation ladder that settles the question: airport mean; the PRU
10th-percentile-per-(stand, runway) reference; a tuned model on one-hot
identifiers plus time features; the same plus geometry; plus congestion; and
geometry *replacing* the one-hots. The third-to-fourth step is the claim.
