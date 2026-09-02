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

## Open questions

**Do OSM stand refs actually match `STAND_mvt`?** Coverage measures whether an
identifier *exists*, not whether it matches the challenge's string. The formats
look plausible and are reported as *shapes* rather than values in
[`../reports/stand_coverage.md`](../reports/stand_coverage.md) — bare numerics
dominate at MUC, BCN, FCO and MAD, a pier letter plus two digits at CDG, AMS and
ZRH, and IST mixes both. Shapes rather than examples because a list of real
identifiers would be an extract of OSM, and this repository carries none.

The join rate is unknown until the buckets arrive. **This is the first thing to
test on real data.** The shapes say what normalisation to expect: zero-padding
where bare numerics run to two and three digits, suffix stripping where a
trailing letter marks a split stand, and case folding throughout.

**Antalya (LTAI) is at 45% coverage.** The one airport where stand→runway
distance may not be derivable for most movements. Options, in preference order:
fall back to apron-centroid geometry; treat the airport with a separate
stand-agnostic feature set; or accept degraded features there and let the model
learn an airport-level offset. Decide once the join rate is known — the real
figure could be better or worse than 45% depending on which stands actually
appear in the movement data.

**Runway assignment is given, not predicted.** `RUNWAY_mvt` is present in the
ranking set, so taxi distance can be computed directly rather than marginalised
over a runway-assignment model. Worth confirming against the ranking file, since
it changes the feature construction substantially.

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
cannot tell the difference. It is built entirely from off-block times and
runway assignments, both given, so it satisfies both leakage conditions above.
This is the strongest idea latent in the work so far and the one most likely to
be genuinely new.

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
page names ATFM as a driver of taxi-out variability, but the challenge data does
not expose regulation identifiers, and inferring them from timestamps risks
leakage from the target. Congestion is instead approached through observable
concurrent movements.

## Leakage watch

Two conditions, both of which a congestion feature must satisfy. The first was
obvious; the second was not, and is the one that would have done real damage.

**Causality.** Every event contributing to a feature must have occurred strictly
before the aircraft's own off-block time. Any window extending past
`BLOCK_TIME_UTC_mvt` leaks the target. This is what rules out the strongest
feature in the literature: Idris's takeoff queue size Q counts departures
between the subject's own pushback and its own take-off, so its value is a
function of the answer. Ravizza and Chen's Q has the same shape. Published RMSE
figures resting on them are not operationally achievable and are not a fair
benchmark.

**Availability.** Every *field* read from those events must be one the ranking
set also exposes for test rows. This is a separate condition and the causal
version of the queue feature fails it. Counting aircraft that have pushed back
but not yet taken off means

    N(t) = #{AOBT_j <= t} - #{AOBT_j + TAXITIME_j <= t}

and `TAXITIME_j` is the hidden target. In January and July 2026 the other
movements in the window are themselves ranking-set rows, so the subtrahend is
missing exactly where it is needed. Training on true values and scoring on a
proxy would make the model lean hard on a feature that is oracle-quality in
training and noise at inference — which shows up as an unexplained gap between
validation and leaderboard, the one failure mode a solo entrant cannot afford to
debug blind.

So congestion features are built from inputs only: departures with off-block in
a trailing window, the same restricted to the subject's runway, the same
restricted to its pier. Whatever is chosen must be constructed identically on
both sides.

Two smaller traps worth writing down before they are hit. A window running
*forward* from the subject's off-block time is technically computable (off-block
times are given) but is not knowable operationally and correlates with the
subject's own queueing — forbidden. And the subject must be excluded from its
own count, with the tie-break for an exactly-equal timestamp fixed explicitly;
that is precisely the class of bug that produces a suspiciously good feature.

Every congestion feature needs an explicit as-of time and a test that asserts
both conditions.

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

- **Blocked and forward in time.** Train Jan–Oct 2025, validate Nov–Dec; and
  train Jan–Jun, validate Jul, because the test set is a January and a July.
  Never random K-fold on rows: congestion is autocorrelated over hours, so
  neighbouring departures share queue state. Block by airport-day at least.
- **A held-out-pair diagnostic.** `GroupKFold` on (stand_id, designator) within
  airport, holding out whole pairs. This is the only split that measures what
  the geometry is for, and the gap between it and the temporal split *is* the
  contribution.
- **Error stratified by pair frequency.** Bucket test rows by how often their
  pair appeared in training: 0, 1–10, 11–100, >100. Geometry should buy most in
  the first two buckets. If it does not, the physical layer is decoration, and
  the write-up should say so — that is still an honest and publishable result.
- **Per-airport RMSE reported alongside the pooled figure.** LTFM and LFPG will
  otherwise dominate, and a pooled-optimal model could be an Istanbul-optimal
  model.

The ablation ladder that settles the question: airport mean; the PRU
10th-percentile-per-(stand, runway) reference; a tuned model on one-hot
identifiers plus time features; the same plus geometry; plus congestion; and
geometry *replacing* the one-hots. The third-to-fourth step is the claim.
