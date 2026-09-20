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

*Added 2 September, Gate 1:*

- Feature frame (`taxiout.features`) with a `FEATURE_SOURCES` registry naming
  the raw columns behind every column, asserted against
  `FORBIDDEN_FEATURE_INPUTS` both as a declaration and by poisoning the
  forbidden fields and checking the frame does not move.
- Unimpeded term (`taxiout.unimpeded`): the PRU 10th-percentile reference and
  the geometric version fitted to reproduce it, with the calibration check
  reported rather than assumed.
- Validation harness (`taxiout.validation`) for the three splits below, and
  the ablation ladder (`taxiout.models`, `scripts/run_baselines.py`) written
  up in [`../reports/baseline.md`](../reports/baseline.md).
- Submission writer (`scripts/make_submission.py`) over Stream B's
  `taxiout.submission`, to a gitignored path the script verifies with git.

Not done, and the honest list: the excess term does not exist yet, so every
figure in the baseline report is a model with no congestion in it being asked
to predict a quantity that is mostly congestion. Gates 2 and 2b are what close
that.

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

*Outcome, 2 September.* The ladder runs end to end and is written up in
[`../reports/baseline.md`](../reports/baseline.md), which the generator
regenerates in under three minutes. Four things it settled, two of them
against the geometry.

The **calibration check** is the weak result. The geometric unimpeded term
reproduces the PRU reference on seen pairs to 252 s RMSE against a median
reference of 728 s — a 35% relative error, with the middle 80% of pairs
between 0.69 and 1.53 times the reference, and a per-airport bias that
changes sign between Frankfurt and Istanbul. A per-airport speed and five
pooled shape coefficients are a coarse instrument, and the report says so.
What the geometry buys has to be argued from the held-out-pair split, not
from this table.

On the **forward splits the geometry buys nothing**: December RMSE 442.2 s
without it and 442.1 s with, July 677.2 against 676.4. That is the result
*Validation* predicted before anything was fitted — a saturated identifier
model has seen every pair it is asked about — and it is the reason a random
K-fold would have been worthless.

On the **held-out-pair split it buys something, and the number is small**.
Over the rows the record-quality term leaves to the physical model, the
identifier model's RMSE rises 35.8 s more on a pair it has never seen than
the airport mean's does; adding the geometry block cuts that to 21.4 s. So
the physical layer recovers about two fifths of what memorising the pair was
worth. Real, measured, and not a headline: 9.7% of the top rung's splits fall
in the geometry block against 36.9% on the stand identifier alone.

The **frequency stratification is the cleanest version of that result**, and
it falls exactly where the validation notes said it should. December
validation RMSE, over the rows the record-quality term leaves alone, with and
without the geometry block: 347.3 against 327.3 for pairs never seen in
training, 446.1 against 426.1 for pairs seen one to ten times, 373.7 against
369.3 for eleven to a hundred, and 316.4 against 316.7 — nothing, and very
slightly worse — above a hundred. The geometry buys 20 s where the identifier
has nothing to remember and nothing where it has plenty, which is the
mechanism the entry claims and not merely its sign.

The **record-quality term dominates the levels**, as the data review said it
would: December RMSE 488.0 s for the airport mean, 391.1 with the term on
top, 380.0 excluding the rows it covers. Fiumicino alone runs at 1,194.5 s
in December and 409.1 excluding those rows.

One environment note, because it cost an hour: pyarrow 19.0.0 cannot read the
challenge parquet files at all (`Repetition level histogram size mismatch`, a
page-index bug fixed in 19.0.1). The floor in `pyproject.toml` is still
`>=15.0` and should be raised.

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

*Outcome, 3 September.* Built as listed, each block a rung, plus the
shared-corridor load the notes had named the strongest idea latent in the
geometry. The construction is written up in
[`../reports/congestion.md`](../reports/congestion.md) — the leakage audit
by phase, the same counts on both sides a year apart, the corridor's
degeneracy test — and what each block is worth is the ladder in
[`../reports/baseline.md`](../reports/baseline.md). Four things it settled.

**Congestion, as the ranking set exposes it, is worth about seventeen
seconds in December and five in July.** Over the rows the record-quality
term leaves to the physical model, the five blocks together take the
December RMSE from 321.5 s to 304.5 s and the July RMSE from 405.6 to 400.3;
on the three July airports, 323.1 to 316.7. Arrivals and demand carry it —
the arrivals' taxi-in times and the scheduled departures around the subject
are worth about five seconds each in December — with throughput about four;
the runway configuration and the corridor buy nothing more on any split.
Congestion is real and it is not large, and the write-up says so in those
terms.

**The shared corridor is a distinct column the model cannot use.** It
correlates with the plain take-off count at 0.29 to 0.56 per airport, so it
is not throughput renamed; and after throughput, arrivals, demand and
configuration it moves the December RMSE by +0.4 s, the July by −1.1 s, and
the held-out-pair contrast by −0.6 s [−1.0, −0.2]. Reported as it fell. The
idea was worth building because it is the one thing here no other entry has;
that the booster finds nothing in it is a finding about the routing model —
a full-length route with no notion of where in the window the other aircraft
was — and not a reason to drop it from the paper.

**None of it moves the held-out-pair claim, which is what it should do.**
Every congestion block's contrast on held-out pairs sits within 1.3 s of
zero: congestion is not a property of the pair, so a block of it buys the
same on a pair the model has seen as on one it has not. The claim contrast
stays +14.5 s [+13.0, +16.3].

**One column the booster splits on freely and learns nothing from.** The
configuration set, a categorical with 12 to 533 levels per airport, takes
12.3% of the top rung's splits and is worth nothing on any split — the
signature of a high-cardinality categorical memorizing airport-hours. The
count of runways in use and the time since the last change, which are what
the set was meant to carry, stay; the write-up notes the set as a feature
that should not be trusted for its split share.

Two smaller things. The parquet files carry microsecond timestamps, which
the window code first read as nanoseconds — a whole year became thirty
thousand "seconds" and every hour-long window spanned six weeks — and the
unit tests, built at nanosecond resolution, never saw it; the sanity check
on the real year did, and a test now feeds microseconds. And the same
calendar month a year apart has the same shape on both sides to within a
count: 36 take-offs an hour at Frankfurt in both Januaries, 39 at Heathrow,
which is the census the data review promised being kept.

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

*Outcome, 3 September.* Fetched, cached and attributed as planned:
`scripts/fetch_metar.py` records the request and the Mesonet's stated terms
— the materials "are in the public domain and may be used freely by anyone
for any lawful purpose" — in a manifest beside the data, every station has
its own report every hour, and the regime table's coverage and flag rates are
in [`../reports/congestion.md`](../reports/congestion.md). Three flags per
airport-hour and nothing continuous; present weather in the vicinity is
dropped, because showers eight kilometres out are not rain on the taxiway.

**Weather is worth ten seconds in December and nothing in July, and it is
worth it where winter is.** The rung takes the December RMSE, over the rows
the record-quality term leaves alone, from 304.5 s to 294.4 s, with the gain
at the airports whose January is a winter: Munich 391 to 357 s, Paris 374 to
352, Schiphol 221 to 202, and Frankfurt, Zurich and Istanbul within a few
seconds. July moves from 400.3 to 400.1 s, and the three July airports from
316.7 to 315.9, because a July has no regime to flag. The held-out-pair
contrast is +0.2 s [−0.4, +0.8] — weather is not a property of the pair —
and the freezing flag takes 2.1% of the top rung's splits, the only weather
column in the top fifteen.

The regime the ranking set will be scored in is the one the training year
saw less of: January 2026 froze for 48% of Frankfurt's hours against 30% in
January 2025, 65% of Zurich's against 41%, 70% of Munich's against 56%, and
Istanbul, which had no freezing hour in January 2025, had them 15% of the
time. That is the one place a model fitted on 2025 is asked to extrapolate,
and the flags are what it extrapolates along.

The top rung of the ladder, and the one `scripts/make_submission.py` writes
from, now stands at 294.4 s in December and 400.1 s in July over the rows
the physical model is answerable for — 277.3 and 315.9 s on the three July
airports — against 322.0 and 407.9 s when Gate 1 closed.

**After Gate 2b — the plan lags, 3 September.** Three planning values are
exposed for every ranking departure and fixed before the aircraft moves:
the flight plan's first and last off-block estimates and the schedule.
Take-off minus any of them is taxi plus the pushback delay against the
plan, and the block carries the subject's own three and the mean of each
over the other departures on its runway within fifteen minutes of its
take-off, the subject removed by identity. It sits on the ladder after the
aircraft and before the congestion blocks, so every congestion rung is read
net of it; the congestion rungs are kept without it as a second chain at
the foot of the ladder, so both readings are in
[`../reports/baseline.md`](../reports/baseline.md).

**It is worth 77 seconds in December and 53 in July, and everything above
it is worth almost nothing more.** Over the rows the record-quality term
leaves to the physical model, the plan lags take December from 321.5 s to
244.8 s and July from 405.6 to 352.9 — the three July airports from 315.9
to 260.2. Net of the lags, the five congestion blocks and the weather are
together worth 7.5 s in December (244.8 to 237.3) and 1.8 s in July; without
the lags the same blocks were worth 27 and 5. The reading is the one the
notes anticipated: the lag the runway is running at *is* most of the excess
term, measured after the fact, and a congestion block is a proxy for it.

**It is not a property of the pair.** On held-out pairs the contrast for
the plan lags is −1.2 s [−5.8, +3.2], zero: the block buys the same on a
pair the model has seen as on one it has not, so the geometry claim stays
where it was, +14.5 s [+13.0, +16.3], and every rung net of the lags sits
within 0.6 s of zero on the same contrast. On pairs never seen in training
the top rung goes from 311.9 s to 233.7.

Two cautions, both recorded in [`../reports/congestion.md`](../reports/congestion.md).
The runway mean of take-off minus the first off-block estimate is higher in
January 2026 than in January 2025 at every airport — Frankfurt 1,238 to
1,377 s at the median, Munich 1,041 to 1,199 — so the ranking month ran
later against its plans than the training year, and whether that is more
taxi or more pushback delay the exposed fields cannot say. And a rung read
net of the plan lags is read net of a good deal of the excess term, which
is why the chain without them is kept.

The top rung now stands at 237.3 s in December and 351.1 s in July, 222.7
and 260.2 s on the three July airports.

**The configuration set, dropped, 3 September.** The raw configuration set
— the designators in use in the trailing hour, as one categorical with 12
to 63 levels per airport — took 12% of the top rung's splits before the
plan lags and 8% after, and was worth nothing on any split. It is now a
diagnostic in [`../reports/congestion.md`](../reports/congestion.md) and
not a feature; the model keeps the count of runways in use and the seconds
since the set last changed. Without it the configuration rung is worth
0.2 s in December net of the plan lags and the top rung stands at 237.9 s
in December and 349.1 s in July (from 237.3 and 351.1), 223.7 and 260.1 s
on the three July airports; the held-out contrast for the configuration
rung is −0.1 s [−0.6, +0.3] net of the lags and +0.5 s [+0.2, +0.9]
without them. A column that costs nothing to drop and was memorizing
airport-hours is better dropped.

**The ceiling, 3 September.** Before any queue model is built, how much
of the top rung's residual sits in the runway sequence at all. An oracle
diagnostic in [`../reports/congestion.md`](../reports/congestion.md),
never a feature: on each forward split's validation rows, over the rows the
record-quality term leaves to the physical model, the top rung's residuals
are read against the residuals of the other departures on the same runway
— their taxi times minus what the model already knew, which no admissible
feature may read. The lag-1 autocorrelation of consecutive same-runway
residuals at a take-off gap under two minutes is 0.20 in December and 0.09
in July net of the plan lags, 0.32 and 0.17 without them. Regressing the
residual on the mean residual of the other same-runway departures within
fifteen minutes — one pooled least-squares fit, in sample, the subject
excluded — takes the December RMSE from 237.9 s to 227.2 s and the July
from 349.1 to 340.4: a ceiling of 10.7 s and 8.7 s on what any queue model
reading the runway sequence could add, net of the plan lags. Without the
plan lags the same ceiling is 32.5 s and 29.8 s. Per airport in December it
runs from 6 s at Schiphol to 20 s at Istanbul and 18 at Munich, and is nil
at Fiumicino, whose residual is the record-quality tail.

The reading for Gate 3: the plan lags have already taken most of what the
sequence carries, and what they leave to a queue model is ten seconds on
a 238-second error. The cheap runway-saturation features are measured
against that bound next; the simulation is not started without a decision.

**Runway saturation, the cheap Gate 3, 3 September — and the stop.** From
the other departures' take-off times on the subject's runway and nothing
else: the share of consecutive take-off gaps under ninety seconds over the
trailing five and fifteen minutes, the smallest gap in the trailing five,
and the time spanned by the last three gaps, capped at thirty minutes and
flagged when fewer than three precede. Registered and tested like every
block; physical on the real year (Heathrow's median share of short gaps is
0.6, Munich's and Zurich's on their parallel runways 0.0; the same shape on
both sides a year apart). One rung at the top of the ladder, measured
against the ceiling.

It is worth nothing. December goes from 237.9 s to 237.7, July from 349.1
to 349.3; without the plan lags, 293.9 to 293.8 and 398.0 to 398.2. The
held-out contrast is −0.5 s [−1.0, −0.0]. And the ceiling is untouched:
after the rung the oracle still finds 10.6 s in December and 8.7 in July
net of the plan lags, 32.5 and 29.9 without them — the saturation columns
took none of what the sequence carries. No saturation column is among the
top rung's split shares.

The reading. The plan lags already carry the lag the runway is running at,
measured after the fact; what the runway sequence holds beyond that is ten
seconds on a 238-second error, and the cheap features cannot reach it. A
queueing simulation is bounded by the same ten seconds. It is not started
without a decision; the numbers that decide it are above.

**The one parameter sweep, 3 September.** Permitted once, on October 2025
alone, and never reported: `scripts/sweep_params.py` fits January to
September and scores October — a month none of the ladder's splits reads —
over one grid of trees, learning rate, leaves and minimum child samples on
the saturation rung's features, the top rung when it ran, by the RMSE over
the rows the record-quality term
leaves to the physical model. The grid ran from 235.0 to 244.8 s and its
top eight cells lie within 0.8 s of each other; the cell that scored 235.0
s — 1,000 trees, learning rate 0.03, 127 leaves, 50 minimum child samples
— is written into `src/taxiout/models.py` with that number beside it, and
the grid's own numbers stay gitignored under `data/processed/`. Nothing
else is tuned, and nothing was chosen on December, July or the held-out
pairs.

Refitted and the ladder run once more, the top rung stands at 236.2 s in
December and 347.0 s in July (from 237.7 and 349.3), 222.5 and 260.0 s on
the three July airports; the rung below it, without the saturation block,
at 235.7 and 346.2. The geometry claim is +14.7 s [+13.2, +16.5], the plan
lags' held-out contrast −1.1 s [−6.0, +3.6], the ceiling 10.0 s in December
and 8.5 in July. The ladder now takes eighty minutes.

**Runway saturation dropped from the model, 4 September.** The saturation
block cost half a second in December and eight tenths in July on the rung it
was added to, and the submission is fitted from that rung. It is out of the
model: `TOP_RUNG` names the weather rung, and the saturation rung stays on the
ladder as the measured negative it is. A cheap queue proxy that was built,
measured against the ceiling, and bought nothing is a result, and it belongs
in the paper beside the ceiling rather than in the predictions.

Over the rows the record-quality term leaves to the physical model, the
submission rung stands at **235.7 s in December and 346.2 s in July**, and
221.4 and 258.8 s on the three July airports, against 236.2 and 347.0 with the
saturation block. The geometry claim is +14.7 s [+13.2, +16.5] and the
saturation rung's own held-out contrast is −0.2 s [−0.7, +0.3]. The one
parameter sweep is now pinned by name to the rung it ran on, which was the
saturation rung; following `TOP_RUNG` down would have quietly re-run it on a
different feature block and stopped reproducing the October numbers recorded
beside the booster's parameters.

**Limitation, 4 September: the ranking months ran late against their plans,
and the exposed fields cannot say why.** Recorded in full, with its numbers, under
*Regime exposure* in [`../reports/congestion.md`](../reports/congestion.md).
Take-off minus a planning value is taxi plus the pushback delay against that
plan, and the ranking months moved: in January 2026 the mean take-off minus
the first off-block estimate rose by 289 s at Istanbul, 224 at Frankfurt and
193 at Munich against the same month of 2025, while July 2026 fell by 135 s
at Frankfurt and 127 at Heathrow. A ranking departure's block time is hidden,
so which of the two moved is unobservable — but it is boundable, and both
halves of the bound are in the report.

The control says which way. Where the plan lag moved most, arrivals' taxi-in
at the same airport-month barely moved: −20 s at Istanbul, +9 at Frankfurt,
+16 at Munich, against arrivals' own lateness moving +856, +453 and +367 s.
Taxi-in is the surface running slower or not; arrivals' lateness is the day
running late or not. The second moves with the plan lag and the first does
not, so the shift is mostly the day running late against its plans rather
than the surface running slower.

The bound is what stands. Taking the whole shift as pushback delay — the
worst case, which the control says it is not — the training-year slope of
taxi-out on the plan lag runs 0.24 to 0.51 seconds per second, so the
per-airport bias reaches 93 s at Istanbul and 71 at Frankfurt, and added in
quadrature to those airports' own ex-flagged RMSE it costs 18.7 s and 12.0 s,
and 5.1 s pooled; in July, 4.5 s and 3.0 s, and 2.7 s pooled. That is a bound
on the exposure and not an estimate of it: the true shift is a mixture, no
exposed field separates the two, and nothing in the model is corrected for
it, because a correction would be a guess at the mixture fitted on nothing.

**Gate 2c — the fields the model never read, 8 September.** A review of where
the remaining error sits ([`review_headroom.md`](review_headroom.md)) found
that the residual of the submission rung is a tail of individual holds rather
than a queue, and that what is left in the *exposed* fields is about eight
seconds on the score. Four blocks were built for it, as a chain of rungs on top
of the encoding arm the submission is fitted from, each read against the one
below it.

**Stand occupancy is the one that matters, and it is a physical bound.** A
stand holds one aircraft, so an arrival that on-blocks at the subject's stand
before the subject takes off means the subject had already left the stand: its
taxi-out is at least the gap. The ranking set gives every arrival's stand and
on-block time in full, so the bound is built from exposed rows — and it bounds
a field the same data hides for departures, which
[`modelling_notes.md`](modelling_notes.md) records rather than leaves to be
noticed. It enters as a feature and never as a floor: below twenty minutes the
bound holds essentially always and the model already respects it, above
forty-five it collapses because a stand identifier can denote a position
holding two aircraft and because aircraft are towed, and the
twenty-to-forty-five-minute band is where the bound is the mechanism. Applied
as a hard floor it loses, at every cap tried.

**Most of the block is not the bound**, and it is described here as what it is:
the stand's recent history — the gap, the previous arrival's lateness and
taxi-in, and whether it was the subject's own aircraft returning. The band
where the bound is the mechanism carries 22% of the block's reduction in
squared error in December and 36% on the three airports July 2026 contains; in
December, 48% comes from gaps past an hour where the previous arrival was the
subject's own type, which bounds nothing and is turnaround pattern. Both halves
are admissible on the same footing. The construction, the bound by five-minute
band, the decomposition of the gain and the negative result are in
[`../reports/occupancy.md`](../reports/occupancy.md).

**What the four are worth.** Over the rows the record-quality term leaves to
the physical model, against the arm's 231.8 s in December and 343.6 s in July:
stand occupancy −8.4 and −7.8, the callsign prefix −3.1 and −5.5, the
destination geometry with the holiday calendar −0.6 and −0.8, the two plan
differences −1.9 and −4.0. Together **231.8 → 218.0 s in December and 343.6 →
325.6 s in July**, and 256.2 → 237.9 s on the three airports July 2026
contains. At the ranking set's own bucket mix, 232.9 → 219.1 s.

That is roughly twice what the review estimated. A residual fit measures what a
correction bolted on top of a frozen prediction would recover, where a refit
restructures every split in the model around the new column, so it bounds a
refit from below — but nothing predicted a factor of two, and reading the
direction as confirmation would be reading a prediction out of a result.

**The leakage audit now reads both of its columns, and stand occupancy misses
its trigger by twenty-nine milliseconds.** Each estimate has two columns,
December over all ten airports and July over the three, and the audit was
applied to the first alone — so an overshoot on the second could not have been
seen. Read against the column it was estimated on, stand occupancy takes
**+15.27 s on the July three against an estimated +10.3**: an overshoot of
**4.971 s** against the 5.000 s the trigger allows. It does not fire. It is
recorded as a near miss rather than as a pass, because a test cleared by
twenty-nine thousandths of a second is not evidence of anything, and because
this gate has already been embarrassed once by a trigger that did not fire
being read as a result.

On the pre-exclusion fit it did fire, at +15.86 s and an overshoot of 5.56 s.
What moved it below the line was taking the recorded-taxi tail out of the fit,
which is worth saying plainly: the margin here is a property of twelve rows.
What the block rests on is therefore not this test at all. It is the source
registry, the poison tests, and the measured concentration of the block's gain
in the band its mechanism names — not the agreement of the two estimates, and
not the band table in `occupancy.md`, which is built on the rung that never
read the gap and reports bias rather than gain.

**It carries to an airport the model has never seen**, which the held-out-pair
split cannot ask and [`../reports/transfer.md`](../reports/transfer.md) can.
The transferable feature set is the submission rung's minus every identifier,
so it now carries the stand-occupancy block and the callsign prefix — both
built from the held-out airport's own movements and from an airline designator
that means the same thing everywhere. Regenerated, the leave-one-airport-out
model beats the grand mean by **23.6% on average against 19.5% before**, best
43.3% at Heathrow against 33.4% at Munich, on 104 columns against 86. The
geometric term's own numbers do not move at all — the fitted speeds stay at
12.0 to 13.2 m/s across the ten fits — because nothing in this gate touches
that fit.

**None of it moves the held-out-pair claim**, which is what it should do: none
of these four is a property of the (airport, stand, runway) pair, so each was
expected within about a second of zero on that contrast. Two of the four came
in outside it, not one. Stand occupancy is at +1.4 s [+0.3, +2.5] — the only
one of the four whose interval is visibly clear of zero — and the plan
differences at −1.1 s [−2.3, −0.0]. The signs point opposite ways and mean
opposite things: positive is more where the pair is *new*, so occupancy is
supplying at unseen stands what the encoding would otherwise have carried,
which is the stand's recent turnaround pattern; negative is more where the pair
was *seen*, which is what a plan quantity interacting with a pair-level
encoding looks like. Both are recorded rather than explained away, and the rule
does not read that column.

**The rule that admitted them, and the knife-edge that is no longer one.** A
rung displaces the rung below it if it is no worse on both forward splits'
ex-flagged RMSE and no worse pooled at the ranking mix; the chain stops at the
first failure. **All four rungs now clear that rule on the literal reading**,
improving both splits and the mix.

They did not before, and the history is kept because it is the more instructive
state. On the pre-exclusion fit the destination rung came out **0.0093 s worse
on July** while improving December by 1.16 s and the ranking mix by 1.13 —
which stops the chain and strands two rungs if `no worse` is read as a
comparison of two real numbers, and is a tie if it is read at the one decimal
place every table prints. The second reading was adopted, after the number was
seen, and `baseline.md` carried a subsection justifying it: the figures to four
decimals, a day-clustered bootstrap whose interval contained zero by two orders
of magnitude, and a computed check that the reading moved that step and no
other. Once the recorded-taxi tail came out of the fit the rung improves July
by 0.78 s and the tolerance decides nothing, so that subsection is no longer
printed. The rule is unchanged and is still read at that tolerance; what
changed is that nothing now depends on it.

That is worth more than the seconds it bought. A methodological compromise this
gate had to argue for at length turned out to be an artefact of a dozen rows
that no aircraft produced, and the honest reading is that the knife-edge was
never really about the destination block.

**From the next gate, the July leg of that rule reads the three-airport
column.** *Recorded 9 September; not applied retroactively, and nothing already
decided is re-decided by it.* The July criterion above is the pooled figure over
all ten airports, and that figure is dominated by rows the record-quality term
does not flag: Rome is 7.8% of the July validation rows and **49% of the
split's ex-flagged squared error**, at an ex-flagged RMSE of 863 s against
roughly 200 s everywhere else. Sixty-five rows with a residual over an hour
carry 41% of it. So a rung can move the July pooled figure by improving Rome's
tail and do nothing anywhere else, or be held back by Rome while helping the
other nine — which is what the per-airport table in `baseline.md` now shows for
the callsign rung, at **−24.0 s [−35.8, −20.7]** at Rome on a rung whose pooled
July step is −5.5: a quarter of the pooled step is one airport that the ranking
month does not contain.

July 2026 contains three airports and Rome is not one of them, so the pooled
column is not even the quantity the ranking month asks about. The three-airport
column already exists in the report, the expected-headroom estimates were made
against it, and the leakage trigger now reads it. The acceptance rule should
too. It is stated here rather than applied now because changing an acceptance
rule after seeing which rungs it would admit is the failure this document
exists to prevent; the change binds the next gate, and Gate 2c stands as it was
decided.

**Seed averaging has been measured three times and is not adopted.** The rung
below this one first, where it gained 0.78 and 0.98 s and did not clear the
threshold. Then the submission rung, where it gained 1.10 and 1.37 s and did.
Then the same rung once the recorded-taxi tail came out of every fit, where it
gains **0.81 and 0.56 s** and does not — so by the rule fixed before any of it
ran, the submission is fitted from a single seed and
`taxiout.models.SEED_AVERAGE_ADOPTED` is `False`.

The three readings together say what the averaging was buying. Most of the
run-to-run variation five fits were smoothing over came from a handful of
day-shift rows in the fitting sample; excluding twelve impossible rows did the
same work for none of the compute. The variance had a cause, and it was not the
booster. It is an estimator choice and not a rung —
it changes how the model is fitted and not what it reads — so the ladder stays
single-seed and only `scripts/make_submission.py` uses it, with
`taxiout.models.BAGGING_SEEDS` recording the decision beside the booster's
parameters. It is worth saying what it is: variance reduction, under half a
percent, and nothing about the aircraft or the airport is understood better for
it. That the same measurement came to 0.78 and 0.98 s one rung below is the
clearest available measure of how much noise sits in a single fit — a hundred
times the difference the literal reading of the rule would have turned on.

**The PRU portal data stays out.** EUROCONTROL's own open data at
`ansperformance.eu/data` carries daily ATFM delay by cause and by airport,
which is the closest thing in existence to the "phased delays" the rationale
page describes. The portal's notice permits copying provided it "is not used
for commercial purposes" and states that the information "may not be modified
without prior written permission" — non-commercial and no-derivatives, which
fails the prize rules' requirement that additional datasets be "openly
available under an open source license", by the same reading that excluded
OPDI. It is left for the paper as what a future edition should open, and the
README records the exclusion beside the sources that were taken.

The submission rung is now `+ plan differences`, and the ladder's only rung
above it is runway saturation, still out and still a measured negative.

**The decisions of 19 September, and the experiments pre-registered on
20 September.** Four decisions were taken on 19 September, each put with its
evidence before any of the numbers it governs existed, and four experiments
were pre-registered on 20 September, criteria first. Both are recorded here,
and this entry is committed ahead of the run that measures them, so that the
record precedes the numbers in the history as it does on disk.

**The seed rule, re-fixed at the floors.** The one-second rule that governed
seed averaging was retired on 19 September, before any measurement under the
rule that replaced it existed. Its stated basis was the run-to-run variation
the ladder showed before the booster was made repeatable; that basis no longer
holds, because the booster now repeats to the bit and what is left is the
seed's own spread, which *The noise floor* measures. The rule fixed in its
place, in the words it was fixed in and as `reports/baseline.md` prints it:
"Adopt it for the submission if, on both forward legs, the five-seed average
beats every one of the five single seeds and its gain over the ladder's seed
clears that leg's seed floor; before it ships it is also measured on the three
July airports and on October, and both are printed."

**The sweep, reopened once, on 19 September.** The "permitted once" clause
above is reopened one time, before any new cell is fitted. The grid:
`learning_rate` 0.015, 0.03 and 0.06; `num_leaves` 63, 127 and 255;
`min_child_samples` 20, 50 and 200; and the trees read at 300, 600, 1,000,
1,500, 2,000 and 3,000 from one 3,000-tree fit per cell, as the first sweep
read its trees from one fit. Twenty-seven cells. The chosen cell (0.03, 127,
50, 1,000) sat on the edge of the first grid on all four axes; on this one it
is interior on three, and the trees axis extends past it. Each new value is
one step past the old edge at the old grid's own spacing, and the far values
of the old grid (0.1, 31, 500) are dropped.

The fit rules are the current ones and not the first sweep's: the submission
rung's features as they stand, the four-hour exclusion, the shrunk encoding on
`fit_rows`, the record-quality model on the training months, and the recency
schedule if one is adopted. Train January to September 2025 and score
October 2025 over the rows the record-quality term leaves to the physical
model, as the first sweep scored. The floor is measured in the same run, from
five single-seed fits of the current cell at `random_state` + 0 to + 4, root
two times their sample standard deviation. The reading: the best cell replaces
`BOOSTER_PARAMS` only if its October figure is below the current cell's figure
from the same run by more than that floor, and a difference inside the floor
keeps the current cell. If the best cell lies on an edge of this grid, that is
recorded and the grid is not extended again. **Nothing is chosen on December,
July or the held-out pairs.** One ladder run follows, under the standing
acceptance rules.

**`AOBT_3_flt` stays excluded by decision, and unmeasured.** It is the third
actual off-block time the challenge exposes, excluded in `taxiout.challenge`
and never measured against the model. The paper says it is excluded and why,
and does not claim to know what it would have been worth. Measuring it now
would be choosing what to exclude by what the exclusion costs, which is the
move this document exists to prevent.

**The July column stays as the 9 September entry above records it.** The July
leg of the acceptance rule reads the three-airport column from the next gate;
nothing already decided is re-decided by it.

**The experiments pre-registered on 20 September, criteria before numbers.**
Every leg below is fitted exactly as the ladder fits the submission rung: the
record-quality model on the leg's training months, `taxiout.models.fit_rows`,
the shrunk encoding fitted on those rows, `BOOSTER_PARAMS` as
`taxiout.models` carries them, and the first seed the ladder's. Scored
ex-flagged, as `taxiout.validation.evaluate` scores. Four legs: December
(train January to November, validate December), July (train January to June,
validate July), October (train January to September, validate October) and
the gap (the July leg's own fitted models scored on December, which is the one
forward leg the data allows in which the recent training months are seasonally
far from the validation month). "No worse" means not worse at the precision
the reports print, one decimal of a second. "Clears the floor" means the gain
on the unrounded figure exceeds that leg's floor, the floor being root two
times the sample standard deviation of five single-seed ex-flagged RMSEs. The
December, July and three-airport floors are 0.34, 0.23 and 1.02 s; October's
and the gap's are measured in the same run. Any step inside a second carries
the day-clustered interval `taxiout.validation.cluster_bootstrap` gives, with
the same draw for both arms.

**Experiment 1, seed averaging on the three and on October.** The five July
seeds scored pooled and on the three, singly and averaged; five October seeds
scored pooled, singly and averaged; October's floor from them. It serves the
rule quoted above, and one guard added on 20 September before either
measurement existed: the average is not adopted if, on the three or on
October, it is worse than the ladder's seed by more than that leg's floor. The
three is the column the July ranking month asks about and October is the one
leg no report has read; a regression clear of noise on either is a reason not
to ship, and the rule as fixed would not have read it. Nothing ships from the
measurement: the rule is read for the submission once the ladder prints the
two measurements under the final parameters.

**Experiment 2, recency weighting.** The confound is stated first, because it
decides what the experiment can show: on every 2025 leg the most recent
training month is also the seasonally nearest to the validation month, so a
gain there cannot separate recency from season. The gap leg is built to break
it. Two schedules, fixed before any fit and no third: weight per training row
`w = 0.5 ** (age / H)`, where `age` is the whole number of calendar months
from the row's take-off month to the leg's last training month, and `H` is
3 months (the strong schedule, which discards most of the year and makes the
confound's sign visible) or 12 months (the mild schedule, where the oldest
month of a full year keeps about half its weight). The weights enter the
booster's fit only, as `sample_weight`; the record-quality model, `fit_rows`,
the encoding and the scoring are unweighted and unchanged. Before any figure
is read, the digest check: `deterministic` was measured without sample
weights, so the mild schedule's October fit is made twice and the two
prediction vectors and the two model dumps must be identical, a disagreement
stopping the experiment. Then the sequence, fixed: each schedule is screened
on October against the unweighted October first seed, and one worse at printed
precision stops there and is fitted on nothing else; each survivor is fitted
on the July leg and read on the gap first, and one worse at printed precision
stops there, its July readings printed since they come from the same fit but
no December fit made and no adoption possible; each survivor of that is fitted
on December and read against 217.6 (December all ten), 237.8 (July the three)
and 218.6 (the ranking mix). Adoption: "A schedule is adopted for the
submission only if it is no worse at printed precision on all five readings
(October, the gap, December all ten, July the three, the ranking mix) and its
gain clears the floor on at least one of December all ten and July the three.
A change to how the model is fitted that moves nothing outside seed noise is
not adopted. If both schedules qualify, the mild one is adopted unless the
strong one beats it on all five readings by more than the respective floors."
A two-model submission, one fit per ranking month with its own weighting, is
not opened whatever the readings: no leg in the data validates the July 2026
geometry with its twin month present, the one-upload rule leaves it
unmeasurable, and it doubles the reproduction the paper defends.

**Experiment 3, the season columns.** The submission rung refitted on December
and on July without `tko_month`, `tko_doy_sin` and `tko_doy_cos`, read against
217.6, 237.8 and 218.6. It is for the paper and not for the model: both
forward legs validate on a month and a day-of-year range the training side
never held, and the ranking months do not, since January 2026 and July 2026
have their twins in the training year. The measurement answers a question the
ranking months do not ask, so **no adoption follows from it in either
direction**, whatever its sign. Its sign and size go to the paper as the
validation design's limitation.

**Experiment 6, Heathrow: nothing is fitted.** The July afternoon and the
no-record class that the residual report measures are structured at intervals
clear of zero, in columns the submitted rung already reads, with no mechanism
the frame lacks, and the oracle's ceiling for the same-runway sequence at EGLL
on July is 9.9 s. They are paper paragraphs and not a feature.

**Not run, and why.** Operator cells at LIRF: closed by the callsign-prefix
section, where the note's test is the record-quality term's own first
condition and a prefix split under the coverage rule opens one cell touching
1 of that airport's 11,061 ranking rows. The January boundary: 839 of 152,719
January 2026 rows carry `occ_truncated` against 853 of 153,706 in January
2025, the training year's own first day; the flag is a column the booster
reads, no forward leg validates a month that begins a block, so the change
could not be measured, and it sits on the scored-file path. A held-out-month
split for the season columns: it would answer the interpolation question the
ranking months ask, and it breaks the forward discipline every validation leg
here keeps; it is named so the paper can say what would have measured it. A
per-day table of Heathrow's afternoons: a search for the days rather than a
description, and the day-clustered interval already says the effect rests on
few of them. The pooled record-quality rule for airports with no covered cell:
recorded in the paper, no instrument in the data measures the alternative, and
it is not reopened on stakes alone. SUMO and the OpenSky scientific datasets:
diagnostics for the paper at most, and nothing in the model.

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
- **"No worse" means at the precision the report prints.** *Fixed 8 September,
  after Gate 2c turned on nine thousandths of a second.* Every RMSE in every
  table is given to one decimal place, and two rungs that print the same number
  are a tie: a rule that ranks them is ranking the random-number generator, not
  the models. The seed-averaging table measures that directly — the same rung
  refitted under a different seed moves by around a second — so a rule reading
  below a tenth of a second is reading noise. This is now the standing reading
  for every comparison in the reports, and it was settled going forward rather
  than left to be re-argued each time it binds.
- **Any step inside a second prints a bootstrap interval.** A difference that
  small is a claim about sampling variation and has to be shown as one. The
  machinery exists and costs nothing: `taxiout.validation.cluster_bootstrap`
  over stored residuals with no model refitted, clustered on the unit the
  question is about — pairs for a pair question, **days** for a
  same-split question, because congestion is autocorrelated over hours and two
  departures an hour apart are not independent draws.
