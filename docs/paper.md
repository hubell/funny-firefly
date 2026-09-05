# Recovering the airport surface: taxi-out prediction from OpenStreetMap geometry

*Working draft. Sections marked **[settled]** are written from reports that
are regenerated and will change only if their generator's numbers change.
Sections marked **[pending]** wait on the final model run and the submission.
Every figure is quoted from a named report, never retyped from memory, and
the freeze on 4 October 2026 is the date after which no figure moves.*

Team `funny-firefly`. PRC Data Challenge 2026.

---

## Abstract [settled; one figure waits on the organizers]

Taxi-out at ten European airports, predicted from a physical decomposition:
unimpeded time recovered from OpenStreetMap taxiway geometry, excess from
the observable traffic and weather regime around a departure, and a
record-quality term for rows whose recorded value no aircraft produced. The
geometry is routed on a graph repaired to a fixed point and verified on a
worked airport where every distance is known by hand. Its value is stated
as a difference in differences on held-out stand–runway pairs: the geometry
block buys **+14.7 s [+13.2, +16.5]** more where the model has never seen
the pair than where it has, and the relation it rests on — a taxi speed of
11.8 to 13.2 m/s recovered ten times over with a different airport left out
each time — transfers to an airport the model has never seen at all, where
an identifier model has nothing to encode. Nearly a tenth of the 2026 test
rows sit on pairs the training year saw a hundred times or fewer, and on
the thinnest of them a pair's own mean is worse than ignoring the pair; the
geometry is the better prior there by some twenty seconds over the airport
mean. The strongest single feature is not geometric but the flight plan's
own delay signal, worth 75 s in December and 57 in July, and once it is in
the model an oracle diagnostic bounds the whole queueing layer at ten
seconds, so the queue model the plan reserved thirty hours for was not
built. A shrunk identifier encoding aimed at the thin pairs, guarded
against four leakage modes, bought its two seconds mostly from the pairs it
was not aimed at; it is in the submitted model and is reported as what it
is. That model scores 233.4 s on a December forward split and 343.7 on
July, 380.0 and 634.7 with the record-quality rows left to it; the
organizers' figure is **[score]**.

## 1. Introduction

### 1.1 The problem, and the form the challenge gives it [settled]

Taxi-out is the interval from off-block to take-off. The challenge asks for
it in a form that inverts the operational one: the take-off time is given
and the off-block time is blanked. This is a post-hoc estimation task, and
the organizers say so — the rationale page names post-operations analysis,
"excess fuel burnt/CO2 production compared to unconstrained status", as the
use. Both framings are stated here, and the entry builds to the challenge's;
the alternative, anchoring only on schedules and filed estimates, is
discussed in §7.2 and was set aside as answering a question the challenge
did not ask.

Source: `docs/modelling_notes.md`, *Leakage watch*; the rationale page.

### 1.2 What this entry claims

Three claims, each with the measurement that supports it and the split on
which it was made:

1. Physical structure the challenge data omits — stand and runway geometry,
   taxiway topology — can be recovered from OpenStreetMap and carries over to
   stand–runway pairs never observed together: **+14.7 s [+13.2, +16.5]**
   more on held-out pairs than on seen ones. The same block is worth 0.1 s on
   the December forward split and 2.6 s on July, and §9.1 argues that this
   gap between the two numbers is the finding rather than a disappointment.
   **[settled: §7.2.]**
2. The target's heavy tail is not what it appears to be, and the honest
   treatment of it is a separate term stated for what it is. **[settled:
   §4.]**
3. The excess term is small once the flight plan's own delay signal is in
   the model — 8.2 s in December and 5.7 in July against that signal's 74.6
   and 56.5 — and the queue layer that would model it further is bounded to
   ten seconds by a diagnostic that needs no model. **[settled: §6.3, §6.4.]**

### 1.3 Prior art, and what is new here [settled]

Building a routable graph from OSM `aeroway` geometry is not new; `traffic`
(Olive, MIT) ships one, and Olive, Waltert & Schultz (ATRD 2025) document
the same defects this pipeline repairs. Ravizza et al. (2013) established
that route shape — turn angle, junction-free running — predicts taxi speed
nearly as well as distance. What appears not to have been done is the
downstream half: these recovered quantities as regression features for
taxi-out in this competition. No 2024 or 2025 entry appears to have used OSM
at all.

Source: `docs/prior_art.md`.

## 2. Data

### 2.1 What the challenge supplies, and what it withholds [settled]

Ten airports, not the eleven listed. January 2026 a census of all ten; July
2026 of three, which are 53% of the test rows. The ranking set blanks
`BLOCK_TIME_UTC_mvt` for *every* departure, not only the subject, which
retires any feature counting other departures' pushbacks. `AOBT_3_flt`
survives and is excluded by decision.

Source: `reports/data_review.md`.

### 2.2 External data, and the license position [settled]

OpenStreetMap (ODbL), OurAirports (public domain), Iowa Environmental
Mesonet METAR (public domain). Each is openly licensed, documented, and
re-fetchable by a scripted, cached client — the prize rules require
additional datasets to be "openly available under an open source license",
and a dataset a reader cannot re-fetch fails the reproduction clause.
OpenSky historical surface tracks were considered and ruled out for the
model: research-use access is not an open-source license.

Source: `docs/prior_art.md`, *External data: licence positions*; README.

### 2.3 Reproducibility and the ODbL boundary [settled]

The repository ships GPLv3 in totality. The distance matrices it generates
are a Derivative Database of OSM and carry ODbL, so they are regenerated
rather than committed; a Zenodo deposit under ODbL gives them a DOI. The
reports are Produced Works and carry the figures. `make licence-check`
enforces the boundary, including that no literal stand identifier from
either source appears in prose.

Source: `data/processed/taxi_distance/ODbL-NOTICE.md`; README, *Data sources*.

## 3. The model, stated as a decomposition [settled]

    taxi-out = unimpeded(geometry) + excess(congestion, regime) + record-quality term

Mapped onto the organizers' own framing of excess over unconstrained status.
The ablation ladder is this decomposition read bottom-up: each rung adds one
term, and the held-out-pair split measures the one claim that distinguishes
the entry.

Source: `docs/modelling_notes.md`, *The model, stated as a decomposition*.

## 4. The tail, and what the scoring rule makes optimal [settled]

*This section is deliberately early and deliberately its own: the 2025
committee praised "a thoughtful analysis of noise in the data", and the
term is the part of this entry most open to misreading.*

Under 0.1% of training departures carry half the airport-mean squared error.
The data review read them as day-shift artifacts. Building the term found
the mechanism behind most of it: at one airport, a departure with no
flight-plan record and a take-off hours after schedule has, nine times in
ten, a recorded off-block time within a minute of the *schedule* — the block
time was defaulted, so the recorded "taxi-out" is the schedule lag, which
every ranking departure exposes. The day-shift is real but small.

The term is the conditional mean per (airport, record, lag-band) cell,
computed knowing that mechanism: `share × lag + (1 − share) × rest_mean`.
Three cells qualify, all at one airport, 107 of 215,876 ranking rows. It is
**the RMSE-optimal prediction under the stated scoring rule for rows whose
recorded value no aircraft produced**, and nothing else. Every RMSE in this
paper is reported with and without the rows it covers.

Sources: `reports/tails.md`; `reports/data_review.md`.

## 5. The unimpeded term

### 5.1 From OSM to a routable surface [settled]

Three defects repaired to a fixed point: coincident-but-unshared nodes,
nodes lying on edges they are not joined to, and ways that cross with no
shared vertex. Stand attachment with the snap distance recorded per stand.
Route descriptors — distance, turn angle on a Douglas–Peucker simplification,
junction-free running, backtrack — rather than one number.

Sources: `reports/taxi_routing.md`; `reports/penalty_sensitivity.md`;
`reports/runway_crosscheck.md`; `docs/modelling_notes.md`, *Graph construction*.

### 5.2 A worked airport: where every number is known [settled]

A fictional field on which every distance, turn and repair is hand-derived,
pushed through the pipeline unchanged. All 32 matrix cells agree to 11 mm
and 0.22°, integer descriptors exact; the runway-penalty crossovers fall
where the arithmetic puts them. It found three things in the pipeline, each
below its working tolerance, each measured: overshoot and undershoot select
different repairs; a merged node takes the last way's coordinates; a
two-feature stand's representative is decided by floating-point rounding.
The field also carries worked traffic in the challenge's own schema, so the
whole pipeline runs end to end in continuous integration without the
challenge data.

*This is the section that answers "how do we know the geometry is right"
without asking a reader to trust 24,000 rows.*

Source: `reports/synthetic_airport.md` and its figure.

### 5.3 What the geometry recovers, and what it does not [settled]

Per airport, a pair's fastest decile follows its routed distance with a
residual of 85–190 s and an intercept of 313–740 s that no distance
explains. The largest structured effect on that decile is wake category —
mediums 35–281 s faster than heavies at the same stand and runway, uniform
across designators, which is a fixed pushback-to-line-up difference and not
intersection use. Intersection departures proper leave a signature of at
most 96 s per designator, constant, and absorbed by any model knowing the
designator; which stands use which entry is unobservable without surface
tracks.

Source: `reports/geometry_gaps.md`.

### 5.4 Where memorization runs out, and what replaces it [settled]

*Added 4 September, after §9.1's first draft asserted the opposite of what
the data says; settled 5 September, when its figures moved from a scratch
calculation into the generator. Two of the scratch figures did not survive
the move — a network-wide count had been attributed to one airport, and the
transfer figures were a stand-in's rather than the report's — which is the
reason this paper quotes generators and nothing else.*

**How much of the test set is thin.** Of the 215,876 ranking departures,
0.52% sit on a (stand, runway) pair the training year never saw, 1.36% on a
pair seen ten times or fewer, and 7.95% on a pair seen between eleven and a
hundred times — **9.8% in total**. The share is concentrated. Frankfurt has
2.37% never-seen and 4.05% at ten or fewer, and in July alone 4.10% and
6.69%; Heathrow, the other large July airport, has 0.18% and 0.28%. July,
the half of the test set with three airports, has 1.46% never-seen against
January's 0.13%. Across the ten airports, 24 stands appear in 2026 that
appear in no 2025 departure, carrying 339 ranking rows: Zurich contributes
nine of them and Frankfurt six, but Frankfurt's six carry 300 of the 339
rows. No runway designator is new. The never-seen bucket splits 339 rows on
a new stand against 776 on a stand and a runway the training year saw
separately and never together; the second kind is what the held-out-pair
split measures, and the first is the case no split of the training data
can pose.

**The pair mean is worse than useless where it is thin.** On the December
forward split, predicting a pair by its training mean scores 571.9 s on
pairs seen one to ten times — worse than ignoring the pair and predicting
the airport mean, 519.0 s. On July the same comparison is 543.1 against
498.3. A mean over a handful of draws from a distribution several hundred
seconds wide is noise wearing the costume of knowledge.

**The geometry is the better prior, and by more than the usual one.**
Shrinking the pair mean toward an estimate — the textbook remedy — depends
on what it is shrunk toward. Toward the airport mean, the standard choice,
the December bucket improves to 454.5 s. Toward the geometric prediction
for that specific pair, with the shrinkage weight estimated per airport by
the same construction the encoding of §7.4 uses, it improves to **433.3
s**. The geometric prior is worth 21.1 s more than the group prior on that
bucket in December and 20.7 s in July, and on the never-seen bucket, where
the two priors stand bare with no pair mean to shrink, 19.7 and 22.9 s. On
pairs seen more than a hundred times every version of the pair mean agrees
to a second, because the count swamps the weight: where memorization
works, the prior does not matter, and where it runs out, which prior it
runs out to is the whole question. The geometry explains 4% to 61% of the
between-pair variance the airport mean leaves, per airport, which is why a
pair's handful of draws is weighed less against it.

**The relation transfers between airports.** Fitted on nine airports and
applied to the tenth, with no stand, pair or airport-level quantity of the
tenth available, the transfer model — geometry plus the clock, the aircraft,
the plan lags, the traffic in the window and the weather, every identifier
removed — beats the only baseline available without it, the grand mean of
the other nine, at **ten airports out of ten**, by a median of 19.5% and by
33.4% at Munich. The geometric term *alone* — two fitted numbers and five
shared modulators — beats that mean at eight of ten, by a median of 3.2%,
and loses at Munich and Heathrow. The speed the ten leave-one-out fits
recover runs from 11.8 to 13.2 m/s, a 12% spread about 12.42 m/s, with the
intercept at 434 to 500 s and the offset at 251 to 277 s. That stability is
the finding: the relation is a transferable physical constant rather than a
per-airport fudge, and it is the one thing in this entry that an identifier
model cannot reproduce even in principle, because at an unseen airport it
has nothing to encode.

Sources: `reports/geometry_gaps.md`, *Where memorization runs out* — the
per-airport and per-month tables, the new-stand table with the shape of
each identifier, both forward splits by bucket, and the shrinkage weights
are there and are not restated; `reports/transfer.md` for the per-airport
transfer table, likewise.

## 6. The excess term

### 6.1 What the ranking set allows, and the registry that enforces it [settled]

The rule is phase-specific: a feature may read what the ranking set exposes
*for the rows it reads from*. Every feature column is registered against its
raw sources and the phase each is read from; the test suite holds each phase
to its own list, and a poison test sets the hidden columns to nonsense and
asserts the frame does not move. Arrivals' on-block and taxi-in times are
exposed at 0.0% null and are read; no departure's block or taxi time is,
ever.

Source: `reports/congestion.md`, *The leakage audit*.

### 6.2 Congestion, corridor, and weather [settled]

Throughput, arrivals, demand, runway configuration, shared-corridor load,
and three weather regime flags per airport-hour. The corridor feature — other
movements weighted by how much of the subject's own taxiway route they share,
each edge weighted by the inverse of the routes using it — is the one feature
here no other entry has, and it is reported as it fell.

Source: `reports/congestion.md`.

### 6.3 The flight plan's own delay signal [settled]

Take-off minus the filed off-block estimate is the strongest single feature
in the model. As a ladder step it is worth 74.6 s in December (318.5 → 243.9
s ex-flagged) and 56.5 in July (408.4 → 351.9); measured at the `+ weather`
rung, by removing it and keeping everything else, 58.2 and 51.1. Either
way it is worth more than every congestion and weather block combined, which
net of it are worth 8.2 s and 5.7.

What it is, said plainly, because a reviewer will ask: take-off minus a
planning value fixed before departure equals the taxi-out plus the pushback
delay against that plan. The model is therefore learning the distribution of
pushback delay around a value it is given — the same construction the
record-quality term uses with the schedule, and admissible for the same
reason: `EOBT_1_flt` and `LOBT_flt` are planning values the ranking set
exposes, not post-hoc records of the blanked field. `AOBT_3_flt`, which is
such a record, is excluded.

Its held-out-pair contrast is −1.1 s [−6.0, +3.6]: it helps equally whether
the pair is new or familiar, which is what a feature about the day rather
than the place should do. The ladder is reported both with it and without
it, and the regime exposure is bounded in §7.3.

Source: `reports/baseline.md`; `reports/congestion.md`, plan-lag and regime
sections.

### 6.4 The queue layer that was not built [settled]

An oracle diagnostic bounds it: regress a departure's residual on the mean
residual of its same-runway neighbors — quantities no admissible feature may
read, and which a queue model is precisely an attempt to approximate. On the
submitted model, the ceiling is **9.3 s in December and 8.7 in July**; the
lag-1 autocorrelation of consecutive same-runway residuals under two
minutes is 0.19 and 0.09.

The plan-lag block is what closed it. At the top of the chain fitted
without that block the same ceiling is 32.2 s and 28.6, and the
autocorrelation 0.31 and 0.17 — so most of what a queue model would have
found was the day's delay against plan, and the cheapest possible feature
had already taken it.

A saturation proxy — the share of sub-90-second gaps, the minimum gap, the
span of the last three gaps on the subject's runway — was then built against
that ceiling and bought nothing: −0.2 s [−0.7, +0.3] on held-out pairs
against the rung it was added to, and the rung carrying it has a ceiling of
10.0 and 8.5, within a second of the model's own. It stays on the ladder as
the measured negative it is, and the submission is not fitted from it. Gate
3 was stopped on that measurement rather than on the schedule, and the stop
is recorded in the plan.

*"The physical layer did not move RMSE once congestion was controlled for"
would have been a publishable finding; so is this. The thirty hours the plan
reserved for a queueing simulation were released by a diagnostic that took
minutes and needed no model.*

Source: `reports/congestion.md`, *The ceiling*; `docs/plan.md`.

## 7. Validation

### 7.1 The design, fixed before anything was fitted [settled]

Two forward splits — December 2025 at all ten airports, July 2025 at the
three the July test set contains — plus a held-out-pair GroupKFold and
pair-frequency strata. Never random K-fold on rows: congestion is
autocorrelated over hours.

The held-out unit is the **pair**, not the stand: every stand and runway is
still seen in other pairs, so this bounds what the geometry buys on a new
combination and does not isolate what it would buy on a new stand, which no
split of this data can measure. The claim is a difference in differences —
what a block buys on held-out pairs minus what it buys on seen ones — so
composition cancels. Intervals are a cluster bootstrap over pairs, 1,000
draws, resampling stored residuals; no model is refitted.

Source: `reports/baseline.md`, *Held-out pairs*.

### 7.2 The ladder, and the claim [settled]

Every rung on both splits, ex-flagged and with the record-quality term, plus
the seen-versus-held-out contrast per rung with its interval. The rung the
submission is fitted from is `+ shrunk encoding, replacing the stand`, at
233.4 s (December) and 343.7 (July), ex-flagged; the control it displaced,
`+ weather`, is 235.7 and 346.2, and every congestion and weather block is
read against that control.

The claim contrast — identifiers + time → identifiers + time + geometry — is
**+14.7 s [+13.2, +16.5]**: the geometry block buys 1.2 s where the model has
seen the pair and 15.9 s where it has not. The interval excludes zero.

Two neighboring rows are worth reading beside it, because they are what
makes the claim mean anything. The step from the geometric unimpeded term to
the identifier model is −38.9 s [−44.1, −33.4]: identifiers beat geometry
outright where the pair is known, which is the null this entry had to face.
And the step into the geometry block from identifiers is where that reverses.
Together they say the two encodings are not competing descriptions of the
same thing — one memorizes a pair, the other computes it — and the split
that separates them is the only one that can.

Three arms above `+ weather` carry a shrunk identifier encoding; they are
the subject of §7.4, one of them is the submitted model, and the claim
contrast is measured near the foot of the ladder where neither rung
carries it.

Source: `reports/baseline.md`, *The ablation ladder* and *Held-out pairs*.

### 7.3 What could go wrong on the ranking months [settled]

Three exposures, each stated as a bound rather than an estimate.

**The plan-lag regime shift.** The ranking months' departures run later
against their plans than their 2025 twins: at Istanbul the mean of take-off
minus the filed estimate moves +289 s, at Frankfurt +224, at Munich +193;
July moves the other way, −135 to −74. The exposed fields cannot decompose
that into taxi and pushback delay, but they can bound it, and a control
that reads no plan at all says which it mostly is. Where the lag moves most,
arrivals' taxi-in barely moves (−20, +9, +16 s at those three airports)
while arrivals' own lateness moves with it (+856, +453, +367): the day ran
late against its plans, the surface did not run slower.

Taking the whole shift as pushback delay anyway — the worst case, which the
control says it is not — and multiplying by each airport's training slope of
taxi-out on that lag (0.24 to 0.51 s per second) gives a per-airport bias of
at most +93 s, costing **+19.1 s at Istanbul, +12.1 at Frankfurt, +5.2
pooled in January and +2.7 in July.** Biases of opposite sign do not cancel
in the pooled figure, because each enters its own airport's squared error.

**The winter regime.** January 2026 froze for 48% of Frankfurt's hours
against 30% in 2025, 65% of Zurich's against 41%; Istanbul went from no
freezing hour to 15%. This is where a model fitted on 2025 extrapolates, and
the weather flags are what it extrapolates along — which is the argument for
regime flags rather than continuous weather drivers.

**The record-quality term's coverage.** Fitted on one airport's 2025 feed.
An airport that begins defaulting block times in 2026 is not covered, and no
rule fitted on 2025 could cover it; §4 gives the arithmetic of that choice
against pooling.

Sources: `reports/congestion.md`, *The regime shift* and *The weather
regime*; `reports/tails.md`.

### 7.4 A block aimed at one place that paid in another: the shrunk encoding [settled]

*The measurement, the decision and the model agree: the arm this section
chooses is the one the submission is fitted from, through an inference-time
encoder that computes the encoding once over the training year and is
tested against the same leakage modes as the fold-wise one.*

**What was built, and against what.** §5.4 says where the identifier
model has nothing to memorize and that the geometry is the better prior
where it runs out. The obvious way to spend that inside the booster is an
empirical-Bayes encoding of the (airport, stand, runway) pair — its
out-of-fold training mean shrunk toward the geometric prediction rather
than toward a group mean, `(n × ȳ + k × ĝ) / (n + k)`, with `k` estimated
per airport as the ratio of within-pair to between-pair variance, the
sampling noise of each pair mean subtracted, and the same construction at
stand level as a second feature. It was pre-registered against a specific
bucket: pairs seen one to ten times, where the identifier exists and its
mean is a handful of draws. The ceiling it was built against was computed
before the arms were run — scale each bucket's airport-mean error by the
ratio the control achieves on the well-observed bucket and call the
difference the attainable headroom — and came to +0.34 s pooled, all of it
in the thin buckets, none by construction in the bucket that carries 91%
of the rows.

**What was guarded.** Four leakage modes, each with a test that plants the
leak and asserts nothing moves: the encoding is recomputed inside every
fold, so a held-out pair sees `n = 0` and encodes to the bare prior; `ȳ` is
out of fold within training too, so a row's own target never enters its
own feature — the single-`k` version failed that test by twelve seconds,
which is how the out-of-fold `k` came to exist; `ȳ` reads unflagged rows
only, so one defaulted block time cannot become a three-row pair's
encoding; and the one overlap that cannot be guarded, the prior being
fitted on hundreds of thousands of rows of which a thin pair contributes a
handful, is stated rather than left to be found. `k` was checked for
route-length dependence by a rule fixed before the run and found
immaterial at a ratio of 1.29 against a threshold of 1.5, and a `k/2`–`2k`
sensitivity moves the arms by at most half a second; `k` is not tuned.

**A gate that was voided rather than failed.** The rule fixed in advance
required a positive difference in differences on the held-out-pair folds.
That condition cannot credit the component the block exists for: a
held-out pair encodes to the prior identically, so the pair encoding
carries none of the pair's own mean exactly where the split measures. The
scope of that is worth stating precisely, because overstating it would be
self-serving. The gate is applied to a whole *arm*, and it validly measures
the rest of one — the stand-level encoding, and the removal of the `stand`
categorical — which it reads as decisively negative. It is not an
instrument incapable of returning a positive number; it is an instrument
blind to the pair encoding. A gate that cannot see the thing under test is
not a test of it, so it was voided as a test of the pair encoding, the
arms took no credit from it, and they were decided on the two instruments
that can see a pair encoding — the forward
splits and the frequency strata — with the ranking set's own bucket mix as
the pooled figure. That
replacement rule was written after the numbers were seen, which is a real
weakening and is said in the report; the condition that does the
rejecting — no worse than the control on either forward split — is the
original rule's own second half, unchanged, and it is what rejects the
first arm at +2.7 s on July.

**What chooses between the two arms that pass is a mechanism, not their
scores.** §5.3 measures what a designator carries that the routed distance
cannot compute: an intersection departure enters the runway at a taxiway
junction rather than at the threshold, and the signature of that is up to
96 s per designator and constant within one — absorbed by any model that
knows the designator, and not recoverable by an encoding shrunk toward a
matrix that routes to the threshold. So the runway categorical stays and
only the stand is replaced, for a reason measured before these arms were
run and independent of their numbers. The never-seen bucket corroborates
and does not decide: it is a few hundred rows, and a decision resting on
it would be a post-hoc rule resting on a thin stratum.

**What it bought, and where.** `+ shrunk encoding, replacing the stand`
clears the rule: 233.4 s against 235.7 in December, 343.7 against 346.2 in
July, 234.4 against 236.8 at the ranking mix, and +5.2 s worse on the
never-seen bucket against +17.1 for the other passing arm. The pooled
figure does not separate the two — 234.4 both ways, four hundredths apart
— so nothing measurable is traded away for the mechanism.
Decomposed by bucket in the sum-of-squares domain, **73% of the ranking-weighted gain
comes from pairs seen more than a hundred times** — the bucket the ceiling
had assigned no headroom at all — against 28% from the two thin buckets it
was aimed at and 1.1% given back on pairs never seen. On the strata it was
aimed at it takes 15.5 s of the 30.9 s of headroom the `1-10` bucket has,
which at that bucket's 0.64% of the rows is worth 0.15 s pooled. The honest
name for the block is smoothed memorization that buys real score on
well-observed pairs. It is reported under that name, and §9.4 says what
the ceiling that missed it was actually measuring.

Two consequences follow. The prior is `ĝ`, so the encoding is
geometry-bearing, and the ladder's bookkeeping now counts it as such; an
ablation that removed the geometry block and left the encoding in would
not have removed the geometry. And the claim contrast of §7.2 is measured
at `identifiers + time → identifiers + time + geometry`, near the foot of
the ladder, where neither rung carries the encoding — so the +14.7 s figure
is untouched by this decision, by construction and not by luck.

Source: `reports/baseline.md`, *The shrunk identifier encoding* and its
subsections; `tests/test_encoding.py` for the four guards.

## 8. Results [settled; one figure waits on the organizers]

*The challenge has no leaderboard: the scored parquet goes up at
submission and the score comes back after the close, so this section
carries what the two forward splits say about the model that is
submitted, per airport and with and without the record-quality term, and
leaves one line for the organizers' figure. Every number is from
`reports/baseline.md` as regenerated on 5 September, for the rung named in
`taxiout.models.SUBMISSION_KEY`, which is now `+ shrunk encoding, replacing
the stand` — the arm §7.4 chose, fitted by the inference-time encoder that
lands with it.*

**Pooled.** On the December split, which stands in for January 2026 at all
ten airports, the submitted model scores **233.4 s** over the rows the
record-quality term leaves to it, 251.2 s with the term substituted on the
88 rows it covers, and 380.0 s with those rows left to the physical model
alone. On the July split, which stands in for July 2026, **343.7 s**,
387.0 and 634.7 in the same order. The whole distance between the first
and third figures is 88 and 335 rows at one airport whose recorded
taxi-out no aircraft produced; the term is what closes it, and §4 says why
that is the RMSE-optimal treatment rather than a trick. Pooled at the
ranking set's own bucket mix rather than December's, the same model is
234.4 s. Bias is +8 s in December and −7 in July.

**Per airport.** The three airports the July test month contains —
Frankfurt, Heathrow and Schiphol, 53% of the test rows — score 222.4 s in
December and 257.2 in July. Frankfurt is 200.4 and 196.9; Schiphol 183.9
and 183.7; Heathrow, the widest field, 269.6 and 359.7. Six of the other
seven run from Madrid's 178.4 to Paris's 277.8 in December, and from
Munich's 196.3 to Paris's 294.6 in July. Fiumicino is the airport the term exists
for: 332.3 s in December and 862.9 in July without its rows, 1,159.5 and
2,073.9 with them and no term. The per-airport claim contrast is positive
at all ten and clear of zero at nine — Barcelona's +4.3 s [−0.9, +9.4] is
the exception — from +5.5 s at Schiphol to +29.9 at Frankfurt, which is
also the airport with the largest thin share.

**What the score will and will not say.** The pooled test figure weights
the three July airports at just over half, so it is closer to a
Frankfurt–Heathrow–Schiphol figure than to a European one; the §7.3 bounds
put the plan-lag regime shift at up to +5.2 s pooled in January and +2.7 in
July, and the winter regime at Frankfurt and Zurich is where the model
extrapolates furthest. The organizers' figure, when it comes: **[score]**.

Source: `reports/baseline.md`, *The ablation ladder*, *Per airport*, *The
claim, per airport* and *Error by pair frequency*.

## 9. Discussion

### 9.1 What the geometry was worth, and where [settled]

The geometry block is worth 0.1 s on the December forward split and 2.6 on
July, and +14.7 s [+13.2, +16.5] on pairs the model has never seen. Both
numbers are the same block; they differ because the two splits ask different
questions.

The first draft of this section explained the gap by asserting that the
ranking months reuse 2025's stands and runways, so the test set has no thin
pairs and cannot reward the physical layer. **That assertion was wrong, and
§5.4 is the measurement that replaced it.** 9.8% of the ranking rows sit on
pairs the training year saw a hundred times or fewer, the share runs from
2% at Heathrow to 19% at Paris and is 15.6% at Frankfurt in July, 24 stands
are new outright, and on pairs seen one to ten times a pair's own mean is
worse than the airport mean by 53 s in December and 45 in July. The gap
between 0.1 s and +14.7 s is therefore not a property of the test set. It
is a property of how the two encodings are combined: a booster handed both
the identifiers and the geometry splits on the identifiers wherever they
are saturated, which on a forward split is nearly everywhere, and the
geometry's 7% of the splits is the residue. §7.4 is the one attempt to
combine them differently; it ships, and it bought its score from the
saturated pairs too.

What survives from the first draft, and is still worth saying: the geometry
block is an injective encoding of (stand, runway), so on a pair seen
hundreds of times a saturated identifier model already knows everything the
distance could tell it, and there the geometry can only lose. The claim was
never that geometry beats memorization where memorization works. It is that
memorization runs out, that the data says where, and that the geometry is
what remains when it does — and, in the leave-one-airport-out fits, that it
is the only thing that remains.

### 9.2 What the honest tail treatment costs and buys [settled]

The term touches 107 ranking rows and is inert on the July half of the test
set. Its whole effect on the leaderboard runs through those rows. The
alternative — predicting only physically plausible taxi-outs and arguing the
point — was set aside because it would place mid-table regardless of model
quality, and the argument can be made just as well in a paper that also
scores.

### 9.3 Limitations [settled, to be assembled]

Assembled from the open questions already recorded: stands OSM does not have
(4% of departures, whole numeric blocks at two airports); the snapped-stand
turn-angle bias, expected on physical grounds and undetectable by the
available proxy; intersection departures at stand level; the unpinned OSM
snapshot, rebuilt in the freeze week; one airport's data feed driving the
entire record-quality term.

Two more that belong here because they concern how the numbers above were
produced rather than what they say:

**The parameter sweep ran on a feature set that differs from the model's by
two blocks.** The booster's parameters were chosen once, on October 2025
held out of the fit, on the ladder's last rung at the time — which carried
the runway saturation block and neither the shrunk encoding nor the stand
it replaces. The saturation block was then dropped and the encoding arm of
§7.4 adopted, so the model is fitted with parameters selected for a
feature set that has one block it lacks and lacks one it has. The
saturation block is worth −0.2 s, the encoding +2.3, and the grid's top
eight cells lie within 0.8 s of each other, so the effect is inside the
noise of the choice; but the sweep is not re-run, because re-running it
after seeing the ladder would convert a one-off selection into a search
over reported results. The sweep is pinned by name to the rung it actually
ran on.

**One ladder run was killed and restarted** after its generator was edited
mid-run; the reported figures are all from the completed run that followed.

Sources: `docs/modelling_notes.md`, *Open questions*; `reports/geometry_gaps.md`;
`src/taxiout/models.py`, the parameter block.

### 9.4 A ceiling that measured the wrong thing [settled]

The headroom estimate of §7.4 was wrong twice, and the two errors are worth
more than the number they were about.

The first was arithmetic. The estimate scales each pair-frequency bucket's
airport-mean error by the ratio the control achieves on the well-observed
bucket and calls the difference the attainable headroom. On the `11-100`
bucket that difference is **−8.3 s**: the control is already ahead of what
the ratio would predict. The first reading of that table described the
bucket as "about zero" and credited its magnitude, which turned a pooled
ceiling of +0.34 s into a claimed one to one and a half seconds. The
generator now computes the pooled figure both ways — a negative bucket held
at zero, +0.34 s, and the signs taken literally, −0.44 s — and prints both,
because the gap between them is the whole of the number. The error ran in
the direction that made a piece of work already proposed look worth doing,
which is the direction such errors run.

The second is the one that matters. The ceiling was not wrong about what it
measured; it measured the wrong thing. It assumed, by construction, that
the bucket carrying 91% of the rows had no headroom at all — the ratio was
defined from it — and looked only where the block was aimed. The block then
bought 73% of its gain from exactly that bucket. A ceiling that can only see
the target of a feature cannot bound a feature whose mechanism is not the
one it was named for, and the honest reading of "smoothed memorization"
in §7.4 is that the mechanism was never the one the ceiling was built to
measure. The lesson is not that the ceiling should have been larger. It is
that a bound computed from where a block is aimed is a bound on the
aiming, and the report that carries it should say so before the arms are
run, not after.

Source: `reports/baseline.md`, *The ceiling this was built against* and
*What the shipped arm is, in its own terms*.

## 10. Reproducing this [settled]

Every figure in this paper is regenerated by a named script from committed
inputs plus the challenge data, which is not redistributable. The geometry
reports and the worked airport regenerate on a clean checkout with no
network and no challenge data. The freeze-week sequence: pinned OSM rebuild,
Zenodo deposit, clean-checkout reproduction of every report, final
submission.

Source: README, *Reproducing*; `docs/plan.md`, freeze section.

## References [settled]

From `docs/prior_art.md`, *Sources*.

---

## Drafting notes, not part of the paper

- **What is already written.** Everything. The ladder, the claim, the
  encoding, the ceiling, the plan-lag block and the regime bound carry
  figures from `reports/baseline.md` and `reports/congestion.md` as
  regenerated on 5 September after the submission rung moved to the
  encoding arm; §5.4 from `reports/geometry_gaps.md` of the same day. Every
  figure will change only if a generator's numbers change.
- **What waits.** Two placeholders, both marked `[score]`: the organizers'
  figure in the abstract and in §8.
- **Which rung each figure is anchored to.** The submitted model is
  `+ shrunk encoding, replacing the stand`. §6.4's ceiling and §7.3's
  worst-case costs are on that rung; §6.3's plan-lag figures and every
  congestion and weather step in §7.2 are read against `+ weather`, the
  control the arm displaced, because that is the rung those blocks were
  added to. The `+ weather` ceiling (10.1 and 8.6 s in the 4 September run)
  is no longer printed by `congestion.md`, whose ceiling section now starts
  at the submission rung; §6.4 was re-anchored rather than left citing a
  number no report prints. If the generator gains the control rung back,
  the saturation proxy's "unchanged ceiling" sentence can return to its
  original form.
- **The number to re-check at the freeze.** Every figure in §6.3, §6.4,
  §7.2, §7.3, §7.4 and §8 was read from `reports/baseline.md` and
  `reports/congestion.md`; §5.4 from `reports/geometry_gaps.md`, whose thin-pair
  section is computed on the raw identifier triple and so does *not* move
  with the matrices, except its geometric-prior columns. If any report is
  regenerated after the pinned OSM rebuild, re-read it rather than
  assuming; the rebuild moves the matrices, and the matrices move §5 and
  §7.2.
- **Figures.** The worked airport's SVG is the one figure the paper already
  has. Candidates for two more: the ladder as a bar chart of ex-flagged RMSE
  per rung on both splits; the ceiling as the autocorrelation-by-gap curve.
  Both from committed report data, generated by a script like everything
  else.
- **Length and venue.** JOAS-style, which the 2025 committee singled out.
  The reports carry the detail; the paper carries the argument and points at
  them.
- **Tone rule inherited from the repository.** Anything about scoring or the
  jury is quoted from the challenge pages verbatim. Every number names its
  report. Nothing is asserted that a generator does not compute.
