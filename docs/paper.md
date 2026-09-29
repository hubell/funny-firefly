# Recovering the airport surface: taxi-out prediction from OpenStreetMap geometry

*Every figure is quoted from a named report or a dated record, never
retyped from memory, and `scripts/check_paper.py` holds it there. The model
froze at the commit tagged `freeze-2026-10-04`, made on 27 September 2026:
no figure moves after it, and the one figure that arrives later, the
organizers' score, is marked where it goes.*

Team `funny-firefly`. PRC Data Challenge 2026.

---

## Abstract

Taxi-out at ten European airports, predicted from a physical decomposition:
unimpeded time recovered from OpenStreetMap taxiway geometry, excess from
the observable traffic and weather regime around a departure, and a
record-quality term for rows whose recorded value no aircraft produced. The
geometry is routed on a graph repaired to a fixed point and verified on a
worked airport where every distance is known by hand. Its value is stated as
a difference in differences on held-out stand–runway pairs: the geometry
block buys **+18.5 s [+16.7, +20.5]** more where the model has never seen
the pair than where it has, and the relation it rests on, a taxi speed of
11.8 to 13.2 m/s recovered ten times over with a different airport left out
each time, transfers to an airport the model has never seen at all, where an
identifier model has nothing to encode. Nearly a tenth of the 2026 test rows
sit on pairs the training year saw a hundred times or fewer, and on the
thinnest of them a pair's own mean is worse than ignoring the pair; the
geometry is the better prior there by some twenty seconds over the airport
mean. The strongest single feature is not geometric but the flight plan's
own delay signal, worth 79 s in December and 57 in July, and once it is in
the model an oracle diagnostic bounds the whole queueing layer at about
eight seconds, so the queue model the plan reserved thirty hours for was not
built. A shrunk identifier encoding aimed at the thin pairs, guarded against
four leakage modes, bought 4.9 s in December and 1.9 in July, and the
December gain came mostly from the pairs it was not aimed at; it is in the
submitted model and is reported as what it is. That model, the mean of five
seeds, scores 213.55 s on a December forward split and 234.20 on the three
airports July 2026 contains, 320.90 over all ten, over the rows the
record-quality term leaves to it; a single seed of it, with the
record-quality rows left to it too, scores 364.8 and 620.5. The organizers'
figure is **[score]**.

## 1. Introduction

### 1.1 The problem, and the form the challenge gives it

Taxi-out is the interval from off-block to take-off. The challenge asks for
it in a form that inverts the operational one: the take-off time is given
and the off-block time is blanked. This is a post-hoc estimation task, and
the organizers say so: the rationale page names post-operations analysis,
"excess fuel burnt/CO2 production compared to unconstrained status", as the
use. Both framings are stated here, and the entry builds to the challenge's;
the alternative, anchoring only on schedules and filed estimates, is
discussed in §7.2 and was set aside as answering a question the challenge
did not ask.

Source: `docs/modelling_notes.md`, *Leakage watch*; the rationale page.

### 1.2 What this entry claims

Three claims, each with the measurement that supports it and the split on
which it was made:

1. Physical structure the challenge data omits (stand and runway geometry,
   taxiway topology) can be recovered from OpenStreetMap and carries over to
   stand–runway pairs never observed together: **+18.5 s [+16.7, +20.5]**
   more on held-out pairs than on seen ones. The same block is worth 2.27 s
   on the December forward split and nothing the seed floor can see on July,
   and §9.1 argues that this gap between the numbers is the finding rather
   than a disappointment (§7.2).
2. The target's heavy tail is not what it appears to be, and the honest
   treatment of it is a separate term stated for what it is (§4).
3. The excess term is small once the flight plan's own delay signal is in
   the model: 7.7 s in December and 5.8 in July against that signal's 79.3
   and 57.4. The queue layer that would model it further is bounded to about
   eight seconds by a diagnostic that needs no model (§6.3, §6.4).

### 1.3 Prior art, and what is new here

Building a routable graph from OSM `aeroway` geometry is not new; `traffic`
(Olive, MIT) ships one, and Olive, Waltert & Schultz (ATRD 2025) document
the same defects this pipeline repairs. Ravizza et al. (2013) established
that route shape (turn angle, junction-free running) predicts taxi speed
nearly as well as distance. What appears not to have been done is the
downstream half: these recovered quantities as regression features for
taxi-out in this competition. No 2024 or 2025 entry appears to have used OSM
at all.

Source: `docs/prior_art.md`.

## 2. Data

### 2.1 What the challenge supplies, and what it withholds

Ten airports, not the eleven listed. January 2026 a census of all ten; July
2026 of three, which are 53% of the test rows. The ranking set blanks
`BLOCK_TIME_UTC_mvt` for *every* departure, not only the subject, which
retires any feature counting other departures' pushbacks. `AOBT_3_flt`
survives and is excluded by decision.

Source: `reports/data_review.md`, *Airports and months* and *Inventory*.

### 2.2 External data, and the license position

OpenStreetMap (ODbL), OurAirports (public domain), Iowa Environmental
Mesonet METAR (public domain). Each is openly licensed, documented, and
re-fetchable by a scripted, cached client: the prize rules require
additional datasets to be "openly available under an open source license",
and a dataset a reader cannot re-fetch fails the reproduction clause.
OpenSky historical surface tracks were considered and ruled out for the
model: research-use access is not an open-source license. OpenSky's ten
published scientific datasets were reviewed on 20 September 2026 against the
same test plus two more, 2025 ground coverage at the ten airports and a
mechanism the exposed fields lack; none passes all three, and the README
records each verdict.

Source: `docs/prior_art.md`, *External data: licence positions*; README.

### 2.3 Reproducibility and the ODbL boundary

The repository ships GPLv3 in totality. The distance matrices it generates
are a Derivative Database of OSM and carry ODbL, so they are regenerated
rather than committed; the Zenodo deposit **[DOI]** carries them under ODbL,
with the eleven Overpass answers they derive from, and beside them the two
public-domain inputs the entry read and a later fetch would not return, the
METAR observations and the OurAirports tables. The reports are Produced
Works and carry the figures. `make licence-check` enforces the boundary,
including that no literal stand identifier from either source appears in
prose.

Source: `data/processed/taxi_distance/ODbL-NOTICE.md`; README, *Data sources*.

## 3. The model, stated as a decomposition

    taxi-out = unimpeded(geometry) + excess(congestion, regime) + record-quality term

Mapped onto the organizers' own framing of excess over unconstrained status.
The ablation ladder is this decomposition read bottom-up: each rung adds one
term, and the held-out-pair split measures the one claim that distinguishes
the entry.

Source: `docs/modelling_notes.md`, *The model, stated as a decomposition*.

## 4. The tail, and what the scoring rule makes optimal

*This section is deliberately early and deliberately its own: the 2025
committee praised "a thoughtful analysis of noise in the data", and the
term is the part of this entry most open to misreading.*

Rows whose error under the airport mean exceeds an hour are 0.086% of
training departures and carry 50.9% of that predictor's squared error.
The data review read them as day-shift artifacts. Building the term found
the mechanism behind most of it: at one airport, a departure with no
flight-plan record and a take-off hours after schedule has, nine times in
ten, a recorded off-block time within a minute of the *schedule*. The block
time was defaulted, so the recorded "taxi-out" is the schedule lag, which
every ranking departure exposes. The day-shift is real but small.

The term is the conditional mean per (airport, record, lag-band) cell,
computed knowing that mechanism: `share × lag + (1 − share) × rest_mean`.
Three cells qualify, all at one airport, 107 of 215,876 ranking rows. It is
**the RMSE-optimal prediction under the stated scoring rule for rows whose
recorded value no aircraft produced**, and nothing else. Every RMSE in this
paper is reported with and without the rows it covers.

**Why an uncovered airport is left to the physical model, and what measures
that.** A cell counts as covered only where enough of its rows carry the
mechanism, and an airport with no covered cell is left to the physical model
entirely. The alternative considered when the term was designed was to pool
cells across airports, and on the 2025 validation legs a pooled design is
worse than the adopted mixture on all three readings: 390 s against 341 on
December, 586 against 461 on July pooled, and 385 against 339 on the three
July airports. That is the arithmetic of the choice and not the whole of the
answer, because it compares a differently shaped design, and because the
case that presses the question is not a validation leg. At Amsterdam the
ranking set lists six consecutive days in January 2026 with the flagged
pattern, and the training year holds no such day at that airport at all; the
longest run anywhere in 2025 is two days, at Madrid.

What settles it is an instrument already in the data, and it measures the
premise pooling rests on: that a bucket's defaulted share is a property of
the defaulting mechanism rather than of the airport, so that a cell learned
where the mechanism is visible may be carried to an airport where it is not.
The fitted cell table measures that premise per airport and per bucket, on
training rows, with no fit and no booster. It says the mechanism is at one
airport: of the 432 training rows the term's own test picks out, 99.1% are at
Fiumicino, and only three of the ten airports carry any at all. In the two
buckets a pooled cell would serve, every other airport reads 0.0% on every
leg, at row counts small enough that a single defaulted row would show.
Tested as arithmetic, each airport's share against two standard errors of the
pooled share at that airport's own row count, the premise fails in both of
those buckets on each of three training halves. Where an airport passes, it
passes on the width of its interval and not on its share: in the larger
bucket over the full year three airports sit inside, each at 0.0%, because at
10 to 14 rows the interval reaches down to zero. The July leg is the thin
one, with 187 and 26 rows pooled in the two buckets, and the premise fails
there too. Amsterdam's zero is a measurement and not a gap: its cells hold 77
and 14 rows, well above the five-row minimum, so one defaulted row would show
as 1.3% or 7.1%, and a share of zero means no defaulted row in a full year.
Pooled, those cells would have predicted 4,512 s against a recorded mean of
824 s and 32,761 s against 1,212 s on Amsterdam's own 2025 rows. So the pooled
rule is not adopted, the arm that would have tested it was not fitted because
its premise failed, and the reason is measured rather than asserted.

One limit remains, narrower than the decision it used to carry. Whether an
uncovered airport's 2026 rows are themselves defaulted block times is a
question about the two fields the ranking set hides, and no instrument
reaches them. §7.3 states the exposure that leaves.

Sources: `reports/tails.md`, *What the tail is made of*, *The cells*, *Why
this form, and not the others*, *What it touches in the ranking set* and *The
pooled record-quality rule, measured*; `reports/data_review.md`, *The tail
owns the metric*, *Days in the ranking set that look like disruptions* and
*The same days in the training year*.

## 5. The unimpeded term

### 5.1 From OSM to a routable surface

Three defects repaired to a fixed point: coincident-but-unshared nodes,
nodes lying on edges they are not joined to, and ways that cross with no
shared vertex. Stand attachment with the snap distance recorded per stand.
Route descriptors (distance, turn angle on a Douglas–Peucker simplification,
junction-free running, backtrack) rather than one number.

Sources: `reports/taxi_routing.md`; `reports/penalty_sensitivity.md`;
`reports/runway_crosscheck.md`; `docs/modelling_notes.md`, *Graph construction*.

### 5.2 A worked airport: where every number is known

A fictional field on which every distance, turn and repair is hand-derived,
pushed through the pipeline unchanged. All 32 matrix cells agree to 11 mm
and 0.10°, integer descriptors exact; the runway-penalty crossovers fall
where the arithmetic puts them. It found three things in the pipeline, each
below its working tolerance, each measured: overshoot and undershoot select
different repairs; a merged node takes the last way's coordinates; a
two-feature stand's representative is decided by floating-point rounding.
The field also carries worked traffic in the challenge's own schema, so the
whole pipeline runs end to end in continuous integration without the
challenge data.

*This is the section that answers "how do we know the geometry is right"
without asking a reader to trust matrices nobody can check by hand.*

Source: `reports/synthetic_airport.md`, *In one paragraph*, *What the field
found* and *Worked traffic*, and its figure.

### 5.3 What the geometry recovers, and what it does not

Per airport, a pair's fastest decile follows its routed distance with a
residual of 85–190 s and an intercept of 313–740 s that no distance
explains. The largest structured effect on that decile is wake category:
mediums 35–281 s faster than heavies at the same stand and runway, uniform
across designators, which is a fixed pushback-to-line-up difference and not
intersection use. Intersection departures proper leave a signature of at
most 96 s per designator, constant, and absorbed by any model knowing the
designator; which stands use which entry is unobservable without surface
tracks.

Source: `reports/geometry_gaps.md`, *In one paragraph*, *The fit*, *The
fastest decile by wake category* and *The intersection signature, per
designator*.

### 5.4 Where memorization runs out, and what replaces it

*Added 4 September, after §9.1's first draft asserted the opposite of what
the data says; settled 5 September, when its figures moved from a scratch
calculation into the generator. Two of the scratch figures did not survive
the move (a network-wide count had been attributed to one airport, and the
transfer figures were a stand-in's rather than the report's), which is the
reason this paper quotes generators and nothing else.*

**How much of the test set is thin.** Of the 215,876 ranking departures,
0.52% sit on a (stand, runway) pair the training year never saw, 1.36% on a
pair seen ten times or fewer, and 7.95% on a pair seen between eleven and a
hundred times: **9.8% in total**. The share is concentrated. Frankfurt has
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
pairs seen one to ten times, worse than ignoring the pair and predicting
the airport mean, 519.0 s. On July the same comparison is 543.1 against
498.3. A mean over a handful of draws from a distribution several hundred
seconds wide is noise wearing the costume of knowledge.

**The geometry is the better prior, and by more than the usual one.**
Shrinking the pair mean toward an estimate, the textbook remedy, depends
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
tenth available, the transfer model (geometry plus the clock, the aircraft,
the plan lags, the traffic in the window and the weather, every identifier
removed) beats the only baseline available without it, the grand mean of
the other nine, at **ten airports out of ten**, by a median of 23.6% and by
as much as 43.4% at Heathrow. The geometric term *alone*, two fitted numbers
and five shared modulators, beats that mean at eight of ten, and loses at
Munich and Heathrow, by 4.1% and 1.5%. The speed the ten leave-one-out fits
recover runs from 11.8 to 13.2 m/s, a 12% spread about 12.42 m/s, with the
intercept at 434 to 500 s and the offset at 251 to 276 s. That stability is
the finding: the relation is a transferable physical constant rather than a
per-airport fudge, and it is the one thing in this entry that an identifier
model cannot reproduce even in principle, because at an unseen airport it
has nothing to encode.

Sources: `reports/geometry_gaps.md`, *Where memorization runs out*, which
carries the per-airport and per-month tables, the new-stand table with the
shape of each identifier, both forward splits by bucket, and the shrinkage
weights, none of them restated here; `reports/transfer.md`, *The speed, ten
times over* and *RMSE, per held-out airport*, for the per-airport transfer
table, likewise.

## 6. The excess term

### 6.1 What the ranking set allows, and the registry that enforces it

The rule is phase-specific: a feature may read what the ranking set exposes
*for the rows it reads from*. Every feature column is registered against its
raw sources and the phase each is read from; the test suite holds each phase
to its own list, and a poison test sets the hidden columns to nonsense and
asserts the frame does not move. Arrivals' on-block and taxi-in times are
exposed at 0.0% null and are read; no departure's block or taxi time is,
ever.

Sources: `reports/congestion.md`, *The leakage audit*; `reports/data_review.md`,
*Null rates*.

### 6.2 Congestion, corridor, and weather

Throughput, arrivals, demand, runway configuration, shared-corridor load,
and three weather regime flags per airport-hour. The corridor feature (other
movements weighted by how much of the subject's own taxiway route they share,
each edge weighted by the inverse of the routes using it) is the one feature
here no other entry has, and it is reported as it fell.

Source: `reports/congestion.md`.

### 6.3 The flight plan's own delay signal

Take-off minus the filed off-block estimate is the strongest single feature
in the model. As a ladder step it is worth 79.3 s in December (320.2 → 240.9
s ex-flagged) and 57.4 in July (407.3 → 349.9); measured at the `+ weather`
rung, by removing it and keeping everything else, 59.3 and 51.7. Either
way it is worth more than every congestion and weather block combined, which
net of it are worth 7.7 s and 5.8.

What it is, said plainly, because a reviewer will ask: take-off minus a
planning value fixed before departure equals the taxi-out plus the pushback
delay against that plan. The model is therefore learning the distribution of
pushback delay around a value it is given, which is the same construction the
record-quality term uses with the schedule, and admissible for the same
reason: `EOBT_1_flt` and `LOBT_flt` are planning values the ranking set
exposes, not post-hoc records of the blanked field. `AOBT_3_flt`, which is
such a record, is excluded.

Its held-out-pair contrast is −1.4 s [−5.8, +3.2]: it helps equally whether
the pair is new or familiar, which is what a feature about the day rather
than the place should do. The ladder is reported both with it and without
it, and the regime exposure is bounded in §7.3.

Sources: `reports/baseline.md`, *The ablation ladder* and *Held-out pairs*;
`reports/congestion.md`, *Regime exposure*.

### 6.4 The queue layer that was not built

An oracle diagnostic bounds it: regress a departure's residual on the mean
residual of its same-runway neighbors, quantities no admissible feature may
read, and which a queue model is precisely an attempt to approximate. On the
submitted rung, the ceiling is **8.2 s in December and 8.3 in July**; the
lag-1 autocorrelation of consecutive same-runway residuals under two
minutes is 0.19 and 0.09.

The plan-lag block is what closed it. At the top of the chain fitted
without that block the same ceiling is 31.5 s and 27.5, and the
autocorrelation 0.31 and 0.17, so most of what a queue model would have
found was the day's delay against plan, and the cheapest possible feature
had already taken it.

A saturation proxy (the share of sub-90-second gaps, the minimum gap, the
span of the last three gaps on the subject's runway) was then built against
that ceiling. Measured on 3 and 4 September, under the parameters then in
force, it bought nothing: after the parameter sweep of that week it cost
half a second in December and eight tenths in July against the rung it was
added to, and the rung carrying it had a ceiling of 10.0 and 8.5, within a
second of the model's own. Gate 3 was stopped on that measurement rather
than on the schedule, the stop is recorded in the plan, and the submission
is not fitted from the block. It stays on the ladder, and under the
parameters that ship it reads differently, which is reported rather than
smoothed over. Against `+ weather`, the rung it was added to, it now scores
232.7 s against 233.2 in December and 343.4 against 344.1 in July, a change
larger than the seed floor on both splits; its held-out-pair contrast is
−0.6 s [−1.2, −0.1]; and its ceiling, 9.2 s and 8.3, sits beside the
submitted rung's 8.2 and 8.3. It has never been measured on top of the rung
the submission is now fitted from, so what it would add there is not known.

*"The physical layer did not move RMSE once congestion was controlled for"
would have been a publishable finding; so is this. The thirty hours the plan
reserved for a queueing simulation were released by a diagnostic that took
minutes and needed no model.*

Sources: `reports/congestion.md`, *The ceiling*; `reports/baseline.md`, *The
ablation ladder*, *Held-out pairs* and *The noise floor*; `docs/plan.md`,
*When the data lands*, a dated record, for the measurements of 3 and 4
September.

### 6.5 The stand's recent history, and the bound inside it

The largest of the four blocks added last reads the arrivals at the
subject's own stand: the gap from the last arrival's on-block to the
subject's take-off, that arrival's lateness and taxi-in, whether it flew the
subject's aircraft type and operator, and the gap to the previous departure
off the same stand. Every field is one the ranking set exposes for arrivals
in full (§6.1). On the ladder it is worth 8.6 s in December (228.3 → 219.7 s
ex-flagged) and 9.1 in July (342.2 → 333.0), the largest single step above
the plan lags on either forward split; on the three airports July 2026
contains, 16.0 s (256.6 → 240.6). Its held-out-pair contrast is +1.8 s
[+0.8, +2.9]: it buys slightly more where the pair is new, which is what a
stand-level quantity should do where the encoding has nothing to say about
the stand.

**The bound, and where it holds.** A stand holds one aircraft, so an arrival
that on-blocks at the subject's stand before the subject takes off means the
subject had already pushed back, and the gap is a lower bound on its
taxi-out. That is why the block was built, and it is true only in a band.
Measured on the December rows of the rung that never read the gap: under
twenty minutes the bound holds in 97–100% of rows and the model already
respects it; at 20–25 minutes it holds in 94% of rows while the model
violates it in 17%, under-predicting those rows by 171 s on average; at
30–35 minutes, 71% and 56%; by 40–45 it holds in 11% of rows, and past
forty-five minutes essentially never, because a stand identifier can denote
a position that holds two aircraft and because aircraft are towed. Applied
as a rule it loses at every cap tried (+0.1 s at twenty minutes, +1.2 at
thirty, +20.6 at forty-five in December), since a floor can only act where
the model is below the gap, and where the bound is true the model already
obeys it. It enters as a feature, and the booster finds the band.

**Where the gain lands, and how much of it is the bound.** The block's
reduction in squared error is a sum over rows, so it can be attributed to
disjoint bands of the gap and the shares add. The 20–45 minute band is 1.0%
of December's rows and 20.9% of the cut; on the July three, 1.9% and 34.9%.
Inside it the previous arrival's type separates the two cases the band
contains. Where the type differs, a different aircraft took the stand after
the subject pushed: 900 December rows carry 17.3% of the whole cut, and
their mean under-prediction falls from +229 s to +5. Where it matches, the
subject's own inbound after a fast turnaround, 686 rows carry 3.6% and go
from +54 s to −23. That is the bound doing what the mechanism says, on the
rows it speaks about, and it is a minority of the block on every leg. Past an
hour the gap bounds nothing, and the block still earns there: 46.4% of
December's cut sits on gaps over an hour where the previous arrival was the
subject's own type, the subject's own inbound, against 10.1% on the July
three, where the short gaps carry the block instead. What it reads at those
gaps is the stand's recent history: the inbound's lateness and taxi-in, the
scheduled ground time, whether the aircraft on the stand was the subject's
own. That is a turnaround-pattern signal, admissible on exactly the same
footing as the bound and a different thing from it. That second share moves
by a factor of several between the legs, so a sentence about what this block
*is* has to name the leg it is true of.

**What survives an operational rule.** The block reads an arrival that
on-blocked *after* the subject's own off-block. Under the challenge's
post-hoc framing (the take-off time is given, and every take-off-anchored
window in the model has the same standing), it is admissible; under an
operational one it is not, and this is one of the two places in the entry
where exploiting the ranking process could be alleged (`AOBT_3_flt`,
excluded, is the other). How much of the block would survive an operational
rule is measured rather than argued. A diagnostic rung off the chain and out
of the submission re-anchors the block on the schedule (the stand's last
arrival at or before the subject's *scheduled* time, every column read from
that choice, so that nothing in it needs the take-off), and it keeps 42% of
the block's gain in December, 27% in July over all ten airports, and **4% on
the three airports July 2026 contains**, where the schedule-anchored gain,
+0.58 s, is inside that leg's seed floor of 1.02 s and cannot be told from
zero. The schedule anchor is stricter than operational, since at pushback
the arrivals up to the actual off-block are known and not only those before
the schedule, so these are lower bounds; but on the airports the ranking's
July month contains, the block is a post-hoc instrument almost entirely, and
this section says so rather than burying it.

**The headroom overshoot.** Each of the four blocks' refit gain is printed
beside the residual-fit headroom estimated for it before it was built
(`docs/review_headroom.md`), and a block that takes more than five seconds
beyond its estimate trips a test. The test was built as a leakage audit, and
it is not one: §7 of `docs/review_headroom.md`, written on 8 September,
before the test ever fired, established that a booster fitted on a frozen
model's stored residuals bounds a refit from below, so the gap between the
two is the value of the interaction rather than evidence of anything wrong.
The report now names the test for what it measures, and the name leakage
audit belongs to the source registry printed as an assertion in
`reports/congestion.md`, which is a leakage test and passes (§6.1). Under the
parameters that ship, on the July three, this block takes +16.0 s against an
estimated +10.3, an overshoot of 5.681 s at 1.55 times the estimate, so the
test fires, by 0.681 s, inside that leg's seed floor of 1.02 s: the verdict is
within what changing the seed alone produces. The block stays in the chain,
whose rule does not read this test, on the reading that an overshoot inside
the floor cannot tell leakage from noise. That reading was adopted after the
number was seen, and this paper says so; what carries the weight is the date
on §7, not the reinterpretation. The other reading is leakage, and it is the
one the test exists for: the registry and the poison tests of §6.1 bound what
the block can read, and the concentration above bounds where what it read
landed. The test has also read differently from one regeneration of the
ladder to the next, each time moved by something other than the block: 1.54
times the estimate on the ladder of 8 September, firing by 0.56 s; 1.48 on 10
September, missing by 0.029 s; 1.52 on 11 September, firing by 0.37 s; 1.42
on 19 September, missing by 0.655 s; and 1.55 under the parameters that ship.
The estimate itself is now regenerable: `reports/headroom.md` recomputes it
from the current residuals at a stated capacity, reads +10.00 s, and draws no
conclusion about the test. What the block rests on is the source registry and
the poison tests of §6.1; the parity of the gap on both sides of the data,
where no airport's share of stands re-taken within twenty minutes moves by
more than two points between 2025 and 2026; the concentration above, which is
consistent with the mechanism and is not proof against another route; and the
schedule-anchored measurement, which puts a number on the part of the block
that is the post-hoc reading.

Three smaller blocks stand on it in the chain (the operator read as the
callsign prefix, the destination's bearing and distance with the holiday
calendar, and the plan's two differences) and are read in
`reports/baseline.md`, *The Gate 2c blocks*, each net of the blocks below it.

Sources: `reports/baseline.md`, *The ablation ladder*, *What each block buys
where the pair is new*, *The Gate 2c blocks*, *The three airports July 2026
contains*, *Where the stand-occupancy gain lands* and *The same block without
the subject's take-off*, the last of which carries the headroom overshoot;
`reports/occupancy.md`, *The gap, on both sides*, *Where the bound holds*,
*The hard floor, and what it costs* and *Where the gain lands*;
`reports/headroom.md`, *The ratios*; `docs/review_headroom.md`, §7, a dated
record of 8 September; `docs/modelling_notes.md`, *Leakage watch*.

## 7. Validation

### 7.1 The design, fixed before anything was fitted

Two forward splits (December 2025 at all ten airports, July 2025 at all ten
with the three the July test set contains read on their own) plus a
held-out-pair GroupKFold and pair-frequency strata. Never random K-fold on
rows: congestion is autocorrelated over hours.

The held-out unit is the **pair**, not the stand: every stand and runway is
still seen in other pairs, so this bounds what the geometry buys on a new
combination and does not isolate what it would buy on a new stand, which no
split of this data can measure. The claim is a difference in differences,
what a block buys on held-out pairs minus what it buys on seen ones, so
composition cancels. Intervals are a cluster bootstrap over pairs, 1,000
draws, resampling stored residuals; no model is refitted.

Source: `reports/baseline.md`, *Held-out pairs*.

### 7.1.1 What the forward legs cannot see, as measured on 20 September

Three limitations of the design above, measured rather than asserted, on
20 September under the parameters then in force. None was repeated under the
parameters that ship, so every figure in this subsection is that day's, read
from the dated record the experiments wrote.

*The season columns.* On 20 September, under the parameters then in force,
the submission rung was refitted on December and on July without
`tko_month`, `tko_doy_sin` and `tko_doy_cos`, first seed, everything else
unchanged, under a rule fixed before the sign was known: no adoption follows
in either direction. Removing them improved December by 0.40 s and the
ranking mix by 0.30, and cost July 1.15 s on the three airports and 0.77 s
pooled over all ten. The sign is not the same on the two legs. Both forward
legs validate on a month and a day-of-year range the training side never
held, December after January to November and July after January to June,
and the ranking months do not: January 2026 and July 2026 each have their
twin in the training year. So the legs extrapolate on these columns where the
final fit interpolates, and the measurement answers a question the ranking
months do not ask. A held-out-month split, training around a month and
validating inside it, would answer the question they do ask, and it breaks
the forward discipline every leg here keeps; it is named because it is what
would have measured this, and it was not run. The columns stay in the model.

*Two instruments, and the first time they disagreed.* This entry sizes a
difference two ways. The seed floor is root two times the standard deviation
of five single-seed fits of one rung on one leg, and asks how far a figure
moves when nothing but the random seed does. The interval is a day-clustered
percentile interval over the evaluation set, days as the clusters and the
same draw for both arms, and asks how far the figure would move on another
sample of days. They measure different things, and they can disagree about
the same contrast. They did on the season columns above: three of the four
steps were outside the floor of the leg they were taken on, while every
interval that printed spanned zero, December all ten at −0.40 s against a
floor of 0.34 with an interval of [−1.6, +0.9], and July all ten at +0.77
against 0.23 with [−0.2, +2.3]. Nothing chose between the two instruments
there, and nothing needed to, because no adoption followed. They disagreed
again on 23 September, and §7.1.2 gives that occasion. **Where a figure in
this paper is called real or called noise, the sentence says which
instrument said so.**

*Recency, and what the gap leg can and cannot say.* On every 2025 leg the
most recent training month is also the seasonally nearest to the month being
scored: November for December, June for July, September for October. A gain
on those legs cannot separate recency from season. For the January 2026
ranking rows the two agree; for the July 2026 rows they do not, since the
most recent training month, December 2025, is seasonally opposite and the
seasonal twin, July 2025, is five months older. The gap leg was built to
break the confound: train January to June 2025 and score December, which is
the July 2026 geometry as nearly as one training year permits, without the
twin month. Two recency schedules were fixed in advance, weights halving
every 3 and every 12 months over the booster's fitting rows, with a screen
fixed at printed precision. On 20 September, under the parameters then in
force, the mild schedule did nothing measurable on the gap leg in either
direction: +0.10 s against that leg's own floor of 0.40 s, with an interval
of [−0.4, +0.6] spanning zero. It gained on the July forward leg past that
leg's floor and with an interval clear of zero, −0.84 s against 0.23 with
[−1.5, −0.4], and −1.61 s on the three against 1.02. It stopped because the
screen reads printed precision and, at that precision, 224.8 is worse than
224.7. **The reason no recency term ships is the screen, not a measured loss
on the gap leg.** The strong schedule is a separate case and should not be
read with it: it was worse on October by +0.37 s against October's own floor
of 0.19, a regression the run could measure. The confound itself stands as a
limitation that no leg inside one training year can resolve.

Source: `reports/preregistered-2026-09-20.md`, *What Experiment 2 measured*
and *What Experiment 3 measured*, a dated record of 20 September.

### 7.1.2 The rules under the parameters that ship, and how those parameters were chosen

*Printed precision, and the seed rule.* Printed precision in this entry is a
tenth of a second, and every seed floor is larger than that: under the
parameters that ship, 0.23 s on December, 0.54 on July over all ten airports,
1.02 on the three and 0.21 on October. So a screen written at printed
precision can fire, in either direction, on a difference the same run calls
seed noise, and on 20 September one did (§7.1.1). That is a property of the
rules as they were fixed; it was not changed after the fact, and the screens
were applied as written. The seed rule of 19 September reads floors and not
printed precision. Under the parameters that ship, the five-seed average
gains 1.11 s over the ladder's seed in December against a floor of 0.23, and
1.46 s in July against 0.54, beats every single seed on both legs, and
clears the rule with a margin. Under the sweep's cell of 21 September the
margin was thin: July gained 0.63 s against a floor of 0.61, and the report
itself says that the standard deviation behind a floor measured from five
seeds is uncertain by about 35%. The cell that ships replaced that margin
rather than resting on it.

*The two instruments, a second time.* On 23 September the floor and the
interval disagreed about the booster cell that ships: against the cell it
replaced it reads 0.4654 s better, 2.56 floors, while its day-clustered
interval, [−0.9492, +0.0009], spans zero by nine ten-thousandths of a second.
The rule that chose the cell reads the floor and nothing else, so the
interval decided nothing, and both are printed. It is the second time the
project has met that, after the season columns of §7.1.1.

*How the booster's parameters were chosen, including the part that was not
planned.* The parameters were chosen on October 2025, a month no rung of the
ladder scores, by two sweeps and one extension of the second, all fitting
January to September 2025. The first sweep, on 3 September, ran on the rung
that was then the top of the ladder and under fit rules that have since
changed, and its figures are not comparable with the second's (§9.3). The
second, pre-registered on 20 September and permitted once, read twenty-seven
cells on the submission rung under the fit rules that stand, one fit each,
and chose 3,000 trees, a learning rate of 0.015, 255 leaves and a minimum
child sample of 20, at 214.90 s against the old cell's 217.28 s, a move of
2.38 s against a seed floor of 0.19 s. That cell sat on an edge of the grid on
all four axes. The pre-registration said the grid would not be extended
again, and the edge was recorded as it required. The author reopened that
rule on 22 September, after the edge was recorded and knowing which way
October was still falling; that is a decision taken with the result in view,
and it is said here in those words rather than presented as the plan. The
extension's criteria were fixed in writing before any figure past the edge
existed. Three of the four axes were probed, one step out each,
independently, and they were chosen from the sweep's own gradients, which is
a design choice and is disclosed as one; the learning rate was the axis left
out. The trees axis won: 6,000 trees read 214.3041 s against 214.9045, 0.6004
s better, 3.31 floors, on one leaf setting, and there the day-clustered
interval agreed with the floor. The other two lost, 511 leaves by
0.2274 s and a minimum child sample of 10 by 0.2381 s. The trees axis was on
an edge again at 6,000, with October still falling at 4,000, 5,000 and 6,000,
and the extension's criteria closed it there.

*The cell that ships was taken on cost, from two the instrument cannot
separate.* On October the better reading is 6,000 trees at a learning rate of
0.015 and 255 leaves, 214.3041 s, against the shipped cell's 214.4391, and it
was not taken. The two differ by 0.1350 s, 0.74 of the floor, and by less
than half of the 0.2872 s the better cell moves between two draws of its own
seed, so this instrument cannot separate them. That is the situation the
one-standard-error rule of Breiman, Friedman, Olshen and Stone (1984) exists
for, and among candidates the evidence cannot separate, it takes the cheaper.
The cell that ships, 6,000 trees at 0.03 and 127 leaves with a minimum child
sample of 20, builds 762,000 terminal nodes against the alternative's
1,530,000 and fits October in 237.5 s against 321.8 s. The alternative is
named here as the equal-performing one, so the choice is visible rather than
implied, and the shipped cell is the worse of the two readings: it is
presented as the cheaper of two cells the evidence cannot separate, not as
the better score.

*The winner's curse, measured.* A search that reads twenty-seven cells at
one seed each and keeps the smallest keeps, in part, the luckiest. At a
floor of 0.19 s, which is root two times the standard deviation one seed
carries, the smallest of twenty-seven such readings is expected to sit about
0.27 s below its cell's own level from the selection alone. That figure is
an expectation, computed under independent, normally distributed seed noise
with every cell at one level, which is the largest the shortfall can be, and
not a measurement. The measurement came before the extension was probed: the
four cells within one floor of the sweep's best were refitted at a second
seed, and the sweep's chosen cell went from first of the four to last, with
four of the six pairs crossing. The trees result is read differently, as a
paired contrast within one seed, which the seed barely moves: going from
3,000 to 6,000 trees read −0.6004 s and −0.6019 s at the two seeds on 255
leaves, and −0.5591 s on 127, while the seed moved the level itself by
0.2872 s. So the trees effect survives the seed where the ranking of the
cells at 3,000 trees did not.

Sources: `reports/baseline.md`, *Seed averaging* and *The noise floor*;
`src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`, which is the
repository's own account of the sweeps, the extension and the cost decision;
`src/taxiout/provenance.py`, the basis of `FLOOR_JULY_ALL`, for the sweep
cell's July reading. The seed-stability check, the interval on the trees
step, and the order in which the extension's criteria and its figures were
fixed are recorded in the header and the message of
`scripts/commit_extension_probe.sh`.

### 7.2 The ladder, and the claim

Every rung on both splits, ex-flagged and with the record-quality term, plus
the seen-versus-held-out contrast per rung with its interval. The submission
is fitted from `+ plan differences`, the top of the chain the four blocks of
§6.5 climb, at 214.7 s (December) and 322.4 (July) ex-flagged on the
ladder's own single seed; §8 gives the figures of the model that ships. Each
rung is read against the rung below it, and the three encoding arms of §7.4
against `+ weather`, at 233.2 and 344.1.

The claim contrast, identifiers + time → identifiers + time + geometry, is
**+18.5 s [+16.7, +20.5]**: the geometry block buys 0.4 s where the model has
seen the pair and 19.0 s where it has not. The interval excludes zero.

Two neighboring rows are worth reading beside it, because they are what
makes the claim mean anything. The step from the geometric unimpeded term,
with the wake class, to the identifier model is −41.3 s [−47.1, −35.7]:
identifiers beat geometry outright where the pair is known, which is the
null this entry had to face. And the step into the geometry block from
identifiers is where that reverses. Together they say the two encodings are
not competing descriptions of the same thing (one memorizes a pair, the
other computes it), and the split that separates them is the only one that
can.

Three arms above `+ weather` carry a shrunk identifier encoding; they are
the subject of §7.4, one of them is carried into the submitted model, and the
claim contrast is measured near the foot of the ladder where neither rung
carries it.

Source: `reports/baseline.md`, *The ablation ladder* and *Held-out pairs*.

### 7.3 What could go wrong on the ranking months

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

Taking the whole shift as pushback delay anyway (the worst case, which the
control says it is not) and multiplying by each airport's training slope of
taxi-out on that lag (0.24 to 0.51 s per second) gives a per-airport bias of
at most +93 s, costing **+20.4 s at Istanbul, +12.8 at Frankfurt, +5.6
pooled in January and +3.0 in July.** Biases of opposite sign do not cancel
in the pooled figure, because each enters its own airport's squared error.

**The winter regime.** January 2026 froze for 48% of Frankfurt's hours
against 30% in 2025, 65% of Zurich's against 41%; Istanbul went from no
freezing hour to 15%. This is where a model fitted on 2025 extrapolates, and
the weather flags are what it extrapolates along, which is the argument for
regime flags rather than continuous weather drivers.

**The record-quality term's coverage.** Fitted on one airport's 2025 feed.
An airport that begins defaulting block times in 2026 is not covered, and no
rule fitted on 2025 could cover it. The alternative, a cell pooled across
airports, is not left as a judgment: the per-airport defaulted share by cell
measures the premise pooling rests on and refutes it in every bucket a pooled
cell would serve, on each of three training halves, which §4 gives with the
arithmetic of the earlier comparison. What remains is the exposure itself,
which is bounded and is stated rather than priced: 235 ranking rows sit where
a pooled rule would have covered them, 138 of them at Amsterdam in January
2026, and each occurrence at an airport whose feed changes costs one
day-scale row.

**And one extrapolation the final fit does not make.** The July leg's
Heathrow figures, the afternoon concentration of §9.5 among them, are read
from a model trained on January to June 2025, which holds no July. The
submitted model is fitted on the whole training year, which holds July
2025's afternoons, with the hour and the runway in its frame. So the leg
measures an extrapolation the submitted model does not make in the same form,
and its Heathrow figures are a diagnostic rather than a bound on the ranking
months.

Sources: `reports/congestion.md`, *Regime exposure* and *The weather regime*;
`reports/tails.md`, *The pooled record-quality rule, measured* and *What the
term is, stated plainly*; `reports/residual.md`, *EGLL's July*.

### 7.4 A block aimed at one place that paid in another: the shrunk encoding

*The measurement, the decision and the model agree: the arm this section
chooses is carried into the rung the submission is fitted from, through an
inference-time encoder that computes the encoding once over the training year
and is tested against the same leakage modes as the fold-wise one.*

**What was built, and against what.** §5.4 says where the identifier
model has nothing to memorize and that the geometry is the better prior
where it runs out. The obvious way to spend that inside the booster is an
empirical-Bayes encoding of the (airport, stand, runway) pair: its
out-of-fold training mean shrunk toward the geometric prediction rather
than toward a group mean, `(n × ȳ + k × ĝ) / (n + k)`, with `k` estimated
per airport as the ratio of within-pair to between-pair variance, the
sampling noise of each pair mean subtracted, and the same construction at
stand level as a second feature. It was pre-registered against a specific
bucket: pairs seen one to ten times, where the identifier exists and its
mean is a handful of draws. The ceiling it was built against was computed
before the arms were run, by scaling each bucket's airport-mean error by the
ratio the control achieves on the well-observed bucket and calling the
difference the attainable headroom. As the report computes it under the
parameters that ship, it comes to +0.32 s pooled, all of it in the thin
buckets, none by construction in the bucket that carries 91% of the rows.

**What was guarded.** Four leakage modes, each with a test that plants the
leak and asserts nothing moves: the encoding is recomputed inside every
fold, so a held-out pair sees `n = 0` and encodes to the bare prior; `ȳ` is
out of fold within training too, so a row's own target never enters its
own feature (the single-`k` version failed that test by twelve seconds,
which is how the out-of-fold `k` came to exist); `ȳ` reads unflagged rows
only, so one defaulted block time cannot become a three-row pair's
encoding; and the one overlap that cannot be guarded, the prior being
fitted on hundreds of thousands of rows of which a thin pair contributes a
handful, is stated rather than left to be found. `k` was checked for
route-length dependence by a rule fixed before the run. Until 19 September,
with the encoding fitted on the unflagged training rows, the check read
immaterial, at a ratio of 1.29 against a threshold of 1.5. With it fitted
since then on the rows every fit reads, the check is material on the
ladder's December leg, at 1.56, so there the pair-level `k` moves with route
length and the stand-level `k` does not; the encoding the submission fits
runs the same check over the whole training year and records its reading in
the manifest written beside the submitted file. A `k/2`–`2k` sensitivity
moves the arms by at most half a second; `k` is not tuned.

**A gate that was voided rather than failed.** The rule fixed in advance
required a positive difference in differences on the held-out-pair folds.
That condition cannot credit the component the block exists for: a held-out
pair encodes to the prior identically, so the pair encoding carries none of
the pair's own mean exactly where the split measures. The scope of that is
worth stating precisely, because overstating it would be self-serving. The
gate is applied to a whole *arm*, and it validly measures the rest of one,
the stand-level encoding and the removal of the `stand` categorical, which
it reads as decisively negative. It is not an instrument incapable of
returning a positive number; it is an instrument blind to the pair encoding.
A gate that cannot see the thing under test is not a test of it, so it was
voided as a test of the pair encoding, the arms took no credit from it, and
they were decided on the two instruments that can see a pair encoding, the
forward splits and the frequency strata, with the ranking set's own bucket
mix as the pooled figure. That replacement rule was written after the
numbers were seen, which is a real weakening and is said in the report; the
condition that does the rejecting (no worse than the control on either
forward split) is the original rule's own second half, unchanged, and it is
what rejects the first arm at +3.2 s on July.

**What chooses between the two arms that pass is a mechanism, not their
scores.** §5.3 measures what a designator carries that the routed distance
cannot compute: an intersection departure enters the runway at a taxiway
junction rather than at the threshold, and the signature of that is up to
96 s per designator and constant within one: absorbed by any model that
knows the designator, and not recoverable by an encoding shrunk toward a
matrix that routes to the threshold. So the runway categorical stays and
only the stand is replaced, for a reason measured before these arms were
run and independent of their numbers. The never-seen bucket corroborates
and does not decide: it is a few hundred rows, and a decision resting on
it would be a post-hoc rule resting on a thin stratum.

**What it bought, and where.** `+ shrunk encoding, replacing the stand`
clears the rule: 228.3 s against 233.2 in December, 342.2 against 344.1 in
July, 229.3 against 234.3 at the ranking mix, and +8.1 s worse on the
never-seen bucket against +14.8 for the other passing arm. The pooled figure
puts the other passing arm ahead, 229.1 s against 229.3, so the mechanism is
bought for 0.19 s pooled, and the report prices it rather than passing over
it. Decomposed by bucket in the sum-of-squares domain, **84% of the
ranking-weighted gain comes from pairs seen more than a hundred times**, the
bucket the ceiling had assigned no headroom at all, against 17% from the
`1-10` and `11-100` buckets together and 0.8% given back on pairs never seen.
On the strata it was aimed at it takes 17.2 s of the 30.8 s of headroom the
`1-10` bucket has, which at that bucket's 0.64% of the rows is worth 0.16 s
pooled. The honest name for the block is smoothed memorization that buys real
score on well-observed pairs. It is reported under that name, and §9.4 says
what the ceiling that missed it was actually measuring.

Two consequences follow. The prior is `ĝ`, so the encoding is
geometry-bearing, and the ladder's bookkeeping now counts it as such; an
ablation that removed the geometry block and left the encoding in would
not have removed the geometry. And the claim contrast of §7.2 is measured
at `identifiers + time → identifiers + time + geometry`, near the foot of
the ladder, where neither rung carries the encoding, so the claim figure of
§7.2 is untouched by this decision, by construction and not by luck.

Sources: `reports/baseline.md`, *The shrunk identifier encoding* and its
subsections; `reports/migration-2026-09-10.md`, *The encoding's rows,
2026-09-19*, a dated record, for the check as it read before that day;
`tests/test_encoding.py` for the four guards.

## 8. Results

*Nothing here is chosen by a leaderboard score. The challenge ranks
entries on a leaderboard: "Teams will be ranked based on the best Root
Mean Square Error (RMSE) achieved across their submissions", in the words
of its ranking page. This entry's rule for it: at most one upload per
frozen version, no upload used to choose a feature, a rung, a parameter
or a rule, and nothing read back from the scorer but `usedPairs` and the
score. So this section carries what the two forward splits say about the
model that is submitted, per airport and with and without the
record-quality term, and leaves one line for the organizers' figure.*

Every number here is from `reports/baseline.md` as regenerated on
24 September under the booster parameters adopted on 23 September (§7.1.2),
for the rung named in `taxiout.models.SUBMISSION_KEY`, `+ plan differences`:
the arm §7.4 chose, with the four blocks of §6.5 above it, fitted without
the recorded-taxi tail. The submitted model is that rung fitted under five
random seeds and averaged, as the report's *Seed averaging* section reads the
rule of 19 September. The average is measured on the pooled figures over the
rows the record-quality term leaves to the model. Every other figure below,
with the term, at the ranking mix and per airport, is the ladder's own single
seed of the same rung, and the text says so where it quotes one.

**Pooled.** On the December split, which stands in for January 2026 at all
ten airports, the submitted model scores **213.55 s** over the rows the
record-quality term leaves to it. On the July split, which stands in for
July 2026, it scores **234.20 s** on the three airports July 2026 contains,
the July figure this paper quotes for the reason §9.5 gives, and 320.90 s
over all ten. The ladder's single seed of the same rung scores 214.7 s,
235.11 and 322.4 on the same rows, so the average is worth 1.11 s, 0.90 and
1.46. On that single seed, with the term substituted on the rows it covers,
December reads 234.0 s and July over all ten 368.2; with those rows left to
the physical model alone, 364.8 and 620.5. The distance between the
ex-flagged figures and the every-row ones is 88 rows in December and 335 in
July, all at one airport, whose recorded taxi-out no aircraft produced; the
term is what closes it, and §4 says why that is the RMSE-optimal treatment
rather than a trick. Pooled at the ranking set's own bucket mix rather than
December's, the single seed is 215.7 s, and its bias is +7 s in December and
−3 in July.

**Per airport.** On the ladder's single seed, the three airports the July
test month contains, Frankfurt, Heathrow and Schiphol, just over half the
test rows, score 203.5 s in December and 235.1 in July. Frankfurt is 188.4
and 189.7; Schiphol 174.1 and 182.4; Heathrow, the widest of the three, 239.4
and 314.4. Six of the other seven run from Madrid's 170.5 to Paris's 252.5 in
December, and from Munich's 189.2 to Barcelona's 269.9 in July. Fiumicino is
the airport the term exists for: 294.4 s in December and 814.5 in July
without its rows, 1,131.9 and 2,043.6 with them and no term. The per-airport
claim contrast is positive and clear of zero at all ten airports, from
+9.0 s [+6.0, +11.9] at Schiphol to +30.7 s [+23.4, +38.0] at Frankfurt,
which is also the airport with the largest never-seen share (§5.4).

**What the score will and will not say.** The pooled test figure weights
the three July airports at just over half, so it is closer to a
Frankfurt–Heathrow–Schiphol figure than to a European one; the §7.3 bounds
put the plan-lag regime shift at up to +5.6 s pooled in January and +3.0 in
July, and the winter regime at Frankfurt and Zurich is where the model
extrapolates furthest. The organizers' figure, when it comes: **[score]**.

Sources: `reports/baseline.md`, *What is fitted, and on what*, *The ablation
ladder*, *The claim, per airport*, *Error by pair frequency*, *Seed averaging*
and *Per airport*; `reports/congestion.md`, *What it bounds*.

## 9. Discussion

### 9.1 What the geometry was worth, and where

The geometry block is worth 2.27 s on the December forward split and nothing
the seed floor can see on July, where it reads 0.14 s worse, and +18.5 s
[+16.7, +20.5] on pairs the model has never seen. The forward-split figures
and the held-out one are the same block; they differ because the splits ask
different questions.

The first draft of this section explained the gap by asserting that the
ranking months reuse 2025's stands and runways, so the test set has no thin
pairs and cannot reward the physical layer. **That assertion was wrong, and
§5.4 is the measurement that replaced it.** 9.8% of the ranking rows sit on
pairs the training year saw a hundred times or fewer, the share runs from
2% at Heathrow to 19% at Paris and is 15.6% at Frankfurt in July, 24 stands
are new outright, and on pairs seen one to ten times a pair's own mean is
worse than the airport mean by 53 s in December and 45 in July. The gap
between 2.27 s and +18.5 s is therefore not a property of the test set. It
is a property of how the two encodings are combined: a booster handed both
the identifiers and the geometry splits on the identifiers wherever they
are saturated, which on a forward split is nearly everywhere, and the 10.9%
of the submitted rung's splits that the geometry block and its quality flags
take is the residue. §7.4 is the one attempt to combine them differently; it
ships, and it bought its score from the saturated pairs too.

What survives from the first draft, and is still worth saying: the geometry
block is an injective encoding of (stand, runway), so on a pair seen
hundreds of times a saturated identifier model already knows everything the
distance could tell it, and there the geometry can only lose. The claim was
never that geometry beats memorization where memorization works. It is that
memorization runs out, that the data says where, and that the geometry is
what remains when it does; and, in the leave-one-airport-out fits, that it
is the only thing that remains.

Sources: `reports/baseline.md`, *Held-out pairs*, *The noise floor* and
*Where `+ plan differences` spends its splits*; `reports/geometry_gaps.md`,
*Where memorization runs out*.

### 9.2 What the honest tail treatment costs and buys

The term touches 107 ranking rows and is inert on the July half of the test
set. Its whole effect on the leaderboard runs through those rows. The
alternative, predicting only physically plausible taxi-outs and arguing the
point, was set aside because it would place mid-table regardless of model
quality, and the argument can be made just as well in a paper that also
scores.

Source: `reports/tails.md`, *What it touches in the ranking set* and *What
the term is, stated plainly*.

### 9.3 Limitations

Assembled from the open questions already recorded: stands OSM does not have
(the join places 96.3% of training departures on a matrix cell, and most of
the rest sit in whole numeric blocks at two airports); the snapped-stand
turn-angle bias, expected on physical grounds and undetectable by the
available proxy; intersection departures at stand level; the OSM snapshot,
fetched unpinned and rebuilt in the freeze week from its recorded
per-airport moments (§10); one airport's data feed driving the entire
record-quality term.

Two more that belong here because they concern how the numbers above were
produced rather than what they say:

**The booster's parameters were searched more than once, and the second
search was extended after its own rule said it would not be.** §7.1.2 gives
the account: a first sweep; a second, pre-registered and permitted once; an
extension of the second that the author reopened after its edge was recorded
and knowing its direction, under criteria fixed before its figures existed;
and a final choice between two cells the instrument cannot separate, taken
on cost. The first sweep, on 3 September, ran on the rung that was then the
top of the ladder, which carried the runway saturation block and neither the
shrunk encoding nor the four-hour exclusion, and its figures are not
comparable with the second's. Its grid ran from 235.0 to 244.8 s and its top
eight cells lie within 0.8 s of each other, and the same parameter cell moves
from 235.0 s to 217.28 s on the change of fit rules alone, before any
parameter changes. `src/taxiout/models.py` keeps both sweeps' accounts beside
the parameters.

**One ladder run was killed and restarted** after its generator was edited
mid-run; the reported figures are all from the completed run that followed.

Sources: `docs/modelling_notes.md`, *What the challenge data turned out to
contain* and *Open questions*; `reports/data_review.md`, *Stands*;
`reports/geometry_gaps.md`, *The intersection signature, per designator*;
`src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`.

### 9.4 A ceiling that measured the wrong thing

The headroom estimate of §7.4 was wrong twice, and the two errors are worth
more than the number they were about.

The first was arithmetic. The estimate scales each pair-frequency bucket's
airport-mean error by the ratio the control achieves on the well-observed
bucket and calls the difference the attainable headroom. On the `11-100`
bucket that difference is negative, **−12.1 s** as the report now computes
it: the control is already ahead of what the ratio would predict. The first
reading of that table described the bucket as "about zero" and credited its
magnitude, which turned a pooled ceiling of about a third of a second into a
claimed one to one and a half seconds. The generator now computes the pooled
figure both ways and prints both, because the gap between them is the whole
of the number: a negative bucket held at zero gives +0.32 s, the signs taken
literally give −0.79 s, and the ceiling is positive only because of that
convention. The error ran in the direction that made a piece of work already
proposed look worth doing, which is the direction such errors run.

The second is the one that matters. The ceiling was not wrong about what it
measured; it measured the wrong thing. It assumed, by construction, that
the bucket carrying 91% of the rows had no headroom at all (the ratio was
defined from it) and looked only where the block was aimed. The block then
bought 84% of its gain from exactly that bucket. A ceiling that can only see
the target of a feature cannot bound a feature whose mechanism is not the
one it was named for, and the honest reading of "smoothed memorization"
in §7.4 is that the mechanism was never the one the ceiling was built to
measure. The lesson is not that the ceiling should have been larger. It is
that a bound computed from where a block is aimed is a bound on the
aiming, and the report that carries it should say so before the arms are
run, not after.

Source: `reports/baseline.md`, *The ceiling this was built against* and
*What the chosen arm is, in its own terms*.

### 9.5 What the residual actually is

The error the model has left is a thin tail of individual holds. Three
measurements say so, and none of them is the absence of a queue model. Every
figure in this section is computed from the stored validation residuals of
the submitted rung's single-seed fit, the ladder's own; the five-seed average
of §8 stores none.

**Its shape.** On the December split, rows with a residual over ten minutes
are 1.78% of the rows and 38.3% of the squared error; over twenty minutes,
0.23% and 17.2%, and four in five of those are under-predictions. On the
three July airports the same tail is 1.82% of the rows and 48.6% of the
squared error, 55% of those rows under-predicted. The under-predicted rows
among those over ten minutes have a median actual taxi-out of 2,221 s against
a median prediction of 1,356 s in December, 2,597 against 1,695 on the July
three, and a ninetieth percentile of 3,466 s in December and 4,430 on the
July three.

**It is not shared.** The oracle of §6.4, a departure's residual regressed on
its same-runway neighbors' mean residual, a quantity no admissible feature
may read, buys 8.2 s in December and 8.3 in July on the submitted rung, and
the lag-1 autocorrelation of consecutive residuals under two minutes apart is
0.19 and 0.09. A queue everyone stands in would show in both. These misses
belong to one aircraft at a time.

**It is enriched where individual holds live.** Heavies are 21.6% of
December's rows and 33.3% of its worst one percent of residuals. Departures
without a Network Manager record (a plan canceled and refiled after a long
delay loses its join, so the missing record is itself a delay signal) are
0.94% of the rows, 4.8% of the worst one percent and 3.6% of the squared
error in December, and 1.99%, 11.0% and 14.8% on the July three, where they
run at 641 s RMSE against 235.1 for the three pooled. Departures whose stand
was taken by an aircraft of another type twenty to forty-five minutes before
take-off, the rows §6.5's bound speaks about, are 0.54% of the rows and 3.1%
of the worst one percent even after the block that reads them; inside that
band the block cut the mean under-prediction from +229 s to +5, which is
`reports/occupancy.md`'s measurement rather than this section's. And the
error grows with the plan lag as a conditional mean between two modes must:
against 181 s at the fifteen-to-thirty-minute lag most rows have, the
December RMSE is 415 s at forty-five to sixty minutes past the filed
off-block and 632 s at one to two hours. Long lag is sometimes gate delay and
sometimes a hold, and no exposed field separates them.

**It is concentrated in one airport's afternoons, on one leg.** On the July
leg, Heathrow's error runs +75.0 s [+17.0, +136.9] above December at the same
airport, 20,790 rows at 314.4 s against 20,000 at 239.4 s. It is not spread
over the day. The six hours from 14:00 to 19:59 UTC hold 60% of July's
squared error on 38% of its rows and run +141.1 s [+40.7, +249.2] above
July's other hours, while the same hours in December run −13.0 s
[−47.3, +19.5] against December's other hours. The afternoon is a July
effect at this airport and not a property of the airport. The second
concentration is by record: the 338 rows with no Network Manager record,
1.6% of July's rows, carry 21.3% of its squared error at 1,138.9 s, which is
+857.7 s [+300.7, +1,207.2] above the rows that have one; in December the
same class runs +207.2 s [+71.3, +389.0] above. Both contrasts are
structured, at day-clustered intervals clear of zero.

**Neither is a feature, and the reason is not modesty.** The hour of
take-off, the runway, the plan-lag band and the presence of a record are
columns the submitted rung already reads, and the callsign prefix is the
`+ callsign prefix` rung's own level, so there is no mechanism here the frame
lacks. What the July leg's model could not learn is what July's afternoons at
Heathrow are like, because that leg trains on January to June and holds no
July (§7.3). The oracle of §6.4 bounds what knowing the same-runway sequence
exactly could still buy at Heathrow on July at 9.7 s, and that is the scale
any queue reading of the afternoon would have to live inside. What is not
shown is which afternoons: the interval on the six-hour contrast runs from
+40.7 to +249.2 s under day clustering, which is what a handful of
afternoons carrying the effect looks like, and a per-day table would turn a
description into a search.

**One row, and which July figure this paper quotes.** Over all ten airports
the July split's ex-flagged RMSE on the ladder's single seed is 322.4 s. One
departure at Fiumicino recording a taxi-out of 87,002 seconds, a day-shift in
a cell the record-quality term does not cover, so that no term touches it,
carries 33.8% of that figure's squared error; the four-hour exclusion is
fit-side only and never reaches the scoring set, and July 2026 contains no
Fiumicino rows. The July figure quoted for the submitted model is therefore
the three-airport one, which §8 gives; the pooled figure stays in
`reports/baseline.md` with this reason beside it, and the rule that admitted
the blocks was read on the pooled figure as it was fixed, not on this one.

What is left is individual, it is not shared, and no field the challenge
exposes predicts it: the fields the model had not read were measured before
they were built (`docs/review_headroom.md` §2), and the four that carried
anything are in the model. That is a bound on this entry and, if the argument
holds, on every entry working from the same columns. What would move it is
information about the hold itself (a regulation, a de-icing queue, a runway
inspection), which the challenge data does not carry and the open dataset
that might, EUROCONTROL's own daily ATFM delay by airport and cause, was
excluded on its license (§2.2).

Sources: `reports/residual.md`, *The shape of the tail*, *What the tail is
enriched in*, *The error by plan lag*, *The largest rows* and *EGLL's July*,
all computed from the submitted rung's stored residuals; `reports/congestion.md`,
*The ceiling*, for the oracle and the autocorrelation; `reports/baseline.md`,
*Per airport*; `reports/occupancy.md`, *Where the gain lands*, for the in-band
before and after; `docs/review_headroom.md` §1–2.

## 10. Reproducing this

A reader brings four things to the repository at the tag: the challenge
data, from the organizers, which is not redistributable; the eleven Overpass
answers, fetched at their pins as described below or taken from the deposit;
and, from the deposit, the METAR observations and the two OurAirports tables
the entry read, cached fetches recorded with their requests and terms, which
a fetch made later would not return because OurAirports is rebuilt nightly.
From those and the committed inputs, every figure in this paper is
regenerated by a named script, or is quoted from a dated record that the
register holds to its digest, and `scripts/check_paper.py` tells the two
apart. One report is of the second kind by necessity:
`reports/determinism.md` is a measurement campaign under the parameters of
its day, whose records hold arrays derived from the challenge data and are
not redistributed, so a clean checkout cannot regenerate it and is not asked
to. The worked airport of §5.2 needs none of the four: it regenerates on a
bare checkout with no network and no data. The freeze ran in this order: the
pinned rebuild, every report regenerated and compared with the committed
one, the submission built, the commit tagged, then the reproduction from a
clean checkout and the deposit.

**What repeats.** The booster is fitted in LightGBM's deterministic mode
with column-wise histograms at a pinned thread count, and a fit repeats to
the bit: `reports/determinism.md` fitted one rung, under the parameters in
force when it was measured, six times in two processes at eight threads and
got one outcome, and the same outcome at two, four and fourteen threads. The
October seed floors that the parameter search of §7.1.2 was read against came
back identical to six places in two separate runs with a commit between
them, and all five single-seed fits of the sweep's cell reproduced to the
last digit; the headers of `scripts/commit_extension_probe.sh` and
`scripts/commit_equivalence.sh` are the record of those two runs. And on
27 September 2026 a fresh clone at the tag, with the challenge data and the
two tables brought in and the geometry fetched at its pins, rebuilt the 35
matrix files and the 52 residual files byte for byte, regenerated fourteen
reports in which no figure moved, passed this paper's own check against
them, and built the submission byte for byte, three times over between the
working copy and the clone; `reports/reproduction-2026-09-27.md` is the
dated record of that run.

**The geometry is rebuilt, not shipped.** The distance matrices are a
Derivative Database of OpenStreetMap under ODbL and are not in the
repository: a clean checkout rebuilds them from a pinned fetch, and the
deposit is the other route to the same files. A fresh fetch returns today's
map, so the rebuild is pinned. The committed figures were built from a fetch
on 1 September 2026 in which each airport was answered from the map as it
stood at its own moment, those moments are recorded per airport in
`taxiout.provenance.OSM_PINS`, and `scripts/fetch_osm.py --pin-from-register
--force` fetches each airport at its own. An Overpass answer's header never
shows whether a pin was applied, so the rebuild is verified by its effect:
`--compare` checks each airport's parsed elements against the answers the
matrices were built from, of which the deposit's copy is identical element
for element, and `--control-date` asks the instance that answered for a date
whose answer is known to differ, which shows that it applies a pin at all.
A header can show the opposite, that a pin could not have been applied,
because an instance cannot serve a state it has not ingested; the client
refuses an answer whose database state precedes the pin, and on 26 and
27 September the two fallback instances it knows answered from databases of
May to July and were refused, every accepted answer coming from the primary.

**The paper checks itself.** `scripts/check_paper.py` reads every figure in
this paper and requires it to appear, at the paper's own precision, in the
file its section's source line cites and, where that line names a section of
the file, in that section. A figure found only in a dated record must be
listed as historical with a reason, and one computed from report figures
must carry its arithmetic. It is a check of presence and not of correctness:
a short figure can match a number in a long report by chance, so a passing
check certifies that every figure appears where its section says it comes
from, and the match it prints for each figure is how a reader tells the
right number from a coincidence.

Sources: README, *Reproducing* and *Data sources*;
`reports/reproduction-2026-09-27.md`, a dated record; `reports/determinism.md`,
*The booster*, a dated record; `scripts/check_paper.py`; `docs/plan.md`,
freeze section.

## References

From `docs/prior_art.md`, *Sources*.
