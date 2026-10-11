# Recovering the airport surface: taxi-out prediction from OpenStreetMap geometry

*Every figure is quoted from a named report or a dated record, never retyped
from memory, and `scripts/check_paper.py` holds it there. The model froze at the
commit tagged `freeze-2026-10-04`, made on 27 September 2026, and nothing fitted
or validated on the training year moves after it. Three things came after the
tag, each dated and each a record. On 28 September the entry took up the ranking
files the organizers had re-issued on 4 September (§2.1), which moved every
figure computed on the ranking set and none computed on the training year. On 1
and 2 October the ladder was refitted with every month of 2025 held out in turn,
a diagnostic that decides nothing and that §8 and the annexes read. On 2 October
one rule was added to the submitted file for a class of three training
departures, under a pre-registration written before it was evaluated (§4); the
booster is untouched by it. The one figure that arrives later, the organizers'
score, is marked where it goes. And on 3 October three candidates left on the
table before the freeze were measured on top of the frozen model under a rule
fixed before their figures existed; the rule adopted none, and the submitted
file is unchanged by them (Annex J). After the organizers' first figure, read on
4 October, the entry ran one more week under pre-registrations written before
the figures they govern, and the file that is scored is now v11: §8.7 gives the
four files after v7, what each added on top of the frozen model and what the
organizers read back for each.*

Team `funny-firefly`. PRC Data Challenge 2026.

---

## Abstract

This entry tests one hypothesis: that the taxiway geometry recovered from
OpenStreetMap carries to stand–runway pairs a model has never seen, where a
memorized pair mean has nothing to carry. Taxi-out at ten European airports is
predicted from a physical decomposition:
unimpeded time recovered from OpenStreetMap taxiway geometry, excess from the
observable traffic and weather regime around a departure, and a record-quality
term for rows whose recorded value no aircraft produced. The geometry is routed
on a graph repaired to a fixed point and verified on a worked airport where
every distance is known by hand. Its value is stated as a difference in
differences on held-out stand–runway pairs: the geometry block buys **+18.5 s
[+16.7, +20.5]** more where the model has never seen the pair than where it has,
in every one of the twelve months of the training year read separately, and the
relation it rests on, a taxi speed of 11.8 to 13.2 m/s recovered ten times over
with a different airport left out each time, transfers to an airport the model
has never seen at all, where an identifier model has nothing to encode. Just
over a tenth of the 2026 test rows sit on pairs the training year saw a hundred
times or fewer, and on the thinnest of them a pair's own mean is worse than
ignoring the pair; the geometry is the better prior there by some twenty seconds
over the airport mean. The strongest single feature is not geometric but the
flight plan's own delay signal, worth 79 s in December and 57 in July, and once
it is in the model an oracle diagnostic bounds the whole queueing layer at about
eight seconds, so the queue model the plan reserved thirty hours for was not
built. A shrunk identifier encoding aimed at the thin pairs, guarded against
four leakage modes, bought 4.9 s in December and 1.9 in July, and the December
gain came mostly from the pairs it was not aimed at; it is in the submitted
model and is reported as what it is. That model, the mean of five seeds, scores
213.55 s on a December forward split and 320.90 on a July one, each over all ten
airports and over the rows the record-quality term leaves to it, and 234.20 on
the three airports the July test month held until the organizers re-issued it
with all ten. Read with every month of 2025 held out in turn, a design that is
not forward in time, the same model pools to 243.7 s over the year on those
rows, 275.9 s over every row with the term, and seven recorded taxi-outs of over
four hours that no term covers separate the two readings from a third, 217.8 s,
printed as a diagnostic. After the organizers' first figure the entry added,
each under a pre-registration fixed before its figures, features built on the
network manager's recorded off-block time, a CatBoost member, and ground events
from open ADS-B and OPDI data fused on top of the booster at the airports where
they help out of fold; §8.7 gives what each was worth on the training year and
the organizers' figure for every file uploaded.

## 1. Introduction

### 1.1 The problem, and the form the challenge gives it

Taxi-out is the interval from off-block to take-off. The challenge asks for it
in a form that inverts the operational one: the take-off time is given and the
off-block time is blanked. This is a post-hoc estimation task, and the
organizers say so: the rationale page names post-operations analysis, "excess
fuel burnt/CO2 production compared to unconstrained status", as the use. Both
framings are stated here, and the entry builds to the challenge's; the
alternative, anchoring only on schedules and filed estimates, is discussed in
§7.2 and was set aside as answering a question the challenge did not ask.

Source: `docs/modelling_notes.md`, *Leakage watch*; the rationale page.

### 1.2 What I set out to test

Every departure in the challenge data carries a stand and a runway, and the
natural model for a problem of this shape learns what each stand–runway pair has
cost before. That works on the pairs a model has seen. It cannot work on a pair
it has not, and a test month always holds some: just over a tenth of the 2026
test rows sit on pairs the training year saw a hundred times or fewer.

My central hypothesis is that the physical structure the challenge data omits,
the route an aircraft taxis from its stand to its runway, can be recovered from
OpenStreetMap and carries over to stand–runway pairs a model has never seen,
where a memorized pair mean has nothing to carry. How it would be judged was
fixed on 2 September, before the first model was fitted: "What
the geometry buys has to be argued from the held-out-pair split, not from this
table" (`docs/plan.md`, the entry for Gate 1). The test is therefore a
difference in differences: how much more the geometry block is worth on pairs
held out of training than on pairs the model has seen.

The hypothesis holds. The geometry block buys **+18.5 s [+16.7, +20.5]** more
on held-out pairs than on seen ones, and the contrast clears zero in every one
of the twelve months of the training year read separately (§8.5). On the
December forward split the same block is worth 2.27 s, and on July nothing the
seed floor can see. It is worth acknowledging that this gap is the finding
rather than a disappointment: the test months reuse the training year's pairs,
so a forward split cannot see what the geometry is for (§7.2, §9.1).

The rest of the model was built around that claim, and each part entered
through a test whose criterion was written down before its figures existed. Two results
shape the paper as much as the geometry does. The target's heavy tail is not
what it appears to be, and the honest treatment of it is a separate term stated
for what it is (§4). The excess term is small once the flight plan's own delay
signal is in the model: 7.7 s in December and 5.8 in July against that signal's
79.3 and 57.4, and the queue layer that would model it further is bounded to
about eight seconds by a diagnostic that needs no model (§6.3, §6.4).

The tests under the central hypothesis, each with its dated anchor in the
repository:

| what was tested | anchor | outcome | where |
| --- | --- | --- | --- |
| congestion from the fields the ranking set exposes, and a queue model only if congestion proved binding | `docs/plan.md`, 2 September, Gates 2 and 3 | real and small; the queue layer bounded and not built | §6 |
| a record-quality term, without which no RMSE means anything | `docs/plan.md`, 2 September, Gate 1 | supported on the training year and, through v8, on the test months | §4, §8.7 |
| seed averaging, recency weighting, season columns | `reports/preregistered-2026-09-20.md` | seed averaging adopted; the other two not | §7.1.1, §7.1.2 |
| three candidates left at the freeze | `reports/preregistered-2026-09-28.md`, `-10-03.md` | none adopted | Annex J |
| a rule for the plan-shifted departures | `reports/preregistered-2026-10-02.md` | fixed and applied | §4, Annex I |
| the two tail rules a liability on the 2026 rows | `reports/preregistered-2026-10-04.md`, §1 | refuted by v8's figure and withdrawn | §8.7 |
| candidates after the first score, and the term's rows predicted per row | `reports/preregistered-2026-10-04.md`, §3 and §4; `-tails.md` | three adopted as one combination; the per-row predictor adopted | §8.7 |
| ground events from open ADS-B and OPDI, fused on top of the booster | `reports/preregistered-2026-10-04.md`, §5, and its four amendments | adopted gated by coverage, then by airport, the second defined after the first's figures were seen | §8.7 |
| candidates read airport by airport, and a selector among them | `reports/preregistered-2026-10-06-last-week.md`, `-airports.md` | the refitted member and the selector adopted; the per-airport models not | §8.7 |
| a blend weight and a larger stacker | `reports/preregistered-2026-10-07-group-c.md` | neither adopted | §8.7 |

Analyses with no criterion fixed in advance, the held-out year and the annexes
built on it among them, are reported as analyses and never as tests. One test
was decided on a score rather than on the training year. On 4 October, after
v7's figure but on an analysis computed before it, the two tail rules were
switched off for the 2026 rows; the organizers' figure for that file
contradicted the judgment, and I withdrew it the same evening. §8.7 states it as the one
exception to the rule this entry otherwise kept.

### 1.3 Prior art, and what is new here

Building a routable graph from OSM `aeroway` geometry is not new; `traffic`
(Olive, MIT) ships one, and Olive, Waltert & Schultz (ATRD 2025) document the
same defects this pipeline repairs. Ravizza et al. (2013) established that route
shape (turn angle, junction-free running) predicts taxi speed nearly as well as
distance. What appears not to have been done is the downstream half: these
recovered quantities as regression features for taxi-out in this competition. No
2024 or 2025 entry appears to have used OSM at all.

Source: `docs/prior_art.md`.

## 2. Data

### 2.1 What the challenge supplies, and what it withholds

Ten airports, the ten the challenge's data page lists, not the eleven of the
entry's own airport list. January 2026 is a census of all ten,
152,719 departures, and July 2026 of all ten, 192,122: 344,841 rows to predict.
It was not always so. The ranking files this entry was built on, taken on 2
September, held July 2026 at three airports, Frankfurt, Heathrow and Schiphol,
and 215,876 rows in all; the organizers replaced both files on 4 September with
versions in which July holds every airport, and in which about 92% of the
departures the two versions share have a movement time moved by up to twelve
seconds. The twelve training files did not change. The entry found the
replacement on 28 September, in the check that precedes an upload, and reads the
files of 4 September from that day: the model does not change, because nothing
in it was chosen on the ranking set, and every figure computed on the ranking
set was regenerated and is what this paper quotes. The three airports keep a
place in the paper as a second July reading, *the three*, because the validation
design of §7.1 was fixed to read them on their own while July was expected to
hold them alone. The ranking set blanks `BLOCK_TIME_UTC_mvt` for *every*
departure, not only the subject, which retires any feature counting other
departures' pushbacks. `AOBT_3_flt` survives and is excluded by decision.

Sources: `reports/data_review.md`, *Airports and months* and *Inventory*;
`reports/ranking-refresh-2026-09-28.md`, a dated record, for the files of 2
September against those of 4 September.

### 2.2 External data, and the license position

OpenStreetMap (data (c) OpenStreetMap contributors, ODbL 1.0), OurAirports
(public domain), Iowa Environmental Mesonet
METAR (public domain), and, for the fusion model of §8.7 alone, the adsb.lol
globe history (ODbL 1.0). Each is openly licensed, documented, and re-fetchable by
a scripted, cached client: the prize rules require additional datasets to be
"openly available under an open source license", and a dataset a reader cannot
re-fetch fails the reproduction clause. The fusion also reads EUROCONTROL's
OPDI flight events, whose site states no data licence; the entry uses OPDI on
the organizers' confirmation that it is open to the challenge, fetches it with
a scripted client like the others, and tracks and deposits no OPDI file and no
table built from one. OpenSky historical surface tracks were
considered and ruled out for the model: research-use access is not an
open-source license. OpenSky's ten published scientific datasets were reviewed
on 20 September 2026 against the same test plus two more, 2025 ground coverage
at the ten airports and a mechanism the exposed fields lack; none passes all
three, and the README records each verdict.

Source: `docs/prior_art.md`, *External data: license positions*; README.

### 2.3 Reproducibility and the ODbL boundary

The repository ships GPLv3 in totality. The distance matrices it generates are a
Derivative Database of OSM and carry ODbL, so they are regenerated rather than
committed; the Zenodo deposit [10.5281/zenodo.23133051](https://doi.org/10.5281/zenodo.23133051) carries them under ODbL, with the eleven
Overpass answers they derive from, and beside them the two public-domain inputs
the entry read and a later fetch would not return, the METAR observations and
the OurAirports tables, and, also under ODbL, the adsb.lol event tables the
fusion reads. The reports are Produced Works and carry the figures.
`make licence-check` enforces the boundary, including that no literal stand
identifier from either source appears in prose.

Source: `data/processed/taxi_distance/ODbL-NOTICE.md`; README, *Data sources*.

## 3. Method

This section states the whole method in one place and in order, so that a reader
can see what the submitted file is a function of before the sections that argue
for each part. Nothing here is new to the paper; §4 to §7 carry the measurements
and the reasons.

### 3.1 The decomposition, and the pipeline in order

    taxi-out = unimpeded(geometry) + excess(congestion, regime) + record-quality term

Mapped onto the organizers' own framing of excess over unconstrained status. The
ablation ladder is this decomposition read bottom-up: each rung adds one term,
and the held-out-pair split measures the one claim that distinguishes the entry.
The file that is scored is produced by these steps, in this order, by
`scripts/make_submission.py`:

1. **The frame.** Every departure of the twelve training files and of the
   ranking file is joined to its stand-to-runway route on the OpenStreetMap
   graph of its airport (§5.1), to the METAR regime of its airport-hour, to its
   destination's bearing and distance from OurAirports, and to the windows of
   other movements around it that the ranking set exposes (§6.1). The same code
   builds both sides; a feature reads nothing on the ranking side that it does
   not read on the training side.
2. **The record-quality term.** Fitted on the training rows alone: per (airport,
   flight-plan record, lag band) cell, the share of rows whose recorded taxi-out
   is the schedule lag, kept where the cell has at least five rows and the share
   at least 5% (§4). Its rows are taken out of every fit below and predicted by
   the term.
3. **The fitting rows.** Training rows the term does not cover, with every
   recorded taxi-out over four hours left out of the fit and never out of the
   scoring (§7.1.2, *The recorded-taxi tail*).
4. **The shrunk encoding.** The (airport, stand, runway) pair's out-of-fold
   training mean shrunk toward the geometric prediction, computed once over the
   fitting rows and overlaid as a feature in place of the stand categorical
   (§7.4).
5. **The booster.** One gradient-boosted model on the fitting rows, the feature
   blocks of §3.2, the parameters of §3.4, fitted five times under consecutive
   random seeds and the five predictions averaged.
6. **The term, substituted.** On the rows the term covers, its prediction
   replaces the booster's.
7. **The plan-shifted departures.** On the rows of one further class, a
   departure with a flight-plan record whose take-off falls more than four hours
   after both of the plan's off-block estimates, a rule of the term's own form
   replaces the booster's prediction (§4, *The plan-shifted departures*). In the
   ranking set it is one row.
8. **Rounding and the clip at zero**, and the file written in the template's own
   order and integer type.

Since v9 the booster of step 5 carries two more feature blocks and is blended
with a CatBoost member, the term's rows of step 6 are predicted per row, and
since v10 a stacker on ground events replaces the prediction on covered rows at
the airports its gate keeps. §8.7 gives each change, and `taxiout.models`
records each decision with its basis.

Sources: `docs/modelling_notes.md`, *The model, stated as a decomposition*;
`reports/tails.md`, *The cells* and *The prediction*; `src/taxiout/models.py`,
`fit_rows`; `scripts/make_submission.py`.

### 3.2 What the model reads

The booster's feature blocks, in the order the ladder adds them. Each block's
rung key is the name the reports use; *what it reads* is the raw fields or
external data behind it; the last column is where that data comes from, which is
the rule of §6.1 (a feature may read what the ranking set exposes for the rows
it reads from).

| block | rung key | what it reads | from |
| --- | --- | --- | --- |
| identifiers + time | `identifiers` | airport, stand, runway, pair; the clock and the calendar of the take-off | the departure's own exposed fields |
| geometry | `identifiers-geometry` | the routed stand-to-runway distance, turn angle, junction-free running and backtrack, and the quality flags of the route | OpenStreetMap, routed per airport (§5) |
| aircraft | `identifiers-geometry-aircraft` | wake class and aircraft type | the departure's own exposed fields |
| plan lags | `plan-lags` | take-off against the plan's off-block estimates and the schedule | the Network Manager record, exposed |
| runway throughput | `throughput` | movements on the subject's runway in windows around its take-off | other movements' exposed take-off and landing times |
| arrivals | `arrivals` | arrivals in the windows, their taxi-in and lateness | arrivals' exposed on-block times |
| demand | `demand` | departures scheduled and planned in the windows | other departures' exposed plans |
| configuration | `configuration` | the runways in use around the take-off | other movements' exposed runways |
| shared corridor | `corridor` | other movements weighted by the share of the subject's own route they use | the routes of §5 and other movements' exposed fields |
| weather | `weather` | three regime flags per airport-hour: freezing, low visibility, precipitation | Iowa Environmental Mesonet METAR |
| shrunk encoding, replacing the stand | `encoding-for-stand` | the pair's training mean shrunk toward the geometry, in place of the stand categorical | the fitting rows (§7.4) |
| stand occupancy | `occupancy` | the last arrival at the subject's stand: the gap to its on-block, its lateness and taxi-in, its type and operator; the previous departure off the stand | arrivals' exposed fields (§6.5) |
| callsign prefix | `callsign` | the operator, read as the callsign's prefix | the departure's own exposed fields |
| destination and holidays | `destination` | the destination's bearing and distance; the public-holiday calendar | OurAirports; the holiday tables |
| plan differences (ships) | `plan-diffs` | the plan's own off-block values against each other | the Network Manager record, exposed |

The ladder also carries rungs the submission is not fitted from: two reference
rungs that fit no booster (the airport mean and the PRU reference), the
geometric unimpeded term alone and with the wake class, two further encoding
arms, a runway-saturation block, a schedule-anchored reading of the
stand-occupancy block, and the congestion chain fitted without the plan lags.
§7.2 and Annex A read them.

Sources: `reports/baseline.md`, *What is fitted, and on what* and *The ablation
ladder*; `reports/congestion.md`, *The leakage audit*; `src/taxiout/models.py`,
the rung table.

### 3.3 Two terms that bypass the booster

The record-quality term and the plan-shifted departures are rules, not fitted
models: each has the form `share × lag + (1 − share) × rest_mean`, where the lag
is a planning value the ranking set exposes and the rest mean is the airport's
ordinary taxi-out. The term reaches 383 of the 344,841 ranking rows, all at one
airport; the plan-shift rule reaches one. §4 gives the mechanism, the evidence
and the decision for each, and §8 reports every pooled figure with and without
the rows they cover, because a reader who disagrees with either rule can
subtract it.

Sources: `reports/tails.md`, *What it touches in the ranking set* and *The
plan-shifted departures*.

### 3.4 The booster, its parameters, and the seed average

LightGBM, `objective` regression, 6,000 trees at a learning rate of 0.03, 127
leaves, a minimum child sample of 20, a column sample of 0.9 and a row sample of
0.8 per tree, categorical splits capped at 64 thresholds with a smoothing of
20.0, no early stopping (stopping on the set being reported is tuning on it), in
LightGBM's deterministic mode with column-wise histograms at eight threads, so
that a fit repeats to the bit (§10). The four parameters that were searched were
chosen on October 2025, a month no rung of the ladder scores, under a rule fixed
before the search and an extension the author reopened after its edge was
recorded; §7.1.2 gives the account. The submitted model is the rung `+ plan
differences` fitted five times under consecutive seeds and averaged, adopted
under the rule of 19 September (§7.1.2): on the December forward split the
average gains 1.11 s over the ladder's seed against a floor of 0.23, and on July
1.46 s against 0.54.

Sources: `src/taxiout/models.py`, `BOOSTER_PARAMS`, `BAGGING_SEEDS` and
`SEED_AVERAGE_ADOPTED`; `reports/baseline.md`, *What is fitted, and on what* and
*Seed averaging*.

### 3.5 The validation design

Six readings, three fixed on 2 September before anything was fitted and three
added later and said to be later. Never random K-fold on rows: congestion is
autocorrelated over hours.

| reading | training rows | scored rows | stands in for, or asks | fixed |
| --- | ---: | ---: | --- | --- |
| December forward split | January to November 2025, 1,919,370 | December 2025, all ten airports, 165,677 | January 2026 | 2 September |
| July forward split | January to June 2025, 1,005,519 | July 2025, all ten, 190,713; *the three* (Frankfurt, Heathrow, Schiphol) read on their own as well | July 2026 | 2 September |
| held-out pairs | the year minus a third of every airport's (stand, runway) pairs, three folds of 1,303,727 to 1,338,046 | the held-out pairs, 676,400 to 712,627 per fold, beside a 5% sample of seen-pair rows | whether a block carries over to a pair never seen; the claim of §7.2 | 2 September |
| October | January to September 2025 | October 2025, all ten | the leg the booster's parameters were read on, scored by no rung of the ladder | 20 September |
| the gap leg | January to June 2025 | December 2025 | recency against season, for the July 2026 rows whose nearest training month is seasonally opposite (§7.1.1) | 20 September |
| every month held out | the other eleven months of 2025 | each month of 2025 in turn, 143,732 to 191,182 rows | how the figures and each block's worth are distributed over the year. Not forward in time, and a diagnostic | 1 October |

The held-out unit of the pair folds is the **pair**, not the stand: every stand
and runway is still seen in other pairs, so the folds bound what the geometry
buys on a new combination and not what it would buy on a new stand, which no
split of this data can measure. The held-out months train on months after the
one scored as well as before, so they cannot replace the forward splits as
proxies for the test months; what they add is twelve readings of every figure
instead of two, and every month of the year the claim can be read in.

Sources: `reports/baseline.md`, *What is fitted, and on what*, *Held-out pairs*
and *Seed averaging*; `reports/monthly_ladder.md`, *What this is* and *The
harness is the ladder's*; `reports/preregistered-2026-09-20.md`, a dated record,
for the gap leg.

### 3.6 The instruments

Three, and the paper says which one speaks wherever a figure is called real or
called noise.

- **The seed floor**, root two times the standard deviation of five single-seed
  fits of one rung on one leg: how far a figure moves when nothing but the
  random seed does. Under the parameters that ship it is 0.23 s on December,
  0.54 on July over all ten airports, 1.02 on the three and 0.21 on October;
  over the twelve held-out months it runs from 0.07 s in April to 1.56 s in
  February, and is 0.16 s over the year pooled. A difference inside the floor is
  within what changing the seed alone produces. The floor is measured on the
  shipped rung and applied to every block's change, an approximation the reports
  state.
- **The day-clustered interval**, a 95% percentile bootstrap over the evaluation
  set with days as the clusters and the same draw for both arms: how far a
  figure would move on another sample of days. It and the floor measure
  different things and have disagreed (§7.1.1, §7.1.2).
- **The pair-clustered interval** of the held-out-pair claim, 1,000 draws
  resampling whole pairs from the stored residuals: no model is refitted.

Printed precision is a tenth of a second, finer than every floor, so a screen
written at printed precision can fire on seed noise, and once did (§7.1.1).

Sources: `reports/baseline.md`, *The noise floor* and *Held-out pairs*;
`reports/monthly_seeds.md`, *Per month and over the year*.

### 3.7 What was fixed before the numbers

Every rule the entry's figures are read against was written down before the
figure it governs existed, with four exceptions that §9.3 lists in a ledger with
what was fixed, what was measured and what was decided after. The written rules
are in the repository as dated records held to their digests: the
pre-registration of 20 September (the season columns, recency and the sweep's
reading rule), the upload rule of 19 September (at most one upload per frozen
version, nothing chosen on a score, nothing read back but `usedPairs` and the
score), the pre-registration of 2 October (the plan-shifted departures), the
pre-registrations of 28 September and 3 October (the three candidates measured
after the freeze, Annex J), and the regeneration record of 28 September. A
figure found only in one of them is marked historical in the paper's own check,
so a reader can tell a figure the repository regenerates from one it only holds.

Sources: `reports/preregistered-2026-09-20.md`,
`reports/preregistered-2026-09-28.md`, `reports/preregistered-2026-10-02.md`,
`reports/preregistered-2026-10-03.md` and
`reports/ranking-refresh-2026-09-28.md`, dated records; `docs/provenance.md`,
*Frozen documents*; `scripts/check_paper.py`.

## 4. The tail, and what the scoring rule makes optimal

*This section is deliberately early and deliberately its own: the 2025 committee
praised "a thoughtful analysis of noise in the data", and the term is the part
of this entry most open to misreading.*

Rows whose error under the airport mean exceeds an hour are 0.086% of training
departures and carry 50.9% of that predictor's squared error. The data review
read them as day-shift artifacts. Building the term found the mechanism behind
most of it: at one airport, a departure with no flight-plan record and a
take-off hours after schedule has, nine times in ten, a recorded off-block time
within a minute of the *schedule*. The block time was defaulted, so the recorded
"taxi-out" is the schedule lag, which every ranking departure exposes. The
day-shift is real but small.

The term is the conditional mean per (airport, record, lag-band) cell, computed
knowing that mechanism: `share × lag + (1 − share) × rest_mean`. Three cells
qualify, all at one airport, 383 of 344,841 ranking rows, 107 in January and 276
in July. It is **the RMSE-optimal prediction under the stated scoring rule for
rows whose recorded value no aircraft produced**, and nothing else. Every RMSE
in this paper is reported with and without the rows it covers.

**Why an uncovered airport is left to the physical model, and what measures
that.** A cell counts as covered only where enough of its rows carry the
mechanism, and an airport with no covered cell is left to the physical model
entirely. The alternative considered when the term was designed was to pool
cells across airports, and on the 2025 validation legs a pooled design is worse
than the adopted mixture on all three readings: 390 s against 341 on December,
586 against 461 on July pooled, and 385 against 339 on the three. That is the
arithmetic of the choice and not the whole of the answer, because it compares a
differently shaped design, and because the case that presses the question is not
a validation leg. At Amsterdam the ranking set lists six consecutive days in
January 2026 with the flagged pattern, and the training year holds no such day
at that airport at all; the longest run anywhere in 2025 is two days, at Madrid.

What settles it is an instrument already in the data, and it measures the
premise pooling rests on: that a bucket's defaulted share is a property of the
defaulting mechanism rather than of the airport, so that a cell learned where
the mechanism is visible may be carried to an airport where it is not. The
fitted cell table measures that premise per airport and per bucket, on training
rows, with no fit and no booster. It says the mechanism is at one airport: of
the 432 training rows the term's own test picks out, 99.1% are at Fiumicino, and
only three of the ten airports carry any at all. In the two buckets a pooled
cell would serve, every other airport reads 0.0% on every leg, at row counts
small enough that a single defaulted row would show. Tested as arithmetic, each
airport's share against two standard errors of the pooled share at that
airport's own row count, the premise fails in both of those buckets on each of
three training halves. Where an airport passes, it passes on the width of its
interval and not on its share: in the larger bucket over the full year three
airports sit inside, each at 0.0%, because at 10 to 14 rows the interval reaches
down to zero. The July leg is the thin one, with 187 and 26 rows pooled in the
two buckets, and the premise fails there too. Amsterdam's zero is a measurement
and not a gap: its cells hold 77 and 14 rows, well above the five-row minimum,
so one defaulted row would show as 1.3% or 7.1%, and a share of zero means no
defaulted row in a full year. Pooled, those cells would have predicted 4,512 s
against a recorded mean of 824 s and 32,761 s against 1,212 s on Amsterdam's own
2025 rows. So the pooled rule is not adopted, the arm that would have tested it
was not fitted because its premise failed, and the reason is measured rather
than asserted.

One limit remains, narrower than the decision it used to carry. Whether an
uncovered airport's 2026 rows are themselves defaulted block times is a question
about the two fields the ranking set hides, and no instrument reaches them. §7.3
states the exposure that leaves.

**The plan-shifted departures, added on 2 October.** One class of departure the
cells cannot reach, because the cells key on the schedule lag and on the absence
of a record and these departures have a record: a take-off more than four hours
after both of the flight plan's off-block estimates, `EOBT_1_flt` and
`LOBT_flt`. The class was seen in the ranking set's exposed fields, after the
freeze, in the reading of the held-out-month ladder: one July 2026 departure at
Fiumicino whose plan values all lie about a day before its take-off, which the
frozen model predicted at 3,120 s. The training year holds three such
departures, one at Paris and two at Fiumicino; two recorded a taxi-out over four
hours, 21,308 s and 87,002 s, each within minutes of take-off minus the plan's
last off-block estimate, and the third 13,865 s, under the threshold. That is
the whole of the evidence, and the rule was written down and its digest recorded
before it was evaluated on any fold or built into any file, so that nothing
about it could be chosen on its effect. The rule is the term's own form: the
share is the add-one estimate over the class, `(long + 1) / (n + 2)`, which on
three rows with two long reads 0.6 rather than the two thirds that three rows
cannot carry; the lag is take-off minus `LOBT_flt`; the rest mean is the
airport's mean recorded taxi-out over departures with a record and a taxi-out of
four hours or less, 1,144.2 s at Fiumicino. It reads nothing else, never
`AOBT_3_flt`, and it moves nothing else: the booster, its rows and its seeds,
and the term's cells, are what they were. Evaluated after it was fixed, on the
twelve held-out months, it touches three validation rows and moves the every-row
figure with the term from 276.6 s to 272.4 s, which the pre-registration itself
says is a statement of the arithmetic on three rows and not independent evidence
that the rule is right (Annex I).

On the one ranking row it predicts 52,550 s, against 3,120 s before. What that
bet is worth is computed here from the report's figures, on a base score of 300
s over the 344,841 rows and under the two readings the row can have, because the
sentence a reviewer will ask for is the one that puts both outcomes in score
terms: if the row recorded the lag, 86,820 s, the frozen model's prediction
would have scored 332.1 s and the rule's 305.6; if it recorded an ordinary
taxi-out, the airport's rest mean, the frozen model's prediction would have
scored 300.0 s and the rule's 312.5. The rule trades the better of two outcomes
for the narrower pair, and the share it rests on says six in ten; which outcome
the row has, no data available to the entry can settle. The constant 300 is an
assumption about the base and not a measurement, and the two cases are the row's
two readings, not a distribution over them.

Sources: `reports/tails.md`, *What the tail is made of*, *The cells*, *Why this
form, and not the others*, *What it touches in the ranking set*, *The
plan-shifted departures* and *The pooled record-quality rule, measured*;
`reports/planshift.md`, *The class in the training year*, *The rule fitted on
the full year*, *The one ranking departure* and *Per held-out month and on the
two forward legs*; `reports/data_review.md`, *The tail owns the metric*, *Days
in the ranking set that look like disruptions* and *The same days in the
training year*.

## 5. The unimpeded term

### 5.1 From OSM to a routable surface

Three defects repaired to a fixed point: coincident-but-unshared nodes, nodes
lying on edges they are not joined to, and ways that cross with no shared
vertex. Stand attachment with the snap distance recorded per stand. Route
descriptors (distance, turn angle on a Douglas–Peucker simplification,
junction-free running, backtrack) rather than one number.

Sources: `reports/taxi_routing.md`; `reports/penalty_sensitivity.md`;
`reports/runway_crosscheck.md`; `docs/modelling_notes.md`, *Graph construction*.

### 5.2 A worked airport: where every number is known

A fictional field on which every distance, turn and repair is hand-derived,
pushed through the pipeline unchanged. All 32 matrix cells agree to 11 mm and
0.10°, integer descriptors exact; the runway-penalty crossovers fall where the
arithmetic puts them. It found three things in the pipeline, each below its
working tolerance, each measured: overshoot and undershoot select different
repairs; a merged node takes the last way's coordinates; a two-feature stand's
representative is decided by floating-point rounding. The field also carries
worked traffic in the challenge's own schema, so the whole pipeline runs end to
end in continuous integration without the challenge data.

*This is the section that answers "how do we know the geometry is right" without
asking a reader to trust matrices nobody can check by hand.*

Source: `reports/synthetic_airport.md`, *In one paragraph*, *What the field
found* and *Worked traffic*, and its figure.

### 5.3 What the geometry recovers, and what it does not

Per airport, a pair's fastest decile follows its routed distance with a residual
of 85–190 s and an intercept of 313–740 s that no distance explains. The largest
structured effect on that decile is wake category: mediums 35–281 s faster than
heavies at the same stand and runway, uniform across designators, which is a
fixed pushback-to-line-up difference and not intersection use. Intersection
departures proper leave a signature of at most 96 s per designator, constant,
and absorbed by any model knowing the designator; which stands use which entry
is unobservable without surface tracks.

Source: `reports/geometry_gaps.md`, *In one paragraph*, *The fit*, *The fastest
decile by wake category* and *The intersection signature, per designator*.

### 5.4 Where memorization runs out, and what replaces it

*Added 4 September, after §9.1's first draft asserted the opposite of what the
data says; settled 5 September, when its figures moved from a scratch
calculation into the generator. Two of the scratch figures did not survive the
move (a network-wide count had been attributed to one airport, and the transfer
figures were a stand-in's rather than the report's), which is the reason this
paper quotes generators and nothing else.*

**How much of the test set is thin.** Of the 344,841 ranking departures, 0.72%
sit on a (stand, runway) pair the training year never saw, 1.20% on a pair seen
ten times or fewer, and 9.08% on a pair seen between eleven and a hundred times:
**11.0% in total**. The share is uneven. Paris has 20.48% of its rows on thin
pairs, 19.07% of them on pairs seen eleven to a hundred times alone, and Madrid
15.57%; Frankfurt has 2.37% never-seen and 4.05% at ten or fewer, and Munich
4.57% never-seen; Heathrow has 0.18% and 0.28%, and 2.08% thin in all. July has
1.20% never-seen against January's 0.13%. Across the ten airports, 46 stands
appear in 2026 that appear in no 2025 departure, at nine of the ten, carrying
1,589 ranking rows: Zurich contributes eleven of them and Munich ten, but
Munich's ten carry 1,182 of the 1,589 rows and Frankfurt's six another 300. One
runway designator is new, on two rows. The never-seen bucket, 2,496 rows, splits
1,589 on a new stand and two on the new designator against 905 on a stand and a
runway the training year saw separately and never together; the last kind is
what the held-out-pair split measures, and the first is the case no split of the
training data can pose.

**The pair mean is worse than useless where it is thin.** On the December
forward split, predicting a pair by its training mean scores 571.9 s on pairs
seen one to ten times, worse than ignoring the pair and predicting the airport
mean, 519.0 s. On July the same comparison is 543.1 against 498.3. A mean over a
handful of draws from a distribution several hundred seconds wide is noise
wearing the costume of knowledge.

**The geometry is the better prior, and by more than the usual one.** Shrinking
the pair mean toward an estimate, the textbook remedy, depends on what it is
shrunk toward. Toward the airport mean, the standard choice, the December bucket
improves to 454.5 s. Toward the geometric prediction for that specific pair,
with the shrinkage weight estimated per airport by the same construction the
encoding of §7.4 uses, it improves to **433.3 s**. The geometric prior is worth
21.1 s more than the group prior on that bucket in December and 20.7 s in July,
and on the never-seen bucket, where the two priors stand bare with no pair mean
to shrink, 19.7 and 22.9 s. On pairs seen more than a hundred times every
version of the pair mean agrees to a second, because the count swamps the
weight: where memorization works, the prior does not matter, and where it runs
out, which prior it runs out to is the whole question. The geometry explains 4%
to 61% of the between-pair variance the airport mean leaves, per airport, which
is why a pair's handful of draws is weighed less against it.

**The relation transfers between airports.** Fitted on nine airports and applied
to the tenth, with no stand, pair or airport-level quantity of the tenth
available, the transfer model (geometry plus the clock, the aircraft, the plan
lags, the traffic in the window and the weather, every identifier removed) beats
the only baseline available without it, the grand mean of the other nine, at
**ten airports out of ten**, by a median of 23.6% and by as much as 43.4% at
Heathrow. The geometric term *alone*, two fitted numbers and five shared
modulators, beats that mean at eight of ten, and loses at Munich and Heathrow,
by 4.1% and 1.5%. The speed the ten leave-one-out fits recover runs from 11.8 to
13.2 m/s, a 12% spread about 12.42 m/s, with the intercept at 434 to 500 s and
the offset at 251 to 276 s. That stability is the finding: the relation is a
transferable physical constant rather than a per-airport fudge, and it is the
one thing in this entry that an identifier model cannot reproduce even in
principle, because at an unseen airport it has nothing to encode.

Sources: `reports/geometry_gaps.md`, *Where memorization runs out*, which
carries the per-airport and per-month tables, the new-stand table with the shape
of each identifier, both forward splits by bucket, and the shrinkage weights,
none of them restated here; `reports/transfer.md`, *The speed, ten times over*
and *RMSE, per held-out airport*, for the per-airport transfer table, likewise.

## 6. The excess term

### 6.1 What the ranking set allows, and the registry that enforces it

The rule is phase-specific: a feature may read what the ranking set exposes *for
the rows it reads from*. Every feature column is registered against its raw
sources and the phase each is read from; the test suite holds each phase to its
own list, and a poison test sets the hidden columns to nonsense and asserts the
frame does not move. Arrivals' on-block and taxi-in times are exposed at 0.0%
null and are read; no departure's block or taxi time is, ever.

Sources: `reports/congestion.md`, *The leakage audit*; `reports/data_review.md`,
*Null rates*.

### 6.2 Congestion, corridor, and weather

Throughput, arrivals, demand, runway configuration, shared-corridor load, and
three weather regime flags per airport-hour. The corridor feature (other
movements weighted by how much of the subject's own taxiway route they share,
each edge weighted by the inverse of the routes using it) is the one feature
here no other entry has, and it is reported as it fell.

Source: `reports/congestion.md`.

### 6.3 The flight plan's own delay signal

Take-off minus the filed off-block estimate is the strongest single feature in
the model. As a ladder step it is worth 79.3 s in December (320.2 → 240.9 s
ex-flagged) and 57.4 in July (407.3 → 349.9); measured at the `+ weather` rung,
by removing it and keeping everything else, 59.3 and 51.7. Either way it is
worth more than every congestion and weather block combined, which net of it are
worth 7.7 s and 5.8.

What it is, said plainly, because a reviewer will ask: take-off minus a planning
value fixed before departure equals the taxi-out plus the pushback delay against
that plan. The model is therefore learning the distribution of pushback delay
around a value it is given, which is the same construction the record-quality
term uses with the schedule, and admissible for the same reason: `EOBT_1_flt`
and `LOBT_flt` are planning values the ranking set exposes, not post-hoc records
of the blanked field. `AOBT_3_flt`, which is such a record, is excluded.

Its held-out-pair contrast is −1.4 s [−5.8, +3.2]: it helps equally whether the
pair is new or familiar, which is what a feature about the day rather than the
place should do. The ladder is reported both with it and without it, and the
regime exposure is bounded in §7.3.

Sources: `reports/baseline.md`, *The ablation ladder* and *Held-out pairs*;
`reports/congestion.md`, *Regime exposure*.

### 6.4 The queue layer that was not built

An oracle diagnostic bounds it: regress a departure's residual on the mean
residual of its same-runway neighbors, quantities no admissible feature may
read, and which a queue model is precisely an attempt to approximate. On the
submitted rung, the ceiling is **8.2 s in December and 8.3 in July**; the lag-1
autocorrelation of consecutive same-runway residuals under two minutes is 0.19
and 0.09.

The plan-lag block is what closed it. At the top of the chain fitted without
that block the same ceiling is 31.5 s and 27.5, and the autocorrelation 0.31 and
0.17, so most of what a queue model would have found was the day's delay against
plan, and the cheapest possible feature had already taken it.

A saturation proxy (the share of sub-90-second gaps, the minimum gap, the span
of the last three gaps on the subject's runway) was then built against that
ceiling. Measured on 3 and 4 September, under the parameters then in force, it
bought nothing: after the parameter sweep of that week it cost half a second in
December and eight tenths in July against the rung it was added to, and the rung
carrying it had a ceiling of 10.0 and 8.5, within a second of the model's own.
Gate 3 was stopped on that measurement rather than on the schedule, the stop is
recorded in the plan, and the submission is not fitted from the block. It stays
on the ladder, and under the parameters that ship it reads differently, which is
reported rather than smoothed over. Against `+ weather`, the rung it was added
to, it now scores 232.7 s against 233.2 in December and 343.4 against 344.1 in
July, a change larger than the seed floor on both splits; its held-out-pair
contrast is −0.6 s [−1.2, −0.1]; and its ceiling, 9.2 s and 8.3, sits beside the
submitted rung's 8.2 and 8.3. It has never been measured on top of the rung the
submission is now fitted from, so what it would add there is not known.

*"The physical layer did not move RMSE once congestion was controlled for" would
have been a publishable finding; so is this. The thirty hours the plan reserved
for a queueing simulation were released by a diagnostic that took minutes and
needed no model.*

Sources: `reports/congestion.md`, *The ceiling*; `reports/baseline.md`, *The
ablation ladder*, *Held-out pairs* and *The noise floor*; `docs/plan.md`, *When
the data lands*, a dated record, for the measurements of 3 and 4 September.

### 6.5 The stand's recent history, and the bound inside it

The largest of the four blocks added last reads the arrivals at the subject's
own stand: the gap from the last arrival's on-block to the subject's take-off,
that arrival's lateness and taxi-in, whether it flew the subject's aircraft type
and operator, and the gap to the previous departure off the same stand. Every
field is one the ranking set exposes for arrivals in full (§6.1). On the ladder
it is worth 8.6 s in December (228.3 → 219.7 s ex-flagged) and 9.1 in July
(342.2 → 333.0), the largest single step above the plan lags on either forward
split; on the three, 16.0 s (256.6 → 240.6). Its held-out-pair contrast is +1.8
s [+0.8, +2.9]: it buys slightly more where the pair is new, which is what a
stand-level quantity should do where the encoding has nothing to say about the
stand.

**The bound, and where it holds.** A stand holds one aircraft, so an arrival
that on-blocks at the subject's stand before the subject takes off means the
subject had already pushed back, and the gap is a lower bound on its taxi-out.
That is why the block was built, and it is true only in a band. Measured on the
December rows of the rung that never read the gap: under twenty minutes the
bound holds in 97–100% of rows and the model already respects it; at 20–25
minutes it holds in 94% of rows while the model violates it in 17%,
under-predicting those rows by 171 s on average; at 30–35 minutes, 71% and 56%;
by 40–45 it holds in 11% of rows, and past forty-five minutes essentially never,
because a stand identifier can denote a position that holds two aircraft and
because aircraft are towed. Applied as a rule it loses at every cap tried (+0.1
s at twenty minutes, +1.2 at thirty, +20.6 at forty-five in December), since a
floor can only act where the model is below the gap, and where the bound is true
the model already obeys it. It enters as a feature, and the booster finds the
band.

**Where the gain lands, and how much of it is the bound.** The block's reduction
in squared error is a sum over rows, so it can be attributed to disjoint bands
of the gap and the shares add. The 20–45 minute band is 1.0% of December's rows
and 20.9% of the cut; on the July three, 1.9% and 34.9%. Inside it the previous
arrival's type separates the two cases the band contains. Where the type
differs, a different aircraft took the stand after the subject pushed: 900
December rows carry 17.3% of the whole cut, and their mean under-prediction
falls from +229 s to +5. Where it matches, the subject's own inbound after a
fast turnaround, 686 rows carry 3.6% and go from +54 s to −23. That is the bound
doing what the mechanism says, on the rows it speaks about, and it is a minority
of the block on every leg. Past an hour the gap bounds nothing, and the block
still earns there: 46.4% of December's cut sits on gaps over an hour where the
previous arrival was the subject's own type, the subject's own inbound, against
10.1% on the July three, where the short gaps carry the block instead. What it
reads at those gaps is the stand's recent history: the inbound's lateness and
taxi-in, the scheduled ground time, whether the aircraft on the stand was the
subject's own. That is a turnaround-pattern signal, admissible on exactly the
same footing as the bound and a different thing from it. That second share moves
by a factor of several between the legs, so a sentence about what this block
*is* has to name the leg it is true of.

**What survives an operational rule.** The block reads an arrival that
on-blocked *after* the subject's own off-block. Under the challenge's post-hoc
framing (the take-off time is given, and every take-off-anchored window in the
model has the same standing), it is admissible; under an operational one it is
not, and this is one of the two places in the entry where exploiting the ranking
process could be alleged (`AOBT_3_flt`, excluded, is the other). How much of the
block would survive an operational rule is measured rather than argued. A
diagnostic rung off the chain and out of the submission re-anchors the block on
the schedule (the stand's last arrival at or before the subject's *scheduled*
time, every column read from that choice, so that nothing in it needs the
take-off), and it keeps 42% of the block's gain in December, 27% in July over
all ten airports, and **4% on the three**, where the schedule-anchored gain,
+0.58 s, is inside that leg's seed floor of 1.02 s and cannot be told from zero.
The schedule anchor is stricter than operational, since at pushback the arrivals
up to the actual off-block are known and not only those before the schedule, so
these are lower bounds; but on the three, the block is a post-hoc instrument
almost entirely, and this section says so rather than burying it.

**The headroom overshoot.** Each of the four blocks' refit gain is printed
beside the residual-fit headroom estimated for it before it was built
(`docs/review_headroom.md`), and a block that takes more than five seconds
beyond its estimate trips a test. The test was built as a leakage audit, and it
is not one: §7 of `docs/review_headroom.md`, written on 8 September, before the
test ever fired, established that a booster fitted on a frozen model's stored
residuals bounds a refit from below, so the gap between the two is the value of
the interaction rather than evidence of anything wrong. The report now names the
test for what it measures, and the name leakage audit belongs to the source
registry printed as an assertion in `reports/congestion.md`, which is a leakage
test and passes (§6.1). Under the parameters that ship, on the July three, this
block takes +16.0 s against an estimated +10.3, an overshoot of 5.681 s at 1.55
times the estimate, so the test fires, by 0.681 s, inside that leg's seed floor
of 1.02 s: the verdict is within what changing the seed alone produces. The
block stays in the chain, whose rule does not read this test, on the reading
that an overshoot inside the floor cannot tell leakage from noise. That reading
was adopted after the number was seen, and this paper says so; what carries the
weight is the date on §7, not the reinterpretation. The other reading is
leakage, and it is the one the test exists for: the registry and the poison
tests of §6.1 bound what the block can read, and the concentration above bounds
where what it read landed. The test has also read differently from one
regeneration of the ladder to the next, each time moved by something other than
the block: 1.54 times the estimate on the ladder of 8 September, firing by 0.56
s; 1.48 on 10 September, missing by 0.029 s; 1.52 on 11 September, firing by
0.37 s; 1.42 on 19 September, missing by 0.655 s; and 1.55 under the parameters
that ship. The estimate itself is now regenerable: `reports/headroom.md`
recomputes it from the current residuals at a stated capacity, reads +10.00 s,
and draws no conclusion about the test. What the block rests on is the source
registry and the poison tests of §6.1; the parity of the gap on both sides of
the data, where no airport's share of stands re-taken within twenty minutes
moves by more than two points between 2025 and 2026; the concentration above,
which is consistent with the mechanism and is not proof against another route;
and the schedule-anchored measurement, which puts a number on the part of the
block that is the post-hoc reading.

Three smaller blocks stand on it in the chain (the operator read as the callsign
prefix, the destination's bearing and distance with the holiday calendar, and
the plan's two differences) and are read in `reports/baseline.md`, *The Gate 2c
blocks*, each net of the blocks below it.

Sources: `reports/baseline.md`, *The ablation ladder*, *What each block buys
where the pair is new*, *The Gate 2c blocks*, *The three airports July 2026
contains*, *Where the stand-occupancy gain lands* and *The same block without
the subject's take-off*, the last of which carries the headroom overshoot;
`reports/occupancy.md`, *The gap, on both sides*, *Where the bound holds*, *The
hard floor, and what it costs* and *Where the gain lands*;
`reports/headroom.md`, *The ratios*; `docs/review_headroom.md`, §7, a dated
record of 8 September; `docs/modelling_notes.md`, *Leakage watch*.

## 7. Validation

### 7.1 The design, fixed before anything was fitted

Two forward splits (December 2025 at all ten airports, standing in for January
2026, and July 2025 at all ten, standing in for July 2026, with the three,
Frankfurt, Heathrow and Schiphol, read on their own as well, because July 2026
was expected to hold those three alone when the design was fixed, §2.1) plus a
held-out-pair GroupKFold and pair-frequency strata. Never random K-fold on rows:
congestion is autocorrelated over hours. The three-airport reading is kept as it
was fixed; where a rule below reads it, the paper says so, and §8 says which
July figure it quotes.

The held-out unit is the **pair**, not the stand: every stand and runway is
still seen in other pairs, so this bounds what the geometry buys on a new
combination and does not isolate what it would buy on a new stand, which no
split of this data can measure. The claim is a difference in differences, what a
block buys on held-out pairs minus what it buys on seen ones, so composition
cancels. Intervals are a cluster bootstrap over pairs, 1,000 draws, resampling
stored residuals; no model is refitted. After the freeze the ladder was also
read with every month of 2025 held out in turn, a diagnostic that is not forward
in time and decides nothing (§3.5); §8 and the annexes carry it, and the forward
splits remain the proxies for the two test months.

Source: `reports/baseline.md`, *Held-out pairs*.

### 7.1.1 What the forward legs cannot see, as measured on 20 September

Three limitations of the design above, measured rather than asserted, on 20
September under the parameters then in force. None was repeated under the
parameters that ship, so every figure in this subsection is that day's, read
from the dated record the experiments wrote.

*The season columns.* On 20 September, under the parameters then in force, the
submission rung was refitted on December and on July without `tko_month`,
`tko_doy_sin` and `tko_doy_cos`, first seed, everything else unchanged, under a
rule fixed before the sign was known: no adoption follows in either direction.
Removing them improved December by 0.40 s and the ranking mix by 0.30, and cost
July 1.15 s on the three and 0.77 s pooled over all ten. The sign is not the
same on the two legs. Both forward legs validate on a month and a day-of-year
range the training side never held, December after January to November and July
after January to June, and the ranking months do not: January 2026 and July 2026
each have their twin in the training year. So the legs extrapolate on these
columns where the final fit interpolates, and the measurement answers a question
the ranking months do not ask. A held-out-month split, training around a month
and validating inside it, would answer the question they do ask, and it breaks
the forward discipline every leg here keeps. It was named here, when this was
written, as what would have measured this, and it was later run as a diagnostic
after the freeze (§3.5, §8.2) without repeating the season-column experiment on
it. The columns stay in the model.

*Two instruments, and the first time they disagreed.* This entry sizes a
difference two ways. The seed floor is root two times the standard deviation of
five single-seed fits of one rung on one leg, and asks how far a figure moves
when nothing but the random seed does. The interval is a day-clustered
percentile interval over the evaluation set, days as the clusters and the same
draw for both arms, and asks how far the figure would move on another sample of
days. They measure different things, and they can disagree about the same
contrast. They did on the season columns above: three of the four steps were
outside the floor of the leg they were taken on, while every interval that
printed spanned zero, December all ten at −0.40 s against a floor of 0.34 with
an interval of [−1.6, +0.9], and July all ten at +0.77 against 0.23 with [−0.2,
+2.3]. Nothing chose between the two instruments there, and nothing needed to,
because no adoption followed. They disagreed again on 23 September, and §7.1.2
gives that occasion. **Where a figure in this paper is called real or called
noise, the sentence says which instrument said so.**

*Recency, and what the gap leg can and cannot say.* On every 2025 leg the most
recent training month is also the seasonally nearest to the month being scored:
November for December, June for July, September for October. A gain on those
legs cannot separate recency from season. For the January 2026 ranking rows the
two agree; for the July 2026 rows they do not, since the most recent training
month, December 2025, is seasonally opposite and the seasonal twin, July 2025,
is five months older. The gap leg was built to break the confound: train January
to June 2025 and score December, which is the July 2026 geometry as nearly as
one training year permits, without the twin month. Two recency schedules were
fixed in advance, weights halving every 3 and every 12 months over the booster's
fitting rows, with a screen fixed at printed precision. On 20 September, under
the parameters then in force, the mild schedule did nothing measurable on the
gap leg in either direction: +0.10 s against that leg's own floor of 0.40 s,
with an interval of [−0.4, +0.6] spanning zero. It gained on the July forward
leg past that leg's floor and with an interval clear of zero, −0.84 s against
0.23 with [−1.5, −0.4], and −1.61 s on the three against 1.02. It stopped
because the screen reads printed precision and, at that precision, 224.8 is
worse than 224.7. **The reason no recency term ships is the screen, not a
measured loss on the gap leg.** The strong schedule is a separate case and
should not be read with it: it was worse on October by +0.37 s against October's
own floor of 0.19, a regression the run could measure. The confound itself
stands as a limitation that no leg inside one training year can resolve.

Source: `reports/preregistered-2026-09-20.md`, *What Experiment 2 measured* and
*What Experiment 3 measured*, a dated record of 20 September.

### 7.1.2 The rules under the parameters that ship, and how those parameters were chosen

*Printed precision, and the seed rule.* Printed precision in this entry is a
tenth of a second, and every seed floor is larger than that: under the
parameters that ship, 0.23 s on December, 0.54 on July over all ten airports,
1.02 on the three and 0.21 on October. So a screen written at printed precision
can fire, in either direction, on a difference the same run calls seed noise,
and on 20 September one did (§7.1.1). That is a property of the rules as they
were fixed; it was not changed after the fact, and the screens were applied as
written. The seed rule of 19 September reads floors and not printed precision.
Under the parameters that ship, the five-seed average gains 1.11 s over the
ladder's seed in December against a floor of 0.23, and 1.46 s in July against
0.54, beats every single seed on both legs, and clears the rule with a margin.
Under the sweep's cell of 21 September the margin was thin: July gained 0.63 s
against a floor of 0.61, and the report itself says that the standard deviation
behind a floor measured from five seeds is uncertain by about 35%. The cell that
ships replaced that margin rather than resting on it.

*The two instruments, a second time.* On 23 September the floor and the interval
disagreed about the booster cell that ships: against the cell it replaced it
reads 0.4654 s better, 2.56 floors, while its day-clustered interval, [−0.9492,
+0.0009], spans zero by nine ten-thousandths of a second. The rule that chose
the cell reads the floor and nothing else, so the interval decided nothing, and
both are printed. It is the second time the project has met that, after the
season columns of §7.1.1.

*How the booster's parameters were chosen, including the part that was not
planned.* The parameters were chosen on October 2025, a month no rung of the
ladder scores, by two sweeps and one extension of the second, all fitting
January to September 2025. The first sweep, on 3 September, ran on the rung that
was then the top of the ladder and under fit rules that have since changed, and
its figures are not comparable with the second's (§9.3). The second,
pre-registered on 20 September and permitted once, read twenty-seven cells on
the submission rung under the fit rules that stand, one fit each, and chose
3,000 trees, a learning rate of 0.015, 255 leaves and a minimum child sample of
20, at 214.90 s against the old cell's 217.28 s, a move of 2.38 s against a seed
floor of 0.19 s. That cell sat on an edge of the grid on all four axes. The
pre-registration said the grid would not be extended again, and the edge was
recorded as it required. The author reopened that rule on 22 September, after
the edge was recorded and knowing which way October was still falling; that is a
decision taken with the result in view, and it is said here in those words
rather than presented as the plan. The extension's criteria were fixed in
writing before any figure past the edge existed. Three of the four axes were
probed, one step out each, independently, and they were chosen from the sweep's
own gradients, which is a design choice and is disclosed as one; the learning
rate was the axis left out. The trees axis won: 6,000 trees read 214.3041 s
against 214.9045, 0.6004 s better, 3.31 floors, on one leaf setting, and there
the day-clustered interval agreed with the floor. The other two lost, 511 leaves
by 0.2274 s and a minimum child sample of 10 by 0.2381 s. The trees axis was on
an edge again at 6,000, with October still falling at 4,000, 5,000 and 6,000,
and the extension's criteria closed it there.

*The cell that ships was taken on cost, from two the instrument cannot
separate.* On October the better reading is 6,000 trees at a learning rate of
0.015 and 255 leaves, 214.3041 s, against the shipped cell's 214.4391, and it
was not taken. The two differ by 0.1350 s, 0.74 of the floor, and by less than
half of the 0.2872 s the better cell moves between two draws of its own seed, so
this instrument cannot separate them. That is the situation the
one-standard-error rule of Breiman, Friedman, Olshen and Stone (1984) exists
for, and among candidates the evidence cannot separate, it takes the cheaper.
The cell that ships, 6,000 trees at 0.03 and 127 leaves with a minimum child
sample of 20, builds 762,000 terminal nodes against the alternative's 1,530,000
and fits October in 237.5 s against 321.8 s. The alternative is named here as
the equal-performing one, so the choice is visible rather than implied, and the
shipped cell is the worse of the two readings: it is presented as the cheaper of
two cells the evidence cannot separate, not as the better score.

*The winner's curse, measured.* A search that reads twenty-seven cells at one
seed each and keeps the smallest keeps, in part, the luckiest. At a floor of
0.19 s, which is root two times the standard deviation one seed carries, the
smallest of twenty-seven such readings is expected to sit about 0.27 s below its
cell's own level from the selection alone. That figure is an expectation,
computed under independent, normally distributed seed noise with every cell at
one level, which is the largest the shortfall can be, and not a measurement. The
measurement came before the extension was probed: the four cells within one
floor of the sweep's best were refitted at a second seed, and the sweep's chosen
cell went from first of the four to last, with four of the six pairs crossing.
The trees result is read differently, as a paired contrast within one seed,
which the seed barely moves: going from 3,000 to 6,000 trees read −0.6004 s and
−0.6019 s at the two seeds on 255 leaves, and −0.5591 s on 127, while the seed
moved the level itself by 0.2872 s. So the trees effect survives the seed where
the ranking of the cells at 3,000 trees did not.

Sources: `reports/baseline.md`, *Seed averaging* and *The noise floor*;
`src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`, which is the
repository's own account of the sweeps, the extension and the cost decision;
`src/taxiout/provenance.py`, the basis of `FLOOR_JULY_ALL`, for the sweep cell's
July reading. The seed-stability check, the interval on the trees step, and the
order in which the extension's criteria and its figures were fixed are recorded
in the header and the message of `scripts/commit_extension_probe.sh`.

### 7.2 The ladder, and the claim

Every rung on both splits, ex-flagged and with the record-quality term, plus the
seen-versus-held-out contrast per rung with its interval. The submission is
fitted from `+ plan differences`, the top of the chain the four blocks of §6.5
climb, at 214.7 s (December) and 322.4 (July) ex-flagged on the ladder's own
single seed; §8 gives the figures of the model that ships. Each rung is read
against the rung below it, and the three encoding arms of §7.4 against `+
weather`, at 233.2 and 344.1.

The claim contrast, identifiers + time → identifiers + time + geometry, is
**+18.5 s [+16.7, +20.5]**: the geometry block buys 0.4 s where the model has
seen the pair and 19.0 s where it has not. The interval excludes zero.

Two neighboring rows are worth reading beside it, because they are what makes
the claim mean anything. The step from the geometric unimpeded term, with the
wake class, to the identifier model is −41.3 s [−47.1, −35.7]: identifiers beat
geometry outright where the pair is known, which is the null this entry had to
face. And the step into the geometry block from identifiers is where that
reverses. Together they say the two encodings are not competing descriptions of
the same thing (one memorizes a pair, the other computes it), and the split that
separates them is the only one that can.

Three arms above `+ weather` carry a shrunk identifier encoding; they are the
subject of §7.4, one of them is carried into the submitted model, and the claim
contrast is measured near the foot of the ladder where neither rung carries it.

**What each source of delay is worth.** The ladder read as a table: each row is
one block, added to the model built from the rows above it. The two RMSE columns
are the ladder's ex-flagged figure once the block is in, on the single seed the
ladder fits; the change is that figure minus the one in the row above (negative:
the block made the model better), so a block's change is net of every block
above it in the table and not what it would be worth alone. The held-out-pair
column is the claim instrument of §7.1: how much more the block buys on pairs
the model has never seen than on pairs it has, with its 95% interval. A change
smaller than the seed floor, 0.23 s in December and 0.54 in July, cannot be told
from what changing the random seed alone does. The steps marked by the report as
inside that floor are listed there by name.

| block added | what it reads | December (s) | change (s) | July (s) | change (s) | held-out-pair difference [95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| identifiers + time | airport, stand, runway, pair; clock and calendar | 324.0 | | 410.9 | | |
| geometry | the routed stand-to-runway distance and route shape | 321.7 | −2.27 | 411.1 | +0.14 | +18.5 [+16.7, +20.5] |
| aircraft | wake class and type | 320.2 | −1.5 | 407.3 | −3.8 | +4.2 [+3.2, +5.3] |
| plan lags | take-off against the plan's off-block estimates and the schedule | 240.9 | −79.3 | 349.9 | −57.4 | −1.4 [−5.8, +3.2] |
| congestion and weather, five blocks | throughput, arrivals, demand, runway configuration, shared corridor, weather flags | 233.2 | −7.7 | 344.1 | −5.8 | |
| shrunk encoding, replacing the stand | the pair's training mean shrunk toward the geometry | 228.3 | −4.9 | 342.2 | −1.9 | −4.4 [−6.1, −2.7] |
| stand occupancy | the last arrival at the subject's stand | 219.7 | −8.60 | 333.0 | −9.12 | +1.8 [+0.8, +2.9] |
| callsign prefix | the operator | 217.6 | −2.11 | 328.4 | −4.64 | +0.7 [−0.8, +2.2] |
| destination and holidays | where the flight goes; the public-holiday calendar | 215.8 | −1.79 | 326.9 | −1.49 | −0.4 [−1.1, +0.4] |
| plan differences (ships) | the plan's own off-block values against each other | 214.7 | −1.14 | 322.4 | −4.55 | −1.0 [−2.5, +0.4] |

Three readings of it. The plan lags are worth more than every other block
together, and they are not about the place (their held-out contrast is inside
its interval of zero). The geometry is worth almost nothing on a forward split
and is the only block whose worth is concentrated on pairs the model has never
seen. The encoding is the reverse: it gains on the forward splits and loses on
held-out pairs, which is what smoothed memorization does (§7.4). The congestion
blocks are read one by one in `reports/baseline.md`; each is under two seconds
net of the plan lags. §8.4 reads the same table over the twelve months of the
training year held out in turn, each block's change pooled over the year and
counted against the floor measured in each month.

Sources: `reports/baseline.md`, *The ablation ladder*, *Held-out pairs*, *The
noise floor*, *The shrunk identifier encoding* and *The Gate 2c blocks*.

### 7.3 What could go wrong on the ranking months

Three exposures, each stated as a bound rather than an estimate.

**The plan-lag regime shift.** The ranking months' departures run later against
their plans than their 2025 twins in January, and earlier in July: at Istanbul
the mean of take-off minus the filed estimate moves +288 s in January, at
Frankfurt +224, at Munich +193; in July it moves the other way at every airport,
from −64 s at Madrid to −480 at Paris. The exposed fields cannot decompose that
into taxi and pushback delay, but they can bound it, and a control that reads no
plan at all says which it mostly is. Where the January lag moves most, arrivals'
taxi-in barely moves (−20, +9, +16 s at Istanbul, Frankfurt and Munich) while
arrivals' own lateness moves with it (+857, +453, +366): the day ran late
against its plans, the surface did not run slower.

Taking the whole shift as pushback delay anyway (the worst case, which the
control says it is not) and multiplying by each airport's training slope of
taxi-out on that lag (0.24 to 0.51 s per second) gives a per-airport bias of at
most 93 s in January and 132 s in July, costing **+20.3 s at Istanbul, +12.8 at
Frankfurt and +5.6 pooled in January, and +31.0 at Paris, +7.1 at Istanbul and
+5.1 pooled in July.** Biases of opposite sign do not cancel in the pooled
figure, because each enters its own airport's squared error. Paris is the
exposure the re-issued July month adds: it was not among the three, and its July
departures take off eight minutes earlier against their filed estimates than in
July 2025, the largest shift at any airport in either month.

Per airport, in seconds: the shift in the mean of take-off minus the filed
off-block estimate between each ranking month and the same month of 2025, and
what that shift would cost the model's RMSE at that airport if all of it were
pushback delay (the worst case):

| airport | January shift | January worst-case cost | July shift | July worst-case cost |
| --- | ---: | ---: | ---: | ---: |
| Frankfurt | +224 | +12.8 | −135 | +4.7 |
| Munich | +193 | +8.0 | −104 | +3.0 |
| Heathrow | +47 | +0.6 | −127 | +3.4 |
| Schiphol | +148 | +3.7 | −74 | +0.9 |
| Barcelona | +23 | +0.1 | −93 | +1.2 |
| Madrid | +92 | +1.7 | −64 | +0.7 |
| Paris | −80 | +1.0 | −480 | +31.0 |
| Fiumicino | −35 | +0.6 | −83 | +1.1 |
| Zurich | +178 | +5.0 | −69 | +0.7 |
| Istanbul | +288 | +20.3 | −192 | +7.1 |
| pooled | | +5.6 | | +5.1 |

**The winter regime.** January 2026 froze for 48% of Frankfurt's hours against
30% in 2025, 65% of Zurich's against 41%; Istanbul went from no freezing hour to
15%. This is where a model fitted on 2025 extrapolates, and the weather flags
are what it extrapolates along, which is the argument for regime flags rather
than continuous weather drivers.

What a winter month outside the model's experience costs is on the record once
in the training year. Istanbul's February 2025 froze for 22.6% of its hours
against a 2015 to 2024 range of 0.0% to 16.7%, and had precipitation in 39.1%
against 9.1% to 23.5%; read with that month held out, the shipped model scores
520.8 s there with a bias of −92 s and no row over four hours, against 173.4 to
237.1 s at six ordinary airports over the year (§8.3). The seven February days
above the month's figure had 10 to 20 freezing hours and plan lags of +1,454 to
+2,486 s. January 2026 at the same airport froze for 14.8% of its hours against
that February's 22.6%, and 3 of its 31 days sit inside those seven days'
freezing range, 10 inside their plan-lag range, 3 inside both; 29 sit inside the
range of the ordinary February days on freezing hours. That is descriptive: the
ranges come from 28 days of one airport-month, and no figure here says what
January 2026 will score (Annex G).

**The record-quality term's coverage.** Fitted on one airport's 2025 feed. An
airport that begins defaulting block times in 2026 is not covered, and no rule
fitted on 2025 could cover it. The alternative, a cell pooled across airports,
is not left as a judgment: the per-airport defaulted share by cell measures the
premise pooling rests on and refutes it in every bucket a pooled cell would
serve, on each of three training halves, which §4 gives with the arithmetic of
the earlier comparison. What remains is the exposure itself, which is bounded
and is stated rather than priced: 279 ranking rows sit where a pooled rule would
have covered them, 138 of them at Amsterdam in January 2026, and each occurrence
at an airport whose feed changes costs one day-scale row.

**And one extrapolation the final fit does not make.** The July leg's Heathrow
figures, the afternoon concentration of §9.5 among them, are read from a model
trained on January to June 2025, which holds no July. The submitted model is
fitted on the whole training year, which holds July 2025's afternoons, with the
hour and the runway in its frame. So the leg measures an extrapolation the
submitted model does not make in the same form, and its Heathrow figures are a
diagnostic rather than a bound on the ranking months.

Sources: `reports/congestion.md`, *Regime exposure* and *The weather regime*;
`reports/tails.md`, *The pooled record-quality rule, measured* and *What the
term is, stated plainly*; `reports/residual.md`, *EGLL's July*;
`reports/monthly_ladder.md`, *Annex E: the historic weather context*;
`reports/monthly_seeds.md`, *Per airport and month: the five-seed average*;
`reports/istanbul.md`, *LTFM, February 2025, day by day*, *LTFM, January 2026
(ranking month), day by day* and *January 2026 against February 2025's days*.

### 7.4 A block aimed at one place that paid in another: the shrunk encoding

*The measurement, the decision and the model agree: the arm this section chooses
is carried into the rung the submission is fitted from, through an
inference-time encoder that computes the encoding once over the training year
and is tested against the same leakage modes as the fold-wise one.*

**What was built, and against what.** §5.4 says where the identifier model has
nothing to memorize and that the geometry is the better prior where it runs out.
The obvious way to spend that inside the booster is an empirical-Bayes encoding
of the (airport, stand, runway) pair: its out-of-fold training mean shrunk
toward the geometric prediction rather than toward a group mean, `(n × ȳ + k ×
ĝ) / (n + k)`, with `k` estimated per airport as the ratio of within-pair to
between-pair variance, the sampling noise of each pair mean subtracted, and the
same construction at stand level as a second feature. It was pre-registered
against a specific bucket: pairs seen one to ten times, where the identifier
exists and its mean is a handful of draws. The ceiling it was built against was
computed before the arms were run, by scaling each bucket's airport-mean error
by the ratio the control achieves on the well-observed bucket and calling the
difference the attainable headroom. As the report computes it under the
parameters that ship, it comes to +0.32 s pooled, all of it in the thin buckets,
none by construction in the bucket that carries 91% of the rows.

**What was guarded.** Four leakage modes, each with a test that plants the leak
and asserts nothing moves: the encoding is recomputed inside every fold, so a
held-out pair sees `n = 0` and encodes to the bare prior; `ȳ` is out of fold
within training too, so a row's own target never enters its own feature (the
single-`k` version failed that test by twelve seconds, which is how the
out-of-fold `k` came to exist); `ȳ` reads unflagged rows only, so one defaulted
block time cannot become a three-row pair's encoding; and the one overlap that
cannot be guarded, the prior being fitted on hundreds of thousands of rows of
which a thin pair contributes a handful, is stated rather than left to be found.
`k` was checked for route-length dependence by a rule fixed before the run.
Until 19 September, with the encoding fitted on the unflagged training rows, the
check read immaterial, at a ratio of 1.29 against a threshold of 1.5. With it
fitted since then on the rows every fit reads, the check is material on the
ladder's December leg, at 1.56, so there the pair-level `k` moves with route
length and the stand-level `k` does not; the encoding the submission fits runs
the same check over the whole training year and records its reading in the
manifest written beside the submitted file. A `k/2`–`2k` sensitivity moves the
arms by at most half a second; `k` is not tuned.

**A gate that was voided rather than failed.** The rule fixed in advance
required a positive difference in differences on the held-out-pair folds. That
condition cannot credit the component the block exists for: a held-out pair
encodes to the prior identically, so the pair encoding carries none of the
pair's own mean exactly where the split measures. The scope of that is worth
stating precisely, because overstating it would be self-serving. The gate is
applied to a whole *arm*, and it validly measures the rest of one, the
stand-level encoding and the removal of the `stand` categorical, which it reads
as decisively negative. It is not an instrument incapable of returning a
positive number; it is an instrument blind to the pair encoding. A gate that
cannot see the thing under test is not a test of it, so it was voided as a test
of the pair encoding, the arms took no credit from it, and they were decided on
the two instruments that can see a pair encoding, the forward splits and the
frequency strata, with the ranking set's own bucket mix as the pooled figure.
That replacement rule was written after the numbers were seen, which is a real
weakening and is said in the report; the condition that does the rejecting (no
worse than the control on either forward split) is the original rule's own
second half, unchanged, and it is what rejects the first arm at +3.2 s on July.

**What chooses between the two arms that pass is a mechanism, not their
scores.** §5.3 measures what a designator carries that the routed distance
cannot compute: an intersection departure enters the runway at a taxiway
junction rather than at the threshold, and the signature of that is up to 96 s
per designator and constant within one: absorbed by any model that knows the
designator, and not recoverable by an encoding shrunk toward a matrix that
routes to the threshold. So the runway categorical stays and only the stand is
replaced, for a reason measured before these arms were run and independent of
their numbers. The never-seen bucket corroborates and does not decide: it is a
few hundred rows, and a decision resting on it would be a post-hoc rule resting
on a thin stratum.

**What it bought, and where.** `+ shrunk encoding, replacing the stand` clears
the rule: 228.3 s against 233.2 in December, 342.2 against 344.1 in July, 229.6
against 234.5 at the ranking mix, and +8.1 s worse on the never-seen bucket
against +14.8 for the other passing arm. The pooled figure puts the other
passing arm ahead, 229.4 s against 229.6, so the mechanism is bought for 0.17 s
pooled, and the report prices it rather than passing over it. Decomposed by
bucket in the sum-of-squares domain, **84% of the ranking-weighted gain comes
from pairs seen more than a hundred times**, the bucket the ceiling had assigned
no headroom at all, against 18% from the `1-10` and `11-100` buckets together
and 1.2% given back on pairs never seen. On the strata it was aimed at it takes
17.2 s of the 30.8 s of headroom the `1-10` bucket has, which at that bucket's
0.64% of the rows is worth 0.16 s pooled. The honest name for the block is
smoothed memorization that buys real score on well-observed pairs. It is
reported under that name, and §9.4 says what the ceiling that missed it was
actually measuring.

Two consequences follow. The prior is `ĝ`, so the encoding is geometry-bearing,
and the ladder's bookkeeping now counts it as such; an ablation that removed the
geometry block and left the encoding in would not have removed the geometry. And
the claim contrast of §7.2 is measured at `identifiers + time → identifiers +
time + geometry`, near the foot of the ladder, where neither rung carries the
encoding, so the claim figure of §7.2 is untouched by this decision, by
construction and not by luck.

Sources: `reports/baseline.md`, *The shrunk identifier encoding* and its
subsections; `reports/migration-2026-09-10.md`, *The encoding's rows,
2026-09-19*, a dated record, for the check as it read before that day;
`tests/test_encoding.py` for the four guards.

## 8. Results

*Nothing here is chosen by a leaderboard score. The challenge ranks entries on a
leaderboard: "Teams will be ranked based on the best Root Mean Square Error
(RMSE) achieved across their submissions", in the words of its ranking page.
This entry's rule for it: at most one upload per frozen version, no upload used
to choose a feature, a rung, a parameter or a rule, and nothing read back from
the scorer but `usedPairs` and the score. So this section carries what the
validation readings of §3.5 say about the model that is submitted, pooled, per
month, per airport and per source of delay, with and without the two rules of
§3.3, and leaves one line for the organizers' figure. §8.1 to §8.6 read the
model frozen on 27 September, the booster every later file is built on; §8.7
reads what the files of 4 to 7 October added to it.*

### 8.1 The model that ships, and what it scores on the forward splits

The submitted file is the rung named in `taxiout.models.SUBMISSION_KEY`, `+ plan
differences`: the arm §7.4 chose, with the four blocks of §6.5 above it, fitted
without the recorded-taxi tail, under five random seeds and averaged, with the
record-quality term on the 383 rows it covers and the plan-shift rule of §4 on
one. Every figure in this subsection is from `reports/baseline.md` as
regenerated on 28 September under the booster parameters adopted on 23 September
(§7.1.2), a regeneration that reproduced the freeze's ladder of 26 September
figure for figure on the validation legs and moved only the figures pooled at
the ranking set's bucket mix (§10). The average is measured on the pooled
figures over the rows the record-quality term leaves to the model; every figure
with the term, at the ranking mix and per airport on the forward splits is the
ladder's own single seed of the same rung, and the text says so where it quotes
one.

On the December split, which stands in for January 2026 at all ten airports, the
submitted model scores **213.55 s** over the rows the record-quality term leaves
to it. On the July split, which stands in for July 2026 at all ten, it scores
**320.90 s**, and 234.20 s on the three. Until the re-issued ranking set was
read (§2.1), this paper quoted the three-airport figure as its July figure,
because July 2026 held those three airports and no other; it holds all ten now,
Fiumicino among them, so the ten-airport figure is the one that stands in for
the scored month, and the three-airport one is kept beside it as the reading the
design of §7.1 fixed. §9.5 says what separates the two: one Fiumicino row. The
ladder's single seed of the same rung scores 214.7 s, 322.4 and 235.11 on the
same rows, so the average is worth 1.11 s, 1.46 and 0.90. On that single seed,
with the term substituted on the rows it covers, December reads 234.0 s and July
over all ten 368.2; with those rows left to the physical model alone, 364.8 and
620.5. The distance between the ex-flagged figures and the every-row ones is 88
rows in December and 335 in July, all at one airport, whose recorded taxi-out no
aircraft produced; the term is what closes it, and §4 says why that is the
RMSE-optimal treatment rather than a trick. Pooled at the ranking set's own
bucket mix rather than December's, the single seed is 216.0 s, and its bias is
+7 s in December and −3 in July.

| RMSE, seconds | December, all ten | July, all ten | July, the three |
| --- | ---: | ---: | ---: |
| submitted model, five seeds, rows the term leaves to it | 213.55 | 320.90 | 234.20 |
| the ladder's single seed, same rows | 214.7 | 322.4 | 235.11 |
| single seed, every row, the term on its rows | 234.0 | 368.2 | |
| single seed, every row, no term | 364.8 | 620.5 | |

The first row is the model that ships, read the way the paper reads every model;
the second is the seed every other figure in this subsection is fitted with; the
last two count every row, as the scorer does, with and without the
record-quality term on the 88 December and 335 July rows it covers. The
plan-shift rule touches no December row and one July row, the 87,002 s departure
of §9.5, and §4 and Annex I give what it does there.

Sources: `reports/baseline.md`, *What is fitted, and on what*, *The ablation
ladder*, *Error by pair frequency*, *Seed averaging* and *Per airport*;
`reports/tails.md`, *What it touches in the ranking set*;
`reports/planshift.md`, *Per held-out month and on the two forward legs*.

### 8.2 Every month of the year

The same model read with each month of 2025 held out in turn and the other
eleven training (§3.5): the five-seed average, the model that ships, refitted on
each split and scored on its month. The design is not forward in time, and the
forward splits of §8.1 remain the proxies for the two test months; what this
adds is the figure in every month, and a floor in every month. Three readings of
every month, as everywhere in this paper: excluding the rows the record-quality
term covers, as the scorer's rule implies; the same over the rows whose recorded
taxi-out is four hours or less, a diagnostic that leaves out the seven day-shift
rows the term does not cover and that the scorer counts (Annex F); and every row
with the term substituted on its rows, which is what the submission does. The
floor is the seed floor of §3.6 measured in that month.

| month | rows scored | five-seed average (s) | 4 h or less, diagnostic (s) | floor (s) | every row, with the term (s) | term rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| January | 153,646 | 346.8 | 231.8 (2 out) | 0.33 | 356.1 | 60 |
| February | 143,674 | 267.9 | 264.0 (1 out) | 1.56 | 282.2 | 58 |
| March | 164,390 | 200.3 | 196.4 (1 out) | 0.36 | 215.8 | 59 |
| April | 175,218 | 188.3 | 188.3 | 0.07 | 205.5 | 70 |
| May | 185,104 | 281.8 | 195.9 (1 out) | 0.13 | 302.2 | 98 |
| June | 182,957 | 218.0 | 218.0 | 0.21 | 318.3 | 185 |
| July | 190,378 | 311.7 | 251.3 (2 out) | 0.58 | 355.5 | 335 |
| August | 190,985 | 216.3 | 216.3 | 0.26 | 273.5 | 197 |
| September | 183,782 | 211.8 | 211.8 | 0.21 | 251.1 | 168 |
| October | 185,559 | 211.4 | 211.4 | 0.29 | 228.4 | 115 |
| November | 162,280 | 210.3 | 210.3 | 0.39 | 229.9 | 52 |
| December | 165,589 | 213.6 | 213.6 | 0.23 | 232.9 | 88 |
| **the year, pooled** | 2,083,562 | **243.7** | 217.8 (7 out) | 0.16 | 275.9 | 1,485 |

The year's row pools the squared errors of the twelve months' rows and is not a
mean of monthly figures; so does every margin in this paper. Held-out December
is the ladder's December split, since both train on January to November, and its
five seeds are the five of §8.1: 213.6 s here is the 213.55 s there at the
report's precision. The three readings are three different statements. The first
is what the scorer's rule implies for the rows the model predicts; the gap
between the first and the second is seven rows in the whole year, recorded
taxi-outs of 15,059 to 87,341 s at Paris, Fiumicino and Zurich that no term
covers; in May one such row is 51.5% of its month's squared error, and in
January two are 17.6% and 37.4%. The gap between the first and the third is the
term's 1,485 rows, all at Fiumicino, scored at the term's prediction. The
five-seed average beats every one of its five single seeds in eleven of the
twelve months; the exception is February, at 267.93 s against the best seed's
267.83, inside that month's floor of 1.56 s, the widest of the twelve.

Sources: `reports/monthly_seeds.md`, *Per month and over the year*;
`reports/monthly_ladder.md`, *Annex F: the rows over four hours*.

### 8.3 Every airport, every month

The same model per airport and month, RMSE excluding the rows the term covers,
the margins pooled from the rows they cover. A bracket is the diagnostic reading
over the rows of four hours or less where an airport has rows over four hours in
that month, with the count it leaves out. The bias per cell is in Annex C.

| airport | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frankfurt | 223.3 | 172.4 | 153.2 | 150.5 | 170.5 | 163.6 | 186.1 | 156.1 | 157.6 | 152.0 | 207.9 | 187.9 | 173.4 |
| Munich | 212.0 | 182.0 | 152.3 | 131.6 | 142.4 | 170.6 | 183.5 | 156.1 | 183.1 | 159.7 | 194.8 | 242.5 | 176.1 |
| Heathrow | 249.6 | 211.9 | 234.8 | 202.3 | 218.9 | 244.4 | 301.4 | 233.0 | 248.9 | 229.3 | 213.6 | 237.9 | 237.1 |
| Schiphol | 250.5 | 196.8 | 186.8 | 151.1 | 146.7 | 160.1 | 177.5 | 152.8 | 159.1 | 166.9 | 187.0 | 172.0 | 176.1 |
| Barcelona | 144.1 | 138.5 | 153.8 | 165.3 | 176.6 | 191.9 | 265.2 | 239.5 | 230.2 | 201.5 | 162.4 | 179.8 | 194.6 |
| Madrid | 163.5 | 151.1 | 187.5 | 168.3 | 176.3 | 184.6 | 192.6 | 182.1 | 181.9 | 191.8 | 172.2 | 170.0 | 177.8 |
| Paris | 810.1 [291.8; 2 out] | 232.2 | 233.9 [203.5; 1 out] | 196.0 | 220.4 | 245.5 | 253.3 [249.3; 1 out] | 233.9 | 230.7 | 227.8 | 276.5 | 251.3 | 317.9 [239.0; 4 out] |
| Fiumicino | 276.4 | 301.3 | 281.9 | 305.1 | 300.5 | 386.6 | 788.4 [434.5; 1 out] | 374.8 | 336.5 | 365.6 | 275.8 | 293.0 | 391.3 [335.7; 1 out] |
| Zurich | 241.1 | 256.2 [179.3; 1 out] | 161.5 | 150.9 | 809.7 [158.9; 1 out] | 168.7 | 188.5 | 161.5 | 151.2 | 153.6 | 195.8 | 181.6 | 298.3 [174.6; 2 out] |
| Istanbul | 212.1 | 520.8 | 202.4 | 210.1 | 207.4 | 202.1 | 257.4 | 210.3 | 197.1 | 215.2 | 192.6 | 203.0 | 244.5 |
| all ten | 346.8 [231.8; 2 out] | 267.9 [264.0; 1 out] | 200.3 [196.4; 1 out] | 188.3 | 281.8 [195.9; 1 out] | 218.0 | 311.7 [251.3; 2 out] | 216.3 | 211.8 | 211.4 | 210.3 | 213.6 | 243.7 [217.8; 7 out] |

What the table shows. Six airports, Frankfurt, Munich, Heathrow, Schiphol,
Barcelona and Madrid, pool to between 173.4 and 237.1 s over the year. Heathrow
is the widest of the six and its July, 301.4 s, the widest of their months,
which §9.5 reads. Three airports carry the year's day-shift rows: at Paris in
January, Zurich in May and Fiumicino in July one recorded taxi-out of a day or
more decides the cell, and the bracket beside it says what the cell reads
without it. Istanbul's February 2025 is the one cell above 500 s that no single
row explains, 520.8 s with no row over four hours and a bias of −92 s: the month
froze for 22.6% of its hours against a 2015 to 2024 range of 0.0% to 16.7%, and
had precipitation in 39.1% against 9.1% to 23.5% (§7.3, Annex G). Fiumicino runs
from 275.8 to 386.6 s in its other eleven months, the widest band of the ten,
and it is where the last three blocks of the chain earn most of their keep
(§9.5).

The forward splits per airport, for the record, on the ladder's single seed: the
three, Frankfurt, Heathrow and Schiphol, a third of the test rows, score 203.5 s
in December and 235.1 in July. Frankfurt is 188.4 and 189.7; Schiphol 174.1 and
182.4; Heathrow, the widest of the three, 239.4 and 314.4. Six of the other
seven run from Madrid's 170.5 to Paris's 252.5 in December, and from Munich's
189.2 to Barcelona's 269.9 in July. Fiumicino is the airport the term exists
for: 294.4 s in December and 814.5 in July without its rows, 1,131.9 and 2,043.6
with them and no term. The per-airport claim contrast is positive and clear of
zero at all ten airports, from +9.0 s [+6.0, +11.9] at Schiphol to +30.7 s
[+23.4, +38.0] at Frankfurt, which is also the airport with the largest
never-seen share after Munich (§5.4).

| airport | December RMSE (s) | December bias (s) | July RMSE (s) | July bias (s) | held-out-pair difference [95%] | ranking rows, January 2026 | ranking rows, July 2026 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frankfurt | 188.4 | +17 | 189.7 | +13 | +30.7 [+23.4, +38.0] | 15,667 | 20,650 |
| Munich | 244.7 | +11 | 189.2 | −17 | +10.8 [+7.2, +15.5] | 10,953 | 15,013 |
| Heathrow | 239.4 | +21 | 314.4 | +22 | +16.4 [+13.4, +19.6] | 19,283 | 20,557 |
| Schiphol | 174.1 | +15 | 182.4 | +13 | +9.0 [+6.0, +11.9] | 16,232 | 21,950 |
| Barcelona | 180.1 | −1 | 269.9 | −12 | +9.3 [+2.3, +16.3] | 12,201 | 17,879 |
| Madrid | 170.5 | +3 | 199.6 | +18 | +19.2 [+14.2, +23.9] | 16,802 | 20,152 |
| Paris | 252.5 | +5 | 265.9 | −6 | +24.2 [+19.5, +30.1] | 18,129 | 21,743 |
| Fiumicino | 294.4 | −48 | 814.5 | −96 | +22.5 [+16.8, +29.3] | 11,061 | 15,838 |
| Zurich | 182.5 | +15 | 193.5 | +7 | +15.8 [+9.6, +27.2] | 9,995 | 13,155 |
| Istanbul | 203.4 | +13 | 267.3 | −1 | +24.9 [+19.5, +30.6] | 22,396 | 25,185 |

RMSE and bias are the ladder's single seed on the forward splits, over the rows
the record-quality term leaves to the model (positive bias: the model predicts
too long). Fiumicino's July figure is one row's (§9.5). The held-out-pair
difference is the claim of §7.2 read at each airport alone. The last two columns
are how many rows of the scored file each airport holds.

Sources: `reports/monthly_seeds.md`, *Per airport and month: the five-seed
average*; `reports/monthly_ladder.md`, *Annex E: the historic weather context*;
`reports/baseline.md`, *The claim, per airport* and *Per airport*;
`reports/data_review.md`, *Airports and months* and *Inventory*;
`reports/tails.md`, *What it touches in the ranking set*.

### 8.4 What each source of delay is worth, over the year

§7.2 reads the ladder on the two forward splits and on the held-out pairs. Read
with every month held out, each block's worth is a change pooled over the year
and a count of months, and the count is read against the floor measured in that
month (§3.6): a block helps in a month when it improves the ex-flagged figure by
more than that month's floor, hurts when it worsens it by more, and otherwise
sits inside the floor, which is within what changing the seed alone produces.
Each block is read against the rung the ladder reads it against, so its change
is net of every block below it.

| block | year, ex-flagged (s) | change (s) | months it helps, beyond the floor | months inside the floor | months it hurts, beyond the floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| identifiers + time | 333.1 | | | | |
| geometry | 331.7 | −1.4 | 9 | 2 (June, July) | 1 (April) |
| aircraft | 328.5 | −3.2 | 11 | 1 (February) | 0 |
| plan lags | 265.7 | −62.9 | 12 | 0 | 0 |
| runway throughput | 262.9 | −2.7 | 12 | 0 | 0 |
| arrivals | 262.6 | −0.3 | 8 | 3 | 1 (February) |
| demand | 262.3 | −0.3 | 7 | 5 | 0 |
| configuration | 262.2 | −0.1 | 5 | 7 | 0 |
| shared corridor | 262.3 | +0.1 | 1 | 11 | 0 |
| weather | 261.5 | −0.9 | 9 | 3 | 0 |
| shrunk encoding, replacing the stand | 258.5 | −3.0 | 12 | 0 | 0 |
| stand occupancy | 251.2 | −7.2 | 12 | 0 | 0 |
| callsign prefix | 248.0 | −3.3 | 11 | 1 (February) | 0 |
| destination and holidays | 246.6 | −1.4 | 12 | 0 | 0 |
| plan differences (ships) | 244.4 | −2.2 | 11 | 1 (February) | 0 |

The year's figures are the ladder's single seed, pooled over the twelve held-out
months excluding the rows the term covers; the five-seed average of the shipped
rung is 243.7 s on the same rows (§8.2). Three readings. The chain that ships
holds up month by month: the plan lags, the throughput, the stand-replacing
encoding, the stand occupancy and the destination block help in every month, the
aircraft block, the callsign prefix and the plan differences in eleven, with
February, the month with the widest floor, the one month each of the three sits
inside it. The geometry block, worth −1.4 s pooled, helps in nine months and
hurts in April; its worth on a forward split was never the claim, which §8.5
reads month by month. And three of the congestion blocks are null over the year
once the floor is applied, the configuration block inside it in seven months,
the shared corridor in eleven, and the arrivals and demand blocks in three and
five: together the five congestion and weather blocks are worth 4.2 s pooled,
net of the plan lags, of which the weather block is 0.9 and the throughput 2.7.
On the chain fitted without the plan lags the weather block is worth 4.6 s
pooled and 16.5 s in January, which is what a plan lag absorbs when it is there
(Annex B).

Sources: `reports/monthly_ladder.md`, *Summary: the year, pooled from the twelve
months*; `reports/monthly_seeds.md`, *Per month and over the year* and *Each
block's months, against each month's floor*.

### 8.5 Where the geometry claim holds

The claim of §7.2, that the geometry block buys more where the pair is new than
where it is seen, read in each month of 2025 from the same three held-out-pair
folds, which train on the whole year and score their held-out pairs in every
month. The statistic is the claim's own, a fold-mean difference in differences
with its pair-clustered interval, restricted to the month's rows; the year's row
is the figure of §7.2.

| month | held-out rows, 3 folds | identifiers + time: seen / held out (s) | with geometry: seen / held out (s) | the claim [95%] |
| --- | ---: | ---: | ---: | ---: |
| January | 153,646 | 328.4 / 433.9 | 327.6 / 416.6 | +16.5 s [+14.0, +20.6] |
| February | 143,674 | 308.3 / 352.8 | 307.6 / 332.6 | +19.5 s [+15.8, +22.9] |
| March | 164,390 | 254.9 / 304.7 | 252.9 / 284.5 | +18.2 s [+16.1, +20.3] |
| April | 175,218 | 240.3 / 289.9 | 240.9 / 270.2 | +20.3 s [+18.3, +22.5] |
| May | 185,104 | 244.3 / 356.2 | 243.0 / 339.3 | +15.6 s [+13.0, +19.9] |
| June | 182,957 | 286.8 / 322.9 | 286.9 / 303.0 | +20.0 s [+17.7, +22.4] |
| July | 190,378 | 332.5 / 412.7 | 332.7 / 396.0 | +16.9 s [+13.2, +21.6] |
| August | 190,985 | 315.4 / 322.3 | 315.2 / 301.8 | +20.3 s [+17.9, +22.6] |
| September | 183,782 | 278.7 / 318.6 | 278.0 / 297.1 | +20.7 s [+18.4, +23.0] |
| October | 185,559 | 315.1 / 318.5 | 315.0 / 298.2 | +20.2 s [+18.2, +22.3] |
| November | 162,280 | 297.5 / 321.6 | 297.2 / 300.6 | +20.7 s [+18.3, +23.2] |
| December | 165,589 | 275.3 / 325.7 | 274.4 / 304.4 | +20.4 s [+17.8, +22.7] |
| **the year** | 2,083,562 | 291.4 / 343.8 | 290.9 / 324.9 | **+18.5 s [+16.7, +20.5]** |

**The claim clears zero in twelve of twelve months**, from +15.6 s in May to
+20.7 s in September and November, and over the rows of four hours or less, a
diagnostic, in twelve of twelve as well. Per airport and month the contrast
clears zero in every cell but three, Schiphol in January and Barcelona in
January and December (Annex D). The block's worth on the forward splits (§7.2),
and over the held-out year, −1.4 s pooled (§8.4), is the other reading of the
same block, and §9.1 says why the two are not in tension. The model that ships
reads 35.1 s [+13.8, +57.0] worse on a pair it has never seen than on one it
has, over the year, and 11.0 s [+5.9, +16.3] over the rows of four hours or
less; the seven day-shift rows, each on a pair held out in one fold, are the
difference between the two readings (Annex D).

Sources: `reports/monthly_pairs.md`, *Per month of 2025*, *Per airport and
month*, *The shipped rung's gap* and *In how many months the claim holds*;
`reports/monthly_ladder.md`, *Summary: the year, pooled from the twelve months*.

### 8.6 What the score will and will not say

The pooled test figure is a census of ten airports over two months, weighted by
their departures, with July 2026 holding 192,122 of the 344,841 rows; the §7.3
bounds put the plan-lag regime shift at up to +5.6 s pooled in January and +5.1
in July, most of July's at Paris, and the winter regime at Frankfurt and Zurich
is where the model extrapolates furthest. Three kinds of row decide the rest,
and §8.2 has shown each kind on the training year. The record-quality term
reaches 383 rows, all at Fiumicino, whose July departures are in the scored set
now; on the held-out year those rows take the figure from 243.7 s to 275.9. A
day-shift row in a cell the term does not cover is scored at its recorded value
and can move the pooled figure by itself: seven such rows in 2025, at three
airports, take the year from 217.8 s to 243.7, and January's figure from 231.8 s
to 346.8 on two of them. And one ranking row is predicted by the rule of §4 at
52,550 s, with the two outcomes that section prices. The organizers' figure is
one number over 344,841 rows, and the three readings above are what it is a
mixture of: 313.169 s, read once on 4 October 2026 over the 344,841 rows the
scorer used. That figure is v7's; §8.7 gives the four files after it.

Sources: `reports/congestion.md`, *What it bounds*; `reports/data_review.md`,
*Airports and months* and *Inventory*; `reports/tails.md`, *What it touches in
the ranking set* and *The plan-shifted departures*; `reports/monthly_seeds.md`,
*Per month and over the year*; `reports/planshift.md`, *The one ranking
departure*; `reports/submission-2026-10-04.md`, *The file*, a dated record.

### 8.7 After the first score: the files of 4 to 7 October

The organizers read back 313.169 s for v7 on 4 October. The entry then ran one
more week, testing what the frozen model had left out under the same discipline
as before: every change was fixed in a pre-registration before any of its
figures existed, and each pre-registration is published in `reports/` as a
dated record. Four more files were uploaded, each once and each when its rule
had spoken, and nothing was read back for any of them but `usedPairs` and the
score.

| file | uploaded | what it changes from the file before | organizers' figure (s) |
| --- | --- | --- | ---: |
| v7 | 4 October | the model of §8.1, with both rules of §3.3 | 313.169 |
| v8 | 4 October | both rules off: the booster predicts every row | 533.1066 |
| v9 | 6 October | both rules restored; two feature blocks and a CatBoost member added; the term's rows predicted per row | 282.399 |
| v10 | 7 October | ground events from open ADS-B and OPDI fused on top, gated by coverage and by airport | 272.0897 |
| v11 | 8 October | a CatBoost member with v9's new columns; the fusion refitted on top of it | 269.1368 |

**v8, and the one decision taken on a score.** The pre-registration of 4 October
judged the two rules of §3.3 a liability on the 2026 rows, on an analysis
computed before v7's figure existed, and v8 was built to that judgment: it
differs from v7 only on the 384 rows the two rules take. Its figure contradicted the
judgment, and I withdrew it the same evening in an amendment that states it as a
choice made after a score and on it. Every later file
substitutes both rules exactly as v7 does. No other decision in this entry reads
a score.

**v9.** Eight candidates fixed on 4 October, and a ninth on 6 October, were each
read against the frozen rung on the legs of §7.1 and the twelve held-out months, and
the ones that cleared alone were then read as one combination. Three cleared:
C1, the network manager's recorded off-block time `AOBT_3_flt` and the queue of
departures it lets the model count; C4, the flight number's own history in the
training year; and C3b, a CatBoost member blended at equal weight. Together they
are better than the frozen rung by 13.81 s on December, by 12.10 on July over
all ten airports, and by 8.25 pooled over the twelve held-out months. A winter
layer, a per-airport recalibration, a gamma-deviance member, thunderstorm flags
from the METAR and row weights toward the target month did not clear.
`AOBT_3_flt` had been excluded by decision since 2 September. It is published in
the ranking set and not blanked, the features built on it are this entry's own,
and admitting it was the owner's originality decision, taken on 6 October once
C1 had cleared and before v9 was built. On the 383 rows the record-quality term
covers, C6b replaces the term's single share per cell with a classifier of
whether a row's block time sits at its schedule: pooled over the twelve months
its mean squared error is 17,687,795 s² against the term's 23,585,522.

**v10: the aircraft seen on the ground.** Two open sources of surface positions
were matched to the departures: the adsb.lol globe history, under the Open
Database License, and the flight events of EUROCONTROL's OPDI, used on the
organizers' confirmation. From them come the times an aircraft is first seen on
the ground, first moves and stops, and the same for the other departures around
it. A stacker reads them on top of v9's out-of-fold prediction. Under its rule as
first written the fusion was better pooled, by 3.89 s, but worse in January and
July, the months where airports no receiver sees carry most of the error, so it
was not adopted. Taken only on rows a source covers (G1), it is better by 7.49 s
pooled and 9.67 on December, and holds in every month. At three airports, EGLL,
LFPG and LIRF, it was still worse on the rows it covered. G1n keeps the fusion at
an airport in a month only where it beat the base on that airport's covered rows
in the other eleven months. **G1n was defined after G1's per-airport figures had
been seen**, and its pre-registration says so. It is better than G1 by 2.40 s
pooled and 4.75 on December, and better than the base by 9.89 pooled. The gate
drops EGLL, LFPG and LIRF in every month and keeps the other seven; no ranking
row at LTFM is covered, so v10 takes the fusion at six airports.

**v11: one night airport by airport.** Six more candidates were read, five of
them fitted on every split overnight. The one with a mechanism behind it carried the result: C3b's
member had been fitted without the columns C1 and C4 added to the booster, and
refitted with them (C11) it is better than v9's configuration by 3.03 s over the
twelve held-out months. One booster per airport and a larger booster both failed
July; the winter layer failed its own January leg; the booster's weight in the
blend was chosen among four values, a selection the report discloses. A selector
then chose, per airport and out of fold, among eight of these predictions.
It is better than v9's configuration by 2.49 s pooled and 4.52 on December, and
for the ranking rows it takes C11's blend at nine airports and the old members at
LIRF. Its forward legs let months after a leg's month inform the choice, and the
report says so. The fusion refitted on this base is better than it by 6.59 s
pooled when gated by coverage, and G1n is better than that by 2.42; the gate is
unchanged. The airport-by-airport approach did not beat a better shared model:
the per-airport boosters lost July, and the selector chose the shared member
everywhere but one airport.

**Group C.** Two last candidates were fixed on 7 October before their figures:
the booster's weight in C11's blend, and a larger stacker for the fusion.
Neither cleared, so there is no v12 and v11 is the entry's file. Their readings
were made in working files outside the repository; their pre-registration is
published as a dated record with the others.

**What the five figures say.** Each file was uploaded once, after its rule had
spoken, and the board ranks a team by its best file, so none of these figures
chose anything. Read in order, every file the rules adopted on the training year
scored lower than the one before it, and the one file built on a judgment that
set the training year's evidence on those rows aside scored far higher.

Sources: `reports/submission-2026-10-04.md`, `reports/submission-2026-10-04-v8.md`,
`reports/submission-2026-10-06-v9.md`, `reports/submission-2026-10-07-v10.md` and
`reports/submission-2026-10-08-v11.md`, dated records;
`reports/candidates-2026-10-05.md`, *Section 4's combination*; `reports/c6.md`,
*Mean squared error, s²*; `reports/fusion.md`, *1. The fusion under §5's rule*,
*2. G1, the fusion gated by coverage* and *3. G1n, the fusion gated by coverage
and by airport, out of fold*; `reports/group-b.md`, *C11, the member with C1 and
C4* and *The combination*; `reports/fusion-new-base.md`, *G1 against the new
base* and *G1n against G1 (its rule)*.

## 9. Discussion

### 9.1 What the geometry was worth, and where

The geometry block is worth 2.27 s on the December forward split and nothing the
seed floor can see on July, where it reads 0.14 s worse, and +18.5 s [+16.7,
+20.5] on pairs the model has never seen. The forward-split figures and the
held-out one are the same block; they differ because the splits ask different
questions.

The first draft of this section explained the gap by asserting that the ranking
months reuse 2025's stands and runways, so the test set has no thin pairs and
cannot reward the physical layer. **That assertion was wrong, and §5.4 is the
measurement that replaced it.** 11.0% of the ranking rows sit on pairs the
training year saw a hundred times or fewer, the share runs from 2.08% at
Heathrow to 20.48% at Paris and is 11.51% in July against 10.35% in January, 46
stands are new outright, and on pairs seen one to ten times a pair's own mean is
worse than the airport mean by 53 s in December and 45 in July. The gap between
2.27 s and +18.5 s is therefore not a property of the test set. It is a property
of how the two encodings are combined: a booster handed both the identifiers and
the geometry splits on the identifiers wherever they are saturated, which on a
forward split is nearly everywhere, and the 10.9% of the submitted rung's splits
that the geometry block and its quality flags take is the residue. §7.4 is the
one attempt to combine them differently; it ships, and it bought its score from
the saturated pairs too.

What survives from the first draft, and is still worth saying: the geometry
block is an injective encoding of (stand, runway), so on a pair seen hundreds of
times a saturated identifier model already knows everything the distance could
tell it, and there the geometry can only lose. The claim was never that geometry
beats memorization where memorization works. It is that memorization runs out,
that the data says where, and that the geometry is what remains when it does;
and, in the leave-one-airport-out fits, that it is the only thing that remains.

The held-out year says the same thing twelve times. The claim contrast clears
zero in every month, from +15.6 s in May to +20.7 s in September and November,
and at every airport in every month but three cells (§8.5, Annex D). On the same
splits the block's forward worth is −1.4 s pooled over the year: it helps by
more than the month's floor in nine months, sits inside it in June and July and
hurts in April (§8.4). A block that is worth a second on the pairs a model has
seen and twenty on the pairs it has not is what the decomposition of §3
predicts, and the two readings are the same measurement taken on two row sets.

Sources: `reports/baseline.md`, *Held-out pairs*, *The noise floor* and *Where
`+ plan differences` spends its splits*; `reports/geometry_gaps.md`, *Where
memorization runs out*; `reports/monthly_pairs.md`, *Per month of 2025*;
`reports/monthly_seeds.md`, *Each block's months, against each month's floor*;
`reports/monthly_ladder.md`, *Summary: the year, pooled from the twelve months*.

### 9.2 What the honest tail treatment costs and buys

The term touches 383 ranking rows, 107 in January and 276 in July, all at
Fiumicino, and its whole effect on the leaderboard runs through those rows. On
the ranking files of 2 September it touched 107 and was inert on July, which
held no Fiumicino departure; July's 276 are the larger part of its reach now,
248 of them in the cell where a departure has no flight-plan record and an
ordinary lag. The alternative, predicting only physically plausible taxi-outs
and arguing the point, was set aside because it would place mid-table regardless
of model quality, and the argument can be made just as well in a paper that also
scores. The rule of §4 for the plan-shifted departures is the same choice made
once more, on one row and three training departures, and the paper prices both
of its outcomes there rather than defending one.

Source: `reports/tails.md`, *What it touches in the ranking set* and *What the
term is, stated plainly*.

### 9.3 Limitations

Assembled from the open questions already recorded: stands OSM does not have
(the join places 96.3% of training departures on a matrix cell, and most of the
rest sit in whole numeric blocks at two airports); the snapped-stand turn-angle
bias, expected on physical grounds and undetectable by the available proxy;
intersection departures at stand level; the OSM snapshot, fetched unpinned and
rebuilt in the freeze week from its recorded per-airport moments (§10); one
airport's data feed driving the entire record-quality term.

Six more that belong here because they concern how the numbers above were
produced rather than what they say:

**The booster's parameters were searched more than once, and the second search
was extended after its own rule said it would not be.** §7.1.2 gives the
account: a first sweep; a second, pre-registered and permitted once; an
extension of the second that the author reopened after its edge was recorded and
knowing its direction, under criteria fixed before its figures existed; and a
final choice between two cells the instrument cannot separate, taken on cost.
The first sweep, on 3 September, ran on the rung that was then the top of the
ladder, which carried the runway saturation block and neither the shrunk
encoding nor the four-hour exclusion, and its figures are not comparable with
the second's. Its grid ran from 235.0 to 244.8 s and its top eight cells lie
within 0.8 s of each other, and the same parameter cell moves from 235.0 s to
217.28 s on the change of fit rules alone, before any parameter changes.
`src/taxiout/models.py` keeps both sweeps' accounts beside the parameters.

**One ladder run was killed and restarted** after its generator was edited
mid-run; the reported figures are all from the completed run that followed.

**The ranking files changed under the entry, and the entry noticed late.** The
organizers replaced the ranking file and the template on 4 September, two days
after the copies the entry was built on were taken, and the entry read the
replaced files for the first time on 28 September, the day after the freeze
(§2.1). Nothing in the model was chosen on the ranking set, so the model did not
move; every ranking-set figure did, the July validation reading was revisited
(§8), and the two submission files built on the earlier template, neither of
them uploaded, were superseded by one built on the later. A check of the
organizers' files against the local copies, which the entry ran only before its
first upload, belonged at the start of every block that read them.

**The held-out year is not a forward reading.** Every month of 2025 held out in
turn trains on months after it as well as before, so the twelve figures of §8.2
and the per-month claim of §8.5 are read inside the training year and not across
its edge; they say how the figures and each block's worth are distributed over
the year, and the two forward splits remain the proxies for the test months. The
five-seed average fails to beat its own best seed in one of the twelve months,
February, by 0.10 s inside a floor of 1.56 s, which is the kind of near miss a
rule read at the floor is built to call noise.

**One rule rests on three rows.** The plan-shifted departures of §4 are a class
of three training departures, and the rule for them was written after one such
departure was seen in the ranking set's exposed fields; its share is an estimate
that three rows cannot pin, and its evaluation on the held-out year is
arithmetic on those rows and not evidence. The paper states both outcomes of the
one row it reaches, in score terms, and the pre-registration is the record that
nothing about the rule was chosen on its effect.

**Three candidates were measured after the freeze, and none was adopted.** The
runway-saturation block on top of the shipped rung, a recency schedule halving
every twelve months, and 8,000 trees in place of 6,000, each measured once
before the freeze and left on the table, were fitted on top of the frozen model
on 3 October and read under conditions fixed in writing before their figures
existed. None held all four: more trees cleared December by 0.28 s against a
floor of 0.23 and failed July; the saturation block was null on both forward
legs; the recency schedule failed all four. The verdict does not turn on the
precision it was read at, and the submitted file is unchanged (Annex J). What
the measurement shows is the ceiling the booster had reached under its own
parameters: the one candidate with a measurable gain on one leg gave it back on
the other.

**What was fixed before the numbers, what was measured, and what was decided
after.** Four points in this paper rest partly on a choice, and a reader should
be able to see which part. Every other rule the paper relies on was written down
before the figure it governs existed.

| point | fixed before the numbers | measured | decided after the numbers were seen |
| --- | --- | --- | --- |
| the geometry is worth 2.27 s on a forward split and +18.5 s on held-out pairs (§7.2, §9.1) | both splits, the held-out-pair design and the claim as a difference in differences, on 2 September | both figures with their intervals | nothing. The gap follows from the model: the geometry is a fixed function of the (stand, runway) pair, so where a pair has been seen hundreds of times the identifiers already carry what it knows, and it can only add where the pair is new |
| the encoding's held-out-pair gate was voided rather than passed (§7.4) | the rule: a positive held-out contrast and no worse on either forward split; the intersection signature of each runway designator, measured in another report before the arms ran | on a held-out pair the encoding equals its prior exactly, so the gate cannot see the thing it tests, whatever the numbers | the replacement rule's wording. Its rejecting half is the original rule's second half, unchanged; the paper and the report both say it was written after the arms' figures existed |
| a rule for the plan-shifted departures was added after the freeze (§4) | the rule itself, in every term, and its digest, before it was evaluated on any fold or built; the four-hour threshold it reads, fixed on 2 September | three training departures in the class, two of them long; one ranking departure; what the rule does to the three, month by month | the decision to add a rule at all, taken after the class was seen in the ranking set and knowing what the frozen model predicted for it; disclosed in §4 with both outcomes priced |
| the parameter sweep was extended after its own rule said it would not be (§7.1.2) | the sweep's grid and its October-only reading rule; the extension's criteria (seed floors, paired contrasts, a second-seed sign check), before any extension figure existed | the chosen cell sat on the edge of the grid on every axis; more trees won by more than three seed floors and held under a seed change; the other axes lost | the decision to reopen the sweep, taken knowing which way the edge pointed. Disclosed here and in §7.1.2; the tree count was then closed |

Nothing in the model is synthetic: the one fictional airport in the repository
(§5.2) tests the geometry pipeline and is held out of every feature, fit and
figure by a check. Of the constants the code holds, the provenance register
classes each as mechanism, quoted, derived or chosen, and a chosen one carries
its reason or is marked as having none on record, rather than given a reason
after the fact (`docs/provenance.md`).

Sources: `docs/modelling_notes.md`, *What the challenge data turned out to
contain* and *Open questions*; `reports/data_review.md`, *Stands*;
`reports/geometry_gaps.md`, *The intersection signature, per designator*;
`src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`;
`reports/ranking-refresh-2026-09-28.md`, a dated record; `reports/baseline.md`,
*The noise floor* and *Held-out pairs*; `docs/provenance.md`;
`reports/monthly_seeds.md`, *Per month and over the year*;
`reports/preregistered-2026-10-02.md`, a dated record;
`reports/candidates-2026-10-03.md`, *What was measured*, *The four conditions of
28 September* and *What the rule adopts*.

### 9.4 A ceiling that measured the wrong thing

The headroom estimate of §7.4 was wrong twice, and the two errors are worth more
than the number they were about.

The first was arithmetic. The estimate scales each pair-frequency bucket's
airport-mean error by the ratio the control achieves on the well-observed bucket
and calls the difference the attainable headroom. On the `11-100` bucket that
difference is negative, **−12.1 s** as the report now computes it: the control
is already ahead of what the ratio would predict. The first reading of that
table described the bucket as "about zero" and credited its magnitude, which
turned a pooled ceiling of about a third of a second into a claimed one to one
and a half seconds. The generator now computes the pooled figure both ways and
prints both, because the gap between them is the whole of the number: a negative
bucket held at zero gives +0.32 s, the signs taken literally give −0.79 s, and
the ceiling is positive only because of that convention. The error ran in the
direction that made a piece of work already proposed look worth doing, which is
the direction such errors run.

The second is the one that matters. The ceiling was not wrong about what it
measured; it measured the wrong thing. It assumed, by construction, that the
bucket carrying 91% of the rows had no headroom at all (the ratio was defined
from it) and looked only where the block was aimed. The block then bought 84% of
its gain from exactly that bucket. A ceiling that can only see the target of a
feature cannot bound a feature whose mechanism is not the one it was named for,
and the honest reading of "smoothed memorization" in §7.4 is that the mechanism
was never the one the ceiling was built to measure. The lesson is not that the
ceiling should have been larger. It is that a bound computed from where a block
is aimed is a bound on the aiming, and the report that carries it should say so
before the arms are run, not after.

Source: `reports/baseline.md`, *The ceiling this was built against* and *What
the chosen arm is, in its own terms*.

### 9.5 What the residual actually is

The error the model has left is a thin tail of individual holds. Three
measurements say so, and none of them is the absence of a queue model. Every
figure in this section is computed from the stored validation residuals of the
submitted rung's single-seed fit, the ladder's own; the five-seed average of §8
stores none.

**Its shape.** On the December split, rows with a residual over ten minutes are
1.78% of the rows and 38.3% of the squared error; over twenty minutes, 0.23% and
17.2%, and four in five of those are under-predictions. On the July leg's three
the same tail is 1.82% of the rows and 48.6% of the squared error, 55% of those
rows under-predicted. The under-predicted rows among those over ten minutes have
a median actual taxi-out of 2,221 s against a median prediction of 1,356 s in
December, 2,597 against 1,695 on the July three, and a ninetieth percentile of
3,466 s in December and 4,430 on the July three.

**It is not shared.** The oracle of §6.4, a departure's residual regressed on
its same-runway neighbors' mean residual, a quantity no admissible feature may
read, buys 8.2 s in December and 8.3 in July on the submitted rung, and the
lag-1 autocorrelation of consecutive residuals under two minutes apart is 0.19
and 0.09. A queue everyone stands in would show in both. These misses belong to
one aircraft at a time.

**It is enriched where individual holds live.** Heavies are 21.6% of December's
rows and 33.3% of its worst one percent of residuals. Departures without a
Network Manager record (a plan canceled and refiled after a long delay loses its
join, so the missing record is itself a delay signal) are 0.94% of the rows,
4.8% of the worst one percent and 3.6% of the squared error in December, and
1.99%, 11.0% and 14.8% on the July three, where they run at 641 s RMSE against
235.1 for the three pooled. Departures whose stand was taken by an aircraft of
another type twenty to forty-five minutes before take-off, the rows §6.5's bound
speaks about, are 0.54% of the rows and 3.1% of the worst one percent even after
the block that reads them; inside that band the block cut the mean
under-prediction from +229 s to +5, which is `reports/occupancy.md`'s
measurement rather than this section's. And the error grows with the plan lag as
a conditional mean between two modes must: against 181 s at the
fifteen-to-thirty-minute lag most rows have, the December RMSE is 415 s at
forty-five to sixty minutes past the filed off-block and 632 s at one to two
hours. Long lag is sometimes gate delay and sometimes a hold, and no exposed
field separates them.

**It is concentrated in one airport's afternoons, on one leg.** On the July leg,
Heathrow's error runs +75.0 s [+17.0, +136.9] above December at the same
airport, 20,790 rows at 314.4 s against 20,000 at 239.4 s. It is not spread over
the day. The six hours from 14:00 to 19:59 UTC hold 60% of July's squared error
on 38% of its rows and run +141.1 s [+40.7, +249.2] above July's other hours,
while the same hours in December run −13.0 s [−47.3, +19.5] against December's
other hours. The afternoon is a July effect at this airport and not a property
of the airport. The second concentration is by record: the 338 rows with no
Network Manager record, 1.6% of July's rows, carry 21.3% of its squared error at
1,138.9 s, which is +857.7 s [+300.7, +1,207.2] above the rows that have one; in
December the same class runs +207.2 s [+71.3, +389.0] above. Both contrasts are
structured, at day-clustered intervals clear of zero.

**Over the year, the last three blocks of the chain are mostly a Fiumicino
correction.** The callsign prefix, the destination block and the plan
differences take 86.0%, 58.3% and 35.4% of their year-level reduction in squared
error at Fiumicino, every part of it on rows with a flight-plan record, and the
first block's at that airport is worth −22.1 s against −3.3 pooled over the ten.
That is the operator's identity doing at one airport what it does nowhere else,
at the airport whose ordinary months run from 275.8 to 386.6 s (§8.3). For the
plan differences the share falls from 35.4% to 26.5% when the one July day-shift
row is left out, so part of that block's year-level worth is one departure
(Annex H). On the forward splits the three blocks are read in §7.2.

**Neither is a feature, and the reason is not modesty.** The hour of take-off,
the runway, the plan-lag band and the presence of a record are columns the
submitted rung already reads, and the callsign prefix is the `+ callsign prefix`
rung's own level, so there is no mechanism here the frame lacks. What the July
leg's model could not learn is what July's afternoons at Heathrow are like,
because that leg trains on January to June and holds no July (§7.3). The oracle
of §6.4 bounds what knowing the same-runway sequence exactly could still buy at
Heathrow on July at 9.7 s, and that is the scale any queue reading of the
afternoon would have to live inside. What is not shown is which afternoons: the
interval on the six-hour contrast runs from +40.7 to +249.2 s under day
clustering, which is what a handful of afternoons carrying the effect looks
like, and a per-day table would turn a description into a search.

**One row, and which July figure this paper quotes.** Over all ten airports the
July split's ex-flagged RMSE on the ladder's single seed is 322.4 s. One
departure at Fiumicino recording a taxi-out of 87,002 seconds, a day-shift in a
cell the record-quality term does not cover, so that no term touches it, carries
33.8% of that figure's squared error; the four-hour exclusion is fit-side only
and never reaches the scoring set. While July 2026 held no Fiumicino row, that
was the reason to quote the three-airport figure, 235.1 s on the same seed, as
the July figure, and this paper did so until the re-issued set was read. July
2026 holds 15,838 Fiumicino departures now (§2.1), and a row of this kind among
them is scored at its recorded value exactly as this one is, so the ten-airport
figure is the one §8 quotes and the three-airport one is kept beside it. The
rule that admitted the blocks was read on the ten-airport figure as it was
fixed, and nothing in it is re-read here.

What is left is individual, it is not shared, and no field the challenge exposes
predicts it: the fields the model had not read were measured before they were
built (`docs/review_headroom.md` §2), and the four that carried anything are in
the model. That is a bound on this entry and, if the argument holds, on every
entry working from the same columns. What would move it is information about the
hold itself (a regulation, a de-icing queue, a runway inspection), which the
challenge data does not carry and the open dataset that might, EUROCONTROL's own
daily ATFM delay by airport and cause, was excluded on its license (§2.2).

Sources: `reports/residual.md`, *The shape of the tail*, *What the tail is
enriched in*, *The error by plan lag*, *The largest rows* and *EGLL's July*, all
computed from the submitted rung's stored residuals; `reports/congestion.md`,
*The ceiling*, for the oracle and the autocorrelation; `reports/baseline.md`,
*Per airport*; `reports/occupancy.md`, *Where the gain lands*, for the in-band
before and after; `reports/data_review.md`, *Airports and months*, for the July
2026 census; `docs/review_headroom.md` §1–2; `reports/fiumicino.md`, *The
held-out-month ladder's year-level changes, recomputed*, *`+ callsign prefix`
against `+ stand occupancy`*, *`+ destination and holidays` against `+ callsign
prefix`* and *`+ plan differences` against `+ destination and holidays`*;
`reports/monthly_seeds.md`, *Per airport and month: the five-seed average*.

### 9.6 What held, and further work

The central hypothesis held where it could be tested. The recovered geometry
carries to stand–runway pairs the model has never seen (§8.5), and that is the
part of the entry an identifier model cannot reproduce. It did not, and could
not, move the test score by much: the test months reuse the training year's
pairs, so the forward splits and the organizers' figures read the identifiers'
strength rather than the geometry's (§9.1).

The larger gains of the last week came from data the geometry does not need:
the network manager's recorded off-block time, a second member refitted with
it, and aircraft seen on the ground by open receivers (§8.7). The last of these
is constrained by where the receivers listen. Over the training year's fitting
days a source covered 0.480 of Heathrow's departures; over the ranking months it
covered 0.923. On the rows it did cover in 2025 the fusion read 237.06 s against
the base's 219.17, so the out-of-fold gate dropped Heathrow and v11 does not use
the fusion there. As the coverage in the test months is nearly double that of
the training year, the stacker was fitted on a regime the test months do not
share, and its reading at Heathrow may reflect the sparse coverage of 2025 as
much as the airport itself.

Three directions follow from these limits. First, the fusion at Heathrow, Paris
and Rome should be refitted on a year in which the receivers see those airports
as densely as they see the test months, before it is read again. Secondly, the
rows at Fiumicino whose recorded block time sits at the schedule remain a source
of error the exposed fields do not identify well (§4); a source that records the
actual pushback there would resolve them directly rather than by a share.
Finally, the queueing layer this entry chose not to build was bounded at about
eight seconds net of the plan lags (§6.4). Ground events observe directly what a
queue model would have inferred, so that bound should be measured again with
the fusion in place before a simulation is reconsidered.

Sources: `reports/fusion.md`, *The rows* and *The gate the ranking rows take*.

## 10. Reproducing this

A reader brings four things to the repository at the commit this paper was
first committed in, tagged `entry-2026-10-04`: the challenge data,
from the organizers, which is not redistributable; the eleven Overpass answers,
fetched at their pins as described below or taken from the deposit; and, from
the deposit, the METAR observations and the two OurAirports tables the entry
read, cached fetches recorded with their requests and terms, which a fetch made
later would not return because OurAirports is rebuilt nightly. The final
phase's file, v12, reads two caches more: the adsb.lol event tables, from the
deposit or fetched again by `scripts/fetch_adsb.py`, and OPDI's flight events
and lists, whose site states no data licence, which the entry uses on the
organizers' confirmation and does not deposit, and which
`scripts/fetch_opdi.py` fetches again. From those and
the committed inputs, every figure in this paper is regenerated by a named
script, or is quoted from a dated record that the register holds to its digest,
and `scripts/check_paper.py` tells the two apart. One report is of the second
kind by necessity: `reports/determinism.md` is a measurement campaign under the
parameters of its day, whose records hold arrays derived from the challenge data
and are not redistributed, so a clean checkout cannot regenerate it and is not
asked to. The worked airport of §5.2 needs none of the four: it regenerates on a
bare checkout with no network and no data. The freeze ran in this order: the
pinned rebuild, every report regenerated and compared with the committed one,
the submission built, the commit tagged, then the reproduction from a clean
checkout and the deposit. One step came after the tag: on 28 September the two
ranking files were replaced by the organizers' versions of 4 September (§2.1),
the nine reports that read them were regenerated in the order their declared
inputs dictate, and the submission was rebuilt on the new template, with the
model, the matrices and the residuals untouched. A reader reproducing this paper
brings those versions of the two files, and
`reports/ranking-refresh-2026-09-28.md` gives the digests of both versions so
that the copy in hand can be told from the copy the entry was first built on.
Two more steps came after it, and the submitted file is built at the commit the
second of them made, tagged `model-v7-2026-10-02`, on top of the freeze tag and
not in place of it. The ladder was refitted on every month of 2025 held out in
turn, and its residuals, the five-seed refits of the shipped rung on the same
splits and the three pair folds refitted for three rungs are stored beside the
ladder's own; the seven reports of the held-out year that §8 and the annexes
quote regenerate from those stored files by their named scripts without a fit,
and the fits themselves repeat by the same scripts. Then the rule of §4 was
added, under its pre-registration, with the booster, the residuals and every
validation figure untouched. Last, the three candidates of Annex J were fitted
and read, and the file was not rebuilt, because the rule adopted none. Three tags mark the
history: `freeze-2026-10-04` the model's freeze, `model-v7-2026-10-02` the rule
of the plan-shifted departures, and `entry-2026-10-04` the file that was scored
first. The files of §8.7 are tagged `model-v8-2026-10-04`, `model-v9-2026-10-06`,
`model-v10-2026-10-07` and `model-v11-2026-10-07`; `scripts/make_submission.py`
at the last of them builds v11, and each file's record gives its digest, so a
rebuild can be told from the scored file byte for byte. v12 carries no tag yet:
`make final` builds it at the commit that landed its code. On 10 and 11 October
2026 a fresh clone at that commit, given the inputs above and no network,
formed the records of group B's selector from sixty fits by the committed
harness and built the final file byte for byte, by the route the README sets
out under *v12 from a clean checkout*.
The organizers' announcement channels, the ranking page's news box, the
challenge's main page and the OSN Discord channel `#prc-data-competition`, were
read on 4 October 2026, before the upload; they carried no final-phase file and
no change to the ranking script, the file format, the naming rule or the
deadline, only a note that a file is submitted with the MinIO command-line
client and not through the web interface.

**What repeats.** The booster is fitted in LightGBM's deterministic mode with
column-wise histograms at a pinned thread count, and a fit repeats to the bit:
`reports/determinism.md` fitted one rung, under the parameters in force when it
was measured, six times in two processes at eight threads and got one outcome,
and the same outcome at two, four and fourteen threads. The October seed floors
that the parameter search of §7.1.2 was read against came back identical to six
places in two separate runs with a commit between them, and all five single-seed
fits of the sweep's cell reproduced to the last digit; the headers of
`scripts/commit_extension_probe.sh` and `scripts/commit_equivalence.sh` are the
record of those two runs. And on 27 September 2026 a fresh clone at the tag,
with the challenge data and the two tables brought in and the geometry fetched
at its pins, rebuilt the 35 matrix files and the 52 residual files byte for
byte, regenerated fourteen reports in which no figure moved, passed this paper's
own check against them, and built the submission byte for byte, three times over
between the working copy and the clone; `reports/reproduction-2026-09-27.md` is
the dated record of that run. The regeneration of 28 September, under the
re-issued ranking files, rebuilt the 52 residual files byte for byte once more
and moved no figure of the validation legs.

**The geometry is rebuilt, not shipped.** The distance matrices are a Derivative
Database of OpenStreetMap under ODbL and are not in the repository: a clean
checkout rebuilds them from a pinned fetch, and the deposit is the other route
to the same files. A fresh fetch returns today's map, so the rebuild is pinned.
The committed figures were built from a fetch on 1 September 2026 in which each
airport was answered from the map as it stood at its own moment, those moments
are recorded per airport in `taxiout.provenance.OSM_PINS`, and
`scripts/fetch_osm.py --pin-from-register --force` fetches each airport at its
own. An Overpass answer's header never shows whether a pin was applied, so the
rebuild is verified by its effect: `--compare` checks each airport's parsed
elements against the answers the matrices were built from, of which the
deposit's copy is identical element for element, and `--control-date` asks the
instance that answered for a date whose answer is known to differ, which shows
that it applies a pin at all. A header can show the opposite, that a pin could
not have been applied, because an instance cannot serve a state it has not
ingested; the client refuses an answer whose database state precedes the pin,
and on 26 and 27 September the two fallback instances it knows answered from
databases of May to July and were refused, every accepted answer coming from the
primary.

**The paper checks itself.** `scripts/check_paper.py` reads every figure in this
paper and requires it to appear, at the paper's own precision, in the file its
section's source line cites and, where that line names a section of the file, in
that section. A figure found only in a dated record must be listed as historical
with a reason, and one computed from report figures must carry its arithmetic.
It is a check of presence and not of correctness: a short figure can match a
number in a long report by chance, so a passing check certifies that every
figure appears where its section says it comes from, and the match it prints for
each figure is how a reader tells the right number from a coincidence.

Sources: README, *Reproducing* and *Data sources*;
`reports/reproduction-2026-09-27.md`, a dated record; `reports/determinism.md`,
*The booster*, a dated record; `reports/ranking-refresh-2026-09-28.md`, a dated
record; `scripts/check_paper.py`; `docs/plan.md`, freeze section.

## Annex A: the ladder, every month held out

Every rung of the ladder on every month of 2025 held out in turn, the other
eleven months training, at the ladder's own seed: the RMSE in seconds excluding
the rows the record-quality term covers, the *year* column pooling the twelve
months' rows. Where a month holds rows whose recorded taxi-out is over four
hours, the bracket is the same figure over the rows of four hours or less, a
diagnostic, with the count it leaves out. The design is not forward in time
(§3.5). Rows per month, excluding the term's rows / of which over four hours /
covered by the term:

|  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rows | 153,646 / 2 / 60 | 143,674 / 1 / 58 | 164,390 / 1 / 59 | 175,218 / 0 / 70 | 185,104 / 1 / 98 | 182,957 / 0 / 185 | 190,378 / 2 / 335 | 190,985 / 0 / 197 | 183,782 / 0 / 168 | 185,559 / 0 / 115 | 162,280 / 0 / 52 | 165,589 / 0 / 88 | 2,083,562 / 7 / 1,485 |

| rung | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `airport mean` | 491.8 [418.9; 2 out] | 470.9 [468.6; 1 out] | 348.7 [345.1; 1 out] | 334.8 | 402.2 [348.2; 1 out] | 361.7 | 465.8 [420.9; 2 out] | 363.0 | 357.5 | 362.2 | 374.8 | 380.0 | 394.0 [377.9; 7 out] |
| `PRU reference` | 532.8 [465.8; 2 out] | 514.7 [512.4; 1 out] | 394.0 [390.6; 1 out] | 381.3 | 449.3 [401.1; 1 out] | 416.9 | 530.8 [491.3; 2 out] | 424.7 | 424.2 | 423.5 | 424.5 | 435.8 | 447.4 [433.0; 7 out] |
| `geometric unimpeded` | 540.8 [474.8; 2 out] | 525.6 [523.5; 1 out] | 402.9 [399.7; 1 out] | 395.2 | 459.3 [412.4; 1 out] | 429.6 | 541.3 [502.7; 2 out] | 440.7 | 438.3 | 437.5 | 438.5 | 448.2 | 459.4 [445.4; 7 out] |
| `geometric unimpeded + wake class` | 531.1 [463.8; 2 out] | 516.6 [514.4; 1 out] | 393.3 [389.9; 1 out] | 384.5 | 449.7 [401.8; 1 out] | 419.3 | 531.6 [492.2; 2 out] | 429.7 | 427.0 | 427.5 | 427.3 | 436.5 | 449.1 [434.9; 7 out] |
| `identifiers + time` | 455.4 [375.5; 2 out] | 407.9 [405.3; 1 out] | 288.3 [283.8; 1 out] | 263.9 | 340.5 [274.0; 1 out] | 296.2 | 404.8 [352.0; 2 out] | 295.2 | 287.9 | 298.4 | 305.3 | 324.0 | 333.1 [313.8; 7 out] |
| `identifiers + time + geometry` | 453.6 [373.4; 2 out] | 403.0 [400.3; 1 out] | 285.5 [281.0; 1 out] | 264.1 | 339.7 [273.1; 1 out] | 296.3 | 404.6 [351.7; 2 out] | 293.3 | 287.3 | 297.2 | 304.2 | 321.7 | 331.7 [312.3; 7 out] |
| `identifiers + time + geometry + aircraft` | 448.3 [366.8; 2 out] | 401.9 [399.2; 1 out] | 282.9 [278.3; 1 out] | 259.8 | 337.4 [270.1; 1 out] | 292.4 | 401.0 [347.6; 2 out] | 288.9 | 283.5 | 293.8 | 301.8 | 320.2 | 328.5 [308.9; 7 out] |
| `+ plan lags` | 365.6 [259.0; 2 out] | 292.1 [288.4; 1 out] | 222.0 [217.3; 1 out] | 207.7 | 297.4 [218.0; 1 out] | 240.1 | 338.9 [283.2; 2 out] | 239.6 | 235.3 | 236.0 | 231.6 | 240.9 | 265.7 [242.0; 7 out] |
| `+ runway throughput` | 364.6 [257.7; 2 out] | 287.0 [283.4; 1 out] | 218.6 [213.6; 1 out] | 206.0 | 296.0 [216.1; 1 out] | 237.8 | 335.3 [278.6; 2 out] | 237.5 | 231.9 | 232.5 | 229.1 | 236.8 | 262.9 [238.9; 7 out] |
| `+ arrivals` | 363.8 [256.6; 2 out] | 289.7 [286.0; 1 out] | 217.7 [212.9; 1 out] | 205.0 | 295.6 [215.5; 1 out] | 237.1 | 334.8 [277.7; 2 out] | 237.1 | 231.9 | 232.7 | 228.4 | 235.8 | 262.6 [238.6; 7 out] |
| `+ demand` | 363.4 [256.0; 2 out] | 289.6 [285.9; 1 out] | 217.0 [212.4; 1 out] | 204.7 | 295.1 [214.8; 1 out] | 236.6 | 335.0 [277.1; 2 out] | 236.5 | 231.7 | 232.7 | 228.5 | 235.5 | 262.3 [238.2; 7 out] |
| `+ configuration` | 363.7 [256.3; 2 out] | 290.4 [286.7; 1 out] | 217.1 [212.4; 1 out] | 204.6 | 295.2 [215.0; 1 out] | 236.6 | 334.3 [276.7; 2 out] | 236.3 | 231.4 | 232.0 | 228.3 | 235.0 | 262.2 [238.1; 7 out] |
| `+ shared corridor` | 363.8 [256.3; 2 out] | 291.8 [288.2; 1 out] | 216.9 [212.1; 1 out] | 204.7 | 294.8 [214.4; 1 out] | 236.4 | 334.5 [276.8; 2 out] | 236.5 | 231.3 | 232.1 | 228.3 | 235.1 | 262.3 [238.2; 7 out] |
| `+ weather` | 360.6 [252.0; 2 out] | 288.9 [285.3; 1 out] | 216.4 [211.6; 1 out] | 204.3 | 294.8 [214.6; 1 out] | 236.2 | 335.0 [276.6; 2 out] | 236.2 | 230.9 | 232.1 | 226.7 | 233.2 | 261.5 [237.2; 7 out] |
| `+ shrunk encoding` | 360.4 [251.7; 2 out] | 288.6 [284.9; 1 out] | 215.4 [210.7; 1 out] | 203.6 | 294.5 [213.9; 1 out] | 235.9 | 336.8 [276.8; 2 out] | 235.9 | 231.0 | 231.6 | 226.7 | 231.4 | 261.3 [236.7; 7 out] |
| `+ shrunk encoding, replacing the stand` | 358.5 [248.9; 2 out] | 284.1 [280.4; 1 out] | 212.9 [208.1; 1 out] | 200.9 | 292.2 [210.7; 1 out] | 233.4 | 333.5 [272.6; 2 out] | 233.1 | 227.6 | 228.3 | 224.5 | 228.3 | 258.5 [233.6; 7 out] |
| `+ shrunk encoding, replacing stand and runway` | 358.8 [249.1; 2 out] | 286.0 [282.3; 1 out] | 213.5 [208.6; 1 out] | 201.3 | 292.4 [211.1; 1 out] | 233.3 | 333.2 [272.4; 2 out] | 233.2 | 227.8 | 228.2 | 224.6 | 228.0 | 258.7 [233.9; 7 out] |
| `+ stand occupancy` | 351.9 [239.6; 2 out] | 272.0 [268.1; 1 out] | 206.4 [201.9; 1 out] | 194.9 | 287.6 [204.2; 1 out] | 226.3 | 325.0 [262.6; 2 out] | 225.8 | 221.1 | 221.2 | 216.8 | 219.7 | 251.2 [225.7; 7 out] |
| `+ stand occupancy, schedule-anchored` (diagnostic) | 356.0 [245.5; 2 out] | 280.5 [276.7; 1 out] | 210.4 [205.7; 1 out] | 197.9 | 290.5 [208.3; 1 out] | 231.2 | 330.3 [269.6; 2 out] | 230.0 | 225.4 | 225.6 | 221.8 | 224.7 | 255.8 [230.8; 7 out] |
| `+ callsign prefix` | 349.5 [236.0; 2 out] | 272.1 [268.2; 1 out] | 203.9 [199.1; 1 out] | 191.3 | 284.5 [199.9; 1 out] | 222.3 | 319.4 [258.1; 2 out] | 220.9 | 217.1 | 217.8 | 213.5 | 217.6 | 248.0 [222.3; 7 out] |
| `+ destination and holidays` | 349.1 [235.3; 2 out] | 269.1 [265.2; 1 out] | 202.9 [198.5; 1 out] | 189.7 | 283.6 [198.5; 1 out] | 221.0 | 318.4 [255.6; 2 out] | 218.7 | 215.4 | 216.0 | 212.7 | 215.8 | 246.6 [220.6; 7 out] |
| `+ plan differences` | 347.5 [233.1; 2 out] | 267.9 [263.9; 1 out] | 201.4 [197.4; 1 out] | 189.1 | 282.3 [196.7; 1 out] | 218.9 | 312.5 [252.3; 2 out] | 217.1 | 212.6 | 212.6 | 211.3 | 214.7 | 244.4 [218.7; 7 out] |
| `+ runway saturation` | 360.8 [252.3; 2 out] | 288.0 [284.3; 1 out] | 215.9 [211.2; 1 out] | 204.1 | 294.7 [214.3; 1 out] | 236.1 | 334.1 [276.8; 2 out] | 236.4 | 230.8 | 232.1 | 226.9 | 232.7 | 261.2 [237.1; 7 out] |
| `+ runway throughput (without plan lags)` | 444.4 [362.2; 2 out] | 400.3 [397.6; 1 out] | 275.8 [271.1; 1 out] | 253.9 | 333.2 [264.7; 1 out] | 287.6 | 396.6 [342.6; 2 out] | 283.6 | 278.3 | 288.4 | 296.4 | 316.7 | 324.0 [304.0; 7 out] |
| `+ arrivals (without plan lags)` | 438.7 [354.9; 2 out] | 396.4 [393.7; 1 out] | 271.5 [266.7; 1 out] | 250.9 | 331.3 [262.4; 1 out] | 284.1 | 393.7 [339.2; 2 out] | 281.4 | 276.8 | 285.1 | 292.7 | 308.5 | 320.4 [300.2; 7 out] |
| `+ demand (without plan lags)` | 433.2 [348.2; 2 out] | 392.4 [389.6; 1 out] | 268.4 [263.6; 1 out] | 246.9 | 329.0 [259.4; 1 out] | 280.2 | 389.8 [334.6; 2 out] | 277.6 | 272.9 | 281.6 | 288.3 | 301.2 | 316.3 [295.8; 7 out] |
| `+ configuration (without plan lags)` | 435.1 [350.4; 2 out] | 391.4 [388.7; 1 out] | 267.8 [263.0; 1 out] | 246.7 | 328.6 [259.0; 1 out] | 279.9 | 389.5 [334.3; 2 out] | 277.4 | 272.8 | 281.4 | 287.8 | 300.4 | 316.1 [295.6; 7 out] |
| `+ shared corridor (without plan lags)` | 433.9 [349.0; 2 out] | 392.9 [390.2; 1 out] | 267.8 [263.0; 1 out] | 246.8 | 328.9 [259.4; 1 out] | 280.0 | 389.4 [334.3; 2 out] | 277.8 | 273.1 | 281.1 | 288.0 | 300.8 | 316.2 [295.8; 7 out] |
| `+ weather (without plan lags)` | 417.4 [328.8; 2 out] | 378.2 [375.4; 1 out] | 265.9 [261.1; 1 out] | 246.0 | 327.8 [257.9; 1 out] | 278.1 | 389.3 [334.1; 2 out] | 277.0 | 272.3 | 280.2 | 282.3 | 292.5 | 311.7 [290.9; 7 out] |
| `+ runway saturation (without plan lags)` | 417.3 [328.7; 2 out] | 378.1 [375.2; 1 out] | 265.7 [260.9; 1 out] | 245.7 | 327.9 [258.1; 1 out] | 278.6 | 389.2 [334.0; 2 out] | 277.1 | 272.0 | 280.1 | 282.4 | 292.0 | 311.6 [290.8; 7 out] |

The shipped rung is `+ plan differences`; `+ stand occupancy, schedule-anchored`
is a diagnostic off the chain (§6.5), and the rungs marked *without plan lags*
are the congestion chain fitted without the plan-lag block (§6.3). Each rung's
change against the rung it is read against, month by month, is the next annex;
the same per airport and month, every rung, is the report's own Annex A.

Sources: `reports/monthly_ladder.md`, *Summary: every month* and *Annex A:
airport by month, every rung*.

## Annex B: each block's change, month by month, against the floor

Each block's change against the rung the ladder reads it against, excluding the
rows the term covers, in seconds, read against the seed floor measured in that
month on the shipped rung (§3.6; the floor is measured on one rung and applied
to every block's change, the same approximation `reports/baseline.md` makes).
**A change in bold sits inside that month's floor**; the year column is read
against the year's floor. A bracket is the same change over the rows of four
hours or less, a diagnostic, with the count it leaves out, read against the
floor measured over those rows. The floors first:

|  | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| floor | 0.33 [0.47] | 1.56 [1.59] | 0.36 [0.29] | 0.07 [0.07] | 0.13 [0.17] | 0.21 [0.21] | 0.58 [0.42] | 0.26 [0.26] | 0.21 [0.21] | 0.29 [0.29] | 0.39 [0.39] | 0.23 [0.23] | 0.16 [0.15] |

| block | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `PRU reference` against `airport mean` | +41.1 [+46.9; 2 out] | +43.8 [+43.8; 1 out] | +45.2 [+45.4; 1 out] | +46.5 | +47.1 [+52.9; 1 out] | +55.3 | +65.0 [+70.4; 2 out] | +61.7 | +66.8 | +61.3 | +49.8 | +55.8 | +53.3 [+55.1; 7 out] |
| `geometric unimpeded` against `PRU reference` | +7.9 [+9.0; 2 out] | +10.9 [+11.1; 1 out] | +8.9 [+9.1; 1 out] | +13.9 | +10.0 [+11.3; 1 out] | +12.6 | +10.5 [+11.4; 2 out] | +16.0 | +14.0 | +14.0 | +13.9 | +12.5 | +12.0 [+12.4; 7 out] |
| `geometric unimpeded + wake class` against `geometric unimpeded` | -9.6 [-11.0; 2 out] | -9.1 [-9.1; 1 out] | -9.6 [-9.7; 1 out] | -10.7 | -9.5 [-10.6; 1 out] | -10.3 | -9.7 [-10.5; 2 out] | -11.0 | -11.2 | -10.0 | -11.1 | -11.7 | -10.2 [-10.5; 7 out] |
| `identifiers + time` against `geometric unimpeded + wake class` | -75.8 [-88.3; 2 out] | -108.6 [-109.1; 1 out] | -105.0 [-106.2; 1 out] | -120.6 | -109.3 [-127.8; 1 out] | -123.0 | -126.8 [-140.3; 2 out] | -134.5 | -139.1 | -129.1 | -122.1 | -112.6 | -116.0 [-121.1; 7 out] |
| `identifiers + time + geometry` against `identifiers + time` | -1.7 [-2.0; 2 out] | -5.0 [-5.0; 1 out] | -2.8 [-2.8; 1 out] | +0.2 | -0.7 [-1.0; 1 out] | **+0.1** | **-0.2** [**-0.3**; 2 out] | -1.8 | -0.7 | -1.2 | -1.1 | -2.3 | -1.4 [-1.5; 7 out] |
| `identifiers + time + geometry + aircraft` against `identifiers + time + geometry` | -5.3 [-6.6; 2 out] | **-1.1** [**-1.1**; 1 out] | -2.6 [-2.6; 1 out] | -4.3 | -2.3 [-2.9; 1 out] | -3.9 | -3.6 [-4.1; 2 out] | -4.4 | -3.8 | -3.4 | -2.4 | -1.5 | -3.2 [-3.4; 7 out] |
| `+ plan lags` against `identifiers + time + geometry + aircraft` | -82.7 [-107.8; 2 out] | -109.8 [-110.7; 1 out] | -60.9 [-61.0; 1 out] | -52.1 | -40.0 [-52.1; 1 out] | -52.3 | -62.1 [-64.5; 2 out] | -49.3 | -48.2 | -57.8 | -70.1 | -79.3 | -62.9 [-66.9; 7 out] |
| `+ runway throughput` against `+ plan lags` | -1.0 [-1.3; 2 out] | -5.0 [-5.1; 1 out] | -3.4 [-3.7; 1 out] | -1.7 | -1.4 [-1.9; 1 out] | -2.3 | -3.6 [-4.5; 2 out] | -2.1 | -3.4 | -3.5 | -2.5 | -4.1 | -2.7 [-3.0; 7 out] |
| `+ arrivals` against `+ runway throughput` | -0.8 [-1.2; 2 out] | +2.7 [+2.7; 1 out] | -1.0 [-0.8; 1 out] | -1.0 | -0.4 [-0.6; 1 out] | -0.8 | **-0.5** [-0.9; 2 out] | -0.4 | **+0.0** | **+0.1** | -0.7 | -1.0 | -0.3 [-0.4; 7 out] |
| `+ demand` against `+ arrivals` | -0.4 [-0.6; 2 out] | **-0.1** [**-0.1**; 1 out] | -0.7 [-0.5; 1 out] | -0.3 | -0.5 [-0.6; 1 out] | -0.5 | **+0.1** [-0.6; 2 out] | -0.6 | **-0.2** | **+0.0** | **+0.0** | -0.4 | -0.3 [-0.4; 7 out] |
| `+ configuration` against `+ demand` | **+0.3** [**+0.3**; 2 out] | **+0.8** [**+0.8**; 1 out] | **+0.2** [**+0.0**; 1 out] | -0.1 | **+0.1** [+0.2; 1 out] | **+0.0** | -0.6 [**-0.4**; 2 out] | **-0.2** | -0.3 | -0.7 | **-0.2** | -0.5 | **-0.1** [**-0.1**; 7 out] |
| `+ shared corridor` against `+ configuration` | **+0.1** [**-0.1**; 2 out] | **+1.5** [**+1.4**; 1 out] | **-0.2** [**-0.3**; 1 out] | **+0.0** | -0.4 [-0.6; 1 out] | **-0.1** | **+0.2** [**+0.1**; 2 out] | **+0.2** | **-0.1** | **+0.1** | **+0.0** | **+0.1** | **+0.1** [**+0.1**; 7 out] |
| `+ weather` against `+ shared corridor` | -3.2 [-4.2; 2 out] | -2.9 [-2.9; 1 out] | -0.5 [-0.6; 1 out] | -0.4 | **+0.0** [**+0.1**; 1 out] | -0.3 | **+0.4** [**-0.2**; 2 out] | -0.3 | -0.4 | **+0.1** | -1.6 | -1.9 | -0.9 [-1.0; 7 out] |
| `+ shrunk encoding` against `+ weather` | **-0.2** [**-0.3**; 2 out] | **-0.3** [**-0.4**; 1 out] | -1.0 [-0.9; 1 out] | -0.7 | -0.3 [-0.7; 1 out] | -0.3 | +1.9 [**+0.1**; 2 out] | -0.3 | **+0.1** | -0.5 | **+0.0** | -1.8 | -0.2 [-0.4; 7 out] |
| `+ shrunk encoding, replacing the stand` against `+ weather` | -2.1 [-3.2; 2 out] | -4.9 [-4.9; 1 out] | -3.5 [-3.5; 1 out] | -3.3 | -2.7 [-3.9; 1 out] | -2.7 | -1.5 [-4.1; 2 out] | -3.1 | -3.3 | -3.8 | -2.2 | -4.9 | -3.0 [-3.6; 7 out] |
| `+ shrunk encoding, replacing stand and runway` against `+ weather` | -1.8 [-2.9; 2 out] | -2.9 [-3.0; 1 out] | -3.0 [-3.0; 1 out] | -3.0 | -2.4 [-3.5; 1 out] | -2.9 | -1.8 [-4.2; 2 out] | -3.0 | -3.1 | -3.9 | -2.1 | -5.2 | -2.8 [-3.3; 7 out] |
| `+ stand occupancy` against `+ shrunk encoding, replacing the stand` | -6.7 [-9.3; 2 out] | -12.1 [-12.3; 1 out] | -6.5 [-6.3; 1 out] | -6.0 | -4.6 [-6.5; 1 out] | -7.1 | -8.4 [-10.0; 2 out] | -7.3 | -6.5 | -7.1 | -7.7 | -8.6 | -7.2 [-7.9; 7 out] |
| `+ stand occupancy, schedule-anchored` (diagnostic) against `+ shrunk encoding, replacing the stand` | -2.5 [-3.3; 2 out] | -3.6 [-3.7; 1 out] | -2.5 [-2.4; 1 out] | -3.1 | -1.7 [-2.4; 1 out] | -2.3 | -3.2 [-2.9; 2 out] | -3.1 | -2.2 | -2.7 | -2.7 | -3.6 | -2.7 [-2.9; 7 out] |
| `+ callsign prefix` against `+ stand occupancy` | -2.3 [-3.6; 2 out] | **+0.1** [**+0.1**; 1 out] | -2.6 [-2.8; 1 out] | -3.6 | -3.1 [-4.3; 1 out] | -4.0 | -5.6 [-4.5; 2 out] | -4.9 | -4.0 | -3.5 | -3.2 | -2.1 | -3.3 [-3.4; 7 out] |
| `+ destination and holidays` against `+ callsign prefix` | -0.5 [-0.7; 2 out] | -3.0 [-3.1; 1 out] | -0.9 [-0.7; 1 out] | -1.6 | -0.9 [-1.4; 1 out] | -1.3 | -1.0 [-2.5; 2 out] | -2.2 | -1.7 | -1.8 | -0.8 | -1.8 | -1.4 [-1.7; 7 out] |
| `+ plan differences` against `+ destination and holidays` | -1.6 [-2.2; 2 out] | **-1.2** [**-1.2**; 1 out] | -1.6 [-1.1; 1 out] | -0.6 | -1.2 [-1.8; 1 out] | -2.2 | -5.9 [-3.3; 2 out] | -1.6 | -2.8 | -3.4 | -1.4 | -1.1 | -2.2 [-2.0; 7 out] |
| `+ runway saturation` against `+ weather` | **+0.2** [**+0.3**; 2 out] | **-1.0** [**-1.0**; 1 out] | -0.5 [-0.4; 1 out] | -0.2 | **-0.1** [-0.3; 1 out] | **-0.1** | -0.9 [**+0.2**; 2 out] | **+0.2** | **-0.1** | **-0.1** | **+0.2** | -0.5 | -0.2 [**-0.1**; 7 out] |
| `+ runway throughput (without plan lags)` against `identifiers + time + geometry + aircraft` | -3.9 [-4.6; 2 out] | **-1.6** [**-1.5**; 1 out] | -7.2 [-7.2; 1 out] | -5.9 | -4.2 [-5.5; 1 out] | -4.8 | -4.4 [-5.0; 2 out] | -5.3 | -5.1 | -5.5 | -5.4 | -3.6 | -4.6 [-4.8; 7 out] |
| `+ arrivals (without plan lags)` against `+ runway throughput (without plan lags)` | -5.8 [-7.3; 2 out] | -3.9 [-4.0; 1 out] | -4.3 [-4.4; 1 out] | -3.0 | -1.8 [-2.3; 1 out] | -3.5 | -2.9 [-3.4; 2 out] | -2.2 | -1.6 | -3.3 | -3.7 | -8.2 | -3.6 [-3.9; 7 out] |
| `+ demand (without plan lags)` against `+ arrivals (without plan lags)` | -5.5 [-6.7; 2 out] | -4.0 [-4.1; 1 out] | -3.1 [-3.1; 1 out] | -4.0 | -2.4 [-3.0; 1 out] | -3.9 | -4.0 [-4.6; 2 out] | -3.8 | -3.9 | -3.5 | -4.5 | -7.3 | -4.1 [-4.4; 7 out] |
| `+ configuration (without plan lags)` against `+ demand (without plan lags)` | +1.9 [+2.2; 2 out] | **-1.0** [**-1.0**; 1 out] | -0.6 [-0.6; 1 out] | -0.2 | -0.3 [-0.4; 1 out] | -0.3 | **-0.3** [**-0.3**; 2 out] | **-0.2** | **-0.1** | **-0.3** | -0.4 | -0.8 | -0.2 [-0.2; 7 out] |
| `+ shared corridor (without plan lags)` against `+ configuration (without plan lags)` | -1.2 [-1.4; 2 out] | **+1.5** [**+1.5**; 1 out] | **+0.0** [**+0.0**; 1 out] | +0.1 | +0.3 [+0.4; 1 out] | **+0.1** | **-0.1** [**+0.0**; 2 out] | +0.4 | +0.3 | **-0.2** | **+0.2** | +0.5 | **+0.1** [+0.2; 7 out] |
| `+ weather (without plan lags)` against `+ shared corridor (without plan lags)` | -16.5 [-20.2; 2 out] | -14.7 [-14.8; 1 out] | -1.9 [-1.9; 1 out] | -0.8 | -1.2 [-1.5; 1 out] | -1.9 | **-0.1** [**-0.2**; 2 out] | -0.8 | -0.8 | -0.9 | -5.8 | -8.3 | -4.6 [-4.9; 7 out] |
| `+ runway saturation (without plan lags)` against `+ weather (without plan lags)` | **+0.0** [**-0.1**; 2 out] | **-0.2** [**-0.2**; 1 out] | **-0.2** [**-0.3**; 1 out] | -0.3 | +0.2 [**+0.2**; 1 out] | +0.5 | **-0.1** [**-0.1**; 2 out] | **+0.1** | -0.3 | **-0.1** | **+0.2** | -0.5 | **-0.1** [**-0.1**; 7 out] |

Per block, the months it helps by more than the floor, hurts by more than it, or
sits inside it; the last column is the same count over the rows of four hours or
less, a diagnostic:

| block | helps by more than the floor | hurts by more than the floor | months | inside the floor | months | 4 h or less: helps / hurts / inside (diagnostic) |
| --- | ---: | ---: | --- | ---: | --- | ---: |
| `PRU reference` against `airport mean` | 0 of 12 | 12 | Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec | 0 | none | 0 / 12 / 0 |
| `geometric unimpeded` against `PRU reference` | 0 of 12 | 12 | Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec | 0 | none | 0 / 12 / 0 |
| `geometric unimpeded + wake class` against `geometric unimpeded` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `identifiers + time` against `geometric unimpeded + wake class` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `identifiers + time + geometry` against `identifiers + time` | 9 of 12 | 1 | Apr | 2 | Jun, Jul | 9 / 1 / 2 |
| `identifiers + time + geometry + aircraft` against `identifiers + time + geometry` | 11 of 12 | 0 | none | 1 | Feb | 11 / 0 / 1 |
| `+ plan lags` against `identifiers + time + geometry + aircraft` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ runway throughput` against `+ plan lags` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ arrivals` against `+ runway throughput` | 8 of 12 | 1 | Feb | 3 | Jul, Sep, Oct | 9 / 1 / 2 |
| `+ demand` against `+ arrivals` | 7 of 12 | 0 | none | 5 | Feb, Jul, Sep, Oct, Nov | 8 / 0 / 4 |
| `+ configuration` against `+ demand` | 5 of 12 | 0 | none | 7 | Jan, Feb, Mar, May, Jun, Aug, Nov | 4 / 1 / 7 |
| `+ shared corridor` against `+ configuration` | 1 of 12 | 0 | none | 11 | Jan, Feb, Mar, Apr, Jun, Jul, Aug, Sep, Oct, Nov, Dec | 1 / 0 / 11 |
| `+ weather` against `+ shared corridor` | 9 of 12 | 0 | none | 3 | May, Jul, Oct | 9 / 0 / 3 |
| `+ shrunk encoding` against `+ weather` | 7 of 12 | 1 | Jul | 4 | Jan, Feb, Sep, Nov | 7 / 0 / 5 |
| `+ shrunk encoding, replacing the stand` against `+ weather` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ shrunk encoding, replacing stand and runway` against `+ weather` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ stand occupancy` against `+ shrunk encoding, replacing the stand` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ stand occupancy, schedule-anchored` (diagnostic) against `+ shrunk encoding, replacing the stand` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ callsign prefix` against `+ stand occupancy` | 11 of 12 | 0 | none | 1 | Feb | 11 / 0 / 1 |
| `+ destination and holidays` against `+ callsign prefix` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ plan differences` against `+ destination and holidays` | 11 of 12 | 0 | none | 1 | Feb | 11 / 0 / 1 |
| `+ runway saturation` against `+ weather` | 4 of 12 | 0 | none | 8 | Jan, Feb, May, Jun, Aug, Sep, Oct, Nov | 4 / 0 / 8 |
| `+ runway throughput (without plan lags)` against `identifiers + time + geometry + aircraft` | 11 of 12 | 0 | none | 1 | Feb | 11 / 0 / 1 |
| `+ arrivals (without plan lags)` against `+ runway throughput (without plan lags)` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ demand (without plan lags)` against `+ arrivals (without plan lags)` | 12 of 12 | 0 | none | 0 | none | 12 / 0 / 0 |
| `+ configuration (without plan lags)` against `+ demand (without plan lags)` | 6 of 12 | 1 | Jan | 5 | Feb, Jul, Aug, Sep, Oct | 6 / 1 / 5 |
| `+ shared corridor (without plan lags)` against `+ configuration (without plan lags)` | 1 of 12 | 5 | Apr, May, Aug, Sep, Dec | 6 | Feb, Mar, Jun, Jul, Oct, Nov | 1 / 5 / 6 |
| `+ weather (without plan lags)` against `+ shared corridor (without plan lags)` | 11 of 12 | 0 | none | 1 | Jul | 11 / 0 / 1 |
| `+ runway saturation (without plan lags)` against `+ weather (without plan lags)` | 3 of 12 | 2 | May, Jun | 7 | Jan, Feb, Mar, Jul, Aug, Oct, Nov | 3 / 1 / 8 |

Sources: `reports/monthly_seeds.md`, *Each block's months, against each month's
floor*.

## Annex C: the shipped model per airport and month, the bias

The five-seed average's bias per airport and month, prediction minus recorded
taxi-out over the rows the term leaves to the model, in seconds; positive means
the model predicts too long. The RMSE of the same cells is the table of §8.3. A
bracket is the diagnostic over the rows of four hours or less, with the count it
leaves out.

| ICAO | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EDDF | -9 | +6 | +9 | -10 | -17 | -7 | +8 | +9 | +6 | -3 | -14 | +16 | -1 |
| EDDM | +1 | -1 | -2 | -3 | -12 | +9 | -8 | +18 | -5 | +8 | -24 | +12 | -1 |
| EGLL | +18 | -13 | +2 | +1 | +2 | -6 | +13 | -1 | +15 | +2 | -18 | +20 | +3 |
| EHAM | -5 | +12 | -12 | -2 | -9 | +0 | +6 | +6 | +7 | +16 | -24 | +14 | +1 |
| LEBL | +0 | +1 | -10 | +8 | -9 | +14 | +1 | +2 | -1 | +8 | +8 | -1 | +2 |
| LEMD | +8 | -1 | -25 | +17 | +1 | -4 | +14 | -5 | -4 | +10 | +11 | +4 | +2 |
| LFPG | +21 [+28; 2 out] | +8 | -11 [-10; 1 out] | +3 | -20 | +11 | +6 [+6; 1 out] | +10 | +5 | +9 | -24 | +5 | +2 [+3; 4 out] |
| LIRF | +1 | +3 | -1 | +2 | -16 | -5 | -2 [+3; 1 out] | +1 | +6 | -15 | +1 | -7 | -3 [-2; 1 out] |
| LSZH | -4 | +7 [+9; 1 out] | -12 | +5 | -19 [-12; 1 out] | +1 | +3 | +3 | +3 | +17 | -17 | +16 | +0 [+1; 2 out] |
| LTFM | +5 | -92 | -29 | +12 | -1 | +3 | -18 | +18 | +10 | +2 | -8 | +12 | -5 |
| all 10 | +5 [+6; 2 out] | -10 [-9; 1 out] | -10 [-10; 1 out] | +3 | -10 [-9; 1 out] | +1 | +2 [+3; 2 out] | +7 | +5 | +5 | -11 | +10 | +0 [+0; 7 out] |

Sources: `reports/monthly_seeds.md`, *Per airport and month: the five-seed
average*.

## Annex D: the geometry claim per airport and month, and the shipped rung's gap

The claim of §7.2 and §8.5, the fold-mean difference in differences of the
geometry block on held-out against seen pairs, restricted to each airport's rows
in each month: the point estimate in seconds, with a note where its 95%
pair-clustered interval does not clear zero, and the year's figure per airport,
which is `reports/baseline.md`'s per-airport claim. The intervals, the rows and
the seen and held-out figures of every cell are in the report's own per-airport
tables.

| airport | Jan | Feb | Mar | Apr | May | Jun | Jul | Aug | Sep | Oct | Nov | Dec | year |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frankfurt | +29.7 | +32.8 | +31.6 | +34.6 | +31.3 | +33.5 | +31.9 | +34.1 | +31.1 | +31.4 | +24.7 | +24.4 | **+30.7** |
| Munich | +9.7 | +9.4 | +10.5 | +10.3 | +13.6 | +10.6 | +11.9 | +13.1 | +11.7 | +13.4 | +9.4 | +8.2 | **+10.8** |
| Heathrow | +13.9 | +16.5 | +12.5 | +17.0 | +15.1 | +17.9 | +16.3 | +15.5 | +15.7 | +18.1 | +18.6 | +19.8 | **+16.4** |
| Schiphol | +5.3 (not clear of zero) | +6.0 | +9.0 | +9.4 | +11.3 | +11.0 | +11.5 | +8.6 | +10.2 | +11.9 | +10.9 | +6.0 | **+9.0** |
| Barcelona | +5.8 (not clear of zero) | +9.1 | +12.9 | +10.2 | +8.7 | +8.3 | +8.3 | +8.8 | +12.2 | +9.4 | +11.9 | +6.4 (not clear of zero) | **+9.3** |
| Madrid | +18.6 | +24.1 | +16.1 | +20.8 | +17.4 | +16.2 | +14.9 | +23.0 | +26.0 | +19.3 | +15.4 | +19.5 | **+19.2** |
| Paris | +17.9 | +18.8 | +20.2 | +24.6 | +20.8 | +29.6 | +41.4 | +29.9 | +25.6 | +25.3 | +31.9 | +28.4 | **+24.2** |
| Fiumicino | +29.4 | +36.9 | +29.6 | +26.3 | +22.9 | +21.3 | +13.0 | +20.7 | +23.6 | +23.5 | +26.0 | +27.1 | **+22.5** |
| Zurich | +19.8 | +23.1 | +19.7 | +21.7 | +9.3 | +21.7 | +19.8 | +22.6 | +24.2 | +24.9 | +23.2 | +25.7 | **+15.8** |
| Istanbul | +31.5 | +20.7 | +22.0 | +24.6 | +21.8 | +28.1 | +19.6 | +26.8 | +30.2 | +23.8 | +25.2 | +30.9 | **+24.9** |

Per airport, the months whose claim clears zero:

| ICAO | months clear of zero | months not clear | months printed as counts |
| --- | ---: | --- | --- |
| EDDF | 12 | none | none |
| EDDM | 12 | none | none |
| EGLL | 12 | none | none |
| EHAM | 11 | Jan | none |
| LEBL | 10 | Jan, Dec | none |
| LEMD | 12 | none | none |
| LFPG | 12 | none | none |
| LIRF | 12 | none | none |
| LSZH | 12 | none | none |
| LTFM | 12 | none | none |

The shipped rung, `+ plan differences`, held out minus seen, per month with its
interval: how much worse the model that ships reads on a pair it has never seen
than on one it has. Context, not a claim; the bracketed figures are over the
rows of four hours or less, a diagnostic.

| month | seen (s) [4 h or less] | held out (s) [4 h or less] | gap [95%] | gap over the rows of 4 h or less [95%] (diagnostic) |
| --- | ---: | ---: | ---: | ---: |
| Jan | 221.3 [221.3] | 327.6 [237.4] | +106.3 s [+13.2, +199.9] | +16.1 s [+7.1, +25.3] |
| Feb | 213.5 [213.5] | 245.1 [240.7] | +31.6 s [+18.3, +46.4] | +27.2 s [+16.1, +38.9] |
| Mar | 186.0 [186.0] | 209.6 [205.8] | +23.6 s [+14.5, +34.2] | +19.7 s [+12.3, +26.8] |
| Apr | 186.5 [186.5] | 202.9 [202.9] | +16.4 s [+3.0, +27.2] | +16.4 s [+3.0, +27.2] |
| May | 187.0 [187.0] | 275.2 [210.1] | +88.3 s [+16.6, +168.8] | +23.1 s [+15.3, +29.5] |
| Jun | 224.3 [224.3] | 231.1 [231.1] | +6.8 s [-8.1, +19.5] | +6.8 s [-8.1, +19.5] |
| Jul | 237.9 [237.9] | 309.4 [259.8] | +71.5 s [+13.5, +143.8] | +21.8 s [+10.3, +32.9] |
| Aug | 239.8 [239.8] | 231.1 [231.1] | -8.7 s [-27.9, +10.4] | -8.7 s [-27.9, +10.4] |
| Sep | 216.6 [216.6] | 225.9 [225.9] | +9.3 s [-1.3, +19.5] | +9.3 s [-1.3, +19.5] |
| Oct | 234.9 [234.9] | 223.8 [223.8] | -11.1 s [-36.1, +12.3] | -11.1 s [-36.1, +12.3] |
| Nov | 212.9 [212.9] | 220.1 [220.1] | +7.2 s [-4.2, +17.7] | +7.2 s [-4.2, +17.7] |
| Dec | 208.2 [208.2] | 223.3 [223.3] | +15.0 s [+6.5, +23.4] | +15.0 s [+6.5, +23.4] |
| **the year** | 215.5 [215.5] | 250.7 [226.6] | +35.1 s [+13.8, +57.0] | +11.0 s [+5.9, +16.3] |

Sources: `reports/monthly_pairs.md`, *Per airport and month* and *The shipped
rung's gap*.

## Annex E: the year's weather against its ten-year norm

Context only: not a weight, not a feature, and in no figure of the model. For
each airport and calendar month, the share of hours in each of the model's three
regimes in 2025 (freezing, low visibility, precipitation, as `taxiout.metar`
flags them) was set against the mean and the range of the same share over 2015
to 2024 from the Iowa Environmental Mesonet archive; Istanbul's station reports
from 2018 only. The months of 2025 that fall outside their ten-year range, per
airport:

| airport | 2025 months outside the 2015 to 2024 range |
| --- | --- |
| Frankfurt | April (freezing below); July (precipitation above); September (precipitation above); November (freezing above); December (precipitation below) |
| Munich | February (precipitation below); July (precipitation above); December (precipitation below) |
| Heathrow | March (precipitation below) |
| Schiphol | January (low visibility above); March (precipitation below); October (precipitation above) |
| Barcelona | March (low visibility above, precipitation above); May (freezing above, low visibility above); June (precipitation below); July (precipitation above); December (precipitation above) |
| Madrid | January (precipitation above); March (precipitation above); May (low visibility above) |
| Paris | February (low visibility above); March (precipitation below); November (freezing above) |
| Fiumicino | June (precipitation below); December (low visibility above) |
| Zurich | November (freezing above) |
| Istanbul | January (freezing below); February (freezing above, precipitation above); March (low visibility above); April (freezing above, low visibility above); June (precipitation below); July (low visibility above, precipitation below) |

The full tables, one per airport with the 2025 share, the ten-year mean and the
range for every month, are the report's Annex E. Istanbul's February, the one
airport-month of §8.3 above 500 s that no single row explains, is above its
range on freezing and on precipitation; Annex G reads it day by day. The two
test months are not in these tables, but §7.3 gives their regime shares against
2025.

Sources: `reports/monthly_ladder.md`, *Annex E: the historic weather context*.

## Annex F: the rows over four hours

Every validation row of the held-out year whose recorded taxi-out exceeds four
hours and which the record-quality term does not cover: out of every fit,
counted by the scorer. For each, the recorded value, the shipped rung's
prediction at the ladder's seed, the residual, its share of the squared error of
its airport-month cell and of its whole month (both excluding the term's rows),
and whether it has a flight-plan record. No movement or stand identifier is
printed. These seven rows are the whole difference between the first and the
second reading of every figure in §8.

| ICAO | month | recorded (s) | predicted (s) | residual (s) | share of the cell's squared error | share of the month's | flight-plan record |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| LFPG | Jan | 58,206 | 1,128 | +57,078 | 27.7% | 17.6% | no |
| LFPG | Jan | 84,240 | 901 | +83,339 | 59.1% | 37.4% | no |
| LFPG | Mar | 21,308 | 5,038 | +16,270 | 24.6% | 4.0% | yes |
| LFPG | Jul | 15,059 | 8,918 | +6,141 | 2.7% | 0.2% | yes |
| LIRF | Jul | 87,002 | 6,743 | +80,259 | 69.3% | 34.6% | yes |
| LSZH | Feb | 18,409 | 1,008 | +17,401 | 50.7% | 2.9% | no |
| LSZH | May | 87,341 | 208 | +87,133 | 96.1% | 51.5% | no |

Per airport over the year:

| ICAO | rows over 4 h, the year |
| --- | ---: |
| EDDF | 0 |
| EDDM | 0 |
| EGLL | 0 |
| EHAM | 0 |
| LEBL | 0 |
| LEMD | 0 |
| LFPG | 4 |
| LIRF | 1 |
| LSZH | 2 |
| LTFM | 0 |
| all | 7 |

Three of the seven have a flight-plan record, and two of those are the long
departures the rule of §4 was written on; the four without a record take off
within minutes of schedule with a recorded block time about a day earlier, and
nothing the ranking set exposes marks them.

Sources: `reports/monthly_ladder.md`, *Annex F: the rows over four hours*;
`reports/planshift.md`, *The class in the training year*.

## Annex G: Istanbul's February 2025, and its January 2026

The one airport-month of the held-out year above 500 s that no single row
explains (§8.3), read day by day in the report beside the same days of the other
eleven months, with the airport's hours in each regime, the mean plan lag and
the arrivals' control; and the ranking month January 2026 at the same airport,
for which no error exists and nothing is predicted, read the same way. Here,
what the report concludes.

February at LTFM pools to 520.8 s over 18,328 departures. 7 of its 28 days read
above that figure (6, 11, 19, 20, 22, 23, 24) and 21 at or below it. Rows over 4
h among them: 0, so the second reading over the rows of 4 h or less is the same
figure.

The regime shares of the two calendar months at the airport, against the history
fetched for Annex E (no reports before 2018 at this airport):

| month | history years | freezing, history | freezing, cache | low visibility, history | low visibility, cache | precipitation, history | precipitation, cache |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| January | 6 (2019, 2020, 2021, 2022, 2023, 2024) | 0.3% to 26.2% | 2025 0.0%; 2026 14.8% | 0.0% to 2.7% | 2025 0.0%; 2026 1.1% | 16.9% to 25.4% | 2025 17.5%; 2026 32.1% |
| February | 6 (2019, 2020, 2021, 2022, 2023, 2024) | 0.0% to 16.7% | 2025 22.6% | 0.1% to 6.4% | 2025 2.5% | 9.1% to 23.5% | 2025 39.1% |

January 2026's days against the February 2025 days above and at or below the
month's figure, on freezing hours and on the mean plan lag:

| February 2025 days | their freezing hours, range | January 2026 days within it | their plan-lag mean, range (s) | January 2026 days within it | within both |
| --- | ---: | ---: | ---: | ---: | ---: |
| February days whose RMSE is above the month's 520.8 s (7) | 10 to 20 | 3 of 31 | +1,454 to +2,486 | 10 of 31 | 3 of 31 |
| February days whose RMSE is at or below the month's 520.8 s (21) | 0 to 18 | 29 of 31 | +1,003 to +1,394 | 10 of 31 | 10 of 31 |

In the report's words: this is descriptive and decides nothing. The ranges come
from 28 days of one airport-month, the error on a day is not its weather or its
plan lag, and no figure here says what January 2026 will score. What it shows is
the shape of the exposure §7.3 bounds: a winter month at this airport that froze
more than any in its record was the model's worst ordinary month, and the
ranking January sits inside the ordinary February days on freezing hours and
divides on the plan lag.

Sources: `reports/istanbul.md`, *LTFM, February 2025, day by day* and *January
2026 against February 2025's days*.

## Annex H: where the last three blocks of the chain earn their keep

The three blocks above the stand occupancy, read against the rung below each,
over the held-out year: their change in RMSE over all ten airports and at
Fiumicino alone, and where each block's year-level reduction in squared error
sits, by airport, by flight-plan record, by plan-lag band and by airline
designator. The report carries every table; here the year-level figures and its
three computed sentences.

| block | airports | here (s) [4 h or less] | `monthly_ladder.md` (s) | same |
| --- | --- | ---: | ---: | --- |
| `+ callsign prefix` against `+ stand occupancy` | all 10 | -3.3 [-3.4] | -3.3 [-3.4] | yes |
| `+ callsign prefix` against `+ stand occupancy` | LIRF | -22.1 [-23.5] | -22.1 [-23.5] | yes |
| `+ destination and holidays` against `+ callsign prefix` | all 10 | -1.4 [-1.7] | -1.4 [-1.7] | yes |
| `+ destination and holidays` against `+ callsign prefix` | LIRF | -6.5 [-8.8] | -6.5 [-8.8] | yes |
| `+ plan differences` against `+ destination and holidays` | all 10 | -2.2 [-2.0] | -2.2 [-2.0] | yes |
| `+ plan differences` against `+ destination and holidays` | LIRF | -6.2 [-4.4] | -6.2 [-4.4] | yes |

**`+ callsign prefix` against `+ stand occupancy`: 86.0% of the year-level
reduction sits at LIRF** [85.2% over the rows of 4 h or less, a diagnostic, as
is every bracket in this sentence]; the next airport is EGLL at 7.0% [7.5%].
Within LIRF, 100.0% [100.0%] of its change is on rows with a flight-plan record,
42.1% [45.5%] in the 15-30 band, the largest of the bands with 30 rows or more,
and 15.4% [16.6%] on one designator, ISR (30.5% [24.9%] on the top three).

**`+ destination and holidays` against `+ callsign prefix`: 58.3% of the
year-level reduction sits at LIRF** [62.8% over the rows of 4 h or less, a
diagnostic, as is every bracket in this sentence]; the next airport is LTFM at
23.0% [21.5%]. Within LIRF, 100.0% [100.0%] of its change is on rows with a
flight-plan record, 57.2% [49.7%] in the 15-30 band, the largest of the bands
with 30 rows or more, and 43.1% [37.5%] on one designator, EXS (76.3% [79.4%] on
the top three).

**`+ plan differences` against `+ destination and holidays`: 35.4% of the
year-level reduction sits at LIRF** [26.5% over the rows of 4 h or less, a
diagnostic, as is every bracket in this sentence]; the next airport is LFPG at
34.7% [36.4%]. Within LIRF, 100.0% [100.0%] of its change is on rows with a
flight-plan record, 24.2% [39.9%] in the 15-30 band, the largest of the bands
with 30 rows or more, and 43.9% [7.7%] on one designator, AEZ (70.9% [52.0%] on
the top three).

The designators are printed as `reports/residual.md` prints them. What the three
sentences say together: the last three blocks are, over the year, mostly a
Fiumicino correction, all of it on rows that have a flight-plan record, and for
the plan differences much of the Fiumicino share rests on the one July day-shift
row of Annex F, since the share falls from 35.4% to 26.5% when that row is left
out. On the forward splits the same blocks are worth 2.11, 1.79 and 1.14 s in
December and 4.64, 1.49 and 4.55 in July (§7.2).

Sources: `reports/fiumicino.md`, *The held-out-month ladder's year-level
changes, recomputed*, *`+ callsign prefix` against `+ stand occupancy`*, *`+
destination and holidays` against `+ callsign prefix`* and *`+ plan differences`
against `+ destination and holidays`*; `reports/baseline.md`, *The Gate 2c
blocks*.

## Annex I: the plan-shifted departures, evaluated after the rule was fixed

The rule of §4 as the report evaluated it on the stored residuals, after the
pre-registration fixed it and before it was built into the submitted file. The
class in the training year, per departure (no identifiers):

| airport | month | flight-plan record | take-off - EOBT_1 (h) | take-off - LOBT (h) | take-off - schedule (h) | recorded taxi-out (s) | long |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| LIRF | Feb 2025 | yes | 4.58 | 4.60 | 3.85 | 13,865 | no |
| LFPG | Mar 2025 | yes | 5.65 | 5.85 | 5.85 | 21,308 | yes |
| LIRF | Jul 2025 | yes | 24.17 | 24.17 | 24.17 | 87,002 | yes |

`n` = 3, `long` = 2, share = (2 + 1) / (3 + 2) = 0.6000. The rest mean per
airport, over the training departures with a flight-plan record and a recorded
taxi-out of 4 h or less:

| airport | rest mean (s) |
| --- | ---: |
| EDDF | 862.5 |
| EDDM | 810.6 |
| EGLL | 1,363.0 |
| EHAM | 787.6 |
| LEBL | 955.6 |
| LEMD | 1,014.3 |
| LFPG | 1,016.7 |
| LIRF | 1,144.2 |
| LSZH | 741.6 |
| LTFM | 1,049.6 |

One departure of the ranking set is in the class: LIRF, Jul 2026, take-off minus
`LOBT_flt` 86,820 s (24.12 h), minus `EOBT_1_flt` 23.97 h, minus the schedule
26.27 h. The rule fitted on the full year predicts 0.6000 x 86,820 + (1 -
0.6000) x 1,144.2 = **52,549.7 s** (52,550 s after the submission's rounding).
v6 predicts **3,120 s** for it, read from
`data/submissions/funny-firefly_v6.parquet` through the template's identifier,
which is not printed. The difference is +49,430 s. What the departure recorded
is hidden, and nothing here says which is right.

On every leg the term and the rule are fitted on the leg's training rows and
applied to its validation rows; *stored* is the booster's prediction on the
class's row at the ladder's seed. The RMSEs are every row, with the term on its
rows (what the submission scores) and with the booster on them, each without and
with the rule; the bracket is over the rows of four hours or less, a diagnostic
that leaves out exactly the rows the rule is for when they are long.

| leg | rows | fitted n / long / share | class rows | the class's rows (s) |
| --- | ---: | ---: | ---: | --- |
| forward Dec | 165,677 | 3 / 2 / 0.600 | 0 | none |
| forward Jul | 190,713 | 2 / 1 / 0.500 | 1 | LIRF: recorded 87,002, rule 44,062, stored 5,214 |
| held out Jan | 153,706 | 3 / 2 / 0.600 | 0 | none |
| held out Feb | 143,732 | 2 / 2 / 0.750 | 1 | LIRF: recorded 13,865, rule 12,710, stored 5,646 |
| held out Mar | 164,449 | 2 / 1 / 0.500 | 1 | LFPG: recorded 21,308, rule 11,048, stored 5,038 |
| held out Apr | 175,288 | 3 / 2 / 0.600 | 0 | none |
| held out May | 185,202 | 3 / 2 / 0.600 | 0 | none |
| held out Jun | 183,142 | 3 / 2 / 0.600 | 0 | none |
| held out Jul | 190,713 | 2 / 1 / 0.500 | 1 | LIRF: recorded 87,002, rule 44,065, stored 6,743 |
| held out Aug | 191,182 | 3 / 2 / 0.600 | 0 | none |
| held out Sep | 183,950 | 3 / 2 / 0.600 | 0 | none |
| held out Oct | 185,674 | 3 / 2 / 0.600 | 0 | none |
| held out Nov | 162,332 | 3 / 2 / 0.600 | 0 | none |
| held out Dec | 165,677 | 3 / 2 / 0.600 | 0 | none |

| leg | with the term | with the term and the rule | change in squared error | booster only | booster only and the rule | change in squared error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| forward Dec | 234.0 [230.0] | 234.0 [230.0] | +0.0 [+0.0; 6 out] | 364.8 [240.3] | 364.8 [240.3] | +0.0 [+0.0; 6 out] |
| forward Jul | 368.2 [300.3] | 331.9 [300.3] | -4,845.4 [+0.0; 37 out] | 620.5 [293.0] | 599.7 [293.0] | -4,845.4 [+0.0; 37 out] |
| held out Jan | 356.8 [239.9] | 356.8 [239.9] | +0.0 [+0.0; 8 out] | 465.7 [241.1] | 465.7 [241.1] | +0.0 [+0.0; 8 out] |
| held out Feb | 282.2 [274.0] | 281.4 [273.1] | -66.2 [-66.2; 7 out] | 463.1 [282.6] | 462.6 [281.8] | -66.2 [-66.2; 7 out] |
| held out Mar | 216.8 [210.0] | 214.6 [210.0] | -159.4 [+0.0; 5 out] | 317.4 [221.6] | 315.9 [221.6] | -159.4 [+0.0; 5 out] |
| held out Apr | 206.2 [203.3] | 206.2 [203.3] | +0.0 [+0.0; 6 out] | 288.7 [209.5] | 288.7 [209.5] | +0.0 [+0.0; 6 out] |
| held out May | 302.7 [214.2] | 302.7 [214.2] | +0.0 [+0.0; 8 out] | 321.6 [222.4] | 321.6 [222.4] | +0.0 [+0.0; 8 out] |
| held out Jun | 318.9 [248.9] | 318.9 [248.9] | +0.0 [+0.0; 10 out] | 424.3 [253.1] | 424.3 [253.1] | +0.0 [+0.0; 10 out] |
| held out Jul | 356.2 [290.1] | 320.6 [290.1] | -4,598.0 [+0.0; 37 out] | 616.3 [284.9] | 596.5 [284.9] | -4,598.0 [+0.0; 37 out] |
| held out Aug | 274.1 [251.7] | 274.1 [251.7] | +0.0 [+0.0; 15 out] | 395.8 [253.8] | 395.8 [253.8] | +0.0 [+0.0; 15 out] |
| held out Sep | 251.7 [240.0] | 251.7 [240.0] | +0.0 [+0.0; 12 out] | 431.3 [241.3] | 431.3 [241.3] | +0.0 [+0.0; 12 out] |
| held out Oct | 229.5 [229.0] | 229.5 [229.0] | +0.0 [+0.0; 4 out] | 251.1 [234.6] | 251.1 [234.6] | +0.0 [+0.0; 4 out] |
| held out Nov | 230.9 [219.9] | 230.9 [219.9] | +0.0 [+0.0; 4 out] | 391.3 [225.8] | 391.3 [225.8] | +0.0 [+0.0; 4 out] |
| held out Dec | 234.0 [230.0] | 234.0 [230.0] | +0.0 [+0.0; 6 out] | 364.8 [240.3] | 364.8 [240.3] | +0.0 [+0.0; 6 out] |
| **the year, twelve held-out months pooled** | 276.6 [239.0] | 272.4 [238.9] | -4,823.6 [-66.2; 122 out] | 405.4 [243.5] | 402.6 [243.5] | -4,823.6 [-66.2; 122 out] |

In the pre-registration's own words, this evaluation shows what the rule does to
three rows and nothing more: each held-out month that holds one of the three
fits the rule on the other two, so the share it applies was computed from the
very rows the rule was written after seeing. Nothing here settles whether the
rule is right for the 2026 departure.

Sources: `reports/planshift.md`, *The class in the training year*, *The rule
fitted on the full year*, *The one ranking departure*, *Per held-out month and
on the two forward legs* and *What this evaluation can show*.

## Annex J: the three candidates measured after the freeze

Three candidates, each measured once before the freeze and left on the table,
were fitted on top of the shipped rung after it and read under a rule fixed in
writing before any of their figures existed: the pre-registration of 28
September, which named the candidates, the legs, the reference and four
conditions, and that of 3 October, which kept all of those, read every figure
over the rows that neither the term nor the plan-shift rule covers, added a
fifth condition on the twelve held-out months, and replaced the re-freeze its
predecessor had priced with an increment. Both are published in the repository
as dated records. The candidates: the runway-saturation block appended to the
shipped rung's features; row weights halving every twelve months over the
fitting rows; 8,000 trees in place of 6,000. The reference is the shipped rung's
own fit on each leg, which reproduced the ladder's stored December and July
residuals to zero and October's printed figure before anything was read.

A candidate is adopted only if, on the paired contrast against the reference at
the same seed, it is better by more than the floor on December and on July over
all ten airports, and not worse by more than the floor on the three July
airports and on October; a candidate that holds those four is then read on the
twelve held-out months against the year's floor, refitted at a second seed for a
sign check, and ships as an increment on the rung. Every contrast is read at the
precision the ladder's report prints, two decimals; negative is better.

| candidate | read on | reference (s) | candidate (s) | contrast (s) | interval (s) | condition, floor (s) | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| saturation on top of `+ plan differences` | December, all ten | 214.66 | 214.83 | +0.17 | [-0.32, +0.74] | better, 0.23 | **fails** |
| saturation on top of `+ plan differences` | December, the three | 203.46 | 203.16 | -0.30 | [-1.25, +0.69] |  | beside the rule |
| saturation on top of `+ plan differences` | July, all ten | 262.25 | 262.25 | +0.00 | [-0.40, +0.41] | better, 0.54 | **fails** |
| saturation on top of `+ plan differences` | July, the three | 235.11 | 235.05 | -0.06 | [-0.80, +0.72] | not worse, 1.02 | holds |
| saturation on top of `+ plan differences` | October, all ten | 214.44 | 214.50 | +0.06 | [-0.34, +0.50] | not worse, 0.21 | holds |
| the mild recency schedule | December, all ten | 214.66 | 214.71 | +0.05 | [-0.43, +0.54] | better, 0.23 | **fails** |
| the mild recency schedule | December, the three | 203.46 | 203.51 | +0.05 | [-0.80, +0.88] |  | beside the rule |
| the mild recency schedule | July, all ten | 262.25 | 262.50 | +0.25 | [-0.21, +0.77] | better, 0.54 | **fails** |
| the mild recency schedule | July, the three | 235.11 | 236.15 | +1.04 | [+0.06, +2.05] | not worse, 1.02 | **fails** |
| the mild recency schedule | October, all ten | 214.44 | 214.67 | +0.23 | [-0.18, +0.70] | not worse, 0.21 | **fails** |
| more trees | December, all ten | 214.66 | 214.38 | -0.28 | [-0.39, -0.18] | better, 0.23 | holds |
| more trees | December, the three | 203.46 | 203.23 | -0.23 | [-0.40, -0.07] |  | beside the rule |
| more trees | July, all ten | 262.25 | 262.34 | +0.09 | [+0.01, +0.17] | better, 0.54 | **fails** |
| more trees | July, the three | 235.11 | 235.34 | +0.23 | [+0.11, +0.36] | not worse, 1.02 | holds |
| more trees | October, all ten | 214.44 | 214.42 | -0.02 | [-0.10, +0.05] | not worse, 0.21 | holds |

**The rule adopts nothing.** No candidate holds the four conditions, so the
fifth was not reached, no second seed was fitted, and v7 ships unchanged. More
trees is the one candidate that clears December, by 0.28 s against a floor of
0.23, and it fails July by 0.09 s in the wrong direction; the saturation block
is null on both forward legs; the recency schedule fails all four, by 1.04 s on
the three July airports against a floor of 1.02. The report also reads every
condition three other ways, on the unrounded contrast and under two roundings to
one decimal, and under every reading the same candidates hold: none. The twelve
fits took 3,283 s of wall time.

Sources: `reports/candidates-2026-10-03.md`, *What was measured*, *Before
anything was read*, *The four conditions of 28 September*, *What the rule
adopts*, *Whether a verdict turns on the precision* and *Every fit*;
`reports/preregistered-2026-09-28.md` and `reports/preregistered-2026-10-03.md`,
dated records.

## References

From `docs/prior_art.md`, *Sources*.
