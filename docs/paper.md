# Recovering the airport surface: taxi-out prediction from OpenStreetMap geometry

*The base configuration was frozen before the ranking months were scored. The
final configuration adds the changes made after the first score, each described
in §8.7, and §1.4 defines the working terms.*

Team `funny-firefly`. PRC Data Challenge 2026.

The code, the reports and this paper are at https://github.com/hubell/funny-firefly, release V1.1, and archived at [10.5281/zenodo.23320026](https://doi.org/10.5281/zenodo.23320026) (Bell, 2026b). The data they read, apart from the challenge data, are at [10.5281/zenodo.23133051](https://doi.org/10.5281/zenodo.23133051) (Bell, 2026a). Paths in backticks are relative to the repository root.

---

## Abstract

This paper predicts taxi-out time, the interval from off-block to take-off, at
ten European airports in the EUROCONTROL Performance Review Commission's Data
Challenge 2026, which withholds the off-block time. The final model is a
gradient-boosted booster with a second learner; among its inputs are the Network
Manager's recorded off-block time, which the challenge leaves in its data, and a
correction from the ground movements seen in open ADS-B and OPDI data. It has a
root-mean-square error of **244.9 s** over the year when each of the twelve
months of 2025 is held out in turn, and 217.7 s on the rows that none of its
special rules covers. On the four months of 2025 that match the four the
challenge ranks in 2026, each held out the same way, it reads 294.4 s against
306.0 for its booster with the added feature groups alone. Several later changes
were chosen on those same months, so these are held-out estimates from the
training year and not a forecast of the challenge score. The challenge's own
scores on January and July 2026, from the configuration frozen before any score
to the last one uploaded, fall in the order the training year predicted
(§8.7.1). The error is concentrated in a tail of extreme
records: the year holds 122 recorded taxi-outs of over four hours, and all but 5
of them lie on rows covered by the record-quality term or the plan-shift rule.

The model rests on one hypothesis: that taxiway geometry recovered from
OpenStreetMap carries to stand and runway pairs a model has never seen, where a
memorized pair mean has nothing to carry. Taxi-out is decomposed into an
unimpeded time routed on the recovered surface, an excess time drawn from the
traffic and weather around a departure, and a record-quality term for rows whose
recorded off-block time was defaulted to the schedule. The routing graph is
verified on a worked airport where every distance is known by hand. The geometry
lowers the error by **18.5 s [16.7, 20.5]** more on held-out pairs than on seen
pairs, in each of the twelve months read separately. It rests on a taxi speed of
11.8 to 13.2 m/s, recovered ten times with a different airport left out each
time, and it transfers to an airport the model has never seen. It matters most
where memory fails: on pairs seen a hundred times or fewer, which hold just over
a tenth of the 2026 test rows, falling back on the geometry rather than the
airport mean is better by some twenty seconds.

Two other findings qualify this. The strongest single feature is not geometric
but the flight plan's own delay signal, worth 62.9 s over the year, and in each
of the twelve months. And a correction built from the times aircraft are first seen
moving, in open ground-movement data, is the largest later gain, 7.7 s over the
year, where a receiver covers the airport. Once the plan's delay signal is in,
an oracle diagnostic bounds what a queueing model could add at about eight
seconds.

## 1. Introduction

### 1.1 The problem, and the form the challenge gives it

Taxi-out is the interval from off-block to take-off. The challenge asks for it
in a form that inverts the operational one: the take-off time is given and the
off-block time is withheld. This makes it a post-hoc estimation task, as the challenge's rationale page states, naming post-operations analysis, "excess fuel
burnt/CO2 production compared to unconstrained status", as the use. An
operational forecast would anchor only on schedules and filed estimates; that
alternative is discussed in §6.5 and not pursued here, because it answers a
different question.

Source: `docs/modelling_notes.md`, *Leakage watch*; the rationale page.

### 1.2 What I set out to test

Every departure in the challenge data carries a stand and a runway, and the
natural model for a problem of this shape learns what each stand and runway pair has
cost before. That works on the pairs a model has seen. It cannot work on a pair
it has not, and a test month always holds some: just over a tenth of the 2026
test rows sit on pairs the training year saw a hundred times or fewer.

My central hypothesis is that the physical structure the challenge data omits,
the route an aircraft taxis from its stand to its runway, can be recovered from
OpenStreetMap and carries over to stand and runway pairs a model has never seen,
where a memorized pair mean has nothing to carry. The criterion was fixed in the project plan, before the first model was fitted: the geometry is
judged on pairs held out of training, not on a forward split. The test is
therefore a difference in differences (Angrist & Pischke, 2009, chap. 5): how much more the geometry block is worth
on held-out pairs than on pairs the model has seen.

The hypothesis holds. The geometry block's held-out-pair contrast is **+18.5 s
[+16.7, +20.5]**: it lowers the error by that much more on held-out pairs than on
seen ones, and the contrast is
positive, with an interval that excludes zero, in every one of the twelve
months of the training year read separately (§8.5). On the December forward
split the same block lowers the error by 2.27 s, and on July it moves it by
less than the seed floor (§3.6). It is worth acknowledging that this gap is the
finding rather than a disappointment: most test rows sit on pairs the training
year saw often, where the identifiers already carry what the geometry knows, so
a forward split cannot see what the geometry is for (§7.2, §9.1).

The rest of the model was built around that claim, and each part entered
through a test whose criterion was written down before its figures existed. Two
results shape the paper as much as the geometry does. First, much of the
target's heavy tail is not taxiing at all but records whose off-block time was
defaulted to the schedule. Those rows are treated by a separate, stated term
rather than left for the model to learn (§4). Secondly, the excess term is
small once the flight plan's own delay signal is in the model: 7.7 s in
December and 5.8 in July against that signal's 79.3 and 57.4, or 62.9 over the year (§8.4). The queue layer
that would model it further is bounded to about eight seconds by a diagnostic
that needs no model (§6.3, §6.4).

The tests under the central hypothesis, each with the record that fixed its
criterion:

| what was tested | criterion fixed in | outcome | where |
| --- | --- | --- | --- |
| congestion from the fields the ranking set exposes, and a queue model only if congestion proved binding | the project plan | real and small; the queue layer bounded and not built | §6 |
| a record-quality term, without which no RMSE means anything | the project plan | supported on the training year | §4 |
| seed averaging, recency weighting, season columns | the pre-registration of the seed, recency and season experiments | seed averaging adopted; the other two not | §7.1.1, §7.1.2 |
| three model changes left untested at the freeze | the pre-registrations of the three candidates | none adopted | Annex J |
| a rule for the plan-shifted departures | the pre-registration of the plan-shift rule | fixed and applied | §4, Annex I |
| changes made after the first challenge score, each under a pre-registration | the pre-registrations listed in §8.7 | adopted in part; each outcome and its reason in §8.7 | §8.7, §8.8 |

Analyses with no criterion fixed in advance, the held-out year and the annexes
built on it among them, are reported as analyses and never as tests. One test
was decided on a score rather than on the training year, and that score refuted
it; §8.7.2 states it as the one exception.

### 1.3 Prior art, and what is new here

Taxi-out has been modeled as a queue (Idris et al., 2002; Simaiakis &
Balakrishnan, 2016) and predicted by regression and machine-learning models
fitted to departure records (Lee et al., 2016; Ravizza et al., 2014). Building a
routable graph from OSM `aeroway` geometry is not new; `traffic` (Olive, 2019)
ships one, and Olive et al. (2025) document the same defects this pipeline
repairs. Ravizza et al. (2013) established that
route shape (turn angle, junction-free running) predicts taxi speed nearly as
well as distance. What appears not to have been done is the downstream half: using these recovered
quantities as regression features for taxi-out.

Source: `docs/prior_art.md`.

### 1.4 Terms used in this paper

The paper keeps a small working vocabulary. Each term is defined where it first
matters, and collected here.

| term | meaning | where |
| --- | --- | --- |
| RMSE | the root-mean-square error in seconds, the error measure used throughout | throughout |
| the base configuration | the booster and its rules as fixed before the ranking months were scored, on which every later change builds | preface, §8.1 |
| the final configuration | the submission over the four ranked months: the base configuration with the feature groups, learners, ground-movement correction and flight-plan window of §8.7 | §8.7 |
| pre-registration | a dated record, fixed before the figures it governs, that names a test and the rule deciding it | §3.7 |
| block | a group of features added to the booster together, such as the geometry block | §3.2 |
| ablation ladder, rung | the sequence of models that each add one block to the one before; a rung is one model on it | §3.1 |
| forward split, leg | a split that trains on earlier months and scores a later one, December 2025 or July 2025; each such reading is a leg | §3.5 |
| the three | Frankfurt, Heathrow and Schiphol, read on their own in July as a second July reading | §2.1, §7.1 |
| held-out pairs | stand-to-runway pairs removed from training as whole pairs, so the model scores them unseen | §3.5, §7.2 |
| pair-frequency bucket | the rows grouped by how often the training year saw their stand-to-runway pair, from never to over a hundred times | §5.4, §7.4 |
| ranking mix | a figure reweighted to the ranking set's own share of rows in each pair-frequency bucket | §7.4 |
| seed floor | how far a figure moves when nothing but the random seed changes; a difference inside it is noise | §3.6 |
| the record-quality term | a rule, not a fitted model, for rows whose recorded off-block time was defaulted to the schedule; its rows are grouped in cells by airport, presence of a flight-plan record and band of plan lag | §3.3, §4 |
| the plan-shift rule | the second rule of §3.3, for departures whose take-off falls far after the plan's off-block time | §3.3, §4 |
| ex-flagged | read on the rows the record-quality term does not take; "every row" includes them | §6.3, §8 |
| core rows | the rows neither rule takes | §8.8 |
| arm, control | the variants a test compares, and the one the others are compared against | §7.4 |
| the ground-movement correction | the second learner of the final configuration that corrects the prediction from adsb.lol and OPDI ground events | §8.7 |

**Signs.** A change is the figure after a step less the figure before it, in
seconds of RMSE, so a negative change means the step lowered the error. The
held-out-pair contrast is the other way round: positive means the block helps
more on unseen pairs than on seen ones. Each table that mixes the two says
which it reports.

**Airports.** The ten airports are named by city in the text and by ICAO code
in some tables: EDDF Frankfurt, EDDM Munich, EGLL London Heathrow, EHAM
Amsterdam Schiphol, LEBL Barcelona, LEMD Madrid, LFPG Paris Charles de Gaulle,
LIRF Rome Fiumicino, LSZH Zurich, LTFM Istanbul.
## 2. Data

### 2.1 What the challenge supplies, and what it withholds

The challenge scores ten airports, the ten its data page lists, not the eleven that this
work's code and data deposit carry. The eleventh, Antalya (LTAI), has no rows
in the challenge data and is not scored. January
2026 is a census of all ten, 152,719 departures, and July 2026 of all ten,
192,122: 344,841 rows to predict. The twelve training files cover 2025. July
2026 is also read at Frankfurt, Heathrow and Schiphol alone, as a second July
reading, *the three*, which the validation design of §7.1 reads on its own. The
ranking file of 2 September held July 2026 at those three airports only; the
organizers' re-issue of 4 September added the other seven, and the validation
design kept the three because it had been fixed to read them on their own. The
ranking set blanks `BLOCK_TIME_UTC_mvt` for *every* departure, not only the one
being predicted, which rules out any feature counting other departures'
pushbacks. `AOBT_3_flt`, which the data page defines as the Actual Off-Block
Time (AOBT) for the flown (M3) trajectory, the Network Manager's recorded
off-block time, is not
blanked; the base configuration left it out although the ranking set exposes it, and
the feature group of §8.7.3 admitted it (§6.3, §8.7).

Sources: `reports/data_review.md`, *Airports and months* and *Inventory*;
`reports/ranking-refresh-2026-09-28.md`, a dated record, *July 2026 gained
seven airports*; `docs/modelling_notes.md`, *What the challenge data turned out
to contain*.

### 2.2 External data, and the license position

OpenStreetMap (data (c) OpenStreetMap contributors, ODbL 1.0; OpenStreetMap
contributors, n.d.), OurAirports (public domain; OurAirports, n.d.), Iowa
Environmental Mesonet METAR (public domain; Iowa State University, n.d.), and,
for the ground-movement correction of §8.7 alone, the adsb.lol globe history
(ODbL 1.0; adsb.lol, n.d.). Each is openly licensed, documented, and re-fetchable
by a scripted, cached client. The prize rules (EUROCONTROL Performance Review
Commission, n.d.) set two conditions on data: every external dataset used must be
"openly accessible/usable and documented", and every additional dataset must be
"openly available under an open source license". This entry reads the second as
governing the datasets it adds, which the deposit carries under ODbL (§2.3), and
a dataset a reader cannot re-fetch fails the reproduction clause. The
ground-movement correction also reads EUROCONTROL's OPDI flight events
(EUROCONTROL, n.d.), an external dataset under the first condition: EUROCONTROL
publishes it openly, it is documented here, and the organizers confirmed that it
is open to the challenge. Its site states no data license, so no OPDI file, and
no table built from one, is tracked or deposited; the events are fetched again
with a scripted client like the others. EUROCONTROL's daily ATFM (Air Traffic
Flow Management) delay by airport and cause was considered and not used: its
portal's notice permits non-commercial copying and forbids modification without
prior written permission, which this entry read as failing the first condition
for a dataset that features are built from, and the organizers were not asked
about it. OpenSky (Schäfer et al., 2014) historical surface tracks were
considered and ruled out for the model, because research-use access is not an
open license. OpenSky's ten published
scientific datasets were reviewed against the same license test plus two more: 2025 ground coverage at the ten airports, and a mechanism
the exposed fields lack. None passes all three, and the README records each
verdict.

Source: `docs/prior_art.md`, *External data: license positions*; README.

### 2.3 Reproducibility and the ODbL boundary

The repository ships under GPLv3 in its entirety. The distance matrices it
generates are a Derivative Database of OSM and carry ODbL, so they are
regenerated rather than committed. The Zenodo deposit
[10.5281/zenodo.23133051](https://doi.org/10.5281/zenodo.23133051) carries them
under ODbL, with the eleven Overpass (Olbricht, n.d.) answers they derive from. Beside them it
holds the two public-domain inputs this work read and a later fetch would not
return, the METAR observations and the OurAirports tables, and, also under
ODbL, the adsb.lol event tables the ground-movement correction reads. The
reports are Produced Works and carry the figures. `make licence-check` enforces
the boundary, including that no literal stand identifier from either source
appears in prose.

Source: `data/processed/taxi_distance/ODbL-NOTICE.md`; README, *Data sources*.

## 3. Method

This section states the whole method in one place and in order, so that a
reader can see what a prediction is a function of before the sections that
argue for each part. Nothing in it is new to the paper; §4 to §7 carry the
measurements and the reasons.

### 3.1 The decomposition, and the pipeline in order

    taxi-out = unimpeded(geometry) + excess(congestion, regime) + record-quality term

The decomposition maps onto the challenge's own framing of excess over
unconstrained status. The ablation ladder is this decomposition read bottom-up:
each rung adds one term, and the held-out-pair split measures the one claim that
distinguishes this work. `scripts/make_submission.py` produces the predictions
in these steps, in this order:

1. **The frame.** Every departure of the twelve training files and of the
   ranking file is joined to its stand-to-runway route on the OpenStreetMap
   graph of its airport (§5.1), to the METAR regime of its airport-hour, to its
   destination's bearing and distance from OurAirports, and to the windows of
   other movements around it that the ranking set exposes (§6.1). The same code
   builds both sides, so a feature reads nothing on the ranking side that it
   does not read on the training side.
2. **The record-quality term.** It is fitted on the training rows alone. For
   each cell of airport, whether a flight-plan record exists, and band of
   schedule lag (take-off less scheduled time), it takes the share of rows whose
   recorded taxi-out equals that lag, kept where the cell has at least five rows
   and the share at least 5% (§4). Its rows are taken out of every fit below and
   predicted by the term.
3. **The fitting rows.** These are the training rows the term does not cover,
   with every recorded taxi-out over four hours left out of the fit and never
   out of the scoring. No aircraft taxis for four hours, so such a row is a
   faulty record, such as a day-shift, in a cell the term does not cover; it
   can be left out of the fit only, because the ranking set hides the
   taxi-out. Two thresholds, four hours and two, were fixed before the run.
   Four hours was adopted: against a fit with nothing left out, it moves the
   RMSE by −4.51 s on December and −3.68 s on July. Two hours failed the rule
   fixed with them: it moves July by +2.26 s, and Paris by +14.2 s and
   Fiumicino by +23.3 s on July (`reports/baseline.md`, *The recorded-taxi
   tail, out of the fit*).
4. **The shrunk encoding.** The (airport, stand, runway) pair's out-of-fold
   training mean, shrunk toward the geometric prediction, is computed once over
   the fitting rows and overlaid as a feature in place of the stand categorical
   (§7.4).
5. **The booster.** One gradient-boosted model is fitted on the fitting rows,
   with the feature blocks of §3.2 and the parameters of §3.4, five times under
   consecutive random seeds, and the five predictions are averaged.
6. **The term, substituted.** On the rows the term covers, its prediction
   replaces the booster's.
7. **The plan-shifted departures.** On the rows of one further class, a
   departure with a flight-plan record whose take-off falls more than four hours
   after both of the plan's off-block estimates, a rule of the term's own form
   replaces the booster's prediction (§4, *The plan-shifted departures*). In the
   ranking set it is one row.
8. **Rounding and the clip at zero**, and the output written in the challenge
   template's own order and integer type.

The final model adds three things to these steps, each read in §8.7 and §8.8.
The booster carries two further feature blocks and is blended with a second
learner, a CatBoost model (Prokhorenkova et al., 2018), with the weight of the blend chosen per airport; the
term's rows of step 6 are predicted row by row. A second model, the
ground-movement correction, reads adsb.lol and OPDI ground events and replaces
the prediction on the rows those sources cover, at the airports where it helped
out of fold. And a clip to the window within which a recorded off-block time lay
from its flight plan's in 2025 is applied last, after every other step.
`taxiout.models` records the basis of each.

Sources: `docs/modelling_notes.md`, *The model, stated as a decomposition*;
`reports/tails.md`, *The cells* and *The prediction*; `reports/baseline.md`,
*The recorded-taxi tail, out of the fit*; `src/taxiout/models.py`,
`fit_rows`; `scripts/make_submission.py`.

### 3.2 What the model reads

The table lists the booster's feature blocks in the order the ladder adds them.
Each block's rung key is the name the reports use; *what it reads* is the raw
fields or external data behind it. The last column is where that data comes
from, which is what the rule of §6.1 governs (a feature may read what the
ranking set exposes for the rows it reads from).

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
| destination and holidays | `destination` | the destination's bearing and distance; the public-holiday calendar (Vacanza Team, n.d.) | OurAirports; the holiday tables |
| plan differences (ships) | `plan-diffs` | the plan's own off-block values against each other | the Network Manager record, exposed |

The ladder also carries rungs the submission is not fitted from: two reference
rungs that fit no booster (the airport mean, and the unimpeded-time reference
of EUROCONTROL's Performance Review Unit), the geometric unimpeded term alone
and with the wake class, two further versions of the shrunk identifier encoding
of §7.4, a runway-saturation block, a schedule-anchored reading of the
stand-occupancy block, and the congestion chain fitted without the plan lags.
§7.2 and Annex A read them.

Sources: `reports/baseline.md`, *What is fitted, and on what* and *The ablation
ladder*; `reports/congestion.md`, *The leakage audit*; `src/taxiout/models.py`,
the rung table.

### 3.3 Two terms that bypass the booster

The record-quality term and the plan-shift rule are rules, not fitted models:
each has the form `share × lag + (1 − share) × rest_mean`, where the lag is a
planning value the ranking set exposes and the rest mean is an ordinary
taxi-out: for the term, the cell's mean over its rows that were not defaulted,
and for the plan-shift rule, the airport's mean over departures with a record
and a taxi-out of four hours or less. The term reaches 383 of the 344,841 ranking rows, all at one
airport; the plan-shift rule reaches one. §4 gives the mechanism, the evidence
and the basis of adoption for each. §8 reports every pooled figure with and
without the rows they cover, so that a reader who disagrees with either rule
can subtract it.

Sources: `reports/tails.md`, *What it touches in the ranking set* and *The
plan-shifted departures*.

### 3.4 The booster, its parameters, and the seed average

The booster is LightGBM (Ke et al., 2017) with `objective` regression: 6,000 trees at a learning
rate of 0.03, 127 leaves, a minimum child sample of 20, a column sample of 0.9
and a row sample of 0.8 per tree, and categorical splits capped at 64
thresholds with a smoothing of 20.0. It uses no early stopping, since stopping
on the set being reported is tuning on it. It runs in LightGBM's deterministic
mode with column-wise histograms at eight threads, so that a fit repeats to the
bit (§10). The four parameters that were searched were tuned on October 2025, a
month no rung of the ladder scores, under a rule fixed before the search. When
the best setting fell on the edge of the searched grid, the grid was widened
after that result was seen; §7.1.2 gives the account. The base configuration is the
rung `+ plan differences` fitted five times under consecutive seeds and
averaged, adopted under a rule that compares its gain with the seed floor
(§7.1.2). On
the December forward split the average lowers the error by 1.11 s against the
single seed the ladder uses, where the seed floor is 0.23; on July it lowers it
by 1.46 s against a floor of 0.54.

Sources: `src/taxiout/models.py`, `BOOSTER_PARAMS`, `BAGGING_SEEDS` and
`SEED_AVERAGE_ADOPTED`; `reports/baseline.md`, *What is fitted, and on what* and
*Seed averaging*.

### 3.5 The validation design

The design has six readings: three fixed before anything was fitted, and three added later and declared as such. None is a random K-fold on
rows, because congestion is autocorrelated over hours.

| reading | training rows | scored rows | stands in for, or asks | fixed |
| --- | ---: | ---: | --- | --- |
| December forward split | January to November 2025, 1,919,370 | December 2025, all ten airports, 165,677 | January 2026 | before any fit |
| July forward split | January to June 2025, 1,005,519 | July 2025, all ten, 190,713; *the three* (Frankfurt, Heathrow, Schiphol) read on their own as well | July 2026 | before any fit |
| held-out pairs | the year minus a third of every airport's (stand, runway) pairs, three folds of 1,303,727 to 1,338,046 | the held-out pairs, 676,400 to 712,627 per fold, beside a 5% sample of seen-pair rows | whether a block carries over to a pair never seen; the claim of §7.2 | before any fit |
| October | January to September 2025 | October 2025, all ten | the leg the booster's parameters were tuned on, scored by no rung of the ladder | added later, declared |
| the gap leg | January to June 2025 | December 2025 | whether a recent month or the same season matters more, for the July 2026 rows whose nearest training month is seasonally opposite (§7.1.1) | added later, declared |
| every month held out | the other eleven months of 2025 | each month of 2025 in turn, 143,732 to 191,182 rows | how the figures and each block's worth are distributed over the year. Not forward in time, and a diagnostic | added later, declared |

The held-out unit of the pair folds is the **pair**, not the stand. Every stand
and runway is still seen in other pairs, so the folds bound what the geometry
buys on a new combination and not what it would buy on a new stand, which no
split of this data can measure. The held-out months train on months after the
one scored as well as before, so they cannot replace the forward splits as
proxies for the test months. What they add is twelve readings of every figure
instead of two, and every month of the year the claim can be read in.

Sources: `reports/baseline.md`, *What is fitted, and on what*, *Held-out pairs*
and *Seed averaging*; `reports/monthly_ladder.md`, *What this is* and *The
harness is the ladder's*; `reports/preregistered-2026-09-20.md`, a dated record,
for the gap leg.

### 3.6 The instruments

The paper uses three instruments, and wherever a figure is called real or
called noise, it says which one speaks.

- **The seed floor**, root two times the standard deviation of five single-seed
  fits of one rung on one leg: how far a figure moves when nothing but the
  random seed does. Under the parameters that ship it is 0.23 s on December,
  0.54 on July over all ten airports, 1.02 on the three and 0.21 on October;
  over the twelve held-out months it runs from 0.07 s in April to 1.56 s in
  February, and is 0.16 s over the year pooled. A difference inside the floor is
  within what changing the seed alone produces. The floor is measured on the
  shipped rung and applied to every block's change, an approximation the reports
  state.
- **The day-clustered interval**, a 95% percentile bootstrap (Efron &
  Tibshirani, 1993) over the evaluation set with days as the clusters (Cameron
  et al., 2008) and the same draw for both arms: how far a
  figure would move on another sample of days. The interval and the floor
  measure different things and have disagreed (§7.1.1, §7.1.2).
- **The pair-clustered interval** of the held-out-pair claim, 1,000 draws
  resampling whole pairs from the stored residuals: no model is refitted.

Printed precision is a tenth of a second, finer than every floor, so a pass or
fail test that compares figures rounded to that precision can trip on seed noise
alone, and once did (§7.1.1). Changes are signed as §1.4 states: negative means
the error went down.

Sources: `reports/baseline.md`, *The noise floor* and *Held-out pairs*;
`reports/monthly_seeds.md`, *Per month and over the year*.

### 3.7 What was fixed before the numbers

Every rule the paper's figures are read against was written down before the
figure it governs existed, except where §9.3 says otherwise. Its table lists
sixteen decisions taken after some of the figures they bear on had been seen,
six before the first challenge score and ten after it, each with its date, what
had been fixed, what had been seen, what was decided and the record that holds
it. The table below gives, for each approach, the question it asks and how that
bears on the central hypothesis, what was fixed before the figures, and what
followed.

| approach | what it asks, and how it bears on the hypothesis | fixed before the figures | outcome |
| --- | --- | --- | --- |
| seed averaging, recency weighting, season columns | whether the forward legs mislead on time and seed choices; none changes the claim, each bounds how the legs are read | the adoption rule for seed averaging; the pass or fail tests for recency and the season columns | seed averaging adopted; the other two not (§7.1.1, §7.1.2) |
| the booster's parameters | whether the setting can be chosen without tuning on a month that is reported | the second sweep's grid and rule; the extension's criteria, before any figure past the grid's edge existed | a cheaper cell taken over a better one the instrument could not separate (§7.1.2) |
| the record-quality term and the plan-shift rule | whether rows whose recorded value no aircraft produced can be handled by a stated rule | the term's form; the rule's form, before it was evaluated on any fold | adopted (§4) |
| three model changes left untested at the freeze | whether any improves the base configuration | the rule for measuring them on top of it | none adopted (Annex J) |
| changes after the first challenge score | whether each improves the base configuration on the training year | a written pre-registration for each change; ten decisions after the first score were taken after figures they bear on, and §9.3 lists them | adopted in part (§8.7, §8.8, Annex K) |

Sources: `reports/preregistered-2026-09-20.md`,
`reports/preregistered-2026-09-28.md`, `reports/preregistered-2026-10-02.md`,
and `reports/preregistered-2026-10-03.md`, dated records.
## 4. The tail, and what the scoring rule makes optimal

*This section comes early and stands alone because the record-quality term is
the part of the method most open to misreading. It is a rule for rows whose
recorded value no aircraft produced, and it is not a fitted model.*

Rows whose error under the airport mean exceeds an hour are 0.086% of training
departures and carry 50.9% of that predictor's squared error. The data review
read them as records shifted by a whole day. Building the term found the
mechanism behind most of it instead: at one airport, a departure with no
flight-plan record and a take-off hours after schedule has, nine times in ten, a
recorded off-block time within a minute of the *schedule*. The block time was
defaulted, so the recorded "taxi-out" is the schedule lag, which every ranking
departure exposes. The day-shift is real but small.

The term is the conditional mean per (airport, record, lag-band) cell, computed
knowing that mechanism. A cell is an airport, whether the departure has a
flight-plan record, and a band of schedule lag: normal, from −1 h to 4 h; 4 to
12 h; over 12 h; and under −1 h. For a row in a covered cell the term predicts
`share × lag + (1 − share) × rest_mean`, where the share is the cell's share of
defaulted rows and the rest mean is the cell's mean taxi-out over its rows that
were not defaulted. A cell is covered when it has at least five training rows
and a defaulted share of at least 5%. Three cells qualify, all at Fiumicino
(LIRF) and all without a record, with 1,485 training rows between them: the
normal band, at a defaulted share of 21.7%, the band of 4 to 12 h, at 84.7%,
and the band over 12 h, at 62.1%. They hold 383 of the 344,841 ranking rows,
107 in January and 276 in July. It is **the RMSE-optimal prediction under the stated
scoring rule for rows whose recorded value no aircraft produced**, and nothing
else. Every RMSE in this paper is reported with and without the rows it covers.

**Why an uncovered airport is left to the physical model, and what measures
that.** A cell counts as covered only where enough of its rows carry the
mechanism, and an airport with no covered cell is left to the physical model
entirely. The alternative considered when the term was designed was to pool
cells across airports. On the 2025 validation legs a pooled design is worse than
the adopted per-airport term on all three readings: 390 s against 341 on
December, 586 against 461 on July pooled, and 385 against 339 on the three. That
comparison is the arithmetic of the choice and not the whole of the answer. It
sets a differently shaped design against the term, and the case that presses the
question is not a validation leg. At Schiphol the ranking set lists six
consecutive days in January 2026 with the defaulted-record pattern, and the
training year holds no such day at that airport; the longest run anywhere in
2025 is two days, at Madrid.

What settles the question is an instrument already in the data, and it measures
the premise pooling rests on. That premise is that the defaulted share of a lag-band
cell, read across airports, is a property of the defaulting mechanism rather than of the airport, so that a
cell learned where the mechanism is visible may be carried to an airport where
it is not. The fitted cell table measures that premise per airport and per
lag-band cell, on training rows, with no fit and no booster. It says the mechanism is
at one airport: of the 432 training rows the term's own test picks out, 99.1%
are at Fiumicino, and only three of the ten airports carry any at all. In the
two lag-band cells a pooled cell would serve, every other airport reads 0.0% on every
leg, at row counts small enough that a single defaulted row would show. The
arithmetic test sets each airport's share against two standard errors of the
pooled share at that airport's own row count. On that test the premise fails in
both of those lag-band cells in each of the three training sets the legs use. Where an
airport passes, it passes on the width of its interval and not on its share: in
the larger lag-band cell over the full year three airports sit inside, each at 0.0%,
because at 10 to 14 rows the interval reaches down to zero. The July leg is the
thin one, with 187 and 26 rows pooled in the two lag-band cells, and the premise fails
there too. Schiphol's zero is a measurement and not a gap. Its cells hold 77
and 14 rows, well above the five-row minimum, so one defaulted row would show as
1.3% or 7.1%, and a share of zero means no defaulted row in a full year. Pooled,
those cells would have predicted 4,512 s against a recorded mean of 824 s and
32,761 s against 1,212 s on Schiphol's own 2025 rows. The pooled rule is
therefore not adopted. The model version that would have tested it was not
fitted, because its premise failed, and the reason is measured rather than
asserted.

One limit remains, narrower than the decision it used to carry. Whether an
uncovered airport's 2026 rows are themselves defaulted block times is a question
about the two fields the ranking set hides, and no instrument reaches them. §7.3
states the exposure that leaves.

**The plan-shifted departures, added after the freeze.** One class of departure lies
outside the cells' reach: a take-off more than four hours after both of the
flight plan's off-block estimates, `EOBT_1_flt` and `LOBT_flt`. The cells key on
the schedule lag and on the absence of a record, and these departures have a
record. The class was seen in the ranking set's exposed fields, after the
freeze, in the reading of the held-out-month ladder: one July 2026 departure at
Fiumicino whose plan values all lie about a day before its take-off, which the base configuration predicted at 3,120 s. The training year holds three such
departures, one at Paris and two at Fiumicino. Two of them recorded a taxi-out
over four hours, 21,308 s and 87,002 s, each within minutes of take-off minus
the plan's last off-block estimate; the third recorded 13,865 s, under the
threshold. That is the whole of the evidence. The rule was written down and its
digest recorded before it was evaluated on any fold or built into any file, so
that nothing about it could be chosen on its effect.

The rule takes the term's own form. The share is the add-one estimate over the
class, `(long + 1) / (n + 2)`, which on three rows with two long reads 0.6
rather than the two thirds that three rows cannot carry. The lag is take-off
minus `LOBT_flt`. The rest mean is the airport's mean recorded taxi-out over
departures with a record and a taxi-out of four hours or less, 1,144.2 s at
Fiumicino. The rule reads nothing else, never `AOBT_3_flt`, and it moves nothing
else: the booster, its rows and its seeds, and the term's cells, are what they
were. Evaluated after it was fixed, on the twelve held-out months, it touches
three validation rows and lowers the RMSE over every row, the term's included,
from 276.6 s to 272.4 s. The pre-registration itself says this is a statement of
the arithmetic on three rows and not independent evidence that the rule is right
(Annex I).

On the one ranking row the rule predicts 52,550 s, against 3,120 s before. Its
effect on the RMSE over the 344,841 ranking rows depends on what the row
recorded, taken on an assumed base of 300 s. If it recorded the lag, 86,820 s,
the RMSE would fall from 332.1 s to 305.6 s. If it recorded an ordinary
taxi-out, the airport's rest mean, the RMSE would rise from 300.0 s to 312.5.
The rule gives up the best case to avoid the worst, and the share it rests on
says six in ten. No record of the row itself settles which outcome it has. The
base of 300 s is an assumption and not a measurement, and the two cases are the
row's two readings, not a distribution over them.

The final configuration goes further on this row. Its flight-plan window
(§8.7.6), applied after both rules, keeps every prediction within 3,618 s of the
plan's lag, because no recorded off-block time of 2025 lies more than 3,606 s
from the plan's last off-block estimate, and the class's July 2025 departure
lies within a second of it. On this row the window lifts the rule's prediction
to the plan's lag less 3,618 s, so the final file takes the reading in which the
row recorded the lag. The two cases above are the base configuration's; in the
final file the first case gains nearly all of the lag reading, and the second
costs more than it does under the rule.

Sources: `reports/tails.md`, *What the tail is made of*, *The cells*, *Why this
form, and not the others*, *What it touches in the ranking set*, *The
plan-shifted departures* and *The pooled record-quality rule, measured*;
`reports/planshift.md`, *The class in the training year*, *The rule fitted on
the full year*, *The one ranking departure* and *Per held-out month and on the
two forward legs*; `reports/window.md`, *The verdict* and *The plan-shift class*; `reports/data_review.md`, *The tail owns the metric*, *Days
in the ranking set that look like disruptions* and *The same days in the
training year*.

## 5. The unimpeded term

### 5.1 From OSM to a routable surface

Three defects are repaired to a fixed point: coincident-but-unshared nodes,
nodes lying on edges they are not joined to, and ways that cross with no shared
vertex. The repairs are not small: Frankfurt needs 808 edge splits, and Zurich
166 node merges and 1,925 splits. They run to a fixed point rather than once
because every repair creates edges that may need repairing in turn: a runway
crossed by several taxiways needs one round per crossing, and Frankfurt takes
25 rounds. Stands are attached to the repaired graph, with the snap distance
recorded per stand. When the router chooses a route, runway pavement costs four
times its length (the runway-edge penalty), so that a route does not use a
runway as a convenient straight taxiway. The value sits on a plateau rather
than at an optimum: from 4 upward, distances and descriptors barely move, while
the share of stand-runway pairs that run along a foreign runway falls from
30.4% at a penalty of 1 to 9.2% at 4. Each route is described by several
descriptors rather than one number: distance, turn angle on a Douglas–Peucker
simplification (Douglas & Peucker, 1973), junction-free running and backtrack.
Backtrack is the distance run along the runway itself to reach a threshold that
no mapped taxiway joins, which happens at 8 designators across 5 airports. On
the repaired graph, 95.2% to 100% of stand-runway pairs are routable at each
airport, and 98% to 100% of stands attach at each of the ten challenge
airports. The runways cross-check against OurAirports: 39 match by designator,
with 0 discrepancies.

Sources: `reports/taxi_routing.md`; `reports/penalty_sensitivity.md`;
`reports/runway_crosscheck.md`; `docs/modelling_notes.md`, *Graph construction*.

### 5.2 A worked airport: where every number is known

A fictional field, on which every distance, turn and repair is hand-derived, is
pushed through the pipeline unchanged. All 32 matrix cells agree to 11 mm and
0.10°, and the integer descriptors are exact; the runway-penalty crossovers fall
where the arithmetic puts them. The field found three things in the pipeline,
each below its working tolerance and each measured: overshoot and undershoot
select different repairs; a merged node takes the last way's coordinates; and a
two-feature stand's representative is decided by floating-point rounding. The
field also carries worked traffic in the challenge's own schema, so the whole
pipeline runs end to end in continuous integration without the challenge data.
The worked field is how the geometry is known to be right without asking a
reader to trust matrices that cannot be checked by hand.

![The worked airport](../reports/synthetic_airport.svg)

**Figure 1.** The worked airport. Runways are gray and taxiways blue; the two worked routes, stand 1 to runway 09 and stand 12 to runway 36, are dashed, and the circles mark the repairs the pipeline makes: a merge, a split and a crossing. Stand 11, in red, does not attach to the network. Every distance in the field is derived by hand before the pipeline runs.

Source: `reports/synthetic_airport.md`, *In one paragraph*, *What the field
found* and *Worked traffic*, and its figure.

### 5.3 What the geometry recovers, and what it does not

Per airport, a pair's fastest decile follows its routed distance with a residual
of 85–190 s and an intercept of 313–740 s that no distance explains. The largest
structured effect on that decile is wake category: mediums are 35–281 s faster
than heavies at the same stand and runway, uniformly across designators. That
pattern is a fixed pushback-to-line-up difference and not intersection use.
Intersection departures proper leave a signature of at most 96 s per designator,
constant, and absorbed by any model knowing the designator. Which stands use
which entry is unobservable without surface tracks.

Source: `reports/geometry_gaps.md`, *In one paragraph*, *The fit*, *The fastest
decile by wake category* and *The intersection signature, per designator*.

### 5.4 Where memorization runs out, and what replaces it

**How much of the test set is thin.** Of the 344,841 ranking departures, 0.72%
sit on a (stand, runway) pair the training year never saw, 1.20% on a pair seen
ten times or fewer, and 9.08% on a pair seen between eleven and a hundred times:
**11.0% in total**. The share is uneven across airports. Paris has 20.48% of its
rows on thin pairs (seen a hundred times or fewer), 19.07% on pairs seen eleven
to a hundred times alone, and Madrid 15.57%. Frankfurt has 2.37% never-seen and
4.05% at ten or fewer, and Munich 4.57% never-seen. Heathrow has 0.18% and
0.28%, and 2.08% thin in all. July has 1.20% never-seen against January's 0.13%.
Across the ten airports, 46 stands appear in 2026 that appear in no 2025
departure. They are at nine of the ten and carry 1,589 ranking rows. Zurich
contributes eleven of them and Munich ten, but Munich's ten carry 1,182 of the
1,589 rows and Frankfurt's six another 300. One runway designator is new, on
two rows. The never-seen bucket, 2,496 rows, splits into 1,589 on a new stand,
two on the new designator, and 905 on a stand and a runway the training year saw
separately and never together. The last kind is what the held-out-pair split
measures; the first is the case no split of the training data can pose.

**A pair's own mean is worse than ignoring the pair where it is thin.** On the December
forward split, predicting a pair by its training mean scores 571.9 s on pairs
seen one to ten times, worse than ignoring the pair and predicting the airport
mean, 519.0 s. On July the same comparison is 543.1 against 498.3. A mean over a
handful of draws from a distribution several hundred seconds wide is noise that
passes for knowledge of the pair.

**The geometry is the better prior, and by more than the usual one.** The
textbook remedy, shrinking the pair mean toward an estimate, depends on what it
is shrunk toward. Toward the airport mean, the standard choice, the December
bucket improves to 454.5 s. Toward the geometric prediction for that specific
pair, with the shrinkage weight estimated per airport by the same construction
the encoding of §7.4 uses, it improves to **433.3 s**. The geometric prior is
worth 21.1 s more than the group prior on that bucket in December and 20.7 s in
July. On the never-seen bucket, where the two priors stand bare with no pair
mean to shrink, the margin is 19.7 and 22.9 s. On pairs seen more than a hundred
times every version of the pair mean agrees to a second, because the count
swamps the weight. Where memorization works, the prior does not matter; where
it runs out, which prior it runs out to is the whole question. Per airport, the
geometry explains 4% to 61% of the between-pair variance the airport mean
leaves, which is why a pair's handful of draws is weighed less against it.

**The relation transfers between airports.** Fitted on nine airports and applied
to the tenth, with no stand, pair or airport-level quantity of the tenth
available, the transfer model beats the only baseline available without it, the
grand mean of the other nine, at **ten airports out of ten**. It wins by a
median of 23.6% and by as much as 43.4% at Heathrow. The transfer model is the
geometry plus the clock, the aircraft, the plan lags, the traffic in the window
and the weather, with every identifier removed. The geometric term is one least-squares
fit of each pair's 10th-percentile taxi-out, over the rows at or below it, on an
intercept per airport and wake class plus the routed distance over a speed per
airport. Five adjustments, pooled across airports, change the time per
kilometer with the route's shape: turn angle per kilometer, junctions per
kilometer, the share of straight running, the detour ratio and the share of
backtrack. At an airport the fit has not seen, it answers with its pooled speed
and intercept. That term *alone*, two fitted numbers (a speed and an intercept)
and the five adjustments, beats that mean at eight of ten and loses at Munich
and Heathrow, by 4.1% and 1.5%. The speed the ten leave-one-out fits recover
runs from 11.8 to 13.2 m/s, a 12% spread about 12.42 m/s, with the intercept at
434 to 500 s and the offset, the constant that carries an unimpeded estimate
onto the scale of an observed taxi-out, at 251 to 276 s. That stability is the finding. The relation behaves as a
transferable physical quantity rather than a per-airport adjustment, and it is
the one thing in this work that an identifier model cannot reproduce even in
principle, because at an unseen airport such a model has nothing to encode.

Sources: `reports/geometry_gaps.md`, *Where memorization runs out*, which
carries the per-airport and per-month tables, the new-stand table with the shape
of each identifier, both forward splits by bucket, and the shrinkage weights,
none of them restated here; `reports/baseline.md`, *The fit* and *The
modulators*, for the form of the geometric term; `reports/transfer.md`, *The
speed, ten times over*
and *RMSE, per held-out airport*, for the per-airport transfer table, likewise.

### 5.5 Taking the model to a new airport

The model needs different amounts from a new airport depending on how much of it
is used. The geometric term needs only a map. The transfer model of §5.4 needs
the airport's traffic, flight plans and weather but no recorded taxi-out. The
full model needs about a year of the airport's own history. The first two were
measured by leaving each of the ten airports out in turn; the third was not
measured at an unseen airport, and its evidence comes from the ten.

**What the airport has to supply.**

| input | used for | where this work gets it | without it |
| --- | --- | --- | --- |
| the aerodrome reference point and a search radius that encloses the movement area | the extent of the map extract | published aerodrome reference points | no map can be built |
| OpenStreetMap aeroway features: taxiways, runways, aprons and parking positions | the routable surface | one cached Overpass extract, with its fetch time pinned | no geometry |
| the stand identifier on each mapped parking position, in the same code the operator's data uses | joining a departure to its route | the `ref`, `aeroway:ref`, `local_ref` and `name` tags | the stand's distance features fall back to airport-level constants; coverage is audited per airport |
| the departure record: stand, runway, take-off time in UTC, aircraft type and wake class, operator, destination | the identifier, time and aircraft blocks | the operator's or the network's movement data | the blocks that read it are absent |
| the flight plan: scheduled time and the plan's off-block estimates | the plan lags, the strongest single feature (62.9 s over the year, §8.4) | the Network Manager record | the model loses its strongest feature |
| the other movements around each take-off: times, runways, arrivals and their on-block times | the congestion, corridor and occupancy blocks | the same movement data, every movement at the airport | those blocks are absent; the plan lags carry most of the congestion signal (§8.4) |
| a METAR archive for the airport's station | the three weather regime flags | the Iowa Environmental Mesonet archive, public domain | the weather block is absent |
| the airport's country | the holiday calendar | a country code per airport | the holiday columns read as missing, not as ordinary days, and the submission writer refuses |
| the local time offset | the daily cycle | not carried; the airport identifier absorbs it | at a new airport the UTC clock misplaces the daily cycle by the offset; supplying it is untested |
| about a year of recorded taxi-outs | the shrunk encoding, the record-quality cells, the per-airport choices | the operator's history | not needed for the first two tiers |
| receiver coverage of ground movements, and the airport's reference point | the ground-movement correction | adsb.lol and OPDI, open data | the correction is not applied |

**The steps, in order.** Each step has its own check, so a failure shows where it
happened.

| step | what is done | what it checks |
| --- | --- | --- |
| 1 | Add the airport's record (ICAO, IATA, name, reference point, radius) to the airport list. | the license and provenance checks, which read the list |
| 2 | Fetch the OpenStreetMap extract, cache it, and pin its fetch time as the other airports' are. | the pin check |
| 3 | Build the graph and the stand-to-runway matrix. The three defects of §5.1 are repaired to a fixed point. | the routing report states what resolved and what did not |
| 4 | Audit stand coverage and the physical runways, and check the runway penalty's sensitivity. | the share of the airport's stands that resolve; the runway thresholds against OurAirports |
| 5 | Add the country; fetch the METAR archive for the station. | the calendar and weather blocks read as present, not missing |
| 6 | Predict with the transfer model. The same code builds the airport's rows as builds the training rows. | the geometric term alone is the floor to quote, as in §5.4 |
| 7 | As movements accumulate, fit with the airport in training, and read it on its own held-out months. | the pair-frequency buckets and the contrast of §7.2, for that airport |
| 8 | Let the record-quality term test the airport: a cell is kept only where it holds at least five rows and a defaulted share of at least 5% (§4). | the pooled-share test of §4 |
| 9 | Add the airport's reference point to the ground-movement tables. Fit and gate the correction (§8.7.4) from the airport's own out-of-fold evidence, and let the selector of §8.7.5 choose the blend. | the airport gate and the selector's choice over the held-out months |

**Crosswalk 1: conventions that change between airports.** Each row is a
convention this work fixed once for ten airports, what can differ at a new one,
and what the difference would do to a claim.

| convention here | what can differ at a new airport | what to do | what it puts at risk |
| --- | --- | --- | --- |
| All times are UTC, and no timezone table is carried; the airport identifier absorbs the offset. | The local offset, and whether the airport observes daylight saving. The ten span UTC+0 to UTC+3; Istanbul and Antalya are three hours ahead, so the first three hours of a local holiday fall on the previous UTC date. | Supply the offset and the daylight-saving rule per airport. Derive the hour of day and the holiday date from local time, and keep UTC for every join. | The daily cycle and the holiday flags, and any comparison of daily patterns across airports. Daylight saving is not examined in this work. |
| The target and the off-block fields are read at the resolution the data gives; the Network Manager's recorded off-block time has minute resolution. | The operator's resolution and clock discipline. | State the resolution of each time field before reading a figure; a coarse off-block field sets a floor under the error. | The size of the smallest effect that can be demonstrated. |
| Airports are keyed by ICAO code, and the weather archive is queried with the ICAO code as the station. | The archive may key a station differently outside Europe. | Confirm the archive returns rows for the station before building the weather block. | The weather block, which would otherwise read as missing. |
| Stand identifiers are joined in three stages: case and whitespace, leading zeros, one trailing letter. | Padding, spaces and lettered half-stands differ per airport. | Inspect what fails to match. Add a stage only on the join rate, never on a validation error. | Whether pair-level geometry exists at the airport at all. |
| Runway designators run 01 to 36 with an optional L, C or R, and thresholds come from `ref`-tagged geometry only. | Non-standard designators, runways used in one direction only, displaced thresholds, and runways split into several map fragments. | Parse the designators, take thresholds from tagged geometry, and check them against OurAirports. | The routed distance of every pair that uses the runway. |
| The holiday calendar is read by country, as ISO 3166-1 alpha-2; subdivisions are not read. | In federal countries the regional holidays matter more. | Keep the country code; treat a subdivision as a better feature, untested here. | The holiday columns at airports in such countries. |
| The weather flags use kilometers and degrees Celsius: 8 km and 1.5 km visibility bands, a 0 degree freezing line. | Another region may report in other units. | Convert before flagging; confirm the units the archive returns. | The weather block. |
| The map is one OpenStreetMap extract per airport, pinned to its fetch time. | The map changes as contributors edit it, and the license travels with the extract. | Pin each fetch, re-fetch deliberately, and carry the ODbL attribution. | Whether a distance can be reproduced. |
| The record-quality term exists where the off-block time was defaulted to the schedule, which here is one airport. | A new operator may default differently, or never. | Run the cell test of §4 on the airport. If no cell qualifies, the airport is left to the physical model. | Whether an RMSE means anything at the airport. |
| The ground-movement correction reads aircraft seen within 8 km of the reference point. | Receiver coverage differs by airport and by day. | Measure coverage per airport before applying the correction. | Where the correction can be claimed to help. |
| Data terms are recorded beside each cache: ODbL for the map and for adsb.lol, public domain for the weather and for OurAirports, MIT for the holiday code (Vacanza Team, n.d.). | A new region's data may restrict redistribution. | Record the terms with the cache manifest, as here. | Whether the result can be reproduced by a reader. |

**Crosswalk 2: the operator's fields and the names used here.** A new airport
maps its own fields onto these, and the mapping is the first thing to check.

| the operator's field | name used here | used for |
| --- | --- | --- |
| departure airport | `ADEP_mvt` | the airport key |
| stand | `STAND_mvt` | the join to the map |
| runway | `RUNWAY_mvt` | the join to the map; values that name no runway are excluded |
| take-off time, UTC | `MVT_TIME_UTC_mvt` | the anchor for every time feature |
| off-block time, recorded | `BLOCK_TIME_UTC_mvt` | the target's start; withheld in the ranking set |
| taxi-out in seconds | `TAXITIME_SEC_mvt` | the target |
| scheduled time | `SCHED_TIME_UTC_mvt` | the schedule lag |
| aircraft type; wake category | `AIRCRAFT_TYPE_mvt`; `WK_TBL_CAT_flt` | the aircraft block |
| callsign; operator | `CALLSIGN_flt`; `AIRCRAFT_OPERATOR_flt` | the callsign-prefix block |
| destination | `ADES_mvt` | the destination block |
| movement phase: departure or arrival | `PHASE_mvt` | separating the traffic around a take-off |
| the flight plan's off-block estimates | `EOBT_1_flt`, `LOBT_flt` | the plan lags |
| the Network Manager's recorded off-block time | `AOBT_3_flt` | admitted with the feature group of §8.7.3 (§6.3) |

**What is measured and what is not.** Steps 1 to 6 are measured by the
leave-one-out result of §5.4: the transfer model beats the pooled mean at ten
airports of ten, and the geometric term alone at eight. Steps 7 to 9 are
measured only at the ten airports in the data. The ten are large European hubs
with mapped taxiways, so the stability of the speed is evidence about airports
like these. A regional field with one runway is not represented, and a hub with
poor stand tagging would first meet step 4. Two conditions fail quietly if they
are not checked: unresolved stands, and an unmapped country. The code treats
both as measurements, so each shows as missing rather than as an ordinary value.

Sources: `reports/transfer.md`, *What each predictor is allowed to read* and
*What this does not show*; `reports/monthly_ladder.md`, *Summary: the year,
pooled from the twelve months*; `reports/tails.md`, *The cells*;
`src/taxiout/airports.py`; `src/taxiout/coverage.py`; `src/taxiout/calendar.py`;
`src/taxiout/features.py`; `src/taxiout/join.py`; `src/taxiout/runways.py`;
`src/taxiout/metar.py`; `src/taxiout/adsb.py`; `src/taxiout/challenge.py`.

## 6. The excess term

### 6.1 What the ranking set allows, and the registry that enforces it

The rule is phase-specific: a feature may read what the ranking set exposes *for
the rows it reads from*. Every feature column is registered against its raw
sources and the phase each is read from. The test suite holds each phase to its
own list, and a poison test sets the hidden columns to nonsense and asserts that
the frame does not move. Arrivals' on-block and taxi-in times are exposed at
0.0% null and are read; no departure's block or taxi time is, ever.

Sources: `reports/congestion.md`, *The leakage audit*; `reports/data_review.md`,
*Null rates*.

### 6.2 Congestion, corridor, and weather

These blocks read throughput, arrivals, demand, runway configuration,
shared-corridor load, and three weather regime flags per airport-hour. The
corridor feature does not appear in the prior art of §1.3. It weights other
movements by how much of the subject's own taxiway route they share, with each
edge weighted by the inverse of the routes using it. Read against
`+ configuration`, it moves the RMSE by +0.13 s on December, inside the seed
floor of §3.6, by +1.16 s on July, beyond that leg's floor of 0.54 s, and by
+0.99 s on the three, inside theirs. On held-out pairs it gains +0.3 s
[−0.3, +0.8] more than on seen ones, an interval that spans zero. Over the
twelve held-out months it moves the year by +0.1 s, inside the year's floor,
and it sits inside the floor in 11 of the 12 months; in the twelfth, May, it
helps by more than the floor.

Sources: `reports/congestion.md`; `reports/baseline.md`, *The noise floor* and
*What each block buys where the pair is new*; `reports/monthly_seeds.md`, *Each
block's months, against each month's floor*.

### 6.3 The flight plan's own delay signal

Take-off minus the filed off-block estimate is the strongest single feature in
the model. As a ladder step it lowers the error by 79.3 s in December (320.2 →
240.9 s ex-flagged, that is, on the rows the record-quality term does not take)
and 57.4 in July (407.3 → 349.9). Measured at the `+ weather` rung, by removing
it and keeping everything else, it is worth 59.3 and 51.7. Either way it is
worth more than every congestion and weather block combined, which net of it are
worth 7.7 s and 5.8.

The feature is admissible for the same reason the record-quality term is.
Take-off minus a planning value fixed before departure equals the taxi-out plus
the pushback delay against that plan. The model is therefore learning the
distribution of pushback delay around a value it is given, which is the same
construction the record-quality term uses with the schedule. `EOBT_1_flt` and
`LOBT_flt` are planning values the ranking set exposes, not post-hoc records of
the blanked field. `AOBT_3_flt`, which is such a record, is excluded from the
base configuration. The final configuration admits it (§8.7.3) under the
challenge's post-hoc framing: the data page documents the field, and the ranking
set blanks only the block time and the taxi time.

Its held-out-pair contrast is −1.4 s [−5.8, +3.2], an interval that spans zero.
It helps equally whether the pair is new or familiar, which is what a feature
about the day rather than the place should do. The ladder is reported both with
it and without it, and the regime exposure is bounded in §7.3.

Sources: `reports/baseline.md`, *The ablation ladder* and *Held-out pairs*;
`reports/congestion.md`, *Regime exposure*.

### 6.4 The queue layer that was not built

An oracle diagnostic bounds what a queue layer could have added. It regresses a
departure's residual on the mean residual of its same-runway neighbors,
quantities no admissible feature may read and which a queue model is precisely
an attempt to approximate. On the base configuration, the ceiling is **8.2 s in
December and 8.3 in July**; the lag-1 autocorrelation of consecutive same-runway
residuals under two minutes is 0.19 and 0.09.

The plan-lag block is what closed that ceiling. On the full model fitted without
that block the same ceiling is 31.5 s and 27.5, and the autocorrelation 0.31 and
0.17. Most of what a queue model would have found was therefore the day's delay
against plan, and the cheapest possible feature had already taken it.

A saturation proxy was then built against that ceiling: the share of
sub-90-second gaps, the minimum gap, and the span of the last three gaps on the
subject's runway. Measured under the parameters then in force, it bought nothing. After the parameter sweep it cost half a
second in December and eight tenths in July against the rung it was added to,
and the rung carrying it had a ceiling of 10.0 and 8.5, within a second of the
model's own. The queue-model work was stopped because of that
measurement, not because time ran out; the stop is recorded in the plan, and the
submission is not fitted from the block.

The block stays on the ladder, and under the parameters that ship it reads
differently, which is reported rather than smoothed over. Against `+ weather`,
the rung it was added to, it now lowers the error slightly, from 233.2 s to
232.7 in December and from 344.1 to 343.4 in July, a change larger than the seed
floor on both splits. Its held-out-pair contrast is −0.6 s [−1.2, −0.1], and its
ceiling, 9.2 s and 8.3, sits beside the base configuration's 8.2 and 8.3. It has never
been measured on top of the rung the submission is now fitted from, so what it
would add there is not known.

The diagnostic needs no model, which is why it could be run before one was
built.

Sources: `reports/congestion.md`, *The ceiling*; `reports/baseline.md`, *The
ablation ladder*, *Held-out pairs* and *The noise floor*; `docs/plan.md`, *When
the data lands*, a dated record, for the measurements of 3 and 4 September.

### 6.5 The stand's recent history, and the bound inside it

The stand-occupancy block is the largest of the four blocks added last (the
others are the three smaller blocks below). It reads the arrivals at the
subject's own stand: the gap from the last arrival's on-block to the subject's
take-off, that arrival's lateness and taxi-in, whether it flew the subject's
aircraft type and operator, and the gap to the previous departure off the same
stand. Every field is one the ranking set exposes for arrivals in full (§6.1).
On the ladder it is worth 8.6 s in December (228.3 → 219.7 s ex-flagged) and 9.1
in July (342.2 → 333.0), the largest single step above the plan lags on either
forward split; on the three, it is worth 16.0 s (256.6 → 240.6). Its
held-out-pair contrast is +1.8 s [+0.8, +2.9]. The block buys slightly more
where the pair is new, which is what a stand-level quantity should do where the
encoding has nothing to say about the stand.

**The bound, and where it holds.** The bound is true only in a band of the gap.
A stand holds one aircraft, so an arrival that on-blocks at the subject's stand
before the subject takes off means the subject had already pushed back, and the
gap is a lower bound on its taxi-out. That is why the block was built. Measured
on the December rows of the rung that never read the gap, the bound behaves as
follows. Under twenty minutes it holds in 97–100% of rows and the model already
respects it. At 20–25 minutes it holds in 94% of rows while the model violates
it in 17%, under-predicting those rows by 171 s on average; at 30–35 minutes,
71% and 56%. By 40–45 it holds in 11% of rows, and past forty-five minutes
essentially never, because a stand identifier can denote a position that holds
two aircraft and because aircraft are towed. Applied as a rule it loses at every
cap tried (+0.1 s at twenty minutes, +1.2 at thirty, +20.6 at forty-five in
December). A floor can only act where the model is below the gap, and where the
bound is true the model already obeys it. Instead, it enters as a feature, and
the booster finds the band.

**Where the gain lands, and how much of it is the bound.** The bound accounts
for a minority of the block on every leg. The block's reduction in squared error
is a sum over rows, so it can be attributed to disjoint bands of the gap and the
shares add. The 20–45 minute band is 1.0% of December's rows and 20.9% of the
block's reduction in squared error (its cut); on the July three, 1.9% and 34.9%.
Inside that band the previous arrival's type separates the two cases the band
contains. Where the type differs, a different aircraft took the stand after the
subject pushed: 900 December rows carry 17.3% of the whole cut, and their mean
under-prediction falls from +229 s to +5. Where it matches, the subject's own
inbound after a fast turnaround, 686 rows carry 3.6% and go from +54 s to −23.
That is the bound doing what the mechanism says, on the rows it speaks about.
Past an hour the gap bounds nothing, and the block still earns there. In
December, 46.4% of the cut sits on gaps over an hour where the previous arrival
was the subject's own type, the subject's own inbound, against 10.1% on the July
three, where the short gaps carry the block instead. What the block reads at
those gaps is the stand's recent history: the inbound's lateness and taxi-in,
the scheduled ground time, and whether the aircraft on the stand was the
subject's own. That is a turnaround-pattern signal, admissible on exactly the
same footing as the bound and a different thing from it. Because that share of
the cut, 46.4% in December against 10.1% on the July three, moves by a factor of
several between the legs, a sentence about what this block *is* has to name the
leg it is true of.

**What survives an operational rule.** The block reads an arrival that
on-blocked *after* the subject's own off-block. Under the challenge's post-hoc
framing (the take-off time is given, and every take-off-anchored window in the
model has the same standing), it is admissible; under an operational one it is
not. This is one of the places in this work where a feature depends on the
challenge's post-hoc framing; the final configuration's features on `AOBT_3_flt`
and its ground-movement correction (§8.7.3, §8.7.4) are the others.
How much of the
block would survive an operational rule is measured rather than argued. A
diagnostic rung, not a step of the ladder and not in the submission, re-anchors
the block on the schedule: it takes the stand's last arrival at or before the
subject's *scheduled* time and reads every column from that choice, so that
nothing in it needs the take-off. It keeps 42% of the block's gain in December,
27% in July over all ten airports, and **4% on the three**, where the
schedule-anchored gain, +0.58 s, is inside that leg's seed floor of 1.02 s and
cannot be told from zero. The schedule anchor is stricter than operational,
since at pushback the arrivals up to the actual off-block are known and not only
those before the schedule, so these figures are lower bounds. On the three,
however, the block is a post-hoc instrument almost entirely, and this section
says so rather than burying it.

**The headroom overshoot.** Before each of the four blocks was built, its likely
gain was estimated from the residuals of a frozen model
(`docs/review_headroom.md`), and a test flags a block whose refit gain beats its
estimate by more than five seconds. On the July three this block takes +16.0 s
against an estimated +10.3, an overshoot of 5.681 s at 1.55 times the estimate,
so the test fires by 0.681 s. That is inside the leg's seed floor of 1.02 s, so
the verdict is within what changing the seed alone produces. The test was built
as a leakage audit and it is not one: the review, written before the test ever fired, established that a booster fitted on a frozen model's
stored residuals bounds a refit from below, so the gap between the two is the
value of an interaction and not evidence of leakage. The leakage test proper is
the source registry and the poison tests of §6.1, and it passes. The block stays
in the model on the reading that an overshoot inside the floor cannot tell
leakage from noise. That reading was adopted after the number was seen, and this
paper says so; what carries the weight is the date of the review, not the
reinterpretation. The other reading is leakage, and it is the one the test
exists for.

The test has also read differently from one regeneration of the ladder to the
next, each time moved by something other than the block: 1.54 times the estimate on the first regeneration of the ladder, firing by 0.56 s; 1.48 on the second, missing by 0.029 s; 1.52 on the third, firing by 0.37 s; 1.42 on the fourth, missing by 0.655 s; and 1.55 under the parameters that ship. The
estimate itself is now regenerable: `reports/headroom.md` recomputes it from the
current residuals at a stated capacity, reads +10.00 s, and draws no conclusion
about the test. What the block rests on is the source registry and the poison
tests of §6.1; the parity of the gap on both sides of the data, where no
airport's share of stands re-taken within twenty minutes moves by more than two
points between 2025 and 2026; the concentration above, which is consistent with
the mechanism and is not proof against another route; and the schedule-anchored
measurement, which puts a number on the part of the block that is the post-hoc
reading.

Three smaller blocks stand on it in the chain (the operator read as the callsign
prefix, the destination's bearing and distance with the holiday calendar, and
the plan's two differences) and are read in §7.2 and §8.4, each net of the
blocks below it.

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

The design has three parts: two forward splits, a held-out-pair split and
pair-frequency strata. In the forward splits, December 2025 at all ten airports
stands in for January 2026, and July 2025 at all ten for July 2026. The three,
Frankfurt, Heathrow and Schiphol, are also read on their own, as a second July reading. The
held-out-pair split is GroupKFold over stand-to-runway pairs. Random K-fold on rows
is never used, because congestion is autocorrelated over hours. The
three-airport reading is kept as it was fixed; any rule below that reads it says
so, and §8 says which July figure it quotes.

The held-out unit is the **pair**, not the stand. Every stand and runway is
still seen in other pairs, so this split bounds what the geometry buys on a new
combination; it does not isolate what the geometry would buy on a new stand,
which no split of this data can measure. The claim is a difference in
differences, what a block is worth on held-out pairs minus what it is worth on
seen ones, so composition cancels. Intervals come from a cluster bootstrap over
pairs, 1,000 draws, resampling stored residuals; no model is refitted. After the
freeze the ladder was also read with every month of 2025 held out in turn. That
diagnostic is not forward in time and decides nothing (§3.5); §8 and the
annexes carry it, and the forward splits remain the proxies for the two test
months.

Source: `reports/baseline.md`, *Held-out pairs*.

#### 7.1.1 What the forward legs cannot see

This subsection gives three limitations of the design above, each measured and
not asserted. None changes the claim of §7.2; each bounds how the forward legs
can be read. The measurements were made under the parameters then in force and
were not repeated under the parameters that ship, so every figure here is read
from the dated record the experiments wrote (§3.7).

*The season columns.* Under the parameters then in force, the rung the base configuration is fitted from was refitted on December and on July
without `tko_month`, `tko_doy_sin` and `tko_doy_cos`, at its first seed and with
everything else unchanged. A rule fixed before the sign was known said no
adoption would follow either way. Removing the columns improved December by
0.40 s and the ranking mix by 0.30, and cost July 1.15 s on the three and 0.77 s
pooled over all ten, so the sign is not the same on the two legs. The
measurement, however, answers a question the ranking months do not ask. Both
forward legs validate on a month and a day-of-year range the training side
never held (December after January to November, and July after January to
June), whereas the ranking months, January 2026 and July 2026, each have their
twin in the training year. The legs therefore extrapolate on these columns
where the final fit interpolates. A held-out-month split, training around a
month and validating inside it, would answer the question the ranking months do
ask, but it breaks the forward discipline every leg here keeps. When this
subsection was written, it named that split as the one that would answer the
question; the split was later run as a diagnostic after the freeze (§3.5,
§8.2), without repeating the season-column experiment on it. The columns stay
in the model.

*Two instruments, and the first time they disagreed.* This paper sizes a difference two ways. The seed floor is root two times the standard deviation of
five single-seed fits of one rung on one leg; it asks how far a figure moves
when nothing but the random seed does. The interval is a day-clustered
percentile interval over the evaluation set, with days as the clusters and the
same draw for both arms; it asks how far the figure would move on another
sample of days. They measure different things, and they can disagree about the
same contrast. They did on the season columns above. Three of the four
differences (December, the ranking mix, July over all ten and the three) were
outside their leg's seed floor, while every interval printed spanned zero:
December all ten at −0.40 s against a floor of 0.34 with an interval of [−1.6,
+0.9], and July all ten at +0.77 against 0.23 with [−0.2, +2.3]. Nothing chose
between the two instruments there, and nothing needed to, because no adoption
followed. They disagreed again over the booster setting, and §7.1.2 gives that occasion.
**Where a figure in this paper is called real or called noise, the sentence says
which instrument said so.**

*Recency, and what the gap leg can and cannot say.* On every 2025 leg the most
recent training month is also the seasonally nearest to the month being scored:
November for December, June for July, September for October. A gain on those
legs therefore cannot separate recency from season. For the January 2026
ranking rows the two agree. For the July 2026 rows they do not: the most recent
training month, December 2025, is seasonally opposite, and the seasonal twin,
July 2025, is five months older. The gap leg was built to break the confound. It
trains on January to June 2025 and scores December, which is the July 2026
geometry as nearly as one training year permits, without the twin month. Two
recency schedules were fixed in advance, with weights halving every 3 and every
12 months over the booster's fitting rows, together with a pass or fail test
fixed at printed precision. Under the parameters then in force, the mild schedule did nothing measurable on the gap leg in either
direction: +0.10 s against that leg's own floor of 0.40 s, with an interval of
[−0.4, +0.6] spanning zero. It gained on the July forward leg past that leg's
floor and with an interval clear of zero, −0.84 s against 0.23 with [−1.5,
−0.4], and −1.61 s on the three against 1.02. It was rejected because the test
compares figures at printed precision, and at that precision its 224.8 s is
worse than the reference's 224.7. **The reason no recency term ships is that
test, not a measured loss on the gap leg.** The strong schedule is a separate
case and should not be read with it: it was worse on October by +0.37 s against
October's own floor of 0.19, a regression the run could measure. The confound
itself stands as a limitation that no leg inside one training year can resolve.

Source: `reports/preregistered-2026-09-20.md`, *What Experiment 2 measured* and
*What Experiment 3 measured*, a dated record of 20 September.

#### 7.1.2 The rules under the parameters that ship, and how those parameters were chosen

*Printed precision, and the seed rule.* Printed precision in this paper is a
tenth of a second, and every seed floor is larger than that: under the
parameters that ship, 0.23 s on December, 0.54 on July over all ten airports,
1.02 on the three and 0.21 on October. A pass or fail test written at printed
precision can therefore trip, in either direction, on a difference the same run
calls seed noise, and in the recency test one did (§7.1.1). That is a property of
the rules as they were fixed; it was not changed after the fact, and the tests
were applied as written. The rule for adopting the seed average, fixed before any measurement under it existed, compares gains with the seed floor, not with rounded figures. Under the
parameters that ship, the five-seed average lowers the error by 1.11 s against
the ladder's single seed in December, where the floor is 0.23, and by 1.46 s in
July against 0.54; it beats every single seed on both legs and passes the rule
with a margin. Under the parameter setting (cell) the second sweep chose, the margin was thin: July gained 0.63 s against a floor of 0.61, and the
report itself says that the standard deviation behind a floor measured from five
seeds is uncertain by about 35%. The setting that ships widened that margin
rather than resting on it.

*The two instruments, a second time.* The floor and the interval disagreed a second time, about the booster cell that ships. Against the cell it
replaced, it reads 0.4654 s better, 2.56 times the seed floor, while its
day-clustered interval, [−0.9492, +0.0009], spans zero by nine ten-thousandths
of a second. The rule that chose the cell reads the floor and nothing else, so
the interval decided nothing; both are printed. It is the second time this work has met such a disagreement, after the season columns of §7.1.1.

*How the booster's parameters were chosen, including the part that was not
planned.* The parameters were tuned on October 2025, a month no rung of the
ladder scores, by two sweeps and one extension of the second, all fitting
January to September 2025. The first sweep ran on the rung that
was then the top of the ladder and under fit rules that have since changed, and
its figures are not comparable with the second's (§9.3). The second sweep was pre-registered and permitted once. It read twenty-seven cells on the rung the base configuration is fitted from under the fit rules that stand, one fit each, and chose
3,000 trees, a learning rate of 0.015, 255 leaves and a minimum child sample of
20, at 214.90 s against the old cell's 217.28 s, a move of 2.38 s against a seed
floor of 0.19 s. That cell sat on an edge of the grid on all four axes. The
pre-registration said the grid would not be extended again, and the edge was
recorded as it required. That rule was reopened after the edge
was recorded and with October's error known to be still falling toward it. The
step was taken with the result in view, and it is stated here as such rather
than presented as the plan. The extension's criteria were fixed in writing
before any figure past the edge existed. Three of the four axes were probed, one
step out each, independently. They were chosen from the sweep's own gradients,
which is a design choice and is disclosed as one; the learning rate was the axis
left out. The trees axis won: 6,000 trees read 214.3041 s against 214.9045,
0.6004 s better, 3.31 times the seed floor, on one leaf setting, and there the
day-clustered interval agreed with the floor. The other two lost, 511 leaves by
0.2274 s and a minimum child sample of 10 by 0.2381 s. At 6,000 the trees axis
was on an edge again, with October still falling at 4,000, 5,000 and 6,000, and
the extension's criteria ended the search there.

*The cell that ships was taken on cost, from two the instrument cannot
separate.* On October the better reading is 6,000 trees at a learning rate of
0.015 and 255 leaves, 214.3041 s, against the shipped cell's 214.4391, and it
was not taken. The two differ by 0.1350 s, 0.74 of the floor, and by less than
half of the 0.2872 s the better cell moves between two draws of its own seed, so
this instrument cannot separate them. That is the situation the
one-standard-error rule of Breiman et al. (1984) exists
for: among candidates the evidence cannot separate, it takes the cheaper. The
cell that ships, 6,000 trees at 0.03 and 127 leaves with a minimum child sample
of 20, builds 762,000 terminal nodes against the alternative's 1,530,000 and
fits October in 237.5 s against 321.8 s. The alternative is named here as the
equal-performing one, so that the choice is visible rather than implied. The
shipped cell is the worse of the two readings, and it is presented as the
cheaper of two cells the evidence cannot separate, not as the better score.

*The winner's curse, measured.* A search that reads twenty-seven cells at one
seed each and keeps the smallest keeps, in part, the luckiest. At a floor of
0.19 s, which is root two times the standard deviation one seed carries, the
smallest of twenty-seven such readings is expected to sit about 0.27 s below its
cell's own level from the selection alone. That figure is an expectation, not a
measurement. It is computed under independent, normally distributed seed noise
with every cell at one level, which is the largest the shortfall can be. The
measurement came before the extension was probed: the four cells within one
floor of the sweep's best were refitted at a second seed, and the sweep's chosen
cell went from first of the four to last, and of the six pairwise orderings
among the four, four reversed. The trees result is read differently, as the
difference between two fits that share a seed, a difference the seed barely
moves. Going from 3,000 to 6,000 trees read −0.6004 s and −0.6019 s at the two
seeds on 255 leaves, and −0.5591 s on 127, while the seed moved the level itself
by 0.2872 s. So the trees effect survives the seed where the ranking of the
cells at 3,000 trees did not.

Sources: `reports/baseline.md`, *Seed averaging* and *The noise floor*;
`src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`, which is the
repository's own account of the sweeps, the extension and the cost decision;
`src/taxiout/provenance.py`, the basis of `FLOOR_JULY_ALL`, for the sweep cell's
July reading. The seed-stability check, the interval on the trees step, and the
order in which the extension's criteria and its figures were fixed are recorded
in the header and the message of `scripts/commit_extension_probe.sh`.

### 7.2 The ladder, and the claim

The ladder reports every rung on both splits, ex-flagged and with the
record-quality term, and the seen-versus-held-out contrast per rung with its
interval. The base configuration is fitted from `+ plan differences`, the last rung,
reached once the four blocks of §6.5 are added. That rung reads 214.7 s
(December) and 322.4 (July) ex-flagged on the ladder's own single seed; §8 gives
the figures of the base configuration. Each rung is read against the rung below it,
and the three versions of the shrunk encoding of §7.4 against `+ weather`, at
233.2 and 344.1.

The claim contrast, identifiers + time → identifiers + time + geometry, is
**+18.5 s [+16.7, +20.5]**: the geometry block lowers the error by 0.4 s where
the model has seen the pair and by 19.0 s where it has not. The interval
excludes zero.

Two neighboring rows are worth reading beside it, because they are what gives
the claim its meaning. The step from the geometric unimpeded term, with the wake
class, to the identifier model is −41.3 s [−47.1, −35.7]: identifiers beat
geometry outright where the pair is known, which is the objection this work has to answer. The step into the geometry block from identifiers is where that
reverses. Together they show that the two encodings are not competing
descriptions of the same thing (one memorizes a pair, the other computes it),
and only the held-out-pair split can tell them apart.

Three versions of the rung above `+ weather` carry a shrunk identifier encoding.
They are the subject of §7.4, and one of them is carried into the base configuration; the claim contrast is measured near the foot of the ladder, where neither
rung carries it.

**What each source of delay is worth.** The ladder read as a table: each row is
one block, added to the model built from the rows above it. The two RMSE columns
are the ladder's ex-flagged figure once the block is in, on the single seed the
ladder fits. The change is that figure minus the one in the row above (negative:
the block made the model better), so a block's change is net of every block
above it in the table and not what it would be worth alone. The held-out-pair
column is the claim instrument of §7.1: how much more the block lowers the error
on pairs the model has never seen than on pairs it has, with its 95% interval. A
change smaller than the seed floor, 0.23 s in December and 0.54 in July, cannot
be told from what changing the random seed alone does; `reports/baseline.md`
lists by name the steps inside that floor.

| block added | what it reads | December (s) | change (s) | July (s) | change (s) | held-out-pair difference [95%] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| identifiers + time | airport, stand, runway, pair; clock and calendar | 324.0 | | 410.9 | | |
| geometry | the routed stand-to-runway distance and route shape | 321.7 | −2.27 | 411.1 | +0.14 | +18.5 [+16.7, +20.5] |
| aircraft | wake class and type | 320.2 | −1.5 | 407.3 | −3.8 | +4.2 [+3.2, +5.3] |
| plan lags | take-off against the plan's off-block estimates and the schedule | 240.9 | −79.3 | 349.9 | −57.4 | −1.4 [−5.8, +3.2] |
| congestion and weather, six blocks | throughput, arrivals, demand, runway configuration, shared corridor, weather flags | 233.2 | −7.7 | 344.1 | −5.8 | |
| shrunk encoding, replacing the stand | the pair's training mean shrunk toward the geometry | 228.3 | −4.9 | 342.2 | −1.9 | −4.4 [−5.8, −3.0] |
| stand occupancy | the last arrival at the subject's stand | 219.7 | −8.60 | 333.0 | −9.12 | +1.8 [+0.8, +2.9] |
| callsign prefix | the operator | 217.6 | −2.11 | 328.4 | −4.64 | +0.7 [−0.8, +2.2] |
| destination and holidays | where the flight goes; the public-holiday calendar | 215.8 | −1.79 | 326.9 | −1.49 | −0.4 [−1.1, +0.4] |
| plan differences (the base configuration) | the plan's own off-block values against each other | 214.7 | −1.14 | 322.4 | −4.55 | −1.0 [−2.5, +0.4] |

The table supports three readings. The plan lags are worth more than every other
block together, and they are not about the place: their held-out contrast has
an interval that includes zero. The geometry is worth almost nothing on a
forward split and is the only block whose worth is concentrated on pairs the
model has never seen. The encoding is the reverse: it gains on the forward
splits and loses on held-out pairs, which is what smoothed memorization does
(§7.4). The congestion blocks are read one by one in `reports/baseline.md`; each
is under two seconds net of the plan lags. §8.4 reads the same table over the
twelve months of the training year held out in turn, with each block's change
pooled over the year and counted against the floor measured in each month.

Sources: `reports/baseline.md`, *The ablation ladder*, *Held-out pairs*, *The
noise floor*, *The shrunk identifier encoding* and *The Gate 2c blocks*.

### 7.3 Exposure to a shift between the training year and the ranking months

Three exposures, each stated as a bound rather than an estimate, and one extrapolation that the final model does not make.

**The plan-lag regime shift.** The ranking months' departures run later against
their plans than their 2025 twins in January, and earlier in July. At Istanbul
the mean of take-off minus the filed estimate moves +288 s in January, at
Frankfurt +224, at Munich +193; in July it moves the other way at every airport,
from −64 s at Madrid to −480 at Paris. The exposed fields cannot decompose that
shift into taxi and pushback delay, but they can bound it, and a control that
reads no plan at all says which it mostly is. Where the January lag moves most,
arrivals' taxi-in barely moves (−20, +9, +16 s at Istanbul, Frankfurt and
Munich), while arrivals' own lateness moves with it (+857, +453, +366). The day
ran late against its plans; the surface did not run slower.

Taking the whole shift as pushback delay anyway (the worst case, which the
control says it is not) and multiplying by each airport's training slope of
taxi-out on that lag (0.24 to 0.51 s per second) gives a per-airport bias of at
most 93 s in January and 132 s in July, costing **+20.3 s at Istanbul, +12.8 at
Frankfurt and +5.6 pooled in January, and +31.0 at Paris, +7.1 at Istanbul and
+5.1 pooled in July.** Biases of opposite sign do not cancel in the pooled
figure, because each enters its own airport's squared error. Paris is the largest July exposure. It is not among the three, and its
July departures take off eight minutes earlier against their filed estimates
than in July 2025, the largest shift at any airport in either month.

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
30% in 2025, and for 65% of Zurich's against 41%; Istanbul went from no freezing
hour to 15%. This is where a model fitted on 2025 extrapolates, and the weather
flags are what it extrapolates along, which is the argument for regime flags
rather than continuous weather drivers.

What a winter month outside the model's experience costs is on the record once
in the training year. Istanbul's February 2025 froze for 22.6% of its hours
against a 2019 to 2024 range of 0.0% to 16.7%, and had precipitation in 39.1%
against 9.1% to 23.5%. Read with that month held out, the shipped model scores
520.8 s there with a bias of −92 s and no row over four hours, against 173.4 to
237.1 s at six ordinary airports over the year (§8.3). The seven February days
above the month's figure had 10 to 20 freezing hours and plan lags of +1,454 to
+2,486 s. January 2026 at the same airport froze for 14.8% of its hours against
that February's 22.6%. Of its 31 days, 3 fall inside those seven days' range of
freezing hours, 10 inside their range of plan lags, and 3 inside both; on
freezing hours, 29 fall inside the range of the ordinary February days. That
comparison is descriptive: the ranges come from 28 days of one airport-month,
and no figure here says what January 2026 will score (Annex G).

**The record-quality term's coverage.** The term is fitted on one airport's 2025
feed. An airport that begins defaulting block times in 2026 is not covered, and
no rule fitted on 2025 could cover it. Pooling the term's cells across airports
would cover more rows, but it assumes that every airport defaults block times
at the same rate. The per-airport rates refute that in every lag-band cell a
pooled rule would serve, in each of the three training sets the legs use, as §4 shows.
What remains is the exposure itself, which is bounded and is stated rather than
turned into a score: 279 ranking rows sit where a pooled rule would have covered
them, 138 of them at Schiphol in January 2026, and each such row at an airport
whose feed starts defaulting would be an error of about a day.

**And one extrapolation the final fit does not make.** The July leg's Heathrow
figures, the afternoon concentration of §9.5 among them, are read from a model
trained on January to June 2025, which holds no July. The final model is fitted on the whole training year, which holds July 2025's afternoons, with the
hour and the runway in its frame. The leg therefore measures an extrapolation
the final model does not make in the same form, and its Heathrow figures are
a diagnostic rather than a bound on the ranking months.

Sources: `reports/congestion.md`, *Regime exposure* and *The weather regime*;
`reports/tails.md`, *The pooled record-quality rule, measured* and *What the
term is, stated plainly*; `reports/residual.md`, *EGLL's July*;
`reports/monthly_ladder.md`, *Annex E: the historic weather context*;
`reports/monthly_seeds.md`, *Per airport and month: the five-seed average*;
`reports/istanbul.md`, *LTFM, February 2025, day by day*, *LTFM, January 2026
(ranking month), day by day* and *January 2026 against February 2025's days*.

### 7.4 A block aimed at one place that paid in another: the shrunk encoding

*The measurement, the decision and the model agree: the version this section
chooses is carried into the rung the base configuration is fitted from, through an
inference-time encoder that computes the encoding once over the training year
and is tested against the same leakage modes as the fold-wise one.*

**What was built, and against what.** §5.4 says where the identifier model has
nothing to memorize, and that the geometry is the better prior where it runs
out. The obvious way to spend that inside the booster is an empirical-Bayes
encoding (Micci-Barreca, 2001; Efron & Morris, 1973) of the (airport, stand, runway) pair: its out-of-fold training mean
shrunk toward the geometric prediction rather than toward a group mean, `(n × ȳ + k ×
ĝ) / (n + k)`, with `k` estimated per airport as the ratio of within-pair to
between-pair variance, the sampling noise of each pair mean subtracted, and the
same construction at stand level as a second feature. It was pre-registered
against a specific bucket: pairs seen one to ten times, where the identifier
exists and its mean is a handful of draws. The ceiling it was built against was
computed before the three versions (arms) were fitted. Each bucket's
airport-mean error was scaled by the ratio that the control, the `+ weather`
rung they are compared against, achieves on the well-observed bucket, and the
difference was called the attainable headroom. As the report computes it under
the parameters that ship, it comes to +0.32 s pooled, all of it in the thin
buckets, none by construction in the bucket that carries 91% of the December split's
rows. The three arms spend the encoding differently: added alongside the
identifier block, at 231.4 s on December and 347.2 s on July; replacing the
stand within it, at 228.3 s and 342.2 s; and replacing both the stand and the
runway, at 228.0 s and 342.1 s. The control reads 233.2 s and 344.1 s.

**What was guarded.** Four leakage modes are guarded, each with a test that
plants the leak and asserts nothing moves. The encoding is recomputed inside
every fold, so a held-out pair sees `n = 0` and encodes to the bare prior. `ȳ`
is out of fold within training too, so a row's own target never enters its own
feature (the single-`k` version failed that test by twelve seconds, which is how
the out-of-fold `k` came to exist). `ȳ` reads unflagged rows only, so one
defaulted block time cannot become a three-row pair's encoding. The one overlap
that cannot be guarded, the prior being fitted on hundreds of thousands of rows
of which a thin pair contributes a handful, is stated rather than left to be
found. `k` was checked for route-length dependence by a rule fixed before the
run. At first, with the encoding fitted on the unflagged training rows, the check read immaterial, at a ratio of 1.29 against a threshold of 1.5.
With it fitted since on the rows every fit reads, the check is material on
the ladder's December leg, at 1.56, so there the pair-level `k` moves with route
length and the stand-level `k` does not. The encoding the final model fits runs
the same check over the whole training year and records its reading in the
manifest written beside the prediction file: 1.60 for `pair`, material, so `k`
moves with the route, and not material for `stand`. What a material reading
does was fixed with the check, in `taxiout.encoding` before it was run: where
the ratio reaches the threshold, `k` varies with route length, and where it
does not, the constant `k` stands. A `k/2`–`2k` sensitivity moves the
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
cannot see the thing under test is not a test of it. It was therefore voided as
a test of the pair encoding, and the arms took no credit from it. They were
decided on the two instruments that can see a pair encoding, the forward splits
and the frequency strata, with the ranking set's own bucket mix as the pooled
figure. It is worth acknowledging that this replacement rule was written after
the numbers were seen, which is a real weakening and is said in the report. The
condition that does the rejecting (no worse than the control on either forward
split) is the original rule's own second half, unchanged, and it is what rejects
the first arm at +3.2 s on July.

**What chooses between the two arms that pass is a mechanism, not their
scores.** §5.3 measures what a designator carries that the routed distance
cannot compute. An intersection departure enters the runway at a taxiway
junction rather than at the threshold, and the signature of that is up to 96 s
per designator and constant within one. It is absorbed by any model that knows
the designator, and it is not recoverable by an encoding shrunk toward a matrix
that routes to the threshold. So the runway categorical stays and only the stand
is replaced, for a reason measured before these versions were fitted and
independent of their numbers. The never-seen bucket corroborates and does not
decide: it is a few hundred rows, and a decision resting on it would be a
post-hoc rule resting on a thin stratum.

**What it bought, and where.** `+ shrunk encoding, replacing the stand` passes
the rule: 228.3 s against 233.2 in December, 342.2 against 344.1 in July, 229.6
against 234.5 at the ranking mix, and +8.1 s worse than the control on the
never-seen bucket against +14.8 for the other passing version. The pooled figure
puts that other version ahead, 229.4 s against 229.6, so choosing the version
with the mechanism costs 0.17 s pooled, and the report states that cost rather
than passing over it. Decomposed by bucket in the sum-of-squares domain, **84%
of the ranking-weighted gain comes from pairs seen more than a hundred times**,
the bucket the ceiling had assigned no headroom at all, against 18% from the
`1-10` and `11-100` buckets together and 1.2% given back on pairs never seen. On
the strata it was aimed at it takes 17.2 s of the 30.8 s of headroom the `1-10`
bucket has, which at that bucket's 0.64% of the December split's rows is worth 0.16 s
pooled. The
honest name for the block is smoothed memorization that buys real score on
well-observed pairs. It is reported under that name, and §9.4 says what the
ceiling that missed it was actually measuring.

Two consequences follow. The prior is `ĝ`, so the encoding is
geometry-bearing, and the ladder's bookkeeping now counts it as such; an
ablation that removed the geometry block and left the encoding in would not have
removed the geometry. And the claim contrast of §7.2 is measured at
`identifiers + time → identifiers + time + geometry`, near the foot of the
ladder, where neither rung carries the encoding, so the claim figure of §7.2 is
untouched by this decision, by construction and not by luck.

Sources: `reports/baseline.md`, *The shrunk identifier encoding* and its
subsections; `reports/migration-2026-09-10.md`, *The encoding's rows,
2026-09-19*, a dated record, for the check as it read before that day;
`tests/test_encoding.py` for the four guards; `src/taxiout/encoding.py`, the
module's docstring, for the length-dependence rule;
`reports/submission-2026-10-11-v12.md`, *The file*, a dated record, for the
final model's reading.

## 8. Results

*This section reports what the validation readings of §3.5 show about the final
model and the models that preceded it, pooled, by month, by airport and by
source of delay, with and without the two rules of §3.3. No upload chose a
feature, a rung or a parameter. One rule decision followed a score; §8.7.2 sets
out what was uploaded, what was read and what was decided. §8.1 to §8.6 read the base configuration, the
booster on which every later change builds. §8.7 sets out each later change in
order, and §8.8 reads the final model step by step.*

### 8.1 The base configuration, and what it scores on the forward splits

The base configuration is the rung named in `taxiout.models.SUBMISSION_KEY`,
`+ plan differences`: the arm §7.4 chose, with the four blocks of §6.5 above
it. It is fitted without the recorded-taxi tail, under five random seeds and
averaged, with the record-quality term on the 383 rows it covers and the
plan-shift rule of §4 on one. Every figure in this subsection is taken from `reports/baseline.md`, under the booster parameters described in §7.1.2. The average
is measured on the pooled figures over the rows the record-quality term leaves
to the model. Every figure with the term, at the ranking mix and per airport on
the forward splits is the ladder's own single seed of the same rung, and the
text says so where it quotes one.

On the December split, which stands in for January 2026 at all ten airports, the base configuration scores **213.55 s** over the rows the record-quality term leaves
to it. On the July split, which stands in for July 2026 at all ten, it scores
**320.90 s**, and 234.20 s on the three. The ten-airport figure is the one that
stands in for the scored month. The three-airport figure is retained beside it as a second July reading (§7.1). §9.5 says what separates the two: one
Fiumicino row. The ladder's single seed of the same rung scores 214.7 s, 322.4
and 235.11 on the same rows, so the average is worth 1.11 s, 1.46 and 0.90. On
that single seed, with the term substituted on the rows it covers, December
reads 234.0 s and July over all ten 368.2; with those rows left to the physical
model alone, 364.8 and 620.5. The gap between the ex-flagged figures and the
every-row ones comes from 88 rows in December and 335 in July,
all at Fiumicino, whose recorded taxi-out no aircraft produced. The term is what
closes it, and §4 says why that is the RMSE-optimal treatment rather than a
trick. Pooled at the ranking set's own bucket mix rather than December's, the
single seed is 216.0 s, and its bias is +7 s in December and −3 in July.

| RMSE, seconds | December, all ten | July, all ten | July, the three |
| --- | ---: | ---: | ---: |
| base configuration, five seeds, rows the term leaves to it | 213.55 | 320.90 | 234.20 |
| the ladder's single seed, same rows | 214.7 | 322.4 | 235.11 |
| single seed, every row, the term on its rows | 234.0 | 368.2 | |
| single seed, every row, no term | 364.8 | 620.5 | |

The first row is the base configuration, read the way the paper reads every model.
The second is the seed every other figure in this subsection is fitted with. The
last two count every row, as the scorer does, with and without the
record-quality term on the 88 December and 335 July rows it covers. The
plan-shift rule touches no December row and one July row, the 87,002 s departure
of §9.5, and §4 and Annex I give what it does there.

Sources: `reports/baseline.md`, *What is fitted, and on what*, *The ablation
ladder*, *Error by pair frequency*, *Seed averaging* and *Per airport*;
`reports/tails.md`, *What it touches in the ranking set*;
`reports/planshift.md`, *Per held-out month and on the two forward legs*.

### 8.2 Every month of the year

Read with each month of 2025 held out in turn and the other eleven training
(§3.5), the same model gives a figure and a floor in every month. The model is
the five-seed average, the base configuration, refitted on each split and scored on
its month. The design is not forward in time, and the forward splits of §8.1
remain the proxies for the two test months. Three readings of every month are
given, as everywhere in this paper: excluding the rows the record-quality term
covers, as the scorer's rule implies; the same over the rows whose recorded
taxi-out is four hours or less, a diagnostic that leaves out the seven rows over
four hours the term does not cover and that the scorer counts (Annex F); and every row
with the term substituted on its rows, which is how the final model predicts. The
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
mean of the monthly figures; every total row in this paper is pooled the same
way. Held-out December is the ladder's December split, since both train on
January to November, and its five seeds are the five of §8.1: 213.6 s here is
the 213.55 s there at the report's precision. The three readings are three
different statements. The first is what the scorer's rule implies for the rows
the model predicts. The gap between the first and the second is seven rows in
the whole year, recorded taxi-outs of 15,059 to 87,341 s at Paris, Fiumicino and
Zurich that no term covers; in May one such row is 51.5% of its month's squared
error, and in January two are 17.6% and 37.4%. The gap between the first and
the third is the term's 1,485 rows, all at Fiumicino, scored at the term's
prediction. The five-seed average beats every one of its five single seeds in
eleven of the twelve months. The exception is February, at 267.93 s against the
best seed's 267.83, inside that month's floor of 1.56 s, the widest of the
twelve.

Sources: `reports/monthly_seeds.md`, *Per month and over the year*;
`reports/monthly_ladder.md`, *Annex F: the rows over four hours*.

### 8.3 Every airport, every month

The table gives the same model per airport and month: RMSE excluding the rows
the term covers, with the margins pooled from the rows they cover. A bracket is
the diagnostic reading over the rows of four hours or less where an airport has
rows over four hours in that month, with the count it leaves out. The bias per
cell is in Annex C.

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

Six airports, Frankfurt, Munich, Heathrow, Schiphol, Barcelona and Madrid, pool
to between 173.4 and 237.1 s over the year. Heathrow is the widest of the six,
and its July, at 301.4 s, is the widest of their months, which §9.5 reads. Three
airports carry the year's day-shift rows: at Paris in January, Zurich in May and
Fiumicino in July, one recorded taxi-out of a day or more decides the cell, and
the bracket beside it says what the cell reads without it. Istanbul's February
2025 is the one cell above 500 s that no single row explains: 520.8 s, with no
row over four hours and a bias of −92 s. That month froze for 22.6% of its hours
against a 2019 to 2024 range of 0.0% to 16.7%, and had precipitation in 39.1%
against 9.1% to 23.5% (§7.3, Annex G). Fiumicino runs from 275.8 to 386.6 s in
its other eleven months, the widest band of the ten, and it is where the last
three blocks of the chain gain the most (§9.5).

On the forward splits, per airport and on the ladder's single seed, the three
(Frankfurt, Heathrow and Schiphol, a third of the test rows) score 203.5 s in
December and 235.1 in July. Frankfurt is 188.4 and 189.7; Schiphol 174.1 and
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
are how many rows of the ranking set each airport holds.

Sources: `reports/monthly_seeds.md`, *Per airport and month: the five-seed
average*; `reports/monthly_ladder.md`, *Annex E: the historic weather context*;
`reports/baseline.md`, *The claim, per airport* and *Per airport*;
`reports/data_review.md`, *Airports and months* and *Inventory*;
`reports/tails.md`, *What it touches in the ranking set*.

### 8.4 What each source of delay is worth, over the year

Read with every month held out, each block's worth becomes a change pooled over
the year and a count of months; §7.2 reads the same ladder on the two forward
splits and on the held-out pairs. The count is read against the floor measured
in that month (§3.6). A block helps in a month when it improves the ex-flagged
figure by more than that month's floor, and hurts when it worsens it by more.
Otherwise it sits inside the floor, which is within what changing the seed alone
produces. Each block is read against the rung the ladder reads it against, so
its change is net of every block below it.

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
months excluding the rows the term covers; the five-seed average of the base configuration is 243.7 s on the same rows (§8.2). The table supports three readings. The
ladder that ends in the base configuration holds up month by month: the plan lags, the
throughput, the stand-replacing encoding, the stand occupancy and the
destination block help in every month, and the aircraft block, the callsign
prefix and the plan differences in eleven. February, the month with the widest
floor, is the one month in which each of those three sits inside it. The
geometry block, worth −1.4 s pooled, helps in nine months and hurts in April;
its worth on a forward split was never the claim, which §8.5 reads month by
month. Two of the congestion blocks are null over the year once the floor is applied: the configuration block sits inside it in seven months and the shared corridor in eleven. The arrivals and demand blocks are small but consistent, helping in eight and seven months and worth 0.3 s each. Together the five congestion blocks and the weather block are worth 4.2 s pooled, net of
the plan lags, of which the weather block is 0.9 and the throughput 2.7. On the
chain fitted without the plan lags the weather block is worth 4.6 s pooled and
16.5 s in January, which is what a plan lag absorbs when it is there (Annex B).

Sources: `reports/monthly_ladder.md`, *Summary: the year, pooled from the twelve
months*; `reports/monthly_seeds.md`, *Per month and over the year* and *Each
block's months, against each month's floor*.

### 8.5 Where the geometry claim holds

The claim of §7.2, that the geometry block buys more where the pair is new than
where it is seen, is read here in each month of 2025 from the same three
held-out-pair folds, which train on the whole year and score their held-out
pairs in every month. The statistic is the claim's own, a fold-mean difference
in differences with its pair-clustered interval, restricted to the rows of the
month in question; the year's row is the figure of §7.2.

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
clears zero in every cell but three: Schiphol in January, and Barcelona in
January and December (Annex D). The block's worth on the forward splits (§7.2)
and over the held-out year, −1.4 s pooled (§8.4), is the other reading of the
same block, and §9.1 says why the two are not in tension. Over the year, the base configuration reads 35.1 s [+13.8, +57.0] worse on a pair it has never seen than
on one it has, and 11.0 s [+5.9, +16.3] worse over the rows of four hours or
less. The seven rows over four hours, each on a pair held out in one fold, are the
difference between the two readings (Annex D).

![The geometry claim in each month of the held-out year](figures/heldout_pairs_by_month.svg)

**Figure 2.** The geometry claim in each month of the held-out year. Each dot is the month's difference in differences with its pair-clustered 95% interval, and the shaded band is the year's interval, +18.5 s [+16.7, +20.5]. The interval clears zero in every month, and the bar at the right marks the middle half of the twelve monthly estimates, so no single month carries the year.

Sources: `reports/monthly_pairs.md`, *Per month of 2025*, *Per airport and
month*, *The shipped rung's gap* and *In how many months the claim holds*;
`reports/monthly_ladder.md`, *Summary: the year, pooled from the twelve months*.

### 8.6 What the challenge score can and cannot show

The pooled test figure is a census of ten airports over two months, weighted by
their departures, with July 2026 holding 192,122 of the 344,841 rows. The §7.3
bounds put the plan-lag regime shift at up to +5.6 s pooled in January and +5.1
in July, most of July's at Paris, and the winter regime at Frankfurt and Zurich
is where the model extrapolates furthest. Three kinds of row decide the rest,
and §8.2 has shown each kind on the training year. The record-quality term
reaches 383 rows, all at Fiumicino, whose July departures are in the scored set; on the held-out year those rows take the figure from 243.7 s to 275.9. A
row over four hours in a cell the term does not cover is scored at its recorded value
and can move the pooled figure by itself: seven such rows in 2025, at three
airports, take the base configuration's held-out year from 217.8 s to 243.7, and January's figure from 231.8 s
to 346.8 on two of them. Finally, one ranking row is predicted by the rule of §4
at 52,550 s, with the two outcomes that section prices; in the final file the
flight-plan window lifts it further (§4). The challenge score is one number over 344,841 rows, a mixture of the three readings above; the base configuration's was 313.169 s. §8.7 gives the files after it.

Sources: `reports/congestion.md`, *What it bounds*; `reports/data_review.md`,
*Airports and months* and *Inventory*; `reports/tails.md`, *What it touches in
the ranking set* and *The plan-shifted departures*; `reports/monthly_seeds.md`,
*Per month and over the year*; `reports/planshift.md`, *The one ranking
departure*; `reports/submission-2026-10-04.md`, *The file*, a dated record.

### 8.7 After the first score: the refinements

#### 8.7.1 From the base configuration to the final one

The base configuration of §8.1, a five-seed LightGBM booster with the two rules of §3.3, was frozen before the first challenge score, 313.169 s, was
read. It was then tested against what it had left out: the two rules of §3.3,
further feature groups, a second learner, the aircraft's own ground movements
and the flight-plan record. Every refinement was read under a written pre-registration, but not every
pre-registration came before the figures it responds to. Four candidates were
defined after figures they respond to had been seen, both rules were restored on
a score (§8.7.2), the adsb.lol source was kept after it failed its gate (§8.7.4),
and the flight-plan window's thresholds were set with an edge other teams had
reported already known (§8.7.6). §9.3 lists each such decision with its date and
its record, and Annex K lists every candidate measured after the freeze with its
verdict. Every pre-registration and amendment written after the first score that
governs a candidate or a configuration is published in `reports/`, with one
exception. The amendment of 10 October, which relaxed two pre-upload checks of the
final file after they had stopped its build, is not published; it changed no
prediction (§9.3). The
challenge returns a count of matched pairs and a score per submission. The
public leaderboard, which shows every team's scores, was also read, on 4 and 6
October; §8.7.2 says where it entered a decision. A final phase then asked for one submission over
four months of 2026, which is the final configuration.

| stage | what it changes from the stage before | challenge score (s) |
| --- | --- | ---: |
| the base configuration | the booster of §8.1, with both rules of §3.3 | 313.169 |
| rules switched off | both rules off: the booster predicts every row | 533.1066 |
| feature groups and a second learner | both rules restored; two feature groups and a CatBoost learner added; the term's rows predicted per row | 282.399 |
| ground-movement correction | a correction from ground movements seen in open ADS-B and OPDI, gated by coverage and by airport | 272.0897 |
| learner refitted, blend per airport | the CatBoost learner refitted with the feature groups, a blend chosen per airport; the correction refitted on top | 269.1368 |
| the final configuration | the final phase's four months; the ground-movement correction refitted on every day of 2025; the flight-plan window | pending the final ranking |

The scores are listed in upload order; each file was fixed before its score
existed. Read in order, lower being better, every refinement the rules adopted
on the training year scored better than the stage before it. The one stage that
set the training year's evidence aside, with the rules switched off, scored far
worse.

#### 8.7.2 The one rule decision that followed a score

Two of the five uploads bear on one question: how to predict the 384 ranking
rows that the two rules of §3.3 take. Before the first upload, an analysis
priced four readings of those rows' recorded values. It judged one implausible
from the public leaderboard: under it, a team predicting those rows as ordinary
departures would have scored far above every score then on the board. On that
analysis, and after the base configuration's score of 313.169 s had been read, a
pre-registration switched both rules off. The file built on it differs from the
base configuration on those 384 rows only. It scored 533.1066 s, which showed
the judgment wrong, and an amendment written after that score restored both
rules as they were fixed before any upload. The same amendment proposed the
classifier of §8.7.3 for the term's rows, which was then adopted on the twelve
months of 2025 alone. Both documents are published in `reports/`. No adoption
decision after them reads a score, though the scores and the public board shaped
which candidates were tried next. The final file applies the same rules, with
that classifier, to all four months, February and June included, which no
upload saw.

#### 8.7.3 Feature groups and a second learner

Nine candidates were each read against the base configuration on the legs of §7.1 and
the twelve held-out months, and those that passed their rule alone were then
read together. Three passed. The first is a feature group built on the network
manager's recorded off-block time, `AOBT_3_flt`, and the queue of departures it
lets the booster count. The second is each flight number's own history in the
training year. The third is a second learner, a CatBoost learner, blended with the
booster at equal weight. Together they are better than the base configuration by
13.81 s on December, by 12.10 on July over all ten airports, and by 8.25 pooled
over the twelve held-out months. A winter layer, a per-airport recalibration, a
gamma-deviance learner, thunderstorm flags from the METAR and row weights toward
the target month did not pass.

`AOBT_3_flt` was first kept out of the features, as a record of the departure
itself. It is published in the ranking set and not blanked, and the features
built on it are this work's own, so it was admitted once the feature group had
passed and before the configuration that uses it was built. On the 383 rows the record-quality term covers,
a classifier of whether a row's block time sits at its schedule replaces the
term's single share per cell. Pooled over the twelve months, its mean squared
error is 17,687,795 s² against the term's 23,585,522.

#### 8.7.4 The ground-movement correction

Two open sources of surface positions were matched to the departures: the
adsb.lol globe history, under the Open Database License, and the flight events
of EUROCONTROL's OPDI. From them come the times an aircraft is first seen on the
ground, first moves and stops, and the same for the other departures around it.

**The adsb.lol source failed the gate its pre-registration set for it, and was
kept.** On two training pilot days, the first ground position an aircraft
shows, the best of the three estimates of its off-block time, lay within 120 s
of the recorded off-block time on 32.73% of the 5,054 covered rows at the seven
airports where coverage reached 30%, against the 50% the gate required. Under
the gate the source stopped there. It was continued after an exploratory check
on the same two days, in which a per-airport linear correction of the
prediction by the time from that first ground position to take-off, fitted on
one day and applied to the other, moved the pooled RMSE from 232.2 to 229.5 s.
The amendment that continued it was written after those numbers and states
itself as a choice made after them, and it leaves the two pilot days out of
every fit and every reading of the correction. The first ground position
therefore does not measure the off-block time to the standard the
pre-registration set, and the source enters the correction only as features
read beside the prediction being corrected.

A second learner, a stacker, reads those times and corrects the out-of-fold
prediction of the configuration with the feature groups and the second
learner. This is the ground-movement correction. Under its rule as first
written, the correction was better pooled, by 3.89 s, but worse in January and
July, the months in which airports that no receiver sees carry most of the
error, so it was not adopted. Applied only to rows a source covers, it is
better by 7.49 s pooled and 9.67 on December, and holds in every month. **That
coverage gate was specified after the rule as first written had failed, with
its figures, the per-airport table among them, already seen**, and its
pre-registration lists them among what existed when it was written. At three
airports, EGLL, LFPG and LIRF (Heathrow, Paris and Fiumicino), it was still
worse on the rows it covered, so a second gate was added: the correction is
kept at an airport in a month only where it beat the uncorrected prediction on
that airport's covered rows in the other eleven months. **That airport gate was
defined after the per-airport figures had been seen**, and its pre-registration
says so. It is better than the coverage gate alone by 2.40 s pooled and 4.75 on
December, and better than the uncorrected prediction by 9.89 pooled. The gate
drops EGLL, LFPG and LIRF in every month and keeps the other seven; no ranking
row at Istanbul (LTFM) is covered, so the correction applies at six airports.
Three candidates read beside the two gates did not pass: the classifier for the
record-quality rows with the same month's arrivals added, a schedule layer for
Fiumicino, and that layer on late departures only, the last defined after the
layer's own figures (Annex K).

#### 8.7.5 The learner refitted, and a choice per airport

Six more candidates were read, five of them fitted on every split. The one with
a mechanism behind it carried the result. The CatBoost learner had been fitted
without the two feature groups the booster had gained; refitted with them, the
blend is better than the stage before by 3.03 s over the twelve held-out months. One booster
per airport and a larger booster both failed July, and the winter layer did
worse on its own January check. The booster's weight in the blend was chosen
among four values, a selection the report discloses. A selector then chose, per
airport and out of fold, among eight of these predictions: the stage before,
the blend at equal weight; the blend at the booster weights 0.3, 0.4, 0.6 and
0.7; the refitted blend; the larger booster of 9,000 trees; and one booster
per airport. For each airport and held-out month it took the one with the
lowest RMSE over the airport's core rows of the other eleven held-out months,
and the stage before unless another was strictly lower. It is better than
that configuration by 2.49 s pooled and 4.52 on December, and for the ranking
rows it takes the refitted blend at nine airports and the old learners at
Fiumicino (LIRF). It should be highlighted that on its forward legs the selector
could see months after the one being scored, and the report says so. The
ground-movement correction refitted on top of it is better than it by 6.59 s
pooled with the coverage gate, and the airport gate adds 2.42; the gate keeps
the same airports. Choosing airport by airport did not beat a better shared
model: the per-airport boosters lost July, and the selector chose the shared
learner everywhere but one airport.

Two last candidates were fixed before their figures: the booster's weight in the
refitted blend, and a larger stacker for the correction. Neither passed, and the
refitted configuration stood until the final phase. Their pre-registration is
published with the others.

#### 8.7.6 The final configuration

The final phase asks for one submission, a file named `<team-name>_final.parquet`,
predicting four months of 2026: January and July, with February and June added.
The final configuration is that submission, uploaded once as
`funny-firefly_final.parquet` on 11 October 2026. It is the refitted configuration, unchanged in every learner, weight, gate, rule and seed, applied to
the four months, and nothing in it was chosen on a figure read on the new rows.
Two changes were made, each under a pre-registration written before any figure
of this entry's own existed for it. First, the ground-movement correction is
refitted, its parameters, features and seed unchanged, on every day of 2025
with an adsb.lol table rather than every second day: 348 days in place of 174.
Second, a flight-plan window is applied, which rests on a measurement of the
training year. Over the 2,062,577 departures of 2025 that carry a flight plan's
off-block time, the recorded off-block time never lies more than 3,606 s from
it. The count does not thin out toward that edge: 4,315 rows lie in the 300 s below
it, against 5,763 in the 300 s before, which is more than half of that count. A record appears to
be associated with a flight plan only inside that window. The window therefore
clips each prediction to the plan's lag within 3,618 s (the edge plus 12 s, the
largest shift of a take-off time this entry measured between two issues of the
ranking file), after everything else,
the two rules included. Tests of records shifted by a day or by a UTC offset
were read under the same pre-registration; the candidate built on them did not
pass, and nothing joined the final configuration from it. **The window and
those tests were prompted by public discussion of the challenge, in other
teams' public write-ups.** Each was taken up only as a property grounded in
documentation: the window in the use that EUROCONTROL's Airport Operator Data
Flow (APDF) specification makes of off-block times to associate records with
flight data, and the shifted records in the next-day example of the Airport
Collaborative Decision Making (A-CDM) manual and in APDF's allowance of times
in UTC or in local time. Each was measured with this entry's own code on its
own 2025 data, with its thresholds fixed in the pre-registration before any
count existed, and no code, threshold or text of another entry was used. **The
edge the other teams reported, 3,606 s, was known when those thresholds were
set**, the validity bound of 3,900 s among them, and this entry's measurement
found the same edge, so the test confirmed on this entry's data a property
others had reported rather than finding it. Two candidates for the final
phase, the surface state read from arriving aircraft's taxi-in times and a
booster held monotone in route length, were fixed before their figures and did
not pass (Annex K). §8.8 reads what each step of the final configuration is
worth.

The final configuration was built twice from a clean tree with identical bytes. Its score
follows the final ranking, and its record is the final submission record in
`reports/`.

#### 8.7.7 The working names, and where to find them

The pre-registrations and reports name each change by a short code, written
before the change had a description. The paper uses the descriptions; this
table gives the code each one carries in the record, and where to read it.

| what the paper calls it | name in the record | where to read it |
| --- | --- | --- |
| the feature group on the Network Manager's recorded off-block time | C1 | `reports/preregistered-2026-10-04.md`, §3; `reports/candidates-2026-10-05.md` |
| the flight number's own history | C4 | the same |
| the CatBoost learner, blended with the booster at equal weight | C3b | the same |
| the classifier for the record-quality rows | C6b | `reports/c6.md` |
| the ground-movement correction | the fusion | `reports/fusion.md` |
| the correction on covered rows only (the coverage gate) | G1 | `reports/fusion.md` |
| the correction kept only at airports where it helped (the airport gate) | G1n | `reports/fusion.md`; `reports/preregistered-2026-10-06-g1n.md` |
| the CatBoost learner refitted with the feature groups of §8.7.3 | C11 | `reports/group-b.md` |
| the per-airport choice of blend | C14 | `reports/group-b.md` |
| the two last candidates | group C | `reports/preregistered-2026-10-07-group-c.md` |
| the correction refitted on every day of 2025 | C17 | `reports/preregistered-2026-10-08-all-days.md`; `reports/v12-ladder.md` |
| the flight-plan window | W | `reports/preregistered-2026-10-09-records.md`, §1; `reports/window.md` |
| the candidate built on shifted records | C20 | `reports/preregistered-2026-10-09-records.md`, §2 |
| the six steps of the final configuration in §8.8 | S0 to S5 | `reports/v12-ladder.md` |

Sources: `reports/preregistered-2026-10-08-all-days.md`,
`reports/preregistered-2026-10-08-four-months.md`,
`reports/preregistered-2026-10-08-final-phase.md` and
`reports/preregistered-2026-10-09-records.md`, dated records; `reports/window.md`,
*The verdict* and *The difference over the training year*; `reports/v12-ladder.md`,
*The steps*;
`reports/submission-2026-10-04.md`, `reports/submission-2026-10-04-v8.md`,
`reports/submission-2026-10-06-v9.md`, `reports/submission-2026-10-07-v10.md` and
`reports/submission-2026-10-08-v11.md` and `reports/submission-2026-10-11-v12.md`,
dated records; `reports/preregistered-2026-10-04-adsb-after-gate.md`, *What
exists as this is written*, and `reports/preregistered-2026-10-06-last-week.md`,
*What exists as this is written*, dated records;
`reports/candidates-2026-10-05.md`, *Section 4's combination*; `reports/c6.md`,
*Mean squared error, s²*; `reports/fusion.md`, *1. The fusion under §5's rule*,
*2. G1, the fusion gated by coverage* and *3. G1n, the fusion gated by coverage
and by airport, out of fold*; `reports/group-b.md`; `reports/fusion-new-base.md`, *G1 against the new
base* and *G1n against G1 (its rule)*.

### 8.8 The final file, read step by step

The final configuration is built in six steps, each adopted under its own rule before this reading
existed. Here every step is read against the one before on the twelve held-out
months of 2025, each month predicted from the other eleven, so that a reader can
see what each part of the submitted file is worth, where, and in which
direction. The design is not forward in time, but it is the only design in which
every step can be read on every month: the forward July leg reads the first
three steps alone, since the ground-movement correction was never fitted on
January to June. Each learner is read at the single seed the stored held-out
fits use. Every value is read as the file writes it, floored at zero and
rounded, with the record-quality term and the plan-shift rule on their rows at
every step.

| step | what it adds | Jan/Jul (s) | Feb/Jun (s) | the four (s) | the year (s) | December leg (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | the booster, with the two feature groups of §8.7.3 | 319.9 | 290.6 | 306.0 | 255.7 | 207.4 |
| 2 | the refitted CatBoost learner, blended in | 318.8 | 290.4 | 305.3 | 255.0 | 206.0 |
| 3 | the per-airport choice of blend: the refitted configuration before its correction | 318.8 | 290.4 | 305.3 | 254.9 | 206.0 |
| 4 | the ground-movement correction, fitted on 174 days: the refitted configuration | 313.1 | 284.7 | 299.6 | 247.3 | 192.6 |
| 5 | the same correction fitted on 348 days | 312.8 | 284.5 | 299.3 | 247.0 | 192.0 |
| 6 | the flight-plan window, last: the final configuration | 303.6 | 284.4 | 294.4 | 244.9 | 192.0 |

Over the four months the challenge ranks, the final configuration reads 294.4 s against 306.0 for
the booster alone. Two steps carry almost all of that, the ground-movement
correction and the window; the other three are each worth less than a second.
Each change below is
the step less the one before, with its 95% interval over days, so a negative
change means the step lowered the error.

![The error after each step](figures/ladder_level.svg)

**Figure 3a.** The error on the four months after each step. Pooled over the four months, the error falls from 306.0 s for the booster alone to 294.4 s for the final configuration. Two steps carry almost all of the fall, the ground-movement correction fitted on 174 days (305.3 s to 299.6 s) and the flight-plan window (299.3 s to 294.4 s).

![What each step changed](figures/ladder_change.svg)

**Figure 3b.** What each step changed on the four months, in seconds. The whisker is the 95% interval over days and the bar is the middle half of the twelve monthly changes. The correction fitted on 174 days lowers the error by 5.67 s [−6.86, −4.76] and the window by 4.91 s [−15.55, −0.09]. The window's interval is wide and its bar narrow because its gain sits in one month: July moved by 17.23 s and no other month by more than 1.53 s. The per-airport choice of blend, −0.05 s [−0.14, +0.04], cannot be told from no change.

**The refitted learner (step 2).** It is better by 0.68 s over the year, with a
95% interval over days of [-0.97, -0.44], and by 1.10 s on January with July. On
February with June it reads 0.21 s better with an interval of [-1.62, +1.00],
which cannot be told apart from no change, because February alone reads worse,
by 0.24 s. The cause is Istanbul: there in February the selector keeps another
learner's prediction, and Istanbul over the four months reads 306.9 s against
the booster's 306.5. The step is a modest, broad gain from a learner that now
sees the off-block and flight-history features the booster gained with the feature groups of §8.7.3.

**The per-airport choice (step 3).** It changes only Fiumicino, where the
selector gives the booster more weight against the CatBoost learner. Over the
year it reads 0.05 s better with an interval of [-0.13, +0.01], which is
indistinguishable from no change. It is in the file because the selector's rule
adopted it as a whole, not for this step.

**The ground-movement correction (step 4).** This is the largest step: 7.69 s
over the year [-8.58, -6.94], 5.70 on January with July, 5.65 on February with
June and 13.37 on December. It works only where a receiver sees the aircraft on
the ground and the airport gate keeps it. Over January with July it moves Munich
from 173.1 s to 147.9, Schiphol from 186.7 to 158.2 and Zurich from 192.4 to
154.6, and leaves Heathrow, Paris, Fiumicino and Istanbul where they were. It
corrects 62,377 of January's 153,706 rows and 89,093 of December's 165,677, and
gains 4.95 s in January against 13.37 in December. The mechanism is direct: the
booster knows the plan and the traffic, and the correction adds when the
aircraft was first seen to move, which no field of the ranking set exposes.

**The correction on every day (step 5).** It is better in every month, by
0.24 s over the year [-0.29, -0.19], 0.57 on December and 0.25 over the four
months. The interval excludes zero in nine of the twelve months. The same model
fitted on twice the days corrects exactly the same rows and is better on them,
most at Schiphol, from 158.2 s to 155.6 over January with July. The gain is
small and consistent: the model was short of days, not of capacity, as the
failure of the larger correction model tested in §8.7.5 had already
suggested.

**The window (step 6).** Over the year it is worth 2.12 s [-5.99, -0.16], and
almost all of it is July: 17.23 s [-49.43, -0.07], where the window moves 25 of
190,713 rows. On February with June it is worth 0.08 s. The interval is wide
because the gain sits on a few rows a day-scale error away from their plan. Over
January with July it moves Fiumicino from 630.6 s to 573.3 and Zurich from
153.5 to 145.1. Read on the core rows, those neither rule covers, the window is
worth 0.27 s over the year and 0.60 in July. Most of its gain falls on the rows
the two rules cover, where most of the rows over four hours sit, and there a
clip to the plan's window bounds an error no model of the taxi itself can.

**Every row and the core rows.** With the rules' rows left out, the final configuration reads
252.3 s over the four months and 217.7 over the year, against 294.4 and 244.9
on every row (the final configuration, not the base configuration's ladder of §8.6). The gap is the tail: the year holds 122 rows over four hours on
every row and 5 on the core rows (the seven of Annex F less the two that the
plan-shift rule covers). The same tail separates the airports. Over the
four months Fiumicino reads 651.5 s with 58 rows over four hours, Paris 432.8
and Istanbul 306.6, while the airports the correction reaches read between
140.3 s (Munich) and 174.5 (Barcelona).

**What this says about 2026.** On 2025, January with July is the harder pair,
303.6 s against 284.4 for February with June, and the reason is January: on the
core rows it reads 324.6 s, against 233.2 for July, 244.4 for February and 203.5
for June. The window narrows the gap between the pairs rather than causing it.
Every step but the last is read out of fold, so its gain is what a fitted part
was worth on months it had not seen. The window is not. Its edge is the largest
distance the training year showed, so the clip cannot move a 2025 prediction
away from its truth, and its 2025 gain holds in 2026 only if 2026's records stay
inside the same window. The correction's gain depends on receivers seeing the
same airports in 2026. None of these figures is a forecast of the challenge
score (§8.6).

Source: `reports/v12-ladder.md`, *The steps*, *Every row: RMSE (s)*, *Every row:
each step less the one before (s), 95% interval over days*, *Core rows: RMSE
(s)*, *Core rows: each step less the one before (s), 95% interval over days*,
*Jan/Jul, every row, RMSE (s)* and *the four, every row, RMSE (s)*.

## 9. Discussion

### 9.1 What the geometry was worth, and where

The forward-split figures and the held-out one describe the same block; they
differ because the splits ask different questions. On the December forward
split the geometry block lowers the error by 2.27 s. On July it raises it by
0.14 s, which is inside the seed floor. On pairs the model has never seen, it
lowers the error by +18.5 s [+16.7, +20.5] more than on pairs it has seen.

![The geometry block on the forward splits](figures/geometry_forward.svg)

**Figure 4a.** What the geometry block adds on the two forward splits. On December it lowers the error by 2.27 s against a seed floor of 0.23 s. On July it raises it by 0.14 s, inside a floor of 0.54 s. A single-seed change has no interval, so the floor is the yardstick.

![The geometry block on seen and held-out pairs](figures/geometry_heldout.svg)

**Figure 4b.** What the geometry block adds on pairs the model has seen and on pairs it has not. It lowers the error by 0.4 s on seen pairs and by 19.0 s on held-out pairs, and the difference, +18.5 s [+16.7, +20.5], is the claim of §7.2, with an interval over pairs that excludes zero. The two panels ask different questions: Figure 4a asks what the block adds on a forward split, and Figure 4b asks where it adds it.

One explanation of the gap is that the ranking months reuse 2025's stands and runways, so the test set has no thin pairs and
cannot reward the physical layer. **That explanation is wrong, and §5.4 is the measurement that rules it out.** Of the ranking rows, 11.0% sit on pairs the
training year saw a hundred times or fewer. The share runs from 2.08% at
Heathrow to 20.48% at Paris and is 11.51% in July against 10.35% in January; 46
stands are new outright; and on pairs seen one to ten times a pair's own mean is
worse than the airport mean by 53 s in December and 45 in July. The gap between
2.27 s and +18.5 s is therefore not a property of the test set. It is a property
of how the two encodings are combined. When a booster is given both the
identifiers and the geometry, it splits on the identifiers wherever a pair has
been seen often enough to be learned. On a forward split that is nearly
everywhere, and the geometry block and its quality flags take what is left:
10.9% of the base configuration's splits. §7.4 is the one attempt to combine them
differently; that version is in the final configuration, and its gain also came
from the often-seen pairs.

One part of that explanation holds and is worth stating. The geometry block
is an injective encoding of (stand, runway), so on a pair seen hundreds of
times a saturated identifier model already knows everything the distance could
tell it, and there the geometry can only lose. The claim was never that geometry
beats memorization where memorization works. It is that memorization runs out,
that the data says where, and that the geometry is what remains when it does;
and, in the leave-one-airport-out fits, that it is the only thing that remains.

The held-out year says the same thing twelve times. The held-out-pair contrast
is positive, with an interval that excludes zero, in every month, from +15.6 s
in May to +20.7 s in September and November, and at every airport in every
month but three cells (§8.5, Annex D). On the same splits the block's change is
−1.4 s pooled over the year, a lower error: it helps by more than the month's
floor in nine months, sits inside it in June and July and hurts in April (§8.4).
A block that is worth a second on the pairs a model has seen and twenty on the
pairs it has not is what the decomposition of §3 predicts, and the two readings
are the same measurement taken on two row sets.

Sources: `reports/baseline.md`, *Held-out pairs*, *The noise floor* and *Where
`+ plan differences` spends its splits*; `reports/geometry_gaps.md`, *Where
memorization runs out*; `reports/monthly_pairs.md`, *Per month of 2025*;
`reports/monthly_seeds.md`, *Each block's months, against each month's floor*;
`reports/monthly_ladder.md`, *Summary: the year, pooled from the twelve months*.

### 9.2 What the record-quality term touches, and why it is kept

The record-quality term touches 383 ranking rows, 107 in January and 276 in
July, all at Fiumicino, and its whole effect on the leaderboard runs through
those rows. Of July's rows, 248 are departures with no flight-plan record and an
ordinary plan lag. The term answers the scoring, not the physics. It lowers the
challenge error on rows where the recorded block time is a schedule artifact, and
a model built only to describe taxi-out would not carry it. The alternative,
predicting only physically plausible taxi-outs, was not taken. The rule of §4 for
the plan-shifted departures takes the same course again, on one ranking row and
three training departures, and §4 gives the score under both of its possible
outcomes rather than defending one; in the final file the flight-plan window
then takes the lag reading for that row (§4). Over the final phase's four
months the term takes 619 ranking rows, 107 in January, 75 in February, 161 in
June and 276 in July 2026, all at Fiumicino, and the classifier of §8.7.3
predicts them in place of the cells.

Sources: `reports/tails.md`, *What it touches in the ranking set* and *What the
term is, stated plainly*; `reports/submission-2026-10-11-v12.md`, *The file*.

### 9.3 Limitations

The open questions recorded in the modeling notes (`docs/modelling_notes.md`)
give the first set of limits: stands OSM
does not have (the join places 96.3% of training departures on a matrix cell,
and most of the rest sit in whole numeric blocks at two airports); the
turn-angle bias of snapped stands, expected on physical grounds and
undetectable by the available proxy; intersection departures at stand level; the OSM snapshot,
fetched unpinned and rebuilt from the map time
recorded for each airport (§10); and one airport's data feed driving the entire
record-quality term. A stand with no mapped apron lead-in is attached to the graph by a
straight line, which carries no turns and no junctions, so the apron maneuver
drops out of exactly the descriptors that describe the route's shape; at Paris
that is 265 of 768 stands. Compared on snapped against unsnapped matrix cells,
turn angle per kilometer moves the wrong way at six of ten airports, Paris among
them, so the bias rests on the mechanism and not on a measurement.

Six more belong here because they concern how the numbers above were produced
rather than what they say.

**The booster's parameters were searched more than once, and the second search
was extended after its own rule said it would not be.** §7.1.2 gives the
account. A first sweep was followed by a second, pre-registered and permitted
once. The second was then extended: it was reopened after its best setting had
been seen to sit on the edge of the grid, with the direction in which the error
was falling already known, and the extension ran under criteria fixed before
its figures existed. A final choice between two cells the instrument cannot
separate was taken on cost. The first sweep's figures are not comparable with
the second's. It ran on the rung that was then the top of the
ladder, which carried the runway saturation block and neither the shrunk
encoding nor the four-hour exclusion. Its grid ran from 235.0 to 244.8 s, and
its top eight cells lie within 0.8 s of each other. Fitting one and the same
parameter setting under the later features and training-row rules lowers its
error from 235.0 s to 217.28 s, before any parameter changes.
`src/taxiout/models.py` keeps both sweeps' accounts beside the parameters.

**The ranking-set figures were recomputed after the challenge replaced the
ranking file and template.** The template grew to 344,841 rows when July
2026 went from three airports to ten; no row was removed. The model was not
chosen on the ranking set, so
nothing in it moved, but every figure read on it did, and the July validation
reading was revisited (§8). A check of the local copies against the challenge's
files should open every piece of work that reads them.

**The held-out year is not a forward reading.** Every month of 2025 held out in
turn trains on months after it as well as before, so the twelve figures of §8.2
and the per-month claim of §8.5 are read inside the training year and not across
its edge. They say how the figures and each block's worth are distributed over
the year; the two forward splits remain the proxies for the test months. The
five-seed average fails to beat its own best seed in one of the twelve months,
February, by 0.10 s inside a floor of 1.56 s. That is the kind of near miss a
rule read at the floor is built to call noise.

**One rule rests on three rows.** The plan-shifted departures of §4 are a class
of three training departures, and the rule for them was written after one such
departure was seen in the ranking set's exposed fields. Its share is an estimate
that three rows cannot pin, and its evaluation on the held-out year is
arithmetic on those rows, not evidence. The paper states both outcomes of the
one row it reaches, in score terms, and the pre-registration is the record that
nothing about the rule was chosen on its effect. In the final file the
flight-plan window overrides the rule on that row (§4), on an edge measured on
the training year that holds the class's July departure.

**Three candidates were measured after the freeze, and none was adopted.** The
runway-saturation block on top of the shipped rung, a recency schedule halving
every twelve months and 8,000 trees in place of 6,000, each measured once before
the freeze and left on the table, were fitted on top of the base configuration and read under conditions fixed in writing before their figures existed.
None met all four conditions: more trees improved December by 0.28 s, more than
its floor of 0.23, and failed July; the saturation block made no measurable
difference on either forward leg; the recency schedule failed all four. The
verdict is the same whether the figures are read unrounded or rounded, and the final configuration is unchanged (Annex J). The measurement shows the ceiling the
booster had reached under its own parameters: the one candidate with a
measurable gain on one leg gave it back on the other.

**What was decided after the numbers were seen.** Sixteen decisions in this work
were taken after some of the figures they bear on had been seen, six before the
first challenge score and ten after it. The table gives each, in the order in
which it was taken, with its date, what had been fixed before, what had been
seen, what was decided and the record that holds it; Annex K gives every
candidate measured after the freeze with its figures and its verdict. The first
row is not one of the sixteen: it is a point a reader may suspect of resting on
a choice, and nothing in it was decided after the numbers. Every other rule the
paper relies on was written down before the figure it governs existed.

| point | date | fixed before the numbers | seen when the decision was taken | decided after the numbers were seen | where it is recorded |
| --- | --- | --- | --- | --- | --- |
| the geometry is worth 2.27 s on a forward split and +18.5 s on held-out pairs (§7.2, §9.1); not one of the sixteen | not applicable | both splits, the held-out-pair design and the claim as a difference in differences | both figures with their intervals | nothing. The gap follows from the model: the geometry is a fixed function of the (stand, runway) pair, so where a pair has been seen hundreds of times the identifiers already carry what it knows, and it can only add where the pair is new | `reports/baseline.md`, *Held-out pairs* |
| the held-out-pair test for the shrunk encoding was set aside rather than passed (§7.4) | early September, before 8 September | the rule: a positive held-out contrast and no worse on either forward split; the intersection signature of each runway designator, measured in another report before the versions were fitted | the versions' figures, and that on a held-out pair the encoding equals its prior exactly, so the gate cannot see the thing it tests, whatever the numbers | the replacement rule's wording. Its rejecting half is the original rule's second half, unchanged; the paper and the report both say it was written after the versions' figures existed | `reports/baseline.md`, *The shrunk identifier encoding* |
| the fields the model had not read were screened on the forward legs that then read them (§6.5, §9.5) | 8 September | the two forward legs, and the rule of reading each block against the rung below it | a small booster fitted on the stored December and July residuals of the rung then shipped, one field group at a time | which blocks to build, among them stand occupancy, the callsign prefix, the destination and the holiday calendar. Each was then read on the same two legs, so its worth there carries the selection; the held-out months other than December and July read them too (Annex B) | `docs/review_headroom.md`, §2 and §6, a dated record of 8 September |
| the seed average's adoption rule was replaced (§3.4, §7.1.2) | 19 September | a gain of at least 1 s on both forward splits | two measurements under that rule, one failing it at 0.78 and 0.98 s and one clearing it at 1.10 and 1.37 s | to retire the rule, because the booster had become repeatable to the bit, and to read the seed floor instead. The replacement was fixed before any measurement under it, and the measurement it read, 1.11 and 1.46 s, also clears the rule it replaced | `reports/baseline.md`, *Seed averaging* |
| the parameter sweep was extended after its own rule said it would not be (§7.1.2) | 22 September | the sweep's grid and its rule of reading October only; the extension's criteria, one step out on each of three axes read against the seed floor, fixed in writing before any figure past the edge existed | the chosen cell on the edge of the grid on every axis, with October still falling toward it | the decision to reopen the sweep, taken knowing that the best setting sat at the high end of the tree count. Disclosed here and in §7.1.2 | `src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`; the extension's working notes are not published |
| the extension's stopping rule was recorded as mis-specified (§7.1.2) | 23 September | the extension's stopping rule, which compared fits made at two seeds | its reading, which set a fit of 6,000 trees at one seed against a fit of 3,000 at another and so mixed what the trees bought with the seed's shift of the level | the rule was recorded as mis-specified, and paired contrasts within one seed were set beside it as its amendment, showing the gain from more trees nearly the same at two seeds and two leaf settings; the tree count was then fixed | `src/taxiout/models.py`, the comment on `BOOSTER_PARAMS` |
| a rule for the plan-shifted departures was added after the freeze (§4) | 2 October | the rule itself, in every term, and its file hash (SHA-256), before it was evaluated on any fold or built; the four-hour threshold it reads, fixed in advance | three training departures in the class, two of them long; one ranking departure, and what the base configuration predicted for it | the decision to add a rule at all, taken after the class was seen in the ranking set; disclosed in §4 with the score under both outcomes | `reports/preregistered-2026-10-02.md` |
| both rules were switched off for one upload (§8.7.2) | 4 October | the analysis that priced the rules' rows under four readings, one of them judged implausible from the public leaderboard, computed before the base configuration's score existed | the base configuration's score, which existed when the decision was written, though no rule read it | to upload one file with both rules off | `reports/preregistered-2026-10-04.md`, *What exists as this is written* and §1 |
| both rules were restored (§8.7.2) | 4 October | both rules, as fixed and validated before any score | the score of the file with both rules off, far worse than the base configuration's (§8.7.1) | to withdraw the decision above and apply both rules in every later file: the one decision taken on a score | `reports/preregistered-2026-10-04-tails.md`, *Why §1 is withdrawn* |
| the adsb.lol source was kept after it failed its gate (§8.7.4) | 4 October | the pilot gate, on coverage and on how close the best estimate came to the recorded off-block time | the gate's failure, and an exploratory check on the two training pilot days | to continue the source into the ground-movement correction as features, with the two pilot days left out of every fit and reading; the gate's verdict stands as a measurement | `reports/preregistered-2026-10-04-adsb-after-gate.md` |
| the network manager's recorded off-block time was admitted (§8.7.3) | 6 October | the feature group and its rule, and that admitting the field would need the author's decision in writing after the verdict | the group's verdict: it cleared | to admit the field, before the configuration that uses it was built | `reports/preregistered-2026-10-04.md`, §3; `reports/candidates-2026-10-05.md`, *What the rule adopts*; the decision itself is in the author's statements, which are not published |
| the row weights toward the target month were corrected for the class ratio (Annex K) | 6 October | the weights, their candidate and its rule | the candidate's failure, with weights set by their clip rather than by their classifier | a corrected weighting, read under the same rule; it failed, and nothing changed | `reports/preregistered-2026-10-06-c8-prior.md`; `reports/candidates-2026-10-07.md`, *What the rule adopts* |
| the ground-movement correction was gated by coverage (§8.7.4) | 6 October | the correction's first rule; the three conditions the coverage gate was then held to | the correction's failure under its first rule in January and July, and its per-airport table | to read the correction on covered rows only; adopted | `reports/preregistered-2026-10-06-last-week.md`, *What exists as this is written* and §2 |
| the correction was gated by airport (§8.7.4) | 6 October | the coverage gate's three conditions, to which the airport gate was held with the coverage gate as its reference | the coverage gate's verdict, and a review of it airport by airport: worse on covered rows at Heathrow, Fiumicino and Paris | an out-of-fold gate by airport; adopted | `reports/preregistered-2026-10-06-g1n.md` |
| Fiumicino's schedule layer was restricted to late departures (Annex K) | 6 October | the layer and its rule | the layer's failure, with a gain on late departures at the schedule and a loss on Fiumicino's other rows | to apply the stored layer to late departures only; it failed all three conditions, and nothing changed | `reports/preregistered-2026-10-06-g1n.md` |
| the flight-plan window's thresholds (§8.7.6) | 9 October | the measurement, the two validity conditions and the clip, before any count of this entry's own | other teams' public write-ups, which had reported the edge at 3,606 s | the thresholds, the validity bound of 3,900 s among them, were set with that edge known; this entry's measurement found the same edge | `reports/preregistered-2026-10-09-records.md`, its provenance paragraph; `reports/window.md`, *The verdict* |
| two pre-upload checks of the final file were amended after they stopped its build (§8.7.6) | 10 October | the checks: every prediction finite and positive, and the January and July predictions equal to a control's | the build stopped on both: on predictions of exactly zero, which the build has written wherever the model's value falls below zero, and on July rows that differed only through the correction's neighboring departures | to count a zero and stop on a negative, and to accept a difference that a neighbor's moved base explains; no prediction changed, and the file is the same bytes before and after | the checks in `reports/preregistered-2026-10-08-final-phase.md`, §2, and `reports/preregistered-2026-10-09-jj-check.md`, §1; the amendment of 10 October, which is not published |

Nothing in the model is synthetic: the one fictional airport in the repository
(§5.2) tests the geometry pipeline and is held out of every feature, fit and
figure by a check. The provenance register classes each constant the code holds
as mechanism, quoted, derived or chosen. A chosen one carries its reason or is
marked as having none on record, rather than given a reason after the fact
(`docs/provenance.md`).

Sources: `docs/modelling_notes.md`, *What the challenge data turned out to
contain* and *Open questions*; `reports/data_review.md`, *Stands*;
`reports/geometry_gaps.md`, *The intersection signature, per designator*;
`src/taxiout/models.py`, the comment on `BOOSTER_PARAMS`;
`reports/ranking-refresh-2026-09-28.md`, a dated record; `reports/baseline.md`,
*The noise floor*, *Held-out pairs*, *Seed averaging* and *The shrunk identifier
encoding*; `docs/provenance.md`; `reports/monthly_seeds.md`, *Per month and over
the year*; `reports/preregistered-2026-10-02.md`, a dated record;
`reports/candidates-2026-10-03.md`, *What was measured*, *The four conditions of
28 September* and *What the rule adopts*; `docs/review_headroom.md`, §2 and §6, a
dated record of 8 September; `reports/preregistered-2026-10-04.md`,
`reports/preregistered-2026-10-04-tails.md`,
`reports/preregistered-2026-10-04-adsb-after-gate.md`,
`reports/preregistered-2026-10-06-c8-prior.md`,
`reports/preregistered-2026-10-06-last-week.md`,
`reports/preregistered-2026-10-06-g1n.md` and
`reports/preregistered-2026-10-09-records.md`, dated records;
`reports/candidates-2026-10-05.md`, *What the rule adopts*;
`reports/candidates-2026-10-07.md`, *What the rule adopts*; `reports/window.md`, *The verdict*.

### 9.4 A ceiling that measured the wrong thing

The headroom estimate of §7.4 was wrong twice, and the two errors are worth more
than the number they were about.

The first was arithmetic. The estimate scales each pair-frequency bucket's
airport-mean error by the ratio the control, the `+ weather` rung, achieves on
pairs seen more than a hundred times, and calls the difference the attainable
headroom. On the `11-100` bucket that difference is negative, **−12.1 s** as the
report now computes it: the control is already ahead of what the ratio would
predict. The first reading of that table described the bucket as "about zero"
but then added its size as if it were positive. That turned a pooled ceiling of
about a third of a second into a claimed one to one and a half seconds. The
generator now computes the pooled figure both ways and prints both, because the
gap between them is the whole of the number: a negative bucket held at zero
gives +0.32 s, the signs taken literally give −0.79 s, and the ceiling is
positive only because of that convention. The error ran in the direction that
made a piece of work already proposed look worth doing, which is the direction
such errors run.

The second is the one that matters. The ceiling was not wrong about what it
measured; it measured the wrong thing. It assumed, by construction, that the
bucket carrying 91% of the December split's rows had no headroom at all (the ratio was defined
from it), and it looked only at the rarely seen pairs the block was designed
for. The block then took 84% of its gain from exactly that bucket. A ceiling
that can only see the target of a feature cannot bound a feature whose mechanism
is not the one it was named for. The honest reading of "smoothed memorization"
in §7.4 is that the mechanism was never the one the ceiling was built to
measure. The lesson is not that the ceiling should have been larger. It is that
a bound computed only where a block is meant to help bounds only that intended
effect, and the report that carries it should say so before the versions are
fitted, not after.

Source: `reports/baseline.md`, *The ceiling this was built against* and *What
the chosen arm is, in its own terms*.

### 9.5 What the residual actually is

The error the model has left is a thin tail of individual holds. Three
measurements say so, and none of them is the absence of a queue model. Every
figure in this section is computed from the stored validation residuals of the
base configuration's rung fitted once, at the ladder's seed; the five-seed average of
§8 stores no residuals.

**Its shape.** On the December split, rows with a residual over ten minutes are
1.78% of the rows and 38.3% of the squared error; over twenty minutes, 0.23% and
17.2%, and four in five of those are under-predictions. On the July leg's three
airports (Frankfurt, Heathrow and Schiphol), the same tail is 1.82% of the rows
and 48.6% of the squared error, with 55% of those rows under-predicted. The
under-predicted rows among those over ten minutes have a median actual taxi-out
of 2,221 s against a median prediction of 1,356 s in December, and 2,597
against 1,695 on the July three. Their ninetieth percentile is 3,466 s in
December and 4,430 on the July three.

![The residual tail](figures/residual_tail.svg)

**Figure 5.** The residual tail. Rows more than twenty minutes off are 0.23% of December's rows and 17.2% of its squared error, and four in five of them are under-predictions. On the three July airports the same rows are 0.33% of the rows, 31.4% of the squared error and two in three under-predictions.

**It is not shared.** The oracle of §6.4 regresses a departure's residual on its
same-runway neighbors' mean residual, a quantity no admissible feature may read.
On the base configuration it would lower the error by 8.2 s in December and 8.3 in
July, and the lag-1 autocorrelation of consecutive residuals under two minutes
apart is 0.19 and 0.09. A queue everyone stands in would show in both. These
misses belong to one aircraft at a time.

**It is enriched where individual holds live.** Heavies are 21.6% of December's
rows and 33.3% of its worst one percent of residuals. Departures without a
Network Manager record are 0.94% of the rows, 4.8% of the worst one percent and
3.6% of the squared error in December, and 1.99%, 11.0% and 14.8% on the July
three, where they run at 641 s RMSE against 235.1 for the three pooled. The
missing record is itself a delay signal, since a plan canceled and refiled after
a long delay loses its join. Departures whose stand was taken by an aircraft of
another type twenty to forty-five minutes before take-off, the rows §6.5's bound
speaks about, are 0.54% of the rows and 3.1% of the worst one percent even after
the block that reads them. Inside that band the block cut the mean
under-prediction from +229 s to +5, which is `reports/occupancy.md`'s
measurement rather than this section's. The error also grows with the plan lag,
as it must when a long lag can mean either an ordinary gate delay or a long hold
and the model can only predict their average. Against 181 s at the
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
of the airport. The second concentration is by record. The 338 rows with no
Network Manager record, 1.6% of July's rows, carry 21.3% of its squared error at
1,138.9 s, which is +857.7 s [+300.7, +1,207.2] above the rows that have one; in
December the same class runs +207.2 s [+71.3, +389.0] above. Both contrasts are
structured, at day-clustered intervals clear of zero.

**Over the year, the last three blocks of the ladder are mostly a Fiumicino
correction.** The callsign prefix, the destination block and the plan
differences take 86.0%, 58.3% and 35.4% of their year-level reduction in squared
error at Fiumicino, every part of it on rows with a flight-plan record. The
first block's change at that airport is −22.1 s, against −3.3 pooled over the
ten. That is the operator's identity doing at one airport what it does nowhere
else, at the airport whose ordinary months run from 275.8 to 386.6 s (§8.3). For
the plan differences the share falls from 35.4% to 26.5% when the one July
day-shift row is left out, so part of that block's year-level worth is one
departure (Annex H). On the forward splits the three blocks are read in §7.2.

**Neither concentration can become a feature.** Heathrow's July afternoons and
the rows without a record are both already visible to the model. The hour of
take-off, the runway, the plan-lag band and the presence of a record are columns the base configuration already reads, and the callsign prefix is the
`+ callsign prefix` rung's own level, so there is no mechanism here the frame
lacks. What the July leg's model could not learn is what July's afternoons at
Heathrow are like, because that leg trains on January to June and holds no July
(§7.3). The oracle of §6.4 bounds what knowing the same-runway sequence
exactly could still buy at Heathrow on July at 9.7 s, and that is the scale any
queue reading of the afternoon would have to live inside. What is not shown is which afternoons: the
interval on the six-hour contrast runs from +40.7 to +249.2 s under day
clustering, which is what a handful of afternoons carrying the effect looks
like. The individual days are not listed, because picking out days after seeing
the errors would be searching for a pattern rather than describing one.

**One row, and which July figure this paper quotes.** Over all ten airports the
July split's RMSE on the rows the record-quality term does not take, at the
ladder's single seed, is 322.4 s. One departure at Fiumicino carries 33.8% of
that figure's squared error. It records a taxi-out of 87,002 seconds, a
day-shift in a cell the record-quality term does not cover, so no term touches
it, and the four-hour exclusion is fit-side only and never reaches the scoring
set. While July 2026 held no Fiumicino row, that was the reason to quote the
three-airport figure, 235.1 s on the same seed, as the July figure, and this
paper did so until the re-issued set was read. July 2026 now holds 15,838
Fiumicino departures (§2.1), and a row of this kind among them is scored at its
recorded value exactly as this one is. The ten-airport figure is therefore the
one §8 quotes, and the three-airport one is kept beside it. The rule that
decided which blocks entered the model was applied to the ten-airport figure as
written in advance, and this section does not revisit it.

What is left is individual and not shared, and no field the base configuration's
screen tested predicts it. Every exposed field the model did not yet read was tested for
predictive value before any feature was built from it (`docs/review_headroom.md`
§2), and the four that carried anything (stand occupancy, the operator read as
its callsign prefix, the destination and the public holidays) are in the model.
That is a bound on the base configuration. It does not bound the final
configuration, whose feature groups of §8.7.3 read the Network Manager's
recorded off-block time, `AOBT_3_flt`, and each flight number's own history,
which the base configuration did not, and lowered the error further. What would move it is information about the hold itself (a regulation,
a de-icing queue, a runway inspection), which the challenge data does not carry.
The open dataset that might carry it, EUROCONTROL's own daily ATFM delay by
airport and cause, was not used, on its portal's notice (§2.2).

Sources: `reports/residual.md`, *The shape of the tail*, *What the tail is
enriched in*, *The error by plan lag*, *The largest rows* and *EGLL's July*, all
computed from the base configuration's stored residuals; `reports/congestion.md`,
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
part of this work an identifier model cannot reproduce. It did not, and could
not, move the test score by much: most test rows sit on pairs the training year
saw often, so the forward splits and the challenge scores read the
identifiers' strength rather than the geometry's (§9.1).

The larger gains came from data the geometry does not need:
the Network Manager's recorded off-block time, a second learner refitted with
it, and aircraft seen on the ground by open receivers (§8.7). The last of these
is constrained by where the receivers listen. Over the training year's fitting
days a source covered 0.480 of Heathrow's departures; over the ranking months it
covered 0.923. On the rows it did cover in 2025 the ground-movement correction
read 237.06 s, worse than the uncorrected prediction's 219.17. The out-of-fold
airport gate therefore dropped Heathrow, and the final configuration does not use the correction there. As the coverage in the test months is nearly double that of
the training year, the correction was fitted on a regime the test months do not
share, and its reading at Heathrow may reflect the sparse coverage of 2025 as
much as the airport itself.

Three directions follow from these limits. First, the correction at Heathrow,
Paris and Fiumicino should be refitted on a year in which the receivers see those
airports as densely as they see the test months, before it is read again.
Secondly, the rows at Fiumicino whose recorded block time sits at the schedule
remain a source of error the exposed fields do not identify well (§4); a source
that records the actual pushback there would resolve them directly rather than
by a share. Finally, the queueing layer, which this work did not build, was
bounded at about eight seconds net of the plan lags (§6.4). Ground events
observe directly what a queue model would have inferred, so that bound should be
measured again with the correction in place before a simulation is reconsidered.

Sources: `reports/fusion.md`, *The rows* and *The gate the ranking rows take*.

## 10. Reproducing this

**What a reader brings.** A reader brings four things to the repository at
release V1.1 (https://github.com/hubell/funny-firefly; archived at 10.5281/zenodo.23320026, Bell, 2026b): the challenge data,
which is not redistributable; the eleven Overpass answers, fetched at their pins
as described below or taken from the data deposit (10.5281/zenodo.23133051, Bell, 2026a); and, from the deposit, the METAR
observations and the two OurAirports tables this work reads. The deposit's
copies are cached fetches recorded with their requests and terms, which a fetch
made later would not return because OurAirports is rebuilt nightly. The final
configuration reads two caches more: the adsb.lol event tables, from the deposit
or fetched again by `scripts/fetch_adsb.py`, and OPDI's flight events and lists.
OPDI is used as an external dataset under the condition of §2.2 and, its site
stating no data license, is not deposited; `scripts/fetch_opdi.py` fetches the
events again. From those and the committed inputs, every figure in this paper is either
regenerated by a named script or quoted from a dated record whose file hash the
provenance register (`docs/provenance.md`) holds, and `scripts/check_paper.py`
tells the two apart. One report is of the second kind by necessity.
`reports/determinism.md` is a measurement campaign under the parameters of its
day, whose records hold arrays derived from the challenge data and are not
redistributed, so a clean checkout cannot regenerate it and is not asked to. The
worked airport of §5.2 needs none of the four: it regenerates on a bare checkout
with no network and no data.

**How the parts rebuild.**

1. **The geometry** is rebuilt from the pinned fetch, as described below.
2. **The held-out year.** The ladder was refitted on every month of 2025 held
   out in turn. Its residuals, the five-seed refits of the base configuration on
   the same splits and the three pair folds refitted for three rungs are stored
   beside the ladder's own. The seven reports of the held-out year that §8 and
   the annexes quote regenerate from those stored files by their named scripts
   without a fit, and the fits themselves repeat by the same scripts.
3. **The ranking file and template** are the versions the challenge later
   replaced. A reader brings those versions, and
   `reports/ranking-refresh-2026-09-28.md` gives the digests of both versions, so
   that the copy in hand can be told from the copy first built on.
4. **The final file.** `make final` builds the final submission at the commit
   that carries its code. A fresh clone at that commit, given the inputs above
   and no network, refitted the sixty models behind the per-airport choice of
   blend (§8.7.5) with the committed code and built the final file byte for
   byte, by the route the README sets out under *v12 from a clean checkout*.

**What repeats.** A fit of the booster repeats to the bit: it is fitted in
LightGBM's deterministic mode with column-wise histograms at a pinned thread
count. `reports/determinism.md` fitted one rung, under the parameters in force
when it was measured, six times in two processes at eight threads and got one
outcome, and the same outcome at two, four and fourteen threads. The seed floors
measured on October that the parameter search of §7.1.2 was read against came
back identical to six places in two separate runs with a commit between them,
and all five single-seed fits of the sweep's cell reproduced to the last digit;
the headers of `scripts/commit_extension_probe.sh` and
`scripts/commit_equivalence.sh` are the record of those two runs. A fresh clone,
with the challenge data and the two tables brought in and the geometry fetched
at its pins, rebuilt the 35 matrix files and the 52 residual files byte for byte
and regenerated fourteen reports in which no figure moved. It passed this
paper's own check against them and built the submission byte for byte, three
times over between the working copy and the clone;
`reports/reproduction-2026-09-27.md` is the record of that run. A regeneration
under the replaced ranking files rebuilt the 52 residual files byte for byte
once more and moved no figure of the validation legs.

**The geometry is rebuilt, not shipped.** The distance matrices are a Derivative
Database of OpenStreetMap under ODbL and are not in the repository: a clean
checkout rebuilds them from a pinned fetch, and the deposit is the other route
to the same files. A fresh fetch returns today's map, so the rebuild reads the pinned fetch.
The committed figures were built from a fetch in which each airport was answered
from the map as it stood at its own moment. Those moments are recorded per
airport in `taxiout.provenance.OSM_PINS`, and
`scripts/fetch_osm.py --pin-from-register --force` fetches each airport at its
own. An Overpass answer's header never shows whether a pin was applied, so the
rebuild is verified by its effect. `--compare` checks each airport's parsed
elements against the answers the matrices were built from, of which the
deposit's copy is identical element for element, and `--control-date` asks the
instance that answered for a date whose answer is known to differ, which shows
that it applies a pin at all. A header can show the opposite, that a pin could
not have been applied, because an instance cannot serve a state it has not
ingested. The client refuses an answer whose database state precedes the pin;
the two fallback instances it knows answered from databases of May to July and
were refused, and every accepted answer came from the primary.

**The paper checks itself.** `scripts/check_paper.py` reads every figure in this
paper and requires it to appear, at the paper's own precision, in the file its
section's source line cites and, where that line names a section of the file, in
that section. A figure found only in a dated record must be listed as historical
with a reason, and one computed from report figures must carry its arithmetic.
It is a check of presence and not of correctness. A short figure can match a
number in a long report by chance, so a passing check certifies that every
figure appears where its section says it comes from, and the match it prints for
each figure is how a reader tells the right number from a coincidence.

Figures 2 to 5 are drawn from the same report tables by
`scripts/report_paper_figures.py`, whose `--check` compares them with the files in
`docs/figures/`. Figure 1 is the worked airport's own drawing.

Sources: README, *Reproducing* and *Data sources*;
`reports/reproduction-2026-09-27.md`, a dated record; `reports/determinism.md`,
*The booster*, a dated record; `reports/ranking-refresh-2026-09-28.md`, a dated
record; `scripts/check_paper.py`.

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

The base configuration's rung is `+ plan differences`; `+ stand occupancy,
schedule-anchored` is a diagnostic variant, not a step of the ladder (§6.5), and
the rungs marked *without plan lags* repeat the congestion steps with the
plan-lag block removed (§6.3). Each rung's
change against the rung it is read against, month by month, is the next annex;
the same per airport and month, every rung, is the report's own Annex A.

Sources: `reports/monthly_ladder.md`, *Summary: every month* and *Annex A:
airport by month, every rung*.

## Annex B: each block's change, month by month, against the floor

Each block's change against the rung the ladder reads it against, excluding the
rows the term covers, in seconds, read against the seed floor measured in that
month on the base configuration's rung (§3.6; the floor is measured on one rung and applied
to every block's change, the same approximation `reports/baseline.md` makes).
**A change in bold sits inside that month's floor**; the year column is read
against the year's floor. A negative change means the block lowered the error. A bracket is the same change over the rows of four
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

## Annex C: the base configuration per airport and month, the bias

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

## Annex D: the geometry claim per airport and month, and the base configuration's gap on unseen pairs

The claim of §7.2 and §8.5, the fold-mean difference in differences of the
geometry block on held-out against seen pairs, restricted to each airport's rows
in each month: the point estimate in seconds, with a note where its 95%
pair-clustered interval does not clear zero, and the year's figure per airport,
which is `reports/baseline.md`'s per-airport claim. A positive figure means the
geometry helps more on unseen pairs than on seen ones. The intervals, the rows and
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

Zurich's year figure, +15.8 s, lies below eleven of its twelve months. The year is the claim computed over the year's rows, not an average of the months. Two Zurich rows with recorded taxi-outs above four hours, one in February and one in May, enter both rungs' errors alike and so shrink the difference between them. In May that row alone brings the held-out error with the geometry block to 604.6 s, against 224.6 s on the rows of four hours or less. Over the rows of four hours or less the year reads +22.1 s [+13.9, +29.0], in line with the months. The claim clears zero on both readings.

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
interval: how much worse the base configuration reads on a pair it has never seen
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
counted by the scorer. For each, the recorded value, the base configuration's
prediction at the ladder's seed, the residual, its share of the squared error of
its airport-month cell and of its whole month (both excluding the term's rows),
and whether it has a flight-plan record. No movement or stand identifier is
printed. These seven rows are the whole difference between each figure of §8 and
its bracketed reading over rows of four hours or less.

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

Three of the seven have a flight-plan record. Two of those are the long
departures the rule of §4 was written on, the Paris row of March and the
Fiumicino row of July, and that rule covers them. The other five are the rows
over four hours that neither the record-quality term nor the rule covers, the
five of the core rows in §8.8. Four of the five have no flight-plan record:
they take off within minutes of schedule with a recorded block time about a day
earlier, and nothing the ranking set exposes marks them.

Sources: `reports/monthly_ladder.md`, *Annex F: the rows over four hours*;
`reports/planshift.md`, *The class in the training year*.

## Annex G: Istanbul's February 2025, and its January 2026

The one airport-month of the held-out year above 500 s that no single row
explains (§8.3), read day by day in the report beside the same days of the other
eleven months, with the airport's hours in each regime, the mean plan lag and
the arrivals' control; and the ranking month January 2026 at the same airport,
for which no error exists and nothing is predicted, read the same way. What
follows is what the report concludes.

February at LTFM pools to 520.8 s over 18,328 departures. Of its 28 days, 7
read above that figure (6, 11, 19, 20, 22, 23, 24) and 21 at or below it. Rows
over 4 h among them: 0, so the second reading over the rows of 4 h or less is
the same figure.

The regime shares of the two calendar months at the airport, against the history
fetched for Annex E (no reports before 2018 at this airport):

| month | history years | freezing, history | freezing, observed | low visibility, history | low visibility, observed | precipitation, history | precipitation, observed |
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

## Annex H: where the last three blocks of the ladder lower the error

The three blocks above the stand occupancy, read against the rung below each,
over the held-out year: their change in RMSE over all ten airports and at
Fiumicino alone, and where each block's year-level reduction in squared error
sits, by airport, by flight-plan record, by plan-lag band and by airline
designator. The report carries every table; here the year-level figures and its
three computed sentences.

| block | airports | `fiumicino.md` (s) [4 h or less] | `monthly_ladder.md` (s) | the two agree |
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
The base configuration's file built before the rule was added predicts **3,120 s** for it, read
from `data/submissions/funny-firefly_v6.parquet` through the template's
identifier, which is not printed. The difference is +49,430 s. What the
departure in question recorded is hidden, and nothing here says which is right.
In the final file the flight-plan window of §8.7.6, applied after the rule,
lifts this row's prediction to the plan's lag less 3,618 s (§4).

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
on the two forward legs* and *What this evaluation can show*;
`reports/window.md`, *The verdict*.

## Annex J: the three candidates measured on the base configuration

Three candidates, each measured once before the freeze and left on the table,
were fitted on top of the base configuration and read under a rule fixed in
writing before any of their figures existed. The first pre-registration named the candidates, the legs, the reference and four conditions.
The second kept all of those, read every figure over the rows that
neither the term nor the plan-shift rule covers, added a fifth condition on the
twelve held-out months, and changed what adoption would mean: a candidate would
be added to the existing model rather than trigger a full refit of the final configuration. Both are
published in the repository as dated records. The candidates: the
runway-saturation block appended to the base configuration's features; row weights
halving every twelve months over the fitting rows; 8,000 trees in place of
6,000. The reference is the shipped rung's own fit on each leg, which reproduced
the ladder's stored December and July residuals to zero and October's printed
figure before anything was read.

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
fifth was not reached, no second seed was fitted, and the submission was not rebuilt. More
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


## Annex K: every candidate measured after the freeze

This annex lists every candidate measured after the base configuration was
frozen, one row each, in the order in which its figures were seen, with the two
decisions taken on figures rather than under a rule fixed before them: the
restoration of both rules on a score, and the continuation of the adsb.lol
source past its failed gate. Each row names the pre-registration that governs
it and says whether its definition was fixed before any of its figures existed
or chosen after figures it reacts to, which is the case for four candidates:
the row weights corrected for the class ratio, the coverage gate, the airport
gate and Fiumicino's schedule layer on late departures. The deciding figures
are quoted from the report the row names; where a result stands only in a
working record that the repository does not publish, the row states the verdict
in words and gives no figure. Two candidates that were to be read only if
another cleared were never measured, and are listed so that every candidate the
pre-registrations name appears here.

Unless a row says otherwise, a candidate is read by the five conditions of
`reports/preregistered-2026-10-04.md`, §4, on the paired contrast against its
reference at the same seed, its RMSE less the reference's in seconds, so that
negative is better: better by more than 0.23 on December and by more than 0.54
on July over all ten airports, not worse by more than 1.02 on the three July
airports (Frankfurt, Heathrow and Schiphol) or by more than 0.21 on October,
and not worse by more than 0.16 over the twelve held-out months, with both
signs kept at a second seed. The ground-movement correction and its gates are
read by three conditions over the days with a source table: better pooled over
the twelve months by more than 0.16, not worse in any month by more than that
month's floor, and better on December by more than 0.23. The candidates for the
record-quality rows are read by mean squared error on the term's rows, each
month predicted from the other eleven. All dates are in 2026.

| what the paper calls it | governed by | fixed before its figures, or chosen after which | date | verdict | the deciding figures, and where they are printed |
| --- | --- | --- | --- | --- | --- |
| the rule for the plan-shifted departures | `reports/preregistered-2026-10-02.md` | the rule fixed before it was evaluated on any fold; the decision to have a rule taken after the class was seen in the ranking set (§4) | 2 October | applied, with no adoption test (Annex I) | over the twelve held-out months it touches three rows, and the RMSE over every row moves from 276.6 to 272.4 s, arithmetic on those rows and not evidence; `reports/planshift.md`, *Per held-out month and on the two forward legs* |
| the runway-saturation block on top of the base configuration | `reports/preregistered-2026-09-28.md`; `reports/preregistered-2026-10-03.md` | fixed before | 3 October | fails | December +0.17, July +0.00, each against a required gain; `reports/candidates-2026-10-03.md`, *The four conditions of 28 September*; Annex J |
| the mild recency schedule | the same | fixed before | 3 October | fails | fails all four, among them the three July airports at +1.04 against 1.02 allowed and October at +0.23 against 0.21; the same report and section |
| 8,000 trees in place of 6,000 | the same | fixed before | 3 October | fails | December -0.28 holds, against a required gain of 0.23; July +0.09 fails, against a required gain of 0.54; the same report and section |
| the upload with both rules switched off | `reports/preregistered-2026-10-04.md`, §1 and §2 | written after the base configuration's score existed, on an analysis computed before it; no rule read the score | 4 October | uploaded once, then withdrawn on its score (next row) | 533.1066 s against the base configuration's 313.169 s over the same pairs; `reports/submission-2026-10-04-v8.md` and `reports/submission-2026-10-04.md`, *The file* |
| both rules restored | `reports/preregistered-2026-10-04-tails.md`, §1 | chosen after the score of the row above, and on it (§8.7.2) | 4 October | adopted: every later file applies both rules, with the classifier for the record-quality rows on the term's rows once it was adopted | the two scores of the row above; `reports/preregistered-2026-10-04-tails.md`, *Why §1 is withdrawn* |
| the adsb.lol pilot gate | `reports/preregistered-2026-10-04.md`, §5, with `reports/preregistered-2026-10-04-adsb-leg.md` | fixed before | 4 October | fails; under the gate the source stops | seven airports reach a covered share of 30% or more; over their 5,054 covered rows the best estimator, the first ground position, lies within 120 s of the recorded off-block time on 32.73% of rows, against the 50% required; `reports/preregistered-2026-10-04-adsb-after-gate.md`, *What exists as this is written* |
| the adsb.lol source continued into the ground-movement correction | `reports/preregistered-2026-10-04-adsb-after-gate.md`, §1 | chosen after the gate's figures and after an exploratory check on the two training pilot days | 4 October | continued as features of the correction, the two pilot days left out of every fit and reading; the gate's verdict stands as a measurement | the exploratory check: a per-airport linear correction by the time from the first ground position to take-off, fitted on one pilot day and applied to the other, moved the pooled RMSE from 232.2 to 229.5 s; the same report and section |
| the schedule share by lag, for the record-quality rows | `reports/preregistered-2026-10-04-tails.md`, §2 | fixed before | 4 October | not adopted: higher than the term in January and July | January 18,146,115 against the term's 16,974,921 s²; July 19,282,867 against 16,724,722 s²; `reports/c6.md`, *The intervals and the conditions* |
| the classifier for the record-quality rows | the same | fixed before | 4 October | adopted; it predicts the term's ranking rows | pooled 17,687,795 s² against the term's 23,585,522, with the interval of the difference wholly below zero and both January and July lower; `reports/c6.md`, *Mean squared error, s²* and *The intervals and the conditions* |
| a winter layer on the prediction | `reports/preregistered-2026-10-04.md`, §3 and §4 | fixed before | 5 to 6 October | fails | December +0.64 and July +0.00, each against a required gain; `reports/candidates-2026-10-05.md`, *C2a, the winter layer* |
| the winter feature block | the same | fixed before | none | not measured: it was to be read only if the winter layer cleared | none |
| a per-airport recalibration | the same | fixed before | 5 to 6 October | fails | December +0.42 and July +0.75, each against a required gain; `reports/candidates-2026-10-05.md`, *C5, per-airport recalibration* |
| the feature group on the network manager's recorded off-block time | the same | fixed before; admitting the field was decided after the group cleared, as §3 of the pre-registration required (§8.7.3) | 5 to 6 October | clears | December -16.62, July -9.21, the three -15.20, October -6.33, the twelve months -9.50; both signs kept at the second seed; `reports/candidates-2026-10-05.md`, *C1, `AOBT_3_flt` and its queue* |
| a gamma-deviance learner, blended with the booster at equal weight | the same | fixed before, its objective chosen by a test fixed in advance | 5 to 6 October | fails | December +13.16, July +30.50, October +1,061.28; `reports/candidates-2026-10-05.md`, *C3a, the variance-matched LightGBM member* |
| the flight number's own history | the same | fixed before | 5 to 6 October | clears | December -0.66, July -1.47, the three +0.17, October -0.09, the twelve months -0.12; both signs kept at the second seed; `reports/candidates-2026-10-05.md`, *C4, the flight's own history* |
| thunderstorm flags from the METAR | `reports/preregistered-2026-10-04-c7c8.md` | fixed before | 5 to 6 October | fails | December +0.25 and July -0.25, each against a required gain; `reports/candidates-2026-10-05.md`, *C7, thunderstorms from the METAR* |
| row weights toward the target month | the same | fixed before | 5 to 6 October | fails | December -0.16, July +2.70, the three +4.05; `reports/candidates-2026-10-05.md`, *C8, the training rows weighted toward the target month* |
| the CatBoost learner, blended with the booster at equal weight | `reports/preregistered-2026-10-04.md`, §3 and §4 | fixed before | 5 to 6 October | clears | December -1.15, July -4.10, the three -4.95, October -2.59, the twelve months -1.03; both signs kept at the second seed; `reports/candidates-2026-10-05.md`, *C3b, the CatBoost member* |
| the configuration with the feature groups and a second learner | the same, §4 | fixed before | 6 October | adopted (§8.7.3) | December -13.81, July -12.10, the three -16.49, October -7.75, the twelve months -8.25; `reports/candidates-2026-10-05.md`, *Section 4's combination* |
| the row weights corrected for the class ratio | `reports/preregistered-2026-10-06-c8-prior.md`, §1 and §2 | chosen after the figures of the row weights toward the target month, whose weights had been set by their clip rather than by their classifier | 6 October | fails; nothing changes | December +0.35, July +2.31, the three +2.30; `reports/candidates-2026-10-07.md`, *C8c, C8's weights corrected for the class ratio* |
| the ground-movement correction on every row, under its rule as first written | `reports/preregistered-2026-10-04.md`, §5, as amended by `reports/preregistered-2026-10-04-adsb-leg.md`, `reports/preregistered-2026-10-04-opdi.md`, `reports/preregistered-2026-10-04-adsb-after-gate.md` and `reports/preregistered-2026-10-04-opdi-tie.md` | fixed before | 6 October | not adopted | pooled -3.89 and December -5.19 hold; January +1.68 against 0.33 allowed and July +3.35 against 0.58 fail; `reports/fusion.md`, *1. The fusion under §5's rule* |
| the coverage gate: the correction on covered rows only | `reports/preregistered-2026-10-06-last-week.md`, §2 | chosen after the figures of the row above, its per-airport table among them | 6 October | adopted | pooled -7.49, December -9.67, no month worse than its floor; beside the rule, worse on covered rows at EGLL by +8.52, LFPG by +0.77 and LIRF by +10.71; `reports/fusion.md`, *2. G1, the fusion gated by coverage* |
| the classifier for the record-quality rows, with the same-month arrivals | the same, §2 | fixed before | 6 October | not adopted: lower than the classifier pooled, and not higher in January or July, but the interval of the pooled difference did not lie wholly below zero | no figure: the result stands in a working record that is not published |
| Fiumicino's schedule layer | the same, §2 | fixed before | 6 October | not adopted | worse than the base pooled by 0.20 s, missing December and failing July and August; `reports/preregistered-2026-10-06-g1n.md`, *What exists as this is written* |
| the airport gate: the correction kept only at airports where it helped, out of fold | `reports/preregistered-2026-10-06-g1n.md`, §1 | chosen after the figures of the coverage gate, Fiumicino's layer and the classifier with arrivals, a review of the coverage gate airport by airport among them | 6 October | adopted | against the coverage gate, pooled -2.40 and December -4.75, no month worse than its floor; against the uncorrected prediction, pooled -9.89; `reports/fusion.md`, *3. G1n, the fusion gated by coverage and by airport, out of fold* |
| Fiumicino's schedule layer on late departures only | the same, §2 | chosen after the figures of Fiumicino's schedule layer, which gained on late departures at the schedule and lost on Fiumicino's other rows | 6 October | not adopted: all three conditions fail | no figure: the result stands in a working record that is not published |
| the winter layer on the winter months | `reports/preregistered-2026-10-06-last-week.md`, §3 | fixed before | 6 to 7 October | fails | January +0.58 against a required gain of 0.47; December +0.65 against 0.23 allowed; the twelve months +0.16 against 0.15 allowed; the December leg +0.65 against 0.23 allowed; `reports/group-b.md`, *W, C2a's winter layer on the base* |
| the winter feature block on the winter months | the same | fixed before | none | not measured: it was to be read only if the winter layer cleared | none |
| the booster's weight in the blend | the same, §3 | fixed before; the weight is a selection among four, as the note fixed | 6 to 7 October | clears at 0.7; it enters only through the per-airport choice, which takes it at Fiumicino | December -2.86, July -0.96, the three -1.06, October -0.53, the twelve months -1.67; `reports/group-b.md` |
| the CatBoost learner refitted with the feature groups of §8.7.3 | the same, §3, and `reports/preregistered-2026-10-06-airports.md`, §1 | fixed before | 6 to 7 October | clears; it ships through the per-airport choice at nine airports | December -4.32, July -2.10, the three -3.20, October -0.45, the twelve months -3.03; `reports/group-b.md` |
| 9,000 trees with the feature groups | the same | fixed before | 6 to 7 October | fails | July -0.05 against a required gain of 0.54; `reports/group-b.md` |
| one booster per airport | `reports/preregistered-2026-10-06-airports.md`, §2 | fixed before | 6 to 7 October | fails | July -0.23 against a required gain of 0.54; `reports/group-b.md` |
| the per-airport choice of blend | the same, §3 and §4 | fixed before; on its forward legs, months after the leg's month informed the choice, as the note fixed and disclosed | 6 to 7 October | adopted, as the combination | December -4.52, July -1.91, the three -3.20, October -0.59, the twelve months -2.49; `reports/group-b.md` and *The combination* |
| the correction and both gates refitted on the per-airport choice | the same, §4 | fixed before | 7 October | both hold; the airport gate ships | the coverage gate against the new base, pooled -6.59 and December -8.05; the airport gate against the coverage gate, -2.42 and -4.96; `reports/fusion-new-base.md`, *G1 against the new base* and *G1n against G1 (its rule)* |
| the booster's weight in the refitted blend | `reports/preregistered-2026-10-07-group-c.md`, §1 | fixed before; a selection among four weights, as the note fixed | 7 October | not adopted: the weight it selected failed the December and July conditions, and its July contrast changed sign at the second seed | no figure: the result stands in a working record that is not published |
| a larger stacker for the correction | the same, §2 | fixed before | 7 October | not adopted | 0.11 s better pooled, short of the 0.16 s floor; `reports/preregistered-2026-10-08-all-days.md`, *What exists as this is written* |
| the correction refitted on every day of 2025 | `reports/preregistered-2026-10-08-all-days.md`, §1 and §2, its §3 replaced by `reports/preregistered-2026-10-08-four-months.md`, §1 | fixed before | 8 October | adopted | the rule's own reading, over the days with an adsb.lol table, stands in a working record that is not published; over the core rows of the held-out year the step reads -0.27, and on December -0.60; `reports/v12-ladder.md`, *Core rows: each step less the one before* |
| the surface state from arrivals' taxi-in times | `reports/preregistered-2026-10-08-c18-c19.md`, §1 and §3 | fixed before | 9 October | not adopted: it failed the December and July conditions; the twelve-month condition held, but that gain lay in one airport and month, Istanbul in February, where the candidates' blend differs from the shipped base's, and not in the candidate | no figure: the result stands in a working record that is not published |
| the booster held monotone in route length | the same, §2 and §3 | fixed before | 9 October | not adopted, on the same conditions and for the same reason as the row above | no figure: the same |
| the flight-plan window | `reports/preregistered-2026-10-09-records.md`, §1 | fixed before any count of this entry's; the edge other teams had reported was known when the thresholds were set (§8.7.6) | 9 October | valid; applied last in the final configuration | the largest difference, 3,606 s, at most 3,900 s; 4,315 rows in the 300 s below it against 5,763 in the 300 s before, so at least half of the latter, 2,881.5; `reports/window.md`, *The verdict* |
| the tests of records shifted by a day or by a UTC offset | the same, §2 | fixed before; the offset test was read as worded after its count had been seen, and before any figure of the row below | 9 October | both survive, so the candidate below was fitted: the day shift in the UTC calendar, at its bound; the offset only on rows whose UTC offset is zero | no figure: the result stands in a working record that is not published |
| the candidate built on shifted records | the same, §2 | fixed before | 9 October | not adopted: the pooled squared error over the twelve months rose, and held-out June and the July leg were worse | no figure: the same |

Every pre-registration and amendment named in the table is published in
`reports/`. Seven results rest only on working records the repository does not
publish: the classifier with the same-month arrivals, Fiumicino's layer on late
departures, the booster's weight in the refitted blend, the two candidates for
the final phase, the tests of shifted records and the candidate built on them.
For those the table gives the verdict and no figure, and the correction refitted
on every day of 2025 is read from the ladder of §8.8 rather than from its rule.

Sources: `reports/planshift.md`, *Per held-out month and on the two forward
legs*; `reports/candidates-2026-10-03.md`, *What was measured*, *The four conditions
of 28 September* and *Every fit*; `reports/submission-2026-10-04.md` and
`reports/submission-2026-10-04-v8.md`, *The file*, dated records;
`reports/preregistered-2026-10-04-adsb-after-gate.md`, *What exists as this is
written*; `reports/c6.md`, *Mean squared error, s²* and *The intervals and the
conditions*; `reports/candidates-2026-10-05.md`, *C2a, the winter layer*, *C5,
per-airport recalibration*, *C1, `AOBT_3_flt` and its queue*, *C3a, the
variance-matched LightGBM member*, *C4, the flight's own history*, *C7,
thunderstorms from the METAR*, *C8, the training rows weighted toward the
target month*, *C3b, the CatBoost member* and *Section 4's combination*;
`reports/candidates-2026-10-07.md`, *C8c, C8's weights corrected for the class
ratio*; `reports/fusion.md`, *1. The fusion under §5's rule*, *2. G1, the
fusion gated by coverage* and *3. G1n, the fusion gated by coverage and by
airport, out of fold*; `reports/preregistered-2026-10-06-g1n.md`, *What exists
as this is written*; `reports/group-b.md`; `reports/fusion-new-base.md`, *G1 against the new base* and *G1n
against G1 (its rule)*; `reports/preregistered-2026-10-08-all-days.md`, *What
exists as this is written*; `reports/v12-ladder.md`, *Core rows: each step less
the one before*; `reports/window.md`, *The verdict*.

## References

adsb.lol. (n.d.). *Globe history* [Data set]. https://github.com/adsblol

Angrist, J. D., & Pischke, J.-S. (2009). *Mostly harmless econometrics: An empiricist's companion*. Princeton University Press.

Bell, H. (2026a). *Data caches for taxi-out prediction at eleven European airports: OpenStreetMap stand-to-runway distance matrices, METAR observations, airport tables and adsb.lol event tables (team funny-firefly, EUROCONTROL PRC Data Challenge 2026)* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.23133051

Bell, H. (2026b). *funny-firefly: Taxi-out prediction from OpenStreetMap geometry (PRC Data Challenge 2026), code, paper and reports* (Version V1.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23320026

Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and regression trees*. Wadsworth.

Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. *The Review of Economics and Statistics, 90*(3), 414–427.

Douglas, D. H., & Peucker, T. K. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. *Cartographica: The International Journal for Geographic Information and Geovisualization, 10*(2), 112–122. https://doi.org/10.3138/FM57-6770-U75U-7727

Efron, B., & Morris, C. (1973). Stein's estimation rule and its competitors: An empirical Bayes approach. *Journal of the American Statistical Association, 68*(341), 117–130.

Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.

EUROCONTROL. (n.d.). *Open Performance Data Initiative (OPDI), version 0.0.2* [Data set]. https://www.eurocontrol.int/performance/data/download/OPDI/v002/

EUROCONTROL Performance Review Commission. (n.d.). *PRC Data Challenge 2026*. https://ansperformance.eu/study/data-challenge/dc2026/

Idris, H., Clarke, J.-P., Bhuva, R., & Kang, L. (2002). Queuing model for taxi-out time estimation. *Air Traffic Control Quarterly, 10*(1), 1–22.

Iowa State University. (n.d.). *Iowa Environmental Mesonet: ASOS, AWOS and METAR observations* [Data set]. https://mesonet.agron.iastate.edu/

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. In *Advances in Neural Information Processing Systems* (Vol. 30, pp. 3146–3154). Curran Associates.

Lee, H., Malik, W., & Jung, Y. C. (2016). *Taxi-out time prediction for departures at Charlotte airport using machine learning techniques* (AIAA Paper 2016-3910). American Institute of Aeronautics and Astronautics.

Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification and prediction problems. *ACM SIGKDD Explorations Newsletter, 3*(1), 27–32. https://doi.org/10.1145/507533.507538

Olbricht, R. (n.d.). *Overpass API* [Computer software]. https://github.com/drolbr/Overpass-API

Olive, X. (2019). traffic, a toolbox for processing and analysing air traffic data. *Journal of Open Source Software, 4*(39), Article 1518. https://doi.org/10.21105/joss.01518

Olive, X., Waltert, M., & Schultz, M. (2025). *Assessing airport surface traffic performance from open sources of aviation data* [Paper presentation]. Air Transportation Research and Development Symposium. https://www.unibw.de/lvk-en/publications/2025_zhr_surface

OpenStreetMap contributors. (n.d.). *OpenStreetMap* [Map data]. https://www.openstreetmap.org/copyright

OurAirports. (n.d.). *OurAirports open data* [Data set]. https://ourairports.com/data/

Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. In *Advances in Neural Information Processing Systems* (Vol. 31). Curran Associates.

Ravizza, S., Atkin, J. A. D., Maathuis, M. H., & Burke, E. K. (2013). A combined statistical approach and ground movement model for improving taxi time estimations at airports. *Journal of the Operational Research Society, 64*(9), 1347–1360.

Ravizza, S., Chen, J., Atkin, J. A. D., Stewart, P., & Burke, E. K. (2014). Aircraft taxi time prediction: Comparisons and insights. *Applied Soft Computing, 14*, 397–406. https://doi.org/10.1016/j.asoc.2013.10.004

Schäfer, M., Strohmeier, M., Lenders, V., Martinovic, I., & Wilhelm, M. (2014). Bringing up OpenSky: A large-scale ADS-B sensor network for research. In *Proceedings of the 13th International Symposium on Information Processing in Sensor Networks* (pp. 83–94). IEEE.

Simaiakis, I., & Balakrishnan, H. (2016). A queuing model of the airport departure process. *Transportation Science, 50*(1), 94–109. https://doi.org/10.1287/trsc.2015.0603 

Vacanza Team. (n.d.). *Open World Holidays Framework* (Version 0.104) [Computer software]. https://doi.org/10.5281/zenodo.14884702
