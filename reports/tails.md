# The record-quality term

Gate 1 of [`../docs/plan.md`](../docs/plan.md), the third term of the
decomposition in [`../docs/modelling_notes.md`](../docs/modelling_notes.md):
what the tail of the target is made of, which cells carry it, what the
term predicts for them, and what it is worth.

**This report has two generators, and each section says which wrote it.**
Everything above *The pooled record-quality rule, measured* is written by
`scripts/report_tails.py` from the data under `data/challenge/` on 2026-10-03,
and every figure in it is computed there. That section is written into this
file afterwards by `scripts/pooled_tails.py --report`, carries its own date,
and its figures are computed there; `make tails` runs the two in that order.
Before this header said so, the file claimed one generator and "every figure
below is computed there", which stopped being true the day it gained the
second. Aggregates only in both: no identifier from the challenge data
appears, and both generators refuse to write one.

The physical model the term sits on is not fitted here. The "with and
without" figures use a stand-in -- the training mean per (airport,
stand, runway), airport mean as fallback, fitted with the flagged rows
excluded -- so the comparison has a concrete baseline;
[`baseline.md`](baseline.md) carries it for the real model.

## In one paragraph

The tail is one airport's data feed. 82.2% of the
training departures recorded above two hours are at LIRF, and for
89.2% of those the block time sits within a minute of
the schedule while the take-off is hours later: the block time was
defaulted, and the recorded taxi-out is the schedule lag, which every
ranking departure exposes. The term covers the
3 (airport, record, lag-band) cells where that happens at a
material rate -- all at LIRF -- and predicts, for
a row in one of them, the lag weighted by the cell's defaulting rate plus
the cell's ordinary mean weighted by the rest. On the December 2025
validation month those are 88 rows of 165,677 carrying
39.2% of the airport-mean squared error, and the stand-in physical
model scores 447 s without the term and 341 s with it;
on July, 335 rows of 190,713 carrying 57.8%, and
689 s against 461 s.
In the ranking set it predicts for 383 rows, 107 in January and 276 in July.
It is the RMSE-optimal prediction under the stated scoring rule for
rows whose recorded value no aircraft produced, and nothing else.

## What the tail is made of

584 training departures record a taxi-out above two hours.
82.2% of them are at LIRF, and
82.4% have no flight-side record. For
74.0% of them -- 89.2% at LIRF
-- the recorded off-block time sits within a minute of the *scheduled*
departure time while the take-off is hours later. The block time was
defaulted to the schedule, so the recorded taxi-out is take-off minus
schedule: the *lag*, which the ranking set exposes for every departure.

Share of LIRF's over-two-hour departures without a record whose block
time is within a given distance of the schedule (439 rows):

| `|block - schedule|` at most (s) | share |
| --- | ---: |
| 0 | 10.3% |
| 5 | 82.9% |
| 30 | 90.2% |
| 60 | 90.2% |
| 300 | 90.7% |
| 1,800 | 91.3% |

`taxiout.tails.defaulted_block` takes the 60 s threshold (the schedule is
minute-resolution) together with a lag above two hours, which is above
every plausible taxi-out and below every artifact. It picks out
432 training rows, 99.1% of them at LIRF and
91.9% without a record; their taxi-out minus lag has
p10/p50/p90 of -5/0/5 s, and 91.4% of them are within 5 s.

A second, smaller population is the day-shift the data review
described: 14 over-two-hour rows that are not defaulted
sit between 20 and 28 hours, 85.7% of them at LIRF,
a median 951 s above one day -- a block time from
the previous day plus an ordinary taxi. The term does not model these
separately: they fall into the same cells and are carried by the cell's
mean over its non-defaulted rows.

Over-two-hour departures per airport, by kind:

| ICAO | rows > 2 h | no flt record | defaulted block | day-shift | other |
| --- | ---: | ---: | ---: | ---: | ---: |
| EGLL | 37 | 29.7% | 3 | 0 | 34 |
| EHAM | 3 | 33.3% | 0 | 0 | 3 |
| LFPG | 37 | 18.9% | 1 | 1 | 35 |
| LIRF | 480 | 91.5% | 428 | 12 | 40 |
| LSZH | 6 | 66.7% | 0 | 1 | 5 |
| LTFM | 21 | 90.5% | 0 | 0 | 21 |

## The cells

A test departure exposes whether the flight-side join found a record
(`LOBT_flt` null) and its lag, banded at the edges of the data review's
lag table: `normal` is [-1 h, 4 h], `4-12h` is (4 h, 12 h], `>12h` and
`<-1h` are open. With the airport, that is a cell. Training rows per
cell and the mean taxi-out in it (rows / mean, seconds):

| ICAO | `flt:normal` | `flt:4-12h` | `flt:>12h` | `flt:<-1h` | `noflt:normal` | `noflt:4-12h` | `noflt:>12h` | `noflt:<-1h` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EDDF | 227,171 / 862 | 602 / 1,056 | 342 / 1,118 | 70 / 814 | 1,892 / 968 | 39 / 1,129 | 18 / 1,409 | 7 / 812 |
| EDDM | 166,138 / 810 | 139 / 1,007 | 23 / 989 | 24 / 710 | 995 / 1,036 | 10 / 1,107 | 3 / 881 | 2 / 748 |
| EGLL | 237,668 / 1,363 | 458 / 1,450 | 2 / 1,192 | 4 / 1,178 | 1,374 / 1,571 | 40 / 1,464 | -- | -- |
| EHAM | 242,715 / 788 | 721 / 933 | 106 / 1,049 | 325 / 470 | 3,977 / 594 | 77 / 824 | 14 / 1,212 | 16 / 406 |
| LEBL | 177,457 / 956 | 373 / 992 | 12 / 1,130 | 1 / 691 | 1,812 / 1,043 | 47 / 1,066 | 2 / 888 | 1 / 394 |
| LEMD | 210,749 / 1,014 | 530 / 1,094 | 7 / 875 | 9 / 753 | 932 / 1,112 | 14 / 1,393 | -- | 1 / 822 |
| LFPG | 235,010 / 1,016 | 733 / 1,163 | 30 / 1,170 | 6 / 1,034 | 3,671 / 1,189 | 88 / 1,039 | 8 / 997 | 6 / 150 |
| LIRF | 158,812 / 1,144 | 381 / 1,252 | 2 / 43,858 | 21 / 818 | 1,358 / 4,253 | 98 / 20,097 | 29 / 67,859 | 3 / 1,422 |
| LSZH | 132,446 / 741 | 146 / 957 | 2 / 696 | 22 / 775 | 2,274 / 681 | 12 / 742 | -- | 5 / 248 |
| LTFM | 267,602 / 1,049 | 1,642 / 1,068 | 25 / 1,321 | 51 / 877 | 3,602 / 1,281 | 41 / 3,109 | 1 / 727 | 1 / 727 |

Share of each cell's rows with a defaulted block time, for the cells
that could carry one; the `flt:normal` cell reaches 0.0% at its highest airport, which is why it is not among them):

| ICAO | `noflt:normal` | `noflt:4-12h` | `noflt:>12h` | `noflt:<-1h` | `flt:4-12h` | `flt:>12h` | `flt:<-1h` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EDDF | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| EDDM | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| EGLL | 0.1% | 0.0% | n/a | n/a | 0.0% | 0.0% | 0.0% |
| EHAM | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| LEBL | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| LEMD | 0.0% | 0.0% | n/a | 0.0% | 0.0% | 0.0% | 0.0% |
| LFPG | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| LIRF | 21.7% | 84.7% | 62.1% | 0.0% | 0.0% | 50.0% | 0.0% |
| LSZH | 0.0% | 0.0% | n/a | 0.0% | 0.0% | 0.0% | 0.0% |
| LTFM | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

A cell is covered by the term when it has at least 5 rows and
a defaulted share of at least 5%; the rule is justified
below. 3 cells qualify, all at
LIRF, 1,485 training rows
between them. For a row in a covered cell the term predicts
`share x lag + (1 - share) x rest_mean`, where `rest_mean` is the cell's
mean over the rows that were not defaulted:

| ICAO | cell | rows | defaulted | cell mean | rest mean | prediction |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| LIRF | `noflt:4-12h` | 98 | 84.7% | 20,097 | 1,127 | 0.847 x lag + 172 |
| LIRF | `noflt:>12h` | 29 | 62.1% | 67,859 | 87,536 | 0.621 x lag + 33,203 |
| LIRF | `noflt:normal` | 1,358 | 21.7% | 4,253 | 2,784 | 0.217 x lag + 2,179 |

Every other row -- every row at the other airports, and every row with
a flight-side record -- gets no prediction from the term and is left to
the physical model, except the plan-shifted departures, which have a
section of their own below.

## Why this form, and not the others

Forward splits as fixed in `docs/modelling_notes.md`: train January to
November 2025 and validate December at all ten airports; train January
to June and validate July, for all airports and for the three July test
airports alone. The physical model is the stand-in described at the top,
fitted with the adopted flags excluded; each design supplies the term
for the rows it covers and the physical model predicts the rest. RMSE
over every validation row, in seconds:

| design | Dec 2025 | Jul 2025, all | Jul 2025, EDDF/EGLL/EHAM |
| --- | ---: | ---: | ---: |
| stand-in physical model alone | 447 | 689 | 339 |
| cell mean, every `noflt` cell, cells under 20 rows pooled over airports (the plan's first proposal) | 390 | 586 | 385 |
| cell mean, every `noflt` cell with 5 rows or more, thin cells to the physical model | 354 | 495 | 341 |
| cell mean, cells with a defaulted share of 5% or more and 5 rows or more | 353 | 494 | 339 |
| mixture, same coverage (**adopted**) | 341 | 461 | 339 |
| mixture, same coverage, with an extra lag edge at 2 h | 340 | 468 | 339 |

Three things the table settles. Pooling a thin cell over airports
carries one airport's artifact rate to airports that have none, and
costs more than it saves: the plan's first proposal is the worst of the five forms on every split. Knowing the mechanism is
worth having: the mixture beats the cell mean with the same coverage
(+12 s on Dec 2025, +33 s on Jul 2025, all) because it predicts each artifact's own value rather than the
cell's average of them. And an extra lag edge at two hours, which
separates the defaulted rows more sharply within the `normal` band,
moves the mixture by +1 s on Dec 2025, -7 s on Jul 2025, all -- the defaulted share in that band shifts
between months by more than the edge gains -- so it is not adopted.

The minimum count. Covered cells at the full-year fit, and RMSE, as the
count moves (defaulted share at least 5% throughout):

| minimum rows | covered cells | Dec 2025 | Jul 2025, all | Jul 2025, EDDF/EGLL/EHAM |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 341 | 461 | 339 |
| 2 | 4 | 341 | 461 | 339 |
| 3 | 3 | 341 | 461 | 339 |
| 5 | 3 | 341 | 461 | 339 |
| 10 | 3 | 341 | 649 | 339 |
| 20 | 3 | 341 | 649 | 339 |
| 50 | 2 | 415 | 687 | 339 |

The cells the term exists for are rare by construction -- they are the
tail -- so a minimum count high enough to look prudent discards them.
The rarest covered cell, `noflt:>12h` at LIRF, has
29 rows in the full year and
27 in January to November and 7 in January to June; a minimum above either count drops it from that split,
which is what the July columns show. Below five, a cell's defaulted
share rests on a handful of rows. 5 is taken; the table shows every
value from one to five scores the same.

The minimum share, at 5 rows:

| minimum defaulted share | covered cells | Dec 2025 | Jul 2025, all | Jul 2025, EDDF/EGLL/EHAM |
| ---: | ---: | ---: | ---: | ---: |
| 1% | 3 | 341 | 461 | 339 |
| 2% | 3 | 341 | 461 | 339 |
| 5% | 3 | 341 | 461 | 339 |
| 10% | 3 | 341 | 461 | 339 |
| 20% | 3 | 341 | 461 | 339 |
| 50% | 2 | 352 | 465 | 339 |

The sweep is flat because no cell with enough rows has a defaulted
share between 0.1% and 21.7%: the artifact is a property of one
airport's feed, present in its cells at 22%-85% and absent
elsewhere. 5% is taken as the round number inside the gap.

## What the term is worth

The share of the airport-mean predictor's squared error that the
flagged rows carry on each validation month, and what the stand-in
physical model scores with and without the term. The last column is
the physical model on the unflagged rows alone, which the term does not
touch; it is the number the physical model is answerable for.

| split | rows | flagged | share of rows | share of airport-mean SSE | RMSE without | RMSE with | RMSE, unflagged rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dec 2025 | 165,677 | 88 | 0.053% | 39.2% | 447 | 341 | 328 |
| Jul 2025, all | 190,713 | 335 | 0.176% | 57.8% | 689 | 461 | 425 |
| Jul 2025, EDDF/EGLL/EHAM | 64,613 | 0 | 0.000% | 0.0% | 339 | 339 | 339 |

Per airport, Dec 2025:

| ICAO | rows | flagged | SSE share | RMSE without | RMSE with | RMSE, unflagged rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EDDF | 17,324 | 0 | 0.0% | 268 | 268 | 268 |
| EDDM | 12,035 | 0 | 0.0% | 452 | 452 | 452 |
| EGLL | 20,000 | 0 | 0.0% | 390 | 390 | 390 |
| EHAM | 19,458 | 0 | 0.0% | 217 | 217 | 217 |
| LEBL | 13,835 | 0 | 0.0% | 246 | 246 | 246 |
| LEMD | 17,985 | 0 | 0.0% | 227 | 227 | 227 |
| LFPG | 19,778 | 0 | 0.0% | 389 | 389 | 389 |
| LIRF | 12,068 | 88 | 85.6% | 1,202 | 542 | 419 |
| LSZH | 10,585 | 0 | 0.0% | 262 | 262 | 262 |
| LTFM | 22,609 | 0 | 0.0% | 335 | 335 | 335 |

Per airport, Jul 2025, all (`*` marks the July test airports):

| ICAO | rows | flagged | SSE share | RMSE without | RMSE with | RMSE, unflagged rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| EDDF * | 21,334 | 0 | 0.0% | 246 | 246 | 246 |
| EDDM | 15,825 | 0 | 0.0% | 243 | 243 | 243 |
| EGLL * | 20,790 | 0 | 0.0% | 499 | 499 | 499 |
| EHAM * | 22,489 | 0 | 0.0% | 208 | 208 | 208 |
| LEBL | 16,752 | 0 | 0.0% | 319 | 319 | 319 |
| LEMD | 18,818 | 0 | 0.0% | 238 | 238 | 238 |
| LFPG | 21,712 | 0 | 0.0% | 359 | 359 | 359 |
| LIRF | 15,211 | 335 | 78.2% | 2,164 | 1,177 | 1,004 |
| LSZH | 12,611 | 0 | 0.0% | 239 | 239 | 239 |
| LTFM | 25,171 | 0 | 0.0% | 435 | 435 | 435 |

## What it touches in the ranking set

Fitted on the full training year, the term predicts for 383 of the
344,841 ranking departures: 107 in January 2026 and
276 in July, all at LIRF.
Rows per covered cell, against the 2025 monthly rate:

| ICAO | month | departures | flagged | covered cells |
| --- | --- | ---: | ---: | --- |
| LIRF | 2026-01 | 11,061 | 107 | `noflt:normal` 92 (113/month in 2025); `noflt:4-12h` 12 (8/month in 2025); `noflt:>12h` 3 (2/month in 2025) |
| LIRF | 2026-07 | 15,838 | 276 | `noflt:normal` 248 (113/month in 2025); `noflt:4-12h` 20 (8/month in 2025); `noflt:>12h` 8 (2/month in 2025) |

Beside the term, the rule of *The plan-shifted departures* below takes
1 row of the ranking set that the term leaves: 1 at LIRF in July 2026.
The two together replace the physical model's prediction on
384 of the 344,841 ranking departures.

## The plan-shifted departures

One class of departure the cells cannot reach: a departure the term does
not cover whose take-off falls more than 4 h after both of its flight
plan's off-block estimates, `EOBT_1_flt` and `LOBT_flt`, both present. The
cells key on the schedule lag and on the absence of a record, and these
departures have one. `taxiout.tails.PlanShiftModel` predicts for them by the
rule `reports/preregistered-2026-10-02.md` fixed before it was evaluated or
built (sha256 `c1aba2f8e7d41626cd2b6fb0bce2025e4f027d056c6cc46c090d269f904f5368`), and that document is the
source of every definition in this section.

**It rests on 3 rows of the training year, and was added after the
class was seen in the ranking set's exposed fields.** The pre-registration
records both, and says of its own evaluation that it shows what the rule
does to those rows and is not independent evidence that the rule is right.

### The class in the training year

Fitted on the full training year, against the term fitted there: the class
holds 3 departures (1 at LFPG, 2 at LIRF),
and 2 of them recorded a taxi-out over 4 h, which the rule
calls *long*. The term covers 0 of the 3 training departures past 4 h after both estimates.

The share is the add-one estimate over the class's departures,
`(long + 1) / (n + 2)`, so a class of a few rows cannot assert a
share of none or all: (2 + 1) / (3 + 2) = **0.6000**.
The rest mean is the mean recorded taxi-out of the airport's training
departures with a flight-plan record (`LOBT_flt` present) and a taxi-out of 4 h
or less. Nothing is left out for being the term's or the class's, since
the pre-registration names no such exclusion. The means hold
0 of the term's 1,485 rows, because
0 of those have a record, and 1 of the class's 3:

| ICAO | departures with a record, 4 h or less | rest mean (s) |
| --- | ---: | ---: |
| EDDF | 228,185 | 862.5 |
| EDDM | 166,324 | 810.6 |
| EGLL | 238,132 | 1,363.0 |
| EHAM | 243,867 | 787.6 |
| LEBL | 177,843 | 955.6 |
| LEMD | 211,295 | 1,014.3 |
| LFPG | 235,777 | 1,016.7 |
| LIRF | 159,215 | 1,144.2 |
| LSZH | 132,616 | 741.6 |
| LTFM | 269,320 | 1,049.6 |

### The class in the ranking set

1 of the 344,841 ranking departures is in the class. The term covers 0 of the 1 ranking departure past 4 h after both estimates.

| ICAO | month | departures in the class |
| --- | --- | ---: |
| LIRF | 2026-07 | 1 |

### The prediction

`share x lag + (1 - share) x rest_mean`, the form the term uses, with the lag
taken as take-off minus `LOBT_flt`, the plan's last known off-block time,
and the rest mean of the departure's airport. It is substituted
(`taxiout.tails.combine`) for the physical model's prediction on the
class's rows, after the term, and before the rounding and the clip at zero
that `scripts/make_submission.py` applies to every prediction. It reads
`MVT_TIME_UTC_mvt`, `EOBT_1_flt`, `LOBT_flt`, `ADEP_mvt` and, at fit time, the recorded taxi-out; never
`AOBT_3_flt`, which stays excluded by decision. The physical model, its rows
and its parameters, and the term's cells and the rows it covers, are what
they were without it.

## The callsign-prefix reading, at LIRF

A review note of 10 September, written up nowhere until now, read several
callsign prefixes at LIRF *with* a Network Manager record as having
taxi times equal to their lag in most of their rows. If that were the
defaulting the term is about, the cells would be under-drawn: the term keys on
(airport, record, lag band) and would be missing an operator dimension. This
section measures the claim, and then measures what acting on it would reach.
It changes nothing -- not the term, not its inputs, not its coverage rule.

Two readings of *the taxi time equals the lag* are printed side by side. The
note's is `|taxi - lag| <= 60 s` and nothing else. The term's is
`taxiout.tails.defaulted_block`: the *block* time within 60 s of the
schedule **and** a lag above 2 hours.

They are not two tests. The target is take-off minus off-block and the lag is
take-off minus schedule, so `taxi - lag` is `schedule - block` exactly, and the
note's test is the term's first condition written another way. The data review
reports the target equal to take-off minus off-block to the second in every
training departure, so the identity should hold on the data and not only in
algebra: over the 2,085,047 training departures the two tests agree on
100.0% of rows, with
0 disagreeing. What the
term adds is the 2-hour condition, and that is the whole of the
difference between the two columns below.

LIRF is the airport the term's covered cells are at, and the one
the note was about.

### LIRF departures with a record

159,216 rows over 190 prefixes, of which
136 have at least 5 rows. The
15 with the largest share under the note's
reading are printed. `median lag` is over that prefix's rows the note's
reading catches, and is what separates the two readings.

| prefix | rows | defaulted, the term (2 h) | taxi within 60 s of the lag, the note | median lag (s) |
| --- | ---: | ---: | ---: | ---: |
| MMO | 10 | 0.0% | 100.0% | 2,726 |
| QFA | 45 | 0.0% | 100.0% | 2,283 |
| TWB | 270 | 0.4% | 100.0% | 1,442 |
| TKJ | 344 | 0.3% | 100.0% | 2,310 |
| ENT | 32 | 0.0% | 100.0% | 2,434 |
| EZZ | 5 | 0.0% | 100.0% | 1,318 |
| FFA | 10 | 0.0% | 100.0% | 2,342 |
| HAT | 38 | 0.0% | 100.0% | 2,279 |
| UZB | 22 | 0.0% | 100.0% | 2,819 |
| ISR | 385 | 2.6% | 99.2% | 2,762 |
| FIE | 64 | 0.0% | 98.4% | 1,796 |
| ANE | 23 | 0.0% | 95.7% | 2,969 |
| TGZ | 83 | 1.2% | 95.2% | 1,798 |
| MEA | 183 | 0.5% | 93.4% | 1,623 |
| LAV | 115 | 0.0% | 93.0% | 2,037 |

Over all 159,216 rows with a record: 0.0% are
defaulted under the term's reading and 20.5%
under the note's, and the rows the note's reading catches have a median lag
of 1,081 s.

### LIRF departures without a record

1,488 rows over 127 prefixes, of which
57 have at least 5 rows. The
15 with the largest share under the note's
reading are printed. `median lag` is over that prefix's rows the note's
reading catches, and is what separates the two readings.

| prefix | rows | defaulted, the term (2 h) | taxi within 60 s of the lag, the note | median lag (s) |
| --- | ---: | ---: | ---: | ---: |
| FIE | 7 | 85.7% | 100.0% | 9,596 |
| TGZ | 5 | 80.0% | 100.0% | 11,335 |
| TKJ | 20 | 40.0% | 100.0% | 6,778 |
| TAR | 6 | 100.0% | 100.0% | 34,408 |
| NMA | 6 | 50.0% | 100.0% | 7,622 |
| MSC | 12 | 91.7% | 100.0% | 9,242 |
| MAB | 6 | 50.0% | 100.0% | 6,987 |
| LAV | 23 | 56.5% | 100.0% | 7,743 |
| HAT | 5 | 80.0% | 100.0% | 8,639 |
| TWB | 19 | 78.9% | 100.0% | 8,637 |
| ISR | 59 | 45.8% | 98.3% | 6,986 |
| KMM | 18 | 55.6% | 94.4% | 7,620 |
| ETH | 24 | 41.7% | 91.7% | 6,601 |
| THY | 11 | 27.3% | 90.9% | 5,670 |
| ASL | 5 | 80.0% | 80.0% | 11,878 |

Over all 1,488 rows without a record: 26.6% are
defaulted under the term's reading and 48.5%
under the note's, and the rows the note's reading catches have a median lag
of 7,532 s.

### The cells a prefix split would open

Every (airport, prefix, bucket) cell at LIRF meeting the term's own
coverage rule -- at least 5 rows and a defaulted share of at
least 5% -- that the term's present cells do not already
cover. 32 cells meet the rule. 31 of them lie
inside the 3 (airport, bucket) cells the term already covers at
LIRF, so every row in them is predicted for already and splitting
by prefix would move nothing there. 1
lies outside.

| prefix | cell | training rows | defaulted | ranking rows it would touch |
| --- | --- | ---: | ---: | ---: |
| ADB | `flt:normal` | 13 | 7.7% | 1 |

1 cell over
13 training rows, touching 1 of the
26,899 LIRF ranking departures.

**What the two readings say.** The note's arithmetic holds and its reading of
it does not. The taxi time does equal the lag in most of the rows of several
prefixes with a record, and what that says is that those operators pushed back
within 60 s of their scheduled off-block time -- which is the same
statement, because the one quantity is the other. On a row where the block time
is the scheduled time, the lag simply *is* the taxi, so the column reads as high
as it does at a punctual operator and says nothing about the record. The median
lag of the rows the note catches is what shows it: with a record it is
1,081 s against
a median recorded taxi-out of 1,024 s over all
of this airport's departures with a record; without a record it is
7,532 s, which is where
the 2-hour condition begins to bite, and
26.6% of those rows are defaulted under the
term's reading against
0.020% of the rows with one. The
2-hour condition is not a refinement of the test; it is the test.

**What the cells say.** The table above is the whole exposure of a prefix
split under the coverage rule as it stands, measured on the training year and
counted against the ranking set. It is printed so the size of the question is
on record. This report does not decide it: the term's key, its coverage rule
and the rows it predicts for are exactly what they were.

## What the term is, stated plainly

The record-quality term is the RMSE-optimal prediction under the stated
scoring rule for rows whose recorded value no aircraft produced. The
ranking page states the rule in these words: teams "will be ranked based
on the best Root Mean Square Error (RMSE) achieved across their
submissions", and "the ranking script will issue an error if ... any rows
are missing", so every row of the template is scored and none is
excluded. For a row whose block time was defaulted to its schedule, the
value the scorer holds is the lag, and the prediction that minimizes
expected squared error is the mixture above -- the lag weighted by how
often the cell defaults, and the cell's ordinary mean weighted by the
rest. That is a conditional mean, not a taxi time; it is presented as
such, kept in its own module, and the physical model is reported
without it wherever it is reported with it.

Two limits, stated so they need not be discovered. First, the term is
fitted per (airport, cell) and covers only cells that showed the
mechanism in 2025 -- LIRF alone. An airport whose
feed starts defaulting block times in 2026 is not covered, and no rule
fitted on 2025 could cover it; the exposure is one day-scale row per
occurrence. Pooling across airports was the alternative, and the
design table shows what it costs on the splits that were run.
Second, it is small: 88 rows of 165,677 in the December
validation month, carrying 39.2% of the airport-mean predictor's
squared error; 335 of 190,713 in July, carrying 57.8%;
and 383 of 344,841 ranking rows, 107 in January and 276 in July. Its
whole effect on the leaderboard runs through those rows.

Beside those rows sits the rule of the plan-shifted departures: the term's
form, with an add-one share, for departures whose take-off left their
flight plan hours behind. Its share is estimated from
3 rows of the training year, 2 of them long, and
its rest mean from the airport's ordinary departures; it is a conditional
mean and not a taxi time. It was added after the class was seen in the ranking
set's exposed fields, was fixed in advance of any evaluation, and takes
1 row of the ranking set, 1 at LIRF in July 2026.
Its whole effect on the leaderboard runs through that row.

## What this report was generated from

Every path this run read, with the sha256 of each as it was read.
`scripts/check_provenance.py` recomputes them and reports this report as
verified, stale or unverifiable; a digest reading `absent` is the
ordinary case on a clean checkout, where the data this repository does
not redistribute is not present.

| input | sha256 |
| --- | --- |
| `data/challenge/movements/training_2025-01-01_2025-02-01.parquet` | `a4dc8f152c082c650cd7eb745fe9c641089f34f74df127e6944a188990a0ed94` |
| `data/challenge/movements/training_2025-02-01_2025-03-01.parquet` | `b6f9ff2cdb1cfe07c5442f4e1f11bbcdbeafe52852dcb0b8175f62ab84c6a066` |
| `data/challenge/movements/training_2025-03-01_2025-04-01.parquet` | `58fde7e5f8b539546af63b41d7f6ec0f2884fdc3d3dd9b6bfca7c59a420dcdd3` |
| `data/challenge/movements/training_2025-04-01_2025-05-01.parquet` | `9ffe0cf5741c4863799c41be81475dfaecedabe0ba687b8f8e4eacba992f9390` |
| `data/challenge/movements/training_2025-05-01_2025-06-01.parquet` | `5481c0642a8fbb4dc395f93bc713c57f4cc16854fb361d404967077e1fce2ff9` |
| `data/challenge/movements/training_2025-06-01_2025-07-01.parquet` | `c3ef5080f2cc636c3416baf1555f7d0ee3519f9046e8043c1c0c8f017a244025` |
| `data/challenge/movements/training_2025-07-01_2025-08-01.parquet` | `34d1c5bba7c8596a1c20c09edd3cb2f63277b5a684fa801911660fe0eaf63c9d` |
| `data/challenge/movements/training_2025-08-01_2025-09-01.parquet` | `6c8d2aa7a1b6c9dff9a0305a37eb02de1e87a68d07557264f41b77738272c540` |
| `data/challenge/movements/training_2025-09-01_2025-10-01.parquet` | `955308a4000947e32a00586d3ee13928949d2231b67f84f6dfb0631893f3754a` |
| `data/challenge/movements/training_2025-10-01_2025-11-01.parquet` | `4a31a35601c1bdcd74b8cd6379231ba516a48324e4c2478bf1c9360cd23e687f` |
| `data/challenge/movements/training_2025-11-01_2025-12-01.parquet` | `ac36b17db7a537e3baeb682fa4763dacaa1fcc5d9bfa6bf32b3ee5d9f91300b0` |
| `data/challenge/movements/training_2025-12-01_2026-01-01.parquet` | `2d357f2f403a86778a69e93f72d8754effe77f1ac0942399b3d1b7728241f724` |
| `data/challenge/ranking.parquet` | `9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86` |
| `data/challenge/submitting.parquet` | `0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b` |
| `reports/preregistered-2026-10-02.md` | `c1aba2f8e7d41626cd2b6fb0bce2025e4f027d056c6cc46c090d269f904f5368` |

## The pooled record-quality rule, measured

An airport with no covered cell is left entirely to the physical model,
and the alternative is that it inherits a cell pooled across airports.
That question was reopened as an adoptive measurement with its criteria
fixed in [`../docs/plan.md`](../docs/plan.md) before any figure here
existed, and this section is the measurement. Generated by
`scripts/pooled_tails.py`. It changes neither the term, nor its inputs,
nor its coverage rule, and the minimum count and the minimum share are
what they were.

It is not the pooled row of the design table above, which is a cell
mean over every `noflt` cell with thin cells pooled: that row differs
from the adopted design in the prediction form and in the coverage rule
at once. The arm this section is about is the adopted mixture at its
present coverage, plus a pooled cell only where an airport has no
covered cell in that bucket, and a reader who takes the design table's
row for it is reading the wrong measurement.

**The instrument.** Pooling rests on one premise: that a bucket's
defaulted share is a property of the defaulting mechanism and not of
the airport, so a cell learned where the mechanism is visible may be
carried to an airport where it is not. The fitted cell table measures
that premise directly, per airport and per bucket, on training rows,
with no fit and no booster. The test is arithmetic rather than a chosen
threshold: an airport's share is read against two standard errors of
the pooled share, `sqrt(p x (1 - p) / n)` at the pooled `p` and the
airport's own `n`, and every airport with at least 5 rows in the
bucket has to lie inside.

### The premise, per leg and per bucket

Every bucket, on each of the three training halves. A pooled cell under
5% is not covered by the term's own rule and reaches nobody,
so the two middle columns say which buckets the question is about at
all.

| leg | bucket | pooled rows | pooled share | covered at 5% | would serve it | airports tested | outside | the premise |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- |
| December leg, train January to November | `flt:4-12h` | 5,192 | 0.0% | no | no | 10 | 0 | holds |
| December leg, train January to November | `flt:<-1h` | 495 | 0.0% | no | no | 8 | 0 | holds |
| December leg, train January to November | `flt:>12h` | 505 | 0.2% | no | no | 7 | 0 | holds |
| December leg, train January to November | `flt:normal` | 1,892,357 | 0.0% | no | no | 10 | 2 | **fails** |
| December leg, train January to November | `noflt:4-12h` | 432 | 18.3% | yes | yes | 10 | 7 | **fails** |
| December leg, train January to November | `noflt:<-1h` | 39 | 0.0% | no | no | 4 | 0 | holds |
| December leg, train January to November | `noflt:>12h` | 67 | 25.4% | yes | yes | 4 | 3 | **fails** |
| December leg, train January to November | `noflt:normal` | 20,283 | 1.3% | no | no | 10 | 10 | **fails** |
| July leg, train January to June | `flt:4-12h` | 2,395 | 0.0% | no | no | 10 | 0 | holds |
| July leg, train January to June | `flt:<-1h` | 272 | 0.0% | no | no | 6 | 0 | holds |
| July leg, train January to June | `flt:>12h` | 262 | 0.0% | no | no | 5 | 0 | holds |
| July leg, train January to June | `flt:normal` | 993,771 | 0.0% | no | no | 10 | 1 | **fails** |
| July leg, train January to June | `noflt:4-12h` | 187 | 16.0% | yes | yes | 9 | 4 | **fails** |
| July leg, train January to June | `noflt:<-1h` | 15 | 0.0% | no | no | 1 | 0 | holds |
| July leg, train January to June | `noflt:>12h` | 26 | 15.4% | yes | yes | 3 | 1 | **fails** |
| July leg, train January to June | `noflt:normal` | 8,591 | 1.4% | no | no | 10 | 10 | **fails** |
| the full training year | `flt:4-12h` | 5,725 | 0.0% | no | no | 10 | 0 | holds |
| the full training year | `flt:<-1h` | 533 | 0.0% | no | no | 8 | 0 | holds |
| the full training year | `flt:>12h` | 551 | 0.2% | no | no | 7 | 0 | holds |
| the full training year | `flt:normal` | 2,055,768 | 0.0% | no | no | 10 | 3 | **fails** |
| the full training year | `noflt:4-12h` | 466 | 17.8% | yes | yes | 10 | 7 | **fails** |
| the full training year | `noflt:<-1h` | 42 | 0.0% | no | no | 4 | 0 | holds |
| the full training year | `noflt:>12h` | 75 | 24.0% | yes | yes | 4 | 3 | **fails** |
| the full training year | `noflt:normal` | 21,887 | 1.4% | no | no | 10 | 10 | **fails** |

The buckets a pooled cell would serve are `noflt:4-12h`, `noflt:>12h`. Every
airport's share in them, against the interval two standard errors of
the pooled share gives at that airport's own row count:

In `noflt:4-12h`:

| leg | ICAO | rows | its defaulted share | two standard errors of the pooled share, at its own rows | the share |
| --- | --- | ---: | ---: | ---: | --- |
| December leg, train January to November | EDDF | 37 | 0.0% | 5.6% to 31.0% | **outside** |
| December leg, train January to November | EDDM | 9 | 0.0% | 0.0% to 44.1% | inside |
| December leg, train January to November | EGLL | 37 | 0.0% | 5.6% to 31.0% | **outside** |
| December leg, train January to November | EHAM | 70 | 0.0% | 9.0% to 27.5% | **outside** |
| December leg, train January to November | LEBL | 44 | 0.0% | 6.6% to 29.9% | **outside** |
| December leg, train January to November | LEMD | 13 | 0.0% | 0.0% to 39.7% | inside |
| December leg, train January to November | LFPG | 79 | 0.0% | 9.6% to 27.0% | **outside** |
| December leg, train January to November | LIRF | 94 | 84.0% | 10.3% to 26.3% | **outside** |
| December leg, train January to November | LSZH | 11 | 0.0% | 0.0% to 41.6% | inside |
| December leg, train January to November | LTFM | 38 | 0.0% | 5.7% to 30.8% | **outside** |
| July leg, train January to June | EDDF | 17 | 0.0% | 0.0% to 33.8% | inside |
| July leg, train January to June | EDDM | 5 | 0.0% | 0.0% to 48.9% | inside |
| July leg, train January to June | EGLL | 13 | 0.0% | 0.0% to 36.4% | inside |
| July leg, train January to June | EHAM | 32 | 0.0% | 3.1% to 29.0% | **outside** |
| July leg, train January to June | LEBL | 13 | 0.0% | 0.0% to 36.4% | inside |
| July leg, train January to June | LEMD | 3 | 0.0% | 0.0% to 58.4% | not tested |
| July leg, train January to June | LFPG | 36 | 0.0% | 3.8% to 28.3% | **outside** |
| July leg, train January to June | LIRF | 34 | 88.2% | 3.5% to 28.6% | **outside** |
| July leg, train January to June | LSZH | 7 | 0.0% | 0.0% to 43.8% | inside |
| July leg, train January to June | LTFM | 27 | 0.0% | 1.9% to 30.2% | **outside** |
| the full training year | EDDF | 39 | 0.0% | 5.6% to 30.1% | **outside** |
| the full training year | EDDM | 10 | 0.0% | 0.0% to 42.0% | inside |
| the full training year | EGLL | 40 | 0.0% | 5.7% to 29.9% | **outside** |
| the full training year | EHAM | 77 | 0.0% | 9.1% to 26.5% | **outside** |
| the full training year | LEBL | 47 | 0.0% | 6.6% to 29.0% | **outside** |
| the full training year | LEMD | 14 | 0.0% | 0.0% to 38.3% | inside |
| the full training year | LFPG | 88 | 0.0% | 9.7% to 26.0% | **outside** |
| the full training year | LIRF | 98 | 84.7% | 10.1% to 25.5% | **outside** |
| the full training year | LSZH | 12 | 0.0% | 0.0% to 39.9% | inside |
| the full training year | LTFM | 41 | 0.0% | 5.9% to 29.8% | **outside** |

In `noflt:>12h`:

| leg | ICAO | rows | its defaulted share | two standard errors of the pooled share, at its own rows | the share |
| --- | --- | ---: | ---: | ---: | --- |
| December leg, train January to November | EDDF | 17 | 0.0% | 4.3% to 46.5% | **outside** |
| December leg, train January to November | EDDM | 2 | 0.0% | 0.0% to 86.9% | not tested |
| December leg, train January to November | EHAM | 12 | 0.0% | 0.2% to 50.5% | **outside** |
| December leg, train January to November | LEBL | 2 | 0.0% | 0.0% to 86.9% | not tested |
| December leg, train January to November | LFPG | 6 | 0.0% | 0.0% to 60.9% | inside |
| December leg, train January to November | LIRF | 27 | 63.0% | 8.6% to 42.1% | **outside** |
| December leg, train January to November | LTFM | 1 | 0.0% | 0.0% to 112.4% | not tested |
| July leg, train January to June | EDDF | 9 | 0.0% | 0.0% to 39.4% | inside |
| July leg, train January to June | EHAM | 6 | 0.0% | 0.0% to 44.8% | inside |
| July leg, train January to June | LFPG | 3 | 0.0% | 0.0% to 57.0% | not tested |
| July leg, train January to June | LIRF | 7 | 57.1% | 0.0% to 42.7% | **outside** |
| July leg, train January to June | LTFM | 1 | 0.0% | 0.0% to 87.5% | not tested |
| the full training year | EDDF | 18 | 0.0% | 3.9% to 44.1% | **outside** |
| the full training year | EDDM | 3 | 0.0% | 0.0% to 73.3% | not tested |
| the full training year | EHAM | 14 | 0.0% | 1.2% to 46.8% | **outside** |
| the full training year | LEBL | 2 | 0.0% | 0.0% to 84.4% | not tested |
| the full training year | LFPG | 8 | 0.0% | 0.0% to 54.2% | inside |
| the full training year | LIRF | 29 | 62.1% | 8.1% to 39.9% | **outside** |
| the full training year | LTFM | 1 | 0.0% | 0.0% to 109.4% | not tested |

### The stake

Ranking rows whose (airport, bucket) is uncovered today and whose
pooled bucket cell would be covered, from `taxiout.tails.flag` and the
full-year cell table alone, reading no field outside `PREDICT_INPUTS`:

| ICAO | month | bucket | ranking rows |
| --- | --- | --- | ---: |
| EDDF | 2026-01 | `noflt:4-12h` | 9 |
| EDDF | 2026-01 | `noflt:>12h` | 3 |
| EDDF | 2026-07 | `noflt:4-12h` | 3 |
| EDDF | 2026-07 | `noflt:>12h` | 1 |
| EDDM | 2026-01 | `noflt:4-12h` | 6 |
| EDDM | 2026-07 | `noflt:4-12h` | 4 |
| EGLL | 2026-01 | `noflt:4-12h` | 10 |
| EGLL | 2026-07 | `noflt:4-12h` | 7 |
| EHAM | 2026-01 | `noflt:4-12h` | 133 |
| EHAM | 2026-01 | `noflt:>12h` | 5 |
| EHAM | 2026-07 | `noflt:4-12h` | 7 |
| EHAM | 2026-07 | `noflt:>12h` | 3 |
| LEBL | 2026-01 | `noflt:4-12h` | 8 |
| LEBL | 2026-07 | `noflt:4-12h` | 16 |
| LEMD | 2026-01 | `noflt:4-12h` | 2 |
| LEMD | 2026-07 | `noflt:4-12h` | 4 |
| LFPG | 2026-01 | `noflt:4-12h` | 22 |
| LFPG | 2026-01 | `noflt:>12h` | 1 |
| LFPG | 2026-07 | `noflt:4-12h` | 12 |
| LFPG | 2026-07 | `noflt:>12h` | 1 |
| LSZH | 2026-01 | `noflt:4-12h` | 10 |
| LSZH | 2026-07 | `noflt:4-12h` | 3 |
| LTFM | 2026-01 | `noflt:4-12h` | 6 |
| LTFM | 2026-07 | `noflt:4-12h` | 3 |

279 rows in all, against the 383 of
344,841 ranking rows the term covers today.
The stake is not zero, so it is the premise and not the stake that
settles this.

The population at EHAM in 2026-01, counted here rather than read off
a column heading in [`data_review.md`](data_review.md): of
16,232 departures, 805 have no flight-side record,
418 have a lag over four hours, 26 over twelve, and
138 have no record *and* a lag over four hours. Those last are
exactly the late no-record buckets, `noflt:4-12h` at 133, `noflt:>12h` at 5.

### What a pooled cell would predict

For one EHAM row in each served bucket, at a stated lag inside the
band. The two share columns are the same pooling arithmetic over the
shares as this report prints them and over the shares as they are
measured, so what the rounding costs is visible; the two prediction
columns follow from them, weighting this airport's own ordinary mean by
the rest.

| bucket | lag used | pooled share, printed | pooled share, measured | predicted, printed | predicted, measured | EHAM's mean | error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noflt:4-12h` | 18,000 s | 17.8% | 17.8% | 3,883 s | 3,883 s | 824 s | +3,059 s |
| `noflt:>12h` | 50,400 s | 24.0% | 24.0% | 13,023 s | 13,017 s | 1,212 s | +11,805 s |

The arm itself would predict higher than either column, because it
weights the pooled cell's own ordinary mean rather than this
airport's: `noflt:4-12h` at 4,260 s, `noflt:>12h` at 25,674 s. The lags above are stated choices inside each
band and are the only chosen numbers in this section. The table
below replaces them with this airport's own rows, whose recorded
taxi-out is known, so the error there is measured and not worked.

What a pooled cell would have predicted on EHAM's own training rows in
those buckets, against what they recorded:

| bucket | rows | median lag | mean taxi-out | mean pooled prediction | mean error | RMSE, pooled cell | RMSE, its own mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `noflt:4-12h` | 77 | 17,988 s | 824 s | 4,512 s | +3,688 s | 3,819 s | 484 s |
| `noflt:>12h` | 14 | 72,588 s | 1,212 s | 32,761 s | +31,549 s | 32,397 s | 822 s |

### The reading

**The mechanism is at one airport.** Of the
432 training rows the term's own test picks out,
99.1% are at LIRF,
and 3 of the 10 airports carry any at all; the covered
cells run from 21.7% to 84.7%. A pooled
share is therefore not an estimate of a rate several airports share. It
is one airport's rate diluted by the row counts of the airports that do
not have it, and the premise test reads as it does for that reason.

That holds at both ends of the table. Where a pooled share rests on a
handful of rows spread over hundreds of thousands, two standard errors
is narrower than the share itself, so an airport with no defaulted row
at all sits outside the interval rather than inside it; and a bucket
whose pooled share is under 5% is not covered and serves nobody
whichever way its premise reads.

**EHAM's zero is a measurement and not a gap.** Its cells in the
served buckets are uncovered on share and not on thinness: 77 rows in `noflt:4-12h`, 14 rows in `noflt:>12h`,
each well above the 5-row minimum. At those counts a single
defaulted row would print as 1.3% and 7.1%, so a share of 0.0% means
exactly zero defaulted rows in a full training year rather than a
cell too small to say.

**So a pooled cell would carry a share learned elsewhere onto rows
measured not to have it.** That is what the two tables above price, and
the second of them prices it on rows whose recorded value is known.
The premise is refuted by the instrument that measures it, on every
leg and in every bucket a pooled cell would serve, so the arm was
not fitted: an arm whose premise has been refuted measures something
already answered.

What is left unmeasurable is narrower, and is stated rather than
widened. Whether an uncovered airport's 2026 rows are themselves
defaulted block times is a question about `BLOCK_TIME_UTC_mvt` and
`TAXITIME_SEC_mvt`, the two fields the ranking set hides, and no
instrument reaches them. That is a limit of the data and not a gap in
this measurement, which is about the 2025 rows and is complete on them.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky
Network under the terms of the PRC Data Challenge 2026. This report uses
no OpenStreetMap data; the repository's distance matrices elsewhere are
derived from it (data (c) OpenStreetMap contributors, ODbL 1.0).
