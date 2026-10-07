# Pre-registration of 6 October 2026: the last week

Written on 6 October 2026 after v9's score, on Hunter's answers recorded as
item 40 of his statements and his "ok go", and fixed by its sha256
before any figure it governs exists. It adds seven candidates in two groups,
each with its rule, and defines v10 and v11. It amends §3 to §6 of
`preregistration_2026-10-04_board.md` (sha256
`568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`) as
amended by `preregistration_2026-10-04_tails_restored.md` (sha256
`4d388f1d65a56ee06a66c45598a3231bbac63e125dc0b229e931cddf6641efeb`),
`preregistration_2026-10-04_opdi.md` (sha256
`3b22e7b49038d4d1216b59af7f9027f1521796a966236165278e034b0bea95c5`) and the
others the fusion model's measurement checked, only where it says so. Verdicts already written
stand as measured: C6b adopted, the fusion not adopted, C2a, C5, C7, C8 and
C8c not adopted.

## What exists as this is written

- **v9**, uploaded once: 282.399 over 344,841 pairs
  (`reports/submission-2026-10-06-v9.md`). The board's top three, as Hunter
  pasted it: 213.9176, 216.0926 and 216.2405 s.
- **The fusion model's measurement** (a working report this repository does not publish): not adopted, failing
  January (+1.68 s) and July (+3.35 s). Its per-airport table (§7) has been
  read: better at LSZH, EDDM, EHAM, LEBL, EDDF and LEMD, worse at EGLL, LIRF,
  LFPG and LTFM, every interval clear of zero. A working note this repository does not publish computed
  from that table what the fusion would read at the six airports where it
  helped (222.41 against 232.38 s). That figure chooses airports after the
  fact; no candidate below does, and no candidate below has been computed.
- **Readings made tonight** (`workingdir/gap_2026-10-06/`, outputs beside the
  scripts): under v9's configuration, LIRF departures with the block time
  within 60 s of the schedule and a take-off 40 min or more after it carry
  35.5% of LIRF's squared error over the held-out year; `AOBT_3_flt` sits at
  the schedule on 20.6% of the at-schedule rows against 6.6% of the rest; the
  share of an operator's arrivals at the schedule correlates with the same
  share for its departures at 0.768 per operator-month and 0.367 per
  operator-day at LIRF, and 0.010 per operator-month at LEBL. The ranking
  file's arrivals at LIRF carry block and scheduled times on every row, in
  January and July 2026.
- **More trees, before C1**: 8,000 trees read December -0.28 s (holds) and
  July +0.09 s (fails) on 3 October (`reports/candidates-2026-10-03.md`).
- **The CatBoost member C3b** was fitted on the rung's 108 columns, without
  C1's nine and C4's two (`reports/candidates-2026-10-05.md`, its timing table).

## 1. Shared definitions

**The base.** v9's configuration at the ladder's seed: the per-row `pred` of
`data/processed/candidates_board/<split>__combo4__seed0.parquet` for the
twelve held-out months and the December, July and October legs, each checked
against its record's `pred_sha256` before use. Its members are
`<split>__combo4_booster__seed0` and `<split>__c3b_member__seed0`; `combo4`
is their mean.

**Rows.** "Core rows": rows neither the record-quality term nor the
plan-shift rule covers. "Short rows": core rows with a recorded taxi-out of
14,400 s or less.

**Floors**, from `reports/monthly_seeds.md` (floors table): over core rows
Jan 0.33, Feb 1.56, Mar 0.36, Apr 0.07, May 0.13, Jun 0.21, Jul 0.58, Aug
0.26, Sep 0.21, Oct 0.29, Nov 0.39, Dec 0.23, year 0.16 s; over short rows
Jan 0.47, Feb 1.59, Dec 0.23, year 0.15 s. The leg floors of §4 of the board
note: December 0.23, July 0.54, the three July airports 1.02, October 0.21 s.

**Readings.** Contrasts are candidate RMSE minus reference RMSE on the same
rows, read with `candidates.holds` at the precision `the_rule()` sets, with
the three readings of `OTHER_READINGS` printed beside, and a day-clustered
95% interval (2,000 draws, seed 20260902) that decides nothing.

**The same-month arrival features.** For a departure at airport a, take-off
in UTC calendar month M and on UTC date D, and callsign prefix P (the first
three letters of `FLIGHT_mvt`, missing if it has none), over the arrivals at
a (`PHASE_mvt` ARR, `ADES_mvt` a) with both `BLOCK_TIME_UTC_mvt` and
`SCHED_TIME_UTC_mvt` present, an arrival being *at the schedule* when the two
lie within 60 s of each other:

- `arr_sched_share_m`, `arr_n_m`: the share at the schedule and the count over
  those of prefix P whose `MVT_TIME_UTC_mvt` falls in month M;
- `arr_sched_share_d`, `arr_n_d`: the same over those on date D;
- `arr_sched_share_ap_m`: the share over all arrivals at a in month M.

A share over no arrival is missing; a count is 0. The arrivals come from the
training file of month M in 2025 and from `ranking.parquet` in 2026. No
departure's block time or taxi-out enters them.

## 2. Group A, for v10

### G1, the fusion gated by coverage

The twelve leave-one-month-out fits of the fusion model's measurement (working
files outside this repository), as they stand, each checked against its record's digests; nothing
is refitted. A row takes the fusion's prediction where `a_covered` or
`o_covered` is 1, and the base everywhere else.

**Rule**: §5's, on the scored rows of the fusion model's measurement (989,825): better pooled over the
twelve months by more than 0.16 s; not worse in any month by more than its
floor; and on December (the fit on January to November) better by more than
0.23 s. Printed beside, deciding nothing: per airport; covered rows only;
short rows.

### C6c, the term's rows with the same-month arrivals

C6b exactly as the tails amendment defines it, with the five arrival features
added to its classifier. **Rule**: C6's, with C6b in place of the term as the
reference: lower pooled MSE than C6b's; the 95% interval of the pooled
difference wholly below zero; not higher than C6b's in January or in July.
Before C6c is read, C6b's pooled figure must reproduce 17,687,795 s²
(`reports/c6.md`), or the block stops. If adopted, C6c predicts the term's
383 ranking rows in place of C6b.

### C9, Fiumicino's schedule layer

A stacker on LIRF's core rows, every day of 2025. LightGBM regression: 500
trees, learning rate 0.03, 15 leaves, `min_data_in_leaf` 50,
`feature_fraction` 0.9, `bagging_fraction` 0.8 with `bagging_freq` 1,
deterministic, column-wise, eight threads, seed 20260902. Target: the
recorded taxi-out. Fitted on LIRF's short rows; scored on LIRF's core rows.
Features: the base; the lag to the schedule (take-off minus
`SCHED_TIME_UTC_mvt`, s); the five arrival features; the scheduled hour and
the take-off hour (UTC); the weekday of the scheduled time; the callsign
prefix and `STAND_mvt` as categoricals, each level with fewer than ten
fitting rows read as `other`. Read leave one month out: each month scored by
the fit on the other eleven. Every other airport's rows keep the base.

**Rule**: G1's three conditions, over the core rows of the twelve held-out
months (2,083,559 rows) and of December, with the year's and the months'
floors over core rows.

### If more than one clears

C6c touches only the term's rows, so it joins whatever else is adopted. If G1
and C9 both clear, a LIRF row takes G1's prediction where it is covered and
C9's where it is not, and the pair is read by G1's rule on the scored
rows of the fusion model's measurement; if the pair fails, the one with the larger pooled gain on its own rule
is adopted alone.

**Verdicts by 7 October 2026, 18:00 UTC**, or nothing in group A is adopted.
**v10** is v9 with every adopted candidate of group A applied: on the ranking
rows, G1 with its final fit on every 2025 fitting day as §5 and the landing of v10
define it, and C9 fitted on every month, each taking v9's five-seed
prediction as its base; the term's rows from C6c if adopted, else C6b; the
plan-shift rule as in v9. If group A adopts nothing, there is no v10.
Uploaded once, on Hunter's word, by 8 October 2026, 21:00 UTC.

## 3. Group B, for v11

### W, C2a's winter layer on the winter leg

C2a exactly as §3 of the board note defines it (the classes, the carry, the
shrinkage), with its means taken from the base's residuals over short rows,
read leave one month out as §4 reads layers. **The winter leg**: held-out
January, February and December 2025, short rows. **Rule**: better on January
by more than 0.47 s; not worse on February by more than 1.59 s or on December
by more than 0.23 s; not worse over the twelve held-out months' short rows by
more than 0.15 s; and on the forward legs, the layer fitted on the months
strictly before each, not worse than the leg floors (December 0.23, July
0.54, the three July airports 1.02, October 0.21 s). If W clears, **W2**, C2b
as defined (features into the booster, the CatBoost member unchanged), is
read by the same rule against W.

### C10, the blend weight

The booster's share w in w x booster + (1 - w) x member, from the stored
member files, no fit: w in {0.3, 0.4, 0.6, 0.7}. The w with the lowest RMSE
over the twelve held-out months' core rows is the candidate, and it is held
to §4's five conditions against w = 0.5; §4's second-seed check applies it to
the second-seed members on December and July, fitted if not stored. The
choice among four weights is disclosed as a selection.

### C11, the member with C1 and C4

C3b's CatBoost member, §3's parameters, with C1's nine and C4's two columns
added to its 108; blended at 0.5 with `combo4_booster`. Held to §4's five
conditions and its second-seed check against the base. Fits in order of cost: the December, July and
October legs first; the twelve held-out months only if the four leg
conditions hold.

### C12, more trees with C1 in

`combo4_booster` refitted at `n_estimators` 9,000, every other parameter of
`BOOSTER_PARAMS` unchanged, blended at 0.5 with `c3b_member`. Held to §4's five
conditions and its second-seed check against the base, legs first as for
C11.

### The combination

Everything in group B that clears is combined into a new base: C12 as the
booster, C11 as the member, the blend at C10's weight if C10 clears and 0.5
otherwise, then W or W2. The combination is held to §4's five conditions
against v9's configuration and, if W or W2 is in it, to W's winter
conditions. If it fails, there is no v11. If it holds, every candidate group
A adopted is refitted on the new base's out-of-fold predictions and re-read
by its own rule; one that no longer holds is left out of v11, and the paper
says so. **Verdicts by 9 October
2026, 18:00 UTC.** **v11**, uploaded once on Hunter's word by 10 October
2026, 21:00 UTC.

## 4. What does not change

§6 of the board note: every uploaded file is a model the paper defends,
uploaded once when its rule has spoken; no score chooses anything; only
`usedPairs` and the score are read back. `AOBT_3_flt` stays excluded outside
C1's module. Nothing is added after a number is seen. The paper reports every
candidate here, the order in which the figures were seen, and each verdict
whichever way it falls.

## What differs from the note

This copy is the working note of 6 October 2026 with eight places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Four of the places lie outside any rule. Four lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `eb35ecd3eb802b077307d5dba6ab07ca0cc5277b6fc7d5522d1b65a86bc9f1d5`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 4 | named the file of Hunter's statements by its working file name | "item 40 of his statements" |
| the opening paragraph, line 13 | named the work block that measured the fusion model | "others the fusion model's measurement checked" |
| What exists as this is written, line 22 | named that work block and its report by its working file name | "**The fusion model's measurement** (a working report this repository does not publish)" |
| What exists as this is written, line 25 | named a working note by its working file name | "A working note this repository does not publish computed" |
| §2, G1, lines 88 and 89 (inside the rule) | named that work block and its fits' working directory | "The twelve leave-one-month-out fits of the fusion model's measurement (working files outside this repository)" |
| §2, G1's rule, line 93 (inside the rule) | named that work block | "on the scored rows of the fusion model's measurement (989,825)" |
| §2, If more than one clears, lines 131 and 132 (inside the rule) | named that work block | "on the scored rows of the fusion model's measurement" |
| §2, v10, lines 137 and 138 (inside the rule) | named the work block that lands v10 | "as §5 and the landing of v10 define it" |

The names of other pre-registrations' working files are left as the note
words them, because they are neither a numbered work block nor a dated
working note's file name. The board pre-registration's published copy is
`reports/preregistered-2026-10-04.md`, the same bytes under the same
digest; each amendment named has a published copy beside this one in
`reports/`, made the same way, whose closing section names that note's
digest. Working
directories that the patterns do not flag are also left as the note words
them, and this repository does not carry them.

The credit paragraph below is the note's own last paragraph, unchanged,
placed after this section so that it stays the last paragraph.

`tests/test_preregistered_copies.py` applies these substitutions to the
note wherever the note is on hand, after checking its digest, and requires
the result to be this copy up to this section, and its last paragraph,
byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This document uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
