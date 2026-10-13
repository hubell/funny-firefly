# Pre-registration of 8 October 2026: C18 and C19, for the final phase

Written on 8 October 2026 at about 15:00 UTC, on Hunter's words "i want to use
the next week to improve the model with this in mind we can hold on final
submission. lets take the time to further train and explore improvement
options with this in mind", and his choice "C18 then C19", and fixed by its
sha256 before any figure it governs exists. "This in mind" is the final phase
as the organizers describe it (statements item 54): a single final
submission, ranked per Jan/Jul, per Feb/Jun and on the four months combined,
to see "how generalizable the predictions for a model can be". It adds two
candidates to `preregistration_2026-10-08_final_phase.md` (sha256
`553e042f94494798c2c7fd1c92823713069514eebdc90ba61a161d9bd12f41fa`), whose
§1 allows a further candidate only in a note of its own, fixed before any
figure, applied to every ranking row. Both are. Neither has a feature, a fit
or a reading as this is written.

## Why these two

- **Where the error is.** Under v9's configuration on held-out January and
  July 2025, LFPG, LIRF, EGLL and LTFM carried 75.5% of the core's squared
  error (a working note this repository does not publish, §2: 44.7, 12.4, 9.3 and 9.1%). G1n's gate drops
  the fusion at LFPG, LIRF and EGLL, and LTFM's coverage touches no ranking
  row: the ground events do not reach the airports where most of the error
  is. Arriving aircraft's taxi-in times do, at every airport, in the training
  year and in the ranking set alike.
- **What generalizes.** A model fitted on 2025 can only average 2025's
  surface. A month with works, a closed taxiway or a new runway pattern
  shows up first in how long arrivals take to reach their stands. C18 reads
  that state from trailing windows only, so it uses nothing a live
  prediction would not have. C19 constrains the booster to a longer route
  never predicting a shorter taxi, the plainest physical relation the
  geometry carries, so a new month cannot meet a learned reversal of it.

## 1. C18, the surface state from arrivals

- **Taxi-in**: for every arrival in a frame (the training year's movements;
  the ranking set's arrivals for the ranking rows), on-block time less
  landing time, kept in [`exposure.TAXI_IN_LO_S`, `exposure.TAXI_IN_HI_S`].
- **Reference**: the 10th percentile (`unimpeded.PRU_PERCENTILE`) of
  taxi-in over the training year's arrivals per (airport, landing runway,
  stand) where that cell has at least `unimpeded.MIN_PAIR_COUNT` arrivals;
  else per (airport, landing runway) on the same condition; else per
  airport. **Excess** is taxi-in less its reference.
- **Five columns** for each departure, from its own airport's arrivals,
  windows trailing and anchored as `congestion.py` anchors its
  `txi_mean_{w}m` windows, arrivals placed by on-block time:
  `txe_mean_180m` and `txe_mean_360m` (mean excess in the trailing 3 h and
  6 h; missing when none), `txe_n_360m` (their count over 6 h),
  `txe_prevday` (mean excess over the previous UTC day), and
  `txe_truncated` (1 where the previous UTC day begins before the row's
  covered block, as `windows.coverage_start` finds it).
- **The model**: `combo4_booster`'s columns and these five, under
  `BOOSTER_PARAMS`, seed and everything else unchanged; the CatBoost member
  (`c11_member`, `c3b_member` at LIRF) unchanged; the base assembled as v11
  assembles it (0.5 and 0.5 at nine airports, 0.7 booster and 0.3
  `c3b_member` at LIRF).

## 2. C19, monotone in route length

`combo4_booster` under `BOOSTER_PARAMS` with LightGBM's
`monotone_constraints` set to +1 on `taxi_m`, `route_m` and
`runway_crossings` (each only if the booster reads it; the block prints the
list before any fit) and 0 elsewhere, `monotone_constraints_method` at its
default; everything else unchanged; the base assembled as for C18.

## 3. The rule

§4 of the board note, unchanged, as group B and C15 were held to it: against
v11's base (the overnight run's `heldout-2025-MM__c14__seed0` and its leg records, each
checked against its record), on core rows, read with `candidates.holds` at
its precision: better by more than 0.23 s on the December leg over all ten;
better by more than 0.54 s on the July leg over all ten; not worse by more
than 1.02 s on the three July airports; not worse by more than 0.21 s on the
October leg; not worse over the twelve held-out months pooled by more than
0.16 s; then the second seed on December and July, both contrasts keeping
their sign. **Beside it, deciding nothing**: the twelve held-out months split
as the organizers rank (January and July pooled, February and June pooled,
the four pooled), each month against its floor, per airport, and the
day-clustered intervals.

**The fusion, after.** A candidate that holds is not adopted on the base
alone. The stacker of the fusion's refit on group B's base is refitted on its base (its out-of-fold
predictions as `base`, the peers recomputed), twelve fits leave one month out
on the days of the fusion's refit on group B's base, G1n's gate recomputed, and read by G1n's three
conditions against the G1n of the fusion's refit on group B's base on v11's base. It is adopted only if that
holds too.

**Both.** If C18 and C19 each hold, the booster with both is read by the same
rule and the fusion step; if it fails, the one with the larger December gain
is adopted alone, as §4 combines.

**Order and time.** C18 first, then C19, then the pair. Fits wait while
the fits of the measurement of C17 run, one fit at a time. Verdicts by **10 October 2026, 06:00
UTC**; what is not read by then is not adopted.

## 4. Into v12

An adopted candidate joins v12's definition (§2 of the four-month note,
which this note amends to that extent): its booster in place of `combo4_booster` at every airport, and the
fusion (C17's if adopted, else that of the fusion's refit on group B's base) refitted on the new base with its
parameters unchanged. v12 is landed and frozen by **12 October 2026, 12:00
UTC**, and uploaded once on Hunter's word on 13 October, as the amendment
fixes. Nothing else changes: one upload, nothing chosen on a score, the
licenses as before, nothing added after a number is seen.

## What differs from the note

This copy is the working note of 8 October 2026 with seven places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
One of the places lies outside any rule. Six lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `78d4a701b8d2e96f6b34d9da6efe8331e24db8f9a26844bd653694757423b7c0`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| Why these two, line 21 | named a working note by its working file name | "(a working note this repository does not publish, §2:" |
| §3, The rule, line 69 (inside the rule) | named the work block of the overnight run | "the overnight run's `heldout-2025-MM__c14__seed0`" |
| §3, The fusion, after, line 82 (inside the rule) | named the work block that refitted the fusion on group B's base | "The stacker of the fusion's refit on group B's base is refitted" |
| §3, The fusion, after, line 84 (inside the rule) | named that work block | "on the days of the fusion's refit on group B's base" |
| §3, The fusion, after, line 85 (inside the rule) | named that work block | "against the G1n of the fusion's refit on group B's base" |
| §3, Order and time, line 93 (inside the rule) | named the work block that measured C17 | "the fits of the measurement of C17 run" |
| §4, Into v12, line 100 (inside the rule) | named the work block that refitted the fusion model | "else that of the fusion's refit on group B's base)" |

The names of other pre-registrations' working files are left as the note
words them, because they are neither a numbered work block nor a dated
working note's file name; each one named has a published copy in
`reports/`, made the same way, whose closing section names that note's
digest.

The credit paragraph below is the note's own last paragraph, unchanged,
placed after this section so that it stays the last paragraph.

`tests/test_preregistered_copies.py` applies these substitutions to the
note wherever the note is on hand, after checking its digest, and requires
the result to be this copy up to this section, and its last paragraph,
byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. Contains data from adsb.lol,
available under the Open Database License (ODbL) 1.0. This document uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
