# Pre-registration of 8 October 2026: the fusion trained on every day of 2025

Written on 8 October 2026 at about 09:00 UTC, on Hunter's words "we have time
for the other candidate if need be but we need to start asap. cpu time is
available through today and tomorrow", and fixed by its sha256 before any
figure it governs exists. It adds one candidate, C17, and the rule that
decides whether v12 is built. It reopens nothing already decided: v11 is group
B's base with G1n refitted on it (the work from the overnight run of 6 October to the landing of v11), and group C adopted
nothing.

## What exists as this is written

- **The fusion in v11** is the stacker of the fusion model's measurement refitted on v11's base by
  the fusion's refit on group B's base and gated by G1n, fitted on the 174 days of 2025 with an adsb.lol
  table (every second day from 1 January, less the two training pilot days
  and the seven days that exist only as `prod-0tmp` releases). On those days
  G1n reads 220.63 s against G1 at 223.05 s and the base at 229.64 s
  (`reports/fusion-new-base.md`).
- **The larger stacker of group C** (3,000 trees at learning rate 0.02) read
  0.11 s better pooled, short of the 0.16 s floor, with the gate unchanged
  (the measurement of group C). Capacity does not appear to bind; the number of fitting days
  has not been varied.
- **OPDI's flight events** are cached for every day of 2025
  (`data/raw/opdi/`). **adsb.lol** tables exist for the 238 days of §5; the
  other days of 2025 have none. The extraction of §5's days measured the fetch at about 9.4
  days per hour on this machine with one download at a time.
- No adsb.lol table for any of the other days of 2025 exists, and no feature
  or fit of C17 exists.

## 1. C17, the fusion on every day of 2025

- **Days**: every day of 2025 for which an adsb.lol table exists by
  **9 October 2026, 06:00 UTC**: the 2025 days of §5 already cached, and the
  other days of 2025 extracted by the committed `taxiout.adsb` runner,
  unchanged, from the `prod` release of `adsblol/globe_history_2025`. A day
  whose `prod` release does not exist, or whose extraction fails twice, has
  no table; the report names each. The two training pilot days stay out, as
  in the fusion model's measurement. Nothing in the extraction is changed after a figure exists.
- **Matching and features**: `taxiout.fusion` as committed (the pilot, the OPDI fetch of 4 October,
  the feature tables of 5 October, the fusion model's measurement and the fusion's refit on group B's base), unchanged, over those days; `base` is v11's out-of-fold
  prediction (the overnight run's `heldout-2025-MM__c14__seed0`), and the peers are
  recomputed with it.
- **The model**: The stacker of the fusion model's measurement, its parameters, features and seed
  unchanged, fitted leave one month out over the fitting rows of all those
  days; G1n's out-of-fold airport gate recomputed from these fits.
- **The reference**: The twelve leave-one-month-out models of the fusion's refit on group B's base, as they
  stand, predicting the same rows, with G1n's gate recomputed from their
  predictions on the same rows. So both arms are read on every scored row of
  every day with a table, and differ only in how many days they were fitted
  on.

## 2. The rule

C17 is adopted only if, against the reference on the same scored rows (rows
neither rule covers, every taxi-out): it is better pooled over the twelve
months by more than 0.16 s; it is not worse in any month by more than that
month's floor (`reports/monthly_seeds.md`, over core rows); and on December,
fitted on January to November, it is better by more than 0.23 s. Read with
`candidates.holds` at its precision, the other readings beside. The verdict
is written by **9 October 2026, 12:00 UTC**, or C17 is not adopted.

## 3. v12

If C17 is adopted, v12 is v11 with the fusion part replaced: C17's model
fitted on every fitting row of those days, its twelve-month gate, applied to
the ranking rows as v11 applies its fusion. Everything else is v11's. It is
landed and built at a clean tree as v10 and v11 were, and uploaded once on
Hunter's word by **10 October 2026, 21:00 UTC**; if it cannot be, there is no
v12 and v11 stands.

## 4. What does not change

§6 of the board note: one upload per frozen version, nothing chosen on a
score, only `usedPairs` and the score read back. The licences: adsb.lol's
tables are a Derivative Database under ODbL and go into the deposit with the
notice; no OPDI file and no table derived from one enters the repository or
the deposit. Nothing is added after a number is seen.

## What differs from the note

This copy is the working note of 8 October 2026 with nine places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Four of the places lie outside any rule. Five lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `695fa658d54abd8aed10c8e5465f941fc013ecebadb980dff46e19d0313f51a9`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 8 | named a range of work blocks | "(the work from the overnight run of 6 October to the landing of v11)" |
| What exists as this is written, lines 13 and 14 | named the work blocks that measured the fusion model and refitted it | "the stacker of the fusion model's measurement refitted on v11's base by the fusion's refit on group B's base and gated" |
| What exists as this is written, line 21 | named the work block that measured group C | "(the measurement of group C)" |
| What exists as this is written, line 25 | named the work block that extracted §5's days | "The extraction of §5's days measured the fetch" |
| §1, Days, lines 37 and 38 (inside the rule) | named the work block that measured the fusion model | "as in the fusion model's measurement." |
| §1, Matching and features, lines 39 and 40 (inside the rule) | named the five work blocks whose code the fusion is | "(the pilot, the OPDI fetch of 4 October, the feature tables of 5 October, the fusion model's measurement and the fusion's refit on group B's base)" |
| §1, Matching and features, line 41 (inside the rule) | named the work block that fitted the base | "(the overnight run's `heldout-2025-MM__c14__seed0`)" |
| §1, The model, line 43 (inside the rule) | named the work block that measured the fusion model | "The stacker of the fusion model's measurement, its parameters" |
| §1, The reference, line 46 (inside the rule) | named the work block that refitted the fusion model | "The twelve leave-one-month-out models of the fusion's refit on group B's base" |

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
