# Amendment of 4 October 2026 to the board pre-registration: adsb.lol's legs into the fusion model after the gate

Written on 4 October 2026 after the pilot's gate failed and after an
exploratory check on the two training pilot days, and fixed by its sha256
before any bulk extraction, any match beyond the pilot, and any fit of the
fusion model. It is a choice made after those numbers, and is stated as one.
It amends §5 of `preregistration_2026-10-04_board.md` (sha256
`568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`), as
amended by `preregistration_2026-10-04_adsb_leg.md` (sha256
`6cdc0f76d32ea314e5df05216a56a8b63963538b4fe60b11a54ed3be2d7bc126`) and
`preregistration_2026-10-04_opdi.md` (sha256
`3b22e7b49038d4d1216b59af7f9027f1521796a966236165278e034b0bea95c5`). It is in
force from Hunter's word, recorded in his statements, item 31.

## What exists as this is written

- **The gate** (the pilot's report, §8): seven airports reach a
  covered share of 30% or more; E1, the best estimator, lies within 120 s of
  the recorded block time on 32.73% of their 5,054 covered rows, against the
  50% §5 requires. Under §5 the adsb.lol track stopped there.
- **What the pilot showed besides** (§9 of that report): E1 comes after the
  recorded off-block by an amount that depends on the airport, a median of
  about 0 to 35 s where receivers see the stand (EHAM, EDDM) and of 4 to 14
  minutes elsewhere, where the first ground position is on a taxiway.
- **The check that motivated this amendment**
  (`workingdir/pivot_2026-10-04/adsb_queue_check.out`): on the two training
  pilot days, a per-airport linear correction of the submission rung with
  take-off minus E1, fitted on one day and applied to the other, moves the
  pooled RMSE from 232.2 to 229.5 s.

## 1. What changes

§5's "Otherwise it stops here" is set aside for one purpose: the fusion model.
The scale-up runs as §5 defines it (every second day of 2025 from 1 January and
every day of January and July 2026), with the runner the pilot built and tested
(a working directory outside this repository), and adsb.lol's legs enter the fusion model
with the features §5 lists. The gate's verdict stands as a measurement: the
paper reports that E1 does not measure the off-block to §5's standard.

The fitting days of the fusion model are §5's covered days of 2025 with both
sources' features, as the OPDI amendment provides for a gate that holds, **less
the two training pilot days, 2025-01-15 and 2025-07-16**, which informed this
amendment and are therefore left out of every fit and every reading of the
fusion model's rule.

One feature is added to §5's, fixed here before any fit: the match's
**confirmation flag**, 1 where the leg's flight field (spaces removed) equals
`CALLSIGN_flt` or `FLIGHT_mvt` (spaces removed) and no other departure matched
the same leg, else 0. The pilot found 42.0% of matched pilot rows sharing their
leg with another departure; the flag lets the model weigh those matches, and
it changes no match.

## 2. What does not change

The leg definition and every event, the matching, the coverage, the fusion
model, its parameters, target, rows and base prediction, the OPDI features,
the rule, the deadline (8 October 2026, 18:00 UTC) and v10, all as §5 and the
two amendments state them.

## What differs from the note

This copy is the working note of 4 October 2026 with six places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Four of the places lie outside any rule. Two lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
One place in the opening paragraph named the tool an exploratory check was
made with, a name this repository does not carry; it reads "an
exploratory check".
The working note this is copied from was digested when it was written,
sha256 `9b08c30924d70a65f842a78d96bb9241e76abe4d74914b4846548a95c30bde65`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 3 | named the pilot's work block | "after the pilot's gate failed" |
| the opening paragraph, lines 3 and 4 | named the tool the exploratory check was made with | "after an exploratory check" |
| the opening paragraph, line 13 | named the file of Hunter's statements by its working file name | "recorded in his statements, item 31." |
| What exists as this is written, line 17 | named the pilot's report by its working file name | "(the pilot's report, §8)" |
| §1, What changes, lines 35 and 36 (inside the rule) | named the pilot's work block and its working directory | "the runner the pilot built and tested (a working directory outside this repository)" |
| §1, What changes, line 49 (inside the rule) | named the pilot's work block | "The pilot found 42.0%" |

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
from it (data (c) OpenStreetMap contributors, ODbL 1.0). Contains figures
computed from data from adsb.lol, available under the Open Database License
(ODbL) 1.0.
