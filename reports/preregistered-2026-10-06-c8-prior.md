# Amendment of 6 October 2026 to the C7 and C8 amendment: C8's weights corrected for the class ratio

Written on 6 October 2026 after the verdict of the reading of the candidates C1 to C5, C7 and C8, on Hunter's word "fix the
design for c8", and fixed by its sha256 before any fit of the candidate it
defines. It is a choice made after C8's figures, and is stated as one. It adds
one candidate, C8c, to `preregistration_2026-10-04_c7_c8.md` (sha256
`4ac17be6925d03255a2e52b1d1ee307e297e913ee42f33c063a03d8e36fbe8cd`), which
amends §3 and §4 of `preregistration_2026-10-04_board.md` (sha256
`568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`). C8's
verdict stands as measured.

## What exists as this is written

- **C8 as specified failed** §4's four leg conditions: December -0.16 s
  against a floor of 0.23, July +2.70, the three July airports +4.05,
  October +0.02 (the report of the reading of the candidates C1 to C5, C7 and C8, `reports/candidates-2026-10-05.md`).
- **Its weights were degenerate.** Printed before each booster was read:
  74.5% (December), 25.1% (July) and 38.7% (October) of training rows were
  clipped low at 0.1; after scaling, the largest weights were 40.1, 39.0 and
  52.1.
- **The cause is in the formula, not the data.** A classifier fitted on n1
  target rows against n0 training rows estimates p with prior odds n1/n0, so
  p/(1 - p) is the density ratio times n1/n0. With the target one month
  against eleven, n1/n0 is about 0.09, so most training rows' odds fall under
  the 0.1 floor whatever their regime, and the clip, not the classifier, sets
  their weight. The standard estimate of the density ratio from a
  classifier multiplies the odds by n0/n1.
- The reading of the candidates C1 to C5, C7 and C8 adopted C1 and C4 in one booster, blended at equal weights with
  C3b's CatBoost member, holding the five conditions (December -13.81 s,
  July -12.10, the three July airports -16.49, October -7.75, the year
  -8.25).

## 1. C8c

C8 exactly as the amendment of 4 October defines it (the target rows, the
classifier, its parameters and features, the two-fold cross-fitting by even
and odd calendar days, the clip to [0.1, 10], the scaling to a mean of 1, the
refit under `BOOSTER_PARAMS`, blend members and layers unweighted, and for a
submission each ranking month's booster weighted toward it), with one change:
each training row's odds are multiplied, before the clip, by n0/n1, where n0
and n1 are the counts of label 0 and label 1 among the rows the classifier
that scored that row was fitted on. So the weight is

> (p / (1 - p)) x (n0 / n1), clipped to [0.1, 10], then scaled to a mean of 1
> over the training rows.

Nothing else changes: no other clip, parameter, feature, fold or row.

## 2. The rule and the combination

§4's rule, unchanged in its legs, its reference (the submission rung at the
ladder's seed), its rows, its floors and its second-seed check, reads C8c
against the submission rung. If C8c holds all five conditions and its
second-seed check, the combination is the adopted configuration of the reading of the candidates C1 to C5, C7 and C8 with
C8c's weights on its booster's fits (C3b's member unweighted), held to the same
five conditions; if it holds them, it is adopted in place of that
configuration; otherwise that configuration stands. If C8c fails, nothing
changes. The weight distribution of every fit (minimum, quartiles, maximum,
share clipped low and high) is printed before its booster is read. The
verdict is written by 7 October 2026, 18:00 UTC, or nothing is adopted. v9 is
the result.

## 3. What it does not do

It does not reopen C7, C2a, C3a or C5, change §4's floors, or refit any
adopted member. The paper reports C8 as specified, its failure, this
amendment and its reason, and C8c's verdict whichever way it falls.

## What differs from the note

This copy is the working note of 6 October 2026 with four places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Three of the places lie outside any rule. One lies inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `81871eeb4f43d82b6a0aca1b5f3302d40a65cd5446efc05cf7289cb3a6fc1e98`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 3 | named the work block that read the candidates C1 to C5, C7 and C8 | "after the verdict of the reading of the candidates C1 to C5, C7 and C8," |
| What exists as this is written, line 16 | named that work block; its report's published path is kept | "(the report of the reading of the candidates C1 to C5, C7 and C8, `reports/candidates-2026-10-05.md`)" |
| What exists as this is written, line 28 | named that work block | "The reading of the candidates C1 to C5, C7 and C8 adopted C1 and C4" |
| §2, The rule and the combination, line 54 (inside the rule) | named that work block | "the combination is the adopted configuration of the reading of the candidates C1 to C5, C7 and C8" |

The names of other pre-registrations' working files are left as the note
words them, because they are neither a numbered work block nor a dated
working note's file name. The board pre-registration's published copy is
`reports/preregistered-2026-10-04.md`, the same bytes under the same
digest; each amendment named has a published copy beside this one in
`reports/`, made the same way, whose closing section names that note's
digest.

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
