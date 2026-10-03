# Pre-registration of 3 October 2026: the three candidates, and how one would ship

Written on 3 October 2026, after v7 was committed (`9feb0c4`, tag
`model-v7-2026-10-02`) and before any candidate has been fitted under this
rule or the rule of 28 September, and before any score exists for any version
of this entry. It supersedes one part of the pre-registration of 28 September
(`reports/preregistered-2026-09-28.md`, sha256 `eee5b947...3e60`): the
paragraph *What adoption costs, stated now*, whose re-freeze chain and
4 October deadline cannot be met. The candidates, the legs, the reference,
the four conditions and their floors are the 28 September note's, unchanged,
and are restated here only where this note adds to them.

## The candidates, unchanged

1. Saturation on top of `+ plan differences`: `SATURATION_FEATURES` added to
   the submission rung.
2. The mild recency schedule: row weights halving every 12 months over the
   fitting rows, as `scripts/preregistered.py` implements them.
3. More trees: `n_estimators` 8,000, every other parameter unchanged.

Nothing else is tried. No fourth candidate is added after a number is seen.

## What v7 changed in the measurement

The model that ships is v7: the five-seed average of `+ plan differences`
under `BOOSTER_PARAMS`, the record-quality term, and the rule of the
plan-shifted departures (`reports/preregistered-2026-10-02.md`). On the legs,
every figure is therefore over the rows that neither the term nor the
plan-shift rule covers, each fitted on the leg's training rows as
`taxiout.tails` fits them. The 28 September note read the ex-flagged figure,
which excludes the term's rows alone; the plan-shift rule did not exist. On
the July leg the rule covers one validation row, recorded 87,002 s, which the
booster would otherwise predict at a few thousand seconds: two candidates
differing by 500 s on that one row would move the July contrast by about
0.7 s through it, more than the leg's floor, on a row v7 does not predict with
the booster. Excluding it is therefore not a choice made on a number but the
definition of what ships. December and October hold no such row.

The reference is the submission rung's own seed-0 fit on each leg, which must
first reproduce the ladder's stored residuals (December and July) to a
largest absolute difference of zero and October's printed figure; the
reference figure is then recomputed over the rows defined above from those
same residuals. The floors are the four `reports/baseline.md` prints, 0.23,
0.54, 1.02 and 0.21, measured over the ex-flagged rows; on July the row set
differs by one row, and the floor is kept as printed rather than remeasured,
which is the same approximation `reports/baseline.md` makes when it applies
one rung's floor to every rung. Beside every figure, the second reading over
the rows of 4 h or less, labelled a diagnostic and deciding nothing.

## The rule, unchanged, plus one condition

A candidate is **adopted** only if all five hold on the paired contrast
against the reference, at the precision `reports/baseline.md` prints:

- December, all ten: better by more than 0.23 s;
- July, all ten: better by more than 0.54 s;
- July, the three: not worse by more than 1.02 s;
- October: not worse by more than 0.21 s;
- **the twelve held-out months** (the splits of the held-out-month ladder,
  each 2025 month held out and the other eleven training), fitted once per
  month at seed 0 and pooled over the year: not worse than the submission
  rung's pooled seed-0 figure by more than the year's floor, 0.16 s, read from
  the five-seed report of the held-out months (a working file outside
  this repository, the year row, until that report is landed in `reports/`). This condition
  is added because those fits now exist for the base rung and cost twelve
  fits; it sizes a regression over the whole year and asks nothing of a
  gain, so a candidate that is better by less than the floor there has not
  failed it.

Combination, if more than one clears: as the 28 September note says, refitted
on the same legs and held to the same five conditions; the combination ships
only if it clears, else the single best-clearing candidate by the December
contrast. Whatever clears is refitted at seed 1 on December and July, and the
paired contrasts at seed 1 must have the same sign as at seed 0 on both legs,
else nothing is adopted.

## How a clearing candidate ships, fixed now

The re-freeze chain of 28 September is replaced by an increment:

- `taxiout.models` gains one recorded adoption, in the form
  `SEED_AVERAGE_ADOPTED` already takes: the adopted variation named, its
  figures quoted from the candidates report, and the default off until the
  rule's verdict is written into the report. The ladder's rung table,
  `SUBMISSION_KEY` and `BOOSTER_PARAMS` do not change; the variation is
  applied by `scripts/make_submission.py` on top of the submission rung when
  the adoption flag is on, and by nothing else.
- v8 is the five-seed average of the adopted arm on the full training year,
  with the term and the plan-shift rule as v7 has them. v8 must equal v7 on
  every row the term or the plan-shift rule covers, and differ from it only on
  the rows the booster predicts.
- The ladder, its reports, and the held-out-month reports are not
  regenerated. They remain the decomposition of the base rung, and the paper
  says that the shipped model is that rung plus one measured increment, with
  the increment's figures from `reports/candidates-2026-10-03.md` alone. The
  per-month figures of the adopted arm, from the fifth condition's twelve
  fits, are printed in that report beside the base rung's.
- The tag `model-v8-2026-10-<dd>` on the commit that adopts; the freeze tag
  and the v7 tag stay where they are. The one upload is then v8, not v7.

## Stops

- If the reproduction check of the reference fails, nothing is read.
- If the floors read from `reports/baseline.md` are not 0.23, 0.54, 1.02 and
  0.21, or the year floor read from the seeds report is not 0.16, stop.
- If the block has no verdict by 7 October 2026, 12:00 UTC, nothing is
  adopted, and the paper says why.
- If nothing clears, nothing changes: v7 ships, and the paper gains one
  paragraph saying that three candidates were measured after the freeze
  under this rule and none was adopted, with the figures.

## What is not decided by this

The score of v7 or v8. Whether the paper reads better for an increment. Any
comparison across the fit rules of earlier sweeps.

## What differs from the note

This copy is the working note of 3 October 2026 with one place changed and
nothing else: no candidate, leg, reference, condition, floor or stop differs
by a word. The note named a file by its path in the working program's
directories, which are named after its numbered work blocks and which this
repository does not publish. The working note this is copied from was
digested when it was written, sha256
`91e3306ef2b5e0efd6ab8d0a5c0ed9e2fc29ed0b0015bb4af8e96c6a2a601662`, before any
candidate was fitted under it; that digest is the proof of the order. What
the note said there, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| The rule, unchanged, plus one condition, lines 63 and 64 | named the five-seed report of the held-out months by its path in a working directory named after a numbered work block | "(a working file outside this repository, the year row, until that report is landed in `reports/`)" |

The digest the opening paragraph gives for the pre-registration of
28 September is that working note's. Its copy in this repository,
`reports/preregistered-2026-09-28.md`, differs from the note in the one place
its own closing section lists.

`tests/test_preregistered_copies.py` applies this substitution to the note
wherever the note is on hand, after checking its digest, and requires the
result to be this copy up to this section, byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This report uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
