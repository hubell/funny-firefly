# Pre-registration of 28 September 2026: three candidates after the freeze

Written on 28 September 2026, before any of the figures it governs exist and
before the score of `funny-firefly_v5.parquet` has been read from the
organizers' scorer. The working note this is copied from was digested when
it was written, sha256
`eee5b947e4c449eb7df5765ca62d25a3ca954f201ecefa23311eec8918813e60`, before
any of the figures it governs existed; that digest is the proof of the order.
This copy differs from the note only in the place listed at its end.

## What is fixed

The model that ships is the five-seed average of `+ plan differences` under
`BOOSTER_PARAMS` as committed at `bfe43f1`, tagged `freeze-2026-10-04`. Three
candidates, each already measured once and left on the table, are fitted on
top of that rung, single seed, and read against it under the rule below. The
leaderboard is not read for any of it: the criteria are here, the verdicts
are computed, and the score of v5, whatever it is, changes nothing in this
file.

## The candidates

1. **Saturation on top of `+ plan differences`.** The runway-saturation block
   (`taxiout.saturation.SATURATION_FEATURES`) added to the submission rung.
   Under the shipped cell it reads 232.7 against 233.2 on December and 343.4
   against 344.1 on July when added to `+ weather`; it has never been fitted
   on top of the rung that ships.
2. **The mild recency schedule.** Row weights halving every 12 months over
   the fitting rows, as `scripts/preregistered.py` implements them, on the
   submission rung. On 20 September, under the parameters then in force, it
   gained 0.84 s on July past that leg's floor and was stopped by the
   printed-precision screen alone.
3. **More trees.** `n_estimators` 8,000 in place of 6,000, every other
   parameter unchanged. October was still falling at 6,000 when the
   extension's criteria closed the axis.

Nothing else is tried. No fourth candidate is added after a number is seen.

## The legs, and the reference

Each candidate is fitted once per leg, seed 0, on the legs the ladder uses:
December (train January to November 2025, validate December, all ten
airports, with the three July airports read from the same fit), July (train
January to June, validate July, all ten and the three), and October (train
January to September, validate October), the leg the parameter search was
read on. The reference is the submission rung's own seed-0 fit on each leg,
which must first reproduce the ladder's stored residuals to a largest absolute
difference of zero, as `scripts/preregistered.py` requires; nothing is read
until it does.

Every figure is the ex-flagged RMSE at the precision `reports/baseline.md`
prints, and every contrast is paired within one seed.

## The rule

A candidate is **adopted** only if all four hold on the paired contrast
against the reference:

- December, all ten: better by more than the floor, 0.23 s;
- July, all ten: better by more than the floor, 0.54 s;
- July, the three: not worse by more than the floor, 1.02 s;
- October: not worse by more than the floor, 0.21 s.

The floors are the four `reports/baseline.md` prints today in *Seed
averaging* and *The noise floor* under the parameters that ship (0.23, 0.54,
1.02 and 0.21), and they are not remeasured for this.

If more than one candidate clears, the clearing candidates are combined into
one arm, refitted on the same three legs at seed 0, and held to the same four
conditions against the same reference; the combination ships only if it
clears, else the single best-clearing candidate by the December contrast
ships. Whatever clears is then refitted at seed 1 on December and July, and
the paired contrasts at seed 1 must have the same sign as at seed 0 on both
legs, else nothing is adopted: that is the winner's-curse check of
22 September, applied as written.

If nothing clears, **nothing changes**: the freeze stands at `bfe43f1`, and
the paper gains one paragraph saying that three candidates were measured
after the freeze under this rule and none was adopted, with the figures.

## What adoption costs, stated now

An adopted candidate means a new model: the ladder and every report
regenerated, the paper's figures moved under `scripts/check_paper.py
--strict`, a new tag (the old one never moves), a clean-checkout
reproduction, and a v6 upload as a second frozen version. That is about two
machine-days and a paper pass, and the deadline is 4 October. If the
candidate block finishes too late for that chain to complete, nothing is
adopted, and the paper says why.

## What is not decided by this

The score v5 receives. Whether the paper reads better or worse for it. Any
comparison across the fit rules of earlier sweeps, which are not comparable.

## What differs from the note

This copy is the working note of 28 September 2026 with one place changed and
nothing else: no candidate, leg, reference, condition, floor or rule differs
by a word. The note named the numbered work block of the working program that
would copy it into this file, and where its digest was recorded, which this
repository does not publish; its digest, above, is the proof of the order.
What the note said there, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 5 to 7 | named the work block that would copy it, unchanged, into this file and freeze it under its digest, and where that digest was recorded before the upload | "The working note this is copied from was digested when it was written, sha256 `eee5b947e4c449eb7df5765ca62d25a3ca954f201ecefa23311eec8918813e60`, before any of the figures it governs existed; that digest is the proof of the order. This copy differs from the note only in the place listed at its end." |

`tests/test_preregistered_copies.py` applies this substitution to the note
wherever the note is on hand, after checking its digest, and requires the
result to be this copy up to this section, byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This report uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
