# Amendment of 6 October 2026 to the last-week pre-registration: G1n and C9b

Written on 6 October 2026 after the verdicts of the measurement of group A, on Hunter's answer
"Amend: G1n and C9b", and fixed by its sha256 before any figure it governs
exists. **Both candidates are chosen after 22d's numbers were seen, and the
paper says so.** It amends §2 of `preregistration_2026-10-06_last_week.md`
(sha256 `eb35ecd3eb802b077307d5dba6ab07ca0cc5277b6fc7d5522d1b65a86bc9f1d5`).
22d's verdicts stand as measured: G1 adopted, C6c and C9 not adopted.

## What exists as this is written

- **The measurement of group A** (a working report this repository does not publish): G1 better than the base
  pooled by 7.49 s and on December by 9.67 s, holding in every month; C9 worse
  pooled by 0.20 s, missing December and failing July and August, gaining
  98 s on LIRF's 2,482 rows at the schedule with a take-off 2,400 s or more
  after it and losing 10.9 s on LIRF's other rows. The v10 candidate
  (a working file outside this repository), sha256
  `15e02afe290744e85838607d785349abb19844577d00a10e9e60c600d307603b`.
- **The review of G1** (`workingdir/gap_2026-10-06/g1_review.py` and `.out`),
  on 22d's out-of-fold rows: on covered rows the gated fusion is worse at EGLL
  (+17.88 s; 10 of 12 months), LIRF (+27.39 s; 7 of 8 months with rows) and
  LFPG (+10.07 s; 6 of 12 months), and better at EDDF, EDDM, EHAM, LEBL,
  LEMD and LSZH. In July 2026 the candidate moves EGLL rows by a
  median 64 s against 26.5 s on EGLL's 2025 covered rows. No nested gate and
  no late-row restriction of C9 has been computed.

## 1. G1n, the fusion gated by coverage and by airport, out of fold

From 22d's rows (a working file outside this repository, sha256
`ee353676370ee59c67887dbe7f0ab418a968a0eda0206400db943103b3b5015d`), whose
`fusion` values are each out of fold for their own month:

- **The gate, per held-out month m and airport a**: over airport a's covered
  rows (`a_covered` or `o_covered` 1) of the other eleven months, the RMSE of
  `fusion` and of `base` against `y`. Airport a keeps the fusion in month m
  only if the fusion's RMSE is strictly lower; otherwise its covered rows in
  month m take the base. An airport with no covered row in the other eleven
  months takes the base.
- **G1n**: a row takes `fusion` where it is covered and its airport keeps the
  fusion in its month; `base` everywhere else.
- **The rule**: G1's three conditions with **G1 as the reference**: better
  than G1 pooled over the twelve months by more than 0.16 s; not worse than G1
  in any month by more than its floor; better than G1 on December by more
  than 0.23 s. On the rows of 22d's G1 reading (989,825).
- **For the ranking rows**, the gate is decided the same way over all twelve
  months' covered rows, and printed per airport before the candidate is
  built.

## 2. C9b, Fiumicino's schedule layer on late departures only

C9's twelve leave-one-month-out predictions as 22d stored them
(a working file outside this repository, sha256
`aee57ec7e361351cae11e0635956349ed71959c2d438e4cdd2da8dbed79ef4eb`),
nothing refitted, applied only to LIRF core rows whose take-off lies 2,400 s
or more after their scheduled time (`taxiout.tails.lag_seconds`; a row
without a scheduled time is not late). Every other row keeps the base.
**The rule**: C9's, against the base, over the core rows of the twelve
held-out months. **For the ranking rows**: C9 fitted on LIRF's short rows of
all twelve months as §2 of the last-week note defines it, applied to the same
late rows.

## 3. v10

- **The fusion part**: G1n if adopted, otherwise G1.
- **C9b joins** only if it clears its rule and, on 22d's G1 rows, the pair
  holds G1's three conditions against the base: a row takes the fusion
  part's value where the fusion part applies, otherwise C9b's value where it
  is a late LIRF core row, otherwise the base.
- The term's rows from C6b and the plan-shift rule as in v9. Verdicts by
  7 October 2026, 18:00 UTC. v10 uploaded once, on Hunter's word, by
  8 October 2026, 21:00 UTC; the landing of v10 reproduces it from committed code
  first.

## 4. What does not change

Group B of the last-week note, its rules and dates. §6 of the board note:
one upload per frozen version, nothing chosen on a score, only `usedPairs`
and the score read back. Nothing is added after a number is seen.

## What differs from the note

This copy is the working note of 6 October 2026 with six places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Three of the places lie outside any rule. Three lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `4e6958eb394e72d663234ce364ee93982447834909436f593c2416e7080805f4`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 3 | named the work block that measured group A | "after the verdicts of the measurement of group A" |
| What exists as this is written, line 12 | named that work block and its report by its working file name | "**The measurement of group A** (a working report this repository does not publish)" |
| What exists as this is written, line 17 | named the candidate's path in that work block's directory | "(a working file outside this repository), sha256" |
| §1, G1n, line 29 (inside the rule) | named the rows' path in that work block's directory | "(a working file outside this repository, sha256" |
| §2, C9b, line 52 (inside the rule) | named the rows' path in that work block's directory | "(a working file outside this repository, sha256" |
| §3, v10, line 71 (inside the rule) | named the work block that lands v10 | "the landing of v10 reproduces it from committed code" |

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
