# Amendment of 4 October 2026 to the board pre-registration: the tail rules restored, and a candidate for their rows

Written on 4 October 2026 after v8's score, and because of it. It is fixed by
its sha256 before any figure it governs exists. It withdraws §1 of
`preregistration_2026-10-04_board.md` (sha256
`568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`), rewrites
the last sentence of its §4 and of its §5, and adds one candidate, C6, with its
own rule. It is in force from Hunter's word, recorded in
his statements, a working note this repository does not publish.

## What exists as this is written

- **v7**, uploaded once: 313.169 over 344,841 pairs
  (`reports/submission-2026-10-04.md`).
- **v8**, uploaded once at 13:17:51Z: 533.1066 over 344,841 pairs, processed
  at 13:18:41Z (`reports/submission-2026-10-04-v8.md`). v8 equals v7 on all
  344,457 rows the two rules did not touch and differs on all 384 they did
  (a working file outside this repository).
- **The two scores read together**, for expectations only:
  `workingdir/v8_reading_2026-10-04/two_scores.py` and its output. Nothing in
  either rule is refitted to them, and no rule below reads them.

## Why §1 is withdrawn

§1 rested on one judgment: that the 384 rows' truth behaving as 2025's did
(reading B), scored in floating point, was the reading the board made
implausible. v7's and v8's scores contradict it.

The two files differ only on the 384 rows, so the difference of their squared
scores, 186,127.8 s², is what substituting the rules is worth there: it lowers
the pooled mean squared error by that much. Of §1's four readings, A scored in
floating point is excluded by v7's score alone (its rules would add 218,351 s²,
more than v7's whole squared score, 98,074.8 s²). From the core v7's score
implies, A with int32 squares puts v8 at 234.0 s and B with int32 squares at
305.2 s; B in floating point puts it at 554.5 s, against 533.1066 s read. With
no reading assumed, in floating point the truth on the 384 rows sits, in the
mean weighted by v7's prediction minus v8's, 0.944 of the way from v8's values
to v7's; reading A gives -0.020 and reading B 1.000.

So a file that predicts those rows with the booster, as v8 does, reads at
least the square root of 186,127.8 s², 431.4 s, whatever its core, and none
could improve on v7. In the reading the scores leave, §1's own table prices
keeping both rules at 25,129 s² and dropping them at 243,448 s².

This is a choice made after a score and on it, and it is stated as one: the
paper reports v7, v8, this amendment and its reason. What it restores was
fixed and validated before any score of this entry existed, and nothing in it
changes.

## 1. The rules restored

Every file uploaded from now (v9, v10) substitutes the record-quality term and
the plan-shift rule exactly as v7 did, as `reports/preregistered-2026-10-02.md`
and the paper's §3.1, steps 6 and 7, describe: the same cells, shares, lags,
rest means, coverage rule and add-one share, fitted on the same rows. Nothing
in either rule is refitted, tuned or chosen on any score. Where §2 of this
amendment adopts C6, C6 replaces the term on the term's rows and on nothing
else.

The last sentence of §4 now reads: "**v9** is the adopted configuration, five
seeds, the record-quality term (or C6 where adopted) and the plan-shift rule
substituted, uploaded once." The last sentence of §5 now reads: "**v10** is
v9's booster with the fusion applied, the two rules' rows as in v9, uploaded
once."

## 2. C6, the term's rows predicted from the schedule

**The rows.** In 2025, the training rows in the term's covered cells:
Fiumicino departures without a flight record (`LOBT_flt` null) in the cells
`noflt:normal`, `noflt:4-12h` and `noflt:>12h`. In the ranking set, the rows
the term covers (383). The plan-shift rule's row is not part of C6.

**What both candidates use.** A row is *at the schedule* when its block time
lies within 60 s of its scheduled time. On such a row the recorded taxi-out
equals the lag to within 60 s, whatever the lag (`reports/tails.md`, *The
callsign-prefix reading*). The term counts a row as defaulted only when its lag
also exceeds 2 h, so rows at the schedule with shorter lags enter its rest mean;
over Fiumicino's departures without a record, 48.5% are at the schedule and
26.6% are counted as defaulted (same section).

**C6a, the schedule share by lag.** The term's form, share × lag + (1 − share)
× rest mean, with two changes. The share is the share of rows at the schedule,
with no lag condition, and the rest mean is over the rows not at the schedule.
And `noflt:normal` is split by lag at 0, 1 h and 2 h into four sub-cells:
[−1 h, 0], (0, 1 h], (1 h, 2 h] and (2 h, 4 h]. A sub-cell with fewer than
five fitting rows takes the values C6a computes for its whole cell. In the
other two cells every lag exceeds 4 h, so their shares and rest means are the
term's.

**C6b, the share per row.** As C6a, but each row's share is the probability
that it is at the schedule, from a LightGBM classifier fitted on the same rows:
objective binary, 300 trees, learning rate 0.03, 15 leaves, minimum child 20,
no bagging, deterministic, column-wise, eight threads, seed 20260902. Features:
the lag, the sub-cell (the four of `noflt:normal` and the other two cells), the
scheduled hour and the take-off hour in UTC, the weekday of the scheduled time,
and three categorical fields, the callsign prefix (the first three letters of
`FLIGHT_mvt`), `STAND_mvt` and `AIRCRAFT_TYPE_mvt`, each level with fewer than
ten fitting rows read as one level, `other`. The rest mean is C6a's for the
row's sub-cell.

Both predictions are floored at 0 s and rounded as the term's are.

**The rule.** Read leave one month out over the twelve months of 2025: each
month's rows are predicted by the candidate and by the term, both fitted on the
other eleven months' rows (cells, coverage, shares, rest means, and C6b's
classifier). Each candidate predicts exactly the rows that fold's term covers,
and only those rows are measured. The measure is the mean squared error in s²
against the recorded taxi-out as it is, long values included. A candidate is
adopted only if (1) pooled over the twelve months its figure is lower than the
term's; (2) the 95% interval of the pooled difference, from 2,000 resamples of
days with seed 20260902, lies wholly below zero; and (3) its figure is not
higher than the term's in January or in July. If both clear, the one with the
lower pooled figure is adopted. The verdict is written with §4's, by 7 October
2026, 12:00 UTC, or the term stays. An adopted candidate is fitted on every
fitting row of the term's cells and predicts the term's ranking rows in v9 and
v10.

## 3. What does not change

§2, which governed v8. §3 and §4 for C1 to C5, their rows and their rule. §5,
but for its last sentence and the separate amendment of its leg start. §6: no
score chooses among the files uploaded, each is uploaded once when its rule has
spoken, and only `usedPairs` and the score are read back.

## What differs from the note

This copy is the working note of 4 October 2026 with two places changed and
nothing else: no rule, threshold, definition or stop differs by a word. The
note named a working note and a working file of the working program by their
file names, which this repository does not publish. The working note this is
copied from was digested when it was written, sha256
`4d388f1d65a56ee06a66c45598a3231bbac63e125dc0b229e931cddf6641efeb`, before any
figure it governs existed; that digest is the proof of the order. What the
note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 8 and 9 | named the file of Hunter's statements by its working file name | "recorded in his statements, a working note this repository does not publish." |
| What exists as this is written, line 18 | named the comparison of v7 and v8 by its path in a working directory named after a numbered work block | "(a working file outside this repository)" |

Two names are left as the note words them, because they are neither a numbered
work block nor a dated working note's file name: the board pre-registration's
working file, in the opening paragraph, whose published copy is
`reports/preregistered-2026-10-04.md` (the same bytes under the same digest);
and, under *What exists as this is written*, the working directory of the
reading of the two scores, which this repository does not carry.

The credit paragraph below is the note's own last paragraph, unchanged, placed
after this section so that it stays the last paragraph.

`tests/test_preregistered_copies.py` applies these substitutions to the note
wherever the note is on hand, after checking its digest, and requires the
result to be this copy up to this section, and its last paragraph, byte for
byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This document uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
