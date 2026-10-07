# Amendment of 6 October 2026 to the last-week pre-registration: airport by airport

Written on 6 October 2026, after the verdicts of the measurement of group A and before the measurement of G1n and C9b,
on Hunter's words "we can take a airport per airport approach to optimize
scores - i plan on leaving it overnight to run", and fixed by its sha256
before any figure it governs exists. It amends §3 (group B) of
`preregistration_2026-10-06_last_week.md` (sha256
`eb35ecd3eb802b077307d5dba6ab07ca0cc5277b6fc7d5522d1b65a86bc9f1d5`). The
fusion's airport-by-airport gate is G1n, in
`preregistration_2026-10-06_g1n_c9b.md` (sha256
`4e6958eb394e72d663234ce364ee93982447834909436f593c2416e7080805f4`), and is
not touched here.

## What exists as this is written

- Group B's candidates (W, W2, C10, C11, C12) are defined and none is
  measured. Group A's verdicts are in (the measurement of group A); G1n and C9b are not yet
  read.
- Per-fit times measured in the reading of the candidates C1 to C5, C7 and C8 (a working report this repository does not publish), mean
  LightGBM fit with prepare and predict: December 347 s, July 189 s, October
  276 s, a held-out month 341 s; the CatBoost member on CPU: December 939 s,
  July 497 s.
- No per-airport fit of any kind exists.

## 1. Every held-out month is fitted

C11 and C12 are fitted on the three legs and all twelve held-out months
whatever their leg readings, so that every menu item of §3 has out-of-fold
predictions everywhere. Their own verdicts are still read by §4's five
conditions and second-seed check, unchanged. This replaces "the twelve
held-out months only if the four leg conditions hold" for C11 and C12.

## 2. C13, one booster per airport

For each of the ten airports, a LightGBM booster under `BOOSTER_PARAMS`,
unchanged, fitted on that airport's fitting rows only, with the columns of
`combo4_booster` (the rung's, C1's nine, C4's two); the airport column stays
in the frame and is constant within a fit. The ten predictions together make
the C13 booster, blended at 0.5 with `c3b_member`. Fitted on the three legs,
the twelve held-out months, and at the second seed on December and July.
Read alone by §4's five conditions and second-seed check against the base.

## 3. C14, the airport-by-airport selector

**The menu**, each an out-of-fold prediction on every split: the base
(`combo4`); the base's members at booster weights 0.3, 0.4, 0.6 and 0.7
(C10's grid); C11's blend; C12's blend; C13's blend. Eight items.

**The choice**, per airport a and held-out month m: among the menu, the item
with the lowest RMSE over airport a's core rows of the other eleven held-out
months; the base unless another is strictly lower. On a forward leg of month
m (December, July, October), the choice uses the held-out months other than
m, the same as for held-out month m; this lets months after a forward leg's
month inform its choice, and the paper says so. For the ranking rows, the
choice per airport uses all twelve held-out months.

**The rule**: §4's five conditions against the base. The second-seed check
applies each airport's seed-0 choice to that item's second-seed predictions
on December and July, fitted if not stored.

The choice table (airport by month) is printed before any contrast.

## 4. The combination, replacing §3's "The combination"

- **If C14 holds its rule**, the new base is C14's choice per airport; C10 to
  C13 enter only through it.
- **Otherwise**, the new base is §3's combination of the last-week note
  (C12 as the booster, C11 as the member, C10's weight) over whichever of
  C10 to C12 cleared alone; C13 enters only through C14.
- **W or W2** then applies on top, its means recomputed from the new base's
  out-of-fold residuals as C2a's layer defines them.
- The combination is held to §4's five conditions against v9's configuration
  and, if W or W2 is in it, to W's winter conditions. If it fails, there is
  no v11. If it holds, group A's adopted pieces (the fusion part and C9b) are
  refitted on the new base's out-of-fold predictions and re-read by their own
  rules; one that no longer holds is left out of v11. Verdicts by
  9 October 2026, 18:00 UTC; v11 uploaded once, on Hunter's word, by
  10 October 2026, 21:00 UTC.

## 5. What does not change

W and W2 as defined, with W2 fitted only if W clears. §4's legs, floors and
second-seed check. §6 of the board note: one upload per frozen version,
nothing chosen on a score. Nothing is added after a number is seen.

## What differs from the note

This copy is the working note of 6 October 2026 with three places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
None of the places lies inside a rule. No rule, threshold, definition or
stop differs by a word.
The working note this is copied from was digested when it was written,
sha256 `9ef5691f36365e2691204380b5f4e984d7049a0e2a8922f80943ed5309724a4a`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 3 | named the work blocks that measured group A and then G1n and C9b | "after the verdicts of the measurement of group A and before the measurement of G1n and C9b," |
| What exists as this is written, line 17 | named the work block that measured group A | "(the measurement of group A)" |
| What exists as this is written, line 19 | named the work block that read those candidates and its report by its working file name | "Per-fit times measured in the reading of the candidates C1 to C5, C7 and C8 (a working report this repository does not publish)" |

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
