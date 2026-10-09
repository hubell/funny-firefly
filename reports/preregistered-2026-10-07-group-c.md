# Pre-registration of 7 October 2026: group C, for a possible v12

Written on 7 October 2026 in the evening, on Hunter's question "anything i can
run sequentially afterward? ... if we can improve the score with a follow up
build let me know and ill stage it", and fixed by its sha256 before any figure
it governs exists. It adds two candidates read on stored predictions and a
rule that decides whether a v12 is built. It reopens nothing already decided:
v11 is group B's combination with G1n refitted on it (the work from the overnight run of 6 October to the landing of v11).

## What exists as this is written

- **v11's base** (C14's choice per airport): C11's blend, 0.5 x
  `combo4_booster` + 0.5 x C11's member, at nine airports; LIRF at
  0.7 x `combo4_booster` + 0.3 x `c3b_member`. Its out-of-fold predictions are
  the overnight run's `heldout-2025-MM__c14__seed0` and the legs' `__c14__seed0`.
- **C10's grid was read on the old members only**, and chose 0.7, the edge
  of its grid (a working report this repository does not publish). No weight other than 0.5 has
  been read on C11's blend.
- **The fusion on the new base** (the fusion's refit on group B's base): the stacker of the fusion model's measurement (1,000 trees,
  learning rate 0.05, 63 leaves), G1n holding against G1 by 2.42 s. No other
  stacker setting has been read on either base.
- Every member prediction both candidates need is stored: `combo4_booster`
  and C11's member on the twelve held-out months and the three legs at seed 0,
  and on December and July at seed 1 (the overnight run's records).

## 1. C15, the booster's share in C11's blend

At the nine airports where v11 takes C11's blend, the blend becomes
w x `combo4_booster` + (1 - w) x C11's member, for w in {0.3, 0.4, 0.6, 0.7};
LIRF keeps v11's 0.7 blend of the old members. The w with the lowest RMSE over
the twelve held-out months' core rows is the candidate, a selection among four
weights disclosed as such. **The rule**: §4 of the board note's five
conditions and its second-seed check, against v11's base (C14) as the
reference. No fit.

## 2. C16, a larger fusion stacker

The stacker of the fusion model's measurement with 3,000 trees at learning rate 0.02, every other
parameter, feature, row and seed unchanged, refitted leave one month out on
the table of the fusion's refit on group B's base (the new base, its peers); G1n's out-of-fold airport gate
recomputed from these fits. **The rule**: G1n's three conditions against
G1n on the new base as the fusion's refit on group B's base fitted it: better pooled over the twelve
months by more than 0.16 s; not worse in any month by more than its floor;
better on December by more than 0.23 s. Twelve small fits.

## 3. v12

- If neither clears, there is no v12 and v11 stands.
- If one clears, v12 is v11 with it. If both clear, C16's stacker is refitted
  on C15's base (its out-of-fold predictions as `base`, the peers recomputed)
  and the pair is read by C16's rule against G1n on C15's base; if the pair
  fails, the candidate with the larger pooled gain on its own rule stands
  alone.
- Verdicts by 9 October 2026, 12:00 UTC. v12 is landed and built at a clean
  tree as v10 and v11 were, and uploaded once on Hunter's word by 10 October
  2026, 21:00 UTC; if the build cannot finish by then, there is no v12.

## 4. What does not change

The rules' rows (C6b and the plan-shift rule), the gate's mechanism, the
members' fits and seeds, and §6 of the board note: one upload per frozen
version, nothing chosen on a score, only `usedPairs` and the score read back.
Nothing is added after a number is seen.

## What differs from the note

This copy is the working note of 7 October 2026 with eight places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Five of the places lie outside any rule. Three lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `cb0ced3de2e6fb1a0a75bb8d9a4a41af29390ecee52356df25bbbf456d1be060`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 8 | named the work blocks from the overnight run of 6 October to the landing of v11 | "(the work from the overnight run of 6 October to the landing of v11)" |
| What exists as this is written, line 15 | named the work block of the overnight run | "the overnight run's `heldout-2025-MM__c14__seed0`" |
| What exists as this is written, line 17 | named the overnight run's report by its working file name | "(a working report this repository does not publish)" |
| What exists as this is written, line 19 | named the work blocks that refitted the fusion on group B's base and that measured the fusion model | "(the fusion's refit on group B's base): the stacker of the fusion model's measurement" |
| What exists as this is written, line 24 | named the work block of the overnight run | "(the overnight run's records)" |
| §2, C16, line 38 (inside the rule) | named the work block that measured the fusion model | "The stacker of the fusion model's measurement with 3,000 trees" |
| §2, C16, lines 39 and 40 (inside the rule) | named the work block that refitted the fusion on group B's base | "on the table of the fusion's refit on group B's base" |
| §2, C16's rule, line 42 (inside the rule) | named the work block that refitted the fusion on group B's base | "as the fusion's refit on group B's base fitted it" |

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
