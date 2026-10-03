# Pre-registration of 2 October 2026: the plan-shifted departures

Written on 2 October 2026, before the rule below has been evaluated on any
fold, before v7 exists, and before any score exists for any version of this
entry. The working note this is copied from was digested when it
was written, sha256
`d62b48f32d3ba9da07aa346cb04a2560feaff9664a9734228d7523e669dda9a1`, before the
rule was evaluated on any fold; that digest is the proof of the order. This
copy differs from the note only in the places listed at its end.

## Why there is a rule at all

Hunter decided on 2 October, after reading the held-out-month ladder, that the record-quality
term is extended to one more class of departure and that the entry ships v7
instead of v6. The class: a departure with a flight-plan record whose take-off
falls hours after both of the plan's off-block estimates. In the training
year three departures have it; two recorded taxi-outs over four hours, each
within minutes of take-off minus its last off-block estimate, and one recorded
an ordinary taxi-out. In the ranking set one departure has it (Fiumicino,
July 2026, both estimates about 24 h before take-off), and v6 predicts 3,120 s
for it. The term does not cover these departures because its cells key on the
schedule lag and on the absence of a record, and they have a record.

This is a rule chosen after the class was seen in the ranking set's exposed
fields, on evidence of three training rows. It is written down here so that
nothing about it is chosen after its effect is measured, and the paper says
both of those things.

## The rule, fixed here

- **The class.** A departure, training or ranking, not covered by the
  record-quality term of `taxiout.tails`, whose take-off minus `EOBT_1_flt`
  and take-off minus `LOBT_flt` both exceed `TAIL_EXCLUSION_ADOPTED_H`, the
  four hours already fixed in `taxiout.models` on the mechanism that no
  aircraft taxis for four hours. Both estimates must be present, which
  requires a flight-plan record.
- **Long.** A training departure of the class whose recorded taxi-out exceeds
  the same four hours.
- **The share.** `(long + 1) / (n + 2)` over the class's departures in the
  rows a fit is given, where `n` is their count: the add-one estimate, because
  the class is far below the five rows `taxiout.tails.MIN_COUNT` requires of a
  cell and a share of `long / n` from one or three rows would assert a
  certainty the data cannot carry.
- **The lag.** Take-off minus `LOBT_flt`, the plan's last known off-block
  time.
- **The rest mean.** The mean recorded taxi-out of the training departures at
  the same airport that have a flight-plan record and a recorded taxi-out of
  four hours or less, in the rows a fit is given.
- **The prediction.** `share x lag + (1 - share) x rest_mean`, the form the
  record-quality term already uses, substituted for the booster's prediction
  on the class's rows, after the term and before the rounding and the clip at
  zero that every prediction goes through.
- **Nothing else.** The booster, its rows, its parameters, its seeds, the
  record-quality term's cells and the rows it covers are unchanged. The rule
  reads only `EOBT_1_flt`, `LOBT_flt`, the take-off time, the departure
  airport and, at fit time, the recorded taxi-out; it never reads
  `AOBT_3_flt`, which stays excluded by decision.

## What is measured, and what it can and cannot show

On the twelve held-out months of the held-out-month ladder and the two forward legs, the rule
fitted on each split's training rows is applied to its validation rows and the
change in squared error is printed per month, every row, with and without the
term. In the training year the class holds three departures, so this
evaluation shows what the rule does to those rows and nothing more: it is a
statement of the arithmetic and not independent evidence that the rule is
right. The paper says so.

## Stops

- If the class in the ranking set holds any departure other than the one
  described above, stop: the rule was written knowing of one.
- If the class in the training year does not hold exactly three departures,
  two of them long, stop: the rule was written knowing of those.
- If v7 differs from v6 in any row other than the class's, stop.

## What is not decided by this

The score of v7. Whether the rule is right for the 2026 departure, which no
data available to the entry can settle.

## What differs from the note

This copy is the working note of 2 October 2026 with three places changed and
nothing else: no rule, threshold, definition or stop differs by a word. The
note named numbered work blocks of the working program and where its digest was
recorded, which this repository does not publish; its digest, above, is the
proof of the order. For each place, what the note said and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 5 to 8 | named the work block that would copy it, unchanged, into this file, and where its digest was recorded | "The working note this is copied from was digested when it was written, sha256 `d62b48f32d3ba9da07aa346cb04a2560feaff9664a9734228d7523e669dda9a1`, before the rule was evaluated on any fold; that digest is the proof of the order. This copy differs from the note only in the places listed at its end." |
| Why there is a rule at all, line 12 | "after reading" a numbered work block | "after reading the held-out-month ladder" |
| What is measured, line 60 | "the twelve held-out months of" a numbered work block | "the twelve held-out months of the held-out-month ladder" |

`tests/test_tails.py` applies these substitutions to the note wherever the note
is on hand, after checking its digest, and requires the result to be this copy
up to this section, byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This report uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
