# Amendment of 4 October 2026 to the board pre-registration: two more candidates, C7 and C8

Written on 4 October 2026 after Hunter's "all go. except pru aggregate"
(his statements, item 26), and fixed by its sha256 before any
figure for either candidate exists. It adds two candidates to §3 of
`preregistration_2026-10-04_board.md` (sha256
`568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`) and states
how they join §4's combination. Everything else in §3 and §4 stands. It is in
force from Hunter's word, recorded in his statements, a working note this repository does not publish.

## What exists as this is written

The record commit of the second upload (`93732cb`) is HEAD; the measurement of
section 3's candidates has not started. No
figure exists for C7 or C8. The catalogue (a working note this repository does not publish) lists
them as E3 and D6.

## C7, thunderstorms, from the METAR already in the deposit

The catalogue's E3 named ERA5's convective fields. A METAR's present weather
observes a thunderstorm at the airport itself rather than on a reanalysis
grid tens of kilometers wide, and it is already fetched and registered
(`METAR.wxcodes`), so C7 reads it alone and fetches nothing. Per airport-hour,
each the worst reading in the hour, carried forward and marked missing exactly
as `taxiout.metar` carries the regime flags:

- `wx_ts`: 1 if a present-weather group at the aerodrome (not prefixed `VC`)
  contains `TS`;
- `wx_vcts`: 1 if a group is `VCTS` and `wx_ts` is 0;
- `wx_heavy`: 1 if a group carries the heavy-intensity prefix `+`;
- `wx_ts_prev3`: how many of the three hours before had `wx_ts` 1.

A departure reads the hour its take-off falls in, as for the regime flags, and
`wx_ts` of the hour before it. The four plus that one are added to the
submission rung's features.

## C8, the training rows weighted toward the target month's regime

For every fit whose target rows are one month's (a forward leg's month, a
held-out month, or for a submission January 2026's or July 2026's ranking
rows):

- a classifier separating the target rows (label 1) from that fit's training
  rows (label 0): LightGBM, objective binary, 200 trees, learning rate 0.05,
  31 leaves, minimum child 500, deterministic, column-wise, eight threads,
  seed 20260902; its features the airport, `WEATHER_FEATURES` and
  `CONGESTION_FEATURES` as the rung builds them, and no calendar or date field;
- each training row's probability p by two-fold cross-fitting, the folds being
  the even and the odd calendar days; its weight p/(1 − p), clipped to
  [0.1, 10], then scaled to a mean of 1 over the training rows;
- the booster refitted with those weights under `BOOSTER_PARAMS`; blend
  members and layers, where adopted, unweighted.

For a submission, January 2026's rows are predicted by the booster weighted
toward them and July 2026's by the one weighted toward July's, five seeds
each.

## The rule and the combination

§4's rule, unchanged in its legs, its reference, its rows, its floors and its
second-seed check, reads C7 and C8 each against the submission rung, C7 first.
If the measurement of section 3's candidates adopts a configuration, the combination is that configuration with
whichever of C7 and C8 cleared (C7's features beside its features, C8's weights
on its booster's fits), built in §4's order and held to the same five
conditions; if it fails, the configuration of the measurement of section 3's candidates stands alone. If the measurement of section 3's candidates
adopts nothing, C7 and C8 combine as §4 combines candidates: both together if
both cleared, held to the five conditions, else the one clearing candidate with
the larger December gain. The verdict is written by 7 October 2026, 12:00 UTC,
or neither is adopted. v9 is the result.

## What differs from the note

This copy is the working note of 4 October 2026 with seven places changed and
nothing else. The note named numbered work blocks of the working program and
two working notes by their file names, which this repository does not publish.
Four of the places lie outside any rule. Three lie inside the rule, in *The
rule and the combination*, where the note named the work block that measured
the candidates of section 3 of the board pre-registration; there Hunter decided
on 6 October 2026 that the copy names that measurement in words instead, "the
measurement of section 3's candidates", and the rule refers to the same
measurement as before. No other word of the rule, and no threshold, definition
or stop, differs. The working note this is copied from was digested when it was
written, sha256
`4ac17be6925d03255a2e52b1d1ee307e297e913ee42f33c063a03d8e36fbe8cd`, before any
figure for either candidate existed; that digest is the proof of the order.
What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 4 | named the file of Hunter's statements by its working file name | "(his statements, item 26)" |
| the opening paragraph, line 9 | the same file, by the same name | "recorded in his statements, a working note this repository does not publish." |
| What exists as this is written, line 13 | named the work block that made the record commit, and the work block that would measure the candidates | "The record commit of the second upload (`93732cb`) is HEAD; the measurement of section 3's candidates has not started." |
| What exists as this is written, line 14 | named the catalogue by its working file name | "(a working note this repository does not publish) lists" |
| The rule and the combination, line 61 (inside the rule) | "If" and the work block that measured the candidates, "adopts a configuration" | "If the measurement of section 3's candidates adopts a configuration" |
| The rule and the combination, line 64 (inside the rule) | that work block's configuration, "stands alone" | "the configuration of the measurement of section 3's candidates stands alone" |
| The rule and the combination, lines 64 and 65 (inside the rule) | "If" and that work block, "adopts nothing" | "If the measurement of section 3's candidates adopts nothing" |

One name is left as the note words it, because it is neither a numbered work
block nor a dated working note's file name: the board pre-registration's
working file, in the opening paragraph, whose published copy is
`reports/preregistered-2026-10-04.md` (the same bytes under the same digest).

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
