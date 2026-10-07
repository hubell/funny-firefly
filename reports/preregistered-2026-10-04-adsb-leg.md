# Amendment of 4 October 2026 to the board pre-registration: where an ADS-B leg starts

Written on 4 October 2026 after the pilot stopped at the end of its Step 3
(a working report this repository does not publish), and fixed by its sha256 before any trace is
extracted, any departure matched, or any recorded block time read for the
pilot. It amends §5 of `preregistration_2026-10-04_board.md` (sha256
`568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`) in the two
places below and nowhere else. It is in force from Hunter's word, recorded in
his statements, a working note this repository does not publish.

## What exists as this is written

The four pilot days, downloaded and checked against their release metadata,
and their format read: the trace layout, the stale and new-leg bits counted on
2025-01-15, and three sample traces (the pilot's report, sections 2 and 3). No leg has
been extracted, no departure matched, and no recorded block time or taxi-out
read for any pilot day.

## 1. Every position is kept

§5 drops no position, and this keeps that. The stale bit (bit 1 of a point's
flags) marks the first position after about 20 s in which none was received;
the position itself is a fresh one (the pilot's report, section 3). Positions carrying
it are read like every other. The pilot's instruction "Stale positions are
dropped before anything is computed" is withdrawn.

## 2. Where a leg starts

§5 starts a leg at the aircraft's first position within the circle after its
previous airborne position there. For an aircraft seen landing earlier in the
day that is its first position on the landing roll, so E1 and E2 fall at the
landing and E3 does not exist (the pilot's report, section 4.2, the sample D-AILL). That
sentence of §5 is replaced by this one:

> A leg starts at the latest of: (i) the aircraft's first position within that
> circle after its previous airborne position there, or its first position of
> the day's file; (ii) its first position within that circle after a gap of
> 20 minutes or more in which no position of the aircraft was received; and
> (iii) the last position of a run of its ground positions within that circle
> below 1 kt lasting 20 minutes or more.

A *run* is a sequence of consecutive positions of the aircraft, each meeting
the run's condition, and its length is the time from its first position to its
last. This reading applies to every run §5 names. Everything else in §5 stands
as written: the take-off, E1 to E3 read within the leg, the counts and the
stationary time, the matching, the coverage, the pilot's days and readings,
the gate and its deadline, the scale-up, and the fusion model and its rule.

Twenty minutes is above a gap in ground coverage along a taxi route and above
most holds before take-off, and below the time an aircraft stands between two
flights. It is chosen from the format and one sample trace, not from any
result. Applied to D-AILL as the pilot's report describes it, the leg starts at
07:48:23Z, the first position after its 58-minute silence; E1 is 07:48:23Z, E3
is 07:50:59Z, the end of its stationary positions, and E2 is 07:51:23Z, the
first at 5 kt or more. The pilot read the first motion by eye at 07:51:15Z.

## What differs from the note

This copy is the working note of 4 October 2026 with eight places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Three of the places lie outside any rule. Five lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `6cdc0f76d32ea314e5df05216a56a8b63963538b4fe60b11a54ed3be2d7bc126`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 3 and 4 | named the pilot's work block and its report by its working file name | "after the pilot stopped at the end of its Step 3 (a working report this repository does not publish)" |
| the opening paragraph, lines 8 and 9 | named the file of Hunter's statements by its working file name | "recorded in his statements, a working note this repository does not publish." |
| What exists as this is written, line 15 | named the pilot's work block | "(the pilot's report, sections 2 and 3)" |
| §1, Every position is kept, line 23 (inside the rule) | named the pilot's work block | "(the pilot's report, section 3)" |
| §1, Every position is kept, line 24 (inside the rule) | named the pilot's work block and its working instructions | "The pilot's instruction" |
| §2, Where a leg starts, line 32 (inside the rule) | named the pilot's work block | "(the pilot's report, section 4.2, the sample D-AILL)" |
| §2, Where a leg starts, line 52 (inside the rule) | named the pilot's work block | "as the pilot's report describes it" |
| §2, Where a leg starts, line 55 (inside the rule) | named the pilot's work block | "The pilot read the first motion" |

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
