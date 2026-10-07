# Amendment of 4 October 2026 to the board pre-registration: OPDI's flight events beside adsb.lol

Written on 4 October 2026 after the organizers confirmed OPDI as open to the
challenge (his statements, items 24 and 25), and fixed by its
sha256 before any OPDI event is matched to a departure or compared with any
recorded block time. It amends §5 of `preregistration_2026-10-04_board.md`
(sha256 `568299d1f2fd63cd049c83d40bfab3a0af6ebe548ad0318d23c7a26a992d3990`),
as amended for its leg start by `preregistration_2026-10-04_adsb_leg.md`, by
adding a second source of ground events and widening the fusion model's
features. Everything else in §5 stands. It is in force from Hunter's word,
recorded in his statements, a working note this repository does not publish.

## What exists as this is written

Two counts on public OPDI files, made without any challenge data
(`workingdir/pivot_2026-10-04/opdi_probe/README.md`). Departures first heard
on the ground at least 9 minutes before their first crossing of FL50, in
January 2025, January 2026 and July 2026: EDDF 20%, 72%, 48%; EHAM 50%, 32%,
32%; LEBL 53%, 44%, 73%; LSZH 48%, 59%, 66%; EGLL 39%, 6%, 24%; EDDM 23%, 8%,
17%; LEMD 2%, 22%, 10%; LFPG, LIRF and LTFM 3.5% or less. Stand, apron and
taxiway events at least 5 minutes before the take-off runway entry are
narrower: LSZH 57% to 96%, EDDF 25% and 38% in 2026 only, LEBL 29% in January
2025, EDDM up to 12%. The same aircraft arrives at the airport within the 12
hours before 54% to 90% of departures. No de-icing pad event appears at the ten
airports. No OPDI event has been matched to a departure.

## 1. The source

OPDI v0.0.2: the flight events (ten-day files at
`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events/`)
and the flight lists (monthly files at
`https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_list/`),
fetched by a scripted, cached client. The site states no data licence; the
entry uses OPDI on the organizers' confirmation (item 25), which the paper
cites. No OPDI file, and no table derived from one, enters the deposit: the
deposit carries the client and the extraction code, which fetch the published
files again.

## 2. Flights, events and matching

An event's `flight_id` is its flight's `id` in the flight list, which gives the
flight's `icao24`, callsign (`flt_id`), `adep`, `ades`, `first_seen` and
`last_seen`.

**The departure's flight.** A departure matches the OPDI flight whose `adep` is
the departure's airport, whose callsign (spaces removed) equals `CALLSIGN_flt`
or `FLIGHT_mvt` (spaces removed), and whose `first_seen` is no later than 600 s
after `MVT_TIME_UTC_mvt` and `last_seen` no earlier than it; the one whose
`first_seen` is nearest that take-off if several.

**Its events**, all at the airport (the event's `osm_airport`, or for O0 the
position) and before the flight's first crossing of FL50: **O0**, the flight's
`first_seen` event where it is a surface position (no altitude) within 4 km of
the airport's reference point (OurAirports, as the pilot uses them); **O1**, the
last `exit-parking_position`; **O2**, the first `entry-taxiway`; **O3**, the
last `exit-apron`; **R**, the last `entry-runway`. A departure is
*OPDI-covered* if matched and O0, O1 or O2 lies at least 120 s before
`MVT_TIME_UTC_mvt`.

**Its inbound.** The latest OPDI flight with the same `icao24` whose `ades` is
the airport and whose `last_seen` lies before the matched flight's
`first_seen`, by at most 12 hours; linked to the challenge arrival at the
airport whose callsign (spaces removed) equals that flight's and whose
`MVT_TIME_UTC_mvt` lies within 30 minutes of its `last_seen`, the nearest if
several.

## 3. The reading, without a gate

OPDI is cheap to fetch for every day, so it has no gate of its own. Its
reading is descriptive and goes into the paper: per airport, over January and
July 2025, on the rows neither rule covers with a recorded taxi-out of 4 h or
less, the matched, covered and inbound-linked shares and, for O0, O1, O2 and
O3, the median and interquartile range of the estimate minus the recorded block
time and the shares within 60, 120 and 300 s; and the matched, covered and
inbound-linked shares for January and July 2026.

## 4. The fusion model

OPDI's features enter the fusion model of §5 whatever adsb.lol's gate says,
and the fusion model is built and judged by §5's rule either way. The features
added:

- take-off minus each of O0, O1, O2 and O3, missing where absent;
  `MVT_TIME_UTC_mvt` minus R; the OPDI match and coverage flags; the
  `opdi_time_in_use_s` of O1's parking position;
- over the other OPDI-covered departures at the airport whose take-off lies
  within 30 minutes of the subject's, the median of take-off minus the first
  present of O1, O0 and O2, minus their out-of-fold prediction, and their
  count; and the same two over those on the subject's runway;
- from the linked inbound: the subject's take-off minus the inbound's recorded
  block time, the inbound's block time minus its scheduled time, and whether
  its stand is the subject's.

The fitting days: §5's covered days with both sources' features if adsb.lol's
gate holds; otherwise every day of 2025, with OPDI's features alone. The model,
its parameters, target, rows and base prediction, its rule, its deadline and
v10 are as §5 states them.

## What differs from the note

This copy is the working note of 4 October 2026 with three places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Two of the places lie outside any rule. One lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `3b22e7b49038d4d1216b59af7f9027f1521796a966236165278e034b0bea95c5`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, line 4 | named the file of Hunter's statements by its working file name | "(his statements, items 24 and 25)" |
| the opening paragraph, line 11 | the same file, by the same name | "recorded in his statements, a working note this repository does not publish." |
| §2, Flights, events and matching, line 54 (inside the rule) | named the pilot's work block | "(OurAirports, as the pilot uses them)" |

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
