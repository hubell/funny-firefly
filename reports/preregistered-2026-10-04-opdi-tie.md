# Amendment of 4 October 2026 to the OPDI amendment: events whose flight id is in no list

Written on 4 October 2026 after the OPDI fetch of 4 October stopped at the end of its Step 3
(a working report this repository does not publish), and fixed by its sha256 before any OPDI
flight is matched to a departure or any recorded block time is read for the
OPDI reading. It amends §2 of `preregistration_2026-10-04_opdi.md` (sha256
`3b22e7b49038d4d1216b59af7f9027f1521796a966236165278e034b0bea95c5`) in the one
place below and nowhere else. It is in force from Hunter's word, "recover by a
new rule, get everything that you can in" (4 October 2026), to be recorded in
his statements, a working note this repository does not publish.

## What exists as this is written

The OPDI files fetched and reduced, with no challenge data read beyond the two
rules' row counts (the OPDI fetch of 4 October). §2 reads "an event's `flight_id` is its
flight's `id` in the flight list". That holds on every day but one stretch:
none of the 247,324 flights first crossing FL50 in
`flight_events_20251201_20251211.parquet` has its id in any list, so the list
departures of 1 to 10 December 2025 (52,777 at the ten airports) carry no
events. Their surface events carry the aircraft's `osn_icao24` and callsign,
and these equal those of list flights of the same day. The other gaps (1 to 5
March, 1 to 4 May, 19 and 20 August and 2 to 11 October 2025; 2 to 10 and 23
January 2026) have no events and no list rows at all, and nothing here
recovers them.

## 1. The tie

An *event flight* is one `flight_id` of the flight-events files. One whose
`flight_id` is the `id` of no flight in the fetched lists (every row of
`flight_list_YYYYMM` for December 2024 to December 2025, January 2026, June
and July 2026) is tied to a list flight, and its events are then read as that
flight's under §2 unchanged:

- its *keys* are the `osn_icao24` and the callsign (`osn_flight_id`, spaces
  removed) carried by its surface events (exit-parking_position, entry-taxiway,
  exit-apron, entry-runway) at the ten airports: the most frequent pair, or
  the pair of its earliest such event if two are equally frequent;
- its *anchor* is its first crossing of FL50, or without one its earliest
  such surface event;
- it is tied to the list flight with `adep` or `ades` among the ten, the same
  `icao24` and the same callsign (spaces removed), whose `first_seen` is
  nearest its anchor, if that lies within 3 hours. If two list flights are
  equally near, or if two event flights tie to the same list flight, nothing
  is tied there.

An event flight with no surface event at the ten is not tied.

## 2. How the tie was checked

On OPDI alone, before this was written: applied blind to the 2,023,000 event
flights whose id does join a ten-airport list flight, the tie picks that
flight for 1,984,153 (98.08%), another flight for 1,751 (0.087%), and none for
37,096. Of the 48,032 event flights of 1 to 11 December 2025 with surface
events at the ten and no list id, it ties 47,118; 28 list flights are tied by
two event flights and so take none. The 3 hours are the window tested, chosen
from the tie's own test, not from any departure or block time.

## 3. What it adds to the reading

Nothing to §3's months: January and July 2025 and 2026 have no event flight
without a list id. It adds 1 to 10 December 2025 to the fusion model's
fitting days. A departure whose OPDI flight was tied is flagged
(`opdi_tied`) in the matched tables; the fusion model's features stay as the
OPDI amendment's §4 lists them.

## What differs from the note

This copy is the working note of 4 October 2026 with three places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
None of the places lies inside a rule. No rule, threshold, definition or
stop differs by a word.
The working note this is copied from was digested when it was written,
sha256 `e3c8f7216722fed756371a4dd8f5d8b8058e86a6d580f1594bbf4155db43fcd3`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 3 and 4 | named the OPDI fetch's work block and its report by its working file name | "after the OPDI fetch of 4 October stopped at the end of its Step 3 (a working report this repository does not publish)" |
| the opening paragraph, lines 9 and 10 | named the file of Hunter's statements by its working file name | "to be recorded in his statements, a working note this repository does not publish." |
| What exists as this is written, line 15 | named the OPDI fetch's work block | "(the OPDI fetch of 4 October)" |

The names of other pre-registrations' working files are left as the note
words them, because they are neither a numbered work block nor a dated
working note's file name. The board pre-registration's published copy is
`reports/preregistered-2026-10-04.md`, the same bytes under the same
digest; each amendment named has a published copy beside this one in
`reports/`, made the same way, whose closing section names that note's
digest. Working
directories that the patterns do not flag are also left as the note words
them, and this repository does not carry them.

The note's credit paragraph names no OpenStreetMap data because the note
uses none. The repository's distance matrices elsewhere are derived from
it (data (c) OpenStreetMap contributors, ODbL 1.0), and that credit is
carried here, in this section, so that every report says it once.

The credit paragraph below is the note's own last paragraph, unchanged,
placed after this section so that it stays the last paragraph.

`tests/test_preregistered_copies.py` applies these substitutions to the
note wherever the note is on hand, after checking its digest, and requires
the result to be this copy up to this section, and its last paragraph,
byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This document uses no
OpenStreetMap data.
