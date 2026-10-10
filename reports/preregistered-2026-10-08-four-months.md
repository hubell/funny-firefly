# Pre-registration of 8 October 2026: four ranking months

Written on 8 October 2026 at about 13:00 UTC, on Hunter's words "so i just
got an email from the organizers - deadline ia now 10amZ 10/14. They also
alsked for 4 months of taxi-out, (Jan/July 2026) plus Feb/June 2026 now", and
fixed by its sha256 before any figure it governs exists. The organizers'
email is recorded verbatim in the statements when he supplies it; nothing
here says how the four months are scored, and if the email says something
this note contradicts, an amendment is written before any figure it governs.

It amends one thing already fixed, §3 of the note of 8 October on the fusion
trained on every day (`preregistration_2026-10-08_all_days.md`, C17), and
defines v12 again, for four months. It reopens nothing decided: v11 is group
B's base with G1n refitted on it, group C adopted nothing, and C17's rule
stands as written.

## What exists as this is written

- **The ranking files on the drive** are those of 4 September (sha256
  `9db75e3a...` and `0d383408...`): 344,841 departures, January and July 2026,
  all ten airports in both months.
- **Inputs for 2026** exist for January and July only: METAR
  (`data/raw/metar/*_2026-01-01_2026-02-01.csv` and `*_2026-07-01_*`), OPDI
  flight events from 31 December 2025 to 9 February 2026 and from 29 June to
  8 August 2026, OPDI flight lists for 202601, 202606 and 202607, and the
  adsb.lol tables of the 62 days of January and July 2026.
- **The code names the two months** in several places (among them
  `fusion.RANKING_MONTHS`, `exposure.MONTH_PAIRS`, `opdi.MATCH_MONTHS` and
  `LIST_MONTHS`, `fetch_metar.PERIODS`). `windows.coverage_start` already
  finds each row's covered block from the months its own frame contains.
- **Held-out 2025 readings** exist for every month, February and June
  included (the overnight run's `heldout-2025-MM__c14__seed0` and the fusion
  fits of the fusion's refit on group B's base), and `reports/monthly_seeds.md` gives each month's floor: January
  0.33 s, February 1.56, June 0.21, July 0.58.
- **C17** is running (the measurement of C17): 61 of its 182 tables written, no feature,
  fit or reading of it exists.

## 1. C17, amended

Its days, matching, model, reference, rule and verdict time are unchanged.
Its §3 is replaced: if C17 is adopted, its model fitted on every fitting row
of its days and its twelve-month gate are fitted and saved, and are the
fusion v12 uses on all four months. No candidate is built on the template
of January and July.

## 2. v12, on four months

- **Definition**: v11's configuration, unchanged in every model, weight,
  gate, rule and seed, applied to the ranking set the organizers publish for
  the four months; with C17's fusion in place of that of the fusion's refit on group B's base if C17 is
  adopted. Nothing in v12 is chosen on a figure read on the new rows.
- **The only code changes** are those that let the ranking months be the
  months the ranking file contains, rather than January and July by name:
  January and February 2026 are one covered block and June and July 2026
  another, as `windows.coverage_start` already reads them. A constant that a
  pre-registration fixed as a rule (rather than as a mechanism) is not
  changed; if one names a month, the block stops and reports it.
- **Inputs extended** by the committed fetchers, unchanged except in the
  periods they are given: METAR for February and June 2026; the OPDI files
  those months need, edges included; adsb.lol tables for every day of
  February and June 2026 with a `prod` release. A day without a table has
  its rows uncovered, and G1n's coverage gate gives them the base, as it
  does now. Every missing day is named.
- **Checks, each a stop**: on the files of 4 September, the changed code
  gives training and ranking feature tables equal in every column to the
  committed code's. On the four-month files, every January and July row's
  prediction equals v11's, except rows whose features change because the
  two months now have neighbors; each such row is listed with its column
  and its distance from the block's edge, and any change that is not
  explained that way stops the build.
- **Landing and upload**: landed and built at a clean tree as v10 and v11
  were, and uploaded once on Hunter's word by **10 October 2026, 21:00
  UTC**. If C17's verdict or final model is not ready in time, v12 uses
  the fusion of the fusion's refit on group B's base and C17 waits for v13.

## 3. Candidates by season, the rule fixed now

Season candidates are named, with their definitions, in a note fixed by
sha256 by **9 October 2026, 18:00 UTC**, before any figure of theirs exists.
At most three per season. A season is winter (January and February) or
summer (June and July); a season candidate changes only its season's
ranking rows. It is adopted only if, against v12's configuration on the same
held-out 2025 rows (rows neither rule covers, every taxi-out):

- over the season's two months pooled, it is better by more than the
  season's floor: the square root of two times the standard deviation of the
  five single-seed readings pooled over those two months, from the seed
  files `reports/monthly_seeds.md` reads, computed and printed before any
  candidate figure;
- in neither month is it worse by more than that month's floor;
- on the forward leg of its season, it is not worse by more than the leg's
  floor: winter on the December leg (0.23 s), summer on the July leg
  (0.54 s).

Read with `candidates.holds` at its precision; the day-clustered intervals,
the per-airport readings and `OTHER_READINGS` beside, deciding nothing. If a
candidate clears, v13 is v12 with it, landed and uploaded once on Hunter's
word by **13 October 2026, 10:00 UTC**; otherwise v12 stands.

## 4. What does not change

§6 of the board note: one upload per frozen version, nothing chosen on a
score, only `usedPairs` and the score read back. The licenses: adsb.lol's
tables are a Derivative Database under ODbL and go into the deposit with the
notice; no OPDI file and no table derived from one enters the repository or
the deposit; challenge data never leaves the machine. Nothing is added after
a number is seen.

## What differs from the note

This copy is the working note of 8 October 2026 with four places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Two of the places lie outside any rule. Two lie inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `8c37c25b6a10c779d946885265d24c9896a5e934b9b424e56217d6cc8b72c372`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| What exists as this is written, lines 32 and 33 | named the work blocks that fitted the base and refitted the fusion | "(the overnight run's `heldout-2025-MM__c14__seed0` and the fusion fits of the fusion's refit on group B's base)" |
| What exists as this is written, line 35 | named the work block that measured C17 | "(the measurement of C17)" |
| §2, Definition, line 50 (inside the rule) | named the work block that refitted the fusion model | "in place of that of the fusion's refit on group B's base if C17" |
| §2, Landing and upload, lines 73 and 74 (inside the rule) | named the work block that refitted the fusion model | "v12 uses the fusion of the fusion's refit on group B's base and" |

The names of other pre-registrations' working files are left as the note
words them, because they are neither a numbered work block nor a dated
working note's file name; each one named has a published copy in
`reports/`, made the same way, whose closing section names that note's
digest.

The credit paragraph below is the note's own last paragraph, unchanged,
placed after this section so that it stays the last paragraph.

`tests/test_preregistered_copies.py` applies these substitutions to the
note wherever the note is on hand, after checking its digest, and requires
the result to be this copy up to this section, and its last paragraph,
byte for byte.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. Contains data from adsb.lol,
available under the Open Database License (ODbL) 1.0. Weather data from the
Iowa Environmental Mesonet of Iowa State University. This document uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
