# Amendment of 9 October 2026: the January and July check, and the final file

Written on 9 October 2026 at about 10:00 UTC, after the report of the measurement of the four months' inputs
(a working report this repository does not publish, sha256 `9b3eaa52...`), and fixed by its
sha256 before any v12 prediction exists. It amends the check of §2 of
`preregistration_2026-10-08_four_months.md` (sha256 `8c37c25b...c372`) and
the pre-upload check of §2 of `preregistration_2026-10-08_final_phase.md`
(sha256 `553e042f...41fa`). No model, weight, gate or rule changes.

## What the measurement of the four months' inputs found

- **The final files** (the bucket, dated 8 October 11:04Z):
  `final_ranking.parquet` (sha256 `4b0692a7...c69f`, 1,341,197 rows) and
  `final_submitting.parquet` (`b3445d80...3924`, 670,790 template rows:
  January 152,719, February 144,158, June 181,791, July 192,122). The
  template begins with the old template's 344,841 rows in their order. The
  twelve training files are unchanged.
- **January and July were redrawn.** The same rows as 4 September's, but
  `MVT_TIME_UTC_mvt` moved on 91.8% of departures and 92.0% of arrivals,
  and arrivals' `BLOCK_TIME_UTC_mvt` likewise, each within ±12 s, as on
  4 September. So 344,840 of 344,841 January and July ranking rows have
  different features from v11's, and the four-month note's check ("every
  January and July row's prediction equals v11's, except rows whose
  features change because the two months now have neighbors") cannot hold
  for a reason that has nothing to do with the code.
- **The neighbors alone**, read against a control (the patched code on the
  final file's January and July rows only): 3,617 January and July rows
  change (January 172, July 3,445), and every one is explained by a
  February or June movement.
- **The ranking page**, verbatim: "Please submit you unique final submission
  using the following filename: `<team-name>_final.parquet`". It names no
  template or destination for the final file; the repository's filename
  check refuses that name.

## 1. The check, replaced

The landing builds v12's models once and predicts two frames with them: the
final file's four months, and the control (its January and July rows
alone). Three checks, each a stop where it says so:

- **(i) Neighbors.** Every January and July row whose features differ
  between the control and the four months is explained by a February or
  June movement, as the measurement of the four months' inputs explained the 3,617; recomputed at the
  landing. An unexplained row stops the build.
- **(ii) Predictions.** v12's January and July predictions equal the
  control's on every row (i) does not list. A difference stops the build.
- **(iii) Against v11, descriptive.** Per row, v12's January and July
  prediction less v11's (`cea5e4bc...51ee`): the distribution, the 50
  largest changes, and every change over 600 s with the columns that
  changed and any change in the rules' rows. The redrawn times, C17's
  fusion and (ii)'s rows are the expected causes; a row whose change no
  changed column explains stops the upload. The 600 s is a threshold for
  listing, not a stop in itself.

The final-phase note's pre-upload condition "every January and July row
equals v11's except the rows the four-month note's check explains" is
replaced by (i) to (iii).

## 2. The final file

`funny-firefly_final.parquet`, with exactly `final_submitting.parquet`'s
identifiers in its order and its columns, to the team's own bucket as every
earlier upload, unless the organizers say otherwise before the upload; what
they say is quoted verbatim in the landing's report. The filename check
learns the final name as a mechanism change with a test; the
`_v<n>` convention stays as it is.

## 3. What does not change

Everything else in the four-month note, its final-phase amendment, the C17
note and the C18 and C19 note. Nothing is chosen on a score.

## What differs from the note

This copy is the working note of 9 October 2026 with three places changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
Two of the places lie outside any rule. One lies inside a rule
or a definition, where the note named the work block that measured or
built what the rule reads; there Hunter decided on 7 October 2026 that the
copy names that work in words instead, and the rule refers to the same
thing as before. No other word of any rule, and no threshold, definition or
stop, differs.
The working note this is copied from was digested when it was written,
sha256 `05214673d1868f88f6215bee204ed43c17e733f527f16b18a56143b89ca0d85e`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in each place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 3 and 4 | named that work block and its report by its working file name | "after the report of the measurement of the four months' inputs (a working report this repository does not publish, sha256" |
| the heading of the first section, line 10 | named that work block | "## What the measurement of the four months' inputs found" |
| §1, (i) Neighbors, line 43 (inside the rule) | named that work block | "as the measurement of the four months' inputs explained the 3,617" |

The names of other pre-registrations' working files are left as the note
words them, because they are neither a numbered work block nor a dated
working note's file name; each one named has a published copy in
`reports/`, made the same way, whose closing section names that note's
digest.

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
under the terms of the PRC Data Challenge 2026. Contains data from adsb.lol,
available under the Open Database License (ODbL) 1.0.
