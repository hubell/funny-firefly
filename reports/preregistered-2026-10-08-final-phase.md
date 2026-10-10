# Amendment of 8 October 2026: the final phase

Written on 8 October 2026 at about 14:00 UTC, after the organizers' email
(recorded verbatim as item 54 of the statements) and on Hunter's answers to
three questions, and fixed by its sha256 before any figure it governs exists.
It amends `preregistration_2026-10-08_four_months.md` (sha256
`8c37c25b6a10c779d946885265d24c9896a5e934b9b424e56217d6cc8b72c372`). No
figure of C17, of v12 or of any season candidate exists as it is written.

## What the organizers wrote

Quoted from the email, verbatim:

- "we introduced a final phase with a single final submission of your
  predictions."
- "The final phase will ask to predict 4 months of taxi-out, the two initial
  months (Jan and July 2026) plus two additional one (Feb and Jun 2026)."
- "We will be able to rank per Jan/Jul, Feb/Jun and the four months combined
  Jan/Feb/Jun/Jul which together with the review of the code and
  documentation would provide (we hope) enough evidence to select the best
  teams."
- "we have extended the deadline to noon 14th Oct CET, or 2026-10-14T10:00Z"

## 1. Season candidates withdrawn

§3 of the four-month note is withdrawn, on Hunter's answer "Drop them". No
season candidate is named, fitted or read, and there is no v13. The
organizers rank Jan/Jul and Feb/Jun, not seasons, and state that the final
phase exists to measure how far a model generalizes; a candidate that
changes some months' rows and not others is the kind of tuning to two
specific months the email names. Any further candidate would need its own
note, fixed before any figure, applied to every ranking row; none is planned.

## 2. v12 is the final submission

- **Unchanged from the four-month note**: v12's definition (v11's
  configuration on the four-month set, with C17's fusion if C17 is adopted
  and its final model is ready, otherwise that of the fusion's refit on group B's base), the code changes it
  allows, the inputs, and the checks against v11.
- **The one upload**: on Hunter's word, on **13 October 2026** (his answer
  "13 Oct"), to the destination and in the form the organizers specify for
  the final phase, read from their own pages or bucket and quoted verbatim in
  the landing's report. It replaces §2's upload by 10 October. v12 is landed
  and frozen before then, as early as its checks allow.
- **Before the upload, because there is no second one**: the file has
  exactly the final-phase template's identifiers, once each, in its order if
  it has one; every prediction is finite and positive, and any prediction
  outside the range of v11's predictions is listed with the rule that set
  it, one set by no rule being a stop; every January and July row equals v11's
  except the rows the four-month note's check explains; it is built twice,
  byte-identical, at a clean tree. Any failure stops the upload.
- **Readings for the paper, deciding nothing**: v12's configuration on the
  held-out 2025 rows, reported the way the organizers rank, per Jan/Jul, per
  Feb/Jun and the four months combined, with the day-clustered intervals.

## 3. What does not change

C17's rule, days and verdict time; §1 of the four-month note; §4 of the
four-month note. Nothing is chosen on a score; if the final phase shows
one, only `usedPairs` and the score are read back.

## What differs from the note

This copy is the working note of 8 October 2026 with one place changed and
nothing else. The note named numbered work blocks of the working program,
their reports and working notes by their working names, which this
repository does not publish.
The place lies inside a rule or a definition, where the note named the
work block that measured or built what the rule reads; there Hunter
decided on 7 October 2026 that the copy names that work in words instead,
and the rule refers to the same thing as before. No other word of any
rule, and no threshold, definition or stop, differs.
The working note this is copied from was digested when it was written,
sha256 `553e042f94494798c2c7fd1c92823713069514eebdc90ba61a161d9bd12f41fa`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in that place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| §2, Unchanged from the four-month note, line 38 (inside the rule) | named the work block that refitted the fusion model | "otherwise that of the fusion's refit on group B's base)" |

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
available under the Open Database License (ODbL) 1.0. This document uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
