# Pre-registration of 9 October 2026: the flight-plan window, the shifted records, the audit

Written on 9 October 2026 at about 10:30 UTC, on Hunter's choices after the
research memo of 9 October (a working note this repository does not publish): "Flight-plan window", "Artifact-row tests", "Feb/Jun crash
audit"; on provenance, "Re-derive and disclose"; and fixed by its sha256
before any figure it governs exists. No count, histogram or fit of anything
below exists as it is written.

**Provenance.** Two of the ideas below were first seen by Hunter in the
memo's account of other teams' public write-ups. Each is tested here only as
a property grounded in documentation, measured with this entry's own code on
its own 2025 data, with thresholds fixed here: the flight-plan window from
the APDF specification's use of off-block times to associate records with
flight data, and the shifted records from the A-CDM manual's next-day
example and APDF's "Time in UTC or LT (Local Time) is sufficient". No code,
threshold or text from another entry is used; the 3,606 s figure the memo
reports is not used. The paper says that public discussion prompted these
tests.

## 1. W, the flight-plan window

- **Measure** on every 2025 departure with `LOBT_flt` present (all
  airports, all months): Δ = `BLOCK_TIME_UTC_mvt` − `LOBT_flt` in seconds.
  W* = max |Δ|. Print per airport, the counts beyond 3,600 and 3,900 s, the
  three plan-shift class rows' Δ, and the histogram of Δ in 30 s bins over
  [−4,200, +4,200].
- **Valid** only if both hold: W* ≤ 3,900 s; and the rows with |Δ| in
  (W* − 300, W*] number at least half of those in (W* − 600, W* − 300] (a
  wall, as a join condition would leave, rather than a tail decaying into the
  edge).
- **If valid**, W is applied to every ranking row with `LOBT_flt` present,
  after every other part of v12 (the rules included): the prediction is
  clipped to [L − W′, L + W′], L = `MVT_TIME_UTC_mvt` − `LOBT_flt` and
  W′ = W* + 12 s (the ranking set's take-off times are redrawn within
  ±12 s). The clip cannot move a prediction away from a truth inside the
  interval. It binds both ways: it raises a prediction far below the lag
  less W′ and caps any prediction, a day-scale hedge included, at the lag
  plus W′. It overrides the plan-shift rule's prediction where the two
  disagree, which amends the output of the 2 October rule; that amendment
  is made here, before any figure, and is disclosed.
- **Read beside, deciding nothing**: on the twelve held-out 2025 months, the
  clip applied to v11's out-of-fold predictions (no month's MSE may rise,
  since W* is the 2025 maximum; a rise is a bug and a stop); on v12's
  four-month predictions, the rows moved per airport and month and
  Σ d² / N per month pair (d the distance moved), a lower bound on the gain
  if the window holds in 2026.

## 2. The shifted records: tests T1 to T4, then C20

The population is the 2025 departures with `LOBT_flt` null (the orphans),
every airport; y the recorded taxi-out; m_a the airport's median y over its
2025 departures with y ≤ 4 h; zones from the IANA database (Python
`zoneinfo`, public domain) for each airport.

- **T1, the date.** Day-shifted (DS): y ≥ 82,800 s. Signature S1 in a
  calendar (local, or UTC): the date of `SCHED_TIME_UTC_mvt` differs from the
  date of `MVT_TIME_UTC_mvt` − m_a. **Survives** if, in one calendar, at
  least 80% of DS orphans carry S1, and among orphans with S1 in that
  calendar at least 5% are DS with at least 5 DS rows (`tails.MIN_SHARE`,
  `tails.MIN_COUNT`). If both calendars pass, the one with the higher DS
  share among S1 rows is used.
- **T2, the offset.** e = y − m_a for orphans and e = y − (`MVT` −
  `AOBT_3_flt`) for departures with `AOBT_3_flt`; o the airport's UTC
  offset on the row's date. **Survives** if, pooled over airports, the rows
  with e within ±300 s of +o number at least three times the mean of the two
  same-width windows centred at o − 1,800 and o + 1,800 s, and at least 20.
- **T3 and T4, diagnostics only**: the ten largest squared errors in the
  core of held-out January and July 2025 and the 2025 rows over 4 h outside
  the term's cells, each classed as k × 86,400, n × 3,600, at schedule, a
  later arrival at the same stand before the take-off, or none; and the DS
  share among orphans by `FLIGHT_RULE_mvt`, by `FLIGHT_ID_mvt` null, and by
  hours from the start of the UTC month. They decide nothing.
- **C20, only if T1 or T2 survives.** A multiclass LightGBM on 2025 orphans
  at every airport (`tails.C6_CLASSIFIER_PARAMS` with objective multiclass,
  300 rounds, seed unchanged): classes S (at schedule, `tails.at_schedule`),
  D (DS, if T1 survives), O (e within ±600 s of +o, if T2 survives), R the
  rest. Features: C6b's, the hours from the start of the take-off's UTC
  month (the monthly files are built month by month, and the final file now
  has two seams inside its blocks), plus S1 in the surviving calendar and
  the minutes from local midnight of the schedule and of the take-off (if T1
  survives).
  k = the date difference S1 measures (1 where S1 is false). Prediction,
  always the conditional mean, never a class:
  - in C6b's covered cells (replacing C6b): p_S·lag + p_D·(86,400·k +
    r) + p_O·(o + r) + p_R·r, r C6a's rest mean over R rows of the cell;
  - on every other orphan (replacing nothing, adding to the physical
    prediction f): f + p_D·86,400·k + p_O·o.
  Absent classes have p = 0.
- **C20's rule**, leave one month out over 2025, every row (no row
  excluded), against v11's configuration out of fold (group B's base, C6b,
  the plan-shift rule): adopted only if the twelve months' pooled MSE falls
  with a day-clustered 95% interval excluding zero; held-out February and
  held-out June 2025 are each not worse; the December and July legs are not
  worse; and over the twelve held-out months Σ p_D on rows with p_D ≥ 0.2
  is at most 1.5 times the DS rows among them (the same for O). Otherwise
  not adopted.

## 3. The audit of the final file (no fit)

Run on v12's four-month file before the freeze, and written into the
landing's report:

- (a) every prediction over 4 h and every row with a day-scale term
  (p_D ≥ 0.01, the plan-shift rule, or the term's lag) listed with its rule,
  its weights, its D and its cell's 2025 count;
- (b) a hedge row whose cell has fewer than 20 rows in 2025 is listed apart;
- (c) any day-scale weight of exactly 0 or 1 on a hedge row is listed apart;
- (d) rows with a flight-plan time on the last day of a month and a take-off
  on the first;
- (e) per airport and month, the count of predictions over 4 h in February
  and June 2026 against the same configuration's count on held-out February
  and June 2025 per thousand rows; a ratio above 2 is reviewed row by row
  with Hunter before the freeze.

A row the rules do not explain stops the upload. The audit changes no
prediction.

## 4. Order, time, and into v12

The tests run first; C20 is fitted only if a test survives. Verdict lines
`W: valid / not valid`, `T1: ...`, `T2: ...`, `C20: ADOPTED / not
adopted`, by **10 October 2026, 18:00 UTC**; what is not read by then is
not adopted. Anything adopted joins v12's definition as the four-month
note's §2 and the C18 and C19 note's §4 allow; v12 is frozen by 12 October
2026, 12:00 UTC, and uploaded once on Hunter's word on 13 October. Nothing
is chosen on a score.

## What differs from the note

This copy is the working note of 9 October 2026 with one place changed and
nothing else. The note named a research memo by its working path, which
names a tool this repository does not carry and which this repository
does not publish.
The place does not lie inside a rule. No rule, threshold, definition or
stop differs by a word.
The working note this is copied from was digested when it was written,
sha256 `908780852884eee96a4619986eb2fca47a8680ab4fd8462cf5029711c92895eb`,
before any figure it governs existed; that digest is the proof of the
order. What the note said in that place, and what this copy says:

| where | the note | this copy |
| --- | --- | --- |
| the opening paragraph, lines 4 and 5 | named the research memo by its working path, which names a tool this repository does not carry | "(a working note this repository does not publish)" |

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
under the terms of the PRC Data Challenge 2026. Time zones from the IANA
time zone database (public domain).
