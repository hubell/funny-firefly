# Review: where the remaining error sits, and what could still move it

Written 8 September 2026, after Gate 2 closed at 233.4 s (December) and
257.2 s (July, the three airports) over the rows the record-quality term
leaves to the physical model. A review note, not a report: the figures
below come from a diagnostic script held outside the repository
(`headroom.py`, with its JSON output) until a decision is taken on what to
build. If any of it is adopted, the script becomes
`scripts/report_headroom.py` and the numbers move to `reports/`.

The method throughout: take the stored validation residuals of the
submission rung (`data/processed/residuals/<split>_encoding-for-stand.parquet`),
join them back to the raw month, build a candidate block, and fit a small
booster *on the residual* with 5-fold cross-validation grouped by day. The
RMSE reduction is the headroom that block could add to the submission rung,
read the same way as the queue ceiling in `reports/congestion.md`. Nothing
here refits the ladder, and nothing is a feature yet.

## 1. The residual is a tail of individual holds, not a queue

Over the ex-flagged rows of the submission rung:

| split | rows | RMSE | rows with \|resid\| > 600 s | their share of SSE | of those, under-predicted |
| --- | ---: | ---: | ---: | ---: | ---: |
| December, all ten | 165,511 | 233.3 | 2.2% | 43% | 60% |
| December, > 1,200 s | | | 0.32% | 21% | 79% |
| July, the three | 64,525 | 255.0 | 2.2% | 54% (EGLL alone 66%) | 58% |
| July, the three, > 1,200 s | | | 0.50% | 36% | 64% |

Three things about the tail. The big misses are under-predictions: actual
taxi-outs of 40 minutes to three hours that the model put at 10 to 40. They
are individual, not shared — the neighbour-residual oracle in
`reports/congestion.md` already showed that knowing every same-runway
neighbour's outcome exactly buys 9–10 s, so these are not a runway queue
everyone is in. And they are enriched in three identifiable classes:
heavies (32% of the top 1% of residuals against 22% of rows); departures
with no Network Manager record (1–2% of rows, 4% of December's SSE and
14% of the three July airports' — EGLL's no-record departures run at
1,216 s RMSE; a flight whose plan was cancelled and refiled after a long
delay loses its join, so this flag is itself a delay signal); and, the
new one, departures whose stand was re-occupied before they took off.

The error also grows with the plan lag, as it must: at a lag of 45–60 min
the December RMSE is 455 s with a bias of −56 s, at 1–2 h it is 769 s with
−164 s. Long lag is mostly gate delay, sometimes a hold, and the
conditional mean sits between the two modes. Nothing about the loss or the
learner fixes that; only information that separates the modes does.

## 2. What the unused exposed fields are worth

Headroom on the residual, in seconds of RMSE, ex-flagged. "The three" is
EDDF, EGLL, EHAM — 53% of the ranking rows.

| block | December, all ten | December, the three | July, the three |
| --- | ---: | ---: | ---: |
| stand occupancy, refined (§3) | +3.9 | +5.0 | **+10.3** (EGLL +21) |
| operator (`AIRCRAFT_OPERATOR_flt`, 486 levels) | +2.4 | | +2.9 |
| callsign prefix (same information, covers the no-record rows) | +2.1 | | +2.8 |
| destination bearing and distance (OurAirports) | +1.4 | | +1.6 |
| destination as a categorical (914 levels) | +1.5 | | +1.9 |
| market segment, flight type, flight rule | +0.6 | | |
| public holidays (`holidays`, MIT) | +0.5 | | +0.1 |
| **stand occupancy + operator + destination + holidays** | **+6.1** | **+5.8** | **+10.8** |
| everything measured, together | +6.9 | | |

Weighted at the ranking mix (January all ten at 47%, July the three at
53%) the recommended set is worth roughly 8 s on the score — about 3%.
That is the honest size of what is left in the exposed fields. The model
already carries what the literature says matters; these are the fields it
never read.

Worth nothing, so that nobody rebuilds them: landings on the subject's own
runway (mixed-mode load) +0.3; the wake class of the preceding departure
and heavies among the last three, +0.3; the minimum, 25th percentile and
spread of the neighbours' plan lags, and the all-runway mean, −0.3 / +0.7;
arrivals' lateness in trailing windows (the regime-exposure control as a
feature), −0.3 / −0.8; arrivals' *excess* taxi-in against a pair reference,
0.0 / −0.6; same-direction departures in the trailing ten minutes (a SID
proxy from destination bearings), +0.2 / +0.1; METAR wind, gust, head- and
crosswind component, −1.3 / −1.9; ceiling, temperature, dew-point spread,
fog, −0.2 / −1.2; thunderstorm and snow flags, +0.2 / +0.1. The weather
block as built is not missing anything the METAR carries.

`ARVT_3_flt − ARVT_1_flt` is worth +1.4 s in December and −0.6 in July and
is in the same class as `AOBT_3_flt` — a flown-trajectory field — so it
stays excluded by the same decision. It is measured so the exclusion costs
a known amount, which is nothing.

## 3. The stand as a shared resource

The ranking set exposes every arrival in full, including its stand and its
on-block time. A stand holds one aircraft. If an arrival on-blocks at the
subject's stand *before the subject takes off*, the subject had already
left the stand, so its taxi-out is at least the gap between that on-block
and its take-off. That is a physical bound built from exposed fields, and
it is the same operational quantity A-CDM calls stand availability — at
Heathrow in particular, aircraft are pushed to free the stand and then held.

Measured on the December validation rows, by minutes from the last arrival
on-block at the stand to the subject's take-off:

| gap | rows | actual ≥ gap | model < gap | residual bias |
| --- | ---: | ---: | ---: | ---: |
| 0–10 min | 9,887 | 100% | 0% | −20 s |
| 10–15 | 2,832 | 99% | 2% | +14 |
| 15–20 | 1,432 | 97% | 6% | +68 |
| 20–30 | 932 | 89% | 23% | **+185** |
| 30–45 | 638 | 29% | 84% | +119 |
| 45–60 | 7,461 | 0% | 100% | −37 |
| 60–120 | 68,789 | 0% | 100% | −17 |

Below 20 minutes the bound holds essentially always and the model already
respects it. The signal is the 20–45 minute band, 1% of rows, where an
arrival at the same stand could be either the subject's own inbound after a
fast turnaround or a different aircraft that took the stand after the
subject pushed — and where the model under-predicts by two to six minutes.
The previous arrival's aircraft type and operator disambiguate: in that
band the bias is +239 s when the type differs and +55 s when it matches
(+219 / +125 by operator). That interaction is what "refined" adds, and it
is where July's EGLL +21 s comes from.

Two limits, stated now. First, the bound is not a rule: above 30 minutes
it collapses even when the type and operator both differ (45–60 min,
different both: actual ≥ gap in 9% of cases), because stand identifiers
denote positions that can hold two aircraft, and because aircraft are towed.
Applying it as a hard floor loses 1–20 s; as a feature the booster learns
where it holds. Second, the block reads an arrival that on-blocked *after*
the subject's own off-block, so it is admissible under the challenge's
post-hoc framing and not under an operational one — the same status as
every take-off-anchored window the model already carries, but it does bound
the hidden field from inside the challenge data, and that is a reading
Hunter should make rather than the note.

Sources it needs: `ARR.STAND_mvt`, `ARR.BLOCK_TIME_UTC_mvt`,
`ARR.SCHED_TIME_UTC_mvt`, `ARR.TAXITIME_SEC_mvt` (all already in
`AVAILABLE_FOR_ARRIVALS`) plus `ARR.AIRCRAFT_TYPE_mvt` and
`ARR.AIRCRAFT_OPERATOR_flt`, null at 0.00–0.01% and 0.76% in the ranking
arrivals by the data review, which would be added to that list. Columns:
gap to the last and second-last arrival on-block at the stand, the schedule
against the last on-block, the last arrival's lateness and taxi-in, the gap
to the previous departure from the same stand, and the two match flags.

## 4. External data, with the rule read verbatim

The eligibility page states: "All used external datasets are openly
accessible/usable and documented" and "All **additional datasets** used are
openly available under an open source license."

| source | what it would carry | licence position | verdict |
| --- | --- | --- | --- |
| OurAirports `airports.csv` | destination coordinates → bearing, distance | public domain; already a dependency | **use** (+1.4 / +1.6 s) |
| `holidays` (python-holidays) | public holidays per airport country | MIT | **use** (+0.5 / +0.1 s; trivial) |
| PRU open data, ansperformance.eu/data: daily airport-arrival ATFM delay by cause, daily ATC and all-causes pre-departure delay by airport, daily ATFM slot adherence, daily en-route ATFM delay by FIR — all covering January and July 2026 | the "phased delays" the question asks about: which days each airport, and each en-route centre its departures cross, was regulated, and why | the portal's copyright notice: "It may be copied in whole or in part, provided that EUROCONTROL is mentioned as the source and it is not used for commercial purposes (i.e. for financial gain). The information in this document may not be modified without prior written permission from EUROCONTROL." Non-commercial and no-derivatives — the same notice that excluded OPDI in `prior_art.md` | **excluded by the same reading**, unless the organisers confirm in writing that their own portal data counts; a question worth asking on the challenge channel, since it is their data and their rule |
| ERA5 (Copernicus CDS) | snowfall, CAPE, lightning proxies, gusts | CC-BY since 2 July 2025, so it now clears the rule | **not pursued**: the nearest test, the METAR's own wind, ceiling, temperature and thunderstorm fields, bought nothing |
| NOTAM archives (notamhistory, Notamify, Aviation Edge) | runway and taxiway closures | commercial APIs; no open archive found | excluded |
| OpenSky historical surface tracks | route validation | research use | diagnostics only, as ruled |
| Wikidata aircraft properties (CC0) | MTOW, wingspan beyond the wake class | open | not needed: `aircraft_type` is already a categorical |

## 5. Other approaches, and what they are worth

- **Seed bagging / a second learner.** Five boosters averaged, or LightGBM
  plus CatBoost, typically buys 1–2% on a tabular RMSE. Cheap on the Mac,
  needs no new feature, no new source, and no rule question. Not measured
  here.
- **`EOBT − SCHED` and `LOBT − EOBT` as explicit columns.** The booster
  sees both lags but has to approximate the difference with splits. Free;
  expected ±1 s.
- **Winsorised fit.** Fit on the target clipped at two hours, score
  unclipped: trades a little tail bias for less variance from the unflagged
  multi-hour rows (22 rows carry 30% of July's pooled SSE, nearly all
  LIRF, which July 2026 does not contain). A cheap check, not a plan.
- **Rotation linkage by flight number.** Airlines often number the inbound
  and outbound legs consecutively; the same-operator-same-type match on the
  stand already captures most of this. Marginal over §3.
- **A queueing simulation, per-airport models, a hurdle model for holds.**
  All bounded by the same 9–10 s ceiling or by the conditional-mean
  argument in §1. Not recommended.
- **The January regime shift.** The exposed fields cannot separate gate
  delay from taxi; arrivals' lateness as a feature was tried above and
  bought nothing within a year. The bound in `reports/congestion.md` stands
  and nothing here changes it.

## 6. Decisions

1. **Stand occupancy** — build it, or rule it out as bounding the hidden
   field? If built: as a block on the ladder above the weather rung, with
   the two arrival columns added to `AVAILABLE_FOR_ARRIVALS`, the poison
   test extended, and the 20–45 minute mechanism written up as in §3.
2. **Operator** — the categorical (486 levels, anonymised) or the callsign
   prefix (covers the 1–2% without a record)? Both together are worth no
   more than either.
3. **OurAirports destinations and holidays** — two small, clean sources;
   include, or keep the external-data footprint at METAR alone for the
   paper's sake?
4. **The PRU portal datasets** — ask the organisers whether their own open
   data clears their rule, or leave it as the paper's "what a future
   edition should open"?
5. **Bagging and the two free columns** — run on the Mac once the feature
   decisions are taken, so the ladder is refitted once.

*Taken, 8 September:* stand occupancy is built as a ladder rung; the
operator enters as the callsign prefix; OurAirports destinations and the
holiday calendar are added; the PRU portal data is left for the paper as
what a future edition should open; bagging and the two difference columns
follow once the ladder is refitted.

None of this changes the paper's central finding. The geometry claim is
measured at the foot of the ladder and these blocks sit at its top;
whichever are adopted, the ex-flagged submission rung would move from about
233 / 257 s to about 227 / 246 s, and the tail that remains is individual
holds no exposed field predicts.

## 7. What the refit found, and the one thing this review got wrong

*Added 8 September, after the four blocks were built and the ladder refitted.
The measurements are in [`../reports/baseline.md`](../reports/baseline.md) and
[`../reports/occupancy.md`](../reports/occupancy.md); this section says only
how they stand against the estimates above.*

All four blocks were built and all four earned their place. Over the
ex-flagged rows, against the arm's 233.4 s in December and 343.7 s in July:
stand occupancy −8.3 and −7.5, the callsign prefix −1.8 and −4.6, the
destination and holidays −1.2 and +0.01, the plan differences −1.5 and −3.2.
Together **233.4 → 220.6 s and 343.7 → 328.3 s**, and 257.2 → 239.4 s on the
three airports July 2026 contains.

**The estimates above are a floor and not an estimate, and that is the
methodological finding.** §2 predicted about +6 s in December and +11 s on the
July three; the refit found 12.8 and 17.8, roughly twice. The reason is in how
the estimates were made. Fitting a small booster *on the stored residuals* of a
frozen model measures what a correction bolted on top of that model would
recover. A refit does something strictly larger: it restructures every split in
the model around the new column and can use it to condition the blocks already
there. So a residual fit bounds a refit from below, and the gap between them is
the value of the interaction rather than evidence of anything wrong. Each
block's own overshoot stays inside the five seconds the leakage-audit trigger
allows, and the registry, the poison tests and the band table in
`occupancy.md` are what stand behind that rather than the arithmetic.

The reading to carry forward: a headroom number produced this way is the answer
to "what would a correction on top of this model recover", which is a smaller
question than the one a gate usually asks. It is still the right instrument for
deciding what to *build* — it is cheap, and it ranks candidates correctly, which
is all §2 was used for. It is the wrong instrument for deciding what a block
*is worth*, and it should not be quoted as one.

Two of §5's other lines were also settled. Seed bagging was measured once and
**adopted**: five seeds averaged gain 1.10 s in December and 1.37 s in July,
both clearing the one second fixed as the threshold beforehand. §5 put it at
"typically 1–2% on a tabular RMSE"; half a percent is what it came to here. The
two free difference columns came in at −1.5 and −3.2 against the "±1 s"
predicted, which is the same lower-bound effect at a smaller scale.
