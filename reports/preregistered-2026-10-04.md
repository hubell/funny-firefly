# Pre-registration of 4 October 2026: the tail rules off, the candidates after the board, and open ADS-B

Written on 4 October 2026 and fixed by its sha256 before any figure it governs
exists. It governs four things: how every file uploaded from now treats the
rows the record-quality term and the plan-shift rule covered; the file uploaded
next (v8); five candidates and the rule that adopts them; and a measurement of
the off-block from open ADS-B data, from a pilot gate to an adoption rule.
Where an earlier pre-registration speaks to the same question, this one
supersedes it only where it says so.

## What exists as this is written

- **The board**, read at about 08:40Z on 4 October: 141 teams, best submission
  per team, first 213.9176 s, third 218.9864 s, last 324.3448 s; 26 teams have
  three submissions or fewer.
- **v7, uploaded once** at 09:04:18Z and scored 313.169 over 344,841 pairs
  (processed 09:04:56Z), recorded in `reports/submission-2026-10-04.md`.
- **The tail analysis**: the 384 ranking rows the two rules predict, priced
  under two readings of their truth and two scorer arithmetics. Its policy
  table was computed at 08:58Z, before the upload, by
  `workingdir/directions_2026-10-04/tail_exposure.py`.

The decision below was taken after v7's score existed. Nothing in this document
is chosen on that score: the analysis that recommends the decision was computed
before it, and no rule below reads it.

## 1. The tail rows: both rules off in every file uploaded from now

Every file uploaded from now predicts every ranking row with the booster. The
record-quality term (`taxiout.tails.TailModel`) and the rule of the plan-shifted
departures (`taxiout.tails.PlanShiftModel`) are still fitted, still exclude
their training rows from the physical fit exactly as now, and are still
reported, but their predictions are not substituted. This supersedes the
substitution steps of `reports/preregistered-2026-10-02.md` and of the
submission described in the paper's §3.1, steps 6 and 7.

The reason, priced in the analysis (added mean squared error in s², then the
score on a stated base of 230 s; truth on these rows taken as Fiumicino's rest
mean, 1,144.2 s, under reading A, and as the 2025 two-point mixture under
reading B):

| policy | A, float | B, float | A, int32 | B, int32 |
| --- | ---: | ---: | ---: | ---: |
| keep both rules | 218,351 / 520.8 | 25,129 / 279.3 | 43,982 / 311.3 | 20,147 / 270.3 |
| drop both rules | 0 / 230.0 | 243,448 / 544.4 | 0 / 230.0 | 24,107 / 277.5 |

Reading B scored in floating point needs every team's best submission to
predict those rows near their lag; with 141 teams at or below 324.3 s, 26 of
them with three submissions or fewer, it is the reading the board makes
implausible. Over the other three, dropping costs at most 7.2 s and keeping up
to 290.8 s. The validation legs cannot inform this: their truth carries the
2025 artifacts.

## 2. v8

The submission rung `+ plan differences` under `BOOSTER_PARAMS`, five seeds,
the fitting rows as now, both rules fitted and not substituted. v8 must equal
v7 on every one of the 344,457 rows the rules did not touch, and may differ
only on the 384 they did. It is uploaded once, on Hunter's word.

## 3. Five candidates

Each is measured on top of the submission rung and adopted only under §4.

**C1, `AOBT_3_flt` and the queue it makes countable.** Features, computed the
same way on both sides, missing where `AOBT_3_flt` is missing:

- `aobt3_lag` = `MVT_TIME_UTC_mvt` minus `AOBT_3_flt`, in seconds;
- `aobt3_vs_lobt` = `AOBT_3_flt` minus `LOBT_flt`; `aobt3_vs_eobt` =
  `AOBT_3_flt` minus `EOBT_1_flt`; `aobt3_missing`, 0 or 1;
- at the subject's airport, over the other departures j with `AOBT_3_flt`
  present (a_j its `AOBT_3_flt`, t_j its take-off; a and t the subject's):
  `q_surface_off` = count of j with a_j <= a < t_j, all runways, and
  `q_surface_off_rwy` on the subject's runway; `q_takeoffs_during` = count of
  j with a <= t_j < t, all runways, and `q_takeoffs_during_rwy`;
  `q_overtakes_rwy` = count of j on the subject's runway with a_j > a and
  t_j < t.

`AOBT_3_flt` stays excluded everywhere else. Admission, if the rule adopts C1,
also needs Hunter's originality decision in writing.

**C2a, a winter layer on the prediction.** From the METAR already in the
deposit, each station-hour takes one class, carried forward up to three hours
as the regime flags are, in this order of priority: `fz_precip` (FZRA, FZDZ or
PL in the present weather); `snow_persistent` (SN, SG or GS in the hour and in
at least four of the previous twelve hours); `snow` (SN, SG or GS); `fz_fog`
(FZFG); `cold_wet` (RA or DZ at or below +3 °C); `frost_risk` (at or below
+1 °C, dew-point spread at most 2 °C, no precipitation code); `other`. For a row
of airport a in class c other than `other`, the layer adds
Z × m(a, c) + (1 − Z) × m(c), with Z = n(a, c) / (n(a, c) + 500), where m is the
mean residual (recorded minus predicted) of the submission rung's
out-of-fold predictions over the fitting months' rows that neither rule covers
and whose recorded taxi-out is 4 h or less, and n counts them. Rows in `other`
are unchanged.

**C2b, the winter block, only if C2a is adopted.** The seven classes one-hot,
the snow hours in the previous 6, 12 and 24 hours, the minimum temperature over
the previous 12 hours, and ERA5 hourly snowfall and snow depth at the airport's
coordinates from the Open-Meteo historical archive (CC BY 4.0, attribution
"Weather data by Open-Meteo.com"), added to the rung's features.

**C3, two blend members at equal weights.** (a) LightGBM under a deviance
matched to the variance: the slope λ of log squared residual on log prediction,
fitted over the submission rung's out-of-fold rows that neither rule covers,
of 4 h or less, with a prediction above zero (the modified Park test), sets the objective: `gamma` if
λ >= 1.75, else `tweedie` with variance power λ clipped to [1.1, 1.9] and
rounded to one decimal; the target is floored at 1 s for this member; every
other parameter is `BOOSTER_PARAMS`. (b) CatBoost on CPU at eight threads:
loss RMSE, 3,000 iterations, learning rate 0.08, depth 8, L2 3, 254 borders,
the rung's categorical columns declared categorical, the ladder's seed. If one
fit on the December leg takes more than 45 minutes, (b) runs on GPU only if two
GPU fits of that leg give bit-identical predictions, and otherwise stops, and
the report says why. Each member is read as the half-and-half blend with the
submission rung at the same seed.

**C4, the flight's own history.** `fn_mean`: the shrunk out-of-fold mean
recorded taxi-out per airport and `FLIGHT_mvt`, over the fitting rows, in the
fold scheme and shrinkage form `taxiout.encoding` uses for the pair, the prior
being the airport's mean over the fitting rows; `fn_n`, the rows behind it.
A row with no `FLIGHT_mvt` takes the prior.

**C5, per-airport recalibration.** Recorded = α(a) + β(a) × predicted, by least
squares per airport over the same out-of-fold rows as C2a, applied to every
row of that airport.

## 4. The rule

The rule of `reports/preregistered-2026-10-03.md`, unchanged in its legs, its
reference, its rows and its floors: a candidate is adopted only if, on the
paired contrast against the submission rung's own fit at the ladder's seed,
read at the precision `reports/baseline.md` prints, it is better by more than
0.23 s on December over all ten airports, better by more than 0.54 s on July
over all ten, not worse by more than 1.02 s on the three July airports, not
worse by more than 0.21 s on October, and not worse than the reference pooled
over the twelve held-out months by more than 0.16 s; a candidate that holds all
five is refitted at the second seed on December and July, and both contrasts
must keep their sign.

For the two layers (C2a, C5), the twelve months are read leave one month out:
each month's layer is fitted on the other eleven months' out-of-fold rows and
applied to that month's out-of-fold predictions. On a forward leg the layer is
fitted on the held-out months strictly before the leg's month (January to
November for December, January to June for July, January to September for
October) and applied to the leg's stored predictions. The layers need no fit
of their own at the second seed: their sign check applies them to the
reference's second-seed predictions on December and July, fitted if not
stored.

Everything that clears is combined (features together, then the layers, then
the blend, in which the booster and every adopted member take equal weights)
and the combination is held to the same five conditions;
if it fails, the single clearing candidate with the largest December gain is
adopted alone. Nothing is added after a number is seen. The order of fitting,
by cost: C2a and C5, then C1, C3a, C4, C2b if C2a clears, C3b. The verdict is
written by 7 October 2026, 12:00 UTC, or nothing is adopted. **v9** is the
adopted configuration, five seeds, both rules off, uploaded once.

## 5. The off-block measured from open ADS-B

**Source.** The adsb.lol globe history (`adsblol/globe_history_2025` and
`_2026` on GitHub, one release per day, the `prod` variant), available under
the Open Database License 1.0. The extracted event tables are a Derivative
Database: they go into the deposit under ODbL with the notice "Contains data
from adsb.lol, available under the Open Database License (ODbL) 1.0", and every
file built from them carries it. The challenge data never leaves the machine;
extraction may run elsewhere because it reads only the open data.

**Events, per aircraft leg ending in a take-off within 8 km of an airport's
reference point** (OurAirports). A leg starts at the aircraft's first position
within that circle after its previous airborne position there, or at its first
position of the day's file: the take-off time (the last ground position
before the first airborne position, with the airborne positions climbing);
E1, the first ground position of the leg; E2, the first ground position with
ground speed of 5 kt or more; E3, the end of the last run of ground positions
below 1 kt lasting at least 120 s that ends before E2; the count of ground
positions before take-off, the longest gap between them, and the total time
stationary (below 1 kt, runs of at least 60 s) between E2 and take-off.

**Matching.** A departure is matched to the leg at its airport whose take-off
lies within 120 s of `MVT_TIME_UTC_mvt`, the nearest if several, ties broken by
the callsign (`CALLSIGN_flt` or `FLIGHT_mvt` against the leg's flight field,
spaces removed). A departure is covered if matched and E1 lies at least 120 s
before take-off. A departure taking off before 01:00 UTC is read only where the
previous day's file is processed too; in the pilot it is left out of every
share.

**The pilot.** Days 2025-01-15 and 2025-07-16 (training, truth available) and
2026-01-14 and 2026-07-15 (ranking, coverage only). Read per airport on the two
training days, over rows neither rule covers with a recorded taxi-out of 4 h
or less: the matched and covered shares; for E1, E2 and E3, the median and
interquartile range of the estimate minus the recorded block time, and the
shares within 60, 120 and 300 s. **The gate**: the track continues only if at
least three airports have a covered share of 30% or more and, over those
airports' covered rows pooled, the best of E1 to E3 lies within 120 s of the
recorded block time on at least 50% of rows. Otherwise it stops here and the
paper records the measurement. Gate verdict by 5 October 2026, 12:00 UTC.

**The scale-up**, if the gate holds: every second day of 2025 from 1 January
(183 days) and every day of January and July 2026 (62 days).

**The fusion model.** LightGBM, regression, 1,000 trees, learning rate 0.05, 63
leaves, minimum child 50, column sample 0.9, row sample 0.8, deterministic,
column-wise, eight threads, seed 20260902; fitted on the covered days' 2025
rows that neither rule covers with a recorded taxi-out of 4 h or less; target
the recorded taxi-out; features: the out-of-fold prediction of the booster that
v9 ships (its twelve held-out-month fits at the ladder's seed, or the
submission rung's five-seed average if §4 adopts nothing), the airport,
take-off minus each of E1, E2 and E3, the coverage and match flags, the leg's
ground-position count, longest gap and stationary time, and three peer features
over the other covered departures at the airport within 30 minutes of the
subject's take-off: the median of take-off minus E1 minus their out-of-fold
prediction, the same with E3 where E3 exists, and their count. A row without a
matched leg enters with those features missing.

**Its rule.** Read leave one month out over the covered 2025 days: for each
month, fitted on the other months' covered days and scored on that month's
covered days, every row neither rule covers, against the out-of-fold prediction
alone on the same rows. Adopted only if it is better pooled over the twelve
months by more than 0.16 s, is not worse in any month by more than that month's
floor (`reports/monthly_seeds.md`), and, fitted on January to November,
improves the same booster's December forward-leg predictions on December's
covered days by more than 0.23 s. Verdict by 8 October 2026, 18:00 UTC. **v10**
is v9's booster with the fusion applied, both rules off, uploaded once.

## 6. Uploads

Every uploaded file is a model the paper defends, uploaded once. The board
ranks a team by its best submission, so no score chooses among v8, v9 and v10:
each is uploaded when its rule has spoken. Only `usedPairs` and the score are
read back. No file is built or uploaded to learn anything from the score. The
last upload is due by 10 October 2026, 21:00 UTC.

Movement and flight data supplied by EUROCONTROL PRC and the OpenSky Network
under the terms of the PRC Data Challenge 2026. This document uses no
OpenStreetMap data; the repository's distance matrices elsewhere are derived
from it (data (c) OpenStreetMap contributors, ODbL 1.0).
