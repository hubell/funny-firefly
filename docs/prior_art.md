# Prior art and external data

What has already been established about taxi-out prediction, what this entry
takes from it, and the license position of every external source considered.

Everything here is cited as reference. No code or text from any of these
sources is used in this repository; where a method is reimplemented, it is
reimplemented from the published description and the citation records where the
idea came from.

## What the literature says drives taxi-out

Ordered by the effect sizes the sources actually report, which is not the order
intuition suggests.

**Surface congestion dominates.** Idris et al. (2002) regressed taxi-out at
Boston against each candidate factor separately: takeoff queue size gave the
largest single R², the number of departures already on the surface at pushback
was next, and aircraft type and airline were close to nothing. Twenty-five
years of subsequent work has not overturned that ordering.

**Distance and turn angle are the second family, and they work on speed rather
than time.** Ravizza et al. (2013) modeled log taxi speed at Stockholm and
Zurich: distance alone gave R²adj 0.479, decomposing distance into components
raised it to 0.604, and total turn angle alone gave 0.470, nearly as much as
distance. This is the single most transferable result for a geometry-derived
approach, and it is why `routing.py` emits a vector of route descriptors rather
than one number.

Two caveats on transferring those numbers, because this entry does transfer
them. They are for taxi *speed*, not taxi *time*, at two airports, on
operator-supplied layouts rather than OSM. Nothing guarantees the same
decomposition helps at eleven airports on crowd-mapped geometry, and the
ablation ladder in [`modelling_notes.md`](modelling_notes.md) is what will
settle whether it does.

**Stand and runway identity carry less than expected, though the published
comparison is not quite the modern one.** At Zurich the operational
stand-to-runway lookup table explained R²adj 0.180 of taxi time against 0.793
for a layout-aware regression (Ravizza et al.). That 0.180 is a lookup table of
means, not a tuned gradient-boosted model on one-hot identifiers plus congestion
and time features, so it is evidence that layout beats a *table*, not that it
beats the baseline this entry actually has to beat. The ablation settles it.

**Aircraft type is consistently weak, and weather is unreliable as a direct
input.** Both fall out of feature-importance analyses repeatedly. Weather acts
on taxi-out through runway throughput and queue length, not through taxi speed,
so it belongs in the model as a regime indicator rather than a continuous
driver.

**Realistic error magnitudes.** Lim et al. (SID 2021) at Atlanta, on a
chronological split: a naive train-mean baseline at 216 s RMSE, the EUROCONTROL
PRU reference method at 199 s, gradient boosting at 78 s, and a graph neural
network at 76 s. The architecture bought 2 s; the feature set bought the rest.
Those figures are on a cleaned single-airport target. On this challenge's
data the airport-mean predictor scores 515 s with the tail and 361 s without
([`../reports/data_review.md`](../reports/data_review.md)), so an absolute
threshold transfers badly; the audit trigger here is a ratio. Lim's gradient
boosting cut the naive baseline by roughly 64%. A model that cuts the ex-tail
airport-mean RMSE by more than that, or that shows a validation-to-leaderboard
gap, gets a leakage audit before celebration.

## The leakage trap this field is full of

The strongest published congestion feature is not computable at prediction
time. Idris's takeoff queue size Q counts departures between the subject
aircraft's own pushback and its own takeoff; Ravizza and Chen's Q, and the
queue variables in several recent papers, are defined over the same interval.
Every one of them uses the target's own outcome. A model built on them cannot
be operated, and its reported error is not comparable to one that can.

The obvious repair is to count aircraft that have pushed back but not yet taken
off, evaluated at the subject's off-block time, the A-CDM "traffic density"
formulation is the same quantity. That is causally clean, and it is *still* not
usable here, which is the trap worth naming explicitly:

    N(t) = #{AOBT_j <= t} - #{AOBT_j + TAXITIME_j <= t}

The subtrahend needs every other movement's taxi time, and in January and July
2026 those movements are themselves ranking-set rows, where taxi time is the
hidden target. Both events precede `t`, so the causality rule passes it; the
field simply is not there at inference. Training on true values and scoring on a
proxy would leave the model leaning on a feature that is oracle-quality in
training and noise in the test months.

So this entry implements neither. The first replacement this paragraph
proposed (departures with off-block in a trailing window, restricted to the
subject's runway or pier) failed the same availability test once the ranking
set was opened: the off-block time is blanked for every departure in the
ranking months, not only the subject's. Congestion is therefore built from
what the ranking set does expose for other movements (take-offs, schedules,
and arrivals' landings, on-blocks and taxi-in times) with the identical
construction on both sides. The availability condition is stated alongside
causality under *Leakage watch* in
[`modelling_notes.md`](modelling_notes.md); it was added after this reading,
not before it, and it caught the plan's own feature second.

## Prior art on OSM airport graphs

Recovering a routable surface graph from OSM `aeroway` geometry is established.
`traffic` (Xavier Olive, MIT) ships an `AirportGraph` with node merging, edge
splitting and component filtering, the same class of repair implemented in
`graph.py`. Olive, Waltert & Schultz (ATRD 2025) document the same three OSM
defects: coincident duplicate nodes, apparent junctions with no topological
connection, and near-absent cross-category topology between parking positions,
taxiways and runways. Brownlee et al. (PATAT 2014) built OSM-plus-AIS airport
layouts explicitly because operator-supplied layouts are NDA-bound.

Two things appear genuinely unestablished, and both are stated as gaps rather
than claims:

**No published study validates OSM shortest-path stand-to-runway distance
against actual taxi routes at scale.** Ravizza et al. assumed the shortest route
because only stand and runway were available, and still got the results above.
Shortest-path distance is a strong proxy, not ground truth.

**No PRC Data Challenge entry, in either the 2024 or 2025 edition, appears to
have used OpenStreetMap.** For a taxi-out problem this is the obvious gap, and
it is where the originality of this entry sits, in the downstream use of
recovered geometry as regression features, not in the graph construction, which
is prior art and is cited as such.

## External data: license positions

Only openly licensed data may be used, so license precision matters more than
usefulness. Assessed rather than assumed.

| Source | License | Position |
| ------ | ------- | -------- |
| OpenStreetMap | ODbL 1.0 | **In use.** Attribution plus share-alike on derived databases, the distance matrices are a Derivative Database and ship under ODbL, not GPLv3 |
| OurAirports | Public domain | **In use** as a cross-check on runway designators and lengths; `le_heading_degT` is degrees *true*, so no magnetic model is needed |
| Iowa State Mesonet ASOS | Public domain | **In use from Gate 2b** (decided on review, 2 September 2026) as per-airport-hour regime flags, fetched by script and cached like the OSM data so a reader can re-fetch it. Covers all 11 ICAOs including LTFM. European caveat: `vsby` saturates at 6.21 mi, precisely where low-visibility procedures matter, so the raw METAR string must be parsed instead |
| ERA5 via CDS | CC-BY-4.0 | **Not used.** It was the source named for the winter block, which was never fitted because the winter layer it depended on did not clear |
| OPDI (opdi.aero) | No data license stated on the site | **In use since 4 October 2026** as an external dataset under the prize rules' second condition, "All used external datasets are openly accessible/usable and documented": EUROCONTROL publishes it openly, it is documented here, and the organizers confirmed that it is open to the challenge. The fourth condition, on an open source license, governs the datasets this entry adds, which the deposit carries under ODbL. Its flight events give stand, apron and taxiway milestones and the first surface position of a flight, read as features of the fusion model (`reports/fusion.md`). No OPDI file, and no table derived from one, enters the repository or the deposit: the client fetches the published files again |
| adsb.lol globe history | ODbL 1.0 | **In use since 4 October 2026.** Daily releases of open ADS-B traces; the ground events extracted from them are a Derivative Database and go into the deposit under ODbL with its notice, outside the repository's GPLv3 tree |
| EUROCONTROL ADRR / DDR | R&D use, no redistribution | **Excluded.** Not an open license |
| AIXM / AMDB / EAD | Restricted or commercial | **Excluded.** No openly licensed pan-European aerodrome mapping database exists; OSM is the only practical source |
| X-Plane `apt.dat` | **Unverifiable** | **Excluded.** Has exactly the taxi network wanted, but the widely repeated GPL-2 claim traces to forum posts, not to the publisher |
| OpenSky historical ADS-B | Research use, citation required, no redistribution of raw data | **Diagnostics only, never a model input** (decided on review, 2 September 2026). The prize rules require additional datasets to be "openly available under an open source license", which research-use is not, so nothing derived from OpenSky enters a feature or a fitted parameter. 2025 surface tracks at the ten airports, access requested, would validate OSM routes against routes flown and report speed distributions and intersection-departure evidence in the paper. Not for the ranking months either, on its license; the ground movements the final configuration reads for the ranking months come from adsb.lol and OPDI (rows above), under the challenge's post-hoc framing. Surface coverage at these airports is unmeasured (published assessments call OpenSky ground coverage limited, and recent surface studies converged on LSZH because a receiver was installed at the airport) so a coverage audit comes first |
| OpenSky scientific datasets (the ten listed at `opensky-network.org/data/scientific` on 20 September 2026) | Per dataset; the overview page states none, and the network's own data terms are research use | **Reviewed 20 September 2026; none taken.** Three tests, in order: an open license; 2025 ground-phase coverage at the ten airports; a mechanism the exposed fields lack, which `reports/residual.md` narrows to separating a gate delay from a hold. The weekly Monday state-vector snapshots are the only set with surface positions (an `onground` flag at 10 s intervals) and fall under the same reading as the historical database unless their own page states an open license, so they are a candidate for the paper's geometry validation at most, one day a week. The 2024 challenge dataset holds 2022 taxi-out times without stand or runway, which the 2025 training year already exceeds at these airports. The raw receiver samples, LocaRDS, the in-flight emergency sets, the climbing-segment set, the GICB capability database, the ADS-C set, the aircraft metadata database (keyed by `icao24`, which the challenge data does not carry) and the COVID-19 flight list carry no ground operations. Any later candidate meets the same three tests before an hour is spent on it |

## Sources

- Idris, Clarke, Bhuva, Kang, *Queuing Model for Taxi-Out Time Estimation*, Air
  Traffic Control Quarterly 10(1), 2002.
- Simaiakis & Balakrishnan, *A Queuing Model of the Airport Departure Process*,
  Transportation Science 50(1), 2016.
- Ravizza, Atkin, Maathuis, Burke, *A combined statistical approach and ground
  movement model for improving taxi time estimations at airports*, JORS 64(9),
  2013.
- Ravizza, Chen, Atkin, Stewart, Burke, *Aircraft taxi time prediction:
  comparisons and insights*, Applied Soft Computing 14, 2014.
- Lim, Tan, Lilith, Alam, *Variable Taxi-Out Time Prediction Using Graph Neural
  Networks*, SESAR Innovation Days, 2021.
- Lee, Malik, Jung, *Taxi-Out Time Prediction for Departures at Charlotte
  Airport Using Machine Learning Techniques*, AIAA AVIATION 2016-3910.
- Olive, Waltert, Schultz, *Assessing Airport Surface Traffic Performance from
  Open Sources of Aviation Data*, US–Europe ATRD Symposium, 2025.
- Olive, Waltert, Mori, Mouyon, *Filtering Aircraft Surface Trajectories Using
  Information on the Taxiway Structure of Airports*, ICRAT 2024.
- Brownlee, Atkin, Woodward, Benlic, Burke, *Airport Ground Movement: Real World
  Data Sets and Approaches to Handling Uncertainty*, PATAT 2014.
- EUROCONTROL, *Specification for Airport Collaborative Decision Making*, 2024:
  the variable taxi time ladder and the de-icing milestone split.
- EUROCONTROL PRU, *Additional Taxi-Out Time Performance Indicator*, 2023, the
  10th-percentile-per-(stand, runway) reference definition, used here as a
  documented baseline to beat.
- EUROCONTROL PRC, *PRC Data Challenge 2026*: rationale, data, ranking and
  eligibility pages, <https://ansperformance.eu/study/data-challenge/dc2026/>;
  and the 2025 outcome page with the selection committee's reviews,
  <https://ansperformance.eu/study/data-challenge/dc2025/outcome.html>.
- `traffic`, <https://github.com/xoolive/traffic> (MIT).
- OpenStreetMap copyright, <https://www.openstreetmap.org/copyright>.
- OurAirports data, <https://ourairports.com/data/>.
