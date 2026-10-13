# Challenge data goes here

**Nothing in this directory is ever committed, and it is not redistributable
under any license.** The PRC Data Challenge datasets are supplied by
EUROCONTROL PRC and the OpenSky Network under the terms of the challenge, which
grant access to participants and nothing further. Unlike the OpenStreetMap
question (where ODbL at least permits redistribution on its own terms) there
is no version of this data that may leave your machine.

Only this README is tracked. `make licence-check` fails if anything else here
becomes trackable.

## Layout

Put the buckets here as they arrive:

```
data/challenge/
  movements/            twelve monthly parquets, full-year 2025
  ranking.parquet       the movements to predict (Jan, Feb, Jun and Jul 2026)
  submitting.parquet    the submission template
```

The final phase publishes these two as `final_ranking.parquet` and
`final_submitting.parquet`; they go here under the names above, which are the
names the code reads.

Roughly 280 MB in total, so it is small enough to keep locally and far too
sensitive to keep anywhere else.

## Credentials

The MinIO/S3 keys arrive by email after team approval. They belong in `.env`,
which is gitignored, and nowhere else, not in a notebook, not in a script
default, not in a shell history you later paste somewhere.

```
mc alias set opensky https://s3.opensky-network.org/ "$BUCKET_KEY" "$BUCKET_SECRET"
```

## Before writing any feature code against this

Two checks came first, recorded in [`../../docs/plan.md`](../../docs/plan.md)
as Gate 0 and answered in
[`../../reports/data_review.md`](../../reports/data_review.md), which
`make data-review` regenerates from these files:

1. **Does the OSM identifier join to `STAND_mvt`?** Yes, after a staged
   normalization (`taxiout.join`): 87–99.9% of departures by airport.

2. **Is the target reconstructible from other columns?** Not cleanly.
   `AOBT_3_flt` survives in `ranking.parquet`, a minute-resolution estimate
   that misses the blanked block time by minutes and, alone, carries no more
   information than the filed EOBT. The base configuration reads nothing built
   on it; the final configuration does (`taxiout.aobt3`), under the
   challenge's post-hoc framing, and the paper says so (§6.3, §8.7.3). The
   constants that say what may not be read are in `taxiout.challenge`.

A third thing the review found that the checks did not ask: the ranking set
blanks `BLOCK_TIME_UTC_mvt` for every departure, so no feature may count other
departures' pushbacks either. Read *Leakage watch* in
[`../../docs/modelling_notes.md`](../../docs/modelling_notes.md) before
building any congestion feature.

## A note on what may be derived from this

Features computed from challenge data stay local, as the data does. A feature
frame that *also* carries the OSM-derived distance columns is additionally a
Derivative Database of OpenStreetMap: see
[`../processed/taxi_distance/ODbL-NOTICE.md`](../processed/taxi_distance/ODbL-NOTICE.md).
Two separate reasons to keep the same file out of the repository.

The predictions you submit are encumbered by neither.
