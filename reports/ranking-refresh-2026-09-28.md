# The ranking set, re-issued on 4 September

A dated record, written on 28 September 2026 from the comparison it describes,
and not regenerated. It records how the two ranking files the organizers
published on 4 September 2026 differ from the copies this entry was built on,
which were taken on 2 September. Every figure here is from that comparison:
the two files as they stood in the competition's data bucket,
`prc-2026-datasets`, downloaded on 28 September and checked against the
bucket's own listing, set beside the copies held since 2 September.

## The files

| file | held since 2 September | in the bucket, dated 4 September |
| --- | ---: | ---: |
| `submitting.parquet` | 1,152,346 bytes | 1,677,097 bytes |
| `ranking.parquet` | 28,002,147 bytes | 43,564,605 bytes |

Their sha256:

- `submitting.parquet`, 2 September:
  `241724d8c234d2c34c96519ef1b6891646e694b142bff2c3504e83a394ee8f73`
- `submitting.parquet`, 4 September:
  `0d383408ed5ac27bfcfd12d6b961cee77c8a6364b10dad9fe3ec81115c79652b`
- `ranking.parquet`, 2 September:
  `25556e61c5f71020287ca130377ce079dc22d2fa7f7f1eaf4a2f22acafab463d`
- `ranking.parquet`, 4 September:
  `9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86`

The bucket dates the two objects 2026-09-04 08:27:23 and 08:30:40 UTC. The
downloads are those objects: each is the size the listing gives, the
template's MD5 is its ETag, and the ranking file's multipart ETag, computed
over 16 MiB parts, is the listing's.

## The rows

| | 2 September | 4 September | added |
| --- | ---: | ---: | ---: |
| template rows, `submitting.parquet` | 215,876 | 344,841 | 128,965 |
| ranking rows, `ranking.parquet` | 431,843 | 689,534 | 257,691 |
| of which departures | 215,876 | 344,841 | 128,965 |
| of which arrivals | 215,967 | 344,693 | 128,726 |

- **No row was removed.** Every movement identifier of the earlier files is
  in the later ones, and in all four files the identifiers are distinct and
  none is null.
- **The template is still the ranking set's departures,** exactly, in both
  versions, and the target column is empty in every template row and every
  ranking departure.
- **The columns did not change:** the same 2 and 30 columns, with the same
  names, types and order.

## July 2026 gained seven airports

January 2026 did not change: 152,719 template rows over the ten airports, the
same count at each. July 2026 held three airports on 2 September, EDDF with
20,650 template rows, EGLL with 20,557 and EHAM with 21,950, and those counts
did not change. On 4 September it holds all ten. The 128,965 added template
rows are all July 2026 departures, from 1 July 00:07:17 to 31 July 23:56:19
UTC, at the seven airports that had none:

| airport | July 2026 template rows added |
| --- | ---: |
| EDDM | 15,013 |
| LEBL | 17,879 |
| LEMD | 20,152 |
| LFPG | 21,743 |
| LIRF | 15,838 |
| LSZH | 13,155 |
| LTFM | 25,185 |
| all seven | 128,965 |

The arrivals follow them: 128,726 July arrivals added at the same seven
airports, and January's 152,871 unchanged.

## The timestamps of the rows both versions hold

Of the 431,843 ranking rows in both versions, 405,623 differ, and in three
columns only: `MVT_TIME_UTC_mvt` in 397,016 rows, `BLOCK_TIME_UTC_mvt` in
198,626 and `TAXITIME_SEC_mvt` in 207,229. The other 26 columns are identical,
with nulls counted equal. Every change below is the later file's value minus
the earlier one's.

- **Departures, 215,876 rows.** `MVT_TIME_UTC_mvt` moved in 198,394 of them
  (91.9%), and never by more than 12 s: minimum -12 s, 10th percentile -7 s,
  median +1 s, 90th percentile +7 s, maximum +12 s, with 50.0% of the moves
  later and 50.0% earlier. The block time and the taxi time are empty for
  every departure in both versions.
- **Arrivals, 215,967 rows.** `MVT_TIME_UTC_mvt` moved in 198,622 of them
  (92.0%) and `BLOCK_TIME_UTC_mvt` in 198,626 (92.0%), each within 12 s
  either way, and `TAXITIME_SEC_mvt` in 207,229 (96.0%), within 24 s either
  way. In every arrival row of both versions the taxi time equals the block
  time minus the movement time, to the second.
- **The same everywhere.** The share of moved movement times lies between
  91.4% and 92.2% in each of the 13 airport-months of departures the two
  versions share, and between 91.5% and 92.5% for arrivals. No value in
  either version carries a fraction of a second.

## What did not change

The twelve monthly training files. Each equals the bucket's object in size
and in its multipart ETag, computed over 16 MiB parts, and the bucket dates
all twelve 13 August 2026. Nothing the model is fitted or validated on
changed.

## The scored files

The entry's scored files `funny-firefly_v4.parquet` and
`funny-firefly_v5.parquet` were built on the template of 2 September: each
holds 215,876 rows, and their identifiers are exactly that template's. Each
lacks 128,965 of the 344,841 rows the template of 4 September lists. That
count, 344,841, is also the `usedPairs` that every scored submission on the
competition's leaderboard reported when the board was read on 18 September
2026.

From 28 September 2026 the entry reads the files of 4 September in place of
those of 2 September.

Distance matrices derived from OpenStreetMap: data (c)
OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by
EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data
Challenge 2026.
