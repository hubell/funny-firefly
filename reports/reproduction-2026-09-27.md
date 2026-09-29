# The frozen commit, reproduced from a clean checkout

A dated record, written on 27 September 2026 from the run it describes, and
not regenerated. It records what a fresh clone of the repository at the tag
`freeze-2026-10-04`, commit `bfe43f127abaeb4403dc892b63ee309caef7edb4`,
produced from its own pinned fetch, the challenge data, and the two cached
tables, against what that commit carries. Every figure here is from the run's
own logs, timed with `/usr/bin/time` under a sleep inhibitor on the machine
the entry was developed on (x86-64, 8 logical CPUs, the same environment the
frozen reports were generated in).

## What a reader brings, and what the repository makes

The clone at the tag has 178 tracked files and, under `data/`, three of them:
`data/challenge/README.md`, `data/processed/taxi_distance/ODbL-NOTICE.md` and
`data/processed/taxi_distance/README.md`. Everything else under `data/` is
either brought or made:

| input | brought or made | how |
| --- | --- | --- |
| the challenge data, 14 files | brought, from the organizers | not redistributable |
| the eleven Overpass answers | made | `scripts/fetch_osm.py --pin-from-register --force`, verified by `--compare`; or brought from the deposit |
| the 35 files of the distance matrices | made | `make reports`, offline, from the answers; or brought from the deposit |
| the METAR cache, 33 files and a manifest | brought | the deposit, or `make metar`, which refetches from the archive |
| the OurAirports tables, `airports.csv` with a manifest and `runways.csv` | brought | the deposit; the table is rebuilt nightly, so a fetch made later returns different files |
| the 52 residual files | made | `make baseline`, the ladder, about ten hours |
| the records under `data/processed/determinism/`, 90 files | neither | a measurement campaign under the parameters of its day; not redistributed, and no target reruns it |

The worked airport needs none of it: in the bare clone, with no network and
nothing under `data/` but the three tracked files, `make synthetic` took
3.88 s and wrote `reports/synthetic_airport.svg` byte-identical to the
committed one, and a report that differs from the committed one in its
generation date alone.

## The geometry

The pinned fetch, with the network, took 18 min 28 s. Every airport was
answered by `overpass-api.de`, every answer's database stamp is later than
its pin, and three fallback answers were refused because their instances'
databases predated the pin: `kumi.systems` at 2026-07-28 for EDDM and at
2026-05-31 for LSZH, `private.coffee` at 2026-07-24 for EHAM. Six of the
eleven airports needed the primary instance more than once.

| airport | pin | database at | elements |
| --- | --- | --- | ---: |
| EDDF | 2026-09-01T13:23:05Z | 2026-09-27T06:14:33Z | 1,324 |
| EDDM | 2026-09-01T13:23:05Z | 2026-09-27T06:18:43Z | 1,194 |
| EGLL | 2026-09-01T13:23:05Z | 2026-09-27T06:22:48Z | 1,083 |
| EHAM | 2026-09-01T13:24:06Z | 2026-09-27T06:24:50Z | 948 |
| LEBL | 2026-09-01T13:26:06Z | 2026-09-27T06:24:50Z | 1,013 |
| LEMD | 2026-09-01T13:27:05Z | 2026-09-27T06:25:51Z | 1,478 |
| LFPG | 2026-09-01T13:27:05Z | 2026-09-27T06:27:51Z | 1,717 |
| LIRF | 2026-09-01T13:29:10Z | 2026-09-27T06:27:51Z | 494 |
| LTAI | 2026-09-01T13:30:11Z | 2026-09-27T06:30:49Z | 801 |
| LTFM | 2026-09-01T13:30:11Z | 2026-09-27T06:30:49Z | 1,631 |
| LSZH | 2026-09-01T13:32:15Z | 2026-09-27T06:32:51Z | 716 |

`--compare` against the answers the committed matrices were built from:
11 compared, 11 identical, in every element, every field and the same order.
The control, LIRF asked of the same instance at 2020-01-01T00:00:00Z:
different, 355 elements against 494, so the instance applies `[date:]`.

`make reports`, `make crosscheck` and `make synthetic`, offline, took 1 min
49 s, 0.32 s and 3.74 s. **All 35 files under `data/processed/taxi_distance/`
came back byte-identical** to the ones the committed reports were built from,
and the four geometry reports they rewrite, `stand_coverage.md`,
`taxi_routing.md`, `penalty_sensitivity.md` and `runway_crosscheck.md`, came
back byte-identical to the committed ones.

## The reports

Regenerated in the order their declared inputs dictate, each verified against
its declared inputs before the next ran:

| report | elapsed | peak memory |
| --- | ---: | ---: |
| `geometry_gaps.md` | 36.67 s | 2,975,376 kB |
| `data_review.md` | 23.91 s | 2,257,464 kB |
| `tails.md` | 52.18 s | 1,907,784 kB |
| `baseline.md`, the ladder | 9 h 36 min 30 s | 8,507,212 kB |
| `congestion.md` | 1 min 42.62 s | 6,998,284 kB |
| `occupancy.md` | 15.27 s | 2,956,064 kB |
| `residual.md` | 20.96 s | 3,353,548 kB |
| `headroom.md` | 2 min 33.72 s | 3,843,448 kB |
| `transfer.md` | 49 min 22.05 s | 7,399,752 kB |

**All 52 residual files the ladder wrote came back byte-identical** to the
ones the committed reports read. Compared figure by figure with the committed
copies, **no figure moved in any of the fourteen regenerated reports**. Nine
differ in one line, their generation date; `headroom.md` differs in its date
and in the days it reads from its two input files' timestamps; four are
byte-identical. Every footer declares the same inputs at the same digests on
both sides, the eleven answers by the digest of their parsed elements.

`reports/determinism.md` was not regenerated, because it cannot be: it reads
90 records of a measurement campaign under `data/processed/determinism/`,
made under the parameters of its day, which hold arrays derived from the
challenge data and are not redistributed. It is a dated record, and stands as
committed.

The two freeze conditions, run in the clone with the regenerated reports in
place: `scripts/check_paper.py --strict` passed, 398 figures, 358 matched, 29
historical, 11 exempt, 0 unmatched. `scripts/check_provenance.py --strict`
failed on one check alone, the declared inputs of `determinism.md`, 90 of its
91 unverifiable because the records are not there; every input of every other
report verified.

## The submission

The five-seed fit of the submission rung gave the same bytes three times:
`funny-firefly_v5.parquet`, 1,491,114 bytes, sha256
`8a10f3acd8d64ee332af78d8be8fbe3e030c9407d942ca42d940274e595cb16a`, 215,876
rows.

| built | where | elapsed |
| --- | --- | ---: |
| 26 September, before the freeze commit, at `195e15f` | the working copy | 29 min 26.84 s |
| 27 September, at the tag, tree clean | the working copy | 29 min 37.17 s |
| 27 September, at the tag, from the clone's own fetch and reports | the clone | 29 min 23.16 s |

The manifest written beside the file records the tagged commit, a clean tree,
five seeds, 2,083,555 fitting rows, and the route-length check over the
training year; the three manifests agree in every field but the file's name
and the time of writing.

## Timing

Twenty-two timed runs, 11 h 53 min 56 s of elapsed time in all, from
05:37:13Z to 17:47:52Z on 27 September 2026. The suspend counter of the host
read the same before and after every run.

Distance matrices derived from OpenStreetMap: data (c)
OpenStreetMap contributors, ODbL 1.0. Movement and flight data supplied by
EUROCONTROL PRC and the OpenSky Network under the terms of the PRC Data
Challenge 2026.
Weather observations from the Iowa Environmental Mesonet of Iowa State
University, and airport tables from OurAirports, both public domain.
