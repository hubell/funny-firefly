#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch OPDI's flight events and flight lists, and match them to the departures.

OPDI v0.0.2, EUROCONTROL's Open Performance Data Initiative, publishes its
flight events as ten-day files and its flight lists as monthly files at
https://www.eurocontrol.int/performance/data/download/OPDI/v002/. The site
states no data licence; the entry uses OPDI on the organizers' confirmation.
So no OPDI file and no table built from one is tracked or deposited: this
script fetches the published files again, into ``data/raw/opdi/``, and builds
every table from them, into ``data/interim/opdi/``. Both are gitignored.

The files fetched are those ``taxiout.opdi`` names for the training year and
the months the ranking file contains (``taxiout.challenge.ranking_months``):
the ten-day event files overlapping those months (``match_months``), and the
monthly lists of those months and of the month before each run of them
(``list_months``); for January and July 2026, 45 event files and 16 lists,
December 2024 to December 2025, January, June and July 2026. A file
already in the cache at the size and sha256 its manifest line records is not
fetched again; ``data/raw/opdi/MANIFEST.json`` records each file's size,
sha256 and fetch time, or the HTTP status it failed with. On 4 October 2026
``flight_events_20251002_20251012`` answered 404, and the pipeline runs
without it, as the matched tables of that day did.

Then, each step reading the previous one's output only (``taxiout.opdi``): the
reduction of each event file, the tie map, the month tables and the match of
every departure of the training year and the ranking months, written as
``data/interim/opdi/matched/<YYYYMM>.parquet`` keyed on ``MVT_ID_mvt``. The
matched tables carry challenge identifiers and stay on the machine that built
them. The whole pipeline takes about five minutes on eight cores once the
14 GB are on disk; the fetch is what takes time.

Usage
-----
    python scripts/fetch_opdi.py              # fetch what is missing, then build
    python scripts/fetch_opdi.py --offline    # build from the cache only
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from taxiout import opdi  # noqa: E402
from taxiout.adsb import sha256_file  # noqa: E402
from taxiout.challenge import ranking_months  # noqa: E402

DATA = REPO / "data" / "challenge"
RAW = REPO / "data" / "raw" / "opdi"
WORK = REPO / "data" / "interim" / "opdi"
MANIFEST = RAW / "MANIFEST.json"


def fetch(names: list[str], offline: bool) -> dict:
    """Every named file in the cache, fetched if missing; the manifest updated."""
    RAW.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": {}}
    for name in names:
        path = RAW / name
        rec = manifest["files"].get(name, {})
        if path.exists():
            size = path.stat().st_size
            if rec.get("bytes") == size and rec.get("sha256"):
                continue
            manifest["files"][name] = {
                "bytes": size,
                "sha256": sha256_file(path),
                "recorded_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
            }
            continue
        if offline:
            manifest["files"][name] = {"status": "not in the cache (offline)"}
            continue
        tmp = path.with_suffix(".part")
        t0 = time.monotonic()
        r = subprocess.run(
            [
                "curl",
                "-sS",
                "-L",
                "--retry",
                "3",
                "-o",
                str(tmp),
                "-w",
                "%{http_code}",
                opdi.url_of(name),
            ],
            capture_output=True,
            text=True,
        )
        code = r.stdout.strip()[-3:]
        if r.returncode == 0 and code == "200":
            tmp.rename(path)
            manifest["files"][name] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "fetch_s": round(time.monotonic() - t0, 3),
            }
        else:
            tmp.unlink(missing_ok=True)
            manifest["files"][name] = {"status": f"HTTP {code}, curl exit {r.returncode}"}
        print(name, manifest["files"][name], flush=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    return manifest


def _reduce(name: str) -> dict:
    return opdi.reduce_file(RAW / name, WORK / "reduced" / name)


def build(workers: int, match: tuple[str, ...]) -> dict:
    """The reduction, the tie, the month tables and the match, from the cache.

    ``match`` is ``opdi.match_months`` of the ranking file's months.
    """
    log = {}
    lists = opdi.list_months(match)
    names = [n for n in opdi.event_files(match) if (RAW / n).exists()]
    log["event_files_missing"] = [
        n for n in opdi.event_files(match) if not (RAW / n).exists()
    ]
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        log["reduction"] = list(pool.map(_reduce, names))
    log["reduction_s"] = round(time.monotonic() - t0, 1)
    reduced = [WORK / "reduced" / n for n in names]

    t0 = time.monotonic()
    tm, log["tie"] = opdi.tie_map(RAW, reduced, lists)
    opdi.write_table(tm, WORK / "meta" / "tie_map.parquet")
    log["tie_s"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    log["months"] = {}
    for ym in lists:
        table, info = opdi.month_table(ym, RAW, reduced, tm)
        opdi.write_table(table, WORK / "events" / f"opdi_{ym}.parquet")
        log["months"][ym] = info
    log["month_tables_s"] = round(time.monotonic() - t0, 1)

    t0 = time.monotonic()
    log["matched"] = {}
    for ym in match:
        frames = [
            pd.read_parquet(WORK / "events" / f"opdi_{m}.parquet")
            for m in opdi.months_around(ym)
            if (WORK / "events" / f"opdi_{m}.parquet").exists()
        ]
        out, info = opdi.match_month(ym, DATA, pd.concat(frames))
        opdi.write_table(out, WORK / "matched" / f"{ym}.parquet")
        info["sha256"] = sha256_file(WORK / "matched" / f"{ym}.parquet")
        log["matched"][ym] = info
        print(ym, info, flush=True)
    log["match_s"] = round(time.monotonic() - t0, 1)
    (WORK / "build_log.json").write_text(json.dumps(log, indent=1, default=int) + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--offline", action="store_true", help="build from the cache only")
    ap.add_argument("-j", "--workers", type=int, default=2, help="reduction workers")
    args = ap.parse_args()
    if not (DATA / "ranking.parquet").exists():
        print(f"no challenge data under {DATA}; the match needs it", file=sys.stderr)
        return 1
    match = opdi.match_months(ranking_months(DATA))
    lists = opdi.list_files(opdi.list_months(match))
    manifest = fetch(opdi.event_files(match) + lists, args.offline)
    missing = [n for n in lists if not (RAW / n).exists()]
    if missing:
        print("flight lists missing: " + " ".join(missing), file=sys.stderr)
        return 1
    log = build(args.workers, match)
    print(
        f"{len(manifest['files'])} files in the manifest; event files missing:"
        f" {log['event_files_missing'] or 'none'}; {len(log['matched'])} months matched"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
