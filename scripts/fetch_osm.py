#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch and cache OSM movement-area geometry for the challenge airports.

Resumable: airports already cached are skipped unless ``--force`` is given, so
this can be run repeatedly until every airport lands.

    python scripts/fetch_osm.py                 # fetch anything missing
    python scripts/fetch_osm.py --only EDDF LTFM
    python scripts/fetch_osm.py --force         # refetch everything
    python scripts/fetch_osm.py --date 2026-01-01T00:00:00Z --force
    python scripts/fetch_osm.py --pin-from-cache --force

The ``--date`` form pins the query to a moment in OSM's history, which is what
makes a rebuild months later reproduce these numbers rather than whatever the
map says then. It is not the default because not every Overpass mirror answers
attic queries; if one refuses, the client rotates as usual and the failure is
reported rather than silently unpinned.

``--pin-from-cache`` is the form that reproduces THIS cache, and it exists
because one ``--date`` cannot. The eleven airports were fetched sequentially
and unpinned, so each was answered from whatever OSM state its own mirror had
ingested at the moment it was served: eleven files carrying seven distinct
``timestamp_osm_base`` values, spanning 2026-09-01T13:23:05Z to 13:32:15Z. A
single date would pin eight of the eleven to a state their matrices were not
built from. This mode reads each airport's own stamp out of its own cache file
and pins that airport's query to it, so nothing has to be kept in sync and the
rebuild reproduces per airport. It requires ``--force``, because without it the
cache is returned unread and a run that changed nothing would look like a
pinned rebuild; and an airport whose stamp cannot be read is a failure for that
airport rather than an unpinned fetch.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS, BY_ICAO  # noqa: E402
from taxiout.osm import POLITE_DELAY_S, cached_osm_base, fetch_airport  # noqa: E402

CACHE = REPO / "data" / "raw"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", metavar="ICAO")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--date",
        metavar="ISO8601",
        help="pin to a moment in OSM history, e.g. 2026-01-01T00:00:00Z",
    )
    parser.add_argument(
        "--pin-from-cache",
        action="store_true",
        help="pin each airport to its own cached timestamp_osm_base",
    )
    args = parser.parse_args()

    if args.pin_from_cache and args.date:
        parser.error("--pin-from-cache and --date pin the same thing two ways")
    if args.pin_from_cache and not args.force:
        parser.error(
            "--pin-from-cache needs --force: without it every airport is served"
            " from the cache and nothing is pinned, which is a run that looks"
            " like a pinned rebuild and is not one"
        )

    targets = [BY_ICAO[i.upper()] for i in args.only] if args.only else list(AIRPORTS)

    failures = []
    fetched = 0
    for ap in targets:
        if fetched:
            time.sleep(POLITE_DELAY_S)
        date = args.date
        if args.pin_from_cache:
            date = cached_osm_base(CACHE, ap.icao)
            if date is None:
                # Never fall through to an unpinned fetch: that would overwrite
                # a cached answer whose OSM state is recorded with one whose is
                # not, and the loss would be silent.
                print(
                    f"  {ap.icao}  FAILED  no timestamp_osm_base cached;"
                    " refusing to fetch unpinned",
                    flush=True,
                )
                failures.append(ap.icao)
                continue
        try:
            payload, cached = fetch_airport(ap, CACHE, force=args.force, date=date)
        except Exception as exc:  # noqa: BLE001
            print(f"  {ap.icao}  FAILED  {exc}", flush=True)
            failures.append(ap.icao)
            continue
        if not cached:
            fetched += 1
        origin = "cache" if cached else "overpass"
        pin = f"  pinned to {date}" if date else ""
        print(
            f"  {ap.icao}  {len(payload['elements']):>6} elements  ({origin}){pin}",
            flush=True,
        )

    if failures:
        print(f"\nincomplete: {' '.join(failures)} -- rerun to retry", flush=True)
        return 1
    print("\nall airports cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
