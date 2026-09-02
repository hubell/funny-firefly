#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch and cache OSM movement-area geometry for the challenge airports.

Resumable: airports already cached are skipped unless ``--force`` is given, so
this can be run repeatedly until every airport lands.

    python scripts/fetch_osm.py                 # fetch anything missing
    python scripts/fetch_osm.py --only EDDF LTFM
    python scripts/fetch_osm.py --force         # refetch everything
    python scripts/fetch_osm.py --date 2026-01-01T00:00:00Z --force

The ``--date`` form pins the query to a moment in OSM's history, which is what
makes a rebuild months later reproduce these numbers rather than whatever the
map says then. It is not the default because not every Overpass mirror answers
attic queries; if one refuses, the client rotates as usual and the failure is
reported rather than silently unpinned.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS, BY_ICAO  # noqa: E402
from taxiout.osm import POLITE_DELAY_S, fetch_airport  # noqa: E402

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
    args = parser.parse_args()

    targets = [BY_ICAO[i.upper()] for i in args.only] if args.only else list(AIRPORTS)

    failures = []
    fetched = 0
    for ap in targets:
        if fetched:
            time.sleep(POLITE_DELAY_S)
        try:
            payload, cached = fetch_airport(ap, CACHE, force=args.force, date=args.date)
        except Exception as exc:  # noqa: BLE001
            print(f"  {ap.icao}  FAILED  {exc}", flush=True)
            failures.append(ap.icao)
            continue
        if not cached:
            fetched += 1
        origin = "cache" if cached else "overpass"
        print(
            f"  {ap.icao}  {len(payload['elements']):>6} elements  ({origin})",
            flush=True,
        )

    if failures:
        print(f"\nincomplete: {' '.join(failures)} -- rerun to retry", flush=True)
        return 1
    print("\nall airports cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
