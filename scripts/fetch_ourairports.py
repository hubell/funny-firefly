#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch and cache OurAirports' airport table for the destination block.

The destination aerodrome is exposed for every ranking departure, and
``taxiout.destination`` turns it into a bearing and a distance rather than a
categorical of 914 levels. That needs a position for every destination the
data names, and the challenge files carry none: OurAirports publishes one,
rebuilt nightly from its contributors' edits, as a single CSV of about twelve
megabytes. The sibling runway table is already cached beside it by
``scripts/check_runways_ourairports.py``.

Scripted and cached exactly as the METAR fetch is, and for the same reason: an
external dataset a reader cannot re-fetch fails the reproduction clause of the
prize rules. The URL, the fetch time, the rows received, the fields the model
reads and the terms are recorded in a manifest beside the data, so a reader
regenerates the cache from the recorded request rather than trusting a copy.
The cache is gitignored under ``data/raw/``.

The terms, as OurAirports states them and as recorded in the manifest: "All
data is released to the Public Domain, and comes with no guarantee of accuracy
or fitness for use." The README's data-sources section carries the same.

Resumable: an existing cache is left alone and reported, and ``--refresh``
refetches it. The response is checked against the header the reader expects
before a byte is written, so a redirect or an error page cannot land in the
cache and be parsed as data later.

Usage
-----
    python scripts/fetch_ourairports.py
    python scripts/fetch_ourairports.py --refresh
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.destination import FIELDS  # noqa: E402
from taxiout.osm import USER_AGENT  # noqa: E402

SOURCE = "https://davidmegginson.github.io/ourairports-data/airports.csv"
CACHE = REPO / "data" / "raw" / "ourairports_airports.csv"
MANIFEST = REPO / "data" / "raw" / "ourairports_MANIFEST.json"

TERMS_URL = "https://ourairports.com/data/"
TERMS = (
    "All data is released to the Public Domain, and comes with no guarantee "
    "of accuracy or fitness for use."
)

PAUSE_S = 2.0
RETRIES = 3
TIMEOUT_S = 120


def fetch(url: str) -> str:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=TIMEOUT_S) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(PAUSE_S * (2**attempt))
    raise RuntimeError(f"gave up after {RETRIES} attempts: {last}")


def header_and_rows(text: str) -> tuple[list[str], int]:
    """The header fields and the number of records, read as CSV.

    Counted through a reader rather than by splitting on newlines, because
    airport names carry commas and quotes and a quoted field may span a line;
    the count goes in the manifest and has to be the number of airports.
    """
    reader = csv.reader(io.StringIO(text))
    return next(reader, []), sum(1 for _ in reader)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refetch the CSV")
    args = parser.parse_args()

    if CACHE.exists() and not args.refresh:
        size_mb = CACHE.stat().st_size / 1e6
        print(
            f"{CACHE.relative_to(REPO)}: cached ({size_mb:.1f} MB); --refresh to refetch"
        )
        return 0

    print(f"{SOURCE}: fetching ...", end=" ", flush=True)
    try:
        text = fetch(SOURCE)
    except Exception as exc:  # noqa: BLE001 - one network call, many failure modes
        print("failed")
        print(f"could not fetch OurAirports data: {exc}", file=sys.stderr)
        return 1

    header, rows = header_and_rows(text)
    absent = [f for f in FIELDS if f not in header]
    if absent:
        print("unexpected response, not cached")
        print(f"header does not name {absent}", file=sys.stderr)
        print(text[:300], file=sys.stderr)
        return 1

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(text, encoding="utf-8")
    MANIFEST.write_text(
        json.dumps(
            {
                "source": "OurAirports: airports.csv",
                "url": SOURCE,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rows": rows,
                "fields": list(FIELDS),
                "terms_url": TERMS_URL,
                "terms": TERMS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"{rows:,} airports")
    print(f"cache: {CACHE.relative_to(REPO)}")
    print(f"manifest: {MANIFEST.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
