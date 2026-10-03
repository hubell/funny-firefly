#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch and cache METARs for the challenge airports from the Iowa State Mesonet.

Gate 2b of ``docs/plan.md``. Weather enters the model as regime flags per
airport-hour (``taxiout.metar``), and the flags need the reports. The Iowa
Environmental Mesonet (IEM) at Iowa State University serves an archive of
ASOS/METAR observations for airports worldwide through a documented request
endpoint, and this script asks it for the training year and the two ranking
months at every listed airport.

Scripted and cached exactly as the OpenStreetMap fetch is, because an
external dataset a reader cannot re-fetch fails the reproduction clause of
the prize rules: the request parameters are recorded in a manifest beside
the data, with the terms the IEM states and the row counts received, so a
reader regenerates the cache from the recorded request rather than trusting
a copy. The cache is gitignored under ``data/raw/metar/``.

The terms, as the IEM's disclaimer page states them and as recorded in the
manifest: the materials "are in the public domain and may be used freely by
anyone for any lawful purpose", offered "without any warranty of accuracy",
and "Attributing the Iowa Environmental Mesonet of Iowa State University
would be appreciated." The README's data-sources section carries the same.

Resumable: a period already cached for a station is skipped. Delete a file
to refetch it. Requests are made one at a time with a pause between them;
the archive is a courtesy, not a service level.

Usage
-----
    python scripts/fetch_metar.py
    python scripts/fetch_metar.py --airports EHAM EGLL
    make metar
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.metar import FIELDS  # noqa: E402

CACHE = REPO / "data" / "raw" / "metar"
MANIFEST = CACHE / "MANIFEST.json"

ENDPOINT = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
TERMS_URL = "https://mesonet.agron.iastate.edu/disclaimer.php"
TERMS = (
    "The materials found on this website are in the public domain and may be "
    "used freely by anyone for any lawful purpose. Attributing the Iowa "
    "Environmental Mesonet of Iowa State University would be appreciated."
)
WARRANTY = (
    "We therefore provide this information without any warranty of accuracy. "
    "Users of this weather/climate data do so at their own risk, and are "
    "advised to use independent judgement as to whether to verify the data "
    "presented."
)

# The training year and the two ranking months, as [start, end) dates.
PERIODS: tuple[tuple[str, str], ...] = (
    ("2025-01-01", "2026-01-01"),
    ("2026-01-01", "2026-02-01"),
    ("2026-07-01", "2026-08-01"),
)

# The archive's request parameters, fixed here so the manifest can record
# them and a reader can repeat them. ``report_type=3`` is routine hourly
# METARs; ``tz=Etc/UTC`` keeps every stamp in UTC like the challenge data;
# ``format=onlycomma`` is plain CSV with no preamble; ``M`` marks missing.
PARAMS: dict[str, str] = {
    "tz": "Etc/UTC",
    "format": "onlycomma",
    "latlon": "no",
    "elev": "no",
    "missing": "M",
    "trace": "T",
    "direct": "no",
    "report_type": "3",
}
# The archive writes the station and the time first on every row whatever is
# asked, so neither is requested as data: the request is the one every cached
# file records in the manifest.
DATA_FIELDS = tuple(f for f in FIELDS if f not in ("station", "valid"))

PAUSE_S = 2.0
RETRIES = 4
TIMEOUT_S = 120


def request_url(icao: str, start: str, end: str) -> str:
    y1, m1, d1 = start.split("-")
    y2, m2, d2 = end.split("-")
    query = [("station", icao)]
    query += [("data", f) for f in DATA_FIELDS]
    query += [
        ("year1", y1),
        ("month1", m1),
        ("day1", d1),
        ("year2", y2),
        ("month2", m2),
        ("day2", d2),
    ]
    query += list(PARAMS.items())
    return ENDPOINT + "?" + urllib.parse.urlencode(query)


def fetch(url: str) -> str:
    last: Exception | None = None
    for attempt in range(RETRIES):
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT_S) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError) as e:
            last = e
            time.sleep(PAUSE_S * (2**attempt))
    raise RuntimeError(f"gave up after {RETRIES} attempts: {last}")


def cache_path(icao: str, start: str, end: str) -> Path:
    return CACHE / f"{icao}_{start}_{end}.csv"


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "source": "Iowa Environmental Mesonet, Iowa State University: ASOS/METAR archive",
        "endpoint": ENDPOINT,
        "parameters": PARAMS,
        # The time and the data asked for, as every manifest so far records them;
        # the station is the request itself.
        "fields": ["valid", *DATA_FIELDS],
        "terms_url": TERMS_URL,
        "terms": TERMS,
        "warranty": WARRANTY,
        "files": {},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airports", nargs="*", default=None, help="ICAO codes; default all")
    args = ap.parse_args()
    airports = [a.icao for a in AIRPORTS]
    if args.airports:
        airports = [a for a in airports if a in set(args.airports)]

    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for icao in airports:
        for start, end in PERIODS:
            path = cache_path(icao, start, end)
            if path.exists():
                print(f"{path.name}: cached")
                continue
            url = request_url(icao, start, end)
            print(f"{path.name}: fetching ...", end=" ", flush=True)
            text = fetch(url)
            lines = text.strip().splitlines()
            if not lines or not lines[0].startswith("station,valid"):
                print("unexpected response, not cached")
                print(text[:300], file=sys.stderr)
                return 1
            path.write_text(text, encoding="utf-8")
            manifest["files"][path.name] = {
                "station": icao,
                "start": start,
                "end": end,
                "url": url,
                "rows": len(lines) - 1,
                "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"{len(lines) - 1:,} reports")
            time.sleep(PAUSE_S)
    print(f"cache: {CACHE.relative_to(REPO)} ({len(manifest['files'])} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
