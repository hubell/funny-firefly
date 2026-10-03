#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Ten years of METARs before the training year, for context only.

A thin wrapper over ``scripts/fetch_metar.py``: the same Iowa Environmental
Mesonet endpoint, the same request parameters, the same fields, the same
terms, the same retry and pause, imported rather than copied. Only the cache
directory and the periods differ. The stations are the airports the challenge
data holds departures for, and the periods are the ten calendar years
2015 to 2024, one request per station and year, written to
``data/raw/metar_history/`` with a manifest of their own in the fetcher's form.

What it is for: one table in ``reports/monthly_ladder.md`` that sets each
airport-month of 2025 against its 2015 to 2024 norm for the three regime
flags of ``taxiout.metar``. It is not a weight, not a feature, and enters no
figure of the ladder. The model's own cache, ``data/raw/metar/``, is never
written: the wrapper refuses to start if its cache resolves anywhere under it.

Resumable: a station-year already cached is skipped.

Usage
-----
    python scripts/fetch_metar_history.py
    python scripts/fetch_metar_history.py --airports EHAM EGLL
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import fetch_metar as fm  # noqa: E402
from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.challenge import load_movements, to_pandas  # noqa: E402

CACHE = REPO / "data" / "raw" / "metar_history"
MANIFEST = CACHE / "MANIFEST.json"
MODEL_CACHE = fm.CACHE  # never written

FIRST_YEAR = 2015
LAST_YEAR = 2024
PERIODS: tuple[tuple[str, str], ...] = tuple(
    (f"{y}-01-01", f"{y + 1}-01-01") for y in range(FIRST_YEAR, LAST_YEAR + 1)
)


def outside_model_cache(path: Path) -> bool:
    model = MODEL_CACHE.resolve()
    target = path.resolve()
    return target != model and model not in target.parents


def stations() -> list[str]:
    """The airports the challenge data holds departures for, in ``AIRPORTS`` order."""
    dep = to_pandas(
        load_movements(REPO / "data" / "challenge", ["ADEP_mvt"], phase="DEP")
    )
    present = set(dep["ADEP_mvt"].astype(str).unique())
    return [a.icao for a in AIRPORTS if a.icao in present]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--airports", nargs="*", default=None, help="ICAO codes; default all")
    args = ap.parse_args(argv)
    if not outside_model_cache(CACHE):
        raise SystemExit(f"refusing: {CACHE} is inside the model's cache {MODEL_CACHE}")
    airports = stations()
    if args.airports:
        airports = [a for a in airports if a in set(args.airports)]

    # The fetcher's cache and manifest names point here for this process; its
    # request, retry and file naming are used as they are.
    fm.CACHE = CACHE
    fm.MANIFEST = MANIFEST
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = fm.load_manifest()
    manifest.setdefault(
        "note",
        "Context only: ten calendar years before the training year, for the "
        "historic weather table of reports/monthly_ladder.md; enters no model.",
    )
    failures = []
    for icao in airports:
        for start, end in PERIODS:
            path = fm.cache_path(icao, start, end)
            if not outside_model_cache(path):
                raise SystemExit(f"refusing to write {path}")
            if path.exists():
                print(f"{path.name}: cached", flush=True)
                continue
            url = fm.request_url(icao, start, end)
            print(f"{path.name}: fetching ...", end=" ", flush=True)
            try:
                text = fm.fetch(url)
            except RuntimeError as e:
                print(f"failed: {e}", flush=True)
                failures.append(path.name)
                continue
            lines = text.strip().splitlines()
            if not lines or not lines[0].startswith("station,valid"):
                print("unexpected response, not cached", flush=True)
                print(text[:300], file=sys.stderr)
                failures.append(path.name)
                continue
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
            print(f"{len(lines) - 1:,} reports", flush=True)
            time.sleep(fm.PAUSE_S)
    print(f"cache: {CACHE} ({len(manifest['files'])} files)")
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
