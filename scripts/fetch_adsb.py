#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch adsb.lol's globe history and cache the per-leg event tables.

The days are §5's: every second day of 2025 from 1 January, and every day of
January and July 2026, 245 in all (``taxiout.fusion.adsb_days``). For each, the
``prod`` release is fetched from ``adsblol/globe_history_<YYYY>`` on GitHub,
streamed through ``taxiout.adsb``'s extraction, and the day's table written to
``data/raw/adsb/<day>.parquet``; the release parts are deleted once the table
is written unless ``--keep-parts``. A day of 2025 is about 2.9 GB of release
and a day of 2026 about 3.6 GB; the fetch, not the parse, sets the time.

Seven days of 2025 (29 and 31 May, and 2 to 10 June every second day) exist
only as ``prod-0tmp`` releases, which carry no digests and are not §5's
``prod`` variant. They are left out, with that reason, in the manifest.

``data/raw/adsb/MANIFEST.json`` records each day's table sha256 and rows, or
the reason it has none, and the release it was built from;
``data/raw/adsb/ODbL-NOTICE.md`` states the licence. The tables are a
Derivative Database of adsb.lol's data under the Open Database License 1.0,
and like every cache under ``data/raw/`` they are not tracked: they travel in
the deposit beside this repository.

``--verify`` reads the cache against the manifest, every table's sha256, rows
and notice, and touches no network: what ``make adsb`` does once the cache is
there. Reads no challenge data.

Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.

Usage
-----
    python scripts/fetch_adsb.py --verify
    python scripts/fetch_adsb.py --manifest
    python scripts/fetch_adsb.py --all -j 4
    python scripts/fetch_adsb.py 2025-01-15 2026-07-15 --keep-parts
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from taxiout import adsb  # noqa: E402
from taxiout.fusion import adsb_days  # noqa: E402

CACHE = REPO / "data" / "raw" / "adsb"
PARTS = REPO / "data" / "interim" / "adsb_parts"
MANIFEST = CACHE / "MANIFEST.json"
NOTICE_FILE = CACHE / "ODbL-NOTICE.md"

# The days with no ``prod`` release, and why none is read.
NO_PROD_RELEASE = {
    d: "only prod-0tmp releases, with no digests: not section 5's prod variant and"
    " not verifiable; left out"
    for d in (
        "2025-05-29",
        "2025-05-31",
        "2025-06-02",
        "2025-06-04",
        "2025-06-06",
        "2025-06-08",
        "2025-06-10",
    )
}

NOTICE_TEXT = """# adsb.lol event tables: licence

The files `<YYYY-MM-DD>.parquet` in this directory are per-leg off-block event
tables extracted from adsb.lol's globe history (`adsblol/globe_history_2025`
and `adsblol/globe_history_2026` on GitHub, the `prod` release of each day) by
`scripts/fetch_adsb.py` and `taxiout.adsb`.

Contains data from adsb.lol, available under the Open Database License (ODbL) 1.0.

They are a Derivative Database of that data and are offered under the same
licence, https://opendatacommons.org/licenses/odbl/1-0/. Every table carries
the notice in its parquet metadata (`licence`), with the release it was built
from (`source`). `MANIFEST.json` lists every day, its table's sha256 and rows,
or the reason it has none.

The code that builds them is GPLv3 and lives in the repository; these tables
are not tracked there, because a second licence does not belong in a tree that
ships GPLv3 in totality. They travel in the deposit beside it.
"""


def write_manifest(days: list[str]) -> dict:
    """The manifest from the tables on disk and the runner's log."""
    import pyarrow.parquet as pq

    log = {}
    log_path = CACHE / "runner_log.jsonl"
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            r = json.loads(line)
            log[r["day"]] = r
    entries = {}
    for d in days:
        p = CACHE / f"{d}.parquet"
        if p.exists():
            md = pq.read_schema(p).metadata or {}
            entries[d] = {
                "table_sha256": adsb.sha256_file(p),
                "rows": pq.read_metadata(p).num_rows,
                "source": md.get(b"source", b"").decode(),
            }
        else:
            entries[d] = {
                "reason": NO_PROD_RELEASE.get(d)
                or log.get(d, {}).get("status", "not fetched yet")
            }
    manifest = {
        "licence": adsb.NOTICE,
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "days": entries,
        "tables": sum("table_sha256" in e for e in entries.values()),
        "without_table": sum("reason" in e for e in entries.values()),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    NOTICE_FILE.write_text(NOTICE_TEXT)
    return manifest


def verify(days: list[str]) -> int:
    """Every table against the manifest; no network."""
    import pyarrow.parquet as pq

    if not MANIFEST.exists():
        print(
            f"no manifest at {MANIFEST.relative_to(REPO)}; fetch first", file=sys.stderr
        )
        return 1
    manifest = json.loads(MANIFEST.read_text())
    bad = []
    for d in days:
        e = manifest["days"].get(d)
        p = CACHE / f"{d}.parquet"
        if e is None:
            bad.append(f"{d}: not in the manifest")
        elif "reason" in e:
            if p.exists():
                bad.append(f"{d}: a table exists where the manifest records none")
        elif not p.exists():
            bad.append(f"{d}: table missing")
        elif adsb.sha256_file(p) != e["table_sha256"]:
            bad.append(f"{d}: sha256 differs from the manifest")
        elif pq.read_metadata(p).num_rows != e["rows"] or not adsb.has_notice(p):
            bad.append(f"{d}: rows or notice differ")
    have = sum("table_sha256" in manifest["days"].get(d, {}) for d in days)
    for b in bad:
        print("  " + b, file=sys.stderr)
    print(
        f"adsb.lol cache: {len(days)} days, {have} tables, {len(days) - have} without"
        f" a table, {len(bad)} problem(s)"
    )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("days", nargs="*", help="days as YYYY-MM-DD")
    ap.add_argument("--all", action="store_true", help="every day of section 5")
    ap.add_argument("--verify", action="store_true", help="check the cache; no network")
    ap.add_argument(
        "--manifest",
        action="store_true",
        help="write the manifest and the notice from the tables on disk; no network",
    )
    ap.add_argument("-j", "--workers", type=int, default=4, help="extraction workers")
    ap.add_argument(
        "--downloads", type=int, default=1, help="days fetched ahead while one extracts"
    )
    ap.add_argument("--inflight", type=int, default=8, help="directories queued at most")
    ap.add_argument("--keep-parts", action="store_true", help="keep the fetched parts")
    ap.add_argument("--parts-dir", default=str(PARTS), help="where parts go or are found")
    args = ap.parse_args()

    every = adsb_days()
    if args.verify:
        return verify(every)
    if args.manifest:
        manifest = write_manifest(every)
        print(
            f"{manifest['tables']} tables, {manifest['without_table']} days without one"
        )
        return verify(every)
    days = list(args.days) or (every if args.all else [])
    if not days:
        ap.error("name days, or pass --all or --verify")
    days = [d for d in days if d not in NO_PROD_RELEASE]
    failed = adsb.run_days(
        days,
        CACHE,
        Path(args.parts_dir),
        workers=args.workers,
        downloads=args.downloads,
        inflight=args.inflight,
        keep_parts=args.keep_parts,
    )
    manifest = write_manifest(every)
    print(f"{manifest['tables']} tables, {manifest['without_table']} days without one")
    if failed:
        print("failed days: " + " ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
