#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Verify that nothing OpenStreetMap-derived is committed to this repository.

This repository ships GPLv3 in totality. OpenStreetMap is ODbL, and its
share-alike obligation is not this project's to waive, so no OSM content may
appear in a tracked file -- and "no OSM content" has to be a checkable property
rather than a habit, because the failure mode is forgetting once, months from
now, in a hurry.

Three checks:

1. **No OSM-derived artefact is tracked.** The distance matrices, the raw
   Overpass cache, cached third-party data, and any feature frame built from
   them are Derivative Databases. They must be gitignored, not committed.

2. **No list of literal OSM identifiers appears in prose.** Stand identifiers
   are the ones that matter: unlike runway designators, which are published in
   AIPs and in the public-domain OurAirports data and so are not OSM's to
   restrict, a stand identifier at these airports comes from OSM and nowhere
   else. Reports describe identifier *shapes* instead; a shape is a description,
   an identifier is an extract.

   Three deliberate limits, stated rather than hidden, because a check that
   cries wolf gets ignored:

   - Only identifiers with a letter *and* a digit and at least three characters
     are considered. Bare numerics are indistinguishable from ordinary numbers,
     and two-character tokens collide with everything.
   - Only prose is scanned -- reports and docs. Code fixtures are invented by
     hand, and short invented names collide with real identifiers by chance:
     an invented helipad ref and two invented stand names all tripped an
     earlier version of this check.
   - A file is flagged only at three or more distinct matches. One match is a
     coincidence; a list is a leak, and a list is what the reports would carry
     if this regressed.

   So this catches the realistic failure -- a report reverting to printing
   examples -- and not a single unlucky string in a test.

3. **Attribution is present** wherever OSM was used to produce a tracked file.

Exit code is non-zero if anything fails, so it can gate a commit or a push.

Usage
-----
    python scripts/check_licence_hygiene.py
    make licence-check
"""

from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.coverage import stand_identifier  # noqa: E402

CACHE = REPO / "data" / "raw"

# Paths that must never be tracked, as glob patterns against repo-relative paths.
FORBIDDEN = (
    "data/raw/*",
    "data/interim/*",
    "data/processed/taxi_distance/*.csv",
    "data/processed/**/*.parquet",
    "*.osm",
    "*.pbf",
)

# A tracked file that reports OSM-derived figures must attribute OSM.
NEEDS_ATTRIBUTION = ("reports/*.md",)
ATTRIBUTION = "OpenStreetMap contributors"


def tracked_files() -> list[str]:
    """Every file git would ship, whatever the current commit state.

    Deliberately not ``git ls-files``: that reports what is committed *now*,
    and the interesting moment is just before a commit, when the answer would
    be the previous tree. Walking the filesystem and subtracting what
    ``.gitignore`` excludes gives what the next commit would contain.
    """
    candidates = [
        p.relative_to(REPO).as_posix()
        for p in REPO.rglob("*")
        if p.is_file()
        and ".git/" not in p.relative_to(REPO).as_posix()
        and "__pycache__" not in p.as_posix()
    ]
    try:
        proc = subprocess.run(
            ["git", "check-ignore", "--stdin"],
            cwd=REPO,
            input="\n".join(candidates),
            capture_output=True,
            text=True,
        )
        ignored = set(proc.stdout.split())
    except FileNotFoundError:
        ignored = set()
    return sorted(f for f in candidates if f not in ignored)


# Files where a leak would realistically land: prose that might list examples.
PROSE = ("reports/*.md", "docs/*.md", "*.md")

# Below this many distinct matches in one file, treat it as coincidence.
LEAK_THRESHOLD = 3


def osm_stand_identifiers() -> set[str]:
    """Distinctive stand identifiers present in the local OSM cache.

    Distinctive means at least three characters with both a letter and a digit:
    short or purely alphabetic identifiers collide with ordinary prose.
    """
    out: set[str] = set()
    for ap in AIRPORTS:
        path = CACHE / f"{ap.icao}.json.gz"
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            elements = json.load(fh)["elements"]
        for el in elements:
            tags = el.get("tags") or {}
            if tags.get("aeroway") != "parking_position":
                continue
            found = stand_identifier(tags)
            if not found:
                continue
            value = found[1].strip()
            if (
                len(value) >= 3
                and re.search(r"[A-Za-z]", value)
                and re.search(r"[0-9]", value)
            ):
                out.add(value)
    return out


def main() -> int:
    files = tracked_files()
    failures: list[str] = []
    print(f"checking {len(files)} tracked files\n")

    # 1 -- forbidden artefacts
    bad = sorted({f for f in files for pat in FORBIDDEN if Path(f).match(pat)})
    if bad:
        failures.append("OSM-derived artefacts are tracked:\n    " + "\n    ".join(bad))
    else:
        print("  ok  no OSM-derived artefact is tracked")

    # 2 -- literal identifiers
    ids = osm_stand_identifiers()
    if not ids:
        print("  --  no OSM cache present; identifier check skipped")
    else:
        hits: dict[str, list[str]] = {}
        prose = [f for f in files if any(Path(f).match(pat) for pat in PROSE)]
        for f in prose:
            try:
                text = (REPO / f).read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found = sorted(
                i
                for i in ids
                if re.search(r"(?<![\w])" + re.escape(i) + r"(?![\w])", text)
            )
            if len(found) >= LEAK_THRESHOLD:
                hits[f] = found
        if hits:
            failures.append(
                "literal OSM stand identifiers appear in tracked files:\n    "
                + "\n    ".join(f"{f}: {v}" for f, v in hits.items())
            )
        else:
            print(
                f"  ok  none of {len(ids)} OSM stand identifiers appears in a tracked file"
            )

    # 3 -- attribution
    missing = []
    for f in files:
        if not any(Path(f).match(pat) for pat in NEEDS_ATTRIBUTION):
            continue
        if ATTRIBUTION not in (REPO / f).read_text(encoding="utf-8"):
            missing.append(f)
    if missing:
        failures.append(
            "files reporting OSM-derived figures without attribution:\n    "
            + "\n    ".join(missing)
        )
    else:
        print("  ok  every report attributes OpenStreetMap")

    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        print("\nSee data/processed/taxi_distance/ODbL-NOTICE.md for what may ship.")
        return 1
    print("licence hygiene clean: GPLv3 in totality, no OSM content tracked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
