#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Verify that nothing OpenStreetMap-derived is committed to this repository.

This repository ships GPLv3 in totality. OpenStreetMap is ODbL, and its
share-alike obligation is not this project's to waive, so no OSM content may
appear in a tracked file -- and "no OSM content" has to be a checkable property
rather than a habit, because the failure mode is forgetting once, months from
now, in a hurry.

Four checks:

1. **No non-redistributable artefact is tracked.** Two classes, for two
   different reasons. OSM-derived artefacts -- the distance matrices, the raw
   Overpass cache, any feature frame built from them -- are Derivative
   Databases carrying ODbL. Challenge data is supplied under terms that permit
   no redistribution whatsoever, which makes it the stricter case and the one
   where a mistake cannot be undone by relicensing.

2. **No list of literal stand identifiers appears in prose.** Stand
   identifiers are the ones that matter: unlike runway designators, which are
   published in AIPs and in the public-domain OurAirports data and so are
   nobody's to restrict, a stand identifier at these airports comes from OSM
   or from the challenge data, and neither may be excerpted here. Reports
   describe identifier *shapes* instead; a shape is a description, an
   identifier is an extract. The challenge side is checked only where the
   data is present locally, which CI is not.

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

4. **No assistant attribution anywhere.** The repository carries no trace of
   what wrote it -- no tool name in a file, a comment, a docstring or a commit
   message, and no co-authorship trailer naming one. This is a property of the
   published artefact, so it is checked over both the tree and the recent
   history rather than trusted to habit.

   The pattern is spelled with bracket classes -- ``cla[u]de`` rather than the
   word itself -- for one reason: a check that names what it forbids matches
   its own source, and every scanner then has to exclude itself by name. Three
   commit scripts learned that the hard way, each having to list the others,
   and one of them failed a run purely because a newer script existed. Written
   this way the check is inert against its own text, so nothing needs an
   exclusion and nothing is silently exempt.

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
# Two distinct reasons, both fatal. The OSM-derived artefacts carry ODbL, which
# this GPLv3-in-totality repository cannot host. The challenge data carries the
# challenge's own terms, which permit no redistribution at all -- so it is the
# stricter of the two, and the one where a mistake is unrecoverable.
FORBIDDEN = (
    # OpenStreetMap-derived: ODbL
    "data/raw/*",
    "data/interim/*",
    "data/processed/taxi_distance/*.csv",
    "*.osm",
    "*.pbf",
    # Challenge data and anything built from it: not redistributable
    "data/challenge/*.parquet",
    "data/challenge/**/*.parquet",
    "challenge_data/*",
    "**/*.parquet",
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

# Everything a reader of the published repository would see. Prose and source
# both, because an attribution is as likely to sit in a docstring as in a
# report.
AUTHORED = PROSE + ("*.py", "*.sh", "*.toml", "*.yml", "*.yaml", "Makefile")

# Spelled in bracket classes so this file does not match its own pattern; the
# docstring says why that matters. Each class holds one letter of the word it
# hides, so the expression is the word and nothing else.
ATTRIBUTION_PATTERN = (
    r"co[w]ork|cla[u]de|anthro[p]ic|copi[l]ot|chat[g]pt|open[a]i|co-authored-b[y]"
)

# How much history to scan. Deep enough to cover a batch of commits made by
# one script, shallow enough to stay fast on every run.
HISTORY_DEPTH = 40

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


def challenge_stand_identifiers() -> set[str]:
    """Distinctive ``STAND_mvt`` values from the local challenge data, if any.

    The challenge data is not redistributable at all, so its identifiers are
    held to the same rule as OSM's. Reads only the one column, and only when
    the data and ``pyarrow`` are both present; CI has neither and skips.
    """
    files = sorted((REPO / "data" / "challenge").rglob("*.parquet"))
    if not files:
        return set()
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return set()
    out: set[str] = set()
    for path in files:
        if "STAND_mvt" not in pq.read_schema(path).names:
            continue
        column = pq.read_table(path, columns=["STAND_mvt"])["STAND_mvt"]
        for value in column.unique().to_pylist():
            if not value:
                continue
            value = str(value).strip()
            if (
                len(value) >= 3
                and re.search(r"[A-Za-z]", value)
                and re.search(r"[0-9]", value)
            ):
                out.add(value)
    return out


def attribution_hits(files: list[str]) -> dict[str, list[str]]:
    """Assistant-attribution matches in tracked files and in recent history.

    Returns a mapping of where to what was found; empty is clean. History is
    read with ``git log``, so a message that names a tool is caught even
    though no file does.
    """
    rx = re.compile(ATTRIBUTION_PATTERN, re.IGNORECASE)
    hits: dict[str, list[str]] = {}
    for f in files:
        if not any(Path(f).match(pat) for pat in AUTHORED):
            continue
        try:
            text = (REPO / f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        found = sorted({m.group(0).lower() for m in rx.finditer(text)})
        if found:
            hits[f] = found
    try:
        proc = subprocess.run(
            ["git", "log", f"-{HISTORY_DEPTH}", "--format=%B%n%an <%ae>%n%cn <%ce>"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return hits
    if proc.returncode == 0:
        found = sorted({m.group(0).lower() for m in rx.finditer(proc.stdout)})
        if found:
            hits[f"the last {HISTORY_DEPTH} commit messages"] = found
    return hits


def main() -> int:
    files = tracked_files()
    failures: list[str] = []
    print(f"checking {len(files)} tracked files\n")

    # 1 -- forbidden artefacts
    bad = sorted({f for f in files for pat in FORBIDDEN if Path(f).match(pat)})
    if bad:
        failures.append(
            "non-redistributable artefacts are tracked:\n    " + "\n    ".join(bad)
        )
    else:
        print("  ok  no OSM-derived or challenge data is tracked")

    # 2 -- literal identifiers, from either non-redistributable source
    osm_ids = osm_stand_identifiers()
    challenge_ids = challenge_stand_identifiers()
    ids = osm_ids | challenge_ids
    if not ids:
        print("  --  no OSM cache or challenge data present; identifier check skipped")
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
                "literal stand identifiers appear in tracked files:\n    "
                + "\n    ".join(f"{f}: {v}" for f, v in hits.items())
            )
        else:
            sources = []
            if osm_ids:
                sources.append(f"{len(osm_ids)} OSM")
            if challenge_ids:
                sources.append(f"{len(challenge_ids)} challenge")
            print(
                "  ok  none of "
                + " and ".join(sources)
                + " stand identifiers appears in a tracked file"
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

    # 4 -- assistant attribution, in the tree and in the history
    attribution = attribution_hits(files)
    if attribution:
        failures.append(
            "assistant attribution appears in the repository:\n    "
            + "\n    ".join(f"{k}: {v}" for k, v in attribution.items())
        )
    else:
        print(
            f"  ok  no assistant attribution in {len(files)} tracked files "
            f"or the last {HISTORY_DEPTH} commit messages"
        )

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
