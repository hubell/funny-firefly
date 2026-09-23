#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Cross-check OSM-derived runway geometry against OurAirports.

Two independent sources agreeing is worth more than one source checked against
itself. OSM gives runway ways tagged with a ``ref``; OurAirports gives runway
records with both designators, a length, and thresholds in decimal degrees.
Where they disagree, something is wrong and the report should say so rather
than the pipeline routing to it quietly.

The check that motivated this: Istanbul's 16L/34R sits 19 degrees off the
nominal heading of a 16, which looks exactly like a mis-merged runway. It is
not -- with five near-parallel strips and only three position letters, the
designator number is offset instead, and OurAirports lists the same six
designators with lengths agreeing to within 1%. A single source could not have
settled that.

OurAirports data is released to the public domain and rebuilt nightly.
Source: https://ourairports.com/data/

Usage
-----
    python scripts/check_runways_ourairports.py
    python scripts/check_runways_ourairports.py --refresh
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS  # noqa: E402
from taxiout.geo import angle_diff_deg, bearing_deg, geodesic_m  # noqa: E402
from taxiout.osm import USER_AGENT, cache_path  # noqa: E402
from taxiout.provenance import report_inputs_footer  # noqa: E402
from taxiout.runways import build_runways, parse_designators  # noqa: E402

SOURCE = "https://davidmegginson.github.io/ourairports-data/runways.csv"
CACHE = REPO / "data" / "raw" / "ourairports_runways.csv"
OUT = REPO / "reports" / "runway_crosscheck.md"

# Tolerances. Length: OSM ways are traced to the marked ends, OurAirports
# publishes the declared distance, so a few percent is normal and 10% is not.
LENGTH_TOL = 0.10
# Heading: magnetic variation across these airports reaches about 7 degrees,
# and designators quantise to 10, so 15 is generous without being useless.
HEADING_TOL_DEG = 15.0


def fetch(refresh: bool = False) -> list[dict]:
    if refresh or not CACHE.exists():
        req = urllib.request.Request(SOURCE, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=60) as resp:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            CACHE.write_bytes(resp.read())
    with CACHE.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="refetch the CSV")
    args = parser.parse_args()

    try:
        rows = fetch(args.refresh)
    except Exception as exc:  # noqa: BLE001 - one network call, many failure modes
        print(f"could not fetch OurAirports data: {exc}")
        return 1

    by_airport: dict[str, list[dict]] = {}
    for row in rows:
        if row.get("closed") == "1":
            continue
        by_airport.setdefault(row.get("airport_ident", ""), []).append(row)

    lines: list[str] = []
    problems = 0
    checked = 0

    for ap in AIRPORTS:
        path = cache_path(REPO / "data" / "raw", ap.icao)
        if not path.exists():
            continue
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            elements = json.load(fh)["elements"]
        osm_runways, _ = build_runways(ap.icao, elements)
        reference = by_airport.get(ap.icao, [])

        osm_designators = {d for r in osm_runways for d in r.designators}
        ref_designators = {
            d
            for row in reference
            for d in parse_designators(row.get("le_ident"))
            + parse_designators(row.get("he_ident"))
        }

        notes: list[str] = []
        only_osm = sorted(osm_designators - ref_designators)
        only_ref = sorted(ref_designators - osm_designators)
        if only_osm:
            notes.append(f"in OSM only: {', '.join(only_osm)}")

        # A designator present only in OurAirports whose reciprocal *is* in OSM
        # is the expected case, not a disagreement: OurAirports lists both ends
        # of every runway as a matter of course, while OSM's ref names only the
        # directions in use. Frankfurt's 18 is departure-only southbound and has
        # no operational 36. Classified rather than described, so the rule holds
        # wherever else it applies.
        expected, unexpected = [], []
        for d in only_ref:
            number, side = int(d[:2]), d[2:]
            reciprocal = f"{((number + 17) % 36) + 1:02d}{side}"
            (expected if reciprocal in osm_designators else unexpected).append(d)
        if expected:
            notes.append(
                f"in OurAirports only: {', '.join(expected)} "
                "(reciprocal of an OSM designator; not in operational use)"
            )
        if unexpected:
            notes.append(f"in OurAirports only: {', '.join(unexpected)}")

        for rw in osm_runways:
            match = next(
                (
                    row
                    for row in reference
                    if set(parse_designators(row.get("le_ident")))
                    | set(parse_designators(row.get("he_ident")))
                    == set(rw.designators)
                ),
                None,
            )
            if match is None:
                continue
            checked += 1

            length_ft = _f(match, "length_ft")
            if length_ft:
                published = length_ft * 0.3048
                delta = abs(rw.length_m - published) / published
                if delta > LENGTH_TOL:
                    notes.append(
                        f"{rw.ref}: OSM {rw.length_m:.0f} m vs published "
                        f"{published:.0f} m ({delta:.0%})"
                    )

            # Threshold positions, where OurAirports carries them.
            le_lat, le_lon = _f(match, "le_latitude_deg"), _f(match, "le_longitude_deg")
            if le_lat is not None and le_lon is not None:
                le = parse_designators(match.get("le_ident"))
                if le:
                    gap = geodesic_m(rw.threshold(le[0]), (le_lat, le_lon))
                    if gap > 300.0:
                        notes.append(
                            f"{rw.ref}: {le[0]} threshold {gap:.0f} m from the "
                            "published position"
                        )

            # Heading, in degrees true on both sides once OSM is measured.
            for designator in rw.designators:
                ident_key = (
                    "le_heading_degT"
                    if designator in parse_designators(match.get("le_ident"))
                    else "he_heading_degT"
                )
                published_hdg = _f(match, ident_key)
                if published_hdg is None:
                    continue
                thr = rw.threshold(designator)
                other = rw.ends[1] if thr == rw.ends[0] else rw.ends[0]
                delta = angle_diff_deg(bearing_deg(thr, other), published_hdg)
                if delta > HEADING_TOL_DEG:
                    notes.append(
                        f"{rw.ref}: {designator} axis {delta:.0f} deg from the "
                        f"published true heading"
                    )

        material = [n for n in notes if "not in operational use" not in n]
        problems += len(material)
        status = "OK" if not material else "CHECK"
        lines.append(
            f"| {ap.icao} | {len(osm_runways)} | {len(reference)} | "
            f"{status} | {'; '.join(notes) if notes else '—'} |"
        )

    body = [
        "# Runway cross-check: OpenStreetMap vs OurAirports",
        "",
        "Generated by `scripts/check_runways_ourairports.py`. Two independent",
        "sources for the same geometry; disagreement is the signal.",
        "",
        f"Tolerances: length {LENGTH_TOL:.0%}, true heading {HEADING_TOL_DEG:.0f} deg,",
        "threshold position 300 m. Runways marked closed in OurAirports are skipped.",
        "",
        "| ICAO | OSM runways | OurAirports runways | status | notes |",
        "| ---- | ----------: | ------------------: | ------ | ----- |",
        *lines,
        "",
        f"{checked} runways matched by designator set; {problems} discrepancies.",
        "",
        "Where the two runway counts differ, the extras are helipad idents such",
        "as `08H/26H`, which `parse_designators` rejects because they are not",
        "runway designators. That is intended: routing a departure to a helipad",
        "would be worse than not routing it.",
        "",
        "",
        *report_inputs_footer(
            REPO,
            [CACHE] + [cache_path(REPO / "data" / "raw", ap.icao) for ap in AIRPORTS],
        ),
        "",
        "OSM data (c) OpenStreetMap contributors, ODbL 1.0.",
        "OurAirports data released to the public domain.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(body) + "\n", encoding="utf-8")
    print("\n".join(body))
    print(f"\nwritten: {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
