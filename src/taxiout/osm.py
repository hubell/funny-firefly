# SPDX-License-Identifier: GPL-3.0-only
"""Overpass client for airport movement-area geometry.

Design notes
------------
Overpass mirrors fail transiently and often (observed: 406, 502, 504, read
timeouts within a single minute). Every fetch therefore rotates mirrors with
exponential backoff, and every successful response is cached to disk so the
extraction is reproducible and reruns cost nothing. Cache files are the unit of
provenance: delete one to refetch that airport.

Data source: OpenStreetMap, (c) OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import NamedTuple

from .airports import Airport

MIRRORS: tuple[str, ...] = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)

USER_AGENT = "prc-dc2026-taxiout/0.1 (PRC Data Challenge 2026 entry; OSM ODbL)"

# aeroway features that make up the surface movement area.
# `apron` is deliberately excluded: the polygons are large, slow to serve, and
# contribute nothing to stand->runway routing. Add it only if apron-area
# features earn their place later.
AEROWAY_KINDS: tuple[str, ...] = (
    "parking_position",
    "taxiway",
    "taxilane",
    "runway",
    "holding_position",
)

# Overpass free tier allows ~2 concurrent slots per IP and throttles hard.
# A short pause between airports is far cheaper than absorbing 429 backoffs.
POLITE_DELAY_S = 5.0


def build_query(ap: Airport, timeout_s: int = 60, date: str | None = None) -> str:
    """Overpass QL for every movement-area feature around an airport.

    ``date`` pins the query to a point in OSM's history (an "attic" query),
    e.g. ``"2026-01-01T00:00:00Z"``. Without it the answer is whatever OSM says
    on the day it is run, which makes the geometry unreproducible after the
    fact and mildly anachronistic against 2025 movements. It is optional rather
    than default because not every Overpass mirror serves attic queries, and a
    pinned query that only one mirror will answer is a worse failure mode than
    an unpinned one -- see ``fetch_airport``, which reports which it used.
    """
    parts = []
    for kind in AEROWAY_KINDS:
        for elem in ("node", "way"):
            parts.append(
                f'{elem}["aeroway"="{kind}"](around:{ap.radius_m},{ap.lat},{ap.lon});'
            )
    body = "\n  ".join(parts)
    stamp = f'[date:"{date}"]' if date else ""
    return f"[out:json][timeout:{timeout_s}]{stamp};\n(\n  {body}\n);\nout body geom;"


def cache_path(cache_dir: Path, icao: str) -> Path:
    return cache_dir / f"{icao}.json.gz"


def cached_osm_base(cache_dir: Path, icao: str) -> str | None:
    """The ``osm3s.timestamp_osm_base`` one cached airport's answer carries.

    Overpass returns the moment of the last edit the answering instance had
    ingested when it answered. **It is per answer and not per run, and it is
    never the pin.** The eleven airports here were fetched sequentially and
    unpinned on 1 September 2026, so their files carry seven different moments,
    and those moments are the pins; but every answer is stamped with the
    instance's own current state whatever ``[date:]`` asked for, so a file
    re-fetched with a pin carries the day of the re-fetch instead.

    **That is why nothing pins from this.** The pins are
    ``taxiout.provenance.OSM_PINS``, and this is read only to check that
    register against the original cache while one exists.

    ``None`` if the file is absent or carries no such stamp.
    """
    import gzip
    import json

    path = cache_path(cache_dir, icao)
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        payload = json.load(fh)
    stamp = payload.get("osm3s", {}).get("timestamp_osm_base")
    return stamp or None


def _post(endpoint: str, query: str, timeout: int) -> dict:
    req = urllib.request.Request(
        endpoint,
        data=("data=" + urllib.parse.quote(query)).encode(),
        headers={
            "User-Agent": USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def fetch_airport(
    ap: Airport,
    cache_dir: Path,
    *,
    force: bool = False,
    attempts_per_mirror: int = 2,
    timeout: int = 60,
    date: str | None = None,
    mirrors: tuple[str, ...] = MIRRORS,
) -> tuple[dict, bool]:
    """Return ``(overpass_json, from_cache)`` for one airport.

    ``date`` pins the query to a moment in OSM's history; see ``build_query``.
    The pinned date, if any, is recorded inside the cached payload so a matrix
    built from it can say what it was built from, and so is the mirror that
    answered. That record is the only one of which instance served an answer
    and what it was asked: the response's own header names the software and the
    instance's current database state, and never the pin.

    ``mirrors`` narrows the rotation. The control probe in
    ``scripts/fetch_osm.py`` asks only the instance that answered the fetch it
    checks, because a control answered by a different instance proves nothing
    about the first.

    Raises ``RuntimeError`` only after every mirror has been tried.
    """
    path = cache_path(cache_dir, ap.icao)
    if path.exists() and not force:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh), True

    query = build_query(ap, date=date)
    errors: list[str] = []
    for round_idx in range(attempts_per_mirror):
        for mirror in mirrors:
            try:
                payload = _post(mirror, query, timeout)
            except urllib.error.HTTPError as exc:
                # 429/504 mean "come back later", not "this mirror is broken".
                errors.append(f"{mirror}: HTTP {exc.code}")
                time.sleep(12 if exc.code == 429 else 3)
                continue
            except Exception as exc:  # noqa: BLE001 - mirrors fail many ways
                errors.append(f"{mirror}: {type(exc).__name__}: {exc}")
                time.sleep(2 + 3 * round_idx)
                continue
            if not payload.get("elements"):
                errors.append(f"{mirror}: empty element list")
                time.sleep(2)
                continue
            payload["taxiout:fetch"] = {
                "mirror": mirror,
                "date_pin": date,
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            return payload, False

    raise RuntimeError(
        f"all Overpass mirrors failed for {ap.icao}:\n  " + "\n  ".join(errors)
    )


def read_payload(path: Path) -> dict:
    """One stored Overpass answer, parsed."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


class ElementComparison(NamedTuple):
    """Two Overpass answers compared element by element, and not byte by byte.

    ``fetched`` and ``reference`` count each side's elements by type.
    ``differing`` is the number of elements on both sides with any field
    unequal, and ``fields`` says which fields and how often. ``duplicates``
    counts a ``(type, id)`` repeated on either side, which fails ``identical``
    because a lookup by key would hide the repeat.
    """

    fetched: dict[str, int]
    reference: dict[str, int]
    only_fetched: int
    only_reference: int
    in_both: int
    differing: int
    fields: dict[str, int]
    duplicates: int
    same_order: bool

    @property
    def same_elements(self) -> bool:
        """No element on one side only, and none on both that differs."""
        return self.only_fetched == 0 and self.only_reference == 0 and self.differing == 0

    @property
    def identical(self) -> bool:
        """The same elements, every field equal, in the same order."""
        return self.same_elements and self.duplicates == 0 and self.same_order


def compare_elements(fetched: dict, reference: dict) -> ElementComparison:
    """Compare the parsed elements of two Overpass answers.

    **The elements and not the payload, because the header differs between any
    two answers.** ``osm3s.timestamp_osm_base`` is the answering instance's own
    database state on the day it answered, pinned or not, and the client adds a
    record of the fetch; two answers carrying the same geometry therefore never
    match byte for byte. Elements are matched on ``(type, id)`` and compared on
    every field, so geometry, node lists and tags all count.
    """
    ours = fetched.get("elements", [])
    theirs = reference.get("elements", [])
    our_keys = [(e.get("type"), e.get("id")) for e in ours]
    their_keys = [(e.get("type"), e.get("id")) for e in theirs]
    by_ours = dict(zip(our_keys, ours))
    by_theirs = dict(zip(their_keys, theirs))
    shared = by_ours.keys() & by_theirs.keys()
    fields: Counter = Counter()
    differing = 0
    for key in shared:
        a, b = by_ours[key], by_theirs[key]
        if a != b:
            differing += 1
            fields.update(f for f in a.keys() | b.keys() if a.get(f) != b.get(f))
    return ElementComparison(
        fetched=dict(Counter(kind for kind, _ in our_keys)),
        reference=dict(Counter(kind for kind, _ in their_keys)),
        only_fetched=len(by_ours.keys() - by_theirs.keys()),
        only_reference=len(by_theirs.keys() - by_ours.keys()),
        in_both=len(shared),
        differing=differing,
        fields=dict(sorted(fields.items())),
        duplicates=len(our_keys) - len(by_ours) + len(their_keys) - len(by_theirs),
        same_order=our_keys == their_keys,
    )
