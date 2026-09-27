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
from collections.abc import Callable
from datetime import datetime, timezone
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


def _utc(stamp: str) -> datetime:
    """An Overpass timestamp, ``2026-09-01T13:32:15Z``, as an aware UTC moment."""
    moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def stamp_precedes_pin(stamp: str | None, pin: str) -> bool:
    """Whether an answer stamped ``stamp`` cannot have been served at ``pin``.

    **The header cannot show that a pin was applied, but it can show that a
    pin could not have been**: ``osm3s.timestamp_osm_base`` is the database
    state the answering instance had ingested, and an instance cannot serve a
    state it has not ingested, so a stamp earlier than the pin proves the
    answer is not the pinned one. On 26 September 2026 both fallback mirrors
    held databases from May, and one of them answered LSZH's September pin
    with LSZH as of May. An answer carrying no stamp shows nothing either way
    and is read the same, since a pinned answer has to show it was possible.

    Compared as moments and never as strings, so an offset written another way
    cannot reorder them.
    """
    return not stamp or _utc(stamp) < _utc(pin)


def fetch_airport(
    ap: Airport,
    cache_dir: Path,
    *,
    force: bool = False,
    attempts_per_mirror: int = 2,
    timeout: int = 60,
    date: str | None = None,
    mirrors: tuple[str, ...] = MIRRORS,
    log: Callable[[str], None] | None = None,
) -> tuple[dict, bool]:
    """Return ``(overpass_json, from_cache)`` for one airport.

    ``date`` pins the query to a moment in OSM's history; see ``build_query``.
    The pinned date, if any, is recorded inside the cached payload so a matrix
    built from it can say what it was built from, and so are the mirror that
    answered and the database state it answered from. That record is the only
    one of which instance served an answer and what it was asked: the
    response's own header names the software and the instance's current
    database state, and never the pin.

    **A pinned answer from a database older than its pin is refused**, by
    ``stamp_precedes_pin``: it is not written, the mirror is set aside for the
    rest of this call as unable to serve the pin, and the rotation moves on. A
    refused answer is never a fetched one. With the fallbacks set aside the
    primary may be the only mirror left, and the free tier throttles it, so a
    pinned call makes as many rounds as the whole rotation has asks, and the
    primary alone can be asked as often as all of them would have been. When
    every mirror has failed or been refused the error names each mirror, and
    each refusal's stamp, and says the pin cannot be served today: a pinned
    rebuild stops there rather than falls back to an answer it cannot vouch
    for.

    ``mirrors`` narrows the rotation. The control probe in
    ``scripts/fetch_osm.py`` asks only the instance that answered the fetch it
    checks, because a control answered by a different instance proves nothing
    about the first. ``log`` receives one line for every failed or refused
    attempt, so a run can say which mirror failed how.

    Raises ``RuntimeError`` only after every mirror has been tried.
    """
    path = cache_path(cache_dir, ap.icao)
    if path.exists() and not force:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return json.load(fh), True

    say = log or (lambda _line: None)
    query = build_query(ap, date=date)
    rounds = attempts_per_mirror * (len(mirrors) if date else 1)
    errors: dict[str, list[str]] = {m: [] for m in mirrors}
    refused: dict[str, str | None] = {}
    for round_idx in range(rounds):
        for mirror in mirrors:
            if mirror in refused:
                continue
            try:
                payload = _post(mirror, query, timeout)
            except urllib.error.HTTPError as exc:
                # 429/504 mean "come back later", not "this mirror is broken".
                errors[mirror].append(f"HTTP {exc.code}")
                say(f"{mirror}: HTTP {exc.code}")
                time.sleep(12 if exc.code == 429 else 3)
                continue
            except Exception as exc:  # noqa: BLE001 - mirrors fail many ways
                errors[mirror].append(f"{type(exc).__name__}: {exc}")
                say(f"{mirror}: {type(exc).__name__}: {exc}")
                time.sleep(2 + 3 * round_idx)
                continue
            stamp = (payload.get("osm3s") or {}).get("timestamp_osm_base")
            if date and stamp_precedes_pin(stamp, date):
                refused[mirror] = stamp
                say(
                    f"{mirror}: REFUSED, its database is at {stamp or 'no stamp'},"
                    f" before the pin {date}, so it cannot serve it; not asked again"
                )
                continue
            if not payload.get("elements"):
                errors[mirror].append("empty element list")
                say(f"{mirror}: empty element list")
                time.sleep(2)
                continue
            payload["taxiout:fetch"] = {
                "mirror": mirror,
                "date_pin": date,
                "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "timestamp_osm_base": stamp,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
            return payload, False
        if len(refused) == len(mirrors):
            break

    lines = [
        f"{m}: refused, its database is at {refused[m] or 'no stamp'}, before the pin"
        if m in refused
        else f"{m}: " + ("; ".join(errors[m]) or "not asked")
        for m in mirrors
    ]
    if date:
        raise RuntimeError(
            f"no Overpass mirror served {ap.icao} at its pin {date}:\n  "
            + "\n  ".join(lines)
            + "\nthe pin cannot be served today, and a pinned fetch stops here"
            " rather than fall back"
        )
    raise RuntimeError(
        f"all Overpass mirrors failed for {ap.icao}:\n  " + "\n  ".join(lines)
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


# The record a fetch leaves beside the cache. **A cached answer cannot say what
# it was pinned to**: Overpass stamps every answer with the answering instance's
# own database state and never with the ``[date:]`` pin, so after a pinned
# rebuild every header carries the day of the rebuild. This file carries what
# the header cannot, one entry per airport, and ``pin_agreement`` reads it.
PINS_SIDECAR = "PINS.json"

PINS_SIDECAR_ABOUT = (
    "Written by scripts/fetch_osm.py beside the cached Overpass answers: for each"
    " airport, the pin its cached answer was fetched at, the instance that"
    " answered, the time of the fetch and the database state that instance"
    " answered from, its osm3s.timestamp_osm_base, copied from the answer's own"
    " fetch record. The answer's header cannot carry the pin, because Overpass"
    " stamps every answer with its own current database state; but a state"
    " earlier than the pin shows the pin cannot have been served, and"
    " taxiout.osm.pin_agreement fails such an entry. A pin of null is an"
    " unpinned fetch."
)

# What an entry records, and what an answer's own fetch record must equal.
SIDECAR_FIELDS: tuple[str, ...] = (
    "date_pin",
    "mirror",
    "fetched_utc",
    "timestamp_osm_base",
)


def sidecar_path(cache_dir: Path) -> Path:
    return cache_dir / PINS_SIDECAR


def read_sidecar(cache_dir: Path) -> dict[str, dict]:
    """The sidecar's entries by airport, or ``{}`` where there is no sidecar."""
    path = sidecar_path(cache_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))["airports"]


def record_fetch(cache_dir: Path, icao: str, payload: dict) -> None:
    """Record one fetched answer's pin, instance, time and stamp in the sidecar.

    Copied from the answer's own ``taxiout:fetch`` record, which
    ``fetch_airport`` writes into the payload it caches, so an entry and the
    file it describes agree for as long as no other fetch has replaced the
    file. **Every fetch is recorded, pinned or not**, so an entry never outlives
    the answer it describes: an unpinned fetch replaces a pinned entry with a
    null pin rather than leaving it to vouch for a file it no longer describes.
    Written through a temporary file and a rename, so an interrupted rebuild
    leaves the entries it finished and never a half-written file.
    """
    rec = payload["taxiout:fetch"]
    entries = read_sidecar(cache_dir)
    entries[icao] = {k: rec.get(k) for k in SIDECAR_FIELDS}
    body = {"about": PINS_SIDECAR_ABOUT, "airports": dict(sorted(entries.items()))}
    path = sidecar_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


class PinCheck(NamedTuple):
    """One cached airport's answer read against its pin, by the evidence it has.

    ``branch`` is ``"original"`` for an answer with no sidecar entry, read by its
    header, and ``"rebuilt"`` for one with an entry, read by the sidecar.
    ``problem`` is ``None`` where the evidence agrees with the pin.
    """

    icao: str
    branch: str
    problem: str | None


def pin_agreement(cache_dir: Path, pins: dict[str, str]) -> list[PinCheck]:
    """Each cached airport's answer against its pin, by whichever evidence it has.

    **Original**: an answer fetched unpinned on 1 September has no sidecar
    entry, and its header stamp is the state it was served from, so the stamp
    must equal the pin. **Rebuilt**: an answer with a sidecar entry carries in
    its header the state of whichever instance answered, which cannot show the
    pin was applied, so the entry's pin must equal the register's, and the
    answer's own fetch record must equal the entry, which is what shows the
    entry describes this file and not one fetched before it. **And the entry's
    stamp must not precede its pin**, because a stamp that does proves the
    answer cannot have been served at the pin, whatever was requested: the
    request is what the entry records, and the stamp is what shows the request
    could have been met. An airport with no cached answer is not read, so an
    empty list means there is no cache here.
    """
    sidecar = read_sidecar(cache_dir)
    out = []
    for icao, pin in pins.items():
        path = cache_path(cache_dir, icao)
        if not path.exists():
            continue
        entry = sidecar.get(icao)
        if entry is None:
            stamp = cached_osm_base(cache_dir, icao)
            problem = (
                None
                if stamp == pin
                else f"its header is stamped {stamp} and its pin is {pin}, and no"
                " sidecar entry says it was fetched at the pin"
            )
            out.append(PinCheck(icao, "original", problem))
            continue
        rec = read_payload(path).get("taxiout:fetch") or {}
        own = {k: rec.get(k) for k in SIDECAR_FIELDS}
        stamp = entry.get("timestamp_osm_base")
        if entry.get("date_pin") != pin:
            problem = (
                f"the sidecar records it fetched at {entry.get('date_pin')} and its"
                f" pin is {pin}"
            )
        elif own != {k: entry.get(k) for k in SIDECAR_FIELDS}:
            problem = (
                "its own fetch record disagrees with its sidecar entry, so the entry"
                " describes another fetch than the one that wrote the file"
            )
        elif not stamp:
            problem = (
                "the sidecar records no database state for its answer, so nothing"
                f" shows the answer can have been served at its pin {pin}"
            )
        elif stamp_precedes_pin(stamp, pin):
            problem = (
                f"its answer came from a database at {stamp}, before its pin {pin},"
                " so the answer cannot have been served at the pin"
            )
        else:
            problem = None
        out.append(PinCheck(icao, "rebuilt", problem))
    return out
