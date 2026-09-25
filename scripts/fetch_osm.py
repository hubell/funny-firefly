#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
"""Fetch and cache OSM movement-area geometry for the challenge airports.

Resumable: airports already cached are skipped unless ``--force`` is given, so
this can be run repeatedly until every airport lands.

    python scripts/fetch_osm.py                 # fetch anything missing
    python scripts/fetch_osm.py --only EDDF LTFM
    python scripts/fetch_osm.py --force         # refetch everything
    python scripts/fetch_osm.py --date 2026-01-01T00:00:00Z --force
    python scripts/fetch_osm.py --pin-from-register --force
    python scripts/fetch_osm.py --compare data/raw /elsewhere/data/raw
    python scripts/fetch_osm.py --control-date 2020-01-01T00:00:00Z --only LIRF

The ``--date`` form pins the query to a moment in OSM's history, which is what
makes a rebuild months later reproduce these numbers rather than whatever the
map says then. It is not the default because not every Overpass mirror answers
attic queries; if one refuses, the client rotates as usual and the failure is
reported rather than silently unpinned.

``--pin-from-register`` is the form that reproduces the cache the committed
matrices were built from, and it exists because one ``--date`` cannot. The
eleven airports were fetched sequentially and unpinned on 1 September 2026, so
each was answered from whatever OSM state its instance had ingested at the
moment it was served: seven distinct moments between 13:23:05Z and 13:32:15Z,
and a single date would pin eight of the eleven to a state their matrices were
not built from. ``taxiout.provenance.OSM_PINS`` records each airport's own
moment, and this mode pins each airport's query to its entry. It requires
``--force``, because without it the cache is returned unread and a run that
changed nothing would look like a pinned rebuild; and an airport the register
has no pin for is a failure for that airport rather than an unpinned fetch.

**The pins come from the register and never from the cache, because the cache
cannot keep them.** Overpass stamps every answer with the answering server's
current database time, whatever ``[date:]`` asked for, so a pinned rebuild
writes each cache file stamped with the day of the rebuild. The mode this one
replaces, ``--pin-from-cache``, read each airport's pin out of that stamp: it
would have worked exactly once, and on every run after a rebuild it would have
pinned all eleven airports to the rebuild day, with no error and a plausible
result. It is retired and refuses. A clean checkout has no cache in any case.

**Nothing in an answer's header proves that a pin was applied**, so the proof
is in two parts, and both are commands here.

- ``--compare FETCHED REFERENCE`` sends nothing. For each airport it matches
  the parsed elements in FETCHED against those in REFERENCE on ``(type, id)``
  and compares every field, never the bytes, because two answers carrying the
  same geometry differ in their headers. Each argument is one payload file or a
  directory of ``<ICAO>.json.gz``. It prints nodes, ways and relations on both
  sides, how many shared elements differ in any field, and the pin and mirror
  each answer's fetch record names, and it exits non-zero unless every airport
  is identical.
- ``--control-date`` fetches the one airport named with ``--only`` at a date
  whose answer is known to differ, from the instance that answered that
  airport's cached fetch, and compares the two. If they differ, that instance
  applies ``[date:]`` today. If they do not, it ignores the date or the airport
  has not changed since then, the pin is unproven, and the exit is non-zero. It
  writes nothing. LIRF at 2020-01-01T00:00:00Z is measured to differ: 355
  elements against 494, on 25 September 2026.

A pinned rebuild writes into this checkout's ``data/raw/``, and ``--force``
writes in place. The reference is the cache the committed matrices were built
from, so the rebuild runs in a clean checkout, which has no cache, and is
compared against the original kept elsewhere. In a checkout that has one, copy
it aside first.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from taxiout.airports import AIRPORTS, BY_ICAO, Airport  # noqa: E402
from taxiout.osm import (  # noqa: E402
    MIRRORS,
    POLITE_DELAY_S,
    ElementComparison,
    cache_path,
    compare_elements,
    fetch_airport,
    read_payload,
)
from taxiout.provenance import OSM_PINS  # noqa: E402

CACHE = REPO / "data" / "raw"

# Printed to whoever still types the retired mode. It is kept as a flag that
# refuses rather than deleted, because a deleted flag fails with "unrecognized
# arguments" and says nothing about why the recorded command stopped working.
PIN_FROM_CACHE_RETIRED = (
    "--pin-from-cache is retired. The cache header reports the Overpass"
    " server's own database time on the day of the fetch, never the pin the"
    " query asked for, so after any rebuild every cache file carries the rebuild"
    " day and this mode would pin every airport to it, with no error. Use"
    " --pin-from-register, which reads the pins from taxiout.provenance.OSM_PINS."
)


def register_pin(icao: str) -> str | None:
    """The airport's entry in ``taxiout.provenance.OSM_PINS``, or ``None``."""
    return next((p.timestamp_osm_base for p in OSM_PINS if p.icao == icao), None)


def _counts(counts: dict[str, int]) -> str:
    return "  ".join(
        f"{counts.get(kind, 0):>5} {kind}s" for kind in ("node", "way", "relation")
    )


def _record(payload: dict) -> str:
    """What an answer's own fetch record says: the pin asked and who answered."""
    rec = payload.get("taxiout:fetch")
    if not rec:
        return "no fetch record"
    return (
        f"requested at {rec.get('date_pin') or 'no pin'},"
        f" answered by {rec.get('mirror')} at {rec.get('fetched_utc')}"
    )


def _summary(c: ElementComparison, ours: str, theirs: str) -> str:
    fields = ", ".join(f"{name} {n}" for name, n in c.fields.items())
    return (
        f"{c.in_both} in both, {c.only_fetched} only {ours},"
        f" {c.only_reference} only {theirs}, {c.differing} differing in any field"
        + (f" ({fields})" if fields else "")
        + f", {c.duplicates} duplicate keys,"
        f" {'the same' if c.same_order else 'a DIFFERENT'} order"
    )


def compare(fetched: Path, reference: Path, targets: list[Airport]) -> int:
    """``--compare``: each airport's parsed elements, fetched against reference."""
    if fetched.resolve() == reference.resolve():
        print(
            "FETCHED and REFERENCE are the same path, and an answer compared with"
            " itself proves nothing"
        )
        return 1
    if fetched.is_file() and reference.is_file():
        pairs = [(fetched.name.split(".")[0], fetched, reference)]
    elif fetched.is_dir() and reference.is_dir():
        pairs = [
            (ap.icao, cache_path(fetched, ap.icao), cache_path(reference, ap.icao))
            for ap in targets
        ]
    else:
        print("FETCHED and REFERENCE must be two payload files or two directories")
        return 1

    failures = []
    for icao, ours, theirs in pairs:
        missing = [str(p) for p in (ours, theirs) if not p.exists()]
        if missing:
            print(f"  {icao}  FAILED  missing: {' '.join(missing)}", flush=True)
            failures.append(icao)
            continue
        got, want = read_payload(ours), read_payload(theirs)
        c = compare_elements(got, want)
        pin = register_pin(icao)
        asked = (got.get("taxiout:fetch") or {}).get("date_pin")
        print(f"  {icao}  {'identical' if c.identical else 'NOT IDENTICAL'}")
        print(f"        fetched    {_counts(c.fetched)}   {_record(got)}")
        print(f"        reference  {_counts(c.reference)}   {_record(want)}")
        print(f"        {_summary(c, 'fetched', 'in the reference')}")
        print(
            f"        the register's pin is {pin or 'absent'}, and the fetched"
            f" answer {'was' if pin and asked == pin else 'was NOT'} requested at it",
            flush=True,
        )
        if not c.identical:
            failures.append(icao)

    tail = f"; not identical: {' '.join(failures)}" if failures else ""
    print(
        f"\n{len(pairs)} compared, {len(pairs) - len(failures)} identical{tail}",
        flush=True,
    )
    return 1 if failures else 0


def control(ap: Airport, date: str) -> int:
    """``--control-date``: does the instance that answered apply ``[date:]`` today?"""
    if date == register_pin(ap.icao):
        print(
            f"  {ap.icao}  FAILED  {date} is this airport's own pin, and a control"
            " needs a date whose answer differs",
            flush=True,
        )
        return 1
    cached = cache_path(CACHE, ap.icao)
    if not cached.exists():
        print(
            f"  {ap.icao}  FAILED  no cached answer to compare against;"
            " run the pinned fetch first",
            flush=True,
        )
        return 1
    pinned = read_payload(cached)
    mirror = (pinned.get("taxiout:fetch") or {}).get("mirror")
    # Into a directory that is deleted on the way out, so the control can never
    # take the place of the answer it is checking.
    with tempfile.TemporaryDirectory() as scratch:
        try:
            probe, _ = fetch_airport(
                ap,
                Path(scratch),
                force=True,
                date=date,
                mirrors=(mirror,) if mirror else MIRRORS,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  {ap.icao}  FAILED  {exc}", flush=True)
            return 1

    c = compare_elements(probe, pinned)
    print(f"  {ap.icao}  control at {date}, against the cached answer")
    print(f"        cached   {_counts(c.reference)}   {_record(pinned)}")
    print(f"        control  {_counts(c.fetched)}   {_record(probe)}")
    if not mirror:
        print(
            "        the cached answer names no mirror, so the control went to the"
            " first that answered"
        )
    print(f"        {_summary(c, 'in the control', 'in the cached answer')}")
    if c.same_elements:
        print(
            f"\nTHE SAME: the instance answered {date} with the cached elements, so"
            f" either it ignores [date:] or {ap.icao} has not changed since then."
            " The pin is unproven.",
            flush=True,
        )
        return 1
    print(
        f"\nDIFFERENT: the instance answers {date} with another state, so it applies"
        " [date:] today.",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", metavar="ICAO")
    parser.add_argument("--force", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--date",
        metavar="ISO8601",
        help="pin to a moment in OSM history, e.g. 2026-01-01T00:00:00Z",
    )
    mode.add_argument(
        "--pin-from-register",
        action="store_true",
        help="pin each airport to its own entry in taxiout.provenance.OSM_PINS",
    )
    mode.add_argument(
        "--compare",
        nargs=2,
        metavar=("FETCHED", "REFERENCE"),
        help="compare each airport's parsed elements with a reference; sends nothing",
    )
    mode.add_argument(
        "--control-date",
        metavar="ISO8601",
        help="fetch the one airport named by --only at a date whose answer is known"
        " to differ, and report whether it does; writes nothing",
    )
    parser.add_argument(
        "--pin-from-cache",
        action="store_true",
        help="retired, and refuses: the cache header does not carry the pin",
    )
    args = parser.parse_args(argv)

    if args.pin_from_cache:
        parser.error(PIN_FROM_CACHE_RETIRED)
    if args.pin_from_register and not args.force:
        parser.error(
            "--pin-from-register needs --force: without it every airport is served"
            " from the cache and nothing is pinned, which is a run that looks"
            " like a pinned rebuild and is not one"
        )
    if (args.compare or args.control_date) and args.force:
        parser.error("--compare and --control-date write nothing, so --force is wrong")

    targets = [BY_ICAO[i.upper()] for i in args.only] if args.only else list(AIRPORTS)

    if args.compare:
        return compare(Path(args.compare[0]), Path(args.compare[1]), targets)
    if args.control_date:
        if len(args.only or []) != 1:
            parser.error("--control-date probes one airport; name it with --only")
        return control(targets[0], args.control_date)

    failures = []
    fetched = 0
    for ap in targets:
        if fetched:
            time.sleep(POLITE_DELAY_S)
        date = args.date
        if args.pin_from_register:
            date = register_pin(ap.icao)
            if date is None:
                # Never fall through to an unpinned fetch: that would overwrite
                # a cached answer whose OSM state is recorded with one whose is
                # not, and the loss would be silent.
                print(
                    f"  {ap.icao}  FAILED  no pin in taxiout.provenance.OSM_PINS;"
                    " refusing to fetch unpinned",
                    flush=True,
                )
                failures.append(ap.icao)
                continue
        try:
            payload, cached = fetch_airport(ap, CACHE, force=args.force, date=date)
        except Exception as exc:  # noqa: BLE001
            print(f"  {ap.icao}  FAILED  {exc}", flush=True)
            failures.append(ap.icao)
            continue
        if not cached:
            fetched += 1
        origin = "cache" if cached else "overpass"
        pin = f"  pinned to {date}" if date else ""
        print(
            f"  {ap.icao}  {len(payload['elements']):>6} elements  ({origin}){pin}",
            flush=True,
        )

    if failures:
        print(f"\nincomplete: {' '.join(failures)} -- rerun to retry", flush=True)
        return 1
    print("\nall airports cached", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
