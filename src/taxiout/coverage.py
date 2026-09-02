# SPDX-License-Identifier: GPL-3.0-only
"""Stand-identifier coverage audit.

This answers the question that gates the whole physically-grounded feature set:
*can a stand ID from the challenge movement data be resolved to a location?*

The challenge supplies ``STAND_mvt`` as an opaque string. OpenStreetMap records
stand identifiers on ``aeroway=parking_position`` features, usually in ``ref``,
sometimes only in ``name``, and at some airports not at all. If an airport has
poor coverage, taxi-distance features degrade to airport-level constants there
and the modelling strategy has to account for it.

Coverage is measured, never assumed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# tags that have been observed to carry a stand identifier, best first
STAND_ID_TAGS: tuple[str, ...] = ("ref", "aeroway:ref", "local_ref", "name")


@dataclass
class AirportCoverage:
    icao: str
    parking_positions: int = 0
    with_any_id: int = 0
    id_tag_hist: Counter = field(default_factory=Counter)
    taxiways: int = 0
    taxilanes: int = 0
    runways: int = 0
    aprons: int = 0
    holding_positions: int = 0
    id_values: list[str] = field(default_factory=list)

    @property
    def id_coverage(self) -> float:
        if not self.parking_positions:
            return 0.0
        return self.with_any_id / self.parking_positions

    @property
    def verdict(self) -> str:
        """Coarse usability band for the taxi-distance feature."""
        if self.parking_positions == 0 or self.runways == 0:
            return "UNUSABLE"
        cov = self.id_coverage
        if cov >= 0.90:
            return "STRONG"
        if cov >= 0.60:
            return "PARTIAL"
        if cov > 0.0:
            return "WEAK"
        return "UNUSABLE"


def identifier_shape(value: str) -> str:
    """Reduce an identifier to its character shape: a pier letter and two
    digits becomes ``@##``.

    Digits become ``#``, letters ``@``, anything else ``.``. The alphabet is
    deliberately non-alphanumeric: an earlier version used ``A`` and ``9``, and
    the resulting shapes were themselves indistinguishable from real stand
    identifiers -- which defeated the point and tripped the licence check that
    exists to catch exactly that. This exists for a
    licensing reason as much as an analytical one. The question that matters is
    whether OSM's identifier *format* matches ``STAND_mvt`` -- whether to expect
    bare numerics, a pier letter, a left/right suffix -- and a shape answers it
    exactly. Listing real identifiers answers it too, but a list of real
    identifiers is an extract of OpenStreetMap's contents, and this repository
    ships GPLv3 with no OSM content in it. A shape is a description, not an
    extract, so it can be published here; the values it describes cannot.
    """
    out = []
    for ch in value.strip():
        out.append("#" if ch.isdigit() else "@" if ch.isalpha() else ".")
    return "".join(out)


def shape_histogram(values: list[str], top: int = 4) -> list[tuple[str, int]]:
    """Most common identifier shapes, commonest first."""
    return Counter(identifier_shape(v) for v in values).most_common(top)


def stand_identifier(tags: dict) -> tuple[str, str] | None:
    """Return ``(tag_used, value)`` for the first usable stand ID tag."""
    for tag in STAND_ID_TAGS:
        value = tags.get(tag)
        if value and str(value).strip():
            return tag, str(value).strip()
    return None


def summarise(icao: str, payload: dict) -> AirportCoverage:
    cov = AirportCoverage(icao=icao)
    counters = {
        "taxiway": "taxiways",
        "taxilane": "taxilanes",
        "runway": "runways",
        "apron": "aprons",
        "holding_position": "holding_positions",
    }

    for element in payload.get("elements", []):
        tags = element.get("tags") or {}
        kind = tags.get("aeroway")
        if kind == "parking_position":
            cov.parking_positions += 1
            found = stand_identifier(tags)
            if found:
                tag, value = found
                cov.with_any_id += 1
                cov.id_tag_hist[tag] += 1
                cov.id_values.append(value)
        elif kind in counters:
            setattr(cov, counters[kind], getattr(cov, counters[kind]) + 1)

    return cov
