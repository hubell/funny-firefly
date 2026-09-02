# SPDX-License-Identifier: GPL-3.0-only
"""Physical runways and their thresholds, recovered from OSM runway ways.

OpenStreetMap splits a runway into however many ways the mapping happened to
produce: Schiphol reads as 10 ways for 6 physical runways, Zurich as 8 for 3.
Counting ways therefore counts nothing meaningful, and routing to a way is
routing to an arbitrary fragment. This module reduces the fragments to physical
runways and then to the thing a departure actually taxis towards: the threshold
of a named designator.

Two decisions worth stating plainly.

*Thresholds come from ``ref``-tagged geometry only.* Untagged runway ways still
become graph edges -- they carry real pavement and real connectivity -- but they
are never allowed to extend a runway's axis, because a mis-tagged taxiway
merged into a runway group would silently move the threshold by hundreds of
metres.

*The two ends are the farthest-apart pair of points in the group*, not the
result of chaining ways by shared endpoints. Chaining assumes the fragments
form a clean path; they routinely do not (overlaps, gaps, spurs). The diameter
of the point set is robust to all three, and a runway is by construction the
longest straight thing in its own group.

Displaced thresholds are the third decision, and the one that took a
cross-check to get right. Where a runway has a displaced threshold, OSM
sometimes splits that stretch out as a separate way tagged
``runway=displaced_threshold`` with no ``ref``, and sometimes leaves it inside
the ``ref``-tagged way. Excluding it -- the first approach here -- therefore
made the same column mean different things at different airports, and shortened
Madrid's 18R/36L by 986 m against its published length.

The displaced pavement is re-attached, because a displacement moves the
*landing* threshold: a departure still lines up at the physical end and uses
the full declared take-off run. So the physical end is the right take-off point
at every airport, and re-attaching makes the geometry agree with OurAirports'
published lengths rather than disagreeing with them by up to 22%. The amount
re-attached is recorded per runway as ``displaced_m``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .geo import Point, angle_diff_deg, bearing_deg, farthest_pair, geodesic_m

# A runway designator: 01-36, optional L/C/R position letter. Anchored, so a
# helipad ref -- which carries a leading letter where a runway carries a digit --
# and any stray free text are rejected rather than silently routed to.
DESIGNATOR_RE = re.compile(r"^(0?[1-9]|[12]\d|3[0-6])([LCR])?$")


def parse_designators(ref: str | None) -> list[str]:
    """Split an OSM runway ``ref`` into normalised designators.

    ``"09L/27R"`` -> ``["09L", "27R"]``; ``"9L"`` -> ``["09L"]``; a helipad ref,
    which carries a letter where a runway carries a digit, -> ``[]``. A
    single-designator ref such as Frankfurt's ``18``
    yields one designator only: runway 18 is departure-only in that direction
    and inventing ``36`` would fabricate a routing target that is never used.
    """
    if not ref:
        return []
    out: list[str] = []
    for token in str(ref).replace("-", "/").split("/"):
        m = DESIGNATOR_RE.match(token.strip().upper())
        if m:
            out.append(f"{int(m.group(1)):02d}{m.group(2) or ''}")
    return out


def designator_heading(designator: str) -> float:
    """Nominal heading in degrees for a designator (``"27L"`` -> 270.0).

    This is a *magnetic* heading by convention. It is used only to decide which
    of two ends a departure starts from -- a choice with 180 degrees of margin --
    so magnetic variation (up to ~7 degrees across these 11 airports) never
    changes the answer. The residual is reported as a QA signal, not corrected.
    """
    return float(int(designator[:2]) * 10)


@dataclass
class Runway:
    """One physical runway, assembled from its OSM ways."""

    icao: str
    ref: str
    way_ids: list[int] = field(default_factory=list)
    ends: tuple[Point, Point] | None = None
    displaced_m: float = 0.0  # displaced-threshold pavement re-attached

    @property
    def length_m(self) -> float:
        return geodesic_m(*self.ends) if self.ends else 0.0

    @property
    def designators(self) -> list[str]:
        return parse_designators(self.ref)

    def threshold(self, designator: str) -> Point:
        """The end a departure on ``designator`` starts from."""
        if self.ends is None:
            raise ValueError(f"{self.icao} {self.ref}: no geometry")
        a, b = self.ends
        target = designator_heading(designator)
        # Depart from `a` means travelling a -> b, so compare outbound bearings.
        return (
            a
            if angle_diff_deg(bearing_deg(a, b), target)
            <= angle_diff_deg(bearing_deg(b, a), target)
            else b
        )

    def heading_error_deg(self, designator: str) -> float:
        """Angle between the runway axis and the designator's nominal heading.

        Expect a few degrees of magnetic variation. Anything beyond ~15 degrees
        means the group is wrong -- a mis-tagged way, or two runways sharing a
        ref -- and the report flags it rather than quietly routing to it.
        """
        thr = self.threshold(designator)
        other = self.ends[1] if thr == self.ends[0] else self.ends[0]
        return angle_diff_deg(bearing_deg(thr, other), designator_heading(designator))


# A displaced-threshold way belongs to a runway if it runs along the same axis
# and starts where that runway ends. Both conditions are required: alignment
# alone would capture a parallel runway, adjacency alone a crossing one.
DISPLACED_ALIGN_DEG = 10.0
DISPLACED_JOIN_M = 80.0


def _way_points(el: dict) -> list[Point]:
    return [(g["lat"], g["lon"]) for g in el.get("geometry", [])]


def _attach_displaced(runways: list[Runway], ways: list[dict]) -> list[dict]:
    """Fold displaced-threshold pavement back into the runway it extends.

    Returns the ways that could not be attached to any runway.
    """
    orphans: list[dict] = []
    for el in ways:
        pts = _way_points(el)
        if len(pts) < 2:
            orphans.append(el)
            continue
        axis = bearing_deg(pts[0], pts[-1])
        best: Runway | None = None
        best_gap = DISPLACED_JOIN_M
        for rw in runways:
            if rw.ends is None:
                continue
            rw_axis = bearing_deg(*rw.ends)
            # Compare modulo 180: neither line has an inherent direction.
            aligned = min(
                angle_diff_deg(axis, rw_axis), angle_diff_deg(axis, rw_axis + 180.0)
            )
            if aligned > DISPLACED_ALIGN_DEG:
                continue
            gap = min(geodesic_m(p, e) for p in (pts[0], pts[-1]) for e in rw.ends)
            if gap < best_gap:
                best, best_gap = rw, gap
        if best is None:
            orphans.append(el)
            continue
        extended = list(best.ends or ()) + pts
        best.ends = farthest_pair(extended)
        best.way_ids = sorted([*best.way_ids, el["id"]])
        best.displaced_m += _path_length(pts)
    return orphans


def _path_length(points: list[Point]) -> float:
    return sum(geodesic_m(p, q) for p, q in zip(points, points[1:]))


def build_runways(icao: str, elements: list[dict]) -> tuple[list[Runway], list[dict]]:
    """Group OSM runway ways into physical runways.

    Returns ``(runways, unusable)`` where ``unusable`` lists the ways that could
    not be placed, each with a reason. Those ways remain available to the graph
    builder as pavement; they are only barred from defining an axis.
    """
    groups: dict[str, list[dict]] = {}
    displaced: list[dict] = []
    unusable: list[dict] = []

    for el in elements:
        tags = el.get("tags") or {}
        if tags.get("aeroway") != "runway" or el.get("type") != "way":
            continue
        ref = (tags.get("ref") or "").strip()
        if not ref:
            if tags.get("runway") == "displaced_threshold":
                displaced.append(el)
            else:
                unusable.append({"way_id": el["id"], "ref": None, "reason": "no ref tag"})
            continue
        if not parse_designators(ref):
            unusable.append(
                {"way_id": el["id"], "ref": ref, "reason": "ref is not a designator"}
            )
            continue
        groups.setdefault(ref, []).append(el)

    runways: list[Runway] = []
    for ref, ways in sorted(groups.items()):
        points: list[Point] = []
        for w in ways:
            points.extend(_way_points(w))
        if len(points) < 2:
            unusable.append(
                {"way_id": ways[0]["id"], "ref": ref, "reason": "no geometry"}
            )
            continue
        runways.append(
            Runway(
                icao=icao,
                ref=ref,
                way_ids=sorted(w["id"] for w in ways),
                ends=farthest_pair(points),
            )
        )

    for el in _attach_displaced(runways, displaced):
        unusable.append(
            {
                "way_id": el["id"],
                "ref": None,
                "reason": "displaced threshold matching no runway axis",
            }
        )
    return runways, unusable
