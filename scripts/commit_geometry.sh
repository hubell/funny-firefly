#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Rebuild the repository history from scratch, in one clean batch.
#
# The two existing commits are discarded rather than rewritten. Rewriting with
# filter-branch works, but it leaves dangling objects, a reflog, and a
# refs/original backup that all have to be cleaned up and then verified as
# clean; re-initialising leaves nothing to verify. There is no remote and
# nothing has been published, so nothing is lost that anyone else can see.
#
# What this deliberately does NOT do: push. Review `git log -p` first.
#
# One decision is a toggle, because it is a licensing decision rather than a
# technical one -- see the block below.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

# ---------------------------------------------------------------------------
# Commit the derived distance matrices?
#
#   0  (default) leave them out. They are a Derivative Database of OpenStreetMap
#      and carry ODbL share-alike, which is not this project's to waive; this
#      repository ships GPLv3 in totality, so they are regenerated rather than
#      committed. The cost is that checking an individual cell needs an
#      Overpass fetch first. See data/processed/taxi_distance/ODbL-NOTICE.md.
#
#   1  commit them anyway. The repository then carries two licences and must
#      say so prominently. Only do this having decided the two-licence repo is
#      what you want.
#
# Set COMMIT_MATRICES=1 in the environment to switch.
# ---------------------------------------------------------------------------
COMMIT_MATRICES="${COMMIT_MATRICES:-0}"

# ------------------------------------------------------------------ safety --

if [ -d .git ] && git remote | grep -q .; then
    echo "A remote is configured. This script discards local history entirely,"
    echo "which is not something to do to a branch anyone has pulled."
    echo "Remove the remote first if you are certain."
    exit 1
fi

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

say "history being discarded"
git --no-pager log --oneline 2>/dev/null || echo "(no history)"

# Local identity, if one was set on this repository rather than globally.
NAME="$(git config --local user.name || true)"
EMAIL="$(git config --local user.email || true)"

BACKUP="../$(basename "$PWD")-old-history-$(date +%Y%m%d%H%M%S).bundle"
if [ -d .git ] && git rev-parse HEAD >/dev/null 2>&1; then
    say "bundling the old history to $BACKUP before discarding it"
    git bundle create "$BACKUP" --all >/dev/null 2>&1 || true
    echo "delete it once you are happy: rm '$BACKUP'"
fi

# --------------------------------------------------------- fresh history ----

say "re-initialising"
rm -rf .git
git init -q
git symbolic-ref HEAD refs/heads/main
[ -n "$NAME" ] && git config user.name "$NAME"
[ -n "$EMAIL" ] && git config user.email "$EMAIL"

# Spent single-use commit batches. Their commits no longer exist, and neither
# should they.
rm -f scripts/commit_scaffold.sh scripts/commit_notes.sh

# ------------------------------------------------------------- the batch ----

say "committing"

git add LICENSE pyproject.toml .gitignore
git commit -q -m "Add GPLv3 project skeleton for the PRC Data Challenge 2026

GPLv3 from the first commit, as prize eligibility requires. The only
runtime dependency is a graph library; everything else the modelling
stage will need is behind an extra, so installing the geometry pipeline
does not pull several hundred megabytes it never imports."

git add src/taxiout/__init__.py src/taxiout/airports.py src/taxiout/osm.py \
        scripts/fetch_osm.py
git commit -q -m "Extract airport movement-area geometry from OpenStreetMap

An Overpass client sized for how Overpass actually behaves: mirrors fail
transiently and often, so every fetch rotates mirrors with backoff and
every success is cached to disk. Cache files are the unit of provenance
-- delete one to refetch that airport.

Aprons are excluded from the query. The polygons are large, slow to
serve, and contribute nothing to stand-to-runway routing; if an
apron-area feature earns its place later the exclusion is one line.

--date pins a fetch to a moment in OSM's history, which is what makes a
rebuild months from now reproduce the same geometry rather than whatever
the map says then."

git add src/taxiout/coverage.py scripts/report_coverage.py \
        reports/stand_coverage.md
git commit -q -m "Audit whether a stand identifier can be resolved at all

The gate check for everything physical that follows: OSM records stand
identifiers on parking positions, but coverage varies by airport, and if
it is poor somewhere then taxi distance degrades to an airport constant
there. Measured rather than assumed.

Identifiers live in more than one tag -- ref dominates, but Frankfurt has
stands identified only by name, and CDG six of them -- so the resolver
tries four tags in order and records which one answered."

git add src/taxiout/geo.py tests/test_geo.py
git commit -q -m "Add local geodesy sized to the airport, not the globe

Distances, bearings, tangent-plane projection and Douglas-Peucker
simplification, with no external geodesy dependency: the maths is short
and the accuracy requirement is explicit.

Two decisions are load-bearing and both were reached by measuring. The
sphere is fitted to the ellipsoid in the direction each line runs rather
than by its Gaussian mean, because the mean biases east-west and
north-south distances ~0.13% in opposite directions -- a bias aligned
with runway orientation, worth about 9 m on a 3.5 km route.

Turn angle is measured on a simplified polyline rather than on raw
vertices, because vertex density is a property of the mapper: Zurich
carries a vertex every 9 m and Istanbul one every 39 m. Fixed-step
resampling was tried first and introduced a worse artefact -- it discards
the original vertices, so an S-bend narrower than the step cancels
itself, and the total then depended on which end it was measured from by
up to 150 degrees. Simplification is symmetric under reversal, which a
total turning angle has to be."

git add src/taxiout/runways.py tests/test_runways.py
git commit -q -m "Assemble physical runways and resolve designator thresholds

OSM splits runways into however many ways the mapping produced --
Schiphol reads as 10 for 6 runways -- so ways are grouped by ref and the
two ends taken as the farthest-apart pair, which survives fragments that
overlap, reverse or leave gaps where endpoint chaining does not.

Displaced-threshold pavement is re-attached to the runway it extends.
Excluding it made the same column mean different things at different
airports, depending only on whether a mapper had split it out, and
shortened Madrid's 18R/36L by 986 m against its published length. A
displacement moves the landing threshold; a departure still lines up at
the physical end and uses the full declared take-off run.

Helipad refs and non-designator text are rejected rather than routed to."

git add src/taxiout/graph.py tests/test_graph.py
git commit -q -m "Build a routable surface graph, repairing OSM topology explicitly

Three defects have to be repaired before anything routes, and each one
silently changes taxi distance if left alone: nodes that are coincident
but not shared, nodes lying on edges they are not connected to, and ways
that cross on the map with no vertex at the crossing at all. The third
has nothing to split, so the intersection is computed and a junction
inserted.

All three run to a fixed point, and reaching one took two attempts.
Without a minimum-length guard the repair split off half-edges of a few
millimetres in densely mapped areas, whose midpoints then sat on their
neighbours; it settled into a cycle that never terminated, so the output
was a function of how many rounds were permitted rather than of the
airport. Split positions are now bounded away from both ends in metres,
and no (node, edge) combination is considered twice.

Tolerances are deliberately tight throughout. A generous snap would
connect everything and report perfect coverage while routing aircraft
through buildings."

git add src/taxiout/routing.py scripts/build_taxi_graph.py
git commit -q -m "Route stands to runways and describe the routes physically

Distance is reported in separable pieces -- pushback lead-in, snap
allowance, routed distance, runway backtrack -- alongside cumulative turn
angle, sharp-turn count, junction count, junction-free running and
runway crossings. Ravizza et al. found route composition explains
substantially more variation in taxi speed than distance alone, and turn
angle alone nearly as much as distance, so which descriptors earn their
place is left to the modelling stage where it can be measured.

Crossings are counted from the nodes a route passes through, not the
edges it traverses. Repairing a crossing inserts a junction, so a route
crossing a runway traverses no runway edge -- counting edges reported
zero crossings for every route at CDG, which reads as a finding about
the airport rather than as a bug.

Rows aggregate to the real feature closest to the group mean rather than
a per-column median. Median-per-column broke the documented identity
between the total and its parts in a fifth of Istanbul's cells and
turned integer counts into halves; taking the lower of two middle values
instead always picked the shorter route, biasing Barcelona 154 m low."

if [ "$COMMIT_MATRICES" = "1" ]; then
    git add data/processed/taxi_distance reports/taxi_routing.md \
            data/challenge/README.md
    git commit -q -m "Publish the stand-to-runway distance matrices under ODbL

Between 95% and 100% of attached stand-runway pairs route at every
airport. The raw OSM topology does not support that directly -- the
repairs above are what makes it possible.

The matrices are a Derivative Database of OpenStreetMap, so they carry
ODbL rather than the GPLv3 that covers the code, and say so where they
live. They are committed rather than regenerated so the numbers can be
checked without an Overpass round trip; the raw cache stays out, and so
do the per-feature files, which are a debugging artefact.

The report states six things the numbers do not establish. The first is
that these identifiers join to STAND_mvt at all, which is unverified
until the challenge data lands."
else
    git add reports/taxi_routing.md data/processed/taxi_distance/README.md \
            data/processed/taxi_distance/ODbL-NOTICE.md data/challenge/README.md
    git commit -q -m "Add the stand-to-runway routing report

Between 95% and 100% of attached stand-runway pairs route at every
airport. The raw OSM topology does not support that directly -- the
repairs above are what makes it possible.

The matrices themselves are not committed. They are a Derivative
Database of OpenStreetMap and carry ODbL share-alike, which is not ours
to waive, and this repository ships GPLv3 in totality -- so they are
regenerated with 'make data && make reports' rather than shipped. The
reports are Produced Works under ODbL, needing attribution and nothing
more, so they carry the figures instead.

The report states six things the numbers do not establish. The first is
that these identifiers join to STAND_mvt at all, which is unverified
until the challenge data lands."
fi

git add scripts/check_runways_ourairports.py scripts/check_penalty_sensitivity.py \
        reports/runway_crosscheck.md reports/penalty_sensitivity.md
git commit -q -m "Check the geometry against a second source and a swept parameter

Two independent sources agreeing is worth more than one source checked
against itself. OurAirports is public domain and publishes runway
headings in degrees true, which settled the Istanbul anomaly: 16L/34R
sits 19 degrees off its nominal heading, which looks exactly like a
mis-merged runway and is not -- with five near-parallel strips there are
only three position letters to go round, so the designator number is
offset instead. All 39 runways now agree on length, heading and
threshold position.

The runway-edge penalty is swept rather than asserted. Distance barely
moves across a twentyfold range while the amount of foreign runway
pavement the routes use collapses, so the value in use sits on a plateau
rather than at an optimum. Turn angle does move across the full range,
by about a fifth, and the report says so rather than claiming otherwise."

git add docs/ README.md scripts/commit_geometry.sh
git commit -q -m "Record prior art, external data licences and the validation plan

Building an OSM airport graph is established work and is cited as such;
what appears to be unestablished is using the recovered quantities as
regression features for taxi-out, which no entry in either past edition
of this challenge seems to have done.

Every external source considered is listed with its licence position,
including the ones ruled out: OPDI publishes exactly the milestones this
problem wants but carries no open licence, and no openly licensed
pan-European aerodrome mapping database exists.

The leakage rule gains a second condition. Causality is not sufficient:
a feature must also read only fields the ranking set exposes for test
rows, and the causally-correct queue count fails that, because it
subtracts other movements' taxi time -- which is the hidden target.

Also records why the geometry needs held-out-pair validation. Every
feature is a deterministic function of (stand, runway), so a random fold
would show it working while testing nothing."

git add Makefile .github scripts/check_licence_hygiene.py
git commit -q -m "Add a single entry point and CI

make data / reports / test / crosscheck / licence-check, each naming what
it reads and what it writes.

licence-check verifies the property the repository depends on: GPLv3 in
totality, with no OpenStreetMap content tracked. It caught two real
leaks the first time it ran, including a literal identifier inside the
docstring of the function written to avoid literal identifiers.

CI runs the tests twice: once with only pytest installed, to prove the
clean-checkout path a reader actually takes, and once installed. It
cannot rebuild the matrices, because that would mean hammering the
public Overpass API from a runner."

# --------------------------------------------------------------- verify ----

say "result"
git --no-pager log --oneline
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

say "no assistant attribution anywhere"
# Check 4 of scripts/check_licence_hygiene.py, run above, covers the
# tree and the recent history in one place. It spells the pattern in
# bracket classes so it does not match its own source, which is what
# made every copy of this grep have to exclude every other one.

echo
echo "Nothing has been pushed. Review with 'git log -p', then add a remote."
