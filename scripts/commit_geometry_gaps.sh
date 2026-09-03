#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the unimpeded-term diagnostics -- reports/geometry_gaps.md and its
# generator -- as one commit on top of the current history. Additive;
# nothing is rewritten. Adds only these two files.
#
# Runs the tests and the licence gate first and stops if either fails. Does
# not push. Review `git log -p -1` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

MINE="scripts/report_geometry_gaps.py reports/geometry_gaps.md scripts/commit_geometry_gaps.sh"
for f in $MINE; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "tests"
python3 -m pytest -q

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "1/1 the unimpeded term against the data"
git add $MINE
git commit -q -m "Measure the unimpeded term against the training year

The geometry supplies a full-length stand-to-threshold distance for
every pair, and the notes left open whether departures taxi to the
threshold at all. reports/geometry_gaps.md asks the training year
three things about that quantity and answers with aggregates.

The fit: per airport, a pair's fastest decile follows its routed
distance with a residual of 85 to 190 s and an intercept of 313 to
740 s that no distance explains -- pushback and line-up, which a speed
fitted through the origin folds into the slope. Barcelona fits worst,
on the narrowest spread of distances. The residuals sit on identifiers
shared by two OpenStreetMap features (-171 s at Istanbul, +80 s at
Paris, on hundreds of pairs each) and on snapped stands, which is the
representative-pick finding of the worked airport showing on real
ground; those tables regenerate after the routing fix.

Wake category: paired on the same stand and runway, medium-wake
departures' fastest decile is 35 to 281 s below heavies', nearly the
same on every designator of an airport -- one to three kilometers at
the fitted speeds, far too large and too uniform to be intersection
departures. It is a fixed pushback-to-line-up and speed difference.
On the mean, with the stand identity in the model, an additive class
offset is worth about one second of forward-split RMSE, because
heavies park at wide-body stands and the stand absorbs the class; at
the same stand and runway the gap on the mean is 60 to 268 s per
airport, which is what a model without the stand -- the geometric
rung, and every never-seen pair -- is missing. Neither the unimpeded
term nor the Gate 1 feature frame carries the category or the aircraft
type, and the ranking set exposes both for every departure.

Intersection departures proper: no designator's wake gap departs from
its airport's by more than 96 s, and the designator residuals beyond a
hundred seconds are the short-runway and heavy-only configurations,
1.2% of departures. Both are constant per designator and absorbed by
a model that knows the designator and the class. The graph lists each
runway's taxiway junctions and their offsets from the threshold; which
stands use which entry is unobservable without surface tracks, and
stays a diagnostic for the paper if they arrive."

say "result"
git --no-pager log --oneline -3
echo
git status --short

echo
echo "Not pushed. Review with: git log -p -1"
