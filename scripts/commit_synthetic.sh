#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the worked airport, its worked traffic, and the record-quality
# section of the modelling notes, as two commits on top of the current
# history. Additive; nothing is rewritten.
#
# The first commit adds docs/modelling_notes.md whole. Only the *Tail
# handling* section (and one spelling in the bullet that cites it) changed
# in this stream, so run this before any other uncommitted edit lands in
# that file, or that edit will be swept into this commit under the wrong
# message. The script refuses if the file's diff touches lines outside the
# section, to make that mistake loud rather than silent.
#
# Runs the tests and the licence gate first and stops if either fails. Does
# not push. Review `git log -p -2` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

MINE="src/taxiout/synthetic.py tests/test_synthetic_airport.py \
      scripts/report_synthetic_airport.py reports/synthetic_airport.md \
      reports/synthetic_airport.svg scripts/commit_synthetic.sh"
for f in $MINE docs/modelling_notes.md; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the modelling notes changed only where this stream wrote"
# Every changed hunk must lie between the Tail handling heading and the
# next heading, save the one-word spelling fix in the bullet that cites it.
git diff -U0 -- docs/modelling_notes.md | python3 - <<'EOF'
import re
import sys
from pathlib import Path

lines = Path("docs/modelling_notes.md").read_text(encoding="utf-8").splitlines()
start = next(i for i, l in enumerate(lines, 1) if l.startswith("## Tail handling"))
end = next(i for i, l in enumerate(lines, 1) if i > start and l.startswith("## "))

hunks, current = [], None
for line in sys.stdin.read().splitlines():
    m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
    if m:
        first = int(m.group(1))
        n = int(m.group(2)) if m.group(2) is not None else 1
        current = {"first": first, "last": first + max(n, 1) - 1, "added": [], "removed": []}
        hunks.append(current)
    elif current is not None and line.startswith("+") and not line.startswith("+++"):
        current["added"].append(line[1:])
    elif current is not None and line.startswith("-") and not line.startswith("---"):
        current["removed"].append(line[1:])

bad = []
for h in hunks:
    inside = start < h["first"] and h["last"] < end
    spelling = (
        len(h["added"]) == 1
        and len(h["removed"]) == 1
        and "flagged artifact rows" in h["added"][0]
        and "flagged artefact rows" in h["removed"][0]
    )
    if not (inside or spelling):
        bad.append(f"lines {h['first']}-{h['last']}")
if bad:
    print("docs/modelling_notes.md has changes outside the Tail handling section:")
    for b in bad:
        print("   ", b)
    print("Commit or park those first; this script will not sweep them in.")
    sys.exit(1)
print(f"ok: {len(hunks)} hunk(s), all inside lines {start}-{end} or the one spelling fix")
EOF

say "tests"
python3 -m pytest -q

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "1/2 the record-quality term, as it turned out to be"
git add docs/modelling_notes.md
git commit -q -m "Record what the record-quality term turned out to be

The Tail handling section was written when the tail read as day-shift
artifacts and the term as a bucket mean. Building the term found the
mechanism behind most of it: at one airport, a departure with no
flight-plan record and a take-off hours after schedule has, nine times
in ten, a recorded off-block time within a minute of the schedule, so
the recorded taxi-out is the schedule lag -- which every ranking
departure exposes. The section now says that, states the term as the
mixture the mechanism makes optimal, records the two forms measured and
set aside, names the one exposure the design carries, and keeps the
sentence the paper has to say: the term is the RMSE-optimal prediction
under the stated scoring rule for rows whose recorded value no aircraft
produced. Figures are quoted from reports/tails.md, which regenerates.

One word in the Validation bullet that cites the section takes the same
spelling."

say "2/2 the worked airport"
git add $MINE
git commit -q -m "Add a worked airport where every number is known

The geometry is validated on the real airports by consistency: pavement
conserved across repairs, fixed points reached, no route shorter than
the straight line, runway lengths against OurAirports. None of that
compares a routed distance with the right answer, because nobody has
one for a real stand. taxiout.synthetic draws a field that does: two
runways, a parallel taxiway, two aprons with single exits, eleven
parking positions covering every attachment rule, and each of the three
topology repairs planted exactly once. It is authored -- no OpenStreetMap
content, no challenge data -- and it runs on a clean checkout, which
makes reports/synthetic_airport.md the one geometry report continuous
integration can regenerate.

Every one of the 32 (stand, designator) cells matches its hand
derivation to 11 mm and 0.22 degrees, with the integer descriptors
exact; the runway-penalty crossovers fall at the values the arithmetic
gives. The report prints both columns side by side and draws the field.

Building it found three things in the pipeline, each below the
tolerances it works to and each now measured: a link that overshoots a
taxiway by 0.8 m is a crossing while one that stops 0.8 m short is a
node on an edge, so the two repairs are told apart by geometry; a merged
node takes the coordinates of the last way that referenced it, and a
later split on an edge at that node books the shift once (0.2 m on a
probe); and with exactly two features on one stand the closest-to-mean
representative is a tie that floating-point rounding decides. The fixes
are small and belong to graph.py and routing.py; the field keeps its
merge and its split on different edges so the tables stay exact, and
draws its duplicated stand three times so the rule's choice is unique.

The field also carries worked traffic: a year of movements in the
challenge's own schema and file layout, from a process every parameter
of which is stated -- a fixed taxi speed over the field's own distances,
a first-in-first-out runway queue, an off-block delay, noise, and a feed
that defaults the block time to the schedule for a stated share of
departures without a flight-plan record. taxiout.challenge loads it
unchanged, the ranking months are blanked as the challenge blanks them,
and the target's components are kept beside the ranking rows. The
record-quality term, the submission writer and the loaders run on it
end to end in the tests; a model that knows the process scores 3,511 s
without the term and 281 s with it, against 44 s on the rows the term
leaves alone, which is the scoring rule in one table.

The field's reference point is on the equator. At 50 N the convergence
of meridians makes a parallel 350 m north of the reference 55 parts per
million shorter than a tangent plane says, which put a 6 cm discrepancy
into a 1,750 m route on the first draft; that is geometry, not error,
but a table that claims to be exact cannot carry it."

say "result"
git --no-pager log --oneline -3
echo
git status --short

echo
echo "Not pushed. Review with: git log -p -2"
