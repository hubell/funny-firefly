#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit, and it is a record rather than a change: the extension of the
# sweep's grid was probed on October and one axis of the three won.
#
# THE TREES AXIS WINS AND THE OTHER TWO LOSE. October, one fit each, read
# against the current cell's 214.9045 from the same run and that run's measured
# floor of 0.1815 s:
#
#   n_estimators 6000   214.3041   -0.6004   3.31 floors   REPLACES
#   n_estimators 5000   214.4218   -0.4826   2.66 floors   REPLACES
#   n_estimators 4000   214.6368   -0.2676   1.47 floors   REPLACES
#   num_leaves 511      215.1319   +0.2274                 keeps the current cell
#   min_child_samples 10  215.1426  +0.2381                keeps the current cell
#
# Only one axis wins, so the pre-registration's combination clause does not
# arise and no factorial was fitted. The day-clustered interval on the 6000-tree
# cell against the current one is -0.6004 s [-0.7333, -0.4694] over 1,000 draws
# and does not span zero, so the two instruments agree here. It decided nothing:
# the rule reads the floor and nothing else.
#
# THE WINNING VALUE IS ON AN EDGE AGAIN, and this is the record the
# pre-registration's third reading rule asks for. 6,000 is the largest tree
# count fitted, and October was still falling monotonically at 4,000, 5,000 and
# 6,000. Under those criteria the extension is not extended a second time and a
# further reopening is not available.
#
# THE GATE PASSED BEFORE ANY PROBE FIGURE WAS READ. The five single-seed October
# fits at the parameters c691026 carried reproduced the records on disk exactly,
# to the stored double and to the prediction digest, on all five seeds:
# 217.282578, 217.647650, 217.431498, 217.374337 and 217.428422, floor 0.189955
# against the 0.19 on record
#
# WHAT THE COST IS, AND WHICH PART OF IT IS MEASURED. Measured, on October, on
# one machine, in one process: the current cell fits 3,000 trees in 189.7 s and
# the adopted cell fits 6,000 in 321.8 s, a ratio of 1.696x. Scaled, and only
# scaled: applying that one ratio to the 7:11:17 ladder measured at the current
# cell gives about 12:11:37, some five hours more per ladder. The scaling
# assumes the ladder's wall is dominated by booster fits under these parameters,
# which is an assumption and not a reading. The per-leg timing that would
# replace the assumption was NOT run here, for the reason below.
#
# THE SEED-STABILITY CHECK CAME FIRST AND THE RANKING DID NOT SURVIVE. The four
# cells within one floor of the best, refit at random_state + 1:
#
#   lr 0.015, 255 leaves, mcs 20    214.9045 -> 215.1932   +0.2887   1st -> 4th
#   lr 0.03,  127 leaves, mcs 20    214.9982 -> 214.9078   -0.0904   2nd -> 2nd
#   lr 0.03,  255 leaves, mcs 20    215.0043 -> 214.8923   -0.1120   3rd -> 1st
#   lr 0.03,  255 leaves, mcs 200   215.0222 -> 215.1868   +0.1646   4th -> 3rd
#
# The cell the wider sweep chose is first at one seed and last at the other, and
# four of the six pairs cross. This commit draws no conclusion from that and
# changes nothing on account of it. It is the owner's reading, and the two
# routes it could take were written down before the figures existed
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it does not move BOOSTER_PARAMS, whatever the probe found. The cost of a
#     ladder at any new cell is the owner's decision on the measurement, and
#     Block 09g is where a move lands.
#   * it does not move SEED_AVERAGE_ADOPTED, which is a separate decision on a
#     separate measurement.
#   * it regenerates no report and changes no figure in one. The assertions
#     below compare reports/, docs/ and src/ against BASE as whole git trees,
#     so a single moved byte anywhere under them fails this script.
#   * it commits no data. data/processed/sweep_extension_october.csv and
#     data/processed/sweep_extension_gate.json are gitignored under
#     data/processed/*, like the wider sweep's own sweep_october.csv.
#   * it adds no entry to the register. The extension introduces no
#     module-level numeric constant in src/ or scripts/, which is what
#     provenance.CONSTANTS covers by its own SCOPE, and the one file this
#     commit adds is a shell script, which the scanner's ast walk does not
#     read. An entry would also require editing src/taxiout/provenance.py,
#     which is the src/ tree this script asserts is unchanged.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

PY="${PYTHON:-$([ -x .venv/bin/python ] && echo "$PWD/.venv/bin/python" || echo python3)}"
RUFF="$([ -x .venv/bin/ruff ] && echo "$PWD/.venv/bin/ruff" || echo ruff)"
case "$PY" in /*) ;; */*) PY="$PWD/$PY" ;; esac

# The commit this block was run on, verified at the start of Block 09f.
BASE="ce8039f"

# The whole of the working-tree change this block makes. One file.
PATHS="scripts/commit_extension_probe.sh"

# The trees nothing in this block may move, compared against BASE as whole
# git tree objects. A tree hash covers every byte of every file beneath it.
FROZEN_TREES="reports docs src scripts/preregistered.py scripts/sweep_params.py"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "this sits on the commit it says it sits on"
if ! git merge-base --is-ancestor "$BASE" HEAD 2>/dev/null; then
    echo "$BASE is not an ancestor of HEAD; this script expects to follow it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' "$BASE")"
if [ "$(git rev-parse HEAD)" != "$(git rev-parse "$BASE")" ]; then
    echo "HEAD has moved past $BASE since Block 09f measured; re-verify before committing"
    exit 1
fi
echo "ok  HEAD is $BASE itself"

say "the working tree holds this one path and nothing else"
expected="$(printf '%s\n' "$PATHS" | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  1 path"

say "1/6  BOOSTER_PARAMS is unchanged, by value and by byte"
git cat-file blob "$BASE:src/taxiout/models.py" > "$work/models.base"
cmp -s "$work/models.base" src/taxiout/models.py || {
    echo "src/taxiout/models.py differs from $BASE; this block moves no parameter"
    exit 1
}
"$PY" - <<'PY'
import sys
sys.path.insert(0, "src")
from taxiout.models import BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED

want = {"n_estimators": 3000, "learning_rate": 0.015,
        "num_leaves": 255, "min_child_samples": 20}
got = {k: BOOSTER_PARAMS[k] for k in want}
if got != want:
    raise SystemExit(f"BOOSTER_PARAMS moved: {got} against the expected {want}")
print(f"  ok  BOOSTER_PARAMS still {got}")
if SEED_AVERAGE_ADOPTED is not False:
    raise SystemExit(f"SEED_AVERAGE_ADOPTED is {SEED_AVERAGE_ADOPTED}, expected False")
print("  ok  SEED_AVERAGE_ADOPTED still False")
PY

say "2/6  no report, no document and no module moved: whole trees against $BASE"
for t in $FROZEN_TREES; do
    a="$(git rev-parse "$BASE:$t")"
    b="$(git rev-parse "HEAD:$t")"
    if [ "$a" != "$b" ]; then
        echo "$t moved between $BASE and HEAD: $a -> $b"
        exit 1
    fi
    echo "  ok  $t $a"
done
echo "and the working copies of those trees are clean, by the status check above"

say "3/6  the probe's own outputs are present and are not tracked"
for f in data/processed/sweep_extension_october.csv \
         data/processed/sweep_extension_gate.json; do
    [ -f "$f" ] || { echo "$f is missing; the probe did not finish"; exit 1; }
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
        echo "$f is tracked, and it must not be"
        exit 1
    fi
    git check-ignore -q "$f" || { echo "$f is not gitignored"; exit 1; }
    echo "  ok  $f present, untracked, ignored"
done

say "4/6  the suite, ruff and licence hygiene, in this tree"
"$PY" -m pytest -q
"$RUFF" check .
"$RUFF" format --check .
"$PY" scripts/check_licence_hygiene.py
"$PY" scripts/check_provenance.py

say "5/6  the same, in a scratch replica built read-only from git ls-files"
# Read-only on this repository: git ls-files is the only thing asked of it, and
# the replica is a copy rather than a worktree, so nothing under .git/ is
# written. data/ is absent, which is the condition 221a7a8 failed under and
# the condition a fresh clone is in.
mkdir -p "$work/t"
git ls-files -z | grep -zv '^data/' | tar --null -cf - -T - | tar -xf - -C "$work/t"
cp -p "$PWD/scripts/commit_extension_probe.sh" "$work/t/scripts/"
if [ -d "$work/t/data" ]; then
    echo "data/ reached the replica and should not have"
    exit 1
fi
(
    cd "$work/t"
    git init -q .
    git add -A
    git -c user.email=probe@invalid -c user.name=probe commit -qm replica
    PYTHONPATH="$work/t/src" "$PY" -m pytest -q
    "$RUFF" check .
    "$RUFF" format --check .
    "$PY" scripts/check_licence_hygiene.py
    "$PY" scripts/check_provenance.py
) > "$work/replica.txt" 2>&1 || {
    echo "--- the replica, with data/ absent ---"
    cat "$work/replica.txt"
    exit 1
}
echo "ok  suite, ruff, licence and provenance all pass with data/ absent"
tail -3 "$work/replica.txt" | sed 's/^/  /'

say "6/6  this script names nothing it must not, and carries no trailer"
if grep -n -i -E '[c]laude|[a]nthropic|[c]hatgpt|[c]opilot|[a]ssistant|[c]o-authored-by|[c]owork' \
        scripts/commit_extension_probe.sh; then
    echo "the script names something it must not"
    exit 1
fi
echo "ok  clean"

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

git commit -F - <<'MSG'
Probe the three hot axes past the sweep's edge, and move no parameter

The wider sweep's chosen cell sat on an edge of its grid on all four axes and
that was recorded, as the sweep's own rule required. The rule was reopened
afterwards, knowing the direction of the edge, and the grid was extended once
under criteria fixed in writing before any figure past the edge existed. That
sequence is disclosed rather than smoothed, and the criteria chose the axes
from the sweep's own gradients, which is a design choice and is disclosed as
one.

Three axes were probed one step out, independently, on October alone. The
trees axis wins by 0.6004 s against a measured floor of 0.1815 s and is still
falling at the largest count fitted. The leaves and minimum-child-samples
axes both lose, by 0.2274 s and 0.2381 s, so the criteria's combination
clause never arises. A capacity analysis made before the probe, from the 162
readings already on disk, predicted about 0.071 s of remaining headroom along
the single direction these three axes share; the measurement is roughly eight
times that, which the criteria named in advance as the outcome worth a second
look rather than the expected one.

The gate ran before any probe figure was read and reproduced the five October
single-seed fits on record to the stored double and the prediction digest,
with the floor coming back at 0.19 s.

BOOSTER_PARAMS does not move here and neither does anything else. No report
is regenerated, no figure in one moves, docs/ and src/ are unchanged as whole
git trees, and the probe's own readings are gitignored beside the wider
sweep's. What a ladder at the winning cell would cost is measured on October
and scaled, and spending it is a decision this commit deliberately leaves
open.
MSG

say "the frozen trees are still frozen, now that the commit exists"
for t in $FROZEN_TREES; do
    a="$(git rev-parse "$BASE:$t")"
    b="$(git rev-parse "HEAD:$t")"
    [ "$a" = "$b" ] || { echo "$t moved in the commit: $a -> $b"; exit 1; }
    echo "  ok  $t unchanged"
done
echo "and the commit touches exactly:"
git --no-pager show --stat --oneline HEAD | tail -n +2

say "result"
git --no-pager log --oneline -3
echo
git -c core.fileMode=false status --short

say "nothing untracked that should be tracked"
git -c core.fileMode=false status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -1"
