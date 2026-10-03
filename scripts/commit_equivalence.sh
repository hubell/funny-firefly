#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit, and like the one before it a record rather than a change: two
# October fits asked whether the adopted cell's capacity is available for half
# the tree-building work, and whether its result survives a second seed.
#
# THE GATE PASSED FIRST, and reproduced the block before it exactly. The five
# October singles at the parameters c691026 carried came back equal to the
# stored double AND the prediction digest on all five seeds, old-cell floor
# 0.189955. The current cell's five singles reproduced to the last digit, so
# this run's floor is 0.181530, the same figure to six places.
#
# RULE 1, COST, TAKEN. The rule is an equivalence test and not a superiority
# test: the cheaper cell replaces the dearer one if it is not worse by more
# than the floor.
#
#   A at 6,000   0.015, 255, 20   214.3041   1,530,000 nodes   321.8 s fit
#   B at 6,000   0.03,  127, 20   214.4391     762,000 nodes   237.5 s fit
#
# B is worse by 0.1350 s, which is 0.74 of the 0.1815 floor, so it is not
# worse by more than the floor and the rule takes it. THIS IS A DECISION
# TAKEN ON COST AND IT IS RECORDED AS ONE: B is the worse of the two October
# readings and is not presented as a better score. What it buys is half the
# terminal nodes, and a measured October fit at 0.736x the time.
#
# RULE 2, THE SEED, FELL IN A GAP THE CRITERIA DID NOT COVER, and this is
# recorded rather than resolved. A at 6,000's second seed reads 214.5913
# against its first at 214.3041, a gap of 0.2872 s, which is 1.58 floors and
# therefore NOT within the floor, so the criteria's "the trees result stands
# as measured" branch did not fire. Neither did the stop branch: both seeds
# sit below the current cell's 214.9045, by 3.31 and 1.73 floors, so they do
# not straddle it and the winner's-curse condition the criteria named did not
# occur. What is measured is that the direction of the trees result survives
# the seed and the size of it does not: 0.60 s on one draw and 0.31 s on the
# other.
#
# Worth setting beside that: the A-to-B difference of 0.1350 s is SMALLER
# than A's own seed-to-seed spread of 0.2872 s, so the instrument cannot
# separate the two cells while it can separate one cell from itself.
#
# THE INTERVALS, WHICH DECIDE NOTHING, and one of them disagrees with the
# floor:
#
#   B at 6,000 minus the current cell        -0.4654  [-0.9492, +0.0009]  spans zero
#   A at 6,000 seed 1 minus the current cell -0.3132  [-0.5898, -0.0349]  clear
#   B at 6,000 minus A at 6,000 seed 0       +0.1350  [-0.3347, +0.5941]  spans zero
#   A seed 1 minus A seed 0                  +0.2872  [+0.0137, +0.5485]  clear
#
# The floor puts B at 2.56 floors better than the current cell; the
# day-clustered interval on the same contrast cannot exclude zero. The two
# instruments disagree, the rule reads the floor and nothing else, and this
# is the second time the project has met that, which is why it is written
# down here rather than found during the documentation pass. The contrasts
# against the current cell are the wider ones because B is a different model
# rather than a longer run of the same one, so its residuals track the
# baseline's less closely.
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it does not move BOOSTER_PARAMS. Which cell ships is a decision on this
#     measurement and the move lands in a later block.
#   * it does not move SEED_AVERAGE_ADOPTED.
#   * it regenerates no report and changes no figure in one. reports/, docs/
#     and src/ are asserted unchanged against BASE as whole git trees, so a
#     single moved byte under any of them fails this script.
#   * it does not probe the trees axis further. That axis is closed at 6,000
#     by the extension's own criteria, whatever these two fits showed.
#   * it does not re-own the PROVENANCE_PENDING entry that still names a
#     closed block. That edit was attempted and reverted, because the register
#     generates docs/provenance.md and check 10 requires the two to agree, so
#     the edit cannot be made without moving docs/. The finding is recorded in
#     this block's note and the edit is a later block's, together with the
#     document regeneration it forces.
#   * it commits no data. data/processed/equivalence_october.json is
#     gitignored under data/processed/*, like the sweep's own readings.

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

# The commit this block was run on, verified at the start of the block.
BASE="fb19c00"

# The whole of the working-tree change this block makes. One file.
PATHS="scripts/commit_equivalence.sh"

# The trees nothing in this block may move, compared against BASE as whole git
# tree objects. A tree hash covers every byte of every file beneath it.
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
    echo "HEAD has moved past $BASE since this block measured; re-verify before committing"
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

say "1/6  BOOSTER_PARAMS and SEED_AVERAGE_ADOPTED are unchanged"
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

say "3/6  the register is still what it was, and still generates its document"
# The entry naming a closed block is deliberately NOT fixed here; see the
# header. What is checked is that the attempt left nothing behind.
if ! grep -q 'two at Block 09f, four at Block 11' src/taxiout/provenance.py; then
    echo "the pending entry has moved; this script expects it untouched"
    exit 1
fi
echo "  ok  the pending entry is unchanged"
cp docs/provenance.md "$work/doc.before"
"$PY" scripts/check_provenance.py --write-doc > /dev/null
cmp -s "$work/doc.before" docs/provenance.md || {
    echo "docs/provenance.md is not stable under regeneration; the register moved"
    cp "$work/doc.before" docs/provenance.md
    exit 1
}
echo "  ok  docs/provenance.md regenerates to itself, byte for byte"

say "4/6  this block's reading is present and is not tracked"
f="data/processed/equivalence_october.json"
[ -f "$f" ] || { echo "$f is missing; the fits did not finish"; exit 1; }
if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    echo "$f is tracked, and it must not be"
    exit 1
fi
git check-ignore -q "$f" || { echo "$f is not gitignored"; exit 1; }
echo "  ok  $f present, untracked, ignored"

say "5/6  the suite, ruff, licence hygiene and the provenance gate, in this tree"
"$PY" -m pytest -q
"$RUFF" check .
"$RUFF" format --check .
"$PY" scripts/check_licence_hygiene.py
"$PY" scripts/check_provenance.py

say "6/6  the same, in a scratch replica built read-only from git ls-files"
# Read-only on this repository: git ls-files is the only thing asked of it, and
# the replica is a copy rather than a worktree, so nothing under .git/ is
# written. data/ is absent, which is the condition a fresh clone is in.
mkdir -p "$work/t"
git ls-files -z | grep -zv '^data/' | tar --null -cf - -T - | tar -xf - -C "$work/t"
cp -p "$PWD/scripts/commit_equivalence.sh" "$work/t/scripts/"
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

say "this script names nothing it must not, and carries no trailer"
# The words the licence gate holds as digests, read by the gate's own function,
# and the one more word this script refuses.
if "$PY" -c 'import importlib.util, sys; s = importlib.util.spec_from_file_location("h", "scripts/check_licence_hygiene.py"); h = importlib.util.module_from_spec(s); s.loader.exec_module(h); sys.exit(not h.attribution_in(open(sys.argv[1], encoding="utf-8").read()))' \
        scripts/commit_equivalence.sh \
   || grep -n -i -E '[a]ssistant' scripts/commit_equivalence.sh; then
    echo "the script names something it must not"
    exit 1
fi
echo "ok  clean"

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

git commit -F - <<'MSG'
Ask whether the adopted capacity is available for half the work, and move nothing

Two October fits, on criteria fixed before either figure existed. The first
asks whether the tied cell the one-standard-error rule points at carries the
same capacity at half the terminal nodes; the second asks whether a result
adopted on one seed survives a second.

On cost the answer is yes, and it is recorded as a decision on cost. The
cheaper cell reads 214.4391 against 214.3041, worse by 0.1350 s where the
floor measured in this run is 0.1815 s, so the equivalence rule takes it. It
builds 762,000 terminal nodes against 1,530,000 and fits October in 0.736 of
the time. The cheaper cell is the worse of the two readings and nothing here
presents it as better.

On the seed the criteria did not cover what was measured. The second seed
reads 214.5913 against the first at 214.3041, a gap of 1.58 floors, so the
branch that would have let the result stand as measured did not fire; and
both seeds sit well below the cell they replaced, so the branch that would
have stopped the block did not fire either. The direction survives the seed
and the size does not, and that is left open rather than decided here.

The gate ran first and reproduced the five October single-seed fits on record
to the stored double and the prediction digest, with both floors coming back
at the figures the previous block measured.

Nothing moves. BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED, reports/, docs/ and src/
are all asserted unchanged as whole git trees, the readings are gitignored
beside the sweep's own, and the trees axis stays closed at 6,000.
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
