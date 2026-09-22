#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: repair 221a7a8, whose own tree failed its own tests.
#
# 221a7a8 added the provenance register and its gate and was committed with a
# tree that does not pass pytest. Two defects, both mine, and both of the same
# shape -- a check that had never been asked the question it exists to ask:
#
#   1. THE GATE HAD NEVER SCANNED ITSELF. The scanner reads git ls-files, which
#      is right, and src/taxiout/provenance.py and scripts/check_provenance.py
#      were both untracked while the gate was validated. Committing them put
#      four assignments and twenty-nine prose figures in scope at once, and
#      checks 1 and 7 failed on the committed tree.
#
#   2. CHECK 10 WAS NEVER WIRED IN. check_document was written and tested
#      directly and never added to the tuple main iterates, so the gate ran nine
#      checks and called itself clean. A check that is not called cannot fail,
#      and its silence reads as a pass, which is the exact failure this block
#      exists to remove.
#
# Both are now fixed, and both are held by a test rather than by care:
# test_the_gate_and_the_register_are_in_their_own_scope and
# test_every_check_the_gate_defines_is_wired_into_its_run. main also refuses to
# run if the number of checks defined and the number wired disagree.
#
# A third defect is fixed here too, in this script rather than in the tree: the
# worktree check at the end of scripts/commit_provenance.sh discarded pytest's
# output, so the failure it found printed one word and no reason. This one shows
# it, and runs the suite in a tree with data/ absent, which is the condition
# under which 221a7a8 failed.
#
# NO REPORT CHANGES AT ALL in this commit, which is stronger than the no-figure
# comparison 221a7a8 needed: the only generated file that moves is
# docs/provenance.md, and it moves because the register did.

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

BROKEN="221a7a8"
REGISTER="src/taxiout/provenance.py"
GATE="scripts/check_provenance.py"
DOC="docs/provenance.md"
TESTS="tests/test_provenance.py"
PATHS="$REGISTER $GATE $DOC $TESTS scripts/commit_provenance_fixes.sh"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "this repairs the commit it says it repairs"
if ! git merge-base --is-ancestor "$BROKEN" HEAD 2>/dev/null; then
    echo "$BROKEN is not an ancestor of HEAD; this script is for repairing it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' "$BROKEN")"

say "the working tree holds these five paths and nothing else"
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  5 paths"

say "no report changes at all"
# Stronger than 221a7a8's numeric comparison, and it can be stronger because
# this commit regenerates nothing: it repairs a gate and a register.
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- reports/)"
if [ -n "$moved" ]; then
    echo "these reports differ from HEAD, and this commit touches no report:"
    echo "$moved"
    exit 1
fi
echo "ok  reports/ is identical to HEAD"

say "no threshold, constant or verdict moves"
if ! grep -qF "HEADROOM_TRIGGER_S = 5.0" scripts/run_baselines.py; then
    echo "scripts/run_baselines.py no longer carries HEADROOM_TRIGGER_S = 5.0"
    exit 1
fi
for f in scripts/run_baselines.py reports/baseline.md; do
    if ! git diff --quiet HEAD -- "$f"; then
        echo "$f differs from HEAD, and it is not this commit's to change"
        exit 1
    fi
done
others="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- src/ \
          | sed 's/^...//' | grep -v "^$REGISTER$" || true)"
if [ -n "$others" ]; then
    echo "these files under src/ also differ from HEAD:"
    echo "$others"
    exit 1
fi
echo "ok  the trigger, the ladder, baseline.md and the rest of src/ are untouched"

say "defect 1: the gate is in its own scope, and categorised"
"$PY" - <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import check_provenance as cp
from taxiout.provenance import CONSTANTS, REGISTER_MODULE

modules = cp.scanned_modules()
for wanted in ("scripts/check_provenance.py", REGISTER_MODULE):
    if wanted not in modules:
        raise SystemExit(f"{wanted} is not in the scanner's own scope")
if not any(c.module == "scripts/check_provenance.py" for c in CONSTANTS):
    raise SystemExit("the gate's own constants are not registered")
if any(c.module == REGISTER_MODULE for c in CONSTANTS):
    raise SystemExit("the register has an entry for itself, which cannot be right")
if REGISTER_MODULE not in cp.UNREGISTERED_MODULES:
    raise SystemExit("the register is not excluded by name")
ok, lines = cp.check_categorised(cp.module_constants())
if not ok:
    print("\n".join(lines), file=sys.stderr)
    raise SystemExit("check 1 does not pass on this tree")
print(f"  both files scanned; {len(CONSTANTS)} entries; check 1 passes")
PY
echo "ok"

say "defect 2: every check the gate defines is wired into its run"
"$PY" - <<'PY'
import inspect
import sys

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import check_provenance as cp

source = inspect.getsource(cp.main)
defined = {
    n for n in dir(cp) if n.startswith("check_") and callable(getattr(cp, n))
} - {"check_scope"}
missing = sorted(n for n in defined if f"{n}(" not in source)
if missing:
    raise SystemExit(f"defined and never called: {missing}")
print(f"  {len(defined)} checks defined, all {len(defined)} called by main")
PY
echo "ok"

say "the gate runs ten checks and passes"
out="$("$PY" "$GATE")" || { echo "$out"; exit 1; }
ran="$(printf '%s\n' "$out" | grep -cE '^  (ok|FAIL) ')"
if [ "$ran" != "10" ]; then
    echo "the gate printed $ran check verdicts, not 10"
    printf '%s\n' "$out" | grep -E '^  (ok|FAIL) ' || true
    exit 1
fi
printf '%s\n' "$out" | tail -1

say "--strict still fails on the pending list, which is the freeze condition"
if "$PY" "$GATE" --strict > /dev/null 2>&1; then
    echo "--strict passes, so the pending list is empty or the flag is inert"
    exit 1
fi
echo "ok"

say "the document is what the register generates, and is stable"
"$PY" "$GATE" --write-doc > /dev/null
if [ -n "$(git -c core.fileMode=false status --porcelain -- "$DOC")" ]; then
    before="$(sha256sum "$DOC" | cut -d' ' -f1)"
    "$PY" "$GATE" --write-doc > /dev/null
    if [ "$before" != "$(sha256sum "$DOC" | cut -d' ' -f1)" ]; then
        echo "$DOC is not stable under regeneration"
        exit 1
    fi
fi
echo "ok  regenerating it twice changes nothing"

say "the fixes carry what this commit is about"
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\""
        exit 1
    fi
}
check "$REGISTER" 'REGISTER_MODULE = "src/taxiout/provenance.py"'
check "$REGISTER" "cannot be an entry in itself"
check "$GATE" "UNREGISTERED_MODULES: frozenset[str]"
check "$GATE" "10 -- the generated document matches the manifests"
check "$GATE" "its silence reads as a"
check "$TESTS" "def test_the_gate_and_the_register_are_in_their_own_scope"
check "$TESTS" "def test_every_check_the_gate_defines_is_wired_into_its_run"
echo "ok"

say "tests"
"$PY" -m pytest -q

say "lint"
"$RUFF" check .
"$RUFF" format --check .

say "licence gate"
"$PY" scripts/check_licence_hygiene.py

for p in ".cla""ude" "CLA""UDE.md"; do
    if [ -e "$p" ] && ! git check-ignore -q "$p"; then
        echo "$p exists and is not ignored -- fix .gitignore first"
        exit 1
    fi
done

BEFORE="$(git rev-parse HEAD)"

say "1/1 the repair"
git add $PATHS
git commit -q -F - <<'MSG'
Put the provenance gate in its own scope, and wire in the check it defined

221a7a8 added the register and the gate and was committed with a tree that does
not pass its own tests. Two defects, both of one shape: a check that had never
been asked the question it exists to ask.

THE GATE HAD NEVER SCANNED ITSELF. The scanner reads git ls-files, which is the
right list, and src/taxiout/provenance.py and scripts/check_provenance.py were
both untracked for the whole of the work. So the gate was validated against a
tree that did not contain it. Committing the two files put four assignments and
twenty-nine prose figures in scope at once: check 1 failed on
scripts/check_provenance.py's own REPO and on the register's three aggregate
assignments, and check 7 failed on the exemption table's own entries and on
every basis that quotes a measurement with its source.

The register's own constants are the one case that cannot be registered. Its
aggregate assignments hold every registered literal in the repository between
them, so an entry for them would be a copy of the register inside itself and
would drift the moment any other entry changed. REGISTER_MODULE names it, the
gate excludes it by name through UNREGISTERED_MODULES, and SCOPE says so and why
-- which is a different exclusion from the authored artifact's and is stated
separately. The gate's own REPO joins the twenty-five other scripts' in the
repo-root pattern, and its prose is excluded for the reason its entry gives: its
figures are the exemption table, which is a list of figures by construction, and
it writes no report. The register's prose is excluded for the reason that a
basis IS a provenance record -- "scored 214.90 s on October against 217.28 s" is
where BOOSTER_PARAMS came from, not an unsourced figure in a report -- and it
writes no report either.

CHECK 10 WAS NEVER WIRED IN. check_document was written, tested directly, and
never added to the tuple main iterates, so the gate ran nine checks, printed
nine verdicts and called itself clean. A check that is not called cannot fail,
and its silence reads as a pass, which is the precise failure this whole block
exists to remove, made by the block itself. It now runs, and main refuses to run
at all if the number of checks defined and the number wired disagree.

Both are held by tests rather than by care, and both tests say in their
docstrings that the failure happened:
test_the_gate_and_the_register_are_in_their_own_scope asserts that the scanner
sees both files and that each is in exactly one state, and
test_every_check_the_gate_defines_is_wired_into_its_run asserts that every
check_ function appears in main and that the printed numbers run from one with
no gaps.

Verified in a tree with data/ absent, which is the condition 221a7a8 failed
under and which a clean checkout and every worktree check share: 693 tests pass
and all ten checks are green. The suite was 692 at 221a7a8's tree once these
fixes are applied and is 693 with the second guard.

NO REPORT CHANGES AT ALL, which is stronger than the numeric comparison 221a7a8
needed and can be, because this repairs a gate and regenerates nothing.
reports/ is byte for byte HEAD's, HEADROOM_TRIGGER_S is 5.0, reports/baseline.md
and scripts/run_baselines.py are untouched, and src/taxiout/ moves only in the
register. The one generated file that changes is docs/provenance.md, and it
changes because the register did.
MSG

say "the commit's own tree passes its own tests, with data absent"
# 221a7a8's worktree check discarded pytest's output, so the failure it found
# printed one word and no reason. This one keeps it. The worktree carries no
# data/ directory, gitignored as it is, which is exactly the condition that
# exposed both defects.
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
for sha in $(git rev-list --reverse "$BEFORE"..HEAD); do
    git worktree add --detach -q "$work/t" "$sha"
    if (cd "$work/t" && PYTHONPATH="$work/t/src" "$PY" -m pytest -q 2>&1) \
            > "$work/pytest.txt"; then
        printf '  ok   %s %s\n' "$(git log -1 --format=%h "$sha")" \
            "$(tail -1 "$work/pytest.txt")"
    else
        printf '  FAIL %s %s\n' "$(git log -1 --format=%h "$sha")" \
            "$(git log -1 --format=%s "$sha")"
        echo "--- pytest, in the worktree ---"
        cat "$work/pytest.txt"
        git worktree remove --force "$work/t"
        exit 1
    fi
    # And the gate itself, in that same tree.
    if (cd "$work/t" && "$PY" scripts/check_provenance.py > "$work/gate.txt" 2>&1); then
        printf '  ok   %s\n' "$(tail -1 "$work/gate.txt")"
    else
        echo "--- the provenance gate, in the worktree ---"
        cat "$work/gate.txt"
        git worktree remove --force "$work/t"
        exit 1
    fi
    git worktree remove --force "$work/t"
done

say "result"
git --no-pager log --oneline -4
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -1"
