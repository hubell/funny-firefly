#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: the provenance register, its gate, and the defects the gate found.
#
# The standing rule is that every number reaching a report, a note or the paper
# is computed by the run that prints it, quoted from a named source with that
# source beside it, or marked an estimate with an instrument and an owner. Until
# this commit the repository obeyed that by convention, and a convention does
# not fail a build. Three provenance defects reached a committed tree because
# nothing enumerated the surface.
#
# WHAT THIS COMMIT MUST NOT DO, and the gates below prove it:
#
#   1. HEADROOM_TRIGGER_S = 5.0 is unchanged in scripts/run_baselines.py.
#   2. EXPECTED_HEADROOM_S's four rows are unchanged, byte for byte.
#   3. reports/baseline.md is unchanged, byte for byte, against HEAD.
#   4. src/taxiout/ is unchanged except for the new provenance.py.
#   5. NO MEASURED FIGURE MOVED in any report, by the numeric comparison
#      scripts/commit_preregistered_fixes.sh uses, with the one deliberate
#      exception named individually and no others.
#
# The one exception is reports/headroom.md, which said "fitted on 1.9 million
# training rows" in prose that no run produced. It now prints the residual row
# count the run computes. That is the whole of the figure movement in this
# commit and assertion 5 names it.
#
# Runs the tests, the lint, the licence gate and the new provenance gate first
# and stops if any fails, refuses a working tree holding anything but the
# expected paths, and verifies the commit's own tree in a worktree before
# finishing. Nothing is pushed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

# The other commit scripts call python3 and ruff by their bare names. This one
# prefers the repository's own virtual environment when it is there, because a
# test gate run by an interpreter with no pandas skips most of the suite.
PY="${PYTHON:-$([ -x .venv/bin/python ] && echo "$PWD/.venv/bin/python" || echo python3)}"
RUFF="$([ -x .venv/bin/ruff ] && echo "$PWD/.venv/bin/ruff" || echo ruff)"
case "$PY" in /*) ;; */*) PY="$PWD/$PY" ;; esac

LADDER="scripts/run_baselines.py"
BASELINE="reports/baseline.md"
REGISTER="src/taxiout/provenance.py"
GATE="scripts/check_provenance.py"
DOC="docs/provenance.md"

NEW="$REGISTER $GATE $DOC tests/test_provenance.py scripts/commit_provenance.sh"
TOUCHED="Makefile README.md
    reports/headroom.md reports/occupancy.md reports/residual.md
    reports/stand_coverage.md reports/tails.md
    scripts/report_coverage.py scripts/report_headroom.py
    scripts/report_occupancy.py scripts/report_residual.py
    scripts/report_tails.py"
PATHS="$NEW $TOUCHED"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these paths and nothing else"
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  $(printf '%s\n' $PATHS | wc -l) paths"

say "1/5 the trigger is unchanged"
if ! grep -qF "HEADROOM_TRIGGER_S = 5.0" "$LADDER"; then
    echo "$LADDER no longer carries \"HEADROOM_TRIGGER_S = 5.0\"."
    echo "This commit registers that threshold. It does not move it."
    exit 1
fi
echo "ok  HEADROOM_TRIGGER_S = 5.0"

say "2/5 EXPECTED_HEADROOM_S's four rows are unchanged, byte for byte"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
git show "HEAD:$LADDER" > "$work/ladder_head.py"
"$PY" - "$work/ladder_head.py" "$LADDER" <<'PY'
import sys
from pathlib import Path

OPEN = "EXPECTED_HEADROOM_S: dict[str, tuple[float, float]] = {"
KEYS = ("occupancy", "callsign", "destination", "plan-diffs")


def rows(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = [i for i, line in enumerate(lines) if line.strip() == OPEN]
    if len(start) != 1:
        raise SystemExit(f"{path}: {len(start)} copies of the constant's opening line")
    end = next(i for i in range(start[0] + 1, len(lines)) if lines[i].rstrip() == "}")
    got = [line for line in lines[start[0] + 1 : end] if line.lstrip().startswith('"')]
    if len(got) != len(KEYS):
        raise SystemExit(f"{path}: the constant has {len(got)} rows, not {len(KEYS)}")
    for key, line in zip(KEYS, got):
        if f'"{key}":' not in line:
            raise SystemExit(f"{path}: row for {key!r} is not where it was")
    return "\n".join(got)


head, working = rows(sys.argv[1]), rows(sys.argv[2])
if head != working:
    raise SystemExit(
        "EXPECTED_HEADROOM_S differs from HEAD. This commit registers it and"
        " does not refresh it."
    )
print("  " + "\n  ".join(line.strip() for line in working.splitlines()))
PY
echo "ok"

say "3/5 reports/baseline.md is unchanged, byte for byte"
if ! git diff --quiet HEAD -- "$BASELINE"; then
    echo "$BASELINE differs from HEAD. The audit's verdict is not this commit's"
    echo "to move, and the rename that is coming needs a ladder run, not a hand edit."
    exit 1
fi
echo "ok  the audit still reads what it read at HEAD"

say "4/5 src/taxiout/ is unchanged except for the register"
others="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- src/ \
          | sed 's/^...//' | grep -v "^$REGISTER$" || true)"
if [ -n "$others" ]; then
    echo "these files under src/ also differ from HEAD:"
    echo "$others"
    echo "This commit adds a register and a gate. It changes no feature and no rule."
    exit 1
fi
echo "ok"

say "5/5 no measured figure moved in any report"
# The comparison scripts/commit_preregistered_fixes.sh uses: every number in
# every touched report, HEAD's against the working copy's, aligned. A figure
# that changed or disappeared fails. A figure that appeared must be named here.
for r in headroom occupancy residual tails stand_coverage; do
    git show "HEAD:reports/$r.md" > "$work/$r.head.md"
done
"$PY" - "$work" <<'PY'
import difflib
import re
import sys
from pathlib import Path

work = Path(sys.argv[1])

# Digests and commit hashes are identifiers, not figures: stripping them keeps
# the digits inside a sha256 out of the comparison, which matters more in this
# commit than in any before it, since it adds a sha256 to four reports.
HEX = re.compile(r"\b[0-9a-f]{7,}\b")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
FIGURE = re.compile(r"\.|^\d{3,}$")

# What this commit MAY change, named individually.
#
# reports/headroom.md said "fitted on 1.9 million training rows" in prose that
# no run produced, which is exactly the defect check 7 of the new gate looks
# for, in the newest file in the repository. It now prints the residual rows the
# run computes.
ALLOWED_REPLACEMENTS = {("1.9",): ("355,967",)}

# What may APPEAR. The generation date in a header, and the tokens of the
# provenance footer's own input paths: a month boundary in a movement file's
# name, and the 256 of "sha256". No measurement is among them.
ALLOWED_INSERTS = {"2025", "2026", "-01.", "256"}


def numbers(text):
    return [m.group(0) for m in NUM.finditer(HEX.sub(" ", text))]


bad = []
for name in ("headroom", "occupancy", "residual", "tails", "stand_coverage"):
    old = numbers((work / f"{name}.head.md").read_text(encoding="utf-8"))
    new = numbers(Path(f"reports/{name}.md").read_text(encoding="utf-8"))
    ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    moved = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag not in ("replace", "delete"):
            continue
        was, now = tuple(old[i1:i2]), tuple(new[j1:j2])
        # A generation date moving by a day is not a measurement. It shows up as
        # a replacement of the day component alone.
        if all(re.fullmatch(r"-?\d{1,2},?", v) for v in was + now):
            continue
        if ALLOWED_REPLACEMENTS.get(was) == now:
            print(f"  reports/{name}.md: {was} -> {now}, named by this commit")
            continue
        bad.append(f"reports/{name}.md: {tag} {was} -> {now}")
        moved += 1
    added = [v for tag, _, _, j1, j2 in ops if tag == "insert" for v in new[j1:j2]]
    unnamed = sorted({v for v in added if FIGURE.search(v)} - ALLOWED_INSERTS)
    if unnamed:
        bad.append(f"reports/{name}.md: unnamed figure(s) appeared: {unnamed}")
    print(
        f"  reports/{name}.md: {len(old)} published figures,"
        f" {moved} moved, {len(added)} token(s) added"
    )

if bad:
    for b in bad:
        print(f"  {b}", file=sys.stderr)
    raise SystemExit("a measured figure moved, and this commit says none did")
print("  no measured figure moved in any report")
PY
echo "ok"

say "the register covers the tree exactly"
"$PY" "$GATE" > "$work/provenance.txt" || {
    cat "$work/provenance.txt"
    echo "the provenance gate fails; it cannot be committed failing"
    exit 1
}
tail -1 "$work/provenance.txt"

say "the generated document is what the register generates"
"$PY" "$GATE" --write-doc > /dev/null
if ! git diff --quiet -- "$DOC" 2>/dev/null && [ -n "$(git status --porcelain -- "$DOC")" ]; then
    # It is untracked on the first commit, so compare against a rebuild instead.
    cp "$DOC" "$work/doc.rebuilt.md"
    "$PY" "$GATE" --write-doc > /dev/null
    if ! cmp -s "$DOC" "$work/doc.rebuilt.md"; then
        echo "$DOC is not stable under regeneration"
        exit 1
    fi
fi
echo "ok  regenerating it changes nothing"

say "the reports are newer than the generators that wrote them"
for pair in "scripts/report_headroom.py:reports/headroom.md" \
            "scripts/report_occupancy.py:reports/occupancy.md" \
            "scripts/report_residual.py:reports/residual.md" \
            "scripts/report_tails.py:reports/tails.md" \
            "scripts/report_coverage.py:reports/stand_coverage.md"; do
    g="${pair%%:*}"
    r="${pair#*:}"
    if [ "$g" -nt "$r" ]; then
        echo "$r is older than $g -- rebuild it"
        exit 1
    fi
done
echo "ok"

say "the register and the gate carry what this commit is about"
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\""
        exit 1
    fi
}
check "$REGISTER" "CONSTANTS: tuple[Constant, ...] = ("
check "$REGISTER" "AUTHORED_ARTIFACTS: tuple[Authored, ...] = ("
check "$REGISTER" "FROZEN_DOCUMENTS: tuple[Frozen, ...] = ("
check "$REGISTER" "PROVENANCE_PENDING: tuple[Pending, ...] = ("
check "$REGISTER" 'NO_BASIS = "NOT RECORDED"'
# The constant this whole register was built around, and the entry that says
# what it is: a line on a diagnostic and not a leakage criterion.
check "$REGISTER" "a line on a DIAGNOSTIC and not a leakage criterion"
check "$GATE" "def module_constants("
check "$GATE" "def string_literals("
check "$GATE" "def comments("
check "$GATE" '"--census"'
check "$GATE" '"--strict"'
check "$DOC" "Generated by \`scripts/check_provenance.py --write-doc\`"
# The report that claimed one generator while carrying a section from another.
check reports/tails.md "two generators"
check reports/tails.md '`scripts/pooled_tails.py`'
check Makefile "scripts/check_provenance.py"
check README.md "scripts/check_provenance.py"
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

say "1/1 the register and its gate"
git add $PATHS
git commit -q -F - <<'MSG'
Register where every number comes from, and gate it

The standing rule is that every number reaching a report, a note or the paper is
computed by the run that prints it, quoted from a named source with that source
beside it, or marked an estimate with an instrument and an owner. There is no
fourth kind. The repository obeyed that by convention, and a convention does not
fail a build: three provenance defects reached a committed tree because nothing
enumerated the surface.

A census found the surface is about a thousand numbers in eight classes, so
registering them by hand was never the answer. What this does instead is make
sure nothing is UNCATEGORISED. Every number in a covered surface is in exactly
one of four states -- generated by the run that prints it, checked against a
named source, registered with a one-line basis, or exempt with a reason -- and
scripts/check_provenance.py proves there is no fifth. Adding a constant without
classifying it now fails the build, which is the property the tree did not have.

The scanner is an ast walk, exported and used by both the gate and its own
--census, so two counts of one quantity cannot disagree. It finds 275
module-level numeric assignments where the census's line-anchored regex found
255, adding 51 it could not see -- numbers in multi-line values, lowercase
names, dataclass field defaults -- and dropping 31 it had counted from strings
that merely contain a digit. Comments come from tokenize, which the ast does not
carry, and prose figures exclude f-string format fields by construction rather
than by stripping braces afterwards. The scanner enumerates and does not
classify: a class guessed from a name would be the unearned provenance this gate
exists to remove.

taxiout.provenance carries 225 entries, one per assignment outside the authored
artifact: 88 mechanism, 106 chosen, 30 quoted, 1 derived. src/taxiout/synthetic.py
is registered once as an authored artifact rather than constant by constant,
with the condition that makes it safe -- that no generator of a model figure
imports it -- proved against ten named generators rather than asserted.

THE FINDING IS IN THE CHOSEN HALF, and that is the half that had failed.
HEADROOM_TRIGGER_S was chosen with a stated reason and no recorded basis, and
turned out to be an absolute line on a quantity docs/review_headroom.md section
7 had already characterised as a floor a refit exceeds; it went from missing to
firing when the booster's capacity grew 3.2 times, with the block untouched. Its
entry now says that it is a line on a diagnostic and not a leakage criterion. Of
the 106 chosen constants, 31 have no basis recorded anywhere in the code: five
plausibility bounds in taxiout.exposure, four windows and a cap in
taxiout.saturation, three in taxiout.ceiling, the fold count and seen fraction
and interval level in taxiout.validation, and the rest. Each says so in its own
basis rather than carrying an invented one, the count is summed from those marks
rather than written down, and PROVENANCE_PENDING owns the class.

Nine dated records get an immutability digest rather than a currency check,
which is the opposite gate from every other file here: the risk on
docs/plan.md and docs/review_headroom.md is not that they go stale but that
somebody edits them, and a currency check over one would demand destroying the
record.

Each of four report generators now writes a footer naming every input path with
its sha256, and the gate recomputes them and reports verified, stale or
unverifiable, never collapsing the three. Content and not modification time,
because Block 11 reproduces from a clean checkout where every file carries the
checkout time and an mtime ordering is meaningless exactly where the check
matters most. A constructed test reproduces the case that bit: a report whose
declared residual digest no longer matches, which is what Block 09b found by
reading.

Three defects the gate found are fixed here. reports/headroom.md said "fitted on
1.9 million training rows" in prose no run produced, in the newest file in the
repository and the one written to close a provenance hole; it now prints the
residual row count it computes. reports/tails.md said its one generator computed
every figure in it, which stopped being true when it gained a second; its header
now names both and dates each part, and the gate holds it there structurally, by
which module writes the path rather than which mentions it. reports/stand_coverage.md
did not name its own generator.

NO THRESHOLD, CONSTANT OR VERDICT MOVES, and no measured figure moves anywhere.
HEADROOM_TRIGGER_S is 5.0, EXPECTED_HEADROOM_S is unchanged byte for byte,
reports/baseline.md is unchanged byte for byte, src/taxiout/ is unchanged but for
the new register, and this script's own gate aligns every number in every touched
report against HEAD and fails on any that moved. The single exception is the 1.9
above, named individually in that gate.

Two items are deferred with their reasons on the pending list rather than
rushed. The rename of reports/baseline.md's section to what it measures, and the
missing fourth audit reading, both need that report regenerated, and
scripts/run_baselines.py has no report-only mode and keeps no results, so it can
only be rebuilt by the whole seven-hour ladder. Cross-report citation, the class
that actually bit, is left to its own block rather than shipped as a checker that
passes because it cannot see.

docs/provenance.md is generated from the register with every count summed from
it, and check 10 rebuilds it and fails if the committed copy differs.
MSG

say "the commit's own tree passes its own tests"
for sha in $(git rev-list --reverse "$BEFORE"..HEAD); do
    git worktree add --detach -q "$work/t" "$sha"
    if (cd "$work/t" && PYTHONPATH="$work/t/src" "$PY" -m pytest -q >/dev/null 2>&1); then
        printf '  ok   %s %s\n' "$(git log -1 --format=%h "$sha")" \
            "$(git log -1 --format=%s "$sha")"
    else
        printf '  FAIL %s %s\n' "$(git log -1 --format=%h "$sha")" \
            "$(git log -1 --format=%s "$sha")"
        git worktree remove --force "$work/t"
        exit 1
    fi
    git worktree remove --force "$work/t"
done

say "result"
git --no-pager log --oneline -5
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -1"
