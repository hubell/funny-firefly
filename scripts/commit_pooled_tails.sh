#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Two commits: the record of the criteria, then the measurement they govern.
#
# The pooled record-quality rule for an airport with no covered cell was closed
# once on the ground that no instrument in the data measures the alternative.
# That ground did not hold. The per-airport defaulted block share by cell,
# which taxiout.tails.TailModel.fit already measures and reports/tails.md
# already prints, is the premise pooling rests on, and it is testable per
# airport and per bucket with no fit and no booster. So the question was
# reopened as an adoptive measurement, its criteria were written into
# docs/plan.md before any of the numbers existed, and the measurement follows.
#
# The order is the point, and this script gates on it: docs/plan.md is older on
# disk than every file the measurement wrote, and it is committed first and by
# itself. A record committed beside the numbers it governs is not a record.
#
# Nothing is adopted. src/taxiout/tails.py is untouched, MIN_COUNT and
# MIN_SHARE are what they were, and this script refuses to run if the working
# tree holds any change under src/. Nothing else regenerates either: every file
# under reports/ but tails.md must be identical to HEAD, and tails.md itself
# must differ from HEAD by the insertion of one section and by nothing else,
# which is checked by rebuilding HEAD's copy from the working one.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# refuses a working tree holding anything but the six expected paths, and
# verifies each commit's own tree in a worktree before finishing. Nothing is
# pushed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

PLAN="docs/plan.md"
REPORT="reports/tails.md"
SCRIPT="scripts/pooled_tails.py"
TESTS="tests/test_pooled_tails.py"
SECTION="## The pooled record-quality rule, measured"

PATHS="$PLAN Makefile $REPORT $SCRIPT $TESTS scripts/commit_pooled_tails.sh"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these six paths and nothing else"
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok"

say "the record precedes the numbers"
# The criteria were fixed before the measurement ran, and on disk that is a
# question about modification times rather than about anyone's word for it.
for f in Makefile "$REPORT" "$SCRIPT" "$TESTS"; do
    if [ "$PLAN" -nt "$f" ]; then
        echo "$PLAN is newer on disk than $f -- the record must precede the numbers"
        exit 1
    fi
done
echo "ok"

say "the plan entry is additive"
# Nothing already in the plan is revised by it. A dated entry that quietly
# rewrote an older one would make the document's history unreadable.
removed="$(git diff --numstat HEAD -- "$PLAN" | cut -f2)"
if [ "${removed:-0}" != "0" ]; then
    echo "$PLAN deletes $removed line(s) -- this entry may only add"
    exit 1
fi
if ! grep -qF "**The pooled record-quality rule, reopened and measured," "$PLAN"; then
    echo "$PLAN does not carry the entry this commit is about"
    exit 1
fi
echo "ok"

say "the term is unchanged"
if [ -n "$(git -c core.fileMode=false status --porcelain -- src/)" ]; then
    echo "src/ differs from HEAD. This measurement adopts nothing and changes no rule."
    exit 1
fi
unchanged() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 no longer carries \"$2\" -- this commit would be adopting something"
        exit 1
    fi
}
unchanged src/taxiout/tails.py "MIN_COUNT = 5"
unchanged src/taxiout/tails.py "MIN_SHARE = 0.05"
unchanged src/taxiout/tails.py "There is no pooled fallback across"
echo "ok"

say "no figure in any other report moved"
others="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- reports/ \
          | sed 's/^...//' | grep -v "^$REPORT$" || true)"
if [ -n "$others" ]; then
    echo "these reports also differ from HEAD, and this commit is not theirs to move:"
    echo "$others"
    exit 1
fi
echo "ok"

say "$REPORT gained one section and changed nothing else"
# The strongest form of that claim is arithmetic rather than a diff read by
# eye: take the section back out of the working copy and what is left has to
# be HEAD's file, byte for byte.
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
git show "HEAD:$REPORT" > "$work/published.md"
python3 - "$work/published.md" "$REPORT" "$SECTION" <<'PY'
import sys
from pathlib import Path

published, working, title = sys.argv[1], sys.argv[2], sys.argv[3]
anchor = "Movement and flight data supplied by EUROCONTROL PRC"

lines = Path(working).read_text(encoding="utf-8").splitlines()
start = [i for i, line in enumerate(lines) if line.strip() == title]
credit = [i for i, line in enumerate(lines) if line.startswith(anchor)]
if len(start) != 1:
    raise SystemExit(f"the working copy carries {len(start)} copies of the section")
if len(credit) != 1:
    raise SystemExit("the working copy has no single data credit to end at")
if not start[0] < credit[0]:
    raise SystemExit("the section is not before the data credit")

head = lines[: start[0]]
while head and not head[-1].strip():
    head.pop()
rebuilt = "\n".join(head + [""] + lines[credit[0] :]) + "\n"
if rebuilt != Path(published).read_text(encoding="utf-8"):
    raise SystemExit("removing the section does not give HEAD's report back")
added = credit[0] - start[0]
print(f"  the section is {added} lines; every other line is HEAD's, byte for byte")
PY
echo "ok"

say "the report is newer than both of its generators"
for g in scripts/report_tails.py "$SCRIPT"; do
    if [ "$g" -nt "$REPORT" ]; then
        echo "$REPORT is older than $g -- rebuild it with 'make tails'"
        exit 1
    fi
done
echo "ok"

say "the report carries what this commit is about"
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\" -- rebuild it with 'make tails'"
        exit 1
    fi
}
check "$REPORT" "$SECTION"
check "$REPORT" "### The premise, per leg and per bucket"
check "$REPORT" "### The stake"
check "$REPORT" "### What a pooled cell would predict"
check "$REPORT" "**The mechanism is at one airport.**"
check "$REPORT" "so the arm was"
check "$REPORT" "### The reading"
# And it still carries what it carried before this commit.
check "$REPORT" "## The callsign-prefix reading, at LIRF"
check "$REPORT" "## What the term is, stated plainly"
# Both generators are wired into the one target, so a rebuild cannot drop the
# section by running only the first of them.
check Makefile "scripts/pooled_tails.py --report"
echo "ok"

say "tests"
python3 -m pytest -q

say "lint"
ruff check .
ruff format --check .

say "licence gate"
python3 scripts/check_licence_hygiene.py

for p in ".cla""ude" "CLA""UDE.md"; do
    if [ -e "$p" ] && ! git check-ignore -q "$p"; then
        echo "$p exists and is not ignored -- fix .gitignore first"
        exit 1
    fi
done

BEFORE="$(git rev-parse HEAD)"

say "1/2 the record"
git add "$PLAN"
git commit -q -F - <<'MSG'
Fix the criteria of the pooled record-quality measurement before making it

The pooled rule for an airport with no covered cell was closed under "Not run,
and why" in the entry above, and one of the grounds given there is that no
instrument in the data measures the alternative. That ground does not hold, and
this entry is the record of reopening the question with its criteria fixed
before any of the numbers they govern exist.

The instrument is the per-airport defaulted block share by cell.
taxiout.tails.TailModel.fit already measures it for every (airport, bucket)
cell it sees, and reports/tails.md already prints the whole table. Pooling
rests on one premise: that a bucket's defaulted share is a property of the
defaulting mechanism and not of the airport, so a cell learned where the
mechanism is visible may be carried to an airport where it is not. That is
directly measurable, per airport and per bucket, on training rows, with no fit
and no booster, and it needs neither the ranking months nor a validation leg.

The entry draws a distinction it would be easy to blur. The share table was
printed in this repository before the question was reopened, so any reading
taken from it now is a reading and is labelled as one. What is pre-registered
is only what the existing numbers do not already answer: the premise test, the
stake, the arm, the adoption rule, that the instrument which decides is the
day-clustered interval and not a seed floor, and the order, which puts the two
screens first and does not fit the arm if the premise fails for every bucket a
pooled cell would serve. The six are quoted in the words they were fixed in.

The entry is committed by itself and ahead of the measurement it governs, and
the commit script gates on the modification times as well as on the order: a
record committed beside the numbers it governs is not a record.
MSG

say "2/2 the measurement"
git add Makefile "$REPORT" "$SCRIPT" "$TESTS" scripts/commit_pooled_tails.sh
git commit -q -F - <<'MSG'
Measure the pooled record-quality rule against its premise, and adopt nothing

The criteria are the entry committed immediately before this one. The
instrument is the per-airport defaulted block share by cell, which
taxiout.tails.TailModel.fit already measures; scripts/pooled_tails.py reads it
on three training halves, counts what a pooled cell would reach in the ranking
set, and writes one section of reports/tails.md. It constructs no booster and
fits nothing but the term itself, so it runs in six seconds.

Screen 1, the premise, fails. A pooled cell is covered by the term's own rule
in two buckets, noflt:4-12h and noflt:>12h, and on all three training halves
the premise fails in both: on the full year, seven of the ten airports tested
in noflt:4-12h and three of the four in noflt:>12h lie outside two standard
errors of the pooled share at their own row counts. The pooled share in those
buckets is 17.8% and 24.0%, and it is one airport's rate diluted by the row
counts of the airports that do not have it: of the 432 training rows the term's
own test picks out, 99.1% are at one airport and only three of the ten carry
any at all. The bucket that holds most of the rows in question, noflt:normal,
pools to 1.4% and is not covered at all, so a pooled cell could not have
reached it whatever its premise said.

Screen 2, the stake, is 235 ranking rows against the 107 the term covers
today, so it is the premise and not the stake that settles this. The largest
single block of them is 138 rows at one airport in January 2026, which is the
case that reopened the question, and the data review's column headed "both" is
confirmed to mean what it was read to mean: no flight-side record and a lag
over four hours, 133 in one late bucket and 5 in the other.

The arm is therefore not fitted, which is what the pre-registered order
requires: fitting an arm whose premise has been refuted would be measuring
something already answered. The section prices what pooling would have done on
rows whose recorded value is known rather than leaving it at an illustration.
On that airport's own 77 training rows in noflt:4-12h a pooled cell predicts
4,512 s against a recorded mean of 824 s, and on its 14 rows in noflt:>12h it
predicts 32,761 s against 1,212 s.

The term is unchanged. src/taxiout/tails.py is untouched, MIN_COUNT and
MIN_SHARE are what they were, no coverage rule moves and nothing is adopted, so
no figure in any other report moves either. The only file that changes is
reports/tails.md, and this commit's own gate takes the new section back out of
it and requires what is left to be the published copy byte for byte.

reports/tails.md now has two generators: scripts/report_tails.py writes it end
to end and scripts/pooled_tails.py writes this one section into it. Running
only the first would drop the section and no figure anywhere would move to say
so, so "make tails" runs both in that order and a test fails if the committed
report has lost it.
MSG

say "each commit's own tree passes its own tests"
for sha in $(git rev-list --reverse "$BEFORE"..HEAD); do
    git worktree add --detach -q "$work/t" "$sha"
    if (cd "$work/t" && PYTHONPATH="$work/t/src" python3 -m pytest -q >/dev/null 2>&1); then
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
echo "Not pushed. Review with: git log -p -2"
