#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit: what the record actually holds behind 31 chosen thresholds, a
# check that a figure quoted beside a report's name is in that report, a check
# that a line a basis cites says what the basis says, and a provenance footer
# for every report cheap enough to regenerate here.
#
# THE COUNT ENDS AT 14 AND THAT IS THE FINISHED STATE. PROVENANCE_PENDING meant
# "not yet accounted for", not "not yet justified". Three of the 31 are fixed
# rather than annotated -- one constant is deleted and imported from the module
# that carries its reason, and two cite the two hours that does carry one.
# Ten record the method, the convention or a neighbouring use and not the
# value. Six are pacing and robustness values that nothing measured depends on,
# and check 13 proves that rather than the basis promising it. The remaining
# fourteen say what the constant does and say plainly that no reason was ever
# written down, which is a REGISTERED state and not a hole. Writing a plausible
# reason after the fact for a threshold nobody argued would be undetectable
# afterwards, and is the failure the register exists to prevent; a test pins the
# count at 14 in both directions so it cannot drift either way.
#
# NO MEASURED FIGURE MOVES. Five reports are regenerated to carry a footer and
# the comparison below finds 0 moved figures in all five, with no exception
# named: the only tokens that appear are the digits inside the footer's own
# input paths and the 256 of sha256. reports/baseline.md is byte for byte
# HEAD's, HEADROOM_TRIGGER_S is still 5.0 and EXPECTED_HEADROOM_S is untouched.
#
# WHAT THIS COMMIT DELIBERATELY DOES NOT DO:
#   * it records no OSM pin. The eleven cache files carry SEVEN different
#     Overpass timestamps spanning 13:23:05Z to 13:32:15Z on 1 September, not
#     one repeated eleven times, so no single --date is the rebuild. The finding
#     is on PROVENANCE_PENDING and the decision is the owner's.
#   * it does not footer reports/geometry_gaps.md or reports/determinism.md,
#     although both are cheap. Neither reproduces from this tree, and a footer
#     records the inputs of the run that wrote the file.
#   * it changes no constant's VALUE anywhere, including the three
#     pre-registration floors. Those are dated quotations of a baseline.md the
#     sweep of 21 September has since regenerated, and only their CITATIONS
#     move, onto the frozen record that still carries the figures.

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

BASE="9e93465"
REPORTS="congestion data_review runway_crosscheck stand_coverage synthetic_airport"
PATHS="docs/provenance.md
reports/congestion.md
reports/data_review.md
reports/runway_crosscheck.md
reports/stand_coverage.md
reports/synthetic_airport.md
scripts/check_provenance.py
scripts/check_runways_ourairports.py
scripts/commit_bases.sh
scripts/report_congestion.py
scripts/report_coverage.py
scripts/report_geometry_gaps.py
scripts/report_synthetic_airport.py
scripts/review_challenge_data.py
src/taxiout/provenance.py
tests/test_provenance.py"

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

say "this sits on the commit it says it sits on"
if ! git merge-base --is-ancestor "$BASE" HEAD 2>/dev/null; then
    echo "$BASE is not an ancestor of HEAD; this script expects to follow it"
    exit 1
fi
echo "ok  $(git log -1 --format='%h %s' "$BASE")"

say "the working tree holds these 16 paths and nothing else"
expected="$(printf '%s\n' "$PATHS" | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok  16 paths"

say "1/6  reports/baseline.md is byte for byte HEAD's"
# The 7 1/4-hour report. Nothing here may touch it, and two pending items wait
# for the run that can.
if ! git diff --quiet HEAD -- reports/baseline.md; then
    echo "reports/baseline.md moved, and this commit spends no ladder run"
    exit 1
fi
echo "ok  unchanged"

say "2/6  HEADROOM_TRIGGER_S is still 5.0"
n="$(grep -c '^HEADROOM_TRIGGER_S = 5\.0$' scripts/run_baselines.py || true)"
if [ "$n" != "1" ]; then
    echo "expected exactly one 'HEADROOM_TRIGGER_S = 5.0'; found $n"
    exit 1
fi
echo "ok  1 hit"

say "3/6  EXPECTED_HEADROOM_S and scripts/run_baselines.py are untouched"
if ! git diff --quiet HEAD -- scripts/run_baselines.py src/taxiout/models.py; then
    echo "the ladder or the model moved, and this commit moves neither"
    exit 1
fi
echo "ok  both unchanged"

say "4/6  src/taxiout/ moves only in the register"
moved="$(git -c core.fileMode=false status --porcelain --untracked-files=all -- src/ \
         | sed 's/^...//' | grep -v '^src/taxiout/provenance.py$' || true)"
if [ -n "$moved" ]; then
    echo "these other modules moved: $moved"
    exit 1
fi
echo "ok  only the register"

say "5/6  no measured figure moved in any report"
# The comparison scripts/commit_preregistered_fixes.sh uses, and the one
# scripts/commit_provenance.sh reused: every number in every touched report,
# HEAD's against the working copy's, aligned. A figure that changed or
# disappeared fails; a figure that appears must be named below.
for r in $REPORTS; do
    git show "HEAD:reports/$r.md" > "$work/$r.head.md"
done
"$PY" - "$work" "$REPORTS" <<'PY'
import difflib
import re
import sys
from pathlib import Path

work, names = Path(sys.argv[1]), sys.argv[2].split()

# A sha256 and a commit hash are identifiers, not figures. Stripping them keeps
# the digits inside a digest out of the comparison, which matters here because
# this commit adds a sha256 table to five reports.
HEX = re.compile(r"\b[0-9a-f]{7,}\b")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
FIGURE = re.compile(r"\.|^\d{3,}$")

# What may CHANGE: nothing. This commit names no replacement, because the
# comparison found none to name.
ALLOWED_REPLACEMENTS: dict[tuple, tuple] = {}

# What may APPEAR: the generation date in a header, and the tokens of the
# provenance footer's own input paths -- a month boundary in a movement or
# METAR file's name, and the 256 of "sha256". No measurement is among them.
ALLOWED_INSERTS = {"2025", "2026", "-01.", "256"}


def numbers(text):
    return [m.group(0) for m in NUM.finditer(HEX.sub(" ", text))]


bad = []
for name in names:
    old = numbers((work / f"{name}.head.md").read_text(encoding="utf-8"))
    new = numbers(Path(f"reports/{name}.md").read_text(encoding="utf-8"))
    ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()
    moved = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag not in ("replace", "delete"):
            continue
        was, now = tuple(old[i1:i2]), tuple(new[j1:j2])
        # A generation date moving by a day is not a measurement: it shows up as
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

say "6/6  the fourteen are fourteen, and the two markers are used as defined"
"$PY" - <<'PY'
import sys
sys.path.insert(0, "src")
from taxiout.provenance import CHOSEN, CONSTANTS, NO_BASIS, VALUE_NOT_JUSTIFIED

unrecorded = [c for c in CONSTANTS if c.cls == CHOSEN and c.basis.startswith(NO_BASIS)]
unjustified = [c for c in CONSTANTS if c.basis.startswith(VALUE_NOT_JUSTIFIED)]
if len(unrecorded) != 14:
    raise SystemExit(
        f"{len(unrecorded)} constants record no reason, and this commit says 14."
        " Neither direction is a correction: fewer means a reason was written"
        " after the fact, more means a constant arrived unargued."
    )
if any(c.module == "scripts/report_geometry_gaps.py" and c.name == "DIRECT_SNAP_M"
       for c in CONSTANTS):
    raise SystemExit("report_geometry_gaps.DIRECT_SNAP_M is back; it must be imported")
print(f"  ok  {len(unrecorded)} record no reason, {len(unjustified)} record the"
      f" method and not the value, {len(CONSTANTS)} entries in all")
PY

say "the gates, in this tree"
"$PY" scripts/check_provenance.py > "$work/provenance.txt" || {
    cat "$work/provenance.txt"
    echo "the provenance gate fails; it cannot be committed failing"
    exit 1
}
grep -cE '^  ok  ' "$work/provenance.txt" | xargs -I{} echo "ok  {} checks green"
if [ "$(grep -cE '^  ok  ' "$work/provenance.txt")" != "13" ]; then
    echo "expected 13 verdicts; a check that is not called cannot fail"
    cat "$work/provenance.txt"
    exit 1
fi

say "--strict still fails, because the pending list is not empty"
if "$PY" scripts/check_provenance.py --strict > "$work/strict.txt" 2>&1; then
    echo "--strict passed, which would mean the freeze condition is met"
    exit 1
fi
echo "ok  $(grep -c '^        [0-9]* --' "$work/strict.txt" || echo '?') check(s) fail under --strict"

say "docs/provenance.md is stable under regeneration"
cp docs/provenance.md "$work/doc.before"
"$PY" scripts/check_provenance.py --write-doc > /dev/null
cmp -s "$work/doc.before" docs/provenance.md || {
    echo "the document is not stable: regenerating it changes it"
    exit 1
}
echo "ok  identical"

say "the suite, ruff and licence hygiene"
"$PY" -m pytest -q
"$RUFF" check .
"$RUFF" format --check .
"$PY" scripts/check_licence_hygiene.py

say "every touched report is newer on disk than the generator that writes it"
# The pairing comes from REPORT_INPUTS_DECLARED and not from which script
# MENTIONS the filename. Generators cite each other's reports in prose
# constantly, so a mention-based pairing names the wrong file -- which is the
# failure check 9 of the gate was rebuilt to avoid, and it was made here first.
"$PY" - "$REPORTS" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, "src")
from taxiout.provenance import REPORT_INPUTS_DECLARED

writes = {report: gen for gen, report in REPORT_INPUTS_DECLARED}
bad = []
for name in sys.argv[1].split():
    report = f"reports/{name}.md"
    gen = writes.get(report)
    if gen is None:
        raise SystemExit(f"{report} is not in REPORT_INPUTS_DECLARED")
    if Path(report).stat().st_mtime < Path(gen).stat().st_mtime:
        bad.append(f"{report} is older than {gen}; regenerate it before committing")
    else:
        print(f"  ok  {report} newer than {gen}")
if bad:
    raise SystemExit("\n".join(bad))
PY

say "the strings this commit is supposed to contain"
check() {
    if ! grep -q "$2" "$1"; then
        echo "missing from $1: $2"
        exit 1
    fi
    echo "ok  $1: $2"
}
check src/taxiout/provenance.py 'VALUE_NOT_JUSTIFIED = "VALUE NOT JUSTIFIED"'
check src/taxiout/provenance.py 'REPORT_CITATION_EXEMPTIONS'
check src/taxiout/provenance.py 'REPORTS_WITHOUT_FOOTER'
check src/taxiout/provenance.py 'the OSM cache is eleven snapshots and not one'
check src/taxiout/provenance.py 'the verdict is not in question'
check scripts/check_provenance.py 'def check_report_citations'
check scripts/check_provenance.py 'def check_register_citations'
check scripts/check_provenance.py 'def check_fetchers_reach_no_report'
check scripts/report_geometry_gaps.py 'from taxiout.graph import DIRECT_SNAP_M'
check tests/test_provenance.py 'assert len(unrecorded) == 14'
check docs/provenance.md '14 is the'

say "staging"
git add -- $PATHS
git -c core.fileMode=false status --short

git commit -F - <<'MSG'
Write down what the record holds behind every chosen threshold, and check the citations

Block 09d registered 31 chosen constants as NOT RECORDED rather than
inventing a basis for them. This is the pass over those 31, and it invents
nothing: for each it says what the record does hold and what it does not.

  - THREE ARE FIXED RATHER THAN ANNOTATED. report_geometry_gaps.py's own
    DIRECT_SNAP_M was a copy of taxiout.graph.DIRECT_SNAP_M, the one that
    carries a reason, so it is deleted and imported; a copy cannot be given
    a basis, only removed. exposure.LAG_HI_S and ARRIVAL_LATE_S cite the two
    hours that does carry a reason, at tails.py line 88, and are cited
    rather than imported because that sentence bounds a taxi and these bound
    a plan lag and a lateness.

  - TEN RECORD THE METHOD AND NOT THE VALUE, marked VALUE NOT JUSTIFIED and
    counted separately: the design is written down and prints the number,
    and no reason for this number rather than a neighbouring one is
    recorded. BOOTSTRAP_LEVEL is the load-bearing one and says so.

  - SIX ARE PACING AND ROBUSTNESS VALUES in the two fetchers, and their
    basis is that nothing measured depends on them. Check 13 proves that
    from the import graph instead of the basis promising it.

  - FOURTEEN SAY WHAT THE CONSTANT DOES AND THAT NO REASON WAS WRITTEN DOWN.
    That is a registered state and the finished number. Driving it to zero
    would mean writing plausible reasons after the fact for thresholds
    nobody argued, which is undetectable afterwards and is the failure this
    register exists to prevent. A test pins it at 14 in both directions.

Two new checks, and both found real defects before they were wired in.

  - Check 11 reads every citation of a report by filename and requires every
    decimal figure in the citation's own sentence to appear in the cited
    file. The scope is the sentence because a paragraph flags figures nobody
    claimed came from the cited file. 80 citations, 4 carrying a figure, 6
    verified and 6 exempt with a reason each; an exemption nothing reaches
    fails too.

  - Check 12 holds the register's own line citations. A basis that names a
    line is a citation and drifts the moment anything above it is edited:
    sixteen of this register's citations were pointing at blank lines and
    unrelated code, six of them broken by the commit that created it, and
    nothing noticed because nothing looked. All sixteen are corrected, and
    ten more in preregistered.py are re-pointed from reports/baseline.md,
    which no longer carries their figures, to the frozen dated record that
    does.

Five reports gain a provenance footer, by the rule that the Makefile times
their generator under five minutes: congestion, data_review,
runway_crosscheck, stand_coverage and synthetic_airport. Six carry none and
are now listed with a cost and an owner, and check 8 fails on a report that
carries none and is on no list.

NO MEASURED FIGURE MOVES. The five regenerated reports compare 0 moved
against HEAD across 4,400 published figures; the only tokens that appear are
the digits inside the footer's own input paths. reports/baseline.md is byte
for byte HEAD's, HEADROOM_TRIGGER_S is still 5.0 and the ladder is untouched.

One finding is recorded and deliberately not acted on: the OSM cache is
eleven snapshots at seven different Overpass timestamps rather than one
moment eleven times, so no single --date reproduces it and no pin is recorded
here. PROVENANCE_PENDING goes from eight items to six.
MSG

say "the commit"
git log -1 --stat | head -40

say "the commit's own tree, in a worktree with data/ absent"
# The condition 221a7a8 failed in, and the check that caught it. This one keeps
# pytest's output and runs the gate as well as the suite.
wt="$work/worktree"
git worktree add -q --detach "$wt" HEAD
(
    cd "$wt"
    "$PY" -m pytest -q 2>&1 | tail -20
    "$PY" scripts/check_provenance.py 2>&1 | grep -E "^  (ok|FAIL)|^provenance|^FAIL"
) || {
    git worktree remove --force "$wt"
    echo "the committed tree does not pass its own gates."
    exit 1
}
git worktree remove --force "$wt"

say "done"
echo "Nothing is pushed. 'git show HEAD' to read it, 'git reset --hard HEAD^' to undo."
