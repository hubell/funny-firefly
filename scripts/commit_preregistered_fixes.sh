#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# One commit, correcting the prose of the pre-registered experiments' report.
#
# The experiments themselves were committed as 16a4f03. No figure in that
# commit is wrong and none moves here: the report is rebuilt from the same 18
# records under data/processed/preregistered/, with no refit, and this script
# refuses to run if any figure that was published has changed or gone. Three
# descriptions in 16a4f03's message are wrong, and that message is not
# rewritten, here or ever: this commit's message says which three and why, and
# the report carries a dated paragraph recording the one that matters.
#
# Nothing is adopted. BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED, the season columns
# and SEED_AVERAGE_PENDING are read out of the modules before anything is
# staged, and the check on SEED_AVERAGE_PENDING now carries the whole line,
# because a check on its declaration alone passes on the empty tuple, which is
# the one change the pre-registration forbids until after the sweep's ladder.
#
# docs/plan.md is not touched. Its entry is the record the criteria were fixed
# in, it is committed as f7cb53c, and this script refuses to run if the working
# tree holds it. Anything ever added to that entry is additive, dated, and
# marked as added after the fact, so that a diff against f7cb53c reads as an
# annotation and not as a revision.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# refuses a working tree holding anything but the four expected paths, and
# verifies the commit's own tree in a worktree before finishing. Nothing is
# pushed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

REPORT="reports/preregistered-2026-09-20.md"
PUBLISHED="16a4f03"

PATHS="scripts/preregistered.py $REPORT tests/test_preregistered.py
scripts/commit_preregistered_fixes.sh"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these four paths and nothing else"
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok"

say "docs/plan.md is untouched"
# The entry is the record that the criteria preceded the numbers. Editing it
# in the same commit as a correction to the report would put the record and
# the thing it governs in one change, and the mtime gate that protected it has
# already run and passed, so nothing but this check stands between the entry
# and a quiet revision.
if [ -n "$(git -c core.fileMode=false status --porcelain -- docs/plan.md)" ]; then
    echo "docs/plan.md differs from HEAD. It is not this commit's business."
    exit 1
fi
echo "ok"

say "no figure moved against the copy published in $PUBLISHED"
# The before image is the published text itself, read out of the object rather
# than kept beside the repository. Every number in it must still be in the
# rebuilt report, in the same order and unchanged; the only permitted
# difference is an insertion, and every inserted figure is named below. A
# rebuild that moved a figure would mean the records were refitted, or read
# differently, which is the one thing this pass must not do.
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
git show "$PUBLISHED:$REPORT" > "$work/published.md"
python3 - "$work/published.md" "$REPORT" <<'PY'
import difflib
import re
import sys
from pathlib import Path

# Digests and commit hashes are identifiers, not figures. Stripping them keeps
# the digits inside a sha256 out of the comparison.
HEX = re.compile(r"\b[0-9a-f]{7,}\b")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
FIGURE = re.compile(r"\.|^\d{3,}$")

# Every figure this pass adds, named. Anything else is a failure.
ALLOWED = {
    # the floors, now printed beside the steps they govern
    "0.19", "0.23", "0.34", "0.40", "1.02",
    # the rule of 19 September on its two forward legs, quoted from baseline.md
    "0.48", "0.52", "217.11", "217.6", "325.8", "325.29",
    "1020", "1021", "1023", "1023,",
    # what the unrounded figures do to the printed subtraction
    "1.00", "1.01", "236.80", "237.80",
    # the corrected reading of the gap step, and its interval
    "0.10", "-0.4", "0.6",
    # printed precision, said where the floors are
    "0.1",
    # the per-fit wall sum, against the run's wall clock
    "863",
    # the paper's section, named where the two instruments disagree
    "7.1",
    # the substitution counts, computed from the manifest
    "13", "18", "15", "10",
    # the year, in the sentence about the July 2026 geometry
    "2026",
    # the digest's algorithm, in the word sha256, which is not a figure
    "256",
}


def numbers(text):
    return [m.group(0) for m in NUM.finditer(HEX.sub(" ", text))]


old = numbers(Path(sys.argv[1]).read_text(encoding="utf-8"))
new = numbers(Path(sys.argv[2]).read_text(encoding="utf-8"))
ops = difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes()

moved = [op for op in ops if op[0] in ("replace", "delete")]
if moved:
    for tag, i1, i2, j1, j2 in moved:
        print(f"  {tag}: {old[i1:i2]} -> {new[j1:j2]}", file=sys.stderr)
    raise SystemExit("a figure that was published has changed or gone")

added = [v for tag, _, _, j1, j2 in ops if tag == "insert" for v in new[j1:j2]]
unnamed = sorted({v for v in added if FIGURE.search(v) and v not in ALLOWED})
if unnamed:
    print("  not named by this pass: " + ", ".join(unnamed), file=sys.stderr)
    raise SystemExit("an unnamed figure appeared in the report")
print(f"  {len(old)} published figures all present and unchanged, {len(added)} added")
PY
echo "ok"

say "this commit adopts nothing"
unchanged() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 no longer carries \"$2\" -- this commit would be adopting something"
        exit 1
    fi
}
unchanged src/taxiout/models.py 'SEED_AVERAGE_ADOPTED: bool = False'
unchanged src/taxiout/models.py '"random_state": 20260902'
unchanged src/taxiout/models.py '"n_estimators": 1000'
unchanged src/taxiout/models.py '"learning_rate": 0.03'
unchanged src/taxiout/models.py '"num_leaves": 127'
unchanged src/taxiout/models.py '"min_child_samples": 50'
# The whole line, and not the declaration. A check on the declaration alone
# passes on an empty tuple, which is exactly the change the pre-registration
# forbids until the ladder that follows the sweep has printed the two
# measurements the rule of 19 September asks for.
unchanged scripts/run_baselines.py \
    'SEED_AVERAGE_PENDING: tuple[str, ...] = ("the three July airports", "October")'
for c in tko_month tko_doy_sin tko_doy_cos; do
    unchanged src/taxiout/features.py "\"$c\","
done
unchanged src/taxiout/models.py "+ TIME_FEATURES"
echo "ok"

say "the report is newer than the generator"
if [ scripts/preregistered.py -nt "$REPORT" ]; then
    echo "$REPORT is older than its generator -- rebuild it with --report"
    exit 1
fi
echo "ok"

say "the report carries what this commit is about"
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\" -- rebuild it with --report"
        exit 1
    fi
}
check "$REPORT" "## What was changed when the pre-registration was copied in"
check "$REPORT" "### A correction to this section, 20 September"
check "$REPORT" "The rule of 19 September, on the two forward legs it reads."
check "$REPORT" "The reason is the screen, not a measured loss on the gap."
check "$REPORT" "The two instruments disagree here"
check "$REPORT" "of wall clock, of which"
# And it still carries what the first version was checked for.
check "$REPORT" "Nothing is decided here and nothing ships from it"
check "$REPORT" "## Definitions, fixed here"
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

say "1/1 the corrections"
git add scripts/preregistered.py "$REPORT" tests/test_preregistered.py \
        scripts/commit_preregistered_fixes.sh
git commit -q -F - <<'MSG'
Correct three readings in the pre-registered report, and move no figure

The experiments are 16a4f03. Nothing is refitted here: the report is rebuilt
from the same 18 records under data/processed/preregistered/, and this
commit's own gate reads every number out of the published copy and requires
each one to still be there, in the same order and unchanged. It is, and the
only differences are the additions named below. No verdict changes either:
nothing was adopted then and nothing is adopted now, which the same gate
checks by reading BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED, the season columns
and SEED_AVERAGE_PENDING out of the modules.

Three descriptions in 16a4f03's message are wrong. That message is not
rewritten, here or ever, so they are named here and the report carries a
dated paragraph recording the one a reader of the report alone would
otherwise inherit.

The first, and the one that matters: "the mild schedule gains where recency
and season agree and loses on the one leg where they do not, which is the
sign the gap leg was built to expose." The gap step is +0.10 s against that
leg's own floor of 0.40 s, measured in the same run from the same five July
fits, with a day-clustered interval of [-0.4, +0.6] that spans zero. A step a
quarter of its leg's floor is not a loss the run can see. What happened is
that the screen is fixed at printed precision, and at printed precision 224.8
is worse than 224.7. The stop is correct and the screen worked as it was
fixed; the reason is the screen, not a measured loss. The mild schedule gains
on July by more than that leg's floor and with an interval clear of zero, and
moves nothing outside seed noise on October or on the gap. The strong
schedule is a different case and is not to be read with it: it is worse on
October by more than October's own floor, so that stop is a regression the
run can measure. The report now prints every step beside the floor of the leg
it was taken on, and says which steps sit inside theirs.

The second: "a gain of 1.01 s against that column's floor of 1.02." On the
three and on October the floor is there for the guard, and the guard sizes a
regression. No test on either column asks the gain to clear anything, so
that gain failed no test. The rule's clears-the-floor test is on the two
forward legs, where the gains are 0.48 and 0.52 against floors of 0.34 and
0.23 and both clear.

The third: "Seventeen fits ... in 928 s." 928 s is the fitting run's wall
clock, which covers the scoring, the paired bootstraps and the reproduction
joins as well. The seventeen fits themselves are 863 s, summed by the
generator from the per-fit walls the report prints, and both figures are now
in the header with what each covers.

Experiment 1 now writes the verdict of the rule as well as the guard, which
the pre-registration asks of it and the first version left half done. The
rule's two forward legs are not fitted here for it, so their figures are
quoted from baseline.md as constants with their lines and the rule is applied
to them: December gains 0.48 s against a floor of 0.34 and beats all five
singles, July gains 0.52 against 0.23 and beats all five. At the present
parameters the rule holds on both forward legs and the guard holds on both
columns, and the decision is still deferred, because the parameters may move
in the wider sweep and SEED_AVERAGE_PENDING is emptied after the ladder that
follows it and not before.

The substitution record is now computed rather than maintained. The copy of
the pre-registration this report carries differs from the working note it
quotes in 13 places: 15 work-block names across 10 of them and 3 note file
names across 3. The first published version described this as two
substitutions, meaning two kinds, and listed twelve places; the thirteenth
was the only one that is not purely a rendering of a name, and it widens a
scoping clause in Experiment 5's fit rules from one block's adoptions to the
adoptions that precede the sweep. Every place is now recorded in
scripts/preregistered.py, holding the published side only, and the counts in
the report are summed from that list, so a sentence and a table cannot drift
apart. Where the note is on hand the generator checks that it rebuilds the
carried text byte for byte with exactly those places changed, and prints the
note's sha256; where it is not, the report says the check was not run rather
than that it passed, and a clean checkout still verifies the list against the
published text alone.

Also in the report: Experiment 3's four steps read against their legs' floors
beside their intervals, because three of the four are outside the floor while
every interval that prints spans zero, and the paper will have to say which
of the two instruments it is quoting; the sign convention stated once for the
whole report, with the one column where positive means better named; twenty
planned fits reconciled with seventeen made; the two floors measured here
told apart, October's from the October fits and the gap's from the July fits
scored on December; the two columns headed "flagged" distinguished, training
rows in one table and scored rows in the other; and the sentence saying the
plan entry "was committed ahead of this run" replaced by what is true, which
is that the entry is older on disk than the earliest record the run wrote and
was committed first of the three, as f7cb53c.

Report assembly has now been edited twice after the fits, and no fitting code
has changed since the run. What shows it: --report rebuilds the whole report,
checks and verdicts included, from the records on disk and gives a
byte-identical file twice over, and the fourteen check and screen lines it
prints are identical to the ones the fitting run printed from the frames it
held in memory.

One test pins what actually defeats the identifier guard. It refused the
first build of this report over H12, a stand identifier at one of the ten
airports, which a fit label had put in the text as a bare word. The code span
the first version credited does nothing, because a backtick is not a word
character and the guard's word-boundary search matches straight through one;
the underscores in july_H12_seed0 are what defeat it. Any label of letters
and digits can collide the same way, and the answer is to rename the label.
MSG

say "the commit's own tree passes its own tests"
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
echo "Not pushed. Review with: git log -p -1"
