#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# The pre-registered experiments. Three commits, in this order and no other:
#
#   1. docs/plan.md: the decisions of 19 September and the criteria fixed on
#      20 September, committed ahead of the run that measures them
#   2. the generator, its tests, the sample-weight argument it needs and the
#      report it wrote
#   3. reports/residual.md: one sentence that said two row sets where the
#      table it describes prints one of them already
#
# The first commit has to be first, and the check below that says so is the
# whole point of the sequence: docs/plan.md is older on disk than the earliest
# record under data/processed/preregistered/, so the criteria were on disk
# before the numbers were. A run that fixed its criteria afterwards would fail
# that check rather than read as though it had not.
#
# Nothing is adopted by any of the three. BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED,
# SEED_AVERAGE_PENDING and the submission rung's features are checked unchanged
# before anything is staged, because a commit that said "nothing was adopted"
# while moving one of them would be the failure this script exists to prevent.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# checks that each generated report is newer than the code that generates it
# and carries the section its commit is about, refuses a working tree holding
# anything but the ten expected paths, and verifies every commit's own tree in
# a worktree before finishing. Nothing is pushed.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

REPORT="reports/preregistered-2026-09-20.md"

PATHS="docs/plan.md
src/taxiout/models.py scripts/preregistered.py tests/test_models.py
tests/test_preregistered.py $REPORT Makefile scripts/commit_preregistered.sh
scripts/report_residual.py reports/residual.md"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these ten paths and nothing else"
# Three commits that each name their own files cannot sweep up a fourth
# change, but a tree holding one is a tree whose gates were run over something
# this script is not committing, and the run should be understood before it
# goes ahead rather than after.
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok"

say "the record precedes the numbers on disk"
# docs/plan.md carries the criteria. Every record under the generator's own
# directory is a measurement made under them. If the entry were the younger
# file, the criteria would have been written with the figures in view, and
# no reading of the commit order would tell you so.
earliest="$(find data/processed/preregistered -name '*.json' -printf '%T@ %p\n' \
            2>/dev/null | sort -n | head -1 | cut -d' ' -f2-)"
if [ -z "$earliest" ]; then
    echo "no records under data/processed/preregistered -- run the generator first"
    exit 1
fi
if [ "docs/plan.md" -nt "$earliest" ]; then
    echo "docs/plan.md is NEWER than $earliest."
    echo "The criteria must be on disk before the measurements made under them."
    exit 1
fi
echo "ok: docs/plan.md predates $earliest"

say "this commit adopts nothing"
# Read from the module rather than promised in a message. Each of these is a
# decision the run was forbidden to take, and a commit that moved one while
# its message said otherwise is exactly the thing the pre-registration is
# written to make impossible.
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
unchanged scripts/run_baselines.py 'SEED_AVERAGE_PENDING: tuple[str, ...] = ('
# Experiment 3 measured the season columns and adopts nothing either way, so
# all three are still in TIME_FEATURES, which every booster rung reads.
for c in tko_month tko_doy_sin tko_doy_cos; do
    unchanged src/taxiout/features.py "\"$c\","
done
unchanged src/taxiout/models.py "+ TIME_FEATURES"
echo "ok"

say "each generated report is newer than the code that generates it"
for pair in "scripts/preregistered.py $REPORT" \
            "scripts/report_residual.py reports/residual.md"; do
    set -- $pair
    if [ "$1" -nt "$2" ]; then
        echo "$2 is older than $1 -- regenerate it"
        exit 1
    fi
done
echo "ok"

say "each file carries the thing its commit is about"
# A report newer than its generator can still be a report generated before the
# section was written, if the generator was touched afterwards and the report
# was not. These greps are what says the files on disk are the ones the
# messages below describe.
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\" -- regenerate it"
        exit 1
    fi
}
check docs/plan.md "**The decisions of 19 September, and the experiments pre-registered on"
check "$REPORT" "## What Experiment 1 measured"
check "$REPORT" "## What Experiment 2 measured"
check "$REPORT" "## What Experiment 3 measured"
check "$REPORT" "## Definitions, fixed here"
check "$REPORT" "Nothing is decided here and nothing ships from it"
check reports/residual.md "One row set that table does not"
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

say "1/3 the record, ahead of the numbers"
git add docs/plan.md
git commit -q -F - <<'MSG'
Fix the criteria of the next three measurements before making them

A dated entry recording the four decisions of 19 September and the four
experiments pre-registered on 20 September. It is committed before the run
that measures any of them, so the record precedes the numbers in the history
as it does on disk, and the commit script that makes these three commits
refuses to run if docs/plan.md is the younger file.

The seed rule, re-fixed at the floors. The one-second rule was retired on
19 September before any measurement under its replacement existed, because
its stated basis -- the run-to-run variation the ladder showed before the
booster was made repeatable -- no longer holds. The rule that replaced it is
quoted in the words it was fixed in, as reports/baseline.md prints it.

The sweep, reopened once. The grid, the fit rules, the October floor measured
in the same run and the reading rule are all written down here before a cell
is fitted, and nothing in it is chosen on December, July or the held-out
pairs. AOBT_3_flt stays excluded by decision and unmeasured, and the July
column stays as the entry of 9 September records it.

The experiments, with their criteria verbatim: seed averaging read on the
three July airports and on October, with the guard added on 20 September
before either measurement existed; recency weighting under two half-lives,
with the confound stated first, the digest check, the screen on October, the
gap leg built to break the confound, and the adoption rule; the season
columns, from which no adoption follows in either direction, whatever the
sign. And the list of what is not run, with the reason for each.

Nothing is measured by this commit and nothing is changed by it. That is the
point of its being first.
MSG

say "2/3 the experiments, and what their own rules decided"
git add src/taxiout/models.py scripts/preregistered.py tests/test_models.py \
        tests/test_preregistered.py "$REPORT" Makefile \
        scripts/commit_preregistered.sh
git commit -q -F - <<'MSG'
Run the pre-registered experiments, and adopt none of them

Seventeen fits of the submission rung under criteria fixed in writing the day
before, on one feature frame, sequential, in 928 s at a peak of 6.9 GiB. Twenty
were counted in advance; three were not made because two schedules stopped at
a screen, which is the sequence working. Every criterion in the report is a
constant copied from the pre-registration and every figure under it is
computed. Nothing is adopted: BOOSTER_PARAMS, SEED_AVERAGE_ADOPTED,
SEED_AVERAGE_PENDING and the submission rung's features stand exactly as they
did, and this commit's own gate checks each of them rather than promising it.

Reproduction first, and nothing was read until it passed. The December and
July first-seed fits reproduce the ladder's stored residuals with a largest
absolute difference of 0 over 165,677 and 190,713 rows, joined on the movement
identifier, which is never printed. They print 217.6 and 325.8 s pooled, 218.6
at the ranking mix and 237.8 on the three; the five July singles span 325.81 to
326.23 s pooled and 237.02 to 238.83 on the three, and their average is
325.29 s. Nine checks, each against the line of baseline.md that fixed it.

Experiment 1, seed averaging on the two columns the rule of 19 September asks
for before the average may ship. On the three the five-seed average is
236.80 s against the ladder's seed at 237.80 and a best single of 237.02, a
gain of 1.01 s against that column's floor of 1.02. On October it is 216.88
against 217.28 and a best single of 217.28, a gain of 0.40 against a floor of
0.19 measured from those same five singles. The average beats every single on
both, and the guard of 20 September -- not adopted if it is worse than the
ladder's seed by more than the floor on either -- holds on both. Nothing ships
from that: the rule is read for the submission once the ladder prints the two
measurements under the final parameters, and SEED_AVERAGE_PENDING is untouched
here.

Experiment 2, recency weighting, and the confound it was built to separate.
The digest check passed first: the mild schedule's October fit, made twice,
gave identical prediction vectors and identical model dumps, so deterministic
holds under sample weights as it does without them. Then the sequence.
H = 3 scores 217.6 s on October against 217.3 unweighted, worse at printed
precision, and stops there; it is fitted on nothing else. H = 12 scores 217.2
on October, no worse, and is fitted on July, where its December reading -- the
gap leg, the one forward leg whose recent training months are seasonally
furthest from the month it scores -- is 224.8 s against 224.7 unweighted. Worse
at printed precision, so it stops before December and cannot be adopted. Its
July readings come from the same fit and are printed: 325.0 s all ten, a step
of -0.84 [-1.5, -0.4], and 236.2 on the three, -1.61. So the mild schedule
gains where recency and season agree and loses on the one leg where they do
not, which is the sign the gap leg was built to expose. Neither schedule is
adopted, make_submission.py is unchanged, and the paper's limitation section
now has the gap-leg reading as the reason no recency term ships.

Experiment 3, the season columns, for the paper and not for the model. Without
tko_month, tko_doy_sin and tko_doy_cos the rung scores 217.2 s on December
against 217.6, a step of -0.40 [-1.6, +0.9]; 218.3 at the ranking mix against
218.6, -0.30 [-1.5, +1.0]; 239.0 on the three against 237.8, +1.15; and 326.6
all ten on July against 325.8, +0.77 [-0.2, +2.3]. Both forward legs validate
on a month the training side never held and the ranking months do not, so the
measurement answers a question the ranking months do not ask and no adoption
follows from it in either direction, which the pre-registration fixed before
the sign was known. The three columns stay in the frame.

taxiout.models gains one argument. Booster.fit and SeedAverage.fit take
sample_weight=None; with None no weight argument reaches the backend at all,
so an unweighted fit is the fit it was, and the array is never kept on the
model. The synthetic tests hold that to two fits agreeing bit for bit, and the
reproduction check above holds it on the real data against the stored
residuals.

The generator writes under data/processed/preregistered/, which is gitignored,
and to the one report it names. reports/ and data/processed/residuals/ are
fingerprinted by size and modification time before it runs and verified after,
in two phases, the report exempted by name only in the phase that writes it;
both came back unchanged over 78 files. It refuses to publish a stand or a
movement identifier, and it refused this report once, over a fit label that
printed the bare token of a real stand identifier. --report rebuilds the whole
report, checks and verdicts included, from the records on disk without
refitting, and does so byte for byte.
MSG

say "3/3 the sentence the lag table had already outgrown"
git add scripts/report_residual.py reports/residual.md
git commit -q -F - <<'MSG'
Say that the lag table already prints its no-estimate row

The by-plan-lag cut of the residual report said it named "two row sets that
table leaves implicit", the two being a take-off with no estimate and a
take-off earlier than its own estimate. Only the second is implicit. The
error by plan lag builds its rows from the lag bands plus no estimate and
prints that row on every leg, so the sentence overstated what the cut adds by
one row set.

It now says the one row set that table does not print, and that no estimate is
already a row there and a row here too. The constant above the two labels said
the same thing and is corrected with it, since that comment is where the
sentence came from. Generator wording only: no figure in the report moves, and
regenerating gives one changed hunk.
MSG

say "each commit's own tree passes its own tests"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
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
git --no-pager log --oneline -4
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -3"
