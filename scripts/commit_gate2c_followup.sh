#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the Gate 2c follow-up -- the leakage trigger read on both of its
# columns, the decomposition of what the stand-occupancy block actually buys,
# the schedule-anchored diagnostic that says how much of it survives an
# operational reading, and the recorded-taxi tail taken out of every fit -- as
# five commits on top of the current history. Additive; nothing is rewritten.
#
# The commits are ordered so that each one leaves a tree whose tests pass: the
# two feature-layer changes first, with their own tests, then the generators
# that read them, then the reports those generators wrote, then the
# documentation that quotes the reports. Unlike the earlier commit scripts in
# this directory, that ordering is CHECKED rather than stated -- see the
# verification pass at the foot of this script, which builds each commit's tree
# in a throwaway worktree and runs the suite against it. Stating it and not
# checking it is what every commit script here has done until now.
#
# Five commits and not the four the follow-up note asks for. The note's split is
# thematic and this repository's history is split by file, and one file --
# scripts/run_baselines.py -- carries three of the note's four themes. Splitting
# it thematically needs partial staging, which produces intermediate trees that
# do not import; splitting by file keeps every tree working and loses nothing
# except a one-to-one correspondence with the note's numbering.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# and refuses to commit a report older than the code that generates it. Does
# not push. Review `git log -p -5` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in src/taxiout/occupancy.py src/taxiout/models.py src/taxiout/features.py \
         tests/test_occupancy.py tests/test_tails.py \
         scripts/run_baselines.py scripts/report_occupancy.py \
         scripts/make_submission.py scripts/report_transfer.py \
         reports/baseline.md reports/occupancy.md reports/transfer.md \
         reports/congestion.md \
         docs/plan.md docs/modelling_notes.md docs/review_headroom.md \
         docs/paper.md; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

# A generated report older than its generator carries figures from code that no
# longer exists. This gate matters more in this change than in any before it:
# the recorded-taxi exclusion moves every fitted number in the repository, so a
# report that predates it is not merely stale but describes a different model.
for pair in "scripts/run_baselines.py reports/baseline.md" \
            "scripts/report_occupancy.py reports/occupancy.md" \
            "scripts/report_congestion.py reports/congestion.md" \
            "scripts/report_transfer.py reports/transfer.md"; do
    set -- $pair
    if [ -n "$(find src/taxiout "$1" -newer "$2" -name '*.py' 2>/dev/null)" ]; then
        echo "$2 is older than the code that generates it."
        echo "Run: python3 $1"
        exit 1
    fi
done

say "tests"
python3 -m pytest -q

say "lint"
ruff check .
ruff format --check .

say "licence gate"
python3 scripts/check_licence_hygiene.py

if git status --porcelain --untracked-files=all data/challenge data/submissions \
        data/raw data/processed | grep -vE 'README.md|ODbL-NOTICE.md' | grep -q .; then
    echo "untracked, unignored files under data/ -- fix .gitignore first"
    exit 1
fi

say "1/5 the stand block without the take-off"
git add src/taxiout/occupancy.py tests/test_occupancy.py
git commit -q -F - <<'MSG'
Re-anchor the stand block on the schedule, as a diagnostic

The stand-occupancy block reads an arrival that on-blocked after the
subject's own off-block. That is admissible under the challenge's
post-hoc framing and not under an operational one, and the obvious way to
ask how much of the block survives the stricter reading is to drop the
three columns whose arithmetic names the take-off. That does not work,
and the reason is worth stating because it is not obvious.

occ_gap1_sched_s already cancels the take-off algebraically: it is SCHED
minus the arrival's on-block. But the take-off also decides WHICH arrival
the subtraction is about, because the stand's last arrival is the last
one to on-block at or before it. Every _gather-ed column inherits that
choice. Holding the schedule and both on-blocks fixed and moving only the
take-off ten minutes moves occ_gap1_sched_s by eighty and flips both
match flags -- a test asserts exactly that, on the fixture the rest of
this module is tested on.

So the anchor moves instead. occupancy_tfree_features selects the stand's
last arrival at or before the subject's SCHEDULED time and reads every
column from that choice, including the coverage flags. Seven columns
under their own occt_ prefix and their own source registry, named apart
from OCCUPANCY_FEATURES so that widening one cannot widen the shipped
model's feature set by accident -- the two are coupled through
ENCODING_STAND_BLOCK + OCCUPANCY_FEATURES on four rungs.

Two numbers in the docstring did not reproduce and are replaced by
measured ones: the share of rows whose stand received no arrival is 0.09%
of the training year and 0.007% of the December validation rows, not
0.6%, and the 20-45 minute band's residual bias is +185 s at 20-30 and
+120 s at 30-45 rather than a single +185. The mechanism paragraph is
rewritten to say what the block is -- the stand's recent history, with a
bound inside it in one band -- rather than calling the whole of it a
bound, which the decomposition in reports/occupancy.md now shows it is
not.
MSG

say "2/5 the recorded-taxi tail, in one place"
git add src/taxiout/models.py src/taxiout/features.py
git commit -q -F - <<'MSG'
Take the recorded-taxi tail out of every fit, from one constant

The record-quality term removes rows by covered cell -- an airport
crossed with whether a Network Manager record exists and how long the
take-off lag is -- and a cell is covered only where enough of its rows
are defaulted. That is blind to a row whose cell is uncovered. The
training year holds a departure recording a taxi of 87,341 s at an
airport whose cells are all uncovered, with a take-off lag of 939 s: a
day-shift that nothing except the recorded taxi identifies, sitting in
the fitting sample at its recorded value.

It cannot be fixed by widening the flag. tails.flag and tails.predict are
restricted by construction to fields the ranking set exposes, because
they run there unchanged, and recorded taxi time is the target and is
hidden for departures. So it is a fit-side exclusion and nothing else:
the target is never clipped and every row is still scored.

taxiout.models.fit_rows is that rule and TAIL_EXCLUSION_ADOPTED_H is its
threshold, four hours, fixed on the mechanism rather than by a sweep --
no aircraft taxis for four hours. It lives in one module because four
places fit models: the ladder, the k-sensitivity diagnostic, the seed
average, the leave-one-airport-out transfer report and the shipped
submission. Two of those had their own fit-row expression, so an
exclusion added to the ladder alone would have put a different model in
the submission than the one reports/baseline.md measures, and nothing
would have said so. A test asserts all three scripts import the shared
rule and that none re-derives the threshold for itself.

The frame also gains the seven occt_ columns and their sources, so that
one diagnostic rung can read them.

The test that holds all three scripts to this rule ships with those
scripts in the next commit rather than here: it reads their source, so in
this commit's tree it would be asserting against files that have not been
changed yet. A commit whose own tests fail is not one to bisect through.
MSG

say "3/5 the generators"
git add scripts/run_baselines.py scripts/report_occupancy.py \
        scripts/make_submission.py scripts/report_transfer.py \
        tests/test_tails.py
git commit -q -F - <<'MSG'
Read the leakage trigger on both of its columns

The expected headroom for each Gate 2c block is two numbers -- December
over all ten airports, July over the three July 2026 contains -- and the
audit tested the first alone, so an overshoot on the second could not be
seen. On the fit this change was written against it fired: stand
occupancy took +15.86 s on the July three against an estimated +10.3.
On the fit that ships it does not, by twenty-nine milliseconds: +15.27 s,
an overshoot of 4.971 against the 5.000 allowed. Both branches were
already written; neither was reachable.

A test cleared by that margin is not evidence of anything, so the audit
now reports the closest approach whether or not it fires, and says in
terms that a near miss is not a pass. What moved it below the line was
twelve rows leaving the fit, which is the measure of how little the test
was carrying.

What the fired audit rests on had to change with it. It used to say that
occupancy.md showed the gain landing where the mechanism predicts, which
that report cannot show: its band table is built on the rung that never
read the gap and reports bias, not gain. The replacement is the
measurement itself -- the block's reduction in squared error attributed
to disjoint bands of the gap, an SSE being additive over rows where an
RMSE is not. The 20-45 minute band is 1.0% of the December rows and 22.4%
of the cut, and 1.9% and 36.3% on the July three. That concentration is
computed and the prose branches on it, so a run where it failed to hold
would say so rather than assert it anyway.

Also here: the 5 s trigger is no longer attributed to docs/prior_art.md,
whose trigger is deliberately a ratio and which says in terms that an
absolute threshold does not transfer between targets; the "no worse"
reading is stated as a 0.1 s tolerance on the ex-flagged RMSE rather than
as "the precision this report prints", which named two precisions in a
table that prints RMSEs to one decimal and the deltas the rule compares
to two; the step interval is given under three clustering units, day,
airport-day and row, so the choice of unit can be seen not to carry the
conclusion; the held-out-pair direction is corrected, a positive
difference-in-differences meaning more where the pair is new; a
per-airport table gives every rung's step with a day-clustered interval
where it clears two seconds; and "what is not here yet" is computed from
the rungs actually scored instead of being a fixed list that went stale
two gates ago.

The occupancy report gains the same decomposition split by the type-match
flag, and a bias column whose prose said prediction minus actual while
the code computed actual minus prediction.

make_submission and report_transfer import the shared fit-row rule, and
tests/test_tails.py holds all three fitting scripts to it: each must
import taxiout.models.fit_rows, and none may re-derive the threshold for
itself. That test lands here rather than with the rule because it reads
these scripts' source.
MSG

say "4/5 the reports"
git add reports/baseline.md reports/occupancy.md reports/transfer.md \
        reports/congestion.md
git commit -q -F - <<'MSG'
Refit the ladder without the recorded-taxi tail

Every fitted number in these four reports moves, because every fit now
excludes recorded taxi times over four hours. congestion.md is here
because its ceiling section reads the submission rung's own stored
residuals, so the oracle bound moves with the refit even though nothing
in that report's own code changed. On the submission rung that
is worth 2.6 s on both forward splits, from twelve rows out of a
1.9-million-row fit -- more than the seed averaging already adopted, and
from rows no aircraft produced.

The recorded-taxi section reports three arms against each other: the
un-excluded fit as the reference, four hours, and two hours. Both
thresholds were fixed before the run and both are reported, so reporting
whichever came out better was not available as a move. Two hours is worse
on July by 2.7 s and is not adopted: the two-to-four hour band holds real
long holds, mostly at Paris and Heathrow with a Network Manager record
and an ordinary schedule lag, and excluding them trades genuine tail
signal for variance. That is why both were run.

The reference arm is refitted rather than read off the ladder, because
the ladder now carries the adopted exclusion and reading an arm against
it would be reading it against itself.

The schedule-anchored diagnostic is here too, and it is the least
comfortable number in the change: on the three airports July 2026
contains, the block keeps 4% of what it buys once the take-off anchor is
taken away -- 42% in December and 34% on July all-ten. Almost all of what
the stand-occupancy block is worth, on the airports the ranking month
actually asks about, depends on an anchor that is admissible only under
the post-hoc reading. It changes nothing about what is submitted; it is
reported because a reviewer is entitled to ask and the answer is a
number.
MSG

say "5/5 the documentation"
git add docs/plan.md docs/modelling_notes.md docs/review_headroom.md docs/paper.md
git commit -q -F - <<'MSG'
Record what the follow-up settled, and what it did not

The plan and the review note said no block beat its headroom by more than
the trigger allows. On the column each estimate was made against, stand
occupancy does, and both now say so. The review note's section 7 also
cited occupancy.md's band table as what stood behind the audit; that
table cannot stand behind it, and the correction names what does.

The stand-occupancy mechanism is reworded in the five places that carry
it. The block is the stand's recent history, and the
twenty-to-forty-five-minute band is where the bound inside it is the
mechanism -- a minority of what the block buys on every leg measured, 22%
in December against 48% from gaps past an hour where the previous arrival
was the subject's own aircraft and the gap bounds nothing. The paper's
6.5 is marked not to be written from the bound, with the five things it
has to carry and the numbers for each.

Seed averaging has now been measured three times against a rule fixed
before the first of them -- 0.78/0.98 s one rung below the submission,
1.10/1.37 s on it, and 0.81/0.56 s on it once the recorded-taxi tail left
the fit -- and only the middle reading cleared the one-second threshold.
It is not adopted, taxiout.models.SEED_AVERAGE_ADOPTED is False, and
make_submission.py reads that constant rather than deciding for itself,
so the shipped model cannot average seeds the report says were declined.
The three readings say what the averaging was buying: most of the
run-to-run variance five fits were smoothing over came from a dozen
day-shift rows in the fitting sample. models.py, plan.md and the review
note had each described it in a way that implied a single measurement. The held-out-pair asymmetry names two rungs and not one, and the
one plan.md named was the wrong one: stand occupancy at +1.4 s [+0.3,
+2.5] is the interval visibly clear of zero, and its sign means the block
buys more where the pair is NEW, which is the opposite of what the
generated prose said.

Recorded for the next gate and not applied to this one: the July leg of
the acceptance rule should read the three-airport column. The pooled July
figure is 49% Rome by squared error on 7.8% of the rows, at an ex-flagged
RMSE of 863 s against roughly 200 elsewhere, so a rung can move it by
improving one airport's tail -- which is what the callsign rung does, at
-25.2 s [-36.8, -22.1] at Rome and +16.9 s [-3.5, +50.1] at Zurich on a
pooled step of -4.6. July 2026 does not contain Rome. Changing an
acceptance rule after seeing which rungs it would admit is the failure
the pre-registration exists to prevent, so it binds the next gate and
Gate 2c stands as it was decided.
MSG

# The claim this script's header makes about commit ordering, checked rather
# than asserted. Each commit is built in a throwaway worktree and the suite is
# run against that tree alone, so a commit that leaves the repository unable to
# import -- a generator committed before the module it imports, say -- fails
# here instead of being found by whoever bisects through it later.
say "each commit's tree passes its own tests"
work="$(mktemp -d)"
trap 'git worktree remove --force "$work/t" 2>/dev/null || true; rm -rf "$work"' EXIT
for sha in $(git rev-list --reverse HEAD~5..HEAD); do
    git worktree add --detach -q "$work/t" "$sha"
    # PYTHONPATH is forced to the worktree's own src. This repository is
    # installed editable, so site-packages carries a .pth pointing at the main
    # checkout; without this, a worktree's tests could import the working tree's
    # modules and every commit would appear to pass by testing code that is not
    # in it. pyproject's `pythonpath = ["src"]` already wins on ordering, and
    # this makes the guarantee not depend on that.
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
git --no-pager log --oneline -6
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -5"
