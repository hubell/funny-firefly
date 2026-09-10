#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Close out Gate 2c: the five prose and code fixes the review left, the new
# report that gives §9.5's figures a generator, and the paper with its daggers
# off. Additive; nothing is rewritten. Not pushed.
#
# The commits are ordered so that each one leaves a tree whose tests pass, and
# that ordering is CHECKED rather than stated -- the verification pass at the
# foot builds each commit in a throwaway worktree and runs the suite against it,
# with PYTHONPATH forced to the worktree's own src, because this repository is
# installed editable and a worktree would otherwise import the working tree's
# modules and every commit would appear to pass by testing code it does not
# contain. The last script found a real ordering bug this way: a test that reads
# three scripts' source shipped one commit before those scripts changed.
#
# The dagger check that `scripts/commit_paper_gate2c.sh` carried is deliberately
# NOT here. It existed to stop the draft claiming a computed source it did not
# have; `reports/residual.md` is now that source, the daggers are off, and a
# check that requires them would fail for the right reason at the wrong time.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# and refuses to commit a report older than the code that generates it.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in scripts/run_baselines.py scripts/report_residual.py \
         tests/test_residual.py src/taxiout/models.py \
         reports/baseline.md reports/residual.md docs/paper.md \
         docs/review_headroom.md Makefile README.md; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

# A generated report older than the code that generates it carries figures from
# code that no longer exists. The residual report joins the four that were
# already gated; it reads stored residuals, so it is cheap to regenerate and
# there is no excuse for a stale one.
for pair in "scripts/run_baselines.py reports/baseline.md" \
            "scripts/report_residual.py reports/residual.md" \
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

say "the paper's figures have a generator, and no dagger is left standing for one"
if ! grep -qE "^## The shape of the tail" reports/residual.md; then
    echo "reports/residual.md has no tail section; run python3 scripts/report_residual.py"
    exit 1
fi
if ! grep -q "reports/residual.md" docs/paper.md; then
    echo "docs/paper.md does not cite reports/residual.md"
    exit 1
fi
echo "ok"

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

BEFORE="$(git rev-parse HEAD)"

say "1/4 the generator and code fixes"
git add scripts/run_baselines.py src/taxiout/models.py reports/baseline.md \
        docs/review_headroom.md
git commit -q -F - <<'MSG'
Say what the audit did before, and what the block does not show

Five things the review left, all of them prose the generator emits rather
than numbers it computes, so the ladder is unchanged and every figure in
reports/baseline.md reproduces the run before it.

The leakage audit now carries its own history. It reports a near miss at
4.971 s against the 5.000 s allowed and never said that the same test
FIRED at 5.56 s on the ladder as committed at ad1eecf, before the
recorded-taxi tail came out of every fit. Twelve rows moved it below the
line. That is quoted here on the same footing as the earlier seed gains:
the run that produced it fitted rows this one does not, so recomputing it
would be a different measurement wearing the same name.

"Concentration" is softened to what it supports. The twenty-to-forty-five
minute band carries about twenty-two times its share of the cut, which is
consistent with the mechanism and is not evidence against a column
reaching the answer another way -- 78% of the December cut and 64% of the
July-three cut is outside that band, and the block helps in every band.
What stands behind the block is the registry, the poison tests, the
both-sides parity of the gap and the schedule-anchored measurement.

The schedule-anchored figures are labelled as the lower bounds they are.
The anchor is stricter than an operational rule: at pushback every arrival
up to the subject's own off-block is known, not only those on-blocking
before its scheduled time, so a block anchored there would keep more than
42%, 34% and 4%.

Seed averaging: models.py still headed the paragraph "adopted 8 September"
above the text saying it is not adopted. And the claim that the day-shift
rows were most of what five seeds smoothed over holds for July, 1.37 to
0.56, and not for December, 1.10 to 0.81. It is about half in December and
most in July, and models.py, the report and the paper's drafting notes now
say that instead.

"Rows out: 12" summed two splits' fits. The two fits exclude different
rows and the column prints them per split.

docs/review_headroom.md is here because it carried the same "most of what
five seeds were smoothing over" sentence as models.py did, and a claim
corrected in three places and left standing in a fourth is not corrected.
MSG

say "2/4 the residual report and its generator"
git add scripts/report_residual.py tests/test_residual.py reports/residual.md \
        Makefile README.md
git commit -q -F - <<'MSG'
Print what the remaining error is, from the stored residuals

The paper's §9.5 argues that what the model has left is a thin tail of
individual holds rather than a queue, and it carried eleven figures no
report printed -- computed for the draft and marked with a dagger. This is
the generator that prints them, so the argument rests on a report rather
than on arithmetic done once.

reports/residual.md reads the submission rung's stored validation
residuals, joins them to the validation month on the movement identifier
and refits nothing, so it runs in seconds. That is why it is its own
report rather than a section of reports/baseline.md as the draft
anticipated: a figure that costs seconds should not sit behind a
two-and-a-half hour ladder, and a reader checking §9.5 should not have to
run one.

Five sections: the shape of the tail at two cuts, with the share of rows,
of squared error and of under-predictions, and the under-predicted rows'
own medians; the three enrichment classes -- heavies, departures with no
Network Manager record, and stands re-occupied by another type in the band
the occupancy bound speaks about -- each against the worst one percent of
residuals; the error by plan lag; the largest rows per leg as airport,
recorded taxi and prediction; and the same tail shares per airport, which
is what the results section's per-airport prose reads.

Three legs, because the pooled July figure answers a question the ranking
month does not ask: one Fiumicino departure recording 87,002 s carries
33.8% of July's squared error and July 2026 contains no Fiumicino rows.
The record-quality term does not cover that cell and the four-hour
exclusion does not reach it either -- that rule is fit-side only and never
touches the scoring set -- so it is scored at its recorded value.

The generator refuses to write a literal stand identifier, as
report_occupancy.py does, and a test holds it to that. make residual, a
README stage line, and the staleness gate extended to the new pair.
MSG

say "3/4 the submission, if the check moved anything"
if git diff --cached --quiet && git diff --quiet -- scripts/make_submission.py \
        src/taxiout/submission.py; then
    echo "  nothing changed: v3 was written from the committed code and the"
    echo "  submissions themselves are gitignored, so there is nothing to commit"
else
    git add scripts/make_submission.py src/taxiout/submission.py
    git commit -q -F - <<'MSG'
Record what the v3 submission's checks found

The submission is regenerated from the committed code and is not tracked;
what changes here is whatever the checks moved.
MSG
fi

say "4/4 the paper, with its daggers off"
# This script commits itself here, as scripts/commit_gate2c.sh did in the
# last commit of the range it created.
git add docs/paper.md scripts/commit_gate2c_closeout.sh
git commit -q -F - <<'MSG'
Give §9.5's figures a generator and take the daggers off

Every figure §9.5 carried with a dagger now comes from
reports/residual.md and has been re-read against it. Three moved, and they
are the case for having built the generator rather than leaving the
draft's own arithmetic in place: the July-three tail is three in five
under-predicted and not two thirds, the plan-lag RMSEs are 421 and 642
rather than 422 and 643, and the Fiumicino row is 33.8% of July's pooled
squared error rather than 34%.

One figure keeps its old source rather than gaining a new one. The mean
under-prediction inside the twenty-to-forty-five minute band, cut from
+235 s to +7, is reports/occupancy.md's measurement of the band before and
after the block, not a residual-report quantity, and the section now says
so instead of implying the new report prints it.

The drafting notes record what moved and why the report went where it did.

This script is here too, as the last one was in the last commit of its own
range.
MSG

say "each commit's tree passes its own tests"
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

# reports/occupancy.md, congestion.md and transfer.md are gated for staleness
# above but staged in no commit, because regenerating them after a run that
# changes no fit should reproduce them byte for byte. If it does not, they are
# left modified and the "untracked" check below would not notice -- it looks for
# ?? and these are tracked. So the claim is asserted rather than printed.
say "nothing tracked is left behind"
if ! git diff --quiet; then
    echo "tracked files are still modified after the last commit:"
    git --no-pager diff --name-only | sed 's/^/  /'
    echo "a regenerated report that changed is either a real difference worth a"
    echo "commit or a bug; either way it is not something to leave in the tree."
    exit 1
fi
echo "ok"

say "result"
git --no-pager log --oneline -6
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -4"
