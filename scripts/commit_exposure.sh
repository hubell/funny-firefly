#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the saturation drop and the regime-exposure bound as two commits on
# top of the current history. Additive; nothing is rewritten.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# and refuses to commit a report older than the code that generates it. Does
# not push. Review `git log -p -2` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in src/taxiout/exposure.py tests/test_exposure.py scripts/report_congestion.py \
         reports/congestion.md reports/baseline.md docs/plan.md; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

# A generated report older than its generator carries figures from code that
# no longer exists. Regenerate before committing.
if [ -n "$(find src/taxiout scripts/run_baselines.py -newer reports/baseline.md \
                -name '*.py' 2>/dev/null)" ]; then
    echo "reports/baseline.md is older than the code that generates it."
    echo "Run: python3 scripts/run_baselines.py"
    exit 1
fi
if [ -n "$(find src/taxiout scripts/report_congestion.py -newer reports/congestion.md \
                -name '*.py' 2>/dev/null)" ]; then
    echo "reports/congestion.md is older than the code that generates it."
    echo "Run: python3 scripts/report_congestion.py"
    exit 1
fi

# The ceiling and regime-exposure sections read the ladder's residuals, so a
# congestion report older than them was written against a different model.
if [ -d data/processed/residuals ] && [ -n "$(find data/processed/residuals \
        -newer reports/congestion.md -name '*.parquet' 2>/dev/null)" ]; then
    echo "reports/congestion.md is older than the residuals it reads."
    echo "Run: python3 scripts/report_congestion.py"
    exit 1
fi

say "tests"
python3 -m pytest -q

say "lint"
ruff check .
ruff format --check .

say "licence gate"
python3 scripts/check_licence_hygiene.py

# A parquet file in a commit is the one mistake no later commit undoes, and
# this stream writes them; the residuals carry the target, and the METAR cache
# is regenerated from the request recorded beside it.
if git status --porcelain --untracked-files=all data/challenge data/submissions \
        data/raw data/processed | grep -vE 'README.md|ODbL-NOTICE.md' | grep -q .; then
    echo "untracked, unignored files under data/ -- fix .gitignore first"
    exit 1
fi

say "1/2 the submission rung"
git add src/taxiout/models.py scripts/make_submission.py scripts/sweep_params.py \
        scripts/run_baselines.py tests/test_features.py Makefile README.md \
        reports/baseline.md
git commit -q -m "Fit the submission from the rung below the saturation block

Runway saturation was built as the cheap queueing proxy, measured against
the ceiling, and found to be worth nothing: it cost half a second in
December and eight tenths in July on the rung it was added to, and the
submission is fitted from that rung. So it is out of the model.
TOP_RUNG is now named rather than taken as the last rung of the ladder --
SUBMISSION_KEY picks the weather rung, LADDER_PROPER is the ladder before
the chain kept without the plan lags, and ABOVE_SUBMISSION is what sits
above the submission rung and is deliberately not in it. The saturation
rung stays on the ladder, and the ceiling section still measures it
against the ceiling of the rung it was added to, because a block that was
built, measured and bought nothing is a result and belongs in the paper.

The one parameter sweep is pinned to the rung it actually ran on. It read
TOP_RUNG, which was the saturation rung when it ran; following TOP_RUNG
down to the weather rung would have quietly re-run the sweep on a
different feature block and stopped reproducing the October numbers
recorded beside BOOTSTRAP_PARAMS, which is the one place the choice of
parameters is accounted for. SWEEP_RUNG names it, and the four places
that described it were reworded.

Also: the submission writer's docstring described the Gate 1 feature set
and not the rung it fits, and did not name the METAR cache it now
requires; the per-airport and split-share sections of the ladder report
name the rung they describe; the bootstrap level is interpolated from its
constant rather than typed; two figures quoted in report prose are
attributed to the report that computes them, and one that no generator
computes any more is stated without a number."

say "2/2 the regime-exposure bound"
git add src/taxiout/exposure.py tests/test_exposure.py src/taxiout/ceiling.py \
        scripts/report_congestion.py reports/congestion.md docs/plan.md \
        scripts/commit_exposure.sh
git commit -q -m "Bound how far the ranking months' regime moved, and say it is a bound

The plan-lag block reads take-off minus a planning value fixed before
departure, which is the taxi plus the pushback delay against that plan.
The ranking months moved against their training-year twins, and a ranking
departure's block time is hidden, so which of the two moved is not
observable. It is boundable, and reports/congestion.md now bounds it.

Three quantities per airport, per ranking month against the same calendar
month of 2025: the shift in the mean of take-off minus each of the three
planning values; a control that reads no departure plan and is exposed on
both sides -- the shift in arrivals' mean taxi-in and in arrivals' mean
landing minus schedule at the same airport-month; and the worst case, the
airport's training-year slope of taxi-out on the plan lag times the shift,
added in quadrature to that airport's own ex-flagged RMSE.

One methodological trap is worth the reader's time, and the module
docstring and the report both carry it. The lag window must be applied to
each lag on its own and not to the three together: a row whose schedule
lag runs long is usually a row that took off late, so a joint window drops
late rows -- and drops more of them on the side that ran later, which is
the side whose shift is being measured. The report prints what a joint
window would have reported instead, computed rather than asserted.

What it bounds. Where the plan lag moves most the surface does not:
arrivals' taxi-in barely moves while arrivals' own lateness moves with the
plan lag, so the shift is mostly the day running late against its plans
rather than the surface running slower. That is a reading of the control,
not a measurement of the split. Taking the whole shift as pushback delay,
which the control says it is not, the cost is tens of seconds of bias at
the worst airports and a few seconds of RMSE pooled. Nothing in the model
is corrected for it: a correction would be a guess at the mixture, fitted
on nothing."

say "result"
git --no-pager log --oneline -3
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -2"
