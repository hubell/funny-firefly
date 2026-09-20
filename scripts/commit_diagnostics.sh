#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Four descriptive sections across three generated reports. Three commits, one
# per generator and its report, additive, nothing rewritten, nothing pushed.
#
#   1. reports/residual.md: where EGLL's July error sits, and what December's
#      three weather-regime flags separate
#   2. reports/tails.md: the callsign-prefix reading of the record-quality
#      term, at the one airport the term covers
#   3. reports/data_review.md: whether the training year held days like the
#      ranking set's disruption days
#
# No model is fitted, no feature is changed and no rule is re-read by any of
# them. Each section describes stored residuals or the challenge data as they
# already are.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# checks that each generated report is newer than the code that generates it
# and that it actually carries its new section, refuses a working tree holding
# anything but the nine expected paths, and verifies every commit's own tree in
# a worktree before finishing.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

PATHS="scripts/report_residual.py reports/residual.md tests/test_residual.py
scripts/report_tails.py reports/tails.md tests/test_report_tails.py
scripts/review_challenge_data.py reports/data_review.md
tests/test_report_data_review.py scripts/commit_diagnostics.sh"

for f in $PATHS; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "the working tree holds these ten paths and nothing else"
# Three commits that each name their own files cannot sweep up a fourth
# change, but a tree holding one is a tree whose gates were run over
# something this script is not committing, and the run should be understood
# before it goes ahead rather than after.
expected="$(printf '%s\n' $PATHS | sort)"
actual="$(git -c core.fileMode=false status --porcelain --untracked-files=all \
          | sed 's/^...//' | sort)"
if [ "$actual" != "$expected" ]; then
    echo "the tree is not what this script expects."
    diff <(echo "$expected") <(echo "$actual") || true
    exit 1
fi
echo "ok"

say "each generated report is newer than the code that generates it"
for pair in "scripts/report_residual.py reports/residual.md" \
            "scripts/report_tails.py reports/tails.md" \
            "scripts/review_challenge_data.py reports/data_review.md"; do
    set -- $pair
    if [ "$1" -nt "$2" ]; then
        echo "$2 is older than $1 -- regenerate it"
        exit 1
    fi
done
echo "ok"

say "each report carries the section its commit is about"
# A report newer than its generator can still be a report generated before the
# section was written, if the generator was touched afterwards and the report
# was not. These greps are what says the file on disk is the one the message
# below describes.
check() {
    if ! grep -qF "$2" "$1"; then
        echo "$1 does not carry \"$2\" -- regenerate it"
        exit 1
    fi
}
check reports/residual.md "## EGLL's July"
check reports/residual.md "## December by weather flag"
check reports/tails.md "## The callsign-prefix reading, at LIRF"
check reports/data_review.md "### The same days in the training year"
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

say "1/3 the residual report's two cuts"
git add scripts/report_residual.py reports/residual.md tests/test_residual.py \
        scripts/commit_diagnostics.sh
git commit -q -F - <<'MSG'
Say where EGLL's July error sits, and what December's weather flags do

Two sections over the submitted rung's stored validation residuals. Both
describe: no model is fitted, no feature is changed, and neither re-reads
a decision the ladder already took. Where the prose states a difference it
carries the day-clustered bootstrap interval taxiout.validation
.cluster_bootstrap gives, the day as the cluster, which is the standing
rule for a question put inside one split.

EGLL's July, five cuts with December at the same airport beside each one.
20,790 ex-flagged July rows at 318.8 s RMSE against 20,000 at 242.4 s,
a difference of +76.5 s [+16.7, +138.0]. It is not spread over the day:
the six hours carrying the most squared error are 14:00 to 19:59 UTC, and
they hold 60% of July's squared error on 38% of its rows, running
+144.6 s [+41.9, +254.0] above July's other hours, while the same hours in
December run -13.7 s [-49.3, +20.4] against December's -- so the afternoon
is a July effect and not a property of the airport. The 338 rows with no
Network Manager record are 1.6% of the leg and carry 20.9% of its squared
error at 1,142 s, +855.8 s [+303.4, +1,205.1] above the rows that have one,
against +203.3 s [+66.8, +390.1] for the same class in December. Runway 27R
carries 58.3% of the squared error on 44.5% of the rows, +88.5 s
[-14.1, +196.7]. One callsign prefix carries 58.0% of it on 43.2% of the
rows. The cuts by plan-lag band name the two row sets the lag table above
them leaves implicit -- no estimate, and a take-off earlier than its own
estimate -- so that the shares add up.

December by weather flag, the three flags taxiout.metar.regime_by_hour
produces, joined on the station and the take-off hour. Pooled over the ten
airports, freezing hours are 14,278 rows at 298 s against 208 s,
+90.0 s [+53.0, +124.0]; hours below 8 km of visibility 25,443 rows at
263 s against 208 s, +54.3 s [+24.9, +86.1]; hours with precipitation
17,696 rows at 250 s against 213 s, +36.0 s [-3.6, +72.3]. The three are
not one thing, and neither is any of them across the feeds: at four of the
ten airports the rows in precipitation score better than the rows without
it, and freezing covers 42.1% of one airport's December rows and 0.1% of
another's. Every December row's hour has a reading, so nothing falls
outside the flags.

A cell whose flagged side rests on fewer than five distinct take-off days
prints that count and no interval. Three do. One day resampled with
replacement is the same day every draw, and the interval it yields is
narrow because it has measured nothing, which is the opposite of what a
narrow interval means; printing the day count says so.
MSG

say "2/3 the tails report's prefix section"
git add scripts/report_tails.py reports/tails.md tests/test_report_tails.py
git commit -q -F - <<'MSG'
Settle the callsign-prefix reading of the record-quality term

A review note of 10 September, in no report until now, read several
callsign prefixes at LIRF with a Network Manager record as having taxi
times equal to their lag in most of their rows. If that were the defaulting
the term is about, its cells would be under-drawn: the term keys on
(airport, record, lag band) and would be missing an operator dimension.
The section measures the claim and what acting on it would reach. The
term, its inputs and its coverage rule are unchanged, and no model is
fitted.

The note's reading is not a second test. The target is take-off minus
off-block and the lag is take-off minus schedule, so taxi - lag is
schedule - block exactly, and |taxi - lag| <= 60 s is the first condition
of taxiout.tails.defaulted_block written another way. Measured rather than
asserted: over the 2,085,047 training departures the two agree on 100.0%
of rows, with 0 disagreeing. What the term adds is the two-hour condition,
and that is the whole of the difference between the two columns.

So the note's arithmetic holds and its reading of it does not. Of LIRF's
159,216 departures with a record, 20.5% are caught by the note's reading
and 0.020% are defaulted under the term's; the rows the note catches have
a median lag of 1,081 s against a median recorded taxi-out of 1,024 s over
all of that airport's departures with a record. The lag is the taxi on
those rows because the block time is the scheduled time, which is what a
punctual operator looks like and not what a defaulted block time looks
like. Without a record the same rows have a median lag of 7,532 s and
26.6% of them are defaulted.

What a prefix split would open, under the coverage rule as it stands: 32
(airport, prefix, bucket) cells at LIRF meet five rows and a 5% defaulted
share, 31 of them inside the three (airport, bucket) cells the term
already covers, where every row is predicted for already. One lies
outside, 13 training rows at a 7.7% defaulted share, and it would touch 1
of the 11,061 LIRF ranking departures. That is the whole exposure, printed
so the size of the question is on record; this commit does not decide it.

The report gains a callsign prefix as a printable quantity. It is the ICAO
airline designator taxiout.callsign already reads for the rung of that
name, an aggregate over an operator's departures and not an identifier of
any of them, and the generator's refusal to write a stand identifier is
unchanged and asserted on the new section's own text.
MSG

say "3/3 the data review's training-year disruption days"
git add scripts/review_challenge_data.py reports/data_review.md \
        tests/test_report_data_review.py
git commit -q -F - <<'MSG'
Ask whether the training year held the ranking set's disruption days

The review lists airport-days in the ranking set with at least twenty
unjoined or four-hours-late departures and a share of them over ten times
the airport's training-year rate, and says they are where de-icing and
recovery queues will sit. It never applied the rule to the training year,
so the list had nothing to be unusual against. Now it does, at the same
threshold and against the same per-airport baseline, with the three
regime flags taxiout.metar.regime_by_hour gives each listed station that
day beside it. Descriptive: no model is fitted and no feature changes.

The training year holds 17 such airport-days at 7 of the 10 airports, and
none at EHAM, which the ranking table lists on 6 days, or at LFPG, which
it lists on 1. They are not the same kind of event. The longest run of
consecutive flagged days in the training year is 2, at LEMD, against 6 at
EHAM in the ranking set; the most four-hours-late departures on any
training day is 36 against 88; and only 2 of the 17 training days reach
the twenty-late threshold at all, against 7 of the ranking set's 8. The
training year's flagged days are, in the main, days the flight-side join
failed on rather than days the airport stopped. The weather columns say
the same: most of them carry no low-visibility hour at all.

Four cells of one other table move, and they are not this change. "What
the coarsest stage collapses" was generated on 2 September at 15:29 and
committed in that state; a2fcfc0, five hours later, fixed two geometry
defects and moved the representative on 1,104 of 23,871 cells, including
52 at EGLL, 80 at LEBL and 285 at LTFM. The report was never regenerated
after it, so its median disagreement and over-250-metre counts at exactly
those three airports were the pre-fix ones. Regenerating brings them into
line with the matrices the repository builds now. No other figure in the
report moves.
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
