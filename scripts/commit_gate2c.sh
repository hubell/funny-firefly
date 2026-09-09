#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit Gate 2c -- the four blocks the model never read, the ladder rungs that
# measure them, the report on the stand as a shared resource, and the plan's
# record of what the gate settled -- as five commits on top of the current
# history. Additive; nothing is rewritten.
#
# The commits are ordered so that each one leaves a tree whose tests pass: the
# blocks land as standalone modules with their own tests, then the wiring that
# registers them and puts them on the ladder, then the reports the refit
# produced, then the documentation. A reader bisecting this range gets a
# working repository at every step rather than four broken ones and a fifth
# that fixes them.
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

for f in src/taxiout/occupancy.py src/taxiout/callsign.py src/taxiout/destination.py \
         src/taxiout/calendar.py tests/test_occupancy.py tests/test_callsign.py \
         tests/test_destination.py tests/test_calendar.py \
         scripts/fetch_ourairports.py scripts/report_occupancy.py \
         reports/occupancy.md reports/baseline.md docs/review_headroom.md \
         docs/plan.md; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

# A generated report older than its generator carries figures from code that no
# longer exists. Regenerate before committing.
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

# A parquet file in a commit is the one mistake no later commit undoes, and this
# stream writes them; the OurAirports cache is the new thing that must never be
# tracked, for the same reason as the METAR one -- a reader regenerates it from
# the recorded request rather than trusting a copy.
if git status --porcelain --untracked-files=all data/challenge data/submissions \
        data/raw data/processed | grep -vE 'README.md|ODbL-NOTICE.md' | grep -q .; then
    echo "untracked, unignored files under data/ -- fix .gitignore first"
    exit 1
fi

say "1/5 the stand as a shared resource"
git add src/taxiout/occupancy.py tests/test_occupancy.py src/taxiout/challenge.py
git commit -q -F - <<'MSG'
Bound the taxi-out from the stand the subject had left

A stand holds one aircraft. If an arrival on-blocks at the subject's
stand before the subject takes off, the subject had already left the
stand, so its taxi-out is at least the gap between that on-block and
its take-off. The ranking set gives every arrival's stand and on-block
time in full, so that is a physical lower bound on the hidden field
built entirely out of exposed rows, and it is the operational quantity
A-CDM calls stand availability. At Heathrow in particular aircraft are
pushed back to free the stand and then held.

It is a feature and never a floor, and the block says why in its own
docstring. Below twenty minutes the bound holds in 99 to 100% of rows
and the model already respects it; the signal is the twenty-to-forty-five
minute band, where the arrival is either the subject's own inbound after
a fast turnaround or a different aircraft that took the stand once the
subject had pushed, and where the model under-predicts by two to six
minutes. The previous arrival's aircraft type and operator tell those two
apart. Above thirty minutes the bound collapses even when both differ,
because a stand identifier can denote a position holding two aircraft and
because aircraft are towed.

Ten columns: the gap to the last and second-last arrival at the stand,
that on-block against the subject's own schedule, the arrival's lateness
and taxi-in, the two match flags, the gap to the previous departure from
the same stand, and two flags separating a stand that received no arrival
from one whose previous arrival may lie before the covered block. No
re-occupation flag: the gap carries it with the threshold left in, and the
threshold is the booster's to find.

The key is the airport and the stand folded for case and whitespace and
nothing else. Both sides of this match come from the movement feed, where
a stand is written one way; the staged normalisation in taxiout.join
exists to bridge that feed to OpenStreetMap's, which is a different
problem and would here merge stands the feed distinguishes.

AVAILABLE_FOR_ARRIVALS gains the aircraft type and the operator, which
this block reads and nothing before it did, and its admission rule is now
written down rather than implied: a column enters when the data review puts
it at 1% or less null in the ranking arrivals, with the rate recorded beside
it. Most are at 0.00%; the operator is at 0.76%, which is the share of
arrivals whose flight-side join found nothing, and the block carries NaN
there rather than a false match. That list moves in this commit rather than
with the wiring, because the test holding this block's arrival sources to it
ships in this commit too, and a commit whose own tests fail is not one to
bisect through.

The registry names every column against the phase-qualified fields it
reads, and the tests hold the arrival sources to what the ranking set
exposes for arrivals, poison the departures' hidden fields and assert the
frame does not move, and poison the arrivals' on-blocks and assert that
the gaps do. Frames are built at microsecond resolution, because the
parquet files are and because a nanosecond-only fixture once hid a unit
bug for a week.
MSG

say "2/5 the operator, the destination and the calendar"
git add src/taxiout/callsign.py tests/test_callsign.py \
        src/taxiout/destination.py tests/test_destination.py \
        src/taxiout/calendar.py tests/test_calendar.py \
        src/taxiout/planlags.py tests/test_planlags.py \
        scripts/fetch_ourairports.py
git commit -q -F - <<'MSG'
Read the operator, the destination and the calendar

Three small blocks and two free columns, none of them clever and all of
them fields the model had never read.

The operator enters as the leading three letters of FLIGHT_mvt, which are
the ICAO airline designator, and not as AIRCRAFT_OPERATOR_flt. The two
carry the same information and are worth the same on the residual. What
decides it is coverage: the flight-side record is missing for 1.08% of
training departures and 1.48% of ranking ones, and those are the worst
rows in the dataset -- Heathrow's no-record departures run at 1,216 s RMSE
against a pooled 233 -- because a plan cancelled and refiled after a long
delay loses its join, so the missing record is itself a delay signal. A
feature keyed on the flight side abstains exactly where it is most needed.
FLIGHT_mvt is null for 0.01% of ranking departures.

A level seen fewer than thirty times in the fitting sample is pooled to
"other", the way wake classes pool in the geometric fit. That threshold is
counted on the fitting sample, so the pooling belongs to the fitted
estimator and not to the feature frame: the frame carries the raw prefix
and the booster does the pooling, which is what keeps the ranking set from
presenting a level the model never saw.

The destination enters as the initial great-circle bearing and the
distance, from OurAirports' public-domain coordinates, rather than as a
categorical of 914 levels. The destination decides the SID, and the SID
decides which runway entry and holding point the aircraft queues at; the
bearing means the same thing at every airport, so one learnt where the
data is thick is still read where it is thin. scripts/fetch_ourairports.py
caches the file with the URL, the fetch time, the row count and the terms
recorded in a manifest beside it, exactly as the METAR fetch does, and the
cache is gitignored for the same reason: a reader regenerates it from the
recorded request rather than trusting a copy.

Public holidays are a code dependency and not a dataset -- holidays
(MIT) computes each country's calendar from published rules, so there is
nothing to fetch, nothing to cache and no external prefix in the registry.
Two flags on the UTC date of take-off, at country level, with a note that
Germany, Spain and Switzerland settle most of theirs by subdivision and
that this deliberately does not read one.

And the two plan differences: EOBT minus the schedule, and LOBT minus
EOBT. The three plan lags are all take-off minus a planning value, so
their pairwise differences cancel the take-off and are properties of the
plan alone. The booster sees both lags and can reach the difference only
by splitting on each in turn. No new missing flag: lag_missing already
says it.
MSG

say "3/5 four rungs above the arm"
git add src/taxiout/features.py src/taxiout/models.py \
        tests/test_features.py tests/test_encoding.py \
        scripts/run_baselines.py scripts/make_submission.py pyproject.toml
git commit -q -F - <<'MSG'
Put the four blocks on the ladder above the arm

Four rungs in a chain on top of the arm the encoding section chose, each
read against the one below it, so a rung's number is what that block is
worth net of the blocks under it. They sit on the arm rather than on the
weather rung because the arm is what the submission is fitted from, and a
block measured against a rung the model does not use would be measuring a
model nobody runs.

They are kept out of the chain the ladder keeps without the plan lags,
and the exclusion is now stated as a property of the features rather than
as a list of keys: every rung carrying the shrunk encoding is out, which
is the three arms and these four. A twin of one of these would have to be
read against an arm without the plan lags, and there is no such rung.

The rule that admits a rung is fixed in the generator before the run, for
the reason the encoding section's own history gives: a rule that lives in
prose can be edited after the numbers, and one that lives in code has to be
edited in a diff. A rung displaces the rung below it if it is no worse on
both forward splits' ex-flagged RMSE and no worse pooled at the ranking
set's bucket mix; the chain stops at the first failure. The held-out-pair
contrast is printed and is not a gate, because none of these four is a
property of the pair and each is expected within a second of zero there --
which is stated before the number rather than after it.

The expected headroom from the residual fits is carried into the report as
a ceiling and not a target, and the generator applies the leakage-audit
trigger: a block that finds materially more than the information it
carries is a reason to audit the block, not to bank the gain.

Seed averaging is measured once, on the submission rung, by the rule fixed
before it ran: adopt only if five seeds averaged gain at least a second on
both forward splits. They gain 1.10 s and 1.37 s, so it is adopted, and
taxiout.models records it beside the booster's parameters as the estimator
choice it is -- it changes how the model is fitted and not what it reads, so
the ladder stays single-seed and only the submission writer uses it. On the
rung below this one it gained 0.78 and 0.98 and would not have been adopted,
which is also the clearest measure available of how much noise sits in a
single fit. The first of the five seeds is the one the ladder fits with, so
the table carries a reproducibility check for free, and it reads zero.
MSG

say "4/5 the report, and the refit"
git add scripts/report_occupancy.py reports/occupancy.md reports/baseline.md \
        reports/congestion.md scripts/report_transfer.py reports/transfer.md \
        Makefile README.md
git commit -q -F - <<'MSG'
Report the stand as a shared resource, and refit the ladder

reports/occupancy.md is the construction; reports/baseline.md is the worth.
The first fits no model. It prints the gap to the stand's last arrival on
both sides of the data, the same calendar month a year apart, so that a
feature built identically on both sides can be seen to have the same shape
on both sides; the bound by five-minute band, split by whether the previous
arrival flew the subject's own type, on the validation rows of a model that
has never read the gap; and the hard floor as the negative result it is.

The bound table is read against the rung the block was added to and not
against whatever rung the submission has since moved to. A model that
already reads the gap will respect the gap, and a table showing that would
be measuring its own feature.

congestion.md and transfer.md are regenerated because both follow the
submission rung and the submission rung moved: the ceiling section reads
its residuals, and the transferable feature set is that rung's minus every
identifier, which now carries the stand-occupancy block and the callsign
prefix. Neither report changed shape; both changed rung.

The Makefile gains occupancy and ourairports targets; the README its
data-sources entry for OurAirports as a model input rather than only a
cross-check, the holidays dependency stated as a dependency and not a
dataset, the two datasets that were considered and ruled out, the stage
rows, the layout tree and the status list.
MSG

say "5/5 what the gate settled"
git add docs/plan.md docs/modelling_notes.md docs/paper.md \
        docs/review_headroom.md scripts/commit_gate2c.sh
git commit -q -F - <<'MSG'
Record what Gate 2c settled

docs/review_headroom.md is the review the gate was decided from: where the
remaining error sits, what each unused exposed field is worth on the
residual, the licence position of every external source considered
including the PRU portal data that is excluded by the same reading that
excluded OPDI, and the decisions taken on 8 September.

The plan records the outcome. The modelling notes record the one thing
about the stand block a reader should be given rather than left to find:
it bounds a field the ranking set hides, from rows the ranking set
exposes, and it reads an arrival that on-blocked after the subject's own
off-block -- admissible under the challenge's post-hoc framing and not
under an operational one, which is the standing of every take-off-anchored
window already in the model.

The paper gains two sections marked pending, for the stand mechanism and
for what the residual actually is, and a drafting note saying exactly
which of its figures move with this refit and which do not. The ones that
do not are the ones anchored below the arms: the claim contrast, the
geometry sections, the tail treatment and the ceiling.
MSG

say "result"
git --no-pager log --oneline -6
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -5"
