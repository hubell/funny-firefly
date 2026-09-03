#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the rest of Gate 2 and Gate 2b -- the weather regime, the report on
# how the excess term was built, and the plan's record of what both gates
# settled -- as three commits on top of the current history. Additive;
# nothing is rewritten.
#
# The Gate 2 feature work before this (the held-out-pair contrast with
# intervals, the aircraft, the windows, the congestion blocks, the corridor)
# is already in the history as separate commits, 22bd660 through 842d19a.
# This script lands what was still in the working tree after them.
#
# Runs the tests, the lint and the licence gate first and stops if any fails,
# and refuses to commit a report older than the code that generates it. Does
# not push. Review `git log -p -3` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in scripts/fetch_metar.py src/taxiout/metar.py tests/test_metar.py \
         scripts/report_congestion.py reports/congestion.md reports/baseline.md \
         docs/plan.md; do
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

say "tests"
python3 -m pytest -q

say "lint"
ruff check .
ruff format --check .

say "licence gate"
python3 scripts/check_licence_hygiene.py

# A parquet file in a commit is the one mistake no later commit undoes, and
# this stream writes them; the METAR cache is the second thing that must
# never be tracked, because a reader regenerates it from the recorded request.
if git status --porcelain --untracked-files=all data/challenge data/submissions \
        data/raw data/processed | grep -vE 'README.md|ODbL-NOTICE.md' | grep -q .; then
    echo "untracked, unignored files under data/ -- fix .gitignore first"
    exit 1
fi

say "1/3 weather as a regime indicator"
git add scripts/fetch_metar.py src/taxiout/metar.py tests/test_metar.py \
        src/taxiout/features.py src/taxiout/models.py \
        scripts/run_baselines.py scripts/make_submission.py tests/test_features.py
git commit -q -m "Enter the weather as three regime flags per airport-hour

Gate 2b. The organisers name airport procedures and load among the
drivers, and the ranking set's first week of January 2026 at Schiphol
has the signature of winter operations. What weather does to taxi-out
is not continuous but a regime: low-visibility procedures space
departures out, precipitation slows the surface, freezing conditions
add de-icing and its queue. So it enters as three flags per airport-hour
and nothing else -- a visibility band with its lower edge at 1.5 km,
where low-visibility procedures are in force; precipitation at the
aerodrome, with present weather in the vicinity dropped, because showers
eight kilometres out are not rain on the taxiway; and freezing, from
the air temperature or a freezing or frozen group. Each hour takes the
worst of its reports, an empty hour inherits the last for up to three
hours and says how old it is, and a subject reads the hour its take-off
falls in.

The reports are METARs, which the airports publish; the Iowa
Environmental Mesonet at Iowa State University serves an archive of
them through a documented request, and scripts/fetch_metar.py asks it
for the training year and the two ranking months at every listed
airport, one request per station and period, resumable, with a pause
between. The request parameters and the terms the Mesonet states are
recorded in a manifest beside the data: the materials are in the public
domain and may be used freely by anyone for any lawful purpose,
offered without any warranty of accuracy, and attributing the Mesonet
would be appreciated, which the README does. The cache is gitignored
with the OpenStreetMap cache for the same reason: a reader regenerates
it from the recorded request rather than trusting a copy. Every station
came back with its own report every hour.

The block is the one that reads outside the challenge data, and the
registry says so: its sources carry the METAR prefix and are held to the
archive's own field list, the same way arrivals' sources are held to
theirs. The runner skips the rung, and says so, when the cache is
absent; the submission writer refuses without it.

What it is worth. Ten seconds in December and nothing in July, and the
ten are where winter is: over the rows the record-quality term leaves to
the physical model, the December RMSE goes from 304.5 s to 294.4 s, with
Munich 391 to 357 s, Paris 374 to 352 and Schiphol 221 to 202, while
Frankfurt, Zurich and Istanbul move by a few seconds. July goes from
400.3 to 400.1 s. On held-out pairs the contrast is +0.2 s [-0.4, +0.8]:
weather is not a property of the pair. The regime the ranking set will
be scored in is the one the training year saw less of -- January 2026
froze for 48% of Frankfurt's hours against 30% in January 2025, 65% of
Zurich's against 41% -- and the flags are what a model fitted on 2025
extrapolates along."

say "2/3 how the excess term was built"
git add scripts/report_congestion.py reports/congestion.md reports/baseline.md \
        Makefile README.md
git commit -q -m "Report how the excess term was built, and what each block is worth

reports/congestion.md is the construction; reports/baseline.md is the
worth. The first fits no model. It prints the leakage audit -- every
block against the phase-qualified columns it reads, and the two lists
each phase is held to -- so a reader checks the rule without running
the tests; the same counts on both sides of the data a year apart,
which agree to within a count at every airport because each ranking
month is a census; the ten training rows and sixteen ranking rows whose
window reaches before their covered block; the corridor's degeneracy
test, its correlation with the plain take-off count per airport; the
runway configurations each airport runs and how long they hold; and the
weather regime's coverage and flag rates, January 2025 against January
2026.

The ladder now has thirteen rungs and says what each congestion block
is worth on the splits fixed before any of it was fitted. Over the rows
the record-quality term leaves to the physical model, the five
congestion blocks take the December RMSE from 321.5 s to 304.5 s and
July from 405.6 to 400.3; arrivals and demand carry it, about five
seconds each in December, throughput about four; the runway
configuration and the shared corridor buy nothing more on any split.
Weather takes December to 294.4 s. The top rung stands at 294.4 s in
December and 400.1 s in July, 277.3 and 315.9 s on the three July
airports, against 322.0 and 407.9 s when Gate 1 closed.

The shared corridor is reported as it fell. It is a distinct column --
it correlates with the take-off count at 0.29 to 0.56 per airport --
and after throughput, arrivals, demand and configuration it moves
December by +0.4 s, July by -1.1 s and the held-out contrast by -0.6 s
[-1.0, -0.2]. That the booster finds nothing in it is a finding about
the routing model, a full-length route with no notion of where in the
window the other aircraft was, and it stays in the paper for that
reason. None of the congestion blocks moves the held-out-pair claim,
which stays at +14.5 s [+13.0, +16.3]; congestion is not a property of
the pair, and every block's contrast sits within 1.3 s of zero.

The Makefile gains congestion and metar targets, and the README its
data-sources entry for the Mesonet, the stage rows, the layout tree and
the status list."

say "3/3 what the gates settled"
git add docs/plan.md scripts/commit_gate2.sh
git commit -q -m "Record what Gates 2 and 2b settled

Congestion, as the ranking set exposes it, is worth about seventeen
seconds in December and five in July; weather ten more in December and
none in July, where winter is. The shared corridor is a distinct column
the model cannot use. None of it moves the held-out-pair claim, which
is what it should do. The configuration set is a categorical the
booster splits on freely and learns nothing from. Whether seventeen
seconds is congestion binding enough for Gate 3 is a decision the plan
leaves open, with the numbers beside it.

Also recorded: the microsecond timestamps the window code first read as
nanoseconds, found on the real year and not by the tests, and the same
calendar month a year apart having the same shape on both sides, which
is the census the data review promised being kept."

say "result"
git --no-pager log --oneline -4
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

echo
echo "Not pushed. Review with: git log -p -3"
