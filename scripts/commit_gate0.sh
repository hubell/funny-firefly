#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the Gate 0 work -- the data review, the identifier join and the plan
# revisions that followed from them -- as four commits on top of the current
# history. Additive, unlike commit_geometry.sh: nothing is rewritten.
#
# Runs the tests and the licence gate first and stops if either fails. Does
# not push. Review `git log -p -4` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in reports/data_review.md src/taxiout/join.py src/taxiout/challenge.py \
         scripts/review_challenge_data.py tests/test_join.py; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "tests"
python3 -m pytest -q

say "licence gate"
python3 scripts/check_licence_hygiene.py

# Anything under data/challenge/ other than the README must be ignored. Belt
# and braces: the licence gate checks this too, but a parquet file in a commit
# is the one mistake that cannot be undone by a later commit.
if git status --porcelain --untracked-files=all data/challenge \
        | grep -v 'data/challenge/README.md' | grep -q .; then
    echo "untracked, unignored files under data/challenge/ -- fix .gitignore first"
    exit 1
fi

say "1/4 licence gate covers the challenge data"
git add .gitignore scripts/check_licence_hygiene.py scripts/commit_geometry.sh
git commit -q -m "Extend the licence gate to the challenge data

The challenge datasets are supplied under terms that permit no
redistribution at all, which makes them the stricter of the two classes
of artefact this repository must never track -- the OpenStreetMap-derived
matrices can at least be relicensed under ODbL; a committed parquet file
cannot be undone by any licence. .gitignore now excludes everything under
data/challenge/ except the README that says what goes there, and the
hygiene check treats any parquet file, anywhere, as forbidden.

The identifier check now covers both sources. A literal STAND_mvt value
is an extract of the challenge data exactly as an OSM ref is an extract
of OpenStreetMap, so the same rule -- shapes in prose, never values --
applies to both, and the check reads the challenge side whenever the
data is present locally. CI has neither and says which it skipped."

say "2/4 join layer"
git add src/taxiout/challenge.py src/taxiout/join.py tests/test_join.py \
        pyproject.toml scripts/build_taxi_graph.py scripts/report_coverage.py
git commit -q -m "Join the challenge identifiers to the distance matrices

STAND_mvt and the OSM ref name the same stands in slightly different
conventions, and the differences are per airport: Frankfurt's OSM refs
carry a space between pier letter and number, Schiphol's movement data
zero-pads where OSM does not, Munich and Istanbul map split stands as
lettered halves where the movement data names the whole. The join runs
in three stages, each applied to both sides and each tried only for the
rows the previous stage left unmatched, so precision is kept where it
exists and only rows that would otherwise be lost fall through to a
coarser key. Exact matching gave 57-99% by airport; the staged join
gives 87-99.9%, 96% of departures overall, and the same on the ranking
set. Each stage was chosen by inspecting what it fixed, never by what
it did to a validation error.

Two things the matrix builder did not have to handle. A few OSM refs
carry two identifiers in one tag and are split rather than
whitespace-stripped into a false match. And a coarser key can fold
several OSM identifiers onto one; the collapsed cell records how many
and how far their taxi distances disagree, exactly as the matrix does
for duplicate features behind one identifier. Nothing decides what to
do with an ambiguous cell here -- it makes the ambiguity visible.

taxiout.challenge reads the datasets and, more importantly, records
which fields the ranking set hides for departures and which one is
excluded by decision, so the feature layer can be tested against a
constant rather than a paragraph. Millions of movements share a few
thousand distinct (airport, stand, runway) triples, so the join runs on
the triples and spreads back by integer code: about four seconds for
the training year.

pandas and pyarrow join the dev extras so CI runs the join tests; they
skip rather than fail where pandas is absent, so the clean-checkout path
still works with pytest alone. The ruff rule set is now stated
explicitly: a newer release had widened the defaults and failed the
lint on code that had not changed, and two files are reformatted for
the current formatter's f-string quoting."

say "3/4 data review"
git add scripts/review_challenge_data.py reports/data_review.md Makefile \
        data/challenge/README.md
git commit -q -m "Review the challenge data before writing a feature

Gate 0 of the plan: profile the data and answer three questions before
any feature code exists. Every number in the report is computed by the
generator; nothing is hand-typed. The report carries aggregates and
identifier shapes only, and the generator refuses to write a literal
stand identifier from either source.

What it found, in the order it matters. The data covers ten airports,
not the eleven listed: Antalya has no rows in training or ranking,
though the row total matches the organisers' figure exactly. The
ranking set blanks the off-block time of every departure, not only the
target, so a test departure carries its take-off time, its schedule
and the flight plan's estimates and nothing else that is actual -- and
no feature can count other departures' pushbacks either, which retires
the congestion feature the plan had called Gate 2. AOBT_3_flt survives
but is a minute-resolution estimate that misses the blanked field by a
median of 51 s with an interquartile range of five minutes; as a single
predictor it is worth no more than the filed EOBT, and it is excluded.
The target has a heavy tail: under 0.1% of training departures carry
half the airport-mean squared error, most of them record-matching
artefacts a day long, and the ranking set carries the two fields that
flag them at the same rate as training; the stated scoring rule is RMSE
over every row of the template. January 2026 is a census of all
ten airports; July 2026 covers Frankfurt, Heathrow and Schiphol only.

The identifiers join -- runways at ~100%, stands at 87-99.9% by airport
-- so the geometry applies directly to every test row."

say "4/4 plan revisions"
git add README.md docs/modelling_notes.md docs/plan.md docs/prior_art.md \
        scripts/commit_gate0.sh
git commit -q -m "Revise the plan for what the data turned out to contain

Three findings changed it. The off-block time is blanked for every
departure in the ranking months, so the leakage rule is restated around
the anchor a test row actually offers -- its take-off time -- and the
congestion layer is rebuilt from what other movements expose: take-offs,
schedules, and arrivals' landing, on-block and taxi-in times. The
challenge is a post-hoc estimation task, not the operational forecast the
first draft assumed; the notes state both framings and build to the
challenge's, and say why the schedule-only alternative was set aside.
AOBT_3_flt is excluded by decision and the reasoning is recorded.

The scoring rule as stated counts every row, and the decision is to
model the artefact rows explicitly, as a separate record-quality component kept
apart from the physical model, with RMSE reported with and without them.

Validation is fixed to two forward splits -- December 2025 at all ten
airports, July 2025 at the three the July test set contains -- with the
held-out-pair diagnostic unchanged. Antalya is recorded as absent from
the data.

A review against the published judging criteria the same day added the
rest. The model is stated as the decomposition the organisers' own
rationale implies -- unimpeded time from geometry, excess from
congestion and regime, a record-quality term kept apart -- which is what
makes the physical layer the organisers' reference generalised rather
than a feature beside it. Claims about the 2025 jury are now quoted from
its outcome page rather than paraphrased. OpenSky surface tracks, if
access arrives, are diagnostics for the paper and never model inputs,
because the prize rules require additional datasets under an open-source
licence and research-use is not that. METAR from the Iowa State Mesonet
enters as a scripted, cached regime indicator. The leakage-audit
threshold in the prior-art notes is restated as a ratio against the
ex-tail baseline, since the absolute figure it quoted was for a cleaned
single-airport target. And the plan gains a freeze date, 4 October, with
a pinned OSM rebuild inside the final week so the committed figures can
be reproduced by whoever forks the repository later."

say "result"
git --no-pager log --oneline -5
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

say "no attribution anywhere in the tree or the new history"
FOUND=0
grep -rniE 'cowork|claude|anthropic|copilot|chatgpt|openai|co-authored-by' \
    --exclude-dir=.git --exclude="$(basename "$0")" \
    --exclude=commit_geometry.sh . && FOUND=1
git log -4 --format='%B%n%an%n%ae%n%cn%n%ce' \
    | grep -niE 'cowork|claude|anthropic|copilot|chatgpt|openai|co-authored-by' && FOUND=1
if [ "$FOUND" -ne 0 ]; then
    echo "FOUND -- inspect before pushing."
    exit 1
fi
echo "clean"

echo
echo "Not pushed. Review with: git log -p -4"
