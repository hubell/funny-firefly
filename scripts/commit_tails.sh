#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the record-quality term and the submission writer -- Stream B of
# Gate 1 -- as three commits on top of the current history. Additive;
# nothing is rewritten. Adds only this stream's seven files, so it can run
# while the other Gate 1 stream's files sit untracked beside them.
#
# Order: commit_gate0.sh, then this, then commit_gate1.sh.
#
# Runs this stream's tests, the whole suite and the licence gate first and
# stops if any fails. The licence gate covers every report in the tree,
# tracked or not, so an unfinished report from the other stream can stop
# this script; the message says which file. Does not push. Review
# `git log -p -3` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

MINE="src/taxiout/tails.py src/taxiout/submission.py scripts/report_tails.py \
      reports/tails.md tests/test_tails.py tests/test_submission.py \
      scripts/commit_tails.sh"
for f in $MINE; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

say "this stream's tests"
python3 -m pytest -q tests/test_tails.py tests/test_submission.py

say "the whole suite"
python3 -m pytest -q

say "licence gate"
python3 scripts/check_licence_hygiene.py

say "1/3 the record-quality term"
git add src/taxiout/tails.py tests/test_tails.py
git commit -q -m "Add the record-quality term

The third term of the decomposition: what to predict for rows whose
recorded taxi-out no aircraft produced. The data review had found the
tail -- under 0.1% of training departures carrying half the airport-mean
squared error -- and had read it as day-shift artifacts. Profiling the
rows for this term found a different mechanism behind most of it. The
over-two-hour departures are 82% Fiumicino, and for nine in ten of those
the recorded off-block time sits within a minute of the scheduled
departure time (83% within five seconds) while the take-off is hours
later. The block time was defaulted to the schedule, so the recorded
taxi-out is take-off minus schedule -- the lag, which the ranking set
exposes for every departure. The day-shift the review described is real
but small: a dozen rows a day plus an ordinary taxi, in the same cells.

The term is the conditional mean per (airport, flight-record presence,
lag band) cell, computed knowing that mechanism. Fitting measures per
cell how often the block time was defaulted and what the remaining rows
average; prediction is share x lag + (1 - share) x rest. A cell is
covered when it has at least five rows and a defaulted share of at
least five percent, which at the full-year fit is three cells, all at
Fiumicino; every other row falls through to the physical model. The
plan's first proposal -- a plain cell mean with thin cells pooled over
airports -- was measured against this on the forward splits and set
aside: pooling carries one airport's artifact rate to airports that have
none and is the worst of the forms tried on every split, and the mixture
beats the cell mean by 12 s on December and 33 s on July because it
predicts each artifact's own value rather than the cell's average. The
plain cell mean is kept as kind=\"mean\" for the ablation.

Interface as fixed between the two Gate 1 streams: flag() reads only
LOBT_flt, MVT_TIME_UTC_mvt and SCHED_TIME_UTC_mvt; TailModel.predict()
adds ADEP_mvt and runs unchanged on the ranking set; fit() additionally
reads exactly the two fields the ranking set hides, on training rows.
The tests assert both against taxiout.challenge, on synthetic frames."

say "2/3 the submission writer"
git add src/taxiout/submission.py tests/test_submission.py
git commit -q -m "Validate and write submission files

The ranking page's words: a submission is scored only if \"each row must
match the corresponding MVT_ID_mvt\", and the ranking script errors if
\"an MVT_ID_mvt does not match; any rows are missing; or extra rows are
included\", with files named <team-name>_v<incremental integer>.parquet.
The template fixes the rest: MVT_ID_mvt as a double, TAXITIME_SEC_mvt as
a 32-bit integer, one row per ranking departure.

validate() raises on the first failure with a count-level message and
never an identifier: identifier set equal to the template's, no nulls or
duplicates, target integer-valued, finite, non-negative and within
int32. write() rounds a float target, validates, puts the rows in the
template's order so that \"each row must match\" holds positionally as
well as by set, writes the template's two typed columns with no pandas
metadata, reads the file back and validates that too, and refuses to
overwrite: the version is in the filename so that a rewritten file is a
new one. Tests on a synthetic template."

say "3/3 the report"
git add scripts/report_tails.py reports/tails.md scripts/commit_tails.sh
git commit -q -m "Report the record-quality term

reports/tails.md, generated: what the tail is made of, the (airport,
bucket) cells with their sizes and conditional means, the coverage rule
and the sweeps that justify it, and what the term is worth on the two
forward splits -- the share of airport-mean squared error the flagged
rows carry and the RMSE a physical model scores with and without it.
The physical model is a stand-in, the training mean per (airport,
stand, runway) fitted with flagged rows excluded, so the comparison has
a baseline before the real one exists; reports/baseline.md carries it
for the real model. Aggregates only; the generator refuses to write a
literal stand identifier.

The report states the term for what it is -- the RMSE-optimal
prediction under the stated scoring rule for rows whose recorded value
no aircraft produced -- quotes the scoring rule from the ranking page,
and states the two limits: it covers only cells that showed the
mechanism in 2025, one airport, and it touches 107 of the 215,876
ranking rows, all in January. The three July test airports have no
covered cell, so the term is inert on that half of the test set."

say "result"
git --no-pager log --oneline -4
echo
git status --short

say "no attribution in the new history"
# The pattern is spelled with bracket classes so that this file does not
# itself match the tree-wide grep in commit_gate0.sh.
PATTERN='co[w]ork|cla[u]de|anthro[p]ic|copi[l]ot|chat[g]pt|open[a]i|co-authored-b[y]'
if git log -3 --format='%B%n%an%n%ae%n%cn%n%ce' | grep -niE "$PATTERN"; then
    echo "FOUND -- inspect before pushing."
    exit 1
fi
echo "clean"

echo
echo "Not pushed. Review with: git log -p -3"
