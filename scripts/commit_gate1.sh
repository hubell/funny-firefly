#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-only
#
# Commit the Gate 1 modelling stream -- the feature frame and its source
# registry, the unimpeded term, the validation harness, the ablation ladder
# and the submission writer -- as six commits on top of the current history.
# Additive, like commit_gate0.sh: nothing is rewritten.
#
# Scope and ordering. This commits only the files this stream owns, and it
# runs last of three: commit_gate0.sh (the join and the data review, which
# also carries the plan's Gate 0 revisions), then commit_tails.sh (Stream B's
# src/taxiout/tails.py and src/taxiout/submission.py, their tests and
# reports/tails.md), then this. The order is not cosmetic -- reports/baseline.md
# is generated with taxiout.tails wired in and scripts/make_submission.py
# imports taxiout.submission, so a tree with this commit and not those does
# not run.
#
# Runs the tests and the licence gate first and stops if either fails. Does
# not push. Review `git log -p -6` afterwards.

set -euo pipefail
cd "$(dirname "$0")/.."

say() { printf '\n== %s\n' "$1"; }

if [ ! -f LICENSE ] || [ ! -f pyproject.toml ]; then
    echo "Run this from the repository, not from somewhere else."
    exit 1
fi

for f in src/taxiout/features.py src/taxiout/unimpeded.py \
         src/taxiout/validation.py src/taxiout/models.py \
         scripts/run_baselines.py scripts/make_submission.py \
         tests/test_features.py tests/test_unimpeded.py \
         tests/test_validation.py reports/baseline.md; do
    if [ ! -f "$f" ]; then
        echo "missing $f -- nothing to commit"
        exit 1
    fi
done

# reports/baseline.md is generated, and a stale one is worse than none: it
# would carry figures from a feature frame that no longer exists. Regenerate
# it before committing if anything it reads has changed since.
if [ -n "$(find src/taxiout/features.py src/taxiout/unimpeded.py \
                src/taxiout/models.py src/taxiout/validation.py \
                scripts/run_baselines.py -newer reports/baseline.md 2>/dev/null)" ]; then
    echo "reports/baseline.md is older than the code that generates it."
    echo "Run: python3 scripts/run_baselines.py"
    exit 1
fi

say "tests"
python3 -m pytest -q

say "lint"
ruff check .
ruff format --check .

say "licence gate"
python3 scripts/check_licence_hygiene.py

# Belt and braces: a parquet file in a commit is the one mistake that cannot
# be undone by a later commit, and this stream writes them.
if git status --porcelain --untracked-files=all data/challenge data/submissions \
        | grep -v 'data/challenge/README.md' | grep -q .; then
    echo "untracked, unignored files under data/challenge or data/submissions"
    exit 1
fi

say "1/6 feature frame"
git add src/taxiout/features.py tests/test_features.py
git commit -q -m "Build the feature frame, and register what every column reads

The Gate 1 frame: the airport, stand and runway identifiers; time
features anchored on the take-off time, which is the only actual stamp
a test departure carries; the distance-matrix cell with four shape
ratios that make the route descriptors comparable between a 400 m taxi
and a 4 km one; the join-quality flags, which say how much to trust the
geometry on a row -- which normalisation stage matched, how many OSM
identifiers the cell stands for and how far their routes disagree, how
far the stand was snapped; and LOBT - IOBT, how far the planned
off-block moved between the Network Manager's first estimate and its
last.

Stands that do not join fall back to airport level rather than carrying
nulls. Four percent of departures are stands OpenStreetMap does not
have, and dropping or nulling them would lose the rows the model is
least able to guess. The fallback is the median matrix cell for that
(airport, runway), or for the airport where the runway names no runway,
and geom_imputed marks every row it fired on. It is computed from the
matrix alone and never from the target, so it is the same table on both
sides of every split and needs no fitting.

FEATURE_SOURCES maps every feature column to the raw columns behind it,
so the leakage rule can be tested rather than asserted. The test has
two halves, because a registry can be right while the code beside it is
wrong: no declared source may be in FORBIDDEN_FEATURE_INPUTS, and
poisoning every forbidden field must leave the built frame unchanged."

say "2/6 unimpeded term"
git add src/taxiout/unimpeded.py tests/test_unimpeded.py
git commit -q -m "Estimate unimpeded taxi-out two ways, and calibrate one on the other

The PRU reference is the organisers' own: the 10th percentile of
observed taxi-out per (airport, stand, runway), over pairs seen at
least thirty times, cascading to (airport, runway) and then to the
airport where a pair has not been. It is exact where it exists and
undefined where it does not, and that gap is the problem the geometry
exists to fill.

The geometric version is the same quantity as a function of the route:
a per-airport speed turns taxi_m into seconds and five pooled
coefficients on standardised route descriptors modulate it, in the
sense of Ravizza et al. The fit is stated as a reproduction rather than
as a regression on taxi-out -- the speed is fitted on the unflagged
lower tail, the rows at or below their own pair's reference, with each
row's target being that reference -- so taxi_m / speed is a least
squares reproduction of the reference by construction, weighted by how
often each pair was observed.

How well it manages that is measured rather than assumed, and the
answer is not flattering: 252 s RMSE against a median reference of
728 s on the seen pairs, the middle eighty percent of pairs between
0.69 and 1.53 times the reference, and a bias that changes sign between
Frankfurt and Istanbul. A per-airport speed and five pooled
coefficients are a coarse instrument. The report says so, and what the
geometry buys is argued from the held-out-pair split instead.

Flagged rows arrive as a boolean mask rather than being recomputed
here, so this module has no opinion about how a row gets flagged."

say "3/6 validation harness"
git add src/taxiout/validation.py tests/test_validation.py
git commit -q -m "Fix the splits before fitting anything

Three splits, and each answers a different question. Two forward in
time: January-November 2025 against December, the winter proxy for a
January test month the training year cannot hold out forward, and
January-June against July, with the three airports the July test set
contains reported separately. Never a random K-fold on rows, because
congestion is autocorrelated over hours.

And the one the entry rests on: whole (airport, stand, runway) pairs
held out of training, dealt round-robin within each airport so every
fold holds out pairs everywhere. Each fold fits one model and scores it
twice -- on the held-out pairs, and on a five percent sample of rows
whose pairs stayed in training and which were themselves excluded from
the fit. One model, two evaluation sets, so the gap is attributable
rather than being two experiments differing in several things at once.

Two corrections turned out to be needed and are in the harness rather
than in the prose. Held-out pairs are the rarer pairs at the quieter
stands, so part of the gap is composition and none of that part is
about generalising; the airport mean knows nothing about a pair, so its
gap measures the composition and every other rung is read net of it.
And a raw RMSE over a five percent sample is dominated by whichever
record-matching artefacts fall in it, so the same comparison is
reported over the rows the record-quality term leaves alone, which is
the stable one.

Error is also stratified by how often the row's pair appeared in
training, and every metric is reported per airport beside the pooled
figure, since the test set is weighted towards three airports."

say "4/6 ablation ladder"
git add src/taxiout/models.py scripts/run_baselines.py reports/baseline.md
git commit -q -m "Run the ablation ladder and report what each term is worth

The ladder is the decomposition read bottom-up: airport mean, the PRU
reference, the geometric unimpeded term alone, a boosted model on
identifiers and time, and the same model with the geometry block added.
Each rung is scored plain and with the record-quality term of
taxiout.tails on top; the physical rungs are fitted with the rows that
term covers excluded and evaluated with them in, which is what the
stated scoring rule counts. Every number in the report is computed by
the generator, which refuses to write a literal stand identifier from
either source.

What it found, including where it goes against the geometry. On the
forward splits the geometry block buys nothing at all -- 442.2 s
against 442.1 in December, 677.2 against 676.4 in July -- which is
exactly what the validation notes predicted before anything was fitted,
because a saturated identifier model has seen every pair it is asked
about. On the held-out-pair split it buys something and the amount is
small: the identifier model's error rises 35.8 s more on a pair it has
never seen than the airport mean's does, and adding the geometry cuts
that to 21.4 s, so the physical layer recovers about two fifths of what
memorising the pair was worth. The geometry block takes 9.7% of the top
rung's splits against 36.9% for the stand identifier alone.

Two rungs need reading with care and the report says how. The PRU
reference and the geometric term estimate unimpeded taxi-out and are
being scored against actual taxi-out, so they are biased low by the
whole excess term; a per-airport offset column, fitted on training rows,
adds that back as the constant Gate 2 exists to replace with a model.

Hyperparameters are written out in BOOSTER_PARAMS and were chosen once,
for a compute budget, before any RMSE existed. There is no early
stopping, because stopping on the set being reported is tuning on it."

say "5/6 submission writer"
git add scripts/make_submission.py
git commit -q -m "Write a submission from the top rung of the ladder

Fits the top rung on the full training year with the rows the
record-quality term covers excluded, combines the two, clips the
handful of negative predictions a boosted model extrapolates on rows
unlike anything it saw, and hands the frame to taxiout.submission,
which owns the validation, the template's schema and the read-back
check. Nothing about the model is chosen by a leaderboard score: the
rung and its parameters are the ones reports/baseline.md was generated
with.

The file goes to data/submissions/<team>_v<n>.parquet, and the script
asks git whether that path is ignored before writing a byte rather than
trusting that it is. A submission is built from data that may not be
redistributed, and .gitignore excluding every parquet file is a
property worth checking at the point of use."

say "6/6 plan"
git add docs/plan.md scripts/commit_gate1.sh
git commit -q -m "Record what Gate 1 settled

Four outcomes, two of them against the geometry, and the plan says so
in the same place it recorded Gate 0's. The calibration of the
geometric unimpeded term against the PRU reference is the weak result
and is stated as one. The forward splits show the geometry buying
nothing, which was predicted. The held-out-pair split shows it
recovering about two fifths of what memorising a pair was worth, which
is real and is not a headline. The record-quality term dominates the
levels, as the data review said it would.

Also recorded because it cost an hour: pyarrow 19.0.0 cannot read the
challenge parquet files at all, on a page-index bug fixed in 19.0.1.
The floor in pyproject.toml is still >=15.0 and should be raised."

say "result"
git --no-pager log --oneline -8
echo
git status --short

say "nothing untracked that should be tracked"
git status --porcelain --untracked-files=all | grep '^??' || echo "clean"

say "no assistant attribution anywhere"
# Check 4 of scripts/check_licence_hygiene.py, run above, covers the
# tree and the recent history in one place. It spells the pattern in
# bracket classes so it does not match its own source, which is what
# made every copy of this grep have to exclude every other one.

echo
echo "Not pushed. Review with: git log -p -6"
